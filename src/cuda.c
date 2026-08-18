// CUDA GPU backend: full single-token forward pass on NVIDIA GPUs.
//
// Uses the CUDA *driver API* loaded dynamically (nvcuda.dll / libcuda.so.1),
// so building and running need no CUDA toolkit — a machine without an NVIDIA
// driver simply falls back to CPU. Kernels live in kernels.cu, compiled to
// PTX at development time (make ptx) and embedded via kernels_ptx.h; the
// driver JIT-compiles them for whatever GPU is present.
//
// Unlike Metal there is no unified memory: weights are copied to VRAM once at
// init, and the fp16 KV cache lives in VRAM with the host copy authoritative.
// CPU-written rows (batched prompt processing) are uploaded lazily — a
// non-contiguous position step means host rows [0, pos) may have changed and
// triggers a resync — and each GPU-written row is copied back to host so the
// CPU path can always take over mid-run.
#include "runner.h"
#include "kernel_args.h"   // shared with src/kernels.cu — see the header

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// clock_gettime/CLOCK_MONOTONIC in prof_now: not transitively available on
// every libc, so include it directly rather than relying on another header
#include <time.h>
// stat() identifies the weight file for the shared-upload registry, and the
// registry itself is mutex-guarded: slots load in parallel with serving
#include <pthread.h>
#include <sys/stat.h>

#ifdef _WIN32
#include <windows.h>
typedef HMODULE dl_t;
static dl_t dl_open(void) { return LoadLibraryA("nvcuda.dll"); }
static void *dl_sym(dl_t h, const char *s) { return (void *)GetProcAddress(h, s); }
#else
#include <dlfcn.h>
typedef void *dl_t;
static dl_t dl_open(void) {
    dl_t h = dlopen("libcuda.so.1", RTLD_NOW);
    return h ? h : dlopen("libcuda.so", RTLD_NOW);
}
static void *dl_sym(dl_t h, const char *s) { return dlsym(h, s); }
#endif

#include "kernels_ptx.h"

// ------------------------------------------------- driver API (what we use)

typedef int       CUresult;   // CUDA_SUCCESS == 0
typedef int       CUdevice;
typedef void     *CUcontext;
typedef void     *CUmodule;
typedef void     *CUfunction;
typedef uint64_t  CUdeviceptr;
typedef void     *CUstream;
typedef void     *CUgraph;
typedef void     *CUgraphExec;

static struct {
    dl_t lib;
    CUresult (*Init)(unsigned);
    CUresult (*DeviceGetCount)(int *);
    CUresult (*DeviceGet)(CUdevice *, int);
    CUresult (*DeviceGetName)(char *, int, CUdevice);
    // _v2 is the one that reports a MIG *instance* UUID; the v1 entry point
    // reports the parent card, which is identical for every slice on it.
    CUresult (*DeviceGetUuid_v2)(unsigned char *, CUdevice);
    CUresult (*DeviceGetUuid)(unsigned char *, CUdevice);
    CUresult (*DeviceGetPCIBusId)(char *, int, CUdevice);
    CUresult (*PrimaryCtxRetain)(CUcontext *, CUdevice);
    CUresult (*PrimaryCtxRelease)(CUdevice);
    CUresult (*CtxSetCurrent)(CUcontext);
    CUresult (*MemGetInfo)(size_t *, size_t *);
    CUresult (*MemAlloc)(CUdeviceptr *, size_t);
    CUresult (*MemFree)(CUdeviceptr);
    CUresult (*MemcpyHtoD)(CUdeviceptr, const void *, size_t);
    CUresult (*MemcpyDtoH)(void *, CUdeviceptr, size_t);
    CUresult (*MemcpyDtoD)(CUdeviceptr, CUdeviceptr, size_t);
    CUresult (*MemsetD8)(CUdeviceptr, unsigned char, size_t);
    CUresult (*ModuleLoadData)(CUmodule *, const void *);
    CUresult (*ModuleUnload)(CUmodule);
    CUresult (*ModuleGetFunction)(CUfunction *, CUmodule, const char *);
    CUresult (*LaunchKernel)(CUfunction, unsigned, unsigned, unsigned,
                             unsigned, unsigned, unsigned,
                             unsigned, void *, void **, void **);
    CUresult (*CtxSynchronize)(void);
    CUresult (*GetErrorString)(CUresult, const char **);
    // optional: CUDA graphs for the single-token decode path (Experiment A).
    // Not required for the backend to function — see cu_graphs_ok().
    CUresult (*StreamCreate)(CUstream *, unsigned);
    CUresult (*StreamDestroy)(CUstream);
    CUresult (*StreamSynchronize)(CUstream);
    CUresult (*StreamBeginCapture)(CUstream, int);
    CUresult (*StreamEndCapture)(CUstream, CUgraph *);
    CUresult (*GraphInstantiateWithFlags)(CUgraphExec *, CUgraph, unsigned long long);
    CUresult (*GraphLaunch)(CUgraphExec, CUstream);
    CUresult (*GraphExecDestroy)(CUgraphExec);
    CUresult (*GraphDestroy)(CUgraph);
    // optional: CUDA events for RUNNER_CUDA_PROFILE phase timing
    CUresult (*EventCreate)(void **, unsigned);
    CUresult (*EventRecord)(void *, CUstream);
    CUresult (*EventSynchronize)(void *);
    CUresult (*EventElapsedTime)(float *, void *, void *);
    CUresult (*EventDestroy)(void *);
} cu;

// ---------------------------------------------------- RUNNER_CUDA_PROFILE
// Coarse GPU phase breakdown behind an env var; OFF by default. Uses CUDA
// events recorded between kernel-group launches on the plain (non-graph)
// path, so it perturbs neither correctness nor the default binary. See
// prof_mark()/prof_flush() and the summary printed from gpu_free().
//
// THREAD SAFETY: the `prof` accumulator below is a single unsynchronized
// global. It is a DEV-ONLY, SINGLE-THREAD instrument — enable it only when
// driving the runner from one decode thread. Under a multi-threaded server
// its counters/event pool would race; that is accepted because it is never on
// in a shipped binary. Do not read prof.* outside that assumption; a lock
// here would sit on the hot launch path for no production benefit.
typedef void *CUevent;
enum { PH_NORM = 0, PH_MATVEC, PH_ROPE, PH_ATTN, PH_ELEM, PH_LOGITS, PH_N };
static const char *PH_NAME[PH_N] = {
    "norms(rms+qk)", "matvec", "rope", "attention", "elementwise", "logits-mv" };
// two independent accumulator sets: mode 0 = prefill/batch tiles (tn>1, _b
// kernels), mode 1 = single-token decode (tn==1, k_mv_* kernels). Set per
// gpu_forward_batch call from n, so one run yields both clean breakdowns.
enum { M_BATCH = 0, M_SINGLE = 1, M_N = 2 };
static struct {
    int      on;            // RUNNER_CUDA_PROFILE set
    int      inited;        // events allocated
    int      capturing;     // inside graph capture: do not mark
    int      cur_mode;      // M_BATCH or M_SINGLE for the in-flight call
    CUevent  ev[4096];      // per-call event pool
    int      ph[4096];      // phase tag of the group ending at ev[k]
    int      nev;           // events used this call
    double   t_ph[M_N][PH_N];      // accumulated GPU ms per phase
    long long total_launches[M_N]; // exact kernel launches
    long long n_tiles[M_N], n_tokens[M_N];
    double   wall_ms[M_N];
    double   t_stage[M_N], t_copyback[M_N], t_logitscp[M_N], t_sync[M_N];
    double   qpc_hz;
} prof;

static double prof_now(void) {
#ifdef _WIN32
    LARGE_INTEGER c; QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1000.0 / prof.qpc_hz;
#else
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
#endif
}

// newer drivers export the 64-bit entry points under _v2 names; the plain
// names are the legacy 32-bit-era ABI and must not be used
static void *sym2(const char *name) {
    char v2[64];
    snprintf(v2, sizeof(v2), "%s_v2", name);
    void *p = dl_sym(cu.lib, v2);
    return p ? p : dl_sym(cu.lib, name);
}

static bool cu_load(void) {
    if (cu.lib) return true;
    cu.lib = dl_open();
    if (!cu.lib) return false;
    cu.Init              = dl_sym(cu.lib, "cuInit");
    cu.DeviceGetCount    = dl_sym(cu.lib, "cuDeviceGetCount");
    cu.DeviceGet         = dl_sym(cu.lib, "cuDeviceGet");
    cu.DeviceGetName     = dl_sym(cu.lib, "cuDeviceGetName");
    // all three optional: without them gpu_device_id falls back a step at a
    // time, and an unidentifiable GPU still gets a (coarser) registry
    cu.DeviceGetUuid_v2  = dl_sym(cu.lib, "cuDeviceGetUuid_v2");
    cu.DeviceGetUuid     = dl_sym(cu.lib, "cuDeviceGetUuid");
    cu.DeviceGetPCIBusId = dl_sym(cu.lib, "cuDeviceGetPCIBusId");
    cu.PrimaryCtxRetain  = dl_sym(cu.lib, "cuDevicePrimaryCtxRetain");
    cu.PrimaryCtxRelease = sym2("cuDevicePrimaryCtxRelease");
    cu.CtxSetCurrent     = dl_sym(cu.lib, "cuCtxSetCurrent");
    cu.MemGetInfo        = sym2("cuMemGetInfo");
    cu.MemAlloc          = sym2("cuMemAlloc");
    cu.MemFree           = sym2("cuMemFree");
    cu.MemcpyHtoD        = sym2("cuMemcpyHtoD");
    cu.MemcpyDtoH        = sym2("cuMemcpyDtoH");
    cu.MemcpyDtoD        = sym2("cuMemcpyDtoD");
    cu.MemsetD8          = sym2("cuMemsetD8");
    cu.ModuleLoadData    = dl_sym(cu.lib, "cuModuleLoadData");
    cu.ModuleUnload      = dl_sym(cu.lib, "cuModuleUnload");
    cu.ModuleGetFunction = dl_sym(cu.lib, "cuModuleGetFunction");
    cu.LaunchKernel      = dl_sym(cu.lib, "cuLaunchKernel");
    cu.CtxSynchronize    = dl_sym(cu.lib, "cuCtxSynchronize");
    cu.GetErrorString    = dl_sym(cu.lib, "cuGetErrorString");
    // optional graph entry points — absence just disables the graph path
    cu.StreamCreate       = dl_sym(cu.lib, "cuStreamCreate");
    cu.StreamDestroy      = sym2("cuStreamDestroy");
    cu.StreamSynchronize  = dl_sym(cu.lib, "cuStreamSynchronize");
    cu.StreamBeginCapture = sym2("cuStreamBeginCapture");
    cu.StreamEndCapture   = sym2("cuStreamEndCapture");
    cu.GraphInstantiateWithFlags = dl_sym(cu.lib, "cuGraphInstantiateWithFlags");
    cu.GraphLaunch        = dl_sym(cu.lib, "cuGraphLaunch");
    cu.GraphExecDestroy   = dl_sym(cu.lib, "cuGraphExecDestroy");
    cu.GraphDestroy       = dl_sym(cu.lib, "cuGraphDestroy");
    cu.EventCreate        = dl_sym(cu.lib, "cuEventCreate");
    cu.EventRecord        = dl_sym(cu.lib, "cuEventRecord");
    cu.EventSynchronize   = dl_sym(cu.lib, "cuEventSynchronize");
    cu.EventElapsedTime   = dl_sym(cu.lib, "cuEventElapsedTime");
    cu.EventDestroy       = sym2("cuEventDestroy");
    return cu.Init && cu.DeviceGetCount && cu.DeviceGet && cu.DeviceGetName &&
           cu.PrimaryCtxRetain && cu.PrimaryCtxRelease && cu.CtxSetCurrent &&
           cu.MemGetInfo && cu.MemAlloc && cu.MemFree && cu.MemcpyHtoD &&
           cu.MemcpyDtoH && cu.MemcpyDtoD && cu.MemsetD8 && cu.ModuleLoadData && cu.ModuleUnload &&
           cu.ModuleGetFunction && cu.LaunchKernel && cu.CtxSynchronize;
}

static bool cu_graphs_ok(void) {
    return cu.StreamCreate && cu.StreamDestroy && cu.StreamSynchronize &&
           cu.StreamBeginCapture && cu.StreamEndCapture &&
           cu.GraphInstantiateWithFlags && cu.GraphLaunch &&
           cu.GraphExecDestroy && cu.GraphDestroy;
}

static const char *cu_err(CUresult r) {
    const char *s = NULL;
    if (cu.GetErrorString) cu.GetErrorString(r, &s);
    return s ? s : "unknown CUDA error";
}

// ------------------------------------------------------------------ backend

// ---------------------------------------------------- shared model weights
//
// Everything a model puts on the device that a forward pass only ever *reads*.
// It is identical for two model_t values loaded from the same file with the
// same parameters, and it is the expensive part — gigabytes of quantized
// weights — so it is uploaded once and refcounted rather than once per slot.
// Before this, `--parallel 4` on an 8B model spent four times the VRAM on four
// byte-identical copies of the same weights.
//
// The registry is keyed on the identity of the file (path, size, mtime, inode)
// plus the geometry and the derived rope tables, because those are the only
// inputs to any of the buffers below. A key mismatch is not an error, just a
// miss: that instance gets its own upload, exactly as before.
// Microbatch width classes. The batched matvec kernels unroll their column
// loop, so each instantiation has a fixed width and a narrower microbatch
// simply runs the smallest one that covers it. Two classes are enough: the
// step cost is flat within a class, so the only widths that pay for a column
// they are not using are 3 (runs 4-wide) and 5..7 (run 8-wide).
enum { BW_4 = 0, BW_8 = 1, BW_N = 2 };
static inline int batch_width_class(int n) { return n <= 4 ? BW_4 : BW_8; }

typedef struct {
    uint64_t    file_off;
    size_t      nbytes;
    CUdeviceptr device;
} gpu_weight_binding;

typedef struct gpu_weights {
    struct gpu_weights *next;
    int         refs;                   // live gpu_t values pointing here
    // --- identity ---
    char       *path;
    uint64_t    fsize, fino;
    int64_t     fmtime;
    int         n_layer, n_embd, n_head, n_head_kv, head_dim, n_ff, n_vocab;
    int         n_ctx, rope_dim, rope_dim_local, kv_q8, v_rmsnorm;
    float       rope_base, rope_mscale;
    float      *rope_inv_freq, *rope_inv_freq_local;  // owned copies, compared
    // --- device state (immutable once built) ---
    CUcontext   ctx;
    CUdevice    dev;
    CUmodule    mod;
    CUfunction  f_rmsnorm, f_qknorm, f_rope, f_store, f_attn, f_silu, f_gelu,
                f_swiglu, f_xielu, f_add, f_scale;
    CUfunction  f_q35_split, f_q35_gate, f_q35_conv, f_q35_delta;
    CUfunction  f_attn_dec, f_attn_merge;   // flash-decoding attention (decode)
    // kernel tables indexed by ggml tensor type; sized past the largest
    // supported type id (T_MXFP4 = 39), not the count of supported types
    #define KT_N 40
    CUfunction  f_mv[KT_N], f_mvb[KT_N];// indexed by ggml type; _b = tile variant
    CUfunction  f_gemm[KT_N];           // prefill tiled-GEMM variants (Q8_0/Q4_K)
    CUfunction  f_gemm_tc[KT_N];        // tensor-core prefill GEMM (opt-in, Q4_K)
    CUfunction  f_gemv[KT_N];           // decode coalesced GEMV variants (Q8_0/Q4_K)
    // cross-sequence decode microbatch (Phase 6): per-column position and KV
    CUfunction  f_rope_seq, f_store_seq, f_attn_dec_seq;
    // multi-column twins of f_gemv, per microbatch width (see BW_* below):
    // f_gemvb[w][type], w indexing the width class, not the width itself
    CUfunction  f_gemvb[BW_N][KT_N];
    // sparse-MoE device routing + indirect expert matvecs (fused-3D layout)
    CUfunction  f_moe_route, f_moe_actmul, f_moe_sum;
    CUfunction  f_moe_mv[KT_N];         // indexed by expert tensor type
    // expert-grouped prefill (P2): gather/scatter around the k_gemm family
    CUfunction  f_moe_gather, f_moe_scatter, f_moe_actmul_plain, f_moe_copy;
    CUdeviceptr weights;
    size_t      weights_len;
    bool        cpu_moe;                // sparse tensor-role placement mode
    int         cpu_moe_layers;         // requested host-expert count (share key)
    gpu_weight_binding *bindings;       // packed non-expert tensors in this mode
    int         n_bindings, cap_bindings;
    int         gpu_layers;             // split decided by the first loader
    bool        no_id;                  // identity unavailable: listed so the
                                        // split guard can see it, never matched
    CUdeviceptr inv_freq, inv_freq_local, out_norm, dummy;
    CUdeviceptr ones;                   // weightless V RMS norm (gemma4)
    CUdeviceptr *attn_norm, *ffn_norm;  // per layer
    CUdeviceptr *pan, *pfn;             // gemma3 sandwich norms, may be 0
    CUdeviceptr *bq, *bk, *bv, *bo;     // per layer, may be 0
    CUdeviceptr *qn, *kn;               // qwen3 per-head q/k norms
    // gemma-4 MoE dual-branch: dense-post/moe-pre/moe-post norms + the router
    // input scale (with 1/sqrt(n_embd) folded in), per MoE layer, may be 0
    CUdeviceptr *g_pn1, *g_prn2, *g_pn2, *g_gis;
    CUdeviceptr *ple_pn;                // gemma4 E-series per-layer post_norm
    // per-expert down-projection scale table per MoE layer (gemma-4), ones
    // when the GGUF has none; and a shared all-ones table for archs without
    // any such scale — k_moe_actmul multiplies unconditionally
    CUdeviceptr *g_dsc;
    CUdeviceptr moe_ones;
    // gpt-oss per layer, may be 0: attention-sink logits [n_head]; router
    // bias [n_expert]; per-expert gate/up ([n_expert][n_ff_exp]) and down
    // ([n_expert][n_embd]) bias tables
    CUdeviceptr *sinks;
    CUdeviceptr *gib, *geb, *ueb, *deb;
    CUdeviceptr *ssm_dt, *ssm_a, *ssm_norm;
} gpu_weights;

// ------------------------------------------------ per-sequence backend state
//
// One per model_t: the KV cache this stream is decoding into, its activation
// scratch, and its own stream/graph. Nothing here is shared, so one sequence
// failing, resetting, or being unloaded cannot touch another's.
typedef struct {
    gpu_weights *sw;                    // shared weights, refcounted
    CUdeviceptr kc, vc;
    CUdeviceptr x, xb, xb2, q, kt, vt, hb, hb2, att, attn_part, logits;
    // sparse-MoE: router logits [MVB][n_expert], per-slot expert down-out
    // [rows][n_embd] (rows = max(used, MVB); the eager path uses column 0)
    CUdeviceptr moe_logits, moe_eout;
    // device routing results [MVB][used] and per-slot expert hidden scratch
    // [rows][gw] (gw = 2*n_ff_exp for gemma's fused gate_up, n_ff_exp else)
    CUdeviceptr moe_sel, moe_selw, moe_hb, moe_hb2;
    // gemma4 E-series: ple holds [token][layer][n_embd_ple] for the batch,
    // uploaded once per forward; ple_g is the per-layer gate scratch.
    CUdeviceptr ple, ple_g;
    // expert-grouped prefill: gathered inputs / down outputs [MVB][n_embd],
    // concatenated per-expert token lists + weights [MVB*used], and the host
    // staging the grouping is built from
    CUdeviceptr moe_gath, moe_dout, moe_gidx, moe_gw;
    int        *h_moe_sel, *h_moe_gidx;
    float      *h_moe_selw, *h_moe_gw;
    bool        moe_fused_ok;           // every MoE layer runs the indirect path
    bool        moe_grouped_ok;         // expert types have width-classed kernels
    CUdeviceptr g_scr;                  // gemma-4-MoE dual-branch scratch: 5*n_embd
    CUdeviceptr q35_mix, q35_cv, q35_z, q35_beta, q35_alpha, q35_gate;
    CUdeviceptr q35_hist, q35_state;    // persistent recurrent state per sequence
    CUdeviceptr q35_hist_prev, q35_state_prev; // pre-forward rollback snapshot
    float       *h_x, *h_logits, *h_moe_logits; // host staging
    float       *h_ple, *h_ple_tmp;             // E-series pre-pass staging
    int          last_pos;              // -2 = nothing synced yet
    // CUDA graphs for the single-token decode path (Experiment A)
    CUstream     stream;
    CUgraph      graph;
    CUgraphExec  gexec;
    CUdeviceptr  pos_dev;
    bool         graph_bad;
} gpu_t;

// max tokens per matvec tile — must match MVB in kernels.cu; every activation
// buffer is allocated this many columns wide
#define MVB 64   // activation buffer columns (keep in sync with kernels.cu)
#define MVT 16   // scalar-kernel fixed tile (keep in sync with MVT there)
#define TC_N    64   // tensor-core GEMM token tile (keep in sync)
#define Q8_COLS 8    // k_gemm_q8_0 column tile (keep in sync)

// flash-decoding KV split count — must match ATTN_SPLITS in kernels.cu; fixed
// (not context-dependent) so the decode CUDA graph stays valid across positions
#define ATTN_SPLITS 8

// output rows per batched-matvec block — must match GEMM_WARPS in kernels.cu
#define GEMM_WARPS 8
#define TC_ROWS    64  // tensor-core GEMM output rows per block (match kernels.cu)

// indirect expert matvec: weight base = w_off + sel[slot]*estride, slot = grid.y
// l_off is a BYTE offset: the cache is fp16 or q8_0 depending on m->kv_q8,
// so element indexing is not enough (see the kv storage block in kernels.cu)

bool gpu_available(char *name, int cap) {
    if (!cu_load()) return false;
    if (cu.Init(0) != 0) return false;
    int n = 0;
    if (cu.DeviceGetCount(&n) != 0 || n < 1) return false;
    if (name) {
        CUdevice d;
        if (cu.DeviceGet(&d, 0) != 0 || cu.DeviceGetName(name, cap, d) != 0)
            snprintf(name, cap, "CUDA GPU");
    }
    return true;
}

// Stable identity for the VRAM registry. On this project's dev box the GPU is a
// 1g.24gb MIG slice: `nvidia-smi -L` shows one card UUID and a separate
// MIG-<uuid> for the slice, and only the latter distinguishes slices. So the
// _v2 UUID entry point comes first, the v1 one (parent card) second, the PCI
// bus id third, and a bare device index last.
bool gpu_device_id(char *id, int cap) {
    if (!id || cap <= 0) return false;
    if (!cu_load() || cu.Init(0) != 0) return false;
    int n = 0;
    if (cu.DeviceGetCount(&n) != 0 || n < 1) return false;
    CUdevice d;
    if (cu.DeviceGet(&d, 0) != 0) return false;

    unsigned char u[16];
    CUresult r = cu.DeviceGetUuid_v2 ? cu.DeviceGetUuid_v2(u, d)
               : cu.DeviceGetUuid    ? cu.DeviceGetUuid(u, d)
                                     : (CUresult)1;
    if (r == 0) {
        // the canonical 8-4-4-4-12 rendering, so the string matches what
        // nvidia-smi prints and a human can grep the registry filename for it
        snprintf(id, cap,
                 "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-"
                 "%02x%02x%02x%02x%02x%02x",
                 u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7],
                 u[8], u[9], u[10], u[11], u[12], u[13], u[14], u[15]);
        return true;
    }
    char bus[64];
    if (cu.DeviceGetPCIBusId && cu.DeviceGetPCIBusId(bus, sizeof(bus), d) == 0) {
        snprintf(id, cap, "%s", bus);
        return true;
    }
    snprintf(id, cap, "cuda:0");
    return true;
}

bool gpu_mem_info(size_t *free_bytes, size_t *total_bytes) {
    if (!cu_load() || cu.Init(0) != 0) return false;
    int n = 0;
    if (cu.DeviceGetCount(&n) != 0 || n < 1) return false;
    CUdevice dev;
    CUcontext ctx;
    if (cu.DeviceGet(&dev, 0) != 0) return false;
    if (cu.PrimaryCtxRetain(&ctx, dev) != 0) return false;
    size_t f = 0, t = 0;
    bool ok = cu.CtxSetCurrent(ctx) == 0 && cu.MemGetInfo(&f, &t) == 0;
    cu.PrimaryCtxRelease(dev);
    if (!ok) return false;
    if (free_bytes) *free_bytes = f;
    if (total_bytes) *total_bytes = t;
    return true;
}

// The CUDA kernels ship as embedded PTX, not shader source; there is no
// equivalent string to hash here. Reporting nothing beats reporting a hash
// of something else.
const char *gpu_shader_source_sha(void) { return NULL; }

bool gpu_max_working_set(size_t *bytes) {
    // CUDA's fit ceiling is device memory, which vram_bytes already reports
    (void)bytes;
    return false;
}

static bool gpu_type_ok(int type) {
    switch (type) {
        case T_F32: case T_F16: case T_Q8_0: case T_Q4_0: case T_Q4_1:
        case T_Q5_0: case T_Q5_1: case T_Q3_K: case T_Q4_K: case T_Q5_K:
        case T_Q6_K: case T_IQ4_NL: case T_IQ4_XS: case T_MXFP4:
            return true;
        default:
            return false;
    }
}

// The --caps answer for this backend, sourced from the admission test above so
// the advertised list and the loader agree by construction.
bool gpu_quant_ok(int type) { return gpu_type_ok(type); }

// Expert tensor types the indirect MoE matvecs (k_moe_mv_*) cover. A type
// outside this set is not an error — that model keeps the eager per-expert
// path, exactly as before the device-routing work.
static bool moe_indirect_type_ok(int type) {
    switch (type) {
        case T_F32: case T_F16: case T_Q8_0: case T_Q4_0:
        case T_Q4_K: case T_Q5_K: case T_Q6_K: case T_MXFP4:
            return true;
        default:
            return false;
    }
}

// Expert tensor types with WIDTH-CLASSED batched kernels (f_gemvb twins +
// a real GEMM). The grouped prefill only pays when the expert matmul's
// work scales with its token count: the fixed-MVB _b/GEMM kernels compute
// all 16 columns regardless of batch, and at the ~1-2 tokens/expert a
// 16-token tile routes, that waste more than cancels the weight-streaming
// win (measured on the 3070: gemma-4 Q4_0 prefill -19% grouped vs fused).
// Types outside this set keep the per-token fused prefill.
static bool moe_grouped_type_ok(int type) {
    switch (type) {
        case T_Q8_0: case T_Q4_K: case T_Q5_K: case T_Q6_K:
            return true;
        default:
            return false;
    }
}

// Per-layer expert placement under tensor-role mode. Outside that mode every
// expert bank is device-resident, so the array is absent and the answer is no.
// test hook state (definition of the setter is beside gpu_tc_force below)
static int g_moe_eager_force = -1;

static bool moe_on_host(const model_t *m, int l) {
    return m->cpu_moe && m->moe_host && m->moe_host[l];
}

static bool moe_any_on_host(const model_t *m) {
    if (!m->cpu_moe) return false;
    for (int l = 0; l < m->n_layer; l++)
        if (m->layers[l].is_moe && moe_on_host(m, l)) return true;
    return false;
}

// Any expert bank the device is expected to run — the trigger for uploading
// expert tensors, validating their types, and allocating MoE scratch.
static bool moe_any_on_device(const model_t *m) {
    if (m->n_expert <= 0) return false;
    for (int l = 0; l < m->n_layer; l++)
        if (m->layers[l].is_moe && !moe_on_host(m, l)) return true;
    return false;
}

// The fused device-routing MoE path needs the fused-3D expert layout (a
// legacy split model has one arbitrarily-placed tensor per expert, which
// in-kernel base+e*stride addressing cannot express) and indirect kernels
// for every expert tensor type. All-or-nothing per model: one eager layer
// would force the graph off anyway.
static bool moe_fused_eligible(const model_t *m) {
    if (m->n_expert <= 0 || m->n_expert > 256) return false;
    if (!moe_any_on_device(m)) return false;
    for (int l = 0; l < m->n_layer; l++) {
        const layer_t *ly = &m->layers[l];
        if (!ly->is_moe || moe_on_host(m, l)) continue;
        if (ly->moe_split) return false;
        if (ly->moe_gemma) {
            if (!moe_indirect_type_ok(ly->ffn_gate_up_exps->type) ||
                !moe_indirect_type_ok(ly->ffn_down_exps->type)) return false;
        } else {
            if (!moe_indirect_type_ok(ly->ffn_gate_exps->type) ||
                !moe_indirect_type_ok(ly->ffn_up_exps->type) ||
                !moe_indirect_type_ok(ly->ffn_down_exps->type)) return false;
        }
    }
    return true;
}

static bool moe_grouped_eligible(const model_t *m) {
    // Default OFF as of 2026-07-29: at MVB=16 tiles (~1-2 tokens/expert) the
    // grouped path loses to the P1 fused prefill on BOTH boxes — 3070:
    // 55.4 ms vs 33.5 ms GPU-busy per tile; Blackwell MIG full-offload
    // end-to-end: qwen3-30b prefill 125.4 tok/s grouped vs 194.3 fused
    // (TC-on-grouped is worse still, 86.5 — the 16-column TC tile wastes
    // more than it amortizes at these expert token counts). Grouping only
    // pays with many tokens per expert (large-tile / batched serving);
    // keep it reachable for those experiments via RUNNER_MOE_GROUPED=1.
    const char *e = getenv("RUNNER_MOE_GROUPED");
    if (!e || !*e || strcmp(e, "0") == 0 || strcmp(e, "off") == 0)
        return false;
    if (!moe_fused_eligible(m)) return false;
    for (int l = 0; l < m->n_layer; l++) {
        const layer_t *ly = &m->layers[l];
        if (!ly->is_moe) continue;
        // per-expert biases (gpt-oss) are wired through the fused path's
        // actmul/sum only; the grouped scatter has no seat for them, so a
        // biased model keeps the fused prefill (refuse > silently wrong)
        if (ly->ffn_gate_exps_b || ly->ffn_up_exps_b || ly->ffn_down_exps_b)
            return false;
        if (ly->moe_gemma) {
            if (!moe_grouped_type_ok(ly->ffn_gate_up_exps->type) ||
                !moe_grouped_type_ok(ly->ffn_down_exps->type)) return false;
        } else {
            if (!moe_grouped_type_ok(ly->ffn_gate_exps->type) ||
                !moe_grouped_type_ok(ly->ffn_up_exps->type) ||
                !moe_grouped_type_ok(ly->ffn_down_exps->type)) return false;
        }
    }
    return true;
}

static void gpu_ctx_free(model_t *m, gpu_t *g);

#define CK(call) do { CUresult _r = (call); if (_r != 0) { \
    fprintf(stderr, "gpu: %s failed: %s\n", #call, cu_err(_r)); goto fail; } } while (0)

static CUdeviceptr f32_dbuf(const float *src, size_t n) {
    if (!src) return 0;
    if (n > SIZE_MAX / sizeof(float)) return 0;
    CUdeviceptr d = 0;
    if (cu.MemAlloc(&d, n * sizeof(float)) != 0) return 0;
    if (cu.MemcpyHtoD(d, src, n * sizeof(float)) != 0) { cu.MemFree(d); return 0; }
    return d;
}

// As f32_dbuf, but a NULL src uploads an all-ones buffer instead of returning 0.
// The CPU rmsnorm() runs weightless (multiply by 1) when its weight is NULL, so
// an optional gemma norm that a GGUF omits must become ones on the GPU — a null
// device pointer would instead fault k_rmsnorm (no NULL guard in the kernel).
static CUdeviceptr f32_dbuf_ones(const float *src, size_t n) {
    if (src) return f32_dbuf(src, n);
    if (n > SIZE_MAX / sizeof(float)) return 0;
    float *ones = malloc(n * sizeof(float));
    if (!ones) return 0;
    for (size_t i = 0; i < n; i++) ones[i] = 1.0f;
    CUdeviceptr d = f32_dbuf(ones, n);
    free(ones);
    return d;
}

bool gpu_kv_q8_ok(void) {
    return true;    // k_store_kv / k_attn / k_attn_dec all read q8_0 blocks
}

bool gpu_moe_ok(void) {
    return true;    // expert banks upload and route on the device
}

bool gpu_eseries_ok(void) {
    // stage_ple ships the per-layer-embedding pre-pass each forward and enc_ple
    // runs the gate/proj/norm branch per layer; gpu_init has no PLE refusal.
    return true;
}

// ------------------------------------------------- shared-weight registry
//
// Resident uploads, refcounted and keyed on model identity. The lock is held
// across the whole build so two instances racing on the same file cannot both
// pay for the upload — model swap and draft-model loads can run while other
// slots are serving.
static gpu_weights *g_shared;
static pthread_mutex_t g_shared_mu = PTHREAD_MUTEX_INITIALIZER;

// File identity beyond the path: a model rebuilt on disk between two loads
// must not be served out of the previous upload. model_file_identity is
// shared with the host-side registry in model.c so one place decides what a
// file IS, and one place reports being unable to tell -- losing it here also
// costs this instance the first one's CPU/GPU split, not just the upload.
static bool file_id(const char *path, uint64_t *size, uint64_t *ino,
                    int64_t *mtime) {
    return model_file_identity(path, "device weights", size, ino, mtime, NULL);
}

// Everything the shared buffers are derived from, compared rather than assumed
// from the path. The rope tables in particular are not a property of the file
// alone: YaRN auto-extension keys off the requested context, phi3 picks its
// LongRoPE factor set the same way, and --rope-base/--rope-scale override both.
// Two loads of one file can legitimately want different tables, and a miss here
// is not an error — that instance just gets its own upload, as before.
static bool shared_config_matches(const gpu_weights *w, const model_t *m) {
    if (w->n_layer != m->n_layer || w->n_embd != m->n_embd ||
        w->n_head  != m->n_head  || w->n_head_kv != m->n_head_kv ||
        w->head_dim != m->head_dim || w->n_ff != m->n_ff ||
        w->n_vocab != m->n_vocab || w->n_ctx != m->n_ctx ||
        w->rope_dim != m->rope_dim || w->rope_dim_local != m->rope_dim_local ||
        w->kv_q8 != (int)m->kv_q8 || w->v_rmsnorm != (int)m->v_rmsnorm ||
        w->rope_base != m->rope_base || w->rope_mscale != m->rope_mscale ||
        w->cpu_moe != m->cpu_moe ||
        w->cpu_moe_layers != m->cpu_moe_layers)
        return false;
    if ((w->rope_inv_freq_local != NULL) != (m->rope_inv_freq_local != NULL))
        return false;
    if (memcmp(w->rope_inv_freq, m->rope_inv_freq,
               sizeof(float) * (size_t)(m->rope_dim / 2)) != 0)
        return false;
    if (m->rope_inv_freq_local &&
        memcmp(w->rope_inv_freq_local, m->rope_inv_freq_local,
               sizeof(float) * (size_t)(m->rope_dim_local / 2)) != 0)
        return false;
    return true;
}

static bool shared_matches(const gpu_weights *w, const model_t *m,
                           uint64_t size, uint64_t ino, int64_t mtime) {
    // a no-identity entry is in the list only for the split guard below; its
    // zeroed identity fields would already refuse a real file, but be explicit
    if (w->no_id) return false;
    if (!w->path || !m->path || strcmp(w->path, m->path) != 0) return false;
    if (w->fsize != size || w->fino != ino || w->fmtime != mtime) return false;
    return shared_config_matches(w, m);
}

// The safety net for the load path shared_matches cannot protect: an instance
// whose file identity is unavailable builds privately and re-decides its own
// CPU/GPU split — usually under the VRAM pressure of the instance already
// resident, so it decides DIFFERENTLY, and two slots of one server then answer
// the same request with different logits. Detect exactly that: same path, same
// configuration (so the splits had no legitimate reason to differ), identity
// missing on either side, different split. Loud stderr rather than refusal:
// failing this load would fall back to CPU, which diverges from the resident
// GPU instance just as silently.
static void split_guard(const gpu_weights *w, const model_t *m) {
    for (const gpu_weights *o = g_shared; o; o = o->next) {
        if (o == w) continue;
        if (!o->path || !m->path || strcmp(o->path, m->path) != 0) continue;
        if (!(w->no_id || o->no_id)) continue;
        if (!shared_config_matches(o, m)) continue;
        if (o->gpu_layers == w->gpu_layers) continue;
        fprintf(stderr,
                "error: %s re-decided the CPU/GPU split without a file "
                "identity: this instance put %d/%d layers on the GPU but a "
                "resident instance of the same path uses %d/%d — two slots "
                "of one server will answer the same request differently\n",
                m->path, w->gpu_layers, w->n_layer,
                o->gpu_layers, o->n_layer);
    }
}

static void shared_destroy(gpu_weights *w) {
    if (!w) return;
    if (w->ctx) cu.CtxSetCurrent(w->ctx);
    for (int l = 0; l < w->n_layer; l++) {
        CUdeviceptr bufs[] = {
            w->attn_norm ? w->attn_norm[l] : 0,
            w->ffn_norm  ? w->ffn_norm[l]  : 0,
            w->bq ? w->bq[l] : 0, w->bk ? w->bk[l] : 0,
            w->bv ? w->bv[l] : 0, w->bo ? w->bo[l] : 0,
            w->qn ? w->qn[l] : 0, w->kn ? w->kn[l] : 0,
            w->pan ? w->pan[l] : 0, w->pfn ? w->pfn[l] : 0,
            w->g_pn1 ? w->g_pn1[l] : 0, w->g_prn2 ? w->g_prn2[l] : 0,
            w->g_pn2 ? w->g_pn2[l] : 0, w->g_gis ? w->g_gis[l] : 0,
            w->g_dsc ? w->g_dsc[l] : 0,
            w->ssm_dt ? w->ssm_dt[l] : 0, w->ssm_a ? w->ssm_a[l] : 0,
            w->ssm_norm ? w->ssm_norm[l] : 0,
            w->ple_pn ? w->ple_pn[l] : 0,
            w->sinks ? w->sinks[l] : 0,
            w->gib ? w->gib[l] : 0, w->geb ? w->geb[l] : 0,
            w->ueb ? w->ueb[l] : 0, w->deb ? w->deb[l] : 0,
        };
        for (size_t i = 0; i < sizeof(bufs) / sizeof(*bufs); i++)
            if (bufs[i]) cu.MemFree(bufs[i]);
    }
    free(w->attn_norm); free(w->ffn_norm);
    free(w->bq); free(w->bk); free(w->bv); free(w->bo);
    free(w->qn); free(w->kn);
    free(w->pan); free(w->pfn); free(w->ple_pn);
    free(w->g_pn1); free(w->g_prn2); free(w->g_pn2); free(w->g_gis);
    free(w->g_dsc);
    free(w->ssm_dt); free(w->ssm_a); free(w->ssm_norm);
    free(w->sinks);
    free(w->gib); free(w->geb); free(w->ueb); free(w->deb);
    for (int i = 0; i < w->n_bindings; i++)
        if (w->bindings[i].device) cu.MemFree(w->bindings[i].device);
    free(w->bindings);
    CUdeviceptr bufs[] = { w->weights, w->inv_freq, w->inv_freq_local,
                           w->out_norm, w->dummy, w->ones, w->moe_ones };
    for (size_t i = 0; i < sizeof(bufs) / sizeof(*bufs); i++)
        if (bufs[i]) cu.MemFree(bufs[i]);
    if (w->mod) cu.ModuleUnload(w->mod);
    if (w->ctx) cu.PrimaryCtxRelease(w->dev);
    free(w->rope_inv_freq); free(w->rope_inv_freq_local);
    free(w->path);
    free(w);
}

static void shared_release(gpu_weights *w) {
    if (!w) return;
    pthread_mutex_lock(&g_shared_mu);
    bool last = --w->refs <= 0;
    if (last)
        // unlink first so a concurrent acquire cannot find a dying entry; the
        // loop is a no-op for an entry that was never listed (no file identity)
        for (gpu_weights **pp = &g_shared; *pp; pp = &(*pp)->next)
            if (*pp == w) { *pp = w->next; break; }
    pthread_mutex_unlock(&g_shared_mu);
    if (last) shared_destroy(w);
}

// Upload one model's immutable device data and decide the CPU/GPU layer split.
// act_bytes/max_hd describe the *per-sequence* scratch one instance needs; they
// only feed the fitting decision, they are not allocated here.
// Total weight bytes of one layer — attention plus the FFN, handling both a
// dense FFN and a sparse-MoE layer (router + every expert, fused or split).
// The dense w_gate/w_up/w_down are NULL on a MoE layer, so accounting only
// those undercounts a MoE layer by ~all of its weight (the experts).
static size_t layer_weight_bytes(const layer_t *ly, int n_expert,
                                 bool host_experts) {
    size_t wb = 0;
    gguf_tensor *att[] = { ly->wq, ly->wk, ly->wv, ly->wo, ly->wqkv,
                           ly->wq_gate, ly->ssm_conv, ly->ssm_beta,
                           ly->ssm_alpha, ly->ssm_out };
    for (size_t i = 0; i < sizeof(att) / sizeof(*att); i++)
        if (att[i]) wb += att[i]->nbytes;
    if (ly->is_moe && host_experts) {
        // The complete expert FFN (router, routed tensors and a coupled
        // Gemma shared branch) executes on the host. Attention remains here.
    } else if (ly->is_moe) {
        wb += model_layer_expert_bytes(ly, n_expert);
    } else {
        gguf_tensor *ffn[] = { ly->w_gate, ly->w_up, ly->w_down };
        for (int i = 0; i < 3; i++) if (ffn[i]) wb += ffn[i]->nbytes;
    }
    // E-series per-layer embedding matrices ride with the layer. They are f32
    // in the published GGUFs and far from negligible (~5 MB/layer on E4B), so
    // leaving them out would under-budget the offload and overcommit VRAM.
    if (ly->ple_gate) wb += ly->ple_gate->nbytes;
    if (ly->ple_proj) wb += ly->ple_proj->nbytes;
    return wb;
}

static uint64_t tensor_file_off(const model_t *m, const gguf_tensor *t) {
    return (uint64_t)((const uint8_t *)t->data - (const uint8_t *)m->gf.map);
}

// Tensor-role placement cannot preserve mmap offsets in one compact device
// allocation: the skipped experts are the holes. Upload each retained tensor
// as a small binding and resolve matvecs by their stable GGUF file offset.
static bool binding_add(gpu_weights *w, const model_t *m, gguf_tensor *t) {
    if (!t) return true;
    uint64_t off = tensor_file_off(m, t);
    for (int i = 0; i < w->n_bindings; i++)
        if (w->bindings[i].file_off == off && w->bindings[i].nbytes >= t->nbytes)
            return true;
    if (w->n_bindings == w->cap_bindings) {
        int cap = w->cap_bindings ? w->cap_bindings * 2 : 64;
        gpu_weight_binding *nb = realloc(w->bindings, sizeof(*nb) * (size_t)cap);
        if (!nb) return false;
        w->bindings = nb;
        w->cap_bindings = cap;
    }
    CUdeviceptr d = 0;
    if (cu.MemAlloc(&d, t->nbytes) != 0) return false;
    if (cu.MemcpyHtoD(d, t->data, t->nbytes) != 0) {
        cu.MemFree(d);
        return false;
    }
    w->bindings[w->n_bindings++] = (gpu_weight_binding){ off, t->nbytes, d };
    w->weights_len += t->nbytes;
    return true;
}

static bool binding_find(const gpu_weights *w, const model_t *m,
                         const gguf_tensor *t, CUdeviceptr *base,
                         uint64_t *relative_off) {
    uint64_t off = tensor_file_off(m, t);
    if (!w->cpu_moe) {
        *base = w->weights;
        *relative_off = off;
        return off <= w->weights_len && t->nbytes <= w->weights_len - (size_t)off;
    }
    for (int i = 0; i < w->n_bindings; i++) {
        const gpu_weight_binding *b = &w->bindings[i];
        if (off >= b->file_off && off - b->file_off <= b->nbytes &&
            t->nbytes <= b->nbytes - (size_t)(off - b->file_off)) {
            *base = b->device;
            *relative_off = off - b->file_off;
            return true;
        }
    }
    return false;
}

// Highest file offset (data end) touched by any of a layer's weight tensors —
// the partial-offload upload must reach this so a GPU MoE layer's expert slices
// are all inside the uploaded prefix.
static size_t layer_weight_end(const layer_t *ly, int n_expert, const uint8_t *map) {
    size_t end = 0;
#define WEND(t) do { gguf_tensor *_t = (t); if (_t) { \
        size_t _e = (size_t)((uint8_t *)_t->data - map) + _t->nbytes; \
        if (_e > end) end = _e; } } while (0)
    WEND(ly->wq); WEND(ly->wk); WEND(ly->wv); WEND(ly->wo);
    WEND(ly->wqkv); WEND(ly->wq_gate); WEND(ly->ssm_conv);
    WEND(ly->ssm_beta); WEND(ly->ssm_alpha); WEND(ly->ssm_out);
    // gemma-4 E-series per-layer embeddings: real mmap tensors (unlike the
    // f32-converted ple_post_norm), so the partial-offload byte-prefix must
    // reach them too or the upload leaves them unmapped -- see cuda.c:1145
    // where the cpu_moe binding path already treats them as required.
    WEND(ly->ple_gate); WEND(ly->ple_proj);
    if (ly->is_moe) {
        WEND(ly->ffn_gate_inp);
        WEND(ly->ffn_gate_up_exps);
        if (ly->moe_split) {
            for (int e = 0; e < n_expert; e++) {
                WEND(ly->moe_g[e]); WEND(ly->moe_u[e]); WEND(ly->moe_d[e]);
            }
        } else {
            WEND(ly->ffn_gate_exps); WEND(ly->ffn_up_exps); WEND(ly->ffn_down_exps);
        }
        if (ly->moe_gemma) { WEND(ly->w_gate); WEND(ly->w_up); WEND(ly->w_down); }
    } else {
        WEND(ly->w_gate); WEND(ly->w_up); WEND(ly->w_down);
    }
#undef WEND
    return end;
}

static gpu_weights *shared_build(model_t *m, size_t act_bytes, int max_hd,
                                 uint64_t fsize, uint64_t fino, int64_t fmtime) {
    gpu_weights *w = calloc(1, sizeof(gpu_weights));
    if (!w) return NULL;
    if (m->path) {
        size_t n = strlen(m->path) + 1;
        w->path = malloc(n);
        if (!w->path) goto fail_quiet;
        memcpy(w->path, m->path, n);
    }
    w->fsize = fsize; w->fino = fino; w->fmtime = fmtime;
    w->n_layer = m->n_layer; w->n_embd = m->n_embd; w->n_head = m->n_head;
    w->n_head_kv = m->n_head_kv; w->head_dim = m->head_dim; w->n_ff = m->n_ff;
    w->n_vocab = m->n_vocab; w->n_ctx = m->n_ctx;
    w->cpu_moe = m->cpu_moe;
    w->cpu_moe_layers = m->cpu_moe_layers;
    w->rope_dim = m->rope_dim; w->rope_dim_local = m->rope_dim_local;
    w->kv_q8 = (int)m->kv_q8; w->v_rmsnorm = (int)m->v_rmsnorm;
    w->rope_base = m->rope_base; w->rope_mscale = m->rope_mscale;
    {   // own copies of the rope tables so a later match can compare against
        // them without reaching into a model_t that may since have been freed
        size_t nb = sizeof(float) * (size_t)(m->rope_dim / 2);
        w->rope_inv_freq = malloc(nb);
        if (!w->rope_inv_freq) goto fail_quiet;
        memcpy(w->rope_inv_freq, m->rope_inv_freq, nb);
        if (m->rope_inv_freq_local) {
            size_t lb = sizeof(float) * (size_t)(m->rope_dim_local / 2);
            w->rope_inv_freq_local = malloc(lb);
            if (!w->rope_inv_freq_local) goto fail_quiet;
            memcpy(w->rope_inv_freq_local, m->rope_inv_freq_local, lb);
        }
    }

    CK(cu.DeviceGet(&w->dev, 0));
    CK(cu.PrimaryCtxRetain(&w->ctx, w->dev));
    CK(cu.CtxSetCurrent(w->ctx));

    {
        size_t vram_free = 0, vram_total = 0;
        CK(cu.MemGetInfo(&vram_free, &vram_total));
        size_t vram_budget = vram_free;
        if (m->reserve_vram_pct > 0) {
            size_t cap = vram_total / 100 * m->reserve_vram_pct;
            if (cap < vram_budget) vram_budget = cap;
        }
        // fixed device overhead regardless of split: activation scratch, the
        // token-embedding weights (always uploaded), and a margin covering the
        // CUDA context + PTX JIT + WDDM reserve
        size_t fixed = act_bytes + (m->cpu_moe ? 0 : m->tok_embd->nbytes) +
                       (512u << 20);
        // decide how many *leading* layers fit — accumulate each layer's weight
        // bytes plus its KV bytes until the budget runs out; the CPU runs the
        // rest (partial offload)
        int G = 0;
        size_t used = fixed;
        for (int l = 0; l < m->n_layer; l++) {
            layer_t *ly = &m->layers[l];
            size_t wb = layer_weight_bytes(ly, m->n_expert, moe_on_host(m, l));
            // KV bytes honour the cache format: a q8_0 cache is ~53% of fp16,
            // so quantized KV directly buys more offloaded layers here
            size_t kv = 2 * (size_t)m->n_ctx * model_kv_row_bytes(m, l);
            if (used + wb + kv > vram_budget) break;
            used += wb + kv;
            G = l + 1;
        }
        // Partial expert offload: the pass above placed attention (and any
        // dense FFN) for the layers that fit. Whatever budget survives is
        // spent on whole expert banks, shallowest first, so a card with room
        // to spare stops idling it — the all-or-nothing flag left 8.8 of 12 GB
        // unused on the first outside install. Explicit counts are honoured as
        // given and were already accounted above.
        if (m->cpu_moe && m->cpu_moe_layers == CPU_MOE_AUTO) {
            // Hold back what a full split still owes before spending the rest
            // on expert banks. `full` below decides whether output.weight is
            // uploaded, but the forward decides whether to compute logits on
            // the device from the LAYER COUNT alone (fwd: gpu_layers <
            // n_layer). Letting a bank eat the last of the budget makes those
            // two disagree: every layer is on the device, so the forward asks
            // for logits, and output.weight was never uploaded. That is the
            // whole of the "--cpu-moe auto dies mid-forward" bug — it was not
            // a VRAM over-commit, and it reproduced on every MoE model.
            // Reserving here is also the better trade: the output projection
            // runs every token over the full vocabulary, an expert bank only
            // when its layer is routed to.
            size_t owed = G == m->n_layer
                        ? (m->cpu_moe ? 0 : m->tok_embd->nbytes) + m->output->nbytes
                        : 0;
            for (int l = 0; l < G; l++) {
                layer_t *ly = &m->layers[l];
                if (!ly->is_moe) continue;
                size_t eb = model_layer_expert_bytes(ly, m->n_expert);
                if (used + eb + owed > vram_budget) break;
                used += eb;
                m->moe_host[l] = false;
            }
        }
        // a full split also needs token_embd + output weights resident
        bool full = G == m->n_layer &&
                    used + (m->cpu_moe ? 0 : m->tok_embd->nbytes) +
                    m->output->nbytes <= vram_budget;
        // Invariant, belt and braces: "the device owns every layer" and "the
        // device owns the output projection" must agree, because the forward
        // reads only the first and the upload honours only the second. If the
        // reserve above could not be met, hand the last layer back to the CPU
        // so the split is honestly partial rather than internally inconsistent.
        if (G == m->n_layer && !full && G > 1) {
            G--;
            fprintf(stderr, "gpu: output weights do not fit alongside the "
                            "layer split — running the last layer on the CPU "
                            "so the boundary is consistent\n");
        }
        // an explicit --gpu-layers overrides the budget-based fit (the user
        // takes responsibility for VRAM; used for testing partial offload and
        // for manual control). It can only lower the count below what fits, or
        // request a specific split.
        int natural_G = G;
        if (m->gpu_layers_override > 0) {
            G = m->gpu_layers_override;
            if (G > m->n_layer) G = m->n_layer;
            full = (G == m->n_layer);
        }
        // Composition warning (no silent worst-of-both). Under tensor-role
        // placement the expert banks are what fill VRAM, so capping the
        // attention split below what already fits moves attention to the CPU
        // and frees almost nothing: measured 10.3 tok/s for
        // `--cpu-moe --gpu-layers 26` against 12.7 (all-host) and 14.6
        // (layer split alone) on a 12 GB card. Say so instead of obeying
        // quietly — the flags stay legal, since the pair is meaningful once a
        // partial expert count is what is being reserved for.
        if (m->cpu_moe && m->gpu_layers_override > 0 && G < natural_G)
            fprintf(stderr,
                    "gpu: --gpu-layers %d caps the attention split below the "
                    "%d layers that fit; under --cpu-moe that mostly moves "
                    "attention to the CPU without freeing useful VRAM. Prefer "
                    "--cpu-moe N (or auto) to spend the headroom on expert "
                    "banks.\n", G, natural_G);
        // The discovery the first outside install could not make: all-host
        // placement left 8.8 GB of a 12 GB card idle and nothing said so.
        if (m->cpu_moe && m->cpu_moe_layers == CPU_MOE_ALL) {
            // Count what the auto fit would actually place, by the same greedy
            // rule, so the advice cannot promise more than it can deliver.
            size_t would_use = used;
            int would_place = 0, moe_layers = 0;
            for (int l = 0; l < G; l++) {
                if (!m->layers[l].is_moe) continue;
                moe_layers++;
                size_t eb = model_layer_expert_bytes(&m->layers[l], m->n_expert);
                if (would_use + eb > vram_budget) continue;
                would_use += eb;
                would_place++;
            }
            if (would_place > 0)
                fprintf(stderr,
                        "gpu: %.2f GB of the budget is unused — `--cpu-moe "
                        "auto` would keep %d of %d expert layers on the GPU.\n",
                        (vram_budget - used) / 1e9, would_place, moe_layers);
        }
        if (G == 0) {
            fprintf(stderr,
                    "gpu: not even one layer fits %.1f GB %s VRAM — using CPU\n",
                    vram_budget / 1e9, m->reserve_vram_pct > 0 ? "reserved" : "free");
            goto fail_quiet;
        }
        int moe_dev = 0, moe_tot = 0;
        for (int l = 0; l < m->n_layer; l++) {
            if (!m->layers[l].is_moe) continue;
            moe_tot++;
            if (!moe_on_host(m, l)) moe_dev++;
        }
        fprintf(stderr, "gpu-split: budget=%.2fGB fixed=%.2fGB G=%d/%d full=%d used=%.2fGB\n", vram_budget/1e9, fixed/1e9, G, m->n_layer, (int)full, used/1e9);
        if (m->cpu_moe && moe_tot)
            fprintf(stderr, "gpu-split: experts %d/%d layers on GPU, %d on host%s\n",
                    moe_dev, moe_tot, moe_tot - moe_dev,
                    m->cpu_moe_layers == CPU_MOE_AUTO ? " (auto fit)" : "");
        w->gpu_layers = full ? m->n_layer : G;
        // weight bytes to upload: whole file for a full split (output/embedding
        // offsets stay valid), else the prefix covering token_embd + [0, G)
        size_t upload_len = m->gf.map_size;
        if (!full) {
            upload_len = (size_t)((uint8_t *)m->tok_embd->data - (uint8_t *)m->gf.map)
                       + m->tok_embd->nbytes;
            for (int l = 0; l < G; l++) {
                size_t end = layer_weight_end(&m->layers[l], m->n_expert,
                                              (const uint8_t *)m->gf.map);
                if (end > upload_len) upload_len = end;
            }
        }
        w->weights_len = m->cpu_moe ? 0 : upload_len;

        CK(cu.ModuleLoadData(&w->mod, k_ptx_src));
        struct { CUfunction *f; const char *name; } fns[] = {
            { &w->f_rmsnorm,    "k_rmsnorm" },   { &w->f_qknorm, "k_qknorm" },
            { &w->f_rope,       "k_rope" },      { &w->f_store,  "k_store_kv" },
            { &w->f_attn,       "k_attn" },      { &w->f_silu,   "k_silu_mul" },
            { &w->f_attn_dec,   "k_attn_dec" },  { &w->f_attn_merge, "k_attn_merge" },
            { &w->f_gelu,       "k_gelu_mul" },  { &w->f_add,    "k_add" },
            { &w->f_swiglu,     "k_swiglu_oai_mul" },
            { &w->f_xielu,      "k_xielu" },
            { &w->f_scale,      "k_scale" },
            { &w->f_q35_split,  "k_q35_split_q" },
            { &w->f_q35_gate,   "k_q35_attn_gate" },
            { &w->f_q35_conv,   "k_q35_conv" },
            { &w->f_q35_delta,  "k_q35_delta" },
            { &w->f_mv[T_F32],  "k_mv_f32" },    { &w->f_mv[T_F16],  "k_mv_f16" },
            { &w->f_mv[T_Q8_0], "k_mv_q8_0" },   { &w->f_mv[T_Q4_0], "k_mv_q4_0" },
            { &w->f_mv[T_Q4_1], "k_mv_q4_1" },   { &w->f_mv[T_Q5_0], "k_mv_q5_0" },
            { &w->f_mv[T_Q5_1], "k_mv_q5_1" },   { &w->f_mv[T_Q4_K], "k_mv_q4_K" },
            { &w->f_mv[T_Q5_K], "k_mv_q5_K" },   { &w->f_mv[T_Q6_K], "k_mv_q6_K" },
            { &w->f_mv[T_Q3_K], "k_mv_q3_K" },
            { &w->f_mv[T_IQ4_NL], "k_mv_iq4_nl" }, { &w->f_mv[T_IQ4_XS], "k_mv_iq4_xs" },
            { &w->f_mv[T_MXFP4], "k_mv_mxfp4" },
            { &w->f_mvb[T_F32],  "k_mv_f32_b" },  { &w->f_mvb[T_F16],  "k_mv_f16_b" },
            { &w->f_mvb[T_Q8_0], "k_mv_q8_0_b" }, { &w->f_mvb[T_Q4_0], "k_mv_q4_0_b" },
            { &w->f_mvb[T_Q4_1], "k_mv_q4_1_b" }, { &w->f_mvb[T_Q5_0], "k_mv_q5_0_b" },
            { &w->f_mvb[T_Q5_1], "k_mv_q5_1_b" }, { &w->f_mvb[T_Q4_K], "k_mv_q4_K_b" },
            { &w->f_mvb[T_Q5_K], "k_mv_q5_K_b" }, { &w->f_mvb[T_Q6_K], "k_mv_q6_K_b" },
            { &w->f_mvb[T_Q3_K], "k_mv_q3_K_b" },
            { &w->f_mvb[T_IQ4_NL], "k_mv_iq4_nl_b" }, { &w->f_mvb[T_IQ4_XS], "k_mv_iq4_xs_b" },
            { &w->f_mvb[T_MXFP4], "k_mv_mxfp4_b" },
            // prefill tiled-GEMM variants (batch>1 fast path for these formats)
            { &w->f_gemm[T_Q8_0], "k_gemm_q8_0" },
            { &w->f_gemm[T_Q4_K], "k_gemm_q4_K" },
            { &w->f_gemm[T_Q5_K], "k_gemm_q5_K" },
            { &w->f_gemm[T_Q6_K], "k_gemm_q6_K" },
            // decode coalesced GEMV variants (batch==1 fast path for these formats)
            { &w->f_gemv[T_Q8_0], "k_gemv_q8_0" },
            { &w->f_gemv[T_Q4_0], "k_gemv_q4_0" },
            { &w->f_gemv[T_Q4_K], "k_gemv_q4_K" },
            { &w->f_gemv[T_Q5_K], "k_gemv_q5_K" },
            { &w->f_gemv[T_Q6_K], "k_gemv_q6_K" },
            // cross-sequence decode microbatch: per-column position and KV,
            // plus the multi-column twins of the decode GEMVs above
            { &w->f_rope_seq,      "k_rope_seq" },
            { &w->f_store_seq,     "k_store_kv_seq" },
            { &w->f_attn_dec_seq,  "k_attn_dec_seq" },
            // twins of f_gemv: same per-column arithmetic, x staged in smem,
            // one instantiation per microbatch width class
            { &w->f_gemvb[BW_4][T_Q8_0], "k_gemvb_q8_0_x4" },
            { &w->f_gemvb[BW_4][T_Q4_K], "k_gemvb_q4_K_x4" },
            { &w->f_gemvb[BW_4][T_Q5_K], "k_gemvb_q5_K_x4" },
            { &w->f_gemvb[BW_4][T_Q6_K], "k_gemvb_q6_K_x4" },
            { &w->f_gemvb[BW_8][T_Q8_0], "k_gemvb_q8_0_x8" },
            { &w->f_gemvb[BW_8][T_Q4_K], "k_gemvb_q4_K_x8" },
            { &w->f_gemvb[BW_8][T_Q5_K], "k_gemvb_q5_K_x8" },
            { &w->f_gemvb[BW_8][T_Q6_K], "k_gemvb_q6_K_x8" },
            // sparse-MoE device routing + indirect expert matvecs
            { &w->f_moe_route,        "k_moe_route" },
            { &w->f_moe_actmul,       "k_moe_actmul" },
            { &w->f_moe_sum,          "k_moe_sum" },
            { &w->f_moe_mv[T_F32],    "k_moe_mv_f32" },
            { &w->f_moe_mv[T_F16],    "k_moe_mv_f16" },
            { &w->f_moe_mv[T_Q8_0],   "k_moe_mv_q8_0" },
            { &w->f_moe_mv[T_Q4_0],   "k_moe_mv_q4_0" },
            { &w->f_moe_mv[T_Q4_K],   "k_moe_mv_q4_K" },
            { &w->f_moe_mv[T_Q5_K],   "k_moe_mv_q5_K" },
            { &w->f_moe_mv[T_Q6_K],   "k_moe_mv_q6_K" },
            { &w->f_moe_mv[T_MXFP4],  "k_moe_mv_mxfp4" },
            // expert-grouped prefill glue
            { &w->f_moe_gather,       "k_moe_gather" },
            { &w->f_moe_scatter,      "k_moe_scatter_add" },
            { &w->f_moe_actmul_plain, "k_moe_actmul_plain" },
            { &w->f_moe_copy,         "k_moe_copy_cols" },
        };
        for (size_t i = 0; i < sizeof(fns) / sizeof(*fns); i++)
            CK(cu.ModuleGetFunction(fns[i].f, w->mod, fns[i].name));
        // Tensor-core kernels: resolved non-fatally so an older embedded PTX
        // (without them) still loads — TC just stays unavailable there.
        cu.ModuleGetFunction(&w->f_gemm_tc[T_Q4_K], w->mod, "k_gemm_q4_K_tc");
        cu.ModuleGetFunction(&w->f_gemm_tc[T_Q6_K], w->mod, "k_gemm_q6_K_tc");
        cu.ModuleGetFunction(&w->f_gemm_tc[T_Q8_0], w->mod, "k_gemm_q8_0_tc");
        cu.ModuleGetFunction(&w->f_gemm_tc[T_Q4_0], w->mod, "k_gemm_q4_0_tc");

        // weights: the file bytes the offloaded layers reference (whole file
        // for a full split, a prefix for partial) so byte offsets stay valid.
        // Every instance sharing this upload indexes it with offsets computed
        // against its own mmap base, which is the same layout by construction.
        if (m->cpu_moe) {
            // Pack only tensors used by the CUDA half. Every MoE FFN stays on
            // the host; attention and dense layers remain device resident.
            for (int l = 0; l < G; l++) {
                layer_t *ly = &m->layers[l];
                if (!binding_add(w, m, ly->ple_gate) ||
                    !binding_add(w, m, ly->ple_proj) ||
                    !binding_add(w, m, ly->wq) ||
                    !binding_add(w, m, ly->wk) ||
                    !binding_add(w, m, ly->wv) ||
                    !binding_add(w, m, ly->wo) ||
                    !binding_add(w, m, ly->wqkv) ||
                    !binding_add(w, m, ly->wq_gate) ||
                    !binding_add(w, m, ly->ssm_conv) ||
                    !binding_add(w, m, ly->ssm_beta) ||
                    !binding_add(w, m, ly->ssm_alpha) ||
                    !binding_add(w, m, ly->ssm_out)) goto fail;
                if (!ly->is_moe &&
                    (!binding_add(w, m, ly->w_gate) ||
                     !binding_add(w, m, ly->w_up) ||
                     !binding_add(w, m, ly->w_down))) goto fail;
                // Partial expert offload: a device-resident bank uploads its
                // router and expert tensors as ordinary bindings, so the same
                // offset-resolved matvec path serves them.
                if (ly->is_moe && !moe_on_host(m, l)) {
                    if (!binding_add(w, m, ly->ffn_gate_inp) ||
                        !binding_add(w, m, ly->ffn_gate_up_exps) ||
                        !binding_add(w, m, ly->ffn_gate_exps) ||
                        !binding_add(w, m, ly->ffn_up_exps) ||
                        !binding_add(w, m, ly->ffn_down_exps)) goto fail;
                    if (ly->moe_split)
                        for (int e = 0; e < m->n_expert; e++)
                            if (!binding_add(w, m, ly->moe_g[e]) ||
                                !binding_add(w, m, ly->moe_u[e]) ||
                                !binding_add(w, m, ly->moe_d[e])) goto fail;
                    if (ly->moe_gemma &&
                        (!binding_add(w, m, ly->w_gate) ||
                         !binding_add(w, m, ly->w_up) ||
                         !binding_add(w, m, ly->w_down))) goto fail;
                }
            }
            if (full && !binding_add(w, m, m->output)) goto fail;
        } else {
            CK(cu.MemAlloc(&w->weights, w->weights_len));
            CK(cu.MemcpyHtoD(w->weights, m->gf.map, w->weights_len));
        }
        CK(cu.MemAlloc(&w->dummy, 4));

        w->inv_freq = f32_dbuf(m->rope_inv_freq, m->rope_dim / 2);
        w->inv_freq_local = f32_dbuf(m->rope_inv_freq_local, m->rope_dim_local / 2);
        w->out_norm = f32_dbuf(m->out_norm_w, m->n_embd);
        if (!w->inv_freq || !w->out_norm ||
            (m->rope_inv_freq_local && !w->inv_freq_local)) goto fail;
        if (m->v_rmsnorm) { // weightless per-head V norm: weight of ones
            float *ones = malloc(sizeof(float) * max_hd);
            if (!ones) goto fail;
            for (int i = 0; i < max_hd; i++) ones[i] = 1.0f;
            w->ones = f32_dbuf(ones, max_hd);
            free(ones);
            if (!w->ones) goto fail;
        }
        w->attn_norm = calloc(m->n_layer, sizeof(CUdeviceptr));
        w->ffn_norm  = calloc(m->n_layer, sizeof(CUdeviceptr));
        w->bq = calloc(m->n_layer, sizeof(CUdeviceptr));
        w->bk = calloc(m->n_layer, sizeof(CUdeviceptr));
        w->bv = calloc(m->n_layer, sizeof(CUdeviceptr));
        w->bo = calloc(m->n_layer, sizeof(CUdeviceptr));
        w->qn = calloc(m->n_layer, sizeof(CUdeviceptr));
        w->kn = calloc(m->n_layer, sizeof(CUdeviceptr));
        w->pan = calloc(m->n_layer, sizeof(CUdeviceptr));
        w->pfn = calloc(m->n_layer, sizeof(CUdeviceptr));
        w->ple_pn = calloc(m->n_layer, sizeof(CUdeviceptr));
        w->g_pn1  = calloc(m->n_layer, sizeof(CUdeviceptr));
        w->g_prn2 = calloc(m->n_layer, sizeof(CUdeviceptr));
        w->g_pn2  = calloc(m->n_layer, sizeof(CUdeviceptr));
        w->g_gis  = calloc(m->n_layer, sizeof(CUdeviceptr));
        w->g_dsc  = calloc(m->n_layer, sizeof(CUdeviceptr));
        if (!w->ple_pn) goto fail;
        w->ssm_dt   = calloc(m->n_layer, sizeof(CUdeviceptr));
        w->ssm_a    = calloc(m->n_layer, sizeof(CUdeviceptr));
        w->ssm_norm = calloc(m->n_layer, sizeof(CUdeviceptr));
        w->sinks    = calloc(m->n_layer, sizeof(CUdeviceptr));
        w->gib      = calloc(m->n_layer, sizeof(CUdeviceptr));
        w->geb      = calloc(m->n_layer, sizeof(CUdeviceptr));
        w->ueb      = calloc(m->n_layer, sizeof(CUdeviceptr));
        w->deb      = calloc(m->n_layer, sizeof(CUdeviceptr));
        if (!w->attn_norm || !w->ffn_norm || !w->bq || !w->bk || !w->bv ||
            !w->bo || !w->qn || !w->kn || !w->pan || !w->pfn ||
            !w->g_pn1 || !w->g_prn2 || !w->g_pn2 || !w->g_gis || !w->g_dsc ||
            !w->ssm_dt || !w->ssm_a || !w->ssm_norm ||
            !w->sinks || !w->gib || !w->geb || !w->ueb || !w->deb)
            goto fail;
        if (moe_any_on_device(m)) {
            // all-ones per-expert scale table for archs without a down-
            // projection scale — k_moe_actmul multiplies unconditionally
            w->moe_ones = f32_dbuf_ones(NULL, (size_t)m->n_expert);
            if (!w->moe_ones) goto fail;
        }
        for (int l = 0; l < m->n_layer; l++) {
            layer_t *ly = &m->layers[l];
            w->attn_norm[l] = f32_dbuf(ly->attn_norm_w, m->n_embd);
            w->ffn_norm[l]  = f32_dbuf(ly->ffn_norm_w, m->n_embd);
            w->bq[l] = f32_dbuf(ly->bq, model_q_dim(m, l));
            w->bk[l] = f32_dbuf(ly->bk, model_kv_dim(m, l));
            w->bv[l] = f32_dbuf(ly->bv, model_kv_dim(m, l));
            w->bo[l] = f32_dbuf(ly->bo, m->n_embd);
            w->qn[l] = f32_dbuf(ly->qnorm_w, model_head_dim(m, l));
            w->kn[l] = f32_dbuf(ly->knorm_w, model_head_dim(m, l));
            w->pan[l] = f32_dbuf(ly->post_attn_norm_w, m->n_embd);
            w->pfn[l] = f32_dbuf(ly->post_ffn_norm_w, m->n_embd);
            if (ly->ple_post_norm)
                w->ple_pn[l] = f32_dbuf(ly->ple_post_norm, m->n_embd);
            if (ly->recurrent) {
                w->ssm_dt[l] = f32_dbuf(ly->ssm_dt, m->ssm_v_heads);
                w->ssm_a[l] = f32_dbuf(ly->ssm_a, m->ssm_v_heads);
                w->ssm_norm[l] = f32_dbuf(ly->ssm_norm_w, m->ssm_state);
            }
            // gpt-oss: per-head sink logits + router/per-expert bias tables
            w->sinks[l] = f32_dbuf(ly->attn_sinks, m->n_head);
            if (ly->is_moe && !moe_on_host(m, l)) {
                w->gib[l] = f32_dbuf(ly->ffn_gate_inp_b, m->n_expert);
                w->geb[l] = f32_dbuf(ly->ffn_gate_exps_b,
                                     (size_t)m->n_expert * m->n_ff_exp);
                w->ueb[l] = f32_dbuf(ly->ffn_up_exps_b,
                                     (size_t)m->n_expert * m->n_ff_exp);
                w->deb[l] = f32_dbuf(ly->ffn_down_exps_b,
                                     (size_t)m->n_expert * m->n_embd);
            }
            if (ly->moe_gemma) {
                // gemma-4 dual-branch norms + router scale. All are optional in
                // the GGUF; the CPU path treats an absent norm as weightless and
                // an absent gate_inp_scale as 1.0, so mirror that here (ones /
                // 1.0-fold) rather than leave a null device ptr — k_rmsnorm has
                // no NULL guard and would fault on device pointer 0.
                w->g_pn1[l]  = f32_dbuf_ones(ly->ffn_post_norm1_w, m->n_embd);
                w->g_prn2[l] = f32_dbuf_ones(ly->ffn_pre_norm2_w,  m->n_embd);
                w->g_pn2[l]  = f32_dbuf_ones(ly->ffn_post_norm2_w, m->n_embd);
                // fold the router's 1/sqrt(n_embd) into the uploaded scale so a
                // plain weighted rmsnorm reproduces the CPU router input exactly
                float inv = 1.0f / sqrtf((float)m->n_embd);
                float *gs = malloc(sizeof(float) * m->n_embd);
                if (!gs) goto fail;
                for (int i = 0; i < m->n_embd; i++)
                    gs[i] = (ly->gate_inp_scale ? ly->gate_inp_scale[i] : 1.0f) * inv;
                w->g_gis[l] = f32_dbuf(gs, m->n_embd);
                free(gs);
                // per-expert down scale for the indirect expert path (ones
                // when the GGUF has none — same 1.0 fold the CPU applies)
                w->g_dsc[l] = f32_dbuf_ones(ly->down_exps_scale, m->n_expert);
            }
            bool moe_dev = ly->is_moe && !moe_on_host(m, l);
            if (!w->attn_norm[l] || !w->ffn_norm[l] ||
                (ly->bq && !w->bq[l]) || (ly->bk && !w->bk[l]) ||
                (ly->bv && !w->bv[l]) || (ly->bo && !w->bo[l]) ||
                (ly->qnorm_w && !w->qn[l]) || (ly->knorm_w && !w->kn[l]) ||
                (ly->post_attn_norm_w && !w->pan[l]) ||
                (ly->post_ffn_norm_w && !w->pfn[l]) ||
                (ly->ple_post_norm && !w->ple_pn[l]) ||
                (ly->attn_sinks && !w->sinks[l]) ||
                (moe_dev && ly->ffn_gate_inp_b  && !w->gib[l]) ||
                (moe_dev && ly->ffn_gate_exps_b && !w->geb[l]) ||
                (moe_dev && ly->ffn_up_exps_b   && !w->ueb[l]) ||
                (moe_dev && ly->ffn_down_exps_b && !w->deb[l]) ||
                (ly->recurrent && (!w->ssm_dt[l] || !w->ssm_a[l] ||
                                    !w->ssm_norm[l])) ||
                (ly->moe_gemma && (!w->g_pn1[l] || !w->g_prn2[l] ||
                                   !w->g_pn2[l] || !w->g_gis[l] ||
                                   !w->g_dsc[l])))
                goto fail;
        }
    }
    w->refs = 1;
    return w;

fail:
fail_quiet:
    shared_destroy(w);
    return NULL;
}

static gpu_weights *shared_acquire(model_t *m, size_t act_bytes, int max_hd) {
    uint64_t fsize = 0, fino = 0;
    int64_t  fmtime = 0;
    bool have_id = m->gf.n_maps <= 1 &&
                   file_id(m->path, &fsize, &fino, &fmtime);
    gpu_weights *w = NULL;
    pthread_mutex_lock(&g_shared_mu);
    if (have_id)
        for (w = g_shared; w; w = w->next)
            if (shared_matches(w, m, fsize, fino, fmtime)) break;
    if (w) {
        w->refs++;
        fprintf(stderr, "gpu: reusing resident weights (%.1f GB, now shared by "
                "%d instances)\n", w->weights_len / 1e9, w->refs);
    } else {
        w = shared_build(m, act_bytes, max_hd, fsize, fino, fmtime);
        // an entry with no file identity stays private — shared_matches never
        // returns it — but it is listed anyway so the split guard can compare
        // splits across it; shared_release unlinks it normally at refs 0
        if (w) {
            w->no_id = !have_id;
            split_guard(w, m);
            w->next = g_shared;
            g_shared = w;
        }
    }
    pthread_mutex_unlock(&g_shared_mu);
    return w;
}

bool gpu_init(model_t *m) {
    // every weight matmul must have a kernel for its quant type (wv may be
    // absent: gemma4 global layers reuse the raw K projection as V).
    // gpt-oss runs here since the sink-aware attention softmax, the MXFP4
    // kernels, swiglu_oai and the router/expert bias plumbing landed
    // (2026-08-01); the old refusal is gone.
    // Generalized MoE routing has no device kernel: k_moe_route is softmax +
    // top-k + renormalize with no bias input, and regenerating kernels_ptx.h
    // is the CUDA-13.3 machine's job. Refuse the backend rather than route
    // with the wrong function and produce confident, wrong output.
    if (m->n_expert > 0 && !model_moe_router_is_plain(m)) {
        fprintf(stderr, "gpu: this model's MoE router (gating=%d groups=%d "
                "norm_w=%d scale=%g) has no device kernel — running on CPU\n",
                m->expert_gating, m->n_expert_groups, (int)m->expert_norm_w,
                (double)m->expert_w_scale);
        goto unsupported;
    }
    for (int l = 0; l < m->n_layer && m->n_expert > 0; l++)
        if (m->layers[l].exp_probs_b) {
            fprintf(stderr, "gpu: MoE selection bias (exp_probs_b) has no "
                    "device kernel — running on CPU\n");
            goto unsupported;
        }
    // Every device MoE path (weight upload sizing, k_moe_route's expert
    // count, the gather/scatter kernels) assumes one uniform n_expert for
    // the whole model. --prune-experts can write a SMALLER expert count for
    // individual layers (model.c reads it per layer into layer_t.n_expert);
    // a model with any such layer would upload the wrong slice or route
    // against the wrong count on-device — refuse rather than risk that.
    for (int l = 0; l < m->n_layer; l++)
        if (m->layers[l].is_moe && m->layers[l].n_expert != m->n_expert) {
            fprintf(stderr, "gpu: blk.%d has %d experts, model declares %d "
                    "(pruned model) — no device kernel assumes non-uniform "
                    "per-layer expert counts, running on CPU\n",
                    l, m->layers[l].n_expert, m->n_expert);
            goto unsupported;
        }
    // The shared always-on expert has no device path: the routed MoE
    // kernels write the FFN output and nothing adds a second dense branch.
    if (m->n_ff_shexp > 0) {
        fprintf(stderr, "gpu: shared-expert MoE has no device path — "
                "running on CPU\n");
        goto unsupported;
    }
    // The DENSE attention output gate (muse-glimmer) rides qwen35's
    // machinery: wq_gate is already in every weight table and k_q35_attn_gate
    // applies exactly the CPU's x *= 1/(1+expf(-g)); only the projection into
    // q35_gate is arch-specific, added in the layer loop below. The gate
    // PLUS sparse MoE (afmoe) stays on the host: that combination has never
    // been run against the identity gates, and enabling it as a side effect
    // of the dense port would be an unmeasured support claim.
    if (m->attn_out_gate && m->n_expert > 0) {
        // afmoe: per-element attention output gate + sparse MoE has no
        // certified device path; running it would be unverified, so the
        // whole model stays on the host.
        goto unsupported;
    }
    if (m->ffn_var) {
        // gemma-4 E2B: per-layer FFN widths differ, and the device FFN path
        // sizes off one global width — computing with it would be silently
        // wrong on every narrow layer.
        fprintf(stderr, "gpu: per-layer FFN widths have no device path — "
                "running on CPU\n");
        goto unsupported;
    }
    if (!gpu_type_ok(m->output->type)) goto unsupported;
    for (int l = 0; l < m->n_layer; l++) {
        layer_t *ly = &m->layers[l];
        gguf_tensor *ws[] = { ly->wq, ly->wk, ly->wv, ly->wo,
                              ly->w_gate, ly->w_up, ly->w_down, ly->wqkv,
                              ly->wq_gate, ly->ssm_beta, ly->ssm_alpha,
                              ly->ssm_out };
        int count = m->cpu_moe && ly->is_moe && moe_on_host(m, l) ? 4 : 12;
        for (int i = 0; i < count; i++)
            if (ws[i] && !gpu_type_ok(ws[i]->type)) goto unsupported;
        // The architecture publishes this tiny depthwise kernel as F32 in its
        // GGUF contract. Keeping it F32 avoids a second quant decoder in the
        // stateful kernel; an unexpected export falls back safely.
        if (ly->ssm_conv && ly->ssm_conv->type != T_F32) goto unsupported;
    }
    // MoE layers leave the dense gate/up/down NULL, so the loop above never
    // checks the router or the per-expert weights — yet enc_mv indexes the
    // KT_N-entry kernel tables (f_mv[]/f_mvb[]) by tensor type on exactly those.
    // An out-of-range or unsupported type would read past the table (function-
    // pointer type confusion) or launch a NULL fn at decode. Validate them here
    // and fall back to CPU, matching how the dense path already rejects.
    for (int l = 0; l < m->n_layer; l++) {
        layer_t *ly = &m->layers[l];
        // A host-resident bank never reaches a device kernel; an AUTO fit may
        // still promote this layer later, so only a pinned host layer skips.
        if (!ly->is_moe || (moe_on_host(m, l) && m->cpu_moe_layers != CPU_MOE_AUTO))
            continue;
        if (ly->ffn_gate_inp && !gpu_type_ok(ly->ffn_gate_inp->type)) goto unsupported;
        if (ly->moe_split) {
            for (int e = 0; e < m->n_expert; e++)
                if (!gpu_type_ok(ly->moe_g[e]->type) ||
                    !gpu_type_ok(ly->moe_u[e]->type) ||
                    !gpu_type_ok(ly->moe_d[e]->type)) goto unsupported;
        } else {
            if ((ly->ffn_gate_exps && !gpu_type_ok(ly->ffn_gate_exps->type)) ||
                (ly->ffn_up_exps   && !gpu_type_ok(ly->ffn_up_exps->type))   ||
                (ly->ffn_down_exps && !gpu_type_ok(ly->ffn_down_exps->type)))
                goto unsupported;
        }
    }

    if (!cu_load() || cu.Init(0) != 0) return false;
    prof.on = getenv("RUNNER_CUDA_PROFILE") != NULL;
    int ndev = 0;
    if (cu.DeviceGetCount(&ndev) != 0 || ndev < 1) return false;

    gpu_t *g = calloc(1, sizeof(gpu_t));
    if (!g) return false;
    g->last_pos = -2;

    {
        // heterogeneous archs (gemma4) vary q/kv widths per layer: size the
        // shared activation buffers to the maxima
        int q_dim = 0, kv_dim = 0, max_hd = 0;
        for (int l = 0; l < m->n_layer; l++) {
            if (model_q_dim(m, l)    > q_dim)  q_dim  = model_q_dim(m, l);
            if (model_kv_dim(m, l)   > kv_dim) kv_dim = model_kv_dim(m, l);
            if (model_head_dim(m, l) > max_hd) max_hd = model_head_dim(m, l);
        }
        int xdim = q_dim > m->n_embd ? q_dim : m->n_embd;
        int q35_convdim = m->qwen35
                        ? 2 * m->ssm_state * m->ssm_groups + m->ssm_inner : 0;
        int q35_mixdim = q35_convdim > 2 * q_dim ? q35_convdim : 2 * q_dim;
        size_t act_bytes = sizeof(float) * (MVB * ((size_t)m->n_embd + 3 * xdim +
                           q_dim + 2 * kv_dim + 2 * m->n_ff +
                           (size_t)m->n_head * m->n_ctx) + m->n_vocab);
        if (m->qwen35) {
            size_t q35_scratch = MVB * ((size_t)q35_mixdim + q35_convdim +
                                 m->ssm_inner + 2 * m->ssm_v_heads + q_dim);
            size_t q35_persistent = (size_t)m->n_layer *
                                    (m->ssm_conv_kernel - 1) * q35_convdim +
                                    (size_t)m->n_layer * m->ssm_v_heads *
                                    m->ssm_state * m->ssm_state;
            act_bytes += sizeof(float) * (q35_scratch + 2 * q35_persistent);
        }

        // the weights (and the CPU/GPU split they imply) come from the registry:
        // a second instance of the same file reuses the first one's upload
        g->sw = shared_acquire(m, act_bytes, max_hd);
        if (!g->sw) goto fail_quiet;
        m->gpu_layers = g->sw->gpu_layers;
        CK(cu.CtxSetCurrent(g->sw->ctx));

        // device KV holds only the offloaded layers [0, gpu_layers) -- the
        // raw cumulative boundary, not layer gpu_layers's own (possibly
        // redirected) storage location. See model_kv_boundary_bytes().
        size_t kv_bytes = model_kv_boundary_bytes(m, m->gpu_layers);
        CK(cu.MemAlloc(&g->kc, kv_bytes));
        CK(cu.MemAlloc(&g->vc, kv_bytes));
        CK(cu.MemsetD8(g->kc, 0, kv_bytes));
        CK(cu.MemsetD8(g->vc, 0, kv_bytes));

        if (m->n_embd_ple > 0) {
            // [token][layer][n_embd_ple] for a whole tile, plus the per-layer
            // gate scratch. The pre-pass itself runs on the host (it reads a
            // bf16 projection and a q5_K table that have no device kernels),
            // so only its result crosses the bus — once per forward, not once
            // per layer.
            size_t per_tok = (size_t)m->n_layer * m->n_embd_ple;
            CK(cu.MemAlloc(&g->ple,   sizeof(float) * MVB * per_tok));
            CK(cu.MemAlloc(&g->ple_g, sizeof(float) * MVB * m->n_embd_ple));
        }
        CK(cu.MemAlloc(&g->x,      sizeof(float) * MVB * m->n_embd));
        CK(cu.MemAlloc(&g->xb,     sizeof(float) * MVB * xdim));
        CK(cu.MemAlloc(&g->xb2,    sizeof(float) * MVB * xdim));
        CK(cu.MemAlloc(&g->q,      sizeof(float) * MVB * q_dim));
        CK(cu.MemAlloc(&g->kt,     sizeof(float) * MVB * kv_dim));
        CK(cu.MemAlloc(&g->vt,     sizeof(float) * MVB * kv_dim));
        // hb/hb2 also hold gemma-4 MoE's fused gate_up (2*n_ff_exp) for one
        // token; MVB*n_ff dwarfs that for real models, but size for the max so a
        // pathological n_ff_exp can never overrun the buffer.
        size_t hb_elems = (size_t)MVB * m->n_ff;
        if (m->moe_gemma && 2 * (size_t)m->n_ff_exp > hb_elems)
            hb_elems = 2 * (size_t)m->n_ff_exp;
        CK(cu.MemAlloc(&g->hb,     sizeof(float) * hb_elems));
        CK(cu.MemAlloc(&g->hb2,    sizeof(float) * hb_elems));
        CK(cu.MemAlloc(&g->att,    sizeof(float) * MVB * (size_t)m->n_head * m->n_ctx));
        // flash-decoding partials: per (token, head, split) -> hd weighted-V + max + sum
        CK(cu.MemAlloc(&g->attn_part, sizeof(float) * MVB * (size_t)m->n_head *
                                      ATTN_SPLITS * (max_hd + 2)));
        CK(cu.MemAlloc(&g->logits, sizeof(float) * m->n_vocab));
        CK(cu.MemAlloc(&g->pos_dev, sizeof(int)));
        if (m->qwen35) {
            size_t hist_elems = (size_t)m->n_layer * (m->ssm_conv_kernel - 1) *
                                q35_convdim;
            size_t hist_alloc = hist_elems ? hist_elems : 1;
            size_t state_elems = (size_t)m->n_layer * m->ssm_v_heads *
                                 m->ssm_state * m->ssm_state;
            CK(cu.MemAlloc(&g->q35_mix, sizeof(float) * MVB * q35_mixdim));
            CK(cu.MemAlloc(&g->q35_cv, sizeof(float) * MVB * q35_convdim));
            CK(cu.MemAlloc(&g->q35_z, sizeof(float) * MVB * m->ssm_inner));
            CK(cu.MemAlloc(&g->q35_beta, sizeof(float) * MVB * m->ssm_v_heads));
            CK(cu.MemAlloc(&g->q35_alpha, sizeof(float) * MVB * m->ssm_v_heads));
            CK(cu.MemAlloc(&g->q35_gate, sizeof(float) * MVB * q_dim));
            CK(cu.MemAlloc(&g->q35_hist, sizeof(float) * hist_alloc));
            CK(cu.MemAlloc(&g->q35_state, sizeof(float) * state_elems));
            CK(cu.MemAlloc(&g->q35_hist_prev, sizeof(float) * hist_alloc));
            CK(cu.MemAlloc(&g->q35_state_prev, sizeof(float) * state_elems));
            CK(cu.MemsetD8(g->q35_hist, 0, sizeof(float) * hist_alloc));
            CK(cu.MemsetD8(g->q35_state, 0, sizeof(float) * state_elems));
            CK(cu.MemsetD8(g->q35_hist_prev, 0, sizeof(float) * hist_alloc));
            CK(cu.MemsetD8(g->q35_state_prev, 0, sizeof(float) * state_elems));
        }
        // gated attention without the rest of qwen35 (afmoe, muse-glimmer):
        // only the gate scratch is needed
        if (m->attn_out_gate && !m->qwen35)
            CK(cu.MemAlloc(&g->q35_gate, sizeof(float) * MVB * q_dim));
        if (m->n_expert > 0) {
            // sparse-MoE buffers, sized for both the fused indirect path
            // (device routing for a whole tile, per-slot expert scratch) and
            // the eager fallback (which uses column/slot 0 only)
            int used_c = m->n_expert_used;
            if (used_c < 1) used_c = 1;
            if (used_c > 256) used_c = 256;
            int rows = used_c > MVB ? used_c : MVB;
            size_t gw = (size_t)m->n_ff_exp * (m->moe_gemma ? 2 : 1);
            if (gw < 1) gw = 1;
            CK(cu.MemAlloc(&g->moe_logits,
                           sizeof(float) * (size_t)MVB * m->n_expert));
            CK(cu.MemAlloc(&g->moe_eout,
                           sizeof(float) * (size_t)rows * m->n_embd));
            CK(cu.MemAlloc(&g->moe_sel,  sizeof(int)   * (size_t)MVB * used_c));
            CK(cu.MemAlloc(&g->moe_selw, sizeof(float) * (size_t)MVB * used_c));
            CK(cu.MemAlloc(&g->moe_hb,   sizeof(float) * (size_t)rows * gw));
            CK(cu.MemAlloc(&g->moe_hb2,  sizeof(float) * (size_t)rows * gw));
            // expert-grouped prefill staging (device + host mirrors)
            CK(cu.MemAlloc(&g->moe_gath, sizeof(float) * (size_t)MVB * m->n_embd));
            CK(cu.MemAlloc(&g->moe_dout, sizeof(float) * (size_t)MVB * m->n_embd));
            CK(cu.MemAlloc(&g->moe_gidx, sizeof(int)   * (size_t)MVB * used_c));
            CK(cu.MemAlloc(&g->moe_gw,   sizeof(float) * (size_t)MVB * used_c));
            g->h_moe_sel  = malloc(sizeof(int)   * (size_t)MVB * used_c);
            g->h_moe_selw = malloc(sizeof(float) * (size_t)MVB * used_c);
            g->h_moe_gidx = malloc(sizeof(int)   * (size_t)MVB * used_c);
            g->h_moe_gw   = malloc(sizeof(float) * (size_t)MVB * used_c);
            if (!g->h_moe_sel || !g->h_moe_selw || !g->h_moe_gidx ||
                !g->h_moe_gw) goto fail;
            g->moe_fused_ok = moe_fused_eligible(m);
            // RUNNER_MOE_EAGER: debug escape to the v0.1.4 host-routing
            // path — the reference side of the RUNNER_DEBUG_MOE bit
            // comparison, and the fused-vs-eager A/B instrument.
            if (g_moe_eager_force >= 0) {
                if (g_moe_eager_force) g->moe_fused_ok = false;
            } else if (getenv("RUNNER_MOE_EAGER")) {
                g->moe_fused_ok = false;
            }
            g->moe_grouped_ok = moe_grouped_eligible(m);
        }
        if (m->moe_gemma)
            // gemma-4-MoE dual-branch scratch: xn, xn2, rin, mout — MVB
            // columns each so the grouped prefill can batch the whole tile
            CK(cu.MemAlloc(&g->g_scr,
                           sizeof(float) * 5 * (size_t)MVB * m->n_embd));
        if (cu_graphs_ok() && cu.StreamCreate(&g->stream, 0) != 0)
            g->stream = NULL;
        // gemma4's V-copy path (ly->wv == NULL) uses a synchronous MemsetD8
        // that cannot be captured into a CUDA graph. MoE with device routing
        // (moe_fused_ok) is fully capture-clean; only the eager fallback
        // (split layout / uncovered expert type) still reads router logits
        // back to the host mid-forward and must keep the graph off.
        for (int l = 0; l < m->gpu_layers; l++)
            if (!m->layers[l].wv ||
                (m->layers[l].is_moe && !g->moe_fused_ok) ||
                m->layers[l].recurrent) g->graph_bad = true;
        if (m->qwen35) g->graph_bad = true;
        if (moe_any_on_host(m)) g->graph_bad = true;

        if (m->n_embd_ple > 0) {
            size_t per_tok = (size_t)m->n_layer * m->n_embd_ple;
            g->h_ple     = malloc(sizeof(float) * MVB * per_tok);
            g->h_ple_tmp = malloc(sizeof(float) * per_tok);
            if (!g->h_ple || !g->h_ple_tmp) goto fail;
        }
        g->h_x      = malloc(sizeof(float) * MVB * m->n_embd);
        g->h_logits = malloc(sizeof(float) * m->n_vocab);
        if (m->n_expert > 0)
            g->h_moe_logits = malloc(sizeof(float) * (size_t)m->n_expert);
        if (!g->h_x || !g->h_logits ||
            (m->n_expert > 0 && !g->h_moe_logits)) goto fail;

        char name[128] = "CUDA GPU";
        cu.DeviceGetName(name, sizeof(name), g->sw->dev);
        if (m->cpu_moe) {
            int on_host = 0, moe_tot = 0;
            for (int l = 0; l < m->n_layer; l++) {
                if (!m->layers[l].is_moe) continue;
                moe_tot++;
                if (moe_on_host(m, l)) on_host++;
            }
            fprintf(stderr, "gpu: CUDA backend on %s (%d/%d attention layers, "
                    "%.1f GB in VRAM; %d/%d expert layers on CPU)\n", name,
                    m->gpu_layers, m->n_layer, g->sw->weights_len / 1e9,
                    on_host, moe_tot);
        }
        else if (m->gpu_layers < m->n_layer)
            fprintf(stderr, "gpu: CUDA backend on %s (%d/%d layers, %.1f GB in "
                    "VRAM; CPU runs the rest)\n", name, m->gpu_layers, m->n_layer,
                    g->sw->weights_len / 1e9);
        else
            fprintf(stderr, "gpu: CUDA backend on %s (%.1f GB weights in VRAM)\n",
                    name, g->sw->weights_len / 1e9);
        // the number Phase 5 is judged on: with weights shared, a second slot
        // should cost only its KV cache and activation scratch
        size_t vfree = 0, vtotal = 0;
        if (cu.MemGetInfo(&vfree, &vtotal) == 0)
            fprintf(stderr, "gpu: VRAM %.2f GB free of %.2f GB after init "
                    "(kv %.2f GB + scratch %.2f GB this instance)\n",
                    vfree / 1e9, vtotal / 1e9,
                    2.0 * model_kv_boundary_bytes(m, m->gpu_layers) / 1e9,
                    act_bytes / 1e9);
    }

    m->gpu = g;
    return true;

fail:
fail_quiet:
    gpu_ctx_free(m, g);
    return false;

unsupported:
    fprintf(stderr, "gpu: model uses a quant type without a CUDA kernel — using CPU\n");
    return false;
}

// ----------------------------------------------------------------- launches

// One-shot diagnostic for the GPU giving up mid-forward. Every enc_* helper
// returns a bare false, so the caller could only ever say "kernel launch
// failed" — which is true of a real launch error, a missing kernel for a quant
// type, and a weight that was never uploaded alike. Those need different
// fixes, so the first one to happen names itself. Once per process: a failing
// forward fails for every layer, and the first is the informative one.
static void gpu_first_error(const char *fmt, ...) {
    static bool reported = false;
    if (reported) return;
    reported = true;
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "gpu: ");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
}

static bool launch(gpu_t *g, CUfunction f, unsigned gx, unsigned gy, unsigned gz,
                   unsigned bx, void **params) {
    if (prof.on) prof.total_launches[prof.cur_mode]++;
    if (!f) { gpu_first_error("no kernel bound for this operation"); return false; }
    CUresult r = cu.LaunchKernel(f, gx, gy, gz, bx, 1, 1, 0, g->stream, params, NULL);
    if (r != 0)
        gpu_first_error("cuLaunchKernel failed: %s (grid %ux%ux%u, block %u)",
                        cu_err(r), gx, gy, gz, bx);
    return r == 0;
}

static void prof_report(void);

// lazily allocate the event pool the first time profiling runs
static void prof_init(void) {
    if (prof.inited || !prof.on) return;
    if (!cu.EventCreate || !cu.EventRecord || !cu.EventElapsedTime ||
        !cu.EventSynchronize) { prof.on = 0; return; }
#ifdef _WIN32
    LARGE_INTEGER f; QueryPerformanceFrequency(&f); prof.qpc_hz = (double)f.QuadPart;
#else
    prof.qpc_hz = 1.0;
#endif
    for (int i = 0; i < 4096; i++)
        if (cu.EventCreate(&prof.ev[i], 0) != 0) {
            // free the events already created so a mid-loop failure doesn't
            // orphan them (prof_report won't run: inited stays 0)
            for (int j = 0; j < i; j++) {
                if (cu.EventDestroy) cu.EventDestroy(prof.ev[j]);
                prof.ev[j] = NULL;
            }
            prof.on = 0;
            return;
        }
    prof.inited = 1;
    atexit(prof_report);   // CLI mode returns without model_free/gpu_free
}

// record an event tagged with the phase of the kernel group just launched;
// no-op unless profiling is on and we are not mid graph-capture
static void prof_mark(gpu_t *g, int phase) {
    if (!prof.on || prof.capturing || !prof.inited) return;
    if (prof.nev >= 4096) return;
    prof.ph[prof.nev] = phase;
    cu.EventRecord(prof.ev[prof.nev], g->stream);
    prof.nev++;
}

// after the call's stream/ctx has been synchronized, fold the per-group
// GPU deltas into the phase accumulators and reset the pool
static void prof_flush(void) {
    if (!prof.on || !prof.inited) return;
    for (int k = 1; k < prof.nev; k++) {
        float ms = 0;
        if (cu.EventElapsedTime(&ms, prof.ev[k - 1], prof.ev[k]) == 0 &&
            prof.ph[k] >= 0 && prof.ph[k] < PH_N)
            prof.t_ph[prof.cur_mode][prof.ph[k]] += ms;
    }
    prof.nev = 0;
}

static bool enc_rmsnorm(gpu_t *g, CUdeviceptr x, CUdeviceptr y, CUdeviceptr w,
                        int n, float eps, int batch, int xs, int ys) {
    void *p[] = { &x, &y, &w, &n, &eps, &xs, &ys };
    return launch(g, g->sw->f_rmsnorm, 1, batch, 1, 256, p);
}

// Tensor-core GEMM (RUNNER_CUDA_TC). Promoted per (type, arch) by the
// tolerance gate (tests/test_tc_tol.c; results in the TC spec): combos in
// tc_promoted() run TC by DEFAULT, everything else stays scalar until gated.
// RUNNER_CUDA_TC=1 forces the path on everywhere (including unpromoted
// combos — how a gate candidate is measured); =0/off forces it off
// everywhere. Parse the VALUE: a bare existence check made =0 mean "on".
enum { TC_ENV_UNSET = -2 };
static int g_tc_state = TC_ENV_UNSET;   // -2 env unread, -1 per-combo, 0/1 forced

// Test hook (the TC tolerance gate compares scalar and TC in one process,
// so an env var read once at first launch cannot express "both"). Pass -1
// to return to the env/promotion default.
void gpu_tc_force(int on) {
    g_tc_state = on < 0 ? TC_ENV_UNSET : (on != 0);
}

// Incremented at the one place the TC GEMM is dispatched, so a gate can ask
// "did it run?" instead of inferring it from the output.
static unsigned long g_tc_dispatches = 0;
unsigned long gpu_tc_dispatches(void) { return g_tc_dispatches; }

// The fast decode matvec is a Metal lever: CUDA decode reaches its throughput
// through the batched/TC paths instead, and the CUDA matvec is not under the
// same byte-identity pin that forced Metal to grow a second kernel. Present so
// the mv gate links and can say "never dispatched here".
void gpu_mv_force(int on) { (void)on; }
unsigned long gpu_mv_dispatches(void) { return 0; }

// Metal-only lever; present so the gate links and reports 'never here'.
void gpu_attn_coop_force(int on) { (void)on; }
unsigned long gpu_attn_coop_dispatches(void) { return 0; }

// Same hook for the MoE routing path: the fused-vs-eager tolerance gate has to
// run both inside one process, which an env var read at first launch cannot
// express. -1 restores the env default (RUNNER_MOE_EAGER).
void gpu_moe_eager_force(int on) { g_moe_eager_force = on < 0 ? -1 : (on != 0); }

// Promoted (type, arch) combos — every row measured by test_tc_tol on real
// weights (Blackwell MIG, 2026-07-29; table in the TC spec). Dense archs
// only, and only those with a gate row: llama 0.003% of range / phi3 0.004%
// / gemma4 0.012% / qwen3 dense, mistral, gemma3, smollm per the same run —
// all 0 top-1 flips in 64 teacher-forced positions. qwen3moe PASSED its
// gate too (0.216%, one near-tie) but stays opt-in by owner decision: MoE
// routing amplifies fp16 noise ~86x over dense, and the promotion decision
// deliberately covers the dense family first. Unmeasured archs (qwen2,
// qwen35, stablelm) are absent, not implied.
static bool tc_promoted(const model_t *m, int type) {
    // Q6_K joined 2026-08-08: the profile showed it was 26% of prefill running
    // on the SCALAR path, because attn_v/ffn_down are Q6_K in every Q4_K_M.
    //
    // Q8_0 joined 2026-08-09, and it is the strongest row in this table:
    // test_tc_tol on Qwen3-4B-Q8_0 reported logits BIT-IDENTICAL across 504
    // counted TC dispatches — not "within tolerance", identical. Prefill went
    // 113.72 -> 475.71 tok/s (4.18x) on the same model and prompt.
    //
    // Bit-identity is why this is promoted for the whole trusted arch list on
    // one model's evidence, where a tolerance-based row would not be. The arch
    // list exists because architectures differ in how much they AMPLIFY
    // numerical difference — MoE routing amplifies fp16 noise ~86x over dense,
    // which is why qwen3moe stays opt-in. With zero difference to amplify, that
    // sensitivity cannot bite. The kernel is exact here because int8 weights
    // convert to fp16 without loss and the accumulation order matches the
    // scalar path; that is a property of the kernel, not of the model.
    //
    // RESOLVED 2026-08-13b, and the caveat's own instruction applies: the
    // sm_120 confirmation was run and Q8_0 is NOT bit-identical there. It is
    // merely near-identical, so this row is hereby DEMOTED from "exact" to
    // tolerance-gated like every other row:
    //
    //   smollm  SmolLM2-135M-Q8_0        0/64 flips, 8e-5 of logit range
    //   phi3    Phi-4-mini-Q8_0          0/64,        2e-5
    //   granite granite-4.1-3b-Q8_0      0/64,        5e-5
    //
    // The old sm_86 bit-identity reading is also retired for a second reason:
    // it was measured 2026-08-09, one day AFTER TC_N widened to 64, on the
    // kernel that computed 16 of its 64 columns. That kernel reported
    // bit-identity on phi3 and gemma4 while producing different free-running
    // text at -b 64 — so "BIT-IDENTICAL" was not evidence of a correct kernel
    // and cannot carry a promotion on its own. Promotion for the whole arch
    // list now rests on three tolerance rows across three archs plus the
    // free-running arm in test_tc_tol, not on an exactness argument.
    // Q4_0 joined 2026-08-13, and only because the bug above it was fixed
    // first. Its TC kernel shares TC_GEMM_32B with Q8_0, which until that day
    // computed 16 of its 64 token columns and published uninitialised shared
    // memory for the rest; Q4_0 forced on scored 28/64 to 59/64 top-1 flips.
    // With the macro fixed, six models across four archs gate clean at
    // N_BATCH=64 — 0/64 flips every row, mean|dlogit| 3e-5 to 8e-5 of range:
    //
    //   qwen3   Qwen3-0.6B-q4_0, Qwen3-1.7B-q4_0, dense-control-stage1-q4_0
    //   phi3    Phi-4-mini-instruct-q4_0
    //   smollm  SmolLM2-135M requantised to q4_0
    //   llama   Llama-3.2-3B-Q4_K_M requantised to q4_0 (Q4_K + Q4_0 mix)
    //
    // This is the largest prefill row in the table because Q4_0 had no GEMM
    // at all on the promoted path: pp512 goes 57.8 -> 977.8 tok/s on
    // Qwen3-1.7B (16.9x) and 21.2 -> 445.4 on Phi-4-mini (21x). gemma4,
    // mistral and gemma3 have no q4_0 gate row of their own and inherit the
    // arch list below, as Q6_K and Q8_0 did; they are the rows to measure
    // next, not rows this evidence covers.
    if (type != T_Q4_K && type != T_Q6_K && type != T_Q8_0 && type != T_Q4_0)
        return false;
    // granite joined 2026-08-13 with a gate row for EVERY promoted type on
    // its own weights — the only arch here that has one — because it was
    // the arch that showed what the missing coverage costs: granite-4.1-8b
    // Q4_0 ran prefill at 8.0 tok/s on a fully offloaded GPU, slower than
    // the same model on CPU, purely because no TC combo admitted it.
    //   Q4_0        granite-4.1-3b-q4_0            0/64, 5e-5 of range
    //   Q8_0        granite-4.1-3b-Q8_0            0/64, 5e-5
    //   Q4_K/Q6_K   granite-3.3-8b-instruct-Q4_K_M 0/64, 3e-5
    // gemma4's 31B QAT Q4_0 row was RE-GATED against the fixed kernel on
    // 2026-08-13b and holds: 0/64 flips, 6e-5 of range, 60/60 offload. The
    // pre-fix "820 dispatches bit-identical" reading it was promoted on is
    // withdrawn — reproduced from a rebuilt pre-fix binary, that verdict is a
    // false pass (same binary, same model, free-running divergence at -b 64).
    // The promotion survives; the evidence behind it did not.
    static const char *archs[] = { "llama", "phi3", "gemma4", "qwen3",
                                   "mistral", "gemma3", "smollm", "granite" };
    for (size_t i = 0; i < sizeof(archs) / sizeof(*archs); i++)
        if (strcmp(m->arch, archs[i]) == 0) return true;
    return false;
}

static bool tc_on(const model_t *m, int type) {
    if (g_tc_state == TC_ENV_UNSET) {
        const char *e = getenv("RUNNER_CUDA_TC");
        if (!e || !*e) g_tc_state = -1;                    // per-combo default
        else g_tc_state = strcmp(e, "0") != 0 && strcmp(e, "off") != 0;
    }
    if (g_tc_state >= 0) return g_tc_state != 0;
    return tc_promoted(m, type);
}

// A kernel with a fixed compile-time column tile cannot take the full prefill
// batch, so a wider batch runs as ceil(batch/tile) launches of the proven
// shape. k_gemm_q8_0 already did exactly this for its 8-column tile; this
// generalises it so the buffer width and the tile width can move apart.
static bool launch_tiled(gpu_t *g, CUfunction f, unsigned grid, unsigned block,
                         CUdeviceptr weights, CUdeviceptr x, CUdeviceptr y,
                         mv_args a, CUdeviceptr bias, int tile) {
    if (a.batch <= tile) {
        void *p[] = { &weights, &x, &y, &a, &bias };
        return launch(g, f, grid, 1, 1, block, p);
    }
    for (int off = 0; off < a.batch; off += tile) {
        mv_args ai = a;
        ai.batch = (a.batch - off) < tile ? (a.batch - off) : tile;
        CUdeviceptr xi = x + (CUdeviceptr)((uint64_t)off * (uint64_t)a.xs * sizeof(float));
        CUdeviceptr yi = y + (CUdeviceptr)((uint64_t)off * (uint64_t)a.ys * sizeof(float));
        void *pi[] = { &weights, &xi, &yi, &ai, &bias };
        if (!launch(g, f, grid, 1, 1, block, pi)) return false;
    }
    return true;
}

static bool enc_mv(gpu_t *g, model_t *m, gguf_tensor *w, CUdeviceptr x,
                   CUdeviceptr y, int n_in, int n_out, CUdeviceptr bias,
                   int batch, int xs, int ys) {
    CUdeviceptr weights = 0;
    uint64_t w_off = 0;
    if (!binding_find(g->sw, m, w, &weights, &w_off)) {
        gpu_first_error("tensor '%s' is not resident on the device "
                        "(no binding covers it)", w->name);
        return false;
    }
    mv_args a = { n_in, n_out, w_off, bias != 0, batch, xs, ys };
    CUdeviceptr b = bias ? bias : g->sw->dummy;
    void *p[] = { &weights, &x, &y, &a, &b };
    // Prefill (batch>1), tensor-core GEMM when promoted for this (type, arch)
    // or forced by RUNNER_CUDA_TC: the block dequantizes a 64-row fp16 weight
    // tile once and its four warps' MMAs share it (TC_ROWS/block, 128 threads).
    if (batch > 1 && tc_on(m, w->type) && g->sw->f_gemm_tc[w->type]) {
        g_tc_dispatches++;
        return launch_tiled(g, g->sw->f_gemm_tc[w->type],
                            (n_out + TC_ROWS - 1) / TC_ROWS, 128,
                            weights, x, y, a, b, TC_N);
    }
    // Prefill (batch>1) uses the tiled-GEMM variant where available (Q8_0/Q4_K):
    // GEMM_WARPS(=8) rows per block, 256 threads, x staged in shared memory.
    if (batch > 1 && g->sw->f_gemm[w->type]) {
        // k_gemm_q8_0 keeps a fixed 8-column tile — a 16-column f32 x-tile
        // exceeds shared memory, and every narrower restructure measured
        // slower — so a wider tile runs as two launches of the proven shape.
        return launch_tiled(g, g->sw->f_gemm[w->type], (n_out + 7) / 8, 8 * 32,
                            weights, x, y, a, b,
                            w->type == T_Q8_0 ? Q8_COLS : MVT);
    }
    // Decode (batch==1) uses the coalesced lane-per-element GEMV where available
    // (Q8_0/Q4_K); same 4-rows/block shape, so capture-compatible with no
    // host-side branching on per-token state.
    if (batch == 1 && g->sw->f_gemv[w->type])
        return launch(g, g->sw->f_gemv[w->type], (n_out + 3) / 4, 1, 1, 128, p);
    // 128 threads = 4 warps = 4 rows per block; the tile variant applies each
    // decoded weight to all columns, the single variant is faster at batch 1
    CUfunction f = batch > 1 ? g->sw->f_mvb[w->type] : g->sw->f_mv[w->type];
    return launch_tiled(g, f, (n_out + 3) / 4, 128, weights, x, y, a, b,
                        batch > 1 ? MVT : MVB);
}

static bool enc_qknorm(gpu_t *g, model_t *m, CUdeviceptr v, CUdeviceptr w,
                       int n_heads, int hd, int batch, int vs) {
    float eps = m->rms_eps;
    void *p[] = { &v, &w, &hd, &eps, &vs };
    return launch(g, g->sw->f_qknorm, n_heads, batch, 1, 64, p);
}

static bool enc_rope(gpu_t *g, model_t *m, CUdeviceptr v, int n_heads,
                     int batch, int vs, int l) {
    // sliding-window layers rope at their own base with no YaRN scale;
    // heterogeneous archs (gemma4) also rotate fewer dims on local layers
    bool local = model_is_swa(m, l);
    rope_args a = { model_head_dim(m, l), n_heads, model_rope_dim(m, l) / 2,
                    m->rope_neox, model_rope_mscale(m, l) };
    CUdeviceptr fr = local ? g->sw->inv_freq_local : g->sw->inv_freq;
    void *p[] = { &v, &fr, &a, &g->pos_dev, &vs };
    return launch(g, g->sw->f_rope, (a.half_dim + 31) / 32, n_heads, batch, 32, p);
}

static bool enc_scale(gpu_t *g, CUdeviceptr x, float s, int n, int batch, int xs) {
    void *p[] = { &x, &s, &n, &xs };
    return launch(g, g->sw->f_scale, (n + 255) / 256, batch, 1, 256, p);
}

// NoPE + attention temperature, mirroring the CPU tail. A NoPE layer skips
// rope and instead scales Q by a per-token factor that grows with position;
// the factor differs per token, so this is one enc_scale per column rather
// than one for the tile. Only reached by a model that declares the knob.
static bool enc_rope_or_temp(gpu_t *g, model_t *m, CUdeviceptr q, int n_heads,
                             int tn, int q_dim, int l, int pos,
                             bool (*rope)(gpu_t *, model_t *, CUdeviceptr, int, int, int, int)) {
    if (model_layer_ropes(m, l)) return rope(g, m, q, n_heads, tn, q_dim, l);
    for (int b = 0; b < tn; b++) {
        float ts = model_attn_temp(m, pos + b);
        if (ts == 1.0f) continue;
        CUdeviceptr col = q + (CUdeviceptr)((size_t)b * q_dim * sizeof(float));
        if (!enc_scale(g, col, ts, q_dim, 1, q_dim)) return false;
    }
    return true;
}


static bool enc_add(gpu_t *g, CUdeviceptr x, CUdeviceptr d, int n,
                    int batch, int xs, int ds) {
    void *p[] = { &x, &d, &n, &xs, &ds };
    return launch(g, g->sw->f_add, (n + 255) / 256, batch, 1, 256, p);
}

// gemma4 E-series per-layer embedding branch, mirroring the CPU tail in
// model.c: gate the post-FFN residual into the PLE width, multiply it by this
// layer's slice of the per-token table, project back and RMS-norm, then add.
// Runs after the FFN residual and before the layer output scale.
//
// xb is free here (its FFN output already went into x) and doubles as the
// projection destination, matching what the CPU path reuses.
static bool enc_ple(gpu_t *g, model_t *m, const layer_t *ly, int l,
                    int tn, int xdim) {
    if (!ly->ple_gate) return true;
    int P = m->n_embd_ple, n_embd = m->n_embd;
    size_t per_tok = (size_t)m->n_layer * P;
    if (!enc_mv(g, m, ly->ple_gate, g->x, g->ple_g, n_embd, P, 0,
                tn, n_embd, P))
        return false;
    // k_gelu_mul is gelu(a)*b elementwise in place on a — the same tanh
    // approximation gated_act(ACT_GELU, ...) uses on the host. The slice
    // stride is per_tok, so each token reads its own layer-l window.
    for (int b = 0; b < tn; b++) {
        CUdeviceptr gate = g->ple_g + (CUdeviceptr)((size_t)b * P * sizeof(float));
        CUdeviceptr slice = g->ple +
            (CUdeviceptr)(((size_t)b * per_tok + (size_t)l * P) * sizeof(float));
        void *p[] = { &gate, &slice, &P };
        if (!launch(g, g->sw->f_gelu, (P + 255) / 256, 1, 1, 256, p))
            return false;
    }
    if (!enc_mv(g, m, ly->ple_proj, g->ple_g, g->xb, P, n_embd, 0,
                tn, P, xdim))
        return false;
    if (!enc_rmsnorm(g, g->xb, g->xb, g->sw->ple_pn[l], n_embd, m->rms_eps,
                     tn, xdim, xdim))
        return false;
    return enc_add(g, g->x, g->xb, n_embd, tn, n_embd, xdim);
}

static bool enc_actmul(gpu_t *g, model_t *m, CUdeviceptr a, CUdeviceptr b, int n) {
    void *p[] = { &a, &b, &n };
    CUfunction f = m->ffn_act == ACT_SWIGLU_OAI ? g->sw->f_swiglu
                 : m->ffn_act == ACT_GELU       ? g->sw->f_gelu
                                                : g->sw->f_silu;
    return launch(g, f, (n + 255) / 256, 1, 1, 256, p);
}

// Apertus's dense FFN is ungated: up -> xIELU -> down. The four effective
// parameters were transformed once at model load and are passed by value for
// the current layer, matching the CPU xielu() helper without device tables.
static bool enc_xielu(gpu_t *g, CUdeviceptr x, int n,
                      float an, float ap, float b, float eps) {
    void *p[] = { &x, &n, &an, &ap, &b, &eps };
    return launch(g, g->sw->f_xielu, (n + 255) / 256, 1, 1, 256, p);
}

// Per-sequence teardown. The shared weights are not touched here beyond
// dropping this instance's reference: another slot may still be decoding
// against them, and only the last release frees them.
static void gpu_ctx_free(model_t *m, gpu_t *g) {
    (void)m;
    if (!g) return;
    if (g->sw && g->sw->ctx) cu.CtxSetCurrent(g->sw->ctx);
    if (g->gexec && cu.GraphExecDestroy) cu.GraphExecDestroy(g->gexec);
    if (g->graph && cu.GraphDestroy) cu.GraphDestroy(g->graph);
    if (g->stream && cu.StreamDestroy) cu.StreamDestroy(g->stream);
    CUdeviceptr bufs[] = { g->kc, g->vc, g->x, g->xb, g->xb2,
                           g->q, g->kt, g->vt, g->hb, g->hb2, g->att,
                           g->attn_part, g->logits, g->pos_dev,
                           g->moe_logits, g->moe_eout, g->moe_sel,
                           g->moe_selw, g->moe_hb, g->moe_hb2,
                           g->moe_gath, g->moe_dout, g->moe_gidx, g->moe_gw,
                           g->g_scr, g->ple, g->ple_g,
                           g->q35_mix, g->q35_cv, g->q35_z, g->q35_beta,
                           g->q35_alpha, g->q35_gate, g->q35_hist,
                           g->q35_state, g->q35_hist_prev, g->q35_state_prev };
    for (size_t i = 0; i < sizeof(bufs) / sizeof(*bufs); i++)
        if (bufs[i]) cu.MemFree(bufs[i]);
    free(g->h_x); free(g->h_logits); free(g->h_moe_logits);
    free(g->h_ple); free(g->h_ple_tmp);
    free(g->h_moe_sel); free(g->h_moe_selw);
    free(g->h_moe_gidx); free(g->h_moe_gw);
    shared_release(g->sw);
    free(g);
}

static void prof_report_mode(int md, const char *label) {
    if (prof.n_tiles[md] == 0) return;
    double gpu_sum = 0;
    for (int i = 0; i < PH_N; i++) gpu_sum += prof.t_ph[md][i];
    double wall = prof.wall_ms[md];
    double nt = (double)prof.n_tiles[md], ntok = (double)prof.n_tokens[md];
    // On the plain path a synchronous cuMemcpyHtoD (inside stage_x) blocks on
    // g->stream, so the GPU-execution wait is charged to whichever host timer
    // issues the next blocking copy — it is NOT extra time. The GPU event sum
    // is the authoritative busy time; residual = wall - GPU-busy is the true
    // launch/host overhead (host copy timers overlap GPU and are shown raw).
    double residual = wall - gpu_sum;
    fprintf(stderr,
      "\n==== RUNNER_CUDA_PROFILE [%s] ====\n"
      "note: CUDA graphs are DISABLED while profiling — a replayed graph never\n"
      "      re-enters the instrumented launch path, so decode would report\n"
      "      0.0 ms for every phase. Decode wall time is therefore pessimistic\n"
      "      (it carries the launch overhead graphs remove); the phase SPLIT is\n"
      "      what these numbers are for.\n"
      "tiles=%lld tokens=%lld launches=%lld (%.1f/tile, %.1f/token)\n"
      "wall: %.1f ms | %.4f ms/tile | %.4f ms/token\n"
      "-- GPU phase (CUDA events, authoritative) --\n",
      label, prof.n_tiles[md], prof.n_tokens[md], prof.total_launches[md],
      prof.total_launches[md] / nt, prof.total_launches[md] / ntok,
      wall, wall / nt, wall / ntok);
    for (int i = 0; i < PH_N; i++)
        fprintf(stderr, "  %-14s %9.1f ms  %5.1f%%  %.4f ms/tile  %.4f ms/tok\n",
                PH_NAME[i], prof.t_ph[md][i],
                100.0 * prof.t_ph[md][i] / (gpu_sum ? gpu_sum : 1),
                prof.t_ph[md][i] / nt, prof.t_ph[md][i] / ntok);
    fprintf(stderr,
      "  %-14s %9.1f ms  (%.1f%% of wall)  %.4f ms/tile\n"
      "-- host-side raw QPC (overlaps GPU via sync memcpy) --\n"
      "  stage_x=%.1f  kv_copyback=%.1f  logitsDtoH=%.1f  sync-wait=%.1f ms\n"
      "  residual(wall-GPUbusy)=%.1f ms (%.1f%% of wall) = launch+host overhead\n",
      "GPU-TOTAL", gpu_sum, 100.0 * gpu_sum / (wall ? wall : 1), gpu_sum / nt,
      prof.t_stage[md], prof.t_copyback[md], prof.t_logitscp[md], prof.t_sync[md],
      residual, 100.0 * residual / (wall ? wall : 1));
}

static void prof_report(void) {
    if (!prof.on || !prof.inited) return;
    prof_report_mode(M_BATCH,  "PREFILL tile tn>1 (_b kernels)");
    prof_report_mode(M_SINGLE, "DECODE  tn==1 (k_mv_* kernels)");
    fprintf(stderr, "=============================\n");
    for (int i = 0; i < 4096; i++)
        if (prof.ev[i] && cu.EventDestroy) cu.EventDestroy(prof.ev[i]);
    prof.inited = 0;
}

void gpu_free(model_t *m) {
    prof_report();
    gpu_t *g = m->gpu;
    m->gpu = NULL;
    gpu_ctx_free(m, g);
}

// A runtime GPU failure on CUDA is fully recoverable on the host: the host KV
// cache is the authoritative copy and the device one is a mirror, so there is
// nothing to rescue before releasing. Freeing rather than orphaning matters
// now that weights are shared — an abandoned context would keep every other
// slot's copy of them alive too.
void gpu_disable(model_t *m) {
    gpu_t *g = m ? m->gpu : NULL;
    // KV is mirrored after every step. Qwen3.5 recurrent state is much larger,
    // so rescue it only on this rare fallback path instead of copying tens of
    // megabytes over PCIe after every generated token.
    if (g && m->qwen35 && g->sw && g->sw->ctx &&
        cu.CtxSetCurrent(g->sw->ctx) == 0) {
        int convdim = 2 * m->ssm_state * m->ssm_groups + m->ssm_inner;
        size_t hist_layer = (size_t)(m->ssm_conv_kernel - 1) * convdim;
        size_t state_layer = (size_t)m->ssm_v_heads * m->ssm_state * m->ssm_state;
        for (int l = 0; l < m->gpu_layers; l++) {
            if (!m->layers[l].recurrent) continue;
            if (hist_layer)
                cu.MemcpyDtoH(m->ssm_conv_state + (size_t)l * hist_layer,
                              g->q35_hist_prev + (size_t)l * hist_layer * sizeof(float),
                              hist_layer * sizeof(float));
            cu.MemcpyDtoH(m->ssm_state_mem + (size_t)l * state_layer,
                          g->q35_state_prev + (size_t)l * state_layer * sizeof(float),
                          state_layer * sizeof(float));
        }
    }
    gpu_free(m);
}

// upload host KV rows [lo, hi) for every layer (CPU prompt processing wrote them)
static bool kv_upload(gpu_t *g, model_t *m, int lo, int hi) {
    if (lo >= hi) return true;
    for (int l = 0; l < m->gpu_layers; l++) {
        size_t row = model_kv_row_bytes(m, l);
        size_t off = model_kv_byte_off(m, l) + (size_t)lo * row;
        size_t len = (size_t)(hi - lo) * row;
        if (cu.MemcpyHtoD(g->kc + off, (uint8_t *)m->kcache + off, len) != 0 ||
            cu.MemcpyHtoD(g->vc + off, (uint8_t *)m->vcache + off, len) != 0)
            return false;
    }
    return true;
}

// copy device KV rows [lo, hi) back to the (authoritative) host cache
static bool kv_copyback(gpu_t *g, model_t *m, int lo, int hi) {
    if (lo >= hi) return true;
    for (int l = 0; l < m->gpu_layers; l++) {
        size_t row = model_kv_row_bytes(m, l);
        size_t off = model_kv_byte_off(m, l) + (size_t)lo * row;
        size_t len = (size_t)(hi - lo) * row;
        if (cu.MemcpyDtoH((uint8_t *)m->kcache + off, g->kc + off, len) != 0 ||
            cu.MemcpyDtoH((uint8_t *)m->vcache + off, g->vc + off, len) != 0)
            return false;
    }
    return true;
}

// stage the tile's token embeddings into g->x (host dequant + one HtoD);
// kept outside fwd_tile so graph capture records kernels only
// E-series: run the per-layer-embedding pre-pass on the host over the freshly
// staged embeddings and ship its result. Once per forward, so the transfer is
// n_tokens * n_layer * n_embd_ple floats — 43 KB for a single E4B token.
static bool stage_ple(gpu_t *g, model_t *m, const int32_t *tokens, int tn) {
    if (m->n_embd_ple <= 0) return true;
    size_t per_tok = (size_t)m->n_layer * m->n_embd_ple;
    model_ple_prepass(m, tokens, tn, g->h_x, g->h_ple, g->h_ple_tmp);
    return cu.MemcpyHtoD(g->ple, g->h_ple,
                         sizeof(float) * (size_t)tn * per_tok) == 0;
}

static bool stage_x(gpu_t *g, model_t *m, const int32_t *tokens, int tn) {
    size_t ers = ggml_row_size(m->tok_embd->type, m->n_embd);
    for (int b = 0; b < tn; b++) {
        float *hx = g->h_x + (size_t)b * m->n_embd;
        dequant_row(m->tok_embd->type,
                    (uint8_t *)m->tok_embd->data + (size_t)tokens[b] * ers,
                    hx, m->n_embd);
        model_embd_transform(m, hx);
    }
    if (cu.MemcpyHtoD(g->x, g->h_x, sizeof(float) * tn * m->n_embd) != 0)
        return false;
    return stage_ple(g, m, tokens, tn);
}

// encode a tile of up to MVB tokens: matvecs read each weight once for the
// whole tile, and every other kernel takes the token column from the launch
// grid — a tile costs the same number of launches as a single token, which
// matters because WDDM launch overhead dominates small-kernel dispatch. The
// vocab-logits matvec (the single most expensive launch) runs only when the
// caller wants logits, and only for the tile's last token. The caller stages
// x (and, for graph capture, uploads pos) before calling this.
// ----------------------------------------------- sparse-MoE (device routing)
// Encoders for the fused indirect MoE path: routing and expert selection stay
// entirely on the device, so a MoE decode step has no host round-trip and is
// CUDA-graph capturable (the moe-gpu-routing spec's P1).

// Device softmax -> top-k -> renormalize for `tokens` routed columns. The
// kernel is serial per token and mirrors moe_route() bit for bit (see the
// contract comment in kernels.cu).
static bool enc_moe_route(gpu_t *g, CUdeviceptr logits, CUdeviceptr sel,
                          CUdeviceptr selw, int ne, int used, int tokens,
                          int ls) {
    void *p[] = { &logits, &sel, &selw, &ne, &used, &tokens, &ls };
    return launch(g, g->sw->f_moe_route, (unsigned)(tokens + 31) / 32, 1, 1, 32, p);
}

// One launch computes every selected expert's matvec: grid.y = expert slot,
// weight base resolved in-kernel from sel[slot] and the fused-3D expert
// byte stride (the moe_expert_weight arithmetic, moved into kernel args).
// base is the whole fused tensor, so binding_find's bounds check covers
// every expert's slice at once.
static bool enc_moe_mv(gpu_t *g, model_t *m, gguf_tensor *base,
                       uint64_t estride, CUdeviceptr x, CUdeviceptr y,
                       CUdeviceptr sel, int n_in, int n_out, int nslots,
                       int xs, int ys) {
    CUdeviceptr weights = 0;
    uint64_t w_off = 0;
    if (!binding_find(g->sw, m, base, &weights, &w_off)) {
        gpu_first_error("expert tensor '%s' is not resident on the device "
                        "(no binding covers it)", base->name);
        return false;
    }
    CUfunction f = g->sw->f_moe_mv[base->type];
    if (!f) {
        gpu_first_error("no indirect MoE kernel for quant type %s",
                        ggml_type_name(base->type));
        return false;
    }
    moe_args a = { n_in, n_out, w_off, estride, xs, ys };
    void *p[] = { &weights, &x, &y, &a, &sel };
    return launch(g, f, (n_out + 3) / 4, nslots, 1, 128, p);
}

// Gated activation + routing-weight fold for every slot in one launch.
// dscale is the per-expert down-scale table (gemma-4) or the shared ones
// table; the arithmetic matches the eager actmul+scale launch pair exactly.
// gb/ub are the per-expert gate/up bias tables (gpt-oss) or 0 — the bias
// lands before the activation, mirroring moe_ffn_token on the CPU.
static bool enc_moe_actmul(gpu_t *g, model_t *m, CUdeviceptr gbuf,
                           CUdeviceptr ubuf, int nff, int gss, int uss,
                           CUdeviceptr selw, CUdeviceptr dscale,
                           CUdeviceptr sel, int nslots,
                           CUdeviceptr gb, CUdeviceptr ub) {
    int act = m->ffn_act;  // ACT_* values match the kernel's selector 1:1
    void *p[] = { &gbuf, &ubuf, &nff, &gss, &uss, &act, &selw, &dscale, &sel,
                  &gb, &ub };
    return launch(g, g->sw->f_moe_actmul, (nff + 255) / 256, nslots, 1, 256, p);
}

// Sum the per-slot down outputs into the token's FFN output, ascending slot
// order — same accumulation order as the eager write-then-add sequence.
// db is the per-expert down-bias table (gpt-oss) or 0; each slot adds
// selw*db[sel] because the routing weight was already folded into the hidden
// before the down matvec (w*(down+db) distributed).
static bool enc_moe_sum(gpu_t *g, CUdeviceptr out, CUdeviceptr eout, int n,
                        int nslots, int es, CUdeviceptr db, CUdeviceptr sel,
                        CUdeviceptr selw) {
    void *p[] = { &out, &eout, &n, &nslots, &es, &db, &sel, &selw };
    return launch(g, g->sw->f_moe_sum, (n + 255) / 256, 1, 1, 256, p);
}

// Fused device-routing MoE FFN (Mixtral/Qwen3 topology, fused-3D layout).
// Reads the RMS-normed input from g->xb (stride xdim) and writes the FFN
// output back into g->xb. Per layer: one router matvec per token (batch-1,
// the certified per-token GEMV arithmetic), one routing kernel for the
// tile, then per token gate/up/actmul/down/sum — six launches a token and
// zero host round-trips, vs the eager path's ~49 launches and one
// sync+DtoH per token.
static bool gpu_moe_ffn_fused(gpu_t *g, model_t *m, const layer_t *ly,
                              int tn, int xdim) {
    int n_embd = m->n_embd, ne = m->n_expert, used = m->n_expert_used;
    int nff = m->n_ff_exp;
    int l = (int)(ly - m->layers);
    enum { MOE_MAX_USED = 256 };
    if (used > MOE_MAX_USED) used = MOE_MAX_USED;
    if (ne  > MOE_MAX_USED) ne  = MOE_MAX_USED;
    uint64_t gstride = (uint64_t)nff *
                       ggml_row_size(ly->ffn_gate_exps->type, n_embd);
    uint64_t ustride = (uint64_t)nff *
                       ggml_row_size(ly->ffn_up_exps->type, n_embd);
    uint64_t dstride = (uint64_t)n_embd *
                       ggml_row_size(ly->ffn_down_exps->type, nff);
    for (int t = 0; t < tn; t++) {
        CUdeviceptr xin = g->xb + (size_t)t * xdim * sizeof(float);
        CUdeviceptr lg  = g->moe_logits + (size_t)t * ne * sizeof(float);
        // gib: the gpt-oss router bias, which applies to the LOGITS before
        // gating (model.c moe_route). k_moe_route takes no bias input, so it
        // has to ride the router matvec — as it already does on the eager and
        // grouped paths and in metal.m.
        if (!enc_mv(g, m, ly->ffn_gate_inp, xin, lg, n_embd, ne,
                    g->sw->gib[l], 1, xdim, ne))
            return false;
    }
    if (!enc_moe_route(g, g->moe_logits, g->moe_sel, g->moe_selw, ne, used,
                       tn, ne))
        return false;
    // RUNNER_DEBUG_MOE: fused-path twin of the eager dump above
    if (getenv("RUNNER_DEBUG_MOE")) {
        static int dumped = 0;
        if (dumped < 2) {
            dumped++;
            cu.StreamSynchronize(g->stream);
            int hs[256]; float hw[256];
            cu.MemcpyDtoH(hs, g->moe_sel, sizeof(int) * used);
            cu.MemcpyDtoH(hw, g->moe_selw, sizeof(float) * used);
            fprintf(stderr, "FUSED sel :");
            for (int s = 0; s < used; s++) fprintf(stderr, " %d", hs[s]);
            fprintf(stderr, "\nFUSED selw:");
            for (int s = 0; s < used; s++) {
                unsigned u; memcpy(&u, &hw[s], 4);
                fprintf(stderr, " %08x", u);
            }
            fprintf(stderr, "\n");
        }
    }
    for (int t = 0; t < tn; t++) {
        CUdeviceptr xin  = g->xb + (size_t)t * xdim * sizeof(float);
        CUdeviceptr sel  = g->moe_sel  + (size_t)t * used * sizeof(int);
        CUdeviceptr selw = g->moe_selw + (size_t)t * used * sizeof(float);
        bool ok = enc_moe_mv(g, m, ly->ffn_gate_exps, gstride, xin, g->moe_hb,
                             sel, n_embd, nff, used, 0, nff)
               && enc_moe_mv(g, m, ly->ffn_up_exps, ustride, xin, g->moe_hb2,
                             sel, n_embd, nff, used, 0, nff)
               && enc_moe_actmul(g, m, g->moe_hb, g->moe_hb2, nff, nff, nff,
                                 selw, g->sw->moe_ones, sel, used,
                                 g->sw->geb[l], g->sw->ueb[l])
               && enc_moe_mv(g, m, ly->ffn_down_exps, dstride, g->moe_hb,
                             g->moe_eout, sel, nff, n_embd, used, nff, n_embd)
               && enc_moe_sum(g, xin, g->moe_eout, n_embd, used, n_embd,
                              g->sw->deb[l], sel, selw);
        if (!ok) return false;
    }
    return true;
}

// ------------------------------------------ expert-grouped prefill (P2)
// CUDA port of the CPU cabdad1 grouping: route the tile on device, read the
// tile's routing back ONCE (prefill is never graph-captured, so the single
// sync+DtoH is legal), build per-expert token lists on the host, then run
// each active expert once over its token set with the batched GEMM kernels
// — each expert's weights stream through the SMs once per tile instead of
// once per (token, slot). Accumulation follows the CPU grouped path:
// ascending expert index, routing weight applied at the scatter.

// Read the tile's routing and build the concatenated per-expert token
// lists (ascending expert), uploading them to moe_gidx/moe_gw. Fills
// e_start/e_cnt (ne entries); when fold_dscale is set the per-expert down
// scale (gemma-4) is folded into the scatter weight, exactly as the CPU
// grouped/eager paths compute sc = selw * down_exps_scale[e]. Returns
// false on a CUDA error.
static bool moe_build_groups(gpu_t *g, const layer_t *ly, int tn, int used,
                             int ne, bool fold_dscale, int *e_start,
                             int *e_cnt) {
    if (cu.StreamSynchronize(g->stream) != 0 ||
        cu.MemcpyDtoH(g->h_moe_sel, g->moe_sel,
                      sizeof(int) * (size_t)tn * used) != 0 ||
        cu.MemcpyDtoH(g->h_moe_selw, g->moe_selw,
                      sizeof(float) * (size_t)tn * used) != 0)
        return false;
    int total = 0;
    for (int e = 0; e < ne; e++) {
        e_start[e] = total;
        for (int b = 0; b < tn; b++) {
            const int   *sel  = g->h_moe_sel  + (size_t)b * used;
            const float *selw = g->h_moe_selw + (size_t)b * used;
            for (int s = 0; s < used; s++) {
                if (sel[s] != e) continue;
                g->h_moe_gidx[total] = b;
                g->h_moe_gw[total] = selw[s] *
                    (fold_dscale && ly->down_exps_scale
                         ? ly->down_exps_scale[e] : 1.0f);
                total++;
                break;                          // top-k experts are distinct
            }
        }
        e_cnt[e] = total - e_start[e];
    }
    return cu.MemcpyHtoD(g->moe_gidx, g->h_moe_gidx,
                         sizeof(int) * (size_t)total) == 0 &&
           cu.MemcpyHtoD(g->moe_gw, g->h_moe_gw,
                         sizeof(float) * (size_t)total) == 0;
}

// defined in the decode-microbatch section below; used here for its
// width-classed kernel selection
static bool enc_mv_batch(gpu_t *g, model_t *m, gguf_tensor *w, CUdeviceptr x,
                         CUdeviceptr y, int n_in, int n_out, CUdeviceptr bias,
                         int batch, int xs, int ys);

// Batched matmul for one expert's token set, cnt-aware. The fixed-width
// GEMM kernels compute all MVB columns whatever the batch, so at the small
// counts grouping produces (~1-2 tokens/expert on a 16-token tile) they do
// ~10x the useful FMA work and lose to the per-token path (measured on the
// 3070). Pick the narrowest kernel that covers cnt: the batch-1 GEMV at
// cnt==1, the width-classed f_gemvb twins up to 8, the full GEMM beyond —
// and the TC GEMM whenever it is promoted/forced for this (type, arch),
// since tensor-core MACs make the unused columns nearly free.
static bool enc_moe_gemm(gpu_t *g, model_t *m, gguf_tensor *w, CUdeviceptr x,
                         CUdeviceptr y, int n_in, int n_out, int cnt,
                         int xs, int ys) {
    if (cnt == 1)
        return enc_mv(g, m, w, x, y, n_in, n_out, 0, 1, xs, ys);
    if (cnt > 8 || (tc_on(m, w->type) && g->sw->f_gemm_tc[w->type]))
        return enc_mv(g, m, w, x, y, n_in, n_out, 0, cnt, xs, ys);
    return enc_mv_batch(g, m, w, x, y, n_in, n_out, 0, cnt, xs, ys);
}

static bool enc_moe_gather(gpu_t *g, CUdeviceptr dst, CUdeviceptr src,
                           CUdeviceptr idx, int n, int cnt, int ss, int ds) {
    void *p[] = { &dst, &src, &idx, &n, &ss, &ds };
    return launch(g, g->sw->f_moe_gather, (n + 255) / 256, cnt, 1, 256, p);
}

static bool enc_moe_scatter(gpu_t *g, CUdeviceptr out, CUdeviceptr src,
                            CUdeviceptr idx, CUdeviceptr wts, int n, int cnt,
                            int os, int ss) {
    void *p[] = { &out, &src, &idx, &wts, &n, &os, &ss };
    return launch(g, g->sw->f_moe_scatter, (n + 255) / 256, cnt, 1, 256, p);
}

// Grouped Mixtral/Qwen3 tile: reads the RMS-normed inputs from g->xb
// (stride xdim), writes the FFN outputs back into g->xb, like the other
// gpu_moe_ffn_* variants.
static bool gpu_moe_ffn_grouped(gpu_t *g, model_t *m, const layer_t *ly,
                                int tn, int xdim) {
    int n_embd = m->n_embd, ne = m->n_expert, used = m->n_expert_used;
    int nff = m->n_ff_exp;
    int l = (int)(ly - m->layers);
    enum { MOE_MAX_USED = 256 };
    if (used > MOE_MAX_USED) used = MOE_MAX_USED;
    if (ne  > MOE_MAX_USED) ne  = MOE_MAX_USED;
    for (int t = 0; t < tn; t++) {
        CUdeviceptr xin = g->xb + (size_t)t * xdim * sizeof(float);
        CUdeviceptr lg  = g->moe_logits + (size_t)t * ne * sizeof(float);
        if (!enc_mv(g, m, ly->ffn_gate_inp, xin, lg, n_embd, ne,
                    g->sw->gib[l], 1, xdim, ne))
            return false;
    }
    if (!enc_moe_route(g, g->moe_logits, g->moe_sel, g->moe_selw, ne, used,
                       tn, ne))
        return false;
    int e_start[MOE_MAX_USED], e_cnt[MOE_MAX_USED];
    if (!moe_build_groups(g, ly, tn, used, ne, false, e_start, e_cnt))
        return false;
    if (cu.MemsetD8(g->moe_eout, 0, sizeof(float) * (size_t)tn * n_embd) != 0)
        return false;
    for (int e = 0; e < ne; e++) {
        int cnt = e_cnt[e];
        if (cnt == 0) continue;
        CUdeviceptr idx = g->moe_gidx + (size_t)e_start[e] * sizeof(int);
        CUdeviceptr wts = g->moe_gw   + (size_t)e_start[e] * sizeof(float);
        gguf_tensor gv = moe_expert_weight(ly, 0, e, n_embd, nff);
        gguf_tensor uv = moe_expert_weight(ly, 1, e, n_embd, nff);
        gguf_tensor dv = moe_expert_weight(ly, 2, e, n_embd, nff);
        bool ok = enc_moe_gather(g, g->moe_gath, g->xb, idx, n_embd, cnt,
                                 xdim, n_embd)
               && enc_moe_gemm(g, m, &gv, g->moe_gath, g->moe_hb, n_embd,
                               nff, cnt, n_embd, nff)
               && enc_moe_gemm(g, m, &uv, g->moe_gath, g->moe_hb2, n_embd,
                               nff, cnt, n_embd, nff)
               && enc_actmul(g, m, g->moe_hb, g->moe_hb2, cnt * nff)
               && enc_moe_gemm(g, m, &dv, g->moe_hb, g->moe_dout, nff,
                               n_embd, cnt, nff, n_embd)
               && enc_moe_scatter(g, g->moe_eout, g->moe_dout, idx, wts,
                                  n_embd, cnt, n_embd, n_embd);
        if (!ok) return false;
    }
    void *p[] = { &g->xb, &g->moe_eout, &n_embd, &xdim, &n_embd };
    return launch(g, g->sw->f_moe_copy, (n_embd + 255) / 256, tn, 1, 256, p);
}

// Grouped gemma-4 dual-branch tile: the dense shared branch batches across
// the whole tile with the ordinary tn-wide kernels (it was per-token
// before), and the routed branch groups by expert over the fused gate_up
// tensor. Reads g->x (un-normed residual), writes dense+routed into g->xb,
// same contract as the other gemma variants.
static bool gpu_gemma_moe_ffn_grouped(gpu_t *g, model_t *m, const layer_t *ly,
                                      int l, int tn, int xdim) {
    int n_embd = m->n_embd, ne = m->n_expert, used = m->n_expert_used;
    int nff = m->n_ff_exp, dff = m->n_ff;
    enum { MOE_MAX_USED = 256 };
    if (used > MOE_MAX_USED) used = MOE_MAX_USED;
    if (ne  > MOE_MAX_USED) ne  = MOE_MAX_USED;
    float eps = m->rms_eps;
    CUdeviceptr xn_b  = g->g_scr;
    CUdeviceptr xn2_b = g->g_scr + (size_t)MVB * n_embd * sizeof(float);
    CUdeviceptr rin_b = g->g_scr + (size_t)2 * MVB * n_embd * sizeof(float);
    // --- dense shared MLP for the whole tile
    bool ok = enc_rmsnorm(g, g->x, xn_b, g->sw->ffn_norm[l], n_embd, eps,
                          tn, n_embd, n_embd)
           && enc_mv(g, m, ly->w_gate, xn_b, g->hb,  n_embd, dff, 0, tn,
                     n_embd, dff)
           && enc_mv(g, m, ly->w_up,   xn_b, g->hb2, n_embd, dff, 0, tn,
                     n_embd, dff)
           && enc_actmul(g, m, g->hb, g->hb2, tn * dff)
           && enc_mv(g, m, ly->w_down, g->hb, g->xb, dff, n_embd, 0, tn,
                     dff, xdim)
           && enc_rmsnorm(g, g->xb, g->xb, g->sw->g_pn1[l], n_embd, eps,
                          tn, xdim, xdim);
    if (!ok) return false;
    // --- routed branch: norms + router for the tile, then group by expert
    ok = enc_rmsnorm(g, g->x, xn2_b, g->sw->g_prn2[l], n_embd, eps, tn,
                     n_embd, n_embd)
      && enc_rmsnorm(g, g->x, rin_b, g->sw->g_gis[l], n_embd, eps, tn,
                     n_embd, n_embd);
    for (int t = 0; ok && t < tn; t++) {
        CUdeviceptr rin = rin_b + (size_t)t * n_embd * sizeof(float);
        CUdeviceptr lg  = g->moe_logits + (size_t)t * ne * sizeof(float);
        ok = enc_mv(g, m, ly->ffn_gate_inp, rin, lg, n_embd, ne, 0, 1,
                    n_embd, ne);
    }
    ok = ok && enc_moe_route(g, g->moe_logits, g->moe_sel, g->moe_selw, ne,
                             used, tn, ne);
    if (!ok) return false;
    int e_start[MOE_MAX_USED], e_cnt[MOE_MAX_USED];
    if (!moe_build_groups(g, ly, tn, used, ne, true, e_start, e_cnt))
        return false;
    if (cu.MemsetD8(g->moe_eout, 0, sizeof(float) * (size_t)tn * n_embd) != 0)
        return false;
    int act = m->ffn_act;  // ACT_* values match the kernel's selector 1:1
    for (int e = 0; e < ne; e++) {
        int cnt = e_cnt[e];
        if (cnt == 0) continue;
        CUdeviceptr idx = g->moe_gidx + (size_t)e_start[e] * sizeof(int);
        CUdeviceptr wts = g->moe_gw   + (size_t)e_start[e] * sizeof(float);
        // fused gate_up expert slice, nbytes clamped (see the eager path)
        gguf_tensor guv = *ly->ffn_gate_up_exps;
        size_t rs = ggml_row_size(guv.type, n_embd);
        guv.data = (uint8_t *)ly->ffn_gate_up_exps->data +
                   (size_t)e * (2 * (size_t)nff) * rs;
        guv.nbytes = (2 * (size_t)nff) * rs;
        gguf_tensor dv = moe_expert_weight(ly, 2, e, n_embd, nff);
        CUdeviceptr up = g->moe_hb + (size_t)nff * sizeof(float);
        int gss = 2 * nff;
        void *pa[] = { &g->moe_hb, &up, &nff, &gss, &gss, &act };
        bool eok = enc_moe_gather(g, g->moe_gath, xn2_b, idx, n_embd, cnt,
                                  n_embd, n_embd)
                && enc_moe_gemm(g, m, &guv, g->moe_gath, g->moe_hb, n_embd,
                                2 * nff, cnt, n_embd, 2 * nff)
                && launch(g, g->sw->f_moe_actmul_plain, (nff + 255) / 256,
                          cnt, 1, 256, pa)
                && enc_moe_gemm(g, m, &dv, g->moe_hb, g->moe_dout, nff,
                                n_embd, cnt, 2 * nff, n_embd)
                && enc_moe_scatter(g, g->moe_eout, g->moe_dout, idx, wts,
                                   n_embd, cnt, n_embd, n_embd);
        if (!eok) return false;
    }
    // post_ffw_norm_2 on the routed accumulator, then out (=dense) += routed
    return enc_rmsnorm(g, g->moe_eout, g->moe_eout, g->sw->g_pn2[l], n_embd,
                       eps, tn, n_embd, n_embd)
        && enc_add(g, g->xb, g->moe_eout, n_embd, tn, xdim, n_embd);
}

// Eager fallback (legacy split expert layout, or an expert tensor type the
// indirect kernels do not cover). Routing — softmax over ALL experts, top-k,
// renormalize (Mixtral/Qwen3) — runs on the host from router logits read
// back per token; the per-expert SwiGLU matmuls run on the GPU by pointing
// enc_mv at each expert's slice. Host-dependent routing is why this path
// forces the eager (graph_bad) decode.
static bool gpu_moe_ffn_eager(gpu_t *g, model_t *m, const layer_t *ly, int tn, int xdim) {
    int n_embd = m->n_embd, ne = m->n_expert, used = m->n_expert_used, nff = m->n_ff_exp;
    int l = (int)(ly - m->layers);
    // belt-and-suspenders: sel[]/selw[] below are fixed MOE_MAX_USED wide. The
    // loader already bounds n_expert_used <= n_expert <= 256 (model.c), but never
    // let a rogue value walk these stack arrays even if a future path reaches here.
    enum { MOE_MAX_USED = 256 };
    if (used > MOE_MAX_USED) used = MOE_MAX_USED;
    if (ne  > MOE_MAX_USED) ne  = MOE_MAX_USED;
    for (int t = 0; t < tn; t++) {
        CUdeviceptr xin  = g->xb  + (size_t)t * xdim * sizeof(float);
        CUdeviceptr aout = g->xb2 + (size_t)t * xdim * sizeof(float);
        if (!enc_mv(g, m, ly->ffn_gate_inp, xin, g->moe_logits, n_embd, ne,
                    g->sw->gib[l], 1, xdim, ne))
            return false;
        if (cu.StreamSynchronize(g->stream) != 0) return false;
        if (cu.MemcpyDtoH(g->h_moe_logits, g->moe_logits, sizeof(float) * (size_t)ne) != 0)
            return false;
        // softmax over all experts, then top-k with renormalized weights
        float *lg = g->h_moe_logits;
        float mx = lg[0];
        for (int e = 1; e < ne; e++)
            if (lg[e] > mx) mx = lg[e];
        float ssum = 0.0f;
        for (int e = 0; e < ne; e++) { float p = expf(lg[e] - mx); lg[e] = p; ssum += p; }
        for (int e = 0; e < ne; e++) lg[e] /= ssum;
        int sel[MOE_MAX_USED];
        float selw[MOE_MAX_USED], denom = 0.0f;
        for (int s = 0; s < used; s++) {
            int best = 0;
            float bp = -1.0f;
            for (int e = 0; e < ne; e++)
                if (lg[e] > bp) { bp = lg[e]; best = e; }
            sel[s] = best;
            selw[s] = bp;
            denom += bp;
            lg[best] = -1.0f;
        }
        // RUNNER_DEBUG_MOE: dump the first two routings' selection + weights
        // as hex — the discriminator that isolated the 2026-07-29 identity
        // regression (device-vs-host expf + reciprocal-math, both 1-ulp).
        // Paired with the same dump in the fused path; bit-compare them.
        if (getenv("RUNNER_DEBUG_MOE")) {
            static int dumped = 0;
            if (dumped < 2 && t == 0) {
                dumped++;
                fprintf(stderr, "EAGER sel :");
                for (int s = 0; s < used; s++) fprintf(stderr, " %d", sel[s]);
                fprintf(stderr, "\nEAGER selw:");
                for (int s = 0; s < used; s++) {
                    float w_ = selw[s] / denom;
                    unsigned u; memcpy(&u, &w_, 4);
                    fprintf(stderr, " %08x", u);
                }
                fprintf(stderr, "\n");
            }
        }
        for (int s = 0; s < used; s++) {
            int e = sel[s];
            float w = selw[s] / denom;
            gguf_tensor gv = moe_expert_weight(ly, 0, e, n_embd, nff);
            gguf_tensor uv = moe_expert_weight(ly, 1, e, n_embd, nff);
            gguf_tensor dv = moe_expert_weight(ly, 2, e, n_embd, nff);
            // gpt-oss per-expert bias slices; 0 for every other arch, which
            // keeps the certified unbiased launch sequence byte-for-byte
            CUdeviceptr gb = g->sw->geb[l]
                ? g->sw->geb[l] + (size_t)e * nff * sizeof(float) : 0;
            CUdeviceptr ub = g->sw->ueb[l]
                ? g->sw->ueb[l] + (size_t)e * nff * sizeof(float) : 0;
            CUdeviceptr db = g->sw->deb[l]
                ? g->sw->deb[l] + (size_t)e * n_embd * sizeof(float) : 0;
            bool ok = enc_mv(g, m, &gv, xin, g->hb, n_embd, nff, gb, 1, xdim, nff)
                   && enc_mv(g, m, &uv, xin, g->hb2, n_embd, nff, ub, 1, xdim, nff)
                   && enc_actmul(g, m, g->hb, g->hb2, nff);
            // Down bias must land BEFORE the routing weight scales the
            // output (CPU: out += w*(down(h)+db)). Folding w into the hidden
            // first would leave the bias unscaled, so the biased sequence
            // rides the bias through the down matvec tail and scales after.
            if (!ok) return false;
            if (!db && !enc_scale(g, g->hb, w, nff, 1, nff))   // fold weight into the hidden
                return false;
            CUdeviceptr dst = s == 0 ? aout : g->moe_eout;
            if (!enc_mv(g, m, &dv, g->hb, dst, nff, n_embd, db, 1, nff, n_embd))
                return false;
            if (db && !enc_scale(g, dst, w, n_embd, 1, n_embd))
                return false;
            if (s > 0 && !enc_add(g, aout, g->moe_eout, n_embd, 1, n_embd, n_embd))
                return false;
        }
    }
    // move the accumulated per-token outputs (in xb2) back into xb for the
    // residual add; sync first so the expert kernels have completed
    if (cu.StreamSynchronize(g->stream) != 0) return false;
    return cu.MemcpyDtoD(g->xb, g->xb2, (size_t)tn * xdim * sizeof(float)) == 0;
}

static bool gpu_moe_ffn(gpu_t *g, model_t *m, const layer_t *ly, int tn,
                        int xdim) {
    if (g->moe_fused_ok)
        return tn > 1 && g->moe_grouped_ok
                      ? gpu_moe_ffn_grouped(g, m, ly, tn, xdim)
                      : gpu_moe_ffn_fused(g, m, ly, tn, xdim);
    return gpu_moe_ffn_eager(g, m, ly, tn, xdim);
}

// gemma-4-MoE dual-branch FFN on GPU: a dense GELU shared expert AND a routed
// top-k GELU expert set, each with its own pre/post RMSNorm sandwich, summed.
// Reads the post-attention residual from g->x (UN-normed — each branch norms it
// itself), writes mlp+moe_out into g->xb; the caller then applies post_ffw_norm
// (pfn) and the residual add. Mirrors the CPU gemma_moe_ffn token-for-token.
//
// Fused variant: routing and expert selection on device (k_moe_route +
// indirect matvecs over the fused gate_up tensor), no host round-trip and no
// stream sync — capture-clean, like gpu_moe_ffn_fused.
static bool gpu_gemma_moe_ffn_fused(gpu_t *g, model_t *m, const layer_t *ly,
                                    int l, int tn, int xdim) {
    int n_embd = m->n_embd, ne = m->n_expert, used = m->n_expert_used;
    int nff = m->n_ff_exp, dff = m->n_ff;   // dff = dense shared-FFN size
    enum { MOE_MAX_USED = 256 };
    if (used > MOE_MAX_USED) used = MOE_MAX_USED;
    if (ne  > MOE_MAX_USED) ne  = MOE_MAX_USED;
    float eps = m->rms_eps;
    // fused gate_up expert block: [n_embd -> 2*nff] per expert
    uint64_t gustride = (uint64_t)(2 * (size_t)nff) *
                        ggml_row_size(ly->ffn_gate_up_exps->type, n_embd);
    uint64_t dstride  = (uint64_t)n_embd *
                        ggml_row_size(ly->ffn_down_exps->type, nff);
    CUdeviceptr xn   = g->g_scr + (size_t)0 * n_embd * sizeof(float);
    CUdeviceptr xn2  = g->g_scr + (size_t)1 * n_embd * sizeof(float);
    CUdeviceptr rin  = g->g_scr + (size_t)2 * n_embd * sizeof(float);
    CUdeviceptr mout = g->g_scr + (size_t)3 * n_embd * sizeof(float);
    for (int t = 0; t < tn; t++) {
        // g->x (the residual stream) is n_embd-strided; g->xb is xdim-strided
        CUdeviceptr attn = g->x  + (size_t)t * n_embd * sizeof(float);
        CUdeviceptr out  = g->xb + (size_t)t * xdim   * sizeof(float);
        // --- dense shared MLP: rmsnorm(ffn_norm) -> GELU SwiGLU -> post_ffw_norm_1.
        //     Written straight into `out`; the routed branch is added on top below.
        bool ok = enc_rmsnorm(g, attn, xn, g->sw->ffn_norm[l], n_embd, eps, 1, xdim, n_embd)
               && enc_mv(g, m, ly->w_gate, xn, g->hb,  n_embd, dff, 0, 1, n_embd, dff)
               && enc_mv(g, m, ly->w_up,   xn, g->hb2, n_embd, dff, 0, 1, n_embd, dff)
               && enc_actmul(g, m, g->hb, g->hb2, dff)
               && enc_mv(g, m, ly->w_down, g->hb, out, dff, n_embd, 0, 1, dff, xdim)
               && enc_rmsnorm(g, out, out, g->sw->g_pn1[l], n_embd, eps, 1, xdim, xdim);
        if (!ok) return false;
        // --- routed experts: pre-norm the branch input (pre_norm2); the router
        //     runs on a SEPARATE weightless-rmsnorm(attn) with (1/sqrt(n_embd) *
        //     gate_inp_scale) folded into the uploaded g_gis weight. Selection
        //     runs on device; per-slot scale = selw * down_exps_scale[expert].
        CUdeviceptr lg   = g->moe_logits + (size_t)t * ne * sizeof(float);
        CUdeviceptr sel  = g->moe_sel  + (size_t)t * used * sizeof(int);
        CUdeviceptr selw = g->moe_selw + (size_t)t * used * sizeof(float);
        CUdeviceptr up   = g->moe_hb + (size_t)nff * sizeof(float);
        ok = enc_rmsnorm(g, attn, xn2, g->sw->g_prn2[l], n_embd, eps, 1, xdim, n_embd)
          && enc_rmsnorm(g, attn, rin, g->sw->g_gis[l],  n_embd, eps, 1, xdim, n_embd)
          && enc_mv(g, m, ly->ffn_gate_inp, rin, lg, n_embd, ne, 0, 1, n_embd, ne)
          && enc_moe_route(g, lg, sel, selw, ne, used, 1, ne)
          && enc_moe_mv(g, m, ly->ffn_gate_up_exps, gustride, xn2, g->moe_hb,
                        sel, n_embd, 2 * nff, used, 0, 2 * nff)
          && enc_moe_actmul(g, m, g->moe_hb, up, nff, 2 * nff, 2 * nff,
                            selw, g->sw->g_dsc[l], sel, used, 0, 0)
          && enc_moe_mv(g, m, ly->ffn_down_exps, dstride, g->moe_hb,
                        g->moe_eout, sel, nff, n_embd, used, 2 * nff, n_embd)
          && enc_moe_sum(g, mout, g->moe_eout, n_embd, used, n_embd, 0, 0, 0);
        if (!ok) return false;
        // post_ffw_norm_2 on the routed branch, then out (=dense) += routed
        if (!enc_rmsnorm(g, mout, mout, g->sw->g_pn2[l], n_embd, eps, 1, n_embd, n_embd) ||
            !enc_add(g, out, mout, n_embd, 1, xdim, n_embd))
            return false;
    }
    return true;
}

static bool gpu_gemma_moe_ffn_eager(gpu_t *g, model_t *m, const layer_t *ly, int l,
                              int tn, int xdim) {
    int n_embd = m->n_embd, ne = m->n_expert, used = m->n_expert_used;
    int nff = m->n_ff_exp, dff = m->n_ff;   // dff = dense shared-FFN size
    enum { MOE_MAX_USED = 256 };
    if (used > MOE_MAX_USED) used = MOE_MAX_USED;
    if (ne  > MOE_MAX_USED) ne  = MOE_MAX_USED;
    float eps = m->rms_eps;
    CUdeviceptr xn   = g->g_scr + (size_t)0 * n_embd * sizeof(float);
    CUdeviceptr xn2  = g->g_scr + (size_t)1 * n_embd * sizeof(float);
    CUdeviceptr rin  = g->g_scr + (size_t)2 * n_embd * sizeof(float);
    CUdeviceptr mout = g->g_scr + (size_t)3 * n_embd * sizeof(float);
    for (int t = 0; t < tn; t++) {
        // g->x (the residual stream) is n_embd-strided; g->xb is xdim-strided
        CUdeviceptr attn = g->x  + (size_t)t * n_embd * sizeof(float);
        CUdeviceptr out  = g->xb + (size_t)t * xdim   * sizeof(float);
        // --- dense shared MLP: rmsnorm(ffn_norm) -> GELU SwiGLU -> post_ffw_norm_1.
        //     Written straight into `out`; the routed branch is added on top below.
        bool ok = enc_rmsnorm(g, attn, xn, g->sw->ffn_norm[l], n_embd, eps, 1, xdim, n_embd)
               && enc_mv(g, m, ly->w_gate, xn, g->hb,  n_embd, dff, 0, 1, n_embd, dff)
               && enc_mv(g, m, ly->w_up,   xn, g->hb2, n_embd, dff, 0, 1, n_embd, dff)
               && enc_actmul(g, m, g->hb, g->hb2, dff)
               && enc_mv(g, m, ly->w_down, g->hb, out, dff, n_embd, 0, 1, dff, xdim)
               && enc_rmsnorm(g, out, out, g->sw->g_pn1[l], n_embd, eps, 1, xdim, xdim);
        if (!ok) return false;
        // --- routed experts: pre-norm the branch input (pre_norm2); the router
        //     runs on a SEPARATE weightless-rmsnorm(attn) with (1/sqrt(n_embd) *
        //     gate_inp_scale) folded into the uploaded g_gis weight
        ok = enc_rmsnorm(g, attn, xn2, g->sw->g_prn2[l], n_embd, eps, 1, xdim, n_embd)
          && enc_rmsnorm(g, attn, rin, g->sw->g_gis[l],  n_embd, eps, 1, xdim, n_embd)
          && enc_mv(g, m, ly->ffn_gate_inp, rin, g->moe_logits, n_embd, ne, 0, 1, n_embd, ne);
        if (!ok) return false;
        if (cu.StreamSynchronize(g->stream) != 0) return false;
        if (cu.MemcpyDtoH(g->h_moe_logits, g->moe_logits, sizeof(float) * (size_t)ne) != 0)
            return false;
        // softmax over all experts, then top-k with renormalized weights
        float *lg = g->h_moe_logits;
        float mx = lg[0];
        for (int e = 1; e < ne; e++) if (lg[e] > mx) mx = lg[e];
        float ssum = 0.0f;
        for (int e = 0; e < ne; e++) { float p = expf(lg[e] - mx); lg[e] = p; ssum += p; }
        for (int e = 0; e < ne; e++) lg[e] /= ssum;
        int sel[MOE_MAX_USED];
        float selw[MOE_MAX_USED], denom = 0.0f;
        for (int s = 0; s < used; s++) {
            int best = 0; float bp = -1.0f;
            for (int e = 0; e < ne; e++) if (lg[e] > bp) { bp = lg[e]; best = e; }
            sel[s] = best; selw[s] = bp; denom += bp; lg[best] = -1.0f;
        }
        for (int s = 0; s < used; s++) {
            int e = sel[s];
            float sc = (selw[s] / denom) *
                       (ly->down_exps_scale ? ly->down_exps_scale[e] : 1.0f);
            // fused gate_up expert weight: [n_embd -> 2*nff], gate=[0,nff) up=[nff,2nff)
            gguf_tensor guv = *ly->ffn_gate_up_exps;
            size_t rs = ggml_row_size(guv.type, n_embd);
            guv.data = (uint8_t *)ly->ffn_gate_up_exps->data +
                       (size_t)e * (2 * (size_t)nff) * rs;
            guv.nbytes = (2 * (size_t)nff) * rs;  // clamp to the slice (see moe_expert_weight)
            gguf_tensor dv = moe_expert_weight(ly, 2, e, n_embd, nff);
            CUdeviceptr up = g->hb + (size_t)nff * sizeof(float);
            ok = enc_mv(g, m, &guv, xn2, g->hb, n_embd, 2 * nff, 0, 1, n_embd, 2 * nff)
              && enc_actmul(g, m, g->hb, up, nff)         // GELU(gate)*up -> hb[0,nff)
              && enc_scale(g, g->hb, sc, nff, 1, nff);    // fold selw*down_scale
            if (!ok) return false;
            if (s == 0) {
                if (!enc_mv(g, m, &dv, g->hb, mout, nff, n_embd, 0, 1, nff, n_embd))
                    return false;
            } else {
                if (!enc_mv(g, m, &dv, g->hb, g->moe_eout, nff, n_embd, 0, 1, nff, n_embd) ||
                    !enc_add(g, mout, g->moe_eout, n_embd, 1, n_embd, n_embd))
                    return false;
            }
        }
        // post_ffw_norm_2 on the routed branch, then out (=dense) += routed
        if (!enc_rmsnorm(g, mout, mout, g->sw->g_pn2[l], n_embd, eps, 1, n_embd, n_embd) ||
            !enc_add(g, out, mout, n_embd, 1, xdim, n_embd))
            return false;
    }
    return cu.StreamSynchronize(g->stream) == 0;
}

static bool gpu_gemma_moe_ffn(gpu_t *g, model_t *m, const layer_t *ly, int l,
                              int tn, int xdim) {
    if (g->moe_fused_ok)
        return tn > 1 && g->moe_grouped_ok
                      ? gpu_gemma_moe_ffn_grouped(g, m, ly, l, tn, xdim)
                      : gpu_gemma_moe_ffn_fused(g, m, ly, l, tn, xdim);
    return gpu_gemma_moe_ffn_eager(g, m, ly, l, tn, xdim);
}

// Qwen3.5 Gated DeltaNet layer. Learned projections use the same quant-aware
// matvec dispatch as dense attention; only convolution and the recurrent state
// transition are architecture-specific CUDA kernels. Prompt columns are
// advanced in order because both history and state are causal.
static bool gpu_q35_recurrent(gpu_t *g, model_t *m, const layer_t *ly, int l,
                              int tn, int xdim) {
    int sk = m->ssm_state, ng = m->ssm_groups, nh = m->ssm_v_heads;
    int inner = m->ssm_inner, convdim = 2 * sk * ng + inner;
    int histn = m->ssm_conv_kernel - 1;
    CUdeviceptr weight_base = 0;
    uint64_t weight_off = 0;
    if (!binding_find(g->sw, m, ly->ssm_conv, &weight_base, &weight_off))
        return false;
    CUdeviceptr conv_w = weight_base + weight_off;

    bool ok = enc_mv(g, m, ly->wqkv, g->xb, g->q35_mix,
                     m->n_embd, convdim, 0, tn, xdim, convdim)
           && enc_mv(g, m, ly->wq_gate, g->xb, g->q35_z,
                     m->n_embd, inner, 0, tn, xdim, inner)
           && enc_mv(g, m, ly->ssm_beta, g->xb, g->q35_beta,
                     m->n_embd, nh, 0, tn, xdim, nh)
           && enc_mv(g, m, ly->ssm_alpha, g->xb, g->q35_alpha,
                     m->n_embd, nh, 0, tn, xdim, nh);
    if (!ok) return false;

    size_t hist_layer = (size_t)histn * convdim;
    size_t state_layer = (size_t)nh * sk * sk;
    CUdeviceptr hist = g->q35_hist + (size_t)l * hist_layer * sizeof(float);
    CUdeviceptr state = g->q35_state + (size_t)l * state_layer * sizeof(float);
    for (int t = 0; t < tn; t++) {
        CUdeviceptr mix = g->q35_mix + (size_t)t * convdim * sizeof(float);
        CUdeviceptr cv = g->q35_cv + (size_t)t * convdim * sizeof(float);
        CUdeviceptr z = g->q35_z + (size_t)t * inner * sizeof(float);
        CUdeviceptr beta = g->q35_beta + (size_t)t * nh * sizeof(float);
        CUdeviceptr alpha = g->q35_alpha + (size_t)t * nh * sizeof(float);
        CUdeviceptr out = g->xb2 + (size_t)t * xdim * sizeof(float);
        int kernel = m->ssm_conv_kernel;
        void *pc[] = { &mix, &cv, &conv_w, &hist, &convdim, &kernel };
        if (!launch(g, g->sw->f_q35_conv, (convdim + 255) / 256,
                    1, 1, 256, pc)) return false;
        float eps = m->rms_eps;
        void *pd[] = { &cv, &z, &beta, &alpha, &g->sw->ssm_dt[l],
                       &g->sw->ssm_a[l], &g->sw->ssm_norm[l], &state,
                       &out, &sk, &ng, &nh, &eps };
        if (!launch(g, g->sw->f_q35_delta, nh, 1, 1, 128, pd)) return false;
    }
    return enc_mv(g, m, ly->ssm_out, g->xb2, g->xb, inner, m->n_embd,
                  0, tn, xdim, xdim);
}

static bool fwd_tile(gpu_t *g, model_t *m, const int32_t *tokens, int tn,
                     int pos, bool want_logits, int l0, int l1) {
    (void)tokens; (void)pos;
    int n_embd = m->n_embd;
    int xdim = n_embd;
    for (int l = 0; l < m->n_layer; l++)
        if (model_q_dim(m, l) > xdim) xdim = model_q_dim(m, l);

    bool ok = true;
    prof_mark(g, -1);   // tile anchor: delta to the first group is charged below
    for (int l = l0; l < l1 && ok; l++) {
        layer_t *ly = &m->layers[l];
        bool local  = model_is_swa(m, l);
        int hd      = model_head_dim(m, l);
        int n_kv    = model_n_head_kv(m, l);
        int q_dim   = model_q_dim(m, l);
        int kv_dim  = model_kv_dim(m, l);

        ok = ok && enc_rmsnorm(g, g->x, g->xb, g->sw->attn_norm[l], n_embd, m->rms_eps,
                               tn, n_embd, xdim);
        prof_mark(g, PH_NORM);
        if (ly->recurrent) {
            ok = ok && gpu_q35_recurrent(g, m, ly, l, tn, xdim);
            prof_mark(g, PH_MATVEC);
            goto attention_done;
        }
        if (m->qwen35) {
            ok = ok && enc_mv(g, m, ly->wq, g->xb, g->q35_mix,
                              n_embd, 2 * q_dim, g->sw->bq[l], tn, xdim,
                              2 * q_dim);
            int heads = m->n_head, packed_stride = 2 * q_dim, q_stride = q_dim;
            void *ps[] = { &g->q35_mix, &g->q, &g->q35_gate, &heads, &hd,
                           &packed_stride, &q_stride };
            ok = ok && launch(g, g->sw->f_q35_split, (q_dim + 255) / 256,
                              tn, 1, 256, ps);
        } else {
            ok = ok && enc_mv(g, m, ly->wq, g->xb, g->q, n_embd, q_dim,
                              g->sw->bq[l], tn, xdim, q_dim);
            // the output gate projects the same normed input as Q, and xb is
            // overwritten by wo below, so this must run before attention
            if (m->attn_out_gate && ly->wq_gate)
                ok = ok && enc_mv(g, m, ly->wq_gate, g->xb, g->q35_gate,
                                  n_embd, q_dim, 0, tn, xdim, q_dim);
        }
        // gemma4 E-series shared-KV layers project Q as usual but compute no
        // K/V: they attend over the cache an earlier layer filled, which the
        // aliased model_kv_byte_off() below already resolves to.
        bool owns_kv = model_kv_owner(m, l) == l;
        if (!owns_kv) goto kv_done;
        ok = ok && enc_mv(g, m, ly->wk, g->xb, g->kt, n_embd, kv_dim, g->sw->bk[l], tn, xdim, kv_dim);
        if (ly->wv) {
            ok = ok && enc_mv(g, m, ly->wv, g->xb, g->vt, n_embd, kv_dim, g->sw->bv[l], tn, xdim, kv_dim);
        } else {
            // gemma4 global layers have no V projection: V is the raw K
            // (zero + add = device-side copy with the kernels we have)
            ok = ok && cu.MemsetD8(g->vt, 0, sizeof(float) * (size_t)tn * kv_dim) == 0;
            ok = ok && enc_add(g, g->vt, g->kt, kv_dim, tn, kv_dim, kv_dim);
        }
        prof_mark(g, PH_MATVEC);
        if (m->v_rmsnorm)
            // weightless per-head RMS norm on V (pre-K-norm values)
            ok = ok && enc_qknorm(g, m, g->vt, g->sw->ones, n_kv, hd, tn, kv_dim);
        if (g->sw->kn[l]) ok = ok && enc_qknorm(g, m, g->kt, g->sw->kn[l], n_kv, hd, tn, kv_dim);
        if (model_layer_ropes(m, l))
            ok = ok && enc_rope(g, m, g->kt, n_kv, tn, kv_dim, l);

        {
            // cache rows for consecutive positions are contiguous, so the
            // kernel indexes both source column and destination row by grid.y
            uint64_t l_off = model_kv_byte_off(m, l);
            int q8 = m->kv_q8;
            // one thread per stored unit: per value for fp16, per 32-value
            // q8_0 block (the whole block shares one amax/scale)
            int units = q8 ? kv_dim / 32 : kv_dim;
            void *ps[] = { &g->kt, &g->vt, &g->kc, &g->vc, &kv_dim, &l_off,
                           &g->pos_dev, &q8 };
            ok = ok && launch(g, g->sw->f_store, (units + 63) / 64, tn, 1, 64, ps);
        }
    kv_done:
        if (g->sw->qn[l]) ok = ok && enc_qknorm(g, m, g->q,  g->sw->qn[l], m->n_head, hd, tn, q_dim);
        prof_mark(g, PH_NORM);
        ok = ok && enc_rope_or_temp(g, m, g->q, m->n_head, tn, q_dim, l,
                                    pos, enc_rope);
        prof_mark(g, PH_ROPE);

        {
            attn_args aa = { hd, m->n_head, n_kv, m->n_ctx,
                             (uint64_t)model_kv_byte_off(m, l),
                             model_attn_scale(m, l), q_dim, xdim,
                             local ? m->swa_window : 0, m->kv_q8 };
            // Decode (tn==1, one query, long KV): flash-decoding — split the KV
            // range across ATTN_SPLITS blocks/head (higher occupancy, coalesced)
            // then merge partials. Prefill (tn>1) already runs n_head*tn blocks,
            // so it keeps the single-pass k_attn. tn is the tile size (constant
            // at graph-capture time), not per-token state, so this branch is
            // capture-safe.
            // gpt-oss attention sinks for this layer (0 elsewhere): the sink
            // joins the softmax at the normalization point — inside k_attn's
            // single reduction, or at k_attn_merge's global merge (the split
            // partials carry no normalization, so k_attn_dec is untouched)
            CUdeviceptr sinks = g->sw->sinks[l];
            if (tn == 1) {
                void *pd[] = { &g->q, &g->kc, &g->vc, &g->att, &g->attn_part,
                               &aa, &g->pos_dev };
                ok = ok && launch(g, g->sw->f_attn_dec, m->n_head, ATTN_SPLITS, 1, 128, pd);
                void *pm[] = { &g->xb2, &g->attn_part, &aa, &g->pos_dev, &sinks };
                ok = ok && launch(g, g->sw->f_attn_merge, m->n_head, 1, 1, 128, pm);
            } else {
                void *pa[] = { &g->q, &g->kc, &g->vc, &g->att, &g->xb2, &aa, &g->pos_dev,
                               &sinks };
                ok = ok && launch(g, g->sw->f_attn, m->n_head, tn, 1, 128, pa);
            }
        }
        prof_mark(g, PH_ATTN);

        if (m->qwen35 || (m->attn_out_gate && ly->wq_gate)) {
            int gate_stride = q_dim;
            void *pg[] = { &g->xb2, &g->q35_gate, &q_dim, &xdim, &gate_stride };
            ok = ok && launch(g, g->sw->f_q35_gate, (q_dim + 255) / 256,
                              tn, 1, 256, pg);
        }

        ok = ok && enc_mv(g, m, ly->wo, g->xb2, g->xb, q_dim, n_embd, g->sw->bo[l], tn, xdim, xdim);
        prof_mark(g, PH_MATVEC);
    attention_done:
        if (g->sw->pan[l])
            ok = ok && enc_rmsnorm(g, g->xb, g->xb, g->sw->pan[l], n_embd, m->post_norm_eps,
                                   tn, xdim, xdim);
        if (m->resid_scale != 1.0f)  // granite muP branch scale
            ok = ok && enc_scale(g, g->xb, m->resid_scale, n_embd, tn, xdim);
        ok = ok && enc_add(g, g->x, g->xb, n_embd, tn, n_embd, xdim);
        prof_mark(g, PH_ELEM);

        if (ly->is_moe && moe_on_host(m, l)) {
            // Tensor-role boundary: attention has updated the residual and KV
            // on-device. Move only the small activation tile to the host, run
            // the sparse expert FFN against mmap-resident weights, and resume
            // the next layer on CUDA. This is the 30B-on-small-VRAM path.
            if (cu.StreamSynchronize(g->stream) != 0 ||
                cu.MemcpyDtoH(m->x, g->x,
                              sizeof(float) * (size_t)tn * n_embd) != 0 ||
                !model_moe_ffn_cpu(m, l, tn) ||
                cu.MemcpyHtoD(g->x, m->x,
                              sizeof(float) * (size_t)tn * n_embd) != 0)
                return false;
            prof_mark(g, PH_MATVEC);
            continue;
        }

        if (ly->moe_gemma) {
            // dual-branch FFN norms itself off g->x (the un-normed residual)
            ok = ok && gpu_gemma_moe_ffn(g, m, ly, l, tn, xdim);
            prof_mark(g, PH_MATVEC);
            goto ffn_done;
        }
        ok = ok && enc_rmsnorm(g, g->x, g->xb, g->sw->ffn_norm[l], n_embd, m->rms_eps,
                               tn, n_embd, xdim);
        prof_mark(g, PH_NORM);
        if (ly->is_moe) {
            ok = ok && gpu_moe_ffn(g, m, ly, tn, xdim);  // writes FFN output into g->xb
            prof_mark(g, PH_MATVEC);
        } else {
        if (!ly->w_gate) {
            ok = ok && enc_mv(g, m, ly->w_up, g->xb, g->hb,
                              n_embd, m->n_ff, 0, tn, xdim, m->n_ff);
            prof_mark(g, PH_MATVEC);
            ok = ok && enc_xielu(g, g->hb, tn * m->n_ff,
                                 m->xielu_an[l], m->xielu_ap[l],
                                 m->xielu_b[l], m->xielu_eps[l]);
            prof_mark(g, PH_ELEM);
        } else {
            ok = ok && enc_mv(g, m, ly->w_gate, g->xb, g->hb,
                              n_embd, m->n_ff, 0, tn, xdim, m->n_ff);
            ok = ok && enc_mv(g, m, ly->w_up, g->xb, g->hb2,
                              n_embd, m->n_ff, 0, tn, xdim, m->n_ff);
            prof_mark(g, PH_MATVEC);
            ok = ok && enc_actmul(g, m, g->hb, g->hb2, tn * m->n_ff);
            prof_mark(g, PH_ELEM);
        }
        ok = ok && enc_mv(g, m, ly->w_down, g->hb, g->xb, m->n_ff, n_embd, 0, tn, m->n_ff, xdim);
        prof_mark(g, PH_MATVEC);
        }
    ffn_done:
        if (g->sw->pfn[l])
            ok = ok && enc_rmsnorm(g, g->xb, g->xb, g->sw->pfn[l], n_embd, m->post_norm_eps,
                                   tn, xdim, xdim);
        if (m->resid_scale != 1.0f)
            ok = ok && enc_scale(g, g->xb, m->resid_scale, n_embd, tn, xdim);
        ok = ok && enc_add(g, g->x, g->xb, n_embd, tn, n_embd, xdim);
        ok = ok && enc_ple(g, m, ly, l, tn, xdim);
        if (ly->out_scale != 1.0f && ly->out_scale != 0.0f)
            // gemma4: whole-layer output scalar, applied after both residuals
            ok = ok && enc_scale(g, g->x, ly->out_scale, n_embd, tn, n_embd);
        prof_mark(g, PH_ELEM);
    }

    if (want_logits) {
        CUdeviceptr xlast = g->x + (size_t)(tn - 1) * n_embd * sizeof(float);
        ok = ok && enc_rmsnorm(g, xlast, g->xb, g->sw->out_norm, n_embd, m->rms_eps,
                               1, 0, 0);
        prof_mark(g, PH_NORM);
        ok = ok && enc_mv(g, m, m->output, g->xb, g->logits, n_embd, m->n_vocab, 0, 1, 0, 0);
        prof_mark(g, PH_LOGITS);
    }
    return ok;
}

// ==========================================================================
// Cross-sequence decode microbatch (Phase 6)
// ==========================================================================
//
// One decode step for N independent sequences in one pass over the weights.
//
// What makes this cheap is that almost nothing new is allocated. The weights
// are already shared (Phase 5), the activation scratch in every gpu_t is
// already MVB columns wide because prompt tiles needed it, and each sequence
// already owns its KV cache. The batch adds three small device arrays — the
// per-column positions and the two per-column KV base pointers — plus room for
// N rows of logits instead of one. Everything else is borrowed.
//
// It borrows the *lead* sequence's scratch, stream and graph slot rather than
// allocating a second set. That is a deliberate constraint, not an oversight:
// a microbatch and the lead's own solo forward would fight over the same
// buffers, so a sequence must not be decoded both ways at once. The scheduler
// this is built for is a single loop issuing one step at a time, which is
// exactly that discipline; the constraint is documented on the public API.
//
// Sequences are named per call rather than enrolled, so the membership of a
// batch can change every step at no cost — which is what makes it CONTINUOUS
// batching rather than static batching. Positions and KV pointers live in
// device memory precisely so that changing them does not invalidate a captured
// graph, and a graph is captured per microbatch width.
struct gpu_batch {
    model_t   **seqs;              // borrowed: the caller owns these
    int         n;                 // sequences enrolled at create time
    gpu_t      *lead;              // whose scratch/stream/KV machinery is used
    gpu_weights *sw;
    int         n_vocab, n_embd, xdim;
    CUdeviceptr logits_n;          // [MVB][n_vocab]
    CUdeviceptr pos_d;             // [MVB] per-column position
    CUdeviceptr kcp_d, vcp_d;      // [MVB] per-column KV base pointers
    float      *h_logits;          // [MVB][n_vocab] host staging
    int         h_pos[MVB];
    uint64_t    h_kcp[MVB], h_vcp[MVB];
    // one captured graph per microbatch width: grid shapes and mv_args.batch
    // are baked in at capture, but positions and KV pointers are not, so a
    // graph stays valid across steps and across a change of membership
    CUgraph     graph[MVB + 1];
    CUgraphExec gexec[MVB + 1];
    bool        graph_bad;
};

// Every member must be decodable in one launch against one weight upload:
// same shared weights, full offload (a partial split has to hand the boundary
// activation back to the CPU per sequence, which is a different kernel shape),
// and the batched kernels present. A no here is not a failure — the caller
// decodes these sequences one at a time, exactly as before.
static bool batch_eligible(model_t **seqs, int n, gpu_t **lead_out) {
    if (n < 1) return false;
    gpu_t *lead = NULL;
    for (int i = 0; i < n; i++) {
        model_t *m = seqs[i];
        if (!m || !m->gpu) return false;
        gpu_t *g = m->gpu;
        if (m->gpu_layers < m->n_layer) return false;
        // fwd_batch runs a dense FFN unconditionally; MoE (sparse experts or
        // the gemma-4 dual-branch) has no path there, and it neither projects
        // the attention output gate nor skips rope on nope_on_full layers —
        // keep all of those on fwd_tile
        if (m->n_expert > 0 || m->moe_gemma || m->qwen35 ||
            m->attn_out_gate || m->nope_on_full) return false;
        if (!g->sw->f_rope_seq || !g->sw->f_store_seq || !g->sw->f_attn_dec_seq)
            return false;
        if (!lead) lead = g;
        // one upload, one geometry: a member from a different registry entry
        // would index a different weight blob with this one's offsets
        else if (g->sw != lead->sw) return false;
    }
    *lead_out = lead;
    return true;
}

gpu_batch *gpu_batch_create(model_t **seqs, int n) {
    gpu_t *lead = NULL;
    if (!batch_eligible(seqs, n, &lead)) return NULL;
    model_t *m0 = seqs[0];

    gpu_batch *b = calloc(1, sizeof(gpu_batch));
    if (!b) return NULL;
    b->seqs = malloc(sizeof(model_t *) * (size_t)n);
    if (!b->seqs) { free(b); return NULL; }
    memcpy(b->seqs, seqs, sizeof(model_t *) * (size_t)n);
    b->n = n;
    b->lead = lead;
    b->sw = lead->sw;
    b->n_vocab = m0->n_vocab;
    b->n_embd  = m0->n_embd;
    b->xdim = m0->n_embd;
    for (int l = 0; l < m0->n_layer; l++)
        if (model_q_dim(m0, l) > b->xdim) b->xdim = model_q_dim(m0, l);

    if (cu.CtxSetCurrent(b->sw->ctx) != 0) goto fail;
    CK(cu.MemAlloc(&b->logits_n, sizeof(float) * (size_t)MVB * b->n_vocab));
    CK(cu.MemAlloc(&b->pos_d, sizeof(int) * MVB));
    CK(cu.MemAlloc(&b->kcp_d, sizeof(uint64_t) * MVB));
    CK(cu.MemAlloc(&b->vcp_d, sizeof(uint64_t) * MVB));
    b->h_logits = malloc(sizeof(float) * (size_t)MVB * b->n_vocab);
    if (!b->h_logits) goto fail;
    // no stream means no graphs; plain launches still work, just slower.
    // gemma4's V-copy path (ly->wv == NULL) uses a synchronous MemsetD8 that
    // cannot be captured, same as the solo decode graph refuses it.
    b->graph_bad = lead->stream == NULL || !cu_graphs_ok();
    for (int l = 0; l < m0->n_layer; l++)
        if (!m0->layers[l].wv) b->graph_bad = true;
    return b;

fail:
    gpu_batch_free(b);
    return NULL;
}

void gpu_batch_free(gpu_batch *b) {
    if (!b) return;
    if (b->sw && b->sw->ctx) cu.CtxSetCurrent(b->sw->ctx);
    for (int i = 0; i <= MVB; i++) {
        if (b->gexec[i] && cu.GraphExecDestroy) cu.GraphExecDestroy(b->gexec[i]);
        if (b->graph[i] && cu.GraphDestroy) cu.GraphDestroy(b->graph[i]);
    }
    CUdeviceptr bufs[] = { b->logits_n, b->pos_d, b->kcp_d, b->vcp_d };
    for (size_t i = 0; i < sizeof(bufs) / sizeof(*bufs); i++)
        if (bufs[i]) cu.MemFree(bufs[i]);
    free(b->h_logits);
    free(b->seqs);
    free(b);
}

// Matvec for a decode microbatch. The selection rule is the whole correctness
// argument: pick the MULTI-COLUMN TWIN of whatever kernel gpu_forward_batch
// would have picked at batch 1 — f_gemvb where f_gemv exists, f_mvb otherwise
// — so each column reproduces the lone-token result bit for bit. It must never
// reach for f_gemm: the prefill GEMM is faster and reassociates the reduction,
// which is exactly the trade this path exists to refuse.
static bool enc_mv_batch(gpu_t *g, model_t *m, gguf_tensor *w, CUdeviceptr x,
                         CUdeviceptr y, int n_in, int n_out, CUdeviceptr bias,
                         int batch, int xs, int ys) {
    CUdeviceptr weights = 0;
    uint64_t w_off = 0;
    if (!binding_find(g->sw, m, w, &weights, &w_off)) {
        gpu_first_error("tensor '%s' is not resident on the device "
                        "(no binding covers it)", w->name);
        return false;
    }
    mv_args a = { n_in, n_out, w_off, bias != 0, batch, xs, ys };
    CUdeviceptr bp = bias ? bias : g->sw->dummy;
    void *p[] = { &weights, &x, &y, &a, &bp };
    // GEMM-shaped twins stage x in shared memory and give 8 rows per block;
    // without that, MVB scattered global x-loads per decoded weight make a
    // microbatch L1-bound and it loses to running the sequences separately.
    CUfunction f = g->sw->f_gemvb[batch_width_class(batch)][w->type];
    if (f) return launch(g, f, (n_out + GEMM_WARPS - 1) / GEMM_WARPS, 1, 1,
                         GEMM_WARPS * 32, p);
    return launch(g, g->sw->f_mvb[w->type], (n_out + 3) / 4, 1, 1, 128, p);
}

static bool enc_rope_batch(gpu_batch *B, model_t *m, CUdeviceptr v, int n_heads,
                           int batch, int vs, int l) {
    gpu_t *g = B->lead;
    bool local = model_is_swa(m, l);
    rope_args a = { model_head_dim(m, l), n_heads, model_rope_dim(m, l) / 2,
                    m->rope_neox, model_rope_mscale(m, l) };
    CUdeviceptr fr = local ? g->sw->inv_freq_local : g->sw->inv_freq;
    void *p[] = { &v, &fr, &a, &B->pos_d, &vs };
    return launch(g, g->sw->f_rope_seq, (a.half_dim + 31) / 32, n_heads, batch, 32, p);
}

// The layer loop, structurally fwd_tile with tn = the microbatch width. Only
// the position-bearing and KV-bearing kernels differ; every other launch is
// the same call with a wider batch, and each already treats its columns
// independently. Logits are produced for EVERY column, not just the last —
// a microbatch has N answers, not one.
static bool fwd_batch(gpu_batch *B, model_t *m, int tn) {
    gpu_t *g = B->lead;
    int n_embd = B->n_embd, xdim = B->xdim;
    bool ok = true;

    for (int l = 0; l < m->n_layer && ok; l++) {
        layer_t *ly = &m->layers[l];
        bool local  = model_is_swa(m, l);
        int hd      = model_head_dim(m, l);
        int n_kv    = model_n_head_kv(m, l);
        int q_dim   = model_q_dim(m, l);
        int kv_dim  = model_kv_dim(m, l);

        ok = ok && enc_rmsnorm(g, g->x, g->xb, g->sw->attn_norm[l], n_embd,
                               m->rms_eps, tn, n_embd, xdim);
        ok = ok && enc_mv_batch(g, m, ly->wq, g->xb, g->q,  n_embd, q_dim,  g->sw->bq[l], tn, xdim, q_dim);
        // shared-KV layers: Q only, then attend over the owning layer's rows
        if (model_kv_owner(m, l) == l) {
        ok = ok && enc_mv_batch(g, m, ly->wk, g->xb, g->kt, n_embd, kv_dim, g->sw->bk[l], tn, xdim, kv_dim);
        if (ly->wv) {
            ok = ok && enc_mv_batch(g, m, ly->wv, g->xb, g->vt, n_embd, kv_dim, g->sw->bv[l], tn, xdim, kv_dim);
        } else {
            ok = ok && cu.MemsetD8(g->vt, 0, sizeof(float) * (size_t)tn * kv_dim) == 0;
            ok = ok && enc_add(g, g->vt, g->kt, kv_dim, tn, kv_dim, kv_dim);
        }
        if (m->v_rmsnorm)
            ok = ok && enc_qknorm(g, m, g->vt, g->sw->ones, n_kv, hd, tn, kv_dim);
        if (g->sw->kn[l]) ok = ok && enc_qknorm(g, m, g->kt, g->sw->kn[l], n_kv, hd, tn, kv_dim);
        ok = ok && enc_rope_batch(B, m, g->kt, n_kv, tn, kv_dim, l);

        {   // each column stores into its own sequence's cache at its own row
            uint64_t l_off = model_kv_byte_off(m, l);
            int q8 = m->kv_q8;
            int units = q8 ? kv_dim / 32 : kv_dim;
            void *ps[] = { &g->kt, &g->vt, &B->kcp_d, &B->vcp_d, &kv_dim,
                           &l_off, &B->pos_d, &q8 };
            ok = ok && launch(g, g->sw->f_store_seq, (units + 63) / 64, tn, 1, 64, ps);
        }
        }
        if (g->sw->qn[l]) ok = ok && enc_qknorm(g, m, g->q,  g->sw->qn[l], m->n_head, hd, tn, q_dim);
        ok = ok && enc_rope_batch(B, m, g->q, m->n_head, tn, q_dim, l);
        {   // flash-decoding over N sequences: grid (head, split, sequence).
            // The extra grid dimension is free occupancy — a solo decode step
            // leaves most of the GPU idle, which is the headroom batching eats.
            attn_args aa = { hd, m->n_head, n_kv, m->n_ctx,
                             (uint64_t)model_kv_byte_off(m, l),
                             model_attn_scale(m, l), q_dim, xdim,
                             local ? m->swa_window : 0, m->kv_q8 };
            void *pd[] = { &g->q, &B->kcp_d, &B->vcp_d, &g->att, &g->attn_part,
                           &aa, &B->pos_d };
            ok = ok && launch(g, g->sw->f_attn_dec_seq, m->n_head, ATTN_SPLITS, tn, 128, pd);
            // k_attn_merge reads neither position nor cache, so the unbatched
            // kernel serves here unchanged
            CUdeviceptr sinks = g->sw->sinks[l];
            void *pm[] = { &g->xb2, &g->attn_part, &aa, &B->pos_d, &sinks };
            ok = ok && launch(g, g->sw->f_attn_merge, m->n_head, tn, 1, 128, pm);
        }

        ok = ok && enc_mv_batch(g, m, ly->wo, g->xb2, g->xb, q_dim, n_embd, g->sw->bo[l], tn, xdim, xdim);
        if (g->sw->pan[l])
            ok = ok && enc_rmsnorm(g, g->xb, g->xb, g->sw->pan[l], n_embd, m->post_norm_eps, tn, xdim, xdim);
        if (m->resid_scale != 1.0f)
            ok = ok && enc_scale(g, g->xb, m->resid_scale, n_embd, tn, xdim);
        ok = ok && enc_add(g, g->x, g->xb, n_embd, tn, n_embd, xdim);

        ok = ok && enc_rmsnorm(g, g->x, g->xb, g->sw->ffn_norm[l], n_embd, m->rms_eps, tn, n_embd, xdim);
        ok = ok && enc_mv_batch(g, m, ly->w_gate, g->xb, g->hb,  n_embd, m->n_ff, 0, tn, xdim, m->n_ff);
        ok = ok && enc_mv_batch(g, m, ly->w_up,   g->xb, g->hb2, n_embd, m->n_ff, 0, tn, xdim, m->n_ff);
        ok = ok && enc_actmul(g, m, g->hb, g->hb2, tn * m->n_ff);
        ok = ok && enc_mv_batch(g, m, ly->w_down, g->hb, g->xb, m->n_ff, n_embd, 0, tn, m->n_ff, xdim);
        if (g->sw->pfn[l])
            ok = ok && enc_rmsnorm(g, g->xb, g->xb, g->sw->pfn[l], n_embd, m->post_norm_eps, tn, xdim, xdim);
        if (m->resid_scale != 1.0f)
            ok = ok && enc_scale(g, g->xb, m->resid_scale, n_embd, tn, xdim);
        ok = ok && enc_add(g, g->x, g->xb, n_embd, tn, n_embd, xdim);
        ok = ok && enc_ple(g, m, ly, l, tn, xdim);
        if (ly->out_scale != 1.0f && ly->out_scale != 0.0f)
            ok = ok && enc_scale(g, g->x, ly->out_scale, n_embd, tn, n_embd);
    }

    ok = ok && enc_rmsnorm(g, g->x, g->xb, g->sw->out_norm, n_embd, m->rms_eps,
                           tn, n_embd, xdim);
    ok = ok && enc_mv_batch(g, m, m->output, g->xb, B->logits_n, n_embd,
                            B->n_vocab, 0, tn, xdim, B->n_vocab);
    return ok;
}

// Stage the microbatch's token embeddings: column i is sequence idx[i]'s token.
// Identical arithmetic to stage_x, just gathering from N different callers.
static bool stage_x_batch(gpu_batch *B, model_t *m, const int32_t *tok, int n) {
    gpu_t *g = B->lead;
    size_t ers = ggml_row_size(m->tok_embd->type, m->n_embd);
    for (int b = 0; b < n; b++) {
        float *hx = g->h_x + (size_t)b * m->n_embd;
        dequant_row(m->tok_embd->type,
                    (uint8_t *)m->tok_embd->data + (size_t)tok[b] * ers,
                    hx, m->n_embd);
        model_embd_transform(m, hx);
    }
    if (cu.MemcpyHtoD(g->x, g->h_x, sizeof(float) * (size_t)n * m->n_embd) != 0)
        return false;
    return stage_ple(g, m, tok, n);
}

bool gpu_batch_decode(gpu_batch *b, const int *idx, const int32_t *tok,
                      const int *pos, int n, float **out) {
    if (!b || n < 1 || n > MVB) return false;
    // idx[] is caller-supplied and selects members of b->seqs[0, b->n): a bogus
    // index would dereference a wild model_t* below. Reject the whole batch
    // rather than launch against garbage — the caller then decodes sequentially.
    for (int i = 0; i < n; i++)
        if (idx[i] < 0 || idx[i] >= b->n) return false;
    // NoPE layers need a per-token Q scale derived from each column's
    // position, and this path keeps positions on the device. Decline rather
    // than scale with the wrong factor; the caller decodes sequentially,
    // which is the same retreat it already takes for a bad index.
    if (b->n > 0 && b->seqs[idx[0]] &&
        (b->seqs[idx[0]]->no_rope_layer_step > 0 ||
         b->seqs[idx[0]]->nope_on_full))
        return false;
    // A microbatch of one is just a decode step with extra staging, and the
    // solo path is better at it: narrower matvec kernels, x straight from
    // global, and a graph that is already warm. Hand it over rather than
    // charge a lightly loaded server for machinery it is not using.
    if (n == 1) {
        model_t *m = b->seqs[idx[0]];
        float *lg = NULL;
        if (!gpu_forward_batch(m, tok, 1, pos[0], true, &lg) || !lg) return false;
        out[0] = lg;
        return true;
    }
    gpu_t *g = b->lead;
    model_t *m0 = b->seqs[0];
    // The lead supplies the scratch and the graphs. If it fell back to the CPU
    // since this batch was created its buffers are gone, and the batch has to
    // decline rather than launch against freed memory — the caller then decodes
    // sequentially, which is where that sequence already is.
    if (m0->gpu != g) return false;
    if (cu.CtxSetCurrent(b->sw->ctx) != 0) return false;

    // Bring each sequence's device KV in line with its host cache. This is the
    // same contract gpu_forward_batch keeps — the host copy is authoritative —
    // applied per member: a sequence that decoded its previous token in this
    // same slot is already contiguous and costs nothing, while one that was
    // just prefilled on the CPU, rewound, or swapped in resyncs [0, pos).
    for (int i = 0; i < n; i++) {
        model_t *m = b->seqs[idx[i]];
        gpu_t   *gi = m->gpu;
        if (!gi || gi->sw != b->sw || m->gpu_layers < m->n_layer) return false;
        if (pos[i] < 0 || pos[i] >= m->n_ctx) return false;
        if (pos[i] != gi->last_pos + 1 && !kv_upload(gi, m, 0, pos[i])) return false;
        b->h_pos[i] = pos[i];
        b->h_kcp[i] = (uint64_t)gi->kc;
        b->h_vcp[i] = (uint64_t)gi->vc;
    }
    // pad the unused columns with column 0's sequence: the kernels compute
    // them (grid dimensions are captured into the graph) and the results are
    // discarded, but they must address real memory
    for (int i = n; i < MVB; i++) {
        b->h_pos[i] = b->h_pos[0];
        b->h_kcp[i] = b->h_kcp[0];
        b->h_vcp[i] = b->h_vcp[0];
    }

    if (!stage_x_batch(b, m0, tok, n) ||
        cu.MemcpyHtoD(b->pos_d, b->h_pos, sizeof(int) * MVB) != 0 ||
        cu.MemcpyHtoD(b->kcp_d, b->h_kcp, sizeof(uint64_t) * MVB) != 0 ||
        cu.MemcpyHtoD(b->vcp_d, b->h_vcp, sizeof(uint64_t) * MVB) != 0)
        return false;

    // Capture once per microbatch width. Launch overhead is the thing batching
    // is fighting — a solo decode step is already graph-captured, so a batched
    // step issuing ~500 plain launches would hand back much of what it won.
    bool ran = false;
    if (!b->graph_bad && !prof.on) {   // see the note on the solo path above
        if (!b->gexec[n]) {
            prof.capturing = 1;
            if (cu.StreamBeginCapture(g->stream, 1) != 0 ||
                !fwd_batch(b, m0, n) ||
                cu.StreamEndCapture(g->stream, &b->graph[n]) != 0 ||
                cu.GraphInstantiateWithFlags(&b->gexec[n], b->graph[n], 0) != 0) {
                CUgraph junk = NULL;
                cu.StreamEndCapture(g->stream, &junk);
                if (junk && junk != b->graph[n]) cu.GraphDestroy(junk);
                fprintf(stderr, "gpu: batch graph capture failed — plain launches\n");
                b->graph_bad = true;
            }
            prof.capturing = 0;
        }
        if (b->gexec[n])
            ran = cu.GraphLaunch(b->gexec[n], g->stream) == 0 &&
                  cu.StreamSynchronize(g->stream) == 0;
        if (!ran && !b->graph_bad) {
            b->graph_bad = true;
            fprintf(stderr, "gpu: batch graph launch failed — plain launches\n");
        }
    }
    if (!ran) {
        if (!fwd_batch(b, m0, n)) {
            fprintf(stderr, "gpu: batched decode failed — falling back\n");
            return false;
        }
        if (cu.CtxSynchronize() != 0) return false;
    }

    // Each sequence's newly written KV row goes back to its own host cache, so
    // any member can leave the batch and continue on the CPU or in a solo
    // forward without noticing it was ever batched.
    for (int i = 0; i < n; i++) {
        model_t *m = b->seqs[idx[i]];
        gpu_t   *gi = m->gpu;
        if (!kv_copyback(gi, m, pos[i], pos[i] + 1)) return false;
        gi->last_pos = pos[i];
    }
    if (cu.MemcpyDtoH(b->h_logits, b->logits_n,
                      sizeof(float) * (size_t)n * b->n_vocab) != 0)
        return false;
    for (int i = 0; i < n; i++)
        out[i] = b->h_logits + (size_t)i * b->n_vocab;
    return true;
}

// Fault-injection hook, debug builds and torture runs only.
// RUNNER_CUDA_INJECT_FAILURE=N fails the Nth device forward and returns its
// ordinal; unset or 0 disables it, and any non-numeric value means 1 so the
// documented `=1` spelling keeps working.
//
// N > 1 is the case worth having. The first device forward of any real
// generation is the prefill tile, so an unconditional hook can only ever test
// a failure before one token boundary has been crossed. Failing at decode is
// what exercises the rollback that matters: the qwen35 recurrent snapshot
// taken at the top of this function, and a device KV the host must resync.
//
// Single-slot only — the counter is a plain static, so a --parallel run would
// race it. That is acceptable for a hook nothing but tests sets.
static int inject_fail_ordinal(void) {
    static int budget = -1;  // -1 unread, 0 disabled
    static int seen = 0;
    if (budget < 0) {
        const char *s = getenv("RUNNER_CUDA_INJECT_FAILURE");
        budget = s ? (atoi(s) > 0 ? atoi(s) : 1) : 0;
    }
    if (budget == 0) return 0;
    seen++;
    return --budget == 0 ? seen : 0;
}

bool gpu_forward_batch(model_t *m, const int32_t *tokens, int n, int pos,
                       bool want_logits, float **logits) {
    gpu_t *g = m->gpu;
    if (logits) *logits = NULL;
    if (n < 1) return true;
    // A prior recoverable GPU failure runs gpu_disable(), which frees the
    // context and clears m->gpu. Decline rather than deref a NULL g (the
    // batch path re-checks m0->gpu for the same reason).
    if (!g) return false;

    if (cu.CtxSetCurrent(g->sw->ctx) != 0) return false;
    if (m->qwen35 && pos == 0) {
        int convdim = 2 * m->ssm_state * m->ssm_groups + m->ssm_inner;
        size_t hist_elems = (size_t)m->n_layer * (m->ssm_conv_kernel - 1) * convdim;
        size_t state_elems = (size_t)m->n_layer * m->ssm_v_heads *
                             m->ssm_state * m->ssm_state;
        if ((hist_elems && cu.MemsetD8(g->q35_hist, 0,
                                      hist_elems * sizeof(float)) != 0) ||
            cu.MemsetD8(g->q35_state, 0, state_elems * sizeof(float)) != 0)
            return false;
    }
    if (m->qwen35) {
        int convdim = 2 * m->ssm_state * m->ssm_groups + m->ssm_inner;
        size_t hist_bytes = sizeof(float) * (size_t)m->n_layer *
                            (m->ssm_conv_kernel - 1) * convdim;
        size_t state_bytes = sizeof(float) * (size_t)m->n_layer *
                             m->ssm_v_heads * m->ssm_state * m->ssm_state;
        if ((hist_bytes && cu.MemcpyDtoD(g->q35_hist_prev, g->q35_hist,
                                        hist_bytes) != 0) ||
            cu.MemcpyDtoD(g->q35_state_prev, g->q35_state, state_bytes) != 0)
            return false;
    }
    if (prof.on && !prof.inited) prof_init();
    prof.cur_mode = (n == 1) ? M_SINGLE : M_BATCH;
    int m_ = prof.cur_mode;
    double t_fwd0 = prof.on ? prof_now() : 0;

    // a non-contiguous position step means the host KV holds rows the device
    // hasn't seen (a CPU-run batch, or a reset + new prompt): resync [0, pos)
    if (pos != g->last_pos + 1) {
        if (!kv_upload(g, m, 0, pos)) return false;
    }

    bool partial = m->gpu_layers < m->n_layer;
    // !prof.on: a replayed CUDA graph never re-enters the instrumented launch
    // path, so every per-phase counter stays 0.0 while the wall clock keeps
    // running — the profiler reported `GPU-TOTAL 0.0 ms` and
    // `residual = 100% of wall` for decode that was demonstrably happening
    // (3070, 2026-08-09). Profiling therefore runs decode EAGERLY. The
    // trade-off is stated in the report header: eager decode carries the
    // launch overhead graphs exist to remove, so the absolute decode wall is
    // pessimistic and the phase SPLIT is what the profile is for.
    if (n == 1 && want_logits && !partial && !g->graph_bad && g->stream &&
        !prof.on && !getenv("RUNNER_CUDA_GRAPH_OFF")) {
        double ts0 = prof.on ? prof_now() : 0;
        if (!stage_x(g, m, tokens, 1) ||
            cu.MemcpyHtoD(g->pos_dev, &pos, sizeof(int)) != 0) return false;
        if (prof.on) prof.t_stage[m_] += prof_now() - ts0;
        if (!g->gexec) {
            prof.capturing = 1;
            // capture records the launch sequence without executing it; the
            // recorded graph is position-invariant because pos lives in
            // device memory, so one capture serves every decode token
            //
            // THREAD_LOCAL (1), not GLOBAL (0): GLOBAL capture forbids
            // synchronous CUDA calls from every other thread for the whole
            // capture window, so another slot's plain-path forward or a
            // second gpu_t warming up on its own thread would fail and
            // permanently downgrade that slot to CPU (model.c sees the
            // failed forward and clears m->gpu)
            if (cu.StreamBeginCapture(g->stream, 1) != 0 ||
                !fwd_tile(g, m, tokens, 1, pos, true, 0, m->gpu_layers) ||
                cu.StreamEndCapture(g->stream, &g->graph) != 0 ||
                cu.GraphInstantiateWithFlags(&g->gexec, g->graph, 0) != 0) {
                CUgraph junk = NULL;
                cu.StreamEndCapture(g->stream, &junk); // ensure capture ended
                if (junk && junk != g->graph) cu.GraphDestroy(junk);
                fprintf(stderr, "gpu: graph capture failed — plain launches\n");
                g->graph_bad = true;
            }
            prof.capturing = 0;
        }
        if (g->gexec) {
            double tg0 = prof.on ? prof_now() : 0;
            bool gl = cu.GraphLaunch(g->gexec, g->stream) == 0 &&
                      cu.StreamSynchronize(g->stream) == 0;
            if (prof.on) prof.t_sync[m_] += prof_now() - tg0;
            if (gl) {
                double tc0 = prof.on ? prof_now() : 0;
                if (!kv_copyback(g, m, pos, pos + 1)) return false;
                if (prof.on) prof.t_copyback[m_] += prof_now() - tc0;
                g->last_pos = pos;
                double tl0 = prof.on ? prof_now() : 0;
                if (cu.MemcpyDtoH(g->h_logits, g->logits, sizeof(float) * m->n_vocab) != 0)
                    return false;
                if (prof.on) {
                    prof.t_logitscp[m_] += prof_now() - tl0;
                    prof.n_tokens[m_] += 1; prof.n_tiles[m_] += 1;
                    prof.wall_ms[m_] += prof_now() - t_fwd0;
                    prof_flush();
                }
                // Steady-state decode returns here, so the hook has to be
                // checked on this path too — otherwise no injected failure can
                // ever land after prefill, which is the only interesting place
                // for one to land.
                int ginj = inject_fail_ordinal();
                if (ginj) {
                    fprintf(stderr, "gpu: injected CUDA runtime failure at "
                                    "device forward %d — falling back to CPU\n",
                            ginj);
                    return false;
                }
                if (logits) *logits = g->h_logits;
                return true;
            }
        }
        if (!g->graph_bad) { g->graph_bad = true;
            fprintf(stderr, "gpu: graph launch failed — plain launches\n"); }
    }

    for (int i = 0; i < n; i += MVB) {
        int tn = n - i < MVB ? n - i : MVB;
        bool last = i + tn == n;
        int p = pos + i;
        // device runs layers [0, gpu_layers); logits only when it owns the whole
        // model (partial hands the boundary activation back to the CPU)
        double ts0 = prof.on ? prof_now() : 0;
        bool sok = stage_x(g, m, tokens + i, tn) &&
                   cu.MemcpyHtoD(g->pos_dev, &p, sizeof(int)) == 0;
        if (prof.on) { prof.t_stage[m_] += prof_now() - ts0; prof.n_tiles[m_] += 1;
                       prof.n_tokens[m_] += tn; }
        if (!sok ||
            !fwd_tile(g, m, tokens + i, tn, p,
                      want_logits && last && !partial, 0, m->gpu_layers)) {
            fprintf(stderr, "gpu: kernel launch failed — falling back to CPU\n");
            return false;
        }
        int inj = inject_fail_ordinal();
        if (inj) {
            // Deterministic ownership/state-machine hook: fail after recurrent
            // state changed, so the caller must restore the pre-forward device
            // snapshot before recomputing this same tile on the CPU.
            fprintf(stderr, "gpu: injected CUDA runtime failure at device "
                            "forward %d — falling back to CPU\n", inj);
            return false;
        }
        // partial: copy this tile's post-boundary activation to the host x
        // buffer so the CPU layer loop can continue from gpu_layers
        if (partial &&
            cu.MemcpyDtoH((uint8_t *)m->x + (size_t)i * m->n_embd * sizeof(float),
                          g->x, sizeof(float) * tn * m->n_embd) != 0)
            return false;
    }
    double tsy0 = prof.on ? prof_now() : 0;
    if (cu.CtxSynchronize() != 0) {
        fprintf(stderr, "gpu: forward failed — falling back to CPU\n");
        return false;
    }
    if (prof.on) { prof.t_sync[m_] += prof_now() - tsy0; prof_flush(); }

    // keep the host KV cache authoritative: copy the offloaded layers' rows
    // back so the CPU path can take over at any point
    double tc0 = prof.on ? prof_now() : 0;
    if (!kv_copyback(g, m, pos, pos + n)) return false;
    if (prof.on) prof.t_copyback[m_] += prof_now() - tc0;
    g->last_pos = pos + n - 1;

    if (want_logits && !partial) {
        double tl0 = prof.on ? prof_now() : 0;
        if (cu.MemcpyDtoH(g->h_logits, g->logits, sizeof(float) * m->n_vocab) != 0)
            return false;
        if (prof.on) prof.t_logitscp[m_] += prof_now() - tl0;
        if (logits) *logits = g->h_logits;
    }
    if (prof.on) prof.wall_ms[m_] += prof_now() - t_fwd0;
    return true;
}
