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

#include <math.h>
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
                f_add, f_scale;
    CUfunction  f_attn_dec, f_attn_merge;   // flash-decoding attention (decode)
    CUfunction  f_mv[32], f_mvb[32];    // indexed by ggml type; _b = tile variant
    CUfunction  f_gemm[32];             // prefill tiled-GEMM variants (Q8_0/Q4_K)
    CUfunction  f_gemm_tc[32];          // tensor-core prefill GEMM (opt-in, Q4_K)
    CUfunction  f_gemv[32];             // decode coalesced GEMV variants (Q8_0/Q4_K)
    // cross-sequence decode microbatch (Phase 6): per-column position and KV
    CUfunction  f_rope_seq, f_store_seq, f_attn_dec_seq;
    // multi-column twins of f_gemv, per microbatch width (see BW_* below):
    // f_gemvb[w][type], w indexing the width class, not the width itself
    CUfunction  f_gemvb[BW_N][32];
    CUdeviceptr weights;
    size_t      weights_len;
    int         gpu_layers;             // split decided by the first loader
    CUdeviceptr inv_freq, inv_freq_local, out_norm, dummy;
    CUdeviceptr ones;                   // weightless V RMS norm (gemma4)
    CUdeviceptr *attn_norm, *ffn_norm;  // per layer
    CUdeviceptr *pan, *pfn;             // gemma3 sandwich norms, may be 0
    CUdeviceptr *bq, *bk, *bv, *bo;     // per layer, may be 0
    CUdeviceptr *qn, *kn;               // qwen3 per-head q/k norms
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
    CUdeviceptr moe_logits, moe_eout;   // sparse-MoE: router logits, expert down-out
    float       *h_x, *h_logits, *h_moe_logits; // host staging
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
#define MVB 8

// flash-decoding KV split count — must match ATTN_SPLITS in kernels.cu; fixed
// (not context-dependent) so the decode CUDA graph stays valid across positions
#define ATTN_SPLITS 8

// output rows per batched-matvec block — must match GEMM_WARPS in kernels.cu
#define GEMM_WARPS 8
#define TC_WPB     4   // tensor-core GEMM warps per block (match kernels.cu)

typedef struct { int n_in, n_out; uint64_t w_off; int has_bias; int batch, xs, ys; } mv_args;
typedef struct { int head_dim, n_heads, half_dim, neox; float mscale; } rope_args;
// l_off is a BYTE offset: the cache is fp16 or q8_0 depending on m->kv_q8,
// so element indexing is not enough (see the kv storage block in kernels.cu)
typedef struct { int head_dim, n_head, n_head_kv, n_ctx; uint64_t l_off; float scale; int qs, os, window, q8; } attn_args;

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

static bool gpu_type_ok(int type) {
    switch (type) {
        case T_F32: case T_F16: case T_Q8_0: case T_Q4_0: case T_Q4_1:
        case T_Q5_0: case T_Q5_1: case T_Q4_K: case T_Q5_K: case T_Q6_K:
        case T_IQ4_NL: case T_IQ4_XS:
            return true;
        default:
            return false;
    }
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

bool gpu_kv_q8_ok(void) {
    return true;    // k_store_kv / k_attn / k_attn_dec all read q8_0 blocks
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
// must not be served out of the previous upload.
static bool file_id(const char *path, uint64_t *size, uint64_t *ino,
                    int64_t *mtime) {
    struct stat st;
    if (!path || stat(path, &st) != 0) return false;
    *size  = (uint64_t)st.st_size;
    *ino   = (uint64_t)st.st_ino;
    *mtime = (int64_t)st.st_mtime;
    return true;
}

// Everything the shared buffers are derived from, compared rather than assumed
// from the path. The rope tables in particular are not a property of the file
// alone: YaRN auto-extension keys off the requested context, phi3 picks its
// LongRoPE factor set the same way, and --rope-base/--rope-scale override both.
// Two loads of one file can legitimately want different tables, and a miss here
// is not an error — that instance just gets its own upload, as before.
static bool shared_matches(const gpu_weights *w, const model_t *m,
                           uint64_t size, uint64_t ino, int64_t mtime) {
    if (!w->path || !m->path || strcmp(w->path, m->path) != 0) return false;
    if (w->fsize != size || w->fino != ino || w->fmtime != mtime) return false;
    if (w->n_layer != m->n_layer || w->n_embd != m->n_embd ||
        w->n_head  != m->n_head  || w->n_head_kv != m->n_head_kv ||
        w->head_dim != m->head_dim || w->n_ff != m->n_ff ||
        w->n_vocab != m->n_vocab || w->n_ctx != m->n_ctx ||
        w->rope_dim != m->rope_dim || w->rope_dim_local != m->rope_dim_local ||
        w->kv_q8 != (int)m->kv_q8 || w->v_rmsnorm != (int)m->v_rmsnorm ||
        w->rope_base != m->rope_base || w->rope_mscale != m->rope_mscale)
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
        };
        for (size_t i = 0; i < sizeof(bufs) / sizeof(*bufs); i++)
            if (bufs[i]) cu.MemFree(bufs[i]);
    }
    free(w->attn_norm); free(w->ffn_norm);
    free(w->bq); free(w->bk); free(w->bv); free(w->bo);
    free(w->qn); free(w->kn);
    free(w->pan); free(w->pfn);
    CUdeviceptr bufs[] = { w->weights, w->inv_freq, w->inv_freq_local,
                           w->out_norm, w->dummy, w->ones };
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
        size_t fixed = act_bytes + m->tok_embd->nbytes + (512u << 20);
        // decide how many *leading* layers fit — accumulate each layer's weight
        // bytes plus its KV bytes until the budget runs out; the CPU runs the
        // rest (partial offload)
        int G = 0;
        size_t used = fixed;
        for (int l = 0; l < m->n_layer; l++) {
            layer_t *ly = &m->layers[l];
            gguf_tensor *ws[] = { ly->wq, ly->wk, ly->wv, ly->wo,
                                  ly->w_gate, ly->w_up, ly->w_down };
            size_t wb = 0;
            for (int i = 0; i < 7; i++) if (ws[i]) wb += ws[i]->nbytes;
            // KV bytes honour the cache format: a q8_0 cache is ~53% of fp16,
            // so quantized KV directly buys more offloaded layers here
            size_t kv = 2 * (size_t)m->n_ctx * model_kv_row_bytes(m, l);
            if (used + wb + kv > vram_budget) break;
            used += wb + kv;
            G = l + 1;
        }
        // a full split also needs token_embd + output weights resident
        bool full = G == m->n_layer &&
                    used + m->tok_embd->nbytes + m->output->nbytes <= vram_budget;
        if (G == 0) {
            fprintf(stderr,
                    "gpu: not even one layer fits %.1f GB %s VRAM — using CPU\n",
                    vram_budget / 1e9, m->reserve_vram_pct > 0 ? "reserved" : "free");
            goto fail_quiet;
        }
        fprintf(stderr, "gpu-split: budget=%.2fGB fixed=%.2fGB G=%d/%d full=%d used=%.2fGB\n", vram_budget/1e9, fixed/1e9, G, m->n_layer, (int)full, used/1e9);
        w->gpu_layers = full ? m->n_layer : G;
        // weight bytes to upload: whole file for a full split (output/embedding
        // offsets stay valid), else the prefix covering token_embd + [0, G)
        size_t upload_len = m->gf.map_size;
        if (!full) {
            upload_len = (size_t)((uint8_t *)m->tok_embd->data - (uint8_t *)m->gf.map)
                       + m->tok_embd->nbytes;
            for (int l = 0; l < G; l++) {
                layer_t *ly = &m->layers[l];
                gguf_tensor *ws[] = { ly->wq, ly->wk, ly->wv, ly->wo,
                                      ly->w_gate, ly->w_up, ly->w_down };
                for (int i = 0; i < 7; i++) {
                    if (!ws[i]) continue;
                    size_t end = (size_t)((uint8_t *)ws[i]->data - (uint8_t *)m->gf.map)
                               + ws[i]->nbytes;
                    if (end > upload_len) upload_len = end;
                }
            }
        }
        w->weights_len = upload_len;

        CK(cu.ModuleLoadData(&w->mod, k_ptx_src));
        struct { CUfunction *f; const char *name; } fns[] = {
            { &w->f_rmsnorm,    "k_rmsnorm" },   { &w->f_qknorm, "k_qknorm" },
            { &w->f_rope,       "k_rope" },      { &w->f_store,  "k_store_kv" },
            { &w->f_attn,       "k_attn" },      { &w->f_silu,   "k_silu_mul" },
            { &w->f_attn_dec,   "k_attn_dec" },  { &w->f_attn_merge, "k_attn_merge" },
            { &w->f_gelu,       "k_gelu_mul" },  { &w->f_add,    "k_add" },
            { &w->f_scale,      "k_scale" },
            { &w->f_mv[T_F32],  "k_mv_f32" },    { &w->f_mv[T_F16],  "k_mv_f16" },
            { &w->f_mv[T_Q8_0], "k_mv_q8_0" },   { &w->f_mv[T_Q4_0], "k_mv_q4_0" },
            { &w->f_mv[T_Q4_1], "k_mv_q4_1" },   { &w->f_mv[T_Q5_0], "k_mv_q5_0" },
            { &w->f_mv[T_Q5_1], "k_mv_q5_1" },   { &w->f_mv[T_Q4_K], "k_mv_q4_K" },
            { &w->f_mv[T_Q5_K], "k_mv_q5_K" },   { &w->f_mv[T_Q6_K], "k_mv_q6_K" },
            { &w->f_mv[T_IQ4_NL], "k_mv_iq4_nl" }, { &w->f_mv[T_IQ4_XS], "k_mv_iq4_xs" },
            { &w->f_mvb[T_F32],  "k_mv_f32_b" },  { &w->f_mvb[T_F16],  "k_mv_f16_b" },
            { &w->f_mvb[T_Q8_0], "k_mv_q8_0_b" }, { &w->f_mvb[T_Q4_0], "k_mv_q4_0_b" },
            { &w->f_mvb[T_Q4_1], "k_mv_q4_1_b" }, { &w->f_mvb[T_Q5_0], "k_mv_q5_0_b" },
            { &w->f_mvb[T_Q5_1], "k_mv_q5_1_b" }, { &w->f_mvb[T_Q4_K], "k_mv_q4_K_b" },
            { &w->f_mvb[T_Q5_K], "k_mv_q5_K_b" }, { &w->f_mvb[T_Q6_K], "k_mv_q6_K_b" },
            { &w->f_mvb[T_IQ4_NL], "k_mv_iq4_nl_b" }, { &w->f_mvb[T_IQ4_XS], "k_mv_iq4_xs_b" },
            // prefill tiled-GEMM variants (batch>1 fast path for these formats)
            { &w->f_gemm[T_Q8_0], "k_gemm_q8_0" },
            { &w->f_gemm[T_Q4_K], "k_gemm_q4_K" },
            { &w->f_gemm[T_Q5_K], "k_gemm_q5_K" },
            { &w->f_gemm[T_Q6_K], "k_gemm_q6_K" },
            // decode coalesced GEMV variants (batch==1 fast path for these formats)
            { &w->f_gemv[T_Q8_0], "k_gemv_q8_0" },
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
        };
        for (size_t i = 0; i < sizeof(fns) / sizeof(*fns); i++)
            CK(cu.ModuleGetFunction(fns[i].f, w->mod, fns[i].name));
        // Tensor-core kernels: resolved non-fatally so an older embedded PTX
        // (without them) still loads — TC just stays unavailable there.
        cu.ModuleGetFunction(&w->f_gemm_tc[T_Q4_K], w->mod, "k_gemm_q4_K_tc");

        // weights: the file bytes the offloaded layers reference (whole file
        // for a full split, a prefix for partial) so byte offsets stay valid.
        // Every instance sharing this upload indexes it with offsets computed
        // against its own mmap base, which is the same layout by construction.
        CK(cu.MemAlloc(&w->weights, w->weights_len));
        CK(cu.MemcpyHtoD(w->weights, m->gf.map, w->weights_len));
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
        if (!w->attn_norm || !w->ffn_norm || !w->bq || !w->bk || !w->bv ||
            !w->bo || !w->qn || !w->kn || !w->pan || !w->pfn)
            goto fail;
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
            if (!w->attn_norm[l] || !w->ffn_norm[l] ||
                (ly->bq && !w->bq[l]) || (ly->bk && !w->bk[l]) ||
                (ly->bv && !w->bv[l]) || (ly->bo && !w->bo[l]) ||
                (ly->qnorm_w && !w->qn[l]) || (ly->knorm_w && !w->kn[l]) ||
                (ly->post_attn_norm_w && !w->pan[l]) ||
                (ly->post_ffn_norm_w && !w->pfn[l]))
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
    bool have_id = file_id(m->path, &fsize, &fino, &fmtime);
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
        // an entry with no file identity is private: never listed, so it is
        // never matched, and shared_release still frees it at refs 0
        if (w && have_id) { w->next = g_shared; g_shared = w; }
    }
    pthread_mutex_unlock(&g_shared_mu);
    return w;
}

bool gpu_init(model_t *m) {
    // every weight matmul must have a kernel for its quant type (wv may be
    // absent: gemma4 global layers reuse the raw K projection as V)
    if (!gpu_type_ok(m->output->type)) goto unsupported;
    for (int l = 0; l < m->n_layer; l++) {
        layer_t *ly = &m->layers[l];
        gguf_tensor *ws[] = { ly->wq, ly->wk, ly->wv, ly->wo,
                              ly->w_gate, ly->w_up, ly->w_down };
        for (int i = 0; i < 7; i++)
            if (ws[i] && !gpu_type_ok(ws[i]->type)) goto unsupported;
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
        size_t act_bytes = sizeof(float) * (MVB * ((size_t)m->n_embd + 3 * xdim +
                           q_dim + 2 * kv_dim + 2 * m->n_ff +
                           (size_t)m->n_head * m->n_ctx) + m->n_vocab);

        // the weights (and the CPU/GPU split they imply) come from the registry:
        // a second instance of the same file reuses the first one's upload
        g->sw = shared_acquire(m, act_bytes, max_hd);
        if (!g->sw) goto fail_quiet;
        m->gpu_layers = g->sw->gpu_layers;
        CK(cu.CtxSetCurrent(g->sw->ctx));

        // device KV holds only the offloaded layers [0, gpu_layers)
        size_t kv_bytes = model_kv_byte_off(m, m->gpu_layers);
        CK(cu.MemAlloc(&g->kc, kv_bytes));
        CK(cu.MemAlloc(&g->vc, kv_bytes));
        CK(cu.MemsetD8(g->kc, 0, kv_bytes));
        CK(cu.MemsetD8(g->vc, 0, kv_bytes));

        CK(cu.MemAlloc(&g->x,      sizeof(float) * MVB * m->n_embd));
        CK(cu.MemAlloc(&g->xb,     sizeof(float) * MVB * xdim));
        CK(cu.MemAlloc(&g->xb2,    sizeof(float) * MVB * xdim));
        CK(cu.MemAlloc(&g->q,      sizeof(float) * MVB * q_dim));
        CK(cu.MemAlloc(&g->kt,     sizeof(float) * MVB * kv_dim));
        CK(cu.MemAlloc(&g->vt,     sizeof(float) * MVB * kv_dim));
        CK(cu.MemAlloc(&g->hb,     sizeof(float) * MVB * m->n_ff));
        CK(cu.MemAlloc(&g->hb2,    sizeof(float) * MVB * m->n_ff));
        CK(cu.MemAlloc(&g->att,    sizeof(float) * MVB * (size_t)m->n_head * m->n_ctx));
        // flash-decoding partials: per (token, head, split) -> hd weighted-V + max + sum
        CK(cu.MemAlloc(&g->attn_part, sizeof(float) * MVB * (size_t)m->n_head *
                                      ATTN_SPLITS * (max_hd + 2)));
        CK(cu.MemAlloc(&g->logits, sizeof(float) * m->n_vocab));
        CK(cu.MemAlloc(&g->pos_dev, sizeof(int)));
        if (m->n_expert > 0) {
            // sparse-MoE router logits + one expert's down output
            CK(cu.MemAlloc(&g->moe_logits, sizeof(float) * (size_t)m->n_expert));
            CK(cu.MemAlloc(&g->moe_eout, sizeof(float) * (size_t)m->n_embd));
        }
        if (cu_graphs_ok() && cu.StreamCreate(&g->stream, 0) != 0)
            g->stream = NULL;
        // gemma4's V-copy path (ly->wv == NULL) uses a synchronous MemsetD8
        // that cannot be captured into a CUDA graph; MoE routing reads router
        // logits back to the host mid-forward, which likewise cannot be captured
        for (int l = 0; l < m->gpu_layers; l++)
            if (!m->layers[l].wv || m->layers[l].is_moe) g->graph_bad = true;

        g->h_x      = malloc(sizeof(float) * MVB * m->n_embd);
        g->h_logits = malloc(sizeof(float) * m->n_vocab);
        if (m->n_expert > 0)
            g->h_moe_logits = malloc(sizeof(float) * (size_t)m->n_expert);
        if (!g->h_x || !g->h_logits) goto fail;

        char name[128] = "CUDA GPU";
        cu.DeviceGetName(name, sizeof(name), g->sw->dev);
        if (m->gpu_layers < m->n_layer)
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
                    2.0 * model_kv_byte_off(m, m->gpu_layers) / 1e9,
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

static bool launch(gpu_t *g, CUfunction f, unsigned gx, unsigned gy, unsigned gz,
                   unsigned bx, void **params) {
    if (prof.on) prof.total_launches[prof.cur_mode]++;
    return cu.LaunchKernel(f, gx, gy, gz, bx, 1, 1, 0, g->stream, params, NULL) == 0;
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
        if (cu.EventCreate(&prof.ev[i], 0) != 0) { prof.on = 0; return; }
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

// Tensor-core GEMM opt-in (RUNNER_CUDA_TC). Read once. Phase 1 of the TC plan;
// stays off by default until the tolerance gate promotes it per (type, arch).
static bool tc_on(void) {
    static int v = -1;
    if (v < 0) v = getenv("RUNNER_CUDA_TC") != NULL;
    return v != 0;
}

static bool enc_mv(gpu_t *g, model_t *m, gguf_tensor *w, CUdeviceptr x,
                   CUdeviceptr y, int n_in, int n_out, CUdeviceptr bias,
                   int batch, int xs, int ys) {
    mv_args a = { n_in, n_out,
                  (uint64_t)((uint8_t *)w->data - (uint8_t *)m->gf.map),
                  bias != 0, batch, xs, ys };
    CUdeviceptr b = bias ? bias : g->sw->dummy;
    void *p[] = { &g->sw->weights, &x, &y, &a, &b };
    // Prefill (batch>1), tensor-core GEMM when enabled and available: one warp
    // per 32-row tile, TC_WPB warps per block.
    if (batch > 1 && tc_on() && g->sw->f_gemm_tc[w->type])
        return launch(g, g->sw->f_gemm_tc[w->type],
                      (n_out + 32 * TC_WPB - 1) / (32 * TC_WPB), 1, 1, 32 * TC_WPB, p);
    // Prefill (batch>1) uses the tiled-GEMM variant where available (Q8_0/Q4_K):
    // GEMM_WARPS(=8) rows per block, 256 threads, x staged in shared memory.
    if (batch > 1 && g->sw->f_gemm[w->type])
        return launch(g, g->sw->f_gemm[w->type], (n_out + 7) / 8, 1, 1, 8 * 32, p);
    // Decode (batch==1) uses the coalesced lane-per-element GEMV where available
    // (Q8_0/Q4_K); same 4-rows/block shape, so capture-compatible with no
    // host-side branching on per-token state.
    if (batch == 1 && g->sw->f_gemv[w->type])
        return launch(g, g->sw->f_gemv[w->type], (n_out + 3) / 4, 1, 1, 128, p);
    // 128 threads = 4 warps = 4 rows per block; the tile variant applies each
    // decoded weight to all columns, the single variant is faster at batch 1
    CUfunction f = batch > 1 ? g->sw->f_mvb[w->type] : g->sw->f_mv[w->type];
    return launch(g, f, (n_out + 3) / 4, 1, 1, 128, p);
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
                    m->rope_neox, local ? 1.0f : m->rope_mscale };
    CUdeviceptr fr = local ? g->sw->inv_freq_local : g->sw->inv_freq;
    void *p[] = { &v, &fr, &a, &g->pos_dev, &vs };
    return launch(g, g->sw->f_rope, (a.half_dim + 31) / 32, n_heads, batch, 32, p);
}

static bool enc_scale(gpu_t *g, CUdeviceptr x, float s, int n, int batch, int xs) {
    void *p[] = { &x, &s, &n, &xs };
    return launch(g, g->sw->f_scale, (n + 255) / 256, batch, 1, 256, p);
}

static bool enc_add(gpu_t *g, CUdeviceptr x, CUdeviceptr d, int n,
                    int batch, int xs, int ds) {
    void *p[] = { &x, &d, &n, &xs, &ds };
    return launch(g, g->sw->f_add, (n + 255) / 256, batch, 1, 256, p);
}

static bool enc_actmul(gpu_t *g, model_t *m, CUdeviceptr a, CUdeviceptr b, int n) {
    void *p[] = { &a, &b, &n };
    CUfunction f = m->ffn_act == ACT_GELU ? g->sw->f_gelu : g->sw->f_silu;
    return launch(g, f, (n + 255) / 256, 1, 1, 256, p);
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
                           g->moe_logits, g->moe_eout };
    for (size_t i = 0; i < sizeof(bufs) / sizeof(*bufs); i++)
        if (bufs[i]) cu.MemFree(bufs[i]);
    free(g->h_x); free(g->h_logits); free(g->h_moe_logits);
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
static bool stage_x(gpu_t *g, model_t *m, const int32_t *tokens, int tn) {
    size_t ers = ggml_row_size(m->tok_embd->type, m->n_embd);
    for (int b = 0; b < tn; b++) {
        float *hx = g->h_x + (size_t)b * m->n_embd;
        dequant_row(m->tok_embd->type,
                    (uint8_t *)m->tok_embd->data + (size_t)tokens[b] * ers,
                    hx, m->n_embd);
        if (m->embd_scale != 1.0f)
            for (int i = 0; i < m->n_embd; i++) hx[i] *= m->embd_scale;
    }
    return cu.MemcpyHtoD(g->x, g->h_x, sizeof(float) * tn * m->n_embd) == 0;
}

// encode a tile of up to MVB tokens: matvecs read each weight once for the
// whole tile, and every other kernel takes the token column from the launch
// grid — a tile costs the same number of launches as a single token, which
// matters because WDDM launch overhead dominates small-kernel dispatch. The
// vocab-logits matvec (the single most expensive launch) runs only when the
// caller wants logits, and only for the tile's last token. The caller stages
// x (and, for graph capture, uploads pos) before calling this.
// Sparse-MoE FFN on the GPU for `tn` tokens. Reads the RMS-normed input from
// g->xb (stride xdim) and writes the FFN output back into g->xb. Routing —
// softmax over ALL experts, top-k, renormalize (Mixtral/Qwen3) — runs on the
// host from router logits read back per token; the per-expert SwiGLU matmuls
// run on the GPU by pointing enc_mv at each expert's slice of the fused 3D
// tensors. The whole model file is uploaded as one buffer, so a slice's mmap
// offset IS its device offset (that is what enc_mv keys on). Host-dependent
// routing is why MoE layers force the eager path (graph_bad).
static bool gpu_moe_ffn(gpu_t *g, model_t *m, const layer_t *ly, int tn, int xdim) {
    int n_embd = m->n_embd, ne = m->n_expert, used = m->n_expert_used, nff = m->n_ff_exp;
    for (int t = 0; t < tn; t++) {
        CUdeviceptr xin  = g->xb  + (size_t)t * xdim * sizeof(float);
        CUdeviceptr aout = g->xb2 + (size_t)t * xdim * sizeof(float);
        if (!enc_mv(g, m, ly->ffn_gate_inp, xin, g->moe_logits, n_embd, ne, 0, 1, xdim, ne))
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
        int sel[256];
        float selw[256], denom = 0.0f;
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
        for (int s = 0; s < used; s++) {
            int e = sel[s];
            float w = selw[s] / denom;
            gguf_tensor gv = moe_expert_weight(ly, 0, e, n_embd, nff);
            gguf_tensor uv = moe_expert_weight(ly, 1, e, n_embd, nff);
            gguf_tensor dv = moe_expert_weight(ly, 2, e, n_embd, nff);
            bool ok = enc_mv(g, m, &gv, xin, g->hb, n_embd, nff, 0, 1, xdim, nff)
                   && enc_mv(g, m, &uv, xin, g->hb2, n_embd, nff, 0, 1, xdim, nff)
                   && enc_actmul(g, m, g->hb, g->hb2, nff)
                   && enc_scale(g, g->hb, w, nff, 1, nff);       // fold weight into the hidden
            if (!ok) return false;
            if (s == 0) {
                if (!enc_mv(g, m, &dv, g->hb, aout, nff, n_embd, 0, 1, nff, n_embd))
                    return false;
            } else {
                if (!enc_mv(g, m, &dv, g->hb, g->moe_eout, nff, n_embd, 0, 1, nff, n_embd) ||
                    !enc_add(g, aout, g->moe_eout, n_embd, 1, n_embd, n_embd))
                    return false;
            }
        }
    }
    // move the accumulated per-token outputs (in xb2) back into xb for the
    // residual add; sync first so the expert kernels have completed
    if (cu.StreamSynchronize(g->stream) != 0) return false;
    return cu.MemcpyDtoD(g->xb, g->xb2, (size_t)tn * xdim * sizeof(float)) == 0;
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
        ok = ok && enc_mv(g, m, ly->wq, g->xb, g->q,  n_embd, q_dim,  g->sw->bq[l], tn, xdim, q_dim);
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
        if (g->sw->qn[l]) ok = ok && enc_qknorm(g, m, g->q,  g->sw->qn[l], m->n_head, hd, tn, q_dim);
        if (m->v_rmsnorm)
            // weightless per-head RMS norm on V (pre-K-norm values)
            ok = ok && enc_qknorm(g, m, g->vt, g->sw->ones, n_kv, hd, tn, kv_dim);
        if (g->sw->kn[l]) ok = ok && enc_qknorm(g, m, g->kt, g->sw->kn[l], n_kv, hd, tn, kv_dim);
        prof_mark(g, PH_NORM);
        ok = ok && enc_rope(g, m, g->q,  m->n_head, tn, q_dim, l);
        ok = ok && enc_rope(g, m, g->kt, n_kv,      tn, kv_dim, l);
        prof_mark(g, PH_ROPE);

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
            if (tn == 1) {
                void *pd[] = { &g->q, &g->kc, &g->vc, &g->att, &g->attn_part,
                               &aa, &g->pos_dev };
                ok = ok && launch(g, g->sw->f_attn_dec, m->n_head, ATTN_SPLITS, 1, 128, pd);
                void *pm[] = { &g->xb2, &g->attn_part, &aa, &g->pos_dev };
                ok = ok && launch(g, g->sw->f_attn_merge, m->n_head, 1, 1, 128, pm);
            } else {
                void *pa[] = { &g->q, &g->kc, &g->vc, &g->att, &g->xb2, &aa, &g->pos_dev };
                ok = ok && launch(g, g->sw->f_attn, m->n_head, tn, 1, 128, pa);
            }
        }
        prof_mark(g, PH_ATTN);

        ok = ok && enc_mv(g, m, ly->wo, g->xb2, g->xb, q_dim, n_embd, g->sw->bo[l], tn, xdim, xdim);
        prof_mark(g, PH_MATVEC);
        if (g->sw->pan[l])
            ok = ok && enc_rmsnorm(g, g->xb, g->xb, g->sw->pan[l], n_embd, m->rms_eps,
                                   tn, xdim, xdim);
        ok = ok && enc_add(g, g->x, g->xb, n_embd, tn, n_embd, xdim);
        prof_mark(g, PH_ELEM);

        ok = ok && enc_rmsnorm(g, g->x, g->xb, g->sw->ffn_norm[l], n_embd, m->rms_eps,
                               tn, n_embd, xdim);
        prof_mark(g, PH_NORM);
        if (ly->is_moe) {
            ok = ok && gpu_moe_ffn(g, m, ly, tn, xdim);  // writes FFN output into g->xb
            prof_mark(g, PH_MATVEC);
        } else {
        ok = ok && enc_mv(g, m, ly->w_gate, g->xb, g->hb,  n_embd, m->n_ff, 0, tn, xdim, m->n_ff);
        ok = ok && enc_mv(g, m, ly->w_up,   g->xb, g->hb2, n_embd, m->n_ff, 0, tn, xdim, m->n_ff);
        prof_mark(g, PH_MATVEC);
        ok = ok && enc_actmul(g, m, g->hb, g->hb2, tn * m->n_ff);
        prof_mark(g, PH_ELEM);
        ok = ok && enc_mv(g, m, ly->w_down, g->hb, g->xb, m->n_ff, n_embd, 0, tn, m->n_ff, xdim);
        prof_mark(g, PH_MATVEC);
        }
        if (g->sw->pfn[l])
            ok = ok && enc_rmsnorm(g, g->xb, g->xb, g->sw->pfn[l], n_embd, m->rms_eps,
                                   tn, xdim, xdim);
        ok = ok && enc_add(g, g->x, g->xb, n_embd, tn, n_embd, xdim);
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
    mv_args a = { n_in, n_out,
                  (uint64_t)((uint8_t *)w->data - (uint8_t *)m->gf.map),
                  bias != 0, batch, xs, ys };
    CUdeviceptr bp = bias ? bias : g->sw->dummy;
    void *p[] = { &g->sw->weights, &x, &y, &a, &bp };
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
                    m->rope_neox, local ? 1.0f : m->rope_mscale };
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
        ok = ok && enc_mv_batch(g, m, ly->wk, g->xb, g->kt, n_embd, kv_dim, g->sw->bk[l], tn, xdim, kv_dim);
        if (ly->wv) {
            ok = ok && enc_mv_batch(g, m, ly->wv, g->xb, g->vt, n_embd, kv_dim, g->sw->bv[l], tn, xdim, kv_dim);
        } else {
            ok = ok && cu.MemsetD8(g->vt, 0, sizeof(float) * (size_t)tn * kv_dim) == 0;
            ok = ok && enc_add(g, g->vt, g->kt, kv_dim, tn, kv_dim, kv_dim);
        }
        if (g->sw->qn[l]) ok = ok && enc_qknorm(g, m, g->q,  g->sw->qn[l], m->n_head, hd, tn, q_dim);
        if (m->v_rmsnorm)
            ok = ok && enc_qknorm(g, m, g->vt, g->sw->ones, n_kv, hd, tn, kv_dim);
        if (g->sw->kn[l]) ok = ok && enc_qknorm(g, m, g->kt, g->sw->kn[l], n_kv, hd, tn, kv_dim);
        ok = ok && enc_rope_batch(B, m, g->q,  m->n_head, tn, q_dim, l);
        ok = ok && enc_rope_batch(B, m, g->kt, n_kv,      tn, kv_dim, l);

        {   // each column stores into its own sequence's cache at its own row
            uint64_t l_off = model_kv_byte_off(m, l);
            int q8 = m->kv_q8;
            int units = q8 ? kv_dim / 32 : kv_dim;
            void *ps[] = { &g->kt, &g->vt, &B->kcp_d, &B->vcp_d, &kv_dim,
                           &l_off, &B->pos_d, &q8 };
            ok = ok && launch(g, g->sw->f_store_seq, (units + 63) / 64, tn, 1, 64, ps);
        }
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
            void *pm[] = { &g->xb2, &g->attn_part, &aa, &B->pos_d };
            ok = ok && launch(g, g->sw->f_attn_merge, m->n_head, tn, 1, 128, pm);
        }

        ok = ok && enc_mv_batch(g, m, ly->wo, g->xb2, g->xb, q_dim, n_embd, g->sw->bo[l], tn, xdim, xdim);
        if (g->sw->pan[l])
            ok = ok && enc_rmsnorm(g, g->xb, g->xb, g->sw->pan[l], n_embd, m->rms_eps, tn, xdim, xdim);
        ok = ok && enc_add(g, g->x, g->xb, n_embd, tn, n_embd, xdim);

        ok = ok && enc_rmsnorm(g, g->x, g->xb, g->sw->ffn_norm[l], n_embd, m->rms_eps, tn, n_embd, xdim);
        ok = ok && enc_mv_batch(g, m, ly->w_gate, g->xb, g->hb,  n_embd, m->n_ff, 0, tn, xdim, m->n_ff);
        ok = ok && enc_mv_batch(g, m, ly->w_up,   g->xb, g->hb2, n_embd, m->n_ff, 0, tn, xdim, m->n_ff);
        ok = ok && enc_actmul(g, m, g->hb, g->hb2, tn * m->n_ff);
        ok = ok && enc_mv_batch(g, m, ly->w_down, g->hb, g->xb, m->n_ff, n_embd, 0, tn, m->n_ff, xdim);
        if (g->sw->pfn[l])
            ok = ok && enc_rmsnorm(g, g->xb, g->xb, g->sw->pfn[l], n_embd, m->rms_eps, tn, xdim, xdim);
        ok = ok && enc_add(g, g->x, g->xb, n_embd, tn, n_embd, xdim);
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
        if (m->embd_scale != 1.0f)
            for (int i = 0; i < m->n_embd; i++) hx[i] *= m->embd_scale;
    }
    return cu.MemcpyHtoD(g->x, g->h_x, sizeof(float) * (size_t)n * m->n_embd) == 0;
}

bool gpu_batch_decode(gpu_batch *b, const int *idx, const int32_t *tok,
                      const int *pos, int n, float **out) {
    if (!b || n < 1 || n > MVB) return false;
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
    if (!b->graph_bad) {
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

bool gpu_forward_batch(model_t *m, const int32_t *tokens, int n, int pos,
                       bool want_logits, float **logits) {
    gpu_t *g = m->gpu;
    if (logits) *logits = NULL;
    if (n < 1) return true;

    if (cu.CtxSetCurrent(g->sw->ctx) != 0) return false;
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
    if (n == 1 && want_logits && !partial && !g->graph_bad && g->stream &&
        !getenv("RUNNER_CUDA_GRAPH_OFF")) {
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
