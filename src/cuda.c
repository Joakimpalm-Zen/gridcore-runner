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
    CUresult (*PrimaryCtxRetain)(CUcontext *, CUdevice);
    CUresult (*PrimaryCtxRelease)(CUdevice);
    CUresult (*CtxSetCurrent)(CUcontext);
    CUresult (*MemGetInfo)(size_t *, size_t *);
    CUresult (*MemAlloc)(CUdeviceptr *, size_t);
    CUresult (*MemFree)(CUdeviceptr);
    CUresult (*MemcpyHtoD)(CUdeviceptr, const void *, size_t);
    CUresult (*MemcpyDtoH)(void *, CUdeviceptr, size_t);
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
    cu.PrimaryCtxRetain  = dl_sym(cu.lib, "cuDevicePrimaryCtxRetain");
    cu.PrimaryCtxRelease = sym2("cuDevicePrimaryCtxRelease");
    cu.CtxSetCurrent     = dl_sym(cu.lib, "cuCtxSetCurrent");
    cu.MemGetInfo        = sym2("cuMemGetInfo");
    cu.MemAlloc          = sym2("cuMemAlloc");
    cu.MemFree           = sym2("cuMemFree");
    cu.MemcpyHtoD        = sym2("cuMemcpyHtoD");
    cu.MemcpyDtoH        = sym2("cuMemcpyDtoH");
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
           cu.MemcpyDtoH && cu.MemsetD8 && cu.ModuleLoadData && cu.ModuleUnload &&
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

typedef struct {
    CUcontext   ctx;
    CUdevice    dev;
    CUmodule    mod;
    CUfunction  f_rmsnorm, f_qknorm, f_rope, f_store, f_attn, f_silu, f_gelu,
                f_add, f_scale;
    CUfunction  f_mv[32], f_mvb[32];    // indexed by ggml type; _b = tile variant
    CUfunction  f_gemm[32];             // prefill tiled-GEMM variants (Q8_0/Q4_K)
    CUdeviceptr weights;
    size_t      weights_len;
    CUdeviceptr kc, vc;
    CUdeviceptr x, xb, xb2, q, kt, vt, hb, hb2, att, logits;
    CUdeviceptr inv_freq, inv_freq_local, out_norm, dummy;
    CUdeviceptr ones;                   // weightless V RMS norm (gemma4)
    CUdeviceptr *attn_norm, *ffn_norm;  // per layer
    CUdeviceptr *pan, *pfn;             // gemma3 sandwich norms, may be 0
    CUdeviceptr *bq, *bk, *bv, *bo;     // per layer, may be 0
    CUdeviceptr *qn, *kn;               // qwen3 per-head q/k norms
    float       *h_x, *h_logits;        // host staging
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

typedef struct { int n_in, n_out; uint64_t w_off; int has_bias; int batch, xs, ys; } mv_args;
typedef struct { int head_dim, n_heads, half_dim, neox; float mscale; } rope_args;
typedef struct { int head_dim, n_head, n_head_kv, n_ctx; uint64_t l_off; float scale; int qs, os, window; } attn_args;

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

    CK(cu.DeviceGet(&g->dev, 0));
    CK(cu.PrimaryCtxRetain(&g->ctx, g->dev));
    CK(cu.CtxSetCurrent(g->ctx));

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
        size_t kv_bytes = m->kv_off[m->n_layer] * sizeof(f16_t);
        size_t act_bytes = sizeof(float) * (MVB * ((size_t)m->n_embd + 3 * xdim +
                           q_dim + 2 * kv_dim + 2 * m->n_ff +
                           (size_t)m->n_head * m->n_ctx) + m->n_vocab);

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
            size_t w = 0;
            for (int i = 0; i < 7; i++) if (ws[i]) w += ws[i]->nbytes;
            size_t kv = 2 * (size_t)m->n_ctx * model_kv_dim(m, l) * sizeof(f16_t);
            if (used + w + kv > vram_budget) break;
            used += w + kv;
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
        m->gpu_layers = full ? m->n_layer : G;
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
        g->weights_len = upload_len;
        // device KV holds only the offloaded layers [0, gpu_layers)
        kv_bytes = m->kv_off[m->gpu_layers] * sizeof(f16_t);

        CK(cu.ModuleLoadData(&g->mod, k_ptx_src));
        struct { CUfunction *f; const char *name; } fns[] = {
            { &g->f_rmsnorm,    "k_rmsnorm" },   { &g->f_qknorm, "k_qknorm" },
            { &g->f_rope,       "k_rope" },      { &g->f_store,  "k_store_kv" },
            { &g->f_attn,       "k_attn" },      { &g->f_silu,   "k_silu_mul" },
            { &g->f_gelu,       "k_gelu_mul" },  { &g->f_add,    "k_add" },
            { &g->f_scale,      "k_scale" },
            { &g->f_mv[T_F32],  "k_mv_f32" },    { &g->f_mv[T_F16],  "k_mv_f16" },
            { &g->f_mv[T_Q8_0], "k_mv_q8_0" },   { &g->f_mv[T_Q4_0], "k_mv_q4_0" },
            { &g->f_mv[T_Q4_1], "k_mv_q4_1" },   { &g->f_mv[T_Q5_0], "k_mv_q5_0" },
            { &g->f_mv[T_Q5_1], "k_mv_q5_1" },   { &g->f_mv[T_Q4_K], "k_mv_q4_K" },
            { &g->f_mv[T_Q5_K], "k_mv_q5_K" },   { &g->f_mv[T_Q6_K], "k_mv_q6_K" },
            { &g->f_mv[T_IQ4_NL], "k_mv_iq4_nl" }, { &g->f_mv[T_IQ4_XS], "k_mv_iq4_xs" },
            { &g->f_mvb[T_F32],  "k_mv_f32_b" },  { &g->f_mvb[T_F16],  "k_mv_f16_b" },
            { &g->f_mvb[T_Q8_0], "k_mv_q8_0_b" }, { &g->f_mvb[T_Q4_0], "k_mv_q4_0_b" },
            { &g->f_mvb[T_Q4_1], "k_mv_q4_1_b" }, { &g->f_mvb[T_Q5_0], "k_mv_q5_0_b" },
            { &g->f_mvb[T_Q5_1], "k_mv_q5_1_b" }, { &g->f_mvb[T_Q4_K], "k_mv_q4_K_b" },
            { &g->f_mvb[T_Q5_K], "k_mv_q5_K_b" }, { &g->f_mvb[T_Q6_K], "k_mv_q6_K_b" },
            { &g->f_mvb[T_IQ4_NL], "k_mv_iq4_nl_b" }, { &g->f_mvb[T_IQ4_XS], "k_mv_iq4_xs_b" },
            // prefill tiled-GEMM variants (batch>1 fast path for these formats)
            { &g->f_gemm[T_Q8_0], "k_gemm_q8_0" },
            { &g->f_gemm[T_Q4_K], "k_gemm_q4_K" },
        };
        for (size_t i = 0; i < sizeof(fns) / sizeof(*fns); i++)
            CK(cu.ModuleGetFunction(fns[i].f, g->mod, fns[i].name));

        // weights: the file bytes the offloaded layers reference (whole file
        // for a full split, a prefix for partial) so byte offsets stay valid
        CK(cu.MemAlloc(&g->weights, g->weights_len));
        CK(cu.MemcpyHtoD(g->weights, m->gf.map, g->weights_len));

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
        CK(cu.MemAlloc(&g->logits, sizeof(float) * m->n_vocab));
        CK(cu.MemAlloc(&g->dummy,  4));
        CK(cu.MemAlloc(&g->pos_dev, sizeof(int)));
        if (cu_graphs_ok() && cu.StreamCreate(&g->stream, 0) != 0)
            g->stream = NULL;
        // gemma4's V-copy path (ly->wv == NULL) uses a synchronous MemsetD8
        // that cannot be captured into a CUDA graph
        for (int l = 0; l < m->gpu_layers; l++)
            if (!m->layers[l].wv) g->graph_bad = true;

        g->inv_freq = f32_dbuf(m->rope_inv_freq, m->rope_dim / 2);
        g->inv_freq_local = f32_dbuf(m->rope_inv_freq_local, m->rope_dim_local / 2);
        g->out_norm = f32_dbuf(m->out_norm_w, m->n_embd);
        if (!g->inv_freq || !g->out_norm ||
            (m->rope_inv_freq_local && !g->inv_freq_local)) goto fail;
        if (m->v_rmsnorm) { // weightless per-head V norm: weight of ones
            float *ones = malloc(sizeof(float) * max_hd);
            if (!ones) goto fail;
            for (int i = 0; i < max_hd; i++) ones[i] = 1.0f;
            g->ones = f32_dbuf(ones, max_hd);
            free(ones);
            if (!g->ones) goto fail;
        }
        g->attn_norm = calloc(m->n_layer, sizeof(CUdeviceptr));
        g->ffn_norm  = calloc(m->n_layer, sizeof(CUdeviceptr));
        g->bq = calloc(m->n_layer, sizeof(CUdeviceptr));
        g->bk = calloc(m->n_layer, sizeof(CUdeviceptr));
        g->bv = calloc(m->n_layer, sizeof(CUdeviceptr));
        g->bo = calloc(m->n_layer, sizeof(CUdeviceptr));
        g->qn = calloc(m->n_layer, sizeof(CUdeviceptr));
        g->kn = calloc(m->n_layer, sizeof(CUdeviceptr));
        g->pan = calloc(m->n_layer, sizeof(CUdeviceptr));
        g->pfn = calloc(m->n_layer, sizeof(CUdeviceptr));
        if (!g->attn_norm || !g->ffn_norm || !g->bq || !g->bk || !g->bv ||
            !g->bo || !g->qn || !g->kn || !g->pan || !g->pfn)
            goto fail;
        for (int l = 0; l < m->n_layer; l++) {
            layer_t *ly = &m->layers[l];
            g->attn_norm[l] = f32_dbuf(ly->attn_norm_w, m->n_embd);
            g->ffn_norm[l]  = f32_dbuf(ly->ffn_norm_w, m->n_embd);
            g->bq[l] = f32_dbuf(ly->bq, model_q_dim(m, l));
            g->bk[l] = f32_dbuf(ly->bk, model_kv_dim(m, l));
            g->bv[l] = f32_dbuf(ly->bv, model_kv_dim(m, l));
            g->bo[l] = f32_dbuf(ly->bo, m->n_embd);
            g->qn[l] = f32_dbuf(ly->qnorm_w, model_head_dim(m, l));
            g->kn[l] = f32_dbuf(ly->knorm_w, model_head_dim(m, l));
            g->pan[l] = f32_dbuf(ly->post_attn_norm_w, m->n_embd);
            g->pfn[l] = f32_dbuf(ly->post_ffn_norm_w, m->n_embd);
            if (!g->attn_norm[l] || !g->ffn_norm[l] ||
                (ly->bq && !g->bq[l]) || (ly->bk && !g->bk[l]) ||
                (ly->bv && !g->bv[l]) || (ly->bo && !g->bo[l]) ||
                (ly->qnorm_w && !g->qn[l]) || (ly->knorm_w && !g->kn[l]) ||
                (ly->post_attn_norm_w && !g->pan[l]) ||
                (ly->post_ffn_norm_w && !g->pfn[l]))
                goto fail;
        }

        g->h_x      = malloc(sizeof(float) * MVB * m->n_embd);
        g->h_logits = malloc(sizeof(float) * m->n_vocab);
        if (!g->h_x || !g->h_logits) goto fail;


        char name[128] = "CUDA GPU";
        cu.DeviceGetName(name, sizeof(name), g->dev);
        if (m->gpu_layers < m->n_layer)
            fprintf(stderr, "gpu: CUDA backend on %s (%d/%d layers, %.1f GB in "
                    "VRAM; CPU runs the rest)\n", name, m->gpu_layers, m->n_layer,
                    g->weights_len / 1e9);
        else
            fprintf(stderr, "gpu: CUDA backend on %s (%.1f GB weights in VRAM)\n",
                    name, g->weights_len / 1e9);
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
    return launch(g, g->f_rmsnorm, 1, batch, 1, 256, p);
}

static bool enc_mv(gpu_t *g, model_t *m, gguf_tensor *w, CUdeviceptr x,
                   CUdeviceptr y, int n_in, int n_out, CUdeviceptr bias,
                   int batch, int xs, int ys) {
    mv_args a = { n_in, n_out,
                  (uint64_t)((uint8_t *)w->data - (uint8_t *)m->gf.map),
                  bias != 0, batch, xs, ys };
    CUdeviceptr b = bias ? bias : g->dummy;
    void *p[] = { &g->weights, &x, &y, &a, &b };
    // Prefill (batch>1) uses the tiled-GEMM variant where available (Q8_0/Q4_K):
    // GEMM_WARPS(=8) rows per block, 256 threads, x staged in shared memory.
    if (batch > 1 && g->f_gemm[w->type])
        return launch(g, g->f_gemm[w->type], (n_out + 7) / 8, 1, 1, 8 * 32, p);
    // 128 threads = 4 warps = 4 rows per block; the tile variant applies each
    // decoded weight to all columns, the single variant is faster at batch 1
    CUfunction f = batch > 1 ? g->f_mvb[w->type] : g->f_mv[w->type];
    return launch(g, f, (n_out + 3) / 4, 1, 1, 128, p);
}

static bool enc_qknorm(gpu_t *g, model_t *m, CUdeviceptr v, CUdeviceptr w,
                       int n_heads, int hd, int batch, int vs) {
    float eps = m->rms_eps;
    void *p[] = { &v, &w, &hd, &eps, &vs };
    return launch(g, g->f_qknorm, n_heads, batch, 1, 64, p);
}

static bool enc_rope(gpu_t *g, model_t *m, CUdeviceptr v, int n_heads,
                     int batch, int vs, int l) {
    // sliding-window layers rope at their own base with no YaRN scale;
    // heterogeneous archs (gemma4) also rotate fewer dims on local layers
    bool local = model_is_swa(m, l);
    rope_args a = { model_head_dim(m, l), n_heads, model_rope_dim(m, l) / 2,
                    m->rope_neox, local ? 1.0f : m->rope_mscale };
    CUdeviceptr fr = local ? g->inv_freq_local : g->inv_freq;
    void *p[] = { &v, &fr, &a, &g->pos_dev, &vs };
    return launch(g, g->f_rope, (a.half_dim + 31) / 32, n_heads, batch, 32, p);
}

static bool enc_scale(gpu_t *g, CUdeviceptr x, float s, int n, int batch, int xs) {
    void *p[] = { &x, &s, &n, &xs };
    return launch(g, g->f_scale, (n + 255) / 256, batch, 1, 256, p);
}

static bool enc_add(gpu_t *g, CUdeviceptr x, CUdeviceptr d, int n,
                    int batch, int xs, int ds) {
    void *p[] = { &x, &d, &n, &xs, &ds };
    return launch(g, g->f_add, (n + 255) / 256, batch, 1, 256, p);
}

static bool enc_actmul(gpu_t *g, model_t *m, CUdeviceptr a, CUdeviceptr b, int n) {
    void *p[] = { &a, &b, &n };
    CUfunction f = m->ffn_act == ACT_GELU ? g->f_gelu : g->f_silu;
    return launch(g, f, (n + 255) / 256, 1, 1, 256, p);
}

static void gpu_ctx_free(model_t *m, gpu_t *g) {
    if (!g) return;
    if (g->ctx) cu.CtxSetCurrent(g->ctx);
    if (g->gexec && cu.GraphExecDestroy) cu.GraphExecDestroy(g->gexec);
    if (g->graph && cu.GraphDestroy) cu.GraphDestroy(g->graph);
    if (g->stream && cu.StreamDestroy) cu.StreamDestroy(g->stream);
    for (int l = 0; l < m->n_layer; l++) {
        CUdeviceptr bufs[] = {
            g->attn_norm ? g->attn_norm[l] : 0,
            g->ffn_norm ? g->ffn_norm[l] : 0,
            g->bq ? g->bq[l] : 0, g->bk ? g->bk[l] : 0,
            g->bv ? g->bv[l] : 0, g->bo ? g->bo[l] : 0,
            g->qn ? g->qn[l] : 0, g->kn ? g->kn[l] : 0,
            g->pan ? g->pan[l] : 0, g->pfn ? g->pfn[l] : 0,
        };
        for (size_t i = 0; i < sizeof(bufs) / sizeof(*bufs); i++)
            if (bufs[i]) cu.MemFree(bufs[i]);
    }
    free(g->attn_norm); free(g->ffn_norm);
    free(g->bq); free(g->bk); free(g->bv); free(g->bo);
    free(g->qn); free(g->kn);
    free(g->pan); free(g->pfn);
    CUdeviceptr bufs[] = { g->weights, g->kc, g->vc, g->x, g->xb, g->xb2,
                           g->q, g->kt, g->vt, g->hb, g->hb2, g->att,
                           g->logits, g->inv_freq, g->inv_freq_local,
                           g->out_norm, g->dummy, g->ones, g->pos_dev };
    for (size_t i = 0; i < sizeof(bufs) / sizeof(*bufs); i++)
        if (bufs[i]) cu.MemFree(bufs[i]);
    if (g->mod) cu.ModuleUnload(g->mod);
    if (g->ctx) cu.PrimaryCtxRelease(g->dev);
    free(g->h_x); free(g->h_logits);
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

// upload host KV rows [lo, hi) for every layer (CPU prompt processing wrote them)
static bool kv_upload(gpu_t *g, model_t *m, int lo, int hi) {
    if (lo >= hi) return true;
    for (int l = 0; l < m->gpu_layers; l++) {
        size_t row = (size_t)model_kv_dim(m, l) * sizeof(f16_t);
        size_t off = m->kv_off[l] * sizeof(f16_t) + (size_t)lo * row;
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
        size_t row = (size_t)model_kv_dim(m, l) * sizeof(f16_t);
        size_t off = m->kv_off[l] * sizeof(f16_t) + (size_t)lo * row;
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

        ok = ok && enc_rmsnorm(g, g->x, g->xb, g->attn_norm[l], n_embd, m->rms_eps,
                               tn, n_embd, xdim);
        prof_mark(g, PH_NORM);
        ok = ok && enc_mv(g, m, ly->wq, g->xb, g->q,  n_embd, q_dim,  g->bq[l], tn, xdim, q_dim);
        ok = ok && enc_mv(g, m, ly->wk, g->xb, g->kt, n_embd, kv_dim, g->bk[l], tn, xdim, kv_dim);
        if (ly->wv) {
            ok = ok && enc_mv(g, m, ly->wv, g->xb, g->vt, n_embd, kv_dim, g->bv[l], tn, xdim, kv_dim);
        } else {
            // gemma4 global layers have no V projection: V is the raw K
            // (zero + add = device-side copy with the kernels we have)
            ok = ok && cu.MemsetD8(g->vt, 0, sizeof(float) * (size_t)tn * kv_dim) == 0;
            ok = ok && enc_add(g, g->vt, g->kt, kv_dim, tn, kv_dim, kv_dim);
        }
        prof_mark(g, PH_MATVEC);
        if (g->qn[l]) ok = ok && enc_qknorm(g, m, g->q,  g->qn[l], m->n_head, hd, tn, q_dim);
        if (m->v_rmsnorm)
            // weightless per-head RMS norm on V (pre-K-norm values)
            ok = ok && enc_qknorm(g, m, g->vt, g->ones, n_kv, hd, tn, kv_dim);
        if (g->kn[l]) ok = ok && enc_qknorm(g, m, g->kt, g->kn[l], n_kv, hd, tn, kv_dim);
        prof_mark(g, PH_NORM);
        ok = ok && enc_rope(g, m, g->q,  m->n_head, tn, q_dim, l);
        ok = ok && enc_rope(g, m, g->kt, n_kv,      tn, kv_dim, l);
        prof_mark(g, PH_ROPE);

        {
            // cache rows for consecutive positions are contiguous, so the
            // kernel indexes both source column and destination row by grid.y
            uint64_t l_off = m->kv_off[l];
            void *ps[] = { &g->kt, &g->vt, &g->kc, &g->vc, &kv_dim, &l_off, &g->pos_dev };
            ok = ok && launch(g, g->f_store, (kv_dim + 63) / 64, tn, 1, 64, ps);
        }

        {
            attn_args aa = { hd, m->n_head, n_kv, m->n_ctx,
                             (uint64_t)m->kv_off[l],
                             model_attn_scale(m, l), q_dim, xdim,
                             local ? m->swa_window : 0 };
            void *pa[] = { &g->q, &g->kc, &g->vc, &g->att, &g->xb2, &aa, &g->pos_dev };
            ok = ok && launch(g, g->f_attn, m->n_head, tn, 1, 128, pa);
        }
        prof_mark(g, PH_ATTN);

        ok = ok && enc_mv(g, m, ly->wo, g->xb2, g->xb, q_dim, n_embd, g->bo[l], tn, xdim, xdim);
        prof_mark(g, PH_MATVEC);
        if (g->pan[l])
            ok = ok && enc_rmsnorm(g, g->xb, g->xb, g->pan[l], n_embd, m->rms_eps,
                                   tn, xdim, xdim);
        ok = ok && enc_add(g, g->x, g->xb, n_embd, tn, n_embd, xdim);
        prof_mark(g, PH_ELEM);

        ok = ok && enc_rmsnorm(g, g->x, g->xb, g->ffn_norm[l], n_embd, m->rms_eps,
                               tn, n_embd, xdim);
        prof_mark(g, PH_NORM);
        ok = ok && enc_mv(g, m, ly->w_gate, g->xb, g->hb,  n_embd, m->n_ff, 0, tn, xdim, m->n_ff);
        ok = ok && enc_mv(g, m, ly->w_up,   g->xb, g->hb2, n_embd, m->n_ff, 0, tn, xdim, m->n_ff);
        prof_mark(g, PH_MATVEC);
        ok = ok && enc_actmul(g, m, g->hb, g->hb2, tn * m->n_ff);
        prof_mark(g, PH_ELEM);
        ok = ok && enc_mv(g, m, ly->w_down, g->hb, g->xb, m->n_ff, n_embd, 0, tn, m->n_ff, xdim);
        prof_mark(g, PH_MATVEC);
        if (g->pfn[l])
            ok = ok && enc_rmsnorm(g, g->xb, g->xb, g->pfn[l], n_embd, m->rms_eps,
                                   tn, xdim, xdim);
        ok = ok && enc_add(g, g->x, g->xb, n_embd, tn, n_embd, xdim);
        if (ly->out_scale != 1.0f && ly->out_scale != 0.0f)
            // gemma4: whole-layer output scalar, applied after both residuals
            ok = ok && enc_scale(g, g->x, ly->out_scale, n_embd, tn, n_embd);
        prof_mark(g, PH_ELEM);
    }

    if (want_logits) {
        CUdeviceptr xlast = g->x + (size_t)(tn - 1) * n_embd * sizeof(float);
        ok = ok && enc_rmsnorm(g, xlast, g->xb, g->out_norm, n_embd, m->rms_eps,
                               1, 0, 0);
        prof_mark(g, PH_NORM);
        ok = ok && enc_mv(g, m, m->output, g->xb, g->logits, n_embd, m->n_vocab, 0, 1, 0, 0);
        prof_mark(g, PH_LOGITS);
    }
    return ok;
}

bool gpu_forward_batch(model_t *m, const int32_t *tokens, int n, int pos,
                       bool want_logits, float **logits) {
    gpu_t *g = m->gpu;
    if (logits) *logits = NULL;
    if (n < 1) return true;

    if (cu.CtxSetCurrent(g->ctx) != 0) return false;
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
