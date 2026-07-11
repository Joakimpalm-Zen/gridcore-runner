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
} cu;

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
    return cu.Init && cu.DeviceGetCount && cu.DeviceGet && cu.DeviceGetName &&
           cu.PrimaryCtxRetain && cu.PrimaryCtxRelease && cu.CtxSetCurrent &&
           cu.MemGetInfo && cu.MemAlloc && cu.MemFree && cu.MemcpyHtoD &&
           cu.MemcpyDtoH && cu.MemsetD8 && cu.ModuleLoadData && cu.ModuleUnload &&
           cu.ModuleGetFunction && cu.LaunchKernel && cu.CtxSynchronize;
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
} gpu_t;

// max tokens per matvec tile — must match MVB in kernels.cu; every activation
// buffer is allocated this many columns wide
#define MVB 8

typedef struct { int n_in, n_out; uint64_t w_off; int has_bias; int batch, xs, ys; } mv_args;
typedef struct { int head_dim, n_heads, half_dim, pos, neox; float mscale; } rope_args;
typedef struct { int head_dim, n_head, n_head_kv, n_ctx, pos; uint64_t l_off; float scale; int qs, os, window; } attn_args;

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

#define CK(call) do { CUresult _r = (call); if (_r != 0) { \
    fprintf(stderr, "gpu: %s failed: %s\n", #call, cu_err(_r)); goto fail; } } while (0)

static CUdeviceptr f32_dbuf(const float *src, size_t n) {
    if (!src) return 0;
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
    int ndev = 0;
    if (cu.DeviceGetCount(&ndev) != 0 || ndev < 1) return false;

    gpu_t *g = calloc(1, sizeof(gpu_t));
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

        g->inv_freq = f32_dbuf(m->rope_inv_freq, m->rope_dim / 2);
        g->inv_freq_local = f32_dbuf(m->rope_inv_freq_local, m->rope_dim_local / 2);
        g->out_norm = f32_dbuf(m->out_norm_w, m->n_embd);
        if (m->v_rmsnorm) { // weightless per-head V norm: weight of ones
            float *ones = malloc(sizeof(float) * max_hd);
            for (int i = 0; i < max_hd; i++) ones[i] = 1.0f;
            g->ones = f32_dbuf(ones, max_hd);
            free(ones);
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
        }

        g->h_x      = malloc(sizeof(float) * MVB * m->n_embd);
        g->h_logits = malloc(sizeof(float) * m->n_vocab);


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
    if (g->mod) cu.ModuleUnload(g->mod);
    if (g->ctx) cu.PrimaryCtxRelease(g->dev);
    free(g);
    return false;

unsupported:
    fprintf(stderr, "gpu: model uses a quant type without a CUDA kernel — using CPU\n");
    return false;
}

// ----------------------------------------------------------------- launches

static bool launch(CUfunction f, unsigned gx, unsigned gy, unsigned gz,
                   unsigned bx, void **params) {
    return cu.LaunchKernel(f, gx, gy, gz, bx, 1, 1, 0, NULL, params, NULL) == 0;
}

static bool enc_rmsnorm(gpu_t *g, CUdeviceptr x, CUdeviceptr y, CUdeviceptr w,
                        int n, float eps, int batch, int xs, int ys) {
    void *p[] = { &x, &y, &w, &n, &eps, &xs, &ys };
    return launch(g->f_rmsnorm, 1, batch, 1, 256, p);
}

static bool enc_mv(gpu_t *g, model_t *m, gguf_tensor *w, CUdeviceptr x,
                   CUdeviceptr y, int n_in, int n_out, CUdeviceptr bias,
                   int batch, int xs, int ys) {
    mv_args a = { n_in, n_out,
                  (uint64_t)((uint8_t *)w->data - (uint8_t *)m->gf.map),
                  bias != 0, batch, xs, ys };
    CUdeviceptr b = bias ? bias : g->dummy;
    void *p[] = { &g->weights, &x, &y, &a, &b };
    // 128 threads = 4 warps = 4 rows per block; the tile variant applies each
    // decoded weight to all columns, the single variant is faster at batch 1
    CUfunction f = batch > 1 ? g->f_mvb[w->type] : g->f_mv[w->type];
    return launch(f, (n_out + 3) / 4, 1, 1, 128, p);
}

static bool enc_qknorm(gpu_t *g, model_t *m, CUdeviceptr v, CUdeviceptr w,
                       int n_heads, int hd, int batch, int vs) {
    float eps = m->rms_eps;
    void *p[] = { &v, &w, &hd, &eps, &vs };
    return launch(g->f_qknorm, n_heads, batch, 1, 64, p);
}

static bool enc_rope(gpu_t *g, model_t *m, CUdeviceptr v, int n_heads, int pos,
                     int batch, int vs, int l) {
    // sliding-window layers rope at their own base with no YaRN scale;
    // heterogeneous archs (gemma4) also rotate fewer dims on local layers
    bool local = model_is_swa(m, l);
    rope_args a = { model_head_dim(m, l), n_heads, model_rope_dim(m, l) / 2,
                    pos, m->rope_neox, local ? 1.0f : m->rope_mscale };
    CUdeviceptr fr = local ? g->inv_freq_local : g->inv_freq;
    void *p[] = { &v, &fr, &a, &vs };
    return launch(g->f_rope, (a.half_dim + 31) / 32, n_heads, batch, 32, p);
}

static bool enc_scale(gpu_t *g, CUdeviceptr x, float s, int n, int batch, int xs) {
    void *p[] = { &x, &s, &n, &xs };
    return launch(g->f_scale, (n + 255) / 256, batch, 1, 256, p);
}

static bool enc_add(gpu_t *g, CUdeviceptr x, CUdeviceptr d, int n,
                    int batch, int xs, int ds) {
    void *p[] = { &x, &d, &n, &xs, &ds };
    return launch(g->f_add, (n + 255) / 256, batch, 1, 256, p);
}

static bool enc_actmul(gpu_t *g, model_t *m, CUdeviceptr a, CUdeviceptr b, int n) {
    void *p[] = { &a, &b, &n };
    CUfunction f = m->ffn_act == ACT_GELU ? g->f_gelu : g->f_silu;
    return launch(f, (n + 255) / 256, 1, 1, 256, p);
}

void gpu_free(model_t *m) {
    gpu_t *g = m->gpu;
    if (!g) return;
    cu.CtxSetCurrent(g->ctx);
    for (int l = 0; l < m->n_layer; l++) {
        CUdeviceptr bufs[] = { g->attn_norm[l], g->ffn_norm[l], g->bq[l],
                               g->bk[l], g->bv[l], g->bo[l], g->qn[l], g->kn[l],
                               g->pan[l], g->pfn[l] };
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
                           g->out_norm, g->dummy, g->ones };
    for (size_t i = 0; i < sizeof(bufs) / sizeof(*bufs); i++)
        if (bufs[i]) cu.MemFree(bufs[i]);
    if (g->mod) cu.ModuleUnload(g->mod);
    cu.PrimaryCtxRelease(g->dev);
    free(g->h_x); free(g->h_logits);
    free(g);
    m->gpu = NULL;
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

// encode a tile of up to MVB tokens: matvecs read each weight once for the
// whole tile, and every other kernel takes the token column from the launch
// grid — a tile costs the same number of launches as a single token, which
// matters because WDDM launch overhead dominates small-kernel dispatch. The
// vocab-logits matvec (the single most expensive launch) runs only when the
// caller wants logits, and only for the tile's last token.
static bool fwd_tile(gpu_t *g, model_t *m, const int32_t *tokens, int tn,
                     int pos, bool want_logits, int l0, int l1) {
    int n_embd = m->n_embd;
    int xdim = n_embd;
    for (int l = 0; l < m->n_layer; l++)
        if (model_q_dim(m, l) > xdim) xdim = model_q_dim(m, l);

    // when starting past the embedding (a partial split's device prefix runs
    // [0, l1); this always starts at l0=0), the caller has already staged x
    size_t ers = ggml_row_size(m->tok_embd->type, n_embd);
    if (l0 == 0) for (int b = 0; b < tn; b++) {
        float *hx = g->h_x + (size_t)b * n_embd;
        dequant_row(m->tok_embd->type,
                    (uint8_t *)m->tok_embd->data + (size_t)tokens[b] * ers,
                    hx, n_embd);
        if (m->embd_scale != 1.0f)
            for (int i = 0; i < n_embd; i++) hx[i] *= m->embd_scale;
    }
    if (l0 == 0 && cu.MemcpyHtoD(g->x, g->h_x, sizeof(float) * tn * n_embd) != 0)
        return false;

    bool ok = true;
    for (int l = l0; l < l1 && ok; l++) {
        layer_t *ly = &m->layers[l];
        bool local  = model_is_swa(m, l);
        int hd      = model_head_dim(m, l);
        int n_kv    = model_n_head_kv(m, l);
        int q_dim   = model_q_dim(m, l);
        int kv_dim  = model_kv_dim(m, l);

        ok = ok && enc_rmsnorm(g, g->x, g->xb, g->attn_norm[l], n_embd, m->rms_eps,
                               tn, n_embd, xdim);
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
        if (g->qn[l]) ok = ok && enc_qknorm(g, m, g->q,  g->qn[l], m->n_head, hd, tn, q_dim);
        if (m->v_rmsnorm)
            // weightless per-head RMS norm on V (pre-K-norm values)
            ok = ok && enc_qknorm(g, m, g->vt, g->ones, n_kv, hd, tn, kv_dim);
        if (g->kn[l]) ok = ok && enc_qknorm(g, m, g->kt, g->kn[l], n_kv, hd, tn, kv_dim);
        ok = ok && enc_rope(g, m, g->q,  m->n_head, pos, tn, q_dim, l);
        ok = ok && enc_rope(g, m, g->kt, n_kv,      pos, tn, kv_dim, l);

        {
            // cache rows for consecutive positions are contiguous, so the
            // kernel indexes both source column and destination row by grid.y
            uint64_t kv_off = (uint64_t)m->kv_off[l] + (uint64_t)pos * kv_dim;
            void *ps[] = { &g->kt, &g->vt, &g->kc, &g->vc, &kv_dim, &kv_off };
            ok = ok && launch(g->f_store, (kv_dim + 63) / 64, tn, 1, 64, ps);
        }

        {
            attn_args aa = { hd, m->n_head, n_kv, m->n_ctx, pos,
                             (uint64_t)m->kv_off[l],
                             model_attn_scale(m, l), q_dim, xdim,
                             local ? m->swa_window : 0 };
            void *pa[] = { &g->q, &g->kc, &g->vc, &g->att, &g->xb2, &aa };
            ok = ok && launch(g->f_attn, m->n_head, tn, 1, 128, pa);
        }

        ok = ok && enc_mv(g, m, ly->wo, g->xb2, g->xb, q_dim, n_embd, g->bo[l], tn, xdim, xdim);
        if (g->pan[l])
            ok = ok && enc_rmsnorm(g, g->xb, g->xb, g->pan[l], n_embd, m->rms_eps,
                                   tn, xdim, xdim);
        ok = ok && enc_add(g, g->x, g->xb, n_embd, tn, n_embd, xdim);

        ok = ok && enc_rmsnorm(g, g->x, g->xb, g->ffn_norm[l], n_embd, m->rms_eps,
                               tn, n_embd, xdim);
        ok = ok && enc_mv(g, m, ly->w_gate, g->xb, g->hb,  n_embd, m->n_ff, 0, tn, xdim, m->n_ff);
        ok = ok && enc_mv(g, m, ly->w_up,   g->xb, g->hb2, n_embd, m->n_ff, 0, tn, xdim, m->n_ff);
        ok = ok && enc_actmul(g, m, g->hb, g->hb2, tn * m->n_ff);
        ok = ok && enc_mv(g, m, ly->w_down, g->hb, g->xb, m->n_ff, n_embd, 0, tn, m->n_ff, xdim);
        if (g->pfn[l])
            ok = ok && enc_rmsnorm(g, g->xb, g->xb, g->pfn[l], n_embd, m->rms_eps,
                                   tn, xdim, xdim);
        ok = ok && enc_add(g, g->x, g->xb, n_embd, tn, n_embd, xdim);
        if (ly->out_scale != 1.0f && ly->out_scale != 0.0f)
            // gemma4: whole-layer output scalar, applied after both residuals
            ok = ok && enc_scale(g, g->x, ly->out_scale, n_embd, tn, n_embd);
    }

    if (want_logits) {
        CUdeviceptr xlast = g->x + (size_t)(tn - 1) * n_embd * sizeof(float);
        ok = ok && enc_rmsnorm(g, xlast, g->xb, g->out_norm, n_embd, m->rms_eps,
                               1, 0, 0);
        ok = ok && enc_mv(g, m, m->output, g->xb, g->logits, n_embd, m->n_vocab, 0, 1, 0, 0);
    }
    return ok;
}

bool gpu_forward_batch(model_t *m, const int32_t *tokens, int n, int pos,
                       bool want_logits, float **logits) {
    gpu_t *g = m->gpu;
    if (logits) *logits = NULL;
    if (n < 1) return true;

    if (cu.CtxSetCurrent(g->ctx) != 0) return false;

    // a non-contiguous position step means the host KV holds rows the device
    // hasn't seen (a CPU-run batch, or a reset + new prompt): resync [0, pos)
    if (pos != g->last_pos + 1) {
        if (!kv_upload(g, m, 0, pos)) return false;
    }

    bool partial = m->gpu_layers < m->n_layer;
    for (int i = 0; i < n; i += MVB) {
        int tn = n - i < MVB ? n - i : MVB;
        bool last = i + tn == n;
        // device runs layers [0, gpu_layers); logits only when it owns the whole
        // model (partial hands the boundary activation back to the CPU)
        if (!fwd_tile(g, m, tokens + i, tn, pos + i,
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
    if (cu.CtxSynchronize() != 0) {
        fprintf(stderr, "gpu: forward failed — falling back to CPU\n");
        return false;
    }

    // keep the host KV cache authoritative: copy the offloaded layers' rows
    // back so the CPU path can take over at any point
    if (!kv_copyback(g, m, pos, pos + n)) return false;
    g->last_pos = pos + n - 1;

    if (want_logits && !partial) {
        if (cu.MemcpyDtoH(g->h_logits, g->logits, sizeof(float) * m->n_vocab) != 0)
            return false;
        if (logits) *logits = g->h_logits;
    }
    return true;
}
