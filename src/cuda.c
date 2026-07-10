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
    CUfunction  f_rmsnorm, f_qknorm, f_rope, f_store, f_attn, f_silu, f_add;
    CUfunction  f_mv[32];               // indexed by ggml type
    CUdeviceptr weights;
    CUdeviceptr kc, vc;
    CUdeviceptr x, xb, xb2, q, kt, vt, hb, hb2, att, logits;
    CUdeviceptr inv_freq, out_norm, dummy;
    CUdeviceptr *attn_norm, *ffn_norm;  // per layer
    CUdeviceptr *bq, *bk, *bv, *bo;     // per layer, may be 0
    CUdeviceptr *qn, *kn;               // qwen3 per-head q/k norms
    float       *h_x, *h_logits;        // host staging
    int          last_pos;              // -2 = nothing synced yet
} gpu_t;

typedef struct { int n_in, n_out; uint64_t w_off; int has_bias; } mv_args;
typedef struct { int head_dim, n_heads, half_dim, pos, neox; float mscale; } rope_args;
typedef struct { int head_dim, n_head, n_head_kv, n_ctx, pos; uint64_t l_off; float scale; } attn_args;

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

static bool gpu_type_ok(int type) {
    switch (type) {
        case T_F32: case T_F16: case T_Q8_0: case T_Q4_0: case T_Q4_1:
        case T_Q5_0: case T_Q5_1: case T_Q4_K: case T_Q5_K: case T_Q6_K:
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
    // every weight matmul must have a kernel for its quant type
    if (!gpu_type_ok(m->output->type)) goto unsupported;
    for (int l = 0; l < m->n_layer; l++) {
        layer_t *ly = &m->layers[l];
        gguf_tensor *ws[] = { ly->wq, ly->wk, ly->wv, ly->wo,
                              ly->w_gate, ly->w_up, ly->w_down };
        for (int i = 0; i < 7; i++)
            if (!gpu_type_ok(ws[i]->type)) goto unsupported;
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
        int q_dim  = m->n_head * m->head_dim;
        int kv_dim = m->n_head_kv * m->head_dim;
        int xdim   = q_dim > m->n_embd ? q_dim : m->n_embd;
        size_t kv_bytes = (size_t)m->n_layer * m->n_ctx * kv_dim * sizeof(f16_t);
        size_t act_bytes = sizeof(float) * ((size_t)m->n_embd + 3 * xdim +
                           q_dim + 2 * kv_dim + 2 * m->n_ff +
                           (size_t)m->n_head * m->n_ctx + m->n_vocab);

        size_t vram_free = 0, vram_total = 0;
        CK(cu.MemGetInfo(&vram_free, &vram_total));
        size_t need = m->gf.map_size + 2 * kv_bytes + act_bytes + (256u << 20);
        if (need > vram_free) {
            fprintf(stderr,
                    "gpu: model needs %.1f GB VRAM, %.1f GB free — using CPU\n",
                    need / 1e9, vram_free / 1e9);
            goto fail_quiet;
        }

        CK(cu.ModuleLoadData(&g->mod, k_ptx_src));
        struct { CUfunction *f; const char *name; } fns[] = {
            { &g->f_rmsnorm,    "k_rmsnorm" },   { &g->f_qknorm, "k_qknorm" },
            { &g->f_rope,       "k_rope" },      { &g->f_store,  "k_store_kv" },
            { &g->f_attn,       "k_attn" },      { &g->f_silu,   "k_silu_mul" },
            { &g->f_add,        "k_add" },
            { &g->f_mv[T_F32],  "k_mv_f32" },    { &g->f_mv[T_F16],  "k_mv_f16" },
            { &g->f_mv[T_Q8_0], "k_mv_q8_0" },   { &g->f_mv[T_Q4_0], "k_mv_q4_0" },
            { &g->f_mv[T_Q4_1], "k_mv_q4_1" },   { &g->f_mv[T_Q5_0], "k_mv_q5_0" },
            { &g->f_mv[T_Q5_1], "k_mv_q5_1" },   { &g->f_mv[T_Q4_K], "k_mv_q4_K" },
            { &g->f_mv[T_Q5_K], "k_mv_q5_K" },   { &g->f_mv[T_Q6_K], "k_mv_q6_K" },
        };
        for (size_t i = 0; i < sizeof(fns) / sizeof(*fns); i++)
            CK(cu.ModuleGetFunction(fns[i].f, g->mod, fns[i].name));

        // weights: one copy of the whole mapped file so tensor byte offsets
        // stay valid on the device
        CK(cu.MemAlloc(&g->weights, m->gf.map_size));
        CK(cu.MemcpyHtoD(g->weights, m->gf.map, m->gf.map_size));

        CK(cu.MemAlloc(&g->kc, kv_bytes));
        CK(cu.MemAlloc(&g->vc, kv_bytes));
        CK(cu.MemsetD8(g->kc, 0, kv_bytes));
        CK(cu.MemsetD8(g->vc, 0, kv_bytes));

        CK(cu.MemAlloc(&g->x,      sizeof(float) * m->n_embd));
        CK(cu.MemAlloc(&g->xb,     sizeof(float) * xdim));
        CK(cu.MemAlloc(&g->xb2,    sizeof(float) * xdim));
        CK(cu.MemAlloc(&g->q,      sizeof(float) * q_dim));
        CK(cu.MemAlloc(&g->kt,     sizeof(float) * kv_dim));
        CK(cu.MemAlloc(&g->vt,     sizeof(float) * kv_dim));
        CK(cu.MemAlloc(&g->hb,     sizeof(float) * m->n_ff));
        CK(cu.MemAlloc(&g->hb2,    sizeof(float) * m->n_ff));
        CK(cu.MemAlloc(&g->att,    sizeof(float) * (size_t)m->n_head * m->n_ctx));
        CK(cu.MemAlloc(&g->logits, sizeof(float) * m->n_vocab));
        CK(cu.MemAlloc(&g->dummy,  4));

        g->inv_freq = f32_dbuf(m->rope_inv_freq, m->rope_dim / 2);
        g->out_norm = f32_dbuf(m->out_norm_w, m->n_embd);
        g->attn_norm = calloc(m->n_layer, sizeof(CUdeviceptr));
        g->ffn_norm  = calloc(m->n_layer, sizeof(CUdeviceptr));
        g->bq = calloc(m->n_layer, sizeof(CUdeviceptr));
        g->bk = calloc(m->n_layer, sizeof(CUdeviceptr));
        g->bv = calloc(m->n_layer, sizeof(CUdeviceptr));
        g->bo = calloc(m->n_layer, sizeof(CUdeviceptr));
        g->qn = calloc(m->n_layer, sizeof(CUdeviceptr));
        g->kn = calloc(m->n_layer, sizeof(CUdeviceptr));
        for (int l = 0; l < m->n_layer; l++) {
            layer_t *ly = &m->layers[l];
            g->attn_norm[l] = f32_dbuf(ly->attn_norm_w, m->n_embd);
            g->ffn_norm[l]  = f32_dbuf(ly->ffn_norm_w, m->n_embd);
            g->bq[l] = f32_dbuf(ly->bq, q_dim);
            g->bk[l] = f32_dbuf(ly->bk, kv_dim);
            g->bv[l] = f32_dbuf(ly->bv, kv_dim);
            g->bo[l] = f32_dbuf(ly->bo, m->n_embd);
            g->qn[l] = f32_dbuf(ly->qnorm_w, m->head_dim);
            g->kn[l] = f32_dbuf(ly->knorm_w, m->head_dim);
        }

        g->h_x      = malloc(sizeof(float) * m->n_embd);
        g->h_logits = malloc(sizeof(float) * m->n_vocab);


        char name[128] = "CUDA GPU";
        cu.DeviceGetName(name, sizeof(name), g->dev);
        fprintf(stderr, "gpu: CUDA backend on %s (%.1f GB weights in VRAM)\n",
                name, m->gf.map_size / 1e9);
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

static bool launch(CUfunction f, unsigned gx, unsigned gy, unsigned bx,
                   void **params) {
    return cu.LaunchKernel(f, gx, gy, 1, bx, 1, 1, 0, NULL, params, NULL) == 0;
}

static bool enc_rmsnorm(gpu_t *g, CUdeviceptr x, CUdeviceptr y, CUdeviceptr w,
                        int n, float eps) {
    void *p[] = { &x, &y, &w, &n, &eps };
    return launch(g->f_rmsnorm, 1, 1, 256, p);
}

static bool enc_mv(gpu_t *g, model_t *m, gguf_tensor *w, CUdeviceptr x,
                   CUdeviceptr y, int n_in, int n_out, CUdeviceptr bias) {
    mv_args a = { n_in, n_out,
                  (uint64_t)((uint8_t *)w->data - (uint8_t *)m->gf.map),
                  bias != 0 };
    CUdeviceptr b = bias ? bias : g->dummy;
    void *p[] = { &g->weights, &x, &y, &a, &b };
    // 128 threads = 4 warps = 4 rows per block
    return launch(g->f_mv[w->type], (n_out + 3) / 4, 1, 128, p);
}

static bool enc_qknorm(gpu_t *g, model_t *m, CUdeviceptr v, CUdeviceptr w,
                       int n_heads) {
    float eps = m->rms_eps;
    int hd = m->head_dim;
    void *p[] = { &v, &w, &hd, &eps };
    return launch(g->f_qknorm, n_heads, 1, 64, p);
}

static bool enc_rope(gpu_t *g, model_t *m, CUdeviceptr v, int n_heads, int pos) {
    rope_args a = { m->head_dim, n_heads, m->rope_dim / 2, pos,
                    m->rope_neox, m->rope_mscale };
    void *p[] = { &v, &g->inv_freq, &a };
    return launch(g->f_rope, (a.half_dim + 31) / 32, n_heads, 32, p);
}

static bool enc_elem(gpu_t *g, CUfunction f, CUdeviceptr a, CUdeviceptr b, int n) {
    void *p[] = { &a, &b, &n };
    return launch(f, (n + 255) / 256, 1, 256, p);
}

void gpu_free(model_t *m) {
    gpu_t *g = m->gpu;
    if (!g) return;
    cu.CtxSetCurrent(g->ctx);
    for (int l = 0; l < m->n_layer; l++) {
        CUdeviceptr bufs[] = { g->attn_norm[l], g->ffn_norm[l], g->bq[l],
                               g->bk[l], g->bv[l], g->bo[l], g->qn[l], g->kn[l] };
        for (size_t i = 0; i < sizeof(bufs) / sizeof(*bufs); i++)
            if (bufs[i]) cu.MemFree(bufs[i]);
    }
    free(g->attn_norm); free(g->ffn_norm);
    free(g->bq); free(g->bk); free(g->bv); free(g->bo);
    free(g->qn); free(g->kn);
    CUdeviceptr bufs[] = { g->weights, g->kc, g->vc, g->x, g->xb, g->xb2,
                           g->q, g->kt, g->vt, g->hb, g->hb2, g->att,
                           g->logits, g->inv_freq, g->out_norm, g->dummy };
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
    int kv_dim = m->n_head_kv * m->head_dim;
    size_t row = (size_t)kv_dim * sizeof(f16_t);
    for (int l = 0; l < m->n_layer; l++) {
        size_t off = ((size_t)l * m->n_ctx + lo) * row;
        size_t len = (size_t)(hi - lo) * row;
        if (cu.MemcpyHtoD(g->kc + off, (uint8_t *)m->kcache + off, len) != 0 ||
            cu.MemcpyHtoD(g->vc + off, (uint8_t *)m->vcache + off, len) != 0)
            return false;
    }
    return true;
}

float *gpu_forward(model_t *m, int token, int pos) {
    gpu_t *g = m->gpu;
    int n_embd = m->n_embd;
    int q_dim  = m->n_head * m->head_dim;
    int kv_dim = m->n_head_kv * m->head_dim;

    if (cu.CtxSetCurrent(g->ctx) != 0) return NULL;

    // a non-contiguous position step means the CPU wrote KV rows we haven't
    // seen (prompt batch, or a reset + new prompt): resync [0, pos)
    if (pos != g->last_pos + 1) {
        if (!kv_upload(g, m, 0, pos)) return NULL;
    }

    // token embedding on CPU (one row), then a tiny HtoD
    size_t ers = ggml_row_size(m->tok_embd->type, n_embd);
    dequant_row(m->tok_embd->type,
                (uint8_t *)m->tok_embd->data + (size_t)token * ers,
                g->h_x, n_embd);
    if (cu.MemcpyHtoD(g->x, g->h_x, sizeof(float) * n_embd) != 0) return NULL;

    bool ok = true;
    for (int l = 0; l < m->n_layer && ok; l++) {
        layer_t *ly = &m->layers[l];
        uint64_t kv_off = ((uint64_t)l * m->n_ctx + pos) * kv_dim;

        ok = ok && enc_rmsnorm(g, g->x, g->xb, g->attn_norm[l], n_embd, m->rms_eps);
        ok = ok && enc_mv(g, m, ly->wq, g->xb, g->q,  n_embd, q_dim,  g->bq[l]);
        ok = ok && enc_mv(g, m, ly->wk, g->xb, g->kt, n_embd, kv_dim, g->bk[l]);
        ok = ok && enc_mv(g, m, ly->wv, g->xb, g->vt, n_embd, kv_dim, g->bv[l]);
        if (g->qn[l]) ok = ok && enc_qknorm(g, m, g->q,  g->qn[l], m->n_head);
        if (g->kn[l]) ok = ok && enc_qknorm(g, m, g->kt, g->kn[l], m->n_head_kv);
        ok = ok && enc_rope(g, m, g->q,  m->n_head,    pos);
        ok = ok && enc_rope(g, m, g->kt, m->n_head_kv, pos);

        {
            void *p[] = { &g->kt, &g->vt, &g->kc, &g->vc, &kv_dim, &kv_off };
            ok = ok && launch(g->f_store, (kv_dim + 63) / 64, 1, 64, p);
        }

        {
            attn_args aa = { m->head_dim, m->n_head, m->n_head_kv, m->n_ctx, pos,
                             (uint64_t)l * m->n_ctx * kv_dim,
                             1.0f / sqrtf((float)m->head_dim) };
            void *p[] = { &g->q, &g->kc, &g->vc, &g->att, &g->xb2, &aa };
            ok = ok && launch(g->f_attn, m->n_head, 1, 128, p);
        }

        ok = ok && enc_mv(g, m, ly->wo, g->xb2, g->xb, q_dim, n_embd, g->bo[l]);
        ok = ok && enc_elem(g, g->f_add, g->x, g->xb, n_embd);

        ok = ok && enc_rmsnorm(g, g->x, g->xb, g->ffn_norm[l], n_embd, m->rms_eps);
        ok = ok && enc_mv(g, m, ly->w_gate, g->xb, g->hb,  n_embd, m->n_ff, 0);
        ok = ok && enc_mv(g, m, ly->w_up,   g->xb, g->hb2, n_embd, m->n_ff, 0);
        ok = ok && enc_elem(g, g->f_silu, g->hb, g->hb2, m->n_ff);
        ok = ok && enc_mv(g, m, ly->w_down, g->hb, g->xb, m->n_ff, n_embd, 0);
        ok = ok && enc_elem(g, g->f_add, g->x, g->xb, n_embd);
    }

    ok = ok && enc_rmsnorm(g, g->x, g->xb, g->out_norm, n_embd, m->rms_eps);
    ok = ok && enc_mv(g, m, m->output, g->xb, g->logits, n_embd, m->n_vocab, 0);

    if (!ok || cu.CtxSynchronize() != 0) {
        fprintf(stderr, "gpu: kernel launch failed — falling back to CPU\n");
        return NULL;
    }

    // keep the host KV cache authoritative: copy this token's row back so the
    // CPU path can take over at any point
    size_t row = (size_t)kv_dim * sizeof(f16_t);
    for (int l = 0; l < m->n_layer; l++) {
        size_t off = ((size_t)l * m->n_ctx + pos) * row;
        if (cu.MemcpyDtoH((uint8_t *)m->kcache + off, g->kc + off, row) != 0 ||
            cu.MemcpyDtoH((uint8_t *)m->vcache + off, g->vc + off, row) != 0)
            return NULL;
    }

    if (cu.MemcpyDtoH(g->h_logits, g->logits, sizeof(float) * m->n_vocab) != 0)
        return NULL;
    g->last_pos = pos;
    return g->h_logits;
}
