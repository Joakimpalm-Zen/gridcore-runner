// Metal GPU backend: full single-token forward pass on Apple GPUs.
// Compiled without ARC; every object lives for the process lifetime.
#import <Metal/Metal.h>

#include "runner.h"
#include "kernels_metal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    id<MTLDevice>       dev;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> p_rmsnorm, p_qknorm, p_rope, p_store, p_attn, p_silu, p_add;
    id<MTLComputePipelineState> p_mv[32];       // indexed by ggml type
    id<MTLBuffer> weights;                      // wraps the model mmap (zero copy)
    bool          weights_copied;
    id<MTLBuffer> kc, vc;
    id<MTLBuffer> x, xb, xb2, q, kt, vt, hb, hb2, att, logits;
    id<MTLBuffer> inv_freq, out_norm, dummy;
    id<MTLBuffer> *attn_norm, *ffn_norm;        // per layer
    id<MTLBuffer> *bq, *bk, *bv, *bo;           // per layer, may be nil
    id<MTLBuffer> *qn, *kn;                     // qwen3 per-head q/k norms
} gpu_t;

typedef struct { int n_in, n_out; uint64_t w_off; int has_bias; } mv_args;
typedef struct { int head_dim, n_heads, half_dim, pos, neox; float mscale; } rope_args;
typedef struct { int head_dim, n_head, n_head_kv, n_ctx, pos; uint64_t l_off; float scale; } attn_args;

bool gpu_available(char *name, int cap) {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) return false;
    if (name) snprintf(name, cap, "%s", dev.name.UTF8String);
    [dev release];
    return true;
}

bool gpu_mem_info(size_t *free_bytes, size_t *total_bytes) {
    // unified memory: the RAM reservation governs; no separate VRAM pool
    (void)free_bytes; (void)total_bytes;
    return false;
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

static id<MTLComputePipelineState> mk_pipeline(id<MTLDevice> dev,
                                               id<MTLLibrary> lib,
                                               NSString *name) {
    id<MTLFunction> fn = [lib newFunctionWithName:name];
    if (!fn) return nil;
    NSError *err = nil;
    id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:fn error:&err];
    [fn release];
    if (!p) fprintf(stderr, "gpu: pipeline %s failed: %s\n",
                    name.UTF8String, err.localizedDescription.UTF8String);
    return p;
}

static id<MTLBuffer> f32_buf(id<MTLDevice> dev, const float *src, size_t n) {
    if (!src) return nil;
    return [dev newBufferWithBytes:src length:n * sizeof(float)
                           options:MTLResourceStorageModeShared];
}

static float *gpu_forward(model_t *m, int token, int pos);

bool gpu_kv_q8_ok(void) {
    // The Metal kernels and the host<->device KV copies still speak fp16 only;
    // porting them is the remaining half of Phase 8. Returning false here keeps
    // `--kv q8` from silently handing q8_0 blocks to an fp16 reader on macOS:
    // the cache falls back to f16 (with a note) unless --gpu off is used.
    return false;
}

bool gpu_init(model_t *m) {
    if (m->swa_window > 0 || m->embd_scale != 1.0f) {
        fprintf(stderr, "gpu: '%s' (sliding-window attention) is not on the metal backend yet — using CPU\n",
                m->arch);
        return false;
    }
    // every weight matmul must have a kernel for its quant type
    if (!gpu_type_ok(m->output->type)) goto unsupported;
    for (int l = 0; l < m->n_layer; l++) {
        layer_t *ly = &m->layers[l];
        gguf_tensor *ws[] = { ly->wq, ly->wk, ly->wv, ly->wo,
                              ly->w_gate, ly->w_up, ly->w_down };
        for (int i = 0; i < 7; i++)
            if (!gpu_type_ok(ws[i]->type)) goto unsupported;
    }

    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) return false;

    NSError *err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:
                              [NSString stringWithUTF8String:k_metal_src]
                                           options:nil
                                             error:&err];
    if (!lib) {
        fprintf(stderr, "gpu: shader compile failed: %s\n",
                err.localizedDescription.UTF8String);
        return false;
    }

    gpu_t *g = calloc(1, sizeof(gpu_t));
    g->dev = dev;
    g->queue = [dev newCommandQueue];
    g->p_rmsnorm      = mk_pipeline(dev, lib, @"k_rmsnorm");
    g->p_qknorm       = mk_pipeline(dev, lib, @"k_qknorm");
    g->p_rope         = mk_pipeline(dev, lib, @"k_rope");
    g->p_store        = mk_pipeline(dev, lib, @"k_store_kv");
    g->p_attn         = mk_pipeline(dev, lib, @"k_attn");
    g->p_silu         = mk_pipeline(dev, lib, @"k_silu_mul");
    g->p_add          = mk_pipeline(dev, lib, @"k_add");
    g->p_mv[T_F32]    = mk_pipeline(dev, lib, @"k_mv_f32");
    g->p_mv[T_F16]    = mk_pipeline(dev, lib, @"k_mv_f16");
    g->p_mv[T_Q8_0]   = mk_pipeline(dev, lib, @"k_mv_q8_0");
    g->p_mv[T_Q4_0]   = mk_pipeline(dev, lib, @"k_mv_q4_0");
    g->p_mv[T_Q4_1]   = mk_pipeline(dev, lib, @"k_mv_q4_1");
    g->p_mv[T_Q5_0]   = mk_pipeline(dev, lib, @"k_mv_q5_0");
    g->p_mv[T_Q5_1]   = mk_pipeline(dev, lib, @"k_mv_q5_1");
    g->p_mv[T_Q4_K]   = mk_pipeline(dev, lib, @"k_mv_q4_K");
    g->p_mv[T_Q5_K]   = mk_pipeline(dev, lib, @"k_mv_q5_K");
    g->p_mv[T_Q6_K]   = mk_pipeline(dev, lib, @"k_mv_q6_K");
    [lib release];
    if (!g->p_rmsnorm || !g->p_rope || !g->p_store || !g->p_attn ||
        !g->p_silu || !g->p_add || !g->p_mv[T_F32] || !g->p_mv[T_Q4_K]) {
        free(g);
        return false;
    }

    // weights: wrap the mmap zero-copy (page aligned; length page-rounded —
    // mmap always maps whole pages, so the rounded tail is valid memory)
    size_t page = 16384;
    size_t wlen = (m->gf.map_size + page - 1) & ~(page - 1);
    g->weights = [dev newBufferWithBytesNoCopy:m->gf.map
                                        length:wlen
                                       options:MTLResourceStorageModeShared
                                   deallocator:nil];
    if (!g->weights) {
        // fallback: copy (costs RAM but still works)
        g->weights = [dev newBufferWithBytes:m->gf.map
                                      length:m->gf.map_size
                                     options:MTLResourceStorageModeShared];
        g->weights_copied = true;
        if (!g->weights) { free(g); return false; }
    }

    int q_dim  = m->n_head * m->head_dim;
    int kv_dim = m->n_head_kv * m->head_dim;
    int xdim   = q_dim > m->n_embd ? q_dim : m->n_embd;
    size_t kv_bytes = (size_t)m->n_layer * m->n_ctx * kv_dim * sizeof(f16_t);

    #define NEWBUF(n) [dev newBufferWithLength:(n) options:MTLResourceStorageModeShared]
    g->kc = NEWBUF(kv_bytes);
    g->vc = NEWBUF(kv_bytes);
    memset(g->kc.contents, 0, kv_bytes);
    memset(g->vc.contents, 0, kv_bytes);
    // CPU batch prompt processing writes the same cache through these pointers
    free(m->kcache); free(m->vcache);
    m->kcache = (f16_t *)g->kc.contents;
    m->vcache = (f16_t *)g->vc.contents;

    g->x      = NEWBUF(sizeof(float) * m->n_embd);
    g->xb     = NEWBUF(sizeof(float) * xdim);
    g->xb2    = NEWBUF(sizeof(float) * xdim);
    g->q      = NEWBUF(sizeof(float) * q_dim);
    g->kt     = NEWBUF(sizeof(float) * kv_dim);
    g->vt     = NEWBUF(sizeof(float) * kv_dim);
    g->hb     = NEWBUF(sizeof(float) * m->n_ff);
    g->hb2    = NEWBUF(sizeof(float) * m->n_ff);
    g->att    = NEWBUF(sizeof(float) * (size_t)m->n_head * m->n_ctx);
    g->logits = NEWBUF(sizeof(float) * m->n_vocab);
    g->dummy  = NEWBUF(4);
    #undef NEWBUF

    g->inv_freq = f32_buf(dev, m->rope_inv_freq, m->rope_dim / 2);
    g->out_norm = f32_buf(dev, m->out_norm_w, m->n_embd);
    g->attn_norm = calloc(m->n_layer, sizeof(id));
    g->ffn_norm  = calloc(m->n_layer, sizeof(id));
    g->bq = calloc(m->n_layer, sizeof(id));
    g->bk = calloc(m->n_layer, sizeof(id));
    g->bv = calloc(m->n_layer, sizeof(id));
    g->bo = calloc(m->n_layer, sizeof(id));
    g->qn = calloc(m->n_layer, sizeof(id));
    g->kn = calloc(m->n_layer, sizeof(id));
    for (int l = 0; l < m->n_layer; l++) {
        layer_t *ly = &m->layers[l];
        g->attn_norm[l] = f32_buf(dev, ly->attn_norm_w, m->n_embd);
        g->ffn_norm[l]  = f32_buf(dev, ly->ffn_norm_w, m->n_embd);
        g->bq[l] = f32_buf(dev, ly->bq, q_dim);
        g->bk[l] = f32_buf(dev, ly->bk, kv_dim);
        g->bv[l] = f32_buf(dev, ly->bv, kv_dim);
        g->bo[l] = f32_buf(dev, ly->bo, m->n_embd);
        g->qn[l] = f32_buf(dev, ly->qnorm_w, m->head_dim);
        g->kn[l] = f32_buf(dev, ly->knorm_w, m->head_dim);
    }

    m->gpu = g;
    // Metal always runs the whole model; without this the dispatcher takes the
    // partial-offload branch (gpu_layers == 0) and re-runs every layer on the
    // CPU, silently discarding the GPU's work
    m->gpu_layers = m->n_layer;
    fprintf(stderr, "gpu: Metal backend on %s%s\n", dev.name.UTF8String,
            g->weights_copied ? " (weights copied)" : " (zero-copy weights)");
    return true;

unsupported:
    fprintf(stderr, "gpu: model uses a quant type without a Metal kernel — using CPU\n");
    return false;
}

// ---------------------------------------------------------------- encoding

static void enc_rmsnorm(gpu_t *g, id<MTLComputeCommandEncoder> e,
                        id<MTLBuffer> x, id<MTLBuffer> y, id<MTLBuffer> w,
                        int n, float eps) {
    [e setComputePipelineState:g->p_rmsnorm];
    [e setBuffer:x offset:0 atIndex:0];
    [e setBuffer:y offset:0 atIndex:1];
    [e setBuffer:w offset:0 atIndex:2];
    [e setBytes:&n length:4 atIndex:3];
    [e setBytes:&eps length:4 atIndex:4];
    [e dispatchThreadgroups:MTLSizeMake(1, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

static void enc_mv(gpu_t *g, id<MTLComputeCommandEncoder> e, model_t *m,
                   gguf_tensor *w, id<MTLBuffer> x, id<MTLBuffer> y,
                   int n_in, int n_out, id<MTLBuffer> bias) {
    mv_args a = { n_in, n_out,
                  (uint64_t)((uint8_t *)w->data - (uint8_t *)m->gf.map),
                  bias != nil };
    [e setComputePipelineState:g->p_mv[w->type]];
    [e setBuffer:g->weights offset:0 atIndex:0];
    [e setBuffer:x offset:0 atIndex:1];
    [e setBuffer:y offset:0 atIndex:2];
    [e setBytes:&a length:sizeof(a) atIndex:3];
    [e setBuffer:bias ? bias : g->dummy offset:0 atIndex:4];
    // 128 threads = 4 simdgroups = 4 rows per threadgroup
    [e dispatchThreadgroups:MTLSizeMake((n_out + 3) / 4, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
}

static void enc_qknorm(gpu_t *g, id<MTLComputeCommandEncoder> e, model_t *m,
                       id<MTLBuffer> v, id<MTLBuffer> w, int n_heads) {
    float eps = m->rms_eps;
    int hd = m->head_dim;
    [e setComputePipelineState:g->p_qknorm];
    [e setBuffer:v offset:0 atIndex:0];
    [e setBuffer:w offset:0 atIndex:1];
    [e setBytes:&hd length:4 atIndex:2];
    [e setBytes:&eps length:4 atIndex:3];
    [e dispatchThreadgroups:MTLSizeMake(n_heads, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
}

static void enc_rope(gpu_t *g, id<MTLComputeCommandEncoder> e, model_t *m,
                     id<MTLBuffer> v, int n_heads, int pos) {
    rope_args a = { m->head_dim, n_heads, m->rope_dim / 2, pos,
                    m->rope_neox, m->rope_mscale };
    [e setComputePipelineState:g->p_rope];
    [e setBuffer:v offset:0 atIndex:0];
    [e setBuffer:g->inv_freq offset:0 atIndex:1];
    [e setBytes:&a length:sizeof(a) atIndex:2];
    [e dispatchThreads:MTLSizeMake(a.half_dim, n_heads, 1)
      threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
}

static void enc_elem(gpu_t *g, id<MTLComputeCommandEncoder> e,
                     id<MTLComputePipelineState> p,
                     id<MTLBuffer> a, id<MTLBuffer> b, int n) {
    [e setComputePipelineState:p];
    [e setBuffer:a offset:0 atIndex:0];
    [e setBuffer:b offset:0 atIndex:1];
    [e setBytes:&n length:4 atIndex:2];
    [e dispatchThreads:MTLSizeMake(n, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

void gpu_free(model_t *m) {
    gpu_t *g = m->gpu;
    if (!g) return;
    // the kv cache pointers alias GPU buffer contents; detach them first
    m->kcache = NULL;
    m->vcache = NULL;
    for (int l = 0; l < m->n_layer; l++) {
        [g->attn_norm[l] release]; [g->ffn_norm[l] release];
        [g->bq[l] release]; [g->bk[l] release];
        [g->bv[l] release]; [g->bo[l] release];
        [g->qn[l] release]; [g->kn[l] release];
    }
    free(g->attn_norm); free(g->ffn_norm);
    free(g->bq); free(g->bk); free(g->bv); free(g->bo);
    free(g->qn); free(g->kn);
    id<MTLBuffer> bufs[] = { g->weights, g->kc, g->vc, g->x, g->xb, g->xb2,
                             g->q, g->kt, g->vt, g->hb, g->hb2, g->att,
                             g->logits, g->inv_freq, g->out_norm, g->dummy };
    for (size_t i = 0; i < sizeof(bufs) / sizeof(*bufs); i++) [bufs[i] release];
    for (int i = 0; i < 32; i++) [g->p_mv[i] release];
    [g->p_rmsnorm release]; [g->p_qknorm release]; [g->p_rope release]; [g->p_store release];
    [g->p_attn release]; [g->p_silu release]; [g->p_add release];
    [g->queue release];
    [g->dev release];
    free(g);
    m->gpu = NULL;
}

bool gpu_forward_batch(model_t *m, const int32_t *tokens, int n, int pos,
                       bool want_logits, float **logits) {
    // Unified memory makes the per-token loop cheap here; a natively batched
    // encoder (one command buffer for the whole batch) is a later optimization.
    float *lg = NULL;
    for (int b = 0; b < n; b++) {
        lg = gpu_forward(m, tokens[b], pos + b);
        if (!lg) return false;
    }
    if (logits) *logits = want_logits ? lg : NULL;
    return true;
}

static float *gpu_forward(model_t *m, int token, int pos) {
    gpu_t *g = m->gpu;
    int n_embd = m->n_embd;
    int q_dim  = m->n_head * m->head_dim;
    int kv_dim = m->n_head_kv * m->head_dim;

    // token embedding on CPU (one row), straight into the shared buffer
    size_t ers = ggml_row_size(m->tok_embd->type, n_embd);
    dequant_row(m->tok_embd->type,
                (uint8_t *)m->tok_embd->data + (size_t)token * ers,
                (float *)g->x.contents, n_embd);

    id<MTLCommandBuffer> cb = [g->queue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];

    for (int l = 0; l < m->n_layer; l++) {
        layer_t *ly = &m->layers[l];
        uint64_t kv_off = ((uint64_t)l * m->n_ctx + pos) * kv_dim;

        enc_rmsnorm(g, e, g->x, g->xb, g->attn_norm[l], n_embd, m->rms_eps);
        enc_mv(g, e, m, ly->wq, g->xb, g->q,  n_embd, q_dim,  g->bq[l]);
        enc_mv(g, e, m, ly->wk, g->xb, g->kt, n_embd, kv_dim, g->bk[l]);
        enc_mv(g, e, m, ly->wv, g->xb, g->vt, n_embd, kv_dim, g->bv[l]);
        if (g->qn[l]) enc_qknorm(g, e, m, g->q,  g->qn[l], m->n_head);
        if (g->kn[l]) enc_qknorm(g, e, m, g->kt, g->kn[l], m->n_head_kv);
        enc_rope(g, e, m, g->q,  m->n_head,    pos);
        enc_rope(g, e, m, g->kt, m->n_head_kv, pos);

        [e setComputePipelineState:g->p_store];
        [e setBuffer:g->kt offset:0 atIndex:0];
        [e setBuffer:g->vt offset:0 atIndex:1];
        [e setBuffer:g->kc offset:0 atIndex:2];
        [e setBuffer:g->vc offset:0 atIndex:3];
        [e setBytes:&kv_dim length:4 atIndex:4];
        [e setBytes:&kv_off length:8 atIndex:5];
        [e dispatchThreads:MTLSizeMake(kv_dim, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];

        attn_args aa = { m->head_dim, m->n_head, m->n_head_kv, m->n_ctx, pos,
                         (uint64_t)l * m->n_ctx * kv_dim,
                         1.0f / sqrtf((float)m->head_dim) };
        [e setComputePipelineState:g->p_attn];
        [e setBuffer:g->q   offset:0 atIndex:0];
        [e setBuffer:g->kc  offset:0 atIndex:1];
        [e setBuffer:g->vc  offset:0 atIndex:2];
        [e setBuffer:g->att offset:0 atIndex:3];
        [e setBuffer:g->xb2 offset:0 atIndex:4];
        [e setBytes:&aa length:sizeof(aa) atIndex:5];
        [e dispatchThreadgroups:MTLSizeMake(m->n_head, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

        enc_mv(g, e, m, ly->wo, g->xb2, g->xb, q_dim, n_embd, g->bo[l]);
        enc_elem(g, e, g->p_add, g->x, g->xb, n_embd);

        enc_rmsnorm(g, e, g->x, g->xb, g->ffn_norm[l], n_embd, m->rms_eps);
        enc_mv(g, e, m, ly->w_gate, g->xb, g->hb,  n_embd, m->n_ff, nil);
        enc_mv(g, e, m, ly->w_up,   g->xb, g->hb2, n_embd, m->n_ff, nil);
        enc_elem(g, e, g->p_silu, g->hb, g->hb2, m->n_ff);
        enc_mv(g, e, m, ly->w_down, g->hb, g->xb, m->n_ff, n_embd, nil);
        enc_elem(g, e, g->p_add, g->x, g->xb, n_embd);
    }

    enc_rmsnorm(g, e, g->x, g->xb, g->out_norm, n_embd, m->rms_eps);
    enc_mv(g, e, m, m->output, g->xb, g->logits, n_embd, m->n_vocab, nil);

    [e endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "gpu: command buffer failed — falling back to CPU\n");
        return NULL;
    }
    return (float *)g->logits.contents;
}
