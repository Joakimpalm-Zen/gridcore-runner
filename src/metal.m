// Metal GPU backend: full single-token forward pass on Apple GPUs.
// Compiled without ARC; every object lives for the process lifetime.
#import <Metal/Metal.h>

#include "runner.h"
#include "kernels_metal.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

typedef struct {
    id<MTLDevice>       dev;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> p_rmsnorm, p_qknorm, p_rope, p_store, p_attn, p_silu, p_add;
    id<MTLComputePipelineState> p_moe_route, p_moe_actmul, p_moe_sum;
    id<MTLComputePipelineState> p_mv[32];       // indexed by ggml type
    id<MTLComputePipelineState> p_moe_mv[32];   // indexed by ggml type
    id<MTLBuffer> weights;                      // wraps the model mmap (zero copy)
    bool          weights_copied;
    id<MTLBuffer> kc, vc;
    id<MTLBuffer> x, xb, xb2, q, kt, vt, hb, hb2, att, logits;
    id<MTLBuffer> moe_logits, moe_sel, moe_selw, moe_hb, moe_hb2, moe_eout;
    id<MTLBuffer> inv_freq, inv_freq_local, out_norm, dummy;
    id<MTLBuffer> *attn_norm, *ffn_norm;        // per layer
    id<MTLBuffer> *bq, *bk, *bv, *bo;           // per layer, may be nil
    id<MTLBuffer> *qn, *kn;                     // qwen3 per-head q/k norms
    int batch_cap;                              // scratch rows allocated
} gpu_t;

typedef struct { int n_in, n_out; uint64_t w_off; int has_bias; } mv_args;
typedef struct { int head_dim, n_heads, half_dim, pos, neox; float mscale; } rope_args;
typedef struct { int head_dim, n_head, n_head_kv, n_ctx, pos; uint64_t l_off; float scale; int q8, window; } attn_args;
typedef struct { int n_in, n_out; uint64_t w_off, estride; int xs, ys; } moe_args;

static void gpu_release_state(gpu_t *g, int n_layer) {
    if (!g) return;
    for (int l = 0; l < n_layer; l++) {
        if (g->attn_norm) [g->attn_norm[l] release];
        if (g->ffn_norm)  [g->ffn_norm[l] release];
        if (g->bq) [g->bq[l] release];
        if (g->bk) [g->bk[l] release];
        if (g->bv) [g->bv[l] release];
        if (g->bo) [g->bo[l] release];
        if (g->qn) [g->qn[l] release];
        if (g->kn) [g->kn[l] release];
    }
    free(g->attn_norm); free(g->ffn_norm);
    free(g->bq); free(g->bk); free(g->bv); free(g->bo);
    free(g->qn); free(g->kn);
    id<MTLBuffer> bufs[] = { g->weights, g->kc, g->vc, g->x, g->xb, g->xb2,
                             g->q, g->kt, g->vt, g->hb, g->hb2, g->att,
                             g->logits, g->moe_logits, g->moe_sel, g->moe_selw,
                             g->moe_hb, g->moe_hb2, g->moe_eout,
                             g->inv_freq, g->inv_freq_local, g->out_norm, g->dummy };
    for (size_t i = 0; i < sizeof(bufs) / sizeof(*bufs); i++) [bufs[i] release];
    for (int i = 0; i < 32; i++) [g->p_mv[i] release];
    for (int i = 0; i < 32; i++) [g->p_moe_mv[i] release];
    [g->p_rmsnorm release]; [g->p_qknorm release]; [g->p_rope release];
    [g->p_store release]; [g->p_attn release]; [g->p_silu release];
    [g->p_add release]; [g->p_moe_route release]; [g->p_moe_actmul release];
    [g->p_moe_sum release];
    [g->queue release];
    [g->dev release];
    free(g);
}

static bool metal_init_injected(const char *point) {
    const char *inject = getenv("RUNNER_METAL_INIT_INJECT_FAILURE");
    return inject && *inject && strcmp(inject, "0") &&
           (!strcmp(inject, "always") || !strcmp(inject, point));
}

static bool metal_env_on(const char *name) {
    const char *v = getenv(name);
    return v && *v && strcmp(v, "0");
}

static bool metal_batch_enabled(void) {
    const char *v = getenv("RUNNER_METAL_BATCH");
    return !(v && *v && !strcmp(v, "0"));
}

static bool gpu_init_fail(model_t *m, gpu_t *g, id<MTLLibrary> lib,
                          const char *why) {
    if (why && *why)
        fprintf(stderr, "gpu: Metal initialization failed (%s) — using CPU\n", why);
    else
        fprintf(stderr, "gpu: Metal initialization failed — using CPU\n");
    if (lib) [lib release];
    gpu_release_state(g, m ? m->n_layer : 0);
    return false;
}

static bool metal_buffer_ok(id<MTLBuffer> b) {
    return b != nil && b.contents != NULL;
}

static bool metal_command_failed(id<MTLCommandBuffer> cb) {
    const char *inject = getenv("RUNNER_METAL_INJECT_FAILURE");
    static int injected_once = 0;
    bool injected = inject && *inject && strcmp(inject, "0") &&
                    (!injected_once || strcmp(inject, "always") == 0);
    if (injected) injected_once = 1;
    return injected || cb.status == MTLCommandBufferStatusError;
}

static id<MTLBuffer> new_f32_scratch(id<MTLDevice> dev, size_t n) {
    if (n > SIZE_MAX / sizeof(float)) return nil;
    return [dev newBufferWithLength:n * sizeof(float)
                            options:MTLResourceStorageModeShared];
}

static void release_buf(id<MTLBuffer> b) {
    [b release];
}

static bool metal_ensure_batch(model_t *m, int n) {
    gpu_t *g = (gpu_t *)m->gpu;
    if (!g || n <= g->batch_cap) return true;
    int q_dim  = m->n_head * m->head_dim;
    int kv_dim = m->n_head_kv * m->head_dim;
    int xdim   = q_dim > m->n_embd ? q_dim : m->n_embd;
    size_t nb = (size_t)n;

    id<MTLBuffer> x      = new_f32_scratch(g->dev, nb * (size_t)m->n_embd);
    id<MTLBuffer> xb     = new_f32_scratch(g->dev, nb * (size_t)xdim);
    id<MTLBuffer> xb2    = new_f32_scratch(g->dev, nb * (size_t)xdim);
    id<MTLBuffer> q      = new_f32_scratch(g->dev, nb * (size_t)q_dim);
    id<MTLBuffer> kt     = new_f32_scratch(g->dev, nb * (size_t)kv_dim);
    id<MTLBuffer> vt     = new_f32_scratch(g->dev, nb * (size_t)kv_dim);
    id<MTLBuffer> hb     = new_f32_scratch(g->dev, nb * (size_t)m->n_ff);
    id<MTLBuffer> hb2    = new_f32_scratch(g->dev, nb * (size_t)m->n_ff);
    id<MTLBuffer> att    = new_f32_scratch(g->dev, nb * (size_t)m->n_head *
                                                   (size_t)m->n_ctx);
    id<MTLBuffer> logits = new_f32_scratch(g->dev, nb * (size_t)m->n_vocab);
    if (!metal_buffer_ok(x) || !metal_buffer_ok(xb) || !metal_buffer_ok(xb2) ||
        !metal_buffer_ok(q) || !metal_buffer_ok(kt) || !metal_buffer_ok(vt) ||
        !metal_buffer_ok(hb) || !metal_buffer_ok(hb2) || !metal_buffer_ok(att) ||
        !metal_buffer_ok(logits)) {
        release_buf(x); release_buf(xb); release_buf(xb2); release_buf(q);
        release_buf(kt); release_buf(vt); release_buf(hb); release_buf(hb2);
        release_buf(att); release_buf(logits);
        return false;
    }

    release_buf(g->x); release_buf(g->xb); release_buf(g->xb2);
    release_buf(g->q); release_buf(g->kt); release_buf(g->vt);
    release_buf(g->hb); release_buf(g->hb2); release_buf(g->att);
    release_buf(g->logits);
    g->x = x; g->xb = xb; g->xb2 = xb2; g->q = q; g->kt = kt; g->vt = vt;
    g->hb = hb; g->hb2 = hb2; g->att = att; g->logits = logits;
    g->batch_cap = n;
    return true;
}

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

// UNVERIFIED — written without a macOS machine to run it on. Nobody on this
// project has executed this function.
//
// Returning false disables the VRAM registry on Metal, which is the correct
// behaviour anyway rather than a placeholder: Apple GPUs share one unified
// memory pool with the CPU, gpu_mem_info already declines to report a separate
// VRAM figure, and the registry's whole arithmetic is "device free bytes minus
// pending claims". With no device-private pool there is no VRAM to account for
// and the RAM reservation (--reserve-ram) governs instead.
//
// If a Metal build ever does want registry accounting, the identity to return
// is [MTLDevice registryID] rendered as a string, and gpu_mem_info would have
// to start reporting recommendedMaxWorkingSetSize / currentAllocatedSize. Both
// need a real Mac to verify before anyone trusts them.
bool gpu_device_id(char *id, int cap) {
    (void)id; (void)cap;
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

static bool metal_moe_type_ok(int type) {
    switch (type) {
        case T_F32: case T_F16: case T_Q8_0: case T_Q4_0:
        case T_Q4_K: case T_Q5_K: case T_Q6_K:
            return true;
        default:
            return false;
    }
}

static bool metal_moe_supported(const model_t *m) {
    if (m->n_expert <= 0) return true;
    if (m->n_expert > 256 || m->n_expert_used < 1 ||
        m->n_expert_used > 256) {
        fprintf(stderr, "gpu: this MoE geometry is outside the Metal router limit — using CPU\n");
        return false;
    }
    if (!model_moe_router_is_plain(m)) {
        fprintf(stderr, "gpu: this model's MoE router has no Metal kernel — using CPU\n");
        return false;
    }
    if (m->n_ff_shexp > 0) {
        fprintf(stderr, "gpu: shared-expert MoE has no Metal path yet — using CPU\n");
        return false;
    }
    if (m->moe_gemma) {
        fprintf(stderr, "gpu: Gemma dual-branch MoE is not on the metal backend yet — using CPU\n");
        return false;
    }
    if (m->ffn_act != ACT_SILU) {
        fprintf(stderr, "gpu: this MoE activation has no Metal kernel yet — using CPU\n");
        return false;
    }
    for (int l = 0; l < m->n_layer; l++) {
        const layer_t *ly = &m->layers[l];
        if (!ly->is_moe) continue;
        if (ly->moe_split) {
            fprintf(stderr, "gpu: split expert layout is not on the metal backend yet — using CPU\n");
            return false;
        }
        if (ly->exp_probs_b || ly->ffn_gate_inp_b ||
            ly->ffn_gate_exps_b || ly->ffn_up_exps_b ||
            ly->ffn_down_exps_b) {
            fprintf(stderr, "gpu: MoE router/expert bias has no Metal path yet — using CPU\n");
            return false;
        }
        if (!ly->ffn_gate_inp || !ly->ffn_gate_exps ||
            !ly->ffn_up_exps || !ly->ffn_down_exps) {
            fprintf(stderr, "gpu: unsupported MoE tensor layout for Metal — using CPU\n");
            return false;
        }
        if (!gpu_type_ok(ly->ffn_gate_inp->type) ||
            !metal_moe_type_ok(ly->ffn_gate_exps->type) ||
            !metal_moe_type_ok(ly->ffn_up_exps->type) ||
            !metal_moe_type_ok(ly->ffn_down_exps->type)) {
            fprintf(stderr, "gpu: MoE tensor type is not on the metal backend yet — using CPU\n");
            return false;
        }
    }
    return true;
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
static float *gpu_forward_native_batch(model_t *m, const int32_t *tokens,
                                       int n, int pos);

// No tensor-core GEMM path on Metal; the TC tolerance gate self-skips.
void gpu_tc_force(int on) { (void)on; }
void gpu_moe_eager_force(int on) { (void)on; }

bool gpu_moe_ok(void) {
    return true;    // plain fused sparse-MoE routes and experts run on Metal
}

bool gpu_kv_q8_ok(void) {
    return true;
}

bool gpu_init(model_t *m) {
    if (m->qwen35) {
        fprintf(stderr, "gpu: qwen35 hybrid path is not on the metal backend yet — using CPU\n");
        return false;
    }
    if (!metal_moe_supported(m))
        return false;
    if (m->embd_scale != 1.0f) {
        fprintf(stderr, "gpu: '%s' scaled embeddings are not on the metal backend yet — using CPU\n",
                m->arch);
        return false;
    }
    if (m->ffn_act != ACT_SILU) {
        fprintf(stderr, "gpu: '%s' FFN activation is not on the metal backend yet — using CPU\n",
                m->arch);
        return false;
    }
    if (m->no_rope_layer_step > 0 || m->attn_temp_scale != 0.0f) {
        fprintf(stderr, "gpu: '%s' NoPE/attention-temperature knobs are not on the metal backend yet — using CPU\n",
                m->arch);
        return false;
    }
    // every weight matmul must have a kernel for its quant type
    if (!gpu_type_ok(m->output->type)) goto unsupported;
    for (int l = 0; l < m->n_layer; l++) {
        layer_t *ly = &m->layers[l];
        if (ly->post_attn_norm_w || ly->post_ffn_norm_w ||
            ly->ple_gate || ly->out_scale != 1.0f || !ly->wv ||
            ly->attn_sinks) {
            fprintf(stderr, "gpu: '%s' layer layout is not on the metal backend yet — using CPU\n",
                    m->arch);
            return false;
        }
        if ((model_head_dim(m, l) != m->head_dim) ||
            (model_n_head_kv(m, l) != m->n_head_kv)) {
            fprintf(stderr, "gpu: '%s' heterogeneous attention geometry is not on the metal backend yet — using CPU\n",
                    m->arch);
            return false;
        }
        if (model_is_swa(m, l) && !m->rope_inv_freq_local) {
            fprintf(stderr, "gpu: '%s' sliding-window rope table is missing — using CPU\n",
                    m->arch);
            return false;
        }
        gguf_tensor *ws[] = { ly->wq, ly->wk, ly->wv, ly->wo,
                              ly->w_gate, ly->w_up, ly->w_down };
        for (int i = 0; i < 7; i++)
            if (ws[i] && !gpu_type_ok(ws[i]->type)) goto unsupported;
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
        [dev release];
        return false;
    }

    gpu_t *g = calloc(1, sizeof(gpu_t));
    if (!g) {
        [lib release];
        [dev release];
        return false;
    }
    g->dev = dev;
    g->queue = [dev newCommandQueue];
    g->p_rmsnorm      = mk_pipeline(dev, lib, @"k_rmsnorm");
    g->p_qknorm       = mk_pipeline(dev, lib, @"k_qknorm");
    g->p_rope         = mk_pipeline(dev, lib, @"k_rope");
    g->p_store        = mk_pipeline(dev, lib, @"k_store_kv");
    g->p_attn         = mk_pipeline(dev, lib, @"k_attn");
    g->p_silu         = mk_pipeline(dev, lib, @"k_silu_mul");
    g->p_add          = mk_pipeline(dev, lib, @"k_add");
    g->p_moe_route    = mk_pipeline(dev, lib, @"k_moe_route");
    g->p_moe_actmul   = mk_pipeline(dev, lib, @"k_moe_actmul");
    g->p_moe_sum      = mk_pipeline(dev, lib, @"k_moe_sum");
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
    g->p_moe_mv[T_F32]  = mk_pipeline(dev, lib, @"k_moe_mv_f32");
    g->p_moe_mv[T_F16]  = mk_pipeline(dev, lib, @"k_moe_mv_f16");
    g->p_moe_mv[T_Q8_0] = mk_pipeline(dev, lib, @"k_moe_mv_q8_0");
    g->p_moe_mv[T_Q4_0] = mk_pipeline(dev, lib, @"k_moe_mv_q4_0");
    g->p_moe_mv[T_Q4_K] = mk_pipeline(dev, lib, @"k_moe_mv_q4_K");
    g->p_moe_mv[T_Q5_K] = mk_pipeline(dev, lib, @"k_moe_mv_q5_K");
    g->p_moe_mv[T_Q6_K] = mk_pipeline(dev, lib, @"k_moe_mv_q6_K");
    [lib release];
    lib = nil;
    if (!g->p_rmsnorm || !g->p_rope || !g->p_store || !g->p_attn ||
        !g->p_silu || !g->p_add || !g->queue || !g->p_qknorm ||
        !g->p_moe_route || !g->p_moe_actmul || !g->p_moe_sum ||
        !g->p_mv[T_F32] || !g->p_mv[T_F16] || !g->p_mv[T_Q8_0] ||
        !g->p_mv[T_Q4_0] || !g->p_mv[T_Q4_1] ||
        !g->p_mv[T_Q5_0] || !g->p_mv[T_Q5_1] ||
        !g->p_mv[T_Q4_K] || !g->p_mv[T_Q5_K] || !g->p_mv[T_Q6_K])
        return gpu_init_fail(m, g, lib, "pipeline allocation");
    if (m->n_expert > 0 &&
        (!g->p_moe_mv[T_F32] || !g->p_moe_mv[T_F16] ||
         !g->p_moe_mv[T_Q8_0] || !g->p_moe_mv[T_Q4_0] ||
         !g->p_moe_mv[T_Q4_K] || !g->p_moe_mv[T_Q5_K] ||
         !g->p_moe_mv[T_Q6_K]))
        return gpu_init_fail(m, g, lib, "MoE pipeline allocation");
    if (metal_init_injected("state"))
        return gpu_init_fail(m, g, lib, "injected state allocation failure");

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
        if (!g->weights)
            return gpu_init_fail(m, g, lib, "weight buffer allocation");
    }

    int q_dim  = m->n_head * m->head_dim;
    int kv_dim = m->n_head_kv * m->head_dim;
    int xdim   = q_dim > m->n_embd ? q_dim : m->n_embd;
    size_t kv_bytes = model_kv_byte_off(m, m->n_layer);

    #define NEWBUF(n) [dev newBufferWithLength:(n) options:MTLResourceStorageModeShared]
    g->kc = NEWBUF(kv_bytes);
    g->vc = NEWBUF(kv_bytes);
    if (!metal_buffer_ok(g->kc) || !metal_buffer_ok(g->vc))
        return gpu_init_fail(m, g, lib, "KV buffer allocation");
    memset(g->kc.contents, 0, kv_bytes);
    memset(g->vc.contents, 0, kv_bytes);
    if (metal_init_injected("after-kv"))
        return gpu_init_fail(m, g, lib, "injected post-KV allocation failure");

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
    if (m->n_expert > 0) {
        size_t used = (size_t)m->n_expert_used;
        g->moe_logits = NEWBUF(sizeof(float) * (size_t)m->n_expert);
        g->moe_sel    = NEWBUF(sizeof(int)   * used);
        g->moe_selw   = NEWBUF(sizeof(float) * used);
        g->moe_hb     = NEWBUF(sizeof(float) * used * (size_t)m->n_ff_exp);
        g->moe_hb2    = NEWBUF(sizeof(float) * used * (size_t)m->n_ff_exp);
        g->moe_eout   = NEWBUF(sizeof(float) * used * (size_t)m->n_embd);
    }
    g->batch_cap = 1;
    #undef NEWBUF
    if (!metal_buffer_ok(g->x) || !metal_buffer_ok(g->xb) ||
        !metal_buffer_ok(g->xb2) || !metal_buffer_ok(g->q) ||
        !metal_buffer_ok(g->kt) || !metal_buffer_ok(g->vt) ||
        !metal_buffer_ok(g->hb) || !metal_buffer_ok(g->hb2) ||
        !metal_buffer_ok(g->att) || !metal_buffer_ok(g->logits) ||
        !metal_buffer_ok(g->dummy) ||
        (m->n_expert > 0 && (!metal_buffer_ok(g->moe_logits) ||
                             !metal_buffer_ok(g->moe_sel) ||
                             !metal_buffer_ok(g->moe_selw) ||
                             !metal_buffer_ok(g->moe_hb) ||
                             !metal_buffer_ok(g->moe_hb2) ||
                             !metal_buffer_ok(g->moe_eout))))
        return gpu_init_fail(m, g, lib, "scratch buffer allocation");

    g->inv_freq = f32_buf(dev, m->rope_inv_freq, m->rope_dim / 2);
    g->inv_freq_local = f32_buf(dev, m->rope_inv_freq_local,
                                m->rope_dim_local / 2);
    g->out_norm = f32_buf(dev, m->out_norm_w, m->n_embd);
    if (!metal_buffer_ok(g->inv_freq) || !metal_buffer_ok(g->out_norm) ||
        (m->rope_inv_freq_local && !metal_buffer_ok(g->inv_freq_local)))
        return gpu_init_fail(m, g, lib, "shared constant allocation");
    g->attn_norm = calloc(m->n_layer, sizeof(id));
    g->ffn_norm  = calloc(m->n_layer, sizeof(id));
    g->bq = calloc(m->n_layer, sizeof(id));
    g->bk = calloc(m->n_layer, sizeof(id));
    g->bv = calloc(m->n_layer, sizeof(id));
    g->bo = calloc(m->n_layer, sizeof(id));
    g->qn = calloc(m->n_layer, sizeof(id));
    g->kn = calloc(m->n_layer, sizeof(id));
    if (!g->attn_norm || !g->ffn_norm || !g->bq || !g->bk || !g->bv ||
        !g->bo || !g->qn || !g->kn)
        return gpu_init_fail(m, g, lib, "per-layer table allocation");
    for (int l = 0; l < m->n_layer; l++) {
        layer_t *ly = &m->layers[l];
        g->attn_norm[l] = f32_buf(dev, ly->attn_norm_w, m->n_embd);
        g->ffn_norm[l]  = f32_buf(dev, ly->ffn_norm_w, m->n_embd);
        int l_hd = model_head_dim(m, l);
        int l_kv_dim = model_kv_dim(m, l);
        g->bq[l] = f32_buf(dev, ly->bq, q_dim);
        g->bk[l] = f32_buf(dev, ly->bk, l_kv_dim);
        g->bv[l] = f32_buf(dev, ly->bv, l_kv_dim);
        g->bo[l] = f32_buf(dev, ly->bo, m->n_embd);
        g->qn[l] = f32_buf(dev, ly->qnorm_w, l_hd);
        g->kn[l] = f32_buf(dev, ly->knorm_w, l_hd);
        if (!metal_buffer_ok(g->attn_norm[l]) ||
            !metal_buffer_ok(g->ffn_norm[l]) ||
            (ly->bq && !metal_buffer_ok(g->bq[l])) ||
            (ly->bk && !metal_buffer_ok(g->bk[l])) ||
            (ly->bv && !metal_buffer_ok(g->bv[l])) ||
            (ly->bo && !metal_buffer_ok(g->bo[l])) ||
            (ly->qnorm_w && !metal_buffer_ok(g->qn[l])) ||
            (ly->knorm_w && !metal_buffer_ok(g->kn[l])))
            return gpu_init_fail(m, g, lib, "per-layer buffer allocation");
    }

    // CPU batch prompt processing writes the same cache through these pointers,
    // but do not detach the malloc-owned cache until every Metal allocation has
    // succeeded. Any failure above falls back with the CPU KV still intact.
    free(m->kcache); free(m->vcache);
    m->kcache = (f16_t *)g->kc.contents;
    m->vcache = (f16_t *)g->vc.contents;
    m->kv_owner = KV_OWNER_GPU_BACKEND;
    m->gpu = g;
    m->gpu_owner = g;
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
                        id<MTLBuffer> x, NSUInteger x_off,
                        id<MTLBuffer> y, NSUInteger y_off, id<MTLBuffer> w,
                        int n, float eps) {
    [e setComputePipelineState:g->p_rmsnorm];
    [e setBuffer:x offset:x_off atIndex:0];
    [e setBuffer:y offset:y_off atIndex:1];
    [e setBuffer:w offset:0 atIndex:2];
    [e setBytes:&n length:4 atIndex:3];
    [e setBytes:&eps length:4 atIndex:4];
    [e dispatchThreadgroups:MTLSizeMake(1, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

static void enc_mv(gpu_t *g, id<MTLComputeCommandEncoder> e, model_t *m,
                   gguf_tensor *w, id<MTLBuffer> x, NSUInteger x_off,
                   id<MTLBuffer> y, NSUInteger y_off,
                   int n_in, int n_out, id<MTLBuffer> bias) {
    mv_args a = { n_in, n_out,
                  (uint64_t)((uint8_t *)w->data - (uint8_t *)m->gf.map),
                  bias != nil };
    [e setComputePipelineState:g->p_mv[w->type]];
    [e setBuffer:g->weights offset:0 atIndex:0];
    [e setBuffer:x offset:x_off atIndex:1];
    [e setBuffer:y offset:y_off atIndex:2];
    [e setBytes:&a length:sizeof(a) atIndex:3];
    [e setBuffer:bias ? bias : g->dummy offset:0 atIndex:4];
    // 128 threads = 4 simdgroups = 4 rows per threadgroup
    [e dispatchThreadgroups:MTLSizeMake((n_out + 3) / 4, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
}

static void enc_qknorm(gpu_t *g, id<MTLComputeCommandEncoder> e, model_t *m,
                       id<MTLBuffer> v, NSUInteger v_off, id<MTLBuffer> w,
                       int n_heads, int hd) {
    float eps = m->rms_eps;
    [e setComputePipelineState:g->p_qknorm];
    [e setBuffer:v offset:v_off atIndex:0];
    [e setBuffer:w offset:0 atIndex:1];
    [e setBytes:&hd length:4 atIndex:2];
    [e setBytes:&eps length:4 atIndex:3];
    [e dispatchThreadgroups:MTLSizeMake(n_heads, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
}

static void enc_rope(gpu_t *g, id<MTLComputeCommandEncoder> e, model_t *m,
                     id<MTLBuffer> v, NSUInteger v_off, int n_heads, int pos,
                     int layer) {
    bool local = model_is_swa(m, layer);
    int hd = model_head_dim(m, layer);
    int rope_dim = model_rope_dim(m, layer);
    rope_args a = { hd, n_heads, rope_dim / 2, pos,
                    m->rope_neox, model_rope_mscale(m, layer) };
    [e setComputePipelineState:g->p_rope];
    [e setBuffer:v offset:v_off atIndex:0];
    [e setBuffer:(local && g->inv_freq_local) ? g->inv_freq_local : g->inv_freq
          offset:0 atIndex:1];
    [e setBytes:&a length:sizeof(a) atIndex:2];
    [e dispatchThreads:MTLSizeMake(a.half_dim, n_heads, 1)
      threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
}

static void enc_elem(gpu_t *g, id<MTLComputeCommandEncoder> e,
                     id<MTLComputePipelineState> p,
                     id<MTLBuffer> a, NSUInteger a_off,
                     id<MTLBuffer> b, NSUInteger b_off, int n) {
    [e setComputePipelineState:p];
    [e setBuffer:a offset:a_off atIndex:0];
    [e setBuffer:b offset:b_off atIndex:1];
    [e setBytes:&n length:4 atIndex:2];
    [e dispatchThreads:MTLSizeMake(n, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

static void enc_moe_route(gpu_t *g, id<MTLComputeCommandEncoder> e,
                          int ne, int used) {
    int tokens = 1, ls = ne;
    [e setComputePipelineState:g->p_moe_route];
    [e setBuffer:g->moe_logits offset:0 atIndex:0];
    [e setBuffer:g->moe_sel offset:0 atIndex:1];
    [e setBuffer:g->moe_selw offset:0 atIndex:2];
    [e setBytes:&ne length:4 atIndex:3];
    [e setBytes:&used length:4 atIndex:4];
    [e setBytes:&tokens length:4 atIndex:5];
    [e setBytes:&ls length:4 atIndex:6];
    [e dispatchThreads:MTLSizeMake(1, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
}

static void enc_moe_mv(gpu_t *g, id<MTLComputeCommandEncoder> e, model_t *m,
                       gguf_tensor *base, uint64_t estride,
                       id<MTLBuffer> x, NSUInteger x_off,
                       id<MTLBuffer> y, NSUInteger y_off,
                       int n_in, int n_out, int nslots, int xs, int ys) {
    moe_args a = { n_in, n_out,
                   (uint64_t)((uint8_t *)base->data - (uint8_t *)m->gf.map),
                   estride, xs, ys };
    [e setComputePipelineState:g->p_moe_mv[base->type]];
    [e setBuffer:g->weights offset:0 atIndex:0];
    [e setBuffer:x offset:x_off atIndex:1];
    [e setBuffer:y offset:y_off atIndex:2];
    [e setBytes:&a length:sizeof(a) atIndex:3];
    [e setBuffer:g->moe_sel offset:0 atIndex:4];
    [e dispatchThreadgroups:MTLSizeMake((n_out + 3) / 4, nslots, 1)
      threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
}

static void enc_moe_actmul(gpu_t *g, id<MTLComputeCommandEncoder> e, int nff,
                           int nslots) {
    [e setComputePipelineState:g->p_moe_actmul];
    [e setBuffer:g->moe_hb offset:0 atIndex:0];
    [e setBuffer:g->moe_hb2 offset:0 atIndex:1];
    [e setBuffer:g->moe_selw offset:0 atIndex:2];
    [e setBytes:&nff length:4 atIndex:3];
    [e dispatchThreads:MTLSizeMake(nff, nslots, 1)
      threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

static void enc_moe_sum(gpu_t *g, id<MTLComputeCommandEncoder> e,
                        id<MTLBuffer> out, NSUInteger out_off,
                        int n, int nslots) {
    [e setComputePipelineState:g->p_moe_sum];
    [e setBuffer:out offset:out_off atIndex:0];
    [e setBuffer:g->moe_eout offset:0 atIndex:1];
    [e setBytes:&n length:4 atIndex:2];
    [e setBytes:&nslots length:4 atIndex:3];
    [e dispatchThreads:MTLSizeMake(n, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

// Metal's KV cache *is* the backend buffer (unified memory: m->kcache points at
// g->kc.contents), and the CPU path is about to read and overwrite those rows.
// Releasing the buffers here would pull them out from under the fallback. Keep
// gpu_owner reachable for model_free(), but make m->gpu NULL so no future
// forward tries to submit Metal work.
void gpu_disable(model_t *m) {
    if (!m) return;
    m->gpu = NULL;
    m->gpu_layers = 0;
}

// Metal has no batched-decode kernels yet, so it declines the microbatch and
// model_batch_decode decodes sequentially. The port is the same shape as the
// CUDA one (per-column position and per-sequence KV buffer, batched twins of
// the batch-1 matvec kernels) and is tracked in FUTURE.md Phase 6; unified
// memory removes the KV upload/copyback half of it entirely.
gpu_batch *gpu_batch_create(model_t **seqs, int n) {
    (void)seqs; (void)n;
    return NULL;
}

void gpu_batch_free(gpu_batch *b) {
    (void)b;
}

bool gpu_batch_decode(gpu_batch *b, const int *idx, const int32_t *tok,
                      const int *pos, int n, float **out) {
    (void)b; (void)idx; (void)tok; (void)pos; (void)n; (void)out;
    return false;
}

void gpu_free(model_t *m) {
    if (!m) return;
    gpu_t *g = m->gpu_owner ? (gpu_t *)m->gpu_owner : (gpu_t *)m->gpu;
    m->gpu = NULL;
    m->gpu_owner = NULL;
    if (!g) return;
    // The KV cache pointers alias MTLBuffer.contents. Detach before releasing
    // the buffers so model_free() never calls free() on borrowed memory.
    if (m->kv_owner == KV_OWNER_GPU_BACKEND) {
        m->kcache = NULL;
        m->vcache = NULL;
        m->kv_owner = KV_OWNER_MALLOC;
    }
    gpu_release_state(g, m->n_layer);
}

bool gpu_forward_batch(model_t *m, const int32_t *tokens, int n, int pos,
                       bool want_logits, float **logits) {
    if (m->n_expert == 0 && n > 1 && metal_batch_enabled()) {
        if (metal_ensure_batch(m, n)) {
            float *lg = gpu_forward_native_batch(m, tokens, n, pos);
            if (!lg) return false;
            if (logits) *logits = want_logits ? lg : NULL;
            return true;
        }
        fprintf(stderr, "gpu: Metal prompt batch scratch allocation failed — "
                "using per-token submits\n");
    }
    float *lg = NULL;
    for (int b = 0; b < n; b++) {
        lg = gpu_forward(m, tokens[b], pos + b);
        if (!lg) return false;
    }
    if (logits) *logits = want_logits ? lg : NULL;
    return true;
}

static NSUInteger foff(size_t elems) {
    return (NSUInteger)(elems * sizeof(float));
}

static void enc_moe_ffn(gpu_t *g, id<MTLComputeCommandEncoder> e, model_t *m,
                        layer_t *ly, NSUInteger xbo) {
    int n_embd = m->n_embd;
    int ne = m->n_expert;
    int used = m->n_expert_used;
    int nff = m->n_ff_exp;
    uint64_t gstride = (uint64_t)nff *
                       ggml_row_size(ly->ffn_gate_exps->type, n_embd);
    uint64_t ustride = (uint64_t)nff *
                       ggml_row_size(ly->ffn_up_exps->type, n_embd);
    uint64_t dstride = (uint64_t)n_embd *
                       ggml_row_size(ly->ffn_down_exps->type, nff);

    enc_mv(g, e, m, ly->ffn_gate_inp, g->xb, xbo, g->moe_logits, 0,
           n_embd, ne, nil);
    enc_moe_route(g, e, ne, used);
    enc_moe_mv(g, e, m, ly->ffn_gate_exps, gstride, g->xb, xbo,
               g->moe_hb, 0, n_embd, nff, used, 0, nff);
    enc_moe_mv(g, e, m, ly->ffn_up_exps, ustride, g->xb, xbo,
               g->moe_hb2, 0, n_embd, nff, used, 0, nff);
    enc_moe_actmul(g, e, nff, used);
    enc_moe_mv(g, e, m, ly->ffn_down_exps, dstride, g->moe_hb, 0,
               g->moe_eout, 0, nff, n_embd, used, nff, n_embd);
    enc_moe_sum(g, e, g->xb, xbo, n_embd, used);
}

static float *gpu_forward_native_batch(model_t *m, const int32_t *tokens,
                                       int n, int pos) {
    gpu_t *g = m->gpu;
    int n_embd = m->n_embd;
    int q_dim  = m->n_head * m->head_dim;
    int kv_dim = m->n_head_kv * m->head_dim;
    int xdim   = q_dim > n_embd ? q_dim : n_embd;

    size_t ers = ggml_row_size(m->tok_embd->type, n_embd);
    for (int b = 0; b < n; b++) {
        dequant_row(m->tok_embd->type,
                    (uint8_t *)m->tok_embd->data + (size_t)tokens[b] * ers,
                    (float *)g->x.contents + (size_t)b * n_embd, n_embd);
    }

    id<MTLCommandBuffer> cb = [g->queue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];

    for (int l = 0; l < m->n_layer; l++) {
        layer_t *ly = &m->layers[l];
        int hd = model_head_dim(m, l);
        int n_kv = model_n_head_kv(m, l);
        int q_dim_l = model_q_dim(m, l);
        int kv_dim_l = model_kv_dim(m, l);
        int window = model_is_swa(m, l) ? m->swa_window : 0;
        for (int b = 0; b < n; b++) {
            int p = pos + b;
            NSUInteger xo = foff((size_t)b * n_embd);
            NSUInteger xbo = foff((size_t)b * xdim);
            NSUInteger xb2o = foff((size_t)b * xdim);
            NSUInteger qo = foff((size_t)b * q_dim);
            NSUInteger kto = foff((size_t)b * kv_dim);
            NSUInteger vto = foff((size_t)b * kv_dim);
            NSUInteger hbo = foff((size_t)b * m->n_ff);
            NSUInteger hb2o = foff((size_t)b * m->n_ff);
            NSUInteger atto = foff((size_t)b * (size_t)m->n_head * m->n_ctx);
            NSUInteger logo = foff((size_t)b * m->n_vocab);
            size_t row_b = model_kv_row_bytes(m, l);
            uint64_t kv_off = model_kv_byte_off(m, l) + (uint64_t)p * row_b;
            int q8 = m->kv_q8;
            int kv_units = q8 ? kv_dim_l / 32 : kv_dim_l;

            enc_rmsnorm(g, e, g->x, xo, g->xb, xbo, g->attn_norm[l],
                        n_embd, m->rms_eps);
            enc_mv(g, e, m, ly->wq, g->xb, xbo, g->q,  qo,
                   n_embd, q_dim_l,  g->bq[l]);
            enc_mv(g, e, m, ly->wk, g->xb, xbo, g->kt, kto,
                   n_embd, kv_dim_l, g->bk[l]);
            enc_mv(g, e, m, ly->wv, g->xb, xbo, g->vt, vto,
                   n_embd, kv_dim_l, g->bv[l]);
            if (g->qn[l]) enc_qknorm(g, e, m, g->q,  qo,  g->qn[l], m->n_head, hd);
            if (g->kn[l]) enc_qknorm(g, e, m, g->kt, kto, g->kn[l], n_kv, hd);
            enc_rope(g, e, m, g->q,  qo,  m->n_head, p, l);
            enc_rope(g, e, m, g->kt, kto, n_kv,      p, l);

            [e setComputePipelineState:g->p_store];
            [e setBuffer:g->kt offset:kto atIndex:0];
            [e setBuffer:g->vt offset:vto atIndex:1];
            [e setBuffer:g->kc offset:0 atIndex:2];
            [e setBuffer:g->vc offset:0 atIndex:3];
            [e setBytes:&kv_dim_l length:4 atIndex:4];
            [e setBytes:&kv_off length:8 atIndex:5];
            [e setBytes:&q8 length:4 atIndex:6];
            [e dispatchThreads:MTLSizeMake(kv_units, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];

            attn_args aa = { hd, m->n_head, n_kv, m->n_ctx, p,
                             (uint64_t)model_kv_byte_off(m, l),
                             model_attn_scale(m, l), q8, window };
            [e setComputePipelineState:g->p_attn];
            [e setBuffer:g->q   offset:qo atIndex:0];
            [e setBuffer:g->kc  offset:0 atIndex:1];
            [e setBuffer:g->vc  offset:0 atIndex:2];
            [e setBuffer:g->att offset:atto atIndex:3];
            [e setBuffer:g->xb2 offset:xb2o atIndex:4];
            [e setBytes:&aa length:sizeof(aa) atIndex:5];
            [e dispatchThreadgroups:MTLSizeMake(m->n_head, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

            enc_mv(g, e, m, ly->wo, g->xb2, xb2o, g->xb, xbo,
                   q_dim_l, n_embd, g->bo[l]);
            enc_elem(g, e, g->p_add, g->x, xo, g->xb, xbo, n_embd);

            enc_rmsnorm(g, e, g->x, xo, g->xb, xbo, g->ffn_norm[l],
                        n_embd, m->rms_eps);
            enc_mv(g, e, m, ly->w_gate, g->xb, xbo, g->hb,  hbo,
                   n_embd, m->n_ff, nil);
            enc_mv(g, e, m, ly->w_up,   g->xb, xbo, g->hb2, hb2o,
                   n_embd, m->n_ff, nil);
            enc_elem(g, e, g->p_silu, g->hb, hbo, g->hb2, hb2o, m->n_ff);
            enc_mv(g, e, m, ly->w_down, g->hb, hbo, g->xb, xbo,
                   m->n_ff, n_embd, nil);
            enc_elem(g, e, g->p_add, g->x, xo, g->xb, xbo, n_embd);
            if (l == m->n_layer - 1) {
                enc_rmsnorm(g, e, g->x, xo, g->xb, xbo, g->out_norm,
                            n_embd, m->rms_eps);
                enc_mv(g, e, m, m->output, g->xb, xbo, g->logits, logo,
                       n_embd, m->n_vocab, nil);
            }
        }
    }

    [e endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (metal_command_failed(cb)) {
        fprintf(stderr, "gpu: command buffer failed — falling back to CPU\n");
        return NULL;
    }
    if (metal_env_on("RUNNER_METAL_STATS"))
        fprintf(stderr, "metal: native prompt batch n=%d command_buffers=1\n", n);
    return (float *)g->logits.contents + (size_t)(n - 1) * m->n_vocab;
}

static float *gpu_forward(model_t *m, int token, int pos) {
    gpu_t *g = m->gpu;
    int n_embd = m->n_embd;

    // token embedding on CPU (one row), straight into the shared buffer
    size_t ers = ggml_row_size(m->tok_embd->type, n_embd);
    dequant_row(m->tok_embd->type,
                (uint8_t *)m->tok_embd->data + (size_t)token * ers,
                (float *)g->x.contents, n_embd);

    id<MTLCommandBuffer> cb = [g->queue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];

    for (int l = 0; l < m->n_layer; l++) {
        layer_t *ly = &m->layers[l];
        int hd = model_head_dim(m, l);
        int n_kv = model_n_head_kv(m, l);
        int q_dim_l = model_q_dim(m, l);
        int kv_dim_l = model_kv_dim(m, l);
        int window = model_is_swa(m, l) ? m->swa_window : 0;
        size_t row_b = model_kv_row_bytes(m, l);
        uint64_t kv_off = model_kv_byte_off(m, l) + (uint64_t)pos * row_b;
        int q8 = m->kv_q8;
        int kv_units = q8 ? kv_dim_l / 32 : kv_dim_l;

        enc_rmsnorm(g, e, g->x, 0, g->xb, 0, g->attn_norm[l], n_embd, m->rms_eps);
        enc_mv(g, e, m, ly->wq, g->xb, 0, g->q,  0, n_embd, q_dim_l,  g->bq[l]);
        enc_mv(g, e, m, ly->wk, g->xb, 0, g->kt, 0, n_embd, kv_dim_l, g->bk[l]);
        enc_mv(g, e, m, ly->wv, g->xb, 0, g->vt, 0, n_embd, kv_dim_l, g->bv[l]);
        if (g->qn[l]) enc_qknorm(g, e, m, g->q,  0, g->qn[l], m->n_head, hd);
        if (g->kn[l]) enc_qknorm(g, e, m, g->kt, 0, g->kn[l], n_kv, hd);
        enc_rope(g, e, m, g->q,  0, m->n_head, pos, l);
        enc_rope(g, e, m, g->kt, 0, n_kv,      pos, l);

        [e setComputePipelineState:g->p_store];
        [e setBuffer:g->kt offset:0 atIndex:0];
        [e setBuffer:g->vt offset:0 atIndex:1];
        [e setBuffer:g->kc offset:0 atIndex:2];
        [e setBuffer:g->vc offset:0 atIndex:3];
        [e setBytes:&kv_dim_l length:4 atIndex:4];
        [e setBytes:&kv_off length:8 atIndex:5];
        [e setBytes:&q8 length:4 atIndex:6];
        [e dispatchThreads:MTLSizeMake(kv_units, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];

        attn_args aa = { hd, m->n_head, n_kv, m->n_ctx, pos,
                         (uint64_t)model_kv_byte_off(m, l),
                         model_attn_scale(m, l), q8, window };
        [e setComputePipelineState:g->p_attn];
        [e setBuffer:g->q   offset:0 atIndex:0];
        [e setBuffer:g->kc  offset:0 atIndex:1];
        [e setBuffer:g->vc  offset:0 atIndex:2];
        [e setBuffer:g->att offset:0 atIndex:3];
        [e setBuffer:g->xb2 offset:0 atIndex:4];
        [e setBytes:&aa length:sizeof(aa) atIndex:5];
        [e dispatchThreadgroups:MTLSizeMake(m->n_head, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

        enc_mv(g, e, m, ly->wo, g->xb2, 0, g->xb, 0, q_dim_l, n_embd, g->bo[l]);
        enc_elem(g, e, g->p_add, g->x, 0, g->xb, 0, n_embd);

        enc_rmsnorm(g, e, g->x, 0, g->xb, 0, g->ffn_norm[l], n_embd, m->rms_eps);
        if (ly->is_moe) {
            enc_moe_ffn(g, e, m, ly, 0);
        } else {
            enc_mv(g, e, m, ly->w_gate, g->xb, 0, g->hb,  0, n_embd, m->n_ff, nil);
            enc_mv(g, e, m, ly->w_up,   g->xb, 0, g->hb2, 0, n_embd, m->n_ff, nil);
            enc_elem(g, e, g->p_silu, g->hb, 0, g->hb2, 0, m->n_ff);
            enc_mv(g, e, m, ly->w_down, g->hb, 0, g->xb, 0, m->n_ff, n_embd, nil);
        }
        enc_elem(g, e, g->p_add, g->x, 0, g->xb, 0, n_embd);
    }

    enc_rmsnorm(g, e, g->x, 0, g->xb, 0, g->out_norm, n_embd, m->rms_eps);
    enc_mv(g, e, m, m->output, g->xb, 0, g->logits, 0, n_embd, m->n_vocab, nil);

    [e endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (metal_command_failed(cb)) {
        fprintf(stderr, "gpu: command buffer failed — falling back to CPU\n");
        return NULL;
    }
    return (float *)g->logits.contents;
}
