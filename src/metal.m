// Metal GPU backend: full single-token forward pass on Apple GPUs.
// Compiled without ARC; every object lives for the process lifetime.
#import <Metal/Metal.h>

#include "runner.h"
#include "kernels_metal.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

enum { METAL_TYPE_SLOTS = T_MXFP4 + 1 };

typedef struct {
    id<MTLDevice>       dev;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> p_rmsnorm, p_qknorm, p_headnorm, p_rope, p_store, p_attn;
    id<MTLComputePipelineState> p_silu, p_gelu, p_add, p_scale, p_head_transform;
    id<MTLComputePipelineState> p_moe_route, p_moe_actmul, p_moe_sum;
    id<MTLComputePipelineState> p_mv[METAL_TYPE_SLOTS];       // indexed by ggml type
    id<MTLComputePipelineState> p_mm[METAL_TYPE_SLOTS];       // tiled prefill GEMM
    id<MTLComputePipelineState> p_moe_mv[METAL_TYPE_SLOTS];   // indexed by ggml type
    id<MTLBuffer> weights;                      // wraps the model mmap (zero copy)
    bool          weights_copied;
    id<MTLBuffer> kc, vc;
    id<MTLBuffer> x, xb, xb2, q, kt, vt, hb, hb2, att, logits;
    id<MTLBuffer> moe_logits, moe_sel, moe_selw, moe_hb, moe_hb2, moe_eout;
    id<MTLBuffer> inv_freq, inv_freq_local, out_norm, dummy;
    id<MTLBuffer> *ppn;                 // gemma4 E-series per-layer post_norm
    id<MTLBuffer> ple, ple_tmp;         // [n][n_layer][P] slices, [n][P] gate
    id<MTLBuffer> *attn_norm, *ffn_norm;        // per layer
    id<MTLBuffer> *bq, *bk, *bv, *bo;           // per layer, may be nil
    id<MTLBuffer> *qn, *kn;                     // qwen3 per-head q/k norms
    id<MTLBuffer> *sinks;                       // gpt-oss attention sinks
    id<MTLBuffer> *gib, *geb, *ueb, *deb;       // gpt-oss MoE biases
    id<MTLBuffer> *pan, *pfn;                   // gemma sandwich norms
    id<MTLBuffer> *gpn1, *gprn2, *gpn2;         // gemma4 MoE branch norms
    id<MTLBuffer> *ggis, *gdsc;                 // gemma4 router/down scales
    id<MTLBuffer> suppress;                     // gemma never-emit token ids
    int batch_cap;                              // scratch rows allocated
} gpu_t;

// mirrors struct mv_args in kernels.metal — keep the field order in step
typedef struct { int n_in, n_out; uint64_t w_off; int has_bias;
                 int n_col, x_stride, y_stride, col_tile; } mv_args;
typedef struct { int n_in, n_out, n_col; uint64_t w_off;
                 int has_bias, x_stride, y_stride; } mm_args;
typedef struct { int n, x_stride, y_stride; float eps; } norm_args;
typedef struct { int kv_dim, q8, stride; uint64_t off, row_b; } store_args;
typedef struct { int head_dim, n_heads, half_dim, pos, neox; float mscale; int stride; } rope_args;
typedef struct { int head_dim, n_head, n_head_kv, n_ctx, pos; uint64_t l_off; float scale; int q8, window, has_sinks;
                 int q_stride, att_stride, out_stride; } attn_args;
typedef struct { int n_in, n_out; uint64_t w_off, estride; int xs, ys, has_bias, bias_stride; } moe_args;

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
        if (g->sinks) [g->sinks[l] release];
        if (g->gib) [g->gib[l] release];
        if (g->geb) [g->geb[l] release];
        if (g->ueb) [g->ueb[l] release];
        if (g->deb) [g->deb[l] release];
        if (g->pan) [g->pan[l] release];
        if (g->ppn) [g->ppn[l] release];
        if (g->pfn) [g->pfn[l] release];
        if (g->gpn1) [g->gpn1[l] release];
        if (g->gprn2) [g->gprn2[l] release];
        if (g->gpn2) [g->gpn2[l] release];
        if (g->ggis) [g->ggis[l] release];
        if (g->gdsc) [g->gdsc[l] release];
    }
    free(g->attn_norm); free(g->ffn_norm);
    free(g->bq); free(g->bk); free(g->bv); free(g->bo);
    free(g->qn); free(g->kn);
    free(g->sinks); free(g->gib); free(g->geb); free(g->ueb); free(g->deb);
    free(g->pan); free(g->pfn);
    free(g->gpn1); free(g->gprn2); free(g->gpn2); free(g->ggis); free(g->gdsc);
    id<MTLBuffer> bufs[] = { g->weights, g->kc, g->vc, g->x, g->xb, g->xb2,
                             g->q, g->kt, g->vt, g->hb, g->hb2, g->att,
                             g->logits, g->moe_logits, g->moe_sel, g->moe_selw,
                             g->moe_hb, g->moe_hb2, g->moe_eout,
                             g->inv_freq, g->inv_freq_local, g->out_norm,
                             g->dummy, g->suppress, g->ple, g->ple_tmp };
    for (size_t i = 0; i < sizeof(bufs) / sizeof(*bufs); i++) [bufs[i] release];
    for (int i = 0; i < METAL_TYPE_SLOTS; i++) [g->p_mv[i] release];
    for (int i = 0; i < METAL_TYPE_SLOTS; i++) [g->p_mm[i] release];
    for (int i = 0; i < METAL_TYPE_SLOTS; i++) [g->p_moe_mv[i] release];
    [g->p_rmsnorm release]; [g->p_qknorm release]; [g->p_headnorm release];
    [g->p_rope release]; [g->p_store release]; [g->p_attn release];
    [g->p_silu release]; [g->p_gelu release]; [g->p_add release];
    [g->p_scale release]; [g->p_head_transform release];
    [g->p_moe_route release]; [g->p_moe_actmul release];
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
    // eligibility keeps heterogeneous models off this path, but size off the
    // per-layer maxima anyway so the two sizing sites cannot drift apart
    int q_dim  = m->n_head * m->head_dim;
    int kv_dim = m->n_head_kv * m->head_dim;
    for (int l = 0; l < m->n_layer; l++) {
        if (model_q_dim(m, l)  > q_dim)  q_dim  = model_q_dim(m, l);
        if (model_kv_dim(m, l) > kv_dim) kv_dim = model_kv_dim(m, l);
    }
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
    int P = m->n_embd_ple;
    id<MTLBuffer> ple = nil, ple_tmp = nil;
    if (P > 0) {
        ple     = new_f32_scratch(g->dev, nb * (size_t)m->n_layer * P);
        ple_tmp = new_f32_scratch(g->dev, nb * (size_t)P);
    }
    if (!metal_buffer_ok(x) || !metal_buffer_ok(xb) || !metal_buffer_ok(xb2) ||
        !metal_buffer_ok(q) || !metal_buffer_ok(kt) || !metal_buffer_ok(vt) ||
        !metal_buffer_ok(hb) || !metal_buffer_ok(hb2) || !metal_buffer_ok(att) ||
        !metal_buffer_ok(logits) ||
        (P > 0 && (!metal_buffer_ok(ple) || !metal_buffer_ok(ple_tmp)))) {
        release_buf(x); release_buf(xb); release_buf(xb2); release_buf(q);
        release_buf(kt); release_buf(vt); release_buf(hb); release_buf(hb2);
        release_buf(att); release_buf(logits);
        release_buf(ple); release_buf(ple_tmp);
        return false;
    }

    release_buf(g->x); release_buf(g->xb); release_buf(g->xb2);
    release_buf(g->q); release_buf(g->kt); release_buf(g->vt);
    release_buf(g->hb); release_buf(g->hb2); release_buf(g->att);
    release_buf(g->logits);
    g->x = x; g->xb = xb; g->xb2 = xb2; g->q = q; g->kt = kt; g->vt = vt;
    g->hb = hb; g->hb2 = hb2; g->att = att; g->logits = logits;
    if (P > 0) {
        release_buf(g->ple); release_buf(g->ple_tmp);
        g->ple = ple; g->ple_tmp = ple_tmp;
    }
    g->batch_cap = n;
    return true;
}

// A device that exists is not a backend that works: if the shader library does
// not compile, gpu_init falls back and every run is CPU-speed. --caps exists so
// a scheduler can place work BEFORE dispatching, so reporting a usable Metal
// backend in that state would be a lie that costs a whole run. Compiling the
// library here is the same work gpu_init does moments later and only happens
// on the --caps path. tests/test_metal_shaders.m is the build-time gate.
bool gpu_available(char *name, int cap) {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) return false;
    NSError *err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:
                              [NSString stringWithUTF8String:k_metal_src]
                                           options:nil
                                             error:&err];
    if (!lib) {
        fprintf(stderr, "gpu: Metal device present but its shader library does "
                "not compile — reporting no GPU backend (every run would be "
                "CPU-speed): %s\n",
                err ? err.localizedDescription.UTF8String : "(no diagnostic)");
        [dev release];
        return false;
    }
    if (name) snprintf(name, cap, "%s", dev.name.UTF8String);
    [lib release];
    [dev release];
    return true;
}

bool gpu_mem_info(size_t *free_bytes, size_t *total_bytes) {
    // unified memory: the RAM reservation governs; no separate VRAM pool
    (void)free_bytes; (void)total_bytes;
    return false;
}

bool gpu_max_working_set(size_t *bytes) {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) return false;
    uint64_t ws = dev.recommendedMaxWorkingSetSize;
    [dev release];
    if (ws == 0) return false;
    if (bytes) *bytes = (size_t)ws;
    return true;
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
        case T_MXFP4:
            return true;
        default:
            return false;
    }
}

static bool metal_moe_type_ok(int type) {
    switch (type) {
        case T_F32: case T_F16: case T_Q8_0: case T_Q4_0:
        case T_Q4_K: case T_Q5_K: case T_Q6_K: case T_MXFP4:
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
    if (m->ffn_act != ACT_SILU &&
        !(m->gptoss && m->ffn_act == ACT_SWIGLU_OAI) &&
        !(m->moe_gemma && m->ffn_act == ACT_GELU)) {
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
        if (ly->exp_probs_b) {
            fprintf(stderr, "gpu: MoE router/expert bias has no Metal path yet — using CPU\n");
            return false;
        }
        if (ly->moe_gemma) {
            if (!ly->ffn_gate_inp || !ly->ffn_gate_up_exps ||
                !ly->ffn_down_exps || !ly->w_gate || !ly->w_up || !ly->w_down) {
                fprintf(stderr, "gpu: unsupported Gemma MoE tensor layout for Metal — using CPU\n");
                return false;
            }
            if (!gpu_type_ok(ly->ffn_gate_inp->type) ||
                !metal_moe_type_ok(ly->ffn_gate_up_exps->type) ||
                !metal_moe_type_ok(ly->ffn_down_exps->type)) {
                fprintf(stderr, "gpu: Gemma MoE tensor type is not on the metal backend yet — using CPU\n");
                return false;
            }
        } else {
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

static id<MTLBuffer> f32_buf_ones(id<MTLDevice> dev, const float *src, size_t n) {
    if (src) return f32_buf(dev, src, n);
    if (n > SIZE_MAX / sizeof(float)) return nil;
    float *tmp = malloc(n * sizeof(float));
    if (!tmp) return nil;
    for (size_t i = 0; i < n; i++) tmp[i] = 1.0f;
    id<MTLBuffer> b = [dev newBufferWithBytes:tmp length:n * sizeof(float)
                                      options:MTLResourceStorageModeShared];
    free(tmp);
    return b;
}


static float *gpu_forward_native_batch(model_t *m, const int32_t *tokens,
                                       int n, int pos);

// Tiled prefill GEMM (k_mm_*): simdgroup matrix units instead of one output
// element per simdgroup. Not bit-identical to the matvec path by construction
// — the weight is dequantized before the multiply and the sum is reassociated
// into 8-element matrix steps — so it answers to tests/test_tc_tol.c, the same
// tolerance gate the CUDA tensor-core prefill answers to, through this hook.
// -1 restores the env default (RUNNER_METAL_MM), 0 pins the matvec path on,
// 1 forces the tiled path wherever a kernel exists for the weight type.
enum { MM_ENV_UNSET = -2 };
static int g_mm_state = MM_ENV_UNSET;

void gpu_tc_force(int on) {
    g_mm_state = on < 0 ? MM_ENV_UNSET : (on != 0);
}

static bool metal_mm_on(void) {
    if (g_mm_state == MM_ENV_UNSET) {
        const char *e = getenv("RUNNER_METAL_MM");
        if (!e || !*e) g_mm_state = -1;                  // promoted default
        else g_mm_state = strcmp(e, "0") && strcmp(e, "off");
    }
    if (g_mm_state >= 0) return g_mm_state != 0;
    return true;   // promoted: every type with a k_mm_* kernel, prefill only
}
void gpu_moe_eager_force(int on) { (void)on; }

bool gpu_moe_ok(void) {
    return true;    // plain fused sparse-MoE routes and experts run on Metal
}

bool gpu_kv_q8_ok(void) {
    return true;
}

bool gpu_init(model_t *m) {
    if (m->attn_out_gate) {
        fprintf(stderr, "gpu: gated attention (afmoe) is not on the metal backend yet — using CPU\n");
        return false;
    }
    if (m->qwen35) {
        fprintf(stderr, "gpu: qwen35 hybrid path is not on the metal backend yet — using CPU\n");
        return false;
    }
    if (!metal_moe_supported(m))
        return false;
    if (m->ffn_act != ACT_SILU && m->ffn_act != ACT_GELU &&
        !(m->n_expert > 0 && m->gptoss && m->ffn_act == ACT_SWIGLU_OAI)) {
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
        if (!ly->wv && !m->v_rmsnorm) {
            fprintf(stderr, "gpu: '%s' layer layout is not on the metal backend yet — using CPU\n",
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
    g->p_headnorm     = mk_pipeline(dev, lib, @"k_head_rmsnorm");
    g->p_rope         = mk_pipeline(dev, lib, @"k_rope");
    g->p_store        = mk_pipeline(dev, lib, @"k_store_kv");
    g->p_attn         = mk_pipeline(dev, lib, @"k_attn");
    g->p_silu         = mk_pipeline(dev, lib, @"k_silu_mul");
    g->p_gelu         = mk_pipeline(dev, lib, @"k_gelu_mul");
    g->p_add          = mk_pipeline(dev, lib, @"k_add");
    g->p_scale        = mk_pipeline(dev, lib, @"k_scale");
    g->p_head_transform = mk_pipeline(dev, lib, @"k_head_transform");
    g->p_moe_route    = mk_pipeline(dev, lib, @"k_moe_route");
    g->p_moe_actmul   = mk_pipeline(dev, lib, @"k_moe_actmul");
    g->p_moe_sum      = mk_pipeline(dev, lib, @"k_moe_sum");
    g->p_mm[T_F32]   = mk_pipeline(dev, lib, @"k_mm_f32");
    g->p_mm[T_F16]   = mk_pipeline(dev, lib, @"k_mm_f16");
    g->p_mm[T_Q8_0]  = mk_pipeline(dev, lib, @"k_mm_q8_0");
    g->p_mm[T_Q4_0]  = mk_pipeline(dev, lib, @"k_mm_q4_0");
    g->p_mm[T_Q4_K]  = mk_pipeline(dev, lib, @"k_mm_q4_K");
    g->p_mm[T_Q6_K]  = mk_pipeline(dev, lib, @"k_mm_q6_K");
    g->p_mm[T_MXFP4] = mk_pipeline(dev, lib, @"k_mm_mxfp4");
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
    g->p_mv[T_MXFP4]  = mk_pipeline(dev, lib, @"k_mv_mxfp4");
    g->p_moe_mv[T_F32]  = mk_pipeline(dev, lib, @"k_moe_mv_f32");
    g->p_moe_mv[T_F16]  = mk_pipeline(dev, lib, @"k_moe_mv_f16");
    g->p_moe_mv[T_Q8_0] = mk_pipeline(dev, lib, @"k_moe_mv_q8_0");
    g->p_moe_mv[T_Q4_0] = mk_pipeline(dev, lib, @"k_moe_mv_q4_0");
    g->p_moe_mv[T_Q4_K] = mk_pipeline(dev, lib, @"k_moe_mv_q4_K");
    g->p_moe_mv[T_Q5_K] = mk_pipeline(dev, lib, @"k_moe_mv_q5_K");
    g->p_moe_mv[T_Q6_K] = mk_pipeline(dev, lib, @"k_moe_mv_q6_K");
    g->p_moe_mv[T_MXFP4] = mk_pipeline(dev, lib, @"k_moe_mv_mxfp4");
    [lib release];
    lib = nil;
    if (!g->p_rmsnorm || !g->p_rope || !g->p_store || !g->p_attn ||
        !g->p_silu || !g->p_gelu || !g->p_add || !g->p_scale ||
        !g->p_head_transform || !g->queue || !g->p_qknorm || !g->p_headnorm ||
        !g->p_moe_route || !g->p_moe_actmul || !g->p_moe_sum ||
        !g->p_mv[T_F32] || !g->p_mv[T_F16] || !g->p_mv[T_Q8_0] ||
        !g->p_mv[T_Q4_0] || !g->p_mv[T_Q4_1] ||
        !g->p_mv[T_Q5_0] || !g->p_mv[T_Q5_1] ||
        !g->p_mv[T_Q4_K] || !g->p_mv[T_Q5_K] || !g->p_mv[T_Q6_K] ||
        !g->p_mv[T_MXFP4])
        return gpu_init_fail(m, g, lib, "pipeline allocation");
    if (m->n_expert > 0 &&
        (!g->p_moe_mv[T_F32] || !g->p_moe_mv[T_F16] ||
         !g->p_moe_mv[T_Q8_0] || !g->p_moe_mv[T_Q4_0] ||
         !g->p_moe_mv[T_Q4_K] || !g->p_moe_mv[T_Q5_K] ||
         !g->p_moe_mv[T_Q6_K] || !g->p_moe_mv[T_MXFP4]))
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
        if (!g->weights) {
            // say what was asked for and what the device allows — the RAM
            // warning two lines below this in a load gives numbers, and a
            // bare "allocation failed" gives a scheduler nothing to reason
            // with (16 GB-Mac field report, 2026-08-05)
            char why[192];
            uint64_t ws = dev.recommendedMaxWorkingSetSize;
            snprintf(why, sizeof why,
                     "weight buffer allocation: %.1f GB requested, device "
                     "working-set limit %.1f GB%s",
                     wlen / 1e9, ws / 1e9,
                     (uint64_t)wlen > ws ? " — model exceeds Metal fit ceiling"
                                         : "");
            return gpu_init_fail(m, g, lib, why);
        }
    }

    // scratch sizes off the per-layer MAXIMA: gemma4 varies q/kv widths per
    // layer, and a global-scalar size would overrun on the widest layer
    int q_dim  = m->n_head * m->head_dim;
    int kv_dim = m->n_head_kv * m->head_dim;
    for (int l = 0; l < m->n_layer; l++) {
        if (model_q_dim(m, l)  > q_dim)  q_dim  = model_q_dim(m, l);
        if (model_kv_dim(m, l) > kv_dim) kv_dim = model_kv_dim(m, l);
    }
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
    if (m->n_embd_ple > 0) {
        g->ple     = NEWBUF(sizeof(float) * (size_t)m->n_layer * m->n_embd_ple);
        g->ple_tmp = NEWBUF(sizeof(float) * (size_t)m->n_embd_ple);
        if (!metal_buffer_ok(g->ple) || !metal_buffer_ok(g->ple_tmp))
            return gpu_init_fail(m, g, lib, "PLE scratch allocation");
    }
    g->hb     = NEWBUF(sizeof(float) * m->n_ff);
    g->hb2    = NEWBUF(sizeof(float) * m->n_ff);
    g->att    = NEWBUF(sizeof(float) * (size_t)m->n_head * m->n_ctx);
    g->logits = NEWBUF(sizeof(float) * m->n_vocab);
    g->dummy  = NEWBUF(4);
    if (m->n_expert > 0) {
        size_t used = (size_t)m->n_expert_used;
        size_t moe_ff = (size_t)m->n_ff_exp * (m->moe_gemma ? 2u : 1u);
        g->moe_logits = NEWBUF(sizeof(float) * (size_t)m->n_expert);
        g->moe_sel    = NEWBUF(sizeof(int)   * used);
        g->moe_selw   = NEWBUF(sizeof(float) * used);
        g->moe_hb     = NEWBUF(sizeof(float) * used * moe_ff);
        g->moe_hb2    = NEWBUF(sizeof(float) * used * moe_ff);
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
    if (m->n_suppress > 0)
        g->suppress = [dev newBufferWithBytes:m->suppress
                                       length:(size_t)m->n_suppress * sizeof(int32_t)
                                      options:MTLResourceStorageModeShared];
    if (!metal_buffer_ok(g->inv_freq) || !metal_buffer_ok(g->out_norm) ||
        (m->rope_inv_freq_local && !metal_buffer_ok(g->inv_freq_local)) ||
        (m->n_suppress > 0 && !metal_buffer_ok(g->suppress)))
        return gpu_init_fail(m, g, lib, "shared constant allocation");
    g->attn_norm = calloc(m->n_layer, sizeof(id));
    g->ffn_norm  = calloc(m->n_layer, sizeof(id));
    g->bq = calloc(m->n_layer, sizeof(id));
    g->bk = calloc(m->n_layer, sizeof(id));
    g->bv = calloc(m->n_layer, sizeof(id));
    g->bo = calloc(m->n_layer, sizeof(id));
    g->qn = calloc(m->n_layer, sizeof(id));
    g->kn = calloc(m->n_layer, sizeof(id));
    g->sinks = calloc(m->n_layer, sizeof(id));
    g->gib = calloc(m->n_layer, sizeof(id));
    g->geb = calloc(m->n_layer, sizeof(id));
    g->ueb = calloc(m->n_layer, sizeof(id));
    g->deb = calloc(m->n_layer, sizeof(id));
    g->ppn = calloc(m->n_layer, sizeof(id));
    g->pan = calloc(m->n_layer, sizeof(id));
    g->pfn = calloc(m->n_layer, sizeof(id));
    g->gpn1 = calloc(m->n_layer, sizeof(id));
    g->gprn2 = calloc(m->n_layer, sizeof(id));
    g->gpn2 = calloc(m->n_layer, sizeof(id));
    g->ggis = calloc(m->n_layer, sizeof(id));
    g->gdsc = calloc(m->n_layer, sizeof(id));
    if (!g->attn_norm || !g->ffn_norm || !g->bq || !g->bk || !g->bv ||
        !g->ppn ||
        !g->bo || !g->qn || !g->kn || !g->sinks || !g->gib || !g->geb ||
        !g->ueb || !g->deb || !g->pan || !g->pfn || !g->gpn1 ||
        !g->gprn2 || !g->gpn2 || !g->ggis || !g->gdsc)
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
        g->sinks[l] = f32_buf(dev, ly->attn_sinks, m->n_head);
        g->pan[l] = f32_buf(dev, ly->post_attn_norm_w, m->n_embd);
        g->ppn[l] = f32_buf(dev, ly->ple_post_norm, m->n_embd);
        g->pfn[l] = f32_buf(dev, ly->post_ffn_norm_w, m->n_embd);
        g->gib[l] = f32_buf(dev, ly->ffn_gate_inp_b, m->n_expert);
        g->geb[l] = f32_buf(dev, ly->ffn_gate_exps_b,
                            (size_t)m->n_expert * (size_t)m->n_ff_exp);
        g->ueb[l] = f32_buf(dev, ly->ffn_up_exps_b,
                            (size_t)m->n_expert * (size_t)m->n_ff_exp);
        g->deb[l] = f32_buf(dev, ly->ffn_down_exps_b,
                            (size_t)m->n_expert * (size_t)m->n_embd);
        if (ly->moe_gemma) {
            g->gpn1[l] = f32_buf_ones(dev, ly->ffn_post_norm1_w, m->n_embd);
            g->gprn2[l] = f32_buf_ones(dev, ly->ffn_pre_norm2_w, m->n_embd);
            g->gpn2[l] = f32_buf_ones(dev, ly->ffn_post_norm2_w, m->n_embd);
            float *gs = malloc(sizeof(float) * (size_t)m->n_embd);
            if (!gs) return gpu_init_fail(m, g, lib, "Gemma router scale allocation");
            float inv = 1.0f / sqrtf((float)m->n_embd);
            for (int i = 0; i < m->n_embd; i++)
                gs[i] = (ly->gate_inp_scale ? ly->gate_inp_scale[i] : 1.0f) * inv;
            g->ggis[l] = f32_buf(dev, gs, m->n_embd);
            free(gs);
            g->gdsc[l] = f32_buf_ones(dev, ly->down_exps_scale, m->n_expert);
        }
        if (!metal_buffer_ok(g->attn_norm[l]) ||
            !metal_buffer_ok(g->ffn_norm[l]) ||
            (ly->bq && !metal_buffer_ok(g->bq[l])) ||
            (ly->bk && !metal_buffer_ok(g->bk[l])) ||
            (ly->bv && !metal_buffer_ok(g->bv[l])) ||
            (ly->bo && !metal_buffer_ok(g->bo[l])) ||
            (ly->qnorm_w && !metal_buffer_ok(g->qn[l])) ||
            (ly->knorm_w && !metal_buffer_ok(g->kn[l])) ||
            (ly->attn_sinks && !metal_buffer_ok(g->sinks[l])) ||
            (ly->post_attn_norm_w && !metal_buffer_ok(g->pan[l])) ||
            (ly->post_ffn_norm_w && !metal_buffer_ok(g->pfn[l])) ||
            (ly->ffn_gate_inp_b && !metal_buffer_ok(g->gib[l])) ||
            (ly->ffn_gate_exps_b && !metal_buffer_ok(g->geb[l])) ||
            (ly->ffn_up_exps_b && !metal_buffer_ok(g->ueb[l])) ||
            (ly->ffn_down_exps_b && !metal_buffer_ok(g->deb[l])) ||
            (ly->moe_gemma &&
             (!metal_buffer_ok(g->gpn1[l]) || !metal_buffer_ok(g->gprn2[l]) ||
              !metal_buffer_ok(g->gpn2[l]) || !metal_buffer_ok(g->ggis[l]) ||
              !metal_buffer_ok(g->gdsc[l]))))
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

static void enc_rmsnorm_n(gpu_t *g, id<MTLComputeCommandEncoder> e,
                          id<MTLBuffer> x, NSUInteger x_off,
                          id<MTLBuffer> y, NSUInteger y_off, id<MTLBuffer> w,
                          int n, float eps,
                          int n_col, int x_stride, int y_stride) {
    norm_args a = { n, x_stride, y_stride, eps };
    [e setComputePipelineState:g->p_rmsnorm];
    [e setBuffer:x offset:x_off atIndex:0];
    [e setBuffer:y offset:y_off atIndex:1];
    [e setBuffer:w offset:0 atIndex:2];
    [e setBytes:&a length:sizeof(a) atIndex:3];
    [e dispatchThreadgroups:MTLSizeMake(n_col, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

static void enc_rmsnorm(gpu_t *g, id<MTLComputeCommandEncoder> e,
                        id<MTLBuffer> x, NSUInteger x_off,
                        id<MTLBuffer> y, NSUInteger y_off, id<MTLBuffer> w,
                        int n, float eps) {
    enc_rmsnorm_n(g, e, x, x_off, y, y_off, w, n, eps, 1, n, n);
}

// n_col columns in one dispatch (see the MV macros in kernels.metal): the
// weight row is walked once per column and stays in cache after the first, so
// a prompt batch stops re-streaming the whole weight matrix per token. Output
// is bit-identical to n_col separate enc_mv calls.
// Columns per threadgroup in the batched matmul. Tunable so the tradeoff can
// be re-measured on other Apple GPUs without a rebuild.
static int metal_col_tile(void) {
    static int v = -1;
    if (v < 0) {
        const char *s = getenv("RUNNER_METAL_COL_TILE");
        v = s && *s ? atoi(s) : 8;
        if (v < 1) v = 1;
    }
    return v;
}

static void enc_mv_n(gpu_t *g, id<MTLComputeCommandEncoder> e, model_t *m,
                     gguf_tensor *w, id<MTLBuffer> x, NSUInteger x_off,
                     id<MTLBuffer> y, NSUInteger y_off,
                     int n_in, int n_out, id<MTLBuffer> bias,
                     int n_col, int x_stride, int y_stride) {
    // Tiled GEMM when this is a real batch and a kernel exists for the weight
    // type. n_in must be a whole number of k-steps (every real model's is) and
    // K-quant kernels index a 256-superblock, so require that too.
    if (n_col > 1 && metal_mm_on() && g->p_mm[w->type] &&
        n_in % 32 == 0 &&
        !((w->type == T_Q4_K || w->type == T_Q6_K) && n_in % 256 != 0)) {
        mm_args ma = { n_in, n_out, n_col,
                       (uint64_t)((uint8_t *)w->data - (uint8_t *)m->gf.map),
                       bias != nil, x_stride, y_stride };
        [e setComputePipelineState:g->p_mm[w->type]];
        [e setBuffer:g->weights offset:0 atIndex:0];
        [e setBuffer:x offset:x_off atIndex:1];
        [e setBuffer:y offset:y_off atIndex:2];
        [e setBytes:&ma length:sizeof(ma) atIndex:3];
        [e setBuffer:bias ? bias : g->dummy offset:0 atIndex:4];
        [e dispatchThreadgroups:MTLSizeMake((n_out + 31) / 32,
                                            (n_col + 15) / 16, 1)
          threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        return;
    }
    int col_tile = metal_col_tile();
    if (col_tile > n_col) col_tile = n_col;
    mv_args a = { n_in, n_out,
                  (uint64_t)((uint8_t *)w->data - (uint8_t *)m->gf.map),
                  bias != nil, n_col, x_stride, y_stride, col_tile };
    [e setComputePipelineState:g->p_mv[w->type]];
    [e setBuffer:g->weights offset:0 atIndex:0];
    [e setBuffer:x offset:x_off atIndex:1];
    [e setBuffer:y offset:y_off atIndex:2];
    [e setBytes:&a length:sizeof(a) atIndex:3];
    [e setBuffer:bias ? bias : g->dummy offset:0 atIndex:4];
    // 128 threads = 4 simdgroups = 4 rows per threadgroup
    [e dispatchThreadgroups:MTLSizeMake((n_out + 3) / 4,
                                        (n_col + col_tile - 1) / col_tile, 1)
      threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
}

static void enc_mv(gpu_t *g, id<MTLComputeCommandEncoder> e, model_t *m,
                   gguf_tensor *w, id<MTLBuffer> x, NSUInteger x_off,
                   id<MTLBuffer> y, NSUInteger y_off,
                   int n_in, int n_out, id<MTLBuffer> bias) {
    enc_mv_n(g, e, m, w, x, x_off, y, y_off, n_in, n_out, bias,
             1, n_in, n_out);
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

static void enc_headnorm(gpu_t *g, id<MTLComputeCommandEncoder> e, model_t *m,
                         id<MTLBuffer> src, NSUInteger src_off,
                         id<MTLBuffer> dst, NSUInteger dst_off,
                         id<MTLBuffer> w, int n_heads, int hd) {
    float eps = m->rms_eps;
    int has_weight = w != nil;
    [e setComputePipelineState:g->p_headnorm];
    [e setBuffer:src offset:src_off atIndex:0];
    [e setBuffer:dst offset:dst_off atIndex:1];
    [e setBuffer:w ? w : g->dummy offset:0 atIndex:2];
    [e setBytes:&hd length:4 atIndex:3];
    [e setBytes:&eps length:4 atIndex:4];
    [e setBytes:&has_weight length:4 atIndex:5];
    [e dispatchThreadgroups:MTLSizeMake(n_heads, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
}

static void enc_rope_n(gpu_t *g, id<MTLComputeCommandEncoder> e, model_t *m,
                       id<MTLBuffer> v, NSUInteger v_off, int n_heads, int pos,
                       int layer, int n_col, int stride) {
    bool local = model_is_swa(m, layer);
    int hd = model_head_dim(m, layer);
    int rope_dim = model_rope_dim(m, layer);
    rope_args a = { hd, n_heads, rope_dim / 2, pos,
                    m->rope_neox, model_rope_mscale(m, layer), stride };
    [e setComputePipelineState:g->p_rope];
    [e setBuffer:v offset:v_off atIndex:0];
    [e setBuffer:(local && g->inv_freq_local) ? g->inv_freq_local : g->inv_freq
          offset:0 atIndex:1];
    [e setBytes:&a length:sizeof(a) atIndex:2];
    [e dispatchThreads:MTLSizeMake(a.half_dim, n_heads, n_col)
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

static void enc_scale(gpu_t *g, id<MTLComputeCommandEncoder> e,
                      id<MTLBuffer> x, NSUInteger x_off,
                      int n, float scale) {
    [e setComputePipelineState:g->p_scale];
    [e setBuffer:x offset:x_off atIndex:0];
    [e setBytes:&scale length:4 atIndex:1];
    [e setBytes:&n length:4 atIndex:2];
    [e dispatchThreads:MTLSizeMake(n, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

// Head transforms (logit softcap / suppressed tokens) deliberately do NOT
// live on the backend: model_forward_batch applies them on the host for both
// the batched and the solo path, so the two cannot drift apart. A Metal-side
// copy existed here and double-applied the softcap.
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
                       int n_in, int n_out, int nslots, int xs, int ys,
                       id<MTLBuffer> bias, int bias_stride) {
    moe_args a = { n_in, n_out,
                   (uint64_t)((uint8_t *)base->data - (uint8_t *)m->gf.map),
                   estride, xs, ys, bias != nil, bias_stride };
    [e setComputePipelineState:g->p_moe_mv[base->type]];
    [e setBuffer:g->weights offset:0 atIndex:0];
    [e setBuffer:x offset:x_off atIndex:1];
    [e setBuffer:y offset:y_off atIndex:2];
    [e setBytes:&a length:sizeof(a) atIndex:3];
    [e setBuffer:g->moe_sel offset:0 atIndex:4];
    [e setBuffer:bias ? bias : g->dummy offset:0 atIndex:5];
    [e dispatchThreadgroups:MTLSizeMake((n_out + 3) / 4, nslots, 1)
      threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
}

static void enc_moe_actmul(gpu_t *g, id<MTLComputeCommandEncoder> e,
                           id<MTLBuffer> gbuf, NSUInteger goff,
                           id<MTLBuffer> ubuf, NSUInteger uoff,
                           int nff, int nslots, int gss, int uss, int act) {
    int args[4] = { nff, gss, uss, act };
    [e setComputePipelineState:g->p_moe_actmul];
    [e setBuffer:gbuf offset:goff atIndex:0];
    [e setBuffer:ubuf offset:uoff atIndex:1];
    [e setBytes:args length:sizeof(args) atIndex:2];
    [e dispatchThreads:MTLSizeMake(nff, nslots, 1)
      threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

static void enc_moe_sum(gpu_t *g, id<MTLComputeCommandEncoder> e,
                        id<MTLBuffer> out, NSUInteger out_off,
                        int n, int nslots, int es, id<MTLBuffer> dscale) {
    int has_dscale = dscale != nil;
    [e setComputePipelineState:g->p_moe_sum];
    [e setBuffer:out offset:out_off atIndex:0];
    [e setBuffer:g->moe_eout offset:0 atIndex:1];
    [e setBuffer:g->moe_selw offset:0 atIndex:2];
    [e setBuffer:dscale ? dscale : g->dummy offset:0 atIndex:3];
    [e setBuffer:g->moe_sel offset:0 atIndex:4];
    [e setBytes:&n length:4 atIndex:5];
    [e setBytes:&nslots length:4 atIndex:6];
    [e setBytes:&es length:4 atIndex:7];
    [e setBytes:&has_dscale length:4 atIndex:8];
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

// The native prompt-batch encoder implements exactly the plain llama shape.
// Every feature beyond it — MoE, GELU dense FFN, embedding scale, weightless
// V norm / V-less layers, sandwich norms, per-layer output scale, logit
// softcap or suppressed tokens, heterogeneous per-layer geometry — is
// implemented only by the per-token path, so a model carrying any of them
// must take that path or the batch would be silently wrong (the gemma3/4
// families hit several of these at once).

bool gpu_forward_batch(model_t *m, const int32_t *tokens, int n, int pos,
                       bool want_logits, float **logits) {
    // One encoder for every n, decode included. There used to be a second,
    // per-token encoder here; keeping two implementations of the same layer
    // loop cost three defects (features silently missing behind an
    // eligibility check, a double-applied logit softcap, and wrong output on
    // real gemma-4 E2B weights) before it was removed. A scratch allocation
    // failure now falls back to the CPU loudly instead of to a second path.
    if (!metal_ensure_batch(m, n)) {
        fprintf(stderr, "gpu: Metal batch scratch allocation failed — "
                "releasing the backend, continuing on CPU\n");
        return false;
    }
    float *lg = gpu_forward_native_batch(m, tokens, n, pos);
    if (!lg) return false;
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

    int l = (int)(ly - m->layers);
    enc_mv(g, e, m, ly->ffn_gate_inp, g->xb, xbo, g->moe_logits, 0,
           n_embd, ne, g->gib[l]);
    enc_moe_route(g, e, ne, used);
    enc_moe_mv(g, e, m, ly->ffn_gate_exps, gstride, g->xb, xbo,
               g->moe_hb, 0, n_embd, nff, used, 0, nff, g->geb[l], nff);
    enc_moe_mv(g, e, m, ly->ffn_up_exps, ustride, g->xb, xbo,
               g->moe_hb2, 0, n_embd, nff, used, 0, nff, g->ueb[l], nff);
    enc_moe_actmul(g, e, g->moe_hb, 0, g->moe_hb2, 0,
                   nff, used, nff, nff, m->ffn_act);
    enc_moe_mv(g, e, m, ly->ffn_down_exps, dstride, g->moe_hb, 0,
               g->moe_eout, 0, nff, n_embd, used, nff, n_embd,
               g->deb[l], n_embd);
    enc_moe_sum(g, e, g->xb, xbo, n_embd, used, n_embd, nil);
}

// xo/xbo select this token's slice of the residual (g->x) and of g->xb. The
// remaining scratch (xb2, q, moe_*) is used at offset 0 by every token: the
// compute encoder is serial, so a token's MoE completes before the next one's
// begins, and by the time the FFN runs the attention has already drained xb2
// and q for the whole batch.
static void enc_gemma_moe_ffn(gpu_t *g, id<MTLComputeCommandEncoder> e,
                              model_t *m, layer_t *ly, int l,
                              NSUInteger xo, NSUInteger xbo) {
    int n_embd = m->n_embd;
    int ne = m->n_expert;
    int used = m->n_expert_used;
    int nff = m->n_ff_exp;
    int dff = m->n_ff;
    uint64_t gustride = (uint64_t)(2 * (size_t)nff) *
                        ggml_row_size(ly->ffn_gate_up_exps->type, n_embd);
    uint64_t dstride = (uint64_t)n_embd *
                       ggml_row_size(ly->ffn_down_exps->type, nff);

    // Dense shared branch: norm(x) -> GELU gate/up -> down -> post_norm_1.
    enc_rmsnorm(g, e, g->x, xo, g->xb2, 0, g->ffn_norm[l], n_embd, m->rms_eps);
    enc_mv(g, e, m, ly->w_gate, g->xb2, 0, g->hb, 0, n_embd, dff, nil);
    enc_mv(g, e, m, ly->w_up, g->xb2, 0, g->hb2, 0, n_embd, dff, nil);
    enc_elem(g, e, g->p_gelu, g->hb, 0, g->hb2, 0, dff);
    enc_mv(g, e, m, ly->w_down, g->hb, 0, g->xb, xbo, dff, n_embd, nil);
    enc_rmsnorm(g, e, g->xb, xbo, g->xb, xbo, g->gpn1[l], n_embd, m->rms_eps);

    // Routed branch: pre_norm_2 feeds experts; a separate weightless norm with
    // the folded 1/sqrt(n_embd)*gate_inp_scale vector feeds the router.
    enc_rmsnorm(g, e, g->x, xo, g->xb2, 0, g->gprn2[l], n_embd, m->rms_eps);
    enc_rmsnorm(g, e, g->x, xo, g->q, 0, g->ggis[l], n_embd, m->rms_eps);
    enc_mv(g, e, m, ly->ffn_gate_inp, g->q, 0, g->moe_logits, 0, n_embd, ne, nil);
    enc_moe_route(g, e, ne, used);
    enc_moe_mv(g, e, m, ly->ffn_gate_up_exps, gustride, g->xb2, 0,
               g->moe_hb, 0, n_embd, 2 * nff, used, 0, 2 * nff, nil, 0);
    enc_moe_actmul(g, e, g->moe_hb, 0, g->moe_hb, foff(nff),
                   nff, used, 2 * nff, 2 * nff, ACT_GELU);
    enc_moe_mv(g, e, m, ly->ffn_down_exps, dstride, g->moe_hb, 0,
               g->moe_eout, 0, nff, n_embd, used, 2 * nff, n_embd, nil, 0);
    enc_moe_sum(g, e, g->q, 0, n_embd, used, n_embd, g->gdsc[l]);
    enc_rmsnorm(g, e, g->q, 0, g->q, 0, g->gpn2[l], n_embd, m->rms_eps);
    enc_elem(g, e, g->p_add, g->xb, xbo, g->q, 0, n_embd);
}

// gemma-4 E-series per-layer embeddings, mirroring the CPU tail in model.c:
// gate(x) -> GELU-gated against this layer's slice of the per-layer embedding
// table -> project back to n_embd -> own RMS norm -> into the residual. Runs
// on the post-FFN residual and BEFORE the layer output scale.
//
// m->ple is filled on the host by model_ple_prepass before the backend is
// called, so the slice is copied into a shared buffer once per forward rather
// than recomputed here.
static void enc_ple(gpu_t *g, id<MTLComputeCommandEncoder> e, model_t *m,
                    layer_t *ly, int l, int n, int xdim) {
    int P = m->n_embd_ple, n_embd = m->n_embd;
    enc_mv_n(g, e, m, ly->ple_gate, g->x, 0, g->ple_tmp, 0,
             n_embd, P, nil, n, n_embd, P);
    // gate *= gelu-gated slice: p_gelu computes g[i] = gelu(g[i]) * u[i],
    // which is exactly gated_act(ACT_GELU, gate, slice)
    for (int b = 0; b < n; b++)
        enc_elem(g, e, g->p_gelu, g->ple_tmp, foff((size_t)b * P),
                 g->ple, foff(((size_t)b * m->n_layer + l) * P), P);
    enc_mv_n(g, e, m, ly->ple_proj, g->ple_tmp, 0, g->xb, 0,
             P, n_embd, nil, n, P, xdim);
    enc_rmsnorm_n(g, e, g->xb, 0, g->xb, 0, g->ppn[l],
                  n_embd, m->rms_eps, n, xdim, xdim);
    for (int b = 0; b < n; b++)
        enc_elem(g, e, g->p_add, g->x, foff((size_t)b * n_embd),
                 g->xb, foff((size_t)b * xdim), n_embd);
}

// RUNNER_METAL_NAN_TRACE=1: submit after every layer and report the first one
// whose residual carries a NaN/Inf. The GPU path has no equivalent of
// RUNNER_DEBUG_ACT — without this, a backend that silently produces NaN logits
// can only be bisected by guesswork. Costs one command buffer per layer, so it
// is opt-in and read once.
static bool metal_nan_trace(void) {
    static int on = -1;
    if (on < 0) { const char *v = getenv("RUNNER_METAL_NAN_TRACE"); on = v && *v && strcmp(v, "0"); }
    return on > 0;
}

static bool metal_scan_bad(const float *p, int n, const char *what, int layer) {
    for (int i = 0; i < n; i++) {
        uint32_t u; memcpy(&u, &p[i], 4);
        if ((u & 0x7f800000u) == 0x7f800000u) {   // NaN or Inf, -ffast-math safe
            fprintf(stderr, "metal-nan: L%d %s[%d] = %s\n", layer, what, i,
                    (u & 0x7fffffu) ? "NaN" : "Inf");
            return true;
        }
    }
    return false;
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
        float *xp = (float *)g->x.contents + (size_t)b * n_embd;
        dequant_row(m->tok_embd->type,
                    (uint8_t *)m->tok_embd->data + (size_t)tokens[b] * ers,
                    xp, n_embd);
        if (m->embd_scale != 1.0f)
            for (int i = 0; i < n_embd; i++) xp[i] *= m->embd_scale;
    }
    // E-series per-layer embedding table. model_forward_batch deliberately
    // skips its own prepass under full offload (CUDA stages the table on the
    // device), so Metal builds it here from the scaled embeddings it just
    // wrote, then hands it over — shared storage, so a copy into unified
    // memory rather than a transfer.
    if (m->n_embd_ple > 0 && m->ple && g->ple) {
        model_ple_prepass(m, tokens, n, (const float *)g->x.contents,
                          m->ple, m->ple_tmp);
        memcpy(g->ple.contents, m->ple,
               sizeof(float) * (size_t)n * m->n_layer * m->n_embd_ple);
    }

    bool nantrace = metal_nan_trace();
    id<MTLCommandBuffer> cb = [g->queue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];

    for (int l = 0; l < m->n_layer; l++) {
        layer_t *ly = &m->layers[l];
        int hd = model_head_dim(m, l);
        int n_kv = model_n_head_kv(m, l);
        int q_dim_l = model_q_dim(m, l);
        int kv_dim_l = model_kv_dim(m, l);
        int window = model_is_swa(m, l) ? m->swa_window : 0;
        // Weight-heavy projections run ONCE for the whole batch: the simdgroup
        // walks its weight row per column, so the row is fetched from device
        // memory on the first token and cached for the rest. Per-token submits
        // instead re-streamed every weight matrix n times, which is why Metal
        // prefill used to run at decode speed. Bit-identical either way.
        enc_rmsnorm_n(g, e, g->x, 0, g->xb, 0, g->attn_norm[l],
                      n_embd, m->rms_eps, n, n_embd, xdim);
        enc_mv_n(g, e, m, ly->wq, g->xb, 0, g->q,  0,
                 n_embd, q_dim_l,  g->bq[l], n, xdim, q_dim);
        // gemma-4 E-series shared-KV layers project Q as usual but compute no
        // K/V of their own: they attend over the cache an earlier layer filled,
        // which model_kv_byte_off() below already aliases to. Projecting and
        // storing here would overwrite that source layer's rows.
        bool owns_kv = model_kv_owner(m, l) == l;
        if (owns_kv) {
            enc_mv_n(g, e, m, ly->wk, g->xb, 0, g->kt, 0,
                     n_embd, kv_dim_l, g->bk[l], n, xdim, kv_dim);
            // gemma-4 global layers publish no V projection: V is the raw K
            // projection, taken before K is normed/roped (as on the CPU path).
            if (ly->wv)
                enc_mv_n(g, e, m, ly->wv, g->xb, 0, g->vt, 0,
                         n_embd, kv_dim_l, g->bv[l], n, xdim, kv_dim);
        }

        for (int b = 0; b < n; b++) {
            NSUInteger qo = foff((size_t)b * q_dim);
            NSUInteger kto = foff((size_t)b * kv_dim);
            NSUInteger vto = foff((size_t)b * kv_dim);
            if (g->qn[l]) enc_qknorm(g, e, m, g->q,  qo,  g->qn[l], m->n_head, hd);
            if (!owns_kv) continue;
            if (m->v_rmsnorm)
                enc_headnorm(g, e, m, ly->wv ? g->vt : g->kt, ly->wv ? vto : kto,
                             g->vt, vto, nil, n_kv, hd);
            if (g->kn[l]) enc_qknorm(g, e, m, g->kt, kto, g->kn[l], n_kv, hd);
        }
        {
            size_t row_b = model_kv_row_bytes(m, l);
            int q8 = m->kv_q8;
            int kv_units = q8 ? kv_dim_l / 32 : kv_dim_l;

            // rope/store/attention each take the batch in one dispatch: the
            // kernels derive their column's position from pos + col, so every
            // token still rotates at, writes to, and attends over exactly the
            // range a per-token submit gave it.
            enc_rope_n(g, e, m, g->q,  0, m->n_head, pos, l, n, q_dim);
            if (owns_kv) {
                enc_rope_n(g, e, m, g->kt, 0, n_kv, pos, l, n, kv_dim);

                store_args sa = { kv_dim_l, q8, kv_dim,
                                  model_kv_byte_off(m, l) + (uint64_t)pos * row_b,
                                  (uint64_t)row_b };
                [e setComputePipelineState:g->p_store];
                [e setBuffer:g->kt offset:0 atIndex:0];
                [e setBuffer:g->vt offset:0 atIndex:1];
                [e setBuffer:g->kc offset:0 atIndex:2];
                [e setBuffer:g->vc offset:0 atIndex:3];
                [e setBytes:&sa length:sizeof(sa) atIndex:4];
                [e dispatchThreads:MTLSizeMake(kv_units, n, 1)
                  threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            }

            attn_args aa = { hd, m->n_head, n_kv, m->n_ctx, pos,
                             (uint64_t)model_kv_byte_off(m, l),
                             model_attn_scale(m, l), q8, window,
                             g->sinks[l] != nil,
                             q_dim, m->n_head * m->n_ctx, xdim };
            [e setComputePipelineState:g->p_attn];
            [e setBuffer:g->q   offset:0 atIndex:0];
            [e setBuffer:g->kc  offset:0 atIndex:1];
            [e setBuffer:g->vc  offset:0 atIndex:2];
            [e setBuffer:g->att offset:0 atIndex:3];
            [e setBuffer:g->xb2 offset:0 atIndex:4];
            [e setBytes:&aa length:sizeof(aa) atIndex:5];
            [e setBuffer:g->sinks[l] ? g->sinks[l] : g->dummy offset:0 atIndex:6];
            [e dispatchThreadgroups:MTLSizeMake(m->n_head, n, 1)
              threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        }

        enc_mv_n(g, e, m, ly->wo, g->xb2, 0, g->xb, 0,
                 q_dim_l, n_embd, g->bo[l], n, xdim, xdim);
        if (g->pan[l])
            enc_rmsnorm_n(g, e, g->xb, 0, g->xb, 0, g->pan[l],
                          n_embd, m->rms_eps, n, xdim, xdim);
        for (int b = 0; b < n; b++)
            enc_elem(g, e, g->p_add, g->x, foff((size_t)b * n_embd),
                     g->xb, foff((size_t)b * xdim), n_embd);

        if (ly->moe_gemma || ly->is_moe) {
            // The MoE FFN itself stays per-token — routing picks different
            // experts for each token, so there is no shared weight tile to
            // amortize. Everything around it (attention, projections, norms)
            // already ran once for the whole batch.
            for (int b = 0; b < n; b++) {
                NSUInteger xo = foff((size_t)b * n_embd);
                NSUInteger xbo = foff((size_t)b * xdim);
                if (ly->moe_gemma) {
                    enc_gemma_moe_ffn(g, e, m, ly, l, xo, xbo);
                } else {
                    enc_rmsnorm(g, e, g->x, xo, g->xb, xbo, g->ffn_norm[l],
                                n_embd, m->rms_eps);
                    enc_moe_ffn(g, e, m, ly, xbo);
                }
            }
        } else {
            // gemma-4 E2B varies the FFN width per layer; hb/hb2 are sized
            // for the max, so pack the batch at THIS layer's width.
            int nff_l = ly->n_ff;
            enc_rmsnorm_n(g, e, g->x, 0, g->xb, 0, g->ffn_norm[l],
                          n_embd, m->rms_eps, n, n_embd, xdim);
            enc_mv_n(g, e, m, ly->w_gate, g->xb, 0, g->hb,  0,
                     n_embd, nff_l, nil, n, xdim, nff_l);
            enc_mv_n(g, e, m, ly->w_up,   g->xb, 0, g->hb2, 0,
                     n_embd, nff_l, nil, n, xdim, nff_l);
            // hb/hb2 are contiguous across the batch, so the activation is one
            // dispatch over the whole batch rather than n
            enc_elem(g, e, m->ffn_act == ACT_GELU ? g->p_gelu : g->p_silu,
                     g->hb, 0, g->hb2, 0, n * nff_l);
            enc_mv_n(g, e, m, ly->w_down, g->hb, 0, g->xb, 0,
                     nff_l, n_embd, nil, n, nff_l, xdim);
        }
        if (g->pfn[l])
            enc_rmsnorm_n(g, e, g->xb, 0, g->xb, 0, g->pfn[l],
                          n_embd, m->rms_eps, n, xdim, xdim);
        for (int b = 0; b < n; b++)
            enc_elem(g, e, g->p_add, g->x, foff((size_t)b * n_embd),
                     g->xb, foff((size_t)b * xdim), n_embd);
        // Order matters: the E-series branch reads the post-FFN residual of
        // EVERY token, so it has to run before any token's output scale.
        if (ly->ple_gate) enc_ple(g, e, m, ly, l, n, xdim);
        if (ly->out_scale != 1.0f && ly->out_scale != 0.0f)
            for (int b = 0; b < n; b++)
                enc_scale(g, e, g->x, foff((size_t)b * n_embd), n_embd,
                          ly->out_scale);
        if (nantrace) {
            [e endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            if (metal_scan_bad((const float *)g->x.contents,
                               n * n_embd, "resid", l))
                return NULL;
            cb = [g->queue commandBuffer];
            e = [cb computeCommandEncoder];
        }
        // Only the last position's logits are ever read (see the return
        // below), and the vocab matmul is the widest in the model — doing it
        // for every prompt token was pure waste.
        if (l == m->n_layer - 1) {
            NSUInteger xo = foff((size_t)(n - 1) * n_embd);
            NSUInteger xbo = foff((size_t)(n - 1) * xdim);
            enc_rmsnorm(g, e, g->x, xo, g->xb, xbo, g->out_norm,
                        n_embd, m->rms_eps);
            enc_mv(g, e, m, m->output, g->xb, xbo, g->logits,
                   foff((size_t)(n - 1) * m->n_vocab),
                   n_embd, m->n_vocab, nil);
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


