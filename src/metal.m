// Metal GPU backend: full single-token forward pass on Apple GPUs.
// Compiled without ARC; every object lives for the process lifetime.
#import <Metal/Metal.h>

#include "runner.h"
#include "kernels_metal.h"
#include "compat.h"   // plat_ram_available_bytes: the residency guard below
#include "metal_admission.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

enum { METAL_TYPE_SLOTS = T_MXFP4 + 1 };
// Weight-mmap wraps. Enough for a 64 GB monolithic file at the M1's 4.29 GB
// per-buffer ceiling, with room to spare; exceeding it is reported, not
// silently truncated.
enum { METAL_MAX_WBUF = 24 };

typedef struct {
    id<MTLDevice>       dev;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> p_rmsnorm, p_qknorm, p_headnorm, p_rope, p_store, p_attn;
    id<MTLComputePipelineState> p_attn_chunk, p_attn_comb;
    id<MTLComputePipelineState> p_attn_coop, p_attn_chunk_coop;  // cooperative KV score read
    id<MTLComputePipelineState> p_silu, p_gelu, p_add, p_scale, p_head_transform;
    id<MTLComputePipelineState> p_sigmul;   // attention output gate (x *= sigmoid(g))
    id<MTLComputePipelineState> p_moe_route, p_moe_actmul, p_moe_sum;
    id<MTLComputePipelineState> p_mv[METAL_TYPE_SLOTS];       // indexed by ggml type
    id<MTLComputePipelineState> p_mm[METAL_TYPE_SLOTS];       // tiled prefill GEMM
    id<MTLComputePipelineState> p_mvf[METAL_TYPE_SLOTS];      // fast decode matvec
    id<MTLComputePipelineState> p_moe_mv[METAL_TYPE_SLOTS];   // indexed by ggml type
    // The model mmap, wrapped zero-copy. Usually ONE buffer; more when the file
    // exceeds maxBufferLength, which on an M1 is 4.29 GB against a 5.73 GB
    // working set (0.75x) and is the ceiling that actually bites for a big
    // monolithic GGUF.
    //
    // The split is at TENSOR boundaries, so no tensor ever straddles two
    // buffers and no kernel had to learn about this: a dispatch reads exactly
    // one weight tensor, so binding the containing buffer and passing an
    // offset within it is all that changes. Adjacent wraps may overlap by one
    // page, because a no-copy wrap must start page-aligned and tensors are
    // not; wrapping the same read-only pages twice costs a page of mapping and
    // keeps every tensor whole.
    id<MTLBuffer> wbuf[METAL_MAX_WBUF];
    uint64_t      wbuf_base[METAL_MAX_WBUF];   // file offset of each wrap's byte 0
    uint64_t      wbuf_end[METAL_MAX_WBUF];    // one past its last file offset
    int           n_wbuf;
    bool          weights_copied;
    id<MTLBuffer> kc, vc;
    id<MTLBuffer> x, xb, xb2, q, kt, vt, hb, hb2, att, logits;
    id<MTLBuffer> agate;             // [n][q_dim] attention output gate scratch
    id<MTLBuffer> att_acc, att_ms;   // chunked-decode partials
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
typedef struct { int head_dim, n_head, n_head_kv, n_ctx, pos; uint64_t l_off; float scale;
                 int q8, window, chunk, n_chunks, q_stride, att_stride; } attn_chunk_args;
typedef struct { int head_dim, n_head, n_chunks, has_sinks, out_stride; } attn_comb_args;

// Chunked decode attention. The split is chosen from the HEAD COUNT, not a
// fixed span: the point of chunking is to fill the GPU, so what matters is
// n_head * n_chunks (the threadgroup count), and n_head varies per model.
// Targeting ~64 threadgroups put an 8-head model at 8 chunks, which is where
// a measured sweep on an M1 landed too -- 3137 tokens of context: single-pass
// 7.32, chunk 256 -> 7.86, 384 -> 7.94, 512 -> 7.85, 1024 -> 7.51, and 384 is
// exactly span/8. Fewer, larger chunks starve the GPU; more, smaller ones pay
// combine cost for parallelism it cannot use.
// RUNNER_METAL_ATTN_CHUNK sets an explicit chunk size for measurement, and 0
// disables the path entirely (the single-pass kernel is the A/B baseline).
// Revised 2026-08-07. The sweep above never tested 128, and 128 wins: at 3259
// tokens on gemma-3-4b, chunk 128 gives 8.130 tok/s against 8.060 for the
// span/8 the target above produces (medians of 3 interleaved rounds, arms
// disjoint: [8.13,8.14] vs [8.06,8.07]). Every point the two sweeps share
// agrees -- 256 and 512 land at 7.96/7.95 here against 7.86/7.85 there -- so
// this is a new point on the curve, not a contradiction of the old one.
//
// The target stays expressed in threadgroups rather than a flat chunk size,
// because the right answer does depend on n_head: a 32-head model already gets
// 32 threadgroups from the single-pass kernel and needs little splitting, while
// an 8-head model on an 8-core GPU is starved. Raising the target to 256 leaves
// many-head models where they were and only makes few-head models finer.
//
// HONEST SIZE OF THIS TUNE. Against the previous target, interleaved, 3 rounds:
// GPU time 119.6-119.7 vs 120.7-120.9 ms/tok, disjoint, ~1.0%. End to end the
// medians differ by +0.87% but the arms OVERLAP, so it is not separable from
// noise on this machine. Kept because the direct measure of the thing changed
// is clean and nothing regresses (short-context decode 9.46 tok/s, unmoved).
// A second A/B meant to settle it was ABANDONED, not reported: macOS began
// rendering an animated aerial wallpaper on the same GPU partway through and
// both arms fell ~7%. Watch for that before trusting any GPU number on an idle
// Mac -- WallpaperAerialsExtension and WindowServer are the processes to check.
//
// For scale, the split itself is worth +9.4% over RUNNER_METAL_ATTN_CHUNK=0
// (7.975 vs 7.290 tok/s at 3259 ctx, arms disjoint). This tunes it, it does
// not replace it.
//
// Provenance note: this change was pushed inside 5d7808b, whose message
// describes only a Makefile fix -- `git add -A` swept it in. The measurement
// record lives here rather than in that commit for exactly that reason.
#define METAL_ATTN_MAX_CHUNKS 64
#define METAL_ATTN_TARGET_GROUPS 256
// Below this many positions per chunk the split is all overhead: at 40 tokens
// of context an 8-head model would otherwise be cut into 8 chunks of 5, and
// short-context decode measured 9.28 -> 9.20 tok/s doing exactly that.
//
// This is a FLOOR on the chunk size, not a veto on chunking. It used to be the
// latter -- a target implying chunks below this disabled the split entirely --
// which is why raising the target alone would have turned the path off at
// exactly the contexts it helps. Short context still falls back to the
// single-pass kernel, but via the n_chunks < 2 test below, which is the
// condition that actually means "nothing to split".
#define METAL_ATTN_MIN_CHUNK 128
static int metal_attn_chunk_override(void) {
    static int v = -2;
    if (v == -2) {
        const char *e = getenv("RUNNER_METAL_ATTN_CHUNK");
        v = -1;
        if (e && *e) { long n = strtol(e, NULL, 10); if (n >= 0) v = (int)n; }
    }
    return v;
}

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
    [g->p_attn_coop release]; [g->p_attn_chunk_coop release];
    for (int i = 0; i < g->n_wbuf; i++) [g->wbuf[i] release];
    id<MTLBuffer> bufs[] = { g->kc, g->vc, g->x, g->xb, g->xb2,
                             g->q, g->kt, g->vt, g->hb, g->hb2, g->att,
                             g->logits, g->moe_logits, g->moe_sel, g->moe_selw,
                             g->moe_hb, g->moe_hb2, g->moe_eout,
                             g->inv_freq, g->inv_freq_local, g->out_norm,
                             g->dummy, g->suppress, g->ple, g->ple_tmp,
                             g->agate };
    for (size_t i = 0; i < sizeof(bufs) / sizeof(*bufs); i++) [bufs[i] release];
    for (int i = 0; i < METAL_TYPE_SLOTS; i++) [g->p_mv[i] release];
    for (int i = 0; i < METAL_TYPE_SLOTS; i++) [g->p_mm[i] release];
    for (int i = 0; i < METAL_TYPE_SLOTS; i++) [g->p_mvf[i] release];
    for (int i = 0; i < METAL_TYPE_SLOTS; i++) [g->p_moe_mv[i] release];
    [g->p_rmsnorm release]; [g->p_qknorm release]; [g->p_headnorm release];
    [g->p_rope release]; [g->p_store release]; [g->p_attn release];
    [g->p_attn_chunk release]; [g->p_attn_comb release];
    [g->p_silu release]; [g->p_gelu release]; [g->p_add release];
    [g->p_sigmul release];
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
    // Chunked decode attention partials: one (max, sum, hd-vector) per
    // (head, chunk). Decode only, so no batch dimension -- n_col > 1 already
    // has grid.y parallelism and keeps the single-pass kernel.
    id<MTLBuffer> att_acc = new_f32_scratch(g->dev, (size_t)m->n_head *
                                            METAL_ATTN_MAX_CHUNKS *
                                            (size_t)model_head_dim(m, 0));
    id<MTLBuffer> att_ms  = new_f32_scratch(g->dev, (size_t)m->n_head *
                                            METAL_ATTN_MAX_CHUNKS * 2);
    int P = m->n_embd_ple;
    id<MTLBuffer> ple = nil, ple_tmp = nil;
    if (P > 0) {
        ple     = new_f32_scratch(g->dev, nb * (size_t)m->n_layer * P);
        ple_tmp = new_f32_scratch(g->dev, nb * (size_t)P);
    }
    id<MTLBuffer> agate = nil;
    if (m->attn_out_gate)
        agate = new_f32_scratch(g->dev, nb * (size_t)q_dim);
    if (!metal_buffer_ok(x) || !metal_buffer_ok(xb) || !metal_buffer_ok(xb2) ||
        !metal_buffer_ok(q) || !metal_buffer_ok(kt) || !metal_buffer_ok(vt) ||
        !metal_buffer_ok(hb) || !metal_buffer_ok(hb2) || !metal_buffer_ok(att) ||
        !metal_buffer_ok(logits) ||
        !metal_buffer_ok(att_acc) || !metal_buffer_ok(att_ms) ||
        (P > 0 && (!metal_buffer_ok(ple) || !metal_buffer_ok(ple_tmp))) ||
        (m->attn_out_gate && !metal_buffer_ok(agate))) {
        release_buf(x); release_buf(xb); release_buf(xb2); release_buf(q);
        release_buf(kt); release_buf(vt); release_buf(hb); release_buf(hb2);
        release_buf(att); release_buf(logits);
        release_buf(att_acc); release_buf(att_ms);
        release_buf(ple); release_buf(ple_tmp); release_buf(agate);
        return false;
    }

    release_buf(g->x); release_buf(g->xb); release_buf(g->xb2);
    release_buf(g->q); release_buf(g->kt); release_buf(g->vt);
    release_buf(g->hb); release_buf(g->hb2); release_buf(g->att);
    release_buf(g->logits); release_buf(g->att_acc); release_buf(g->att_ms);
    g->x = x; g->xb = xb; g->xb2 = xb2; g->q = q; g->kt = kt; g->vt = vt;
    g->hb = hb; g->hb2 = hb2; g->att = att; g->logits = logits;
    g->att_acc = att_acc; g->att_ms = att_ms;
    if (P > 0) {
        release_buf(g->ple); release_buf(g->ple_tmp);
        g->ple = ple; g->ple_tmp = ple_tmp;
    }
    if (m->attn_out_gate) {
        release_buf(g->agate);
        g->agate = agate;
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

// k_metal_sha is generated into kernels_metal.h alongside the source string
// itself, so it can only agree with the shaders this binary actually holds.
const char *gpu_shader_source_sha(void) { return k_metal_sha; }

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
        case T_F32: case T_F16: case T_BF16: case T_Q8_0: case T_Q4_0:
        case T_Q4_1: case T_Q5_0: case T_Q5_1: case T_Q2_K: case T_Q3_K:
        case T_Q4_K:
        case T_Q5_K: case T_Q6_K: case T_IQ4_NL: case T_IQ4_XS: case T_MXFP4:
            return true;
        default:
            return false;
    }
}

// The --caps answer for this backend, sourced from the admission test above so
// the advertised list and the loader agree by construction.
bool gpu_quant_ok(int type) { return gpu_type_ok(type); }

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
    // Threadgroup memory is the occupancy currency: an Apple GPU core has
    // 32 KB of it, so a kernel asking 14 KB gets 2 resident threadgroups and
    // one asking 8 KB gets 4. That ratio is invisible from throughput alone,
    // and a tile-shape sweep that cannot see it is guessing.
    if (p && metal_env_on("RUNNER_METAL_STATS"))
        fprintf(stderr, "metal-pipeline %-16s tgmem=%5lu B  max_threads=%lu\n",
                name.UTF8String,
                (unsigned long)p.staticThreadgroupMemoryLength,
                (unsigned long)p.maxTotalThreadsPerThreadgroup);
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

// Metal has no tensor-core GEMM; the counter exists so the gate links and can
// state "never dispatched here" rather than failing to build.
unsigned long gpu_tc_dispatches(void) { return 0; }

static bool metal_mm_on(void) {
    if (g_mm_state == MM_ENV_UNSET) {
        const char *e = getenv("RUNNER_METAL_MM");
        if (!e || !*e) g_mm_state = -1;                  // promoted default
        else g_mm_state = strcmp(e, "0") && strcmp(e, "off");
    }
    if (g_mm_state >= 0) return g_mm_state != 0;
    return true;   // promoted: every type with a k_mm_* kernel, prefill only
}
// Fast decode matvec (k_mvf_*): the reassociating twin of the identity matvec,
// selected at n_col == 1 only. See the kernel-family comment in kernels.metal
// for what it does that identity forbids, and tests/test_mv_tol.c for the gate.
//
// Default: OFF. It ships opt-in until a measurement promotes it, exactly as
// the fused CPU int8 dot does — a lever that has not cleared its bar on a
// machine is not a default anywhere. `RUNNER_METAL_MV=1` opts in; -1 through
// gpu_mv_force restores the env default.
enum { MV_ENV_UNSET = -2 };
static int g_mv_state = MV_ENV_UNSET;

void gpu_mv_force(int on) {
    g_mv_state = on < 0 ? MV_ENV_UNSET : (on != 0);
}

// Cooperative KV score read (k_attn_coop / k_attn_chunk_coop). It reassociates
// the per-row dot into a simd_sum over 32 lane partials, so it cannot live on
// the identity route -- it answers to tests/test_attn_tol.c the way the tiled
// prefill GEMM answers to test_tc_tol.c.
//
// PROMOTED 2026-08-17, and it is opt-OUT now: `RUNNER_METAL_ATTN_COOP=0` pins
// the byte-identical kernel back, exactly as `RUNNER_METAL_MM=0` does for
// prefill. The cert that bought this is in the suite's
// m1-night-results-2026-08-17.md -- 0/64 teacher-forced flips on every local
// model that engages the route, both KV cache formats, at a measured
// +3.0-4.3% decode across 2.3k-8.1k spans.
enum { COOP_ENV_UNSET = -2 };
static int g_coop_state = COOP_ENV_UNSET;
static unsigned long g_coop_dispatches = 0;

void gpu_attn_coop_force(int on) {
    g_coop_state = on < 0 ? COOP_ENV_UNSET : (on != 0);
}
unsigned long gpu_attn_coop_dispatches(void) { return g_coop_dispatches; }

static bool metal_attn_coop_on(void) {
    if (g_coop_state == COOP_ENV_UNSET) {
        const char *e = getenv("RUNNER_METAL_ATTN_COOP");
        if (!e || !*e) g_coop_state = -1;                 // promoted default
        else g_coop_state = strcmp(e, "0") && strcmp(e, "off");
    }
    if (g_coop_state >= 0) return g_coop_state != 0;
    return true;   // promoted: decode attention, every arch that reaches it
}

static unsigned long g_mv_dispatches = 0;
unsigned long gpu_mv_dispatches(void) { return g_mv_dispatches; }

static bool metal_mv_fast_on(void) {
    // Cache the env read: this sits in the per-matvec dispatch path, and
    // gpu_mv_force(-1) resets to UNSET so the next call re-reads it.
    if (g_mv_state == MV_ENV_UNSET) {
        const char *e = getenv("RUNNER_METAL_MV");
        g_mv_state = e && *e && strcmp(e, "0") && strcmp(e, "off");
    }
    return g_mv_state > 0;
}

void gpu_moe_eager_force(int on) { (void)on; }

bool gpu_moe_ok(void) {
    return true;    // plain fused sparse-MoE routes and experts run on Metal
}

bool gpu_eseries_ok(void) {
    // The per-layer-embedding branch landed in 0.1.11 and is gated by
    // `make test-metal-eseries`. Before that this was honestly false, which is
    // exactly the window a scheduler had no way to see.
    return true;
}

bool gpu_kv_q8_ok(void) {
    return true;
}

// Largest K such that the prologue plus layers [0,K) fit `budget` bytes, and
// whether a prefix wrap is legal for this file at all.
//
// A zero-copy wrap must be one contiguous range, and every kernel indexes
// g->weights by a FILE offset, so the only shape needing no arithmetic change
// is a prefix: [0, end of blk.K-1). That requires non-layer tensors to sit
// before blk.0 and blocks to be ordered -- true of gemma-4-26B, gemma-3-4b,
// e2b, Trinity-Nano and gpt-oss, five architectures, but a property of how
// converters write files rather than of the format. So it is CHECKED, and a
// file that violates it gets full offload or CPU, never a truncated wrap
// pointed at the wrong weights.
// Would a partial split actually help? Only if the weights the CPU keeps still
// fit in available RAM. Wiring K layers into the Metal working set makes them
// non-evictable, which is the whole point -- but it also removes that memory
// from the page cache the remaining layers stream through, so on a heavily
// oversubscribed machine it makes the tail page harder than before. Explicit
// --gpu-layers bypasses this: an operator asking for a specific split gets it.
static bool metal_wrap_check(gpu_t *g, model_t *m, uint64_t wlen);

// Per-buffer ceiling for the weight wraps. RUNNER_METAL_MAX_BUF overrides the
// device's maxBufferLength so the multi-buffer path can be exercised on a
// machine whose models all fit one buffer -- which is every machine this was
// developed on. Without a knob the split path would ship untested.
static uint64_t metal_wbuf_cap(id<MTLDevice> dev) {
    const char *e = getenv("RUNNER_METAL_MAX_BUF");
    if (e && *e) {
        long long v = atoll(e);
        if (v > 0) return (uint64_t)v;
    }
    return (uint64_t)dev.maxBufferLength;
}

// Wrap [0, wlen) of the model mmap in as few zero-copy buffers as the
// per-buffer ceiling allows, cutting only at TENSOR boundaries so no tensor is
// ever split across two of them. That is what lets every kernel stay unchanged:
// a dispatch reads exactly one weight tensor, so the caller binds the
// containing buffer and passes an offset within it.
//
// Wraps may overlap by up to a page. A no-copy wrap must begin page-aligned
// and tensors are not page-aligned, so a cut rounds its base DOWN to a page,
// which re-includes the tail of the previous group. Those bytes are read-only
// and never addressed through the second wrap; the cost is one page of extra
// mapping per cut, and the alternative (cutting mid-tensor) is a correctness
// bug.
static bool metal_wrap_weights(gpu_t *g, model_t *m, id<MTLDevice> dev,
                               uint64_t wlen, size_t page) {
    uint64_t cap = metal_wbuf_cap(dev);
    g->n_wbuf = 0;
    if (wlen == 0 || cap == 0) return false;

    // Tensor extents inside the wrapped prefix, in file order. GGUF writes
    // them ascending, but nothing in the format promises it, and a descending
    // file would silently produce overlapping groups -- so sort.
    uint64_t n = m->gf.n_tensors;
    typedef struct { uint64_t beg, end; } ext;
    ext *ex = calloc(n ? n : 1, sizeof(ext));
    if (!ex) return false;
    uint64_t n_ex = 0;
    for (uint64_t i = 0; i < n; i++) {
        gguf_tensor *t = &m->gf.tensors[i];
        if (!t->data) continue;
        uint64_t beg = (uint64_t)((uint8_t *)t->data - (uint8_t *)m->gf.map);
        uint64_t end = beg + t->nbytes;
        if (beg >= wlen) continue;          // beyond the wrapped prefix
        if (end > wlen) end = wlen;
        ex[n_ex].beg = beg;
        ex[n_ex].end = end;
        n_ex++;
    }
    if (n_ex == 0) { free(ex); return false; }
    for (uint64_t i = 1; i < n_ex; i++) {   // insertion sort: near-sorted input
        ext k = ex[i];
        uint64_t j = i;
        while (j > 0 && ex[j - 1].beg > k.beg) { ex[j] = ex[j - 1]; j--; }
        ex[j] = k;
    }

    uint64_t gi = 0;
    while (gi < n_ex) {
        if (g->n_wbuf >= METAL_MAX_WBUF) {
            fprintf(stderr, "gpu: this model needs more than %d weight buffers "
                    "at a %.2f GB per-buffer limit — not wrapping\n",
                    METAL_MAX_WBUF, cap / 1e9);
            free(ex);
            return false;
        }
        uint64_t base = ex[gi].beg & ~(uint64_t)(page - 1);
        uint64_t last = ex[gi].end;
        if (last - base > cap) {
            // One tensor bigger than the ceiling. Nothing here can split it,
            // and pretending otherwise would wrap it short.
            fprintf(stderr, "gpu: a single %.2f GB tensor exceeds the %.2f GB "
                    "per-buffer limit — not wrapping\n",
                    (double)(ex[gi].end - ex[gi].beg) / 1e9, cap / 1e9);
            free(ex);
            return false;
        }
        uint64_t gj = gi + 1;
        while (gj < n_ex && ex[gj].end - base <= cap) { last = ex[gj].end; gj++; }

        uint64_t len = (last - base + page - 1) & ~(uint64_t)(page - 1);
        if (base + len > m->gf.map_size) len = m->gf.map_size - base;
        id<MTLBuffer> b = [dev newBufferWithBytesNoCopy:(uint8_t *)m->gf.map + base
                                                 length:(NSUInteger)len
                                                options:MTLResourceStorageModeShared
                                            deallocator:nil];
        if (!b) {
            for (int i = 0; i < g->n_wbuf; i++) { [g->wbuf[i] release]; g->wbuf[i] = nil; }
            g->n_wbuf = 0;
            free(ex);
            return false;
        }
        g->wbuf[g->n_wbuf] = b;
        g->wbuf_base[g->n_wbuf] = base;
        g->wbuf_end[g->n_wbuf] = base + len;
        g->n_wbuf++;
        gi = gj;
    }
    free(ex);
    if (!metal_wrap_check(g, m, wlen)) {
        for (int i = 0; i < g->n_wbuf; i++) { [g->wbuf[i] release]; g->wbuf[i] = nil; }
        g->n_wbuf = 0;
        return false;
    }
    if (g->n_wbuf > 1)
        fprintf(stderr, "gpu: weights wrapped in %d buffers (%.2f GB "
                "per-buffer limit)\n", g->n_wbuf, cap / 1e9);
    return true;
}

// Which wrap holds [off, off+len), and where inside it. The LENGTH is part of
// the question: a kernel reads a whole tensor, and a wrap that contains the
// first byte but not the last would read past its own mapping. Linear over at
// most METAL_MAX_WBUF entries, and one compare in the single-buffer case.
//
// Returns nil only when no single wrap contains the range, which
// metal_wrap_check() has already ruled out at init -- so a nil here is a
// programming error, and the callers say so rather than binding something
// plausible and returning wrong numbers.
static id<MTLBuffer> metal_wbuf_for(gpu_t *g, uint64_t off, uint64_t len,
                                    uint64_t *within) {
    for (int i = 0; i < g->n_wbuf; i++)
        if (off >= g->wbuf_base[i] && off + len <= g->wbuf_end[i]) {
            *within = off - g->wbuf_base[i];
            return g->wbuf[i];
        }
    *within = 0;
    return nil;
}

// Bind helper for the three sites that address the weight mmap. Keeps the
// fallback in ONE place: with a single wrap the resolver always succeeds, so
// this degrades to exactly the old behaviour and the diagnostic can never fire.
static uint64_t metal_bind_weights(gpu_t *g, id<MTLComputeCommandEncoder> e,
                                   uint64_t off, uint64_t len) {
    uint64_t within = 0;
    id<MTLBuffer> wb = metal_wbuf_for(g, off, len, &within);
    if (!wb) {
        static bool told;
        if (!told) {
            told = true;
            fprintf(stderr, "gpu: internal — no weight wrap holds [%llu,+%llu); "
                    "results from this model are not trustworthy\n",
                    (unsigned long long)off, (unsigned long long)len);
        }
        wb = g->wbuf[0];
        within = off;
    }
    [e setBuffer:wb offset:0 atIndex:0];
    return within;
}

// Every range a kernel will address must sit inside ONE wrap. The split is
// made at tensor boundaries so this holds by construction; checking it turns a
// silent wrong-answer bug into a refusal that falls back to a copied buffer.
// MoE expert banks are checked at full tensor length because the kernel strides
// across experts from the base offset, so the whole stack has to be reachable.
static bool metal_wrap_check(gpu_t *g, model_t *m, uint64_t wlen) {
    for (uint64_t i = 0; i < m->gf.n_tensors; i++) {
        gguf_tensor *t = &m->gf.tensors[i];
        if (!t->data) continue;
        uint64_t beg = (uint64_t)((uint8_t *)t->data - (uint8_t *)m->gf.map);
        if (beg >= wlen) continue;
        uint64_t len = t->nbytes;
        if (beg + len > wlen) len = wlen - beg;
        uint64_t within;
        if (!metal_wbuf_for(g, beg, len, &within)) {
            fprintf(stderr, "gpu: tensor %s straddles a weight-buffer boundary "
                    "— not wrapping\n", t->name);
            return false;
        }
    }
    return true;
}

static bool metal_partial_pays(const model_t *m, uint64_t wrapped) {
    uint64_t have = plat_ram_available_bytes();
    if (!have) return false;                 // cannot tell: do not gamble
    // The WRAPPED part must fit in RAM. This is the condition that matters and
    // it is not the one this guard originally checked.
    //
    // The design behind partial offload argued the win was residency: wiring
    // weights into the Metal working set makes them non-evictable, so an
    // oversubscribed model would stop paging. That is wrong, and physics says
    // why -- you cannot pin more bytes than the machine has. Measured twice on
    // an 8 GB M1 with ~2.8 GB available:
    //   gemma-4-26B, 3.8 GB wrapped:  1.5-1.7 -> 0.21 tok/s   (8x slower)
    //   e4b-q4km,    4.3 GB wrapped:  5.6     -> 0.15 tok/s   (35x slower)
    // Both passed the old tail-fits check. Neither could hold its wrap
    // resident, so the pinned pages thrashed against everything else.
    //
    // What survives is narrow and honest: partial offload helps only when the
    // WHOLE model fits in RAM but its weights plus KV/scratch exceed Metal's
    // aggregate working-set budget. maxBufferLength no longer enters this
    // decision: the multi-buffer wrapper removes that per-resource ceiling.
    // In the fits-in-RAM case the split costs nothing in residency and gains
    // the ordinary compute blend. An oversubscribed model is refused however
    // the split is drawn, because whatever is pinned is taken from what the
    // rest must stream through.
    uint64_t total = m->gf.map_size;
    uint64_t margin = have / 8;              // KV, scratch, the OS
    return total + margin <= have && wrapped <= total;
}

static int metal_fit_layers(model_t *m, uint64_t budget, int cap,
                            bool *layout_ok, uint64_t *wrap_end) {
    *layout_ok = false;
    *wrap_end = 0;
    uint64_t *end_of = calloc((size_t)m->n_layer + 1, sizeof(uint64_t));
    if (!end_of) return 0;
    uint64_t prologue_end = 0;
    int prev_layer = -1;
    bool ok = true;
    for (uint64_t t = 0; t < m->gf.n_tensors && ok; t++) {
        gguf_tensor *w = &m->gf.tensors[t];
        uint64_t beg = (uint64_t)((uint8_t *)w->data - (uint8_t *)m->gf.map);
        uint64_t end = beg + w->nbytes;
        int li = strncmp(w->name, "blk.", 4) ? -1 : atoi(w->name + 4);
        if (li < 0) {
            if (prev_layer >= 0) { ok = false; break; }   // non-layer after a layer
            if (end > prologue_end) prologue_end = end;
        } else if (li >= m->n_layer || li < prev_layer) {
            ok = false;                                    // out of range or unordered
        } else {
            prev_layer = li;
            if (end > end_of[li + 1]) end_of[li + 1] = end;
        }
    }
    int K = 0;
    if (ok && end_of[m->n_layer] > 0) {
        end_of[0] = prologue_end;
        for (int i = 1; i <= m->n_layer; i++)
            if (end_of[i] < end_of[i - 1]) end_of[i] = end_of[i - 1];
        *layout_ok = true;
        int top = (cap > 0 && cap < m->n_layer) ? cap : m->n_layer;
        for (int i = 1; i <= top; i++)
            if (end_of[i] <= budget) { K = i; *wrap_end = end_of[i]; } else break;
    }
    free(end_of);
    return K;
}

bool gpu_init(model_t *m) {
    if (m->qwen35) {
        fprintf(stderr, "gpu: qwen35 hybrid path is not on the metal backend yet — using CPU\n");
        return false;
    }
    if (m->attn_out_gate && m->n_expert > 0) {
        // the dense output gate (muse-glimmer) is implemented below; gate +
        // sparse MoE (afmoe) has never been run against the identity gates,
        // so it keeps its CPU fallback rather than gaining silent support
        fprintf(stderr, "gpu: gated attention with sparse MoE (afmoe) is not on the metal backend yet — using CPU\n");
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
                              ly->w_gate, ly->w_up, ly->w_down, ly->wq_gate };
        for (int i = 0; i < 8; i++)
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
    g->p_attn_coop    = mk_pipeline(dev, lib, @"k_attn_coop");
    g->p_attn_chunk_coop = mk_pipeline(dev, lib, @"k_attn_chunk_coop");
    g->p_attn_chunk   = mk_pipeline(dev, lib, @"k_attn_chunk");
    g->p_attn_comb    = mk_pipeline(dev, lib, @"k_attn_combine");
    g->p_silu         = mk_pipeline(dev, lib, @"k_silu_mul");
    g->p_gelu         = mk_pipeline(dev, lib, @"k_gelu_mul");
    g->p_add          = mk_pipeline(dev, lib, @"k_add");
    g->p_sigmul       = mk_pipeline(dev, lib, @"k_sigmoid_mul");
    g->p_scale        = mk_pipeline(dev, lib, @"k_scale");
    g->p_head_transform = mk_pipeline(dev, lib, @"k_head_transform");
    g->p_moe_route    = mk_pipeline(dev, lib, @"k_moe_route");
    g->p_moe_actmul   = mk_pipeline(dev, lib, @"k_moe_actmul");
    g->p_moe_sum      = mk_pipeline(dev, lib, @"k_moe_sum");
    g->p_mm[T_F32]   = mk_pipeline(dev, lib, @"k_mm_f32");
    g->p_mm[T_F16]   = mk_pipeline(dev, lib, @"k_mm_f16");
    g->p_mm[T_Q8_0]  = mk_pipeline(dev, lib, @"k_mm_q8_0");
    g->p_mm[T_Q4_0]  = mk_pipeline(dev, lib, @"k_mm_q4_0");
    g->p_mm[T_Q2_K]  = mk_pipeline(dev, lib, @"k_mm_q2_K");
    g->p_mm[T_Q3_K]  = mk_pipeline(dev, lib, @"k_mm_q3_K");
    g->p_mm[T_BF16]  = mk_pipeline(dev, lib, @"k_mm_bf16");
    g->p_mm[T_IQ4_NL] = mk_pipeline(dev, lib, @"k_mm_iq4_nl");
    g->p_mm[T_IQ4_XS] = mk_pipeline(dev, lib, @"k_mm_iq4_xs");
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
    g->p_mv[T_Q2_K]   = mk_pipeline(dev, lib, @"k_mv_q2_K");
    g->p_mv[T_Q3_K]   = mk_pipeline(dev, lib, @"k_mv_q3_K");
    g->p_mv[T_BF16]   = mk_pipeline(dev, lib, @"k_mv_bf16");
    g->p_mv[T_IQ4_NL] = mk_pipeline(dev, lib, @"k_mv_iq4_nl");
    g->p_mv[T_IQ4_XS] = mk_pipeline(dev, lib, @"k_mv_iq4_xs");
    g->p_mv[T_Q4_K]   = mk_pipeline(dev, lib, @"k_mv_q4_K");
    g->p_mv[T_Q5_K]   = mk_pipeline(dev, lib, @"k_mv_q5_K");
    g->p_mv[T_Q6_K]   = mk_pipeline(dev, lib, @"k_mv_q6_K");
    g->p_mv[T_MXFP4]  = mk_pipeline(dev, lib, @"k_mv_mxfp4");

    // Fast (reassociating) decode matvec, gated by RUNNER_METAL_MV. Only the
    // two types that carry real decode traffic on the models this was measured
    // against; a type with no entry here simply keeps the identity kernel, so
    // the coverage can grow one measured format at a time.
    g->p_mvf[T_Q4_0]  = mk_pipeline(dev, lib, @"k_mvf_q4_0");
    g->p_mvf[T_Q8_0]  = mk_pipeline(dev, lib, @"k_mvf_q8_0");
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
        !g->p_attn_chunk || !g->p_attn_comb ||
        !g->p_silu || !g->p_gelu || !g->p_add || !g->p_scale || !g->p_sigmul ||
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

    // Choose K. Full offload stays the default and the fast path; a partial
    // split is only entered when the whole file cannot be wrapped, because a
    // model that fits should never pay a CPU tail.
    //
    // The ceiling covers KV and scratch as well as weights, not weights alone
    // -- measured: a 5.17 GB wrap was refused under a 5.73 GB limit. So the
    // budget subtracts what this load will also allocate, and keeps a margin
    // on top rather than trusting the remainder to be exact.
    int gpu_K = m->n_layer;
    uint64_t ws_limit = dev.recommendedMaxWorkingSetSize;
    {
        // recommendedMaxWorkingSetSize is the aggregate residency ceiling.
        // The budget subtracts KV and scratch and keeps a safety margin.
        // maxBufferLength is smaller (4.29 GB against 5.73 GB on an M1), but
        // it is only the size of each individual wrap: metal_wrap_weights()
        // cuts a larger file into several tensor-boundary buffers. Clamping
        // this budget to maxBufferLength preserved the old ceiling in the
        // admission decision even after the multi-buffer implementation had
        // removed it from allocation.
        uint64_t other = (uint64_t)model_kv_byte_off(m, m->n_layer) * 2
                       + 512ull * 1024 * 1024;
        metal_weight_limits limits = {
            .working_set = ws_limit,
            .max_buffer = (uint64_t)dev.maxBufferLength,
        };
        uint64_t budget = metal_full_weight_budget(limits, other);
        int cap = m->gpu_layers_override;
        bool need_split = wlen > budget || (cap > 0 && cap < m->n_layer);
        if (need_split) {
            bool layout_ok = false;
            uint64_t wrap_end = 0;
            int fit = metal_fit_layers(m, budget, cap, &layout_ok, &wrap_end);
            if (!layout_ok) {
                if (wlen > budget)
                    return gpu_init_fail(m, g, lib,
                        "model exceeds the Metal working set and its tensor "
                        "layout forbids a partial split (non-layer tensors "
                        "after blk.0, or blocks out of order)");
                fprintf(stderr, "gpu: --gpu-layers ignored — tensor layout "
                        "does not allow a partial split\n");
            } else if (fit < 1) {
                return gpu_init_fail(m, g, lib,
                    "not even one layer fits the Metal working set");
            } else if (wlen > budget && cap <= 0 &&
                       !metal_partial_pays(m, wrap_end)) {
                // Auto-selected partial offload that would make things WORSE.
                // Measured on an 8 GB M1 with gemma-4-26B: wrapping 3.8 GB of
                // 14.4 left the remaining 10.6 GB of CPU-side weights fighting
                // an already-starved page cache, and decode went 1.5-1.7 ->
                // 0.21 tok/s. An 8x regression, enabled by default, on the
                // exact model this feature was built for.
                //
                // Residency only pays if the tail the CPU still owns fits in
                // RAM. When it does not, pinning part of the model just takes
                // memory from the part that is streaming.
                return gpu_init_fail(m, g, lib,
                    "model is larger than available RAM, so no split can be "
                    "held resident — CPU-only is faster (--gpu-layers forces "
                    "a split anyway)");
            } else {
                gpu_K = fit;
                if (gpu_K < m->n_layer) {
                    wlen = (size_t)((wrap_end + page - 1) & ~(uint64_t)(page - 1));
                    if (wlen > m->gf.map_size)
                        wlen = (m->gf.map_size + page - 1) & ~(size_t)(page - 1);
                }
            }
        }
    }

    if (!metal_wrap_weights(g, m, dev, wlen, page)) {
        // fallback: copy (costs RAM but still works). One buffer by
        // definition -- a copy this large only happens for a model well under
        // the per-buffer ceiling, since the ceiling is what forced the split.
        g->wbuf[0] = [dev newBufferWithBytes:m->gf.map
                                      length:m->gf.map_size
                                     options:MTLResourceStorageModeShared];
        g->wbuf_base[0] = 0;
        g->wbuf_end[0] = m->gf.map_size;
        g->n_wbuf = g->wbuf[0] ? 1 : 0;
        g->weights_copied = true;
        if (!g->wbuf[0]) {
            // say what was asked for and what the device allows — the RAM
            // warning two lines below this in a load gives numbers, and a
            // bare "allocation failed" gives a scheduler nothing to reason
            // with (16 GB-Mac field report, 2026-08-05)
            char why[192];
            uint64_t ws = dev.recommendedMaxWorkingSetSize;
            // Name BOTH limits: a message quoting only the working set reads
            // as self-contradictory whenever maxBufferLength is what refused
            // the allocation, and that is the common case (it is 0.75x the
            // working set). A field report spent its entire run trying to
            // reconcile "10.3 requested, 12.7 limit, failed".
            uint64_t mb = (uint64_t)dev.maxBufferLength;
            snprintf(why, sizeof why,
                     "weight buffer allocation: %.1f GB requested; device max "
                     "single buffer %.1f GB, working set %.1f GB%s",
                     wlen / 1e9, mb / 1e9, ws / 1e9,
                     (uint64_t)wlen > mb ? " — exceeds the per-buffer limit"
                     : ((uint64_t)wlen > ws ? " — exceeds the working set" : ""));
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
    // K layers on the device. Full offload (K == n_layer) returns logits and
    // never reaches the CPU loop; a partial split leaves the boundary
    // activation in the host x buffer and the CPU finishes from layer K.
    // Setting this wrong in either direction is silent: too high and the
    // dispatcher asks for logits the device never computed, too low and every
    // layer is re-run on the CPU, discarding the GPU's work.
    m->gpu_layers = gpu_K;
    fprintf(stderr, "gpu: Metal backend on %s%s\n", dev.name.UTF8String,
            g->weights_copied ? " (weights copied)" : " (zero-copy weights)");
    if (gpu_K < m->n_layer)
        fprintf(stderr, "gpu: partial offload — %d of %d layers on the GPU "
                "(%.1f GB wrapped of %.1f GB; working set %.1f GB). The "
                "remaining %d run on the CPU.\n",
                gpu_K, m->n_layer, wlen / 1e9, m->gf.map_size / 1e9,
                ws_limit / 1e9, m->n_layer - gpu_K);
    return true;

unsupported:
    fprintf(stderr, "gpu: model uses a quant type without a Metal kernel — using CPU\n");
    return false;
}

// ---------------------------------------------------------------- encoding

// ------------------------------------------------------- dispatch census
// "Where does the batch dimension collapse?" is not answerable from
// throughput: a kernel that takes the batch in grid.y and a kernel encoded n
// times cost the same to WRITE and look identical from outside. This counts
// dispatches by kind, so a prefill census at two batch sizes shows directly
// which kinds scale with n (collapsed) and which do not (batched).
//
// Off unless RUNNER_METAL_STATS is set... except that the counters themselves
// are unconditional. They are non-atomic increments of a static struct on the
// encoding thread, which is the same thread for a whole command buffer, and
// the alternative — a branch per dispatch — costs more than the increment.
static struct {
    unsigned long mm, mv, mvf, rmsnorm, qknorm, headnorm, rope, store,
                  attn, attn_chunk, elem, moe, ple;
    // KV bytes an attention dispatch will read, accumulated for DECODE only
    // (n == 1), split by whether the layer slides. Decode-only because at
    // n > 1 each column attends over its own growing range and a single
    // `pos` does not describe the read; counting prefill here would produce
    // an underestimate that looks authoritative.
    unsigned long long kv_global, kv_swa;
    unsigned long      kv_layers_global, kv_layers_swa;
} g_disp;

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
    g_disp.rmsnorm++;
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
        !((w->type == T_Q4_K || w->type == T_Q6_K || w->type == T_Q2_K ||
           w->type == T_Q3_K || w->type == T_IQ4_XS) && n_in % 256 != 0)) {
        [e setComputePipelineState:g->p_mm[w->type]];
        mm_args ma = { n_in, n_out, n_col,
                       metal_bind_weights(g, e,
                           (uint64_t)((uint8_t *)w->data - (uint8_t *)m->gf.map),
                           w->nbytes),
                       bias != nil, x_stride, y_stride };
        [e setBuffer:x offset:x_off atIndex:1];
        [e setBuffer:y offset:y_off atIndex:2];
        [e setBytes:&ma length:sizeof(ma) atIndex:3];
        [e setBuffer:bias ? bias : g->dummy offset:0 atIndex:4];
        // MM_TILE_M/MM_TILE_N mirror MM_TM/MM_TN in kernels.metal -- this file
        // cannot see that #define (the shader is compiled from an embedded
        // source string at runtime, not by clang alongside this file), so the
        // two are kept in step by hand, the same way mv_args/mm_args already
        // are (see the struct comment above).
        enum { MM_TILE_M = 64, MM_TILE_N = 32 };
        g_disp.mm++;
        [e dispatchThreadgroups:MTLSizeMake((n_out + MM_TILE_M - 1) / MM_TILE_M,
                                            (n_col + MM_TILE_N - 1) / MM_TILE_N, 1)
          threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        return;
    }
    int col_tile = metal_col_tile();
    if (col_tile > n_col) col_tile = n_col;
    mv_args a = { n_in, n_out, 0,
                  bias != nil, n_col, x_stride, y_stride, col_tile };
    // Fast decode matvec: single column only. At n_col > 1 either the tiled
    // GEMM above already took the work, or this type has no mm kernel and the
    // identity matvec is what the batch falls back to — in both cases the
    // fast kernel would be answering a question nobody asked, and it would put
    // a non-identity result on the PREFILL path, where the tolerance gate
    // (decode-only by construction) has never looked.
    id<MTLComputePipelineState> mv = g->p_mv[w->type];
    if (n_col == 1 && metal_mv_fast_on() && g->p_mvf[w->type]) {
        mv = g->p_mvf[w->type];
        g_mv_dispatches++;
        g_disp.mvf++;
    } else {
        g_disp.mv++;
    }
    [e setComputePipelineState:mv];
    a.w_off = metal_bind_weights(g, e,
                  (uint64_t)((uint8_t *)w->data - (uint8_t *)m->gf.map),
                  w->nbytes);
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

// n_col columns in one dispatch (grid.y), stride elements apart. Each
// (head, column) pair is an independent reduction, so this is bit-identical to
// n_col separate enc_qknorm calls -- see the kernel comment. It exists because
// the per-token encoding of this op was 50n of the 240n per-token dispatches
// measured in docs/metal-dispatch-census-2026-08-13.md.
static void enc_qknorm_n(gpu_t *g, id<MTLComputeCommandEncoder> e, model_t *m,
                         id<MTLBuffer> v, NSUInteger v_off, id<MTLBuffer> w,
                         int n_heads, int hd, int n_col, int stride) {
    float eps = m->rms_eps;
    [e setComputePipelineState:g->p_qknorm];
    [e setBuffer:v offset:v_off atIndex:0];
    [e setBuffer:w offset:0 atIndex:1];
    [e setBytes:&hd length:4 atIndex:2];
    [e setBytes:&eps length:4 atIndex:3];
    [e setBytes:&stride length:4 atIndex:4];
    g_disp.qknorm++;
    [e dispatchThreadgroups:MTLSizeMake(n_heads, n_col, 1)
      threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
}


// Batched twin of enc_qknorm_n, same bit-identity argument. src and dst share
// one stride: the K and V staging buffers are both strided by kv_dim.
static void enc_headnorm_n(gpu_t *g, id<MTLComputeCommandEncoder> e, model_t *m,
                           id<MTLBuffer> src, NSUInteger src_off,
                           id<MTLBuffer> dst, NSUInteger dst_off,
                           id<MTLBuffer> w, int n_heads, int hd,
                           int n_col, int stride) {
    float eps = m->rms_eps;
    int has_weight = w != nil;
    [e setComputePipelineState:g->p_headnorm];
    [e setBuffer:src offset:src_off atIndex:0];
    [e setBuffer:dst offset:dst_off atIndex:1];
    [e setBuffer:w ? w : g->dummy offset:0 atIndex:2];
    [e setBytes:&hd length:4 atIndex:3];
    [e setBytes:&eps length:4 atIndex:4];
    [e setBytes:&has_weight length:4 atIndex:5];
    [e setBytes:&stride length:4 atIndex:6];
    g_disp.headnorm++;
    [e dispatchThreadgroups:MTLSizeMake(n_heads, n_col, 1)
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
    g_disp.rope++;
    [e dispatchThreads:MTLSizeMake(a.half_dim, n_heads, n_col)
      threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
}


// n_col columns in one dispatch (grid.y), a_stride/b_stride elements apart.
// Bit-identical to n_col separate calls -- every element was already
// independent, so only the encoding changes. See the kernel-family comment in
// kernels.metal for why this exists.
static void enc_elem_n(gpu_t *g, id<MTLComputeCommandEncoder> e,
                       id<MTLComputePipelineState> p,
                       id<MTLBuffer> a, NSUInteger a_off,
                       id<MTLBuffer> b, NSUInteger b_off, int n,
                       int n_col, int a_stride, int b_stride) {
    [e setComputePipelineState:p];
    [e setBuffer:a offset:a_off atIndex:0];
    [e setBuffer:b offset:b_off atIndex:1];
    [e setBytes:&n length:4 atIndex:2];
    [e setBytes:&a_stride length:4 atIndex:3];
    [e setBytes:&b_stride length:4 atIndex:4];
    g_disp.elem++;
    [e dispatchThreads:MTLSizeMake(n, n_col, 1)
      threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

// Single column. Still used by the MoE FFN, which stays per-token because
// routing picks different experts per token.
static void enc_elem(gpu_t *g, id<MTLComputeCommandEncoder> e,
                     id<MTLComputePipelineState> p,
                     id<MTLBuffer> a, NSUInteger a_off,
                     id<MTLBuffer> b, NSUInteger b_off, int n) {
    enc_elem_n(g, e, p, a, a_off, b, b_off, n, 1, 0, 0);
}

static void enc_scale_n(gpu_t *g, id<MTLComputeCommandEncoder> e,
                        id<MTLBuffer> x, NSUInteger x_off,
                        int n, float scale, int n_col, int stride) {
    [e setComputePipelineState:g->p_scale];
    [e setBuffer:x offset:x_off atIndex:0];
    [e setBytes:&scale length:4 atIndex:1];
    [e setBytes:&n length:4 atIndex:2];
    [e setBytes:&stride length:4 atIndex:3];
    g_disp.elem++;
    [e dispatchThreads:MTLSizeMake(n, n_col, 1)
      threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

static void enc_scale(gpu_t *g, id<MTLComputeCommandEncoder> e,
                      id<MTLBuffer> x, NSUInteger x_off,
                      int n, float scale) {
    enc_scale_n(g, e, x, x_off, n, scale, 1, 0);
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
    g_disp.moe++;
    [e dispatchThreads:MTLSizeMake(1, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
}

static void enc_moe_mv(gpu_t *g, id<MTLComputeCommandEncoder> e, model_t *m,
                       gguf_tensor *base, uint64_t estride,
                       id<MTLBuffer> x, NSUInteger x_off,
                       id<MTLBuffer> y, NSUInteger y_off,
                       int n_in, int n_out, int nslots, int xs, int ys,
                       id<MTLBuffer> bias, int bias_stride) {
    [e setComputePipelineState:g->p_moe_mv[base->type]];
    moe_args a = { n_in, n_out,
                   metal_bind_weights(g, e,
                       (uint64_t)((uint8_t *)base->data - (uint8_t *)m->gf.map),
                       base->nbytes),
                   estride, xs, ys, bias != nil, bias_stride };
    [e setBuffer:x offset:x_off atIndex:1];
    [e setBuffer:y offset:y_off atIndex:2];
    [e setBytes:&a length:sizeof(a) atIndex:3];
    [e setBuffer:g->moe_sel offset:0 atIndex:4];
    [e setBuffer:bias ? bias : g->dummy offset:0 atIndex:5];
    g_disp.moe++;
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
    g_disp.moe++;
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
    g_disp.moe++;
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

// Partial split hand-off: the CPU loop resumes at layer gpu_layers and reads
// the boundary hidden state from the host x buffer. On Metal this is a copy
// within one address space -- g->x is shared storage, so it IS host memory --
// and it is small (n * n_embd floats; 11 KB at decode). The KV needs nothing:
// m->kcache/m->vcache already point at g->kc/g->vc contents, so the layers
// the device just wrote are already visible to the CPU.
static void metal_handoff_boundary(model_t *m, int n) {
    gpu_t *g = (gpu_t *)m->gpu;
    if (!g || !m->x) return;
    memcpy(m->x, g->x.contents, sizeof(float) * (size_t)n * m->n_embd);
}

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
    bool partial = m->gpu_layers > 0 && m->gpu_layers < m->n_layer;
    float *lg = gpu_forward_native_batch(m, tokens, n, pos);
    if (!lg) return false;
    if (partial) {
        metal_handoff_boundary(m, n);
        if (logits) *logits = NULL;
        return true;
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
    // ple is [token][layer][P], so this layer's slice starts at l*P and the
    // per-column stride is the whole layer row.
    enc_elem_n(g, e, g->p_gelu, g->ple_tmp, 0,
               g->ple, foff((size_t)l * P), P, n, P, m->n_layer * P);
    enc_mv_n(g, e, m, ly->ple_proj, g->ple_tmp, 0, g->xb, 0,
             P, n_embd, nil, n, P, xdim);
    enc_rmsnorm_n(g, e, g->xb, 0, g->xb, 0, g->ppn[l],
                  n_embd, m->rms_eps, n, xdim, xdim);
    enc_elem_n(g, e, g->p_add, g->x, 0, g->xb, 0, n_embd, n, n_embd, xdim);
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
        model_embd_transform(m, xp);
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
    // RUNNER_METAL_TIMING=1: split a forward into the CPU time spent encoding
    // dispatches and the GPU time actually executing them. Decode on Metal is
    // slower than CPU on an M1 and the two candidate explanations -- encode
    // overhead at ~420 dispatches per token, versus the kernels themselves --
    // are indistinguishable from throughput alone.
    static int timing = -1;
    if (timing < 0) {
        const char *t = getenv("RUNNER_METAL_TIMING");
        timing = t && *t && strcmp(t, "0") ? 1 : 0;
    }
    double t_enc0 = timing ? CFAbsoluteTimeGetCurrent() : 0;
    id<MTLCommandBuffer> cb = [g->queue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];

    // Partial split: only the leading gpu_layers run here.
    int n_gpu_layers = m->gpu_layers > 0 && m->gpu_layers < m->n_layer
                     ? m->gpu_layers : m->n_layer;
    for (int l = 0; l < n_gpu_layers; l++) {
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
        // the output gate projects the same normed input as Q; xb is
        // overwritten by the wo matvec below, so project it here
        if (m->attn_out_gate && ly->wq_gate)
            enc_mv_n(g, e, m, ly->wq_gate, g->xb, 0, g->agate, 0,
                     n_embd, q_dim_l, nil, n, xdim, q_dim);
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

        // Batched in grid.y rather than encoded once per token. Bit-identical:
        // each (head, column) pair was always an independent reduction, so
        // only the encoding changes. This was 65n of the 240n per-token
        // dispatches the census measured; the elementwise ops are the rest.
        if (g->qn[l])
            enc_qknorm_n(g, e, m, g->q, 0, g->qn[l], m->n_head, hd, n, q_dim);
        if (owns_kv) {
            if (m->v_rmsnorm)
                enc_headnorm_n(g, e, m, ly->wv ? g->vt : g->kt, 0,
                               g->vt, 0, nil, n_kv, hd, n, kv_dim);
            if (g->kn[l])
                enc_qknorm_n(g, e, m, g->kt, 0, g->kn[l], n_kv, hd, n, kv_dim);
        }
        {
            size_t row_b = model_kv_row_bytes(m, l);
            int q8 = m->kv_q8;
            int kv_units = q8 ? kv_dim_l / 32 : kv_dim_l;

            // rope/store/attention each take the batch in one dispatch: the
            // kernels derive their column's position from pos + col, so every
            // token still rotates at, writes to, and attends over exactly the
            // range a per-token submit gave it.
            if (model_layer_ropes(m, l)) {
                enc_rope_n(g, e, m, g->q,  0, m->n_head, pos, l, n, q_dim);
            } else if (m->attn_temp_scale != 0.0f) {
                // NoPE layer: no rotation, and THIS is where the llama-4
                // attention temperature applies -- on the layers that skipped
                // rope, not on every layer (matching the CPU path and
                // llama.cpp). The factor is per POSITION, so unlike everything
                // else here it cannot be one batched dispatch; it is one scale
                // per column, the same shape the CUDA backend uses.
                for (int b = 0; b < n; b++) {
                    float ts = model_attn_temp(m, pos + b);
                    if (ts != 1.0f)
                        enc_scale(g, e, g->q, foff((size_t)b * q_dim),
                                  q_dim_l, ts);
                }
            }
            if (owns_kv) {
                if (model_layer_ropes(m, l))
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
                g_disp.store++;
                [e dispatchThreads:MTLSizeMake(kv_units, n, 1)
                  threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            }

            // Chunked path: decode only (n == 1), long enough range to be
            // worth splitting, and a head_dim the partial buffer was sized
            // for. Everything else keeps the single-pass kernel, including
            // all of prefill -- which already has grid.y parallelism from the
            // batch and is the validated fast path.
            int a_t0 = (window > 0 && pos - window + 1 > 0) ? pos - window + 1 : 0;
            int a_span = pos - a_t0 + 1;
            if (n == 1) {
                // K and V, both read across the span this layer attends over.
                unsigned long long b = 2ull * (unsigned long long)a_span
                                     * (unsigned long long)model_kv_row_bytes(m, l);
                if (window > 0) { g_disp.kv_swa += b; g_disp.kv_layers_swa++; }
                else            { g_disp.kv_global += b; g_disp.kv_layers_global++; }
            }
            int a_ov = metal_attn_chunk_override();
            int a_chunk, a_nch;
            if (a_ov == 0) { a_chunk = 0; a_nch = 0; }        // path disabled
            else if (a_ov > 0) {
                a_chunk = a_ov;
                a_nch = (a_span + a_chunk - 1) / a_chunk;
            } else {
                int want = METAL_ATTN_TARGET_GROUPS / (m->n_head > 0 ? m->n_head : 1);
                if (want < 1) want = 1;
                if (want > METAL_ATTN_MAX_CHUNKS) want = METAL_ATTN_MAX_CHUNKS;
                a_chunk = (a_span + want - 1) / want;
                // floor, not veto: a target implying tiny chunks means "use the
                // smallest chunk worth having", and short context then drops to
                // the single-pass kernel through the n_chunks < 2 test below
                if (a_chunk < METAL_ATTN_MIN_CHUNK) a_chunk = METAL_ATTN_MIN_CHUNK;
                a_nch = (a_span + a_chunk - 1) / a_chunk;
            }
            if (n == 1 && a_nch >= 2 && a_nch <= METAL_ATTN_MAX_CHUNKS &&
                hd == model_head_dim(m, 0)) {
                attn_chunk_args ca = { hd, m->n_head, n_kv, m->n_ctx, pos,
                                       (uint64_t)model_kv_byte_off(m, l),
                                       model_attn_scale(m, l), q8, window,
                                       a_chunk, a_nch, q_dim,
                                       m->n_head * m->n_ctx };
                bool coopc = metal_attn_coop_on() && g->p_attn_chunk_coop;
                if (coopc) g_coop_dispatches++;
                [e setComputePipelineState:coopc ? g->p_attn_chunk_coop
                                                 : g->p_attn_chunk];
                [e setBuffer:g->q       offset:0 atIndex:0];
                [e setBuffer:g->kc      offset:0 atIndex:1];
                [e setBuffer:g->vc      offset:0 atIndex:2];
                [e setBuffer:g->att     offset:0 atIndex:3];
                [e setBuffer:g->att_acc offset:0 atIndex:4];
                [e setBuffer:g->att_ms  offset:0 atIndex:5];
                [e setBytes:&ca length:sizeof(ca) atIndex:6];
                g_disp.attn_chunk++;
                [e dispatchThreadgroups:MTLSizeMake(m->n_head, a_nch, 1)
                  threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                attn_comb_args cb = { hd, m->n_head, a_nch,
                                      g->sinks[l] != nil, xdim };
                [e setComputePipelineState:g->p_attn_comb];
                [e setBuffer:g->att_acc offset:0 atIndex:0];
                [e setBuffer:g->att_ms  offset:0 atIndex:1];
                [e setBuffer:g->xb2     offset:0 atIndex:2];
                [e setBytes:&cb length:sizeof(cb) atIndex:3];
                [e setBuffer:g->sinks[l] ? g->sinks[l] : g->dummy offset:0 atIndex:4];
                g_disp.attn_chunk++;
                [e dispatchThreadgroups:MTLSizeMake(m->n_head, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                goto attn_done;
            }

            attn_args aa = { hd, m->n_head, n_kv, m->n_ctx, pos,
                             (uint64_t)model_kv_byte_off(m, l),
                             model_attn_scale(m, l), q8, window,
                             g->sinks[l] != nil,
                             q_dim, m->n_head * m->n_ctx, xdim };
            // Cooperative twin at decode only: the score loop is what
            // scatters, and at n > 1 each column has its own KV range, which
            // the coop form's simdgroup-per-row split does not describe.
            bool coop = (n == 1) && metal_attn_coop_on() && g->p_attn_coop;
            if (coop) g_coop_dispatches++;
            [e setComputePipelineState:coop ? g->p_attn_coop : g->p_attn];
            [e setBuffer:g->q   offset:0 atIndex:0];
            [e setBuffer:g->kc  offset:0 atIndex:1];
            [e setBuffer:g->vc  offset:0 atIndex:2];
            [e setBuffer:g->att offset:0 atIndex:3];
            [e setBuffer:g->xb2 offset:0 atIndex:4];
            [e setBytes:&aa length:sizeof(aa) atIndex:5];
            [e setBuffer:g->sinks[l] ? g->sinks[l] : g->dummy offset:0 atIndex:6];
            // Widest power-of-two threadgroup this pipeline allows, capped
            // at the red[] scratch in the kernel. The reduction halves tpg
            // each step, so a non-power-of-two would drop lanes silently.
            NSUInteger amax = (coop ? g->p_attn_coop : g->p_attn)
                                  .maxTotalThreadsPerThreadgroup;
            if (amax > 256) amax = 256;   // measured optimum; red[] is sized to match
            NSUInteger atpg = 1;
            while (atpg * 2 <= amax) atpg *= 2;
            if (timing) {
                static bool told;
                if (!told) {
                    told = true;
                    fprintf(stderr, "metal-timing attn threadgroups=%d x %llu "
                            "threads (pipeline max %llu)\n", m->n_head,
                            (unsigned long long)atpg,
                            (unsigned long long)g->p_attn.maxTotalThreadsPerThreadgroup);
                }
            }
            g_disp.attn++;
            [e dispatchThreadgroups:MTLSizeMake(m->n_head, n, 1)
              threadsPerThreadgroup:MTLSizeMake(atpg, 1, 1)];
            attn_done: ;
        }

        if (m->attn_out_gate && ly->wq_gate)
            enc_elem_n(g, e, g->p_sigmul, g->xb2, 0, g->agate, 0,
                       q_dim_l, n, xdim, q_dim);
        enc_mv_n(g, e, m, ly->wo, g->xb2, 0, g->xb, 0,
                 q_dim_l, n_embd, g->bo[l], n, xdim, xdim);
        if (g->pan[l])
            enc_rmsnorm_n(g, e, g->xb, 0, g->xb, 0, g->pan[l],
                          n_embd, m->post_norm_eps, n, xdim, xdim);
        if (m->resid_scale != 1.0f)  // granite muP branch scale
            enc_scale_n(g, e, g->xb, 0, n_embd, m->resid_scale, n, xdim);
        enc_elem_n(g, e, g->p_add, g->x, 0, g->xb, 0, n_embd, n, n_embd, xdim);

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
                          n_embd, m->post_norm_eps, n, xdim, xdim);
        if (m->resid_scale != 1.0f)
            enc_scale_n(g, e, g->xb, 0, n_embd, m->resid_scale, n, xdim);
        enc_elem_n(g, e, g->p_add, g->x, 0, g->xb, 0, n_embd, n, n_embd, xdim);
        // Order matters: the E-series branch reads the post-FFN residual of
        // EVERY token, so it has to run before any token's output scale.
        if (ly->ple_gate) enc_ple(g, e, m, ly, l, n, xdim);
        if (ly->out_scale != 1.0f && ly->out_scale != 0.0f)
            enc_scale_n(g, e, g->x, 0, n_embd, ly->out_scale, n, n_embd);
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
        if (l == m->n_layer - 1 && n_gpu_layers == m->n_layer) {
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
    double t_enc1 = timing ? CFAbsoluteTimeGetCurrent() : 0;
    [cb commit];
    [cb waitUntilCompleted];
    if (timing) {
        double now = CFAbsoluteTimeGetCurrent();
        double gpu = [cb GPUEndTime] - [cb GPUStartTime];
        fprintf(stderr, "metal-timing n=%d encode=%.2fms submit+wait=%.2fms "
                "gpu=%.2fms total=%.2fms\n", n,
                (t_enc1 - t_enc0) * 1e3, (now - t_enc1) * 1e3, gpu * 1e3,
                (now - t_enc0) * 1e3);
    }
    if (metal_command_failed(cb)) {
        fprintf(stderr, "gpu: command buffer failed — falling back to CPU\n");
        return NULL;
    }
    if (metal_env_on("RUNNER_METAL_STATS")) {
        unsigned long tot = g_disp.mm + g_disp.mv + g_disp.mvf +
                            g_disp.rmsnorm + g_disp.qknorm + g_disp.headnorm +
                            g_disp.rope + g_disp.store + g_disp.attn +
                            g_disp.attn_chunk + g_disp.elem + g_disp.moe;
        fprintf(stderr, "metal: native prompt batch n=%d command_buffers=1\n", n);
        // Per-kind census for this forward, so "which kinds scale with n"
        // can be read straight off two runs at different batch sizes rather
        // than inferred from throughput. Counters are cumulative over the
        // process; take differences across forwards.
        if (g_disp.kv_global || g_disp.kv_swa)
            fprintf(stderr, "metal-kv decode-cumulative: global %llu B over %lu "
                    "layer-dispatches | sliding %llu B over %lu | sliding share "
                    "%.1f%%\n",
                    g_disp.kv_global, g_disp.kv_layers_global,
                    g_disp.kv_swa, g_disp.kv_layers_swa,
                    100.0 * (double)g_disp.kv_swa
                        / (double)(g_disp.kv_swa + g_disp.kv_global));
        fprintf(stderr, "metal-census n=%d total=%lu | mm=%lu mv=%lu mvf=%lu "
                "rmsnorm=%lu qknorm=%lu headnorm=%lu rope=%lu store=%lu "
                "attn=%lu attn_chunk=%lu(coop %lu) elem=%lu moe=%lu\n",
                n, tot, g_disp.mm, g_disp.mv, g_disp.mvf, g_disp.rmsnorm,
                g_disp.qknorm, g_disp.headnorm, g_disp.rope, g_disp.store,
                g_disp.attn, g_disp.attn_chunk, g_coop_dispatches,
                g_disp.elem, g_disp.moe);
    }
    return (float *)g->logits.contents + (size_t)(n - 1) * m->n_vocab;
}
