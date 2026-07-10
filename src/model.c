// Llama-family transformer: weight wiring + batched forward pass.
#include "runner.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---------------------------------------------------------------- helpers

static float *tensor_to_f32(gguf_tensor *t) {
    if (!t) return NULL;
    int64_t n = (int64_t)(t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3]);
    float *out = malloc(sizeof(float) * n);
    if (t->type == T_F32) {
        memcpy(out, t->data, sizeof(float) * n);
    } else {
        size_t rs = ggml_row_size(t->type, t->ne[0]);
        int64_t rows = n / (int64_t)t->ne[0];
        for (int64_t r = 0; r < rows; r++)
            dequant_row(t->type, (uint8_t *)t->data + r * rs, out + r * (int64_t)t->ne[0], (int)t->ne[0]);
    }
    return out;
}

static gguf_tensor *need_tensor(gguf_file *g, const char *fmt, int i, bool *ok) {
    char name[128];
    snprintf(name, sizeof(name), fmt, i);
    gguf_tensor *t = gguf_find_tensor(g, name);
    if (!t) { fprintf(stderr, "error: missing tensor %s\n", name); *ok = false; return NULL; }
    if (!ggml_type_supported(t->type)) {
        fprintf(stderr, "error: tensor %s has unsupported type %d (%s)\n",
                name, t->type, ggml_type_name(t->type));
        *ok = false;
        return NULL;
    }
    return t;
}

static gguf_tensor *opt_tensor(gguf_file *g, const char *fmt, int i) {
    char name[128];
    snprintf(name, sizeof(name), fmt, i);
    return gguf_find_tensor(g, name);
}

// ---------------------------------------------------------------- rope setup

// YaRN correction dimension (llama.cpp rope_yarn_corr_dim)
static float yarn_corr_dim(int n_dims, int n_ctx_orig, float n_rot, float base) {
    return n_dims * logf(n_ctx_orig / (n_rot * 2 * (float)M_PI)) / (2 * logf(base));
}

static void rope_setup(model_t *m, gguf_file *g, const char *arch,
                       float base_ovr, float scale_ovr) {
    char key[128];
    #define RK(fmt) (snprintf(key, sizeof(key), "%s." fmt, arch), key)
    if (base_ovr > 0) m->rope_base = base_ovr;
    m->rope_mscale = 1.0f;

    int half = m->rope_dim / 2;
    m->rope_inv_freq = malloc(sizeof(float) * half);
    for (int j = 0; j < half; j++)
        m->rope_inv_freq[j] = powf(m->rope_base, -2.0f * j / m->rope_dim);

    // per-dimension frequency factors (llama 3.x long-context models)
    gguf_tensor *ff = gguf_find_tensor(g, "rope_freqs.weight");
    if (ff && ff->type == T_F32 && (int)ff->ne[0] >= half) {
        const float *f = ff->data;
        for (int j = 0; j < half; j++) m->rope_inv_freq[j] /= f[j];
        fprintf(stderr, "rope: using model frequency factors (rope_freqs.weight)\n");
    }

    const char *sct = gguf_get_str(g, RK("rope.scaling.type"), "");
    float factor  = gguf_get_f32(g, RK("rope.scaling.factor"), 0.0f);
    int   orig    = (int)gguf_get_u32(g, RK("rope.scaling.original_context_length"),
                                      m->n_ctx_train);
    if (orig <= 0) orig = m->n_ctx_train;

    enum { RS_NONE, RS_LINEAR, RS_YARN } mode = RS_NONE;
    if (scale_ovr > 0) {
        mode = RS_LINEAR; factor = scale_ovr;
        fprintf(stderr, "rope: forced linear scaling x%.2f\n", factor);
    } else if (strcmp(sct, "linear") == 0 && factor > 1.0f) {
        mode = RS_LINEAR;
        fprintf(stderr, "rope: linear scaling x%.2f (model metadata)\n", factor);
    } else if (strcmp(sct, "yarn") == 0 && factor > 1.0f) {
        mode = RS_YARN;
        fprintf(stderr, "rope: YaRN scaling x%.2f (model metadata)\n", factor);
    } else if (m->n_ctx > m->n_ctx_train) {
        // automatic context extension for models trained on short contexts
        mode = RS_YARN;
        factor = (float)m->n_ctx / m->n_ctx_train;
        orig = m->n_ctx_train;
        fprintf(stderr,
                "rope: requested ctx %d > training ctx %d — applying YaRN x%.2f\n",
                m->n_ctx, m->n_ctx_train, factor);
    }

    if (mode == RS_LINEAR) {
        for (int j = 0; j < half; j++) m->rope_inv_freq[j] /= factor;
    } else if (mode == RS_YARN) {
        // NTK-by-parts: interpolate long wavelengths, keep short ones intact
        float lo = floorf(yarn_corr_dim(m->rope_dim, orig, 32.0f, m->rope_base));
        float hi = ceilf(yarn_corr_dim(m->rope_dim, orig, 1.0f, m->rope_base));
        if (lo < 0) lo = 0;
        if (hi > m->rope_dim - 1) hi = m->rope_dim - 1;
        for (int j = 0; j < half; j++) {
            float y = (j - lo) / (hi - lo > 0.001f ? hi - lo : 0.001f);
            float keep = 1.0f - fminf(1.0f, fmaxf(0.0f, y)); // 1 = extrapolate (keep)
            float s = keep + (1.0f - keep) / factor;
            m->rope_inv_freq[j] *= s;
        }
        m->rope_mscale = 1.0f + 0.1f * logf(factor);
    }
    #undef RK
}

// ---------------------------------------------------------------- load

bool model_load(model_t *m, const char *path, const model_params *p) {
    memset(m, 0, sizeof(*m));
    if (!gguf_open(&m->gf, path)) return false;
    gguf_file *g = &m->gf;

    const char *arch = gguf_get_str(g, "general.architecture", "?");
    snprintf(m->arch, sizeof(m->arch), "%s", arch);
    if (strcmp(arch, "llama") != 0 && strcmp(arch, "qwen2") != 0 &&
        strcmp(arch, "mistral") != 0 && strcmp(arch, "smollm") != 0 &&
        strcmp(arch, "stablelm") != 0) {
        fprintf(stderr, "warning: architecture '%s' untested, attempting llama-style load\n", arch);
    }
    char key[128];
    #define AK(fmt) (snprintf(key, sizeof(key), "%s." fmt, arch), key)

    m->n_layer     = (int)gguf_get_u32(g, AK("block_count"), 0);
    m->n_embd      = (int)gguf_get_u32(g, AK("embedding_length"), 0);
    m->n_head      = (int)gguf_get_u32(g, AK("attention.head_count"), 0);
    m->n_head_kv   = (int)gguf_get_u32(g, AK("attention.head_count_kv"), m->n_head);
    m->n_ff        = (int)gguf_get_u32(g, AK("feed_forward_length"), 0);
    m->n_ctx_train = (int)gguf_get_u32(g, AK("context_length"), 2048);
    m->head_dim    = (int)gguf_get_u32(g, AK("attention.key_length"),
                                       m->n_head ? m->n_embd / m->n_head : 0);
    m->rope_dim    = (int)gguf_get_u32(g, AK("rope.dimension_count"), m->head_dim);
    m->rms_eps     = gguf_get_f32(g, AK("attention.layer_norm_rms_epsilon"), 1e-5f);
    m->rope_base   = gguf_get_f32(g, AK("rope.freq_base"), 10000.0f);
    // llama-arch GGUFs have Q/K permuted at conversion for adjacent-pair rope;
    // qwen2 (and other HF-layout archs) need NeoX-style half-split rotation
    m->rope_neox   = strcmp(arch, "llama") != 0 && strcmp(arch, "mistral") != 0;
    if (gguf_get_u32(g, AK("expert_count"), 0) > 0) {
        fprintf(stderr, "error: MoE models are not supported\n");
        return false;
    }
    if (gguf_get(g, AK("ssm.conv_kernel"))) {
        fprintf(stderr, "error: '%s' is a hybrid SSM/attention architecture — "
                "only pure-transformer llama-family models are supported\n", arch);
        return false;
    }
    if (m->n_layer <= 0 || m->n_embd <= 0 || m->n_head <= 0 || m->n_ff <= 0) {
        fprintf(stderr, "error: missing model hyperparameters for arch '%s'\n", arch);
        return false;
    }

    bool ok = true;
    m->tok_embd = need_tensor(g, "token_embd.weight", 0, &ok);
    if (!ok) return false;
    m->n_vocab = (int)m->tok_embd->ne[1];

    gguf_tensor *out_norm = need_tensor(g, "output_norm.weight", 0, &ok);
    if (!ok) return false;
    m->out_norm_w = tensor_to_f32(out_norm);

    m->output = gguf_find_tensor(g, "output.weight");
    if (!m->output) m->output = m->tok_embd; // tied embeddings
    if (!ggml_type_supported(m->output->type)) {
        fprintf(stderr, "error: output tensor type %s unsupported\n", ggml_type_name(m->output->type));
        return false;
    }

    m->layers = calloc(m->n_layer, sizeof(layer_t));
    for (int i = 0; i < m->n_layer; i++) {
        layer_t *l = &m->layers[i];
        gguf_tensor *an = need_tensor(g, "blk.%d.attn_norm.weight", i, &ok);
        gguf_tensor *fn = need_tensor(g, "blk.%d.ffn_norm.weight", i, &ok);
        l->wq     = need_tensor(g, "blk.%d.attn_q.weight", i, &ok);
        l->wk     = need_tensor(g, "blk.%d.attn_k.weight", i, &ok);
        l->wv     = need_tensor(g, "blk.%d.attn_v.weight", i, &ok);
        l->wo     = need_tensor(g, "blk.%d.attn_output.weight", i, &ok);
        l->w_gate = need_tensor(g, "blk.%d.ffn_gate.weight", i, &ok);
        l->w_up   = need_tensor(g, "blk.%d.ffn_up.weight", i, &ok);
        l->w_down = need_tensor(g, "blk.%d.ffn_down.weight", i, &ok);
        if (!ok) return false;
        l->attn_norm_w = tensor_to_f32(an);
        l->ffn_norm_w  = tensor_to_f32(fn);
        l->bq = tensor_to_f32(opt_tensor(g, "blk.%d.attn_q.bias", i));
        l->bk = tensor_to_f32(opt_tensor(g, "blk.%d.attn_k.bias", i));
        l->bv = tensor_to_f32(opt_tensor(g, "blk.%d.attn_v.bias", i));
        l->bo = tensor_to_f32(opt_tensor(g, "blk.%d.attn_output.bias", i));
    }

    // runtime buffers
    int n_ctx = p->n_ctx;
    if (n_ctx <= 0) n_ctx = m->n_ctx_train < 4096 ? m->n_ctx_train : 4096;
    m->n_ctx = n_ctx;
    m->n_batch = p->n_batch > 0 ? p->n_batch : 64;
    if (m->n_batch > n_ctx) m->n_batch = n_ctx;
    int q_dim  = m->n_head * m->head_dim;
    int kv_dim = m->n_head_kv * m->head_dim;
    int xdim   = q_dim > m->n_embd ? q_dim : m->n_embd;
    int B      = m->n_batch;

    size_t kv_bytes = (size_t)m->n_layer * n_ctx * kv_dim * sizeof(f16_t);
    m->kcache = calloc(1, kv_bytes);
    m->vcache = calloc(1, kv_bytes);
    m->x      = malloc(sizeof(float) * (size_t)B * m->n_embd);
    m->xb     = malloc(sizeof(float) * (size_t)B * xdim);
    m->xb2    = malloc(sizeof(float) * (size_t)B * xdim);
    m->q      = malloc(sizeof(float) * (size_t)B * q_dim);
    m->k_tmp  = malloc(sizeof(float) * (size_t)B * kv_dim);
    m->v_tmp  = malloc(sizeof(float) * (size_t)B * kv_dim);
    m->hb     = malloc(sizeof(float) * (size_t)B * m->n_ff);
    m->hb2    = malloc(sizeof(float) * (size_t)B * m->n_ff);
    m->att    = malloc(sizeof(float) * (size_t)m->n_head * n_ctx);
    m->logits = malloc(sizeof(float) * m->n_vocab);
    if (!m->kcache || !m->vcache || !m->hb2 || !m->att) {
        fprintf(stderr, "error: cannot allocate buffers (ctx %d needs %.1f MB KV cache)\n",
                n_ctx, 2.0 * kv_bytes / 1e6);
        return false;
    }

    m->tp = tpool_create(p->n_threads > 0 ? p->n_threads : 1);
    rope_setup(m, g, arch, p->rope_base, p->rope_scale);

    if (p->verbose) {
        fprintf(stderr, "%-24s %s\n", "architecture", m->arch);
        fprintf(stderr, "%-24s %d\n", "layers", m->n_layer);
        fprintf(stderr, "%-24s %d\n", "embedding dim", m->n_embd);
        fprintf(stderr, "%-24s %d (%d kv)\n", "heads", m->n_head, m->n_head_kv);
        fprintf(stderr, "%-24s %d\n", "head dim", m->head_dim);
        fprintf(stderr, "%-24s %d\n", "ffn dim", m->n_ff);
        fprintf(stderr, "%-24s %d\n", "vocab", m->n_vocab);
        fprintf(stderr, "%-24s %d (train %d)\n", "context", m->n_ctx, m->n_ctx_train);
        fprintf(stderr, "%-24s %.1f MB (fp16)\n", "kv cache", 2.0 * kv_bytes / 1e6);
        fprintf(stderr, "%-24s %d\n", "batch", m->n_batch);
        fprintf(stderr, "%-24s %s\n", "weight type", ggml_type_name(m->layers[0].wq->type));
        fprintf(stderr, "%-24s %.1f\n", "rope base", m->rope_base);
    }
    return true;
}

// ---------------------------------------------------------------- math ops

static void rmsnorm(float *o, const float *x, const float *w, int n, float eps) {
    float ss = 0;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float r = 1.0f / sqrtf(ss / n + eps);
    for (int i = 0; i < n; i++) o[i] = x[i] * r * w[i];
}

static void softmax(float *x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float s = 0;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); s += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= s;
}

typedef struct {
    const gguf_tensor *w;
    const float *x, *bias;
    float *y;
    int n_in, n_batch, x_stride, y_stride;
    size_t rsz;
} mv_job;

static void mv_rows(void *ctx, int i0, int i1) {
    mv_job *j = ctx;
    const uint8_t *base = j->w->data;
    int type = j->w->type, n_in = j->n_in;

    if (j->n_batch == 1) {
        for (int r = i0; r < i1; r++) {
            float v = vec_dot(type, base + (size_t)r * j->rsz, j->x, n_in);
            j->y[r] = j->bias ? v + j->bias[r] : v;
        }
        return;
    }
    // batched: dequantize each weight row once, reuse for every token
    float *buf = type == T_F32 ? NULL : malloc(sizeof(float) * n_in);
    for (int r = i0; r < i1; r++) {
        const float *wrow;
        if (type == T_F32) {
            wrow = (const float *)(base + (size_t)r * j->rsz);
        } else {
            dequant_row(type, base + (size_t)r * j->rsz, buf, n_in);
            wrow = buf;
        }
        float b0 = j->bias ? j->bias[r] : 0.0f;
        for (int b = 0; b < j->n_batch; b++) {
            const float *xb = j->x + (size_t)b * j->x_stride;
            float s = 0;
            for (int i = 0; i < n_in; i++) s += wrow[i] * xb[i];
            j->y[(size_t)b * j->y_stride + r] = s + b0;
        }
    }
    free(buf);
}

// Y[b] = W X[b] (+ bias) for b in [0, n_batch)
static void matvec_b(tpool *tp, float *y, int y_stride, const gguf_tensor *w,
                     const float *x, int x_stride, int n_in, int n_out,
                     const float *bias, int n_batch) {
    mv_job j = { w, x, bias, y, n_in, n_batch, x_stride, y_stride,
                 ggml_row_size(w->type, n_in) };
    tpool_run(tp, mv_rows, &j, n_out);
}

static void rope_apply(model_t *m, float *v, int n_heads, int pos) {
    int half = m->rope_dim / 2;
    float ms = m->rope_mscale;
    for (int j = 0; j < half; j++) {
        float a = pos * m->rope_inv_freq[j];
        float c = cosf(a) * ms, s = sinf(a) * ms;
        for (int h = 0; h < n_heads; h++) {
            float *p = v + h * m->head_dim;
            float *p0 = m->rope_neox ? p + j : p + 2 * j;
            float *p1 = m->rope_neox ? p + j + half : p0 + 1;
            float x0 = *p0, x1 = *p1;
            *p0 = x0 * c - x1 * s;
            *p1 = x0 * s + x1 * c;
        }
    }
}

typedef struct {
    model_t *m;
    const f16_t *kc, *vc;   // this layer's cache
    const float *q;         // this token's query [q_dim]
    float *out;             // attention output [q_dim]
    int pos;
} attn_job;

static void attn_heads(void *ctx, int h0, int h1) {
    attn_job *j = ctx;
    model_t *m = j->m;
    int hd = m->head_dim;
    int kv_dim = m->n_head_kv * hd;
    int kv_mul = m->n_head / m->n_head_kv;
    float scale = 1.0f / sqrtf((float)hd);

    for (int h = h0; h < h1; h++) {
        const float *qh = j->q + h * hd;
        float *att = m->att + (size_t)h * m->n_ctx;
        int kvh = h / kv_mul;
        for (int t = 0; t <= j->pos; t++) {
            const f16_t *kt = j->kc + (size_t)t * kv_dim + kvh * hd;
            float s = 0;
            for (int i = 0; i < hd; i++) s += qh[i] * f16_load(kt + i);
            att[t] = s * scale;
        }
        softmax(att, j->pos + 1);
        float *out = j->out + h * hd;
        memset(out, 0, sizeof(float) * hd);
        for (int t = 0; t <= j->pos; t++) {
            const f16_t *vt = j->vc + (size_t)t * kv_dim + kvh * hd;
            float a = att[t];
            for (int i = 0; i < hd; i++) out[i] += a * f16_load(vt + i);
        }
    }
}

// ---------------------------------------------------------------- forward

float *model_forward_batch(model_t *m, const int32_t *tokens, int n, int pos,
                           bool want_logits) {
    int n_embd = m->n_embd;
    int q_dim  = m->n_head * m->head_dim;
    int kv_dim = m->n_head_kv * m->head_dim;
    int xdim   = q_dim > n_embd ? q_dim : n_embd;

    size_t ers = ggml_row_size(m->tok_embd->type, n_embd);
    for (int b = 0; b < n; b++)
        dequant_row(m->tok_embd->type,
                    (uint8_t *)m->tok_embd->data + (size_t)tokens[b] * ers,
                    m->x + (size_t)b * n_embd, n_embd);

    for (int l = 0; l < m->n_layer; l++) {
        layer_t *ly = &m->layers[l];
        f16_t *kc_l = m->kcache + (size_t)l * m->n_ctx * kv_dim;
        f16_t *vc_l = m->vcache + (size_t)l * m->n_ctx * kv_dim;

        // attention
        for (int b = 0; b < n; b++)
            rmsnorm(m->xb + (size_t)b * xdim, m->x + (size_t)b * n_embd,
                    ly->attn_norm_w, n_embd, m->rms_eps);
        matvec_b(m->tp, m->q,     q_dim,  ly->wq, m->xb, xdim, n_embd, q_dim,  ly->bq, n);
        matvec_b(m->tp, m->k_tmp, kv_dim, ly->wk, m->xb, xdim, n_embd, kv_dim, ly->bk, n);
        matvec_b(m->tp, m->v_tmp, kv_dim, ly->wv, m->xb, xdim, n_embd, kv_dim, ly->bv, n);
        for (int b = 0; b < n; b++) {
            rope_apply(m, m->q + (size_t)b * q_dim, m->n_head, pos + b);
            rope_apply(m, m->k_tmp + (size_t)b * kv_dim, m->n_head_kv, pos + b);
            f16_t *kc = kc_l + (size_t)(pos + b) * kv_dim;
            f16_t *vc = vc_l + (size_t)(pos + b) * kv_dim;
            for (int i = 0; i < kv_dim; i++) {
                kc[i] = f32_to_f16(m->k_tmp[(size_t)b * kv_dim + i]);
                vc[i] = f32_to_f16(m->v_tmp[(size_t)b * kv_dim + i]);
            }
        }
        for (int b = 0; b < n; b++) {
            attn_job aj = { m, kc_l, vc_l, m->q + (size_t)b * q_dim,
                            m->xb2 + (size_t)b * xdim, pos + b };
            tpool_run(m->tp, attn_heads, &aj, m->n_head);
        }
        matvec_b(m->tp, m->xb, xdim, ly->wo, m->xb2, xdim, q_dim, n_embd, ly->bo, n);
        for (int b = 0; b < n; b++)
            for (int i = 0; i < n_embd; i++)
                m->x[(size_t)b * n_embd + i] += m->xb[(size_t)b * xdim + i];

        // feed-forward (SwiGLU)
        for (int b = 0; b < n; b++)
            rmsnorm(m->xb + (size_t)b * xdim, m->x + (size_t)b * n_embd,
                    ly->ffn_norm_w, n_embd, m->rms_eps);
        matvec_b(m->tp, m->hb,  m->n_ff, ly->w_gate, m->xb, xdim, n_embd, m->n_ff, NULL, n);
        matvec_b(m->tp, m->hb2, m->n_ff, ly->w_up,   m->xb, xdim, n_embd, m->n_ff, NULL, n);
        for (size_t i = 0; i < (size_t)n * m->n_ff; i++) {
            float g = m->hb[i];
            m->hb[i] = (g / (1.0f + expf(-g))) * m->hb2[i]; // silu(g) * up
        }
        matvec_b(m->tp, m->xb, xdim, ly->w_down, m->hb, m->n_ff, m->n_ff, n_embd, NULL, n);
        for (int b = 0; b < n; b++)
            for (int i = 0; i < n_embd; i++)
                m->x[(size_t)b * n_embd + i] += m->xb[(size_t)b * xdim + i];
    }

    if (!want_logits) return NULL;
    rmsnorm(m->xb, m->x + (size_t)(n - 1) * n_embd, m->out_norm_w, n_embd, m->rms_eps);
    matvec_b(m->tp, m->logits, m->n_vocab, m->output, m->xb, xdim, n_embd, m->n_vocab, NULL, 1);
    return m->logits;
}

float *model_forward(model_t *m, int token, int pos) {
    int32_t t = token;
    return model_forward_batch(m, &t, 1, pos, true);
}
