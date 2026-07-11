// Llama-family transformer: weight wiring + batched forward pass.
#include "runner.h"
#include "compat.h"

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

    if (m->swa_window > 0) {
        // sliding-window layers rope at their own (short-context) base with
        // no scaling — gemma locals use 10k while globals run 1M + scaling;
        // gemma4 locals also rotate fewer dims (rope_dim_local)
        float local_base = gguf_get_f32(g, RK("rope.local.freq_base"),
                           gguf_get_f32(g, RK("rope.freq_base_swa"), 10000.0f));
        int lhalf = m->rope_dim_local / 2;
        m->rope_inv_freq_local = malloc(sizeof(float) * lhalf);
        for (int j = 0; j < lhalf; j++)
            m->rope_inv_freq_local[j] = powf(local_base, -2.0f * j / m->rope_dim_local);
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
    // architectures whose weights load fine llama-style but whose math is
    // silently wrong without arch-specific handling (scalar multipliers,
    // logit softcapping): refuse instead of generating plausible gibberish
    if (strcmp(arch, "granite") == 0 || strcmp(arch, "gemma2") == 0 ||
        strcmp(arch, "gemma") == 0) {
        fprintf(stderr, "error: unsupported architecture '%s' — it would load "
                "but produce incorrect output without its scaling/softcapping\n", arch);
        return false;
    }
    if (strcmp(arch, "llama") != 0 && strcmp(arch, "qwen2") != 0 &&
        strcmp(arch, "qwen3") != 0 && strcmp(arch, "mistral") != 0 &&
        strcmp(arch, "smollm") != 0 && strcmp(arch, "stablelm") != 0 &&
        strcmp(arch, "gemma3") != 0 && strcmp(arch, "gemma4") != 0) {
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
    m->embd_scale  = 1.0f;
    m->rope_dim_local = m->rope_dim;
    if (strcmp(arch, "gemma3") == 0) {
        // gemma3: scaled embeddings, GELU ffn, sliding-window attention on 5
        // of every 6 layers (the 6th is global), locals rope at their own base
        m->embd_scale  = sqrtf((float)m->n_embd);
        m->ffn_act     = ACT_GELU;
        m->swa_window  = (int)gguf_get_u32(g, AK("attention.sliding_window"), 0);
        m->rms_eps     = gguf_get_f32(g, AK("attention.layer_norm_rms_epsilon"), 1e-6f);
        // gemma3 ropes global layers at 1M; some exports omit the key
        m->rope_base   = gguf_get_f32(g, AK("rope.freq_base"), 1000000.0f);
        int pattern    = (int)gguf_get_u32(g, AK("attention.sliding_window_pattern"), 6);
        m->l_is_swa    = calloc(m->n_layer, sizeof(bool));
        for (int i = 0; i < m->n_layer; i++)
            m->l_is_swa[i] = m->swa_window > 0 && ((i + 1) % pattern) != 0;
    }
    if (strcmp(arch, "gemma4") == 0) {
        // gemma4 (reference: llama.cpp src/models/gemma4.cpp): heterogeneous
        // layers — per-layer kv heads and head dims (global 512 / sliding
        // 256), global layers may have no V projection (V reuses the raw K
        // projection), every V gets a weightless per-head RMS norm, attention
        // scale is fixed 1.0, each layer's output is scaled by a per-layer
        // scalar, and final logits are softcapped. Verified against llama.cpp
        // b9964: greedy raw completions are token-identical on the official
        // ggml-org gemma-4-12B-it Q4_K_M, and chat-formatted prompts answer
        // correctly. Note the model is thinking-tuned: raw untemplated
        // completions are legitimately degenerate, and llama.cpp additionally
        // biases tokenizer.ggml.suppress_tokens to -inf (not done here).
        if (gguf_get_u32(g, AK("attention.shared_kv_layers"), 0) > 0 ||
            gguf_get_u32(g, AK("embedding_length_per_layer_input"), 0) > 0) {
            fprintf(stderr, "error: unsupported architecture variant '%s' — "
                    "shared-KV / per-layer-embedding gemma4 models (e.g. E2B/E4B) "
                    "are not supported yet\n", arch);
            return false;
        }
        fprintf(stderr, "gemma4: dense variant verified against llama.cpp (token-identical greedy output on the official ggml-org GGUF); unofficial dequant conversions may still produce garbage — prefer official files\n");
        m->embd_scale    = sqrtf((float)m->n_embd);
        m->ffn_act       = ACT_GELU;
        m->v_rmsnorm     = true;
        m->attn_scale    = 1.0f;
        // thinking-tuned: responses interleave <|channel>thought ... <channel|>
        // reasoning blocks with the answer text (split out by think_feed)
        m->think_open    = "<|channel>thought";
        m->think_close   = "<channel|>";
        m->logit_softcap = gguf_get_f32(g, AK("final_logit_softcapping"), 0.0f);
        m->swa_window    = (int)gguf_get_u32(g, AK("attention.sliding_window"), 0);
        m->rms_eps       = gguf_get_f32(g, AK("attention.layer_norm_rms_epsilon"), 1e-6f);
        m->rope_base     = gguf_get_f32(g, AK("rope.freq_base"), 1000000.0f);
        m->head_dim      = (int)gguf_get_u32(g, AK("attention.key_length"), 512);
        m->rope_dim      = (int)gguf_get_u32(g, AK("rope.dimension_count"), m->head_dim);
        m->rope_dim_local = (int)gguf_get_u32(g, AK("rope.dimension_count_swa"), m->rope_dim);
        int hd_swa       = (int)gguf_get_u32(g, AK("attention.key_length_swa"), m->head_dim);
        m->l_is_swa   = calloc(m->n_layer, sizeof(bool));
        m->l_head_kv  = calloc(m->n_layer, sizeof(int));
        m->l_head_dim = calloc(m->n_layer, sizeof(int));
        m->l_rope_dim = calloc(m->n_layer, sizeof(int));
        gguf_kv *swa_arr = gguf_get(g, AK("attention.sliding_window_pattern"));
        gguf_kv *kv_arr  = gguf_get(g, AK("attention.head_count_kv"));
        for (int i = 0; i < m->n_layer; i++) {
            bool swa = swa_arr && swa_arr->arr_raw && (uint64_t)i < swa_arr->arr_n
                       ? ((const uint8_t *)swa_arr->arr_raw)[i] != 0 : false;
            m->l_is_swa[i]   = swa && m->swa_window > 0;
            m->l_head_dim[i] = swa ? hd_swa : m->head_dim;
            m->l_rope_dim[i] = swa ? m->rope_dim_local : m->rope_dim;
            m->l_head_kv[i]  = kv_arr && kv_arr->arr_raw && (uint64_t)i < kv_arr->arr_n
                               ? (int)((const uint32_t *)kv_arr->arr_raw)[i] : m->n_head_kv;
        }
    }
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
        l->wv     = m->v_rmsnorm ? opt_tensor(g, "blk.%d.attn_v.weight", i)
                                 : need_tensor(g, "blk.%d.attn_v.weight", i, &ok);
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
        l->qnorm_w = tensor_to_f32(opt_tensor(g, "blk.%d.attn_q_norm.weight", i));
        l->knorm_w = tensor_to_f32(opt_tensor(g, "blk.%d.attn_k_norm.weight", i));
        l->post_attn_norm_w = tensor_to_f32(opt_tensor(g, "blk.%d.post_attention_norm.weight", i));
        l->post_ffn_norm_w  = tensor_to_f32(opt_tensor(g, "blk.%d.post_ffw_norm.weight", i));
        l->out_scale = 1.0f;
        gguf_tensor *osc = opt_tensor(g, "blk.%d.layer_output_scale.weight", i);
        if (osc && osc->type == T_F32) l->out_scale = ((const float *)osc->data)[0];
    }

    // gemma4 dropped the plus-one norm convention (unlike gemma1-3): its
    // RMSNorm is the standard x_normed * weight, so norm weights are used raw.

    // checkpoint workaround (gemma4 ships one): ids the model must never emit;
    // their logits are forced to -inf after every forward pass
    gguf_kv *sup = gguf_get(g, "tokenizer.ggml.suppress_tokens");
    if (sup && sup->arr_raw && sup->arr_type == GGUF_T_I32 && sup->arr_n > 0) {
        const int32_t *ids = (const int32_t *)sup->arr_raw;
        m->suppress = malloc(sizeof(int32_t) * sup->arr_n);
        for (uint64_t i = 0; i < sup->arr_n; i++)
            if (ids[i] >= 0 && ids[i] < m->n_vocab)
                m->suppress[m->n_suppress++] = ids[i];
    }

    // runtime buffers
    m->reserve_vram_pct = p->reserve_vram_pct;
    int n_ctx = p->n_ctx;
    if (n_ctx <= 0 && (p->reserve_vram_pct > 0 || p->reserve_ram_pct > 0)) {
        // reservation auto-fit: size the context to fill whatever the
        // reservation leaves after the weights, so small models grow their
        // window into the reserved room instead of idling at the default
        size_t kv_per_tok = 0;
        for (int l = 0; l < m->n_layer; l++)
            kv_per_tok += 2ull * model_kv_dim(m, l) * sizeof(f16_t);
        size_t head = 256u << 20;                 // activations + slack
        long long best = -1;
        if (p->reserve_ram_pct > 0) {
            // host budget covers the mmap'd weights plus the host KV copy
            size_t budget = plat_ram_bytes() / 100 * p->reserve_ram_pct;
            long long room = (long long)budget - (long long)m->gf.map_size - (long long)head;
            long long fit = room > 0 ? room / (long long)kv_per_tok : 0;
            best = fit;
        }
        if (p->reserve_vram_pct > 0 && p->gpu_mode == GPU_AUTO) {
            size_t vfree = 0, vtotal = 0;
            if (gpu_mem_info(&vfree, &vtotal)) {
                // device budget covers a weights copy plus the device KV
                size_t budget = vtotal / 100 * p->reserve_vram_pct;
                long long room = (long long)budget - (long long)m->gf.map_size - (long long)head;
                long long fit = room > 0 ? room / (long long)kv_per_tok : 0;
                if (best < 0 || fit < best) best = fit;
            }
        }
        if (best > 0) {
            n_ctx = best > m->n_ctx_train ? m->n_ctx_train : (int)best;
            if (n_ctx < 512) n_ctx = 512;
            fprintf(stderr, "reservation: auto-fit context %d (train %d)\n",
                    n_ctx, m->n_ctx_train);
        }
    }
    if (n_ctx <= 0) n_ctx = m->n_ctx_train < 4096 ? m->n_ctx_train : 4096;
    m->n_ctx = n_ctx;
    m->n_batch = p->n_batch > 0 ? p->n_batch : 64;
    if (m->n_batch > n_ctx) m->n_batch = n_ctx;
    int q_dim = 0, kv_dim = 0;
    for (int l = 0; l < m->n_layer; l++) {
        if (model_q_dim(m, l) > q_dim)   q_dim  = model_q_dim(m, l);
        if (model_kv_dim(m, l) > kv_dim) kv_dim = model_kv_dim(m, l);
    }
    int xdim   = q_dim > m->n_embd ? q_dim : m->n_embd;
    int B      = m->n_batch;

    // per-layer element offsets into the (possibly heterogeneous) KV cache
    m->kv_off = malloc(sizeof(size_t) * (m->n_layer + 1));
    m->kv_off[0] = 0;
    for (int l = 0; l < m->n_layer; l++)
        m->kv_off[l + 1] = m->kv_off[l] + (size_t)n_ctx * model_kv_dim(m, l);
    size_t kv_bytes = m->kv_off[m->n_layer] * sizeof(f16_t);
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

    if (p->gpu_mode == GPU_AUTO) gpu_init(m); // sets m->gpu on success

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

void model_free(model_t *m) {
    gpu_free(m); // nulls kcache/vcache if the GPU owned them
    for (int i = 0; i < m->n_layer; i++) {
        layer_t *l = &m->layers[i];
        free(l->attn_norm_w); free(l->ffn_norm_w);
        free(l->bq); free(l->bk); free(l->bv); free(l->bo);
        free(l->qnorm_w); free(l->knorm_w);
        free(l->post_attn_norm_w); free(l->post_ffn_norm_w);
    }
    free(m->l_head_kv); free(m->l_head_dim); free(m->l_rope_dim);
    free(m->l_is_swa); free(m->kv_off); free(m->suppress);
    free(m->layers);
    free(m->out_norm_w);
    free(m->rope_inv_freq);
    free(m->rope_inv_freq_local);
    free(m->kcache); free(m->vcache);
    free(m->x); free(m->xb); free(m->xb2); free(m->q);
    free(m->k_tmp); free(m->v_tmp);
    free(m->hb); free(m->hb2); free(m->att); free(m->logits);
    tpool_destroy(m->tp);
    gguf_close(&m->gf);
    memset(m, 0, sizeof(*m));
}

// ---------------------------------------------------------------- math ops

static void rmsnorm(float *o, const float *x, const float *w, int n, float eps) {
    float ss = 0;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float r = 1.0f / sqrtf(ss / n + eps);
    if (w) for (int i = 0; i < n; i++) o[i] = x[i] * r * w[i];
    else   for (int i = 0; i < n; i++) o[i] = x[i] * r;      // weightless (gemma4 V)
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

// qwen3-style per-head RMSNorm on Q or K (weight is one head_dim vector)
static void qk_norm(float *v, const float *w, int n_heads, int head_dim, float eps) {
    for (int h = 0; h < n_heads; h++)
        rmsnorm(v + h * head_dim, v + h * head_dim, w, head_dim, eps);
}

static void rope_apply(model_t *m, float *v, int n_heads, int pos, int layer) {
    bool local = model_is_swa(m, layer);
    int hd   = model_head_dim(m, layer);
    int half = model_rope_dim(m, layer) / 2;
    const float *fr = local ? m->rope_inv_freq_local : m->rope_inv_freq;
    float ms = local ? 1.0f : m->rope_mscale;
    for (int j = 0; j < half; j++) {
        float a = pos * fr[j];
        float c = cosf(a) * ms, s = sinf(a) * ms;
        for (int h = 0; h < n_heads; h++) {
            float *p = v + h * hd;
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
    int t0;                 // first attended position (sliding window)
    int hd, kv_dim;         // this layer's head dim / kv row width
    float scale;
} attn_job;

static void attn_heads(void *ctx, int h0, int h1) {
    attn_job *j = ctx;
    model_t *m = j->m;
    int hd = j->hd;
    int kv_dim = j->kv_dim;
    int kv_mul = m->n_head / (kv_dim / hd);
    float scale = j->scale;

    for (int h = h0; h < h1; h++) {
        const float *qh = j->q + h * hd;
        float *att = m->att + (size_t)h * m->n_ctx;
        int kvh = h / kv_mul;
        for (int t = j->t0; t <= j->pos; t++) {
            const f16_t *kt = j->kc + (size_t)t * kv_dim + kvh * hd;
            float s = 0;
            for (int i = 0; i < hd; i++) s += qh[i] * f16_load(kt + i);
            att[t] = s * scale;
        }
        softmax(att + j->t0, j->pos + 1 - j->t0);
        float *out = j->out + h * hd;
        memset(out, 0, sizeof(float) * hd);
        for (int t = j->t0; t <= j->pos; t++) {
            const f16_t *vt = j->vc + (size_t)t * kv_dim + kvh * hd;
            float a = att[t];
            for (int i = 0; i < hd; i++) out[i] += a * f16_load(vt + i);
        }
    }
}

// ---------------------------------------------------------------- forward

// suppress_tokens checkpoint workaround: a large finite constant instead of
// -INFINITY because the binary is built with -ffast-math (finite-math-only)
static void suppress_logits(const model_t *m, float *logits) {
    for (int i = 0; i < m->n_suppress; i++)
        logits[m->suppress[i]] = -1e30f;
}

float *model_forward_batch(model_t *m, const int32_t *tokens, int n, int pos,
                           bool want_logits) {
    // GPU handles the leading gpu_layers. A full split (gpu_layers == n_layer)
    // returns logits directly; a partial split runs [0, gpu_layers) on the GPU,
    // leaves the boundary activation in the host x buffer + the offloaded
    // layers' KV in the host cache, and the CPU loop below finishes the rest.
    int start = 0;
    if (m->gpu) {
        if (m->gpu_layers >= m->n_layer) {
            float *lg = NULL;
            if (gpu_forward_batch(m, tokens, n, pos, want_logits, &lg)) {
                if (lg && m->n_suppress) suppress_logits(m, lg);
                return lg;
            }
            m->gpu = NULL; // GPU failed at runtime: fall back to CPU permanently
        } else if (gpu_forward_batch(m, tokens, n, pos, false, NULL)) {
            start = m->gpu_layers;
        } else {
            m->gpu = NULL;
        }
    }
    int n_embd = m->n_embd;
    int xdim = n_embd;
    for (int l = 0; l < m->n_layer; l++)
        if (model_q_dim(m, l) > xdim) xdim = model_q_dim(m, l);

    if (start == 0) {
        size_t ers = ggml_row_size(m->tok_embd->type, n_embd);
        for (int b = 0; b < n; b++) {
            dequant_row(m->tok_embd->type,
                        (uint8_t *)m->tok_embd->data + (size_t)tokens[b] * ers,
                        m->x + (size_t)b * n_embd, n_embd);
            if (m->embd_scale != 1.0f)
                for (int i = 0; i < n_embd; i++)
                    m->x[(size_t)b * n_embd + i] *= m->embd_scale;
        }
    }

    for (int l = start; l < m->n_layer; l++) {
        layer_t *ly = &m->layers[l];
        bool local   = model_is_swa(m, l);
        int hd       = model_head_dim(m, l);
        int n_kv     = model_n_head_kv(m, l);
        int q_dim    = model_q_dim(m, l);
        int kv_dim   = model_kv_dim(m, l);
        float scale  = model_attn_scale(m, l);
        f16_t *kc_l = m->kcache + m->kv_off[l];
        f16_t *vc_l = m->vcache + m->kv_off[l];

        // attention
        for (int b = 0; b < n; b++)
            rmsnorm(m->xb + (size_t)b * xdim, m->x + (size_t)b * n_embd,
                    ly->attn_norm_w, n_embd, m->rms_eps);
        matvec_b(m->tp, m->q,     q_dim,  ly->wq, m->xb, xdim, n_embd, q_dim,  ly->bq, n);
        matvec_b(m->tp, m->k_tmp, kv_dim, ly->wk, m->xb, xdim, n_embd, kv_dim, ly->bk, n);
        if (ly->wv)
            matvec_b(m->tp, m->v_tmp, kv_dim, ly->wv, m->xb, xdim, n_embd, kv_dim, ly->bv, n);
        else
            // gemma4 global layers have no V projection: V is the raw K
            memcpy(m->v_tmp, m->k_tmp, sizeof(float) * (size_t)n * kv_dim);
        for (int b = 0; b < n; b++) {
            if (ly->qnorm_w)
                qk_norm(m->q + (size_t)b * q_dim, ly->qnorm_w, m->n_head,
                        hd, m->rms_eps);
            if (m->v_rmsnorm)
                // gemma4: weightless per-head RMS norm on V (pre-K-norm values)
                qk_norm(m->v_tmp + (size_t)b * kv_dim, NULL, n_kv, hd, m->rms_eps);
            if (ly->knorm_w)
                qk_norm(m->k_tmp + (size_t)b * kv_dim, ly->knorm_w, n_kv,
                        hd, m->rms_eps);
            rope_apply(m, m->q + (size_t)b * q_dim, m->n_head, pos + b, l);
            rope_apply(m, m->k_tmp + (size_t)b * kv_dim, n_kv, pos + b, l);
            f16_t *kc = kc_l + (size_t)(pos + b) * kv_dim;
            f16_t *vc = vc_l + (size_t)(pos + b) * kv_dim;
            for (int i = 0; i < kv_dim; i++) {
                kc[i] = f32_to_f16(m->k_tmp[(size_t)b * kv_dim + i]);
                vc[i] = f32_to_f16(m->v_tmp[(size_t)b * kv_dim + i]);
            }
        }
        for (int b = 0; b < n; b++) {
            int p = pos + b;
            int t0 = local && p - m->swa_window + 1 > 0 ? p - m->swa_window + 1 : 0;
            attn_job aj = { m, kc_l, vc_l, m->q + (size_t)b * q_dim,
                            m->xb2 + (size_t)b * xdim, p, t0, hd, kv_dim, scale };
            tpool_run(m->tp, attn_heads, &aj, m->n_head);
        }
        matvec_b(m->tp, m->xb, xdim, ly->wo, m->xb2, xdim, q_dim, n_embd, ly->bo, n);
        if (ly->post_attn_norm_w)
            for (int b = 0; b < n; b++)
                rmsnorm(m->xb + (size_t)b * xdim, m->xb + (size_t)b * xdim,
                        ly->post_attn_norm_w, n_embd, m->rms_eps);
        for (int b = 0; b < n; b++)
            for (int i = 0; i < n_embd; i++)
                m->x[(size_t)b * n_embd + i] += m->xb[(size_t)b * xdim + i];

        // feed-forward (gated: silu for llama-family, gelu for gemma)
        for (int b = 0; b < n; b++)
            rmsnorm(m->xb + (size_t)b * xdim, m->x + (size_t)b * n_embd,
                    ly->ffn_norm_w, n_embd, m->rms_eps);
        matvec_b(m->tp, m->hb,  m->n_ff, ly->w_gate, m->xb, xdim, n_embd, m->n_ff, NULL, n);
        matvec_b(m->tp, m->hb2, m->n_ff, ly->w_up,   m->xb, xdim, n_embd, m->n_ff, NULL, n);
        if (m->ffn_act == ACT_GELU) {
            for (size_t i = 0; i < (size_t)n * m->n_ff; i++) {
                float g = m->hb[i];
                float t = tanhf(0.7978845608f * (g + 0.044715f * g * g * g));
                m->hb[i] = 0.5f * g * (1.0f + t) * m->hb2[i]; // gelu(g) * up
            }
        } else {
            for (size_t i = 0; i < (size_t)n * m->n_ff; i++) {
                float g = m->hb[i];
                m->hb[i] = (g / (1.0f + expf(-g))) * m->hb2[i]; // silu(g) * up
            }
        }
        matvec_b(m->tp, m->xb, xdim, ly->w_down, m->hb, m->n_ff, m->n_ff, n_embd, NULL, n);
        if (ly->post_ffn_norm_w)
            for (int b = 0; b < n; b++)
                rmsnorm(m->xb + (size_t)b * xdim, m->xb + (size_t)b * xdim,
                        ly->post_ffn_norm_w, n_embd, m->rms_eps);
        for (int b = 0; b < n; b++)
            for (int i = 0; i < n_embd; i++)
                m->x[(size_t)b * n_embd + i] += m->xb[(size_t)b * xdim + i];

        if (ly->out_scale != 1.0f && ly->out_scale != 0.0f)
            // gemma4: whole-layer output scalar, applied after both residuals
            for (int b = 0; b < n; b++)
                for (int i = 0; i < n_embd; i++)
                    m->x[(size_t)b * n_embd + i] *= ly->out_scale;
    }

    if (!want_logits) return NULL;
    rmsnorm(m->xb, m->x + (size_t)(n - 1) * n_embd, m->out_norm_w, n_embd, m->rms_eps);
    matvec_b(m->tp, m->logits, m->n_vocab, m->output, m->xb, xdim, n_embd, m->n_vocab, NULL, 1);
    if (m->logit_softcap > 0)
        for (int i = 0; i < m->n_vocab; i++)
            m->logits[i] = m->logit_softcap * tanhf(m->logits[i] / m->logit_softcap);
    if (m->n_suppress) suppress_logits(m, m->logits);
    return m->logits;
}

float *model_forward(model_t *m, int token, int pos) {
    int32_t t = token;
    return model_forward_batch(m, &t, 1, pos, true);
}
