// Llama-family transformer: weight wiring + batched forward pass.
#include "runner.h"
#include "compat.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---------------------------------------------------------------- helpers

static int64_t stat_mtime_ns(const struct stat *st) {
#if defined(__APPLE__) || defined(__FreeBSD__) || defined(__OpenBSD__) || defined(__NetBSD__)
    return (int64_t)st->st_mtimespec.tv_sec * 1000000000ll +
           (int64_t)st->st_mtimespec.tv_nsec;
#elif defined(__linux__)
    return (int64_t)st->st_mtim.tv_sec * 1000000000ll +
           (int64_t)st->st_mtim.tv_nsec;
#else
    return (int64_t)st->st_mtime * 1000000000ll;
#endif
}

static int64_t stat_ctime_ns(const struct stat *st) {
#if defined(__APPLE__) || defined(__FreeBSD__) || defined(__OpenBSD__) || defined(__NetBSD__)
    return (int64_t)st->st_ctimespec.tv_sec * 1000000000ll +
           (int64_t)st->st_ctimespec.tv_nsec;
#elif defined(__linux__)
    return (int64_t)st->st_ctim.tv_sec * 1000000000ll +
           (int64_t)st->st_ctim.tv_nsec;
#else
    return (int64_t)st->st_ctime * 1000000000ll;
#endif
}

#ifdef _WIN32
static int64_t filetime_unix_ns(FILETIME ft) {
    ULARGE_INTEGER ticks;
    ticks.LowPart = ft.dwLowDateTime;
    ticks.HighPart = ft.dwHighDateTime;
    // FILETIME is 100 ns since 1601; normalize to Unix epoch so the value
    // remains inside signed 64-bit while retaining NTFS timestamp precision.
    const uint64_t unix_epoch = 116444736000000000ull;
    return ticks.QuadPart >= unix_epoch
        ? (int64_t)((ticks.QuadPart - unix_epoch) * 100ull) : 0;
}
#endif

static void model_record_file_id(model_t *m, const char *path) {
#ifdef _WIN32
    // MinGW's struct stat exposes only whole-second timestamps. That can make
    // two different GGUF revisions loaded in the same second share a prefix-
    // cache key. Native file information gives the stable file index and the
    // filesystem's 100 ns last-write timestamp without hashing multi-GB files.
    HANDLE h = path ? CreateFileA(path, FILE_READ_ATTRIBUTES,
                                  FILE_SHARE_READ | FILE_SHARE_WRITE |
                                  FILE_SHARE_DELETE, NULL, OPEN_EXISTING,
                                  FILE_ATTRIBUTE_NORMAL, NULL)
                    : INVALID_HANDLE_VALUE;
    BY_HANDLE_FILE_INFORMATION info;
    if (h != INVALID_HANDLE_VALUE && GetFileInformationByHandle(h, &info)) {
        m->file_id_ok = true;
        m->file_size = ((uint64_t)info.nFileSizeHigh << 32) |
                       info.nFileSizeLow;
        m->file_ino = ((uint64_t)info.nFileIndexHigh << 32) |
                      info.nFileIndexLow;
        m->file_mtime_ns = filetime_unix_ns(info.ftLastWriteTime);
        m->file_ctime_ns = filetime_unix_ns(info.ftCreationTime);
        CloseHandle(h);
        return;
    }
    if (h != INVALID_HANDLE_VALUE) CloseHandle(h);
#endif
    struct stat st;
    if (path && stat(path, &st) == 0) {
        m->file_id_ok = true;
        m->file_size = (uint64_t)st.st_size;
#ifndef _WIN32
        m->file_ino = (uint64_t)st.st_ino;
#else
        m->file_ino = 0;
#endif
        m->file_mtime_ns = stat_mtime_ns(&st);
        m->file_ctime_ns = stat_ctime_ns(&st);
    } else {
        // Still include the mapped length in identities on platforms where
        // stat failed, rather than falling back to path alone.
        m->file_id_ok = false;
        m->file_size = (uint64_t)m->gf.map_size;
        m->file_ino = 0;
        m->file_mtime_ns = 0;
        m->file_ctime_ns = 0;
    }
}

// Materialize a tensor into a freshly-allocated f32 buffer. Returns NULL when
// t is absent (a legitimate result for optional tensors). If t is present but
// the allocation fails, sets *ok = false and returns NULL, so an out-of-memory
// condition is never silently mistaken for an absent optional tensor (which
// would change the forward-pass math without any error).
static float *tensor_to_f32(gguf_tensor *t, bool *ok) {
    if (!t) return NULL;
    // The element count comes from untrusted GGUF dimensions. Compute it with
    // overflow checks (a raw ne[0]*ne[1]*ne[2]*ne[3] can wrap) and make sure the
    // f32 allocation size cannot overflow either. gguf_open already bounded
    // t->nbytes to the mapping; cross-check that the bytes this dequant/copy
    // will touch (rows * row_size) actually fit inside that mapped extent so a
    // shape larger than the stored data can never be read past.
    uint64_t n64, rows;
    if (!checked_u64_mul(t->ne[0], t->ne[1], &n64) ||
        !checked_u64_mul(n64, t->ne[2], &n64) ||
        !checked_u64_mul(n64, t->ne[3], &n64) ||
        n64 == 0 || n64 > SIZE_MAX / sizeof(float)) {
        *ok = false; return NULL;
    }
    size_t rs = ggml_row_size(t->type, (int64_t)t->ne[0]);
    rows = n64 / t->ne[0];
    if (rs == 0 || rows > t->nbytes / rs) { *ok = false; return NULL; }
    int64_t n = (int64_t)n64;
    float *out = malloc(sizeof(float) * (size_t)n);
    if (!out) { *ok = false; return NULL; }
    if (t->type == T_F32) {
        memcpy(out, t->data, sizeof(float) * (size_t)n);
    } else {
        for (uint64_t r = 0; r < rows; r++)
            dequant_row(t->type, (uint8_t *)t->data + r * rs,
                        out + r * t->ne[0], (int)t->ne[0]);
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

// Reject a weight tensor whose dimensions do not match the geometry the forward
// pass will drive it with. matvec reads n_out rows of n_in elements each, at a
// stride derived from n_in, so ne[0] must equal n_in exactly (a wrong row length
// mis-strides every row) and ne[1] must cover the n_out rows that will be read
// (a shorter tensor would be read past its mapped bytes). A NULL tensor is a
// legitimately-absent optional weight (e.g. gemma4 V) and is accepted here.
static bool check_shape(gguf_tensor *t, int n_in, int n_out,
                        const char *what, int layer) {
    if (!t) return true;
    if ((int64_t)t->ne[0] != n_in || (int64_t)t->ne[1] < n_out) {
        fprintf(stderr, "error: tensor %s in blk.%d has shape [%llu,%llu], "
                "expected [%d,>=%d] for this model geometry\n",
                what, layer, (unsigned long long)t->ne[0],
                (unsigned long long)t->ne[1], n_in, n_out);
        return false;
    }
    return true;
}

// A fused 3D MoE expert tensor: ne[0]/ne[1] must be EXACT (they are the row
// length and per-expert row count the forward pass slices with — see
// moe_expert_weight — so a mismatch drifts every expert's offset, not just the
// last), and ne[2] must hold at least n_expert expert blocks. Without this a
// crafted GGUF whose expert tensors are smaller than the declared geometry
// loads clean and is then read out of bounds at decode time (RNC-1).
static bool check_shape3(gguf_tensor *t, int ne0, int ne1, int n_expert,
                         const char *what, int layer) {
    if (!t) return true;
    if ((int64_t)t->ne[0] != ne0 || (int64_t)t->ne[1] != ne1 ||
        (int64_t)t->ne[2] < n_expert) {
        fprintf(stderr, "error: MoE tensor %s in blk.%d has shape "
                "[%llu,%llu,%llu], expected [%d,%d,>=%d]\n",
                what, layer, (unsigned long long)t->ne[0],
                (unsigned long long)t->ne[1], (unsigned long long)t->ne[2],
                ne0, ne1, n_expert);
        return false;
    }
    return true;
}

// Carve a row range out of a tensor without copying. A row is a whole number
// of quantization blocks (row length is always a multiple of the block size),
// so the slice starts at an exact byte offset into the same mmapped data.
static gguf_tensor *slice_rows(gguf_tensor *src, gguf_tensor *dst,
                               int64_t row0, int64_t nrows) {
    if (!src) return NULL;
    size_t rs = ggml_row_size(src->type, (int64_t)src->ne[0]);
    *dst = *src;
    dst->ne[1] = (uint64_t)nrows;
    dst->ne[2] = dst->ne[3] = 1;
    dst->data  = (uint8_t *)src->data + (size_t)row0 * rs;
    dst->nbytes = (uint64_t)nrows * rs;
    return dst;
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

static bool rope_setup(model_t *m, gguf_file *g, const char *arch,
                       float base_ovr, float scale_ovr) {
    char key[128];
    #define RK(fmt) (snprintf(key, sizeof(key), "%s." fmt, arch), key)
    if (base_ovr > 0) m->rope_base = base_ovr;
    m->rope_mscale = 1.0f;

    int half = m->rope_dim / 2;
    m->rope_inv_freq = malloc(sizeof(float) * half);
    if (!m->rope_inv_freq) return false;
    for (int j = 0; j < half; j++)
        m->rope_inv_freq[j] = powf(m->rope_base, -2.0f * j / m->rope_dim);

    // per-dimension frequency factors (llama 3.x long-context models)
    gguf_tensor *ff = gguf_find_tensor(g, "rope_freqs.weight");
    if (ff && ff->type == T_F32 && (int)ff->ne[0] >= half) {
        const float *f = ff->data;
        for (int j = 0; j < half; j++) m->rope_inv_freq[j] /= f[j];
        fprintf(stderr, "rope: using model frequency factors (rope_freqs.weight)\n");
    }

    // phi3 LongRoPE ships two factor sets and which applies depends on the
    // context in use, not on the model: short factors are not identity, so
    // ignoring them mis-rotates even well inside the original window
    if (strcmp(arch, "phi3") == 0) {
        int orig_ctx = (int)gguf_get_u32(g, RK("rope.scaling.original_context_length"),
                                         m->n_ctx_train);
        bool use_long = orig_ctx > 0 && m->n_ctx > orig_ctx;
        gguf_tensor *rf = gguf_find_tensor(g, use_long ? "rope_factors_long.weight"
                                                       : "rope_factors_short.weight");
        if (rf && rf->type == T_F32 && (int)rf->ne[0] >= half) {
            const float *f = rf->data;
            for (int j = 0; j < half; j++) m->rope_inv_freq[j] /= f[j];
            m->rope_mscale = gguf_get_f32(g, RK("rope.scaling.attn_factor"), 1.0f);
            fprintf(stderr, "rope: phi3 LongRoPE %s factors, mscale %.4f\n",
                    use_long ? "long" : "short", (double)m->rope_mscale);
        }
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
        if (!m->rope_inv_freq_local) return false;
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
    return true;
}

// ---------------------------------------------------------------- load

// ------------------------------------------------- vram registry integration
//
// The registry needs three things this file already knows: which GPU, how much
// this instance intends to hold, and what it ended up holding.

static uint64_t vram_free_now(void *ud) {
    (void)ud;
    size_t f = 0, t = 0;
    return gpu_mem_info(&f, &t) ? (uint64_t)f : 0;
}

static uint64_t model_cuda_weight_estimate(const model_t *m,
                                           const model_params *p) {
    if (!p->cpu_moe || m->n_expert <= 0) return (uint64_t)m->gf.map_size;
    uint64_t total = m->output ? m->output->nbytes : 0;
    for (int l = 0; l < m->n_layer; l++) {
        const layer_t *ly = &m->layers[l];
        gguf_tensor *att[] = { ly->wq, ly->wk, ly->wv, ly->wo };
        for (int i = 0; i < 4; i++) if (att[i]) total += att[i]->nbytes;
        if (!ly->is_moe) {
            gguf_tensor *ffn[] = { ly->w_gate, ly->w_up, ly->w_down };
            for (int i = 0; i < 3; i++) if (ffn[i]) total += ffn[i]->nbytes;
        }
    }
    return total;
}

// Register the intended footprint before a byte of it is allocated.
//
// Returns false only to abort the load: that happens when the request does not
// fit AND the registry can name who is holding the memory. An unattributable
// shortfall is left to the backend's existing adaptive split, which trims
// layers onto the CPU — refusing there would regress every legitimate
// partial-offload run on a small GPU into a hard failure.
static bool model_vram_claim(model_t *m, const model_params *p, size_t kv_bytes) {
    char gpu_id[128];
    if (!gpu_device_id(gpu_id, sizeof(gpu_id))) return true;   // no GPU: nothing to account
    size_t vfree = 0, vtotal = 0;
    if (!gpu_mem_info(&vfree, &vtotal)) return true;           // unified memory (Metal)

    // What a full offload would hold: the weights, this instance's KV cache,
    // and the same fixed margin cuda.c budgets for context + JIT + activations.
    // An estimate is the right resolution here — it decides fit, and the exact
    // figure replaces it at commit time.
    uint64_t need = model_cuda_weight_estimate(m, p) +
                    (uint64_t)kv_bytes * 2 + (512ull << 20);
    if (p->reserve_vram_pct > 0) {
        uint64_t cap = (uint64_t)vtotal / 100 * (uint64_t)p->reserve_vram_pct;
        if (cap < need) need = cap;   // --reserve-vram already caps the ask
    }

    char err[1024];
    vram_status st = {0};
    m->vram = vram_claim(gpu_id, m->path, need, vram_free_now, NULL,
                         p->vram_wait_secs, p->load_cancel, &st, err, sizeof(err));
    if (m->vram) return true;

    if (st.holders > 0) {
        fprintf(stderr, "error: %s\n", err);
        return false;
    }
    // Nobody to blame: claim what is actually available so the next runner can
    // still see this instance, and let the adaptive split size itself down.
    m->vram = vram_claim(gpu_id, m->path, st.available, vram_free_now, NULL,
                         0, NULL, NULL, NULL, 0);
    return true;
}

// The split is decided and uploaded: replace the estimate with what the device
// really lost, measured rather than predicted. `before` is the free figure from
// immediately before gpu_init.
//
// Measuring the delta is what keeps this honest across every path at once — a
// full offload, a partial offload that trimmed layers onto the CPU, a backend
// that declined the model entirely, and the shared-weights case where a second
// instance of the same file uploads nothing because the first already did. All
// four report the truth without this file knowing which one happened.
static void model_vram_commit(model_t *m, size_t before) {
    if (!m->vram) return;
    size_t vfree = 0, vtotal = 0;
    if (!gpu_mem_info(&vfree, &vtotal)) { vram_commit(m->vram, 0); return; }
    // A load that freed memory nets to zero rather than underflowing.
    vram_commit(m->vram, before > vfree ? (uint64_t)(before - vfree) : 0);
}

const char *const *model_supported_archs(size_t *count) {
    // One source of truth for the admission allowlist (see model_load below and
    // --caps in main.c). Keep in sync with any new per-family handling added
    // here; wrong-math archs (granite/gemma2/gemma) are intentionally excluded.
    static const char *const arches[] = {
        "llama", "qwen2", "qwen3", "qwen35", "qwen3moe", "mistral",
        "smollm", "stablelm", "gemma3", "gemma4", "phi3",
    };
    if (count) *count = sizeof(arches) / sizeof(arches[0]);
    return arches;
}

static bool model_load_inner(model_t *m, const char *path, const model_params *p);

bool model_load(model_t *m, const char *path, const model_params *p) {
    memset(m, 0, sizeof(*m));
    if (!model_load_inner(m, path, p)) {
        model_free(m);
        return false;
    }
    return true;
}

static bool model_load_inner(model_t *m, const char *path, const model_params *p) {
    if (!gguf_open(&m->gf, path)) return false;
    gguf_file *g = &m->gf;
    // kept for the backend's shared-weight registry: two instances of the same
    // file are what let `--parallel N` upload the weights once
    size_t plen = strlen(path) + 1;
    m->path = malloc(plen);
    if (!m->path) return false;
    memcpy(m->path, path, plen);
    model_record_file_id(m, path);

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
    size_t n_arch;
    const char *const *ok_archs = model_supported_archs(&n_arch);
    bool arch_known = false;
    for (size_t i = 0; i < n_arch; i++)
        if (strcmp(arch, ok_archs[i]) == 0) { arch_known = true; break; }
    if (!arch_known) {
        // Admission is an allowlist: tensor-name compatibility is not proof of
        // mathematical compatibility (Q/K layout, norms, rope, activations,
        // softcapping all vary per family). Running an unknown architecture
        // through llama-style math produces plausible-looking WRONG output —
        // worse than a clear refusal, especially under an agent. Experimental
        // llama-style loading stays available behind an explicit opt-in that
        // is never represented as supported.
        if (!getenv("RUNNER_ALLOW_UNKNOWN_ARCH")) {
            fprintf(stderr, "error: unsupported architecture '%s' — refusing "
                    "to run it through llama-style math (set "
                    "RUNNER_ALLOW_UNKNOWN_ARCH=1 to try anyway, EXPERIMENTAL: "
                    "output may be silently wrong)\n", arch);
            return false;
        }
        fprintf(stderr, "warning: architecture '%s' is UNSUPPORTED; "
                "RUNNER_ALLOW_UNKNOWN_ARCH is set — attempting llama-style "
                "load, output may be silently wrong\n", arch);
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
        if (!m->l_is_swa) return false;
        for (int i = 0; i < m->n_layer; i++)
            m->l_is_swa[i] = m->swa_window > 0 && ((i + 1) % pattern) != 0;
    }
    if (strcmp(arch, "qwen3") == 0 || strcmp(arch, "qwen3moe") == 0) {
        // Thinking-tuned Qwen3 (dense and sparse-MoE) responses wrap hidden
        // reasoning before the visible answer; shared CLI/server output
        // handling splits this pair. qwen3moe = qwen3 attention (qk-norm, GQA,
        // NeoX rope) with a sparse-MoE FFN, handled by the is_moe layer path.
        m->think_open  = "<think>";
        m->think_close = "</think>";
    }
    if (strcmp(arch, "qwen35") == 0) {
        // Qwen3.5 dense (the architecture used by Ornith-1.0-9B) alternates
        // three Gated DeltaNet layers with one conventional attention layer.
        // Current GGUFs include their auxiliary NextN/MTP predictor layers in
        // block_count.  Those blocks have a different tensor layout and do not
        // participate in ordinary autoregressive decoding, so retain only the
        // backbone depth.  Older exports omit the key and remain unchanged.
        int nextn = (int)gguf_get_u32(g, AK("nextn_predict_layers"), 0);
        if (nextn < 0 || nextn >= m->n_layer) {
            fprintf(stderr, "error: invalid qwen35 NextN layer count %d for %d blocks\n",
                    nextn, m->n_layer);
            return false;
        }
        m->n_layer -= nextn;
        m->qwen35            = true;
        m->think_open        = "<think>";
        m->think_close       = "</think>";
        m->full_attn_interval = (int)gguf_get_u32(g, AK("full_attention_interval"), 4);
        m->ssm_conv_kernel   = (int)gguf_get_u32(g, AK("ssm.conv_kernel"), 0);
        m->ssm_inner         = (int)gguf_get_u32(g, AK("ssm.inner_size"), 0);
        m->ssm_state         = (int)gguf_get_u32(g, AK("ssm.state_size"), 0);
        m->ssm_v_heads       = (int)gguf_get_u32(g, AK("ssm.time_step_rank"), 0);
        m->ssm_groups        = (int)gguf_get_u32(g, AK("ssm.group_count"), 0);
        if (m->full_attn_interval <= 0 || m->ssm_conv_kernel <= 0 ||
            m->ssm_inner <= 0 || m->ssm_state <= 0 ||
            m->ssm_v_heads <= 0 || m->ssm_groups <= 0 ||
            m->ssm_inner % m->ssm_v_heads != 0 ||
            m->ssm_inner / m->ssm_v_heads != m->ssm_state ||
            m->ssm_v_heads % m->ssm_groups != 0) {
            fprintf(stderr, "error: invalid qwen35 Gated DeltaNet geometry\n");
            return false;
        }
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
        if (!m->l_is_swa || !m->l_head_kv || !m->l_head_dim || !m->l_rope_dim)
            return false;
        gguf_kv *swa_arr = gguf_get(g, AK("attention.sliding_window_pattern"));
        gguf_kv *kv_arr  = gguf_get(g, AK("attention.head_count_kv"));
        // per-layer arrays must have the element width we index with — a
        // converter that writes e.g. I32 booleans would misparse every layer
        if (swa_arr && swa_arr->arr_type != GGUF_T_BOOL &&
            swa_arr->arr_type != GGUF_T_U8 && swa_arr->arr_type != GGUF_T_I8)
            swa_arr = NULL;
        if (kv_arr && kv_arr->arr_type != GGUF_T_U32 && kv_arr->arr_type != GGUF_T_I32)
            kv_arr = NULL;
        for (int i = 0; i < m->n_layer; i++) {
            bool swa = swa_arr && swa_arr->arr_raw && (uint64_t)i < swa_arr->arr_n
                       ? ((const uint8_t *)swa_arr->arr_raw)[i] != 0 : false;
            m->l_is_swa[i]   = swa && m->swa_window > 0;
            m->l_head_dim[i] = swa ? hd_swa : m->head_dim;
            m->l_rope_dim[i] = swa ? m->rope_dim_local : m->rope_dim;
            m->l_head_kv[i]  = kv_arr && kv_arr->arr_raw && (uint64_t)i < kv_arr->arr_n
                               ? (int)((const uint32_t *)kv_arr->arr_raw)[i] : m->n_head_kv;
            // Per-layer kv-head / head-dim come from untrusted per-layer arrays
            // and feed the same attention divisors as the scalar path (kv_dim /
            // hd and n_head / n_head_kv). A zero or non-dividing value would
            // divide-by-zero on the first token — reject at load instead.
            if (m->l_head_dim[i] < 1 || m->l_head_kv[i] < 1 ||
                m->n_head < 1 || m->l_head_kv[i] > m->n_head ||
                m->n_head % m->l_head_kv[i] != 0) {
                fprintf(stderr, "error: invalid gemma4 per-layer geometry at "
                        "blk.%d (head_dim=%d head_count_kv=%d, n_head=%d)\n",
                        i, m->l_head_dim[i], m->l_head_kv[i], m->n_head);
                return false;
            }
        }
    }
    m->n_expert = (int)gguf_get_u32(g, AK("expert_count"), 0);
    if (m->n_expert > 0) {
        // sparse-MoE (Mixtral / Qwen3-MoE): softmax-over-all router, top-k
        // selection, renormalized weights, per-expert SwiGLU, weighted sum.
        m->n_expert_used = (int)gguf_get_u32(g, AK("expert_used_count"), 0);
        // Mixtral omits expert_feed_forward_length and uses feed_forward_length
        m->n_ff_exp = (int)gguf_get_u32(g, AK("expert_feed_forward_length"), m->n_ff);
        if (m->n_expert_used < 1 || m->n_expert_used > m->n_expert ||
            m->n_ff_exp <= 0 || m->n_expert > 256) {
            fprintf(stderr, "error: invalid MoE geometry (experts=%d used=%d ff_exp=%d)\n",
                    m->n_expert, m->n_expert_used, m->n_ff_exp);
            return false;
        }
        // Only the Mixtral/Qwen3 convention is implemented: softmax-over-all
        // gating, top-k, renormalized weights, no shared expert. A shared-expert
        // MoE (Qwen2-MoE / DeepSeek) would load its standard experts and
        // silently ignore the shared expert — refuse rather than compute wrong.
        if (gguf_get_u32(g, AK("expert_shared_count"), 0) > 0 ||
            gguf_find_tensor(g, "blk.0.ffn_gate_inp_shexp.weight")) {
            fprintf(stderr, "error: shared-expert MoE is not supported "
                    "(only Mixtral/Qwen3-style top-k MoE)\n");
            return false;
        }
        // GELU-gated MoE is implemented only for gemma-4's dual-branch layout
        // (fused gate_up experts + a dense shared FFN per layer). Any other
        // non-SiLU MoE is refused rather than run through SiLU math.
        bool gemma_moe = strcmp(arch, "gemma4") == 0 &&
                         gguf_find_tensor(g, "blk.0.ffn_gate_up_exps.weight");
        if (m->ffn_act != ACT_SILU && !gemma_moe) {
            fprintf(stderr, "error: MoE is only supported with SiLU-gated "
                    "experts (this model uses a different activation)\n");
            return false;
        }
    }
    if (!m->qwen35 && gguf_get(g, AK("ssm.conv_kernel"))) {
        fprintf(stderr, "error: '%s' is a hybrid SSM/attention architecture — "
                "only pure-transformer llama-family models are supported\n", arch);
        return false;
    }
    if (m->n_layer <= 0 || m->n_embd <= 0 || m->n_head <= 0 || m->n_ff <= 0) {
        fprintf(stderr, "error: missing model hyperparameters for arch '%s'\n", arch);
        return false;
    }
    // Everything above is only checked for presence (> 0). These fields come
    // straight from an untrusted GGUF and go on to drive allocation sizes, loop
    // bounds and divisors in the forward pass, so bound them here — before any
    // size math — against a generous ceiling (real models are far smaller) and
    // enforce the GQA relationships attention divides by. head_dim == 0 or
    // n_head_kv == 0 would divide-by-zero on the first token (kv_dim / hd and
    // n_head / n_head_kv); n_head_kv > n_head makes kv_mul == 0; an oversized
    // dim would overflow the byte products below. This gate closes those.
    #define MDL_DIM_MAX 1048576   /* 2^20 elements per axis */
    if (m->head_dim < 1        || m->head_dim > MDL_DIM_MAX ||
        m->n_embd   > MDL_DIM_MAX || m->n_ff  > MDL_DIM_MAX ||
        m->n_head   > MDL_DIM_MAX || m->n_layer > 100000 ||
        m->rope_dim < 0        || m->rope_dim > m->head_dim ||
        m->n_head_kv < 1       || m->n_head_kv > m->n_head ||
        m->n_head % m->n_head_kv != 0) {
        fprintf(stderr, "error: invalid model geometry for arch '%s' "
                "(head_dim=%d rope_dim=%d n_head=%d n_head_kv=%d "
                "n_embd=%d n_ff=%d n_layer=%d)\n",
                arch, m->head_dim, m->rope_dim, m->n_head, m->n_head_kv,
                m->n_embd, m->n_ff, m->n_layer);
        return false;
    }
    #undef MDL_DIM_MAX

    bool ok = true;
    m->tok_embd = need_tensor(g, "token_embd.weight", 0, &ok);
    if (!ok) return false;
    // The embedding table is [n_embd, n_vocab]; the forward pass dequantizes one
    // row of n_embd per token, so ne[0] must be exactly n_embd or every row
    // lands at the wrong offset. n_vocab is defined by this tensor and bounds
    // every token-id index below, so it must be present and sane.
    m->n_vocab = (int)m->tok_embd->ne[1];
    if ((int64_t)m->tok_embd->ne[0] != m->n_embd ||
        m->n_vocab < 1 || (int64_t)m->tok_embd->ne[1] > INT_MAX) {
        fprintf(stderr, "error: token_embd.weight shape [%llu,%llu] is "
                "inconsistent with n_embd=%d / n_vocab\n",
                (unsigned long long)m->tok_embd->ne[0],
                (unsigned long long)m->tok_embd->ne[1], m->n_embd);
        return false;
    }

    gguf_tensor *out_norm = need_tensor(g, "output_norm.weight", 0, &ok);
    if (!ok) return false;
    m->out_norm_w = tensor_to_f32(out_norm, &ok);
    if (!ok) return false;

    m->output = gguf_find_tensor(g, "output.weight");
    if (!m->output) m->output = m->tok_embd; // tied embeddings
    if (!ggml_type_supported(m->output->type)) {
        fprintf(stderr, "error: output tensor type %s unsupported\n", ggml_type_name(m->output->type));
        return false;
    }
    // The final projection reads n_vocab rows of n_embd from output.weight. A
    // distinct output tensor with fewer than n_vocab rows (or a wrong row
    // length) would be read past its mapped bytes at logit time. Tied
    // embeddings reuse tok_embd, already shape-checked above, so this only
    // needs to gate a separate output matrix.
    if (m->output != m->tok_embd &&
        !check_shape(m->output, m->n_embd, m->n_vocab, "output.weight", 0))
        return false;

    m->layers = calloc(m->n_layer, sizeof(layer_t));
    if (!m->layers) return false;
    // phi3 fuses Q/K/V into attn_qkv and gate/up into ffn_up: five slice
    // descriptors per layer, pointing into the mmapped weights
    bool fused_qkv = strcmp(arch, "phi3") == 0;
    if (fused_qkv) {
        m->fused_splits = calloc((size_t)m->n_layer * 5, sizeof(gguf_tensor));
        if (!m->fused_splits) return false;
    }
    for (int i = 0; i < m->n_layer; i++) {
        layer_t *l = &m->layers[i];
        gguf_tensor *an = need_tensor(g, "blk.%d.attn_norm.weight", i, &ok);
        gguf_tensor *fn = m->qwen35
            ? need_tensor(g, "blk.%d.post_attention_norm.weight", i, &ok)
            : need_tensor(g, "blk.%d.ffn_norm.weight", i, &ok);
        l->recurrent = m->qwen35 && ((i + 1) % m->full_attn_interval != 0);
        if (l->recurrent) {
            l->wqkv      = need_tensor(g, "blk.%d.attn_qkv.weight", i, &ok);
            l->wq_gate   = need_tensor(g, "blk.%d.attn_gate.weight", i, &ok);
            l->ssm_conv  = need_tensor(g, "blk.%d.ssm_conv1d.weight", i, &ok);
            l->ssm_beta  = need_tensor(g, "blk.%d.ssm_beta.weight", i, &ok);
            l->ssm_alpha = need_tensor(g, "blk.%d.ssm_alpha.weight", i, &ok);
            l->ssm_out   = need_tensor(g, "blk.%d.ssm_out.weight", i, &ok);
            // llama.cpp-era Ornith exports used `ssm_dt`; current Qwen3.5
            // GGUFs use `ssm_dt.bias`. They carry the same per-head vector.
            gguf_tensor *dt = opt_tensor(g, "blk.%d.ssm_dt.bias", i);
            if (!dt) dt = need_tensor(g, "blk.%d.ssm_dt", i, &ok);
            l->ssm_dt    = tensor_to_f32(dt, &ok);
            l->ssm_a     = tensor_to_f32(need_tensor(g, "blk.%d.ssm_a", i, &ok), &ok);
            l->ssm_norm_w = tensor_to_f32(need_tensor(g, "blk.%d.ssm_norm.weight", i, &ok), &ok);
            l->w_gate = need_tensor(g, "blk.%d.ffn_gate.weight", i, &ok);
            l->w_up   = need_tensor(g, "blk.%d.ffn_up.weight", i, &ok);
            l->w_down = need_tensor(g, "blk.%d.ffn_down.weight", i, &ok);
            if (!ok) return false;
            l->attn_norm_w = tensor_to_f32(an, &ok);
            l->ffn_norm_w = tensor_to_f32(fn, &ok);
            if (!ok) return false;
            l->out_scale = 1.0f;
            continue;
        }
        if (fused_qkv) {
            // phi3: [Q rows | K rows | V rows] in attn_qkv, and the FFN's
            // gate and up halves stacked in ffn_up (HF's gate_up_proj)
            gguf_tensor *qkv = need_tensor(g, "blk.%d.attn_qkv.weight", i, &ok);
            gguf_tensor *gu  = need_tensor(g, "blk.%d.ffn_up.weight", i, &ok);
            if (!ok) return false;
            int64_t q_rows = (int64_t)m->n_head * m->head_dim;
            int64_t kv_rows = (int64_t)m->n_head_kv * m->head_dim;
            // Both fused tensors project from n_embd, so ne[0] must be n_embd;
            // the row counts must match the slice boundaries exactly (a short
            // tensor would let the slices below point past the mapped bytes)
            // and the gate/up half must be n_ff wide.
            if ((int64_t)qkv->ne[0] != m->n_embd || (int64_t)gu->ne[0] != m->n_embd ||
                (int64_t)qkv->ne[1] != q_rows + 2 * kv_rows ||
                (int64_t)gu->ne[1] != 2 * (int64_t)m->n_ff) {
                fprintf(stderr, "error: unexpected fused tensor shape in blk.%d\n", i);
                return false;
            }
            gguf_tensor *sl = &m->fused_splits[i * 5];
            l->wq     = slice_rows(qkv, &sl[0], 0, q_rows);
            l->wk     = slice_rows(qkv, &sl[1], q_rows, kv_rows);
            l->wv     = slice_rows(qkv, &sl[2], q_rows + kv_rows, kv_rows);
            l->w_gate = slice_rows(gu,  &sl[3], 0, (int64_t)gu->ne[1] / 2);
            l->w_up   = slice_rows(gu,  &sl[4], (int64_t)gu->ne[1] / 2,
                                   (int64_t)gu->ne[1] / 2);
        } else {
            l->wq     = need_tensor(g, "blk.%d.attn_q.weight", i, &ok);
            l->wk     = need_tensor(g, "blk.%d.attn_k.weight", i, &ok);
            l->wv     = m->v_rmsnorm ? opt_tensor(g, "blk.%d.attn_v.weight", i)
                                     : need_tensor(g, "blk.%d.attn_v.weight", i, &ok);
            if (m->n_expert == 0) {
                l->w_gate = need_tensor(g, "blk.%d.ffn_gate.weight", i, &ok);
                l->w_up   = need_tensor(g, "blk.%d.ffn_up.weight", i, &ok);
            }
        }
        l->wo     = need_tensor(g, "blk.%d.attn_output.weight", i, &ok);
        if (m->n_expert > 0) {
            // sparse-MoE FFN: a router plus fused 3D expert tensors replace the
            // dense gate/up/down for this layer
            l->is_moe = true;
            l->ffn_gate_inp  = need_tensor(g, "blk.%d.ffn_gate_inp.weight", i, &ok);
            l->ffn_gate_up_exps = opt_tensor(g, "blk.%d.ffn_gate_up_exps.weight", i);
            if (l->ffn_gate_up_exps) {
                // gemma-4 dual-branch MoE: gate+up fused in one 3D tensor, a
                // per-expert down scale, and — every MoE layer ALSO runs a dense
                // GELU FFN as a shared expert, plus two branch sandwich norms.
                l->moe_gemma = true;
                m->moe_gemma = true;   // dual-branch: dense shared FFN + routed experts
                l->ffn_down_exps   = need_tensor(g, "blk.%d.ffn_down_exps.weight", i, &ok);
                l->down_exps_scale = tensor_to_f32(opt_tensor(g, "blk.%d.ffn_down_exps.scale", i), &ok);
                l->gate_inp_scale  = tensor_to_f32(opt_tensor(g, "blk.%d.ffn_gate_inp.scale", i), &ok);
                l->w_gate = need_tensor(g, "blk.%d.ffn_gate.weight", i, &ok);
                l->w_up   = need_tensor(g, "blk.%d.ffn_up.weight",   i, &ok);
                l->w_down = need_tensor(g, "blk.%d.ffn_down.weight", i, &ok);
                l->ffn_post_norm1_w = tensor_to_f32(opt_tensor(g, "blk.%d.post_ffw_norm_1.weight", i), &ok);
                l->ffn_pre_norm2_w  = tensor_to_f32(opt_tensor(g, "blk.%d.pre_ffw_norm_2.weight",  i), &ok);
                l->ffn_post_norm2_w = tensor_to_f32(opt_tensor(g, "blk.%d.post_ffw_norm_2.weight", i), &ok);
            } else if ((l->ffn_gate_exps = opt_tensor(g, "blk.%d.ffn_gate_exps.weight", i))) {
                // modern fused 3D expert tensors
                l->ffn_up_exps   = need_tensor(g, "blk.%d.ffn_up_exps.weight", i, &ok);
                l->ffn_down_exps = need_tensor(g, "blk.%d.ffn_down_exps.weight", i, &ok);
            } else {
                // legacy split layout: one 2D tensor per expert (older Mixtral)
                l->moe_split = true;
                l->moe_g = calloc((size_t)m->n_expert, sizeof(gguf_tensor *));
                l->moe_u = calloc((size_t)m->n_expert, sizeof(gguf_tensor *));
                l->moe_d = calloc((size_t)m->n_expert, sizeof(gguf_tensor *));
                if (!l->moe_g || !l->moe_u || !l->moe_d) return false;
                for (int e = 0; e < m->n_expert; e++) {
                    char nm[128];
                    snprintf(nm, sizeof(nm), "blk.%d.ffn_gate.%d.weight", i, e);
                    l->moe_g[e] = gguf_find_tensor(g, nm);
                    snprintf(nm, sizeof(nm), "blk.%d.ffn_up.%d.weight", i, e);
                    l->moe_u[e] = gguf_find_tensor(g, nm);
                    snprintf(nm, sizeof(nm), "blk.%d.ffn_down.%d.weight", i, e);
                    l->moe_d[e] = gguf_find_tensor(g, nm);
                    if (!l->moe_g[e] || !l->moe_u[e] || !l->moe_d[e]) {
                        fprintf(stderr, "error: missing MoE expert tensor "
                                "(neither fused ffn_gate_exps nor split "
                                "ffn_gate.%d) in blk.%d\n", e, i);
                        return false;
                    }
                }
            }
        } else {
            l->w_down = need_tensor(g, "blk.%d.ffn_down.weight", i, &ok);
        }
        if (!ok) return false;
        // Validate the projection weights against the geometry the forward pass
        // will drive them with, so a tensor smaller than the declared shape is
        // rejected here rather than read past at decode time. attn_output and
        // ffn_down share their layout across every dense variant; Q/K/V and
        // gate/up are fused for phi3 (validated above) or doubled for qwen35,
        // so shape-check those only for the plain llama-family layout.
        {
            int qd = model_q_dim(m, i);    // n_head * head_dim (per layer)
            int kd = model_kv_dim(m, i);   // n_head_kv * head_dim (per layer)
            if (!check_shape(l->wo, qd, m->n_embd, "attn_output", i))
                return false;
            if (!fused_qkv && !m->qwen35) {
                if (!check_shape(l->wq, m->n_embd, qd, "attn_q", i) ||
                    !check_shape(l->wk, m->n_embd, kd, "attn_k", i) ||
                    !check_shape(l->wv, m->n_embd, kd, "attn_v", i))
                    return false;
                if (!l->is_moe &&
                    (!check_shape(l->w_gate, m->n_embd, m->n_ff, "ffn_gate", i) ||
                     !check_shape(l->w_up,   m->n_embd, m->n_ff, "ffn_up",   i)))
                    return false;
            }
            if (!l->is_moe &&
                !check_shape(l->w_down, m->n_ff, m->n_embd, "ffn_down", i))
                return false;
            if (l->is_moe) {
                // The MoE forward path slices each expert from the fused 3D
                // tensors (or the array of 2D split tensors) using offsets
                // computed from n_embd/n_ff_exp/n_expert metadata, NOT the
                // tensors' own ne[]. Validate that geometry here so the offsets
                // stay in bounds (RNC-1). The router reads n_expert rows of
                // length n_embd; each expert's gate/up is {n_embd, n_ff_exp}
                // and down is {n_ff_exp, n_embd}.
                if (!check_shape(l->ffn_gate_inp, m->n_embd, m->n_expert,
                                 "ffn_gate_inp", i))
                    return false;
                if (l->moe_gemma) {
                    // gate and up are fused: {n_embd, 2*n_ff_exp, n_expert}
                    if (!check_shape3(l->ffn_gate_up_exps, m->n_embd,
                                      2 * m->n_ff_exp, m->n_expert,
                                      "ffn_gate_up_exps", i) ||
                        !check_shape3(l->ffn_down_exps, m->n_ff_exp, m->n_embd,
                                      m->n_expert, "ffn_down_exps", i))
                        return false;
                } else if (l->moe_split) {
                    for (int e = 0; e < m->n_expert; e++) {
                        if (!check_shape(l->moe_g[e], m->n_embd, m->n_ff_exp,
                                         "ffn_gate.<e>", i) ||
                            !check_shape(l->moe_u[e], m->n_embd, m->n_ff_exp,
                                         "ffn_up.<e>", i) ||
                            !check_shape(l->moe_d[e], m->n_ff_exp, m->n_embd,
                                         "ffn_down.<e>", i))
                            return false;
                    }
                } else {
                    if (!check_shape3(l->ffn_gate_exps, m->n_embd, m->n_ff_exp,
                                      m->n_expert, "ffn_gate_exps", i) ||
                        !check_shape3(l->ffn_up_exps, m->n_embd, m->n_ff_exp,
                                      m->n_expert, "ffn_up_exps", i) ||
                        !check_shape3(l->ffn_down_exps, m->n_ff_exp, m->n_embd,
                                      m->n_expert, "ffn_down_exps", i))
                        return false;
                }
            }
        }
        l->attn_norm_w = tensor_to_f32(an, &ok);
        l->ffn_norm_w  = tensor_to_f32(fn, &ok);
        l->bq = tensor_to_f32(opt_tensor(g, "blk.%d.attn_q.bias", i), &ok);
        l->bk = tensor_to_f32(opt_tensor(g, "blk.%d.attn_k.bias", i), &ok);
        l->bv = tensor_to_f32(opt_tensor(g, "blk.%d.attn_v.bias", i), &ok);
        l->bo = tensor_to_f32(opt_tensor(g, "blk.%d.attn_output.bias", i), &ok);
        l->qnorm_w = tensor_to_f32(opt_tensor(g, "blk.%d.attn_q_norm.weight", i), &ok);
        l->knorm_w = tensor_to_f32(opt_tensor(g, "blk.%d.attn_k_norm.weight", i), &ok);
        // Qwen3.5's post_attention_norm is the FFN input norm (loaded as
        // ffn_norm_w above), not a sandwich norm on the attention projection.
        l->post_attn_norm_w = m->qwen35 ? NULL
            : tensor_to_f32(opt_tensor(g, "blk.%d.post_attention_norm.weight", i), &ok);
        l->post_ffn_norm_w  = tensor_to_f32(opt_tensor(g, "blk.%d.post_ffw_norm.weight", i), &ok);
        l->out_scale = 1.0f;
        gguf_tensor *osc = opt_tensor(g, "blk.%d.layer_output_scale.weight", i);
        if (osc && osc->type == T_F32 && osc->ne[0] >= 1)
            l->out_scale = ((const float *)osc->data)[0];
        if (!ok) return false;  // a norm/bias materialization OOMed this layer
    }

    // gemma4 dropped the plus-one norm convention (unlike gemma1-3): its
    // RMSNorm is the standard x_normed * weight, so norm weights are used raw.

    // checkpoint workaround (gemma4 ships one): ids the model must never emit;
    // their logits are forced to -inf after every forward pass
    gguf_kv *sup = gguf_get(g, "tokenizer.ggml.suppress_tokens");
    if (sup && sup->arr_raw && sup->arr_type == GGUF_T_I32 && sup->arr_n > 0) {
        const int32_t *ids = (const int32_t *)sup->arr_raw;
        // A failed suppress list is not optional to skip: these ids are tokens
        // the checkpoint must never emit, so run without it would be silently
        // wrong. Fail the load instead.
        m->suppress = malloc(sizeof(int32_t) * sup->arr_n);
        if (!m->suppress) return false;
        for (uint64_t i = 0; i < sup->arr_n; i++)
            if (ids[i] >= 0 && ids[i] < m->n_vocab)
                m->suppress[m->n_suppress++] = ids[i];
    }

    // KV storage format. q8_0 halves the cache again and is supported by both
    // the CPU path and the CUDA kernels, so it is decided here — before the
    // reservation auto-fit — and the sizing below uses the real per-token cost.
    // Requires per-HEAD block alignment, not just per-row: attention slices the
    // row at kvh*head_dim, which must land on a q8 block boundary.
    if (p->kv_q8) {
        bool aligned = true;
        for (int l = 0; l < m->n_layer; l++)
            if (model_head_dim(m, l) % 32 != 0) aligned = false;
        // a GPU backend without q8 attention kernels would read the blocks as
        // fp16, so fall back rather than corrupt
        char gname[128];
        bool gpu_path = p->gpu_mode == GPU_AUTO && gpu_available(gname, sizeof(gname));
        if (!aligned)
            fprintf(stderr, "kv: head_dim not a multiple of 32 — keeping f16\n");
        else if (gpu_path && !gpu_kv_q8_ok())
            fprintf(stderr, "kv: this GPU backend has no q8_0 attention kernels "
                            "— keeping f16 (use --gpu off for a q8 cache)\n");
        else
            m->kv_q8 = true;
    }

    // runtime buffers
    m->reserve_vram_pct = p->reserve_vram_pct;
    m->gpu_layers_override = p->gpu_layers_override;
    m->cpu_moe = p->cpu_moe && m->n_expert > 0;
    int n_ctx = p->n_ctx;
    if (n_ctx <= 0 && (p->reserve_vram_pct > 0 || p->reserve_ram_pct > 0)) {
        // reservation auto-fit: size the context to fill whatever the
        // reservation leaves after the weights, so small models grow their
        // window into the reserved room instead of idling at the default
        size_t kv_per_tok = 0;
        for (int l = 0; l < m->n_layer; l++) {
            int d = model_kv_dim(m, l);
            kv_per_tok += 2ull * (m->kv_q8 ? (size_t)(d / 32) * 34
                                           : (size_t)d * sizeof(f16_t));
        }
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
                long long room = (long long)budget -
                                 (long long)model_cuda_weight_estimate(m, p) -
                                 (long long)head;
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
    m->xdim    = xdim;   // cached so the forward paths reuse it (RNC-3)
    int B      = m->n_batch;

    // per-layer element offsets into the (possibly heterogeneous) KV cache
    m->kv_off = malloc(sizeof(size_t) * (m->n_layer + 1));
    if (!m->kv_off) return false;
    m->kv_off[0] = 0;
    for (int l = 0; l < m->n_layer; l++)
        m->kv_off[l + 1] = m->kv_off[l] + (size_t)n_ctx * model_kv_dim(m, l);
    size_t kv_bytes = model_kv_byte_off(m, m->n_layer);
    m->kcache = calloc(1, kv_bytes);
    m->vcache = calloc(1, kv_bytes);
    m->kv_owner = KV_OWNER_MALLOC;
    m->x      = malloc(sizeof(float) * (size_t)B * m->n_embd);
    m->xb     = malloc(sizeof(float) * (size_t)B * xdim);
    m->xb2    = malloc(sizeof(float) * (size_t)B * xdim);
    m->q      = malloc(sizeof(float) * (size_t)B * q_dim);
    m->k_tmp  = malloc(sizeof(float) * (size_t)B * kv_dim);
    m->v_tmp  = malloc(sizeof(float) * (size_t)B * kv_dim);
    if (m->qwen35) {
        int conv_dim = 2 * m->ssm_state * m->ssm_groups + m->ssm_inner;
        int hv = m->ssm_inner / m->ssm_v_heads;
        m->q_gate = malloc(sizeof(float) * (size_t)B * q_dim);
        // ssm_qkv is shared by two uses: recurrent layers write conv_dim floats
        // per token, but full-attention layers write the fused Q/gate with stride
        // 2*q_dim (model.c qwen35 attention path). Nothing relates conv_dim to
        // 2*q_dim, so size for the larger of the two or the attention write
        // overflows the buffer.
        int qkv_dim = conv_dim > 2 * q_dim ? conv_dim : 2 * q_dim;
        m->ssm_qkv = malloc(sizeof(float) * (size_t)B * qkv_dim);
        m->ssm_z = malloc(sizeof(float) * (size_t)B * m->ssm_inner);
        m->ssm_aux = malloc(sizeof(float) * (size_t)B *
                            (conv_dim + m->ssm_v_heads));
        m->ssm_conv_state = calloc((size_t)m->n_layer *
                                   (m->ssm_conv_kernel - 1) * conv_dim,
                                   sizeof(float));
        m->ssm_state_mem = calloc((size_t)m->n_layer * m->ssm_v_heads *
                                  hv * hv, sizeof(float));
        // one dequantized conv-kernel row, reused every conv step (the forward
        // pass is single-threaded here, so one buffer suffices) — preallocated
        // so the hot path never mallocs and an OOM fails the load, not a token
        m->ssm_cw = malloc(sizeof(float) * m->ssm_conv_kernel);
    }
    // hb/hb2 hold the dense FFN's n_ff-wide output for B token columns; gemma-4
    // MoE also reuses hb to hold one token's fused gate_up of width 2*n_ff_exp,
    // so the buffer must cover both — otherwise --batch 1 with 2*n_ff_exp > n_ff
    // overflows the heap.
    size_t ff_scratch = (size_t)B * m->n_ff;
    if (m->moe_gemma && 2 * (size_t)m->n_ff_exp > ff_scratch)
        ff_scratch = 2 * (size_t)m->n_ff_exp;
    m->hb     = malloc(sizeof(float) * ff_scratch);
    m->hb2    = malloc(sizeof(float) * ff_scratch);
    m->att    = malloc(sizeof(float) * (size_t)m->n_head * n_ctx);
    m->logits = malloc(sizeof(float) * m->n_vocab);
    m->spec_batch = 16; // all_logits rows, allocated lazily on first use
    if (m->n_expert > 0) {
        // per-token MoE scratch (decode routes each token independently)
        m->moe_logits = malloc(sizeof(float) * (size_t)m->n_expert);
        m->moe_gate   = malloc(sizeof(float) * (size_t)m->n_ff_exp);
        m->moe_up     = malloc(sizeof(float) * (size_t)m->n_ff_exp);
        m->moe_dexp   = malloc(sizeof(float) * (size_t)m->n_embd);
        m->moe_out    = malloc(sizeof(float) * (size_t)m->n_embd);
        // grouped-by-expert prefill scratch (a whole batch at once)
        size_t nb = (size_t)m->n_batch, ne = (size_t)m->n_embd;
        size_t nf = (size_t)m->n_ff_exp, nu = (size_t)m->n_expert_used;
        m->moe_out_b  = malloc(sizeof(float) * nb * ne);
        m->moe_gath   = malloc(sizeof(float) * nb * ne);
        m->moe_gate_b = malloc(sizeof(float) * nb * nf);
        m->moe_up_b   = malloc(sizeof(float) * nb * nf);
        m->moe_dexp_b = malloc(sizeof(float) * nb * ne);
        m->moe_sel    = malloc(sizeof(int)   * nb * nu);
        m->moe_selw   = malloc(sizeof(float) * nb * nu);
        m->moe_gidx   = malloc(sizeof(int)   * nb);
        m->moe_gw     = malloc(sizeof(float) * nb);
    }
    if (!m->kv_off || !m->kcache || !m->vcache || !m->x || !m->xb ||
        !m->xb2 || !m->q || !m->k_tmp || !m->v_tmp || !m->hb ||
        !m->hb2 || !m->att || !m->logits ||
        (m->n_expert > 0 && (!m->moe_logits || !m->moe_gate || !m->moe_up ||
                             !m->moe_dexp || !m->moe_out || !m->moe_out_b ||
                             !m->moe_gath || !m->moe_gate_b || !m->moe_up_b ||
                             !m->moe_dexp_b || !m->moe_sel || !m->moe_selw ||
                             !m->moe_gidx || !m->moe_gw)) ||
        (m->qwen35 && (!m->q_gate || !m->ssm_qkv || !m->ssm_z ||
                       !m->ssm_aux || !m->ssm_conv_state ||
                       !m->ssm_state_mem || !m->ssm_cw))) {
        fprintf(stderr, "error: cannot allocate buffers (ctx %d needs %.1f MB KV cache)\n",
                n_ctx, 2.0 * kv_bytes / 1e6);
        return false; // model_load unwinds the partial allocation before returning
    }

    int pool_threads = p->n_threads > 0 ? p->n_threads : 1;
    if (m->qwen35 && p->cpu_fallback_threads > pool_threads) {
        // This architecture decodes on the CPU alone (no recurrent GPU
        // kernels, below); the defaulted thread count is sized for a
        // GPU-fed workload and starves it — measured on a 128-cpu host,
        // 8 threads gave 1.7 tok/s where 64 gave 5.1.
        pool_threads = p->cpu_fallback_threads;
    }
    m->tp = tpool_create(pool_threads);
    if (!m->tp) {
        fprintf(stderr, "error: cannot create thread pool\n");
        return false;
    }
    if (!rope_setup(m, g, arch, p->rope_base, p->rope_scale)) return false;

    if (p->gpu_mode == GPU_AUTO) {
        // Register the intended VRAM footprint before allocating any of it, so
        // a concurrent runner sees this claim rather than discovering it as a
        // mysteriously shrunken free figure. CPU-only runs never get here, so
        // they are never accounted and never refused.
        if (!model_vram_claim(m, p, kv_bytes)) return false;
        size_t vfree_before = 0, vtotal_before = 0;
        gpu_mem_info(&vfree_before, &vtotal_before);
        gpu_init(m);                        // sets m->gpu on success
        model_vram_commit(m, vfree_before);
    }

    if (p->verbose) {
        fprintf(stderr, "%-24s %s\n", "architecture", m->arch);
        fprintf(stderr, "%-24s %d\n", "layers", m->n_layer);
        fprintf(stderr, "%-24s %d\n", "embedding dim", m->n_embd);
        fprintf(stderr, "%-24s %d (%d kv)\n", "heads", m->n_head, m->n_head_kv);
        fprintf(stderr, "%-24s %d\n", "head dim", m->head_dim);
        fprintf(stderr, "%-24s %d\n", "ffn dim", m->n_ff);
        fprintf(stderr, "%-24s %d\n", "vocab", m->n_vocab);
        fprintf(stderr, "%-24s %d (train %d)\n", "context", m->n_ctx, m->n_ctx_train);
        fprintf(stderr, "%-24s %.1f MB (%s)\n", "kv cache", 2.0 * kv_bytes / 1e6,
                m->kv_q8 ? "q8_0" : "fp16");
        fprintf(stderr, "%-24s %d\n", "batch", m->n_batch);
        gguf_tensor *w0 = m->layers[0].recurrent ? m->layers[0].wqkv
                                                  : m->layers[0].wq;
        fprintf(stderr, "%-24s %s\n", "weight type", ggml_type_name(w0->type));
        fprintf(stderr, "%-24s %.1f\n", "rope base", m->rope_base);
    }
    return true;
}

void model_free(model_t *m) {
    gpu_free(m); // nulls kcache/vcache if the GPU owned them
    // Deregister on the clean path. The unclean paths (SIGKILL, crash) are
    // covered by dead-pid reaping in the next runner's claim, which is how the
    // orphans that motivated the registry would have been cleared.
    vram_release(m->vram);
    m->vram = NULL;
    // partial load: n_layer is read from GGUF metadata long before m->layers
    // is allocated, so a load that fails in between (unsupported tensor
    // type, missing token_embd/output_norm, etc.) reaches here with
    // m->layers still NULL — guard against dereferencing it.
    for (int i = 0; m->layers && i < m->n_layer; i++) {
        layer_t *l = &m->layers[i];
        free(l->attn_norm_w); free(l->ffn_norm_w);
        free(l->bq); free(l->bk); free(l->bv); free(l->bo);
        free(l->qnorm_w); free(l->knorm_w);
        free(l->post_attn_norm_w); free(l->post_ffn_norm_w);
        free(l->down_exps_scale); free(l->gate_inp_scale);
        free(l->ffn_pre_norm2_w); free(l->ffn_post_norm1_w); free(l->ffn_post_norm2_w);
        free(l->ssm_dt); free(l->ssm_a); free(l->ssm_norm_w);
        free(l->moe_g); free(l->moe_u); free(l->moe_d);  // split-MoE pointer arrays
    }
    free(m->l_head_kv); free(m->l_head_dim); free(m->l_rope_dim);
    free(m->l_is_swa); free(m->kv_off); free(m->suppress);
    free(m->layers);
    free(m->path);
    free(m->fused_splits);
    free(m->out_norm_w);
    free(m->rope_inv_freq);
    free(m->rope_inv_freq_local);
    if (m->kv_owner == KV_OWNER_MALLOC) {
        free(m->kcache);
        free(m->vcache);
    }
    m->kcache = NULL;
    m->vcache = NULL;
    m->kv_owner = KV_OWNER_MALLOC;
    free(m->x); free(m->xb); free(m->xb2); free(m->q);
    free(m->k_tmp); free(m->v_tmp);
    free(m->q_gate); free(m->ssm_qkv); free(m->ssm_z); free(m->ssm_aux);
    free(m->ssm_cw);
    free(m->ssm_conv_state); free(m->ssm_state_mem);
    free(m->hb); free(m->hb2); free(m->att); free(m->logits); free(m->all_logits);
    free(m->moe_logits); free(m->moe_gate); free(m->moe_up);
    free(m->moe_dexp); free(m->moe_out);
    free(m->moe_out_b); free(m->moe_gath); free(m->moe_gate_b);
    free(m->moe_up_b); free(m->moe_dexp_b); free(m->moe_sel);
    free(m->moe_selw); free(m->moe_gidx); free(m->moe_gw);
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
    // batched: dequantize each weight row once, reuse for every token; the
    // multi-column dot shares each weight load across 4 activation columns
    float *buf = type == T_F32 ? NULL : malloc(sizeof(float) * n_in);
    // If that scratch could not be allocated (OOM mid-inference), fall back to
    // the buffer-free fused dequant-and-dot used by the n_batch==1 path — one
    // column at a time. Slower, but never a NULL write and never silently
    // wrong output. This is the only in-band recovery a void thread-pool
    // worker has.
    bool no_scratch = type != T_F32 && !buf;
    float outs[64];
    for (int r = i0; r < i1; r++) {
        float b0 = j->bias ? j->bias[r] : 0.0f;
        if (no_scratch) {
            for (int c = 0; c < j->n_batch; c++) {
                float v = vec_dot(type, base + (size_t)r * j->rsz,
                                  j->x + (size_t)c * j->x_stride, n_in);
                j->y[(size_t)c * j->y_stride + r] = v + b0;
            }
            continue;
        }
        const float *wrow;
        if (type == T_F32) {
            wrow = (const float *)(base + (size_t)r * j->rsz);
        } else {
            dequant_row(type, base + (size_t)r * j->rsz, buf, n_in);
            wrow = buf;
        }
        for (int c = 0; c < j->n_batch; c += 64) {
            int nb = j->n_batch - c < 64 ? j->n_batch - c : 64;
            vec_dot_f32_multi(wrow, j->x + (size_t)c * j->x_stride,
                              j->x_stride, nb, n_in, outs);
            for (int b = 0; b < nb; b++)
                j->y[(size_t)(c + b) * j->y_stride + r] = outs[b] + b0;
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
    const uint8_t *kc, *vc; // this layer's cache (f16 rows or q8_0 blocks)
    const float *q;         // this token's query [q_dim]
    float *out;             // attention output [q_dim]
    int pos;
    int t0;                 // first attended position (sliding window)
    int hd, kv_dim;         // this layer's head dim / kv row width
    size_t row_b;           // bytes per cached row
    bool q8;                // rows are q8_0 blocks
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
        size_t hoff = j->q8 ? (size_t)(kvh * hd / 32) * 34
                            : (size_t)kvh * hd * sizeof(f16_t);
        for (int t = j->t0; t <= j->pos; t++) {
            const uint8_t *kt = j->kc + (size_t)t * j->row_b + hoff;
            float s;
            if (j->q8) {
                s = vec_dot(T_Q8_0, kt, qh, hd);
            } else {
                const f16_t *kh = (const f16_t *)kt;
                s = 0;
                for (int i = 0; i < hd; i++) s += qh[i] * f16_load(kh + i);
            }
            att[t] = s * scale;
        }
        softmax(att + j->t0, j->pos + 1 - j->t0);
        float *out = j->out + h * hd;
        memset(out, 0, sizeof(float) * hd);
        for (int t = j->t0; t <= j->pos; t++) {
            const uint8_t *vt = j->vc + (size_t)t * j->row_b + hoff;
            float a = att[t];
            if (j->q8) {
                q8_accum_row(vt, a, out, hd);
            } else {
                const f16_t *vh = (const f16_t *)vt;
                for (int i = 0; i < hd; i++) out[i] += a * f16_load(vh + i);
            }
        }
    }
}

// ------------------------------------------------- activation tracing (debug)
// RUNNER_DEBUG_ACT=1 dumps per-layer activation statistics for the first
// forward pass to stderr. Off by default; zero cost when unset (one cached
// getenv + a predictable branch per dumped tensor).

static int dbg_act_mode(void) {
    static int mode = -1;
    if (mode < 0) {
        const char *e = getenv("RUNNER_DEBUG_ACT");
        mode = e && *e && strcmp(e, "0") != 0 ? atoi(e) : 0;
        if (mode == 0 && e && *e && strcmp(e, "0") != 0) mode = 1;
    }
    return mode;
}
static int dbg_act_pass = 0; // forward passes seen so far

static void dbg_stat(const char *tag, int layer, const float *v, size_t n) {
    float mn = INFINITY, mx = -INFINITY, absmx = 0;
    double sum = 0;
    size_t n_inf = 0, n_nan = 0, n_zero = 0;
    for (size_t i = 0; i < n; i++) {
        float x = v[i];
        if (x != x) { n_nan++; continue; }
        if (x > 3.0e38f || x < -3.0e38f) { n_inf++; continue; }
        if (x == 0.0f) n_zero++;
        if (x < mn) mn = x;
        if (x > mx) mx = x;
        float a = x < 0 ? -x : x;
        if (a > absmx) absmx = a;
        sum += x;
    }
    size_t good = n - n_inf - n_nan;
    fprintf(stderr, "ACT L%-3d %-16s n=%-7zu min=%+.4g max=%+.4g mean=%+.4g absmax=%.4g inf=%zu nan=%zu zero=%zu\n",
            layer, tag, n, good ? mn : 0.0f, good ? mx : 0.0f,
            good ? sum / (double)good : 0.0, absmx, n_inf, n_nan, n_zero);
}

// same, over an f16 buffer already written to the KV cache
static void dbg_stat_f16(const char *tag, int layer, const f16_t *v, size_t n) {
    float absmx = 0;
    size_t n_inf = 0, n_nan = 0;
    for (size_t i = 0; i < n; i++) {
        float x = f16_load(v + i);
        if (x != x) { n_nan++; continue; }
        if (x > 3.0e38f || x < -3.0e38f) { n_inf++; continue; }
        float a = x < 0 ? -x : x;
        if (a > absmx) absmx = a;
    }
    fprintf(stderr, "ACT L%-3d %-16s n=%-7zu absmax=%.4g inf=%zu nan=%zu (f16 cache)\n",
            layer, tag, n, absmx, n_inf, n_nan);
}

// ---------------------------------------------------------------- forward

// suppress_tokens checkpoint workaround: a large finite constant instead of
// -INFINITY because the binary is built with -ffast-math (finite-math-only)
static void suppress_logits(const model_t *m, float *logits) {
    for (int i = 0; i < m->n_suppress; i++)
        logits[m->suppress[i]] = -1e30f;
}

// The output-head transforms every forward path applies to its logits, in one
// place so a batched step and a solo step cannot drift apart (RNC-2): the
// final-logit softcap (Gemma-style tanh squashing) then suppression of the
// never-emit tokens. No-op on a NULL buffer (a want_logits==false step).
static void apply_head_transforms(const model_t *m, float *logits) {
    if (!logits) return;
    if (m->logit_softcap > 0)
        for (int i = 0; i < m->n_vocab; i++)
            logits[i] = m->logit_softcap * tanhf(logits[i] / m->logit_softcap);
    if (m->n_suppress) suppress_logits(m, logits);
}

// One CPU Gated DeltaNet layer for Qwen3.5. State is stored transposed
// ([value_column][key_row]), matching the reference operator: decay, delta
// correction, outer-product update, then query readout.
static void qwen35_linear(model_t *m, layer_t *ly, int layer, int n, int xdim) {
    int sk = m->ssm_state, ng = m->ssm_groups, nh = m->ssm_v_heads;
    int inner = m->ssm_inner, hv = inner / nh;
    int keydim = sk * ng, convdim = 2 * keydim + inner;
    int histn = m->ssm_conv_kernel - 1;

    matvec_b(m->tp, m->ssm_qkv, convdim, ly->wqkv,
             m->xb, xdim, m->n_embd, convdim, NULL, n);
    matvec_b(m->tp, m->ssm_z, inner, ly->wq_gate,
             m->xb, xdim, m->n_embd, inner, NULL, n);
    // beta and alpha are small per-head projections.
    matvec_b(m->tp, m->q_gate, nh, ly->ssm_beta,
             m->xb, xdim, m->n_embd, nh, NULL, n);
    float *alphas = m->ssm_aux + (size_t)n * convdim;
    matvec_b(m->tp, alphas, nh, ly->ssm_alpha,
             m->xb, xdim, m->n_embd, nh, NULL, n);

    float *hist = m->ssm_conv_state + (size_t)layer * histn * convdim;
    float *states = m->ssm_state_mem + (size_t)layer * nh * hv * hv;
    size_t wrs = ggml_row_size(ly->ssm_conv->type, m->ssm_conv_kernel);
    float *cw = m->ssm_cw;   // preallocated at load
    for (int b = 0; b < n; b++) {
        float *mix = m->ssm_qkv + (size_t)b * convdim;
        // Causal depthwise convolution over the persistent history and input.
        for (int c = 0; c < convdim; c++) {
            dequant_row(ly->ssm_conv->type,
                        (const uint8_t *)ly->ssm_conv->data + (size_t)c * wrs,
                        cw, m->ssm_conv_kernel);
            float sum = cw[histn] * mix[c];
            for (int k = 0; k < histn; k++)
                sum += cw[k] * hist[(size_t)k * convdim + c];
            m->ssm_aux[(size_t)b * convdim + c] =
                sum / (1.0f + expf(-sum));
        }
        if (histn) {
            memmove(hist, hist + convdim,
                    sizeof(float) * (size_t)(histn - 1) * convdim);
            memcpy(hist + (size_t)(histn - 1) * convdim, mix,
                   sizeof(float) * convdim);
        }
        float *cv = m->ssm_aux + (size_t)b * convdim;
        // Q and K use L2 norm (not RMS norm).
        for (int g = 0; g < ng; g++) {
            float *q = cv + g * sk;
            float *k = cv + keydim + g * sk;
            float qs = m->rms_eps, ks = m->rms_eps;
            for (int j = 0; j < sk; j++) {
                qs += q[j] * q[j]; ks += k[j] * k[j];
            }
            qs = 1.0f / sqrtf(qs); ks = 1.0f / sqrtf(ks);
            for (int j = 0; j < sk; j++) { q[j] *= qs; k[j] *= ks; }
        }
        const float *vv = cv + 2 * keydim;
        float *out = m->xb2 + (size_t)b * xdim;
        for (int h = 0; h < nh; h++) {
            // llama.cpp's Qwen3.5 GDN kernel tiles the key-head axis across
            // value heads (0..G-1, 0..G-1).
            int group = h % ng;
            const float *q = cv + group * sk;
            const float *k = cv + keydim + group * sk;
            const float *v = vv + h * hv;
            float *st = states + (size_t)h * hv * hv;
            float beta = 1.0f / (1.0f + expf(-m->q_gate[(size_t)b * nh + h]));
            float a = alphas[(size_t)b * nh + h] + ly->ssm_dt[h];
            float softplus = a > 20.0f ? a : log1pf(expf(a));
            float decay = expf(ly->ssm_a[h] * softplus);
            for (int j = 0; j < hv; j++)
                for (int i = 0; i < hv; i++) st[j * hv + i] *= decay;
            for (int j = 0; j < hv; j++) {
                float pred = 0;
                for (int i = 0; i < hv; i++) pred += st[j * hv + i] * k[i];
                float delta = (v[j] - pred) * beta;
                for (int i = 0; i < hv; i++) st[j * hv + i] += delta * k[i];
                float y = 0;
                for (int i = 0; i < hv; i++) y += st[j * hv + i] * q[i];
                // The DeltaNet recurrence uses normalized Q/K and applies the
                // conventional 1/sqrt(key_dim) query scale.
                out[h * hv + j] = y / sqrtf((float)sk);
            }
            rmsnorm(out + h * hv, out + h * hv, ly->ssm_norm_w,
                    hv, m->rms_eps);
            for (int j = 0; j < hv; j++) {
                float z = m->ssm_z[(size_t)b * inner + h * hv + j];
                out[h * hv + j] *= z / (1.0f + expf(-z));
            }
        }
    }
    matvec_b(m->tp, m->xb, xdim, ly->ssm_out, m->xb2, xdim,
             inner, m->n_embd, NULL, n);
}

gguf_tensor moe_expert_weight(const layer_t *ly, int which, int e,
                              int n_embd, int n_ff_exp) {
    if (ly->moe_split) {
        gguf_tensor *t = which == 0 ? ly->moe_g[e]
                       : which == 1 ? ly->moe_u[e] : ly->moe_d[e];
        return *t;
    }
    gguf_tensor *base = which == 0 ? ly->ffn_gate_exps
                      : which == 1 ? ly->ffn_up_exps : ly->ffn_down_exps;
    // fused 3D: gate/up are {n_embd, n_ff_exp, n_expert}; down is
    // {n_ff_exp, n_embd, n_expert}. Expert e is a contiguous 2D block.
    int64_t rows = which == 2 ? n_embd : n_ff_exp;   // rows of this expert's block
    int64_t rowlen = which == 2 ? n_ff_exp : n_embd; // ne[0] (row length)
    size_t rs = ggml_row_size(base->type, rowlen);
    gguf_tensor v = *base;
    v.data = (uint8_t *)base->data + (size_t)e * rows * rs;
    // clamp to the slice: enc_mv's binding bounds check reads nbytes, and the
    // full fused-tensor size pushes expert e>=1 past the upload end (silent
    // CPU fallback for the whole forward)
    v.nbytes = (size_t)rows * rs;
    return v;
}

// Route one token: router matvec -> softmax over ALL experts -> top-k select
// (largest probs; ties to the lowest index) -> renormalize the selected weights
// to sum to 1 (Mixtral / Qwen3 convention). Writes sel[used]/selw[used].
static void moe_route(model_t *m, const layer_t *ly, const float *xin,
                      int n_embd, int ne, int used, int *sel, float *selw) {
    matvec_b(m->tp, m->moe_logits, ne, ly->ffn_gate_inp, xin,
             n_embd, n_embd, ne, NULL, 1);
    float mx = m->moe_logits[0];
    for (int e = 1; e < ne; e++)
        if (m->moe_logits[e] > mx) mx = m->moe_logits[e];
    float ssum = 0.0f;
    for (int e = 0; e < ne; e++) {
        float p = expf(m->moe_logits[e] - mx);
        m->moe_logits[e] = p;
        ssum += p;
    }
    for (int e = 0; e < ne; e++) m->moe_logits[e] /= ssum;
    float denom = 0.0f;
    for (int t = 0; t < used; t++) {
        int best = 0;
        float bp = -1.0f;
        for (int e = 0; e < ne; e++)
            if (m->moe_logits[e] > bp) { bp = m->moe_logits[e]; best = e; }
        sel[t] = best;
        selw[t] = bp;
        denom += bp;
        m->moe_logits[best] = -1.0f;                     // exclude from next round
    }
    for (int t = 0; t < used; t++) selw[t] /= denom;     // renormalized weight
}

// Decode path: one token, per selected expert SwiGLU -> weighted sum. Reads the
// normed input from xin and writes the FFN output back in place.
// Gated activation for a (Sw|Ge)GLU FFN: act(g) * u. act is SiLU for the
// llama family, the tanh-GELU approximation for gemma. One definition shared by
// the dense FFN and both MoE expert-FFN paths so a gemma-style GELU MoE cannot
// silently run SiLU math. The GPU path already selects f_gelu/f_silu the same
// way (enc_actmul / dense f-select).
static inline float gated_act(int act, float g, float u) {
    if (act == ACT_GELU) {
        float t = tanhf(0.7978845608f * (g + 0.044715f * g * g * g));
        return 0.5f * g * (1.0f + t) * u;
    }
    return (g / (1.0f + expf(-g))) * u;
}

static void moe_ffn_token(model_t *m, const layer_t *ly, float *xin) {
    int n_embd = m->n_embd, ne = m->n_expert, used = m->n_expert_used;
    int nff = m->n_ff_exp;
    int   sel[256];
    float selw[256];
    moe_route(m, ly, xin, n_embd, ne, used, sel, selw);
    for (int i = 0; i < n_embd; i++) m->moe_out[i] = 0.0f;
    for (int t = 0; t < used; t++) {
        int e = sel[t];
        float w = selw[t];
        gguf_tensor gv = moe_expert_weight(ly, 0, e, n_embd, nff);
        gguf_tensor uv = moe_expert_weight(ly, 1, e, n_embd, nff);
        gguf_tensor dv = moe_expert_weight(ly, 2, e, n_embd, nff);
        matvec_b(m->tp, m->moe_gate, nff, &gv, xin, n_embd, n_embd, nff, NULL, 1);
        matvec_b(m->tp, m->moe_up,   nff, &uv, xin, n_embd, n_embd, nff, NULL, 1);
        for (int j = 0; j < nff; j++)
            m->moe_gate[j] = gated_act(m->ffn_act, m->moe_gate[j], m->moe_up[j]);
        matvec_b(m->tp, m->moe_dexp, n_embd, &dv, m->moe_gate,
                 nff, nff, n_embd, NULL, 1);
        for (int i = 0; i < n_embd; i++) m->moe_out[i] += w * m->moe_dexp[i];
    }
    for (int i = 0; i < n_embd; i++) xin[i] = m->moe_out[i];
}

// Prefill path: route every token, then run each expert ONCE over all the
// tokens routed to it as a single batched matmul (the weight rows dequantize
// once and stream across every token) instead of one token at a time. Same
// math as the per-token path — the router, SwiGLU, and weights are identical,
// and each token accumulates its selected experts in ascending-expert order,
// which for the dense-oracle configs equals the per-token order — so greedy
// output is preserved. Big prefill throughput win; decode is untouched.
static void moe_ffn_grouped(model_t *m, const layer_t *ly, int n, int xdim) {
    int n_embd = m->n_embd, ne = m->n_expert, used = m->n_expert_used;
    int nff = m->n_ff_exp;
    for (int b = 0; b < n; b++) {
        float *xin = m->xb + (size_t)b * xdim;
        moe_route(m, ly, xin, n_embd, ne, used,
                  m->moe_sel + (size_t)b * used, m->moe_selw + (size_t)b * used);
        float *out = m->moe_out_b + (size_t)b * n_embd;
        for (int i = 0; i < n_embd; i++) out[i] = 0.0f;
    }
    for (int e = 0; e < ne; e++) {
        int cnt = 0;
        for (int b = 0; b < n; b++) {
            const int   *sel  = m->moe_sel  + (size_t)b * used;
            const float *selw = m->moe_selw + (size_t)b * used;
            for (int t = 0; t < used; t++) {
                if (sel[t] != e) continue;
                m->moe_gidx[cnt] = b;
                m->moe_gw[cnt]   = selw[t];
                memcpy(m->moe_gath + (size_t)cnt * n_embd,
                       m->xb + (size_t)b * xdim, (size_t)n_embd * sizeof(float));
                cnt++;
                break;                                   // top-k experts are distinct
            }
        }
        if (cnt == 0) continue;
        gguf_tensor gv = moe_expert_weight(ly, 0, e, n_embd, nff);
        gguf_tensor uv = moe_expert_weight(ly, 1, e, n_embd, nff);
        gguf_tensor dv = moe_expert_weight(ly, 2, e, n_embd, nff);
        matvec_b(m->tp, m->moe_gate_b, nff, &gv, m->moe_gath,
                 n_embd, n_embd, nff, NULL, cnt);
        matvec_b(m->tp, m->moe_up_b,   nff, &uv, m->moe_gath,
                 n_embd, n_embd, nff, NULL, cnt);
        for (int c = 0; c < cnt; c++) {
            float *g = m->moe_gate_b + (size_t)c * nff;
            const float *u = m->moe_up_b + (size_t)c * nff;
            for (int j = 0; j < nff; j++)
                g[j] = gated_act(m->ffn_act, g[j], u[j]);
        }
        matvec_b(m->tp, m->moe_dexp_b, n_embd, &dv, m->moe_gate_b,
                 nff, nff, n_embd, NULL, cnt);
        for (int c = 0; c < cnt; c++) {
            float *out = m->moe_out_b + (size_t)m->moe_gidx[c] * n_embd;
            const float *dx = m->moe_dexp_b + (size_t)c * n_embd;
            float w = m->moe_gw[c];
            for (int i = 0; i < n_embd; i++) out[i] += w * dx[i];
        }
    }
    for (int b = 0; b < n; b++)
        memcpy(m->xb + (size_t)b * xdim, m->moe_out_b + (size_t)b * n_embd,
               (size_t)n_embd * sizeof(float));
}

// gemma-4 fused gate_up expert slice for expert e: {n_embd, 2*n_ff_exp} (first
// n_ff_exp rows are gate, next n_ff_exp are up). Offset from n_embd/n_ff_exp/e
// metadata (shape-validated at load, RNC-1).
static gguf_tensor gemma_gate_up_weight(const layer_t *ly, int e, int n_embd,
                                        int n_ff_exp) {
    gguf_tensor *base = ly->ffn_gate_up_exps;
    size_t rs = ggml_row_size(base->type, n_embd);
    gguf_tensor v = *base;
    v.data = (uint8_t *)base->data + (size_t)e * (2 * (size_t)n_ff_exp) * rs;
    return v;
}

// gemma-4 dual-branch MoE FFN for one layer: a dense GELU shared expert AND a
// routed top-k GELU expert set, each with its own pre/post RMSNorm sandwich,
// summed. attn_out is m->x (post-attention residual); writes the sum to m->xb
// (the caller then applies post_ffw_norm + the residual add). Per token; decode
// is n==1. Verified token-identical to llama.cpp gemma4.cpp.
static void gemma_moe_ffn(model_t *m, const layer_t *ly, int n, int xdim) {
    int n_embd = m->n_embd, ne = m->n_expert, used = m->n_expert_used;
    int nff = m->n_ff_exp, dff = m->n_ff;           // dff = dense shared-FFN size
    float inv = 1.0f / sqrtf((float)n_embd);
    for (int b = 0; b < n; b++) {
        const float *attn = m->x + (size_t)b * n_embd;
        float xn[n_embd], mlp[n_embd], xn2[n_embd], rin[n_embd];
        // --- dense shared MLP: rmsnorm(ffn_norm) -> GELU SwiGLU -> post_ffw_norm_1
        rmsnorm(xn, attn, ly->ffn_norm_w, n_embd, m->rms_eps);
        matvec_b(m->tp, m->hb,  dff, ly->w_gate, xn, n_embd, n_embd, dff, NULL, 1);
        matvec_b(m->tp, m->hb2, dff, ly->w_up,   xn, n_embd, n_embd, dff, NULL, 1);
        for (int j = 0; j < dff; j++) m->hb[j] = gated_act(ACT_GELU, m->hb[j], m->hb2[j]);
        matvec_b(m->tp, mlp, n_embd, ly->w_down, m->hb, dff, dff, n_embd, NULL, 1);
        rmsnorm(mlp, mlp, ly->ffn_post_norm1_w, n_embd, m->rms_eps);
        // --- routed experts: pre-norm the branch input; router runs on a
        //     SEPARATE weightless-rmsnorm(attn_out) * (1/sqrt(n_embd)) * gate_inp_scale
        rmsnorm(xn2, attn, ly->ffn_pre_norm2_w, n_embd, m->rms_eps);
        float rss = 0.0f;
        for (int i = 0; i < n_embd; i++) rss += attn[i] * attn[i];
        rss = 1.0f / sqrtf(rss / n_embd + m->rms_eps);
        for (int i = 0; i < n_embd; i++)
            rin[i] = attn[i] * rss * inv *
                     (ly->gate_inp_scale ? ly->gate_inp_scale[i] : 1.0f);
        matvec_b(m->tp, m->moe_logits, ne, ly->ffn_gate_inp, rin, n_embd, n_embd, ne, NULL, 1);
        // softmax over all experts, top-k, renormalized (same as moe_route)
        float mx = m->moe_logits[0];
        for (int e = 1; e < ne; e++) if (m->moe_logits[e] > mx) mx = m->moe_logits[e];
        float ssum = 0.0f;
        for (int e = 0; e < ne; e++) { float p = expf(m->moe_logits[e] - mx); m->moe_logits[e] = p; ssum += p; }
        for (int e = 0; e < ne; e++) m->moe_logits[e] /= ssum;
        int sel[256]; float selw[256], denom = 0.0f;
        for (int t = 0; t < used; t++) {
            int best = 0; float bp = -1.0f;
            for (int e = 0; e < ne; e++) if (m->moe_logits[e] > bp) { bp = m->moe_logits[e]; best = e; }
            sel[t] = best; selw[t] = bp; denom += bp; m->moe_logits[best] = -1.0f;
        }
        for (int t = 0; t < used; t++) selw[t] /= denom;
        for (int i = 0; i < n_embd; i++) m->moe_out[i] = 0.0f;
        for (int t = 0; t < used; t++) {
            int e = sel[t];
            gguf_tensor guv = gemma_gate_up_weight(ly, e, n_embd, nff);
            matvec_b(m->tp, m->hb, 2 * nff, &guv, xn2, n_embd, n_embd, 2 * nff, NULL, 1);
            for (int j = 0; j < nff; j++)
                m->moe_gate[j] = gated_act(ACT_GELU, m->hb[j], m->hb[nff + j]);
            gguf_tensor dv = moe_expert_weight(ly, 2, e, n_embd, nff);
            matvec_b(m->tp, m->moe_dexp, n_embd, &dv, m->moe_gate, nff, nff, n_embd, NULL, 1);
            float sc = selw[t] * (ly->down_exps_scale ? ly->down_exps_scale[e] : 1.0f);
            for (int i = 0; i < n_embd; i++) m->moe_out[i] += sc * m->moe_dexp[i];
        }
        rmsnorm(m->moe_out, m->moe_out, ly->ffn_post_norm2_w, n_embd, m->rms_eps);
        // --- combine
        float *out = m->xb + (size_t)b * xdim;
        for (int i = 0; i < n_embd; i++) out[i] = mlp[i] + m->moe_out[i];
    }
}

// Sparse-MoE FFN for one layer. Decode (n==1) keeps the exact per-token path;
// prefill (n>1) groups tokens by shared expert for throughput.
static void moe_ffn(model_t *m, const layer_t *ly, int n, int xdim) {
    if (n <= 1) { moe_ffn_token(m, ly, m->xb); return; }
    moe_ffn_grouped(m, ly, n, xdim);
}

bool model_moe_ffn_cpu(model_t *m, int layer, int n) {
    if (!m || layer < 0 || layer >= m->n_layer || n < 1 || n > m->n_batch)
        return false;
    layer_t *ly = &m->layers[layer];
    if (!ly->is_moe) return false;
    int ne = m->n_embd, xs = m->xdim;
    if (ly->moe_gemma) {
        // Gemma's validated MoE is a coupled dense+routed branch with its own
        // sandwich norms, so the correctness boundary is the whole FFN.
        gemma_moe_ffn(m, ly, n, xs);
    } else {
        for (int b = 0; b < n; b++)
            rmsnorm(m->xb + (size_t)b * xs,
                    m->x + (size_t)b * ne,
                    ly->ffn_norm_w, ne, m->rms_eps);
        moe_ffn(m, ly, n, xs);
    }
    if (ly->post_ffn_norm_w)
        for (int b = 0; b < n; b++)
            rmsnorm(m->xb + (size_t)b * xs,
                    m->xb + (size_t)b * xs,
                    ly->post_ffn_norm_w, ne, m->rms_eps);
    for (int b = 0; b < n; b++)
        for (int i = 0; i < ne; i++)
            m->x[(size_t)b * ne + i] += m->xb[(size_t)b * xs + i];
    if (ly->out_scale != 1.0f && ly->out_scale != 0.0f)
        for (int b = 0; b < n; b++)
            for (int i = 0; i < ne; i++)
                m->x[(size_t)b * ne + i] *= ly->out_scale;
    return true;
}

float *model_forward_batch(model_t *m, const int32_t *tokens, int n, int pos,
                           bool want_logits) {
    if (m->qwen35 && pos == 0) {
        int convdim = 2 * m->ssm_state * m->ssm_groups + m->ssm_inner;
        int hv = m->ssm_inner / m->ssm_v_heads;
        memset(m->ssm_conv_state, 0, sizeof(float) * (size_t)m->n_layer *
               (m->ssm_conv_kernel - 1) * convdim);
        memset(m->ssm_state_mem, 0, sizeof(float) * (size_t)m->n_layer *
               m->ssm_v_heads * hv * hv);
    }
    // GPU handles the leading gpu_layers. A full split (gpu_layers == n_layer)
    // returns logits directly; a partial split runs [0, gpu_layers) on the GPU,
    // leaves the boundary activation in the host x buffer + the offloaded
    // layers' KV in the host cache, and the CPU loop below finishes the rest.
    int start = 0;
    int dbg = dbg_act_mode() && dbg_act_pass == 0;
    if (dbg_act_mode()) dbg_act_pass++;
    if (dbg)
        fprintf(stderr, "ACT ==== forward n=%d pos=%d arch=%s embd_scale=%.5f "
                "rms_eps=%.3g attn_scale=%.4f softcap=%.3f v_rmsnorm=%d "
                "kv_q8=%d n_suppress=%d\n",
                n, pos, m->arch, m->embd_scale, m->rms_eps, m->attn_scale,
                m->logit_softcap, (int)m->v_rmsnorm, (int)m->kv_q8, m->n_suppress);
    if (m->gpu) {
        if (m->gpu_layers >= m->n_layer) {
            float *lg = NULL;
            if (gpu_forward_batch(m, tokens, n, pos, want_logits, &lg)) {
                apply_head_transforms(m, lg);
                return lg;
            }
            // GPU failed at runtime: fall back to CPU permanently. Release the
            // backend rather than just forgetting it — with shared weights an
            // orphaned context also pins every other slot's copy of them.
            gpu_disable(m);
        } else if (gpu_forward_batch(m, tokens, n, pos, false, NULL)) {
            start = m->gpu_layers;
        } else {
            gpu_disable(m);
        }
    }
    int n_embd = m->n_embd;
    int xdim = m->xdim;   // cached at load (RNC-3)

    if (start == 0) {
        size_t ers = ggml_row_size(m->tok_embd->type, n_embd);
        for (int b = 0; b < n; b++) {
            // tokens[] is untrusted here (prompt ids, a corrupt cache, a
            // mismatched tokenizer): an id outside [0, n_vocab) would index the
            // embedding table out of its mapped rows. Clamp to 0 so a bad id
            // degrades output rather than reading past the mapping. (The
            // speculative path guards its drafted ids separately in engine.c.)
            int32_t id = tokens[b];
            if (id < 0 || id >= m->n_vocab) id = 0;
            dequant_row(m->tok_embd->type,
                        (uint8_t *)m->tok_embd->data + (size_t)id * ers,
                        m->x + (size_t)b * n_embd, n_embd);
            if (m->embd_scale != 1.0f)
                for (int i = 0; i < n_embd; i++)
                    m->x[(size_t)b * n_embd + i] *= m->embd_scale;
        }
        if (dbg) dbg_stat("post-embd", -1, m->x + (size_t)(n - 1) * n_embd, n_embd);
    }

    for (int l = start; l < m->n_layer; l++) {
        layer_t *ly = &m->layers[l];
        bool local   = model_is_swa(m, l);
        int hd       = model_head_dim(m, l);
        int n_kv     = model_n_head_kv(m, l);
        int q_dim    = model_q_dim(m, l);
        int kv_dim   = model_kv_dim(m, l);
        float scale  = model_attn_scale(m, l);
        uint8_t *kc_l = (uint8_t *)m->kcache + model_kv_byte_off(m, l);
        uint8_t *vc_l = (uint8_t *)m->vcache + model_kv_byte_off(m, l);
        size_t row_b = model_kv_row_bytes(m, l);

        // attention
        for (int b = 0; b < n; b++)
            rmsnorm(m->xb + (size_t)b * xdim, m->x + (size_t)b * n_embd,
                    ly->attn_norm_w, n_embd, m->rms_eps);
        if (dbg) {
            fprintf(stderr, "ACT L%-3d cfg swa=%d hd=%d n_kv=%d q_dim=%d kv_dim=%d "
                    "scale=%.5f out_scale=%.6f wv=%d qn=%d kn=%d pan=%d pfn=%d "
                    "anorm[absmax]=", l, (int)local, hd, n_kv, q_dim, kv_dim, scale,
                    ly->out_scale, ly->wv != NULL, ly->qnorm_w != NULL,
                    ly->knorm_w != NULL, ly->post_attn_norm_w != NULL,
                    ly->post_ffn_norm_w != NULL);
            float a = 0;
            for (int i = 0; i < n_embd; i++) {
                float t = ly->attn_norm_w[i] < 0 ? -ly->attn_norm_w[i] : ly->attn_norm_w[i];
                if (t > a) a = t;
            }
            fprintf(stderr, "%.4g\n", a);
            dbg_stat("post-attn-norm", l, m->xb + (size_t)(n - 1) * xdim, n_embd);
        }
        if (ly->recurrent) {
            qwen35_linear(m, ly, l, n, xdim);
        } else {
        if (m->qwen35) {
            matvec_b(m->tp, m->ssm_qkv, 2 * q_dim, ly->wq,
                     m->xb, xdim, n_embd, 2 * q_dim, ly->bq, n);
            for (int b = 0; b < n; b++)
                for (int h = 0; h < m->n_head; h++) {
                    memcpy(m->q + (size_t)b * q_dim + h * hd,
                           m->ssm_qkv + (size_t)b * 2 * q_dim + h * 2 * hd,
                           sizeof(float) * hd);
                    memcpy(m->q_gate + (size_t)b * q_dim + h * hd,
                           m->ssm_qkv + (size_t)b * 2 * q_dim + h * 2 * hd + hd,
                           sizeof(float) * hd);
                }
        } else {
            matvec_b(m->tp, m->q, q_dim, ly->wq, m->xb, xdim,
                     n_embd, q_dim, ly->bq, n);
        }
        matvec_b(m->tp, m->k_tmp, kv_dim, ly->wk, m->xb, xdim, n_embd, kv_dim, ly->bk, n);
        if (ly->wv)
            matvec_b(m->tp, m->v_tmp, kv_dim, ly->wv, m->xb, xdim, n_embd, kv_dim, ly->bv, n);
        else
            // gemma4 global layers have no V projection: V is the raw K
            memcpy(m->v_tmp, m->k_tmp, sizeof(float) * (size_t)n * kv_dim);
        if (dbg) {
            dbg_stat("q-raw", l, m->q + (size_t)(n - 1) * q_dim, q_dim);
            dbg_stat("k-raw", l, m->k_tmp + (size_t)(n - 1) * kv_dim, kv_dim);
            dbg_stat("v-raw", l, m->v_tmp + (size_t)(n - 1) * kv_dim, kv_dim);
        }
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
            uint8_t *kc = kc_l + (size_t)(pos + b) * row_b;
            uint8_t *vc = vc_l + (size_t)(pos + b) * row_b;
            if (m->kv_q8) {
                q8_quant_row(m->k_tmp + (size_t)b * kv_dim, kc, kv_dim);
                q8_quant_row(m->v_tmp + (size_t)b * kv_dim, vc, kv_dim);
            } else {
                f16_t *kh = (f16_t *)kc, *vh = (f16_t *)vc;
                for (int i = 0; i < kv_dim; i++) {
                    kh[i] = f32_to_f16(m->k_tmp[(size_t)b * kv_dim + i]);
                    vh[i] = f32_to_f16(m->v_tmp[(size_t)b * kv_dim + i]);
                }
            }
        }
        if (dbg) {
            dbg_stat("q-post-rope", l, m->q + (size_t)(n - 1) * q_dim, q_dim);
            dbg_stat("k-post-rope", l, m->k_tmp + (size_t)(n - 1) * kv_dim, kv_dim);
            dbg_stat("v-post-norm", l, m->v_tmp + (size_t)(n - 1) * kv_dim, kv_dim);
            if (!m->kv_q8) {
                dbg_stat_f16("k-cached", l,
                    (const f16_t *)(kc_l + (size_t)(pos + n - 1) * row_b), kv_dim);
                dbg_stat_f16("v-cached", l,
                    (const f16_t *)(vc_l + (size_t)(pos + n - 1) * row_b), kv_dim);
            }
        }
        for (int b = 0; b < n; b++) {
            int p = pos + b;
            int t0 = local && p - m->swa_window + 1 > 0 ? p - m->swa_window + 1 : 0;
            attn_job aj = { m, kc_l, vc_l, m->q + (size_t)b * q_dim,
                            m->xb2 + (size_t)b * xdim, p, t0, hd, kv_dim,
                            row_b, m->kv_q8, scale };
            tpool_run(m->tp, attn_heads, &aj, m->n_head);
            if (m->qwen35)
                for (int i = 0; i < q_dim; i++) {
                    float g = m->q_gate[(size_t)b * q_dim + i];
                    m->xb2[(size_t)b * xdim + i] *= 1.0f / (1.0f + expf(-g));
                }
        }
        if (dbg) dbg_stat("attn-out", l, m->xb2 + (size_t)(n - 1) * xdim, q_dim);
        matvec_b(m->tp, m->xb, xdim, ly->wo, m->xb2, xdim, q_dim, n_embd, ly->bo, n);
        }
        if (dbg) dbg_stat("wo-out", l, m->xb + (size_t)(n - 1) * xdim, n_embd);
        if (ly->post_attn_norm_w)
            for (int b = 0; b < n; b++)
                rmsnorm(m->xb + (size_t)b * xdim, m->xb + (size_t)b * xdim,
                        ly->post_attn_norm_w, n_embd, m->rms_eps);
        for (int b = 0; b < n; b++)
            for (int i = 0; i < n_embd; i++)
                m->x[(size_t)b * n_embd + i] += m->xb[(size_t)b * xdim + i];
        if (dbg) {
            dbg_stat("post-attn-res", l, m->x + (size_t)(n - 1) * n_embd, n_embd);
        }

        // feed-forward (gated: silu for llama-family, gelu for gemma). MoE
        // layers route each token to a few experts instead of one dense FFN.
        // gemma-4 MoE is a dual branch that does its own norms off m->x directly.
        if (ly->moe_gemma) {
            gemma_moe_ffn(m, ly, n, xdim);  // reads m->x, writes dense⊕routed to m->xb
            goto ffn_done;
        }
        for (int b = 0; b < n; b++)
            rmsnorm(m->xb + (size_t)b * xdim, m->x + (size_t)b * n_embd,
                    ly->ffn_norm_w, n_embd, m->rms_eps);
        if (ly->is_moe) {
            moe_ffn(m, ly, n, xdim);   // reads normed m->xb, writes FFN out to m->xb
        } else {
        matvec_b(m->tp, m->hb,  m->n_ff, ly->w_gate, m->xb, xdim, n_embd, m->n_ff, NULL, n);
        matvec_b(m->tp, m->hb2, m->n_ff, ly->w_up,   m->xb, xdim, n_embd, m->n_ff, NULL, n);
        for (size_t i = 0; i < (size_t)n * m->n_ff; i++)
            m->hb[i] = gated_act(m->ffn_act, m->hb[i], m->hb2[i]);
        if (dbg) dbg_stat("ffn-act", l, m->hb + (size_t)(n - 1) * m->n_ff, m->n_ff);
        matvec_b(m->tp, m->xb, xdim, ly->w_down, m->hb, m->n_ff, m->n_ff, n_embd, NULL, n);
        }
        ffn_done:
        if (dbg) dbg_stat("ffn-down", l, m->xb + (size_t)(n - 1) * xdim, n_embd);
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
        if (dbg) dbg_stat("layer-out", l, m->x + (size_t)(n - 1) * n_embd, n_embd);
    }

    if (!want_logits) return NULL;
    rmsnorm(m->xb, m->x + (size_t)(n - 1) * n_embd, m->out_norm_w, n_embd, m->rms_eps);
    if (dbg) dbg_stat("final-norm", m->n_layer, m->xb, n_embd);
    matvec_b(m->tp, m->logits, m->n_vocab, m->output, m->xb, xdim, n_embd, m->n_vocab, NULL, 1);
    if (dbg) dbg_stat("logits-raw", m->n_layer, m->logits, m->n_vocab);
    apply_head_transforms(m, m->logits);
    if (dbg) {
        dbg_stat("logits-final", m->n_layer, m->logits, m->n_vocab);
        int am = 0;
        for (int i = 1; i < m->n_vocab; i++)
            if (m->logits[i] > m->logits[am]) am = i;
        fprintf(stderr, "ACT top1 token=%d logit=%.5f\n", am, m->logits[am]);
    }
    return m->logits;
}

// forward a small batch keeping every row's hidden state in x (speculative
// verify). CPU path only: full GPU offload keeps hidden states on-device.
bool model_forward_batch_keep(model_t *m, const int32_t *tokens, int n, int pos) {
    if (m->gpu && m->gpu_layers >= m->n_layer) return false;
    if (!m->all_logits)
        m->all_logits = malloc(sizeof(float) * (size_t)m->spec_batch * m->n_vocab);
    if (n > m->spec_batch || !m->all_logits) return false;
    model_forward_batch(m, tokens, n, pos, false); // leaves hidden states in x
    return true;
}

// logits for one row of the last model_forward_batch_keep — computed lazily
// so an early draft rejection skips the remaining output projections (the
// lm_head is the single most expensive matvec in the model). Valid until
// the next forward.
float *model_spec_row_logits(model_t *m, int b) {
    int n_embd = m->n_embd, xdim = m->xdim;   // cached at load (RNC-3)
    float *lg = m->all_logits + (size_t)b * m->n_vocab;
    rmsnorm(m->xb, m->x + (size_t)b * n_embd, m->out_norm_w, n_embd, m->rms_eps);
    matvec_b(m->tp, lg, m->n_vocab, m->output, m->xb, xdim, n_embd,
             m->n_vocab, NULL, 1);
    apply_head_transforms(m, lg);
    return lg;
}

float *model_forward(model_t *m, int token, int pos) {
    int32_t t = token;
    return model_forward_batch(m, &t, 1, pos, true);
}

// ------------------------------------------------ continuous batching (Phase 6)
//
// The host half of the batched decode primitive: it owns the sequence list and
// the fallback, and it owns the decision of when a microbatch is worth forming.
// The arithmetic lives in the backend (gpu_batch_* in cuda.c), because that is
// the only place a decode step for N sequences can actually be *one* pass over
// the weights.
//
// The fallback is the whole reason this layer exists as a type rather than as a
// free function. Without a batching backend — CPU, Metal, partial offload, a
// model whose quant types have no batched kernel — the answer is still correct,
// it is just computed one sequence at a time. Callers get one API and one code
// path, and a scheduler written against it keeps working on a laptop.
struct model_batch {
    model_t **seqs;
    int       n;
    gpu_batch *gb;      // NULL = decode sequentially
};

int model_batch_max(void) {
    return MODEL_BATCH_MAX;
}

model_batch *model_batch_create(model_t **seqs, int n) {
    if (n < 1) return NULL;
    model_batch *b = calloc(1, sizeof(model_batch));
    if (!b) return NULL;
    b->seqs = malloc(sizeof(model_t *) * (size_t)n);
    if (!b->seqs) { free(b); return NULL; }
    memcpy(b->seqs, seqs, sizeof(model_t *) * (size_t)n);
    b->n = n;
    // A NULL here is not a failure: it means this build, backend or model has
    // no microbatch, and every decode falls through to the sequential path.
    b->gb = gpu_batch_create(seqs, n);
    return b;
}

void model_batch_free(model_batch *b) {
    if (!b) return;
    gpu_batch_free(b->gb);
    free(b->seqs);
    free(b);
}

bool model_batch_decode(model_batch *b, const int *idx, const int32_t *tok,
                        const int *pos, int n, float **out) {
    if (!b || n < 1) return false;
    for (int i = 0; i < n; i++)
        if (idx[i] < 0 || idx[i] >= b->n) return false;

    // Split into microbatches the backend can take in one launch. A caller
    // with more ready sequences than MODEL_BATCH_MAX gets consecutive passes
    // rather than an error, so the scheduler above never has to know the size.
    int done = 0;
    while (done < n) {
        int take = n - done;
        if (take > MODEL_BATCH_MAX) take = MODEL_BATCH_MAX;
        bool ok = b->gb && gpu_batch_decode(b->gb, idx + done, tok + done,
                                            pos + done, take, out + done);
        if (ok) {
            // The backend returns raw logits, exactly as gpu_forward_batch
            // does; the head transforms live here for both paths so a batched
            // step and a solo step cannot drift apart on them.
            for (int i = done; i < done + take; i++) {
                model_t *m = b->seqs[idx[i]];
                apply_head_transforms(m, out[i]);
            }
        }
        if (!ok) {
            // Sequential fallback. Also the recovery path: a backend that
            // fails mid-run has already released its GPU state, and these
            // forwards land on the CPU without the caller noticing.
            for (int i = done; i < done + take; i++) {
                out[i] = model_forward(b->seqs[idx[i]], tok[i], pos[i]);
                if (!out[i]) return false;
            }
        }
        done += take;
    }
    return true;
}

// mean-pooled, L2-normalized embedding of toks (final layer, output-normed).
// Clobbers KV slots [0, n) — the caller owns resetting its engine state.
bool model_embed(model_t *m, const int32_t *toks, int n, float *out) {
    if (n <= 0 || n > m->n_ctx) return false;
    void *save_gpu = m->gpu;
    if (m->gpu && m->gpu_layers >= m->n_layer)
        m->gpu = NULL; // full offload keeps hidden states on-device; go CPU
    memset(out, 0, sizeof(float) * m->n_embd);
    float *tmp = malloc(sizeof(float) * m->n_embd);
    if (!tmp) { m->gpu = save_gpu; return false; }
    for (int i = 0; i < n; ) {
        int chunk = n - i < m->n_batch ? n - i : m->n_batch;
        model_forward_batch(m, toks + i, chunk, i, false);
        for (int b = 0; b < chunk; b++) {
            rmsnorm(tmp, m->x + (size_t)b * m->n_embd, m->out_norm_w,
                    m->n_embd, m->rms_eps);
            for (int j = 0; j < m->n_embd; j++) out[j] += tmp[j];
        }
        i += chunk;
    }
    free(tmp);
    m->gpu = save_gpu;
    float ss = 0;
    for (int j = 0; j < m->n_embd; j++) { out[j] /= n; ss += out[j] * out[j]; }
    if (ss > 0) {
        float inv = 1.0f / sqrtf(ss);
        for (int j = 0; j < m->n_embd; j++) out[j] *= inv;
    }
    return true;
}
