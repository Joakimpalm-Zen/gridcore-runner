// Llama-family transformer: weight wiring + batched forward pass.
#include "model.h"

#include <errno.h>
#include "gpu.h"
#include "vramreg.h"
#include "compat.h"
#include "template.h"   // HARMONY_THINK_OPEN/CLOSE for the gpt-oss channel split

#include <math.h>
#include <float.h>
#include <pthread.h>
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

// Generous per-axis ceiling for geometry read out of an untrusted GGUF: real
// models are orders of magnitude smaller, and every axis below this bound
// keeps the int products the loader and forward pass compute from it
// (n_head*head_dim, 2*n_ff_exp, batch*width) far inside their type.
#define MDL_DIM_MAX 1048576   /* 2^20 elements per axis */

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
    // One notion of what a file *is*, shared with the weight registries: the
    // prefix-cache key and the sharing key deciding identity differently is
    // how the 2026-08-04 defect stayed invisible (the cache keyed real
    // checkpoints fine while the sharing lookup silently missed on them).
    // registry=NULL keeps this call quiet — on this same load path the loss
    // has already been reported by the host-weights lookup.
    uint64_t size = 0, ino = 0;
    int64_t mtime = 0, fctime = 0;
    if (m->gf.n_maps > 1) {
        // A stat tuple for one shard is not the identity of the set. Until a
        // whole-set key exists, disable persistent-prefix identity rather than
        // reusing state after an unselected shard changed on disk.
        m->file_id_ok = false;
        m->file_size = gguf_mapped_size(&m->gf);
        m->file_ino = 0;
        m->file_mtime_ns = 0;
        m->file_ctime_ns = 0;
    } else if (model_file_identity(path, NULL, &size, &ino, &mtime, &fctime)) {
        m->file_id_ok = true;
        m->file_size = size;
        m->file_ino = ino;
        m->file_mtime_ns = mtime;
        m->file_ctime_ns = fctime;
    } else {
        // Still include the mapped length in identities on platforms where
        // stat failed, rather than falling back to path alone.
        m->file_id_ok = false;
        m->file_size = gguf_mapped_size(&m->gf);
        m->file_ino = 0;
        m->file_mtime_ns = 0;
        m->file_ctime_ns = 0;
    }
}

// Materialize a tensor into a freshly-allocated f32 buffer holding at least
// `need` elements. Returns NULL when t is absent (a legitimate result for
// optional tensors). If t is present but the allocation fails, sets *ok =
// false and returns NULL, so an out-of-memory condition is never silently
// mistaken for an absent optional tensor (which would change the forward-pass
// math without any error).
//
// `need` is what the FORWARD PASS will index this vector with, and it is
// mandatory because the two are otherwise unrelated: every consumer of these
// buffers (norms over n_embd or a head, biases over q_dim/kv_dim, sinks over
// n_head, expert scales over n_expert) reads a count derived from geometry
// metadata, never from the tensor's own length. A file whose vector is
// shorter than the geometry that indexes it read past the buffer — so the
// count is stated here, once, at the point the buffer is created.
static float *tensor_to_f32(gguf_tensor *t, int64_t need, bool *ok) {
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
    if (need > 0 && n64 < (uint64_t)need) {
        fprintf(stderr, "error: tensor %s holds %llu values but this model "
                "geometry indexes %lld of them\n", t->name,
                (unsigned long long)n64, (long long)need);
        *ok = false; return NULL;
    }
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

// A weight whose ggml type this build cannot decode. need_tensor already
// refuses one; the OPTIONAL weights (gemma4's V, apertus's ffn_gate, the fused
// expert banks) arrive through opt_tensor, which cannot report and must not
// silently drop a tensor that IS present. dequant_block ignores a type it does
// not know, so an unchecked one leaves the scratch buffer at whatever it
// already held and the model runs on uninitialized memory — plausible output,
// silently wrong. Every weight the forward pass drives goes through the shape
// checks below, so the type check belongs with them.
static bool check_type(const gguf_tensor *t, const char *what, int layer) {
    if (!t || ggml_type_supported(t->type)) return true;
    fprintf(stderr, "error: tensor %s in blk.%d has unsupported type %d (%s)\n",
            what, layer, t->type, ggml_type_name(t->type));
    return false;
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
    if (!check_type(t, what, layer)) return false;
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
// last), and ne[2] must hold EXACTLY n_expert expert blocks — the caller
// passes the layer's own n_expert (from its router tensor, possibly smaller
// than the model's declared expert_count when --prune-experts wrote this
// layer), and every one of a layer's expert tensors must agree on it exactly:
// a mismatch between, say, ffn_gate_exps and ffn_down_exps silently drifts
// one of them out of alignment with the router's selection indices. Without
// this a crafted (or partially-pruned) GGUF whose expert tensors disagree
// with the declared geometry loads clean and is then read out of bounds or
// misaligned at decode time (RNC-1).
static bool check_shape3(gguf_tensor *t, int ne0, int ne1, int n_expert,
                         const char *what, int layer) {
    if (!t) return true;
    if (!check_type(t, what, layer)) return false;
    if ((int64_t)t->ne[0] != ne0 || (int64_t)t->ne[1] != ne1 ||
        (int64_t)t->ne[2] != n_expert) {
        fprintf(stderr, "error: MoE tensor %s in blk.%d has shape "
                "[%llu,%llu,%llu], expected [%d,%d,%d]\n",
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

// `attention.sliding_window_pattern` is published two ways. Dense gemma3/4
// give an integer PERIOD (every period-th layer is full attention); the
// Gemma-4 E-series instead gives a BOOLEAN ARRAY, one entry per layer. Read
// as a u32 an array key yields the default, which would silently mis-mark
// every layer, so both forms are handled here and the array wins when present.
// Returns true when an array form was consumed into out[].
static bool swa_pattern_array(gguf_file *g, const char *key, bool *out, int n) {
    gguf_kv *kv = gguf_get(g, key);
    if (!kv || kv->type != GGUF_T_ARR || kv->arr_type != GGUF_T_BOOL) return false;
    if ((int64_t)kv->arr_n != n || !kv->arr_raw) return false;
    const uint8_t *raw = kv->arr_raw;
    for (int i = 0; i < n; i++) out[i] = raw[i] != 0;
    return true;
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
        // gemma locals rope at 10k while globals run 1M; gpt-oss instead
        // inherits the GLOBAL base for its sliding layers (llama.cpp seeds
        // rope_freq_base_train_swa = rope_freq_base_train), so the 10k
        // default must not apply there.
        float local_base = gguf_get_f32(g, RK("rope.local.freq_base"),
                           gguf_get_f32(g, RK("rope.freq_base_swa"),
                                        m->gptoss ? m->rope_base : 10000.0f));
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
    // The local table above was built from the raw base, BEFORE the block that
    // scales the global frequencies. For an arch whose sliding layers share
    // the global rope regime that is wrong twice over — unscaled frequencies
    // and, via model_rope_mscale, a dropped YaRN magnitude factor. Copy the
    // scaled table across so the two regimes are genuinely identical.
    if (m->swa_window > 0 && m->swa_rope_global && m->rope_inv_freq_local) {
        if (m->rope_dim_local != m->rope_dim) {
            fprintf(stderr, "error: swa_rope_global needs matching rope dims "
                    "(local %d, global %d)\n", m->rope_dim_local, m->rope_dim);
            return false;
        }
        memcpy(m->rope_inv_freq_local, m->rope_inv_freq,
               sizeof(float) * (size_t)(m->rope_dim / 2));
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

// Byte cost of one MoE layer's expert half: router, routed expert tensors in
// either layout, and gemma-4's coupled dense shared branch. Shared by the VRAM
// estimate and the CUDA fit so the two cannot disagree about what an expert
// bank costs.
// Expert-granular prefetch.
//
// On a model larger than RAM the engine's cost is not bandwidth and not
// arithmetic — it is the NUMBER of I/O operations. Reaching an expert through
// the mmap costs ~200 synchronous 16 KB faults; measured on gemma-4-26B on an
// 8 GB M1 that is ~17,000 faults per token at ~45-60 us each, which is the
// whole token budget (thread scaling 1.32x for 4x threads, and warming the
// page cache halved the faults without moving throughput at all).
//
// The router has just named the 8 experts this layer will read. Handing those
// byte ranges to the OS as whole blocks turns ~200 faults per expert into one
// readahead. A standalone probe replaying real routing traces against the real
// GGUF measured the difference as 0.70 s/token of fault stall against
// 0.20-0.25 s/token of expert-granular reads.
//
// This is deliberately NOT a cache: no residency set, no eviction, no hit-rate
// policy. The same probe showed caching contributes little and does not
// generalize across workloads (13.7% cross-workload hit rate at top-8), while
// granularity alone captures most of the win. It also keeps the feature
// architecture-agnostic: the ONLY thing it takes from the model is the list of
// expert ids a router just produced, which every MoE arch has. The rejected
// expert-cache tier hooked one specific FFN path and therefore never engaged on
// gemma-4 at all (docs/negative-result-expert-cache.md); this hook is fed by
// whichever router ran.
//
// Advisory throughout: it cannot change a single output bit, only how long the
// read that follows takes.
static bool moe_prefetch_enabled(const model_t *m) {
    if (!m->moe_prefetch) return false;
    if (m->gpu && m->gpu_layers >= m->n_layer) return false;  // weights on device
    return true;
}

void model_moe_prefetch(const model_t *m, const layer_t *ly,
                        const int *sel, int used) {
    if (!moe_prefetch_enabled(m) || !ly->is_moe) return;
    int ne = ly->n_expert;
    for (int i = 0; i < used; i++) {
        int e = sel[i];
        if (e < 0 || e >= ne) continue;
        if (ly->moe_split) {           // legacy: one tensor per expert
            if (ly->moe_g[e]) plat_willneed(ly->moe_g[e]->data, ly->moe_g[e]->nbytes);
            if (ly->moe_u[e]) plat_willneed(ly->moe_u[e]->data, ly->moe_u[e]->nbytes);
            if (ly->moe_d[e]) plat_willneed(ly->moe_d[e]->data, ly->moe_d[e]->nbytes);
            continue;
        }
        // fused 3D {.., .., n_expert}: expert e is one contiguous stride in
        gguf_tensor *fused[3] = { ly->ffn_gate_up_exps ? ly->ffn_gate_up_exps
                                                       : ly->ffn_gate_exps,
                                  ly->ffn_gate_up_exps ? NULL : ly->ffn_up_exps,
                                  ly->ffn_down_exps };
        for (int t = 0; t < 3; t++) {
            gguf_tensor *w = fused[t];
            if (!w || ne <= 0) continue;
            size_t stride = (size_t)(w->nbytes / (uint64_t)ne);
            plat_willneed((const uint8_t *)w->data + stride * (size_t)e, stride);
        }
    }
}

// The routed expert banks alone — the only part of a layer that sparsity
// applies to. The router and any always-on shared expert are deliberately not
// counted here: every token touches those, so they belong to the hot set.
static uint64_t layer_routed_expert_bytes(const layer_t *ly, int n_expert) {
    uint64_t wb = 0;
    if (!ly->is_moe) return 0;
    if (ly->ffn_gate_up_exps) wb += ly->ffn_gate_up_exps->nbytes;  // gemma-4 fused
    if (ly->moe_split) {
        for (int e = 0; e < n_expert; e++)
            wb += ly->moe_g[e]->nbytes + ly->moe_u[e]->nbytes + ly->moe_d[e]->nbytes;
    } else {
        if (ly->ffn_gate_exps) wb += ly->ffn_gate_exps->nbytes;
        if (ly->ffn_up_exps)   wb += ly->ffn_up_exps->nbytes;
        if (ly->ffn_down_exps) wb += ly->ffn_down_exps->nbytes;
    }
    return wb;
}

uint64_t model_layer_expert_bytes(const layer_t *ly, int n_expert) {
    if (!ly->is_moe) return 0;
    uint64_t wb = layer_routed_expert_bytes(ly, n_expert);
    if (ly->ffn_gate_inp) wb += ly->ffn_gate_inp->nbytes;
    if (ly->moe_gemma) {   // dense GELU shared expert, evaluated with the layer
        if (ly->w_gate) wb += ly->w_gate->nbytes;
        if (ly->w_up)   wb += ly->w_up->nbytes;
        if (ly->w_down) wb += ly->w_down->nbytes;
    }
    return wb;
}

// ------------------------------------------------- pre-download fit check
//
// "Will this model run on this machine?" answered from a GGUF HEADER, before
// the weights exist locally. The arithmetic below deliberately re-derives what
// model_hot_set_bytes() and model_kv_row_bytes() compute, because those take a
// BOUND model_t -- they need the layer table, which needs the tensors, which
// is exactly what a header-only file does not have.
//
// Deliberately NOT in this file's job: fetching anything. The runner does not
// speak HTTP, and a fit check is not a reason to teach it. `--fit` reads a
// local path; the README documents the one-line ranged read that produces a
// header-only file from a URL.
static uint64_t fit_routed_expert_bytes(const gguf_file *g) {
    // Header-only, so this matches on NAME rather than the layer table. The
    // stacked expert banks are the `..._exps.weight` tensors; the router
    // (ffn_gate_inp) and any shared expert are excluded for the same reason
    // layer_routed_expert_bytes() excludes them -- every token touches those.
    uint64_t wb = 0;
    for (uint64_t i = 0; i < g->n_tensors; i++) {
        const char *n = g->tensors[i].name;
        size_t len = strlen(n);
        const char *suffix = "_exps.weight";
        size_t sl = strlen(suffix);
        if (len >= sl && strcmp(n + len - sl, suffix) == 0)
            wb += g->tensors[i].nbytes;
    }
    return wb;
}

bool model_fit_report(gguf_file *g, int n_ctx_want, model_fit *out) {
    memset(out, 0, sizeof(*out));
    char key[128];
    const char *arch = gguf_get_str(g, "general.architecture", "");
    if (!arch || !*arch) return false;
    #define FK(fmt) (snprintf(key, sizeof(key), "%s." fmt, arch), key)

    out->arch      = arch;
    out->n_layer   = (int)gguf_get_u32(g, FK("block_count"), 0);
    int n_embd     = (int)gguf_get_u32(g, FK("embedding_length"), 0);
    int n_head     = (int)gguf_get_u32(g, FK("attention.head_count"), 0);
    int n_head_kv  = (int)gguf_get_u32(g, FK("attention.head_count_kv"), n_head);
    int head_dim   = (int)gguf_get_u32(g, FK("attention.key_length"),
                                       n_head > 0 ? n_embd / n_head : 0);
    int train_ctx  = (int)gguf_get_u32(g, FK("context_length"), 0);
    out->n_expert      = (int)gguf_get_u32(g, FK("expert_count"), 0);
    out->n_expert_used = (int)gguf_get_u32(g, FK("expert_used_count"), 0);
    out->train_ctx     = train_ctx;

    if (out->n_layer <= 0 || head_dim <= 0 || n_head_kv <= 0) return false;

    // Context: what the caller asked for, else the loader's own default.
    out->n_ctx = n_ctx_want > 0 ? n_ctx_want
               : (train_ctx > 0 && train_ctx < 4096 ? train_ctx : 4096);

    out->weights = gguf_mapped_size(g);

    uint64_t routed = fit_routed_expert_bytes(g);
    if (out->n_expert > 0 && out->n_expert_used > 0 &&
        out->n_expert_used < out->n_expert && routed && routed <= out->weights) {
        out->hot = out->weights - routed +
                   routed * (uint64_t)out->n_expert_used / (uint64_t)out->n_expert;
        out->sparse = true;
    } else {
        out->hot = out->weights;
    }

    // Same shape as model_kv_row_bytes: K and V, per layer, per token. A model
    // with per-layer KV geometry (gemma-4 shared KV, MLA) will differ from
    // this; it is stated as an upper-bound estimate rather than quietly
    // presented as exact.
    uint64_t kv_dim = (uint64_t)n_head_kv * (uint64_t)head_dim;
    out->kv_f16_per_tok = 2ull * (uint64_t)out->n_layer * kv_dim * 2ull;
    out->kv_q8_per_tok  = kv_dim % 32 == 0
        ? 2ull * (uint64_t)out->n_layer * (kv_dim / 32) * 34ull
        : 0;                                   // q8 KV needs head_dim % 32 == 0
    out->kv_f16 = out->kv_f16_per_tok * (uint64_t)out->n_ctx;
    out->kv_q8  = out->kv_q8_per_tok  * (uint64_t)out->n_ctx;

    out->available = plat_ram_available_bytes();
    #undef FK
    return true;
}

const char *model_fit_verdict(const model_fit *f) {
    if (!f->available) return "UNKNOWN";
    if (f->hot + f->kv_f16 <= f->available) return "FITS";
    if (f->kv_q8 && f->hot + f->kv_q8 <= f->available) return "FITS WITH --kv q8";
    return "PAGES";
}

uint64_t model_hot_set_bytes(const model_t *m) {
    if (m->n_expert <= 0 || m->n_expert_used <= 0 ||
        m->n_expert_used >= m->n_expert) return 0;      // dense, or not sparse
    uint64_t routed = 0;
    for (int l = 0; l < m->n_layer; l++)
        routed += layer_routed_expert_bytes(&m->layers[l], m->n_expert);
    uint64_t mapped = gguf_mapped_size(&m->gf);
    if (!routed || routed > mapped) return 0;   // nothing to discount
    uint64_t hot = routed * (uint64_t)m->n_expert_used / (uint64_t)m->n_expert;
    return mapped - routed + hot;
}

// Which residency warning to print, split out from the printing so both
// branches can be tested. Available RAM cannot be forced on a build machine;
// the choice between the wordings can.
//
// `hot` is the per-token hot set (0 for a dense model, where the file is the
// hot set). The distinction matters because the dense wording is wrong in
// degree for a sparse MoE: gemma-4-26B-A4B was measured at 8+ tok/s on a 16 GB
// Mac while being told every token would page, because only 8 of its 128
// experts per layer are ever touched.
bool model_residency_warning(uint64_t need, uint64_t hot, uint64_t have,
                             bool locked, char *buf, size_t n) {
    if (!need || !have || need <= have) return false;
    const char *fix = locked ? ""
        : " --mlock pins them; a smaller model is the other fix.";
    if (!hot || hot >= need) {                          // dense, or not sparse
        snprintf(buf, n,
                 "warning: weights are %.1f GB but only %.1f GB of RAM is"
                 " available — expect the model to be evicted and every token"
                 " to page from disk.%s",
                 (double)need / 1e9, (double)have / 1e9, fix);
    } else if (hot > have) {                            // sparse, still too big
        snprintf(buf, n,
                 "warning: weights are %.1f GB but only %.1f GB of RAM is"
                 " available, and even the %.1f GB this sparse MoE touches per"
                 " token does not fit — expect every token to page from"
                 " disk.%s",
                 (double)need / 1e9, (double)have / 1e9, (double)hot / 1e9, fix);
    } else {                                            // sparse and it fits
        snprintf(buf, n,
                 "warning: weights are %.1f GB but only %.1f GB of RAM is"
                 " available. This is a sparse MoE and touches about %.1f GB"
                 " per token, which does fit — expect a slow start and disk"
                 " reads whenever routing reaches a cold expert, not a stall"
                 " on every token.",
                 (double)need / 1e9, (double)have / 1e9, (double)hot / 1e9);
        // Deliberately no --mlock hint here: pinning a file that does not fit
        // is not the fix for a hot set that does.
        (void)fix;
    }
    return true;
}

// Publish the per-layer expert placement for a requested host-layer count.
// CPU_MOE_ALL/CPU_MOE_AUTO and any count at or above the number of MoE layers
// host every expert FFN (the original all-or-nothing meaning); a smaller N
// hosts the *deepest* N, leaving the shallower banks device-resident so the
// GPU-resident run stays leading-aligned like the layer split. AUTO starts
// all-host and is narrowed by the CUDA upload once the budget is known.
void model_moe_place_host(model_t *m, int host_layers) {
    if (!m->moe_host) return;
    int n_moe = 0;
    for (int l = 0; l < m->n_layer; l++) if (m->layers[l].is_moe) n_moe++;
    bool all = host_layers < 0 || host_layers >= n_moe;
    int seen = 0;
    for (int l = 0; l < m->n_layer; l++) {
        if (!m->layers[l].is_moe) { m->moe_host[l] = false; continue; }
        seen++;
        m->moe_host[l] = all || seen > n_moe - host_layers;
    }
}

static uint64_t model_cuda_weight_estimate(const model_t *m,
                                           const model_params *p) {
    if (!p->cpu_moe || m->n_expert <= 0) return gguf_mapped_size(&m->gf);
    // An explicit partial split leaves some expert banks device-resident, so
    // the estimate cannot assume every expert stays on the host. AUTO fits
    // into whatever is free, so it estimates as the all-host lower bound.
    int host_layers = p->cpu_moe_layers;
    int n_moe = 0;
    for (int l = 0; l < m->n_layer; l++) if (m->layers[l].is_moe) n_moe++;
    int device_moe = host_layers >= 0 && host_layers < n_moe
                       ? n_moe - host_layers : 0;
    uint64_t total = m->output ? m->output->nbytes : 0;
    int seen_moe = 0;
    for (int l = 0; l < m->n_layer; l++) {
        const layer_t *ly = &m->layers[l];
        gguf_tensor *att[] = { ly->wq, ly->wk, ly->wv, ly->wo };
        for (int i = 0; i < 4; i++) if (att[i]) total += att[i]->nbytes;
        if (!ly->is_moe) {
            gguf_tensor *ffn[] = { ly->w_gate, ly->w_up, ly->w_down };
            for (int i = 0; i < 3; i++) if (ffn[i]) total += ffn[i]->nbytes;
        } else if (seen_moe++ < device_moe) {
            total += model_layer_expert_bytes(ly, m->n_expert);
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
    m->vram = vram_claim(gpu_id, m->path, need, p->vram_priority,
                         vram_free_now, NULL,
                         p->vram_wait_secs, p->load_cancel, &st, err, sizeof(err));
    if (m->vram) return true;

    if (st.holders > 0) {
        fprintf(stderr, "error: %s\n", err);
        return false;
    }
    // Nobody to blame: claim what is actually available so the next runner can
    // still see this instance, and let the adaptive split size itself down.
    m->vram = vram_claim(gpu_id, m->path, st.available, p->vram_priority,
                         vram_free_now, NULL, 0, NULL, NULL, NULL, 0);
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
        "smollm", "stablelm", "gemma3", "gemma4", "phi3", "gpt-oss",
        "apertus", "afmoe", "muse-glimmer", "granite", "granitehybrid",
        "nemotron_h", "nemotron_h_moe",
    };
    if (count) *count = sizeof(arches) / sizeof(arches[0]);
    return arches;
}

static bool model_load_inner(model_t *m, const char *path, const model_params *p);
static bool model_bind_weights(model_t *m, const char *path, const model_params *p);
static bool model_alloc_runtime(model_t *m, const model_params *p);

static bool profile_integer(const gguf_kv *kv) {
    if (!kv) return false;
    return kv->type == GGUF_T_U8 || kv->type == GGUF_T_I8 ||
           kv->type == GGUF_T_U16 || kv->type == GGUF_T_I16 ||
           kv->type == GGUF_T_U32 || kv->type == GGUF_T_I32 ||
           kv->type == GGUF_T_U64 || kv->type == GGUF_T_I64;
}

// ------------------------------------------------------ shared host weights
//
// Everything model_bind_weights produces is derived from the GGUF and never
// written again: the mmap and its parsed metadata, the layer array, the f32
// conversion of every norm and bias, the per-layer geometry. Two model_t
// values loaded from the same file with the same weight-side parameters hold
// bit-identical copies of all of it.
//
// Until now they held literal copies, and the bill was not the weights — those
// are mmap'd — but the parse. Measured on Qwen2.5-7B-Instruct-Q4_K_M,
// `--serve --parallel 4` cost 29.7 MB of touched host memory per extra slot,
// of which 15.3 MB is the tokenizer vocabulary and merge list: 303,454
// separate string allocations, remade per slot, for a vocabulary the slots
// never read — they share one tokenizer built from slot 0's file.
//
// src/cuda.c has done the device-side version of this since the MoE work: one
// upload, refcounted, keyed on file identity. This is that, on the host. The
// record owns the buffers and every model_t sharing it holds aliasing
// pointers, so field access is unchanged and only ownership moved — which is
// what keeps the change out of the backends.
typedef struct model_weights {
    int      refs;
    model_t  proto;         // the model exactly as the bind phase left it
    // What these buffers were derived from, compared rather than assumed from
    // the path: a model rebuilt on disk between two loads must not be served
    // out of the previous parse.
    uint64_t fsize, fino;
    int64_t  fmtime, fctime;
    // The only two parameters the bind phase reads. Everything else the load
    // takes from `p` is consumed after the seam, by the per-instance half, so
    // two slots may legitimately differ on it and still share these buffers.
    bool     want_kv_q8;
    int      gpu_mode;
    struct model_weights *next;
} model_weights;

static model_weights *g_weights;
static pthread_mutex_t g_weights_mu = PTHREAD_MUTEX_INITIALIZER;

static bool path_looks_like_split_part(const char *path) {
    if (!path) return false;
    size_t n = strlen(path);
    if (n < 20 || strcmp(path + n - 5, ".gguf") != 0) return false;
    const char *of = path + n - 13; // "of-00000.gguf"
    if (memcmp(of, "of-", 3) != 0) return false;
    for (int i = 3; i < 8; i++) if (of[i] < '0' || of[i] > '9') return false;
    const char *part = of - 7;      // "-00000-of-00000.gguf"
    if (part[0] != '-' || part[6] != '-') return false;
    for (int i = 1; i < 6; i++) if (part[i] < '0' || part[i] > '9') return false;
    return true;
}

// Losing the identity is not fatal — the model still loads — so this is the
// only place it is ever reported. Keep it a warning on stderr rather than a
// verbose-only line: it means this instance stopped sharing weights and now
// picks its own CPU/GPU split, which two slots of one server must not do.
static void warn_no_file_id(const char *path, const char *registry,
                            const char *why) {
    fprintf(stderr,
            "warning: %s cannot be keyed by file identity (%s) — loading it "
            "privately: %s are not shared with another instance of this model, "
            "and it re-decides its own CPU/GPU split\n",
            path ? path : "(null path)", why, registry);
}

bool model_file_identity(const char *path, const char *registry,
                         uint64_t *size, uint64_t *ino,
                         int64_t *mtime, int64_t *ctime) {
    // Deliberate test hook: on a machine whose stat() works, nothing in a load
    // can provoke this branch, so the fallback would be untestable and the
    // sharing gate unfalsifiable. Same role as RUNNER_TEST_GPU_OFF.
    if (getenv("RUNNER_TEST_NO_FILE_ID")) {
        if (registry)
            warn_no_file_id(path, registry,
                            "injected by RUNNER_TEST_NO_FILE_ID");
        return false;
    }
    if (!path) {
        if (registry) warn_no_file_id(path, registry, "no path");
        return false;
    }
#ifdef _WIN32
    // MinGW's stat() is the wrong tool here twice over: st_size is 32-bit, so
    // every real checkpoint (>2 GB) fails with EOVERFLOW and the identity is
    // lost — every --parallel slot then loads privately and re-decides its
    // own CPU/GPU split, which is the 2026-08-04 defect — and its timestamps
    // are whole seconds, which can alias two GGUF revisions written in the
    // same second. Native file information gives the stable 64-bit file index
    // and the filesystem's 100 ns timestamps without hashing multi-GB files.
    HANDLE h = CreateFileA(path, FILE_READ_ATTRIBUTES,
                           FILE_SHARE_READ | FILE_SHARE_WRITE |
                           FILE_SHARE_DELETE, NULL, OPEN_EXISTING,
                           FILE_ATTRIBUTE_NORMAL, NULL);
    BY_HANDLE_FILE_INFORMATION info;
    if (h != INVALID_HANDLE_VALUE && GetFileInformationByHandle(h, &info)) {
        *size  = ((uint64_t)info.nFileSizeHigh << 32) | info.nFileSizeLow;
        *ino   = ((uint64_t)info.nFileIndexHigh << 32) | info.nFileIndexLow;
        *mtime = filetime_unix_ns(info.ftLastWriteTime);
        if (ctime) *ctime = filetime_unix_ns(info.ftCreationTime);
        CloseHandle(h);
        return true;
    }
    if (h != INVALID_HANDLE_VALUE) CloseHandle(h);
#endif
    struct stat st;
    if (stat(path, &st) != 0) {
        // ENOENT is not a sharing problem. The load is about to fail with its
        // own, clearer error, and a warning about weight sharing and CPU/GPU
        // splits for a file that cannot load at all buries the real cause —
        // the header comment above justifies this warning on the grounds that
        // "the model still loads", which is exactly false when the file is
        // gone. Every other stat failure still warrants it.
        if (registry && errno != ENOENT) {
            char why[128];
            snprintf(why, sizeof(why), "stat: %s", strerror(errno));
            warn_no_file_id(path, registry, why);
        }
        return false;
    }
    *size  = (uint64_t)st.st_size;
#ifdef _WIN32
    // stat succeeded where native file information did not (non-NTFS volume,
    // exotic redirector). No stable index on this path; the remaining fields
    // still key, just ino=0 for every such file.
    *ino   = 0;
#else
    *ino   = (uint64_t)st.st_ino;
#endif
    *mtime = stat_mtime_ns(&st);
    if (ctime) *ctime = stat_ctime_ns(&st);
    return true;
}

// Caller holds g_weights_mu.
static model_weights *mw_find(const char *path, const model_params *p,
                              uint64_t size, uint64_t ino,
                              int64_t mtime, int64_t ctime) {
    for (model_weights *w = g_weights; w; w = w->next) {
        if (!w->proto.path || strcmp(w->proto.path, path) != 0) continue;
        if (w->fsize != size || w->fino != ino ||
            w->fmtime != mtime || w->fctime != ctime) continue;
        if (w->want_kv_q8 != p->kv_q8 || w->gpu_mode != p->gpu_mode) continue;
        return w;
    }
    return NULL;
}

// The immutable half. Freed by the last holder, or directly by model_free for
// a load that failed before it could publish.
static void model_free_weights(model_t *m) {
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
        free(l->attn_sinks);
        free(l->ffn_gate_inp_b); free(l->ffn_gate_exps_b); free(l->exp_probs_b);
        free(l->ffn_up_exps_b);  free(l->ffn_down_exps_b);
        free(l->ffn_pre_norm2_w); free(l->ffn_post_norm1_w); free(l->ffn_post_norm2_w);
        free(l->ssm_dt); free(l->ssm_a); free(l->ssm_norm_w);
        free(l->ssm_conv1d_b); free(l->ssm_d);
        free(l->moe_g); free(l->moe_u); free(l->moe_d);  // split-MoE pointer arrays
        free(l->ple_post_norm);
    }
    free(m->l_head_kv); free(m->l_head_dim); free(m->l_rope_dim);
    free(m->l_is_swa); free(m->suppress); free(m->kv_src);
    free(m->ple_proj_norm);
    free(m->xielu_an); free(m->xielu_ap); free(m->xielu_b); free(m->xielu_eps);
    free(m->layers);
    free(m->path);
    free(m->fused_splits);
    free(m->out_norm_w);
    // Unlock before unmapping, and only the mapping this model_t actually
    // locked: with shared weights several model_t alias one map, and munlock
    // on a mapping we never locked is a silent no-op that would hide a failed
    // lock rather than report it.
    if (m->weights_locked) {
        for (uint32_t i = 0; i < gguf_map_count(&m->gf); i++) {
            size_t size;
            void *map = gguf_map_part(&m->gf, i, &size);
            if (map && size) plat_munlock(map, size);
        }
        m->weights_locked = false;
    }
    gguf_close(&m->gf);
}

static void mw_release(model_weights *w) {
    if (!w) return;
    pthread_mutex_lock(&g_weights_mu);
    bool last = --w->refs == 0;
    if (last)
        for (model_weights **pp = &g_weights; *pp; pp = &(*pp)->next)
            if (*pp == w) { *pp = w->next; break; }
    pthread_mutex_unlock(&g_weights_mu);
    if (last) {
        model_free_weights(&w->proto);
        free(w);
    }
}

static bool model_load_inner(model_t *m, const char *path, const model_params *p) {
    uint64_t size = 0, ino = 0;
    int64_t  mtime = 0, ctime = 0;
    bool id_ok = model_file_identity(path, "host weights", &size, &ino,
                                     &mtime, &ctime);
    // Standard split names are the only multi-part layout gguf_open accepts.
    // One part's stat tuple cannot key buffers derived from every part.
    if (path_looks_like_split_part(path)) id_ok = false;

    // The lock is held across the whole bind, so two slots racing on one file
    // cannot both pay for the parse — the same reason cuda.c holds its
    // registry lock across the upload. It serializes concurrent *loads*, which
    // is all it can block: the server loads slots in sequence, and swap and
    // draft loads already hold their own locks.
    pthread_mutex_lock(&g_weights_mu);
    model_weights *w = id_ok ? mw_find(path, p, size, ino, mtime, ctime) : NULL;
    if (w) {
        w->refs++;
        *m = w->proto;      // scalars, plus aliases into the shared buffers
        m->W = w;
    } else {
        if (!model_bind_weights(m, path, p)) {
            pthread_mutex_unlock(&g_weights_mu);
            return false;
        }
        // A file that cannot be stat'd cannot be keyed, so it is loaded
        // privately rather than shared under an identity nothing can confirm.
        // Same if the record itself cannot be allocated: sharing is an
        // optimization and must never be the reason a load fails.
        if (id_ok && (w = calloc(1, sizeof(*w))) != NULL) {
            w->refs       = 1;
            w->proto      = *m;   // the per-instance fields are all still zero
            w->fsize      = size;
            w->fino       = ino;
            w->fmtime     = mtime;
            w->fctime     = ctime;
            w->want_kv_q8 = p->kv_q8;
            w->gpu_mode   = p->gpu_mode;
            w->next       = g_weights;
            g_weights     = w;
            m->W          = w;
        }
    }
    pthread_mutex_unlock(&g_weights_mu);
    return model_alloc_runtime(m, p);
}

double model_resident_fraction(const model_t *m) {
    if (!m || !gguf_mapped_size(&m->gf)) return -1.0;
    double weighted = 0.0;
    uint64_t measured = 0;
    for (uint32_t i = 0; i < gguf_map_count(&m->gf); i++) {
        size_t size;
        void *map = gguf_map_part(&m->gf, i, &size);
        double fraction = plat_resident_fraction(map, size);
        if (fraction < 0.0) return -1.0;
        weighted += fraction * (double)size;
        measured += size;
    }
    return measured ? weighted / (double)measured : -1.0;
}

// Say so at load time when the weights cannot fit in what the machine has
// left, because every signal downstream stays green while they page: the tok/s
// counter, /health and --caps all look fine while a five-token reply takes a
// minute. Reported from a 16 GB Mac where a 1.2k-token prompt returned nothing
// in 300 s at 0% CPU. Advisory only -- the load continues either way, since
// the estimate can be wrong and a model that fits today may not fit at noon.
// Expert prefetch hands routed experts to the OS as whole blocks instead of
// letting them arrive as ~16 KB fault-by-fault reads. The default is decided
// PER MACHINE CLASS, because that is what the measurements decided:
//
// - Apple Silicon, weights over available RAM: ON. Two machines agree —
//   interleaved A/B measured decode 1.37 -> 1.96 tok/s (1.43x) on an 8 GB M1
//   at ~4x oversubscription, and 2.68 -> 3.37 (1.26x, arms fully disjoint
//   across three interleaved rounds) on a 16 GB M2 Pro at ~3.2x.
// - Everywhere else: OFF. On Linux with XFS at 425 MB/s the same feature
//   measured nothing at any oversubscription from 1.35x to 3.36x, against a
//   RAM cap verified to hold, replicated. The M2 Pro ran at HIGHER
//   oversubscription than that null, so the differentiator is fault cost on
//   the storage class, not the oversubscription ratio: many small synchronous
//   faults are expensive on Apple NVMe via mmap and absorbed on fast direct
//   storage. A win on one storage class does not set another's default.
//
// Precedence: --moe-prefetch flag, then RUNNER_MOE_PREFETCH env, then the
// per-class default above. The flag exists because env-only opt-in dies on
// GUI relaunch: a tray started by a login LaunchAgent runs with launchd's
// environment, and a measured win that evaporates on reboot reads as a
// regression (workmac deployment report, 2026-08-07).
static bool moe_prefetch_default(const model_t *m, int flag) {
    if (flag > 0) return true;
    if (flag < 0) return false;
    const char *e = getenv("RUNNER_MOE_PREFETCH");
    if (e && *e) return strcmp(e, "0") && strcmp(e, "off");
#if defined(__APPLE__) && defined(__aarch64__)
    uint64_t need = gguf_mapped_size(&m->gf), have = plat_ram_available_bytes();
    return need && have && need > have;
#else
    // Windows gained a working prefetch primitive on 2026-08-07
    // (PrefetchVirtualMemory; it was a no-op before). That makes the feature
    // AVAILABLE there, not on: no Windows A/B has been run yet, and the whole
    // reason this default is per-class is that a win on one class does not
    // transfer to another. `--moe-prefetch on` is how you opt in and how the
    // A/B gets run.
    (void)m;
    return false;
#endif
}

static void warn_if_it_will_not_stay_resident(const model_t *m, bool locked) {
    char msg[512];
    if (model_residency_warning(gguf_mapped_size(&m->gf), model_hot_set_bytes(m),
                                plat_ram_available_bytes(), locked,
                                msg, sizeof(msg)))
        fprintf(stderr, "%s\n", msg);
}

bool model_load(model_t *m, const char *path, const model_params *p) {
    if (!m) return false;
    memset(m, 0, sizeof(*m));
    if (!path || !p) return false;
    if (!model_load_inner(m, path, p)) {
        model_free(m);
        return false;
    }
    bool locked = false;
    if (p && p->mlock && gguf_mapped_size(&m->gf)) {
        uint32_t n = gguf_map_count(&m->gf), done = 0;
        locked = true;
        for (; done < n; done++) {
            size_t size;
            void *map = gguf_map_part(&m->gf, done, &size);
            if (!map || !size || !plat_mlock(map, size)) { locked = false; break; }
        }
        if (!locked)
            while (done > 0) {
                size_t size;
                void *map = gguf_map_part(&m->gf, --done, &size);
                plat_munlock(map, size);
            }
        if (locked) {
            m->weights_locked = true;
            fprintf(stderr, "mlock: %.1f GB of weights wired into RAM\n",
                    (double)gguf_mapped_size(&m->gf) / 1e9);
        } else {
            // Not fatal, and not silent. A refusal here usually means the
            // process cannot raise its locked-memory limit (RLIMIT_MEMLOCK),
            // which is the common case on a stock Linux box.
            fprintf(stderr,
                    "warning: --mlock could not wire %.1f GB (%s); continuing"
                    " without it, so the weights can still be evicted\n",
                    (double)gguf_mapped_size(&m->gf) / 1e9, strerror(errno));
        }
    }
    warn_if_it_will_not_stay_resident(m, locked);
    m->moe_prefetch = m->n_expert > 0 && !locked &&
                      moe_prefetch_default(m, p->moe_prefetch);
    // Refuse to announce a prefetch the platform cannot perform. Without this
    // an A/B on such a machine compares the feature against itself and comes
    // back flat, which reads exactly like an honest negative result — the
    // same trap as a CPU-vs-CPU identity check that "passes".
    if (m->moe_prefetch && !plat_willneed_available()) {
        m->moe_prefetch = false;
        fprintf(stderr, "moe: expert prefetch requested but this platform has "
                "no prefetch primitive (Windows needs 8 or newer); running "
                "without it\n");
    }
    if (m->moe_prefetch)
        fprintf(stderr, "moe: prefetching routed experts as whole blocks "
                "(--moe-prefetch off disables)\n");
    return true;
}

// ---------------------------------------------------------- hybrid SSM admission
//
// The Mamba-2 hybrid families — Granite-4 h-series (`granitehybrid`) and
// Bind one Nemotron-H block. Each block is EXACTLY ONE of {SSM, attention,
// MLP} (mutually exclusive), typed off head_count_kv / feed_forward_length,
// with a single pre-norm (attn_norm) and a single residual add. Fail closed on
// any missing or mis-dimensioned tensor (the hostile-GGUF discipline). The AK
// arch-key macro is local to model_bind_weights, so this helper spells the one
// per-layer array key it needs in full.
static bool nemotron_bind_layer(model_t *m, gguf_file *g, layer_t *l, int i) {
    bool ok = true;
    gguf_tensor *an = need_tensor(g, "blk.%d.attn_norm.weight", i, &ok);
    if (!ok) return false;
    l->attn_norm_w = tensor_to_f32(an, m->n_embd, &ok);
    if (!ok) return false;
    l->out_scale = 1.0f;
    int kv = m->l_head_kv[i];
    char ffkey[64];
    snprintf(ffkey, sizeof ffkey, "%s.feed_forward_length",
             gguf_get_str(g, "general.architecture", "nemotron_h"));
    int ff = (int)gguf_get_u32_idx(g, ffkey, (uint64_t)i, 0);
    if (kv == 0 && ff == 0) {
        // ---- recurrent (Mamba-2) block: mixer only, no FFN ----
        l->recurrent = true;
        l->skip_ffn  = true;
        int nh = m->ssm_v_heads, ds = m->ssm_state, ng = m->ssm_groups;
        int inner = m->ssm_inner;
        int conv_dim  = inner + 2 * ng * ds;
        int d_in_proj = 2 * inner + 2 * ng * ds + nh;
        l->ssm_in   = need_tensor(g, "blk.%d.ssm_in.weight", i, &ok);
        l->ssm_conv = need_tensor(g, "blk.%d.ssm_conv1d.weight", i, &ok);
        gguf_tensor *sconvb = need_tensor(g, "blk.%d.ssm_conv1d.bias", i, &ok);
        gguf_tensor *sdtb   = need_tensor(g, "blk.%d.ssm_dt.bias", i, &ok);
        gguf_tensor *sa     = need_tensor(g, "blk.%d.ssm_a", i, &ok);
        gguf_tensor *sd     = need_tensor(g, "blk.%d.ssm_d", i, &ok);
        gguf_tensor *sn     = need_tensor(g, "blk.%d.ssm_norm.weight", i, &ok);
        l->ssm_out  = need_tensor(g, "blk.%d.ssm_out.weight", i, &ok);
        if (!ok) return false;
        // ssm_norm is [d_inner/n_group, n_group] here (granite's is 1 group);
        // both flatten to `inner` contiguous floats the grouped RMS norm reads,
        // group g's weights at offset g*(inner/n_group).
        if (!check_shape(l->ssm_in, m->n_embd, d_in_proj, "ssm_in", i) ||
            !check_shape(l->ssm_conv, m->ssm_conv_kernel, conv_dim, "ssm_conv1d", i) ||
            !check_shape(sconvb, conv_dim, 1, "ssm_conv1d.bias", i) ||
            !check_shape(sdtb, nh, 1, "ssm_dt.bias", i) ||
            !check_shape(sa, 1, nh, "ssm_a", i) ||
            !check_shape(sd, 1, nh, "ssm_d", i) ||
            !check_shape(sn, inner / ng, ng, "ssm_norm", i) ||
            !check_shape(l->ssm_out, inner, m->n_embd, "ssm_out", i))
            return false;
        l->ssm_conv1d_b = tensor_to_f32(sconvb, conv_dim, &ok);
        l->ssm_dt       = tensor_to_f32(sdtb, nh, &ok);
        l->ssm_a        = tensor_to_f32(sa, nh, &ok);
        l->ssm_d        = tensor_to_f32(sd, nh, &ok);
        l->ssm_norm_w   = tensor_to_f32(sn, inner, &ok);
        if (!ok) return false;
        l->n_ff = m->n_ff;   // unused (skip_ffn); kept in-range for any bound
    } else if (kv > 0) {
        // ---- attention block: mixer only, no FFN. NoPE, no biases, no qk-norm ----
        l->skip_ffn = true;
        int hd = m->head_dim;
        l->wq = need_tensor(g, "blk.%d.attn_q.weight", i, &ok);
        l->wk = need_tensor(g, "blk.%d.attn_k.weight", i, &ok);
        l->wv = need_tensor(g, "blk.%d.attn_v.weight", i, &ok);
        l->wo = need_tensor(g, "blk.%d.attn_output.weight", i, &ok);
        if (!ok) return false;
        if (!check_shape(l->wq, m->n_embd, m->n_head * hd, "attn_q", i) ||
            !check_shape(l->wk, m->n_embd, kv * hd, "attn_k", i) ||
            !check_shape(l->wv, m->n_embd, kv * hd, "attn_v", i) ||
            !check_shape(l->wo, m->n_head * hd, m->n_embd, "attn_output", i))
            return false;
        l->n_ff = m->n_ff;   // unused (skip_ffn)
    } else {
        // ---- MLP block: no mixer. Dense gate-less squared-ReLU FFN, or — for
        //      nemotron_h_moe — a gate-less squared-ReLU MoE (router + routed
        //      experts + an always-on gate-less shared expert). ----
        l->skip_mixer = true;
        l->n_ff = ff;
        // The block's single pre-norm doubles as the FFN input norm. Convert a
        // SEPARATE copy (not an alias of attn_norm_w): model_free_weights frees
        // both fields, so sharing one pointer would double-free.
        l->ffn_norm_w = tensor_to_f32(an, m->n_embd, &ok);
        if (!ok) return false;
        if (m->n_expert > 0) {
            l->is_moe   = true;
            l->n_expert = m->n_expert;
            l->ffn_gate_inp  = need_tensor(g, "blk.%d.ffn_gate_inp.weight", i, &ok);
            l->ffn_up_exps   = need_tensor(g, "blk.%d.ffn_up_exps.weight", i, &ok);
            l->ffn_down_exps = need_tensor(g, "blk.%d.ffn_down_exps.weight", i, &ok);
            if (!ok) return false;
            if ((int64_t)l->ffn_gate_inp->ne[0] != m->n_embd ||
                (int64_t)l->ffn_gate_inp->ne[1] != m->n_expert) {
                fprintf(stderr, "error: blk.%d ffn_gate_inp is [%lld,%lld], "
                        "expected [%d,%d]\n", i,
                        (long long)l->ffn_gate_inp->ne[0],
                        (long long)l->ffn_gate_inp->ne[1], m->n_embd, m->n_expert);
                return false;
            }
            if (m->n_ff_shexp > 0) {
                l->w_up_shexp   = need_tensor(g, "blk.%d.ffn_up_shexp.weight", i, &ok);
                l->w_down_shexp = need_tensor(g, "blk.%d.ffn_down_shexp.weight", i, &ok);
                if (!ok) return false;
                if (!check_shape(l->w_up_shexp, m->n_embd, m->n_ff_shexp,
                                 "ffn_up_shexp", i) ||
                    !check_shape(l->w_down_shexp, m->n_ff_shexp, m->n_embd,
                                 "ffn_down_shexp", i))
                    return false;
            }
        } else {
            l->w_gate = NULL;
            l->w_up   = need_tensor(g, "blk.%d.ffn_up.weight", i, &ok);
            l->w_down = need_tensor(g, "blk.%d.ffn_down.weight", i, &ok);
            if (!ok) return false;
            if (l->n_ff <= 0 || l->n_ff > m->n_ff ||
                !check_shape(l->w_up, m->n_embd, l->n_ff, "ffn_up", i) ||
                !check_shape(l->w_down, l->n_ff, m->n_embd, "ffn_down", i))
                return false;
        }
    }
    return true;
}

static bool model_bind_weights(model_t *m, const char *path, const model_params *p) {
    if (!gguf_open(&m->gf, path)) return false;
    gguf_file *g = &m->gf;
    // A profile is opt-in metadata: legacy/dense GGUFs remain admitted exactly
    // as before. Once any profile key is present, however, the contract is
    // atomic and fail-closed. Validate it before path/state/tensor allocation.
    static const char *const profile_keys[] = {
        "gridcore.agent.protocol_version", "gridcore.agent.tokenizer_version",
        "gridcore.agent.schema_id", "gridcore.agent.schema_digest",
        "gridcore.agent.required_features",
    };
    bool profile = false;
    for (size_t i = 0; i < sizeof(profile_keys) / sizeof(*profile_keys); i++)
        profile |= gguf_get(g, profile_keys[i]) != NULL;
    if (profile) {
        gguf_kv *pv = gguf_get(g, profile_keys[0]);
        gguf_kv *tv = gguf_get(g, profile_keys[1]);
        gguf_kv *rf = gguf_get(g, profile_keys[4]);
        const char *sid = gguf_get_str(g, profile_keys[2], NULL);
        const char *dig = gguf_get_str(g, profile_keys[3], NULL);
        uint32_t protocol = gguf_get_u32(g, profile_keys[0], 0);
        uint32_t tokenizer = gguf_get_u32(g, profile_keys[1], 0);
        if (!profile_integer(pv) || !profile_integer(tv) ||
            !sid || !*sid || !dig || strlen(dig) != 64 ||
            !rf || rf->type != GGUF_T_ARR || rf->arr_type != GGUF_T_STR) {
            fprintf(stderr, "error: invalid gridcore agent profile metadata\n");
            return false;
        }
        for (int i = 0; i < 64; i++)
            if (!((dig[i] >= '0' && dig[i] <= '9') ||
                  (dig[i] >= 'a' && dig[i] <= 'f'))) {
                fprintf(stderr, "error: invalid gridcore agent schema digest\n");
                return false;
            }
        if (protocol != 1) {
            fprintf(stderr, "error: unsupported gridcore agent protocol version %u\n",
                    protocol);
            return false;
        }
        if (tokenizer != 1) {
            fprintf(stderr, "error: unsupported gridcore agent tokenizer version %u\n",
                    tokenizer);
            return false;
        }
        static const char *const features[] = {
            "dense", "json_schema", "continuous_batching", "prefix_cache",
            "spec_decode",
        };
        for (uint64_t i = 0; i < rf->arr_n; i++) {
            const char *want = rf->arr_str[i].s;
            bool known = false;
            for (size_t j = 0; j < sizeof(features) / sizeof(*features); j++)
                if (!strcmp(want, features[j])) { known = true; break; }
            // Named-but-unimplemented features get their own reason: a profile
            // that REQUIRES multi-token-prediction consumption is asking for a
            // verifier this build does not have, which is different from an
            // unrecognized string and must not read as a typo.
            if (!strcmp(want, "mtp")) {
                fprintf(stderr, "error: this profile requires MTP consumption, "
                        "which this build does not implement (MTP tensors are "
                        "admitted as training-only; see docs)\n");
                return false;
            }
            if (!known) {
                fprintf(stderr, "error: unsupported required agent feature '%s'\n", want);
                return false;
            }
        }
        m->agent_profile = true;
        m->agent_protocol_version = protocol;
        m->agent_tokenizer_version = tokenizer;
        m->agent_schema_id = sid;
        m->agent_schema_digest = dig;
        m->agent_required_features = rf->arr_str;
        m->n_agent_required_features = rf->arr_n;
    }
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
    if (strcmp(arch, "gemma2") == 0 ||
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
    // Bounded HERE, not at the general geometry gate below: the architecture
    // blocks between the two allocate n_layer-length arrays (SWA patterns,
    // per-layer head geometry, the xIELU parameters), so the gate is too late
    // to keep a hostile count out of a size. Read as a u32 into an int, 2^31
    // arrives negative and calloc(negative, ...) asked for ~2^64 bytes — the
    // load then failed with no message at all.
    if (m->n_layer < 1 || m->n_layer > 100000) {
        fprintf(stderr, "error: '%s' declares an unusable block_count (%u)\n",
                arch, gguf_get_u32(g, AK("block_count"), 0));
        return false;
    }
    // "every layer owns its KV" is the default; only gemma4 E-series lowers it
    m->kv_from_start = m->n_layer;
    m->n_embd      = (int)gguf_get_u32(g, AK("embedding_length"), 0);
    m->n_head      = (int)gguf_get_u32(g, AK("attention.head_count"), 0);
    m->n_head_kv   = (int)gguf_get_u32(g, AK("attention.head_count_kv"), m->n_head);
    // feed_forward_length is a scalar in almost every export, but gemma-4 E2B
    // publishes real per-layer width variation (6144/12288) as an ARRAY-typed
    // value. m->n_ff carries the MAX (scratch buffers size off it); the
    // per-layer widths land on each layer_t at bind time via the same
    // per-index getter, and heterogeneous widths mark ffn_var so the device
    // backends can refuse rather than compute with one global width.
    m->n_ff        = (int)gguf_get_u32(g, AK("feed_forward_length"), 0);
    m->ffn_var     = false;
    if (m->n_ff == 0 && m->n_layer > 0) {
        uint32_t mx = 0, mn = UINT32_MAX;
        for (int i = 0; i < m->n_layer; i++) {
            uint32_t w = gguf_get_u32_idx(g, AK("feed_forward_length"),
                                          (uint64_t)i, 0);
            if (w == 0) { mx = 0; break; }   // absent or malformed entry
            if (w > mx) mx = w;
            if (w < mn) mn = w;
        }
        if (mx > 0) { m->n_ff = (int)mx; m->ffn_var = mn != mx; }
    }
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
    m->logit_scale = 1.0f;
    m->resid_scale = 1.0f;
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
        if (pattern < 1) pattern = 6;   // it is a divisor below (every other
                                        // arch block clamps its own period)
        m->l_is_swa    = calloc(m->n_layer, sizeof(bool));
        if (!m->l_is_swa) return false;
        if (!swa_pattern_array(g, AK("attention.sliding_window_pattern"),
                               m->l_is_swa, m->n_layer))
            for (int i = 0; i < m->n_layer; i++)
                m->l_is_swa[i] = m->swa_window > 0 && ((i + 1) % pattern) != 0;
        else if (m->swa_window <= 0)
            for (int i = 0; i < m->n_layer; i++) m->l_is_swa[i] = false;
    }
    if (strcmp(arch, "gpt-oss") == 0) {
        // gpt-oss (OpenAI MoE). Transcribed from llama.cpp
        // src/models/openai-moe.cpp + build_moe_ffn, not inferred:
        //   * attn_norm pre-attention, post_attention_norm as the FFN norm
        //     (the qwen35 loading shape, not llama's ffn_norm);
        //   * per-head attention SINKS in the softmax denominator;
        //   * clamped alpha-sigmoid GLU (ACT_SWIGLU_OAI), not SwiGLU;
        //   * router bias + per-expert gate/up/down biases;
        //   * SWA period 2 with NO separate SWA rope base — llama.cpp seeds
        //     rope_freq_base_train_swa from the main base and only overrides
        //     it if the key exists, so the locals rope at 150k here. The
        //     runner's generic SWA path defaults that to 10k, which would be
        //     silently wrong, so it is pinned explicitly below.
        m->gptoss     = true;
        // Harmony's analysis channel, split out of content by think_feed the
        // same way muse's ` to=self` reasoning turn is. Suppressed from
        // content by default and surfaced as reasoning_content.
        m->think_open  = HARMONY_THINK_OPEN;
        m->think_close = HARMONY_THINK_CLOSE;
        m->swa_rope_global = true;   // sliding layers rope like the global ones
        m->ffn_act    = ACT_SWIGLU_OAI;
        m->swa_window = (int)gguf_get_u32(g, AK("attention.sliding_window"), 0);
        int swa_period = (int)gguf_get_u32(g, AK("attention.sliding_window_pattern"), 2);
        if (swa_period < 1) swa_period = 2;
        m->l_is_swa = calloc(m->n_layer, sizeof(bool));
        if (!m->l_is_swa) return false;
        // llama.cpp set_swa_pattern(p): is_swa[il] = il % p < p-1, which for
        // p=2 marks the EVEN layers — identical to the runner's existing
        // ((i + 1) % p) != 0 form used by gemma3.
        for (int i = 0; i < m->n_layer; i++)
            m->l_is_swa[i] = m->swa_window > 0 && ((i + 1) % swa_period) != 0;
        m->rms_eps = gguf_get_f32(g, AK("attention.layer_norm_rms_epsilon"), 1e-5f);
    }
    if (strcmp(arch, "afmoe") == 0) {
        // afmoe (Arcee Trinity, AfmoeForCausalLM). Transcribed from llama.cpp
        // src/models/afmoe.cpp (PR #16477) + the Trinity GGUF headers:
        //   * muP embedding scale sqrt(n_embd), hardcoded exactly as llama.cpp
        //     does for this arch;
        //   * Qwen-G1 output-gated attention: a separate blk.N.attn_gate
        //     projection of the normed block input, sigmoid, elementwise on
        //     the concatenated heads before attn_output (the qwen35 gate math
        //     with a standalone tensor instead of a fused Q slice);
        //   * SWA 2048 at period 4 (3 local : 1 global); the global layers
        //     are NoPE — the Llama-4 no_rope knob with the SAME period, set
        //     after the generic key read below since the GGUF carries no key;
        //   * sigmoid routing with a DeepSeek-style selection-only bias
        //     (exp_probs_b.bias), renormalized weights, then x route_scale —
        //     all existing generalized-router behavior driven by keys;
        //   * the first leading_dense_block_count layers are plain dense FFN.
        m->attn_out_gate = true;
        m->embd_scale    = sqrtf((float)m->n_embd);
        m->swa_window    = (int)gguf_get_u32(g, AK("attention.sliding_window"), 2048);
        int period       = (int)gguf_get_u32(g, AK("attention.sliding_window_pattern"), 4);
        if (period < 1) period = 4;
        m->l_is_swa = calloc(m->n_layer, sizeof(bool));
        if (!m->l_is_swa) return false;
        for (int i = 0; i < m->n_layer; i++)
            m->l_is_swa[i] = m->swa_window > 0 && ((i + 1) % period) != 0;
        m->swa_rope_global = true;   // single 10k rope base; globals skip rope
        m->n_dense_lead  = (int)gguf_get_u32(g, AK("leading_dense_block_count"), 0);
        int gating = (int)gguf_get_u32(g, AK("expert_gating_func"), 2);
        m->expert_gating  = gating == 1 ? EXPERT_GATE_SOFTMAX : EXPERT_GATE_SIGMOID;
        m->expert_w_scale = gguf_get_f32(g, AK("expert_weights_scale"), 0.0f);
        m->expert_norm_w  = gguf_get_bool(g, AK("expert_weights_norm"), false);
    }
    if (strcmp(arch, "qwen3") == 0 || strcmp(arch, "qwen3moe") == 0) {
        // Thinking-tuned Qwen3 (dense and sparse-MoE) responses wrap hidden
        // reasoning before the visible answer; shared CLI/server output
        // handling splits this pair. qwen3moe = qwen3 attention (qk-norm, GQA,
        // NeoX rope) with a sparse-MoE FFN, handled by the is_moe layer path.
        m->think_open  = "<think>";
        m->think_close = "</think>";
    }
    // Multi-token-prediction heads, admitted as training-only for EVERY
    // architecture that declares them. An export whose block_count includes
    // auxiliary NextN/MTP predictor blocks says so with
    // `<arch>.nextn_predict_layers`; those blocks have a different tensor
    // layout and take no part in ordinary autoregressive decoding, so the
    // backbone depth excludes them and dense decoding is bit-for-bit
    // unchanged. Consuming them (speculation off the MTP heads) is a separate,
    // unimplemented feature — a profile that *requires* it is rejected above.
    // This generalizes the qwen35-only handling; qwen35 exports read the same
    // key and behave exactly as before.
    m->mtp_layers = (int)gguf_get_u32(g, AK("nextn_predict_layers"), 0);
    if (m->mtp_layers < 0 || m->mtp_layers >= m->n_layer) {
        fprintf(stderr, "error: invalid NextN/MTP layer count %d for %d blocks\n",
                m->mtp_layers, m->n_layer);
        return false;
    }
    m->n_layer -= m->mtp_layers;
    if (!m->l_is_swa) {
        int sw = (int)gguf_get_u32(g, AK("attention.sliding_window"), 0);
        if (sw > 0) {
            int pattern = (int)gguf_get_u32(g, AK("attention.sliding_window_pattern"), 2);
            if (pattern < 1) pattern = 2;
            m->swa_window = sw;
            m->l_is_swa = calloc(m->n_layer, sizeof(bool));
            if (!m->l_is_swa) return false;
            if (!swa_pattern_array(g, AK("attention.sliding_window_pattern"),
                                   m->l_is_swa, m->n_layer))
                for (int i = 0; i < m->n_layer; i++)
                    m->l_is_swa[i] = ((i + 1) % pattern) != 0;
        }
    }
    if (strcmp(arch, "qwen35") == 0) {
        // Qwen3.5 dense (the architecture used by Ornith-1.0-9B) alternates
        // three Gated DeltaNet layers with one conventional attention layer.
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
    if (strcmp(arch, "apertus") == 0) {
        // Apertus: ungated MLP (no ffn_gate) with the xIELU activation, whose
        // four parameters are published per layer.
        m->ffn_act = ACT_XIELU;
        if (m->n_layer < 1 || m->n_layer > 100000) {
            fprintf(stderr, "error: apertus block_count %d out of range\n", m->n_layer);
            return false;
        }
        size_t nl = (size_t)(unsigned)m->n_layer;
        m->xielu_an  = malloc(sizeof(float) * nl);
        m->xielu_ap  = malloc(sizeof(float) * nl);
        m->xielu_b   = malloc(sizeof(float) * nl);
        m->xielu_eps = malloc(sizeof(float) * nl);
        if (!m->xielu_an || !m->xielu_ap || !m->xielu_b || !m->xielu_eps)
            return false;
        // The keys are NOT architecture-prefixed (llama.cpp spells them
        // "xielu.alpha_n", not "apertus.xielu.alpha_n") and may be a scalar
        // shared by every layer or a per-layer array.
        static const struct { const char *key; size_t off; float dflt; } XK[] = {
            { "xielu.alpha_n", 0, 0.8f }, { "xielu.alpha_p", 1, 0.8f },
            { "xielu.beta",    2, 0.5f }, { "xielu.eps",     3, -1e-6f },
        };
        float *dst[4] = { m->xielu_an, m->xielu_ap, m->xielu_b, m->xielu_eps };
        for (int k = 0; k < 4; k++) {
            gguf_kv *a = gguf_get(g, XK[k].key);
            float scalar = gguf_get_f32(g, XK[k].key, XK[k].dflt);
            for (int i = 0; i < m->n_layer; i++) {
                float v = scalar;
                // same alignment rule as every other array read here: the
                // elements are packed at the file's offset, not the type's
                if (a && a->arr_raw && a->arr_type == GGUF_T_F32 &&
                    (uint64_t)i < a->arr_n)
                    memcpy(&v, (const uint8_t *)a->arr_raw + (size_t)i * 4, 4);
                dst[XK[k].off][i] = v;
            }
        }
        // The file's alpha_n and alpha_p are NOT what the activation consumes.
        // ggml_xielu transforms them when it builds the node — alpha_p becomes
        // softplus(alpha_p) and alpha_n becomes beta + softplus(alpha_n) —
        // and only then does op_xielu see them. Transcribing op_xielu without
        // that step, which is what runner did, feeds the raw values straight
        // in. It is invisible wherever the raw value is large, because
        // softplus saturates to the identity above ~20 and most of Apertus-8B's
        // alphas are in the tens or hundreds; it is catastrophic in the middle
        // layers, where layer 15 has alpha_n 0.00296 against an effective
        // 1.19463 — a factor of 403. The model generated fluent-looking
        // gibberish.
        //
        // Folded here rather than in xielu() so the hot path is unchanged and
        // the transform happens once per layer instead of once per element.
        for (int i = 0; i < m->n_layer; i++) {
            m->xielu_ap[i] = softplus_f32(m->xielu_ap[i]);
            m->xielu_an[i] = m->xielu_b[i] + softplus_f32(m->xielu_an[i]);
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
        // The general geometry validation runs after the arch blocks, but this
        // one sizes several per-layer arrays from block_count first.
        if (m->n_layer < 1 || m->n_layer > 100000) {
            fprintf(stderr, "error: gemma4 block_count %d out of range\n", m->n_layer);
            return false;
        }
        int shared_kv = (int)gguf_get_u32(g, AK("attention.shared_kv_layers"), 0);
        // Same rule as expert_count: above INT_MAX this lands negative, and a
        // negative width reads as "no per-layer embeddings" — the E-series
        // branch is skipped, its tensors are ignored, and the model answers as
        // a different architecture without saying so.
        uint32_t ple_raw = gguf_get_u32(g, AK("embedding_length_per_layer_input"), 0);
        if (ple_raw > MDL_DIM_MAX) {
            fprintf(stderr, "error: gemma4 per-layer embedding size %u is out "
                    "of range\n", ple_raw);
            return false;
        }
        m->n_embd_ple    = (int)ple_raw;
        m->kv_from_start = m->n_layer;
        if (shared_kv > 0) {
            // llama-model.cpp's reuse callback: layers at or past
            // n_layer - shared_kv_layers own no cache and read the LAST KV
            // layer of their own sliding/full type, which is
            // (kv_from_start - 2) for sliding and (kv_from_start - 1) for full.
            m->kv_from_start = m->n_layer - shared_kv;
            if (m->kv_from_start < 2) {
                fprintf(stderr, "error: gemma4 shared_kv_layers=%d leaves no "
                        "KV-owning layers\n", shared_kv);
                return false;
            }
        }
        if (m->n_embd_ple > 0 || m->kv_from_start < m->n_layer)
            // E-series: per-layer embeddings and/or shared-KV layers. Verified
            // against llama.cpp b10076 on gemma-4-E4B-it Q4_K_M — greedy
            // agreement is at the quantisation noise floor (the same profile a
            // long-verified dense model shows), not exact token identity.
            // CPU and CUDA, byte-identical to each other.
            fprintf(stderr, "gemma4: E-series (per-layer embeddings%s) — "
                    "verified against llama.cpp at the Q4_K noise floor "
                    "rather than token-identically\n",
                    m->kv_from_start < m->n_layer ? " + shared KV" : "");
        else
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
        // per-layer arrays must have the element width we index with — a
        // converter that writes e.g. I32 booleans would misparse every layer.
        // (head_count_kv goes through gguf_get_u32_idx below, which does its
        // own type validation and reads elements without assuming alignment.)
        if (swa_arr && swa_arr->arr_type != GGUF_T_BOOL &&
            swa_arr->arr_type != GGUF_T_U8 && swa_arr->arr_type != GGUF_T_I8)
            swa_arr = NULL;
        for (int i = 0; i < m->n_layer; i++) {
            bool swa = swa_arr && swa_arr->arr_raw && (uint64_t)i < swa_arr->arr_n
                       ? ((const uint8_t *)swa_arr->arr_raw)[i] != 0 : false;
            m->l_is_swa[i]   = swa && m->swa_window > 0;
            m->l_head_dim[i] = swa ? hd_swa : m->head_dim;
            // The rotated-dim count must come from the SAME choice that picks
            // the frequency table: rope_apply reads rope_inv_freq_local only
            // for a layer model_is_swa() says is sliding, and that predicate
            // is window-gated. Selecting on the raw pattern bit instead would
            // pair the local dim count with the global table on a file that
            // declares a sliding pattern and no window.
            m->l_rope_dim[i] = m->l_is_swa[i] ? m->rope_dim_local : m->rope_dim;
            // A GGUF array is packed at whatever offset it landed on, so it
            // is NOT aligned for the element type: casting arr_raw to
            // uint32_t* and indexing it is a misaligned load (UBSan flagged
            // exactly this on the committed gemma4-hetero fixture). The
            // per-index getter reads elements with memcpy and validates the
            // element type and range in one place, which is where that
            // knowledge belongs.
            m->l_head_kv[i]  = (int)gguf_get_u32_idx(
                g, AK("attention.head_count_kv"), (uint64_t)i,
                (uint32_t)m->n_head_kv);
            // Per-layer kv-head / head-dim / rope-dim come from untrusted
            // per-layer keys and feed the same attention divisors and index
            // arithmetic as the scalar path (kv_dim / hd, n_head / n_head_kv,
            // and the pair index inside a head). A zero or non-dividing value
            // would divide-by-zero on the first token; rope rotates
            // rope_dim/2 PAIRS inside each head, so a rope dim wider than the
            // head runs off the end of the last head's q/k row (the scalar
            // gate below bounds rope_dim against head_dim, but never saw
            // rope.dimension_count_swa against key_length_swa) — reject at
            // load instead.
            if (m->l_head_dim[i] < 1 || m->l_head_kv[i] < 1 ||
                m->n_head < 1 || m->l_head_kv[i] > m->n_head ||
                m->n_head % m->l_head_kv[i] != 0 ||
                m->l_rope_dim[i] < 0 || m->l_rope_dim[i] > m->l_head_dim[i] ||
                (int64_t)m->n_head * m->l_head_dim[i] > MDL_DIM_MAX) {
                fprintf(stderr, "error: invalid gemma4 per-layer geometry at "
                        "blk.%d (head_dim=%d head_count_kv=%d rope_dim=%d, "
                        "n_head=%d)\n",
                        i, m->l_head_dim[i], m->l_head_kv[i], m->l_rope_dim[i],
                        m->n_head);
                return false;
            }
        }
        // Shared-KV map. Identity for the layers that own a cache; every
        // later layer points at the last KV-owning layer of its own type.
        // n_layer is bounded above; the local keeps that range visible to the
        // allocator call across the intervening stores to *m.
        size_t nl = (size_t)(unsigned)m->n_layer;
        m->kv_src = calloc(nl ? nl : 1, sizeof(int));
        if (!m->kv_src) return false;
        for (int i = 0; i < m->n_layer; i++) {
            m->kv_src[i] = i;
            if (i < m->kv_from_start) continue;
            int src = m->kv_from_start - (m->l_is_swa[i] ? 2 : 1);
            if (src < 0 || m->l_head_dim[src] != m->l_head_dim[i] ||
                m->l_head_kv[src] != m->l_head_kv[i]) {
                // a source whose KV geometry differs would be read with the
                // wrong row stride — refuse rather than reinterpret bytes
                fprintf(stderr, "error: gemma4 shared-KV layer %d maps to %d "
                        "with mismatched KV geometry\n", i, src);
                return false;
            }
            m->kv_src[i] = src;
        }
    }
    if (strcmp(arch, "granite") == 0) {
        // granite (IBM Granite dense, 3.x/4.1; reference: llama.cpp b10353
        // src/models/granite.cpp + build_inp_embd). The four muP scalars are
        // the whole family quirk — and running the arch llama-style without
        // them is exactly why it sat on the wrong-math refusal list:
        //   * embeddings x embedding_scale (build_inp_embd);
        //   * kq_scale is the FIXED attention.scale value, not 1/sqrt(hd);
        //   * BOTH branch outputs x residual_scale before their residual
        //     adds (granite.cpp:241/301);
        //   * final logits x 1/logit_scale — llama.cpp DIVIDES where the
        //     runner's knob multiplies, so store the reciprocal.
        // Rope is adjacent-pair (LLAMA_ROPE_TYPE_NORM, like llama), gated by
        // rope.scaling.finetuned which defaults ON; a NoPE granite export is
        // refused rather than mis-rotated. granitemoe/granitehybrid are
        // different arch ids and stay unadmitted.
        m->rope_neox   = false;
        m->embd_scale  = gguf_get_f32(g, AK("embedding_scale"), 1.0f);
        m->attn_scale  = gguf_get_f32(g, AK("attention.scale"), 0.0f);
        m->resid_scale = gguf_get_f32(g, AK("residual_scale"), 1.0f);
        float ls = gguf_get_f32(g, AK("logit_scale"), 0.0f);
        if (ls > 0.0f) m->logit_scale = 1.0f / ls;
        if (!gguf_get_bool(g, AK("rope.scaling.finetuned"), true)) {
            fprintf(stderr, "error: granite export disables rope "
                    "(rope.scaling.finetuned=false) — unsupported layout\n");
            return false;
        }
    }
    if (strcmp(arch, "granitehybrid") == 0) {
        // Granite-4 h-series (granitehybrid): Mamba-2 recurrent layers
        // interleaved with GQA attention, per-layer typed off the
        // attention.head_count_kv ARRAY (0 => recurrent). Reference: llama.cpp
        // b10353 src/models/granite-hybrid.cpp + mamba-base.cpp
        // build_mamba2_layer + ggml_ssm_scan (scalar-per-head branch). The four
        // granite muP scalars apply exactly as the dense arch
        // (embedding/attention/residual/logit); the MoE FFN (softmax top-k over
        // fused experts + an always-on shared expert) is configured by the
        // generic MoE block below. Rope is gated by rope.scaling.finetuned —
        // NoPE when false, which the certified granite-4.0-h-small sets; when
        // true the attention layers rope adjacent-pair like dense granite.
        m->granite_hybrid = true;
        m->rope_neox   = false;
        m->embd_scale  = gguf_get_f32(g, AK("embedding_scale"), 1.0f);
        m->attn_scale  = gguf_get_f32(g, AK("attention.scale"), 0.0f);
        m->resid_scale = gguf_get_f32(g, AK("residual_scale"), 1.0f);
        float ls = gguf_get_f32(g, AK("logit_scale"), 0.0f);
        if (ls > 0.0f) m->logit_scale = 1.0f / ls;
        // rope gate: llama.cpp builds no positions and applies no rope when
        // rope.scaling.finetuned is false (granite-4.0-h sets it false, so its
        // four attention layers are NoPE — position comes from the Mamba
        // layers). Express that with the NoPE-every-layer knob; attention
        // temperature stays off (attn_temp_scale == 0), so model_layer_ropes
        // returning false is the whole effect. Set AFTER a value the generic
        // Llama-4 no_rope read below would otherwise clobber (that read now
        // defaults to the current value, preserving this — see its comment).
        if (!gguf_get_bool(g, AK("rope.scaling.finetuned"), true))
            m->no_rope_layer_step = 1;
        // Mamba-2 geometry (NOT the Gated DeltaNet ratios qwen35 checks):
        // inner = 2*n_embd = n_head * head_dim, groups partition the heads and
        // evenly divide the inner dim.
        m->ssm_conv_kernel = (int)gguf_get_u32(g, AK("ssm.conv_kernel"), 0);
        m->ssm_inner       = (int)gguf_get_u32(g, AK("ssm.inner_size"), 0);
        m->ssm_state       = (int)gguf_get_u32(g, AK("ssm.state_size"), 0);
        m->ssm_v_heads     = (int)gguf_get_u32(g, AK("ssm.time_step_rank"), 0);
        m->ssm_groups      = (int)gguf_get_u32(g, AK("ssm.group_count"), 0);
        if (m->ssm_conv_kernel <= 0 || m->ssm_conv_kernel > 8 ||
            m->ssm_inner <= 0 || m->ssm_inner > MDL_DIM_MAX ||
            m->ssm_state <= 0 || m->ssm_state > MDL_DIM_MAX ||
            m->ssm_v_heads <= 0 || m->ssm_groups <= 0 ||
            m->ssm_inner % m->ssm_v_heads != 0 ||
            m->ssm_v_heads % m->ssm_groups != 0 ||
            m->ssm_inner % m->ssm_groups != 0 ||
            m->ssm_inner != 2 * m->n_embd) {
            fprintf(stderr, "error: invalid granitehybrid Mamba-2 geometry "
                    "(conv_kernel=%d inner=%d state=%d heads=%d groups=%d, "
                    "n_embd=%d)\n", m->ssm_conv_kernel, m->ssm_inner,
                    m->ssm_state, m->ssm_v_heads, m->ssm_groups, m->n_embd);
            return false;
        }
        // Per-layer attention typing from the head_count_kv ARRAY. Recurrent
        // layers publish 0 (no KV rows); attention layers a real GQA count.
        // Build the per-layer geometry the KV allocator and attention path read
        // (identity head_dim, no shared KV), and set the scalar n_head_kv to a
        // representative attention value so the general geometry gate (which
        // predates the per-layer arrays) still passes.
        size_t nl = (size_t)(unsigned)m->n_layer;
        m->l_head_kv  = calloc(nl, sizeof(int));
        m->l_head_dim = calloc(nl, sizeof(int));
        m->l_rope_dim = calloc(nl, sizeof(int));
        if (!m->l_head_kv || !m->l_head_dim || !m->l_rope_dim) return false;
        int attn_kv = 0;
        for (int i = 0; i < m->n_layer; i++) {
            int kv = (int)gguf_get_u32_idx(g, AK("attention.head_count_kv"),
                                           (uint64_t)i, 0);
            m->l_head_kv[i]  = kv;
            m->l_head_dim[i] = m->head_dim;
            m->l_rope_dim[i] = m->rope_dim;
            if (kv > 0) {
                if (kv > m->n_head || m->n_head % kv != 0) {
                    fprintf(stderr, "error: granitehybrid blk.%d head_count_kv "
                            "%d does not divide n_head %d\n", i, kv, m->n_head);
                    return false;
                }
                if (attn_kv == 0) attn_kv = kv;
                else if (kv != attn_kv) {
                    fprintf(stderr, "error: granitehybrid attention layers "
                            "disagree on head_count_kv (%d vs %d)\n",
                            attn_kv, kv);
                    return false;
                }
            }
        }
        if (attn_kv == 0) {
            fprintf(stderr, "error: granitehybrid has no attention layers "
                    "(every attention.head_count_kv entry is 0)\n");
            return false;
        }
        m->n_head_kv = attn_kv;
        int n_attn = 0;
        for (int i = 0; i < m->n_layer; i++) if (m->l_head_kv[i] > 0) n_attn++;
        fprintf(stderr, "granitehybrid: Mamba-2 hybrid (%d recurrent + %d "
                "attention layers) — greedy output verified against llama.cpp "
                "b10353 on granite-4.0-h-small at Q4_K and Q8_0: token-"
                "identical on high-confidence prompts, diverging only at "
                "genuine near-ties (a ~0.03-nat argmax coin-flip at Q8_0; "
                "runner and oracle rank the same top-2 — math verified)\n",
                m->n_layer - n_attn, n_attn);
    }
    if (strcmp(arch, "nemotron_h") == 0 || strcmp(arch, "nemotron_h_moe") == 0) {
        bool nh_moe = strcmp(arch, "nemotron_h_moe") == 0;
        // Nemotron-H (NVIDIA Nemotron-Nano-9B-v2): a Mamba-2 / attention / MLP
        // hybrid. Each block is EXACTLY ONE of three kinds (mutually exclusive,
        // one pre-norm + one residual), typed off two per-layer arrays:
        //   recurrent (SSM) : head_count_kv[i] == 0 && feed_forward_length[i] == 0
        //   attention       : head_count_kv[i]  > 0  (feed_forward_length 0)
        //   MLP             : feed_forward_length[i] > 0
        // Unlike granitehybrid this family is NON-MoE (dense gate-less
        // squared-ReLU MLP) and carries NO muP scalars — embedding/attention/
        // residual/logit scales stay at their off defaults; attention kq_scale
        // is the plain 1/sqrt(head_dim). Reference: llama.cpp b10353
        // src/models/nemotron-h.cpp + mamba-base.cpp — the SAME ggml_ssm_scan
        // the granite path certifies, here exercising the n_group>1 grouped
        // B/C broadcast (Nano: n_group=8), which mamba2_ssd_step already
        // implements (g = h / (n_head/n_group)).
        m->nemotron_h = true;
        m->ffn_relu2  = true;      // gate-less MLP: down(relu(up(x))^2)
        m->rope_neox  = false;
        // NoPE: nemotron-h.cpp applies no rope to its attention layers (position
        // comes from the Mamba layers), and the export marks this with
        // rope.scaling.finetuned=false — the SAME gate llama.cpp's granite path
        // uses. A finetuned-rope export would rope the attention layers, a
        // layout this tracer has not certified: refuse rather than mis-rotate.
        if (gguf_get_bool(g, AK("rope.scaling.finetuned"), false)) {
            fprintf(stderr, "error: nemotron_h with rope.scaling.finetuned=true "
                    "(rope on attention layers) is not certified\n");
            return false;
        }
        m->no_rope_layer_step = 1;   // NoPE every layer
        // Mamba-2 geometry. NOTE: nemotron does NOT satisfy granite's
        // inner==2*n_embd (Nano-9B-v2: inner=10240, n_embd=4480), so that
        // assertion is deliberately absent here.
        m->ssm_conv_kernel = (int)gguf_get_u32(g, AK("ssm.conv_kernel"), 0);
        m->ssm_inner       = (int)gguf_get_u32(g, AK("ssm.inner_size"), 0);
        m->ssm_state       = (int)gguf_get_u32(g, AK("ssm.state_size"), 0);
        m->ssm_v_heads     = (int)gguf_get_u32(g, AK("ssm.time_step_rank"), 0);
        m->ssm_groups      = (int)gguf_get_u32(g, AK("ssm.group_count"), 0);
        if (m->ssm_conv_kernel <= 0 || m->ssm_conv_kernel > 8 ||
            m->ssm_inner <= 0 || m->ssm_inner > MDL_DIM_MAX ||
            m->ssm_state <= 0 || m->ssm_state > MDL_DIM_MAX ||
            m->ssm_v_heads <= 0 || m->ssm_groups <= 0 ||
            m->ssm_inner % m->ssm_v_heads != 0 ||
            m->ssm_v_heads % m->ssm_groups != 0 ||
            m->ssm_inner % m->ssm_groups != 0) {
            fprintf(stderr, "error: invalid nemotron_h Mamba-2 geometry "
                    "(conv_kernel=%d inner=%d state=%d heads=%d groups=%d)\n",
                    m->ssm_conv_kernel, m->ssm_inner, m->ssm_state,
                    m->ssm_v_heads, m->ssm_groups);
            return false;
        }
        // nemotron_h_moe (Nemotron-3.5 Lightning): the MLP blocks are a gate-less
        // squared-ReLU MoE — a router, gate-less routed experts (relu(up)^2, no
        // gate branch), and an always-on gate-less shared expert. Routing reuses
        // the general moe_route (group/scale/norm); only the activation differs.
        if (nh_moe) {
            m->n_expert        = (int)gguf_get_u32(g, AK("expert_count"), 0);
            m->n_expert_used   = (int)gguf_get_u32(g, AK("expert_used_count"), 0);
            m->n_ff_exp        = (int)gguf_get_u32(g, AK("expert_feed_forward_length"), 0);
            m->n_ff_shexp      = (int)gguf_get_u32(g, AK("expert_shared_feed_forward_length"), 0);
            m->n_expert_groups = (int)gguf_get_u32(g, AK("expert_group_count"), 1);
            m->n_group_used    = (int)gguf_get_u32(g, AK("expert_group_used_count"), 1);
            m->expert_gating   = (int)gguf_get_u32(g, AK("expert_gating_func"),
                                                   EXPERT_GATE_SOFTMAX);
            if (m->expert_gating == EXPERT_GATE_NONE)
                m->expert_gating = EXPERT_GATE_SOFTMAX;
            m->expert_w_scale  = gguf_get_f32(g, AK("expert_weights_scale"), 1.0f);
            m->expert_norm_w   = gguf_get_bool(g, AK("expert_weights_norm"), true);
            if (m->n_expert <= 0 || m->n_expert > 256 ||
                m->n_expert_used <= 0 || m->n_expert_used > m->n_expert ||
                m->n_ff_exp <= 0 || m->n_ff_exp > MDL_DIM_MAX ||
                m->n_ff_shexp < 0 || m->n_ff_shexp > MDL_DIM_MAX ||
                m->expert_gating < EXPERT_GATE_NONE ||
                m->expert_gating > EXPERT_GATE_SQRT_SOFTPLUS) {
                fprintf(stderr, "error: invalid nemotron_h_moe expert config "
                        "(count=%d used=%d exp_ff=%d shexp_ff=%d gating=%d)\n",
                        m->n_expert, m->n_expert_used, m->n_ff_exp,
                        m->n_ff_shexp, m->expert_gating);
                return false;
            }
        }
        // Per-layer typing off the head_count_kv ARRAY (0 => not attention).
        size_t nl = (size_t)(unsigned)m->n_layer;
        m->l_head_kv  = calloc(nl, sizeof(int));
        m->l_head_dim = calloc(nl, sizeof(int));
        m->l_rope_dim = calloc(nl, sizeof(int));
        if (!m->l_head_kv || !m->l_head_dim || !m->l_rope_dim) return false;
        int attn_kv = 0, n_attn = 0, n_rec = 0, n_mlp = 0, max_ff = 0;
        for (int i = 0; i < m->n_layer; i++) {
            int kv = (int)gguf_get_u32_idx(g, AK("attention.head_count_kv"),
                                           (uint64_t)i, 0);
            int ff = (int)gguf_get_u32_idx(g, AK("feed_forward_length"),
                                           (uint64_t)i, 0);
            m->l_head_kv[i]  = kv;
            m->l_head_dim[i] = m->head_dim;
            m->l_rope_dim[i] = m->rope_dim;
            if (kv > 0) {
                if (kv > m->n_head || m->n_head % kv != 0) {
                    fprintf(stderr, "error: nemotron_h blk.%d head_count_kv %d "
                            "does not divide n_head %d\n", i, kv, m->n_head);
                    return false;
                }
                if (attn_kv == 0) attn_kv = kv;
                else if (kv != attn_kv) {
                    fprintf(stderr, "error: nemotron_h attention layers disagree "
                            "on head_count_kv (%d vs %d)\n", attn_kv, kv);
                    return false;
                }
                n_attn++;
            } else if (ff > 0) { n_mlp++; if (ff > max_ff) max_ff = ff; }
            else n_rec++;
        }
        // feed_forward_length is a per-layer array with 0 on the SSM/attention
        // blocks; the generic array-max fallback bails on any 0 entry, so set
        // the model-wide FFN width (buffer sizing + the >0 hyperparam gate)
        // from the max over the MLP blocks here.
        m->n_ff = max_ff;
        m->ffn_var = true;
        if (attn_kv == 0) {
            fprintf(stderr, "error: nemotron_h has no attention layers\n");
            return false;
        }
        if (n_rec == 0) {
            fprintf(stderr, "error: nemotron_h has no recurrent (Mamba-2) "
                    "layers — not a hybrid\n");
            return false;
        }
        m->n_head_kv = attn_kv;
        if (nh_moe)
            fprintf(stderr, "nemotron_h_moe: Mamba-2 hybrid (%d recurrent + %d "
                    "attention + %d MoE layers, n_group=%d grouped scan) — %d/%d "
                    "gate-less squared-ReLU experts + always-on shared expert, "
                    "NoPE attention\n", n_rec, n_attn, n_mlp, m->ssm_groups,
                    m->n_expert_used, m->n_expert);
        else
            fprintf(stderr, "nemotron_h: Mamba-2 hybrid (%d recurrent + %d attention "
                    "+ %d MLP layers, n_group=%d grouped scan) — NON-MoE, NoPE "
                    "attention, gate-less squared-ReLU MLP; greedy output verified "
                    "against llama.cpp b10353 on Nemotron-Nano-9B-v2 Q8_0\n",
                    n_rec, n_attn, n_mlp, m->ssm_groups);
    }
    if (strcmp(arch, "muse-glimmer") == 0) {
        // muse-glimmer (Meta Muse Glimmer 30B). Transcribed from llama.cpp
        // b10353 src/models/muse-glimmer.cpp, not inferred:
        //   * weightless RMS norm on the embedding row, no embedding scale;
        //   * afmoe's output-gated attention (blk.N.attn_gate, sigmoid
        //     between SDPA out and o_proj) plus qwen3-style per-head QK
        //     norms (conversion folds qk_scale_factor into attn_q_norm and
        //     ships attn_k_norm as identity);
        //   * sandwich norms at a FIXED 1e-8 eps (llama.cpp hardcodes
        //     post_norm_eps) while the pre-norms keep the declared epsilon;
        //   * SWA with a per-layer bool pattern array (30B: 3 local : 1
        //     global); rope runs ONLY on the sliding layers — llama.cpp maps
        //     this arch to LLAMA_ROPE_TYPE_NORM, i.e. adjacent-pair like
        //     llama — at freq_base_swa, seeded from the MAIN base exactly
        //     like gpt-oss; the full-attention layers are NoPE;
        //   * final logits: x logit_scale (a required key), then softcap.
        m->attn_out_gate = true;
        m->nope_on_full  = true;
        m->embd_norm     = true;
        m->rope_neox     = false;
        m->post_norm_eps = 1e-8f;
        m->logit_softcap = gguf_get_f32(g, AK("final_logit_softcapping"), 0.0f);
        m->logit_scale   = gguf_get_f32(g, AK("logit_scale"), 0.0f);
        if (m->logit_scale <= 0.0f) {
            // llama.cpp reads the key as required; a build that silently
            // defaulted it would emit uncapped-scale logits on every token
            fprintf(stderr, "error: muse-glimmer requires %s\n",
                    AK("logit_scale"));
            return false;
        }
        m->swa_window = (int)gguf_get_u32(g, AK("attention.sliding_window"), 2048);
        int period = (int)gguf_get_u32(g, AK("attention.sliding_window_pattern"), 4);
        if (period < 1) period = 4;
        m->l_is_swa = calloc(m->n_layer, sizeof(bool));
        if (!m->l_is_swa) return false;
        if (!swa_pattern_array(g, AK("attention.sliding_window_pattern"),
                               m->l_is_swa, m->n_layer))
            for (int i = 0; i < m->n_layer; i++)
                m->l_is_swa[i] = m->swa_window > 0 && ((i + 1) % period) != 0;
        // sliding layers share the global rope regime unless the export
        // declares a genuinely different local base
        float sbase = gguf_get_f32(g, AK("rope.freq_base_swa"), 0.0f);
        m->swa_rope_global = sbase <= 0.0f || sbase == m->rope_base;
        // Reasoning is a separate assistant turn addressed to the model
        // itself: ` to=self<|message|>THINKING<|eom|><|start|>assistant
        // to=RECIPIENT<|message|>ANSWER`. `<|eom|>`, `<|start|>` and
        // `<|message|>` decode to no bytes, so on an unconstrained byte
        // stream the pair below is exactly what brackets the thinking text
        // and nothing else marks where the answer begins. Narrowing the
        // close to ` to=` (to make tool recipients closable) leaked the
        // recipient word into content and `assistant` into reasoning on
        // plain thinking chats — measured live, content came back "user391".
        // Constrained runs never wait for these bytes: engine.c recognizes
        // the <|eom|> control id and feeds this close to the splitter
        // itself (constraint_finish_think), then the recipient-turn
        // automaton owns everything that follows.
        m->think_open  = " to=self";
        m->think_close = "assistant to=user";
    }
    // The sandwich norms run at rms_eps everywhere except where an arch block
    // above pinned them (muse-glimmer's fixed 1e-8); resolve the default after
    // every block that can still change rms_eps has run.
    if (m->post_norm_eps <= 0.0f) m->post_norm_eps = m->rms_eps;
    // "This layer slides" and "the window is 0" cannot both hold, and the
    // forward pass cannot express the pair: a sliding layer selects the local
    // rope table, which rope_setup only builds when there IS a window (NULL,
    // dereferenced on the first token), and its attended range starts one past
    // its last position. The pattern and the window are separate keys and the
    // arch blocks above read them in several different orders, so resolve it
    // here, once, for all of them: no window means no sliding layers.
    if (m->l_is_swa && m->swa_window <= 0)
        for (int i = 0; i < m->n_layer; i++) m->l_is_swa[i] = false;
    // Llama-4 attention knobs, off unless the GGUF asks for them. The default
    // preserves any value an arch block above already set (granitehybrid marks
    // every layer NoPE via step 1 when rope.scaling.finetuned is false); every
    // other arch leaves the field at its zero-init, so behavior is unchanged.
    m->no_rope_layer_step   = (int)gguf_get_u32(g, AK("attention.no_rope_layer_step"),
                                                (uint32_t)m->no_rope_layer_step);
    // afmoe's GGUF carries no no_rope key; llama.cpp defaults the step to the
    // SWA period (4) for this arch, so the global layers are the NoPE layers.
    // muse-glimmer also gates its attention but derives NoPE from the pattern
    // array (nope_on_full) — the afmoe default must not stack onto it.
    if (m->attn_out_gate && !m->nope_on_full && m->no_rope_layer_step == 0)
        m->no_rope_layer_step = 4;
    m->attn_temp_floor_scale = (int)gguf_get_u32(g, AK("attention.attn_temp_floor_scale"), 0);
    m->attn_temp_scale       = gguf_get_f32(g, AK("attention.attn_temp_scale"), 0.0f);
    m->attn_temp_offset      = gguf_get_f32(g, AK("attention.attn_temp_offset"), 1.0f);
    if (m->no_rope_layer_step < 0) {
        fprintf(stderr, "error: negative no_rope_layer_step\n");
        return false;
    }
    // Bounded as a u32, before the cast: above INT_MAX this lands negative,
    // and a negative count is neither "> 0" (so the MoE block is skipped) nor
    // "== 0" (so the dense FFN is skipped too) — the layer then loaded no FFN
    // at all and the forward pass ran a matvec on a NULL tensor.
    uint32_t n_expert_raw = gguf_get_u32(g, AK("expert_count"), 0);
    if (n_expert_raw > 256) {
        fprintf(stderr, "error: expert_count %u is out of range (max 256)\n",
                n_expert_raw);
        return false;
    }
    m->n_expert = (int)n_expert_raw;
    if (m->n_expert > 0 && !m->nemotron_h) {
        // sparse-MoE (Mixtral / Qwen3-MoE): softmax-over-all router, top-k
        // selection, renormalized weights, per-expert SwiGLU, weighted sum.
        // nemotron_h_moe is EXCLUDED: its gate-less squared-ReLU MoE (routed +
        // an always-on gate-less shared expert) is configured wholly in the
        // nemotron_h loader above; this block's ffn_gate_shexp probe and the
        // SiLU-only guard below would wrongly reject it.
        m->n_expert_used = (int)gguf_get_u32(g, AK("expert_used_count"), 0);
        // Mixtral omits expert_feed_forward_length and uses feed_forward_length
        m->n_ff_exp = (int)gguf_get_u32(g, AK("expert_feed_forward_length"), m->n_ff);
        if (m->n_expert_used < 1 || m->n_expert_used > m->n_expert ||
            m->n_ff_exp <= 0 || m->n_ff_exp > MDL_DIM_MAX || m->n_expert > 256) {
            fprintf(stderr, "error: invalid MoE geometry (experts=%d used=%d ff_exp=%d)\n",
                    m->n_expert, m->n_expert_used, m->n_ff_exp);
            return false;
        }
        // Generalized router knobs. Every default below reproduces the
        // Mixtral/Qwen3 path exactly, so a GGUF that declares none of them
        // routes bit-for-bit as before.
        m->expert_gating   = (int)gguf_get_u32(g, AK("expert_gating_func"),
                                               EXPERT_GATE_SOFTMAX);
        if (m->expert_gating == EXPERT_GATE_NONE)
            m->expert_gating = EXPERT_GATE_SOFTMAX;
        m->n_expert_groups = (int)gguf_get_u32(g, AK("expert_group_count"), 1);
        m->n_group_used    = (int)gguf_get_u32(g, AK("expert_group_used_count"), 1);
        m->expert_w_scale  = gguf_get_f32(g, AK("expert_weights_scale"), 1.0f);
        m->expert_norm_w   = gguf_get_bool(g, AK("expert_weights_norm"), true);
        if (m->expert_gating < EXPERT_GATE_NONE ||
            m->expert_gating > EXPERT_GATE_SQRT_SOFTPLUS) {
            fprintf(stderr, "error: unknown expert_gating_func %d\n",
                    m->expert_gating);
            return false;
        }
        // Group-limited top-k needs the experts to divide evenly into groups
        // and to keep at least enough groups to hold n_expert_used picks.
        if (m->n_expert_groups < 1 || m->n_group_used < 1 ||
            m->n_group_used > m->n_expert_groups ||
            m->n_expert % m->n_expert_groups != 0 ||
            (long)m->n_group_used * (m->n_expert / m->n_expert_groups)
                < m->n_expert_used) {
            fprintf(stderr, "error: invalid MoE expert grouping "
                    "(experts=%d groups=%d used_groups=%d used=%d)\n",
                    m->n_expert, m->n_expert_groups, m->n_group_used,
                    m->n_expert_used);
            return false;
        }

        // Shared always-on expert (Qwen2-MoE / DeepSeek): a dense FFN over the
        // same normed input, summed with the routed output. Supported when the
        // tensors are present; the width falls back to the routed expert width
        // exactly as llama.cpp does.
        char shexp_probe[64];
        // probe the first MoE layer: afmoe's leading blocks are dense and
        // carry no shared-expert tensors at all
        snprintf(shexp_probe, sizeof shexp_probe,
                 "blk.%d.ffn_gate_shexp.weight", m->n_dense_lead);
        if (gguf_find_tensor(g, shexp_probe)) {
            int nsh = (int)gguf_get_u32(g, AK("expert_shared_count"), 1);
            if (nsh < 1) nsh = 1;
            // afmoe publishes no shexp width key: the shared expert is
            // expert_shared_count routed-expert widths wide, per llama.cpp.
            // Both factors come from the file, so the product is computed wide
            // — as int it overflowed (UBSan: "signed integer overflow: 64 *
            // 1073741824") before the width could be judged out of range.
            int64_t dflt = (int64_t)m->n_ff_exp * nsh;
            if (dflt > MDL_DIM_MAX) dflt = MDL_DIM_MAX + 1;   // refused below
            m->n_ff_shexp = (int)gguf_get_u32(
                g, AK("expert_shared_feed_forward_length"), (uint32_t)dflt);
            if (m->n_ff_shexp > MDL_DIM_MAX) {
                fprintf(stderr, "error: shared-expert FFN width %d is out of "
                        "range (expert_shared_count=%d, expert width=%d)\n",
                        m->n_ff_shexp, nsh, m->n_ff_exp);
                return false;
            }
        }
        else if (gguf_get_u32(g, AK("expert_shared_count"), 0) > 0) {
            fprintf(stderr, "error: expert_shared_count is set but the shared "
                    "expert tensors are absent\n");
            return false;
        }
        // GELU-gated MoE is implemented only for gemma-4's dual-branch layout
        // (fused gate_up experts + a dense shared FFN per layer). Any other
        // non-SiLU MoE is refused rather than run through SiLU math.
        bool gemma_moe = strcmp(arch, "gemma4") == 0 &&
                         gguf_find_tensor(g, "blk.0.ffn_gate_up_exps.weight");
        // gpt-oss's ACT_SWIGLU_OAI is implemented in BOTH CPU MoE paths
        // (per-token and grouped prefill), so it is admitted here by name
        // rather than widening the guard to "any non-SiLU".
        if (m->ffn_act != ACT_SILU && !gemma_moe && !m->gptoss) {
            fprintf(stderr, "error: MoE is only supported with SiLU-gated "
                    "experts (this model uses a different activation)\n");
            return false;
        }
    }
    if (!m->qwen35 && !m->granite_hybrid && !m->nemotron_h &&
        gguf_get(g, AK("ssm.conv_kernel"))) {
        fprintf(stderr, "error: '%s' is a hybrid SSM/attention architecture — "
                "only pure-transformer llama-family models are supported\n", arch);
        return false;
    }
    if (m->n_layer <= 0 || m->n_embd <= 0 || m->n_head <= 0 || m->n_ff <= 0) {
        fprintf(stderr, "error: missing model hyperparameters for arch '%s'\n", arch);
        return false;
    }
    // context_length is read as a u32 into an int, so anything above INT_MAX
    // arrives NEGATIVE — and it caps the default window, sizes the cache, and
    // seeds the YaRN extension ratio. The load did fail, but with "cannot
    // allocate buffers", which describes the machine rather than the file.
    if (m->n_ctx_train < 1) {
        fprintf(stderr, "error: '%s' declares an unusable context_length "
                "(%u)\n", arch, gguf_get_u32(g, AK("context_length"), 0));
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
    if (m->head_dim < 1        || m->head_dim > MDL_DIM_MAX ||
        m->n_embd   > MDL_DIM_MAX || m->n_ff  > MDL_DIM_MAX ||
        m->n_head   > MDL_DIM_MAX || m->n_layer > 100000 ||
        m->rope_dim < 0        || m->rope_dim > m->head_dim ||
        m->n_head_kv < 1       || m->n_head_kv > m->n_head ||
        m->n_head % m->n_head_kv != 0 ||
        (int64_t)m->n_head * m->head_dim > MDL_DIM_MAX) {
        fprintf(stderr, "error: invalid model geometry for arch '%s' "
                "(head_dim=%d rope_dim=%d n_head=%d n_head_kv=%d "
                "n_embd=%d n_ff=%d n_layer=%d)\n",
                arch, m->head_dim, m->rope_dim, m->n_head, m->n_head_kv,
                m->n_embd, m->n_ff, m->n_layer);
        return false;
    }

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
    m->out_norm_w = tensor_to_f32(out_norm, m->n_embd, &ok);
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

    if (m->n_embd_ple > 0) {
        // Gemma-4 E-series per-layer embeddings. The table holds one
        // n_embd_ple slice per layer per token, concatenated along ne[0].
        // The bound comes FIRST: it used to sit one line below the
        // multiplication it guards, so the overflow had already happened by
        // the time it was tested (UBSan caught the int product).
        if (m->n_embd_ple > MDL_DIM_MAX ||
            m->n_embd_ple > INT_MAX / m->n_layer) {
            fprintf(stderr, "error: gemma4 per-layer embedding size (%d x %d "
                    "layers) is out of range\n", m->n_embd_ple, m->n_layer);
            return false;
        }
        int per_tok = m->n_layer * m->n_embd_ple;
        m->ple_tok_embd   = need_tensor(g, "per_layer_token_embd.weight", 0, &ok);
        m->ple_model_proj = need_tensor(g, "per_layer_model_proj.weight", 0, &ok);
        gguf_tensor *pn   = need_tensor(g, "per_layer_proj_norm.weight", 0, &ok);
        if (!ok) return false;
        if (!check_shape(m->ple_tok_embd, per_tok, m->n_vocab,
                         "per_layer_token_embd.weight", 0) ||
            !check_shape(m->ple_model_proj, m->n_embd, per_tok,
                         "per_layer_model_proj.weight", 0) ||
            !check_shape(pn, m->n_embd_ple, 1, "per_layer_proj_norm.weight", 0))
            return false;
        m->ple_proj_norm = tensor_to_f32(pn, m->n_embd_ple, &ok);
        if (!ok) return false;
    }

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
        if (m->nemotron_h) {
            // Nemotron-H blocks are three mutually-exclusive kinds with a single
            // norm and no ffn_norm tensor; bind each fully and skip the shared
            // recurrent/attention/FFN binding below.
            if (!nemotron_bind_layer(m, g, l, i)) return false;
            continue;
        }
        // per-layer FFN width: the ARRAY form answers per index, the scalar
        // form (every other export) answers every index with m->n_ff
        l->n_ff = (int)gguf_get_u32_idx(g, AK("feed_forward_length"),
                                        (uint64_t)i, (uint32_t)m->n_ff);
        if (l->n_ff <= 0 || l->n_ff > m->n_ff) {
            fprintf(stderr, "error: layer %d FFN width %d outside (0, %d]\n",
                    i, l->n_ff, m->n_ff);
            return false;
        }
        gguf_tensor *an = need_tensor(g, "blk.%d.attn_norm.weight", i, &ok);
        // gpt-oss shares qwen35's shape here: post_attention_norm IS the FFN
        // input norm and there is no ffn_norm tensor at all.
        gguf_tensor *fn = (m->qwen35 || m->gptoss)
            ? need_tensor(g, "blk.%d.post_attention_norm.weight", i, &ok)
            : need_tensor(g, "blk.%d.ffn_norm.weight", i, &ok);
        if (m->gptoss)
            l->attn_sinks = tensor_to_f32(
                need_tensor(g, "blk.%d.attn_sinks.weight", i, &ok),
                m->n_head, &ok);
        l->recurrent = m->qwen35 ? ((i + 1) % m->full_attn_interval != 0)
                     : m->granite_hybrid ? (m->l_head_kv && m->l_head_kv[i] == 0)
                     : false;
        if (l->recurrent && m->qwen35) {
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
            gguf_tensor *sa = need_tensor(g, "blk.%d.ssm_a", i, &ok);
            gguf_tensor *sn = need_tensor(g, "blk.%d.ssm_norm.weight", i, &ok);
            l->w_gate = need_tensor(g, "blk.%d.ffn_gate.weight", i, &ok);
            l->w_up   = need_tensor(g, "blk.%d.ffn_up.weight", i, &ok);
            l->w_down = need_tensor(g, "blk.%d.ffn_down.weight", i, &ok);
            if (!ok) return false;
            // The DeltaNet geometry keys index every tensor above, and until
            // this check nothing tied the two together: ssm.inner_size,
            // state_size, group_count and time_step_rank passed only an
            // internal ratio test, so a file could declare a wider
            // convolution or more heads than its own tensors hold and the
            // recurrence would read past them.
            {
                int keydim  = m->ssm_state * m->ssm_groups;
                int convdim = 2 * keydim + m->ssm_inner;
                int hv      = m->ssm_inner / m->ssm_v_heads;
                if (!check_shape(l->wqkv, m->n_embd, convdim, "attn_qkv", i) ||
                    !check_shape(l->wq_gate, m->n_embd, m->ssm_inner,
                                 "attn_gate", i) ||
                    // read as convdim rows of conv_kernel, one per channel
                    !check_shape(l->ssm_conv, m->ssm_conv_kernel, convdim,
                                 "ssm_conv1d", i) ||
                    !check_shape(l->ssm_beta, m->n_embd, m->ssm_v_heads,
                                 "ssm_beta", i) ||
                    !check_shape(l->ssm_alpha, m->n_embd, m->ssm_v_heads,
                                 "ssm_alpha", i) ||
                    !check_shape(l->ssm_out, m->ssm_inner, m->n_embd,
                                 "ssm_out", i) ||
                    !check_shape(dt, m->ssm_v_heads, 1, "ssm_dt", i) ||
                    !check_shape(sa, m->ssm_v_heads, 1, "ssm_a", i) ||
                    !check_shape(sn, hv, 1, "ssm_norm", i) ||
                    !check_shape(l->w_gate, m->n_embd, l->n_ff, "ffn_gate", i) ||
                    !check_shape(l->w_up, m->n_embd, l->n_ff, "ffn_up", i) ||
                    !check_shape(l->w_down, l->n_ff, m->n_embd, "ffn_down", i))
                    return false;
            }
            l->ssm_dt     = tensor_to_f32(dt, m->ssm_v_heads, &ok);
            l->ssm_a      = tensor_to_f32(sa, m->ssm_v_heads, &ok);
            l->ssm_norm_w = tensor_to_f32(sn, m->ssm_inner / m->ssm_v_heads, &ok);
            if (!ok) return false;
            l->attn_norm_w = tensor_to_f32(an, m->n_embd, &ok);
            l->ffn_norm_w = tensor_to_f32(fn, m->n_embd, &ok);
            if (!ok) return false;
            l->out_scale = 1.0f;
            continue;
        }
        if (l->recurrent && m->granite_hybrid) {
            // Granite-4 h-series Mamba-2 mixer. attn_norm (an) is bound above;
            // ffn_norm (fn) is this layer's MoE input norm and is converted in
            // the shared norm tail below, so this branch deliberately does NOT
            // continue — it binds only the mixer and falls through to the
            // shared FFN/MoE binding. Tensor shapes per llama.cpp b10353
            // granite-hybrid.cpp::load_arch_tensors.
            int nh = m->ssm_v_heads, ds = m->ssm_state, ng = m->ssm_groups;
            int inner = m->ssm_inner;
            int conv_dim  = inner + 2 * ng * ds;
            int d_in_proj = 2 * inner + 2 * ng * ds + nh;
            l->ssm_in   = need_tensor(g, "blk.%d.ssm_in.weight", i, &ok);
            l->ssm_conv = need_tensor(g, "blk.%d.ssm_conv1d.weight", i, &ok);
            gguf_tensor *sconvb = need_tensor(g, "blk.%d.ssm_conv1d.bias", i, &ok);
            gguf_tensor *sdtb   = need_tensor(g, "blk.%d.ssm_dt.bias", i, &ok);
            gguf_tensor *sa2    = need_tensor(g, "blk.%d.ssm_a", i, &ok);
            gguf_tensor *sd2    = need_tensor(g, "blk.%d.ssm_d", i, &ok);
            gguf_tensor *sn2    = need_tensor(g, "blk.%d.ssm_norm.weight", i, &ok);
            l->ssm_out  = need_tensor(g, "blk.%d.ssm_out.weight", i, &ok);
            if (!ok) return false;
            // Tie every mixer tensor to the ssm.* geometry so a mislabelled or
            // truncated hybrid fails closed here, not by reading past the map
            // in the scan (the hostile-GGUF discipline the admission stub kept).
            if (!check_shape(l->ssm_in, m->n_embd, d_in_proj, "ssm_in", i) ||
                !check_shape(l->ssm_conv, m->ssm_conv_kernel, conv_dim,
                             "ssm_conv1d", i) ||
                !check_shape(sconvb, conv_dim, 1, "ssm_conv1d.bias", i) ||
                !check_shape(sdtb, nh, 1, "ssm_dt.bias", i) ||
                !check_shape(sa2, 1, nh, "ssm_a", i) ||
                !check_shape(sd2, 1, nh, "ssm_d", i) ||
                !check_shape(sn2, inner, 1, "ssm_norm", i) ||
                !check_shape(l->ssm_out, inner, m->n_embd, "ssm_out", i))
                return false;
            l->ssm_conv1d_b = tensor_to_f32(sconvb, conv_dim, &ok);
            l->ssm_dt       = tensor_to_f32(sdtb, nh, &ok);
            l->ssm_a        = tensor_to_f32(sa2, nh, &ok);
            l->ssm_d        = tensor_to_f32(sd2, nh, &ok);
            l->ssm_norm_w   = tensor_to_f32(sn2, inner, &ok);
            if (!ok) return false;
            l->out_scale = 1.0f;
        }
        if (!l->recurrent) {
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
                (int64_t)gu->ne[1] != 2 * (int64_t)l->n_ff) {
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
            if (m->n_expert == 0 || i < m->n_dense_lead) {
                // Apertus has no ffn_gate: its MLP is up -> xielu -> down.
                // Every other dense arch here is gated, so the tensor stays
                // required unless the activation is the ungated one.
                l->w_gate = m->ffn_act == ACT_XIELU
                            ? opt_tensor(g, "blk.%d.ffn_gate.weight", i)
                            : need_tensor(g, "blk.%d.ffn_gate.weight", i, &ok);
                l->w_up   = need_tensor(g, "blk.%d.ffn_up.weight", i, &ok);
            }
        }
        l->wo     = need_tensor(g, "blk.%d.attn_output.weight", i, &ok);
        if (m->attn_out_gate) {
            l->wq_gate = need_tensor(g, "blk.%d.attn_gate.weight", i, &ok);
            // same shape as the plain Q projection; a mismatch means a
            // conversion this code has never seen — refuse, not garble
            if (l->wq_gate && l->wq &&
                (l->wq_gate->ne[0] != l->wq->ne[0] ||
                 l->wq_gate->ne[1] != l->wq->ne[1])) {
                fprintf(stderr, "error: attn_gate shape differs from attn_q "
                        "in layer %d\n", i);
                ok = false;
            }
        }
        }  // end !l->recurrent attention binding
        if (m->n_expert > 0 && i >= m->n_dense_lead) {
            // sparse-MoE FFN: a router plus fused 3D expert tensors replace the
            // dense gate/up/down for this layer
            l->is_moe = true;
            if (m->n_ff_shexp > 0) {
                l->w_gate_shexp = need_tensor(g, "blk.%d.ffn_gate_shexp.weight", i, &ok);
                l->w_up_shexp   = need_tensor(g, "blk.%d.ffn_up_shexp.weight", i, &ok);
                l->w_down_shexp = need_tensor(g, "blk.%d.ffn_down_shexp.weight", i, &ok);
                // optional: Qwen2-MoE gates the branch, DeepSeek does not
                l->ffn_gate_inp_shexp =
                    opt_tensor(g, "blk.%d.ffn_gate_inp_shexp.weight", i);
            }
            l->ffn_gate_inp  = need_tensor(g, "blk.%d.ffn_gate_inp.weight", i, &ok);
            if (!ok) return false;
            // THIS layer's expert count, resolved here because every
            // per-expert vector below is indexed with it and must therefore
            // state it. It comes from the layer's own router rather than the
            // model-wide expert_count: --prune-experts writes a shorter router
            // (and shorter expert tensors) for a pruned layer — same names,
            // same widths, fewer blocks.
            if ((int64_t)l->ffn_gate_inp->ne[0] != m->n_embd) {
                fprintf(stderr, "error: ffn_gate_inp in blk.%d has ne[0]=%llu, "
                        "expected %d\n", i,
                        (unsigned long long)l->ffn_gate_inp->ne[0], m->n_embd);
                return false;
            }
            l->n_expert = (int)l->ffn_gate_inp->ne[1];
            if (l->n_expert < m->n_expert_used || l->n_expert > m->n_expert) {
                fprintf(stderr, "error: blk.%d declares %d experts via "
                        "ffn_gate_inp, outside [n_expert_used=%d, "
                        "expert_count=%d]\n", i, l->n_expert,
                        m->n_expert_used, m->n_expert);
                return false;
            }
            // Optional selection-only bias; absent on every arch certified so
            // far. DeepSeek exports name it .weight; afmoe's converter renames
            // expert_bias so the GGUF carries a .bias suffix.
            gguf_tensor *epb = opt_tensor(g, "blk.%d.exp_probs_b.weight", i);
            if (!epb) epb = opt_tensor(g, "blk.%d.exp_probs_b.bias", i);
            l->exp_probs_b = tensor_to_f32(epb, l->n_expert, &ok);
            l->ffn_gate_up_exps = opt_tensor(g, "blk.%d.ffn_gate_up_exps.weight", i);
            if (l->ffn_gate_up_exps) {
                // gemma-4 dual-branch MoE: gate+up fused in one 3D tensor, a
                // per-expert down scale, and — every MoE layer ALSO runs a dense
                // GELU FFN as a shared expert, plus two branch sandwich norms.
                l->moe_gemma = true;
                m->moe_gemma = true;   // dual-branch: dense shared FFN + routed experts
                l->ffn_down_exps   = need_tensor(g, "blk.%d.ffn_down_exps.weight", i, &ok);
                l->down_exps_scale = tensor_to_f32(opt_tensor(g, "blk.%d.ffn_down_exps.scale", i),
                                                   l->n_expert, &ok);
                l->gate_inp_scale  = tensor_to_f32(opt_tensor(g, "blk.%d.ffn_gate_inp.scale", i),
                                                   m->n_embd, &ok);
                l->w_gate = need_tensor(g, "blk.%d.ffn_gate.weight", i, &ok);
                l->w_up   = need_tensor(g, "blk.%d.ffn_up.weight",   i, &ok);
                l->w_down = need_tensor(g, "blk.%d.ffn_down.weight", i, &ok);
                l->ffn_post_norm1_w = tensor_to_f32(opt_tensor(g, "blk.%d.post_ffw_norm_1.weight", i),
                                                    m->n_embd, &ok);
                l->ffn_pre_norm2_w  = tensor_to_f32(opt_tensor(g, "blk.%d.pre_ffw_norm_2.weight",  i),
                                                    m->n_embd, &ok);
                l->ffn_post_norm2_w = tensor_to_f32(opt_tensor(g, "blk.%d.post_ffw_norm_2.weight", i),
                                                    m->n_embd, &ok);
            } else if ((l->ffn_gate_exps = opt_tensor(g, "blk.%d.ffn_gate_exps.weight", i))) {
                // modern fused 3D expert tensors
                l->ffn_up_exps   = need_tensor(g, "blk.%d.ffn_up_exps.weight", i, &ok);
                l->ffn_down_exps = need_tensor(g, "blk.%d.ffn_down_exps.weight", i, &ok);
                if (m->gptoss) {
                    // gpt-oss router + per-expert FFN biases (all F32). The
                    // expert biases are indexed as [expert][width], so they
                    // must cover the whole rectangle, not just one expert.
                    int64_t ne_l = l->n_expert;
                    l->ffn_gate_inp_b  = tensor_to_f32(
                        need_tensor(g, "blk.%d.ffn_gate_inp.bias", i, &ok),
                        ne_l, &ok);
                    l->ffn_gate_exps_b = tensor_to_f32(
                        need_tensor(g, "blk.%d.ffn_gate_exps.bias", i, &ok),
                        ne_l * m->n_ff_exp, &ok);
                    l->ffn_up_exps_b   = tensor_to_f32(
                        need_tensor(g, "blk.%d.ffn_up_exps.bias", i, &ok),
                        ne_l * m->n_ff_exp, &ok);
                    l->ffn_down_exps_b = tensor_to_f32(
                        need_tensor(g, "blk.%d.ffn_down_exps.bias", i, &ok),
                        ne_l * m->n_embd, &ok);
                }
            } else {
                // legacy split layout: one 2D tensor per expert (older Mixtral)
                l->moe_split = true;
                // This layout has no pruned form: the loop below wires exactly
                // expert_count tensors, so the router must have that many rows.
                if (l->n_expert != m->n_expert) {
                    fprintf(stderr, "error: blk.%d has the split expert layout "
                            "with a %d-row router, but expert_count is %d\n",
                            i, l->n_expert, m->n_expert);
                    return false;
                }
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
            // Recurrent (Mamba-2) layers carry no attention projections; their
            // mixer tensors were shape-checked at bind time. Only the MoE
            // checks below apply to them.
            if (!l->recurrent) {
            if (!check_shape(l->wo, qd, m->n_embd, "attn_output", i))
                return false;
            if (!fused_qkv && !m->qwen35) {
                if (!check_shape(l->wq, m->n_embd, qd, "attn_q", i) ||
                    !check_shape(l->wk, m->n_embd, kd, "attn_k", i) ||
                    !check_shape(l->wv, m->n_embd, kd, "attn_v", i))
                    return false;
                if (!l->is_moe &&
                    (!check_shape(l->w_gate, m->n_embd, l->n_ff, "ffn_gate", i) ||
                     !check_shape(l->w_up,   m->n_embd, l->n_ff, "ffn_up",   i)))
                    return false;
            }
            // qwen35's full-attention layers are the same layout with Q and its
            // output gate fused into one tensor of 2*q_dim rows — a different
            // width, not an unvalidated one.
            if (m->qwen35 &&
                (!check_shape(l->wq, m->n_embd, 2 * qd, "attn_q", i) ||
                 !check_shape(l->wk, m->n_embd, kd, "attn_k", i) ||
                 !check_shape(l->wv, m->n_embd, kd, "attn_v", i)))
                return false;
            }  // end !l->recurrent attention shape checks
            if (!l->is_moe &&
                !check_shape(l->w_down, l->n_ff, m->n_embd, "ffn_down", i))
                return false;
            if (l->is_moe) {
                // The MoE forward path slices each expert from the fused 3D
                // tensors (or the array of 2D split tensors) using offsets
                // computed from n_embd/n_ff_exp/n_expert metadata, NOT the
                // tensors' own ne[]. Validate that geometry here so the offsets
                // stay in bounds (RNC-1). The router reads n_expert rows of
                // length n_embd; each expert's gate/up is {n_embd, n_ff_exp}
                // and down is {n_ff_exp, n_embd}.
                //
                // l->n_expert is THIS layer's real count, resolved from its own
                // router tensor when it was loaded (above) rather than assumed
                // to equal the model-wide m->n_expert: --prune-experts writes a
                // shorter router (and shorter fused expert tensors) for a
                // pruned layer, same names, same n_embd/n_ff_exp, fewer expert
                // blocks. Every expert tensor in the layer must agree on this
                // count exactly (check_shape3 below requires ne[2] ==
                // l->n_expert, not merely >=).
                // The always-on shared branch is driven by n_ff_shexp, which
                // is metadata (expert_shared_feed_forward_length, else the
                // routed width times expert_shared_count) and decides how many
                // rows of gate/up the FFN reads and how long a down row is.
                // Nothing tied it to the tensors it indexes, so a width past
                // them read off the end of the mapping.
                if (m->n_ff_shexp > 0 &&
                    (!check_shape(l->w_gate_shexp, m->n_embd, m->n_ff_shexp,
                                  "ffn_gate_shexp", i) ||
                     !check_shape(l->w_up_shexp, m->n_embd, m->n_ff_shexp,
                                  "ffn_up_shexp", i) ||
                     !check_shape(l->w_down_shexp, m->n_ff_shexp, m->n_embd,
                                  "ffn_down_shexp", i) ||
                     !check_shape(l->ffn_gate_inp_shexp, m->n_embd, 1,
                                  "ffn_gate_inp_shexp", i)))
                    return false;
                if (l->moe_gemma) {
                    // gate and up are fused: {n_embd, 2*n_ff_exp, n_expert}.
                    // The dual branch also runs an ORDINARY dense FFN at this
                    // layer's feed_forward_length, and the dense shape checks
                    // below are guarded by !is_moe — which this layer is not —
                    // so its three tensors state their geometry here instead.
                    if (!check_shape3(l->ffn_gate_up_exps, m->n_embd,
                                      2 * m->n_ff_exp, l->n_expert,
                                      "ffn_gate_up_exps", i) ||
                        !check_shape3(l->ffn_down_exps, m->n_ff_exp, m->n_embd,
                                      l->n_expert, "ffn_down_exps", i) ||
                        !check_shape(l->w_gate, m->n_embd, l->n_ff, "ffn_gate", i) ||
                        !check_shape(l->w_up, m->n_embd, l->n_ff, "ffn_up", i) ||
                        !check_shape(l->w_down, l->n_ff, m->n_embd, "ffn_down", i))
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
                                      l->n_expert, "ffn_gate_exps", i) ||
                        !check_shape3(l->ffn_up_exps, m->n_embd, m->n_ff_exp,
                                      l->n_expert, "ffn_up_exps", i) ||
                        !check_shape3(l->ffn_down_exps, m->n_ff_exp, m->n_embd,
                                      l->n_expert, "ffn_down_exps", i))
                        return false;
                }
            }
        }
        // Each of these is read at a count the geometry fixes, not at its
        // own length: the norms over n_embd or one head, the projection biases
        // over the rows of the matvec they ride (qwen35's Q bias covers the
        // fused Q+gate, so 2*q_dim).
        int qd_l = model_q_dim(m, i), kd_l = model_kv_dim(m, i);
        l->attn_norm_w = tensor_to_f32(an, m->n_embd, &ok);
        l->ffn_norm_w  = tensor_to_f32(fn, m->n_embd, &ok);
        l->bq = tensor_to_f32(opt_tensor(g, "blk.%d.attn_q.bias", i),
                              m->qwen35 ? 2 * (int64_t)qd_l : qd_l, &ok);
        l->bk = tensor_to_f32(opt_tensor(g, "blk.%d.attn_k.bias", i), kd_l, &ok);
        l->bv = tensor_to_f32(opt_tensor(g, "blk.%d.attn_v.bias", i), kd_l, &ok);
        l->bo = tensor_to_f32(opt_tensor(g, "blk.%d.attn_output.bias", i),
                              m->n_embd, &ok);
        l->qnorm_w = tensor_to_f32(opt_tensor(g, "blk.%d.attn_q_norm.weight", i),
                                   model_head_dim(m, i), &ok);
        l->knorm_w = tensor_to_f32(opt_tensor(g, "blk.%d.attn_k_norm.weight", i),
                                   model_head_dim(m, i), &ok);
        // Qwen3.5's post_attention_norm is the FFN input norm (loaded as
        // ffn_norm_w above), not a sandwich norm on the attention projection.
        // qwen35 AND gpt-oss already consumed post_attention_norm as the FFN
        // input norm above; loading it here too would apply it a second time
        // to the attention output (the gemma-style placement) and quietly
        // corrupt every layer.
        l->post_attn_norm_w = (m->qwen35 || m->gptoss) ? NULL
            : tensor_to_f32(opt_tensor(g, "blk.%d.post_attention_norm.weight", i),
                            m->n_embd, &ok);
        l->post_ffn_norm_w  = tensor_to_f32(opt_tensor(g, "blk.%d.post_ffw_norm.weight", i),
                                            m->n_embd, &ok);
        if (m->n_embd_ple > 0) {
            // blk.N.post_norm is the PLE branch's own norm — attention and FFN
            // carry post_attention_norm / post_ffw_norm separately.
            l->ple_gate = need_tensor(g, "blk.%d.inp_gate.weight", i, &ok);
            l->ple_proj = need_tensor(g, "blk.%d.proj.weight", i, &ok);
            gguf_tensor *pn = need_tensor(g, "blk.%d.post_norm.weight", i, &ok);
            if (!ok) return false;
            // check_shape prints "tensor <what> in blk.<n>", so `what` is the
            // bare role, not a name template — passing the format string put a
            // literal "blk.%d." in the error text.
            if (!check_shape(l->ple_gate, m->n_embd, m->n_embd_ple,
                             "inp_gate", i) ||
                !check_shape(l->ple_proj, m->n_embd_ple, m->n_embd,
                             "proj", i) ||
                !check_shape(pn, m->n_embd, 1, "post_norm", i))
                return false;
            l->ple_post_norm = tensor_to_f32(pn, m->n_embd, &ok);
            if (!ok) return false;
        }
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
        // memcpy per element: a GGUF array sits at whatever offset it landed
        // on, so indexing arr_raw as int32_t* is a misaligned load (UBSan
        // flagged it on the --suppress-all-but-eos fixture).
        const uint8_t *raw = sup->arr_raw;
        // A failed suppress list is not optional to skip: these ids are tokens
        // the checkpoint must never emit, so run without it would be silently
        // wrong. Fail the load instead.
        m->suppress = malloc(sizeof(int32_t) * sup->arr_n);
        if (!m->suppress) return false;
        for (uint64_t i = 0; i < sup->arr_n; i++) {
            int32_t id;
            memcpy(&id, raw + i * sizeof(int32_t), sizeof id);
            if (id >= 0 && id < m->n_vocab) m->suppress[m->n_suppress++] = id;
        }
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

    return true;
}

// The per-instance half: everything sized by this sequence's context, batch
// and thread count, allocated fresh for every model_t even when the weights
// above are shared. The split is at exactly the line where the load stops
// reading the file and starts sizing buffers -- `path` and the gguf handle are
// not referenced past here, which is what makes the seam a clean one.
// Reservation auto-fit arithmetic, split out from model_alloc_runtime so it can
// be tested at all. The regime it exists for -- a budget tight enough that the
// context has to shrink to fit -- is unreachable on a development machine with
// a fixture model: the per-slot head term alone is 256 MB, so on an 8 GB host
// even a 1% reservation leaves negative room and the branch never runs. Every
// number that matters here belongs to 7B-and-up models on 24 GB cards, so the
// only way to gate it is to feed those numbers in directly.
long long model_autofit_tokens(uint64_t budget, uint64_t weights,
                               uint64_t head_per_seq, uint64_t kv_per_tok,
                               int n_seq) {
    if (n_seq < 1) n_seq = 1;
    if (kv_per_tok == 0) return 0;
    // The reservation is a budget for the SERVER, not for one sequence, and the
    // two halves of the bill scale differently: weights are uploaded once and
    // shared by every slot, while the KV cache and the activation head are paid
    // per slot. Billing both once over-committed a multi-slot server by nearly
    // the slot count -- see the Qwen2.5-7B case in tests/test_autofit.c.
    long long head   = (long long)head_per_seq * n_seq;
    long long kv_all = (long long)kv_per_tok * n_seq;
    long long room   = (long long)budget - (long long)weights - head;
    return room > 0 ? room / kv_all : 0;
}

// Whether the KV-evicts-weights trade note applies. Split out for the same
// reason as the auto-fit above: the situation only arises on a partial GPU
// split caused by the KV cache rather than by the model simply being larger
// than the card, which takes a model bigger than any fixture here.
bool model_kv_trade_note(int gpu_layers, int n_layer, uint64_t kv_dev,
                         uint64_t weights) {
    // a full offload has made no trade, and neither has a load with nothing on
    // the device at all
    if (gpu_layers <= 0 || gpu_layers >= n_layer) return false;
    if (kv_dev == 0 || weights == 0) return false;
    // only when the KV is a big share of what is on the device: a split forced
    // by the model's own size is not a trade the user can take back with -c
    return kv_dev * 4 > weights;
}

int model_autofit_clamp(long long best, int n_ctx_train) {
    int n = best > (long long)n_ctx_train ? n_ctx_train : (int)best;
    if (n < 512) n = 512;   // a floor: below this the window is not usable
    return n;
}

// Default prompt batch when no `-b` is given. The Metal tiled prefill GEMM
// (kernels.metal k_mm_*, MM_TN columns per threadgroup) gets more reuse out
// of every staged weight tile the more columns are in flight per dispatch, so
// a bigger batch is a real prefill win on the GPU path: measured on an M1
// (e2b-q40, -c 1024, pre-MM_TN-widening kernel), `-b 512` gave +9%
// prompt_tok_s over the old flat default of 64. CPU-only decode has no tiled
// kernel to feed, so a bigger default there would only cost scratch memory
// for nothing -- it stays at 64.
//
// That scratch is real, though: every B-scaled buffer below (x/xb/xb2/q/
// k_tmp/v_tmp/hb/hb2, and for MoE the grouped-prefill buffers) grows linearly
// with the batch, and measured peak process footprint on e2b-q40 went
// 250 MB (b=64) -> 524 MB (b=256) -> 897 MB (b=512). On the shared 8 GB M1
// this was measured on, free RAM alone swung from ~2.4 GB to ~470 MB just
// from other desktop activity between two `vm_stat` calls minutes apart, so
// unconditionally defaulting to 512 would be a real eviction risk exactly on
// the machine this lever targets. Scale the default down instead of ignoring
// the box's actual headroom.
static int model_gpu_batch_default(void) {
    uint64_t avail = plat_ram_available_bytes();
    if (!avail) return 512;                       // unmeasurable: assume fine
    if (avail < (uint64_t)1536 << 20) return 64;   // < 1.5 GB free: no change
    if (avail < (uint64_t)4096 << 20) return 256;  // tight: half the win
    return 512;
}

static bool model_alloc_runtime(model_t *m, const model_params *p) {
    // The gguf handle and the architecture name come off the model rather than
    // out of the bind function's locals, because on a shared-weights hit the
    // bind function never ran for this instance.
    gguf_file *g = &m->gf;
    const char *arch = m->arch;
    // runtime buffers
    m->reserve_vram_pct = p->reserve_vram_pct;
    m->gpu_layers_override = p->gpu_layers_override;
    m->cpu_moe = p->cpu_moe && m->n_expert > 0;
    m->cpu_moe_layers = p->cpu_moe_layers;
    if (m->cpu_moe) {
        // Per-layer expert placement. The default (and every non-CUDA build)
        // is the original all-on-host meaning; a CUDA upload may narrow it to
        // the requested count or to what the VRAM budget actually fits, and
        // publishes the outcome here so the forward path reads one array.
        m->moe_host = calloc((size_t)m->n_layer, sizeof(bool));
        if (!m->moe_host) return false;
        model_moe_place_host(m, m->cpu_moe_layers);
    }
    int n_ctx = p->n_ctx;
    if (n_ctx <= 0 && (p->reserve_vram_pct > 0 || p->reserve_ram_pct > 0)) {
        // Reservation auto-fit: size the context to fill whatever the
        // reservation leaves after the weights, so small models grow their
        // window into the reserved room instead of idling at the default.
        //
        // The reservation is a budget for the SERVER, not for one sequence,
        // and the two halves of the bill scale differently: the weights are
        // uploaded once and shared by every slot, while the KV cache and the
        // activation head are paid per slot. Billing both once — which is what
        // this did until Phase 5 — over-committed a multi-slot server by
        // nearly the slot count. Measured on Qwen2.5-7B with
        // `--reserve-vram 40 --parallel 4 -c 0`: every slot independently
        // auto-fit to 32768 and allocated its own 1.88 GB cache, for 12.47 GB
        // against a 10.15 GB reservation, and the only thing that kept it from
        // being far worse was the train context capping the window.
        int n_seq = p->n_seq > 0 ? p->n_seq : 1;
        size_t kv_per_tok = 0;
        for (int l = 0; l < m->n_layer; l++) {
            if (model_kv_owner(m, l) != l) continue;  // shared-KV: no rows here
            int d = model_kv_dim(m, l);
            kv_per_tok += 2ull * (m->kv_q8 ? (size_t)(d / 32) * 34
                                           : (size_t)d * sizeof(f16_t));
        }
        long long best = -1;
        if (p->reserve_ram_pct > 0) {
            // host budget covers the mmap'd weights plus the host KV copies
            size_t budget = plat_ram_bytes() / 100 * p->reserve_ram_pct;
            best = model_autofit_tokens(budget, gguf_mapped_size(&m->gf),
                                        MODEL_AUTOFIT_HEAD, kv_per_tok, n_seq);
        }
        if (p->reserve_vram_pct > 0 && p->gpu_mode == GPU_AUTO) {
            size_t vfree = 0, vtotal = 0;
            if (gpu_mem_info(&vfree, &vtotal)) {
                // device budget covers one weights copy plus every slot's KV
                size_t budget = vtotal / 100 * p->reserve_vram_pct;
                long long fit = model_autofit_tokens(
                    budget, model_cuda_weight_estimate(m, p),
                    MODEL_AUTOFIT_HEAD, kv_per_tok, n_seq);
                if (best < 0 || fit < best) best = fit;
            }
        }
        if (best > 0) {
            n_ctx = model_autofit_clamp(best, m->n_ctx_train);
            if (n_seq > 1)
                fprintf(stderr, "reservation: auto-fit context %d (train %d, "
                        "%d slots sharing the budget)\n",
                        n_ctx, m->n_ctx_train, n_seq);
            else
                fprintf(stderr, "reservation: auto-fit context %d (train %d)\n",
                        n_ctx, m->n_ctx_train);
        }
    }
    if (n_ctx <= 0) n_ctx = m->n_ctx_train < 4096 ? m->n_ctx_train : 4096;
    m->n_ctx = n_ctx;
    int batch_default = p->gpu_mode == GPU_AUTO ? model_gpu_batch_default() : 64;
    m->n_batch = p->n_batch > 0 ? p->n_batch : batch_default;
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
        // A shared-KV layer reserves nothing: model_kv_byte_off resolves it to
        // its owner's rows. Its own kv_off entry is never read as a start.
        m->kv_off[l + 1] = m->kv_off[l] +
            (model_kv_owner(m, l) == l ? (size_t)n_ctx * model_kv_dim(m, l) : 0);
    size_t kv_bytes = model_kv_boundary_bytes(m, m->n_layer);
    m->kcache = calloc(1, kv_bytes);
    m->vcache = calloc(1, kv_bytes);
    m->kv_owner = KV_OWNER_MALLOC;
    m->x      = malloc(sizeof(float) * (size_t)B * m->n_embd);
    m->xb     = malloc(sizeof(float) * (size_t)B * xdim);
    m->xb2    = malloc(sizeof(float) * (size_t)B * xdim);
    m->q      = malloc(sizeof(float) * (size_t)B * q_dim);
    m->k_tmp  = malloc(sizeof(float) * (size_t)B * kv_dim);
    m->v_tmp  = malloc(sizeof(float) * (size_t)B * kv_dim);
    if (m->n_embd_ple > 0) {
        // [token][layer][n_embd_ple] for the batch, plus one scratch of the
        // same width (the pre-pass dequantises a table row into it; the layer
        // loop reuses its first B*n_embd_ple as the gate output)
        size_t per_tok = (size_t)m->n_layer * m->n_embd_ple;
        m->ple     = malloc(sizeof(float) * (size_t)B * per_tok);
        m->ple_tmp = malloc(sizeof(float) * (size_t)B * per_tok);
        if (!m->ple || !m->ple_tmp) return false;
    }
    if (m->attn_out_gate && !m->q_gate) {
        m->q_gate = malloc(sizeof(float) * (size_t)B * q_dim);
        if (!m->q_gate) return false;
    }
    if (m->qwen35) {
        int conv_dim = 2 * m->ssm_state * m->ssm_groups + m->ssm_inner;
        int hv = m->ssm_inner / m->ssm_v_heads;
        // q_gate is shared the same way ssm_qkv is below: full-attention layers
        // write q_dim floats per token (the attention output gate), recurrent
        // ones write ssm_v_heads (the DeltaNet beta). Nothing relates the two.
        int gate_dim = q_dim > m->ssm_v_heads ? q_dim : m->ssm_v_heads;
        m->q_gate = malloc(sizeof(float) * (size_t)B * gate_dim);
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
        // recurrent-state snapshot (tracer 4): same shape as the live buffers
        m->ssm_conv_snap = calloc((size_t)m->n_layer *
                                  (m->ssm_conv_kernel - 1) * conv_dim,
                                  sizeof(float));
        m->ssm_state_snap = calloc((size_t)m->n_layer * m->ssm_v_heads *
                                   hv * hv, sizeof(float));
        m->ssm_snap_pos = -1;
        // one dequantized conv-kernel row, reused every conv step (the forward
        // pass is single-threaded here, so one buffer suffices) — preallocated
        // so the hot path never mallocs and an OOM fails the load, not a token
        m->ssm_cw = malloc(sizeof(float) * m->ssm_conv_kernel);
    }
    if (m->granite_hybrid || m->nemotron_h) {
        // Mamba-2 recurrent scratch. Per token: the in_proj output zxBCdt
        // (ssm_qkv, d_in_proj wide), the post-conv xBC (ssm_aux, conv_dim),
        // and the gated+normed inner y (ssm_z, inner). Per layer, held across
        // the whole sequence (no cache seam yet): the conv ring and the SSD
        // state. ssm_cw holds one dequantized conv-kernel row.
        int nh = m->ssm_v_heads, ds = m->ssm_state, ng = m->ssm_groups;
        int inner = m->ssm_inner, hd = inner / nh;
        int conv_dim  = inner + 2 * ng * ds;
        int d_in_proj = 2 * inner + 2 * ng * ds + nh;
        m->ssm_qkv = malloc(sizeof(float) * (size_t)B * d_in_proj);
        m->ssm_aux = malloc(sizeof(float) * (size_t)B * conv_dim);
        m->ssm_z   = malloc(sizeof(float) * (size_t)B * inner);
        m->ssm_conv_state = calloc((size_t)m->n_layer *
                                   (m->ssm_conv_kernel - 1) * conv_dim,
                                   sizeof(float));
        m->ssm_state_mem = calloc((size_t)m->n_layer * nh * hd * ds,
                                  sizeof(float));
        // recurrent-state snapshot (tracer 4): same shape as the live buffers
        m->ssm_conv_snap = calloc((size_t)m->n_layer *
                                  (m->ssm_conv_kernel - 1) * conv_dim,
                                  sizeof(float));
        m->ssm_state_snap = calloc((size_t)m->n_layer * nh * hd * ds,
                                   sizeof(float));
        m->ssm_snap_pos = -1;
        m->ssm_cw = malloc(sizeof(float) * m->ssm_conv_kernel);
        if (!m->ssm_qkv || !m->ssm_aux || !m->ssm_z ||
            !m->ssm_conv_state || !m->ssm_state_mem || !m->ssm_cw ||
            !m->ssm_conv_snap || !m->ssm_state_snap)
            return false;
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
        m->moe_sel_scores = malloc(sizeof(float) * (size_t)m->n_expert);
        if (m->n_ff_shexp > 0) {
            size_t nb = (size_t)m->n_batch;
            m->shexp_in = malloc(sizeof(float) * nb * (size_t)m->n_embd);
            m->shexp_o  = malloc(sizeof(float) * nb * (size_t)m->n_embd);
            m->shexp_g  = malloc(sizeof(float) * nb * (size_t)m->n_ff_shexp);
            m->shexp_u  = malloc(sizeof(float) * nb * (size_t)m->n_ff_shexp);
            if (!m->shexp_in || !m->shexp_o || !m->shexp_g || !m->shexp_u)
                return false;
        }
        m->moe_group_score = malloc(sizeof(float) * (size_t)m->n_expert_groups);
        if (!m->moe_sel_scores || !m->moe_group_score) return false;
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
        m->moe_trace_norms = malloc(sizeof(float) * nb * nu);
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
                             !m->moe_trace_norms || !m->moe_gidx || !m->moe_gw)) ||
        (m->qwen35 && (!m->q_gate || !m->ssm_qkv || !m->ssm_z ||
                       !m->ssm_aux || !m->ssm_conv_state ||
                       !m->ssm_state_mem || !m->ssm_cw ||
                       !m->ssm_conv_snap || !m->ssm_state_snap))) {
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
        // A context that FITS but evicts weights is the silent case. Refusing
        // one that cannot fit is loud and correct — `-c 1000000` says it needs
        // 131072 MB of KV cache and exits non-zero. But a context that merely
        // costs layers is accepted without comment, and the bill arrives as
        // throughput: `-c 32768` on an 8B model takes decode from 66.5 to
        // 8.6 tok/s on an 8 GB card, because a 4.3 GB cache pushed layers onto
        // the host. The split line already reports the placement; nothing
        // connected it to the context that caused it.
        //
        // Only when the KV is actually a big share of what is on the device —
        // a partial split for any other reason (a model simply larger than the
        // card) is not a trade the user can take back by lowering -c.
        if (m->gpu) {
            size_t kv_dev = model_kv_boundary_bytes(m, m->gpu_layers) * 2;
            uint64_t wb = model_cuda_weight_estimate(m, p);
            if (model_kv_trade_note(m->gpu_layers, m->n_layer, kv_dev, wb)) {
                fprintf(stderr,
                        "note: the KV cache for ctx %d is %.2f GB on the device"
                        " and %d of %d layers ran out of room because of it —"
                        " a smaller -c%s moves layers back\n",
                        m->n_ctx, kv_dev / 1e9, m->n_layer - m->gpu_layers,
                        m->n_layer, m->kv_q8 ? "" : " or --kv q8 (about half)");
            }
        }
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
    free(m->moe_host);
    m->moe_host = NULL;
    // Deregister on the clean path. The unclean paths (SIGKILL, crash) are
    // covered by dead-pid reaping in the next runner's claim, which is how the
    // orphans that motivated the registry would have been cleared.
    vram_release(m->vram);
    m->vram = NULL;
    // ---- the per-instance half. Everything below belongs to this sequence
    // alone and is freed unconditionally; the weights are refcounted at the
    // bottom. The rope tables are per-instance and not shared: YaRN
    // auto-extension keys off the requested context and phi3 picks its
    // LongRoPE factor set the same way, so two slots of one file with
    // different -c legitimately want different tables.
    free(m->kv_off); free(m->ple); free(m->ple_tmp);
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
    free(m->ssm_conv_snap); free(m->ssm_state_snap);
    free(m->hb); free(m->hb2); free(m->att); free(m->logits); free(m->all_logits);
    free(m->shexp_in); free(m->shexp_o); free(m->shexp_g); free(m->shexp_u);
    free(m->moe_logits); free(m->moe_sel_scores); free(m->moe_group_score);
    free(m->moe_gate); free(m->moe_up);
    free(m->moe_dexp); free(m->moe_out);
    free(m->moe_out_b); free(m->moe_gath); free(m->moe_gate_b);
    free(m->moe_up_b); free(m->moe_dexp_b); free(m->moe_sel);
    free(m->moe_selw); free(m->moe_trace_norms); free(m->moe_gidx); free(m->moe_gw);
    free(m->moe_probe_hist);
    tpool_destroy(m->tp);
    // ---- the shared half. A published load hands its reference back and the
    // last holder frees the buffers; an unpublished one (stat failed, or the
    // load failed inside the bind phase) owns them outright and frees them
    // here. Nothing else may free them: every other model_t sharing this
    // record still holds aliasing pointers.
    if (m->W) mw_release(m->W);
    else      model_free_weights(m);
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

// Attention softmax with a learned sink logit (gpt-oss). Transcribed from
// llama.cpp's ggml_compute_forward_soft_max: the sink participates in the max
// and in the denominator, and has NO output row — so the probabilities over
// real positions sum to less than one and the head's output shrinks. Note the
// sink is compared against ALREADY-SCALED scores; it is not itself scaled.
static void softmax_sink(float *x, int n, float sink) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    if (sink > mx) mx = sink;
    float s = 0;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); s += x[i]; }
    s += expf(sink - mx);
    for (int i = 0; i < n; i++) x[i] /= s;
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
    const void *xq;   // int8-quantized activations, or NULL for the f32 route
} mv_job;

static void mv_rows(void *ctx, int i0, int i1) {
    mv_job *j = ctx;
    const uint8_t *base = j->w->data;
    int type = j->w->type, n_in = j->n_in;

    if (j->n_batch == 1) {
        if (j->xq) {
            for (int r = i0; r < i1; r++) {
                float v = vec_dot_i8(type, base + (size_t)r * j->rsz, j->xq, n_in);
                j->y[r] = j->bias ? v + j->bias[r] : v;
            }
            return;
        }
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
//
// Single-column (decode) matvecs on a promoted quant type take the fused int8
// route: the activation column is quantized ONCE here and every row dots
// against it in int8 (quants.c "fused int8 dot"). The quantization is O(n_in)
// against O(n_in*n_out) of matvec work, so it is only worth it once the output
// is wide enough to amortize — a 1-row projection would pay for a whole
// activation quant to save one dot.
enum { I8_MIN_ROWS = 32 };

static void matvec_b(tpool *tp, float *y, int y_stride, const gguf_tensor *w,
                     const float *x, int x_stride, int n_in, int n_out,
                     const float *bias, int n_batch) {
    mv_job j = { w, x, bias, y, n_in, n_batch, x_stride, y_stride,
                 ggml_row_size(w->type, n_in), NULL };
    void *xq = NULL;
    if (n_batch == 1 && n_out >= I8_MIN_ROWS && i8_dot_enabled() &&
        i8_dot_ok(w->type, n_in) && (xq = malloc(i8_act_size(n_in)))) {
        i8_quant_act(x, xq, n_in);
        j.xq = xq;
    }
    tpool_run(tp, mv_rows, &j, n_out);
    free(xq);
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
    float ms = model_rope_mscale(m, layer);
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
    const float *sinks;     // gpt-oss per-head sink logits, or NULL
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
        if (j->sinks) softmax_sink(att + j->t0, j->pos + 1 - j->t0, j->sinks[h]);
        else          softmax(att + j->t0, j->pos + 1 - j->t0);
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

// Is the pass currently running the one RUNNER_DEBUG_ACT selected? The layer
// loop keeps this in a local; helpers called from inside it need to ask.
static bool dbg_act_now(void) {
    return dbg_act_mode() && dbg_act_pass == dbg_act_mode();
}

static void dbg_stat(const char *tag, int layer, const float *v, size_t n) {
    float mn = FLT_MAX, mx = -FLT_MAX, absmx = 0;
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
    fprintf(stderr, "ACT L%-3d %-16s n=%-7zu min=%+.4g max=%+.4g mean=%+.4g absmax=%.4g inf=%zu nan=%zu zero=%zu sum=%+.6f",
            layer, tag, n, good ? mn : 0.0f, good ? mx : 0.0f,
            good ? sum / (double)good : 0.0, absmx, n_inf, n_nan, n_zero, sum);
    // First and last three values in llama.cpp's eval-callback layout, so a
    // trace from either engine can be diffed row by row without reformatting.
    if (n >= 6)
        fprintf(stderr, " [%+.4f %+.4f %+.4f ... %+.4f %+.4f %+.4f]",
                v[0], v[1], v[2], v[n - 3], v[n - 2], v[n - 1]);
    fprintf(stderr, "\n");
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

// ------------------------------------------------------ MoE routing trace
// RUNNER_MOE_TRACE=path appends one JSONL record per routed token per MoE
// layer: {"pos":N,"layer":L,"experts":[...],"gates":[...]}. Off by default;
// zero cost when unset (one cached getenv + a cached FILE*, same shape as
// dbg_act_mode above). CPU path only — the CUDA MoE kernels route on-device
// and never reach this call site.
static FILE *moe_trace_file(void) {
    static FILE *fp = NULL;
    static int opened = 0;
    if (!opened) {
        opened = 1;
        const char *path = getenv("RUNNER_MOE_TRACE");
        if (path && *path) {
            fp = fopen(path, "a");
            if (!fp)
                fprintf(stderr, "warning: RUNNER_MOE_TRACE=%s: could not open for append\n", path);
        }
    }
    return fp;
}

// `norms` is the L2 norm of each selected expert's own FFN output (after its
// down-projection + bias, before the routing weight scales and sums it) —
// NULL is written as an empty array, so an older consumer parsing only
// "experts"/"gates" is unaffected by the new field (JC-R3 saliency: gate
// mass alone ranks experts by how often/hard they're picked; gate x norm
// also weighs how much each pick actually moved the residual stream).
static void moe_trace_emit(int pos, int layer, const int *sel, const float *selw,
                           const float *norms, int used) {
    FILE *fp = moe_trace_file();
    if (!fp) return;
    fprintf(fp, "{\"pos\":%d,\"layer\":%d,\"experts\":[", pos, layer);
    for (int t = 0; t < used; t++) fprintf(fp, "%s%d", t ? "," : "", sel[t]);
    fprintf(fp, "],\"gates\":[");
    for (int t = 0; t < used; t++) fprintf(fp, "%s%.6g", t ? "," : "", selw[t]);
    fprintf(fp, "],\"norms\":[");
    if (norms)
        for (int t = 0; t < used; t++) fprintf(fp, "%s%.6g", t ? "," : "", norms[t]);
    fprintf(fp, "]}\n");
    fflush(fp);   // durable across a killed server: --serve traces run for the
                   // process's whole lifetime, not to a matching fclose
}

// ------------------------------------------------------ MoE gate-reuse probe
// RUNNER_MOE_PROBE=path appends one JSONL record per (token, layer, lookahead
// in 1..3): {"pos":N,"layer":L,"lookahead":K,"predicted":[8],"actual":[8]}.
// "predicted" is layer L's OWN router applied to the hidden state from K
// layers back instead of L's own — the real gate-reuse/prefetch question,
// not trip 1's raw expert-set-overlap proxy (that proxy checked whether two
// DIFFERENT layers' routers happened to agree on the SAME hidden state; this
// checks whether ONE layer's router already knows the answer EARLY, from an
// OLDER hidden state — the thing an actual prefetcher would exploit). Off by
// default, zero cost when unset. gemma-4 only (gemma_moe_ffn below) — the
// only architecture this trip's dev task targets.
static FILE *moe_probe_file(void) {
    static FILE *fp = NULL;
    static int opened = 0;
    if (!opened) {
        opened = 1;
        const char *path = getenv("RUNNER_MOE_PROBE");
        if (path && *path) {
            fp = fopen(path, "a");
            if (!fp)
                fprintf(stderr, "warning: RUNNER_MOE_PROBE=%s: could not open for append\n", path);
        }
    }
    return fp;
}

static void moe_probe_emit(int pos, int layer, int lookahead,
                            const int *predicted, const int *actual, int used) {
    FILE *fp = moe_probe_file();
    if (!fp) return;
    fprintf(fp, "{\"pos\":%d,\"layer\":%d,\"lookahead\":%d,\"predicted\":[",
            pos, layer, lookahead);
    for (int t = 0; t < used; t++) fprintf(fp, "%s%d", t ? "," : "", predicted[t]);
    fprintf(fp, "],\"actual\":[");
    for (int t = 0; t < used; t++) fprintf(fp, "%s%d", t ? "," : "", actual[t]);
    fprintf(fp, "]}\n");
    fflush(fp);
}

// Shift the probe's 3-deep hidden-state ring and push this layer's post-
// attention hidden state (the whole current batch) as the newest entry.
// Lazily sized to [3][n_batch][n_embd] on first use, so a model loaded with
// RUNNER_MOE_PROBE unset never allocates it. `stride` is the caller's
// per-token spacing in `x` (gemma's m->x is packed at n_embd; the generic
// path's m->xb is packed at m->xdim, which can exceed n_embd) — the ring
// itself always stores compact n_embd rows regardless of the source stride.
static void moe_probe_push(model_t *m, const float *x, int n, int n_embd, int stride) {
    if (!m->moe_probe_hist) {
        m->moe_probe_hist = malloc(sizeof(float) * 3 * (size_t)m->n_batch * n_embd);
        if (!m->moe_probe_hist) return;
    }
    size_t slot = (size_t)m->n_batch * n_embd;
    memmove(m->moe_probe_hist + slot, m->moe_probe_hist, sizeof(float) * slot * 2);
    for (int b = 0; b < n; b++)
        memcpy(m->moe_probe_hist + (size_t)b * n_embd, x + (size_t)b * stride,
               sizeof(float) * n_embd);
    if (m->moe_probe_depth < 3) m->moe_probe_depth++;
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
    // muse-glimmer scales BEFORE the softcap (llama.cpp: ggml_scale, then
    // the tanh squashing); the order is observable through the cap
    if (m->logit_scale != 1.0f && m->logit_scale != 0.0f)
        for (int i = 0; i < m->n_vocab; i++)
            logits[i] *= m->logit_scale;
    if (m->logit_softcap > 0)
        for (int i = 0; i < m->n_vocab; i++)
            logits[i] = m->logit_softcap * tanhf(logits[i] / m->logit_softcap);
    if (m->n_suppress) suppress_logits(m, logits);
}

// See model.h: the one definition of what happens to a dequantized embedding
// row, shared by the CPU forward and the CUDA/Metal host-side staging copies.
void model_embd_transform(const model_t *m, float *row) {
    if (m->embd_scale != 1.0f)
        for (int i = 0; i < m->n_embd; i++) row[i] *= m->embd_scale;
    if (m->embd_norm)
        rmsnorm(row, row, NULL, m->n_embd, m->rms_eps);
}

// One CPU Mamba-2 (SSD) recurrent layer for Granite-4 h-series. Transcribed
// from llama.cpp b10353 build_mamba2_layer (mamba-base.cpp) + the scalar-per-
// head branch of ggml_compute_forward_ssm_scan_f32 (ops.cpp), which this
// reproduces exactly as the single-token recurrent step:
//
//   zxBCdt = in_proj(x)                      split z[inner], xBC[conv_dim], dt[H]
//   xBC   = silu(conv1d(ring ++ xBC) + b)    causal depthwise conv, then split
//                                            x[inner], B[G*S], C[G*S]
//   per head h (group g = h/(H/G)):
//     dt_h = softplus(dt[h] + dt_bias[h]);  dA = exp(dt_h * A[h])
//     state[h,p,n] = state[h,p,n]*dA + B[g,n]*x[h,p]*dt_h    (in place)
//     y[h,p]       = sum_n state[h,p,n]*C[g,n] + D[h]*x[h,p]
//   y = silu(z) * y                          gated,
//   y = rmsnorm(y) * ssm_norm                THEN grouped RMS norm (per group),
//   out = out_proj(y)
//
// Prefill (n>1) has two math-identical paths (tracer 3): a SERIAL sweep of the
// step above (the reference, and the decode path), and a CHUNKED scan. The
// chunked path (a) hoists the conv-weight dequant out of the token loop and
// computes the causal conv for every token from the known pre-conv inputs, then
// (b) tiles the token axis into chunks of SSM_PREFILL_CHUNK and, within a chunk,
// runs the per-head SSD recurrence in PARALLEL across heads (each head an
// independent lane), carrying the SSD state and conv ring across chunk
// boundaries (inter-chunk recurrent). It never reassociates a reduction and
// never changes the per-lane op order -- it only reorders INDEPENDENT lanes and
// hoists a redundant dequant -- so it is BIT-IDENTICAL to the serial sweep, not
// merely within tolerance. That is the tracer-3 gate: chunked == serial, pinned
// by a test, so chunking changes speed, not output. XR_SSM_SERIAL=1 forces the
// serial path (the gate runs the same prompt both ways); XR_SSM_CHUNK overrides
// the chunk size. Decode (n==1) always takes the serial path. State lives in the
// per-model conv ring / SSD buffers and is reset at pos 0, so a correct decode
// runs the whole sequence in order.
//
// Built with fast-math DISABLED (unlike the rest of the engine): the SSD
// recurrence carries state token-to-token, so a per-step reassociation or an
// approximate expf/division COMPOUNDS through the sequence and drifts the
// running-count kind of state that long-range tasks depend on. ggml compiles
// its scan with accurate math; matching it is what makes greedy decoding
// token-identical rather than merely fluent. The grouped RMS norm below also
// accumulates its sum-of-squares in double, exactly as ggml_compute_forward_
// rms_norm_f32 does.
//
// GCC and Clang spell per-function FP control differently.  Clang accepts
// GCC's optimize attribute only by ignoring it, which used to leave this
// whole block under the translation unit's -ffast-math setting on macOS.
// Keep the compiler split explicit: GCC gets its function option and Clang
// gets the documented compound-statement precise mode.
#if defined(__clang__)
#define SSM_PRECISE_ATTR
#define SSM_PRECISE_SCOPE _Pragma("float_control(precise, on)")
#elif defined(__GNUC__)
#define SSM_PRECISE_ATTR __attribute__((optimize("no-fast-math")))
#define SSM_PRECISE_SCOPE
#else
#define SSM_PRECISE_ATTR
#define SSM_PRECISE_SCOPE
#endif

// Largest conv kernel the chunked path's per-thread dequant buffer supports
// (Mamba-2 hybrids use 4). Above this the dispatcher falls back to serial.
#define SSM_MAX_CONV_KERNEL 16
// Token-chunk size for the chunked SSD prefill scan (design: ~128-256).
#ifndef SSM_PREFILL_CHUNK
#define SSM_PREFILL_CHUNK 256
#endif

static int ssm_prefill_chunk(void) {
    static int c = -1;
    if (c < 0) {
        const char *e = getenv("XR_SSM_CHUNK");
        c = e ? atoi(e) : SSM_PREFILL_CHUNK;
        if (c < 1) c = SSM_PREFILL_CHUNK;
    }
    return c;
}
static int ssm_force_serial(void) {
    static int f = -1;
    if (f < 0) {
        const char *e = getenv("XR_SSM_SERIAL");
        f = (e && e[0] && e[0] != '0') ? 1 : 0;
    }
    return f;
}

// ---- serial reference core: the exact per-token sweep (also the decode path).
// This is the ground truth the chunked path is pinned bit-identical against.
SSM_PRECISE_ATTR
static void mamba2_ssd_core_serial(model_t *m, layer_t *ly, int layer, int n) {
    SSM_PRECISE_SCOPE
    int nh = m->ssm_v_heads;                 // n_ssm_head
    int inner = m->ssm_inner, hd = inner / nh; // head_dim = inner / n_head
    int ds = m->ssm_state, ng = m->ssm_groups;
    int gsz = nh / ng;                       // heads per group (repeat_interleave)
    int conv_dim  = inner + 2 * ng * ds;
    int d_in_proj = 2 * inner + 2 * ng * ds + nh;
    int histn = m->ssm_conv_kernel - 1;

    float *convstate = m->ssm_conv_state +
                       (size_t)layer * histn * conv_dim;
    float *states = m->ssm_state_mem + (size_t)layer * nh * hd * ds;
    size_t wrs = ggml_row_size(ly->ssm_conv->type, m->ssm_conv_kernel);
    float *cw = m->ssm_cw;                    // one dequantized conv row, reused

    for (int b = 0; b < n; b++) {
        float *proj = m->ssm_qkv + (size_t)b * d_in_proj;
        const float *z   = proj;              // gate [inner]
        const float *xBC0 = proj + inner;     // pre-conv xBC [conv_dim]
        const float *dtr = proj + inner + conv_dim; // dt raw [nh]
        float *xBC = m->ssm_aux + (size_t)b * conv_dim; // post-conv [conv_dim]
        // Causal depthwise conv over the ring (oldest..newest) + current, then
        // bias and silu. Weight row c is [conv_kernel]; tap histn multiplies
        // the current input (llama concats the ring BEFORE the new column).
        for (int c = 0; c < conv_dim; c++) {
            dequant_row(ly->ssm_conv->type,
                        (const uint8_t *)ly->ssm_conv->data + (size_t)c * wrs,
                        cw, m->ssm_conv_kernel);
            float sum = ly->ssm_conv1d_b[c] + cw[histn] * xBC0[c];
            for (int k = 0; k < histn; k++)
                sum += cw[k] * convstate[(size_t)k * conv_dim + c];
            xBC[c] = sum / (1.0f + expf(-sum));    // silu
        }
        // advance the ring: drop the oldest column, append the PRE-conv input
        if (histn) {
            memmove(convstate, convstate + conv_dim,
                    sizeof(float) * (size_t)(histn - 1) * conv_dim);
            memcpy(convstate + (size_t)(histn - 1) * conv_dim, xBC0,
                   sizeof(float) * conv_dim);
        }
        const float *xh = xBC;                // x  [inner] = [head_dim, n_head]
        const float *Bb = xBC + inner;        // B  [n_group, d_state]
        const float *Cb = xBC + inner + ng * ds; // C [n_group, d_state]
        float *y = m->ssm_z + (size_t)b * inner;  // gated + normed inner output
        for (int h = 0; h < nh; h++) {
            int g = h / gsz;
            const float *Bg = Bb + (size_t)g * ds;
            const float *Cg = Cb + (size_t)g * ds;
            float dt = softplus_f32(dtr[h] + ly->ssm_dt[h]);
            float dA = expf(dt * ly->ssm_a[h]);
            float D  = ly->ssm_d[h];
            float *st = states + (size_t)h * hd * ds;
            for (int p = 0; p < hd; p++) {
                float x_dt = xh[(size_t)h * hd + p] * dt;
                float *sp = st + (size_t)p * ds;
                float sumf = 0.0f;
                for (int j = 0; j < ds; j++) {
                    float s = sp[j] * dA + Bg[j] * x_dt;
                    sumf += s * Cg[j];
                    sp[j] = s;
                }
                y[(size_t)h * hd + p] = sumf + D * xh[(size_t)h * hd + p];
            }
        }
        // gate by silu(z), then grouped RMS norm (n_group groups over inner),
        // matching Mamba-2's RMSNormGated: normalize the GATED activations.
        for (int i = 0; i < inner; i++) {
            float zz = z[i];
            y[i] *= zz / (1.0f + expf(-zz));
        }
        int per_g = inner / ng;
        for (int g = 0; g < ng; g++) {
            float *yg = y + (size_t)g * per_g;
            const float *wg = ly->ssm_norm_w + (size_t)g * per_g;
            double ss = 0.0;                       // double accumulation, as ggml
            for (int i = 0; i < per_g; i++) ss += (double)(yg[i] * yg[i]);
            float mean = (float)(ss / per_g);
            float scale = 1.0f / sqrtf(mean + m->rms_eps);
            for (int i = 0; i < per_g; i++) yg[i] = yg[i] * scale * wg[i];
        }
    }
}

// ---- chunked-prefill workers. Each reproduces the serial core's per-lane op
// order EXACTLY; only independent lanes (conv rows / heads / tokens) are spread
// across threads, so the result is bit-identical to mamba2_ssd_core_serial.

// phase 1: causal depthwise conv1d for every token, one dequant per conv row.
typedef struct {
    const model_t *m; const layer_t *ly;
    int n, inner, conv_dim, d_in_proj, histn, conv_kernel;
    size_t wrs;
    const float *convstate;   // initial conv ring (read-only here)
} ssm_conv_job;

SSM_PRECISE_ATTR
static void ssm_conv1d_worker(void *vp, int c0, int c1) {
    SSM_PRECISE_SCOPE
    const ssm_conv_job *J = (const ssm_conv_job *)vp;
    const model_t *m = J->m; const layer_t *ly = J->ly;
    int n = J->n, inner = J->inner, conv_dim = J->conv_dim, d_in = J->d_in_proj;
    int histn = J->histn;
    for (int c = c0; c < c1; c++) {
        float cw[SSM_MAX_CONV_KERNEL];        // per-thread dequant buffer
        dequant_row(ly->ssm_conv->type,
                    (const uint8_t *)ly->ssm_conv->data + (size_t)c * J->wrs,
                    cw, J->conv_kernel);
        float bias = ly->ssm_conv1d_b[c];
        for (int b = 0; b < n; b++) {
            // xBC0[t][c] = pre-conv input of token t, channel c
            float cur = m->ssm_qkv[(size_t)b * d_in + inner + c];
            float sum = bias + cw[histn] * cur;
            for (int k = 0; k < histn; k++) {
                int tau = b - histn + k;      // ring tap -> absolute token
                float v = (tau >= 0)
                    ? m->ssm_qkv[(size_t)tau * d_in + inner + c]
                    : J->convstate[(size_t)(b + k) * conv_dim + c];
                sum += cw[k] * v;
            }
            m->ssm_aux[(size_t)b * conv_dim + c] = sum / (1.0f + expf(-sum));
        }
    }
}

// phase 2: SSD scan for one token chunk [t0,t1), parallel across heads.
typedef struct {
    const model_t *m; const layer_t *ly;
    int inner, hd, ds, ng, gsz, d_in_proj, conv_dim;
    float *states;
    int t0, t1;               // token chunk
} ssm_scan_job;

SSM_PRECISE_ATTR
static void ssm_scan_worker(void *vp, int h0, int h1) {
    SSM_PRECISE_SCOPE
    const ssm_scan_job *J = (const ssm_scan_job *)vp;
    const model_t *m = J->m; const layer_t *ly = J->ly;
    int inner = J->inner, hd = J->hd, ds = J->ds, ng = J->ng, gsz = J->gsz;
    int d_in = J->d_in_proj, conv_dim = J->conv_dim;
    for (int h = h0; h < h1; h++) {
        int g = h / gsz;
        float A = ly->ssm_a[h], D = ly->ssm_d[h], dtb = ly->ssm_dt[h];
        float *st = J->states + (size_t)h * hd * ds;
        for (int b = J->t0; b < J->t1; b++) {
            const float *proj = m->ssm_qkv + (size_t)b * d_in;
            const float *xBC = m->ssm_aux + (size_t)b * conv_dim;
            const float *Bg = xBC + inner + (size_t)g * ds;
            const float *Cg = xBC + inner + ng * ds + (size_t)g * ds;
            float dt = softplus_f32(proj[inner + conv_dim + h] + dtb);
            float dA = expf(dt * A);
            float *y = m->ssm_z + (size_t)b * inner;
            for (int p = 0; p < hd; p++) {
                float x = xBC[(size_t)h * hd + p];
                float x_dt = x * dt;
                float *sp = st + (size_t)p * ds;
                float sumf = 0.0f;
                for (int j = 0; j < ds; j++) {
                    float s = sp[j] * dA + Bg[j] * x_dt;
                    sumf += s * Cg[j];
                    sp[j] = s;
                }
                y[(size_t)h * hd + p] = sumf + D * x;
            }
        }
    }
}

// phase 3: gate by silu(z) then grouped RMS norm, parallel across tokens.
typedef struct {
    const model_t *m; const layer_t *ly;
    int inner, ng, per_g, d_in_proj; float rms_eps;
} ssm_norm_job;

SSM_PRECISE_ATTR
static void ssm_gate_norm_worker(void *vp, int b0, int b1) {
    SSM_PRECISE_SCOPE
    const ssm_norm_job *J = (const ssm_norm_job *)vp;
    const model_t *m = J->m; const layer_t *ly = J->ly;
    int inner = J->inner, ng = J->ng, per_g = J->per_g;
    for (int b = b0; b < b1; b++) {
        const float *z = m->ssm_qkv + (size_t)b * J->d_in_proj; // gate [inner]
        float *y = m->ssm_z + (size_t)b * inner;
        for (int i = 0; i < inner; i++) {
            float zz = z[i];
            y[i] *= zz / (1.0f + expf(-zz));
        }
        for (int g = 0; g < ng; g++) {
            float *yg = y + (size_t)g * per_g;
            const float *wg = ly->ssm_norm_w + (size_t)g * per_g;
            double ss = 0.0;
            for (int i = 0; i < per_g; i++) ss += (double)(yg[i] * yg[i]);
            float mean = (float)(ss / per_g);
            float scale = 1.0f / sqrtf(mean + J->rms_eps);
            for (int i = 0; i < per_g; i++) yg[i] = yg[i] * scale * wg[i];
        }
    }
}

// ---- chunked core: intra-chunk parallel, inter-chunk recurrent. Bit-identical
// to mamba2_ssd_core_serial (see the ssd-step comment above and the pinned gate).
SSM_PRECISE_ATTR
static void mamba2_ssd_core_chunked(model_t *m, layer_t *ly, int layer, int n) {
    SSM_PRECISE_SCOPE
    int nh = m->ssm_v_heads;
    int inner = m->ssm_inner, hd = inner / nh;
    int ds = m->ssm_state, ng = m->ssm_groups, gsz = nh / ng;
    int conv_dim = inner + 2 * ng * ds;
    int d_in_proj = 2 * inner + 2 * ng * ds + nh;
    int histn = m->ssm_conv_kernel - 1;
    float *convstate = m->ssm_conv_state + (size_t)layer * histn * conv_dim;
    float *states = m->ssm_state_mem + (size_t)layer * nh * hd * ds;

    // phase 1: conv1d for all tokens (thread over conv rows), reading the
    // initial ring for the first histn taps of the leading tokens.
    ssm_conv_job cj = { m, ly, n, inner, conv_dim, d_in_proj, histn,
                        m->ssm_conv_kernel,
                        ggml_row_size(ly->ssm_conv->type, m->ssm_conv_kernel),
                        convstate };
    tpool_run(m->tp, ssm_conv1d_worker, &cj, conv_dim);

    // advance the conv ring to the last histn PRE-conv inputs -- the same state
    // the per-token memmove/append leaves, so a following decode continues bit-
    // identically.
    if (histn) {
        if (n >= histn) {
            for (int r = 0; r < histn; r++)
                memcpy(convstate + (size_t)r * conv_dim,
                       m->ssm_qkv + (size_t)(n - histn + r) * d_in_proj + inner,
                       sizeof(float) * conv_dim);
        } else {
            memmove(convstate, convstate + (size_t)n * conv_dim,
                    sizeof(float) * (size_t)(histn - n) * conv_dim);
            for (int r = 0; r < n; r++)
                memcpy(convstate + (size_t)(histn - n + r) * conv_dim,
                       m->ssm_qkv + (size_t)r * d_in_proj + inner,
                       sizeof(float) * conv_dim);
        }
    }

    // phase 2: chunked SSD scan -- tile tokens, run heads in parallel per chunk,
    // carry SSD state across chunk boundaries.
    int chunk = ssm_prefill_chunk();
    ssm_scan_job sj = { m, ly, inner, hd, ds, ng, gsz, d_in_proj, conv_dim,
                        states, 0, 0 };
    for (int t0 = 0; t0 < n; t0 += chunk) {
        sj.t0 = t0;
        sj.t1 = (t0 + chunk < n) ? t0 + chunk : n;
        tpool_run(m->tp, ssm_scan_worker, &sj, nh);
    }

    // phase 3: gate + grouped RMS norm (thread over tokens).
    ssm_norm_job nj = { m, ly, inner, ng, inner / ng, d_in_proj, m->rms_eps };
    tpool_run(m->tp, ssm_gate_norm_worker, &nj, n);
}

#undef SSM_PRECISE_SCOPE
#undef SSM_PRECISE_ATTR

// Dispatcher: in_proj (batched), the mixer core (chunked prefill or serial),
// then out_proj (batched). Decode (n==1) and XR_SSM_SERIAL take the serial core.
static void mamba2_ssd_step(model_t *m, layer_t *ly, int layer, int n, int xdim) {
    int nh = m->ssm_v_heads, inner = m->ssm_inner, ng = m->ssm_groups;
    int d_in_proj = 2 * inner + 2 * ng * m->ssm_state + nh;

    // in_proj: normed input (m->xb) -> zxBCdt (m->ssm_qkv)
    matvec_b(m->tp, m->ssm_qkv, d_in_proj, ly->ssm_in,
             m->xb, xdim, m->n_embd, d_in_proj, NULL, n);

    if (n > 1 && !ssm_force_serial() &&
        m->ssm_conv_kernel <= SSM_MAX_CONV_KERNEL)
        mamba2_ssd_core_chunked(m, ly, layer, n);
    else
        mamba2_ssd_core_serial(m, ly, layer, n);

    // out_proj: y (m->ssm_z, stride inner) -> m->xb (stride xdim)
    matvec_b(m->tp, m->xb, xdim, ly->ssm_out, m->ssm_z, inner,
             inner, m->n_embd, NULL, n);
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
                      int n_embd, int ne, int used, int *sel, float *selw,
                      bool dbg_dump) {
    float *probs = m->moe_logits;
    matvec_b(m->tp, probs, ne, ly->ffn_gate_inp, xin, n_embd, n_embd, ne, NULL, 1);
    if (dbg_dump && dbg_act_now())
        dbg_stat("moe-logits-raw", (int)(ly - m->layers), probs, ne);
    // Router bias (gpt-oss) applies to the LOGITS, before gating.
    if (ly->ffn_gate_inp_b)
        for (int e = 0; e < ne; e++) probs[e] += ly->ffn_gate_inp_b[e];

    // --- gating: logits -> probabilities. SOFTMAX_WEIGHT deliberately leaves
    // the logits alone here; its softmax runs over the SELECTED weights below.
    switch (m->expert_gating) {
    case EXPERT_GATE_SIGMOID:
        for (int e = 0; e < ne; e++) probs[e] = 1.0f / (1.0f + expf(-probs[e]));
        break;
    case EXPERT_GATE_SQRT_SOFTPLUS:
        for (int e = 0; e < ne; e++) {
            // log1p(exp(x)) with the standard large-x guard: for big x the
            // softplus is x to within fp32, and expf would overflow.
            float x = probs[e];
            probs[e] = sqrtf(x > 20.0f ? x : log1pf(expf(x)));
        }
        break;
    case EXPERT_GATE_SOFTMAX_WEIGHT:
        break;
    default: {   // EXPERT_GATE_SOFTMAX
        float mx = probs[0];
        for (int e = 1; e < ne; e++) if (probs[e] > mx) mx = probs[e];
        float ssum = 0.0f;
        for (int e = 0; e < ne; e++) { float p = expf(probs[e] - mx); probs[e] = p; ssum += p; }
        for (int e = 0; e < ne; e++) probs[e] /= ssum;
        break;
    }
    }

    // --- selection scores. exp_probs_b biases SELECTION ONLY: the weights the
    // chosen experts are scaled by come from the unbiased probabilities, which
    // is the whole point of DeepSeek V3's aux-loss-free balancing.
    float *scores = probs;
    if (ly->exp_probs_b || m->n_expert_groups > 1) {
        scores = m->moe_sel_scores;
        for (int e = 0; e < ne; e++)
            scores[e] = probs[e] + (ly->exp_probs_b ? ly->exp_probs_b[e] : 0.0f);
    }

    // --- group-limited top-k: keep the n_group_used groups with the largest
    // sum of their top-2 scores, and mask the rest out of selection entirely.
    if (m->n_expert_groups > 1) {
        int per = ne / m->n_expert_groups;
        float *gs = m->moe_group_score;
        for (int gi = 0; gi < m->n_expert_groups; gi++) {
            float b1 = -FLT_MAX, b2 = -FLT_MAX;
            for (int e = gi * per; e < (gi + 1) * per; e++) {
                if (scores[e] > b1) { b2 = b1; b1 = scores[e]; }
                else if (scores[e] > b2) b2 = scores[e];
            }
            gs[gi] = b1 + (per > 1 ? b2 : 0.0f);
        }
        for (int k = 0; k < m->n_expert_groups - m->n_group_used; k++) {
            int worst = -1;
            for (int gi = 0; gi < m->n_expert_groups; gi++)
                if (gs[gi] != -FLT_MAX && (worst < 0 || gs[gi] < gs[worst])) worst = gi;
            if (worst < 0) break;
            gs[worst] = -FLT_MAX;
            for (int e = worst * per; e < (worst + 1) * per; e++) scores[e] = -FLT_MAX;
        }
    }

    // --- top-k over the selection scores, weights read from `probs`
    float denom = 0.0f;
    for (int t = 0; t < used; t++) {
        int best = 0;
        float bs = -FLT_MAX;
        for (int e = 0; e < ne; e++)
            if (scores[e] > bs) { bs = scores[e]; best = e; }
        sel[t]  = best;
        selw[t] = probs[best];
        denom  += selw[t];
        scores[best] = -FLT_MAX;                // exclude from the next round
    }

    if (m->expert_gating == EXPERT_GATE_SOFTMAX_WEIGHT) {
        float mx = selw[0];
        for (int t = 1; t < used; t++) if (selw[t] > mx) mx = selw[t];
        float ssum = 0.0f;
        for (int t = 0; t < used; t++) { selw[t] = expf(selw[t] - mx); ssum += selw[t]; }
        for (int t = 0; t < used; t++) selw[t] /= ssum;
    } else if (m->expert_norm_w) {
        // Clamped exactly as the reference: the smallest normal fp16, so a
        // degenerate all-zero row cannot divide by zero.
        if (denom < 6.103515625e-5f) denom = 6.103515625e-5f;
        for (int t = 0; t < used; t++) selw[t] /= denom;
    }
    if (m->expert_w_scale != 0.0f && m->expert_w_scale != 1.0f)
        for (int t = 0; t < used; t++) selw[t] *= m->expert_w_scale;
}

// Gemma-4 E-series per-layer embeddings, computed once per batch before the
// layer loop (llama.cpp gemma4.cpp build_inp_per_layer + project_per_layer_inputs).
// m->ple ends up laid out [token][layer][n_embd_ple], which is exactly the
// shape per_layer_model_proj already produces, so no permute is needed.
// `x` is the scaled input embedding for the batch (stride n_embd) and `out`
// receives [token][layer][n_embd_ple]. Split out from the CPU forward so the
// CUDA path can run the same arithmetic over its own staging buffers — the
// pre-pass needs a per-token row gather+dequant from the quantized token
// table, which has no device kernel (the projection matvec alone would not
// be worth the round trip), so it stays on the host for both backends.
void model_ple_prepass(model_t *m, const int32_t *tokens, int n,
                       const float *x, float *out, float *scratch) {
    int P = m->n_embd_ple, n_embd = m->n_embd;
    size_t per_tok = (size_t)m->n_layer * P;
    const float proj_scale = 1.0f / sqrtf((float)n_embd);
    const float tok_scale  = sqrtf((float)P);
    const float mix_scale  = 1.0f / sqrtf(2.0f);

    matvec_b(m->tp, out, (int)per_tok, m->ple_model_proj, x, n_embd,
             n_embd, (int)per_tok, NULL, n);
    size_t ers = ggml_row_size(m->ple_tok_embd->type, (int)per_tok);
    for (int b = 0; b < n; b++) {
        float *dst = out + (size_t)b * per_tok;
        // Same clamp as the main embedding table: token ids are untrusted.
        int32_t id = tokens[b];
        if (id < 0 || id >= m->n_vocab) id = 0;
        dequant_row(m->ple_tok_embd->type,
                    (uint8_t *)m->ple_tok_embd->data + (size_t)id * ers,
                    scratch, (int)per_tok);
        for (int l = 0; l < m->n_layer; l++) {
            float *slice = dst + (size_t)l * P;
            for (int i = 0; i < P; i++) slice[i] *= proj_scale;
            rmsnorm(slice, slice, m->ple_proj_norm, P, m->rms_eps);
            for (int i = 0; i < P; i++)
                slice[i] = (slice[i] + scratch[(size_t)l * P + i] * tok_scale)
                           * mix_scale;
        }
    }
}

// Decode path: one token, per selected expert SwiGLU -> weighted sum. Reads the
// normed input from xin and writes the FFN output back in place.
// Gated activation for a (Sw|Ge)GLU FFN: act(g) * u. act is SiLU for the
// llama family, the tanh-GELU approximation for gemma. One definition shared by
// the dense FFN and both MoE expert-FFN paths so a gemma-style GELU MoE cannot
// silently run SiLU math. The GPU path already selects f_gelu/f_silu the same
// way (enc_actmul / dense f-select).
static inline float gated_act(int act, float g, float u) {
    if (act == ACT_SWIGLU_OAI) {
        // gpt-oss, transcribed from llama.cpp's swiglu_oai kernel: the gate is
        // clamped ABOVE only, the up branch on BOTH sides, and the up branch
        // carries a +1 shift. alpha/limit are the constants that file pins.
        const float alpha = 1.702f, limit = 7.0f;
        float x = g < limit ? g : limit;
        float y = u < -limit ? -limit : (u > limit ? limit : u);
        // same overflow guard as silu below: -alpha*x past ~88 is UB under
        // -ffast-math. x <= -50 makes the sigmoid factor < 1e-36 anyway.
        float gl = x < -50.0f ? 0.0f : x / (1.0f + expf(alpha * -x));
        return gl * (y + 1.0f);
    }
    if (act == ACT_GELU) {
        float t = tanhf(0.7978845608f * (g + 0.044715f * g * g * g));
        return 0.5f * g * (1.0f + t) * u;
    }
    // silu(g) = g / (1 + e^{-g}). fp32 expf overflows past ~x=88, and this
    // build compiles with -ffast-math, under which that overflow is UB: the
    // auto-vectorized libmvec expf returns garbage rather than +inf, and the
    // huge negative gate leaks through (observed: TildeOpen-30b's last-layer
    // gates legitimately reach |g| ~ 2.7e3, corrupting every CPU decode step
    // into <unk> emissions — GPU CUDA expf saturates properly and was
    // unaffected). Below -80, |silu| < 1.5e-33: identically zero for every
    // downstream purpose, with no UB-adjacent expf call. The positive side is
    // safe as-is: expf(-g) underflows to 0, which is defined even here.
    if (g < -80.0f) return 0.0f;
    return (g / (1.0f + expf(-g))) * u;
}

// Shared always-on expert. A dense FFN over the SAME normed input the router
// saw, summed into the routed output. Qwen2-MoE scales it by sigmoid of a
// scalar router (llama.cpp writes that sigmoid as silu(x)/x); DeepSeek has no
// router tensor and adds the branch unscaled.
//
// `in` is the normed input the routed path consumed, kept aside because
// moe_ffn overwrites m->xb with its own output.
static void shexp_add(model_t *m, const layer_t *ly, const float *in,
                      int n, int xdim) {
    if (!ly->w_up_shexp) return;
    int ne = m->n_embd, nf = m->n_ff_shexp;
    matvec_b(m->tp, m->shexp_u, nf, ly->w_up_shexp,   in, ne, ne, nf, NULL, n);
    if (m->ffn_relu2 && !ly->w_gate_shexp) {
        // gate-less squared-ReLU shared expert (nemotron_h_moe): relu(up)^2, no
        // gate projection and no per-branch router (always-on, unscaled).
        for (size_t i = 0; i < (size_t)n * nf; i++) {
            float r = m->shexp_u[i] > 0.0f ? m->shexp_u[i] : 0.0f;
            m->shexp_g[i] = r * r;
        }
    } else {
        matvec_b(m->tp, m->shexp_g, nf, ly->w_gate_shexp, in, ne, ne, nf, NULL, n);
        for (size_t i = 0; i < (size_t)n * nf; i++)
            m->shexp_g[i] = gated_act(m->ffn_act, m->shexp_g[i], m->shexp_u[i]);
    }
    matvec_b(m->tp, m->shexp_o, ne, ly->w_down_shexp, m->shexp_g, nf, nf, ne, NULL, n);
    for (int b = 0; b < n; b++) {
        float gate = 1.0f;
        if (ly->ffn_gate_inp_shexp) {
            float logit = 0.0f;
            matvec_b(m->tp, &logit, 1, ly->ffn_gate_inp_shexp,
                     in + (size_t)b * ne, ne, ne, 1, NULL, 1);
            gate = 1.0f / (1.0f + expf(-logit));
        }
        float *out = m->xb + (size_t)b * xdim;
        const float *sh = m->shexp_o + (size_t)b * ne;
        for (int i = 0; i < ne; i++) out[i] += gate * sh[i];
    }
}

static void moe_ffn_token(model_t *m, const layer_t *ly, float *xin) {
    int n_embd = m->n_embd, ne = ly->n_expert, used = m->n_expert_used;
    int nff = m->n_ff_exp;
    bool probe = moe_probe_file() != NULL;
    bool trace = moe_trace_file() != NULL;
    int layer_idx = (int)(ly - m->layers);
    int   sel[256];
    float selw[256];
    float norms[256];
    moe_route(m, ly, xin, n_embd, ne, used, sel, selw, true);
    model_moe_prefetch(m, ly, sel, used);
    // RUNNER_MOE_PROBE: same lookback replay as gemma_route's caller — see
    // gemma_moe_ffn's comment. xin gets overwritten in place by this
    // function's last line, so the push below must happen before that, and
    // the probe's extra moe_route calls (reading history) must happen before
    // xin's real content is touched at all — both hold here since neither
    // occurs until after this block.
    if (probe && m->moe_probe_depth > 0) {
        size_t slotsz = (size_t)m->n_batch * n_embd;
        int max_d = m->moe_probe_depth < 3 ? m->moe_probe_depth : 3;
        for (int d = 1; d <= max_d; d++) {
            const float *h_old = m->moe_probe_hist + (size_t)(d - 1) * slotsz;
            int psel[256]; float pselw[256];
            moe_route(m, ly, h_old, n_embd, ne, used, psel, pselw, false);
            moe_probe_emit(m->fwd_pos, layer_idx, d, psel, sel, used);
        }
    }
    if (probe) moe_probe_push(m, xin, 1, n_embd, n_embd);
    for (int i = 0; i < n_embd; i++) m->moe_out[i] = 0.0f;
    for (int t = 0; t < used; t++) {
        int e = sel[t];
        float w = selw[t];
        gguf_tensor uv = moe_expert_weight(ly, 1, e, n_embd, nff);
        gguf_tensor dv = moe_expert_weight(ly, 2, e, n_embd, nff);
        matvec_b(m->tp, m->moe_up,   nff, &uv, xin, n_embd, n_embd, nff, NULL, 1);
        if (m->ffn_relu2) {
            // nemotron_h_moe: gate-less squared-ReLU experts, no gate branch —
            // relu(up)^2 into moe_gate (the down-projection input), same shape
            // the dense nemotron MLP uses.
            for (int j = 0; j < nff; j++) {
                float r = m->moe_up[j] > 0.0f ? m->moe_up[j] : 0.0f;
                m->moe_gate[j] = r * r;
            }
        } else {
        gguf_tensor gv = moe_expert_weight(ly, 0, e, n_embd, nff);
        matvec_b(m->tp, m->moe_gate, nff, &gv, xin, n_embd, n_embd, nff, NULL, 1);
        // gpt-oss per-expert biases: added to this expert's own gate/up before
        // the activation, and to its down output BEFORE the routing weight
        // scales it (llama.cpp adds down_exps_b, then multiplies by weights).
        if (ly->ffn_gate_exps_b)
            for (int j = 0; j < nff; j++)
                m->moe_gate[j] += ly->ffn_gate_exps_b[(size_t)e * nff + j];
        if (ly->ffn_up_exps_b)
            for (int j = 0; j < nff; j++)
                m->moe_up[j] += ly->ffn_up_exps_b[(size_t)e * nff + j];
        for (int j = 0; j < nff; j++)
            m->moe_gate[j] = gated_act(m->ffn_act, m->moe_gate[j], m->moe_up[j]);
        }
        matvec_b(m->tp, m->moe_dexp, n_embd, &dv, m->moe_gate,
                 nff, nff, n_embd, NULL, 1);
        if (ly->ffn_down_exps_b)
            for (int i = 0; i < n_embd; i++)
                m->moe_dexp[i] += ly->ffn_down_exps_b[(size_t)e * n_embd + i];
        if (trace) {
            float ss = 0.0f;
            for (int i = 0; i < n_embd; i++) ss += m->moe_dexp[i] * m->moe_dexp[i];
            norms[t] = sqrtf(ss);
        }
        for (int i = 0; i < n_embd; i++) m->moe_out[i] += w * m->moe_dexp[i];
    }
    moe_trace_emit(m->fwd_pos, layer_idx, sel, selw, trace ? norms : NULL, used);
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
    int n_embd = m->n_embd, ne = ly->n_expert, used = m->n_expert_used;
    int nff = m->n_ff_exp;
    bool probe = moe_probe_file() != NULL;
    bool trace = moe_trace_file() != NULL;
    int layer_idx = (int)(ly - m->layers);
    for (int b = 0; b < n; b++) {
        float *xin = m->xb + (size_t)b * xdim;
        moe_route(m, ly, xin, n_embd, ne, used,
                  m->moe_sel + (size_t)b * used, m->moe_selw + (size_t)b * used, true);
        // RUNNER_MOE_PROBE: replay against 1/2/3 layers back, before m->xb
        // is overwritten with this layer's FFN output further below.
        if (probe && m->moe_probe_depth > 0) {
            size_t slotsz = (size_t)m->n_batch * n_embd;
            int max_d = m->moe_probe_depth < 3 ? m->moe_probe_depth : 3;
            for (int d = 1; d <= max_d; d++) {
                const float *h_old = m->moe_probe_hist +
                                     (size_t)(d - 1) * slotsz + (size_t)b * n_embd;
                int psel[256]; float pselw[256];
                moe_route(m, ly, h_old, n_embd, ne, used, psel, pselw, false);
                moe_probe_emit(m->fwd_pos + b, layer_idx, d, psel,
                               m->moe_sel + (size_t)b * used, used);
            }
        }
        float *out = m->moe_out_b + (size_t)b * n_embd;
        for (int i = 0; i < n_embd; i++) out[i] = 0.0f;
    }
    if (probe) moe_probe_push(m, m->xb, n, n_embd, xdim);
    // Prefill knows the WHOLE batch's routing before any expert runs, so this
    // is the one place the prefetch gets real lead time: hand the union of
    // selected experts to the OS up front and the reads overlap the compute
    // that follows. Decode has no such luxury — routing there is inherently
    // just-in-time.
    if (m->moe_prefetch) {
        int un[256], nun = 0;
        for (int b = 0; b < n; b++) {
            const int *sel = m->moe_sel + (size_t)b * used;
            for (int t = 0; t < used; t++) {
                int e = sel[t], seen = 0;
                for (int k = 0; k < nun; k++) if (un[k] == e) { seen = 1; break; }
                if (!seen && nun < (int)(sizeof(un) / sizeof(*un))) un[nun++] = e;
            }
        }
        model_moe_prefetch(m, ly, un, nun);
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
        gguf_tensor uv = moe_expert_weight(ly, 1, e, n_embd, nff);
        gguf_tensor dv = moe_expert_weight(ly, 2, e, n_embd, nff);
        matvec_b(m->tp, m->moe_up_b,   nff, &uv, m->moe_gath,
                 n_embd, n_embd, nff, NULL, cnt);
        const float *db = ly->ffn_down_exps_b
                            ? ly->ffn_down_exps_b + (size_t)e * n_embd : NULL;
        if (m->ffn_relu2) {
            // gate-less squared-ReLU experts (nemotron_h_moe): relu(up)^2 into
            // moe_gate_b, no gate projection or per-expert gate/up biases.
            for (int c = 0; c < cnt; c++) {
                float *g = m->moe_gate_b + (size_t)c * nff;
                float *u = m->moe_up_b + (size_t)c * nff;
                for (int j = 0; j < nff; j++) {
                    float r = u[j] > 0.0f ? u[j] : 0.0f;
                    g[j] = r * r;
                }
            }
        } else {
        gguf_tensor gv = moe_expert_weight(ly, 0, e, n_embd, nff);
        matvec_b(m->tp, m->moe_gate_b, nff, &gv, m->moe_gath,
                 n_embd, n_embd, nff, NULL, cnt);
        // per-expert biases, identical ordering to the per-token path above:
        // gate/up before the activation, down before the routing weight
        const float *gb = ly->ffn_gate_exps_b
                            ? ly->ffn_gate_exps_b + (size_t)e * nff : NULL;
        const float *ub = ly->ffn_up_exps_b
                            ? ly->ffn_up_exps_b + (size_t)e * nff : NULL;
        for (int c = 0; c < cnt; c++) {
            float *g = m->moe_gate_b + (size_t)c * nff;
            float *u = m->moe_up_b + (size_t)c * nff;
            if (gb) for (int j = 0; j < nff; j++) g[j] += gb[j];
            if (ub) for (int j = 0; j < nff; j++) u[j] += ub[j];
            for (int j = 0; j < nff; j++)
                g[j] = gated_act(m->ffn_act, g[j], u[j]);
        }
        }
        matvec_b(m->tp, m->moe_dexp_b, n_embd, &dv, m->moe_gate_b,
                 nff, nff, n_embd, NULL, cnt);
        for (int c = 0; c < cnt; c++) {
            int b = m->moe_gidx[c];
            float *out = m->moe_out_b + (size_t)b * n_embd;
            const float *dx = m->moe_dexp_b + (size_t)c * n_embd;
            float w = m->moe_gw[c];
            if (trace) {
                float ss = 0.0f;
                if (db) for (int i = 0; i < n_embd; i++) { float v = dx[i] + db[i]; ss += v * v; }
                else    for (int i = 0; i < n_embd; i++) ss += dx[i] * dx[i];
                // this expert is at whichever slot t routed token b picked it
                // (sel[t]==e) — moe_route's slots are few, a linear search is
                // cheap and only runs when tracing is already on
                const int *sel = m->moe_sel + (size_t)b * used;
                for (int t = 0; t < used; t++)
                    if (sel[t] == e) { m->moe_trace_norms[(size_t)b * used + t] = sqrtf(ss); break; }
            }
            if (db) for (int i = 0; i < n_embd; i++) out[i] += w * (dx[i] + db[i]);
            else    for (int i = 0; i < n_embd; i++) out[i] += w * dx[i];
        }
    }
    if (trace)
        for (int b = 0; b < n; b++)
            moe_trace_emit(m->fwd_pos + b, layer_idx, m->moe_sel + (size_t)b * used,
                           m->moe_selw + (size_t)b * used,
                           m->moe_trace_norms + (size_t)b * used, used);
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

// Gemma-4's routed-branch router, alone: weightless-rmsnorm(h) [rss over h,
// no learned weight] * (1/sqrt(n_embd)) * gate_inp_scale -> matvec
// ffn_gate_inp -> softmax -> top-k. Factored out of gemma_moe_ffn's inline
// router so RUNNER_MOE_PROBE can replay layer `ly`'s router against an
// earlier layer's hidden state `h` without a second, driftable copy of the
// math. `dbg_dump` reproduces the original moe-logits ACT dump exactly where
// the inline version fired it (pre-softmax, real router call only — probe
// replays never dump). Uses m->moe_logits as scratch, like the original.
static void gemma_route(model_t *m, const layer_t *ly, const float *h,
                         int n_embd, int ne, int used, int *sel, float *selw,
                         bool dbg_dump) {
    float rin[n_embd];
    float inv = 1.0f / sqrtf((float)n_embd);
    float rss = 0.0f;
    for (int i = 0; i < n_embd; i++) rss += h[i] * h[i];
    rss = 1.0f / sqrtf(rss / n_embd + m->rms_eps);
    for (int i = 0; i < n_embd; i++)
        rin[i] = h[i] * rss * inv * (ly->gate_inp_scale ? ly->gate_inp_scale[i] : 1.0f);
    matvec_b(m->tp, m->moe_logits, ne, ly->ffn_gate_inp, rin, n_embd, n_embd, ne, NULL, 1);
    // Router logits before the softmax, matching llama.cpp's
    // ffn_moe_logits-N so an expert-selection flip can be seen directly.
    if (dbg_dump && dbg_act_now())
        dbg_stat("moe-logits", (int)(ly - m->layers), m->moe_logits, ne);
    float mx = m->moe_logits[0];
    for (int e = 1; e < ne; e++) if (m->moe_logits[e] > mx) mx = m->moe_logits[e];
    float ssum = 0.0f;
    for (int e = 0; e < ne; e++) { float p = expf(m->moe_logits[e] - mx); m->moe_logits[e] = p; ssum += p; }
    for (int e = 0; e < ne; e++) m->moe_logits[e] /= ssum;
    float denom = 0.0f;
    for (int t = 0; t < used; t++) {
        int best = 0; float bp = -1.0f;
        for (int e = 0; e < ne; e++) if (m->moe_logits[e] > bp) { bp = m->moe_logits[e]; best = e; }
        sel[t] = best; selw[t] = bp; denom += bp; m->moe_logits[best] = -1.0f;
    }
    for (int t = 0; t < used; t++) selw[t] /= denom;
}

// gemma-4 dual-branch MoE FFN for one layer: a dense GELU shared expert AND a
// routed top-k GELU expert set, each with its own pre/post RMSNorm sandwich,
// summed. attn_out is m->x (post-attention residual); writes the sum to m->xb
// (the caller then applies post_ffw_norm + the residual add). Per token; decode
// is n==1. Verified token-identical to llama.cpp gemma4.cpp.
static void gemma_moe_ffn(model_t *m, const layer_t *ly, int n, int xdim) {
    int n_embd = m->n_embd, ne = ly->n_expert, used = m->n_expert_used;
    int nff = m->n_ff_exp, dff = ly->n_ff;          // dff = dense shared-FFN size
    bool probe = moe_probe_file() != NULL;
    bool trace = moe_trace_file() != NULL;
    int layer_idx = (int)(ly - m->layers);
    for (int b = 0; b < n; b++) {
        const float *attn = m->x + (size_t)b * n_embd;
        float xn[n_embd], mlp[n_embd], xn2[n_embd];
        // --- dense shared MLP: rmsnorm(ffn_norm) -> GELU SwiGLU -> post_ffw_norm_1
        rmsnorm(xn, attn, ly->ffn_norm_w, n_embd, m->rms_eps);
        matvec_b(m->tp, m->hb,  dff, ly->w_gate, xn, n_embd, n_embd, dff, NULL, 1);
        matvec_b(m->tp, m->hb2, dff, ly->w_up,   xn, n_embd, n_embd, dff, NULL, 1);
        for (int j = 0; j < dff; j++) m->hb[j] = gated_act(ACT_GELU, m->hb[j], m->hb2[j]);
        matvec_b(m->tp, mlp, n_embd, ly->w_down, m->hb, dff, dff, n_embd, NULL, 1);
        rmsnorm(mlp, mlp, ly->ffn_post_norm1_w, n_embd, m->rms_eps);
        // --- routed experts: pre-norm the branch input (still needed as the
        //     expert FFN's own input — the router below uses a SEPARATE
        //     weightless norm of attn_out, computed inside gemma_route)
        rmsnorm(xn2, attn, ly->ffn_pre_norm2_w, n_embd, m->rms_eps);
        int sel[256]; float selw[256], norms[256];
        gemma_route(m, ly, attn, n_embd, ne, used, sel, selw, b == n - 1);
        model_moe_prefetch(m, ly, sel, used);
        // RUNNER_MOE_PROBE: replay THIS layer's router against the 1/2/3
        // layers-back hidden state instead of attn, and record whether that
        // earlier-and-cheaper prediction covers what attn's router actually
        // picked. moe_probe_depth counts how many prior layers this forward
        // call has pushed so far (capped at 3; 0 at layer 0).
        if (probe && m->moe_probe_depth > 0) {
            size_t slotsz = (size_t)m->n_batch * n_embd;
            int max_d = m->moe_probe_depth < 3 ? m->moe_probe_depth : 3;
            for (int d = 1; d <= max_d; d++) {
                const float *h_old = m->moe_probe_hist +
                                     (size_t)(d - 1) * slotsz + (size_t)b * n_embd;
                int psel[256]; float pselw[256];
                gemma_route(m, ly, h_old, n_embd, ne, used, psel, pselw, false);
                moe_probe_emit(m->fwd_pos + b, layer_idx, d, psel, sel, used);
            }
        }
        if (dbg_act_now() && b == n - 1) {
            fprintf(stderr, "ACT L%-3d %-16s", (int)(ly - m->layers), "moe-experts");
            for (int t = 0; t < used; t++)
                fprintf(stderr, " %d(%.4f)", sel[t], selw[t]);
            fprintf(stderr, "\n");
        }
        for (int i = 0; i < n_embd; i++) m->moe_out[i] = 0.0f;
        for (int t = 0; t < used; t++) {
            int e = sel[t];
            gguf_tensor guv = gemma_gate_up_weight(ly, e, n_embd, nff);
            matvec_b(m->tp, m->hb, 2 * nff, &guv, xn2, n_embd, n_embd, 2 * nff, NULL, 1);
            for (int j = 0; j < nff; j++)
                m->moe_gate[j] = gated_act(ACT_GELU, m->hb[j], m->hb[nff + j]);
            gguf_tensor dv = moe_expert_weight(ly, 2, e, n_embd, nff);
            matvec_b(m->tp, m->moe_dexp, n_embd, &dv, m->moe_gate, nff, nff, n_embd, NULL, 1);
            if (trace) {
                float ss = 0.0f;
                for (int i = 0; i < n_embd; i++) ss += m->moe_dexp[i] * m->moe_dexp[i];
                norms[t] = sqrtf(ss);
            }
            float sc = selw[t] * (ly->down_exps_scale ? ly->down_exps_scale[e] : 1.0f);
            for (int i = 0; i < n_embd; i++) m->moe_out[i] += sc * m->moe_dexp[i];
        }
        moe_trace_emit(m->fwd_pos + b, layer_idx, sel, selw, trace ? norms : NULL, used);
        rmsnorm(m->moe_out, m->moe_out, ly->ffn_post_norm2_w, n_embd, m->rms_eps);
        // --- combine
        float *out = m->xb + (size_t)b * xdim;
        for (int i = 0; i < n_embd; i++) out[i] = mlp[i] + m->moe_out[i];
    }
    if (probe) moe_probe_push(m, m->x, n, n_embd, n_embd);
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
        if (ly->w_gate_shexp)
            for (int b = 0; b < n; b++)
                memcpy(m->shexp_in + (size_t)b * ne,
                       m->xb + (size_t)b * xs, sizeof(float) * (size_t)ne);
        moe_ffn(m, ly, n, xs);
        shexp_add(m, ly, m->shexp_in, n, xs);
    }
    if (ly->post_ffn_norm_w)
        for (int b = 0; b < n; b++)
            rmsnorm(m->xb + (size_t)b * xs,
                    m->xb + (size_t)b * xs,
                    ly->post_ffn_norm_w, ne, m->post_norm_eps);
    for (int b = 0; b < n; b++)
        for (int i = 0; i < ne; i++)
            m->x[(size_t)b * ne + i] += m->xb[(size_t)b * xs + i];
    if (ly->out_scale != 1.0f && ly->out_scale != 0.0f)
        for (int b = 0; b < n; b++)
            for (int i = 0; i < ne; i++)
                m->x[(size_t)b * ne + i] *= ly->out_scale;
    return true;
}

// ---- recurrent-state cache seam (SSM tracer 4) -------------------------
//
// The persistent recurrent state is two buffers, both indexed [layer][...] and
// summed only within a layer, so the whole-buffer byte size is all a snapshot
// needs. conv_dim is identical in value across the three arches; the SSD/
// DeltaNet state per head differs: qwen35 keeps a [hv x hv] DeltaNet matrix,
// Mamba-2 (granite/nemotron) keeps an [hd x ds] state (hd == hv). These match
// the two calloc sites in model_alloc_buffers exactly.
static size_t recurrent_conv_bytes(const model_t *m) {
    int conv_dim = 2 * m->ssm_state * m->ssm_groups + m->ssm_inner;
    return sizeof(float) * (size_t)m->n_layer *
           (size_t)(m->ssm_conv_kernel - 1) * (size_t)conv_dim;
}
static size_t recurrent_state_bytes(const model_t *m) {
    int hv = m->ssm_inner / m->ssm_v_heads;
    size_t per_head = m->qwen35 ? (size_t)hv * (size_t)hv
                                : (size_t)hv * (size_t)m->ssm_state;
    return sizeof(float) * (size_t)m->n_layer *
           (size_t)m->ssm_v_heads * per_head;
}

bool model_has_recurrent(const model_t *m) {
    return m && (m->qwen35 || m->granite_hybrid || m->nemotron_h) &&
           m->ssm_conv_state && m->ssm_state_mem;
}

void model_recurrent_reset(model_t *m) {
    if (!model_has_recurrent(m)) return;
    memset(m->ssm_conv_state, 0, recurrent_conv_bytes(m));
    memset(m->ssm_state_mem, 0, recurrent_state_bytes(m));
    m->ssm_snap_pos = -1;   // a fresh sequence: no earlier fold to restore
}

bool model_recurrent_snapshot(model_t *m, int pos) {
    if (!model_has_recurrent(m) || !m->ssm_conv_snap || !m->ssm_state_snap)
        return false;
    memcpy(m->ssm_conv_snap, m->ssm_conv_state, recurrent_conv_bytes(m));
    memcpy(m->ssm_state_snap, m->ssm_state_mem, recurrent_state_bytes(m));
    m->ssm_snap_pos = pos;
    return true;
}

bool model_recurrent_restore(model_t *m, int pos) {
    if (!model_has_recurrent(m) || !m->ssm_conv_snap || !m->ssm_state_snap)
        return false;
    if (pos < 0 || m->ssm_snap_pos != pos) return false;   // fold not sliceable
    memcpy(m->ssm_conv_state, m->ssm_conv_snap, recurrent_conv_bytes(m));
    memcpy(m->ssm_state_mem, m->ssm_state_snap, recurrent_state_bytes(m));
    return true;
}

// Serialize/restore the recurrent fold to/from a caller-owned byte buffer, so
// the prefix cache can store the fold beside the KV rows and restore it on an
// EXACT full-prefix hit (tracer 5). The fold is a fixed-size, position-keyed
// blob (conv ring + SSD state), not sliceable — so a caller may only load it at
// the exact position it was saved. 0 bytes when the model has no recurrent state.
size_t model_recurrent_blob_bytes(const model_t *m) {
    if (!model_has_recurrent(m)) return 0;
    return recurrent_conv_bytes(m) + recurrent_state_bytes(m);
}

bool model_recurrent_blob_save(const model_t *m, uint8_t *dst) {
    if (!model_has_recurrent(m)) return false;
    size_t cb = recurrent_conv_bytes(m);
    memcpy(dst, m->ssm_conv_state, cb);
    memcpy(dst + cb, m->ssm_state_mem, recurrent_state_bytes(m));
    return true;
}

bool model_recurrent_blob_load(model_t *m, const uint8_t *src) {
    if (!model_has_recurrent(m)) return false;
    size_t cb = recurrent_conv_bytes(m);
    memcpy(m->ssm_conv_state, src, cb);
    memcpy(m->ssm_state_mem, src + cb, recurrent_state_bytes(m));
    m->ssm_snap_pos = -1;   // a freshly installed fold has no earlier snapshot
    return true;
}

float *model_forward_batch(model_t *m, const int32_t *tokens, int n, int pos,
                           bool want_logits) {
    m->fwd_pos = pos;
    m->moe_probe_depth = 0;   // the ring is scoped to one top-to-bottom layer
                              // pass, not carried across forward() calls
    // A sequence starting at pos 0 zeroes the fold. This is the ONE reset the
    // recurrent state gets: for any pos>0 the state is carried in place from the
    // previous forward, so a rewind to pos>0 must have restored it first (see
    // engine_rewind / model_recurrent_restore). Centralized so all three
    // recurrent arches — including nemotron_h, which the per-arch blocks missed
    // — reset identically.
    if (pos == 0) model_recurrent_reset(m);
    // GPU handles the leading gpu_layers. A full split (gpu_layers == n_layer)
    // returns logits directly; a partial split runs [0, gpu_layers) on the GPU,
    // leaves the boundary activation in the host x buffer + the offloaded
    // layers' KV in the host cache, and the CPU loop below finishes the rest.
    int start = 0;
    // RUNNER_DEBUG_ACT=N dumps the N-th forward pass (1-based): =1 keeps the
    // historical dump-the-first-pass behavior; =3 reaches the second DECODE
    // step — the first pass that reads KV a previous decode step wrote, which
    // is where the TildeOpen CPU-path corruption first became observable.
    int dbg = dbg_act_mode() && dbg_act_pass == dbg_act_mode() - 1;
    if (dbg_act_mode()) dbg_act_pass++;
    if (dbg)
        fprintf(stderr, "ACT ==== forward n=%d pos=%d arch=%s embd_scale=%.5f "
                "rms_eps=%.3g attn_scale=%.4f softcap=%.3f v_rmsnorm=%d "
                "kv_q8=%d n_suppress=%d\n",
                n, pos, m->arch, m->embd_scale, m->rms_eps, m->attn_scale,
                m->logit_softcap, (int)m->v_rmsnorm, (int)m->kv_q8, m->n_suppress);
    int n_embd = m->n_embd;
    // gemma-4 E-series: a partial GPU split hands off to the CPU loop below
    // starting at layer `gpu_layers`, and that hand-off overwrites m->x with
    // the boundary hidden state -- by the time the loop reads m->ple the raw
    // embedding it must be built from is already gone. Compute it up front,
    // before dispatch, with the same host arithmetic the CUDA backend's own
    // stage_ple() runs for its leading layers (model_ple_prepass depends only
    // on tokens, so the two copies agree). Full offload never reaches the
    // loop below and stages its own copy on the device side, so this would
    // be wasted work there -- skip it. Previously this table was only filled
    // when `start == 0` (CPU-only or GPU-off), so a partial split silently
    // fed the CPU-continued layers a stale/zero per-layer-embedding table.
    if (m->n_embd_ple > 0 && m->gpu && m->gpu_layers > 0 &&
        m->gpu_layers < m->n_layer) {
        size_t ers = ggml_row_size(m->tok_embd->type, n_embd);
        for (int b = 0; b < n; b++) {
            int32_t id = tokens[b];
            if (id < 0 || id >= m->n_vocab) id = 0;
            dequant_row(m->tok_embd->type,
                        (uint8_t *)m->tok_embd->data + (size_t)id * ers,
                        m->x + (size_t)b * n_embd, n_embd);
            model_embd_transform(m, m->x + (size_t)b * n_embd);
        }
        model_ple_prepass(m, tokens, n, m->x, m->ple, m->ple_tmp);
    }
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
            // Say so: a silent fallback produces correct output at a fraction
            // of the speed and no gate can see it (the fallback IS the CPU
            // oracle) — the 69b8085 MoE slice-nbytes defect hid exactly here.
            fprintf(stderr, "gpu: forward failed at runtime — releasing the "
                    "backend, continuing on CPU\n");
            gpu_disable(m);
        } else if (gpu_forward_batch(m, tokens, n, pos, false, NULL)) {
            start = m->gpu_layers;
        } else {
            fprintf(stderr, "gpu: forward failed at runtime — releasing the "
                    "backend, continuing on CPU\n");
            gpu_disable(m);
        }
    }
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
            model_embd_transform(m, m->x + (size_t)b * n_embd);
        }
        if (dbg) dbg_stat("post-embd", -1, m->x + (size_t)(n - 1) * n_embd, n_embd);
        if (m->n_embd_ple > 0)
            model_ple_prepass(m, tokens, n, m->x, m->ple, m->ple_tmp);
    }

    for (int l = start; l < m->n_layer; l++) {
        layer_t *ly = &m->layers[l];
        bool local   = model_is_swa(m, l);
        int hd       = model_head_dim(m, l);
        int n_kv     = model_n_head_kv(m, l);
        int q_dim    = model_q_dim(m, l);
        int kv_dim   = model_kv_dim(m, l);
        float scale  = model_attn_scale(m, l);
        // RUNNER_LAYER_SIM=1: per-layer input/output cosine over the residual
        // stream (ShortGPT-style block influence is 1 - cos). Diagnostic in
        // the RUNNER_ACT family: measurement the depth-prune planning needs,
        // printed per forward, aggregated by whoever is reading stderr.
        static float *sim_snap;
        static size_t sim_cap;
        bool sim = getenv("RUNNER_LAYER_SIM") != NULL;
        if (sim) {
            size_t need = (size_t)n * n_embd;
            if (need > sim_cap) {
                free(sim_snap);
                sim_snap = malloc(sizeof(float) * need);
                sim_cap = sim_snap ? need : 0;
            }
            if (sim_snap)
                memcpy(sim_snap, m->x, sizeof(float) * (size_t)n * n_embd);
            else
                sim = false;
        }
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
        if (ly->skip_mixer) goto nemo_ffn;   // nemotron_h MLP-only block
        if (ly->recurrent) {
            if (m->granite_hybrid || m->nemotron_h)
                                   mamba2_ssd_step(m, ly, l, n, xdim);
            else                   qwen35_linear(m, ly, l, n, xdim);
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
            // afmoe output gate: projected from the SAME normed input as Q,
            // consumed after attn_heads by the shared q_gate multiply below
            if (m->attn_out_gate && ly->wq_gate)
                matvec_b(m->tp, m->q_gate, q_dim, ly->wq_gate, m->xb, xdim,
                         n_embd, q_dim, NULL, n);
        }
        // gemma4 E-series shared-KV layers project Q as usual but compute no
        // K/V at all: they attend over the cache an earlier layer already
        // filled (kc_l/vc_l above resolve to that layer's rows). Their wk/wv
        // tensors exist in the file and are deliberately never read.
        bool owns_kv = model_kv_owner(m, l) == l;
        if (owns_kv) {
            matvec_b(m->tp, m->k_tmp, kv_dim, ly->wk, m->xb, xdim, n_embd, kv_dim, ly->bk, n);
            if (ly->wv)
                matvec_b(m->tp, m->v_tmp, kv_dim, ly->wv, m->xb, xdim, n_embd, kv_dim, ly->bv, n);
            else
                // gemma4 global layers have no V projection: V is the raw K
                memcpy(m->v_tmp, m->k_tmp, sizeof(float) * (size_t)n * kv_dim);
        }
        if (dbg) {
            dbg_stat("q-raw", l, m->q + (size_t)(n - 1) * q_dim, q_dim);
            if (owns_kv) {
                dbg_stat("k-raw", l, m->k_tmp + (size_t)(n - 1) * kv_dim, kv_dim);
                dbg_stat("v-raw", l, m->v_tmp + (size_t)(n - 1) * kv_dim, kv_dim);
            } else {
                fprintf(stderr, "ACT L%-3d shared-kv src=%d\n", l, model_kv_owner(m, l));
            }
        }
        for (int b = 0; b < n; b++) {
            if (ly->qnorm_w)
                qk_norm(m->q + (size_t)b * q_dim, ly->qnorm_w, m->n_head,
                        hd, m->rms_eps);
            if (model_layer_ropes(m, l)) {
                rope_apply(m, m->q + (size_t)b * q_dim, m->n_head, pos + b, l);
            } else {
                // NoPE layer: no rotation, and THIS is where the attention
                // temperature applies — llama.cpp scales Q only on the layers
                // that skipped rope, not on every layer.
                float ts = model_attn_temp(m, pos + b);
                if (ts != 1.0f)
                    for (int i = 0; i < q_dim; i++)
                        m->q[(size_t)b * q_dim + i] *= ts;
            }
            if (!owns_kv) continue;
            if (m->v_rmsnorm)
                // gemma4: weightless per-head RMS norm on V (pre-K-norm values)
                qk_norm(m->v_tmp + (size_t)b * kv_dim, NULL, n_kv, hd, m->rms_eps);
            if (ly->knorm_w)
                qk_norm(m->k_tmp + (size_t)b * kv_dim, ly->knorm_w, n_kv,
                        hd, m->rms_eps);
            if (model_layer_ropes(m, l))
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
        if (dbg && owns_kv) {
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
                            row_b, m->kv_q8, scale, ly->attn_sinks };
            tpool_run(m->tp, attn_heads, &aj, m->n_head);
            if (m->qwen35 || (m->attn_out_gate && ly->wq_gate))
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
                        ly->post_attn_norm_w, n_embd, m->post_norm_eps);
        if (m->resid_scale != 1.0f)
            for (int b = 0; b < n; b++)
                for (int i = 0; i < n_embd; i++)
                    m->xb[(size_t)b * xdim + i] *= m->resid_scale;
        for (int b = 0; b < n; b++)
            for (int i = 0; i < n_embd; i++)
                m->x[(size_t)b * n_embd + i] += m->xb[(size_t)b * xdim + i];
        if (dbg) {
            dbg_stat("post-attn-res", l, m->x + (size_t)(n - 1) * n_embd, n_embd);
        }

        // feed-forward (gated: silu for llama-family, gelu for gemma). MoE
        // layers route each token to a few experts instead of one dense FFN.
        // gemma-4 MoE is a dual branch that does its own norms off m->x directly.
    nemo_ffn:
        if (ly->skip_ffn) goto nemo_layer_end;   // nemotron_h SSM/attention block
        if (ly->moe_gemma) {
            gemma_moe_ffn(m, ly, n, xdim);  // reads m->x, writes dense⊕routed to m->xb
            goto ffn_done;
        }
        for (int b = 0; b < n; b++)
            rmsnorm(m->xb + (size_t)b * xdim, m->x + (size_t)b * n_embd,
                    ly->ffn_norm_w, n_embd, m->rms_eps);
        if (ly->is_moe) {
            if (ly->w_up_shexp)
                for (int b = 0; b < n; b++)
                    memcpy(m->shexp_in + (size_t)b * n_embd,
                           m->xb + (size_t)b * xdim,
                           sizeof(float) * (size_t)n_embd);
            moe_ffn(m, ly, n, xdim);   // reads normed m->xb, writes FFN out to m->xb
            shexp_add(m, ly, m->shexp_in, n, xdim);
        } else {
        {
        int nff = ly->n_ff;   // per-layer width (gemma-4 E2B varies it)
        if (!ly->w_gate) {
            // ungated MLP: up -> activation -> down, no gate branch
            matvec_b(m->tp, m->hb, nff, ly->w_up, m->xb, xdim, n_embd, nff, NULL, n);
            if (m->ffn_relu2) {
                // nemotron_h: gate-less squared ReLU, down(relu(up(x))^2)
                for (size_t i = 0; i < (size_t)n * nff; i++) {
                    float r = m->hb[i] > 0.0f ? m->hb[i] : 0.0f;
                    m->hb[i] = r * r;
                }
            } else {
                // Apertus xielu
                int l_i = (int)(ly - m->layers);
                float an = m->xielu_an[l_i], ap = m->xielu_ap[l_i];
                float bb = m->xielu_b[l_i],  ep = m->xielu_eps[l_i];
                for (size_t i = 0; i < (size_t)n * nff; i++)
                    m->hb[i] = xielu(m->hb[i], an, ap, bb, ep);
            }
        } else {
        matvec_b(m->tp, m->hb,  nff, ly->w_gate, m->xb, xdim, n_embd, nff, NULL, n);
        matvec_b(m->tp, m->hb2, nff, ly->w_up,   m->xb, xdim, n_embd, nff, NULL, n);
        for (size_t i = 0; i < (size_t)n * nff; i++)
            m->hb[i] = gated_act(m->ffn_act, m->hb[i], m->hb2[i]);
        }
        if (dbg) dbg_stat("ffn-act", l, m->hb + (size_t)(n - 1) * nff, nff);
        matvec_b(m->tp, m->xb, xdim, ly->w_down, m->hb, nff, nff, n_embd, NULL, n);
        }
        }
        ffn_done:
        if (dbg) dbg_stat("ffn-down", l, m->xb + (size_t)(n - 1) * xdim, n_embd);
        if (ly->post_ffn_norm_w)
            for (int b = 0; b < n; b++)
                rmsnorm(m->xb + (size_t)b * xdim, m->xb + (size_t)b * xdim,
                        ly->post_ffn_norm_w, n_embd, m->post_norm_eps);
        if (m->resid_scale != 1.0f)
            for (int b = 0; b < n; b++)
                for (int i = 0; i < n_embd; i++)
                    m->xb[(size_t)b * xdim + i] *= m->resid_scale;
        for (int b = 0; b < n; b++)
            for (int i = 0; i < n_embd; i++)
                m->x[(size_t)b * n_embd + i] += m->xb[(size_t)b * xdim + i];

    nemo_layer_end:;   // nemotron_h SSM/attention blocks land here (no FFN)
        // Per-layer embedding branch (E-series). Runs on the post-FFN residual
        // and before the layer output scale, matching gemma4.cpp's ordering.
        // ple_tmp is free once the pre-pass is done, and xb once the FFN output
        // has been added into x just above.
        if (ly->ple_gate) {
            int P = m->n_embd_ple;
            size_t per_tok = (size_t)m->n_layer * P;
            matvec_b(m->tp, m->ple_tmp, P, ly->ple_gate, m->x, n_embd,
                     n_embd, P, NULL, n);
            for (int b = 0; b < n; b++) {
                const float *slice = m->ple + (size_t)b * per_tok + (size_t)l * P;
                float *g = m->ple_tmp + (size_t)b * P;
                for (int i = 0; i < P; i++)
                    g[i] = gated_act(ACT_GELU, g[i], slice[i]);
            }
            matvec_b(m->tp, m->xb, xdim, ly->ple_proj, m->ple_tmp, P,
                     P, n_embd, NULL, n);
            for (int b = 0; b < n; b++) {
                float *u = m->xb + (size_t)b * xdim;
                rmsnorm(u, u, ly->ple_post_norm, n_embd, m->rms_eps);
                for (int i = 0; i < n_embd; i++)
                    m->x[(size_t)b * n_embd + i] += u[i];
            }
            if (dbg) dbg_stat("ple-out", l, m->xb + (size_t)(n - 1) * xdim, n_embd);
        }

        if (ly->out_scale != 1.0f && ly->out_scale != 0.0f)
            // gemma4: whole-layer output scalar, applied after both residuals
            for (int b = 0; b < n; b++)
                for (int i = 0; i < n_embd; i++)
                    m->x[(size_t)b * n_embd + i] *= ly->out_scale;
        if (dbg) dbg_stat("layer-out", l, m->x + (size_t)(n - 1) * n_embd, n_embd);
        if (sim) {
            double cs = 0;
            for (int b = 0; b < n; b++) {
                const float *pre = sim_snap + (size_t)b * n_embd;
                const float *post = m->x + (size_t)b * n_embd;
                double dp = 0, na = 0, nb2 = 0;
                for (int i = 0; i < n_embd; i++) {
                    dp += (double)pre[i] * post[i];
                    na += (double)pre[i] * pre[i];
                    nb2 += (double)post[i] * post[i];
                }
                if (na > 0 && nb2 > 0) cs += dp / (sqrt(na) * sqrt(nb2));
            }
            fprintf(stderr, "LAYERSIM l=%d n=%d cos=%.6f\n", l, n, cs / n);
        }
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
bool model_spec_verify_ok(const model_t *m) {
    return !(m->gpu && m->gpu_layers >= m->n_layer);
}

bool model_forward_batch_keep(model_t *m, const int32_t *tokens, int n, int pos) {
    if (!model_spec_verify_ok(m)) return false;
    if (!m->all_logits)
        m->all_logits = malloc(sizeof(float) * (size_t)m->spec_batch * m->n_vocab);
    // n_batch bounds the activation buffers (x/xb/q/...), spec_batch bounds
    // all_logits; a batch beyond EITHER would write past a heap block. The
    // draft window used to be clamped to spec_batch only — with a small -b
    // (n_batch < 16) that overflowed x by the difference.
    if (n > m->spec_batch || n > m->n_batch || !m->all_logits) return false;
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
