// GGUF requantizer: rewrite a model with its weight matrices converted to a
// smaller quantization so one downloaded model can be fitted to each node's
// RAM/VRAM. Metadata is copied verbatim; norms/biases/1-D tensors stay F32.
#include "quants.h"
#include "fp16.h"
#include "gguf.h"
#include "compat.h"
#include "json.h"

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <sys/types.h>
#endif

// ---------------------------------------------------------------- rows

typedef struct { f16_t d; uint8_t qs[16]; } wblock_q4_0;
typedef struct { f16_t d; int8_t qs[32]; }  wblock_q8_0;

static void quantize_row_q8_0(const float *x, uint8_t *dst, int n) {
    wblock_q8_0 *b = (wblock_q8_0 *)dst;
    for (int i = 0; i < n / 32; i++, b++) {
        const float *xp = x + i * 32;
        float amax = 0;
        for (int j = 0; j < 32; j++) {
            float v = fabsf(xp[j]);
            if (v > amax) amax = v;
        }
        float d = amax / 127.0f;
        float id = d ? 1.0f / d : 0.0f;
        b->d = f32_to_f16(d);
        for (int j = 0; j < 32; j++) b->qs[j] = (int8_t)roundf(xp[j] * id);
    }
}

static void quantize_row_q4_0(const float *x, uint8_t *dst, int n) {
    wblock_q4_0 *b = (wblock_q4_0 *)dst;
    for (int i = 0; i < n / 32; i++, b++) {
        const float *xp = x + i * 32;
        float amax = 0, vmax = 0;
        for (int j = 0; j < 32; j++) {
            float v = fabsf(xp[j]);
            if (v > amax) { amax = v; vmax = xp[j]; }
        }
        float d = vmax / -8.0f;
        float id = d ? 1.0f / d : 0.0f;
        b->d = f32_to_f16(d);
        for (int j = 0; j < 16; j++) {
            int q0 = (int)(xp[j] * id + 8.5f);
            int q1 = (int)(xp[j + 16] * id + 8.5f);
            if (q0 > 15) q0 = 15;
            if (q1 > 15) q1 = 15;
            if (q0 < 0) q0 = 0;
            if (q1 < 0) q1 = 0;
            b->qs[j] = (uint8_t)(q0 | (q1 << 4));
        }
    }
}

static void quantize_row_f16(const float *x, uint8_t *dst, int n) {
    f16_t *h = (f16_t *)dst;
    for (int i = 0; i < n; i++) h[i] = f32_to_f16(x[i]);
}

// ---------------------------------------------------------------- writer

typedef struct {
    FILE *f;
    bool ok;
} writer;

static void wr(writer *w, const void *p, size_t n) {
    if (w->ok && n > 0 && fwrite(p, 1, n, w->f) != n) w->ok = false;
}
static void wr_u32(writer *w, uint32_t v) { wr(w, &v, 4); }
static void wr_u64(writer *w, uint64_t v) { wr(w, &v, 8); }
static void wr_str(writer *w, const char *s, uint64_t n) {
    wr_u64(w, n);
    if (n > SIZE_MAX) { w->ok = false; return; }
    wr(w, s, (size_t)n);
}

static int64_t wr_tell(writer *w) {
    if (!w->ok) return -1;
#ifdef _WIN32
    __int64 pos = _ftelli64(w->f);
#else
    off_t pos = ftello(w->f);
#endif
    if (pos < 0) w->ok = false;
    return (int64_t)pos;
}

static void wr_pad_to(writer *w, uint64_t target) {
    int64_t signed_pos = wr_tell(w);
    if (signed_pos < 0) return;
    uint64_t pos = (uint64_t)signed_pos;
    if (pos > target) { w->ok = false; return; }
    static const uint8_t zeros[4096] = {0};
    while (w->ok && pos < target) {
        size_t n = target - pos < sizeof(zeros) ? (size_t)(target - pos)
                                                 : sizeof(zeros);
        wr(w, zeros, n);
        pos += n;
    }
}

static bool wr_close(writer *w) {
    bool ok = w->ok;
    if (fclose(w->f) != 0) ok = false;
    w->f = NULL;
    w->ok = ok;
    return ok;
}

static const size_t kv_scalar_size[] = {
    [GGUF_T_U8] = 1, [GGUF_T_I8] = 1, [GGUF_T_U16] = 2, [GGUF_T_I16] = 2,
    [GGUF_T_U32] = 4, [GGUF_T_I32] = 4, [GGUF_T_F32] = 4, [GGUF_T_BOOL] = 1,
    [GGUF_T_U64] = 8, [GGUF_T_I64] = 8, [GGUF_T_F64] = 8,
};

static void wr_scalar(writer *w, const gguf_kv *kv, uint32_t type) {
    uint8_t b1; uint16_t b2; uint32_t b4; uint64_t b8;
    switch (type) {
        case GGUF_T_U8:   b1 = (uint8_t)kv->v.u64;  wr(w, &b1, 1); break;
        case GGUF_T_I8:   b1 = (uint8_t)(int8_t)kv->v.i64; wr(w, &b1, 1); break;
        case GGUF_T_BOOL: b1 = kv->v.b ? 1 : 0;     wr(w, &b1, 1); break;
        case GGUF_T_U16:  b2 = (uint16_t)kv->v.u64; wr(w, &b2, 2); break;
        case GGUF_T_I16:  b2 = (uint16_t)(int16_t)kv->v.i64; wr(w, &b2, 2); break;
        case GGUF_T_U32:  b4 = (uint32_t)kv->v.u64; wr(w, &b4, 4); break;
        case GGUF_T_I32:  b4 = (uint32_t)(int32_t)kv->v.i64; wr(w, &b4, 4); break;
        case GGUF_T_F32:  { uint32_t u = (uint32_t)kv->raw; wr(w, &u, 4); break; }
        case GGUF_T_U64:  b8 = kv->v.u64;           wr(w, &b8, 8); break;
        case GGUF_T_I64:  b8 = (uint64_t)kv->v.i64; wr(w, &b8, 8); break;
        case GGUF_T_F64:  wr(w, &kv->raw, 8); break;
    }
}

static void wr_kv(writer *w, const gguf_kv *kv) {
    wr_str(w, kv->key, strlen(kv->key));
    wr_u32(w, kv->type);
    if (kv->type == GGUF_T_STR) {
        wr_str(w, kv->str.s, kv->str.n);
    } else if (kv->type == GGUF_T_ARR) {
        wr_u32(w, kv->arr_type);
        wr_u64(w, kv->arr_n);
        if (kv->arr_type == GGUF_T_STR) {
            for (uint64_t i = 0; i < kv->arr_n; i++)
                wr_str(w, kv->arr_str[i].s, kv->arr_str[i].n);
        } else {
            uint64_t bytes;
            if (!checked_u64_mul(kv_scalar_size[kv->arr_type], kv->arr_n, &bytes) ||
                bytes > SIZE_MAX) w->ok = false;
            else wr(w, kv->arr_raw, (size_t)bytes);
        }
    } else {
        wr_scalar(w, kv, kv->type);
    }
}

// ------------------------------------------------------------- expert prune

// --prune-experts LIST.json: {"layer_N": [kept expert ids...], ...}. A layer
// absent from the file keeps every expert. Stacked-layout pruning needs no
// per-expert tensor split: every per-expert tensor (router weight/bias, the
// three fused expert projections and their biases, gemma's fused gate+up and
// per-expert down scale) stores expert e as a contiguous block along its
// LAST dimension (see moe_expert_weight/model.c's check_shape3 comment) —
// pruning is just "keep these blocks, in this order, drop the rest."
typedef struct { int layer; int *ids; int n_ids; } prune_layer_t;
typedef struct { prune_layer_t *layers; int n; } prune_plan_t;

static void prune_plan_free(prune_plan_t *p) {
    if (!p) return;
    for (int i = 0; i < p->n; i++) free(p->layers[i].ids);
    free(p->layers);
    p->layers = NULL;
    p->n = 0;
}

// Returns false (plan left zeroed) on any parse/validation failure — a
// malformed plan must error out, not silently prune nothing or prune the
// wrong experts. Per-layer id bounds are re-checked later against each
// tensor's own declared expert count (not known until it's read).
static bool prune_plan_load(const char *path, prune_plan_t *plan) {
    memset(plan, 0, sizeof(*plan));
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "error: cannot open --prune-experts file %s\n", path); return false; }
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return false; }
    long sz = ftell(f);
    if (sz < 0 || fseek(f, 0, SEEK_SET) != 0) { fclose(f); return false; }
    if (sz > 64 * 1024 * 1024) {
        fclose(f);
        fprintf(stderr, "error: %s is too large to be a prune plan\n", path);
        return false;
    }
    char *buf = malloc((size_t)sz + 1);
    if (!buf) { fclose(f); return false; }
    size_t rd = fread(buf, 1, (size_t)sz, f);
    fclose(f);
    buf[rd] = 0;
    jv *root = json_parse(buf, rd);
    free(buf);
    if (!root || root->type != J_OBJ) {
        fprintf(stderr, "error: %s is not a JSON object\n", path);
        if (root) jv_free(root);
        return false;
    }
    plan->layers = root->n > 0 ? calloc((size_t)root->n, sizeof(prune_layer_t)) : NULL;
    if (root->n > 0 && !plan->layers) { jv_free(root); return false; }
    bool ok = true;
    for (int i = 0; ok && i < root->n; i++) {
        int layer;
        if (sscanf(root->keys[i], "layer_%d", &layer) != 1 || layer < 0) {
            fprintf(stderr, "error: prune-plan key \"%s\" is not \"layer_N\"\n", root->keys[i]);
            ok = false;
            break;
        }
        jv *arr = root->items[i];
        if (arr->type != J_ARR || arr->n < 1) {
            fprintf(stderr, "error: prune-plan[\"layer_%d\"] must be a "
                    "non-empty array of expert ids\n", layer);
            ok = false;
            break;
        }
        int *ids = malloc(sizeof(int) * (size_t)arr->n);
        if (!ids) { ok = false; break; }
        for (int j = 0; j < arr->n; j++) {
            if (arr->items[j]->type != J_NUM || arr->items[j]->num < 0) {
                fprintf(stderr, "error: prune-plan[\"layer_%d\"][%d] is not "
                        "a non-negative integer\n", layer, j);
                free(ids);
                ok = false;
                break;
            }
            ids[j] = (int)arr->items[j]->num;
        }
        if (!ok) break;
        for (int a = 0; ok && a < arr->n; a++)
            for (int b = a + 1; b < arr->n; b++)
                if (ids[a] == ids[b]) {
                    fprintf(stderr, "error: prune-plan[\"layer_%d\"] lists "
                            "expert %d twice\n", layer, ids[a]);
                    ok = false;
                    break;
                }
        if (!ok) { free(ids); break; }
        plan->layers[plan->n].layer = layer;
        plan->layers[plan->n].ids = ids;
        plan->layers[plan->n].n_ids = arr->n;
        plan->n++;
    }
    jv_free(root);
    if (!ok) { prune_plan_free(plan); return false; }
    return true;
}

static const prune_layer_t *prune_plan_find(const prune_plan_t *plan, int layer) {
    if (!plan) return NULL;
    for (int i = 0; i < plan->n; i++)
        if (plan->layers[i].layer == layer) return &plan->layers[i];
    return NULL;
}

// Every per-expert tensor role this quantizer knows how to prune, matched by
// name (not shape — a coincidental last-dim size is not license to slice a
// tensor this tool doesn't understand). ffn_gate_inp.scale is deliberately
// absent: it is gemma-4's per-EMBEDDING-CHANNEL router input scale
// ([n_embd]), not per-expert, and must never be sliced.
static bool is_expert_tensor_suffix(const char *suffix) {
    static const char *names[] = {
        "ffn_gate_inp.weight", "ffn_gate_inp.bias",
        "ffn_gate_exps.weight", "ffn_gate_exps.bias",
        "ffn_up_exps.weight", "ffn_up_exps.bias",
        "ffn_down_exps.weight", "ffn_down_exps.bias",
        "ffn_gate_up_exps.weight",
        "ffn_down_exps.scale",
        "exp_probs_b.weight",
    };
    for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); i++)
        if (!strcmp(suffix, names[i])) return true;
    return false;
}

// "blk.N.suffix" -> layer=N, *suffix -> "suffix". False for anything not
// per-layer (token embedding, output head, ...).
static bool parse_blk_name(const char *name, int *layer, const char **suffix) {
    if (strncmp(name, "blk.", 4) != 0) return false;
    char *end;
    long v = strtol(name + 4, &end, 10);
    if (end == name + 4 || *end != '.' || v < 0 || v > INT_MAX) return false;
    *layer = (int)v;
    *suffix = end + 1;
    return true;
}

// Per-expert tensors store expert e as one contiguous block of `rows` rows
// (ggml_row_size(type, row_len) bytes each) at block index e along their
// LAST dimension — uniformly across the 1D (bias/scale: row_len=1, rows=1),
// 2D (router/exps-bias: row_len=ne[0], rows=1), and 3D (fused expert
// weights: row_len=ne[0], rows=ne[1]) shapes this tool prunes.
static void expert_block_geometry(const gguf_tensor *t, int64_t *row_len, int64_t *rows) {
    *row_len = t->n_dims >= 2 ? t->ne[0] : 1;
    *rows    = t->n_dims >= 3 ? t->ne[1] : 1;
}

// ---------------------------------------------------------------- main

static bool should_quantize(const gguf_tensor *t) {
    if (t->n_dims < 2) return false;                 // norms, biases
    if (t->ne[0] % 32 != 0) return false;
    size_t nl = strlen(t->name);
    if (nl < 7 || strcmp(t->name + nl - 7, ".weight") != 0) return false;
    if (strstr(t->name, "_norm.")) return false;
    if (strstr(t->name, "rope_freqs")) return false;
    return true;
}

static bool quantize_install_injected(void) {
    const char *inject = getenv("RUNNER_QUANTIZE_INSTALL_FAIL");
    return inject && *inject && strcmp(inject, "0");
}

static int install_finished_file(const char *tmp_path, const char *out_path) {
    if (quantize_install_injected()) return -1;
#ifdef _WIN32
    return MoveFileExA(tmp_path, out_path,
                       MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) ? 0 : -1;
#else
    return rename(tmp_path, out_path);
#endif
}

// If tensor `t` is a per-expert tensor for a layer the (already fully
// validated — see the pre-pass in quantize_gguf) prune plan covers, returns
// that layer's kept-id list; else NULL ("not pruned, write unchanged").
static const prune_layer_t *resolve_prune(const gguf_tensor *t, const prune_plan_t *plan) {
    if (!plan || plan->n == 0) return NULL;
    int layer; const char *suffix;
    if (!parse_blk_name(t->name, &layer, &suffix)) return NULL;
    if (!is_expert_tensor_suffix(suffix)) return NULL;
    return prune_plan_find(plan, layer);
}

int quantize_gguf(const char *in_path, const char *out_path, int target,
                  const char *prune_path) {
    if (target != T_KEEP && target != T_Q8_0 && target != T_Q4_0 && target != T_F16) {
        fprintf(stderr, "error: quantize target must be q8_0, q4_0, or f16\n");
        return 1;
    }
    prune_plan_t plan = {0};
    if (prune_path && !prune_plan_load(prune_path, &plan)) return 1;

    gguf_file g;
    if (!gguf_open(&g, in_path)) { prune_plan_free(&plan); return 1; }
    for (uint64_t i = 0; i < g.n_tensors; i++) {
        if (!ggml_type_supported(g.tensors[i].type)) {
            fprintf(stderr, "error: tensor %s has unsupported type %s\n",
                    g.tensors[i].name, ggml_type_name(g.tensors[i].type));
            gguf_close(&g);
            prune_plan_free(&plan);
            return 1;
        }
    }

    // Prune-plan pre-pass: resolve every plan entry against its layer's own
    // router tensor (the one tensor every is_moe layer always has), validate
    // ids are in range, and decide whether the WHOLE model ends up with one
    // uniform post-prune expert count — the only case expert_count metadata
    // can honestly be rewritten to a single new value. Done before any
    // output is written so a bad plan fails clean, before touching the
    // destination file at all.
    bool uniform_prune = false;
    int uniform_count = 0;
    if (plan.n > 0) {
        int n_moe_layers = 0, n_moe_seen = 0;
        bool any_pruned = false, all_equal = true;
        int first_count = -1;
        for (uint64_t i = 0; i < g.n_tensors; i++) {
            int layer; const char *suffix;
            if (!parse_blk_name(g.tensors[i].name, &layer, &suffix)) continue;
            if (strcmp(suffix, "ffn_gate_inp.weight") != 0) continue;
            n_moe_layers++;
            gguf_tensor *router = &g.tensors[i];
            if (router->n_dims < 2) {
                fprintf(stderr, "error: %s has %d dims, expected 2\n",
                        router->name, router->n_dims);
                gguf_close(&g); prune_plan_free(&plan); return 1;
            }
            int64_t orig_n = router->ne[router->n_dims - 1];
            const prune_layer_t *pl = prune_plan_find(&plan, layer);
            int final_count = (int)orig_n;
            if (pl) {
                n_moe_seen++;
                for (int j = 0; j < pl->n_ids; j++)
                    if (pl->ids[j] >= orig_n) {
                        fprintf(stderr, "error: prune-plan layer_%d lists "
                                "expert %d, but blk.%d only has %lld experts\n",
                                layer, pl->ids[j], layer, (long long)orig_n);
                        gguf_close(&g); prune_plan_free(&plan); return 1;
                    }
                final_count = pl->n_ids;
                if (final_count < orig_n) any_pruned = true;
            }
            if (first_count < 0) first_count = final_count;
            else if (final_count != first_count) all_equal = false;
        }
        if (n_moe_seen < plan.n) {
            // at least one "layer_N" in the plan matched no MoE layer in
            // this file at all — almost certainly the plan was built for a
            // different model or has a typo'd layer index
            fprintf(stderr, "error: --prune-experts lists %d layer(s) but "
                    "only %d matched an MoE layer in %s\n",
                    plan.n, n_moe_seen, in_path);
            gguf_close(&g); prune_plan_free(&plan); return 1;
        }
        if (any_pruned && all_equal && n_moe_seen == n_moe_layers) {
            uniform_prune = true;
            uniform_count = first_count;
        }
    }

    // Honor the file's declared data alignment. The reader derives every
    // tensor's data offset from general.alignment (gguf.c), and we copy that
    // KV verbatim — so the data MUST be laid out at the same alignment, or
    // every offset past the first misaligned tensor is read at the wrong
    // address. (RNR-002: hardcoding 32 silently corrupts any model that
    // declares 64/128.)
    uint64_t align = gguf_get_u32(&g, "general.alignment", 32);
    if (align == 0 || (align & (align - 1))) {
        fprintf(stderr, "error: general.alignment=%llu is not a power of two\n",
                (unsigned long long)align);
        gguf_close(&g);
        prune_plan_free(&plan);
        return 1;
    }
    uint64_t amask = align - 1;

    // Write beside the destination and rename on success, so a failed or
    // interrupted requant never destroys an existing good model — and an
    // in-place requant (in_path == out_path) no longer truncates its own
    // input. (RNR-015.)
    size_t tlen = strlen(out_path) + sizeof(".partial");
    char *tmp_path = malloc(tlen);
    if (!tmp_path) { gguf_close(&g); prune_plan_free(&plan); return 1; }
    snprintf(tmp_path, tlen, "%s.partial", out_path);

    FILE *f = fopen(tmp_path, "wb");
    if (!f) {
        fprintf(stderr, "error: cannot write %s\n", tmp_path);
        free(tmp_path);
        gguf_close(&g);
        prune_plan_free(&plan);
        return 1;
    }
    writer w = { f, true };

    // Rewrite {arch}.expert_count in place (still in memory, not yet
    // written) when pruning left every MoE layer at the SAME new count —
    // the only case a single global KV can honestly describe. A plan that
    // prunes layers to different counts, or leaves some layers unpruned,
    // makes expert_count ambiguous by construction; runner reads each
    // layer's real count from its own router tensor regardless (model.c),
    // so leaving the metadata at its original value there is correct, not
    // stale — it's still every layer's true ceiling, just no longer tight
    // for the pruned ones.
    if (uniform_prune) {
        const char *arch = gguf_get_str(&g, "general.architecture", "");
        char ec_key[128];
        snprintf(ec_key, sizeof(ec_key), "%s.expert_count", arch);
        gguf_kv *ec = gguf_get(&g, ec_key);
        if (ec && ec->type == GGUF_T_U32) {
            fprintf(stderr, "quantize: %s %llu -> %d (uniform prune)\n",
                    ec_key, (unsigned long long)ec->v.u64, uniform_count);
            ec->v.u64 = (uint64_t)uniform_count;
        }
    }

    wr_u32(&w, 0x46554747);
    wr_u32(&w, 3);
    wr_u64(&w, g.n_tensors);
    wr_u64(&w, g.n_kv);
    for (uint64_t i = 0; i < g.n_kv; i++) wr_kv(&w, &g.kv[i]);

    // tensor table with new types/offsets (data alignment `align`)
    uint64_t off = 0;
    int *out_type = malloc(sizeof(int) * g.n_tensors);
    uint64_t *out_off = malloc(sizeof(uint64_t) * g.n_tensors);
    // effective (possibly pruned) ne[] this tensor is WRITTEN with — equal
    // to t->ne[] unless resolve_prune() slices its expert axis
    int64_t (*eff_ne)[4] = malloc(sizeof(int64_t[4]) * (g.n_tensors ? g.n_tensors : 1));
    if ((g.n_tensors > 0 && (!out_type || !out_off || !eff_ne))) w.ok = false;
    for (uint64_t i = 0; i < g.n_tensors; i++) {
        gguf_tensor *t = &g.tensors[i];
        if (!w.ok) break;
        for (int d = 0; d < 4; d++) eff_ne[i][d] = t->ne[d];
        const prune_layer_t *pl = resolve_prune(t, &plan);
        if (pl) eff_ne[i][t->n_dims - 1] = pl->n_ids;

        if (target == T_KEEP) {
            out_type[i] = t->type;
        } else {
            // diagnostic: RUNNER_REQUANT_ONLY=<substr> converts just the
            // tensors whose name contains <substr> and leaves every other
            // tensor at its ORIGINAL type, so a model can be bisected group
            // by group
            const char *only = getenv("RUNNER_REQUANT_ONLY");
            bool filtered = only && *only && !strstr(t->name, only);
            if (filtered)
                out_type[i] = t->type;
            else
                out_type[i] = should_quantize(t) ? target
                            : (t->type == T_F16 ? T_F16 : T_F32);
            // never grow a tensor that is already smaller than the target —
            // RUNNER_FORCE_REQUANT overrides this to build same-size
            // conversions (e.g. Q4_K -> Q4_0) for diagnosing per-quant-type
            // correctness bugs
            if (!filtered && should_quantize(t) && !getenv("RUNNER_FORCE_REQUANT") &&
                ggml_row_size(t->type, t->ne[0]) <= ggml_row_size(target, t->ne[0]))
                out_type[i] = t->type;
        }
        uint64_t rows, bytes, end;
        if (!checked_u64_mul((uint64_t)eff_ne[i][1], (uint64_t)eff_ne[i][2], &rows) ||
            !checked_u64_mul(rows, (uint64_t)eff_ne[i][3], &rows) ||
            !checked_u64_mul(ggml_row_size(out_type[i], eff_ne[i][0]), rows, &bytes) ||
            !checked_u64_add(off, bytes, &end) ||
            !checked_u64_add(end, amask, &end)) {
            w.ok = false;
            break;
        }
        out_off[i] = off;
        off = end & ~amask;

        wr_str(&w, t->name, strlen(t->name));
        wr_u32(&w, t->n_dims);
        for (uint32_t d = 0; d < t->n_dims; d++) wr_u64(&w, (uint64_t)eff_ne[i][d]);
        wr_u32(&w, (uint32_t)out_type[i]);
        wr_u64(&w, out_off[i]);
    }
    // pad to data section
    int64_t hdr_end = wr_tell(&w);
    uint64_t data_start = hdr_end >= 0 ? ((uint64_t)hdr_end + amask) & ~amask : 0;
    if (hdr_end >= 0 && data_start < (uint64_t)hdr_end) w.ok = false;
    wr_pad_to(&w, data_start);

    // tensor data
    size_t max_row = 0;
    for (uint64_t i = 0; i < g.n_tensors; i++)
        if (g.tensors[i].ne[0] > max_row) max_row = g.tensors[i].ne[0];
    float *rowf = malloc(sizeof(float) * max_row);
    uint8_t *rowq = malloc(ggml_row_size(T_F32, max_row) + 64);
    if (!rowf || !rowq) w.ok = false;

    uint64_t quantized = 0, kept = 0;
    for (uint64_t i = 0; w.ok && i < g.n_tensors; i++) {
        gguf_tensor *t = &g.tensors[i];
        uint64_t want;
        if (!checked_u64_add(data_start, out_off[i], &want)) {
            w.ok = false;
            break;
        }
        wr_pad_to(&w, want);

        const prune_layer_t *pl = resolve_prune(t, &plan);
        if (pl) {
            // Pruned: copy/convert only the kept experts' blocks, IN PLAN
            // ORDER, instead of the tensor's full contiguous row range —
            // each expert is one contiguous block of exp_rows rows (1 for
            // the router/exps-bias 2D shapes, 1 element for the 1D
            // bias/scale shapes, n_ff_exp rows for the fused 3D expert
            // weights), so this is the pruning itself: the survivors, in
            // the plan's order, is the new expert index space — no
            // separate id remapping happens anywhere at runtime.
            int64_t row_len, exp_rows;
            expert_block_geometry(t, &row_len, &exp_rows);
            size_t in_row = ggml_row_size(t->type, row_len);
            size_t per_expert_in = in_row * (size_t)exp_rows;
            if ((uint32_t)out_type[i] == t->type) {
                for (int j = 0; w.ok && j < pl->n_ids; j++)
                    wr(&w, (const uint8_t *)t->data +
                           (size_t)pl->ids[j] * per_expert_in, per_expert_in);
                kept++;
                continue;
            }
            size_t out_rs = ggml_row_size(out_type[i], row_len);
            for (int j = 0; w.ok && j < pl->n_ids; j++) {
                for (int64_t r = 0; r < exp_rows; r++) {
                    int64_t src_row = (int64_t)pl->ids[j] * exp_rows + r;
                    dequant_row(t->type, (const uint8_t *)t->data +
                                (size_t)src_row * in_row, rowf, (int)row_len);
                    switch (out_type[i]) {
                        case T_Q8_0: quantize_row_q8_0(rowf, rowq, (int)row_len); break;
                        case T_Q4_0: quantize_row_q4_0(rowf, rowq, (int)row_len); break;
                        case T_F16:  quantize_row_f16(rowf, rowq, (int)row_len); break;
                        default:     memcpy(rowq, rowf, sizeof(float) * (size_t)row_len); break;
                    }
                    wr(&w, rowq, out_rs);
                }
            }
            quantized++;
            continue;
        }

        uint64_t rows = t->ne[1] * t->ne[2] * t->ne[3];
        size_t in_rs = ggml_row_size(t->type, t->ne[0]);
        size_t out_rs = ggml_row_size(out_type[i], t->ne[0]);
        if ((uint32_t)out_type[i] == t->type) {
            uint64_t bytes;
            if (!checked_u64_mul(in_rs, rows, &bytes) || bytes > SIZE_MAX)
                w.ok = false;
            else
                wr(&w, t->data, (size_t)bytes);
            kept++;
            continue;
        }
        for (uint64_t r = 0; r < rows; r++) {
            dequant_row(t->type, (uint8_t *)t->data + r * in_rs, rowf, (int)t->ne[0]);
            switch (out_type[i]) {
                case T_Q8_0: quantize_row_q8_0(rowf, rowq, (int)t->ne[0]); break;
                case T_Q4_0: quantize_row_q4_0(rowf, rowq, (int)t->ne[0]); break;
                case T_F16:  quantize_row_f16(rowf, rowq, (int)t->ne[0]); break;
                default:     memcpy(rowq, rowf, sizeof(float) * t->ne[0]); break;
            }
            wr(&w, rowq, out_rs);
        }
        quantized++;
    }
    bool write_ok = wr_close(&w);
    free(rowf); free(rowq); free(out_type); free(out_off); free(eff_ne);
    gguf_close(&g);
    prune_plan_free(&plan);

    if (!write_ok) {
        remove(tmp_path);
        fprintf(stderr, "error: failed writing quantized model %s "
                "(destination left untouched)\n", out_path);
        free(tmp_path);
        return 1;
    }

    // Install the finished file only after the temp is complete. POSIX rename()
    // atomically replaces an existing destination; Windows needs MoveFileExA
    // with replace semantics. Never remove the destination first: if the final
    // install fails, the caller must still have the previous model.
    if (install_finished_file(tmp_path, out_path) != 0) {
        remove(tmp_path);
        fprintf(stderr, "error: wrote %s but could not install it as %s "
                "(destination unchanged)\n", tmp_path, out_path);
        free(tmp_path);
        return 1;
    }
    free(tmp_path);

    fprintf(stderr, "quantize: %s -> %s (%s): %llu tensors converted, %llu kept\n",
            in_path, out_path, target == T_KEEP ? "per-tensor unchanged" : ggml_type_name(target),
            (unsigned long long)quantized, (unsigned long long)kept);
    return 0;
}
