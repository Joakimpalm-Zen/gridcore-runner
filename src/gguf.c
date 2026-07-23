// GGUF v2/v3 file parser (mmap based).
#include "runner.h"
#include "compat.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>

typedef struct {
    const uint8_t *p, *end;
    bool ok;
} cursor;

static bool need(cursor *c, size_t n) {
    if (!c->ok || (size_t)(c->end - c->p) < n) { c->ok = false; return false; }
    return true;
}

static uint8_t  rd_u8 (cursor *c) { if (!need(c, 1)) return 0; return *c->p++; }
static uint16_t rd_u16(cursor *c) { uint16_t v = 0; if (need(c, 2)) { memcpy(&v, c->p, 2); c->p += 2; } return v; }
static uint32_t rd_u32(cursor *c) { uint32_t v = 0; if (need(c, 4)) { memcpy(&v, c->p, 4); c->p += 4; } return v; }
static uint64_t rd_u64(cursor *c) { uint64_t v = 0; if (need(c, 8)) { memcpy(&v, c->p, 8); c->p += 8; } return v; }
static float    rd_f32(cursor *c) { uint32_t u = rd_u32(c); float f; memcpy(&f, &u, 4); return f; }
static double   rd_f64(cursor *c) { uint64_t u = rd_u64(c); double f; memcpy(&f, &u, 8); return f; }

static bool rd_str(cursor *c, gg_str *s) {
    uint64_t n = rd_u64(c);
    if (n > SIZE_MAX - 1 || !need(c, (size_t)n)) return false;
    s->n = n;
    s->s = malloc((size_t)n + 1);
    if (!s->s) return false;
    memcpy(s->s, c->p, n);
    s->s[n] = 0;
    c->p += n;
    return true;
}

static const size_t gguf_scalar_size[] = {
    [GGUF_T_U8] = 1, [GGUF_T_I8] = 1, [GGUF_T_U16] = 2, [GGUF_T_I16] = 2,
    [GGUF_T_U32] = 4, [GGUF_T_I32] = 4, [GGUF_T_F32] = 4, [GGUF_T_BOOL] = 1,
    [GGUF_T_U64] = 8, [GGUF_T_I64] = 8, [GGUF_T_F64] = 8,
};

static bool rd_kv_value(cursor *c, gguf_kv *kv, uint32_t type) {
    switch (type) {
        case GGUF_T_U8:   kv->v.u64 = rd_u8(c);  break;
        case GGUF_T_I8:   kv->v.i64 = (int8_t)rd_u8(c); break;
        case GGUF_T_U16:  kv->v.u64 = rd_u16(c); break;
        case GGUF_T_I16:  kv->v.i64 = (int16_t)rd_u16(c); break;
        case GGUF_T_U32:  kv->v.u64 = rd_u32(c); break;
        case GGUF_T_I32:  kv->v.i64 = (int32_t)rd_u32(c); break;
        case GGUF_T_F32:  kv->v.f64 = rd_f32(c); break;
        case GGUF_T_BOOL: kv->v.b   = rd_u8(c) != 0; break;
        case GGUF_T_U64:  kv->v.u64 = rd_u64(c); break;
        case GGUF_T_I64:  kv->v.i64 = (int64_t)rd_u64(c); break;
        case GGUF_T_F64:  kv->v.f64 = rd_f64(c); break;
        case GGUF_T_STR:  return rd_str(c, &kv->str);
        case GGUF_T_ARR: {
            kv->arr_type = rd_u32(c);
            kv->arr_n    = rd_u64(c);
            if (kv->arr_type == GGUF_T_STR) {
                // Every element carries at least a u64 length prefix, so a
                // count larger than an eighth of what is left cannot be
                // satisfied by this file. Without this the count is trusted
                // as far as SIZE_MAX/sizeof(gg_str) and a 49-byte file
                // declaring arr_n = 2^32 spends ~9s reserving 64 GiB before
                // being correctly rejected. The scalar branch below already
                // bounds itself this way via need().
                if (kv->arr_n > (uint64_t)(c->end - c->p) / 8) return false;
                if (kv->arr_n > SIZE_MAX / sizeof(gg_str)) return false;
                kv->arr_str = calloc((size_t)kv->arr_n, sizeof(gg_str));
                if (kv->arr_n > 0 && !kv->arr_str) return false;
                for (uint64_t i = 0; i < kv->arr_n; i++)
                    if (!rd_str(c, &kv->arr_str[i])) return false;
            } else if (kv->arr_type < GGUF_T_F64 + 1 && kv->arr_type != GGUF_T_ARR) {
                size_t es = gguf_scalar_size[kv->arr_type];
                if (es == 0 || kv->arr_n > SIZE_MAX / es ||
                    !need(c, es * (size_t)kv->arr_n)) return false;
                kv->arr_raw = c->p;
                c->p += es * (size_t)kv->arr_n;
            } else {
                return false; // nested arrays unsupported
            }
            break;
        }
        default: return false;
    }
    return c->ok;
}

static int cmp_cstr(const void *a, const void *b) {
    return strcmp(*(const char *const *)a, *(const char *const *)b);
}

// Sorts `s` in place and returns the first duplicated string, or NULL. O(n log n).
static const char *first_duplicate_inplace(const char **s, uint64_t n) {
    if (n < 2) return NULL;
    qsort(s, n, sizeof(*s), cmp_cstr);
    for (uint64_t i = 1; i < n; i++)
        if (strcmp(s[i - 1], s[i]) == 0) return s[i];
    return NULL;
}

bool gguf_open(gguf_file *g, const char *path) {
    memset(g, 0, sizeof(*g));
    g->map = plat_mmap_ro(path, &g->map_size);
    if (!g->map || g->map_size < 24) {
        fprintf(stderr, "error: cannot open %s as a GGUF file\n", path);
        goto fail;
    }

    cursor c = { g->map, (const uint8_t *)g->map + g->map_size, true };
    if (rd_u32(&c) != 0x46554747) { // "GGUF"
        fprintf(stderr, "error: %s is not a GGUF file\n", path);
        goto fail;
    }
    g->version = rd_u32(&c);
    if (g->version < 2 || g->version > 3) {
        fprintf(stderr, "error: unsupported GGUF version %u\n", g->version);
        goto fail;
    }
    g->n_tensors = rd_u64(&c);
    g->n_kv      = rd_u64(&c);
    if (g->n_tensors > 100000 || g->n_kv > 100000) goto fail;

    g->kv = calloc(g->n_kv, sizeof(gguf_kv));
    if (g->n_kv > 0 && !g->kv) goto fail;
    for (uint64_t i = 0; i < g->n_kv; i++) {
        gg_str key = {0};
        if (!rd_str(&c, &key)) { fprintf(stderr, "error: bad GGUF metadata\n"); goto fail; }
        g->kv[i].key  = key.s;
        g->kv[i].type = rd_u32(&c);
        if (!rd_kv_value(&c, &g->kv[i], g->kv[i].type)) {
            fprintf(stderr, "error: bad GGUF metadata value for %s\n", key.s);
            goto fail;
        }
    }

    g->tensors = calloc(g->n_tensors, sizeof(gguf_tensor));
    if (g->n_tensors > 0 && !g->tensors) goto fail;
    for (uint64_t i = 0; i < g->n_tensors; i++) {
        gguf_tensor *t = &g->tensors[i];
        gg_str name = {0};
        if (!rd_str(&c, &name)) goto fail;
        if (name.n >= sizeof(t->name)) {
            free(name.s);
            fprintf(stderr, "error: invalid tensor metadata (name too long)\n");
            goto fail;
        }
        snprintf(t->name, sizeof(t->name), "%s", name.s);
        free(name.s);
        t->n_dims = rd_u32(&c);
        if (t->n_dims == 0 || t->n_dims > 4) {
            fprintf(stderr, "error: invalid tensor metadata for %s\n", t->name);
            goto fail;
        }
        t->ne[0] = t->ne[1] = t->ne[2] = t->ne[3] = 1;
        for (uint32_t d = 0; d < t->n_dims; d++) t->ne[d] = rd_u64(&c);
        t->type = rd_u32(&c);
        uint64_t off = rd_u64(&c);
        t->data = (void *)(uintptr_t)off; // fixed up below
    }
    if (!c.ok) { fprintf(stderr, "error: truncated GGUF header\n"); goto fail; }

    // Reject duplicate metadata keys and tensor names rather than silently
    // resolving to the first entry — a crafted file could otherwise shadow a
    // real geometry key or tensor with an earlier decoy (RNR-010).
    if (g->n_kv > 1) {
        const char **keys = malloc((size_t)g->n_kv * sizeof(*keys));
        if (keys) {
            for (uint64_t i = 0; i < g->n_kv; i++) keys[i] = g->kv[i].key;
            const char *d = first_duplicate_inplace(keys, g->n_kv);
            if (d) fprintf(stderr, "error: duplicate GGUF metadata key '%s'\n", d);
            free(keys);
            if (d) goto fail;
        }
    }
    if (g->n_tensors > 1) {
        const char **names = malloc((size_t)g->n_tensors * sizeof(*names));
        if (names) {
            for (uint64_t i = 0; i < g->n_tensors; i++) names[i] = g->tensors[i].name;
            const char *d = first_duplicate_inplace(names, g->n_tensors);
            if (d) fprintf(stderr, "error: duplicate GGUF tensor name '%s'\n", d);
            free(names);
            if (d) goto fail;
        }
    }

    uint64_t align = gguf_get_u32(g, "general.alignment", 32);
    if (align == 0 || (align & (align - 1))) align = 32;
    uint64_t header_end = (uint64_t)(c.p - (const uint8_t *)g->map);
    uint64_t aligned_end;
    if (!checked_u64_add(header_end, align - 1, &aligned_end)) goto invalid;
    uint64_t data_start = aligned_end & ~(align - 1);
    if (data_start > g->map_size) goto invalid;

    for (uint64_t i = 0; i < g->n_tensors; i++) {
        gguf_tensor *t = &g->tensors[i];
        if (!ggml_type_supported(t->type)) continue; // checked at use time
        int bs = ggml_block_size(t->type);
        uint64_t rows, row_blocks, row_bytes;
        if (t->ne[0] == 0 || t->ne[0] > INT_MAX || t->ne[0] % (uint64_t)bs != 0 ||
            t->ne[1] == 0 || t->ne[1] > INT_MAX ||
            t->ne[2] == 0 || t->ne[2] > INT_MAX ||
            t->ne[3] == 0 || t->ne[3] > INT_MAX ||
            !checked_u64_mul(t->ne[1], t->ne[2], &rows) ||
            !checked_u64_mul(rows, t->ne[3], &rows)) {
            fprintf(stderr, "error: invalid tensor metadata for %s\n", t->name);
            goto fail;
        }
        row_blocks = t->ne[0] / (uint64_t)bs;
        if (!checked_u64_mul(row_blocks, ggml_type_size(t->type), &row_bytes) ||
            !checked_u64_mul(row_bytes, rows, &t->nbytes)) {
            fprintf(stderr, "error: invalid tensor metadata for %s\n", t->name);
            goto fail;
        }
        uint64_t off = (uint64_t)(uintptr_t)t->data;
        if (off > g->map_size - data_start ||
            t->nbytes > g->map_size - data_start - off) {
            fprintf(stderr, "error: invalid tensor metadata for %s\n", t->name);
            goto fail;
        }
        t->data = (uint8_t *)g->map + (size_t)data_start + (size_t)off;
    }
    return true;

invalid:
    fprintf(stderr, "error: invalid tensor metadata\n");
fail:
    gguf_close(g);
    return false;
}

void gguf_close(gguf_file *g) {
    for (uint64_t i = 0; g->kv && i < g->n_kv; i++) {
        gguf_kv *kv = &g->kv[i];
        free(kv->key);
        free(kv->str.s);
        if (kv->arr_str) {
            for (uint64_t j = 0; j < kv->arr_n; j++) free(kv->arr_str[j].s);
            free(kv->arr_str);
        }
    }
    free(g->kv);
    free(g->tensors);
    plat_munmap(g->map, g->map_size);
    memset(g, 0, sizeof(*g));
}

gguf_kv *gguf_get(gguf_file *g, const char *key) {
    for (uint64_t i = 0; i < g->n_kv; i++)
        if (strcmp(g->kv[i].key, key) == 0) return &g->kv[i];
    return NULL;
}

static bool kv_is_unsigned(uint32_t t) {
    return t == GGUF_T_U8 || t == GGUF_T_U16 || t == GGUF_T_U32 || t == GGUF_T_U64;
}
static bool kv_is_signed(uint32_t t) {
    return t == GGUF_T_I8 || t == GGUF_T_I16 || t == GGUF_T_I32 || t == GGUF_T_I64;
}
// Bit-pattern finiteness/sign checks: the engine builds with -ffast-math, which
// lets the compiler assume no NaN/inf and fold float comparisons like `d>=0` or
// isfinite() to constants — so those cannot be trusted for validating untrusted
// metadata. Inspecting the IEEE bits directly is immune to that.
static bool dbl_finite(double d)  { uint64_t u; memcpy(&u, &d, 8); return (u & 0x7ff0000000000000ull) != 0x7ff0000000000000ull; }
static bool dbl_negative(double d){ uint64_t u; memcpy(&u, &d, 8); return (u >> 63) != 0; }
static bool flt_finite(float f)   { uint32_t u; memcpy(&u, &f, 4); return (u & 0x7f800000u) != 0x7f800000u; }

// Typed getters validate the source type, sign, range, and finiteness rather
// than reinterpreting whatever the union happens to hold. A negative,
// out-of-range, NaN, or type-confused metadata value from an untrusted GGUF
// becomes the caller's default (i.e. "not usable"), never a huge or
// implementation-defined geometry value that slips past a later sanity check
// (RNR-010).
uint32_t gguf_get_u32(gguf_file *g, const char *key, uint32_t dflt) {
    gguf_kv *kv = gguf_get(g, key);
    if (!kv) return dflt;
    if (kv_is_unsigned(kv->type))
        return kv->v.u64 <= UINT32_MAX ? (uint32_t)kv->v.u64 : dflt;
    if (kv_is_signed(kv->type))
        return (kv->v.i64 >= 0 && kv->v.i64 <= (int64_t)UINT32_MAX)
                 ? (uint32_t)kv->v.i64 : dflt;
    if (kv->type == GGUF_T_F32 || kv->type == GGUF_T_F64) {
        double d = kv->v.f64;   // must be a finite, non-negative, in-range integer
        if (!dbl_finite(d) || dbl_negative(d) || d > (double)UINT32_MAX ||
            d != (double)(uint32_t)d)
            return dflt;
        return (uint32_t)d;
    }
    return dflt;   // ARR, STR, BOOL are not a u32
}

float gguf_get_f32(gguf_file *g, const char *key, float dflt) {
    gguf_kv *kv = gguf_get(g, key);
    if (!kv) return dflt;
    if (kv->type == GGUF_T_F32 || kv->type == GGUF_T_F64) {
        float f = (float)kv->v.f64;              // may overflow a huge double to inf
        return flt_finite(f) ? f : dflt;         // rejects NaN input and overflow
    }
    if (kv_is_unsigned(kv->type)) return (float)kv->v.u64;
    if (kv_is_signed(kv->type))   return (float)kv->v.i64;
    return dflt;   // ARR, STR, BOOL are not a scalar number
}

bool gguf_get_bool(gguf_file *g, const char *key, bool dflt) {
    gguf_kv *kv = gguf_get(g, key);
    return (kv && kv->type == GGUF_T_BOOL) ? kv->v.b : dflt;
}

const char *gguf_get_str(gguf_file *g, const char *key, const char *dflt) {
    gguf_kv *kv = gguf_get(g, key);
    return (kv && kv->type == GGUF_T_STR) ? kv->str.s : dflt;
}

gguf_tensor *gguf_find_tensor(gguf_file *g, const char *name) {
    for (uint64_t i = 0; i < g->n_tensors; i++)
        if (strcmp(g->tensors[i].name, name) == 0) return &g->tensors[i];
    return NULL;
}
