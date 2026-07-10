// GGUF v2/v3 file parser (mmap based).
#include "runner.h"
#include "compat.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    if (!need(c, n)) return false;
    s->n = n;
    s->s = malloc(n + 1);
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
                kv->arr_str = calloc(kv->arr_n, sizeof(gg_str));
                for (uint64_t i = 0; i < kv->arr_n; i++)
                    if (!rd_str(c, &kv->arr_str[i])) return false;
            } else if (kv->arr_type < GGUF_T_F64 + 1 && kv->arr_type != GGUF_T_ARR) {
                size_t es = gguf_scalar_size[kv->arr_type];
                if (es == 0 || !need(c, es * kv->arr_n)) return false;
                kv->arr_raw = c->p;
                c->p += es * kv->arr_n;
            } else {
                return false; // nested arrays unsupported
            }
            break;
        }
        default: return false;
    }
    return c->ok;
}

bool gguf_open(gguf_file *g, const char *path) {
    memset(g, 0, sizeof(*g));
    g->map = plat_mmap_ro(path, &g->map_size);
    if (!g->map || g->map_size < 24) {
        fprintf(stderr, "error: cannot open %s as a GGUF file\n", path);
        return false;
    }

    cursor c = { g->map, (const uint8_t *)g->map + g->map_size, true };
    if (rd_u32(&c) != 0x46554747) { // "GGUF"
        fprintf(stderr, "error: %s is not a GGUF file\n", path);
        return false;
    }
    g->version = rd_u32(&c);
    if (g->version < 2 || g->version > 3) {
        fprintf(stderr, "error: unsupported GGUF version %u\n", g->version);
        return false;
    }
    g->n_tensors = rd_u64(&c);
    g->n_kv      = rd_u64(&c);
    if (g->n_tensors > 100000 || g->n_kv > 100000) return false;

    g->kv = calloc(g->n_kv, sizeof(gguf_kv));
    for (uint64_t i = 0; i < g->n_kv; i++) {
        gg_str key = {0};
        if (!rd_str(&c, &key)) { fprintf(stderr, "error: bad GGUF metadata\n"); return false; }
        g->kv[i].key  = key.s;
        g->kv[i].type = rd_u32(&c);
        if (!rd_kv_value(&c, &g->kv[i], g->kv[i].type)) {
            fprintf(stderr, "error: bad GGUF metadata value for %s\n", key.s);
            return false;
        }
    }

    g->tensors = calloc(g->n_tensors, sizeof(gguf_tensor));
    for (uint64_t i = 0; i < g->n_tensors; i++) {
        gguf_tensor *t = &g->tensors[i];
        gg_str name = {0};
        if (!rd_str(&c, &name)) return false;
        snprintf(t->name, sizeof(t->name), "%s", name.s);
        free(name.s);
        t->n_dims = rd_u32(&c);
        if (t->n_dims > 4) return false;
        t->ne[0] = t->ne[1] = t->ne[2] = t->ne[3] = 1;
        for (uint32_t d = 0; d < t->n_dims; d++) t->ne[d] = rd_u64(&c);
        t->type = rd_u32(&c);
        uint64_t off = rd_u64(&c);
        t->data = (void *)(uintptr_t)off; // fixed up below
    }
    if (!c.ok) { fprintf(stderr, "error: truncated GGUF header\n"); return false; }

    uint64_t align = gguf_get_u32(g, "general.alignment", 32);
    if (align == 0 || (align & (align - 1))) align = 32;
    size_t header_end = (size_t)(c.p - (const uint8_t *)g->map);
    size_t data_start = (header_end + align - 1) & ~(align - 1);

    for (uint64_t i = 0; i < g->n_tensors; i++) {
        gguf_tensor *t = &g->tensors[i];
        if (!ggml_type_supported(t->type)) continue; // checked at use time
        uint64_t n_elem = t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3];
        int bs = ggml_block_size(t->type);
        if (t->ne[0] % bs != 0) { fprintf(stderr, "error: tensor %s bad shape\n", t->name); return false; }
        t->nbytes = ggml_row_size(t->type, t->ne[0]) * (n_elem / t->ne[0]);
        uint64_t off = (uint64_t)(uintptr_t)t->data;
        if (data_start + off + t->nbytes > g->map_size) {
            fprintf(stderr, "error: tensor %s out of bounds\n", t->name);
            return false;
        }
        t->data = (uint8_t *)g->map + data_start + off;
    }
    return true;
}

void gguf_close(gguf_file *g) {
    plat_munmap(g->map, g->map_size);
    g->map = NULL;
}

gguf_kv *gguf_get(gguf_file *g, const char *key) {
    for (uint64_t i = 0; i < g->n_kv; i++)
        if (strcmp(g->kv[i].key, key) == 0) return &g->kv[i];
    return NULL;
}

uint32_t gguf_get_u32(gguf_file *g, const char *key, uint32_t dflt) {
    gguf_kv *kv = gguf_get(g, key);
    if (!kv) return dflt;
    if (kv->type == GGUF_T_F32 || kv->type == GGUF_T_F64) return (uint32_t)kv->v.f64;
    return (uint32_t)kv->v.u64;
}

float gguf_get_f32(gguf_file *g, const char *key, float dflt) {
    gguf_kv *kv = gguf_get(g, key);
    if (!kv) return dflt;
    if (kv->type == GGUF_T_F32 || kv->type == GGUF_T_F64) return (float)kv->v.f64;
    return (float)kv->v.u64;
}

bool gguf_get_bool(gguf_file *g, const char *key, bool dflt) {
    gguf_kv *kv = gguf_get(g, key);
    return kv ? kv->v.b : dflt;
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
