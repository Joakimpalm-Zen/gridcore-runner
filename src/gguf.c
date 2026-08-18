// GGUF v2/v3 file parser (mmap based).
#include "gguf.h"
#include "quants.h"
#include "compat.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <sys/stat.h>

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
static float    f32_from_bits(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static double   f64_from_bits(uint64_t u) { double f; memcpy(&f, &u, 8); return f; }

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
        case GGUF_T_F32:  kv->raw = rd_u32(c); kv->v.f64 = f32_from_bits((uint32_t)kv->raw); break;
        case GGUF_T_BOOL: kv->v.b   = rd_u8(c) != 0; break;
        case GGUF_T_U64:  kv->v.u64 = rd_u64(c); break;
        case GGUF_T_I64:  kv->v.i64 = (int64_t)rd_u64(c); break;
        case GGUF_T_F64:  kv->raw = rd_u64(c); kv->v.f64 = f64_from_bits(kv->raw); break;
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

static bool gguf_open_one_x(gguf_file *g, const char *path, bool header_only) {
    memset(g, 0, sizeof(*g));
    g->header_only = header_only;
    g->map = plat_mmap_ro(path, &g->map_size);
    if (!g->map || g->map_size < 24) {
        // "cannot open X as a GGUF file" reads as "this file is corrupt", but
        // plat_mmap_ro collapses four different failures into one NULL: the
        // file is absent, it is not a regular file, it is too short to hold a
        // header, or it could not be mapped. A deleted checkpoint reported as
        // a malformed one sends the reader to the wrong place entirely, so ask
        // the filesystem which failure this was instead of blaming contents we
        // never read.
        struct stat st;
        if (stat(path, &st) != 0)
            fprintf(stderr, "error: cannot open %s: %s\n", path,
                    strerror(errno));
        else if (!S_ISREG(st.st_mode))
            fprintf(stderr, "error: %s is not a regular file\n", path);
        else if ((uint64_t)st.st_size < 24)
            fprintf(stderr,
                    "error: %s is too small to be a GGUF file (%llu bytes)\n",
                    path, (unsigned long long)st.st_size);
        else
            fprintf(stderr,
                    "error: cannot map %s (%llu bytes) — check permissions\n",
                    path, (unsigned long long)st.st_size);
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
    // real geometry key or tensor with an earlier decoy (RNR-010). The scratch
    // array is part of the check, so failing to allocate it fails the open:
    // skipping the scan would admit exactly the file the check exists to
    // refuse, and a defense that switches itself off under memory pressure is
    // not one.
    if (g->n_kv > 1) {
        const char **keys = malloc((size_t)g->n_kv * sizeof(*keys));
        if (!keys) goto fail;
        for (uint64_t i = 0; i < g->n_kv; i++) keys[i] = g->kv[i].key;
        const char *d = first_duplicate_inplace(keys, g->n_kv);
        if (d) fprintf(stderr, "error: duplicate GGUF metadata key '%s'\n", d);
        free(keys);
        if (d) goto fail;
    }
    if (g->n_tensors > 1) {
        const char **names = malloc((size_t)g->n_tensors * sizeof(*names));
        if (!names) goto fail;
        for (uint64_t i = 0; i < g->n_tensors; i++) names[i] = g->tensors[i].name;
        const char *d = first_duplicate_inplace(names, g->n_tensors);
        if (d) fprintf(stderr, "error: duplicate GGUF tensor name '%s'\n", d);
        free(names);
        if (d) goto fail;
    }

    uint64_t align = gguf_get_u32(g, "general.alignment", 32);
    if (align == 0 || (align & (align - 1))) align = 32;
    uint64_t header_end = (uint64_t)(c.p - (const uint8_t *)g->map);
    uint64_t aligned_end;
    if (!checked_u64_add(header_end, align - 1, &aligned_end)) goto invalid;
    uint64_t data_start = aligned_end & ~(align - 1);
    // The header itself must still be entirely present, in BOTH modes: a
    // header-only parse that read past its own mapping would be reading
    // garbage, not being tolerant.
    if (data_start > g->map_size && !header_only) goto invalid;
    if (data_start > g->map_size && header_only) goto invalid;

    for (uint64_t i = 0; i < g->n_tensors; i++) {
        gguf_tensor *t = &g->tensors[i];
        if (!ggml_type_supported(t->type)) {
            // The descriptor stays (the loader reports an unusable type by
            // name), but the stored offset cannot be validated without a block
            // size this build does not have — so it must not survive as a
            // pointer. It is an arbitrary 64-bit value out of an untrusted
            // file; leaving it in place parks an unbounded address in a field
            // whose every consumer treats NULL as "nothing to read here".
            t->data = NULL;
            t->nbytes = 0;
            continue;   // the type itself is checked at use time
        }
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
        if (header_only) {
            // The point of this mode: the bytes need not be here. Record what
            // the header claims the data section spans, and leave `data` NULL
            // so nothing can read through it by accident. Still reject
            // arithmetic that cannot describe a real file.
            uint64_t end;
            if (!checked_u64_add(off, t->nbytes, &end)) {
                fprintf(stderr, "error: invalid tensor metadata for %s\n", t->name);
                goto fail;
            }
            if (end > g->data_bytes) g->data_bytes = end;
            t->data = NULL;
            continue;
        }
        if (off > g->map_size - data_start ||
            t->nbytes > g->map_size - data_start - off) {
            fprintf(stderr, "error: invalid tensor metadata for %s\n", t->name);
            goto fail;
        }
        t->data = (uint8_t *)g->map + (size_t)data_start + (size_t)off;
    }
    if (header_only) {
        uint64_t total;
        if (!checked_u64_add(data_start, g->data_bytes, &total)) goto invalid;
        g->mapped_size = total;      // what the WHOLE file would be
        return true;
    }
    g->mapped_size = g->map_size;
    return true;

invalid:
    fprintf(stderr, "error: invalid tensor metadata\n");
fail:
    gguf_close(g);
    return false;
}

static bool gguf_open_one(gguf_file *g, const char *path) {
    return gguf_open_one_x(g, path, false);
}

bool gguf_open_header(gguf_file *g, const char *path) {
    return gguf_open_one_x(g, path, true);
}

static bool split_part_path(char *out, size_t cap, const char *path,
                            uint32_t current, uint32_t count, uint32_t wanted) {
    char suffix[64];
    snprintf(suffix, sizeof suffix, "-%05u-of-%05u.gguf", current + 1, count);
    size_t n = strlen(path), sn = strlen(suffix);
    if (n < sn || strcmp(path + n - sn, suffix) != 0) return false;
    int wrote = snprintf(out, cap, "%.*s-%05u-of-%05u.gguf",
                         (int)(n - sn), path, wanted + 1, count);
    return wrote > 0 && (size_t)wrote < cap;
}

bool gguf_open(gguf_file *g, const char *path) {
    memset(g, 0, sizeof(*g));
    gguf_file probe;
    if (!gguf_open_one(&probe, path)) return false;
    uint32_t count = gguf_get_u32(&probe, "split.count", 0);
    if (count <= 1) { *g = probe; return true; }
    uint32_t current = gguf_get_u32(&probe, "split.no", UINT32_MAX);
    uint32_t expected = gguf_get_u32(&probe, "split.tensors.count", UINT32_MAX);
    if (current >= count || expected == UINT32_MAX || expected > 100000 ||
        count > expected || count > 10000) {
        fprintf(stderr, "error: invalid split GGUF metadata in %s\n", path);
        gguf_close(&probe);
        return false;
    }

    char first[4096];
    if (!split_part_path(first, sizeof first, path, current, count, 0)) {
        fprintf(stderr, "error: split GGUF filename does not follow "
                "<prefix>-00001-of-%05u.gguf: %s\n", count, path);
        gguf_close(&probe);
        return false;
    }
    gguf_close(&probe);

    gguf_file *parts = calloc(count, sizeof(*parts));
    if (!parts) return false;
    uint64_t total = 0, mapped = 0;
    bool ok = true;
    for (uint32_t i = 0; i < count; i++) {
        char part_path[4096] = {0};
        if (!split_part_path(part_path, sizeof part_path, first, 0, count, i) ||
            !gguf_open_one(&parts[i], part_path)) {
            fprintf(stderr, "error: missing or unreadable split GGUF part %u/%u: %s\n",
                    i + 1, count, part_path);
            ok = false; break;
        }
        uint32_t no = gguf_get_u32(&parts[i], "split.no", UINT32_MAX);
        uint32_t nc = gguf_get_u32(&parts[i], "split.count", 0);
        uint32_t nt = gguf_get_u32(&parts[i], "split.tensors.count", UINT32_MAX);
        const char *arch0 = gguf_get_str(&parts[0], "general.architecture", NULL);
        const char *archi = gguf_get_str(&parts[i], "general.architecture", NULL);
        uint32_t align0 = gguf_get_u32(&parts[0], "general.alignment", 32);
        uint32_t aligni = gguf_get_u32(&parts[i], "general.alignment", 32);
        if (parts[i].version != parts[0].version ||
            !arch0 || !archi || strcmp(arch0, archi) != 0 || align0 != aligni ||
            no != i || nc != count || nt != expected ||
            parts[i].n_tensors > expected ||
            total > (uint64_t)expected - parts[i].n_tensors ||
            mapped > UINT64_MAX - parts[i].map_size) {
            fprintf(stderr, "error: inconsistent split GGUF metadata in %s\n",
                    part_path);
            ok = false; break;
        }
        total += parts[i].n_tensors;
        mapped += parts[i].map_size;
    }
    if (ok && total != expected) {
        fprintf(stderr, "error: split GGUF declares %u tensors but parts contain %llu\n",
                expected, (unsigned long long)total);
        ok = false;
    }
    if (!ok) {
        for (uint32_t i = 0; i < count; i++) gguf_close(&parts[i]);
        free(parts);
        return false;
    }

    gguf_tensor *merged = calloc((size_t)total, sizeof(*merged));
    void **maps = calloc(count, sizeof(*maps));
    size_t *sizes = calloc(count, sizeof(*sizes));
    if ((total && !merged) || !maps || !sizes) {
        free(merged); free(maps); free(sizes);
        for (uint32_t i = 0; i < count; i++) gguf_close(&parts[i]);
        free(parts);
        return false;
    }
    uint64_t at = 0;
    for (uint32_t i = 0; i < count; i++) {
        memcpy(merged + at, parts[i].tensors,
               (size_t)parts[i].n_tensors * sizeof(*merged));
        at += parts[i].n_tensors;
        maps[i] = parts[i].map;
        sizes[i] = parts[i].map_size;
        parts[i].map = NULL;
        parts[i].map_size = 0;
    }

    *g = parts[0];
    free(g->tensors);
    g->tensors = merged;
    g->n_tensors = total;
    g->maps = maps;
    g->map_sizes = sizes;
    g->n_maps = count;
    g->mapped_size = mapped;
    g->map = maps[0];
    g->map_size = sizes[0];
    for (uint32_t i = 1; i < count; i++) gguf_close(&parts[i]);
    free(parts);

    if (g->n_tensors > 1) {
        const char **names = malloc((size_t)g->n_tensors * sizeof(*names));
        if (!names) { gguf_close(g); return false; }
        for (uint64_t i = 0; i < g->n_tensors; i++) names[i] = g->tensors[i].name;
        const char *d = first_duplicate_inplace(names, g->n_tensors);
        if (d) fprintf(stderr, "error: duplicate tensor name across split GGUF parts: '%s'\n", d);
        free(names);
        if (d) { gguf_close(g); return false; }
    }
    return true;
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
    if (g->maps) {
        for (uint32_t i = 0; i < g->n_maps; i++)
            plat_munmap(g->maps[i], g->map_sizes[i]);
        free(g->maps);
        free(g->map_sizes);
    } else {
        plat_munmap(g->map, g->map_size);
    }
    memset(g, 0, sizeof(*g));
}

uint64_t gguf_mapped_size(const gguf_file *g) {
    return g ? g->mapped_size : 0;
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
static bool raw_f32_finite(uint32_t u) { return (u & 0x7f800000u) != 0x7f800000u; }
static bool raw_f32_negative(uint32_t u) { return (u >> 31) != 0; }
static bool raw_f64_finite(uint64_t u) { return (u & 0x7ff0000000000000ull) != 0x7ff0000000000000ull; }
static bool raw_f64_negative(uint64_t u) { return (u >> 63) != 0; }

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
        bool finite = kv->type == GGUF_T_F32
                    ? raw_f32_finite((uint32_t)kv->raw)
                    : raw_f64_finite(kv->raw);
        bool negative = kv->type == GGUF_T_F32
                      ? raw_f32_negative((uint32_t)kv->raw)
                      : raw_f64_negative(kv->raw);
        double d = kv->v.f64;   // finite, non-negative, in-range integer
        if (!finite || negative || d > (double)UINT32_MAX ||
            d != (double)(uint32_t)d)
            return dflt;
        return (uint32_t)d;
    }
    return dflt;   // ARR, STR, BOOL are not a u32
}

// Per-index variant for metadata some exports publish per layer as an
// ARRAY-typed value where others write one scalar (gemma-4 E2B's
// feed_forward_length is the motivating case: real 6144/12288 per-layer
// variation). A scalar answers every index; an integer-typed array answers
// in-range indexes with the same sign/range validation as the scalar getter;
// float/bool/string arrays, out-of-range indexes and negatives are the
// caller's default — never a read past the array or garbage geometry.
uint32_t gguf_get_u32_idx(gguf_file *g, const char *key, uint64_t idx,
                          uint32_t dflt) {
    gguf_kv *kv = gguf_get(g, key);
    if (!kv) return dflt;
    if (kv->type != GGUF_T_ARR) return gguf_get_u32(g, key, dflt);
    if (idx >= kv->arr_n || !kv->arr_raw) return dflt;
    const unsigned char *p = kv->arr_raw;
    switch (kv->arr_type) {
    case GGUF_T_U8:  return p[idx];
    case GGUF_T_I8:  { int8_t v = (int8_t)p[idx];
                       return v >= 0 ? (uint32_t)v : dflt; }
    case GGUF_T_U16: { uint16_t v; memcpy(&v, p + idx * 2, 2); return v; }
    case GGUF_T_I16: { int16_t v; memcpy(&v, p + idx * 2, 2);
                       return v >= 0 ? (uint32_t)v : dflt; }
    case GGUF_T_U32: { uint32_t v; memcpy(&v, p + idx * 4, 4); return v; }
    case GGUF_T_I32: { int32_t v; memcpy(&v, p + idx * 4, 4);
                       return v >= 0 ? (uint32_t)v : dflt; }
    case GGUF_T_U64: { uint64_t v; memcpy(&v, p + idx * 8, 8);
                       return v <= UINT32_MAX ? (uint32_t)v : dflt; }
    case GGUF_T_I64: { int64_t v; memcpy(&v, p + idx * 8, 8);
                       return (v >= 0 && v <= (int64_t)UINT32_MAX)
                                ? (uint32_t)v : dflt; }
    default: return dflt;   // F32/F64/BOOL/STR arrays are not integer geometry
    }
}

float gguf_get_f32(gguf_file *g, const char *key, float dflt) {
    gguf_kv *kv = gguf_get(g, key);
    if (!kv) return dflt;
    if (kv->type == GGUF_T_F32) {
        uint32_t u = (uint32_t)kv->raw;
        return raw_f32_finite(u) ? f32_from_bits(u) : dflt;
    }
    if (kv->type == GGUF_T_F64) {
        if (!raw_f64_finite(kv->raw)) return dflt;
        float f = (float)kv->v.f64;              // may overflow a huge double to inf
        uint32_t u; memcpy(&u, &f, 4);
        return raw_f32_finite(u) ? f : dflt;     // rejects overflow to inf
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
