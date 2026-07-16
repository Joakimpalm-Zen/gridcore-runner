// GGUF requantizer: rewrite a model with its weight matrices converted to a
// smaller quantization so one downloaded model can be fitted to each node's
// RAM/VRAM. Metadata is copied verbatim; norms/biases/1-D tensors stay F32.
#include "runner.h"
#include "compat.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef _WIN32
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
    uint8_t b1; uint16_t b2; uint32_t b4; uint64_t b8; float f4; double f8;
    switch (type) {
        case GGUF_T_U8:   b1 = (uint8_t)kv->v.u64;  wr(w, &b1, 1); break;
        case GGUF_T_I8:   b1 = (uint8_t)(int8_t)kv->v.i64; wr(w, &b1, 1); break;
        case GGUF_T_BOOL: b1 = kv->v.b ? 1 : 0;     wr(w, &b1, 1); break;
        case GGUF_T_U16:  b2 = (uint16_t)kv->v.u64; wr(w, &b2, 2); break;
        case GGUF_T_I16:  b2 = (uint16_t)(int16_t)kv->v.i64; wr(w, &b2, 2); break;
        case GGUF_T_U32:  b4 = (uint32_t)kv->v.u64; wr(w, &b4, 4); break;
        case GGUF_T_I32:  b4 = (uint32_t)(int32_t)kv->v.i64; wr(w, &b4, 4); break;
        case GGUF_T_F32:  f4 = (float)kv->v.f64;    wr(w, &f4, 4); break;
        case GGUF_T_U64:  b8 = kv->v.u64;           wr(w, &b8, 8); break;
        case GGUF_T_I64:  b8 = (uint64_t)kv->v.i64; wr(w, &b8, 8); break;
        case GGUF_T_F64:  f8 = kv->v.f64;           wr(w, &f8, 8); break;
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

int quantize_gguf(const char *in_path, const char *out_path, int target) {
    if (target != T_Q8_0 && target != T_Q4_0 && target != T_F16) {
        fprintf(stderr, "error: quantize target must be q8_0, q4_0, or f16\n");
        return 1;
    }
    gguf_file g;
    if (!gguf_open(&g, in_path)) return 1;
    for (uint64_t i = 0; i < g.n_tensors; i++) {
        if (!ggml_type_supported(g.tensors[i].type)) {
            fprintf(stderr, "error: tensor %s has unsupported type %s\n",
                    g.tensors[i].name, ggml_type_name(g.tensors[i].type));
            gguf_close(&g);
            return 1;
        }
    }

    FILE *f = fopen(out_path, "wb");
    if (!f) {
        fprintf(stderr, "error: cannot write %s\n", out_path);
        gguf_close(&g);
        return 1;
    }
    writer w = { f, true };

    wr_u32(&w, 0x46554747);
    wr_u32(&w, 3);
    wr_u64(&w, g.n_tensors);
    wr_u64(&w, g.n_kv);
    for (uint64_t i = 0; i < g.n_kv; i++) wr_kv(&w, &g.kv[i]);

    // tensor table with new types/offsets (alignment 32)
    uint64_t off = 0;
    int *out_type = malloc(sizeof(int) * g.n_tensors);
    uint64_t *out_off = malloc(sizeof(uint64_t) * g.n_tensors);
    if ((g.n_tensors > 0 && (!out_type || !out_off))) w.ok = false;
    for (uint64_t i = 0; i < g.n_tensors; i++) {
        gguf_tensor *t = &g.tensors[i];
        if (!w.ok) break;
        out_type[i] = should_quantize(t) ? target : (t->type == T_F16 ? T_F16 : T_F32);
        // never grow a tensor that is already smaller than the target
        if (should_quantize(t) &&
            ggml_row_size(t->type, t->ne[0]) <= ggml_row_size(target, t->ne[0]))
            out_type[i] = t->type;
        uint64_t rows, bytes, end;
        if (!checked_u64_mul(t->ne[1], t->ne[2], &rows) ||
            !checked_u64_mul(rows, t->ne[3], &rows) ||
            !checked_u64_mul(ggml_row_size(out_type[i], t->ne[0]), rows, &bytes) ||
            !checked_u64_add(off, bytes, &end) ||
            !checked_u64_add(end, 31, &end)) {
            w.ok = false;
            break;
        }
        out_off[i] = off;
        off = end & ~31ull;

        wr_str(&w, t->name, strlen(t->name));
        wr_u32(&w, t->n_dims);
        for (uint32_t d = 0; d < t->n_dims; d++) wr_u64(&w, t->ne[d]);
        wr_u32(&w, (uint32_t)out_type[i]);
        wr_u64(&w, out_off[i]);
    }
    // pad to data section
    int64_t hdr_end = wr_tell(&w);
    uint64_t data_start = hdr_end >= 0 ? ((uint64_t)hdr_end + 31) & ~31ull : 0;
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
    free(rowf); free(rowq); free(out_type); free(out_off);
    gguf_close(&g);

    if (!write_ok) {
        fprintf(stderr, "error: failed writing quantized model %s\n", out_path);
        return 1;
    }

    fprintf(stderr, "quantize: %s -> %s (%s): %llu tensors converted, %llu kept\n",
            in_path, out_path, ggml_type_name(target),
            (unsigned long long)quantized, (unsigned long long)kept);
    return 0;
}
