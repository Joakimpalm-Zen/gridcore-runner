// SIMD dot/dequant kernels vs an independent reference.
//
// vec_dot dispatches to AVX2 or NEON kernels depending on the build; this
// test checks every supported format against a double-precision dot over
// dequantized weights. For the formats whose *block dequant* also has a SIMD
// path (Q8_0, Q4_K, Q6_K, MXFP4), the reference decode is reimplemented here
// from the format spec so the test does not trust the code under test.
// q8_quant_row must be byte-identical to the scalar definition on every
// platform (the KV cache is compared across runs), so that one is exact.
#include "quants.h"
#include "fp16.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define QK 32
#define QK_K 256

// block layouts, copied from quants.c (kept in sync by the size asserts below)
typedef struct { f16_t d; uint8_t qs[QK / 2]; }                  block_q4_0;
typedef struct { f16_t d, m; uint8_t qs[QK / 2]; }               block_q4_1;
typedef struct { f16_t d; uint8_t qh[4]; uint8_t qs[QK / 2]; }   block_q5_0;
typedef struct { f16_t d, m; uint8_t qh[4]; uint8_t qs[QK / 2]; } block_q5_1;
typedef struct { f16_t d; int8_t qs[QK]; }                       block_q8_0;
typedef struct { uint8_t scales[QK_K / 16]; uint8_t qs[QK_K / 4]; f16_t d, dmin; } block_q2_K;
typedef struct { uint8_t hmask[QK_K / 8]; uint8_t qs[QK_K / 4]; uint8_t scales[12]; f16_t d; } block_q3_K;
typedef struct { f16_t d, dmin; uint8_t scales[12]; uint8_t qs[QK_K / 2]; } block_q4_K;
typedef struct { f16_t d, dmin; uint8_t scales[12]; uint8_t qh[QK_K / 8]; uint8_t qs[QK_K / 2]; } block_q5_K;
typedef struct { uint8_t ql[QK_K / 2]; uint8_t qh[QK_K / 4]; int8_t scales[QK_K / 16]; f16_t d; } block_q6_K;
typedef struct { f16_t d; uint8_t qs[QK / 2]; }                  block_iq4_nl;
typedef struct { f16_t d; uint16_t scales_h; uint8_t scales_l[QK_K / 64]; uint8_t qs[QK_K / 2]; } block_iq4_xs;
typedef struct { uint8_t e; uint8_t qs[QK / 2]; }                block_mxfp4;
typedef struct { f16_t d; uint16_t qs[QK_K / 8]; } block_iq2_xxs;
typedef struct { f16_t d; uint16_t qs[QK_K / 8]; uint8_t scales[QK_K / 32]; } block_iq2_xs;
typedef struct { f16_t d; uint8_t qs[QK_K / 4]; uint8_t qh[QK_K / 32]; uint8_t scales[QK_K / 32]; } block_iq2_s;
typedef struct { f16_t d; uint8_t qs[3 * QK_K / 8]; } block_iq3_xxs;
typedef struct { f16_t d; uint8_t qs[QK_K / 4]; uint8_t qh[QK_K / 32]; uint8_t signs[QK_K / 8]; uint8_t scales[QK_K / 64]; } block_iq3_s;
typedef struct { f16_t d; uint8_t qs[QK_K / 8]; uint16_t qh[QK_K / 32]; } block_iq1_s;
typedef struct { uint8_t qs[QK_K / 8]; uint8_t qh[QK_K / 16]; uint8_t scales[QK_K / 32]; } block_iq1_m;
// the i-quant references need the same codebooks the kernels read; the
// header is data extracted verbatim from llama.cpp, shared, not duplicated
#include "../src/quants_iq_grids.h"
#define IQ1S_DELTA 0.125

static const double kv_mxfp4[16] = {
    0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6,
};

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { \
    fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
    fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); g_fail++; } } while (0)

// deterministic PRNG so failures reproduce
static uint64_t g_rng = 0x243F6A8885A308D3ull;
static uint32_t rnd32(void) {
    g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17;
    return (uint32_t)(g_rng >> 32);
}
static float frnd(void) { return (float)(int32_t)rnd32() / 2147483648.0f; } // [-1,1)
static f16_t sane_f16(void) { return f32_to_f16(0.01f + 0.5f * fabsf(frnd())); }

// ------------------------------------------------- independent block decode

static void ref_dq_iq2_xxs(const block_iq2_xxs *b, double *y) {
    double d = f16_to_f32(b->d);
    uint32_t aux32[2];
    const uint8_t *aux8 = (const uint8_t *)aux32;
    for (int ib32 = 0; ib32 < QK_K / 32; ib32++) {
        memcpy(aux32, b->qs + 4 * ib32, 8);
        double db = d * (0.5 + (aux32[1] >> 28)) * 0.25;
        for (int l = 0; l < 4; l++) {
            const uint8_t *grid = (const uint8_t *)(iq2xxs_grid + aux8[l]);
            uint8_t signs = ksigns_iq2xs[(aux32[1] >> 7 * l) & 127];
            for (int j = 0; j < 8; j++)
                y[j] = db * grid[j] * (signs & kmask_iq2xs[j] ? -1.0 : 1.0);
            y += 8;
        }
    }
}

static void ref_dq_iq2_xs(const block_iq2_xs *b, double *y) {
    double d = f16_to_f32(b->d);
    for (int ib32 = 0; ib32 < QK_K / 32; ib32++) {
        double db[2] = { d * (0.5 + (b->scales[ib32] & 0xf)) * 0.25,
                         d * (0.5 + (b->scales[ib32] >> 4)) * 0.25 };
        for (int l = 0; l < 4; l++) {
            const uint8_t *grid =
                (const uint8_t *)(iq2xs_grid + (b->qs[4 * ib32 + l] & 511));
            uint8_t signs = ksigns_iq2xs[b->qs[4 * ib32 + l] >> 9];
            for (int j = 0; j < 8; j++)
                y[j] = db[l / 2] * grid[j] * (signs & kmask_iq2xs[j] ? -1.0 : 1.0);
            y += 8;
        }
    }
}

static void ref_dq_iq2_s(const block_iq2_s *b, double *y) {
    double d = f16_to_f32(b->d);
    const uint8_t *qs = b->qs, *qh = b->qh, *signs = b->qs + QK_K / 8;
    for (int ib32 = 0; ib32 < QK_K / 32; ib32++) {
        double db[2] = { d * (0.5 + (b->scales[ib32] & 0xf)) * 0.25,
                         d * (0.5 + (b->scales[ib32] >> 4)) * 0.25 };
        for (int l = 0; l < 4; l++) {
            const uint8_t *grid = (const uint8_t *)
                (iq2s_grid + (qs[l] | ((qh[ib32] << (8 - 2 * l)) & 0x300)));
            for (int j = 0; j < 8; j++)
                y[j] = db[l / 2] * grid[j] * (signs[l] & kmask_iq2xs[j] ? -1.0 : 1.0);
            y += 8;
        }
        qs += 4;
        signs += 4;
    }
}

static void ref_dq_iq3_xxs(const block_iq3_xxs *b, double *y) {
    double d = f16_to_f32(b->d);
    const uint8_t *qs = b->qs, *sas = b->qs + QK_K / 4;
    uint32_t aux32;
    for (int ib32 = 0; ib32 < QK_K / 32; ib32++) {
        memcpy(&aux32, sas + 4 * ib32, 4);
        double db = d * (0.5 + (aux32 >> 28)) * 0.5;
        for (int l = 0; l < 4; l++) {
            uint8_t signs = ksigns_iq2xs[(aux32 >> 7 * l) & 127];
            const uint8_t *g1 = (const uint8_t *)(iq3xxs_grid + qs[2 * l]);
            const uint8_t *g2 = (const uint8_t *)(iq3xxs_grid + qs[2 * l + 1]);
            for (int j = 0; j < 4; j++) {
                y[j]     = db * g1[j] * (signs & kmask_iq2xs[j] ? -1.0 : 1.0);
                y[j + 4] = db * g2[j] * (signs & kmask_iq2xs[j + 4] ? -1.0 : 1.0);
            }
            y += 8;
        }
        qs += 8;
    }
}

static void ref_dq_iq3_s(const block_iq3_s *b, double *y) {
    double d = f16_to_f32(b->d);
    const uint8_t *qs = b->qs, *qh = b->qh, *signs = b->signs;
    for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
        double db1 = d * (1 + 2 * (b->scales[ib32 / 2] & 0xf));
        double db2 = d * (1 + 2 * (b->scales[ib32 / 2] >> 4));
        for (int k = 0; k < 2; k++) {
            double db = k ? db2 : db1;
            for (int l = 0; l < 4; l++) {
                const uint8_t *g1 = (const uint8_t *)
                    (iq3s_grid + (qs[2 * l] | ((qh[k] << (8 - 2 * l)) & 256)));
                const uint8_t *g2 = (const uint8_t *)
                    (iq3s_grid + (qs[2 * l + 1] | ((qh[k] << (7 - 2 * l)) & 256)));
                for (int j = 0; j < 4; j++) {
                    y[j]     = db * g1[j] * (signs[l] & kmask_iq2xs[j] ? -1.0 : 1.0);
                    y[j + 4] = db * g2[j] * (signs[l] & kmask_iq2xs[j + 4] ? -1.0 : 1.0);
                }
                y += 8;
            }
            qs += 8;
            signs += 4;
        }
        qh += 2;
    }
}

static void ref_dq_iq1_s(const block_iq1_s *b, double *y) {
    double d = f16_to_f32(b->d);
    const uint8_t *qs = b->qs;
    const uint16_t *qh = b->qh;
    for (int ib = 0; ib < QK_K / 32; ib++) {
        double dl = d * (2 * ((qh[ib] >> 12) & 7) + 1);
        double delta = qh[ib] & 0x8000 ? -IQ1S_DELTA : IQ1S_DELTA;
        for (int l = 0; l < 4; l++) {
            const int8_t *grid = (const int8_t *)
                (iq1s_grid + (qs[l] | (((qh[ib] >> 3 * l) & 7) << 8)));
            for (int j = 0; j < 8; j++) y[j] = dl * (grid[j] + delta);
            y += 8;
        }
        qs += 4;
    }
}

static void ref_dq_iq1_m(const block_iq1_m *b, double *y) {
    const uint16_t *sc = (const uint16_t *)b->scales;
    f16_t sd = (f16_t)((sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) |
                       ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000));
    double d = f16_to_f32(sd);
    const uint8_t *qs = b->qs, *qh = b->qh;
    for (int ib = 0; ib < QK_K / 32; ib++) {
        double dl1 = d * (2 * ((sc[ib / 2] >> (6 * (ib % 2) + 0)) & 0x7) + 1);
        double dl2 = d * (2 * ((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 0x7) + 1);
        uint16_t idx[4] = { (uint16_t)(qs[0] | ((qh[0] << 8) & 0x700)),
                            (uint16_t)(qs[1] | ((qh[0] << 4) & 0x700)),
                            (uint16_t)(qs[2] | ((qh[1] << 8) & 0x700)),
                            (uint16_t)(qs[3] | ((qh[1] << 4) & 0x700)) };
        double delta[4] = { qh[0] & 0x08 ? -IQ1S_DELTA : IQ1S_DELTA,
                            qh[0] & 0x80 ? -IQ1S_DELTA : IQ1S_DELTA,
                            qh[1] & 0x08 ? -IQ1S_DELTA : IQ1S_DELTA,
                            qh[1] & 0x80 ? -IQ1S_DELTA : IQ1S_DELTA };
        for (int l = 0; l < 4; l++) {
            const int8_t *grid = (const int8_t *)(iq1s_grid + idx[l]);
            double dl = l < 2 ? dl1 : dl2;
            for (int j = 0; j < 8; j++) y[j] = dl * (grid[j] + delta[l]);
            y += 8;
        }
        qs += 4;
        qh += 2;
    }
}


static void ref_dq_q8_0(const block_q8_0 *b, double *y) {
    double d = f16_to_f32(b->d);
    for (int j = 0; j < QK; j++) y[j] = d * b->qs[j];
}

static void ref_dq_mxfp4(const block_mxfp4 *b, double *y) {
    double d = ldexp(1.0, (int)b->e - 127);
    for (int j = 0; j < 16; j++) {
        y[j]      = kv_mxfp4[b->qs[j] & 0xF] * d;
        y[j + 16] = kv_mxfp4[b->qs[j] >> 4]  * d;
    }
}

static void ref_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4)  | ((q[j    ] >> 6) << 4);
    }
}

static void ref_dq_q4_K(const block_q4_K *b, double *y) {
    double d = f16_to_f32(b->d), dmin = f16_to_f32(b->dmin);
    const uint8_t *q = b->qs;
    int is = 0;
    for (int j = 0; j < QK_K; j += 64) {
        uint8_t sc, mn;
        ref_scale_min_k4(is + 0, b->scales, &sc, &mn);
        double d1 = d * sc, m1 = dmin * mn;
        ref_scale_min_k4(is + 1, b->scales, &sc, &mn);
        double d2 = d * sc, m2 = dmin * mn;
        for (int l = 0; l < 32; l++) y[l]      = d1 * (q[l] & 0xF) - m1;
        for (int l = 0; l < 32; l++) y[l + 32] = d2 * (q[l] >> 4)  - m2;
        q += 32; is += 2; y += 64;
    }
}

static void ref_dq_q6_K(const block_q6_K *b, double *y) {
    double d = f16_to_f32(b->d);
    const uint8_t *ql = b->ql, *qh = b->qh;
    const int8_t *sc = b->scales;
    for (int half = 0; half < 2; half++) {
        for (int l = 0; l < 32; l++) {
            int is = l / 16;
            int q1 = (int)((ql[l]      & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            int q2 = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            int q3 = (int)((ql[l]      >> 4)  | (((qh[l] >> 4) & 3) << 4)) - 32;
            int q4 = (int)((ql[l + 32] >> 4)  | (((qh[l] >> 6) & 3) << 4)) - 32;
            y[l]      = d * sc[is + 0] * q1;
            y[l + 32] = d * sc[is + 2] * q2;
            y[l + 64] = d * sc[is + 4] * q3;
            y[l + 96] = d * sc[is + 6] * q4;
        }
        y += 128; ql += 64; qh += 32; sc += 8;
    }
}

// ------------------------------------------------------------ row builders

// fill a row of blocks with random payloads and sane scale fields
static void make_row(int type, uint8_t *row, int n) {
    int bs = ggml_block_size(type);
    size_t ts = ggml_type_size(type);
    for (int i = 0; i < n / bs; i++) {
        uint8_t *p = row + i * ts;
        for (size_t j = 0; j < ts; j++) p[j] = (uint8_t)rnd32();
        switch (type) {
            case T_F32: case T_F16: case T_BF16: break;
            case T_Q4_0: ((block_q4_0 *)p)->d = sane_f16(); break;
            case T_Q4_1: ((block_q4_1 *)p)->d = sane_f16();
                         ((block_q4_1 *)p)->m = sane_f16(); break;
            case T_Q5_0: ((block_q5_0 *)p)->d = sane_f16(); break;
            case T_Q5_1: ((block_q5_1 *)p)->d = sane_f16();
                         ((block_q5_1 *)p)->m = sane_f16(); break;
            case T_Q8_0: ((block_q8_0 *)p)->d = sane_f16(); break;
            case T_Q2_K: ((block_q2_K *)p)->d = sane_f16();
                         ((block_q2_K *)p)->dmin = sane_f16(); break;
            case T_Q3_K: ((block_q3_K *)p)->d = sane_f16(); break;
            case T_Q4_K: ((block_q4_K *)p)->d = sane_f16();
                         ((block_q4_K *)p)->dmin = sane_f16(); break;
            case T_Q5_K: ((block_q5_K *)p)->d = sane_f16();
                         ((block_q5_K *)p)->dmin = sane_f16(); break;
            case T_Q6_K: ((block_q6_K *)p)->d = sane_f16(); break;
            case T_IQ4_NL: ((block_iq4_nl *)p)->d = sane_f16(); break;
            case T_IQ4_XS: ((block_iq4_xs *)p)->d = sane_f16(); break;
            case T_MXFP4: ((block_mxfp4 *)p)->e = (uint8_t)(117 + rnd32() % 21); break;
            case T_IQ2_XXS: ((block_iq2_xxs *)p)->d = sane_f16(); break;
            case T_IQ2_XS: ((block_iq2_xs *)p)->d = sane_f16(); break;
            case T_IQ2_S: ((block_iq2_s *)p)->d = sane_f16(); break;
            case T_IQ3_XXS: ((block_iq3_xxs *)p)->d = sane_f16(); break;
            case T_IQ3_S: ((block_iq3_s *)p)->d = sane_f16(); break;
            case T_IQ1_S: ((block_iq1_s *)p)->d = sane_f16(); break;
            case T_IQ1_M: {
                // the fp16 block scale hides in the top nibbles of the four
                // scale words; plant a sane one there or the reference is NaN
                uint16_t *sc = (uint16_t *)((block_iq1_m *)p)->scales;
                uint16_t d16 = sane_f16();
                sc[0] = (uint16_t)((sc[0] & 0x0fff) | ((d16 & 0x000f) << 12));
                sc[1] = (uint16_t)((sc[1] & 0x0fff) | ((d16 & 0x00f0) << 8));
                sc[2] = (uint16_t)((sc[2] & 0x0fff) | ((d16 & 0x0f00) << 4));
                sc[3] = (uint16_t)((sc[3] & 0x0fff) | (d16 & 0xf000));
                break;
            }
        }
    }
    if (type == T_F32) { float *f = (float *)row; for (int i = 0; i < n; i++) f[i] = frnd(); }
    if (type == T_F16) { f16_t *h = (f16_t *)row; for (int i = 0; i < n; i++) h[i] = f32_to_f16(frnd()); }
    if (type == T_BF16) {
        uint16_t *h = (uint16_t *)row;
        for (int i = 0; i < n; i++) {
            union { float f; uint32_t u; } v = { .f = frnd() };
            h[i] = (uint16_t)(v.u >> 16);
        }
    }
}

// reference weights: independent decode where a SIMD dequant exists, the
// (scalar on the platform under test, or exact) dequant_row elsewhere
static void ref_weights(int type, const uint8_t *row, double *w, int n) {
    int bs = ggml_block_size(type);
    size_t ts = ggml_type_size(type);
    switch (type) {
        case T_Q8_0:
            for (int i = 0; i < n; i += bs) ref_dq_q8_0((const block_q8_0 *)(row + (i / bs) * ts), w + i);
            return;
        case T_Q4_K:
            for (int i = 0; i < n; i += bs) ref_dq_q4_K((const block_q4_K *)(row + (i / bs) * ts), w + i);
            return;
        case T_Q6_K:
            for (int i = 0; i < n; i += bs) ref_dq_q6_K((const block_q6_K *)(row + (i / bs) * ts), w + i);
            return;
        case T_MXFP4:
            for (int i = 0; i < n; i += bs) ref_dq_mxfp4((const block_mxfp4 *)(row + (i / bs) * ts), w + i);
            return;
        case T_IQ2_XXS:
            for (int i = 0; i < n; i += bs) ref_dq_iq2_xxs((const block_iq2_xxs *)(row + (i / bs) * ts), w + i);
            return;
        case T_IQ2_XS:
            for (int i = 0; i < n; i += bs) ref_dq_iq2_xs((const block_iq2_xs *)(row + (i / bs) * ts), w + i);
            return;
        case T_IQ2_S:
            for (int i = 0; i < n; i += bs) ref_dq_iq2_s((const block_iq2_s *)(row + (i / bs) * ts), w + i);
            return;
        case T_IQ3_XXS:
            for (int i = 0; i < n; i += bs) ref_dq_iq3_xxs((const block_iq3_xxs *)(row + (i / bs) * ts), w + i);
            return;
        case T_IQ3_S:
            for (int i = 0; i < n; i += bs) ref_dq_iq3_s((const block_iq3_s *)(row + (i / bs) * ts), w + i);
            return;
        case T_IQ1_S:
            for (int i = 0; i < n; i += bs) ref_dq_iq1_s((const block_iq1_s *)(row + (i / bs) * ts), w + i);
            return;
        case T_IQ1_M:
            for (int i = 0; i < n; i += bs) ref_dq_iq1_m((const block_iq1_m *)(row + (i / bs) * ts), w + i);
            return;
        default: {
            float *tmp = malloc((size_t)n * sizeof(float));
            dequant_row(type, row, tmp, n);
            for (int i = 0; i < n; i++) w[i] = tmp[i];
            free(tmp);
        }
    }
}

// ------------------------------------------------------------------- tests

static void test_vec_dot(int type, int n) {
    size_t rowsz = ggml_row_size(type, n);
    uint8_t *row = malloc(rowsz);
    float *x = malloc((size_t)n * sizeof(float));
    double *w = malloc((size_t)n * sizeof(double));
    for (int trial = 0; trial < 8; trial++) {
        make_row(type, row, n);
        for (int i = 0; i < n; i++) x[i] = frnd();
        ref_weights(type, row, w, n);
        double ref = 0, mag = 0;
        for (int i = 0; i < n; i++) { ref += w[i] * x[i]; mag += fabs(w[i] * x[i]); }
        float got = vec_dot(type, row, x, n);
        double tol = 5e-5 * mag + 1e-4;
        CHECK(fabs(got - ref) <= tol, "vec_dot %s n=%d trial=%d: got %g ref %g (tol %g)",
              ggml_type_name(type), n, trial, (double)got, ref, tol);
    }
    free(row); free(x); free(w);
}

static void test_dequant(int type, int n) {
    size_t rowsz = ggml_row_size(type, n);
    uint8_t *row = malloc(rowsz);
    float *got = malloc((size_t)n * sizeof(float));
    double *ref = malloc((size_t)n * sizeof(double));
    for (int trial = 0; trial < 8; trial++) {
        make_row(type, row, n);
        ref_weights(type, row, ref, n);
        dequant_row(type, row, got, n);
        for (int i = 0; i < n; i++) {
            double tol = 1e-5 * fabs(ref[i]) + 1e-6;
            CHECK(fabs(got[i] - ref[i]) <= tol,
                  "dequant %s trial=%d i=%d: got %g ref %g",
                  ggml_type_name(type), trial, i, (double)got[i], ref[i]);
            if (g_fail > 20) return;
        }
    }
    free(row); free(got); free(ref);
}

// scalar q8_quant_row, verbatim semantics — the SIMD path must match exactly
static void ref_q8_quant_row(const float *x, block_q8_0 *b, int n) {
    for (int i = 0; i < n / QK; i++, b++, x += QK) {
        float amax = 0;
        for (int j = 0; j < QK; j++) {
            float a = fabsf(x[j]);
            if (a > amax) amax = a;
        }
        float d = amax / 127.0f;
        float id = d > 0 ? 1.0f / d : 0.0f;
        b->d = f32_to_f16(d);
        for (int j = 0; j < QK; j++) b->qs[j] = (int8_t)roundf(x[j] * id);
    }
}

static void test_q8_kv(int n) {
    float *x = malloc((size_t)n * sizeof(float));
    block_q8_0 *got = malloc(ggml_row_size(T_Q8_0, n));
    block_q8_0 *ref = malloc(ggml_row_size(T_Q8_0, n));
    for (int trial = 0; trial < 16; trial++) {
        for (int i = 0; i < n; i++) x[i] = frnd() * (trial + 1);
        if (trial == 3) memset(x, 0, 32 * sizeof(float)); // all-zero block
        // exact half-integer products to exercise tie rounding
        if (trial == 4) for (int i = 0; i < n; i++) x[i] = (float)((int)(rnd32() % 255) - 127) * 0.5f;
        q8_quant_row(x, got, n);
        ref_q8_quant_row(x, ref, n);
        CHECK(memcmp(got, ref, ggml_row_size(T_Q8_0, n)) == 0,
              "q8_quant_row trial=%d: SIMD and scalar rows differ", trial);

        float acc_got[512], acc_ref[512];
        for (int i = 0; i < n; i++) acc_got[i] = acc_ref[i] = frnd();
        float a = frnd();
        q8_accum_row(got, a, acc_got, n);
        const block_q8_0 *b = ref;
        for (int i = 0; i < n / QK; i++, b++)
            for (int j = 0; j < QK; j++)
                acc_ref[i * QK + j] += a * f16_to_f32(b->d) * b->qs[j];
        for (int i = 0; i < n; i++)
            CHECK(fabsf(acc_got[i] - acc_ref[i]) <= 1e-5f * fabsf(acc_ref[i]) + 1e-6f,
                  "q8_accum_row trial=%d i=%d: got %g ref %g",
                  trial, i, (double)acc_got[i], (double)acc_ref[i]);
    }
    free(x); free(got); free(ref);
}

// ------------------------------------------------------- fused int8 dot route
//
// vec_dot_i8 quantizes the ACTIVATIONS to int8, so unlike vec_dot it is not
// held to "same value as an f64 dot over exact weights within fp32 noise". Its
// contract is (a) the int8 arithmetic itself is exact — the only error is the
// activation rounding — and (b) that error stays inside the bound an 8-bit
// per-32-block activation quantizer can produce. Both are checked here:
//   * i8_quant_act must match a scalar reference block-for-block, exactly;
//   * the dot must match a reference that quantizes the activations the SAME
//     way and then sums in double — that comparison has NO quantization error
//     left in it, so it is held to fp32 rounding only. A wrong nibble order, a
//     dropped min term or a bad scale fails it by orders of magnitude;
//   * against the true f32 dot, the error must stay under the analytic
//     activation-quantization bound (per block: |x|max/254 * sum|w|).
typedef struct { float d; int32_t s; int8_t qs[QK]; } ref_block_i8a;

static void ref_i8_quant_act(const float *x, ref_block_i8a *b, int n) {
    for (int i = 0; i < n / QK; i++, b++, x += QK) {
        float amax = 0;
        for (int j = 0; j < QK; j++) {
            float a = fabsf(x[j]);
            if (a > amax) amax = a;
        }
        float d = amax / 127.0f;
        float id = d > 0 ? 1.0f / d : 0.0f;
        b->d = d;
        int32_t s = 0;
        for (int j = 0; j < QK; j++) {
            // round half away from zero with a single rounding — see the
            // i8_quant_act contract in quants.c
            b->qs[j] = (int8_t)fmaf(x[j], id, copysignf(0.5f, x[j]));
            s += b->qs[j];
        }
        b->s = s;
    }
}

// The vectorized quantizer against the scalar definition, at volume and on
// exact ties: a .5 product is where a fused multiply-add and a two-step
// round would part company, so those are constructed rather than hoped for.
static void test_i8_quant_act(void) {
    enum { N = 4096 };
    float *x = malloc(N * sizeof(float));
    void *got = malloc((size_t)(N / QK) * sizeof(ref_block_i8a));
    ref_block_i8a *ref = malloc((size_t)(N / QK) * sizeof(ref_block_i8a));
    for (int trial = 0; trial < 200; trial++) {
        for (int i = 0; i < N; i++) x[i] = frnd() * (float)(1 << (trial % 20));
        if (trial % 4 == 1) {
            // exact half-integer products: amax fixes d to a power of two, so
            // (k + 0.5) * d is representable and x*id lands exactly on k+0.5
            float u = ldexpf(1.0f, (trial % 9) - 4);
            for (int b = 0; b < N / QK; b++) {
                x[b * QK] = 127.0f * u;
                for (int j = 1; j < QK; j++)
                    x[b * QK + j] = ((float)(int)(rnd32() % 253 - 126) + 0.5f) * u;
            }
        }
        if (trial % 4 == 2) memset(x, 0, N * sizeof(float));
        i8_quant_act(x, got, N);
        ref_i8_quant_act(x, ref, N);
        CHECK(memcmp(got, ref, (size_t)(N / QK) * sizeof(ref_block_i8a)) == 0,
              "i8_quant_act trial=%d: differs from the scalar definition", trial);
        if (g_fail > 20) break;
    }
    free(x); free(got); free(ref);
}

static int g_i8_checked = 0;   // guards against a vacuous pass (see main)

static void test_i8_dot(int type, int n) {
    if (!i8_dot_ok(type, n)) return;
    g_i8_checked++;
    size_t rowsz = ggml_row_size(type, n);
    uint8_t *row = malloc(rowsz);
    float *x = malloc((size_t)n * sizeof(float));
    double *w = malloc((size_t)n * sizeof(double));
    void *xq = malloc(i8_act_size(n));
    ref_block_i8a *xr = malloc((size_t)(n / QK) * sizeof(ref_block_i8a));
    CHECK(i8_act_size(n) == (size_t)(n / QK) * sizeof(ref_block_i8a),
          "i8_act_size(%d) = %zu, reference layout is %zu",
          n, i8_act_size(n), (size_t)(n / QK) * sizeof(ref_block_i8a));
    for (int trial = 0; trial < 8; trial++) {
        make_row(type, row, n);
        for (int i = 0; i < n; i++) x[i] = frnd();
        // trial 3: a zero activation block (d == 0 divides by nothing)
        if (trial == 3) memset(x, 0, QK * sizeof(float));
        // trial 4: one block far larger than the rest — per-block scaling is
        // the whole point, a row-wide scale would lose the small blocks
        if (trial == 4) for (int i = 0; i < QK; i++) x[i] = frnd() * 4096.0f;
        ref_weights(type, row, w, n);

        i8_quant_act(x, xq, n);
        ref_i8_quant_act(x, xr, n);
        CHECK(memcmp(xq, xr, i8_act_size(n)) == 0,
              "i8_quant_act %s trial=%d: differs from the scalar reference",
              ggml_type_name(type), trial);

        // (b) same-quantization reference: only fp32 rounding may differ
        double qref = 0, qmag = 0;
        for (int i = 0; i < n; i++) {
            double xv = (double)xr[i / QK].d * xr[i / QK].qs[i % QK];
            qref += w[i] * xv;
            qmag += fabs(w[i] * xv);
        }
        float got = vec_dot_i8(type, row, xq, n);
        double qtol = 5e-5 * qmag + 1e-4;
        CHECK(fabs(got - qref) <= qtol,
              "vec_dot_i8 %s n=%d trial=%d: got %g vs same-quant ref %g (tol %g)",
              ggml_type_name(type), n, trial, (double)got, qref, qtol);

        // (c) against the true dot, within the activation-quantization bound
        double ref = 0, bound = 0;
        for (int b = 0; b < n / QK; b++) {
            double amax = 0, sw = 0;
            for (int j = 0; j < QK; j++) {
                double xv = x[b * QK + j];
                ref += w[b * QK + j] * xv;
                if (fabs(xv) > amax) amax = fabs(xv);
                sw += fabs(w[b * QK + j]);
            }
            bound += amax / 254.0 * sw;   // half a quantization step per element
        }
        CHECK(fabs(got - ref) <= bound + 1e-4,
              "vec_dot_i8 %s n=%d trial=%d: got %g true ref %g exceeds the "
              "activation-quantization bound %g",
              ggml_type_name(type), n, trial, (double)got, ref, bound);
        if (g_fail > 20) break;
    }
    free(row); free(x); free(w); free(xq); free(xr);
}

static void test_multi(void) {
    // nb=7 exercises the 4-column block and the remainder; odd stride and
    // n not a multiple of the vector width exercise the tails
    int n = 133, nb = 7, stride = n + 3;
    float *w = malloc((size_t)n * sizeof(float));
    float *x = malloc((size_t)nb * stride * sizeof(float));
    float out[7];
    for (int trial = 0; trial < 8; trial++) {
        for (int i = 0; i < n; i++) w[i] = frnd();
        for (int i = 0; i < nb * stride; i++) x[i] = frnd();
        vec_dot_f32_multi(w, x, stride, nb, n, out);
        for (int b = 0; b < nb; b++) {
            double ref = 0, mag = 0;
            for (int i = 0; i < n; i++) {
                ref += (double)w[i] * x[b * stride + i];
                mag += fabs((double)w[i] * x[b * stride + i]);
            }
            CHECK(fabs(out[b] - ref) <= 5e-5 * mag + 1e-4,
                  "vec_dot_f32_multi trial=%d col=%d: got %g ref %g",
                  trial, b, (double)out[b], ref);
        }
    }
    free(w); free(x);
}

int main(void) {
    f16_init();

    // layout drift between this file's copies and quants.c would invalidate
    // every reference below
    CHECK(ggml_type_size(T_Q4_0) == sizeof(block_q4_0), "q4_0 size");
    CHECK(ggml_type_size(T_Q4_1) == sizeof(block_q4_1), "q4_1 size");
    CHECK(ggml_type_size(T_Q5_0) == sizeof(block_q5_0), "q5_0 size");
    CHECK(ggml_type_size(T_Q5_1) == sizeof(block_q5_1), "q5_1 size");
    CHECK(ggml_type_size(T_Q8_0) == sizeof(block_q8_0), "q8_0 size");
    CHECK(ggml_type_size(T_Q2_K) == sizeof(block_q2_K), "q2_K size");
    CHECK(ggml_type_size(T_Q3_K) == sizeof(block_q3_K), "q3_K size");
    CHECK(ggml_type_size(T_Q4_K) == sizeof(block_q4_K), "q4_K size");
    CHECK(ggml_type_size(T_Q5_K) == sizeof(block_q5_K), "q5_K size");
    CHECK(ggml_type_size(T_Q6_K) == sizeof(block_q6_K), "q6_K size");
    CHECK(ggml_type_size(T_IQ4_NL) == sizeof(block_iq4_nl), "iq4_nl size");
    CHECK(ggml_type_size(T_IQ4_XS) == sizeof(block_iq4_xs), "iq4_xs size");
    CHECK(ggml_type_size(T_MXFP4) == sizeof(block_mxfp4), "mxfp4 size");
    CHECK(ggml_type_size(T_IQ2_XXS) == sizeof(block_iq2_xxs), "iq2_xxs size");
    CHECK(ggml_type_size(T_IQ2_XS) == sizeof(block_iq2_xs), "iq2_xs size");
    CHECK(ggml_type_size(T_IQ2_S) == sizeof(block_iq2_s), "iq2_s size");
    CHECK(ggml_type_size(T_IQ3_XXS) == sizeof(block_iq3_xxs), "iq3_xxs size");
    CHECK(ggml_type_size(T_IQ3_S) == sizeof(block_iq3_s), "iq3_s size");
    CHECK(ggml_type_size(T_IQ1_S) == sizeof(block_iq1_s), "iq1_s size");
    CHECK(ggml_type_size(T_IQ1_M) == sizeof(block_iq1_m), "iq1_m size");

    static const int types[] = {
        T_F32, T_F16, T_BF16, T_Q4_0, T_Q4_1, T_Q5_0, T_Q5_1, T_Q8_0,
        T_Q2_K, T_Q3_K, T_Q4_K, T_Q5_K, T_Q6_K, T_IQ4_NL, T_IQ4_XS, T_MXFP4,
        T_IQ2_XXS, T_IQ2_XS, T_IQ2_S, T_IQ3_XXS, T_IQ3_S, T_IQ1_S, T_IQ1_M,
    };
    for (size_t t = 0; t < sizeof(types) / sizeof(types[0]); t++) {
        test_vec_dot(types[t], 4096);
        test_vec_dot(types[t], 256);   // single K-block / few 32-blocks
    }
    test_dequant(T_Q8_0, 4096);
    test_dequant(T_Q4_K, 4096);
    test_dequant(T_Q6_K, 4096);
    test_dequant(T_MXFP4, 4096);
    test_dequant(T_IQ2_XXS, 4096);
    test_dequant(T_IQ2_XS, 4096);
    test_dequant(T_IQ2_S, 4096);
    test_dequant(T_IQ3_XXS, 4096);
    test_dequant(T_IQ3_S, 4096);
    test_dequant(T_IQ1_S, 4096);
    test_dequant(T_IQ1_M, 4096);
    test_q8_kv(512);
    test_multi();

    test_i8_quant_act();
    // The fused int8 route: promoted formats at a full row and at one block,
    // plus the formats it must decline (i8_dot_ok false => test_i8_dot returns)
    for (size_t t = 0; t < sizeof(types) / sizeof(types[0]); t++) {
        test_i8_dot(types[t], 4096);
        test_i8_dot(types[t], 256);
    }
    CHECK(!i8_dot_ok(T_Q4_K, 128), "i8_dot_ok must decline a partial K-block");
    CHECK(!i8_dot_ok(T_IQ2_XXS, 4096), "i8_dot_ok must decline ungated formats");
    // On a build with the fused route compiled in, all three promoted formats
    // must have been exercised at both row lengths. Reporting "OK" because
    // every combo declined itself is the failure mode this catches.
#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
    CHECK(g_i8_checked == 6, "fused int8 route: %d combos exercised, expected 6",
          g_i8_checked);
#endif

    if (g_fail) { fprintf(stderr, "test_quants_simd: %d FAILURES\n", g_fail); return 1; }
    printf("test_quants_simd: OK\n");
    return 0;
}
