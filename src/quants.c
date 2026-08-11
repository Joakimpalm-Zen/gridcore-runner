// Quantization block formats (ggml-compatible), dot kernels, threadpool.
#include "quants.h"
#include "fp16.h"
#include "tpool.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

// ---------------------------------------------------------------- fp16 table

float g_f16_table[65536];

static float f16_decode(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign;
        } else { // subnormal
            uint32_t e = 113;
            while (!(mant & 0x400)) { mant <<= 1; e--; }
            mant &= 0x3FF;
            f = sign | (e << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7F800000u | (mant << 13);
    } else {
        f = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    union { uint32_t u; float fl; } v = { f };
    return v.fl;
}

void f16_init(void) {
    for (uint32_t i = 0; i < 65536; i++) g_f16_table[i] = f16_decode((uint16_t)i);
}

// ---------------------------------------------------------------- block defs

#define QK 32
#define QK_K 256

typedef struct { f16_t d; uint8_t qs[QK / 2]; }                  block_q4_0; // 18
typedef struct { f16_t d, m; uint8_t qs[QK / 2]; }               block_q4_1; // 20
typedef struct { f16_t d; uint8_t qh[4]; uint8_t qs[QK / 2]; }   block_q5_0; // 22
typedef struct { f16_t d, m; uint8_t qh[4]; uint8_t qs[QK / 2]; } block_q5_1; // 24
typedef struct { f16_t d; int8_t qs[QK]; }                       block_q8_0; // 34
typedef struct { uint8_t scales[QK_K / 16]; uint8_t qs[QK_K / 4]; f16_t d, dmin; } block_q2_K; // 84
typedef struct { uint8_t hmask[QK_K / 8]; uint8_t qs[QK_K / 4]; uint8_t scales[12]; f16_t d; } block_q3_K; // 110
typedef struct { f16_t d, dmin; uint8_t scales[12]; uint8_t qs[QK_K / 2]; } block_q4_K; // 144
typedef struct { f16_t d, dmin; uint8_t scales[12]; uint8_t qh[QK_K / 8]; uint8_t qs[QK_K / 2]; } block_q5_K; // 176
typedef struct { uint8_t ql[QK_K / 2]; uint8_t qh[QK_K / 4]; int8_t scales[QK_K / 16]; f16_t d; } block_q6_K; // 210
typedef struct { f16_t d; uint8_t qs[QK / 2]; }                  block_iq4_nl; // 18
typedef struct { f16_t d; uint16_t scales_h; uint8_t scales_l[QK_K / 64]; uint8_t qs[QK_K / 2]; } block_iq4_xs; // 136
// OCP microscaling FP4: one E8M0 (power-of-two) block scale byte + 32 packed
// E2M1 4-bit codes (the gpt-oss expert-tensor format).
typedef struct { uint8_t e; uint8_t qs[QK / 2]; }               block_mxfp4; // 17
// codebook i-quants (llama.cpp b10353 ggml-common.h shapes, byte-identical)
typedef struct { f16_t d; uint16_t qs[QK_K / 8]; } block_iq2_xxs; // 66
typedef struct { f16_t d; uint16_t qs[QK_K / 8]; uint8_t scales[QK_K / 32]; } block_iq2_xs; // 74
typedef struct { f16_t d; uint8_t qs[QK_K / 4]; uint8_t qh[QK_K / 32]; uint8_t scales[QK_K / 32]; } block_iq2_s; // 82
typedef struct { f16_t d; uint8_t qs[3 * QK_K / 8]; } block_iq3_xxs; // 98
typedef struct { f16_t d; uint8_t qs[QK_K / 4]; uint8_t qh[QK_K / 32]; uint8_t signs[QK_K / 8]; uint8_t scales[QK_K / 64]; } block_iq3_s; // 110
typedef struct { f16_t d; uint8_t qs[QK_K / 8]; uint16_t qh[QK_K / 32]; } block_iq1_s; // 50
typedef struct { uint8_t qs[QK_K / 8]; uint8_t qh[QK_K / 16]; uint8_t scales[QK_K / 32]; } block_iq1_m; // 56
#define IQ1S_DELTA 0.125f
#include "quants_iq_grids.h"


static const int8_t kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
};

// E2M1 magnitudes {0,.5,1,1.5,2,3,4,6} with sign in the high nibble half;
// numerically identical to ggml's integer table halved (matches gpt-oss GGUFs).
static const float kvalues_mxfp4[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
     0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

// the same E2M1 table doubled to integers so SIMD nibble lookups can run in
// int8 (pshufb / tbl); the 0.5 factor is folded into the block scale
static const int8_t kvalues_mxfp4_i8[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,
};

int ggml_block_size(int type) {
    switch (type) {
        case T_F32: case T_F16: case T_BF16: return 1;
        case T_Q4_0: case T_Q4_1: case T_Q5_0: case T_Q5_1: case T_Q8_0:
        case T_IQ4_NL: case T_MXFP4: return QK;
        case T_Q4_K: case T_Q5_K: case T_Q6_K: case T_Q2_K: case T_Q3_K:
        case T_IQ4_XS:
        case T_IQ2_XXS: case T_IQ2_XS: case T_IQ2_S:
        case T_IQ3_XXS: case T_IQ3_S:
        case T_IQ1_S: case T_IQ1_M: return QK_K;
        default: return 1;
    }
}

size_t ggml_type_size(int type) {
    switch (type) {
        case T_F32:  return 4;
        case T_F16:  return 2;
        case T_BF16: return 2;
        case T_Q4_0: return sizeof(block_q4_0);
        case T_Q4_1: return sizeof(block_q4_1);
        case T_Q5_0: return sizeof(block_q5_0);
        case T_Q5_1: return sizeof(block_q5_1);
        case T_Q8_0: return sizeof(block_q8_0);
        case T_Q4_K: return sizeof(block_q4_K);
        case T_Q5_K: return sizeof(block_q5_K);
        case T_Q6_K: return sizeof(block_q6_K);
        case T_Q2_K: return sizeof(block_q2_K);
        case T_Q3_K: return sizeof(block_q3_K);
        case T_IQ4_NL: return sizeof(block_iq4_nl);
        case T_IQ4_XS: return sizeof(block_iq4_xs);
        case T_IQ2_XXS: return sizeof(block_iq2_xxs);
        case T_IQ2_XS: return sizeof(block_iq2_xs);
        case T_IQ2_S: return sizeof(block_iq2_s);
        case T_IQ3_XXS: return sizeof(block_iq3_xxs);
        case T_IQ3_S: return sizeof(block_iq3_s);
        case T_IQ1_S: return sizeof(block_iq1_s);
        case T_IQ1_M: return sizeof(block_iq1_m);
        case T_MXFP4: return sizeof(block_mxfp4);
        default:     return 1;
    }
}

const char *ggml_type_name(int type) {
    switch (type) {
        case T_F32: return "F32";   case T_F16: return "F16";  case T_BF16: return "BF16";
        case T_Q4_0: return "Q4_0"; case T_Q4_1: return "Q4_1";
        case T_Q5_0: return "Q5_0"; case T_Q5_1: return "Q5_1";
        case T_Q8_0: return "Q8_0";
        case T_Q4_K: return "Q4_K"; case T_Q5_K: return "Q5_K"; case T_Q6_K: return "Q6_K";
        case T_Q2_K: return "Q2_K"; case T_Q3_K: return "Q3_K";
        case T_IQ4_NL: return "IQ4_NL"; case T_IQ4_XS: return "IQ4_XS";
        case T_IQ2_XXS: return "IQ2_XXS"; case T_IQ2_XS: return "IQ2_XS";
        case T_IQ2_S: return "IQ2_S";
        case T_IQ3_XXS: return "IQ3_XXS"; case T_IQ3_S: return "IQ3_S";
        case T_IQ1_S: return "IQ1_S"; case T_IQ1_M: return "IQ1_M";
        case T_MXFP4: return "MXFP4";
        default: return "?";
    }
}

bool ggml_type_supported(int type) {
    switch (type) {
        case T_F32: case T_F16: case T_BF16:
        case T_Q4_0: case T_Q4_1: case T_Q5_0: case T_Q5_1: case T_Q8_0:
        case T_Q2_K: case T_Q3_K: case T_Q4_K: case T_Q5_K: case T_Q6_K:
        case T_IQ4_NL: case T_IQ4_XS: case T_MXFP4:
        case T_IQ2_XXS: case T_IQ2_XS: case T_IQ2_S:
        case T_IQ3_XXS: case T_IQ3_S: case T_IQ1_S: case T_IQ1_M:
            return true;
        default:
            return false;
    }
}

// ---------------------------------------------------------------- dequant

static void dq_mxfp4(const block_mxfp4 *b, float *y) {
    // E8M0 block scale: the byte is a biased power-of-two exponent, 2^(e-127).
    float d = ldexpf(1.0f, (int)b->e - 127);
    for (int j = 0; j < 16; j++) {
        y[j]      = kvalues_mxfp4[b->qs[j] & 0xF] * d;
        y[j + 16] = kvalues_mxfp4[b->qs[j] >> 4]  * d;
    }
}

static void dq_q4_0(const block_q4_0 *b, float *y) {
    float d = f16_to_f32(b->d);
    for (int j = 0; j < 16; j++) {
        y[j]      = ((b->qs[j] & 0xF) - 8) * d;
        y[j + 16] = ((b->qs[j] >> 4)  - 8) * d;
    }
}

static void dq_q4_1(const block_q4_1 *b, float *y) {
    float d = f16_to_f32(b->d), m = f16_to_f32(b->m);
    for (int j = 0; j < 16; j++) {
        y[j]      = (b->qs[j] & 0xF) * d + m;
        y[j + 16] = (b->qs[j] >> 4)  * d + m;
    }
}

static void dq_q5_0(const block_q5_0 *b, float *y) {
    float d = f16_to_f32(b->d);
    uint32_t qh; memcpy(&qh, b->qh, 4);
    for (int j = 0; j < 16; j++) {
        int x0 = (b->qs[j] & 0xF) | (((qh >> j) & 1) << 4);
        int x1 = (b->qs[j] >> 4)  | (((qh >> (j + 16)) & 1) << 4);
        y[j]      = (x0 - 16) * d;
        y[j + 16] = (x1 - 16) * d;
    }
}

static void dq_q5_1(const block_q5_1 *b, float *y) {
    float d = f16_to_f32(b->d), m = f16_to_f32(b->m);
    uint32_t qh; memcpy(&qh, b->qh, 4);
    for (int j = 0; j < 16; j++) {
        int x0 = (b->qs[j] & 0xF) | (((qh >> j) & 1) << 4);
        int x1 = (b->qs[j] >> 4)  | (((qh >> (j + 16)) & 1) << 4);
        y[j]      = x0 * d + m;
        y[j + 16] = x1 * d + m;
    }
}

static void dq_q8_0(const block_q8_0 *b, float *y) {
    float d = f16_to_f32(b->d);
    for (int j = 0; j < QK; j++) y[j] = b->qs[j] * d;
}

static void dq_q2_K(const block_q2_K *b, float *y) {
    float d = f16_to_f32(b->d), dmin = f16_to_f32(b->dmin);
    const uint8_t *q = b->qs;
    int is = 0;
    for (int n = 0; n < QK_K; n += 128) {
        int shift = 0;
        for (int j = 0; j < 4; j++) {
            uint8_t sc = b->scales[is++];
            float dl = d * (sc & 0xF), ml = dmin * (sc >> 4);
            for (int l = 0; l < 16; l++) *y++ = dl * ((q[l] >> shift) & 3) - ml;
            sc = b->scales[is++];
            dl = d * (sc & 0xF); ml = dmin * (sc >> 4);
            for (int l = 0; l < 16; l++) *y++ = dl * ((q[l + 16] >> shift) & 3) - ml;
            shift += 2;
        }
        q += 32;
    }
}

static void dq_q3_K(const block_q3_K *b, float *y) {
    const uint32_t kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f;
    uint32_t aux[4];
    const int8_t *scales = (const int8_t *)aux;
    memcpy(aux, b->scales, 12);
    uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

    float d_all = f16_to_f32(b->d);
    const uint8_t *q = b->qs, *hm = b->hmask;
    uint8_t m = 1;
    int is = 0;
    for (int n = 0; n < QK_K; n += 128) {
        int shift = 0;
        for (int j = 0; j < 4; j++) {
            float dl = d_all * (scales[is++] - 32);
            for (int l = 0; l < 16; l++)
                *y++ = dl * (((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
            dl = d_all * (scales[is++] - 32);
            for (int l = 0; l < 16; l++)
                *y++ = dl * (((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
            shift += 2; m <<= 1;
        }
        q += 32;
    }
}

static inline void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4)  | ((q[j    ] >> 6) << 4);
    }
}

static void dq_q4_K(const block_q4_K *b, float *y) {
    float d = f16_to_f32(b->d), dmin = f16_to_f32(b->dmin);
    const uint8_t *q = b->qs;
    int is = 0;
    for (int j = 0; j < QK_K; j += 64) {
        uint8_t sc, mn;
        get_scale_min_k4(is + 0, b->scales, &sc, &mn);
        float d1 = d * sc, m1 = dmin * mn;
        get_scale_min_k4(is + 1, b->scales, &sc, &mn);
        float d2 = d * sc, m2 = dmin * mn;
        for (int l = 0; l < 32; l++) y[l]      = d1 * (q[l] & 0xF) - m1;
        for (int l = 0; l < 32; l++) y[l + 32] = d2 * (q[l] >> 4)  - m2;
        q += 32; is += 2; y += 64;
    }
}

static void dq_q5_K(const block_q5_K *b, float *y) {
    float d = f16_to_f32(b->d), dmin = f16_to_f32(b->dmin);
    const uint8_t *q = b->qs, *qh = b->qh;
    int is = 0;
    uint8_t u1 = 1, u2 = 2;
    for (int j = 0; j < QK_K; j += 64) {
        uint8_t sc, mn;
        get_scale_min_k4(is + 0, b->scales, &sc, &mn);
        float d1 = d * sc, m1 = dmin * mn;
        get_scale_min_k4(is + 1, b->scales, &sc, &mn);
        float d2 = d * sc, m2 = dmin * mn;
        for (int l = 0; l < 32; l++) y[l]      = d1 * ((q[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - m1;
        for (int l = 0; l < 32; l++) y[l + 32] = d2 * ((q[l] >> 4)  + ((qh[l] & u2) ? 16 : 0)) - m2;
        q += 32; is += 2; y += 64; u1 <<= 2; u2 <<= 2;
    }
}

static void dq_q6_K(const block_q6_K *b, float *y) {
    float d = f16_to_f32(b->d);
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

static void dq_iq4_nl(const block_iq4_nl *b, float *y) {
    float d = f16_to_f32(b->d);
    for (int j = 0; j < 16; j++) {
        y[j]      = d * kvalues_iq4nl[b->qs[j] & 0xF];
        y[j + 16] = d * kvalues_iq4nl[b->qs[j] >> 4];
    }
}

static void dq_iq4_xs(const block_iq4_xs *b, float *y) {
    float d = f16_to_f32(b->d);
    const uint8_t *qs = b->qs;
    for (int ib = 0; ib < QK_K / 32; ib++) {
        int ls = ((b->scales_l[ib / 2] >> 4 * (ib % 2)) & 0xF) |
                 (((b->scales_h >> 2 * ib) & 3) << 4);
        float dl = d * (ls - 32);
        for (int j = 0; j < 16; j++) {
            y[j]      = dl * kvalues_iq4nl[qs[j] & 0xF];
            y[j + 16] = dl * kvalues_iq4nl[qs[j] >> 4];
        }
        qs += 16; y += 32;
    }
}

// ------------------------------------------------- codebook i-quant dequant
// Transcribed per-block from llama.cpp b10353 ggml-quants.c
// dequantize_row_iq* (the outer nb loop lives in dequant_row here); the
// codebook grids and sign tables are in quants_iq_grids.h, verbatim.

static void dq_iq2_xxs(const block_iq2_xxs *b, float *y) {
    float d = f16_to_f32(b->d);
    uint32_t aux32[2];
    const uint8_t *aux8 = (const uint8_t *)aux32;
    for (int ib32 = 0; ib32 < QK_K / 32; ib32++) {
        memcpy(aux32, b->qs + 4 * ib32, 2 * sizeof(uint32_t));
        float db = d * (0.5f + (aux32[1] >> 28)) * 0.25f;
        for (int l = 0; l < 4; l++) {
            const uint8_t *grid = (const uint8_t *)(iq2xxs_grid + aux8[l]);
            uint8_t signs = ksigns_iq2xs[(aux32[1] >> 7 * l) & 127];
            for (int j = 0; j < 8; j++)
                y[j] = db * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
            y += 8;
        }
    }
}

static void dq_iq2_xs(const block_iq2_xs *b, float *y) {
    float d = f16_to_f32(b->d);
    float db[2];
    for (int ib32 = 0; ib32 < QK_K / 32; ib32++) {
        db[0] = d * (0.5f + (b->scales[ib32] & 0xf)) * 0.25f;
        db[1] = d * (0.5f + (b->scales[ib32] >> 4)) * 0.25f;
        for (int l = 0; l < 4; l++) {
            const uint8_t *grid =
                (const uint8_t *)(iq2xs_grid + (b->qs[4 * ib32 + l] & 511));
            uint8_t signs = ksigns_iq2xs[b->qs[4 * ib32 + l] >> 9];
            for (int j = 0; j < 8; j++)
                y[j] = db[l / 2] * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
            y += 8;
        }
    }
}

static void dq_iq2_s(const block_iq2_s *b, float *y) {
    float d = f16_to_f32(b->d);
    const uint8_t *qs = b->qs, *qh = b->qh, *signs = b->qs + QK_K / 8;
    float db[2];
    for (int ib32 = 0; ib32 < QK_K / 32; ib32++) {
        db[0] = d * (0.5f + (b->scales[ib32] & 0xf)) * 0.25f;
        db[1] = d * (0.5f + (b->scales[ib32] >> 4)) * 0.25f;
        for (int l = 0; l < 4; l++) {
            float dl = db[l / 2];
            const uint8_t *grid = (const uint8_t *)
                (iq2s_grid + (qs[l] | ((qh[ib32] << (8 - 2 * l)) & 0x300)));
            for (int j = 0; j < 8; j++)
                y[j] = dl * grid[j] * (signs[l] & kmask_iq2xs[j] ? -1.f : 1.f);
            y += 8;
        }
        qs += 4;
        signs += 4;
    }
}

static void dq_iq3_xxs(const block_iq3_xxs *b, float *y) {
    float d = f16_to_f32(b->d);
    const uint8_t *qs = b->qs, *scales_and_signs = b->qs + QK_K / 4;
    uint32_t aux32;
    for (int ib32 = 0; ib32 < QK_K / 32; ib32++) {
        memcpy(&aux32, scales_and_signs + 4 * ib32, sizeof(uint32_t));
        float db = d * (0.5f + (aux32 >> 28)) * 0.5f;
        for (int l = 0; l < 4; l++) {
            uint8_t signs = ksigns_iq2xs[(aux32 >> 7 * l) & 127];
            const uint8_t *grid1 = (const uint8_t *)(iq3xxs_grid + qs[2 * l + 0]);
            const uint8_t *grid2 = (const uint8_t *)(iq3xxs_grid + qs[2 * l + 1]);
            for (int j = 0; j < 4; j++) {
                y[j + 0] = db * grid1[j] * (signs & kmask_iq2xs[j + 0] ? -1.f : 1.f);
                y[j + 4] = db * grid2[j] * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f);
            }
            y += 8;
        }
        qs += 8;
    }
}

static void dq_iq3_s(const block_iq3_s *b, float *y) {
    float d = f16_to_f32(b->d);
    const uint8_t *qs = b->qs, *qh = b->qh, *signs = b->signs;
    for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
        float db1 = d * (1 + 2 * (b->scales[ib32 / 2] & 0xf));
        float db2 = d * (1 + 2 * (b->scales[ib32 / 2] >> 4));
        for (int l = 0; l < 4; l++) {
            const uint8_t *grid1 = (const uint8_t *)
                (iq3s_grid + (qs[2 * l + 0] | ((qh[0] << (8 - 2 * l)) & 256)));
            const uint8_t *grid2 = (const uint8_t *)
                (iq3s_grid + (qs[2 * l + 1] | ((qh[0] << (7 - 2 * l)) & 256)));
            for (int j = 0; j < 4; j++) {
                y[j + 0] = db1 * grid1[j] * (signs[l] & kmask_iq2xs[j + 0] ? -1.f : 1.f);
                y[j + 4] = db1 * grid2[j] * (signs[l] & kmask_iq2xs[j + 4] ? -1.f : 1.f);
            }
            y += 8;
        }
        qs += 8;
        signs += 4;
        for (int l = 0; l < 4; l++) {
            const uint8_t *grid1 = (const uint8_t *)
                (iq3s_grid + (qs[2 * l + 0] | ((qh[1] << (8 - 2 * l)) & 256)));
            const uint8_t *grid2 = (const uint8_t *)
                (iq3s_grid + (qs[2 * l + 1] | ((qh[1] << (7 - 2 * l)) & 256)));
            for (int j = 0; j < 4; j++) {
                y[j + 0] = db2 * grid1[j] * (signs[l] & kmask_iq2xs[j + 0] ? -1.f : 1.f);
                y[j + 4] = db2 * grid2[j] * (signs[l] & kmask_iq2xs[j + 4] ? -1.f : 1.f);
            }
            y += 8;
        }
        qh += 2;
        qs += 8;
        signs += 4;
    }
}

static void dq_iq1_s(const block_iq1_s *b, float *y) {
    float d = f16_to_f32(b->d);
    const uint8_t *qs = b->qs;
    const uint16_t *qh = b->qh;
    for (int ib = 0; ib < QK_K / 32; ib++) {
        float dl = d * (2 * ((qh[ib] >> 12) & 7) + 1);
        float delta = qh[ib] & 0x8000 ? -IQ1S_DELTA : IQ1S_DELTA;
        for (int l = 0; l < 4; l++) {
            const int8_t *grid = (const int8_t *)
                (iq1s_grid + (qs[l] | (((qh[ib] >> 3 * l) & 7) << 8)));
            for (int j = 0; j < 8; j++) y[j] = dl * (grid[j] + delta);
            y += 8;
        }
        qs += 4;
    }
}

static void dq_iq1_m(const block_iq1_m *b, float *y) {
    // the block-level fp16 scale is scattered across the top nibbles of the
    // four 16-bit scale words (llama.cpp iq1m_scale_t)
    const uint16_t *sc = (const uint16_t *)b->scales;
    f16_t sd = (f16_t)((sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) |
                       ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000));
    float d = f16_to_f32(sd);
    const uint8_t *qs = b->qs, *qh = b->qh;
    float delta[4];
    uint16_t idx[4];
    for (int ib = 0; ib < QK_K / 32; ib++) {
        float dl1 = d * (2 * ((sc[ib / 2] >> (6 * (ib % 2) + 0)) & 0x7) + 1);
        float dl2 = d * (2 * ((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 0x7) + 1);
        idx[0] = qs[0] | ((qh[0] << 8) & 0x700);
        idx[1] = qs[1] | ((qh[0] << 4) & 0x700);
        idx[2] = qs[2] | ((qh[1] << 8) & 0x700);
        idx[3] = qs[3] | ((qh[1] << 4) & 0x700);
        delta[0] = qh[0] & 0x08 ? -IQ1S_DELTA : IQ1S_DELTA;
        delta[1] = qh[0] & 0x80 ? -IQ1S_DELTA : IQ1S_DELTA;
        delta[2] = qh[1] & 0x08 ? -IQ1S_DELTA : IQ1S_DELTA;
        delta[3] = qh[1] & 0x80 ? -IQ1S_DELTA : IQ1S_DELTA;
        for (int l = 0; l < 2; l++) {
            const int8_t *grid = (const int8_t *)(iq1s_grid + idx[l]);
            for (int j = 0; j < 8; j++) y[j] = dl1 * (grid[j] + delta[l]);
            y += 8;
        }
        for (int l = 2; l < 4; l++) {
            const int8_t *grid = (const int8_t *)(iq1s_grid + idx[l]);
            for (int j = 0; j < 8; j++) y[j] = dl2 * (grid[j] + delta[l]);
            y += 8;
        }
        qs += 4;
        qh += 2;
    }
}

#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
#include <immintrin.h>
// vectorized block dequant for the batched prompt path (dequant once, dot
// many); the same unpack tricks as the vec_dot kernels below, storing floats
static inline void st8i(float *y, __m128i v8lo) {
    _mm256_storeu_ps(y, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v8lo)));
}
static inline void st16i(float *y, __m128i v16) {
    st8i(y, v16);
    st8i(y + 8, _mm_srli_si128(v16, 8));
}
static inline void scale16(float *y, float d) {
    __m256 dv = _mm256_set1_ps(d);
    _mm256_storeu_ps(y,     _mm256_mul_ps(_mm256_loadu_ps(y),     dv));
    _mm256_storeu_ps(y + 8, _mm256_mul_ps(_mm256_loadu_ps(y + 8), dv));
}
static void dq_q8_0_avx2(const block_q8_0 *b, float *y) {
    float d = f16_to_f32(b->d);
    st16i(y,      _mm_loadu_si128((const __m128i *)b->qs));
    st16i(y + 16, _mm_loadu_si128((const __m128i *)(b->qs + 16)));
    scale16(y, d); scale16(y + 16, d);
}
static void dq_q4_K_avx2(const block_q4_K *b, float *y) {
    const __m128i mF = _mm_set1_epi8(0xF);
    float d = f16_to_f32(b->d), dmin = f16_to_f32(b->dmin);
    const uint8_t *q = b->qs;
    int is = 0;
    for (int j = 0; j < QK_K; j += 64) {
        uint8_t sc, mn;
        get_scale_min_k4(is + 0, b->scales, &sc, &mn);
        float d1 = d * sc, m1 = dmin * mn;
        get_scale_min_k4(is + 1, b->scales, &sc, &mn);
        float d2 = d * sc, m2 = dmin * mn;
        __m256 d1v = _mm256_set1_ps(d1), m1v = _mm256_set1_ps(m1);
        __m256 d2v = _mm256_set1_ps(d2), m2v = _mm256_set1_ps(m2);
        for (int l = 0; l < 32; l += 16) {
            __m128i qv = _mm_loadu_si128((const __m128i *)(q + l));
            __m128i lo = _mm_and_si128(qv, mF);
            __m128i hi = _mm_and_si128(_mm_srli_epi16(qv, 4), mF);
            st16i(y + l, lo);
            st16i(y + 32 + l, hi);
            for (int o = 0; o < 16; o += 8) {
                _mm256_storeu_ps(y + l + o, _mm256_fmsub_ps(
                    _mm256_loadu_ps(y + l + o), d1v, m1v));
                _mm256_storeu_ps(y + 32 + l + o, _mm256_fmsub_ps(
                    _mm256_loadu_ps(y + 32 + l + o), d2v, m2v));
            }
        }
        q += 32; is += 2; y += 64;
    }
}
static void dq_q6_K_avx2(const block_q6_K *b, float *y) {
    const __m128i mF = _mm_set1_epi8(0xF), m3 = _mm_set1_epi8(3),
                  m32 = _mm_set1_epi8(32);
    float d = f16_to_f32(b->d);
    const uint8_t *ql = b->ql, *qh = b->qh;
    const int8_t *sc = b->scales;
    for (int half = 0; half < 2; half++) {
        for (int base = 0; base < 32; base += 16) {
            int is = base / 16;
            __m128i l0 = _mm_loadu_si128((const __m128i *)(ql + base));
            __m128i l1 = _mm_loadu_si128((const __m128i *)(ql + base + 32));
            __m128i h  = _mm_loadu_si128((const __m128i *)(qh + base));
            __m128i q1 = _mm_sub_epi8(_mm_or_si128(_mm_and_si128(l0, mF),
                    _mm_slli_epi16(_mm_and_si128(h, m3), 4)), m32);
            __m128i q2 = _mm_sub_epi8(_mm_or_si128(_mm_and_si128(l1, mF),
                    _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(h, 2), m3), 4)), m32);
            __m128i q3 = _mm_sub_epi8(_mm_or_si128(
                    _mm_and_si128(_mm_srli_epi16(l0, 4), mF),
                    _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(h, 4), m3), 4)), m32);
            __m128i q4 = _mm_sub_epi8(_mm_or_si128(
                    _mm_and_si128(_mm_srli_epi16(l1, 4), mF),
                    _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(h, 6), m3), 4)), m32);
            st16i(y + base, q1);        scale16(y + base,      d * sc[is]);
            st16i(y + 32 + base, q2);   scale16(y + 32 + base, d * sc[is + 2]);
            st16i(y + 64 + base, q3);   scale16(y + 64 + base, d * sc[is + 4]);
            st16i(y + 96 + base, q4);   scale16(y + 96 + base, d * sc[is + 6]);
        }
        y += 128; ql += 64; qh += 32; sc += 8;
    }
}

// ---- codebook i-quants (same shapes as the NEON versions below: signed
// byte negate-select applies the sign table, then one widen+scale pass)
static const uint8_t avx2_iq_kmask[16] = { 1, 2, 4, 8, 16, 32, 64, 128,
                                           1, 2, 4, 8, 16, 32, 64, 128 };

static inline void avx2_iq_sign16(const uint8_t *mag, uint8_t s0, uint8_t s1,
                                  float *y, float db) {
    __m128i m = _mm_loadu_si128((const __m128i *)mag);
    __m128i sv = _mm_set_epi64x((long long)(0x0101010101010101ULL * s1),
                                (long long)(0x0101010101010101ULL * s0));
    __m128i km = _mm_loadu_si128((const __m128i *)avx2_iq_kmask);
    __m128i neg = _mm_cmpeq_epi8(_mm_and_si128(sv, km), km);
    __m128i v = _mm_blendv_epi8(m, _mm_sub_epi8(_mm_setzero_si128(), m), neg);
    st16i(y, v);
    scale16(y, db);
}

static inline void avx2_iq1_16(const int8_t *g16, float dl, float d0, float d1,
                               float *y) {
    __m128i v = _mm_loadu_si128((const __m128i *)g16);
    __m256 lo = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v));
    __m256 hi = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(v, 8)));
    lo = _mm256_mul_ps(_mm256_add_ps(lo, _mm256_set1_ps(d0)), _mm256_set1_ps(dl));
    hi = _mm256_mul_ps(_mm256_add_ps(hi, _mm256_set1_ps(d1)), _mm256_set1_ps(dl));
    _mm256_storeu_ps(y, lo);
    _mm256_storeu_ps(y + 8, hi);
}

static void dq_iq2_xxs_avx2(const block_iq2_xxs *b, float *y) {
    float d = f16_to_f32(b->d);
    uint32_t aux32[2];
    const uint8_t *aux8 = (const uint8_t *)aux32;
    uint8_t mag[16];
    for (int ib32 = 0; ib32 < QK_K / 32; ib32++) {
        memcpy(aux32, b->qs + 4 * ib32, 2 * sizeof(uint32_t));
        float db = d * (0.5f + (aux32[1] >> 28)) * 0.25f;
        for (int half = 0; half < 2; half++) {
            memcpy(mag,     iq2xxs_grid + aux8[2 * half],     8);
            memcpy(mag + 8, iq2xxs_grid + aux8[2 * half + 1], 8);
            avx2_iq_sign16(mag,
                           ksigns_iq2xs[(aux32[1] >> (14 * half)) & 127],
                           ksigns_iq2xs[(aux32[1] >> (14 * half + 7)) & 127],
                           y, db);
            y += 16;
        }
    }
}

static void dq_iq2_xs_avx2(const block_iq2_xs *b, float *y) {
    float d = f16_to_f32(b->d);
    uint8_t mag[16];
    for (int ib32 = 0; ib32 < QK_K / 32; ib32++) {
        float db[2] = { d * (0.5f + (b->scales[ib32] & 0xf)) * 0.25f,
                        d * (0.5f + (b->scales[ib32] >> 4)) * 0.25f };
        for (int half = 0; half < 2; half++) {
            uint16_t q0 = b->qs[4 * ib32 + 2 * half];
            uint16_t q1 = b->qs[4 * ib32 + 2 * half + 1];
            memcpy(mag,     iq2xs_grid + (q0 & 511), 8);
            memcpy(mag + 8, iq2xs_grid + (q1 & 511), 8);
            avx2_iq_sign16(mag, ksigns_iq2xs[q0 >> 9], ksigns_iq2xs[q1 >> 9],
                           y, db[half]);
            y += 16;
        }
    }
}

static void dq_iq2_s_avx2(const block_iq2_s *b, float *y) {
    float d = f16_to_f32(b->d);
    const uint8_t *qs = b->qs, *qh = b->qh, *signs = b->qs + QK_K / 8;
    uint8_t mag[16];
    for (int ib32 = 0; ib32 < QK_K / 32; ib32++) {
        float db[2] = { d * (0.5f + (b->scales[ib32] & 0xf)) * 0.25f,
                        d * (0.5f + (b->scales[ib32] >> 4)) * 0.25f };
        for (int half = 0; half < 2; half++) {
            int l0 = 2 * half, l1 = 2 * half + 1;
            memcpy(mag, iq2s_grid + (qs[l0] | ((qh[ib32] << (8 - 2 * l0)) & 0x300)), 8);
            memcpy(mag + 8, iq2s_grid + (qs[l1] | ((qh[ib32] << (8 - 2 * l1)) & 0x300)), 8);
            avx2_iq_sign16(mag, signs[l0], signs[l1], y, db[half]);
            y += 16;
        }
        qs += 4;
        signs += 4;
    }
}

static void dq_iq3_xxs_avx2(const block_iq3_xxs *b, float *y) {
    float d = f16_to_f32(b->d);
    const uint8_t *qs = b->qs, *sas = b->qs + QK_K / 4;
    uint32_t aux32;
    uint8_t mag[16];
    for (int ib32 = 0; ib32 < QK_K / 32; ib32++) {
        memcpy(&aux32, sas + 4 * ib32, sizeof(uint32_t));
        float db = d * (0.5f + (aux32 >> 28)) * 0.5f;
        for (int half = 0; half < 2; half++) {
            const uint8_t *q = qs + 4 * half;
            memcpy(mag,      iq3xxs_grid + q[0], 4);
            memcpy(mag + 4,  iq3xxs_grid + q[1], 4);
            memcpy(mag + 8,  iq3xxs_grid + q[2], 4);
            memcpy(mag + 12, iq3xxs_grid + q[3], 4);
            avx2_iq_sign16(mag,
                           ksigns_iq2xs[(aux32 >> (14 * half)) & 127],
                           ksigns_iq2xs[(aux32 >> (14 * half + 7)) & 127],
                           y, db);
            y += 16;
        }
        qs += 8;
    }
}

static void dq_iq3_s_avx2(const block_iq3_s *b, float *y) {
    float d = f16_to_f32(b->d);
    const uint8_t *qs = b->qs, *qh = b->qh, *signs = b->signs;
    uint8_t mag[16];
    for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
        float db1 = d * (1 + 2 * (b->scales[ib32 / 2] & 0xf));
        float db2 = d * (1 + 2 * (b->scales[ib32 / 2] >> 4));
        for (int k = 0; k < 2; k++) {
            float db = k ? db2 : db1;
            for (int half = 0; half < 2; half++) {
                int l0 = 2 * half, l1 = 2 * half + 1;
                memcpy(mag, iq3s_grid + (qs[2 * l0] | ((qh[k] << (8 - 2 * l0)) & 256)), 4);
                memcpy(mag + 4, iq3s_grid + (qs[2 * l0 + 1] | ((qh[k] << (7 - 2 * l0)) & 256)), 4);
                memcpy(mag + 8, iq3s_grid + (qs[2 * l1] | ((qh[k] << (8 - 2 * l1)) & 256)), 4);
                memcpy(mag + 12, iq3s_grid + (qs[2 * l1 + 1] | ((qh[k] << (7 - 2 * l1)) & 256)), 4);
                avx2_iq_sign16(mag, signs[l0], signs[l1], y, db);
                y += 16;
            }
            qs += 8;
            signs += 4;
        }
        qh += 2;
    }
}

static void dq_iq1_s_avx2(const block_iq1_s *b, float *y) {
    float d = f16_to_f32(b->d);
    const uint8_t *qs = b->qs;
    const uint16_t *qh = b->qh;
    int8_t g16[16];
    for (int ib = 0; ib < QK_K / 32; ib++) {
        float dl = d * (2 * ((qh[ib] >> 12) & 7) + 1);
        float delta = qh[ib] & 0x8000 ? -IQ1S_DELTA : IQ1S_DELTA;
        for (int half = 0; half < 2; half++) {
            int l0 = 2 * half, l1 = 2 * half + 1;
            memcpy(g16, iq1s_grid + (qs[l0] | (((qh[ib] >> 3 * l0) & 7) << 8)), 8);
            memcpy(g16 + 8, iq1s_grid + (qs[l1] | (((qh[ib] >> 3 * l1) & 7) << 8)), 8);
            avx2_iq1_16(g16, dl, delta, delta, y);
            y += 16;
        }
        qs += 4;
    }
}

static void dq_iq1_m_avx2(const block_iq1_m *b, float *y) {
    const uint16_t *sc = (const uint16_t *)b->scales;
    f16_t sd = (f16_t)((sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) |
                       ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000));
    float d = f16_to_f32(sd);
    const uint8_t *qs = b->qs, *qh = b->qh;
    int8_t g16[16];
    for (int ib = 0; ib < QK_K / 32; ib++) {
        float dl1 = d * (2 * ((sc[ib / 2] >> (6 * (ib % 2) + 0)) & 0x7) + 1);
        float dl2 = d * (2 * ((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 0x7) + 1);
        memcpy(g16,     iq1s_grid + (qs[0] | ((qh[0] << 8) & 0x700)), 8);
        memcpy(g16 + 8, iq1s_grid + (qs[1] | ((qh[0] << 4) & 0x700)), 8);
        avx2_iq1_16(g16, dl1,
                    qh[0] & 0x08 ? -IQ1S_DELTA : IQ1S_DELTA,
                    qh[0] & 0x80 ? -IQ1S_DELTA : IQ1S_DELTA, y);
        y += 16;
        memcpy(g16,     iq1s_grid + (qs[2] | ((qh[1] << 8) & 0x700)), 8);
        memcpy(g16 + 8, iq1s_grid + (qs[3] | ((qh[1] << 4) & 0x700)), 8);
        avx2_iq1_16(g16, dl2,
                    qh[1] & 0x08 ? -IQ1S_DELTA : IQ1S_DELTA,
                    qh[1] & 0x80 ? -IQ1S_DELTA : IQ1S_DELTA, y);
        y += 16;
        qs += 4;
        qh += 2;
    }
}
#endif

// NEON kernels (aarch64). Same math as the scalar code — only the
// accumulation order differs (4-lane FMA), so results can drift in the last
// ulps but stay well inside quantization noise, exactly like the AVX2 path.
#if defined(__aarch64__) && defined(__ARM_NEON)
#define RUNNER_NEON 1
#include <arm_neon.h>

// widen 16 signed bytes into four float32x4 lanes-in-order
static inline void i8_to_f32x4(int8x16_t q, float32x4_t f[4]) {
    int16x8_t lo = vmovl_s8(vget_low_s8(q));
    int16x8_t hi = vmovl_s8(vget_high_s8(q));
    f[0] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo)));
    f[1] = vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo)));
    f[2] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi)));
    f[3] = vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi)));
}
// widen 16 unsigned bytes (nibble values) the same way
static inline void u8_to_f32x4(uint8x16_t q, float32x4_t f[4]) {
    uint16x8_t lo = vmovl_u8(vget_low_u8(q));
    uint16x8_t hi = vmovl_u8(vget_high_u8(q));
    f[0] = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo)));
    f[1] = vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo)));
    f[2] = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi)));
    f[3] = vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi)));
}
// dot of a full 32-value block (two 16-byte halves, already widened order):
// four independent mul+fma chains, summed at the end — a single accumulator
// serializes eight FMAs and halves throughput on wide cores
static inline float32x4_t dot32(int8x16_t qa, int8x16_t qb, const float *x) {
    float32x4_t fa[4], fb[4];
    i8_to_f32x4(qa, fa);
    i8_to_f32x4(qb, fb);
    float32x4_t t0 = vmulq_f32(fa[0], vld1q_f32(x));
    float32x4_t t1 = vmulq_f32(fa[1], vld1q_f32(x + 4));
    float32x4_t t2 = vmulq_f32(fa[2], vld1q_f32(x + 8));
    float32x4_t t3 = vmulq_f32(fa[3], vld1q_f32(x + 12));
    t0 = vfmaq_f32(t0, fb[0], vld1q_f32(x + 16));
    t1 = vfmaq_f32(t1, fb[1], vld1q_f32(x + 20));
    t2 = vfmaq_f32(t2, fb[2], vld1q_f32(x + 24));
    t3 = vfmaq_f32(t3, fb[3], vld1q_f32(x + 28));
    return vaddq_f32(vaddq_f32(t0, t1), vaddq_f32(t2, t3));
}
// same shape for unsigned nibble values
static inline float32x4_t dot32u(uint8x16_t qa, uint8x16_t qb, const float *x) {
    return dot32(vreinterpretq_s8_u8(qa), vreinterpretq_s8_u8(qb), x);
}
static inline float32x4_t sum32(const float *x) {
    float32x4_t t0 = vaddq_f32(vld1q_f32(x),      vld1q_f32(x + 16));
    float32x4_t t1 = vaddq_f32(vld1q_f32(x + 4),  vld1q_f32(x + 20));
    float32x4_t t2 = vaddq_f32(vld1q_f32(x + 8),  vld1q_f32(x + 24));
    float32x4_t t3 = vaddq_f32(vld1q_f32(x + 12), vld1q_f32(x + 28));
    return vaddq_f32(vaddq_f32(t0, t1), vaddq_f32(t2, t3));
}
// dot of one 16-byte vector with x[0..15], two chains
static inline float32x4_t dot16(int8x16_t q, const float *x) {
    float32x4_t f[4];
    i8_to_f32x4(q, f);
    float32x4_t t0 = vmulq_f32(f[0], vld1q_f32(x));
    float32x4_t t1 = vmulq_f32(f[1], vld1q_f32(x + 4));
    t0 = vfmaq_f32(t0, f[2], vld1q_f32(x + 8));
    t1 = vfmaq_f32(t1, f[3], vld1q_f32(x + 12));
    return vaddq_f32(t0, t1);
}
static inline void st16f(float *y, const float32x4_t f[4], float d) {
    vst1q_f32(y,      vmulq_n_f32(f[0], d));
    vst1q_f32(y + 4,  vmulq_n_f32(f[1], d));
    vst1q_f32(y + 8,  vmulq_n_f32(f[2], d));
    vst1q_f32(y + 12, vmulq_n_f32(f[3], d));
}
// y = d*q - m for 16 widened bytes
static inline void st16fm(float *y, const float32x4_t f[4], float d, float m) {
    float32x4_t dv = vdupq_n_f32(d), nm = vdupq_n_f32(-m);
    vst1q_f32(y,      vfmaq_f32(nm, f[0], dv));
    vst1q_f32(y + 4,  vfmaq_f32(nm, f[1], dv));
    vst1q_f32(y + 8,  vfmaq_f32(nm, f[2], dv));
    vst1q_f32(y + 12, vfmaq_f32(nm, f[3], dv));
}

static void dq_q8_0_neon(const block_q8_0 *b, float *y) {
    float d = f16_to_f32(b->d);
    float32x4_t f[4];
    i8_to_f32x4(vld1q_s8(b->qs), f);      st16f(y, f, d);
    i8_to_f32x4(vld1q_s8(b->qs + 16), f); st16f(y + 16, f, d);
}

static void dq_mxfp4_neon(const block_mxfp4 *b, float *y) {
    const int8x16_t tbl = vld1q_s8(kvalues_mxfp4_i8);
    const uint8x16_t mF = vdupq_n_u8(0xF);
    float d = 0.5f * ldexpf(1.0f, (int)b->e - 127);
    uint8x16_t q = vld1q_u8(b->qs);
    float32x4_t f[4];
    i8_to_f32x4(vqtbl1q_s8(tbl, vandq_u8(q, mF)), f);      st16f(y, f, d);
    i8_to_f32x4(vqtbl1q_s8(tbl, vshrq_n_u8(q, 4)), f);     st16f(y + 16, f, d);
}

static void dq_q4_K_neon(const block_q4_K *b, float *y) {
    const uint8x16_t mF = vdupq_n_u8(0xF);
    float d = f16_to_f32(b->d), dmin = f16_to_f32(b->dmin);
    const uint8_t *q = b->qs;
    int is = 0;
    for (int j = 0; j < QK_K; j += 64) {
        uint8_t sc, mn;
        get_scale_min_k4(is + 0, b->scales, &sc, &mn);
        float d1 = d * sc, m1 = dmin * mn;
        get_scale_min_k4(is + 1, b->scales, &sc, &mn);
        float d2 = d * sc, m2 = dmin * mn;
        for (int l = 0; l < 32; l += 16) {
            uint8x16_t qv = vld1q_u8(q + l);
            float32x4_t f[4];
            u8_to_f32x4(vandq_u8(qv, mF), f);   st16fm(y + l, f, d1, m1);
            u8_to_f32x4(vshrq_n_u8(qv, 4), f);  st16fm(y + 32 + l, f, d2, m2);
        }
        q += 32; is += 2; y += 64;
    }
}

static void dq_q6_K_neon(const block_q6_K *b, float *y) {
    const uint8x16_t mF = vdupq_n_u8(0xF), m3 = vdupq_n_u8(3);
    const int8x16_t m32 = vdupq_n_s8(32);
    float d = f16_to_f32(b->d);
    const uint8_t *ql = b->ql, *qh = b->qh;
    const int8_t *sc = b->scales;
    for (int half = 0; half < 2; half++) {
        for (int base = 0; base < 32; base += 16) {
            int is = base / 16;
            uint8x16_t l0 = vld1q_u8(ql + base);
            uint8x16_t l1 = vld1q_u8(ql + base + 32);
            uint8x16_t h  = vld1q_u8(qh + base);
            int8x16_t q1 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(
                    vandq_u8(l0, mF),
                    vshlq_n_u8(vandq_u8(h, m3), 4))), m32);
            int8x16_t q2 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(
                    vandq_u8(l1, mF),
                    vshlq_n_u8(vandq_u8(vshrq_n_u8(h, 2), m3), 4))), m32);
            int8x16_t q3 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(
                    vshrq_n_u8(l0, 4),
                    vshlq_n_u8(vandq_u8(vshrq_n_u8(h, 4), m3), 4))), m32);
            int8x16_t q4 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(
                    vshrq_n_u8(l1, 4),
                    vshlq_n_u8(vshrq_n_u8(h, 6), 4))), m32);
            float32x4_t f[4];
            i8_to_f32x4(q1, f); st16f(y + base,      f, d * sc[is]);
            i8_to_f32x4(q2, f); st16f(y + 32 + base, f, d * sc[is + 2]);
            i8_to_f32x4(q3, f); st16f(y + 64 + base, f, d * sc[is + 4]);
            i8_to_f32x4(q4, f); st16f(y + 96 + base, f, d * sc[is + 6]);
        }
        y += 128; ql += 64; qh += 32; sc += 8;
    }
}

// ---- codebook i-quants. The grids hold u8 magnitudes <= 62, so a signed
// byte negate-select applies the sign table before the one widen+scale pass.
// kmask duplicated across both halves: lane j tests bit j%8 of its sign byte.
static const uint8_t neon_iq_kmask[16] = { 1, 2, 4, 8, 16, 32, 64, 128,
                                           1, 2, 4, 8, 16, 32, 64, 128 };

// y[0..15] = db * +/-mag[0..15], sign bit j of s0 (first 8) / s1 (second 8)
static inline void neon_iq_sign16(const uint8_t *mag, uint8_t s0, uint8_t s1,
                                  float *y, float db) {
    int8x16_t m = vld1q_s8((const int8_t *)mag);
    uint8x16_t neg = vtstq_u8(vcombine_u8(vdup_n_u8(s0), vdup_n_u8(s1)),
                              vld1q_u8(neon_iq_kmask));
    int8x16_t v = vbslq_s8(neg, vnegq_s8(m), m);
    float32x4_t f[4];
    i8_to_f32x4(v, f);
    st16f(y, f, db);
}

static void dq_iq2_xxs_neon(const block_iq2_xxs *b, float *y) {
    float d = f16_to_f32(b->d);
    uint32_t aux32[2];
    const uint8_t *aux8 = (const uint8_t *)aux32;
    uint8_t mag[16];
    for (int ib32 = 0; ib32 < QK_K / 32; ib32++) {
        memcpy(aux32, b->qs + 4 * ib32, 2 * sizeof(uint32_t));
        float db = d * (0.5f + (aux32[1] >> 28)) * 0.25f;
        for (int half = 0; half < 2; half++) {
            memcpy(mag,     iq2xxs_grid + aux8[2 * half],     8);
            memcpy(mag + 8, iq2xxs_grid + aux8[2 * half + 1], 8);
            neon_iq_sign16(mag,
                           ksigns_iq2xs[(aux32[1] >> (14 * half)) & 127],
                           ksigns_iq2xs[(aux32[1] >> (14 * half + 7)) & 127],
                           y, db);
            y += 16;
        }
    }
}

static void dq_iq2_xs_neon(const block_iq2_xs *b, float *y) {
    float d = f16_to_f32(b->d);
    uint8_t mag[16];
    for (int ib32 = 0; ib32 < QK_K / 32; ib32++) {
        float db[2] = { d * (0.5f + (b->scales[ib32] & 0xf)) * 0.25f,
                        d * (0.5f + (b->scales[ib32] >> 4)) * 0.25f };
        for (int half = 0; half < 2; half++) {
            uint16_t q0 = b->qs[4 * ib32 + 2 * half];
            uint16_t q1 = b->qs[4 * ib32 + 2 * half + 1];
            memcpy(mag,     iq2xs_grid + (q0 & 511), 8);
            memcpy(mag + 8, iq2xs_grid + (q1 & 511), 8);
            neon_iq_sign16(mag, ksigns_iq2xs[q0 >> 9], ksigns_iq2xs[q1 >> 9],
                           y, db[half]);
            y += 16;
        }
    }
}

static void dq_iq2_s_neon(const block_iq2_s *b, float *y) {
    float d = f16_to_f32(b->d);
    const uint8_t *qs = b->qs, *qh = b->qh, *signs = b->qs + QK_K / 8;
    uint8_t mag[16];
    for (int ib32 = 0; ib32 < QK_K / 32; ib32++) {
        float db[2] = { d * (0.5f + (b->scales[ib32] & 0xf)) * 0.25f,
                        d * (0.5f + (b->scales[ib32] >> 4)) * 0.25f };
        for (int half = 0; half < 2; half++) {
            int l0 = 2 * half, l1 = 2 * half + 1;
            memcpy(mag, iq2s_grid + (qs[l0] | ((qh[ib32] << (8 - 2 * l0)) & 0x300)), 8);
            memcpy(mag + 8, iq2s_grid + (qs[l1] | ((qh[ib32] << (8 - 2 * l1)) & 0x300)), 8);
            neon_iq_sign16(mag, signs[l0], signs[l1], y, db[half]);
            y += 16;
        }
        qs += 4;
        signs += 4;
    }
}

static void dq_iq3_xxs_neon(const block_iq3_xxs *b, float *y) {
    float d = f16_to_f32(b->d);
    const uint8_t *qs = b->qs, *sas = b->qs + QK_K / 4;
    uint32_t aux32;
    uint8_t mag[16];
    for (int ib32 = 0; ib32 < QK_K / 32; ib32++) {
        memcpy(&aux32, sas + 4 * ib32, sizeof(uint32_t));
        float db = d * (0.5f + (aux32 >> 28)) * 0.5f;
        for (int half = 0; half < 2; half++) {
            const uint8_t *q = qs + 4 * half;
            memcpy(mag,      iq3xxs_grid + q[0], 4);
            memcpy(mag + 4,  iq3xxs_grid + q[1], 4);
            memcpy(mag + 8,  iq3xxs_grid + q[2], 4);
            memcpy(mag + 12, iq3xxs_grid + q[3], 4);
            neon_iq_sign16(mag,
                           ksigns_iq2xs[(aux32 >> (14 * half)) & 127],
                           ksigns_iq2xs[(aux32 >> (14 * half + 7)) & 127],
                           y, db);
            y += 16;
        }
        qs += 8;
    }
}

static void dq_iq3_s_neon(const block_iq3_s *b, float *y) {
    float d = f16_to_f32(b->d);
    const uint8_t *qs = b->qs, *qh = b->qh, *signs = b->signs;
    uint8_t mag[16];
    for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
        float db1 = d * (1 + 2 * (b->scales[ib32 / 2] & 0xf));
        float db2 = d * (1 + 2 * (b->scales[ib32 / 2] >> 4));
        for (int k = 0; k < 2; k++) {
            float db = k ? db2 : db1;
            for (int half = 0; half < 2; half++) {
                int l0 = 2 * half, l1 = 2 * half + 1;
                memcpy(mag, iq3s_grid + (qs[2 * l0] | ((qh[k] << (8 - 2 * l0)) & 256)), 4);
                memcpy(mag + 4, iq3s_grid + (qs[2 * l0 + 1] | ((qh[k] << (7 - 2 * l0)) & 256)), 4);
                memcpy(mag + 8, iq3s_grid + (qs[2 * l1] | ((qh[k] << (8 - 2 * l1)) & 256)), 4);
                memcpy(mag + 12, iq3s_grid + (qs[2 * l1 + 1] | ((qh[k] << (7 - 2 * l1)) & 256)), 4);
                neon_iq_sign16(mag, signs[l0], signs[l1], y, db);
                y += 16;
            }
            qs += 8;
            signs += 4;
        }
        qh += 2;
    }
}

// iq1: signed int8 grid values plus a per-group delta, y = dl * (g + delta)
static inline void neon_iq1_16(const int8_t *g16, float dl, float d0, float d1,
                               float *y) {
    float32x4_t f[4];
    i8_to_f32x4(vld1q_s8(g16), f);
    f[0] = vaddq_f32(f[0], vdupq_n_f32(d0));
    f[1] = vaddq_f32(f[1], vdupq_n_f32(d0));
    f[2] = vaddq_f32(f[2], vdupq_n_f32(d1));
    f[3] = vaddq_f32(f[3], vdupq_n_f32(d1));
    st16f(y, f, dl);
}

static void dq_iq1_s_neon(const block_iq1_s *b, float *y) {
    float d = f16_to_f32(b->d);
    const uint8_t *qs = b->qs;
    const uint16_t *qh = b->qh;
    int8_t g16[16];
    for (int ib = 0; ib < QK_K / 32; ib++) {
        float dl = d * (2 * ((qh[ib] >> 12) & 7) + 1);
        float delta = qh[ib] & 0x8000 ? -IQ1S_DELTA : IQ1S_DELTA;
        for (int half = 0; half < 2; half++) {
            int l0 = 2 * half, l1 = 2 * half + 1;
            memcpy(g16, iq1s_grid + (qs[l0] | (((qh[ib] >> 3 * l0) & 7) << 8)), 8);
            memcpy(g16 + 8, iq1s_grid + (qs[l1] | (((qh[ib] >> 3 * l1) & 7) << 8)), 8);
            neon_iq1_16(g16, dl, delta, delta, y);
            y += 16;
        }
        qs += 4;
    }
}

static void dq_iq1_m_neon(const block_iq1_m *b, float *y) {
    const uint16_t *sc = (const uint16_t *)b->scales;
    f16_t sd = (f16_t)((sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) |
                       ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000));
    float d = f16_to_f32(sd);
    const uint8_t *qs = b->qs, *qh = b->qh;
    int8_t g16[16];
    for (int ib = 0; ib < QK_K / 32; ib++) {
        float dl1 = d * (2 * ((sc[ib / 2] >> (6 * (ib % 2) + 0)) & 0x7) + 1);
        float dl2 = d * (2 * ((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 0x7) + 1);
        memcpy(g16,     iq1s_grid + (qs[0] | ((qh[0] << 8) & 0x700)), 8);
        memcpy(g16 + 8, iq1s_grid + (qs[1] | ((qh[0] << 4) & 0x700)), 8);
        neon_iq1_16(g16, dl1,
                    qh[0] & 0x08 ? -IQ1S_DELTA : IQ1S_DELTA,
                    qh[0] & 0x80 ? -IQ1S_DELTA : IQ1S_DELTA, y);
        y += 16;
        memcpy(g16,     iq1s_grid + (qs[2] | ((qh[1] << 8) & 0x700)), 8);
        memcpy(g16 + 8, iq1s_grid + (qs[3] | ((qh[1] << 4) & 0x700)), 8);
        neon_iq1_16(g16, dl2,
                    qh[1] & 0x08 ? -IQ1S_DELTA : IQ1S_DELTA,
                    qh[1] & 0x80 ? -IQ1S_DELTA : IQ1S_DELTA, y);
        y += 16;
        qs += 4;
        qh += 2;
    }
}
#endif // RUNNER_NEON

static void dequant_block(int type, const void *src, float *dst) {
#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
    switch (type) {
        case T_Q8_0: dq_q8_0_avx2(src, dst); return;
        case T_Q4_K: dq_q4_K_avx2(src, dst); return;
        case T_Q6_K: dq_q6_K_avx2(src, dst); return;
        case T_IQ2_XXS: dq_iq2_xxs_avx2(src, dst); return;
        case T_IQ2_XS: dq_iq2_xs_avx2(src, dst); return;
        case T_IQ2_S: dq_iq2_s_avx2(src, dst); return;
        case T_IQ3_XXS: dq_iq3_xxs_avx2(src, dst); return;
        case T_IQ3_S: dq_iq3_s_avx2(src, dst); return;
        case T_IQ1_S: dq_iq1_s_avx2(src, dst); return;
        case T_IQ1_M: dq_iq1_m_avx2(src, dst); return;
    }
#elif RUNNER_NEON
    switch (type) {
        case T_Q8_0:  dq_q8_0_neon(src, dst); return;
        case T_Q4_K:  dq_q4_K_neon(src, dst); return;
        case T_Q6_K:  dq_q6_K_neon(src, dst); return;
        case T_MXFP4: dq_mxfp4_neon(src, dst); return;
        case T_IQ2_XXS: dq_iq2_xxs_neon(src, dst); return;
        case T_IQ2_XS: dq_iq2_xs_neon(src, dst); return;
        case T_IQ2_S: dq_iq2_s_neon(src, dst); return;
        case T_IQ3_XXS: dq_iq3_xxs_neon(src, dst); return;
        case T_IQ3_S: dq_iq3_s_neon(src, dst); return;
        case T_IQ1_S: dq_iq1_s_neon(src, dst); return;
        case T_IQ1_M: dq_iq1_m_neon(src, dst); return;
    }
#endif
    switch (type) {
        case T_Q4_0: dq_q4_0(src, dst); break;
        case T_Q4_1: dq_q4_1(src, dst); break;
        case T_Q5_0: dq_q5_0(src, dst); break;
        case T_Q5_1: dq_q5_1(src, dst); break;
        case T_Q8_0: dq_q8_0(src, dst); break;
        case T_Q2_K: dq_q2_K(src, dst); break;
        case T_Q3_K: dq_q3_K(src, dst); break;
        case T_Q4_K: dq_q4_K(src, dst); break;
        case T_Q5_K: dq_q5_K(src, dst); break;
        case T_Q6_K: dq_q6_K(src, dst); break;
        case T_IQ4_NL: dq_iq4_nl(src, dst); break;
        case T_IQ4_XS: dq_iq4_xs(src, dst); break;
        case T_IQ2_XXS: dq_iq2_xxs(src, dst); break;
        case T_IQ2_XS: dq_iq2_xs(src, dst); break;
        case T_IQ2_S: dq_iq2_s(src, dst); break;
        case T_IQ3_XXS: dq_iq3_xxs(src, dst); break;
        case T_IQ3_S: dq_iq3_s(src, dst); break;
        case T_IQ1_S: dq_iq1_s(src, dst); break;
        case T_IQ1_M: dq_iq1_m(src, dst); break;
        case T_MXFP4: dq_mxfp4(src, dst); break;
    }
}

void dequant_row(int type, const void *src, float *dst, int n) {
    if (type == T_F32) { memcpy(dst, src, (size_t)n * 4); return; }
    if (type == T_F16) {
        const f16_t *h = src;
        for (int i = 0; i < n; i++) dst[i] = f16_to_f32(h[i]);
        return;
    }
    if (type == T_BF16) {
        const uint16_t *h = src;
        for (int i = 0; i < n; i++) dst[i] = bf16_to_f32(h[i]);
        return;
    }
    int bs = ggml_block_size(type);
    size_t ts = ggml_type_size(type);
    const uint8_t *p = src;
    for (int i = 0; i < n; i += bs, p += ts) dequant_block(type, p, dst + i);
}

// ------------------------------------------------------------- batched dots

// out[b] = dot(w, x + b*x_stride) for nb activation columns sharing one
// dequantized weight row — the inner loop of batched prompt eval. Register
// blocking (4 columns per pass) reuses each weight load four times.
void vec_dot_f32_multi(const float *w, const float *x, int x_stride,
                       int nb, int n, float *out) {
#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
    int b = 0;
    for (; b + 4 <= nb; b += 4) {
        const float *x0 = x + (size_t)b * x_stride;
        const float *x1 = x0 + x_stride, *x2 = x1 + x_stride, *x3 = x2 + x_stride;
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
        __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
        int i = 0;
        for (; i + 8 <= n; i += 8) {
            __m256 wv = _mm256_loadu_ps(w + i);
            a0 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x0 + i), a0);
            a1 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x1 + i), a1);
            a2 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x2 + i), a2);
            a3 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x3 + i), a3);
        }
        float s0, s1, s2, s3;
        {
            __m128 l0 = _mm_add_ps(_mm256_castps256_ps128(a0), _mm256_extractf128_ps(a0, 1));
            __m128 l1 = _mm_add_ps(_mm256_castps256_ps128(a1), _mm256_extractf128_ps(a1, 1));
            __m128 l2 = _mm_add_ps(_mm256_castps256_ps128(a2), _mm256_extractf128_ps(a2, 1));
            __m128 l3 = _mm_add_ps(_mm256_castps256_ps128(a3), _mm256_extractf128_ps(a3, 1));
            l0 = _mm_add_ps(l0, _mm_movehl_ps(l0, l0)); l0 = _mm_add_ss(l0, _mm_shuffle_ps(l0, l0, 1));
            l1 = _mm_add_ps(l1, _mm_movehl_ps(l1, l1)); l1 = _mm_add_ss(l1, _mm_shuffle_ps(l1, l1, 1));
            l2 = _mm_add_ps(l2, _mm_movehl_ps(l2, l2)); l2 = _mm_add_ss(l2, _mm_shuffle_ps(l2, l2, 1));
            l3 = _mm_add_ps(l3, _mm_movehl_ps(l3, l3)); l3 = _mm_add_ss(l3, _mm_shuffle_ps(l3, l3, 1));
            s0 = _mm_cvtss_f32(l0); s1 = _mm_cvtss_f32(l1);
            s2 = _mm_cvtss_f32(l2); s3 = _mm_cvtss_f32(l3);
        }
        for (; i < n; i++) {
            float wv = w[i];
            s0 += wv * x0[i]; s1 += wv * x1[i];
            s2 += wv * x2[i]; s3 += wv * x3[i];
        }
        out[b] = s0; out[b + 1] = s1; out[b + 2] = s2; out[b + 3] = s3;
    }
    for (; b < nb; b++) {
        const float *xb = x + (size_t)b * x_stride;
        __m256 acc = _mm256_setzero_ps();
        int i = 0;
        for (; i + 8 <= n; i += 8)
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(w + i),
                                  _mm256_loadu_ps(xb + i), acc);
        __m128 l = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
        l = _mm_add_ps(l, _mm_movehl_ps(l, l));
        l = _mm_add_ss(l, _mm_shuffle_ps(l, l, 1));
        float s = _mm_cvtss_f32(l);
        for (; i < n; i++) s += w[i] * xb[i];
        out[b] = s;
    }
#elif defined(__aarch64__) && defined(__ARM_NEON)
    int b = 0;
    for (; b + 4 <= nb; b += 4) {
        const float *x0 = x + (size_t)b * x_stride;
        const float *x1 = x0 + x_stride, *x2 = x1 + x_stride, *x3 = x2 + x_stride;
        float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0);
        float32x4_t a2 = vdupq_n_f32(0), a3 = vdupq_n_f32(0);
        int i = 0;
        for (; i + 4 <= n; i += 4) {
            float32x4_t wv = vld1q_f32(w + i);
            a0 = vfmaq_f32(a0, wv, vld1q_f32(x0 + i));
            a1 = vfmaq_f32(a1, wv, vld1q_f32(x1 + i));
            a2 = vfmaq_f32(a2, wv, vld1q_f32(x2 + i));
            a3 = vfmaq_f32(a3, wv, vld1q_f32(x3 + i));
        }
        float s0 = vaddvq_f32(a0), s1 = vaddvq_f32(a1);
        float s2 = vaddvq_f32(a2), s3 = vaddvq_f32(a3);
        for (; i < n; i++) {
            float wv = w[i];
            s0 += wv * x0[i]; s1 += wv * x1[i];
            s2 += wv * x2[i]; s3 += wv * x3[i];
        }
        out[b] = s0; out[b + 1] = s1; out[b + 2] = s2; out[b + 3] = s3;
    }
    for (; b < nb; b++) {
        const float *xb = x + (size_t)b * x_stride;
        float32x4_t acc = vdupq_n_f32(0);
        int i = 0;
        for (; i + 4 <= n; i += 4)
            acc = vfmaq_f32(acc, vld1q_f32(w + i), vld1q_f32(xb + i));
        float s = vaddvq_f32(acc);
        for (; i < n; i++) s += w[i] * xb[i];
        out[b] = s;
    }
#else
    for (int b = 0; b < nb; b++) {
        const float *xb = x + (size_t)b * x_stride;
        float s = 0;
        for (int i = 0; i < n; i++) s += w[i] * xb[i];
        out[b] = s;
    }
#endif
}

// ------------------------------------------------------------- q8 KV cache

// quantize a row of floats into q8_0 blocks (n must be a multiple of 32)
void q8_quant_row(const float *x, void *dst, int n) {
    block_q8_0 *b = dst;
#if defined(__aarch64__) && defined(__ARM_NEON)
    // vcvtaq rounds ties away from zero — identical to the scalar roundf, so
    // the two paths produce byte-identical blocks
    for (int i = 0; i < n / QK; i++, b++, x += QK) {
        float32x4_t mx = vabsq_f32(vld1q_f32(x));
        for (int j = 4; j < QK; j += 4)
            mx = vmaxq_f32(mx, vabsq_f32(vld1q_f32(x + j)));
        float amax = vmaxvq_f32(mx);
        float d = amax / 127.0f;
        float id = d > 0 ? 1.0f / d : 0.0f;
        b->d = f32_to_f16(d);
        for (int j = 0; j < QK; j += 16) {
            int32x4_t i0 = vcvtaq_s32_f32(vmulq_n_f32(vld1q_f32(x + j), id));
            int32x4_t i1 = vcvtaq_s32_f32(vmulq_n_f32(vld1q_f32(x + j + 4), id));
            int32x4_t i2 = vcvtaq_s32_f32(vmulq_n_f32(vld1q_f32(x + j + 8), id));
            int32x4_t i3 = vcvtaq_s32_f32(vmulq_n_f32(vld1q_f32(x + j + 12), id));
            int16x8_t lo = vcombine_s16(vmovn_s32(i0), vmovn_s32(i1));
            int16x8_t hi = vcombine_s16(vmovn_s32(i2), vmovn_s32(i3));
            vst1q_s8(b->qs + j, vcombine_s8(vmovn_s16(lo), vmovn_s16(hi)));
        }
    }
#else
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
#endif
}

// out[i] += a * dequant(row)[i] — the V accumulation in q8 attention
void q8_accum_row(const void *src, float a, float *out, int n) {
    const block_q8_0 *b = src;
#if defined(__aarch64__) && defined(__ARM_NEON)
    for (int i = 0; i < n / QK; i++, b++, out += QK) {
        float ad = a * f16_to_f32(b->d);
        float32x4_t f[4];
        i8_to_f32x4(vld1q_s8(b->qs), f);
        for (int j = 0; j < 4; j++)
            vst1q_f32(out + 4 * j,
                      vfmaq_n_f32(vld1q_f32(out + 4 * j), f[j], ad));
        i8_to_f32x4(vld1q_s8(b->qs + 16), f);
        for (int j = 0; j < 4; j++)
            vst1q_f32(out + 16 + 4 * j,
                      vfmaq_n_f32(vld1q_f32(out + 16 + 4 * j), f[j], ad));
    }
#else
    for (int i = 0; i < n / QK; i++, b++, out += QK) {
        float ad = a * f16_to_f32(b->d);
        for (int j = 0; j < QK; j++) out[j] += ad * b->qs[j];
    }
#endif
}

// ---------------------------------------------------------------- vec_dot

// AVX2 kernels for the hot quant formats. Same math as the scalar code —
// only the accumulation order differs (8-lane FMA), so results can drift in
// the last ulps but stay well inside quantization noise. NEON kernels below
// mirror this set for aarch64; scalar code remains the fallback elsewhere.
#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
#define RUNNER_AVX2 1
#include <immintrin.h>

static inline float hsum8(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    return _mm_cvtss_f32(s);
}
// low/high 8 bytes of a 16-byte vector as 8 floats (sign-extended)
static inline __m256 i8lo_ps(__m128i v) {
    return _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v));
}
static inline __m256 i8hi_ps(__m128i v) {
    return _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(v, 8)));
}

static float dot_f16_avx2(const f16_t *w, const float *x, int n) {
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        a0 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w + i))),
                             _mm256_loadu_ps(x + i), a0);
        a1 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w + i + 8))),
                             _mm256_loadu_ps(x + i + 8), a1);
    }
    float s = hsum8(_mm256_add_ps(a0, a1));
    for (; i < n; i++) s += f16_to_f32(w[i]) * x[i];
    return s;
}

static float dot_bf16_avx2(const uint16_t *w, const float *x, int n) {
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256i wi = _mm256_slli_epi32(
            _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(w + i))), 16);
        acc = _mm256_fmadd_ps(_mm256_castsi256_ps(wi), _mm256_loadu_ps(x + i), acc);
    }
    float s = hsum8(acc);
    for (; i < n; i++) s += bf16_to_f32(w[i]) * x[i];
    return s;
}

static float dot_q8_0_avx2(const block_q8_0 *b, const float *x, int n) {
    __m256 acc = _mm256_setzero_ps();
    for (int i = 0; i < n / QK; i++) {
        const __m128i q0 = _mm_loadu_si128((const __m128i *)b[i].qs);
        const __m128i q1 = _mm_loadu_si128((const __m128i *)(b[i].qs + 16));
        const float *xp = x + i * QK;
        __m256 t = _mm256_mul_ps(i8lo_ps(q0), _mm256_loadu_ps(xp));
        t = _mm256_fmadd_ps(i8hi_ps(q0), _mm256_loadu_ps(xp + 8), t);
        t = _mm256_fmadd_ps(i8lo_ps(q1), _mm256_loadu_ps(xp + 16), t);
        t = _mm256_fmadd_ps(i8hi_ps(q1), _mm256_loadu_ps(xp + 24), t);
        acc = _mm256_fmadd_ps(_mm256_set1_ps(f16_to_f32(b[i].d)), t, acc);
    }
    return hsum8(acc);
}

static float dot_q4_0_avx2(const block_q4_0 *b, const float *x, int n) {
    const __m128i mF = _mm_set1_epi8(0xF), m8 = _mm_set1_epi8(8);
    __m256 acc = _mm256_setzero_ps();
    for (int i = 0; i < n / QK; i++) {
        __m128i q  = _mm_loadu_si128((const __m128i *)b[i].qs);
        __m128i lo = _mm_sub_epi8(_mm_and_si128(q, mF), m8);
        __m128i hi = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(q, 4), mF), m8);
        const float *xp = x + i * QK;
        __m256 t = _mm256_mul_ps(i8lo_ps(lo), _mm256_loadu_ps(xp));
        t = _mm256_fmadd_ps(i8hi_ps(lo), _mm256_loadu_ps(xp + 8), t);
        t = _mm256_fmadd_ps(i8lo_ps(hi), _mm256_loadu_ps(xp + 16), t);
        t = _mm256_fmadd_ps(i8hi_ps(hi), _mm256_loadu_ps(xp + 24), t);
        acc = _mm256_fmadd_ps(_mm256_set1_ps(f16_to_f32(b[i].d)), t, acc);
    }
    return hsum8(acc);
}

static float dot_q4_K_avx2(const block_q4_K *b, const float *x, int n) {
    const __m128i mF = _mm_set1_epi8(0xF);
    float s = 0;
    for (int i = 0; i < n / QK_K; i++, b++) {
        float d = f16_to_f32(b->d), dmin = f16_to_f32(b->dmin);
        const uint8_t *q = b->qs;
        const float *xp = x + i * QK_K;
        int is = 0;
        for (int j = 0; j < QK_K; j += 64) {
            uint8_t sc, mn;
            get_scale_min_k4(is + 0, b->scales, &sc, &mn);
            float d1 = d * sc, m1 = dmin * mn;
            get_scale_min_k4(is + 1, b->scales, &sc, &mn);
            float d2 = d * sc, m2 = dmin * mn;
            __m256 t1 = _mm256_setzero_ps(), t2 = _mm256_setzero_ps();
            __m256 s1 = _mm256_setzero_ps(), s2 = _mm256_setzero_ps();
            for (int l = 0; l < 32; l += 16) {
                __m128i qv = _mm_loadu_si128((const __m128i *)(q + l));
                __m128i lo = _mm_and_si128(qv, mF);
                __m128i hi = _mm_and_si128(_mm_srli_epi16(qv, 4), mF);
                __m256 x0 = _mm256_loadu_ps(xp + l),      x1 = _mm256_loadu_ps(xp + l + 8);
                __m256 x2 = _mm256_loadu_ps(xp + 32 + l), x3 = _mm256_loadu_ps(xp + 32 + l + 8);
                t1 = _mm256_fmadd_ps(i8lo_ps(lo), x0, t1);
                t1 = _mm256_fmadd_ps(i8hi_ps(lo), x1, t1);
                t2 = _mm256_fmadd_ps(i8lo_ps(hi), x2, t2);
                t2 = _mm256_fmadd_ps(i8hi_ps(hi), x3, t2);
                s1 = _mm256_add_ps(s1, _mm256_add_ps(x0, x1));
                s2 = _mm256_add_ps(s2, _mm256_add_ps(x2, x3));
            }
            s += d1 * hsum8(t1) - m1 * hsum8(s1) + d2 * hsum8(t2) - m2 * hsum8(s2);
            q += 32; is += 2; xp += 64;
        }
    }
    return s;
}

static float dot_q5_K_avx2(const block_q5_K *b, const float *x, int n) {
    const __m128i mF = _mm_set1_epi8(0xF), m16 = _mm_set1_epi8(16);
    float s = 0;
    for (int i = 0; i < n / QK_K; i++, b++) {
        float d = f16_to_f32(b->d), dmin = f16_to_f32(b->dmin);
        const uint8_t *q = b->qs, *qh = b->qh;
        const float *xp = x + i * QK_K;
        int is = 0;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < QK_K; j += 64) {
            uint8_t sc, mn;
            get_scale_min_k4(is + 0, b->scales, &sc, &mn);
            float d1 = d * sc, m1 = dmin * mn;
            get_scale_min_k4(is + 1, b->scales, &sc, &mn);
            float d2 = d * sc, m2 = dmin * mn;
            __m128i u1v = _mm_set1_epi8((char)u1), u2v = _mm_set1_epi8((char)u2);
            __m256 t1 = _mm256_setzero_ps(), t2 = _mm256_setzero_ps();
            __m256 s1 = _mm256_setzero_ps(), s2 = _mm256_setzero_ps();
            for (int l = 0; l < 32; l += 16) {
                __m128i qv  = _mm_loadu_si128((const __m128i *)(q + l));
                __m128i qhv = _mm_loadu_si128((const __m128i *)(qh + l));
                __m128i lo = _mm_and_si128(qv, mF);
                __m128i hi = _mm_and_si128(_mm_srli_epi16(qv, 4), mF);
                lo = _mm_add_epi8(lo, _mm_and_si128(
                        _mm_cmpeq_epi8(_mm_and_si128(qhv, u1v), u1v), m16));
                hi = _mm_add_epi8(hi, _mm_and_si128(
                        _mm_cmpeq_epi8(_mm_and_si128(qhv, u2v), u2v), m16));
                __m256 x0 = _mm256_loadu_ps(xp + l),      x1 = _mm256_loadu_ps(xp + l + 8);
                __m256 x2 = _mm256_loadu_ps(xp + 32 + l), x3 = _mm256_loadu_ps(xp + 32 + l + 8);
                t1 = _mm256_fmadd_ps(i8lo_ps(lo), x0, t1);
                t1 = _mm256_fmadd_ps(i8hi_ps(lo), x1, t1);
                t2 = _mm256_fmadd_ps(i8lo_ps(hi), x2, t2);
                t2 = _mm256_fmadd_ps(i8hi_ps(hi), x3, t2);
                s1 = _mm256_add_ps(s1, _mm256_add_ps(x0, x1));
                s2 = _mm256_add_ps(s2, _mm256_add_ps(x2, x3));
            }
            s += d1 * hsum8(t1) - m1 * hsum8(s1) + d2 * hsum8(t2) - m2 * hsum8(s2);
            q += 32; is += 2; xp += 64; u1 <<= 2; u2 <<= 2;
        }
    }
    return s;
}

static float dot_q6_K_avx2(const block_q6_K *b, const float *x, int n) {
    const __m128i mF = _mm_set1_epi8(0xF), m3 = _mm_set1_epi8(3), m32 = _mm_set1_epi8(32);
    float s = 0;
    for (int i = 0; i < n / QK_K; i++, b++) {
        float d = f16_to_f32(b->d);
        const uint8_t *ql = b->ql, *qh = b->qh;
        const int8_t *sc = b->scales;
        const float *xp = x + i * QK_K;
        for (int half = 0; half < 2; half++) {
            float t[8];
            for (int base = 0; base < 32; base += 16) {
                int is = base / 16;
                __m128i l0 = _mm_loadu_si128((const __m128i *)(ql + base));
                __m128i l1 = _mm_loadu_si128((const __m128i *)(ql + base + 32));
                __m128i h  = _mm_loadu_si128((const __m128i *)(qh + base));
                __m128i q1 = _mm_sub_epi8(_mm_or_si128(_mm_and_si128(l0, mF),
                        _mm_slli_epi16(_mm_and_si128(h, m3), 4)), m32);
                __m128i q2 = _mm_sub_epi8(_mm_or_si128(_mm_and_si128(l1, mF),
                        _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(h, 2), m3), 4)), m32);
                __m128i q3 = _mm_sub_epi8(_mm_or_si128(
                        _mm_and_si128(_mm_srli_epi16(l0, 4), mF),
                        _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(h, 4), m3), 4)), m32);
                __m128i q4 = _mm_sub_epi8(_mm_or_si128(
                        _mm_and_si128(_mm_srli_epi16(l1, 4), mF),
                        _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(h, 6), m3), 4)), m32);
                __m256 a1 = _mm256_mul_ps(i8lo_ps(q1), _mm256_loadu_ps(xp + base));
                a1 = _mm256_fmadd_ps(i8hi_ps(q1), _mm256_loadu_ps(xp + base + 8), a1);
                __m256 a2 = _mm256_mul_ps(i8lo_ps(q2), _mm256_loadu_ps(xp + 32 + base));
                a2 = _mm256_fmadd_ps(i8hi_ps(q2), _mm256_loadu_ps(xp + 32 + base + 8), a2);
                __m256 a3 = _mm256_mul_ps(i8lo_ps(q3), _mm256_loadu_ps(xp + 64 + base));
                a3 = _mm256_fmadd_ps(i8hi_ps(q3), _mm256_loadu_ps(xp + 64 + base + 8), a3);
                __m256 a4 = _mm256_mul_ps(i8lo_ps(q4), _mm256_loadu_ps(xp + 96 + base));
                a4 = _mm256_fmadd_ps(i8hi_ps(q4), _mm256_loadu_ps(xp + 96 + base + 8), a4);
                t[is * 4 + 0] = hsum8(a1);
                t[is * 4 + 1] = hsum8(a2);
                t[is * 4 + 2] = hsum8(a3);
                t[is * 4 + 3] = hsum8(a4);
            }
            s += d * (sc[0] * t[0] + sc[2] * t[1] + sc[4] * t[2] + sc[6] * t[3] +
                      sc[1] * t[4] + sc[3] * t[5] + sc[5] * t[6] + sc[7] * t[7]);
            ql += 64; qh += 32; sc += 8; xp += 128;
        }
    }
    return s;
}
// IQ4: the nibble is an index into the 16-entry kvalues table — pshufb does
// the whole lookup in one instruction
static float dot_iq4_nl_avx2(const block_iq4_nl *b, const float *x, int n) {
    const __m128i tbl = _mm_loadu_si128((const __m128i *)kvalues_iq4nl);
    const __m128i mF  = _mm_set1_epi8(0xF);
    __m256 acc = _mm256_setzero_ps();
    for (int i = 0; i < n / QK; i++) {
        __m128i q  = _mm_loadu_si128((const __m128i *)b[i].qs); // 16B, 32 vals
        __m128i lo = _mm_shuffle_epi8(tbl, _mm_and_si128(q, mF));
        __m128i hi = _mm_shuffle_epi8(tbl, _mm_and_si128(_mm_srli_epi16(q, 4), mF));
        const float *xp = x + i * QK;
        __m256 t = _mm256_mul_ps(i8lo_ps(lo), _mm256_loadu_ps(xp));
        t = _mm256_fmadd_ps(i8hi_ps(lo), _mm256_loadu_ps(xp + 8), t);
        t = _mm256_fmadd_ps(i8lo_ps(hi), _mm256_loadu_ps(xp + 16), t);
        t = _mm256_fmadd_ps(i8hi_ps(hi), _mm256_loadu_ps(xp + 24), t);
        acc = _mm256_fmadd_ps(_mm256_set1_ps(f16_to_f32(b[i].d)), t, acc);
    }
    return hsum8(acc);
}

// MXFP4: nibble → E2M1 value via the doubled-integer table (pshufb), the 0.5
// folded into the block scale
static float dot_mxfp4_avx2(const block_mxfp4 *b, const float *x, int n) {
    const __m128i tbl = _mm_loadu_si128((const __m128i *)kvalues_mxfp4_i8);
    const __m128i mF  = _mm_set1_epi8(0xF);
    __m256 acc = _mm256_setzero_ps();
    for (int i = 0; i < n / QK; i++) {
        __m128i q  = _mm_loadu_si128((const __m128i *)b[i].qs);
        __m128i lo = _mm_shuffle_epi8(tbl, _mm_and_si128(q, mF));
        __m128i hi = _mm_shuffle_epi8(tbl, _mm_and_si128(_mm_srli_epi16(q, 4), mF));
        const float *xp = x + i * QK;
        __m256 t = _mm256_mul_ps(i8lo_ps(lo), _mm256_loadu_ps(xp));
        t = _mm256_fmadd_ps(i8hi_ps(lo), _mm256_loadu_ps(xp + 8), t);
        t = _mm256_fmadd_ps(i8lo_ps(hi), _mm256_loadu_ps(xp + 16), t);
        t = _mm256_fmadd_ps(i8hi_ps(hi), _mm256_loadu_ps(xp + 24), t);
        float d = 0.5f * ldexpf(1.0f, (int)b[i].e - 127);
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), t, acc);
    }
    return hsum8(acc);
}

static float dot_iq4_xs_avx2(const block_iq4_xs *b, const float *x, int n) {
    const __m128i tbl = _mm_loadu_si128((const __m128i *)kvalues_iq4nl);
    const __m128i mF  = _mm_set1_epi8(0xF);
    float s = 0;
    for (int i = 0; i < n / QK_K; i++, b++) {
        float d = f16_to_f32(b->d);
        const uint8_t *qs = b->qs;
        const float *xp = x + i * QK_K;
        for (int ib = 0; ib < QK_K / 32; ib++) {
            int ls = ((b->scales_l[ib / 2] >> 4 * (ib % 2)) & 0xF) |
                     (((b->scales_h >> 2 * ib) & 3) << 4);
            __m128i q  = _mm_loadu_si128((const __m128i *)qs);
            __m128i lo = _mm_shuffle_epi8(tbl, _mm_and_si128(q, mF));
            __m128i hi = _mm_shuffle_epi8(tbl, _mm_and_si128(_mm_srli_epi16(q, 4), mF));
            __m256 t = _mm256_mul_ps(i8lo_ps(lo), _mm256_loadu_ps(xp));
            t = _mm256_fmadd_ps(i8hi_ps(lo), _mm256_loadu_ps(xp + 8), t);
            t = _mm256_fmadd_ps(i8lo_ps(hi), _mm256_loadu_ps(xp + 16), t);
            t = _mm256_fmadd_ps(i8hi_ps(hi), _mm256_loadu_ps(xp + 24), t);
            s += d * (ls - 32) * hsum8(t);
            qs += 16; xp += 32;
        }
    }
    return s;
}
#endif // RUNNER_AVX2

// NEON dot kernels — the aarch64 mirror of the AVX2 set above. Helpers
// (i8_to_f32x4 / u8_to_f32x4 / fma16) live next to the NEON dequant kernels.
#if RUNNER_NEON

// No NEON kernels for F16, BF16, Q8_0 or Q4_K on purpose: their scalar loops
// are simple enough that clang auto-vectorizes them with more parallel
// accumulator chains than the hand-written versions had — every one was
// measured slower or at par as an intrinsic kernel. The formats kept below
// (nibble unpacks and table lookups) are the ones the compiler cannot
// vectorize on its own.

static float dot_q4_0_neon(const block_q4_0 *b, const float *x, int n) {
    const uint8x16_t mF = vdupq_n_u8(0xF);
    const int8x16_t m8 = vdupq_n_s8(8);
    float32x4_t acc = vdupq_n_f32(0);
    for (int i = 0; i < n / QK; i++) {
        uint8x16_t q = vld1q_u8(b[i].qs);
        int8x16_t lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(q, mF)), m8);
        int8x16_t hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(q, 4)), m8);
        acc = vfmaq_n_f32(acc, dot32(lo, hi, x + i * QK), f16_to_f32(b[i].d));
    }
    return vaddvq_f32(acc);
}

static float dot_mxfp4_neon(const block_mxfp4 *b, const float *x, int n) {
    const int8x16_t tbl = vld1q_s8(kvalues_mxfp4_i8);
    const uint8x16_t mF = vdupq_n_u8(0xF);
    float32x4_t acc = vdupq_n_f32(0);
    for (int i = 0; i < n / QK; i++) {
        uint8x16_t q = vld1q_u8(b[i].qs);
        float32x4_t t = dot32(vqtbl1q_s8(tbl, vandq_u8(q, mF)),
                              vqtbl1q_s8(tbl, vshrq_n_u8(q, 4)), x + i * QK);
        acc = vfmaq_n_f32(acc, t, 0.5f * ldexpf(1.0f, (int)b[i].e - 127));
    }
    return vaddvq_f32(acc);
}

static float dot_q5_K_neon(const block_q5_K *b, const float *x, int n) {
    const uint8x16_t mF = vdupq_n_u8(0xF), m16 = vdupq_n_u8(16);
    float s = 0;
    for (int i = 0; i < n / QK_K; i++, b++) {
        float d = f16_to_f32(b->d), dmin = f16_to_f32(b->dmin);
        const uint8_t *q = b->qs, *qh = b->qh;
        const float *xp = x + i * QK_K;
        int is = 0;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < QK_K; j += 64) {
            uint8_t sc, mn;
            get_scale_min_k4(is + 0, b->scales, &sc, &mn);
            float d1 = d * sc, m1 = dmin * mn;
            get_scale_min_k4(is + 1, b->scales, &sc, &mn);
            float d2 = d * sc, m2 = dmin * mn;
            uint8x16_t u1v = vdupq_n_u8(u1), u2v = vdupq_n_u8(u2);
            uint8x16_t q0 = vld1q_u8(q),  q1 = vld1q_u8(q + 16);
            uint8x16_t h0 = vld1q_u8(qh), h1 = vld1q_u8(qh + 16);
            uint8x16_t lo0 = vaddq_u8(vandq_u8(q0, mF),
                    vandq_u8(vtstq_u8(h0, u1v), m16));
            uint8x16_t lo1 = vaddq_u8(vandq_u8(q1, mF),
                    vandq_u8(vtstq_u8(h1, u1v), m16));
            uint8x16_t hi0 = vaddq_u8(vshrq_n_u8(q0, 4),
                    vandq_u8(vtstq_u8(h0, u2v), m16));
            uint8x16_t hi1 = vaddq_u8(vshrq_n_u8(q1, 4),
                    vandq_u8(vtstq_u8(h1, u2v), m16));
            float32x4_t t1 = dot32u(lo0, lo1, xp);
            float32x4_t t2 = dot32u(hi0, hi1, xp + 32);
            s += d1 * vaddvq_f32(t1) - m1 * vaddvq_f32(sum32(xp))
               + d2 * vaddvq_f32(t2) - m2 * vaddvq_f32(sum32(xp + 32));
            q += 32; is += 2; xp += 64; u1 <<= 2; u2 <<= 2;
        }
    }
    return s;
}

static float dot_q6_K_neon(const block_q6_K *b, const float *x, int n) {
    const uint8x16_t mF = vdupq_n_u8(0xF), m3 = vdupq_n_u8(3);
    const int8x16_t m32 = vdupq_n_s8(32);
    float s = 0;
    for (int i = 0; i < n / QK_K; i++, b++) {
        float d = f16_to_f32(b->d);
        const uint8_t *ql = b->ql, *qh = b->qh;
        const int8_t *sc = b->scales;
        const float *xp = x + i * QK_K;
        for (int half = 0; half < 2; half++) {
            float t[8];
            for (int base = 0; base < 32; base += 16) {
                int is = base / 16;
                uint8x16_t l0 = vld1q_u8(ql + base);
                uint8x16_t l1 = vld1q_u8(ql + base + 32);
                uint8x16_t h  = vld1q_u8(qh + base);
                int8x16_t q1 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(
                        vandq_u8(l0, mF),
                        vshlq_n_u8(vandq_u8(h, m3), 4))), m32);
                int8x16_t q2 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(
                        vandq_u8(l1, mF),
                        vshlq_n_u8(vandq_u8(vshrq_n_u8(h, 2), m3), 4))), m32);
                int8x16_t q3 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(
                        vshrq_n_u8(l0, 4),
                        vshlq_n_u8(vandq_u8(vshrq_n_u8(h, 4), m3), 4))), m32);
                int8x16_t q4 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(
                        vshrq_n_u8(l1, 4),
                        vshlq_n_u8(vshrq_n_u8(h, 6), 4))), m32);
                t[is * 4 + 0] = vaddvq_f32(dot16(q1, xp + base));
                t[is * 4 + 1] = vaddvq_f32(dot16(q2, xp + 32 + base));
                t[is * 4 + 2] = vaddvq_f32(dot16(q3, xp + 64 + base));
                t[is * 4 + 3] = vaddvq_f32(dot16(q4, xp + 96 + base));
            }
            s += d * (sc[0] * t[0] + sc[2] * t[1] + sc[4] * t[2] + sc[6] * t[3] +
                      sc[1] * t[4] + sc[3] * t[5] + sc[5] * t[6] + sc[7] * t[7]);
            ql += 64; qh += 32; sc += 8; xp += 128;
        }
    }
    return s;
}

static float dot_iq4_nl_neon(const block_iq4_nl *b, const float *x, int n) {
    const int8x16_t tbl = vld1q_s8(kvalues_iq4nl);
    const uint8x16_t mF = vdupq_n_u8(0xF);
    float32x4_t acc = vdupq_n_f32(0);
    for (int i = 0; i < n / QK; i++) {
        uint8x16_t q = vld1q_u8(b[i].qs);
        float32x4_t t = dot32(vqtbl1q_s8(tbl, vandq_u8(q, mF)),
                              vqtbl1q_s8(tbl, vshrq_n_u8(q, 4)), x + i * QK);
        acc = vfmaq_n_f32(acc, t, f16_to_f32(b[i].d));
    }
    return vaddvq_f32(acc);
}

static float dot_iq4_xs_neon(const block_iq4_xs *b, const float *x, int n) {
    const int8x16_t tbl = vld1q_s8(kvalues_iq4nl);
    const uint8x16_t mF = vdupq_n_u8(0xF);
    float s = 0;
    for (int i = 0; i < n / QK_K; i++, b++) {
        float d = f16_to_f32(b->d);
        const uint8_t *qs = b->qs;
        const float *xp = x + i * QK_K;
        for (int ib = 0; ib < QK_K / 32; ib++) {
            int ls = ((b->scales_l[ib / 2] >> 4 * (ib % 2)) & 0xF) |
                     (((b->scales_h >> 2 * ib) & 3) << 4);
            uint8x16_t q = vld1q_u8(qs);
            float32x4_t t = dot32(vqtbl1q_s8(tbl, vandq_u8(q, mF)),
                                  vqtbl1q_s8(tbl, vshrq_n_u8(q, 4)), xp);
            s += d * (ls - 32) * vaddvq_f32(t);
            qs += 16; xp += 32;
        }
    }
    return s;
}
#endif // RUNNER_NEON

float vec_dot(int type, const void *row, const float *x, int n) {
    switch (type) {
        case T_F32: {
            const float *w = row;
            float s = 0;
            for (int i = 0; i < n; i++) s += w[i] * x[i];
            return s;
        }
        case T_F16: {
            const f16_t *w = row;
            float s = 0;
#if RUNNER_AVX2
            return dot_f16_avx2(w, x, n);
#elif defined(__ARM_FP16_FORMAT_IEEE)
            // no NEON kernel on purpose: this loop auto-vectorizes to
            // fcvtl+fmla and outruns a hand-written 4-wide version
            const __fp16 *h = (const __fp16 *)w;
            for (int i = 0; i < n; i++) s += (float)h[i] * x[i];
#else
            for (int i = 0; i < n; i++) s += f16_to_f32(w[i]) * x[i];
#endif
            return s;
        }
        case T_BF16: {
            const uint16_t *w = row;
#if RUNNER_AVX2
            return dot_bf16_avx2(w, x, n);
#endif
            float s = 0;
            for (int i = 0; i < n; i++) s += bf16_to_f32(w[i]) * x[i];
            return s;
        }
        case T_Q8_0: {
#if RUNNER_AVX2
            return dot_q8_0_avx2(row, x, n);
#endif
            const block_q8_0 *b = row;
            float s = 0;
            for (int i = 0; i < n / QK; i++) {
                const int8_t *q = b[i].qs;
                const float *xp = x + i * QK;
                float t = 0;
                for (int j = 0; j < QK; j++) t += q[j] * xp[j];
                s += f16_to_f32(b[i].d) * t;
            }
            return s;
        }
        case T_Q4_0: {
#if RUNNER_AVX2
            return dot_q4_0_avx2(row, x, n);
#elif RUNNER_NEON
            return dot_q4_0_neon(row, x, n);
#endif
            const block_q4_0 *b = row;
            float s = 0;
            for (int i = 0; i < n / QK; i++) {
                const uint8_t *q = b[i].qs;
                const float *xp = x + i * QK;
                float t = 0;
                for (int j = 0; j < 16; j++)
                    t += ((q[j] & 0xF) - 8) * xp[j] + ((q[j] >> 4) - 8) * xp[j + 16];
                s += f16_to_f32(b[i].d) * t;
            }
            return s;
        }
        case T_Q4_K: {
#if RUNNER_AVX2
            return dot_q4_K_avx2(row, x, n);
#endif
            const block_q4_K *b = row;
            float s = 0;
            for (int i = 0; i < n / QK_K; i++, b++) {
                float d = f16_to_f32(b->d), dmin = f16_to_f32(b->dmin);
                const uint8_t *q = b->qs;
                const float *xp = x + i * QK_K;
                int is = 0;
                for (int j = 0; j < QK_K; j += 64) {
                    uint8_t sc, mn;
                    get_scale_min_k4(is + 0, b->scales, &sc, &mn);
                    float d1 = d * sc, m1 = dmin * mn;
                    get_scale_min_k4(is + 1, b->scales, &sc, &mn);
                    float d2 = d * sc, m2 = dmin * mn;
                    float t1 = 0, t2 = 0, sx1 = 0, sx2 = 0;
                    for (int l = 0; l < 32; l++) {
                        t1 += (q[l] & 0xF) * xp[l];      sx1 += xp[l];
                        t2 += (q[l] >> 4)  * xp[l + 32]; sx2 += xp[l + 32];
                    }
                    s += d1 * t1 - m1 * sx1 + d2 * t2 - m2 * sx2;
                    q += 32; is += 2; xp += 64;
                }
            }
            return s;
        }
#if RUNNER_AVX2
        case T_Q5_K:
            return dot_q5_K_avx2(row, x, n);
        case T_IQ4_NL:
            return dot_iq4_nl_avx2(row, x, n);
        case T_IQ4_XS:
            return dot_iq4_xs_avx2(row, x, n);
#elif RUNNER_NEON
        case T_Q5_K:
            return dot_q5_K_neon(row, x, n);
        case T_IQ4_NL:
            return dot_iq4_nl_neon(row, x, n);
        case T_IQ4_XS:
            return dot_iq4_xs_neon(row, x, n);
#endif
        case T_MXFP4: {
#if RUNNER_AVX2
            return dot_mxfp4_avx2(row, x, n);
#elif RUNNER_NEON
            return dot_mxfp4_neon(row, x, n);
#endif
            const block_mxfp4 *b = row;
            float s = 0;
            for (int i = 0; i < n / QK; i++) {
                const uint8_t *q = b[i].qs;
                const float *xp = x + i * QK;
                float t = 0;
                for (int j = 0; j < 16; j++)
                    t += kvalues_mxfp4[q[j] & 0xF] * xp[j]
                       + kvalues_mxfp4[q[j] >> 4]  * xp[j + 16];
                s += ldexpf(1.0f, (int)b[i].e - 127) * t;
            }
            return s;
        }
        case T_Q6_K: {
#if RUNNER_AVX2
            return dot_q6_K_avx2(row, x, n);
#elif RUNNER_NEON
            return dot_q6_K_neon(row, x, n);
#endif
            const block_q6_K *b = row;
            float s = 0;
            for (int i = 0; i < n / QK_K; i++, b++) {
                float d = f16_to_f32(b->d);
                const uint8_t *ql = b->ql, *qh = b->qh;
                const int8_t *sc = b->scales;
                const float *xp = x + i * QK_K;
                for (int half = 0; half < 2; half++) {
                    float t[8] = {0};
                    for (int l = 0; l < 32; l++) {
                        int is = (l / 16) & 1;
                        int q1 = (int)((ql[l]      & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                        int q2 = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                        int q3 = (int)((ql[l]      >> 4)  | (((qh[l] >> 4) & 3) << 4)) - 32;
                        int q4 = (int)((ql[l + 32] >> 4)  | (((qh[l] >> 6) & 3) << 4)) - 32;
                        t[is * 4 + 0] += q1 * xp[l];
                        t[is * 4 + 1] += q2 * xp[l + 32];
                        t[is * 4 + 2] += q3 * xp[l + 64];
                        t[is * 4 + 3] += q4 * xp[l + 96];
                    }
                    s += d * (sc[0] * t[0] + sc[2] * t[1] + sc[4] * t[2] + sc[6] * t[3] +
                              sc[1] * t[4] + sc[3] * t[5] + sc[5] * t[6] + sc[7] * t[7]);
                    ql += 64; qh += 32; sc += 8; xp += 128;
                }
            }
            return s;
        }
        default: {
            // generic: dequantize block by block
            int bs = ggml_block_size(type);
            size_t ts = ggml_type_size(type);
            const uint8_t *p = row;
            float buf[QK_K];
            float s = 0;
            for (int i = 0; i < n; i += bs, p += ts) {
                dequant_block(type, p, buf);
                for (int j = 0; j < bs; j++) s += buf[j] * x[i + j];
            }
            return s;
        }
    }
}

// ---------------------------------------------------------------- threadpool

#define TP_MAX 64

struct tpool {
    pthread_t th[TP_MAX];
    int n_threads;          // total workers incl. calling thread
    pthread_mutex_t mu;
    pthread_cond_t cv_work, cv_done;
    tp_fn fn;
    void *ctx;
    int n_items;
    uint64_t gen;
    int n_done;
    bool stop;
};

typedef struct { tpool *tp; int idx; } tp_arg;

static void tp_slice(int idx, int n_threads, int n_items, int *i0, int *i1) {
    int per = (n_items + n_threads - 1) / n_threads;
    *i0 = idx * per;
    *i1 = *i0 + per;
    if (*i0 > n_items) *i0 = n_items;
    if (*i1 > n_items) *i1 = n_items;
}

static void *tp_worker(void *argp) {
    tp_arg *a = argp;
    tpool *tp = a->tp;
    int idx = a->idx; // 1..n_threads-1 (0 is the calling thread)
    free(a);
    uint64_t seen = 0;
    for (;;) {
        pthread_mutex_lock(&tp->mu);
        while (tp->gen == seen && !tp->stop) pthread_cond_wait(&tp->cv_work, &tp->mu);
        if (tp->stop) { pthread_mutex_unlock(&tp->mu); return NULL; }
        seen = tp->gen;
        tp_fn fn = tp->fn; void *ctx = tp->ctx; int n = tp->n_items;
        pthread_mutex_unlock(&tp->mu);

        int i0, i1;
        tp_slice(idx, tp->n_threads, n, &i0, &i1);
        if (i0 < i1) fn(ctx, i0, i1);

        pthread_mutex_lock(&tp->mu);
        tp->n_done++;
        if (tp->n_done == tp->n_threads - 1) pthread_cond_signal(&tp->cv_done);
        pthread_mutex_unlock(&tp->mu);
    }
}

int tpool_size(const tpool *tp) { return tp ? tp->n_threads : 0; }

tpool *tpool_create(int n_threads) {
    if (n_threads < 1) n_threads = 1;
    // Say so. Silently discarding an explicit -t value is the wrong failure
    // mode: the caller asked for N, got TP_MAX, and had no way to find out.
    // Reported by the Syntetik-MoE run, which lost time to `-t 128` behaving
    // exactly like `-t 64`. Once per process — this is called per instance and
    // a --parallel server would otherwise repeat it per slot.
    if (n_threads > TP_MAX) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            fprintf(stderr, "warning: %d threads requested, using %d "
                            "(compile-time TP_MAX)\n", n_threads, TP_MAX);
        }
        n_threads = TP_MAX;
    }
    tpool *tp = calloc(1, sizeof(tpool));
    if (!tp) return NULL;
    tp->n_threads = n_threads;
    if (pthread_mutex_init(&tp->mu, NULL) != 0) { free(tp); return NULL; }
    if (pthread_cond_init(&tp->cv_work, NULL) != 0) {
        pthread_mutex_destroy(&tp->mu);
        free(tp);
        return NULL;
    }
    if (pthread_cond_init(&tp->cv_done, NULL) != 0) {
        pthread_cond_destroy(&tp->cv_work);
        pthread_mutex_destroy(&tp->mu);
        free(tp);
        return NULL;
    }
    int created = 0;
    for (int i = 1; i < n_threads; i++) {
        tp_arg *a = malloc(sizeof(tp_arg));
        if (!a) goto fail;
        a->tp = tp; a->idx = i;
        if (pthread_create(&tp->th[i], NULL, tp_worker, a) != 0) {
            free(a);
            goto fail;
        }
        created++;
    }
    return tp;

fail:
    pthread_mutex_lock(&tp->mu);
    tp->stop = true;
    pthread_cond_broadcast(&tp->cv_work);
    pthread_mutex_unlock(&tp->mu);
    for (int i = 1; i <= created; i++) pthread_join(tp->th[i], NULL);
    pthread_cond_destroy(&tp->cv_done);
    pthread_cond_destroy(&tp->cv_work);
    pthread_mutex_destroy(&tp->mu);
    free(tp);
    return NULL;
}

void tpool_destroy(tpool *tp) {
    if (!tp) return;
    pthread_mutex_lock(&tp->mu);
    tp->stop = true;
    pthread_cond_broadcast(&tp->cv_work);
    pthread_mutex_unlock(&tp->mu);
    for (int i = 1; i < tp->n_threads; i++) pthread_join(tp->th[i], NULL);
    pthread_mutex_destroy(&tp->mu);
    pthread_cond_destroy(&tp->cv_work);
    pthread_cond_destroy(&tp->cv_done);
    free(tp);
}

void tpool_run(tpool *tp, tp_fn fn, void *ctx, int n_items) {
    if (n_items <= 0) return;
    if (!tp || tp->n_threads <= 1 || n_items < 2) {
        fn(ctx, 0, n_items);
        return;
    }
    pthread_mutex_lock(&tp->mu);
    tp->fn = fn; tp->ctx = ctx; tp->n_items = n_items;
    tp->n_done = 0;
    tp->gen++;
    pthread_cond_broadcast(&tp->cv_work);
    pthread_mutex_unlock(&tp->mu);

    int i0, i1;
    tp_slice(0, tp->n_threads, n_items, &i0, &i1);
    if (i0 < i1) fn(ctx, i0, i1);

    pthread_mutex_lock(&tp->mu);
    while (tp->n_done < tp->n_threads - 1) pthread_cond_wait(&tp->cv_done, &tp->mu);
    pthread_mutex_unlock(&tp->mu);
}
