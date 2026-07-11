// Quantization block formats (ggml-compatible), dot kernels, threadpool.
#include "runner.h"

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

static const int8_t kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
};

int ggml_block_size(int type) {
    switch (type) {
        case T_F32: case T_F16: case T_BF16: return 1;
        case T_Q4_0: case T_Q4_1: case T_Q5_0: case T_Q5_1: case T_Q8_0:
        case T_IQ4_NL: return QK;
        case T_Q4_K: case T_Q5_K: case T_Q6_K: case T_Q2_K: case T_Q3_K:
        case T_IQ4_XS: return QK_K;
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
        default: return "?";
    }
}

bool ggml_type_supported(int type) {
    switch (type) {
        case T_F32: case T_F16: case T_BF16:
        case T_Q4_0: case T_Q4_1: case T_Q5_0: case T_Q5_1: case T_Q8_0:
        case T_Q2_K: case T_Q3_K: case T_Q4_K: case T_Q5_K: case T_Q6_K:
        case T_IQ4_NL: case T_IQ4_XS:
            return true;
        default:
            return false;
    }
}

// ---------------------------------------------------------------- dequant

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

static void dequant_block(int type, const void *src, float *dst) {
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

// ---------------------------------------------------------------- vec_dot

// AVX2 kernels for the hot quant formats. Same math as the scalar code —
// only the accumulation order differs (8-lane FMA), so results can drift in
// the last ulps but stay well inside quantization noise. Scalar code below
// remains the fallback for every other platform (ARM macs, older x86).
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
#endif // RUNNER_AVX2

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
#endif
        case T_Q6_K: {
#if RUNNER_AVX2
            return dot_q6_K_avx2(row, x, n);
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

tpool *tpool_create(int n_threads) {
    if (n_threads < 1) n_threads = 1;
    if (n_threads > TP_MAX) n_threads = TP_MAX;
    tpool *tp = calloc(1, sizeof(tpool));
    tp->n_threads = n_threads;
    pthread_mutex_init(&tp->mu, NULL);
    pthread_cond_init(&tp->cv_work, NULL);
    pthread_cond_init(&tp->cv_done, NULL);
    for (int i = 1; i < n_threads; i++) {
        tp_arg *a = malloc(sizeof(tp_arg));
        a->tp = tp; a->idx = i;
        pthread_create(&tp->th[i], NULL, tp_worker, a);
    }
    return tp;
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
    if (i0 < i1) fn(ctx, 0 < 1 ? i0 : i0, i1);

    pthread_mutex_lock(&tp->mu);
    while (tp->n_done < tp->n_threads - 1) pthread_cond_wait(&tp->cv_done, &tp->mu);
    pthread_mutex_unlock(&tp->mu);
}
