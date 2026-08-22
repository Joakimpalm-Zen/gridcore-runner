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
// ------------------------------------------------ IEEE subnormals for quant
// ggml computes quant block scales with IEEE subnormals (it is built without
// -ffast-math). The runner engine links crtfastmath.o (-ffast-math), which
// sets FTZ/DAZ in MXCSR process-wide; a subnormal scale d = amax/127 then
// flushes to zero and that block's codes come out zero. Clearing FTZ/DAZ for
// the offline quantize makes runner's Q8_0/q4_0 bytes match ggml's exactly.
#if defined(__x86_64__) || defined(__i386__)
#include <xmmintrin.h>
typedef unsigned fp_denormal_state;
static inline fp_denormal_state fp_denormals_disable(void) {
    unsigned csr = _mm_getcsr();
    _mm_setcsr(csr & ~0x8040u); // clear FTZ (bit 15) and DAZ (bit 6)
    return csr;
}
static inline void fp_denormals_restore(fp_denormal_state s) { _mm_setcsr(s); }
#elif defined(__aarch64__)
typedef unsigned long fp_denormal_state;
static inline fp_denormal_state fp_denormals_disable(void) {
    unsigned long fpcr; __asm__ volatile("mrs %0, fpcr" : "=r"(fpcr));
    __asm__ volatile("msr fpcr, %0" :: "r"(fpcr & ~(1UL << 24))); // clear FZ
    return fpcr;
}
static inline void fp_denormals_restore(fp_denormal_state s) { __asm__ volatile("msr fpcr, %0" :: "r"(s)); }
#else
typedef int fp_denormal_state;
static inline fp_denormal_state fp_denormals_disable(void) { return 0; }
static inline void fp_denormals_restore(fp_denormal_state s) { (void)s; }
#endif


typedef struct { f16_t d; uint8_t qs[16]; } wblock_q4_0;
typedef struct { f16_t d; int8_t qs[32]; }  wblock_q8_0;
typedef struct {
    uint8_t hmask[32], qs[64], scales[12];
    f16_t d;
} wblock_q3_K;
typedef struct {
    f16_t d, dmin;
    uint8_t scales[12], qs[128];
} wblock_q4_K;
typedef struct {
    uint8_t ql[128], qh[64];
    int8_t scales[16];
    f16_t d;
} wblock_q6_K;

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

// QAT grid-exact repack.
//
// A quantization-aware-trained checkpoint's weights already sit on the q4_0
// grid: every value is d*(q-8) for one per-block scale d and integer q in
// [0,15]. Requantizing such a block should be LOSSLESS, because the answer is
// already there to be read off.
//
// The derived-scale path below cannot read it off. `d = vmax / -8` recovers
// the true scale only when the block actually contains q == 0; a QAT block
// whose codes never reach 0 gets a scale that is too small, and the values at
// the far end of the range then saturate at q == 15 and come back wrong. On a
// 64-block fixture with codes 1..15 that is 1770 of 2048 values changed, worst
// error 0.4375 -- on a repack that should have been exact.
//
// So try to RECOVER the grid first. The smallest nonzero magnitude in the
// block is m*d for some integer m in [1,8], which gives eight candidate
// scales; take the largest that works, since a coarser grid that still
// reproduces every value is equally exact and leaves the codes better centred.
//
// The acceptance test is the whole safety argument: a candidate is accepted
// only if the value the DEQUANTIZER will compute -- f16_to_f32 of the stored
// scale, times the code -- equals the source float bit for bit, for all 32
// values. That folds the fp16 storage of the scale into the same check (a
// bf16-scaled source is exactly representable, since bf16's 7 mantissa bits
// fit fp16's 10, but only inside fp16's narrower exponent range), and it means
// imprecision anywhere can only cause a REJECTION, never a wrong write. On any
// rejection the derived-scale path runs unchanged, so a non-QAT model gets
// byte-identical output to before.
static bool q4_0_grid_exact(const float *xp, wblock_q4_0 *b) {
    float amin = 0;
    bool any = false;
    for (int j = 0; j < 32; j++) {
        float a = fabsf(xp[j]);
        if (a != 0.0f && (!any || a < amin)) { amin = a; any = true; }
    }
    if (!any) return false;      // all-zero block: the derived path is already exact

    for (int m = 1; m <= 8; m++) {
        f16_t dh = f32_to_f16(amin / (float)m);
        float ds = f16_to_f32(dh);
        if (ds == 0.0f) continue;            // underflowed fp16, or no f16 table
        uint8_t q[32];
        bool ok = true;
        for (int j = 0; j < 32; j++) {
            float kf = roundf(xp[j] / ds);
            if (kf < -8.0f || kf > 7.0f) { ok = false; break; }
            int k = (int)kf;
            if (ds * (float)k != xp[j]) { ok = false; break; }
            q[j] = (uint8_t)(k + 8);
        }
        if (!ok) continue;
        b->d = dh;
        for (int j = 0; j < 16; j++) b->qs[j] = (uint8_t)(q[j] | (q[j + 16] << 4));
        return true;
    }
    return false;
}

static void quantize_row_q4_0(const float *x, uint8_t *dst, int n) {
    wblock_q4_0 *b = (wblock_q4_0 *)dst;
    for (int i = 0; i < n / 32; i++, b++) {
        const float *xp = x + i * 32;
        if (q4_0_grid_exact(xp, b)) continue;
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

static int nearest_int(float v) {
    // Exact ggml reference rounding for K-quants (nearest, ties-to-even for
    // the default IEEE environment), avoiding a libm implementation choice.
    float biased = v + 12582912.0f;
    int bits;
    memcpy(&bits, &biased, sizeof(bits));
    return (bits & 0x007fffff) - 0x00400000;
}

static float make_q3_quants(const float *x, int8_t *q) {
    float vmax = 0.0f, amax = 0.0f;
    for (int i = 0; i < 16; i++) {
        float a = fabsf(x[i]);
        if (a > amax) { amax = a; vmax = x[i]; }
    }
    if (amax < 1e-15f) {
        memset(q, 0, 16);
        return 0.0f;
    }
    float iscale = -4.0f / vmax;
    float sumlx = 0.0f, suml2 = 0.0f;
    for (int i = 0; i < 16; i++) {
        int l = nearest_int(iscale * x[i]);
        if (l < -4) l = -4;
        if (l > 3) l = 3;
        q[i] = (int8_t)l;
        float w = x[i] * x[i];
        sumlx += w * x[i] * l;
        suml2 += w * l * l;
    }
    for (int itry = 0; itry < 5; itry++) {
        int changed = 0;
        for (int i = 0; i < 16; i++) {
            float w = x[i] * x[i];
            float slx = sumlx - w * x[i] * q[i];
            if (slx <= 0.0f) continue;
            float sl2 = suml2 - w * q[i] * q[i];
            int l = nearest_int(x[i] * sl2 / slx);
            if (l < -4) l = -4;
            if (l > 3) l = 3;
            if (l == q[i]) continue;
            slx += w * x[i] * l;
            sl2 += w * l * l;
            if (sl2 > 0.0f && slx * slx * suml2 > sumlx * sumlx * sl2) {
                q[i] = (int8_t)l;
                sumlx = slx;
                suml2 = sl2;
                changed++;
            }
        }
        if (!changed) break;
    }
    for (int i = 0; i < 16; i++) q[i] += 4;
    return suml2 > 0.0f ? sumlx / suml2 : 0.0f;
}

// GGML Q3_K's 256-value super-block: sixteen signed 3-bit groups, with the
// group scales themselves quantized to signed 6-bit values.  The layout is
// deliberately the inverse of dq_q3_K in quants.c; the hermetic writer test
// drives that existing reader, so a layout disagreement cannot pass.
static void quantize_row_q3_K(const float *x, uint8_t *dst, int n) {
    wblock_q3_K *out = (wblock_q3_K *)dst;
    for (int ib = 0; ib < n / 256; ib++, out++, x += 256) {
        int8_t q[256];
        float scales[16];
        float max_scale = 0.0f, amax = 0.0f;
        for (int j = 0; j < 16; j++) {
            scales[j] = make_q3_quants(x + 16 * j, q + 16 * j);
            float a = fabsf(scales[j]);
            if (a > amax) { amax = a; max_scale = scales[j]; }
        }
        memset(out, 0, sizeof(*out));
        if (max_scale != 0.0f) {
            float iscale = -32.0f / max_scale;
            for (int j = 0; j < 16; j++) {
                int l = nearest_int(iscale * scales[j]);
                if (l < -32) l = -32;
                if (l > 31) l = 31;
                l += 32;
                if (j < 8) out->scales[j] = (uint8_t)(l & 15);
                else out->scales[j - 8] |= (uint8_t)((l & 15) << 4);
                out->scales[j % 4 + 8] |= (uint8_t)((l >> 4) << (2 * (j / 4)));
            }
            out->d = f32_to_f16(1.0f / iscale);
        }
        for (int j = 0; j < 16; j++) {
            int sc = j < 8 ? out->scales[j] & 15 : out->scales[j - 8] >> 4;
            sc = (sc | (((out->scales[8 + j % 4] >> (2 * (j / 4))) & 3) << 4)) - 32;
            float d = f16_to_f32(out->d) * sc;
            if (d == 0.0f) continue;
            for (int k = 0; k < 16; k++) {
                int l = nearest_int(x[16 * j + k] / d);
                if (l < -4) l = -4;
                if (l > 3) l = 3;
                q[16 * j + k] = (int8_t)(l + 4);
            }
        }
        int m = 0;
        uint8_t hm = 1;
        for (int j = 0; j < 256; j++) {
            if (q[j] > 3) { out->hmask[m] |= hm; q[j] -= 4; }
            if (++m == 32) { m = 0; hm <<= 1; }
        }
        for (int j = 0; j < 256; j += 128)
            for (int k = 0; k < 32; k++)
                out->qs[j / 4 + k] = (uint8_t)(q[j + k] | (q[j + k + 32] << 2) |
                    (q[j + k + 64] << 4) | (q[j + k + 96] << 6));
    }
}

// ggml's make_qx_quants specialized to the one call Q6_K makes of it
// (x^2 error weighting, the full ±9 candidate-scale search). Faithful port:
// a K-quant tensor this file writes must carry the bytes llama.cpp's own
// quantizer would emit for the same floats.
static float make_q6_quants(int n, int nmax, const float *x, int8_t *L) {
    float vmax = 0.0f, amax = 0.0f;
    for (int i = 0; i < n; i++) {
        float ax = fabsf(x[i]);
        if (ax > amax) { amax = ax; vmax = x[i]; }
    }
    if (amax < 1e-15f) {
        memset(L, 0, (size_t)n);
        return 0.0f;
    }
    float iscale = -nmax / vmax;
    float sumlx = 0.0f, suml2 = 0.0f;
    for (int i = 0; i < n; i++) {
        int l = nearest_int(iscale * x[i]);
        if (l < -nmax) l = -nmax;
        if (l > nmax - 1) l = nmax - 1;
        L[i] = (int8_t)(l + nmax);
        float w = x[i] * x[i];
        sumlx += w * x[i] * l;
        suml2 += w * l * l;
    }
    float scale = suml2 > 0.0f ? sumlx / suml2 : 0.0f;
    float best = scale * sumlx;
    for (int is = -9; is <= 9; is++) {
        if (is == 0) continue;
        iscale = -(nmax + 0.1f * is) / vmax;
        sumlx = suml2 = 0.0f;
        for (int i = 0; i < n; i++) {
            int l = nearest_int(iscale * x[i]);
            if (l < -nmax) l = -nmax;
            if (l > nmax - 1) l = nmax - 1;
            float w = x[i] * x[i];
            sumlx += w * x[i] * l;
            suml2 += w * l * l;
        }
        if (suml2 > 0.0f && sumlx * sumlx > best * suml2) {
            for (int i = 0; i < n; i++) {
                int l = nearest_int(iscale * x[i]);
                if (l < -nmax) l = -nmax;
                if (l > nmax - 1) l = nmax - 1;
                L[i] = (int8_t)(l + nmax);
            }
            scale = sumlx / suml2;
            best = scale * sumlx;
        }
    }
    return scale;
}

// GGML Q6_K: sixteen 16-value groups of signed 6-bit codes, group scales
// quantized to signed 8-bit against one f16 super-scale. Inverse of
// dq_q6_K in quants.c; the hermetic round-trip test drives that reader.
static void quantize_row_q6_K(const float *x, uint8_t *dst, int n) {
    wblock_q6_K *out = (wblock_q6_K *)dst;
    for (int ib = 0; ib < n / 256; ib++, out++, x += 256) {
        int8_t L[256];
        float scales[16];
        float max_scale = 0.0f, max_abs = 0.0f;
        for (int j = 0; j < 16; j++) {
            scales[j] = make_q6_quants(16, 32, x + 16 * j, L + 16 * j);
            float a = fabsf(scales[j]);
            if (a > max_abs) { max_abs = a; max_scale = scales[j]; }
        }
        memset(out, 0, sizeof(*out));
        if (max_abs < 1e-15f) continue;      // all-zero block, d stays 0
        float iscale = -128.0f / max_scale;
        out->d = f32_to_f16(1.0f / iscale);
        for (int j = 0; j < 16; j++) {
            int l = nearest_int(iscale * scales[j]);
            out->scales[j] = (int8_t)(l < 127 ? l : 127);
        }
        for (int j = 0; j < 16; j++) {
            // a group whose quantized scale rounded to 0 keeps its
            // make_q6_quants codes (they decode to 0 either way) — the
            // reference does the same, and byte fidelity to it matters
            float d = f16_to_f32(out->d) * (float)out->scales[j];
            if (d == 0.0f) continue;
            for (int k = 0; k < 16; k++) {
                int l = nearest_int(x[16 * j + k] / d);
                if (l < -32) l = -32;
                if (l > 31) l = 31;
                L[16 * j + k] = (int8_t)(l + 32);
            }
        }
        uint8_t *ql = out->ql, *qh = out->qh;
        for (int j = 0; j < 256; j += 128) {
            for (int k = 0; k < 32; k++) {
                uint8_t q1 = L[j + k]      & 0xF;
                uint8_t q2 = L[j + k + 32] & 0xF;
                uint8_t q3 = L[j + k + 64] & 0xF;
                uint8_t q4 = L[j + k + 96] & 0xF;
                ql[k]      = (uint8_t)(q1 | (q3 << 4));
                ql[k + 32] = (uint8_t)(q2 | (q4 << 4));
                qh[k] = (uint8_t)((L[j + k] >> 4) |
                                  ((L[j + k + 32] >> 4) << 2) |
                                  ((L[j + k + 64] >> 4) << 4) |
                                  ((L[j + k + 96] >> 4) << 6));
            }
            ql += 64; qh += 32;
        }
    }
}

// ggml's make_qkx2_quants with Q4_K's fixed search parameters (rmin -1,
// rdelta 0.1, 20 steps, squared-error weighting).
static float make_qkx2_quants(int n, int nmax, const float *x,
                              const float *weights, uint8_t *L,
                              float *the_min, uint8_t *Laux) {
    float min = x[0], max = x[0];
    float sum_w = weights[0], sum_x = weights[0] * x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] < min) min = x[i];
        if (x[i] > max) max = x[i];
        sum_w += weights[i];
        sum_x += weights[i] * x[i];
    }
    if (min > 0.0f) min = 0.0f;
    if (max == min) {
        memset(L, 0, (size_t)n);
        *the_min = -min;
        return 0.0f;
    }
    float iscale = nmax / (max - min);
    float scale = 1.0f / iscale;
    float best_mad = 0.0f;
    for (int i = 0; i < n; i++) {
        int l = nearest_int(iscale * (x[i] - min));
        if (l < 0) l = 0;
        if (l > nmax) l = nmax;
        L[i] = (uint8_t)l;
        float diff = scale * l + min - x[i];
        best_mad += weights[i] * diff * diff;
    }
    for (int is = 0; is <= 20; is++) {
        iscale = (-1.0f + 0.1f * is + nmax) / (max - min);
        float sum_l = 0.0f, sum_l2 = 0.0f, sum_xl = 0.0f;
        for (int i = 0; i < n; i++) {
            int l = nearest_int(iscale * (x[i] - min));
            if (l < 0) l = 0;
            if (l > nmax) l = nmax;
            Laux[i] = (uint8_t)l;
            sum_l  += weights[i] * l;
            sum_l2 += weights[i] * l * l;
            sum_xl += weights[i] * l * x[i];
        }
        float D = sum_w * sum_l2 - sum_l * sum_l;
        if (D <= 0.0f) continue;
        float this_scale = (sum_w * sum_xl - sum_x * sum_l) / D;
        float this_min = (sum_l2 * sum_x - sum_l * sum_xl) / D;
        if (this_min > 0.0f) {
            this_min = 0.0f;
            this_scale = sum_xl / sum_l2;   // D>0 guarantees sum_l2>0
        }
        float mad = 0.0f;
        for (int i = 0; i < n; i++) {
            float diff = this_scale * Laux[i] + this_min - x[i];
            mad += weights[i] * diff * diff;
        }
        if (mad < best_mad) {
            memcpy(L, Laux, (size_t)n);
            best_mad = mad;
            scale = this_scale;
            min = this_min;
        }
    }
    *the_min = -min;
    return scale;
}

// mirror of quants.c get_scale_min_k4 — the reader this writer must invert
static void scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4)  | ((q[j    ] >> 6) << 4);
    }
}

// GGML Q4_K: eight 32-value groups of unsigned 4-bit codes with per-group
// scale+min pairs packed to 6 bits against f16 d/dmin super-scales.
// Inverse of dq_q4_K in quants.c.
static void quantize_row_q4_K(const float *x, uint8_t *dst, int n) {
    wblock_q4_K *out = (wblock_q4_K *)dst;
    for (int ib = 0; ib < n / 256; ib++, out++, x += 256) {
        uint8_t L[256], Laux[32];
        float weights[32], mins[8], scales[8];
        float max_scale = 0.0f, max_min = 0.0f;
        for (int j = 0; j < 8; j++) {
            float sum_x2 = 0.0f;
            for (int l = 0; l < 32; l++)
                sum_x2 += x[32 * j + l] * x[32 * j + l];
            float av_x = sqrtf(sum_x2 / 32.0f);
            for (int l = 0; l < 32; l++)
                weights[l] = av_x + fabsf(x[32 * j + l]);
            scales[j] = make_qkx2_quants(32, 15, x + 32 * j, weights,
                                         L + 32 * j, &mins[j], Laux);
            if (scales[j] > max_scale) max_scale = scales[j];
            if (mins[j] > max_min) max_min = mins[j];
        }
        float inv_scale = max_scale > 0.0f ? 63.0f / max_scale : 0.0f;
        float inv_min = max_min > 0.0f ? 63.0f / max_min : 0.0f;
        memset(out, 0, sizeof(*out));
        for (int j = 0; j < 8; j++) {
            uint8_t ls = (uint8_t)nearest_int(inv_scale * scales[j]);
            uint8_t lm = (uint8_t)nearest_int(inv_min * mins[j]);
            if (ls > 63) ls = 63;
            if (lm > 63) lm = 63;
            if (j < 4) {
                out->scales[j] = ls;
                out->scales[j + 4] = lm;
            } else {
                out->scales[j + 4] = (uint8_t)((ls & 0xF) | ((lm & 0xF) << 4));
                out->scales[j - 4] |= (uint8_t)((ls >> 4) << 6);
                out->scales[j]     |= (uint8_t)((lm >> 4) << 6);
            }
        }
        out->d = f32_to_f16(max_scale / 63.0f);
        out->dmin = f32_to_f16(max_min / 63.0f);
        for (int j = 0; j < 8; j++) {
            uint8_t sc, mn;
            scale_min_k4(j, out->scales, &sc, &mn);
            float d = f16_to_f32(out->d) * sc;
            if (d == 0.0f) continue;
            float dm = f16_to_f32(out->dmin) * mn;
            for (int l = 0; l < 32; l++) {
                int q = nearest_int((x[32 * j + l] + dm) / d);
                if (q < 0) q = 0;
                if (q > 15) q = 15;
                L[32 * j + l] = (uint8_t)q;
            }
        }
        uint8_t *qs = out->qs;
        for (int j = 0; j < 256; j += 64) {
            for (int l = 0; l < 32; l++)
                qs[l] = (uint8_t)(L[j + l] | (L[j + l + 32] << 4));
            qs += 32;
        }
    }
}

static void quantize_row_f16(const float *x, uint8_t *dst, int n) {
    f16_t *h = (f16_t *)dst;
    for (int i = 0; i < n; i++) h[i] = f32_to_f16(x[i]);
}

// ggml_compute_fp32_to_bf16 semantics: round to nearest even, NaN forced
// quiet — a truncation here would not match what a bf16 GGUF built by the
// reference converter carries.
static void quantize_row_bf16(const float *x, uint8_t *dst, int n) {
    uint16_t *h = (uint16_t *)dst;
    for (int i = 0; i < n; i++) {
        uint32_t u;
        memcpy(&u, &x[i], 4);
        if ((u & 0x7fffffff) > 0x7f800000)
            h[i] = (uint16_t)((u >> 16) | 64);
        else
            h[i] = (uint16_t)((u + (0x7fffu + ((u >> 16) & 1))) >> 16);
    }
}

// One row, f32 -> `type`. Everything this file can write goes through here;
// type_from_name, the whole-file target validator and quantize_row_writable
// must stay in sync with this switch.
static void quantize_row_any(int type, const float *rowf, uint8_t *rowq, int n) {
    switch (type) {
        case T_Q8_0: quantize_row_q8_0(rowf, rowq, n); break;
        case T_Q4_0: quantize_row_q4_0(rowf, rowq, n); break;
        case T_Q3_K: quantize_row_q3_K(rowf, rowq, n); break;
        case T_Q4_K: quantize_row_q4_K(rowf, rowq, n); break;
        case T_Q6_K: quantize_row_q6_K(rowf, rowq, n); break;
        case T_F16:  quantize_row_f16(rowf, rowq, n); break;
        case T_BF16: quantize_row_bf16(rowf, rowq, n); break;
        default:     memcpy(rowq, rowf, sizeof(float) * (size_t)n); break;
    }
}

static bool quantize_row_writable(int type) {
    return type == T_F32 || type == T_F16 || type == T_BF16 ||
           type == T_Q8_0 || type == T_Q4_0 || type == T_Q3_K ||
           type == T_Q4_K || type == T_Q6_K;
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

// ------------------------------------------------------- selective precision
//
// A --type-plan is a per-TENSOR type override. Per-tensor is the finest
// granularity GGUF can express, and that limit is the whole story of the
// hot/cold expert experiment this was written for: MoE experts are stored
// STACKED, one 3-D tensor per layer holding every expert
// (blk.N.ffn_down_exps.weight is (768, 2048, 128) on Qwen3-30B-A3B), and a
// tensor carries exactly one type. Giving hot and cold experts different
// precisions would mean splitting that tensor into 128, which no loader
// expects and which would stop the file being a GGUF anyone can run. So
// per-expert precision is not a thing this format does, and the lever that IS
// available is per tensor class: keep attention and embeddings high, move the
// expert bulk down.
//
//   {"default": "keep",
//    "rules": [{"match": "_exps.weight", "type": "q4_0"}]}
//
// Rules are tried in order and the first substring match wins; "default"
// applies to everything unmatched. Types are the ones this quantizer can
// actually write (q8_0, q4_0, q3_k, q4_k, q6_k, f16, bf16, keep) — naming
// one it cannot is an error at load, not a silent fallback.
typedef struct { char *match; int type; } type_rule_t;
typedef struct { type_rule_t *rules; int n; int dflt; } type_plan_t;

static int type_from_name(const char *s) {
    if (!s) return -2;
    if (!strcmp(s, "keep")) return T_KEEP;
    if (!strcmp(s, "q8_0")) return T_Q8_0;
    if (!strcmp(s, "q4_0")) return T_Q4_0;
    if (!strcmp(s, "q3_k")) return T_Q3_K;
    if (!strcmp(s, "q4_k")) return T_Q4_K;
    if (!strcmp(s, "q6_k")) return T_Q6_K;
    if (!strcmp(s, "f16"))  return T_F16;
    if (!strcmp(s, "bf16")) return T_BF16;
    return -2;
}

static void type_plan_free(type_plan_t *p) {
    for (int i = 0; i < p->n; i++) free(p->rules[i].match);
    free(p->rules);
    memset(p, 0, sizeof(*p));
}

static bool type_plan_load(const char *path, type_plan_t *plan) {
    memset(plan, 0, sizeof(*plan));
    plan->dflt = T_KEEP;
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "error: cannot open --type-plan file %s\n", path); return false; }
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return false; }
    long sz = ftell(f);
    if (sz < 0 || sz > 8 * 1024 * 1024 || fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        fprintf(stderr, "error: %s is not a readable type plan\n", path);
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
    bool ok = true;
    jv *d = jv_get(root, "default");
    if (d && d->type == J_STR) {
        plan->dflt = type_from_name(jv_str(d, NULL));
        if (plan->dflt == -2) {
            fprintf(stderr, "error: type-plan default \"%s\" is not one of "
                    "keep|q8_0|q4_0|q3_k|f16\n", jv_str(d, "?"));
            ok = false;
        }
    }
    jv *rules = jv_get(root, "rules");
    if (ok && rules && rules->type == J_ARR && rules->n > 0) {
        plan->rules = calloc((size_t)rules->n, sizeof(type_rule_t));
        if (!plan->rules) { jv_free(root); return false; }
        for (int i = 0; ok && i < rules->n; i++) {
            jv *r = rules->items[i];
            jv *m = r && r->type == J_OBJ ? jv_get(r, "match") : NULL;
            jv *t = r && r->type == J_OBJ ? jv_get(r, "type") : NULL;
            if (!m || m->type != J_STR || !t || t->type != J_STR) {
                fprintf(stderr, "error: type-plan rule %d needs string "
                        "\"match\" and \"type\"\n", i);
                ok = false;
                break;
            }
            int ty = type_from_name(jv_str(t, NULL));
            if (ty == -2) {
                fprintf(stderr, "error: type-plan rule %d type \"%s\" is not "
                        "one of keep|q8_0|q4_0|q3_k|f16\n", i, jv_str(t, "?"));
                ok = false;
                break;
            }
            plan->rules[plan->n].match = strdup(jv_str(m, ""));
            if (!plan->rules[plan->n].match) { ok = false; break; }
            plan->rules[plan->n].type = ty;
            plan->n++;
        }
    }
    jv_free(root);
    if (!ok) type_plan_free(plan);
    return ok;
}

// first matching rule wins; T_KEEP means "leave this tensor exactly as it is"
static int type_plan_pick(const type_plan_t *plan, const char *name) {
    for (int i = 0; i < plan->n; i++)
        if (strstr(name, plan->rules[i].match)) return plan->rules[i].type;
    return plan->dflt;
}

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
        long long parsed_layer;
        const char *key = root->keys[i];
        if (strncmp(key, "layer_", 6) != 0 ||
            key[6] < '0' || key[6] > '9' ||
            !parse_i64(key + 6, 0, INT_MAX, &parsed_layer)) {
            fprintf(stderr, "error: prune-plan key \"%s\" is not \"layer_N\"\n", root->keys[i]);
            ok = false;
            break;
        }
        int layer = (int)parsed_layer;
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
            double id = arr->items[j]->type == J_NUM
                      ? arr->items[j]->num : -1.0;
            if (!isfinite(id) || id < 0 || id > INT_MAX || trunc(id) != id) {
                fprintf(stderr, "error: prune-plan[\"layer_%d\"][%d] is not "
                        "a non-negative integer\n", layer, j);
                free(ids);
                ok = false;
                break;
            }
            ids[j] = (int)id;
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
        // Both on-disk spellings of the selection bias: model.c reads
        // `exp_probs_b.weight` OR `exp_probs_b.bias` with the layer's expert
        // count, and nemotron_h_moe ships the `.bias` one. Slicing only
        // `.weight` left a full-length bias beside a pruned router, which the
        // loader then (correctly) refuses.
        "exp_probs_b.weight", "exp_probs_b.bias",
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

// ---------------------------------------------------------------- lora merge
// --merge-lora: fold a trained adapter into the base weights, producing a
// standalone GGUF (W' = W + (alpha/r)*B·A per adapted projection). The
// merged file runs anywhere a GGUF runs — but merging into a quantized
// type ROUNDS THE DELTA through that type's grid, so base + --lora stays
// the exact form; a merged artifact's fidelity is a measurement, not a
// given. Provenance (.merge.json, written by the caller) carries the
// base/adapter shas that make the merged blob auditable.

typedef struct {
    const float *a, *b;   // mmap'd F32 pair: a is r rows of n_in, b is
                          // n_out rows of r — the orientation lora_apply
                          // consumes, so the merged weight computes
                          // W'x = Wx + scale*B(Ax) up to summation order
    int r;                // 0 = this base tensor is untouched
    float scale;          // (alpha / r) * user scale
    bool zero;            // delta is exactly zero (b all-zero or scale 0):
                          // copy the base tensor verbatim instead of
                          // churning it through dequant->requant
} merge_pair_t;

typedef struct {
    gguf_file g;          // adapter file; stays open while rows stream
    bool open;
    merge_pair_t *map;    // one entry per BASE tensor index
    uint64_t n_base;      // base tensor count the map was resolved against
    int n_pairs, n_zero;
} merge_set_t;

// mirror of model.c lora_slot_name — the projections --lora hooks. The
// merge accepts exactly what serving accepts, so a mergeable adapter and a
// servable adapter are the same set.
static const char *const merge_slot_name[] = {
    "attn_q", "attn_k", "attn_v", "attn_output",
    "ffn_gate", "ffn_up", "ffn_down",
};

static void merge_set_free(merge_set_t *ms) {
    if (ms->open) gguf_close(&ms->g);
    free(ms->map);
    memset(ms, 0, sizeof(*ms));
}

// model_lora_load's validation, run against the base FILE instead of a
// loaded model: every adapter tensor must name a hooked projection that
// exists in the base with matching geometry, or the whole merge refuses —
// a silently skipped tensor would emit a merged model that is not
// base+adapter.
static bool merge_set_resolve(merge_set_t *ms, const char *adapter_path,
                              float user_scale, gguf_file *base) {
    memset(ms, 0, sizeof(*ms));
    if (!gguf_open(&ms->g, adapter_path)) {
        fprintf(stderr, "error: cannot open adapter %s\n", adapter_path);
        return false;
    }
    ms->open = true;
    const char *ftype = gguf_get_str(&ms->g, "general.type", "");
    const char *atype = gguf_get_str(&ms->g, "adapter.type", "");
    if (strcmp(ftype, "adapter") != 0 || strcmp(atype, "lora") != 0) {
        fprintf(stderr, "error: %s is not a LoRA adapter GGUF "
                "(general.type=%s adapter.type=%s)\n", adapter_path, ftype,
                atype);
        return false;
    }
    const char *aarch = gguf_get_str(&ms->g, "general.architecture", "");
    const char *barch = gguf_get_str(base, "general.architecture", "");
    if (strcmp(aarch, barch) != 0) {
        fprintf(stderr, "error: adapter architecture '%s' does not match "
                "the model's '%s'\n", aarch, barch);
        return false;
    }
    float alpha = gguf_get_f32(&ms->g, "adapter.lora.alpha", 0.0f);
    if (!(alpha > 0.0f)) {
        fprintf(stderr, "error: adapter.lora.alpha missing or not "
                "positive\n");
        return false;
    }
    ms->n_base = base->n_tensors;
    ms->map = calloc(base->n_tensors ? base->n_tensors : 1,
                     sizeof(merge_pair_t));
    if (!ms->map) return false;
    for (uint64_t ti = 0; ti < ms->g.n_tensors; ti++) {
        const gguf_tensor *t = &ms->g.tensors[ti];
        int l = -1;
        char pname[64];
        char side = 0;
        if (sscanf(t->name, "blk.%d.%40[a-z_].weight.lora_%c", &l, pname,
                   &side) != 3 || (side != 'a' && side != 'b') || l < 0) {
            fprintf(stderr, "error: adapter tensor '%s' does not name a "
                    "hooked projection\n", t->name);
            return false;
        }
        bool hooked = false;
        for (size_t k = 0; k < sizeof(merge_slot_name) /
                               sizeof(*merge_slot_name); k++)
            if (strcmp(pname, merge_slot_name[k]) == 0) { hooked = true; break; }
        char bname[96];
        snprintf(bname, sizeof(bname), "blk.%d.%s.weight", l, pname);
        const gguf_tensor *bt = NULL;
        uint64_t bi = 0;
        for (uint64_t j = 0; hooked && j < base->n_tensors; j++)
            if (strcmp(base->tensors[j].name, bname) == 0) {
                bt = &base->tensors[j];
                bi = j;
                break;
            }
        if (!bt || bt->n_dims != 2) {
            fprintf(stderr, "error: adapter tensor '%s' targets a "
                    "projection this model does not carry\n", t->name);
            return false;
        }
        if (t->type != T_F32) {
            fprintf(stderr, "error: adapter tensor '%s' is not F32\n",
                    t->name);
            return false;
        }
        int64_t n_in = bt->ne[0], n_out = bt->ne[1];
        int64_t r = side == 'a' ? t->ne[1] : t->ne[0];
        int64_t want0 = side == 'a' ? n_in : r;
        int64_t want1 = side == 'a' ? r : n_out;
        // 512 = LORA_R_MAX, model.c — the merge accepts what --lora serves
        if (r < 1 || r > 512 || (int64_t)t->ne[0] != want0 ||
            (int64_t)t->ne[1] != want1) {
            fprintf(stderr, "error: adapter tensor '%s' is [%lld,%lld]; "
                    "base '%s' is [%lld,%lld]\n", t->name,
                    (long long)t->ne[0], (long long)t->ne[1], bt->name,
                    (long long)n_in, (long long)n_out);
            return false;
        }
        merge_pair_t *mp = &ms->map[bi];
        if (mp->r && mp->r != (int)r) {
            fprintf(stderr, "error: adapter '%s' rank %lld disagrees with "
                    "its pair (rank %d)\n", t->name, (long long)r, mp->r);
            return false;
        }
        mp->r = (int)r;
        if (side == 'a') mp->a = (const float *)t->data;
        else             mp->b = (const float *)t->data;
    }
    for (uint64_t i = 0; i < base->n_tensors; i++) {
        merge_pair_t *mp = &ms->map[i];
        if (!mp->r) continue;
        if (!mp->a || !mp->b) {
            fprintf(stderr, "error: adapter has only half of the "
                    "lora_a/lora_b pair for %s\n", base->tensors[i].name);
            return false;
        }
        mp->scale = alpha / (float)mp->r * user_scale;
        int64_t n_out = base->tensors[i].ne[1];
        mp->zero = mp->scale == 0.0f;
        if (!mp->zero) {
            mp->zero = true;
            const uint32_t *bw = (const uint32_t *)mp->b;
            for (int64_t c = 0; c < (int64_t)mp->r * n_out; c++)
                if (bw[c] != 0) { mp->zero = false; break; }
        }
        ms->n_pairs++;
        if (mp->zero) ms->n_zero++;
    }
    if (!ms->n_pairs) {
        fprintf(stderr, "error: adapter carries no lora_a/lora_b pairs\n");
        return false;
    }
    return true;
}

// Fold the low-rank delta into one dequantized row: as a weight update,
// row j of scale*B(Ax) is rowf[i] += scale * sum_k B[j][k]*A[k][i].
// Fixed-order fmaf chain — the merged bytes must be reproducible across
// runs and hosts, the same discipline as lora_apply.
static void merge_row(const merge_pair_t *mp, uint64_t j, float *rowf,
                      int n_in) {
    const float *br = mp->b + (size_t)j * mp->r;
    for (int i = 0; i < n_in; i++) {
        float acc = 0.0f;
        for (int k = 0; k < mp->r; k++)
            acc = fmaf(br[k], mp->a[(size_t)k * n_in + i], acc);
        rowf[i] = fmaf(mp->scale, acc, rowf[i]);
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

// Tensors that leave a requant with their on-disk type untouched, whatever
// the target or the plan says. The MoE router picks WHICH expert runs: an
// error there swaps a whole FFN instead of perturbing one output, and it is
// the one weight matrix whose cost of staying exact is negligible
// (n_embd x n_expert — 0.3% of a 30B-A3B checkpoint). That is why llama.cpp
// excludes ffn_gate_inp from quantization and why every reference MoE GGUF
// ships it at F32 beside 4-bit experts. `ffn_gate_inp_shexp` is the same
// decision for the shared expert, so the substring covers both.
static bool keep_source_type(const gguf_tensor *t) {
    return strstr(t->name, "ffn_gate_inp") != NULL;
}

// A quantized row must divide into a whole number of blocks. should_quantize
// only clears ne[0] % 32, which is the block width of q8_0/q4_0 but not of the
// 256-wide K-quants a --type-plan can name: writing a 288-wide row as Q3_K
// emits one block (256 values, 32 silently dropped) under a header that still
// says 288, and gguf.c refuses ne[0] % block_size != 0 at open — so the requant
// would report success and produce a file nothing can load.
static bool type_fits_row(int type, int64_t n) {
    return n % ggml_block_size(type) == 0;
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

static void quantize_plans_free(prune_plan_t *prune, type_plan_t *types) {
    prune_plan_free(prune);
    type_plan_free(types);
}

int quantize_gguf(const char *in_path, const char *out_path, int target,
                  const char *prune_path) {
    return quantize_gguf_plan(in_path, out_path, target, prune_path, NULL);
}

int quantize_type_from_name(const char *s) {
    return type_from_name(s);
}

static int quantize_gguf_plan_inner(const char *in_path, const char *out_path, int target,
                       const char *prune_path, const char *type_plan_path,
                       merge_set_t *ms);

// FTZ/DAZ are process-wide (set by the engine's -ffast-math startup); clear
// them around the quantize so a subnormal block scale matches ggml, and
// restore on exit so no later caller sees perturbed denormal handling.
int quantize_gguf_plan(const char *in_path, const char *out_path, int target,
                       const char *prune_path, const char *type_plan_path) {
    fp_denormal_state fpst = fp_denormals_disable();
    int rc = quantize_gguf_plan_inner(in_path, out_path, target, prune_path,
                                      type_plan_path, NULL);
    fp_denormals_restore(fpst);
    return rc;
}

// The adapter is resolved against a first open of the base, which then
// closes; the inner writer re-opens the same path itself. Tensor indices
// are stable across the two opens of one file, and the adapter's mmap (the
// a/b pointers in the map) stays alive in `ms` for the whole write.
int merge_lora_gguf(const char *in_path, const char *adapter_path,
                    float user_scale, const char *out_path, int target) {
    gguf_file base;
    if (!gguf_open(&base, in_path)) return 1;
    merge_set_t ms;
    bool ok = merge_set_resolve(&ms, adapter_path, user_scale, &base);
    gguf_close(&base);
    if (!ok) {
        merge_set_free(&ms);
        return 1;
    }
    fp_denormal_state fpst = fp_denormals_disable();
    int rc = quantize_gguf_plan_inner(in_path, out_path, target, NULL, NULL,
                                      &ms);
    fp_denormals_restore(fpst);
    merge_set_free(&ms);
    return rc;
}

static int quantize_gguf_plan_inner(const char *in_path, const char *out_path, int target,
                       const char *prune_path, const char *type_plan_path,
                       merge_set_t *ms) {
    if (target != T_KEEP && !quantize_row_writable(target)) {
        fprintf(stderr, "error: quantize target must be q8_0, q4_0, q3_k, "
                "q4_k, q6_k, f16, or bf16\n");
        return 1;
    }
    prune_plan_t plan = {0};
    if (prune_path && !prune_plan_load(prune_path, &plan)) return 1;
    type_plan_t tplan;
    memset(&tplan, 0, sizeof(tplan));
    if (type_plan_path && !type_plan_load(type_plan_path, &tplan)) {
        quantize_plans_free(&plan, &tplan);
        return 1;
    }

    gguf_file g;
    if (!gguf_open(&g, in_path)) {
        quantize_plans_free(&plan, &tplan);
        return 1;
    }
    for (uint64_t i = 0; i < g.n_tensors; i++) {
        if (!ggml_type_supported(g.tensors[i].type)) {
            fprintf(stderr, "error: tensor %s has unsupported type %s\n",
                    g.tensors[i].name, ggml_type_name(g.tensors[i].type));
            gguf_close(&g);
            quantize_plans_free(&plan, &tplan);
            return 1;
        }
    }
    if (ms && ms->n_base != g.n_tensors) {
        // the base changed between the resolve open and this one
        fprintf(stderr, "error: %s changed while merging\n", in_path);
        gguf_close(&g);
        quantize_plans_free(&plan, &tplan);
        return 1;
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
                gguf_close(&g);
                quantize_plans_free(&plan, &tplan);
                return 1;
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
                        gguf_close(&g);
                        quantize_plans_free(&plan, &tplan);
                        return 1;
                    }
                // Every tensor resolve_prune() will slice must describe the
                // same expert axis as this router. Validating ids against the
                // router alone is insufficient: a shorter bias/bank would
                // make the copy loop index beyond that tensor's declared
                // data and produce an artifact that cannot load.
                for (uint64_t j = 0; j < g.n_tensors; j++) {
                    int tensor_layer; const char *tensor_suffix;
                    gguf_tensor *et = &g.tensors[j];
                    if (!parse_blk_name(et->name, &tensor_layer,
                                        &tensor_suffix) ||
                        tensor_layer != layer ||
                        !is_expert_tensor_suffix(tensor_suffix))
                        continue;
                    int64_t tensor_n = et->ne[et->n_dims - 1];
                    if (tensor_n != orig_n) {
                        fprintf(stderr, "error: %s expert axis has %lld "
                                "entries, but the blk.%d router has %lld\n",
                                et->name, (long long)tensor_n, layer,
                                (long long)orig_n);
                        gguf_close(&g);
                        quantize_plans_free(&plan, &tplan);
                        return 1;
                    }
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
            gguf_close(&g);
            quantize_plans_free(&plan, &tplan);
            return 1;
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
        quantize_plans_free(&plan, &tplan);
        return 1;
    }
    uint64_t amask = align - 1;

    // Write beside the destination and rename on success, so a failed or
    // interrupted requant never destroys an existing good model — and an
    // in-place requant (in_path == out_path) no longer truncates its own
    // input. (RNR-015.)
    size_t tlen = strlen(out_path) + sizeof(".partial");
    char *tmp_path = malloc(tlen);
    if (!tmp_path) {
        gguf_close(&g);
        quantize_plans_free(&plan, &tplan);
        return 1;
    }
    snprintf(tmp_path, tlen, "%s.partial", out_path);

    FILE *f = fopen(tmp_path, "wb");
    if (!f) {
        fprintf(stderr, "error: cannot write %s\n", tmp_path);
        free(tmp_path);
        gguf_close(&g);
        quantize_plans_free(&plan, &tplan);
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

    // Output types are decided BEFORE the header goes out, because
    // general.file_type must describe the OUTPUT: copying the parent's
    // declaration shipped a mixed Q4_0/Q4_K artifact labeled by its parent
    // (finding B of the 2026-08-11 external evaluation — the same
    // mixed-tensor trap this project's own cert matrix fails third parties
    // for). The declared value is computed from the histogram of what is
    // actually written, and the histogram itself is printed so a mixed
    // result is visible in the log, not just in a header field.
    int *out_type = malloc(sizeof(int) * g.n_tensors);
    if (g.n_tensors > 0 && !out_type) w.ok = false;
    uint64_t declined_width = 0;
    bool merge_unwritable = false;
    for (uint64_t i = 0; w.ok && i < g.n_tensors; i++) {
        gguf_tensor *t = &g.tensors[i];
        if (type_plan_path) {
            // The plan decides per tensor. T_KEEP leaves the tensor byte for
            // byte; anything else goes through the same should_quantize,
            // block-width and never-grow guards as a whole-file target, so a
            // plan cannot make a tensor bigger, quantize something that must
            // stay f32, or write a row the target type cannot describe.
            int want = type_plan_pick(&tplan, t->name);
            if (keep_source_type(t) || want == T_KEEP) {
                // "keep" means exactly that, for every tensor. Falling through
                // to the whole-file rule below pushed a non-f16 tensor (a bf16
                // norm) to F32 — a converted, grown tensor in a plan whose
                // whole instruction was to convert nothing.
                out_type[i] = t->type;
            } else if (!should_quantize(t)) {
                out_type[i] = t->type == T_F16 ? T_F16 : T_F32;
            } else if (!type_fits_row(want, t->ne[0])) {
                out_type[i] = t->type;
                declined_width++;
            } else if (ggml_row_size(t->type, t->ne[0]) <=
                       ggml_row_size(want, t->ne[0])) {
                out_type[i] = t->type;   // never grow
            } else {
                out_type[i] = want;
            }
        } else if (target == T_KEEP) {
            out_type[i] = t->type;
        } else {
            const char *only = ms ? NULL : getenv("RUNNER_REQUANT_ONLY");
            bool filtered = keep_source_type(t) ||
                            (only && *only && !strstr(t->name, only));
            if (filtered)
                out_type[i] = t->type;
            else if (should_quantize(t) && !type_fits_row(target, t->ne[0])) {
                out_type[i] = t->type;
                filtered = true;          // and skip the never-grow rule below
                declined_width++;
            } else
                out_type[i] = should_quantize(t) ? target
                            : (t->type == T_F16 ? T_F16 : T_F32);
            // never grow a tensor that is already smaller than the target —
            // RUNNER_FORCE_REQUANT overrides this to build same-size
            // conversions (e.g. Q4_K -> Q4_0) for diagnosing per-quant-type
            // correctness bugs. A merge to a wider type is exempt too: asking
            // for a merged f16 from a 4-bit base is a preservation request,
            // not an accident.
            if (!filtered && !ms && should_quantize(t) && !getenv("RUNNER_FORCE_REQUANT") &&
                ggml_row_size(t->type, t->ne[0]) <= ggml_row_size(target, t->ne[0]))
                out_type[i] = t->type;
        }
        // an adapted tensor cannot keep a type this file has no quantizer
        // for — refuse before writing a byte rather than skip the delta
        if (ms && ms->map[i].r && !ms->map[i].zero &&
            !quantize_row_writable(out_type[i])) {
            fprintf(stderr, "error: cannot merge into %s — no %s "
                    "quantizer; pass --quant to pick a writable output "
                    "type\n", t->name, ggml_type_name(out_type[i]));
            merge_unwritable = true;
        }
    }
    if (merge_unwritable) {
        free(out_type);
        fclose(f);
        remove(tmp_path);
        free(tmp_path);
        gguf_close(&g);
        quantize_plans_free(&plan, &tplan);
        return 1;
    }
    if (w.ok && declined_width)
        fprintf(stderr, "quantize: %llu tensor(s) kept their own type — the "
                "requested type's block does not divide their row width\n",
                (unsigned long long)declined_width);
    uint32_t ftype_out = 0;
    if (w.ok && g.n_tensors > 0) {
        // dominant non-f32 output type -> its MOSTLY_* code. The K-quant
        // S/M/L distinction is not recoverable from a one-type histogram;
        // the M code is the closest honest label and the printed histogram
        // carries the exact truth.
        static const struct { int t; uint32_t ftype; const char *name; } FT[] = {
            { T_F16, 1, "F16" }, { T_BF16, 32, "BF16" },
            { T_Q4_0, 2, "Q4_0" }, { T_Q4_1, 3, "Q4_1" },
            { T_Q5_0, 8, "Q5_0" }, { T_Q5_1, 9, "Q5_1" },
            { T_Q8_0, 7, "Q8_0" },
            { T_Q2_K, 10, "Q2_K" }, { T_Q3_K, 12, "Q3_K" },
            { T_Q4_K, 15, "Q4_K" }, { T_Q5_K, 17, "Q5_K" },
            { T_Q6_K, 18, "Q6_K" },
            { T_IQ2_XXS, 19, "IQ2_XXS" }, { T_IQ2_XS, 20, "IQ2_XS" },
            { T_IQ2_S, 28, "IQ2_S" }, { T_IQ3_XXS, 23, "IQ3_XXS" },
            { T_IQ3_S, 26, "IQ3_S" }, { T_IQ1_S, 24, "IQ1_S" },
            { T_IQ1_M, 31, "IQ1_M" }, { T_IQ4_NL, 25, "IQ4_NL" },
            { T_IQ4_XS, 30, "IQ4_XS" }, { T_MXFP4, 38, "MXFP4" },
        };
        uint64_t counts[sizeof(FT) / sizeof(*FT)] = {0};
        uint64_t f32s = 0;
        for (uint64_t i = 0; i < g.n_tensors; i++) {
            if (out_type[i] == T_F32) { f32s++; continue; }
            for (size_t k = 0; k < sizeof(FT) / sizeof(*FT); k++)
                if (FT[k].t == out_type[i]) { counts[k]++; break; }
        }
        size_t dom = 0;
        uint64_t best = 0;
        for (size_t k = 0; k < sizeof(FT) / sizeof(*FT); k++)
            if (counts[k] > best) { best = counts[k]; dom = k; }
        ftype_out = best ? FT[dom].ftype : 0;   // all-f32 -> ALL_F32
        fprintf(stderr, "quantize: general.file_type %u (MOSTLY_%s) — output histogram:",
                ftype_out, best ? FT[dom].name : "F32");
        if (f32s) fprintf(stderr, " F32:%llu", (unsigned long long)f32s);
        for (size_t k = 0; k < sizeof(FT) / sizeof(*FT); k++)
            if (counts[k]) fprintf(stderr, " %s:%llu", FT[k].name,
                                   (unsigned long long)counts[k]);
        fprintf(stderr, "\n");
    }
    bool had_ftype = gguf_get(&g, "general.file_type") != NULL;

    wr_u32(&w, 0x46554747);
    wr_u32(&w, 3);
    wr_u64(&w, g.n_tensors);
    wr_u64(&w, g.n_kv - (had_ftype ? 1 : 0) + 1);
    for (uint64_t i = 0; i < g.n_kv; i++) {
        if (!strcmp(g.kv[i].key, "general.file_type")) continue;
        wr_kv(&w, &g.kv[i]);
    }
    wr_str(&w, "general.file_type", strlen("general.file_type"));
    wr_u32(&w, 4);            // GGUF_T_U32
    wr_u32(&w, ftype_out);

    // tensor table with new types/offsets (data alignment `align`)
    uint64_t off = 0;
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
        // out_type[i] was decided in the pre-pass above, before the header
        // (general.file_type is derived from it)
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
                    quantize_row_any(out_type[i], rowf, rowq, (int)row_len);
                    wr(&w, rowq, out_rs);
                }
            }
            quantized++;
            continue;
        }

        // a zero-delta pair leaves its tensor on the verbatim path below —
        // copying the base bytes is exactly the merge result, without a
        // gratuitous dequant->requant of untouched weights
        const merge_pair_t *mp = ms && ms->map[i].r && !ms->map[i].zero
                                 ? &ms->map[i] : NULL;
        uint64_t rows = t->ne[1] * t->ne[2] * t->ne[3];
        size_t in_rs = ggml_row_size(t->type, t->ne[0]);
        size_t out_rs = ggml_row_size(out_type[i], t->ne[0]);
        if (!mp && (uint32_t)out_type[i] == t->type) {
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
            if (mp) merge_row(mp, r, rowf, (int)t->ne[0]);
            quantize_row_any(out_type[i], rowf, rowq, (int)t->ne[0]);
            wr(&w, rowq, out_rs);
        }
        quantized++;
    }
    bool write_ok = wr_close(&w);
    free(rowf); free(rowq); free(out_type); free(out_off); free(eff_ne);
    gguf_close(&g);
    quantize_plans_free(&plan, &tplan);

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
    if (ms)
        fprintf(stderr, "merge: %d adapted projection%s folded into the "
                "weights%s\n", ms->n_pairs - ms->n_zero,
                ms->n_pairs - ms->n_zero == 1 ? "" : "s",
                ms->n_zero ? " (zero-delta pairs copied verbatim)" : "");
    return 0;
}
