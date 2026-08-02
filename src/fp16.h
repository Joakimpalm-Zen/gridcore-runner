// fp16/bf16 conversion, shared by every backend.
#ifndef RUNNER_FP16_H
#define RUNNER_FP16_H

#include <stdint.h>
#include <string.h>

typedef uint16_t f16_t;
extern float g_f16_table[65536];
void f16_init(void);

static inline float f16_to_f32(f16_t h) { return g_f16_table[h]; }

static inline float bf16_to_f32(uint16_t h) {
    union { uint32_t u; float f; } v;
    v.u = (uint32_t)h << 16;
    return v.f;
}

#if defined(__ARM_FP16_FORMAT_IEEE)
static inline f16_t f32_to_f16(float f) {
    __fp16 h = (__fp16)f;
    f16_t r;
    __builtin_memcpy(&r, &h, 2);
    return r;
}
static inline float f16_load(const f16_t *p) { return (float)*(const __fp16 *)p; }
#else
static inline f16_t f32_to_f16(float f) {
    union { float fl; uint32_t u; } v = { .fl = f };
    uint32_t sign = (v.u >> 16) & 0x8000;
    int32_t  exp  = (int32_t)((v.u >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = v.u & 0x7FFFFF;
    if (exp >= 31) return (f16_t)(sign | 0x7C00);           // inf/overflow
    if (exp <= 0) {                                          // subnormal/zero
        if (exp < -10) return (f16_t)sign;
        mant |= 0x800000;
        uint32_t shift = 14 - exp;
        uint32_t h = mant >> shift;
        if ((mant >> (shift - 1)) & 1) h++;                  // round
        return (f16_t)(sign | h);
    }
    uint32_t h = (uint32_t)(exp << 10) | (mant >> 13);
    if ((mant >> 12) & 1) h++;                               // round to nearest
    return (f16_t)(sign | h);
}
static inline float f16_load(const f16_t *p) { return g_f16_table[*p]; }
#endif

#endif // RUNNER_FP16_H
