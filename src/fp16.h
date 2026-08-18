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

// Portable f32 -> f16. Round to nearest, TIES TO EVEN, and NaN stays NaN:
// the same result the aarch64 `fcvt` below produces, bit for bit, for all 2^32
// inputs, and the same rule ggml/llama.cpp use — a block scale or an f16 KV
// entry must not depend on which ISA the binary was built for. Integer-only on
// purpose: the float-add rounding trick would be miscompiled under -ffast-math,
// which most of this engine is built with.
// Exposed (rather than left as the #else branch) so the gate can check the
// portable path on an aarch64 host, where it is otherwise unreachable.
static inline f16_t f32_to_f16_soft(float f) {
    union { float fl; uint32_t u; } v = { .fl = f };
    uint32_t sign = (v.u >> 16) & 0x8000;
    int32_t  exp  = (int32_t)((v.u >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = v.u & 0x7FFFFF;
    if (exp >= 31) {                                         // inf/nan/overflow
        if (exp == 143 && mant)                              // quiet the NaN,
            return (f16_t)(sign | 0x7C00 | (mant >> 13) | 0x0200);  // keep payload
        return (f16_t)(sign | 0x7C00);
    }
    if (exp <= 0) {                                          // subnormal/zero
        if (exp < -10) return (f16_t)sign;
        mant |= 0x800000;
        uint32_t shift = (uint32_t)(14 - exp);               // 14..24
        uint32_t h    = mant >> shift;
        uint32_t rem  = mant & ((1u << shift) - 1);
        uint32_t half = 1u << (shift - 1);
        if (rem > half || (rem == half && (h & 1))) h++;
        return (f16_t)(sign | h);
    }
    uint32_t h   = (uint32_t)(exp << 10) | (mant >> 13);
    uint32_t rem = mant & 0x1FFF;
    if (rem > 0x1000 || (rem == 0x1000 && (h & 1))) h++;     // carries into exp
    return (f16_t)(sign | h);
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
static inline f16_t f32_to_f16(float f) { return f32_to_f16_soft(f); }
static inline float f16_load(const f16_t *p) { return g_f16_table[*p]; }
#endif

#endif // RUNNER_FP16_H
