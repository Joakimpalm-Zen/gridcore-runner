// GGUF tensor element types and the quantized dot products.
#ifndef RUNNER_QUANTS_H
#define RUNNER_QUANTS_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

// tensor data types (subset of ggml)
enum ggml_type {
    T_F32 = 0, T_F16 = 1, T_Q4_0 = 2, T_Q4_1 = 3,
    T_Q5_0 = 6, T_Q5_1 = 7, T_Q8_0 = 8,
    T_Q2_K = 10, T_Q3_K = 11, T_Q4_K = 12, T_Q5_K = 13, T_Q6_K = 14,
    // codebook i-quants (sub-4-bit; dequant transcribed from llama.cpp
    // b10353 ggml-quants.c, grids in quants_iq_grids.h). CPU only: the
    // device backends refuse these types and fall back loudly.
    T_IQ2_XXS = 16, T_IQ2_XS = 17, T_IQ3_XXS = 18, T_IQ1_S = 19,
    T_IQ4_NL = 20,
    T_IQ3_S = 21, T_IQ2_S = 22,
    T_IQ4_XS = 23,
    T_IQ1_M = 29,
    T_BF16 = 30,
    T_MXFP4 = 39,   // OCP microscaling FP4 (E2M1 codes + per-block E8M0 scale); gpt-oss
};
int         ggml_block_size(int type);   // elements per block
size_t      ggml_type_size(int type);    // bytes per block
const char *ggml_type_name(int type);
bool        ggml_type_supported(int type);
static inline size_t ggml_row_size(int type, int64_t n) {
    return (size_t)(n / ggml_block_size(type)) * ggml_type_size(type);
}

// dequantize a full row of n elements
void  dequant_row(int type, const void *src, float *dst, int n);
// target == T_KEEP: every tensor keeps its own on-disk type (--prune-experts
// with no --quant — geometry changes, precision doesn't).
#define T_KEEP (-1)
// Requantize a GGUF at in_path to out_path (written beside + atomically
// renamed), optionally pruning MoE experts per prune_path first (NULL =
// no pruning; see quantize.c's prune_plan_load for the LIST.json shape).
// Returns 0 on success, nonzero on failure with the destination left
// untouched. Declared here so tests can drive it without main.c.
int   quantize_gguf(const char *in_path, const char *out_path, int target,
                    const char *prune_path);
// dot(row, x) over n elements
float vec_dot(int type, const void *row, const float *x, int n);
// out[b] = dot(w, x + b*x_stride) for nb columns sharing one weight row
void  vec_dot_f32_multi(const float *w, const float *x, int x_stride,
                        int nb, int n, float *out);
void  q8_quant_row(const float *x, void *dst, int n); // n % 32 == 0
void  q8_accum_row(const void *src, float a, float *out, int n);

// ------------------------------------------------ fused int8 dot (CPU lever 1)
//
// The scalar/SIMD vec_dot above keeps f32 activations and converts each weight
// quant to f32 before the FMA. The fused route instead quantizes the ACTIVATION
// row to int8 once per matvec and does the whole dot in int8 with an int32
// accumulator (AVX-512 VNNI `_mm512_dpbusd_epi32`, AVX2 `maddubs` fallback).
// Quantizing the activations means it can never be bit-identical to vec_dot;
// it is a tolerance-gated fast route in the same sense as the CUDA tensor-core
// prefill path. It is OFF by default — measured 2026-08-13, no (format, model)
// combo cleared the 0/64 teacher-forced flip bar with a decode gain worth
// taking; `RUNNER_CPU_I8=1` opts in. The promotion record is in quants.c and
// the gate is test-i8-tol.
//
//   if (i8_dot_ok(type, n)) { i8_quant_act(x, buf, n);        // once per matvec
//                             for each row: vec_dot_i8(type, row, buf, n); }
//
bool   i8_dot_enabled(void);              // RUNNER_CPU_I8 pin, cached
bool   i8_dot_ok(int type, int n);        // fused kernel exists for (type, n)
size_t i8_act_size(int n);                // scratch bytes for n activations
void   i8_quant_act(const float *x, void *dst, int n);
float  vec_dot_i8(int type, const void *row, const void *xq, int n);
// Force the route on (1) or off (0) regardless of RUNNER_CPU_I8; -1 returns
// to the env default. Mirrors gpu_tc_force so one harness can drive both.
void   i8_dot_force(int on);
// Matvecs that took the fused route since process start. A tolerance gate
// that sees zero is comparing the scalar path with itself.
unsigned long i8_dot_dispatches(void);

#endif // RUNNER_QUANTS_H
