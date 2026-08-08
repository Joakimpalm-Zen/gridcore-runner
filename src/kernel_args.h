// Kernel argument structs, shared by the HOST and the DEVICE.
//
// These are passed to CUDA kernels BY VALUE, so gcc (compiling src/cuda.c) and
// nvcc (compiling src/kernels.cu to PTX) must agree on their layout exactly.
// They used to be declared twice — once in each file — with nothing checking
// that the two stayed in step.
//
// That is a silent failure: adding a field to one side compiles cleanly on
// both, and the kernel then reads its arguments at the wrong offsets. It
// produces fluent, confident, WRONG output rather than a crash — on 2026-08-08
// exactly that happened while adding a field to attn_args, and the GPU emitted
// "The capital of France isnas!!!!!!!!" while the CPU path stayed correct.
// Without a CPU arm to compare against, that reads like a broken model file.
//
// One definition removes the failure mode entirely.
//
// Keep this header free of anything CUDA-specific: it is compiled as C by gcc
// and as C++ by nvcc, so plain structs and fixed-width integers only.
#ifndef RUNNER_KERNEL_ARGS_H
#define RUNNER_KERNEL_ARGS_H

// `unsigned long long` rather than uint64_t: nvcc compiles kernels.cu without
// the host's <stdint.h> in scope, and the two spellings are the same 8-byte
// type with the same alignment on every platform this builds for.
typedef unsigned long long ka_u64;

typedef struct {
    int    n_in;
    int    n_out;
    ka_u64 w_off;    // tensor byte offset inside the weight buffer
    int    has_bias;
    int    batch;    // 1..MVB token columns (k_mv_* ignores it)
    int    xs, ys;   // element stride between x / y columns
} mv_args;

typedef struct {
    int    n_in;
    int    n_out;
    ka_u64 w_off;      // fused expert tensor base byte offset
    ka_u64 estride;    // bytes per expert block within the fused tensor
    int    xs, ys;     // x / y element stride between expert slots
} moe_args;

typedef struct {
    int   head_dim, n_heads, half_dim, neox;
    float mscale;
} rope_args;

typedef struct {
    int    head_dim, n_head, n_head_kv, n_ctx;
    ka_u64 l_off;    // this layer's BYTE offset into the kv cache
    float  scale;
    int    qs, os;   // q / out element stride per token column
    int    window;   // sliding-window size for this layer (0 = full)
    int    q8;       // cache rows are q8_0 blocks rather than fp16
} attn_args;

#endif // RUNNER_KERNEL_ARGS_H
