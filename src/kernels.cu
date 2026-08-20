// CUDA compute kernels: the full forward pass, one token or a prompt tile.
// 1:1 port of kernels.metal for the layout, with two additions for prompt
// tiles of up to MVB tokens:
//   - k_mv_*_b matvec variants decode each weight element once and FMA it
//     against every token column (weight-bandwidth reuse; generation keeps
//     the single-column k_mv_* variants, which are faster at batch 1)
//   - every small kernel takes the token index from blockIdx.y/z, so a tile
//     costs the same number of kernel launches as a single token — launch
//     overhead (severe under Windows WDDM) does not scale with tile size
// Compiled to PTX at development time (make ptx) and embedded via
// kernels_ptx.h; the driver JIT-compiles for the resident GPU.
#include <cuda_fp16.h>
#include <mma.h>

// mv_args / moe_args / rope_args / attn_args: ONE definition, shared with
// src/cuda.c. They are passed to kernels by value, so host and device must
// agree on the layout exactly (see the header).
#include "kernel_args.h"

typedef unsigned char  uchar;
typedef unsigned short ushort16;
typedef unsigned int   uint;
typedef unsigned long long ulong64;

static __device__ __forceinline__ float f16f(const uchar *p) {
    return __half2float(*(const __half *)p);
}

// ---------------------------------------------------------------- rmsnorm
// grid.y = token column; xs/ys = element stride between columns

extern "C" __global__ void k_rmsnorm(const float *x, float *y, const float *w,
                                     int n, float eps, int xs, int ys) {
    __shared__ float red[256];
    int tid = threadIdx.x, tpg = blockDim.x;
    x += (ulong64)blockIdx.y * xs;
    y += (ulong64)blockIdx.y * ys;
    float s = 0;
    for (int i = tid; i < n; i += tpg) s += x[i] * x[i];
    red[tid] = s;
    __syncthreads();
    for (int off = tpg / 2; off > 0; off >>= 1) {
        if (tid < off) red[tid] += red[tid + off];
        __syncthreads();
    }
    float r = rsqrtf(red[0] / n + eps);
    for (int i = tid; i < n; i += tpg) y[i] = x[i] * r * w[i];
}

// per-head RMSNorm (qwen3 Q/K norm): one block per (head, token)
// LAUNCH INVARIANT: the tree reduction below (off = tpg/2; off >>= 1) and the
// red[] indexing require blockDim.x to be a power of two and <= 128 (red[]'s
// size). Host launches this with 64 — do not exceed the buffer or break pow2.
extern "C" __global__ void k_qknorm(float *v, const float *w, int hd, float eps,
                                    int vs) {
    __shared__ float red[128];
    int tid = threadIdx.x, tpg = blockDim.x;
    float *x = v + (ulong64)blockIdx.y * vs + blockIdx.x * hd;
    float s = 0;
    for (int i = tid; i < hd; i += tpg) s += x[i] * x[i];
    red[tid] = s;
    __syncthreads();
    for (int off = tpg / 2; off > 0; off >>= 1) {
        if (tid < off) red[tid] += red[tid + off];
        __syncthreads();
    }
    float r = rsqrtf(red[0] / hd + eps);
    for (int i = tid; i < hd; i += tpg) x[i] = x[i] * r * w[i];
}

// ---------------------------------------------------------------- matvec
// One warp (32 lanes) per output row; lanes stride over blocks.
// 128 threads = 4 warps = 4 rows per block (same shape as the Metal version).
// k_mv_* handles one column (generation); k_mv_*_b applies each decoded
// weight to all MVB columns (x buffers are always MVB columns wide, so the
// unguarded reads for t >= batch touch valid, ignored memory).

#define MVT 16  // scalar-kernel tile: register arrays and xsm size
#define MVB 64  // activation BUFFER columns (keep in sync with cuda.c)


static __device__ __forceinline__ float warp_sum(float s) {
    for (int off = 16; off > 0; off >>= 1)
        s += __shfl_down_sync(0xffffffffu, s, off);
    return s;
}

#define MV_HEAD \
    unsigned row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5); \
    unsigned lane = threadIdx.x & 31; \
    if (row >= (unsigned)a.n_out) return;

#define MV_TAIL \
    s = warp_sum(s); \
    if (lane == 0) y[row] = a.has_bias ? s + bias[row] : s;

#define MV_HEAD_B \
    MV_HEAD; \
    float s[MVT] = {0};

// apply one decoded weight w at element index idx to every token column
#define MV_FMA(w, idx) do { \
    float _w = (w); ulong64 _i = (idx); \
    _Pragma("unroll") \
    for (int t = 0; t < MVT; t++) s[t] += _w * x[(ulong64)t * a.xs + _i]; \
} while (0)

#define MV_TAIL_B \
    for (int t = 0; t < a.batch; t++) { \
        float r = warp_sum(s[t]); \
        if (lane == 0) y[(ulong64)t * a.ys + row] = a.has_bias ? r + bias[row] : r; \
    }

#define MV_PARAMS const uchar *wb, const float *x, float *y, mv_args a, const float *bias

extern "C" __global__ void k_mv_f32(MV_PARAMS) {
    MV_HEAD;
    const float *rw = (const float *)(wb + a.w_off) + (ulong64)row * a.n_in;
    float s = 0;
    for (int i = lane; i < a.n_in; i += 32) s += rw[i] * x[i];
    MV_TAIL;
}

extern "C" __global__ void k_mv_f32_b(MV_PARAMS) {
    MV_HEAD_B;
    const float *rw = (const float *)(wb + a.w_off) + (ulong64)row * a.n_in;
    for (int i = lane; i < a.n_in; i += 32) MV_FMA(rw[i], i);
    MV_TAIL_B;
}

extern "C" __global__ void k_mv_f16(MV_PARAMS) {
    MV_HEAD;
    const __half *rw = (const __half *)(wb + a.w_off) + (ulong64)row * a.n_in;
    float s = 0;
    for (int i = lane; i < a.n_in; i += 32) s += __half2float(rw[i]) * x[i];
    MV_TAIL;
}

extern "C" __global__ void k_mv_f16_b(MV_PARAMS) {
    MV_HEAD_B;
    const __half *rw = (const __half *)(wb + a.w_off) + (ulong64)row * a.n_in;
    for (int i = lane; i < a.n_in; i += 32) MV_FMA(__half2float(rw[i]), i);
    MV_TAIL_B;
}

extern "C" __global__ void k_mv_q8_0(MV_PARAMS) {
    MV_HEAD;
    int nb = a.n_in / 32;
    const uchar *rw = wb + a.w_off + (ulong64)row * nb * 34;
    float s = 0;
    for (int b = lane; b < nb; b += 32) {
        const uchar *blk = rw + (ulong64)b * 34;
        float d = f16f(blk);
        const signed char *q = (const signed char *)(blk + 2);
        const float *xp = x + b * 32;
        float t = 0;
        for (int j = 0; j < 32; j++) t += (float)q[j] * xp[j];
        s += d * t;
    }
    MV_TAIL;
}

extern "C" __global__ void k_mv_q8_0_b(MV_PARAMS) {
    MV_HEAD_B;
    int nb = a.n_in / 32;
    const uchar *rw = wb + a.w_off + (ulong64)row * nb * 34;
    for (int b = lane; b < nb; b += 32) {
        const uchar *blk = rw + (ulong64)b * 34;
        float d = f16f(blk);
        const signed char *q = (const signed char *)(blk + 2);
        ulong64 base = (ulong64)b * 32;
        for (int j = 0; j < 32; j++) MV_FMA(d * (float)q[j], base + j);
    }
    MV_TAIL_B;
}

// -------------------------------------------------------------- prefill GEMM
// Real tiled GEMM replacements for the two prefill formats that matter on this
// machine (Q8_0, Q4_K). The batch _b kernels are compute/latency bound: each
// decoded weight issues MVB scattered global x-loads. These variants stage the
// x-tile columns into shared memory once per block, so every decoded weight
// FMAs against smem instead of global memory, and multiple warps (rows) reuse
// the same staged x.
//
// Correctness: the reduction is kept BIT-IDENTICAL to the _b kernels — lane b
// still owns k-blocks b, b+32, ...; the inner j-order is 0..31; the per-weight
// term is the same d*(float)q[j]; the final warp_sum tree is unchanged. Only
// the *source* of x changes (smem vs global), so results match the _b kernels
// exactly and greedy tokens are identical.

#define GEMM_WARPS 8            // output rows per block (warps)
#define Q8_CHUNK   32           // q8 blocks staged per k-iteration (== warp lanes)
#define SMPAD      33           // 32 + 1: makes per-lane smem reads conflict-free

// MVB=16 note: a 16-column f32 x-tile of 32 q8 blocks does not fit shared
// memory (16*1024*4 = 64 KB), and every narrower restructure measured slower
// than this proven 8-column shape (the kernel is smem-FMA bound, so per-weight
// work scales with columns and wider tiles buy nothing on this path). The
// kernel therefore keeps its 8-column tile and cuda.c splits a 16-token tile
// into two launches — identical arithmetic to the MVB=8 build.
#define Q8_COLS 8               // fixed column width of this kernel's tile

// xsm[t][blk_in_chunk*SMPAD + j] holds x column t, element (chunk*1024)+blk*32+j
extern "C" __global__ void k_gemm_q8_0(MV_PARAMS) {
    __shared__ float xsm[Q8_COLS][Q8_CHUNK * SMPAD];
    unsigned warp = threadIdx.x >> 5;
    unsigned lane = threadIdx.x & 31;
    unsigned row  = blockIdx.x * GEMM_WARPS + warp;
    int nb = a.n_in / 32;
    float s[Q8_COLS] = {0};
    const uchar *rw = wb + a.w_off + (ulong64)(row < (unsigned)a.n_out ? row : 0) * nb * 34;

    for (int cs = 0; cs < nb; cs += Q8_CHUNK) {
        int cblocks = nb - cs < Q8_CHUNK ? nb - cs : Q8_CHUNK;
        int celems  = cblocks * 32;
        int base_e  = cs * 32;
        // coalesced cooperative load of this chunk's x into padded smem
        #pragma unroll
        for (int t = 0; t < Q8_COLS; t++) {
            const float *xg = x + (ulong64)t * a.xs + base_e;
            for (int e = threadIdx.x; e < celems; e += blockDim.x)
                xsm[t][(e >> 5) * SMPAD + (e & 31)] = xg[e];
        }
        __syncthreads();
        if (row < (unsigned)a.n_out && lane < (unsigned)cblocks) {
            const uchar *blk = rw + (ulong64)(cs + lane) * 34;
            float d = f16f(blk);
            const signed char *q = (const signed char *)(blk + 2);
            int soff = (int)lane * SMPAD;
            #pragma unroll
            for (int j = 0; j < 32; j++) {
                float w = d * (float)q[j];
                #pragma unroll
                for (int t = 0; t < Q8_COLS; t++) s[t] += w * xsm[t][soff + j];
            }
        }
        __syncthreads();
    }
    if (row < (unsigned)a.n_out) {
        int cols = a.batch < Q8_COLS ? a.batch : Q8_COLS;
        for (int t = 0; t < cols; t++) {
            float r = warp_sum(s[t]);
            if (lane == 0) y[(ulong64)t * a.ys + row] = a.has_bias ? r + bias[row] : r;
        }
    }
}

// -------------------------------------------------------------- decode GEMV
// Batch-1 (decode) matvec replacements for the two formats that matter here
// (Q8_0, Q4_K). The generic k_mv_* decode kernel maps one lane to a whole
// quant block, so consecutive lanes read 34-byte-strided (Q8) addresses: the
// loads never coalesce into 32-byte segments and the kernel tops out at ~18 %
// of peak weight bandwidth (memory-latency/coalescing bound, see diagnosis).
//
// These variants flip the mapping to LANE-PER-ELEMENT: within each block lane
// l owns element l, so the 32 lanes read 32 consecutive bytes -> one coalesced
// transaction per block. Each lane accumulates its own element-position across
// all blocks, then a single warp_sum reduces at the end. This REORDERS the
// k-reduction relative to k_mv_* (per-element partials summed once, vs the
// per-block d*(Sum q*x) of the originals), so identity is not bitwise and is
// established empirically by kernel-verify on the real models. Same block
// shape as k_mv_* (4 rows/block, 128 threads) so occupancy is unchanged; the
// win is purely coalescing.

// v2 (decode-bandwidth pass): the warp covers FOUR blocks per iteration —
// lane l owns bytes [(l&7)*4, +4) of block b0+(l>>3) — so each lane issues one
// aligned float4 x-load and two ushort quant loads per iteration instead of a
// single scalar element. Quart of the iterations, 4x the loads in flight; the
// per-lane partial is reduced by the same warp_sum. Reduction order differs
// from v1 (per-lane running s over its byte-quarter vs per-element), so
// identity vs CPU is established empirically by kernel-verify, like v1 was.
extern "C" __global__ void k_gemv_q8_0(MV_PARAMS) {
    MV_HEAD;
    int nb = a.n_in / 32;
    const uchar *rw = wb + a.w_off + (ulong64)row * nb * 34;
    int bsub = (int)(lane >> 3);          // which of the 4 blocks this lane works
    int boff = ((int)lane & 7) * 4;       // this lane's 4-element chunk
    float s = 0;
    int b4 = nb & ~3;
    for (int b0 = 0; b0 < b4; b0 += 4) {
        const uchar *blk = rw + (ulong64)(b0 + bsub) * 34;
        float d = f16f(blk);
        // 34-byte stride keeps quants only 2-aligned: two ushort loads, then
        // sign-extend the four int8 lanes
        const uchar *qp = blk + 2 + boff;
        ushort16 u0 = *(const ushort16 *)qp, u1 = *(const ushort16 *)(qp + 2);
        int q0 = (int)(signed char)(u0 & 0xFF), q1 = (int)(signed char)(u0 >> 8);
        int q2 = (int)(signed char)(u1 & 0xFF), q3 = (int)(signed char)(u1 >> 8);
        const float4 xv = *(const float4 *)(x + (ulong64)(b0 + bsub) * 32 + boff);
        s += d * ((float)q0 * xv.x + (float)q1 * xv.y +
                  (float)q2 * xv.z + (float)q3 * xv.w);
    }
    // tail blocks (nb not a multiple of 4): v1's element-per-lane mapping
    for (int b = b4; b < nb; b++) {
        const uchar *blk = rw + (ulong64)b * 34;
        float d = f16f(blk);
        const signed char *q = (const signed char *)(blk + 2);
        s += d * ((float)q[lane] * x[(ulong64)b * 32 + lane]);
    }
    MV_TAIL;
}

// Q4_0 decode GEMV — the twin k_gemv_q8_0 never got (2026-08-13). Without it
// Q4_0 decode fell through to k_mv_q4_0, where one lane walks a whole 32-element
// block with a serial 16-iteration scalar loop: measured ~55 GB/s of implied
// weight bandwidth against 250-330 GB/s for every Q4_K/Q8_0 model on the same
// slice, a 5x gap that was pure kernel coverage, not arithmetic.
//
// Same shape as k_gemv_q8_0: four blocks in flight across the warp, eight lanes
// per block. A q4_0 block packs element j in the low nibble of byte j and
// element j+16 in the high nibble, so a lane taking two adjacent quant bytes
// owns four elements — two contiguous at boff and two contiguous at boff+16 —
// which is two aligned float2 activation loads and one 2-aligned ushort weight
// load (the 18-byte block stride never gives more than 2-alignment).
//
// The reduction is reordered relative to k_mv_q4_0, exactly as k_gemv_q4_K is
// relative to k_mv_q4_K, so identity is an empirical gate (kernel-verify +
// cpu_cuda_check), not an accumulation-order argument.
extern "C" __global__ void k_gemv_q4_0(MV_PARAMS) {
    MV_HEAD;
    int nb = a.n_in / 32;
    const uchar *rw = wb + a.w_off + (ulong64)row * nb * 18;
    int bsub = (int)(lane >> 3);          // which of the 4 blocks this lane works
    int boff = ((int)lane & 7) * 2;       // this lane's 2 quant bytes
    float s = 0;
    int b4 = nb & ~3;
    for (int b0 = 0; b0 < b4; b0 += 4) {
        const uchar *blk = rw + (ulong64)(b0 + bsub) * 18;
        float d = f16f(blk);
        ushort16 u = *(const ushort16 *)(blk + 2 + boff);
        int q0 = (int)( u        & 0xF) - 8;   // byte boff  low  -> elem boff
        int q1 = (int)((u >>  4) & 0xF) - 8;   // byte boff  high -> elem boff+16
        int q2 = (int)((u >>  8) & 0xF) - 8;   // byte boff+1 low  -> elem boff+1
        int q3 = (int)((u >> 12) & 0xF) - 8;   // byte boff+1 high -> elem boff+17
        const float *xp = x + (ulong64)(b0 + bsub) * 32 + boff;
        float2 xlo = *(const float2 *)xp;
        float2 xhi = *(const float2 *)(xp + 16);
        s += d * ((float)q0 * xlo.x + (float)q2 * xlo.y +
                  (float)q1 * xhi.x + (float)q3 * xhi.y);
    }
    // tail blocks (nb not a multiple of 4): one element per lane
    for (int b = b4; b < nb; b++) {
        const uchar *blk = rw + (ulong64)b * 18;
        float d = f16f(blk);
        const uchar *q = blk + 2;
        int j = (int)lane & 15, hi = (int)lane >> 4;
        int qv = hi ? (q[j] >> 4) : (q[j] & 0xF);
        s += d * (float)(qv - 8) * x[(ulong64)b * 32 + lane];
    }
    MV_TAIL;
}

extern "C" __global__ void k_mv_q4_0(MV_PARAMS) {
    MV_HEAD;
    int nb = a.n_in / 32;
    const uchar *rw = wb + a.w_off + (ulong64)row * nb * 18;
    float s = 0;
    for (int b = lane; b < nb; b += 32) {
        const uchar *blk = rw + (ulong64)b * 18;
        float d = f16f(blk);
        const uchar *q = blk + 2;
        const float *xp = x + b * 32;
        float t = 0;
        for (int j = 0; j < 16; j++)
            t += ((int)(q[j] & 0xF) - 8) * xp[j] + ((int)(q[j] >> 4) - 8) * xp[j + 16];
        s += d * t;
    }
    MV_TAIL;
}

extern "C" __global__ void k_mv_q4_0_b(MV_PARAMS) {
    MV_HEAD_B;
    int nb = a.n_in / 32;
    const uchar *rw = wb + a.w_off + (ulong64)row * nb * 18;
    for (int b = lane; b < nb; b += 32) {
        const uchar *blk = rw + (ulong64)b * 18;
        float d = f16f(blk);
        const uchar *q = blk + 2;
        ulong64 base = (ulong64)b * 32;
        for (int j = 0; j < 16; j++) {
            MV_FMA(d * (float)((int)(q[j] & 0xF) - 8), base + j);
            MV_FMA(d * (float)((int)(q[j] >> 4)  - 8), base + j + 16);
        }
    }
    MV_TAIL_B;
}

extern "C" __global__ void k_mv_q4_1(MV_PARAMS) {
    MV_HEAD;
    int nb = a.n_in / 32;
    const uchar *rw = wb + a.w_off + (ulong64)row * nb * 20;
    float s = 0;
    for (int b = lane; b < nb; b += 32) {
        const uchar *blk = rw + (ulong64)b * 20;
        float d  = f16f(blk);
        float mm = f16f(blk + 2);
        const uchar *q = blk + 4;
        const float *xp = x + b * 32;
        float t = 0, sx = 0;
        for (int j = 0; j < 16; j++) {
            t += (float)(q[j] & 0xF) * xp[j] + (float)(q[j] >> 4) * xp[j + 16];
            sx += xp[j] + xp[j + 16];
        }
        s += d * t + mm * sx;
    }
    MV_TAIL;
}

extern "C" __global__ void k_mv_q4_1_b(MV_PARAMS) {
    MV_HEAD_B;
    int nb = a.n_in / 32;
    const uchar *rw = wb + a.w_off + (ulong64)row * nb * 20;
    for (int b = lane; b < nb; b += 32) {
        const uchar *blk = rw + (ulong64)b * 20;
        float d  = f16f(blk);
        float mm = f16f(blk + 2);
        const uchar *q = blk + 4;
        ulong64 base = (ulong64)b * 32;
        for (int j = 0; j < 16; j++) {
            MV_FMA(d * (float)(q[j] & 0xF) + mm, base + j);
            MV_FMA(d * (float)(q[j] >> 4)  + mm, base + j + 16);
        }
    }
    MV_TAIL_B;
}

extern "C" __global__ void k_mv_q5_0(MV_PARAMS) {
    MV_HEAD;
    int nb = a.n_in / 32;
    const uchar *rw = wb + a.w_off + (ulong64)row * nb * 22;
    float s = 0;
    for (int b = lane; b < nb; b += 32) {
        const uchar *blk = rw + (ulong64)b * 22;
        float d = f16f(blk);
        uint qh = (uint)blk[2] | ((uint)blk[3] << 8) |
                  ((uint)blk[4] << 16) | ((uint)blk[5] << 24);
        const uchar *q = blk + 6;
        const float *xp = x + b * 32;
        float t = 0;
        for (int j = 0; j < 16; j++) {
            int x0 = (int)((q[j] & 0xF) | (((qh >> j) & 1u) << 4)) - 16;
            int x1 = (int)((q[j] >> 4)  | (((qh >> (j + 16)) & 1u) << 4)) - 16;
            t += x0 * xp[j] + x1 * xp[j + 16];
        }
        s += d * t;
    }
    MV_TAIL;
}

extern "C" __global__ void k_mv_q5_0_b(MV_PARAMS) {
    MV_HEAD_B;
    int nb = a.n_in / 32;
    const uchar *rw = wb + a.w_off + (ulong64)row * nb * 22;
    for (int b = lane; b < nb; b += 32) {
        const uchar *blk = rw + (ulong64)b * 22;
        float d = f16f(blk);
        uint qh = (uint)blk[2] | ((uint)blk[3] << 8) |
                  ((uint)blk[4] << 16) | ((uint)blk[5] << 24);
        const uchar *q = blk + 6;
        ulong64 base = (ulong64)b * 32;
        for (int j = 0; j < 16; j++) {
            int x0 = (int)((q[j] & 0xF) | (((qh >> j) & 1u) << 4)) - 16;
            int x1 = (int)((q[j] >> 4)  | (((qh >> (j + 16)) & 1u) << 4)) - 16;
            MV_FMA(d * (float)x0, base + j);
            MV_FMA(d * (float)x1, base + j + 16);
        }
    }
    MV_TAIL_B;
}

extern "C" __global__ void k_mv_q5_1(MV_PARAMS) {
    MV_HEAD;
    int nb = a.n_in / 32;
    const uchar *rw = wb + a.w_off + (ulong64)row * nb * 24;
    float s = 0;
    for (int b = lane; b < nb; b += 32) {
        const uchar *blk = rw + (ulong64)b * 24;
        float d  = f16f(blk);
        float mm = f16f(blk + 2);
        uint qh = (uint)blk[4] | ((uint)blk[5] << 8) |
                  ((uint)blk[6] << 16) | ((uint)blk[7] << 24);
        const uchar *q = blk + 8;
        const float *xp = x + b * 32;
        float t = 0, sx = 0;
        for (int j = 0; j < 16; j++) {
            t += (float)((q[j] & 0xF) | (((qh >> j) & 1u) << 4)) * xp[j] +
                 (float)((q[j] >> 4)  | (((qh >> (j + 16)) & 1u) << 4)) * xp[j + 16];
            sx += xp[j] + xp[j + 16];
        }
        s += d * t + mm * sx;
    }
    MV_TAIL;
}

extern "C" __global__ void k_mv_q5_1_b(MV_PARAMS) {
    MV_HEAD_B;
    int nb = a.n_in / 32;
    const uchar *rw = wb + a.w_off + (ulong64)row * nb * 24;
    for (int b = lane; b < nb; b += 32) {
        const uchar *blk = rw + (ulong64)b * 24;
        float d  = f16f(blk);
        float mm = f16f(blk + 2);
        uint qh = (uint)blk[4] | ((uint)blk[5] << 8) |
                  ((uint)blk[6] << 16) | ((uint)blk[7] << 24);
        const uchar *q = blk + 8;
        ulong64 base = (ulong64)b * 32;
        for (int j = 0; j < 16; j++) {
            MV_FMA(d * (float)((q[j] & 0xF) | (((qh >> j) & 1u) << 4)) + mm, base + j);
            MV_FMA(d * (float)((q[j] >> 4)  | (((qh >> (j + 16)) & 1u) << 4)) + mm, base + j + 16);
        }
    }
    MV_TAIL_B;
}

static __device__ __forceinline__ void get_scale_min_k4(int j, const uchar *q,
                                                        uchar *d, uchar *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4)  | ((q[j    ] >> 6) << 4);
    }
}

// bf16 is the top 16 bits of an f32: widening is a shift and a reinterpret,
// with no rounding, so this matches bf16_to_f32() in fp16.h exactly. Read as
// ushort and widened by hand rather than via __nv_bfloat16, which would pull
// in a header the PTX build does not otherwise need.
static __device__ __forceinline__ float bf16f(unsigned short h) {
    return __uint_as_float((unsigned int)h << 16);
}

extern "C" __global__ void k_mv_bf16(MV_PARAMS) {
    MV_HEAD;
    const unsigned short *rw =
        (const unsigned short *)(wb + a.w_off) + (ulong64)row * a.n_in;
    float s = 0;
    for (int i = lane; i < a.n_in; i += 32) s += bf16f(rw[i]) * x[i];
    MV_TAIL;
}

extern "C" __global__ void k_mv_bf16_b(MV_PARAMS) {
    MV_HEAD_B;
    const unsigned short *rw =
        (const unsigned short *)(wb + a.w_off) + (ulong64)row * a.n_in;
    for (int i = lane; i < a.n_in; i += 32) MV_FMA(bf16f(rw[i]), i);
    MV_TAIL_B;
}

// ports quants.c dq_q2_K into the warp-per-row dot. Unlike q3_K the scale byte
// carries BOTH a scale (low nibble) and a min (high nibble), and the min is
// subtracted from the dequantized weight rather than scaling it -- so the value
// handed to the accumulate is (dl*q - ml), not a product.
#define Q2K_SETUP \
    float d = f16f(blk + 80), dmin = f16f(blk + 82); \
    const uchar *q2scales = blk; \
    const uchar *qbase = blk + 16;

extern "C" __global__ void k_mv_q2_K(MV_PARAMS) {
    MV_HEAD;
    int nb = a.n_in / 256;
    const uchar *rw = wb + a.w_off + (ulong64)row * nb * 84;
    float s = 0;
    for (int b = lane; b < nb; b += 32) {
        const uchar *blk = rw + (ulong64)b * 84;
        Q2K_SETUP;
        const float *xp = x + (ulong64)b * 256;
        int pos = 0, is = 0; const uchar *q = qbase;
        for (int n = 0; n < 256; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                uchar scb = q2scales[is++];
                float dl = d * (scb & 0xF), ml = dmin * (scb >> 4);
                for (int l = 0; l < 16; l++)
                    s += (dl * (float)((q[l] >> shift) & 3) - ml) * xp[pos++];
                scb = q2scales[is++];
                dl = d * (scb & 0xF); ml = dmin * (scb >> 4);
                for (int l = 0; l < 16; l++)
                    s += (dl * (float)((q[l+16] >> shift) & 3) - ml) * xp[pos++];
                shift += 2;
            }
            q += 32;
        }
    }
    MV_TAIL;
}

extern "C" __global__ void k_mv_q2_K_b(MV_PARAMS) {
    MV_HEAD_B;
    int nb = a.n_in / 256;
    const uchar *rw = wb + a.w_off + (ulong64)row * nb * 84;
    for (int b = lane; b < nb; b += 32) {
        const uchar *blk = rw + (ulong64)b * 84;
        Q2K_SETUP;
        ulong64 base = (ulong64)b * 256;
        int pos = 0, is = 0; const uchar *q = qbase;
        for (int n = 0; n < 256; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                uchar scb = q2scales[is++];
                float dl = d * (scb & 0xF), ml = dmin * (scb >> 4);
                for (int l = 0; l < 16; l++)
                    MV_FMA(dl * (float)((q[l] >> shift) & 3) - ml, base + pos++);
                scb = q2scales[is++];
                dl = d * (scb & 0xF); ml = dmin * (scb >> 4);
                for (int l = 0; l < 16; l++)
                    MV_FMA(dl * (float)((q[l+16] >> shift) & 3) - ml, base + pos++);
                shift += 2;
            }
            q += 32;
        }
    }
    MV_TAIL_B;
}

// Q3_K: 110-byte block, 256 elements. Layout: hmask[32] (one high bit per
// weight), qs[64] (2-bit low bits), scales[12] (packed 16x 6-bit, bias -32),
// d (f16 super-scale). Weight = d*(scale-32)*((2bit) - (highbit?0:4)). This
// ports quants.c dq_q3_K into the warp-per-row dot (dequant fused into the
// accumulate), matching the CPU reference bit-for-bit modulo reduction order.
#define Q3K_UNPACK_SCALES \
    const uint kmask1 = 0x03030303u, kmask2 = 0x0f0f0f0fu; \
    const uchar *sc = blk + 96; \
    uint a0 = sc[0] | (sc[1]<<8) | (sc[2]<<16) | ((uint)sc[3]<<24); \
    uint a1 = sc[4] | (sc[5]<<8) | (sc[6]<<16) | ((uint)sc[7]<<24); \
    uint a2 = sc[8] | (sc[9]<<8) | (sc[10]<<16) | ((uint)sc[11]<<24); \
    uint xs[4]; \
    xs[2] = ((a0 >> 4) & kmask2) | (((a2 >> 4) & kmask1) << 4); \
    xs[3] = ((a1 >> 4) & kmask2) | (((a2 >> 6) & kmask1) << 4); \
    xs[0] = ( a0       & kmask2) | (((a2 >> 0) & kmask1) << 4); \
    xs[1] = ( a1       & kmask2) | (((a2 >> 2) & kmask1) << 4); \
    const signed char *q3scales = (const signed char *)xs; \
    float d_all = f16f(blk + 108); \
    const uchar *hm = blk; \
    const uchar *qbase = blk + 32;

extern "C" __global__ void k_mv_q3_K(MV_PARAMS) {
    MV_HEAD;
    int nb = a.n_in / 256;
    const uchar *rw = wb + a.w_off + (ulong64)row * nb * 110;
    float s = 0;
    for (int b = lane; b < nb; b += 32) {
        const uchar *blk = rw + (ulong64)b * 110;
        Q3K_UNPACK_SCALES;
        const float *xp = x + (ulong64)b * 256;
        int pos = 0, is = 0; uchar mbit = 1; const uchar *q = qbase;
        for (int n = 0; n < 256; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                float dl = d_all * (q3scales[is++] - 32);
                for (int l = 0; l < 16; l++)
                    s += dl * (float)(((q[l] >> shift) & 3) - ((hm[l] & mbit) ? 0 : 4)) * xp[pos++];
                dl = d_all * (q3scales[is++] - 32);
                for (int l = 0; l < 16; l++)
                    s += dl * (float)(((q[l+16] >> shift) & 3) - ((hm[l+16] & mbit) ? 0 : 4)) * xp[pos++];
                shift += 2; mbit <<= 1;
            }
            q += 32;
        }
    }
    MV_TAIL;
}

extern "C" __global__ void k_mv_q3_K_b(MV_PARAMS) {
    MV_HEAD_B;
    int nb = a.n_in / 256;
    const uchar *rw = wb + a.w_off + (ulong64)row * nb * 110;
    for (int b = lane; b < nb; b += 32) {
        const uchar *blk = rw + (ulong64)b * 110;
        Q3K_UNPACK_SCALES;
        ulong64 base = (ulong64)b * 256;
        int pos = 0, is = 0; uchar mbit = 1; const uchar *q = qbase;
        for (int n = 0; n < 256; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                float dl = d_all * (q3scales[is++] - 32);
                for (int l = 0; l < 16; l++)
                    MV_FMA(dl * (float)(((q[l] >> shift) & 3) - ((hm[l] & mbit) ? 0 : 4)), base + pos++);
                dl = d_all * (q3scales[is++] - 32);
                for (int l = 0; l < 16; l++)
                    MV_FMA(dl * (float)(((q[l+16] >> shift) & 3) - ((hm[l+16] & mbit) ? 0 : 4)), base + pos++);
                shift += 2; mbit <<= 1;
            }
            q += 32;
        }
    }
    MV_TAIL_B;
}

extern "C" __global__ void k_mv_q4_K(MV_PARAMS) {
    MV_HEAD;
    int nb = a.n_in / 256;
    const uchar *rw = wb + a.w_off + (ulong64)row * nb * 144;
    float s = 0;
    for (int b = lane; b < nb; b += 32) {
        const uchar *blk = rw + (ulong64)b * 144;
        float d    = f16f(blk);
        float dmin = f16f(blk + 2);
        const uchar *sc = blk + 4;
        const uchar *q  = blk + 16;
        const float *xp = x + b * 256;
        int is = 0;
        for (int j = 0; j < 256; j += 64) {
            uchar s1, m1, s2, m2;
            get_scale_min_k4(is + 0, sc, &s1, &m1);
            get_scale_min_k4(is + 1, sc, &s2, &m2);
            float d1 = d * s1, mm1 = dmin * m1;
            float d2 = d * s2, mm2 = dmin * m2;
            float t1 = 0, t2 = 0, sx1 = 0, sx2 = 0;
            const uint4 *q16 = (const uint4 *)q;   // blk+16 is 16B-aligned
            for (int v = 0; v < 2; v++) {
                uint4 w = q16[v];
                uint ws[4] = { w.x, w.y, w.z, w.w };
                #pragma unroll
                for (int c = 0; c < 4; c++) {
                    #pragma unroll
                    for (int k = 0; k < 4; k++) {
                        int l = v * 16 + c * 4 + k;
                        uint b8 = (ws[c] >> (8 * k)) & 0xFFu;
                        t1 += (float)(b8 & 0xF) * xp[l];      sx1 += xp[l];
                        t2 += (float)(b8 >> 4)  * xp[l + 32]; sx2 += xp[l + 32];
                    }
                }
            }
            s += d1 * t1 - mm1 * sx1 + d2 * t2 - mm2 * sx2;
            q += 32; is += 2; xp += 64;
        }
    }
    MV_TAIL;
}

extern "C" __global__ void k_mv_q4_K_b(MV_PARAMS) {
    MV_HEAD_B;
    int nb = a.n_in / 256;
    const uchar *rw = wb + a.w_off + (ulong64)row * nb * 144;
    for (int b = lane; b < nb; b += 32) {
        const uchar *blk = rw + (ulong64)b * 144;
        float d    = f16f(blk);
        float dmin = f16f(blk + 2);
        const uchar *sc = blk + 4;
        const uchar *q  = blk + 16;
        ulong64 base = (ulong64)b * 256;
        int is = 0;
        for (int j = 0; j < 256; j += 64) {
            uchar s1, m1, s2, m2;
            get_scale_min_k4(is + 0, sc, &s1, &m1);
            get_scale_min_k4(is + 1, sc, &s2, &m2);
            float d1 = d * s1, mm1 = dmin * m1;
            float d2 = d * s2, mm2 = dmin * m2;
            const uint4 *q16 = (const uint4 *)q;   // blk+16 is 16B-aligned
            for (int v = 0; v < 2; v++) {
                uint4 w = q16[v];
                uint ws[4] = { w.x, w.y, w.z, w.w };
                #pragma unroll
                for (int c = 0; c < 4; c++) {
                    #pragma unroll
                    for (int k = 0; k < 4; k++) {
                        int l = v * 16 + c * 4 + k;
                        uint b8 = (ws[c] >> (8 * k)) & 0xFFu;
                        MV_FMA(d1 * (float)(b8 & 0xF) - mm1, base + j + l);
                        MV_FMA(d2 * (float)(b8 >> 4)  - mm2, base + j + l + 32);
                    }
                }
            }
            q += 32; is += 2;
        }
    }
    MV_TAIL_B;
}

// Q4_K prefill GEMM. Unlike Q8_0 the 256-element block is too wide to keep
// lane==k-block (32 blocks -> 8192 x elements won't fit in smem), so the warp
// instead walks k-blocks sequentially and its 32 lanes cooperatively reduce the
// 256 elements of each block (8 elements/lane). x for the current block is
// staged into smem (small, 256*MVB floats) so decoded weights FMA against smem.
// This REORDERS the k-reduction relative to k_mv_q4_K_b, so it is not bitwise
// identical — token-identity is verified empirically by kernel-verify on the
// real Q4_K model. Lane l owns elements [l*8, l*8+8): all in scale group l/4,
// quant segment l/8, byte offset (l&3)*8, lower nibble iff group even.
extern "C" __global__ void k_gemm_q4_K(MV_PARAMS) {
    __shared__ float xsm[MVT][256];
    unsigned warp = threadIdx.x >> 5;
    unsigned lane = threadIdx.x & 31;
    unsigned row  = blockIdx.x * GEMM_WARPS + warp;
    int nb = a.n_in / 256;
    const uchar *rw = wb + a.w_off +
                      (ulong64)(row < (unsigned)a.n_out ? row : 0) * nb * 144;
    float s[MVT] = {0};
    int g     = (int)(lane >> 2);         // scale/min group 0..7
    int ji    = (int)(lane >> 3);         // 32-byte quant segment 0..3
    int lo    = (((int)lane >> 2) & 1) == 0;
    int bbase = ((int)lane & 3) * 8;      // byte offset within the segment

    for (int b = 0; b < nb; b++) {
        const uchar *blk = rw + (ulong64)b * 144;
        int base_e = b * 256;
        #pragma unroll
        for (int t = 0; t < MVT; t++) {
            const float *xg = x + (ulong64)t * a.xs + base_e;
            for (int e = threadIdx.x; e < 256; e += blockDim.x) xsm[t][e] = xg[e];
        }
        __syncthreads();
        if (row < (unsigned)a.n_out) {
            float dd   = f16f(blk);
            float dmin = f16f(blk + 2);
            const uchar *sc = blk + 4;
            const uchar *q  = blk + 16 + ji * 32;
            uchar sg, mg;
            get_scale_min_k4(g, sc, &sg, &mg);
            float dg = dd * (float)sg, mmg = dmin * (float)mg;
            int el = (int)lane * 8;
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                uchar byte = q[bbase + k];
                int nib = lo ? (byte & 0xF) : (byte >> 4);
                float w = dg * (float)nib - mmg;
                #pragma unroll
                for (int t = 0; t < MVT; t++) s[t] += w * xsm[t][el + k];
            }
        }
        __syncthreads();
    }
    if (row < (unsigned)a.n_out) {
        for (int t = 0; t < a.batch; t++) {
            float r = warp_sum(s[t]);
            if (lane == 0) y[(ulong64)t * a.ys + row] = a.has_bias ? r + bias[row] : r;
        }
    }
}

// ---------------------------------------------------- tensor-core Q4_K GEMM
// v2, MMQ-style (suite plan P1-prefill lever b; supersedes the per-warp v1,
// which re-dequantized per 16-element K-step through a scalar index helper and
// measured ~6-7x SLOWER than the scalar GEMM at 8 tokens). The economics that
// make this one fast:
//   - the whole BLOCK cooperates on one 64-row x 128-K fp16 weight tile per
//     step, dequantized ONCE into shared memory with 8-byte quant loads —
//     two threads per row, each unpacking one contiguous 64-element segment
//     (exactly two scale groups) — then all four warps' MMAs reuse it;
//   - activations are staged as a 128-K x 16-token fp16 tile (vectorized
//     float4 reads, zero-padded past a.batch), tiny enough to live in L2
//     across the row-block sweep;
//   - each warp owns a 16-row m16n16k16 fragment strip: 4 warps x 16 = 64
//     rows per block, accumulated in fp32 across the full K without ever
//     leaving registers (the "register-accumulate" of the plan item).
// Numerics: identical per-element weight values to k_gemm_q4_K, rounded to
// fp16 operands with fp32 accumulation — same numeric class as v1, still
// behind the RUNNER_CUDA_TC opt-in + tolerance gate, never the default path.
// Compiles to compute_75 PTX (fp16 WMMA is Turing+; no bf16 in sm_75).

#define TC_ROWS 64    // output rows per block (4 warps x 16-row fragments)
#define TC_K    128   // K-elements staged per step (half a q4_K super-block)
#define TC_N    64    // token columns per tile (was 16; widened to amortise the weight dequantisation)

extern "C" __global__ void k_gemm_q4_K_tc(MV_PARAMS) {
    using namespace nvcuda::wmma;
    const int tid  = threadIdx.x;         // 128 threads = 4 warps
    const int warp = tid >> 5;
    const unsigned row0 = blockIdx.x * TC_ROWS;
    __shared__ __half sh_w[TC_ROWS * TC_K];   // row-major, ldm = TC_K
    __shared__ __half sh_x[TC_N * TC_K];      // col-major (k,t) at [t*TC_K+k]
    // The epilogue tile ALIASES the weight tile: sh_w is 64*128 halves = 16 KB
    // and sh_c needs TC_ROWS*TC_N floats, and their lifetimes are disjoint
    // (weights are dead once the last MMA has run). Separate arrays would push
    // the block past the 48 KB static shared cap at a wide tile.
    float *sh_c = (float *)sh_w;

    fragment<matrix_a, 16, 16, 16, __half, row_major> fa;
    fragment<matrix_b, 16, 16, 16, __half, col_major> fb;
    fragment<accumulator, 16, 16, 16, float> fc[TC_N / 16];
    #pragma unroll
    for (int n = 0; n < TC_N / 16; n++) fill_fragment(fc[n], 0.0f);

    int nb = a.n_in / 256;
    // this thread stages one 64-element segment of one row per K-step:
    // row srow, segment sseg (0 or 1) of the 128-K tile
    int srow = tid >> 1, sseg = tid & 1;

    for (int b = 0; b < nb; b++) {
        #pragma unroll
        for (int koff = 0; koff < 256; koff += TC_K) {
            // ---- stage weights: 64 rows x 128 K, dequantized once ----
            {
                unsigned gr = row0 + srow;
                __half *dst = sh_w + srow * TC_K + sseg * 64;
                if (gr < (unsigned)a.n_out) {
                    const uchar *blk = wb + a.w_off +
                                       (ulong64)gr * nb * 144 + (ulong64)b * 144;
                    int e0 = koff + sseg * 64;         // segment base element
                    float dd   = f16f(blk);
                    float dmin = f16f(blk + 2);
                    uchar sg0, mg0, sg1, mg1;
                    get_scale_min_k4((e0 >> 5) + 0, blk + 4, &sg0, &mg0);
                    get_scale_min_k4((e0 >> 5) + 1, blk + 4, &sg1, &mg1);
                    float dg0 = dd * (float)sg0, mm0 = dmin * (float)mg0;
                    float dg1 = dd * (float)sg1, mm1 = dmin * (float)mg1;
                    // 64 consecutive elements live in ONE 32-byte quant
                    // segment: bytes b8 carry the low nibbles of elements
                    // 0..31 and the high nibbles of elements 32..63
                    const uint2 *q8 = (const uint2 *)(blk + 16 + (e0 >> 6) * 32);
                    #pragma unroll
                    for (int v = 0; v < 4; v++) {
                        uint2 qv = q8[v];
                        #pragma unroll
                        for (int w = 0; w < 2; w++) {
                            uint bits = w ? qv.y : qv.x;
                            int base = v * 8 + w * 4;
                            uint lo = bits & 0x0F0F0F0Fu;
                            uint hi = (bits >> 4) & 0x0F0F0F0Fu;
                            dst[base + 0]  = __float2half(dg0 * (float)(lo & 0xFF) - mm0);
                            dst[base + 1]  = __float2half(dg0 * (float)((lo >> 8) & 0xFF) - mm0);
                            dst[base + 2]  = __float2half(dg0 * (float)((lo >> 16) & 0xFF) - mm0);
                            dst[base + 3]  = __float2half(dg0 * (float)(lo >> 24) - mm0);
                            dst[base + 32] = __float2half(dg1 * (float)(hi & 0xFF) - mm1);
                            dst[base + 33] = __float2half(dg1 * (float)((hi >> 8) & 0xFF) - mm1);
                            dst[base + 34] = __float2half(dg1 * (float)((hi >> 16) & 0xFF) - mm1);
                            dst[base + 35] = __float2half(dg1 * (float)(hi >> 24) - mm1);
                        }
                    }
                } else {
                    #pragma unroll
                    for (int e = 0; e < 64; e++) dst[e] = __float2half(0.0f);
                }
            }
            // ---- stage activations: 128 K x 16 tokens, vectorized ----
            {
                int col = tid / 2, part = tid % 2;
                __half *dst = sh_x + col * TC_K + part * 64;
                if (col < a.batch) {
                    const float *xg = x + (ulong64)col * a.xs + b * 256 + koff
                                      + part * 64;
                    #pragma unroll
                    for (int v = 0; v < 16; v++) {
                        float4 xv = *(const float4 *)(xg + v * 4);
                        dst[v * 4 + 0] = __float2half(xv.x);
                        dst[v * 4 + 1] = __float2half(xv.y);
                        dst[v * 4 + 2] = __float2half(xv.z);
                        dst[v * 4 + 3] = __float2half(xv.w);
                    }
                } else {
                    #pragma unroll
                    for (int e = 0; e < 64; e++) dst[e] = __float2half(0.0f);
                }
            }
            __syncthreads();
            // ---- 8 MMA K-steps over the staged tile ----
            const __half *wt = sh_w + warp * 16 * TC_K;
            #pragma unroll
            for (int k = 0; k < TC_K; k += 16) {
                load_matrix_sync(fa, wt + k, TC_K);
                // one staged weight tile, reused across every n-tile: this is
                // the entire point of widening TC_N
                #pragma unroll
                for (int n = 0; n < TC_N / 16; n++) {
                    load_matrix_sync(fb, sh_x + n * 16 * TC_K + k, TC_K);
                    mma_sync(fc[n], fa, fb, fc[n]);
                }
            }
            __syncthreads();
        }
    }
    // sh_c aliases sh_w: every warp must be done reading weights first
    __syncthreads();
    #pragma unroll
    for (int n = 0; n < TC_N / 16; n++)
        store_matrix_sync(sh_c + warp * 16 * TC_N + n * 16, fc[n], TC_N,
                          mem_row_major);
    __syncthreads();
    for (int idx = tid; idx < TC_ROWS * TC_N; idx += blockDim.x) {
        int rr = idx / TC_N, tt = idx % TC_N;
        unsigned gr = row0 + rr;
        if (gr < (unsigned)a.n_out && tt < a.batch) {
            float r = sh_c[rr * TC_N + tt];
            y[(ulong64)tt * a.ys + gr] = a.has_bias ? r + bias[gr] : r;
        }
    }
}

// ------------------------------------- tensor-core Q8_0 / Q4_0 GEMM twins
// P3 of the moe-gpu-routing spec: the same MMQ-style structure as
// k_gemm_q4_K_tc — a 64-row x 128-K fp16 weight tile dequantized once per
// step by the whole block, a 128-K x 16-token fp16 activation tile, four
// warps of m16n16k16 MMA strips with fp32 accumulation. Per-element weight
// values match the scalar kernels' dequantization exactly, rounded to fp16
// operands. Both are OPT-IN (RUNNER_CUDA_TC / per-(type, arch) promotion
// via make test-tc-tol); tc_promoted() does not list them, so the default
// path is untouched.
//
// Unlike Q4_K (256-element super-blocks, so every K-step is block-aligned),
// these formats have 32-element blocks and real models carry K dims that
// are not 128-multiples (gemma-4-MoE n_ff_exp = 704), so the K loop is
// tail-safe: elements past a.n_in stage as zeros, which the MMA then
// accumulates harmlessly.

// Stage one 64-element segment of one row: two 32-element quant blocks,
// dequantized with the scalar kernels' exact per-element arithmetic.
static __device__ __forceinline__ void tc_stage_q8_0(__half *dst,
                                                     const uchar *rw, int nb,
                                                     int e0, int n_in) {
    #pragma unroll
    for (int half = 0; half < 2; half++) {
        int base = e0 + half * 32;
        if (base >= n_in) {
            #pragma unroll
            for (int j = 0; j < 32; j++) dst[half * 32 + j] = __float2half(0.0f);
            continue;
        }
        const uchar *blk = rw + (ulong64)(base / 32) * 34;
        float d = f16f(blk);
        const signed char *q = (const signed char *)(blk + 2);
        #pragma unroll
        for (int j = 0; j < 32; j++)
            dst[half * 32 + j] = __float2half(d * (float)q[j]);
    }
    (void)nb;
}

static __device__ __forceinline__ void tc_stage_q4_0(__half *dst,
                                                     const uchar *rw, int nb,
                                                     int e0, int n_in) {
    #pragma unroll
    for (int half = 0; half < 2; half++) {
        int base = e0 + half * 32;
        if (base >= n_in) {
            #pragma unroll
            for (int j = 0; j < 32; j++) dst[half * 32 + j] = __float2half(0.0f);
            continue;
        }
        const uchar *blk = rw + (ulong64)(base / 32) * 18;
        float d = f16f(blk);
        const uchar *q = blk + 2;
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            dst[half * 32 + j]      = __float2half(d * (float)((int)(q[j] & 0xF) - 8));
            dst[half * 32 + j + 16] = __float2half(d * (float)((int)(q[j] >> 4)  - 8));
        }
    }
    (void)nb;
}

// 32-byte-block twin of k_gemm_q4_K_tc (Q8_0, Q4_0): same 64-row x TC_K fp16
// weight tile, staged through STAGE instead of the q4_K super-block decoder.
//
// FIXED 2026-08-13 — this macro was left at the original TC_N=16 shape when
// TC_N was widened 16 -> 64 in 6cf8c70 (2026-08-08 — NOT 2026-07-29, which is
// the Q4_K promotion date; the correction matters because it sets the exposure
// window: 2026-08-08 to 2026-08-13, and only for a type promoted or forced in
// it — Q8_0 from 2026-08-09, gemma4 Q4_0 from 2026-08-12). It staged 16
// activation columns,
// accumulated a single 16-wide fragment and stored one 16x16 tile, then the
// epilogue wrote sh_c columns 16..batch-1 — never written, so UNINITIALISED
// shared memory — into y. Q8_0 is promoted by default, so the default CUDA
// prefill path produced corrupt logits for every prompt batch above 16 (the
// runner's own default is -b 64). It reproduced as: greedy output identical
// to the scalar path at -b 16, divergent at -b 32 and -b 64.
//
// The tolerance gate did not catch it, and "it was measured before the
// widening" turned out to be the wrong explanation: re-run on 2026-08-13, the
// pre-fix kernel PASSES the teacher-forced gate on phi3 and gemma4 q4_0
// ("BIT-IDENTICAL over 448/820 dispatches") while the same binary diverges in
// free-running greedy at -b 64. The gate ran at n_ctx = n_tok + 8; at a
// production context the block inherits ZEROED shared memory and the
// corruption surfaces. test_tc_tol now carries a free-running arm at ctx 4096
// that fails against this kernel.
#define TC_GEMM_32B(NAME, STAGE, BLKBYTES)                                     \
extern "C" __global__ void NAME(MV_PARAMS) {                                   \
    using namespace nvcuda::wmma;                                              \
    const int tid  = threadIdx.x;                                              \
    const int warp = tid >> 5;                                                 \
    const unsigned row0 = blockIdx.x * TC_ROWS;                                \
    __shared__ __half sh_w[TC_ROWS * TC_K];                                    \
    __shared__ __half sh_x[TC_N * TC_K];                                       \
    __shared__ float  sh_c[TC_ROWS * TC_N];                                    \
    fragment<matrix_a, 16, 16, 16, __half, row_major> fa;                      \
    fragment<matrix_b, 16, 16, 16, __half, col_major> fb;                      \
    fragment<accumulator, 16, 16, 16, float> fc[TC_N / 16];                    \
    _Pragma("unroll")                                                          \
    for (int n = 0; n < TC_N / 16; n++) fill_fragment(fc[n], 0.0f);            \
    int nb = a.n_in / 32;                                                      \
    int srow = tid >> 1, sseg = tid & 1;                                       \
    for (int ks = 0; ks < a.n_in; ks += TC_K) {                                \
        {                                                                      \
            unsigned gr = row0 + srow;                                         \
            __half *dst = sh_w + srow * TC_K + sseg * 64;                      \
            if (gr < (unsigned)a.n_out) {                                      \
                const uchar *rw = wb + a.w_off +                               \
                                  (ulong64)gr * nb * BLKBYTES;                 \
                STAGE(dst, rw, nb, ks + sseg * 64, a.n_in);                    \
            } else {                                                           \
                _Pragma("unroll")                                              \
                for (int e = 0; e < 64; e++) dst[e] = __float2half(0.0f);      \
            }                                                                  \
        }                                                                      \
        {                                                                      \
            /* TC_N columns x TC_K elements with 128 threads: two 64-element */\
            /* halves per thread, guarded at 32-element granularity because  */\
            /* n_in is a 32-multiple (quant blocks) but need not be a        */\
            /* 64-multiple — a 64-wide part can straddle the end of the row. */\
            int col = tid >> 1, part = tid & 1;                                \
            __half *dst = sh_x + col * TC_K + part * 64;                       \
            _Pragma("unroll")                                                  \
            for (int h = 0; h < 2; h++) {                                      \
                __half *d2 = dst + h * 32;                                     \
                int e0 = ks + part * 64 + h * 32;                              \
                if (col < a.batch && e0 < a.n_in) {                            \
                    const float *xg = x + (ulong64)col * a.xs + e0;            \
                    _Pragma("unroll")                                          \
                    for (int v = 0; v < 8; v++) {                              \
                        float4 xv = *(const float4 *)(xg + v * 4);             \
                        d2[v * 4 + 0] = __float2half(xv.x);                    \
                        d2[v * 4 + 1] = __float2half(xv.y);                    \
                        d2[v * 4 + 2] = __float2half(xv.z);                    \
                        d2[v * 4 + 3] = __float2half(xv.w);                    \
                    }                                                          \
                } else {                                                       \
                    _Pragma("unroll")                                          \
                    for (int e = 0; e < 32; e++) d2[e] = __float2half(0.0f);   \
                }                                                              \
            }                                                                  \
        }                                                                      \
        __syncthreads();                                                       \
        const __half *wt = sh_w + warp * 16 * TC_K;                            \
        _Pragma("unroll")                                                      \
        for (int k = 0; k < TC_K; k += 16) {                                   \
            load_matrix_sync(fa, wt + k, TC_K);                                \
            _Pragma("unroll")                                                  \
            for (int n = 0; n < TC_N / 16; n++) {                              \
                load_matrix_sync(fb, sh_x + n * 16 * TC_K + k, TC_K);          \
                mma_sync(fc[n], fa, fb, fc[n]);                                \
            }                                                                  \
        }                                                                      \
        __syncthreads();                                                       \
    }                                                                          \
    _Pragma("unroll")                                                          \
    for (int n = 0; n < TC_N / 16; n++)                                        \
        store_matrix_sync(sh_c + warp * 16 * TC_N + n * 16, fc[n], TC_N,       \
                          mem_row_major);                                      \
    __syncthreads();                                                           \
    for (int idx = tid; idx < TC_ROWS * TC_N; idx += blockDim.x) {             \
        int rr = idx / TC_N, tt = idx % TC_N;                                  \
        unsigned gr = row0 + rr;                                               \
        if (gr < (unsigned)a.n_out && tt < a.batch) {                          \
            float r = sh_c[rr * TC_N + tt];                                    \
            y[(ulong64)tt * a.ys + gr] = a.has_bias ? r + bias[gr] : r;        \
        }                                                                      \
    }                                                                          \
}

TC_GEMM_32B(k_gemm_q8_0_tc, tc_stage_q8_0, 34)
TC_GEMM_32B(k_gemm_q4_0_tc, tc_stage_q4_0, 18)

// Q4_K decode GEMV: lane-per-element coalesced variant of k_mv_q4_K. Lane l
// owns the 8 elements [l*8, l*8+8), reusing the exact per-element weight
// geometry of k_gemm_q4_K (which passed identity empirically) -> group l/4,
// quant segment l/8, byte offset (l&3)*8, lower nibble iff group even. x is
// read from global (single decode column, small + L1-cached); the 128-byte
// quant region of each block is read coalesced across the warp. Reduction is
// reordered vs k_mv_q4_K -> identity is verified empirically.
// v2 (decode-bandwidth pass): same lane-per-8-elements geometry, but the 8
// quant bytes arrive as one aligned uint2 (blk+16 + ji*32 + bbase is 8-aligned
// within the 144-byte block) and the 8 x elements as two aligned float4s. The
// per-group affine is factored out — s += dg*sum(nib*x) - mmg*sum(x) — so the
// inner 8 elements are pure FMAs on unpacked nibbles. Per-element weights are
// unchanged from v1; only load shape and summation order differ, so identity
// vs CPU stays empirically gated by kernel-verify.
extern "C" __global__ void k_gemv_q4_K(MV_PARAMS) {
    MV_HEAD;
    int nb = a.n_in / 256;
    const uchar *rw = wb + a.w_off + (ulong64)row * nb * 144;
    int g     = (int)(lane >> 2);         // scale/min group 0..7
    int ji    = (int)(lane >> 3);         // 32-byte quant segment 0..3
    int sh    = ((((int)lane >> 2) & 1) == 0) ? 0 : 4;   // low or high nibble
    int bbase = ((int)lane & 3) * 8;      // byte offset within the segment
    float s = 0;
    for (int b = 0; b < nb; b++) {
        const uchar *blk = rw + (ulong64)b * 144;
        float dd   = f16f(blk);
        float dmin = f16f(blk + 2);
        uchar sg, mg;
        get_scale_min_k4(g, blk + 4, &sg, &mg);
        float dg = dd * (float)sg, mmg = dmin * (float)mg;
        uint2 qv = *(const uint2 *)(blk + 16 + ji * 32 + bbase);
        const float *xp = x + (ulong64)b * 256 + (int)lane * 8;
        float4 x0 = *(const float4 *)xp, x1 = *(const float4 *)(xp + 4);
        uint v0 = (qv.x >> sh) & 0x0F0F0F0Fu, v1 = (qv.y >> sh) & 0x0F0F0F0Fu;
        float t  = (float)(v0 & 0xFF)         * x0.x
                 + (float)((v0 >>  8) & 0xFF) * x0.y
                 + (float)((v0 >> 16) & 0xFF) * x0.z
                 + (float)((v0 >> 24)       ) * x0.w
                 + (float)(v1 & 0xFF)         * x1.x
                 + (float)((v1 >>  8) & 0xFF) * x1.y
                 + (float)((v1 >> 16) & 0xFF) * x1.z
                 + (float)((v1 >> 24)       ) * x1.w;
        float sx = x0.x + x0.y + x0.z + x0.w + x1.x + x1.y + x1.z + x1.w;
        s += dg * t - mmg * sx;
    }
    MV_TAIL;
}

extern "C" __global__ void k_mv_q5_K(MV_PARAMS) {
    MV_HEAD;
    int nb = a.n_in / 256;
    const uchar *rw = wb + a.w_off + (ulong64)row * nb * 176;
    float s = 0;
    for (int b = lane; b < nb; b += 32) {
        const uchar *blk = rw + (ulong64)b * 176;
        float d    = f16f(blk);
        float dmin = f16f(blk + 2);
        const uchar *sc = blk + 4;
        const uchar *qh = blk + 16;
        const uchar *q  = blk + 48;
        const float *xp = x + b * 256;
        int is = 0;
        uchar u1 = 1, u2 = 2;
        for (int j = 0; j < 256; j += 64) {
            uchar s1, m1, s2, m2;
            get_scale_min_k4(is + 0, sc, &s1, &m1);
            get_scale_min_k4(is + 1, sc, &s2, &m2);
            float d1 = d * s1, mm1 = dmin * m1;
            float d2 = d * s2, mm2 = dmin * m2;
            for (int l = 0; l < 32; l++) {
                s += (d1 * (float)((q[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - mm1) * xp[l];
                s += (d2 * (float)((q[l] >> 4)  + ((qh[l] & u2) ? 16 : 0)) - mm2) * xp[l + 32];
            }
            q += 32; is += 2; xp += 64; u1 <<= 2; u2 <<= 2;
        }
    }
    MV_TAIL;
}

extern "C" __global__ void k_mv_q5_K_b(MV_PARAMS) {
    MV_HEAD_B;
    int nb = a.n_in / 256;
    const uchar *rw = wb + a.w_off + (ulong64)row * nb * 176;
    for (int b = lane; b < nb; b += 32) {
        const uchar *blk = rw + (ulong64)b * 176;
        float d    = f16f(blk);
        float dmin = f16f(blk + 2);
        const uchar *sc = blk + 4;
        const uchar *qh = blk + 16;
        const uchar *q  = blk + 48;
        ulong64 base = (ulong64)b * 256;
        int is = 0;
        uchar u1 = 1, u2 = 2;
        for (int j = 0; j < 256; j += 64) {
            uchar s1, m1, s2, m2;
            get_scale_min_k4(is + 0, sc, &s1, &m1);
            get_scale_min_k4(is + 1, sc, &s2, &m2);
            float d1 = d * s1, mm1 = dmin * m1;
            float d2 = d * s2, mm2 = dmin * m2;
            for (int l = 0; l < 32; l++) {
                MV_FMA(d1 * (float)((q[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - mm1, base + j + l);
                MV_FMA(d2 * (float)((q[l] >> 4)  + ((qh[l] & u2) ? 16 : 0)) - mm2, base + j + l + 32);
            }
            q += 32; is += 2; u1 <<= 2; u2 <<= 2;
        }
    }
    MV_TAIL_B;
}

// Q5_K decode GEMV: Q4_K's lane-per-element geometry with Q5_K's extra high
// bit. Lane l owns eight consecutive elements in scale/min group l/4. The
// nibble comes from quant segment l/8 and the fifth bit is bit l/4 of qh.
// A warp therefore processes each 256-element block cooperatively, coalescing
// the 128-byte qs region and 32-byte qh region instead of reading 176-byte-
// strided blocks across lanes.
// v2 (decode-bandwidth pass): the Q4_K v2 load shape — one aligned uint2 for
// the 8 nibbles, one aligned uint2 for the 8 qh bytes (blk+16+bbase is
// 8-aligned in the 176-byte block), two float4 x loads, and the factored
// s += dg*sum(qv*x) - mmg*sum(x). Weights per element unchanged; identity vs
// CPU empirically gated by kernel-verify as before.
extern "C" __global__ void k_gemv_q5_K(MV_PARAMS) {
    MV_HEAD;
    int nb = a.n_in / 256;
    const uchar *rw = wb + a.w_off + (ulong64)row * nb * 176;
    int g     = (int)(lane >> 2);         // scale/min group 0..7
    int ji    = (int)(lane >> 3);         // 32-byte quant segment 0..3
    int sh    = ((((int)lane >> 2) & 1) == 0) ? 0 : 4;
    int bbase = ((int)lane & 3) * 8;      // byte offset within segment/qh
    int hshift = g;                       // qh bit for this group
    float s = 0;
    for (int b = 0; b < nb; b++) {
        const uchar *blk = rw + (ulong64)b * 176;
        float dd   = f16f(blk);
        float dmin = f16f(blk + 2);
        uchar sg, mg;
        get_scale_min_k4(g, blk + 4, &sg, &mg);
        float dg = dd * (float)sg, mmg = dmin * (float)mg;
        uint2 qv = *(const uint2 *)(blk + 48 + ji * 32 + bbase);
        uint2 hv = *(const uint2 *)(blk + 16 + bbase);
        const float *xp = x + (ulong64)b * 256 + (int)lane * 8;
        float4 x0 = *(const float4 *)xp, x1 = *(const float4 *)(xp + 4);
        uint v0 = (qv.x >> sh) & 0x0F0F0F0Fu, v1 = (qv.y >> sh) & 0x0F0F0F0Fu;
        // fifth bit: bit `g` of each qh byte, moved to value 16
        uint h0 = ((hv.x >> hshift) & 0x01010101u) << 4;
        uint h1 = ((hv.y >> hshift) & 0x01010101u) << 4;
        v0 += h0; v1 += h1;
        float t  = (float)(v0 & 0xFF)         * x0.x
                 + (float)((v0 >>  8) & 0xFF) * x0.y
                 + (float)((v0 >> 16) & 0xFF) * x0.z
                 + (float)((v0 >> 24)       ) * x0.w
                 + (float)(v1 & 0xFF)         * x1.x
                 + (float)((v1 >>  8) & 0xFF) * x1.y
                 + (float)((v1 >> 16) & 0xFF) * x1.z
                 + (float)((v1 >> 24)       ) * x1.w;
        float sx = x0.x + x0.y + x0.z + x0.w + x1.x + x1.y + x1.z + x1.w;
        s += dg * t - mmg * sx;
    }
    MV_TAIL;
}

// Q5_K prefill GEMM: the decode geometry above with the current x block staged
// in shared memory. Eight warps reuse that tile for eight output rows; each
// warp reduces one 256-element weight block cooperatively.
extern "C" __global__ void k_gemm_q5_K(MV_PARAMS) {
    __shared__ float xsm[MVT][256];
    unsigned warp = threadIdx.x >> 5;
    unsigned lane = threadIdx.x & 31;
    unsigned row  = blockIdx.x * GEMM_WARPS + warp;
    int nb = a.n_in / 256;
    const uchar *rw = wb + a.w_off +
                      (ulong64)(row < (unsigned)a.n_out ? row : 0) * nb * 176;
    int g     = (int)(lane >> 2);         // scale/min group 0..7
    int ji    = (int)(lane >> 3);         // 32-byte quant segment 0..3
    int lo    = (((int)lane >> 2) & 1) == 0;
    int bbase = ((int)lane & 3) * 8;      // byte offset within segment/qh
    int hmask = 1 << g;
    float s[MVT] = {0};

    for (int b = 0; b < nb; b++) {
        const uchar *blk = rw + (ulong64)b * 176;
        int base_e = b * 256;
        #pragma unroll
        for (int t = 0; t < MVT; t++) {
            const float *xg = x + (ulong64)t * a.xs + base_e;
            for (int e = threadIdx.x; e < 256; e += blockDim.x) xsm[t][e] = xg[e];
        }
        __syncthreads();
        if (row < (unsigned)a.n_out) {
            float dd   = f16f(blk);
            float dmin = f16f(blk + 2);
            const uchar *sc = blk + 4;
            const uchar *qh = blk + 16;
            const uchar *q  = blk + 48 + ji * 32;
            uchar sg, mg;
            get_scale_min_k4(g, sc, &sg, &mg);
            float dg = dd * (float)sg, mmg = dmin * (float)mg;
            int el = (int)lane * 8;
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                uchar byte = q[bbase + k];
                int qv = (lo ? (byte & 0xF) : (byte >> 4)) +
                         ((qh[bbase + k] & hmask) ? 16 : 0);
                float w = dg * (float)qv - mmg;
                #pragma unroll
                for (int t = 0; t < MVT; t++) s[t] += w * xsm[t][el + k];
            }
        }
        __syncthreads();
    }
    if (row < (unsigned)a.n_out) {
        for (int t = 0; t < a.batch; t++) {
            float r = warp_sum(s[t]);
            if (lane == 0) y[(ulong64)t * a.ys + row] = a.has_bias ? r + bias[row] : r;
        }
    }
}

extern "C" __global__ void k_mv_q6_K(MV_PARAMS) {
    MV_HEAD;
    int nb = a.n_in / 256;
    const uchar *rw = wb + a.w_off + (ulong64)row * nb * 210;
    float s = 0;
    for (int b = lane; b < nb; b += 32) {
        const uchar *blk = rw + (ulong64)b * 210;
        const uchar *ql = blk;
        const uchar *qh = blk + 128;
        const signed char *sc = (const signed char *)(blk + 192);
        float d = f16f(blk + 208);
        const float *xp = x + b * 256;
        for (int half_i = 0; half_i < 2; half_i++) {
            float t[8] = {0, 0, 0, 0, 0, 0, 0, 0};
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
    MV_TAIL;
}

extern "C" __global__ void k_mv_q6_K_b(MV_PARAMS) {
    MV_HEAD_B;
    int nb = a.n_in / 256;
    const uchar *rw = wb + a.w_off + (ulong64)row * nb * 210;
    for (int b = lane; b < nb; b += 32) {
        const uchar *blk = rw + (ulong64)b * 210;
        const uchar *ql = blk;
        const uchar *qh = blk + 128;
        const signed char *sc = (const signed char *)(blk + 192);
        float d = f16f(blk + 208);
        ulong64 base = (ulong64)b * 256;
        for (int half_i = 0; half_i < 2; half_i++) {
            for (int l = 0; l < 32; l++) {
                int is = (l / 16) & 1;
                int q1 = (int)((ql[l]      & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int q2 = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int q3 = (int)((ql[l]      >> 4)  | (((qh[l] >> 4) & 3) << 4)) - 32;
                int q4 = (int)((ql[l + 32] >> 4)  | (((qh[l] >> 6) & 3) << 4)) - 32;
                MV_FMA(d * (float)(sc[is] * q1),     base + l);
                MV_FMA(d * (float)(sc[2 + is] * q2), base + l + 32);
                MV_FMA(d * (float)(sc[4 + is] * q3), base + l + 64);
                MV_FMA(d * (float)(sc[6 + is] * q4), base + l + 96);
            }
            ql += 64; qh += 32; sc += 8; base += 128;
        }
    }
    MV_TAIL_B;
}

// Q6_K decode GEMV: lane-per-element coalesced variant of k_mv_q6_K. The
// generic k_mv_q6_K maps one lane to a whole 210-byte block, so consecutive
// lanes read 210-byte-strided addresses (uncoalesced). This variant makes the
// warp process each block cooperatively: lane l owns the four sub-positions
// {l, l+32, l+64, l+96} within each of the block's two 128-element halves (8
// elements total). Then ql[l]/ql[l+32]/qh[l] each read 32 consecutive bytes
// across the warp -> coalesced. Per-element weight d*sc[base+is]*q matches
// k_mv_q6_K exactly; only the k-reduction is reordered (per-lane partials +
// one warp_sum), so identity is verified empirically by kernel-verify.
// v2 (decode-bandwidth pass): same cooperative geometry — the per-lane loads
// already coalesce across the warp — but two blocks per iteration with
// independent accumulators so twice as many loads are in flight per loop trip
// (the 210-byte stride is only 2-aligned, so wider per-lane loads are not
// available). Identity vs CPU stays empirically gated by kernel-verify.
static __device__ __forceinline__ float q6k_block_dot(const uchar *blk,
                                                      const float *xb,
                                                      int lane, int is) {
    float d = f16f(blk + 208);
    float acc = 0;
    #pragma unroll
    for (int half = 0; half < 2; half++) {
        const uchar *ql = blk + half * 64;
        const uchar *qh = blk + 128 + half * 32;
        const signed char *sc = (const signed char *)(blk + 192) + half * 8;
        int q1 = (int)((ql[lane]      & 0xF) | (((qh[lane] >> 0) & 3) << 4)) - 32;
        int q2 = (int)((ql[lane + 32] & 0xF) | (((qh[lane] >> 2) & 3) << 4)) - 32;
        int q3 = (int)((ql[lane]      >> 4)  | (((qh[lane] >> 4) & 3) << 4)) - 32;
        int q4 = (int)((ql[lane + 32] >> 4)  | (((qh[lane] >> 6) & 3) << 4)) - 32;
        const float *xp = xb + half * 128;
        acc += d * ((float)(sc[0 + is] * q1) * xp[lane] +
                    (float)(sc[2 + is] * q2) * xp[lane + 32] +
                    (float)(sc[4 + is] * q3) * xp[lane + 64] +
                    (float)(sc[6 + is] * q4) * xp[lane + 96]);
    }
    return acc;
}

extern "C" __global__ void k_gemv_q6_K(MV_PARAMS) {
    MV_HEAD;
    int nb = a.n_in / 256;
    const uchar *rw = wb + a.w_off + (ulong64)row * nb * 210;
    int is = (int)(lane >> 4);          // (lane/16)&1 for lane 0..31 -> 0 or 1
    float s0 = 0, s1 = 0;
    int b2 = nb & ~1;
    for (int b = 0; b < b2; b += 2) {
        s0 += q6k_block_dot(rw + (ulong64)b * 210,       x + (ulong64)b * 256,
                            (int)lane, is);
        s1 += q6k_block_dot(rw + (ulong64)(b + 1) * 210, x + (ulong64)(b + 1) * 256,
                            (int)lane, is);
    }
    if (b2 < nb)
        s0 += q6k_block_dot(rw + (ulong64)b2 * 210, x + (ulong64)b2 * 256,
                            (int)lane, is);
    float s = s0 + s1;
    MV_TAIL;
}

// Q6_K prefill GEMM: same shared-memory x staging as k_gemm_q4_K, with the
// lane-per-element geometry of k_gemv_q6_K. Warp walks blocks sequentially;
// its 32 lanes cooperatively reduce the 256 elements of each block (8/lane,
// positions {l,l+32,l+64,l+96} per half). x for the current block is staged in
// smem so decoded weights FMA against smem. Reordered k-reduction vs
// k_mv_q6_K_b -> token identity verified empirically.
extern "C" __global__ void k_gemm_q6_K(MV_PARAMS) {
    __shared__ float xsm[MVT][256];
    unsigned warp = threadIdx.x >> 5;
    unsigned lane = threadIdx.x & 31;
    unsigned row  = blockIdx.x * GEMM_WARPS + warp;
    int nb = a.n_in / 256;
    const uchar *rw = wb + a.w_off +
                      (ulong64)(row < (unsigned)a.n_out ? row : 0) * nb * 210;
    int is = (int)(lane >> 4);
    float s[MVT] = {0};

    for (int b = 0; b < nb; b++) {
        const uchar *blk = rw + (ulong64)b * 210;
        int base_e = b * 256;
        #pragma unroll
        for (int t = 0; t < MVT; t++) {
            const float *xg = x + (ulong64)t * a.xs + base_e;
            for (int e = threadIdx.x; e < 256; e += blockDim.x) xsm[t][e] = xg[e];
        }
        __syncthreads();
        if (row < (unsigned)a.n_out) {
            float d = f16f(blk + 208);
            #pragma unroll
            for (int half = 0; half < 2; half++) {
                const uchar *ql = blk + half * 64;
                const uchar *qh = blk + 128 + half * 32;
                const signed char *sc = (const signed char *)(blk + 192) + half * 8;
                int q1 = (int)((ql[lane]      & 0xF) | (((qh[lane] >> 0) & 3) << 4)) - 32;
                int q2 = (int)((ql[lane + 32] & 0xF) | (((qh[lane] >> 2) & 3) << 4)) - 32;
                int q3 = (int)((ql[lane]      >> 4)  | (((qh[lane] >> 4) & 3) << 4)) - 32;
                int q4 = (int)((ql[lane + 32] >> 4)  | (((qh[lane] >> 6) & 3) << 4)) - 32;
                float w1 = d * (float)(sc[0 + is] * q1);
                float w2 = d * (float)(sc[2 + is] * q2);
                float w3 = d * (float)(sc[4 + is] * q3);
                float w4 = d * (float)(sc[6 + is] * q4);
                int e0 = half * 128;
                #pragma unroll
                for (int t = 0; t < MVT; t++) {
                    const float *xs = xsm[t];
                    s[t] += w1 * xs[e0 + lane]      + w2 * xs[e0 + lane + 32] +
                            w3 * xs[e0 + lane + 64] + w4 * xs[e0 + lane + 96];
                }
            }
        }
        __syncthreads();
    }
    if (row < (unsigned)a.n_out) {
        for (int t = 0; t < a.batch; t++) {
            float r = warp_sum(s[t]);
            if (lane == 0) y[(ulong64)t * a.ys + row] = a.has_bias ? r + bias[row] : r;
        }
    }
}

// IQ4: the nibble indexes a fixed 16-entry codebook
static __device__ const signed char kv_iq4[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
};

extern "C" __global__ void k_mv_iq4_nl(MV_PARAMS) {
    MV_HEAD;
    int nb = a.n_in / 32;
    const uchar *rw = wb + a.w_off + (ulong64)row * nb * 18;
    float s = 0;
    for (int b = lane; b < nb; b += 32) {
        const uchar *blk = rw + (ulong64)b * 18;
        float d = f16f(blk);
        const uchar *q = blk + 2;
        const float *xp = x + b * 32;
        float t = 0;
        for (int j = 0; j < 16; j++) {
            t += (float)kv_iq4[q[j] & 0xF] * xp[j];
            t += (float)kv_iq4[q[j] >> 4]  * xp[j + 16];
        }
        s += d * t;
    }
    MV_TAIL;
}

extern "C" __global__ void k_mv_iq4_nl_b(MV_PARAMS) {
    MV_HEAD_B;
    int nb = a.n_in / 32;
    const uchar *rw = wb + a.w_off + (ulong64)row * nb * 18;
    for (int b = lane; b < nb; b += 32) {
        const uchar *blk = rw + (ulong64)b * 18;
        float d = f16f(blk);
        const uchar *q = blk + 2;
        ulong64 base = (ulong64)b * 32;
        for (int j = 0; j < 16; j++) {
            MV_FMA(d * (float)kv_iq4[q[j] & 0xF], base + j);
            MV_FMA(d * (float)kv_iq4[q[j] >> 4],  base + j + 16);
        }
    }
    MV_TAIL_B;
}

extern "C" __global__ void k_mv_iq4_xs(MV_PARAMS) {
    MV_HEAD;
    int nb = a.n_in / 256;
    const uchar *rw = wb + a.w_off + (ulong64)row * nb * 136;
    float s = 0;
    for (int b = lane; b < nb; b += 32) {
        const uchar *blk = rw + (ulong64)b * 136;
        float d = f16f(blk);
        unsigned sh = (unsigned)blk[2] | ((unsigned)blk[3] << 8);
        const uchar *sl = blk + 4;
        const uchar *q  = blk + 8;
        const float *xp = x + b * 256;
        for (int ib = 0; ib < 8; ib++) {
            int ls = ((sl[ib / 2] >> 4 * (ib % 2)) & 0xF) | (((sh >> 2 * ib) & 3) << 4);
            float dl = d * (ls - 32);
            float t = 0;
            for (int j = 0; j < 16; j++) {
                t += (float)kv_iq4[q[j] & 0xF] * xp[j];
                t += (float)kv_iq4[q[j] >> 4]  * xp[j + 16];
            }
            s += dl * t;
            q += 16; xp += 32;
        }
    }
    MV_TAIL;
}

extern "C" __global__ void k_mv_iq4_xs_b(MV_PARAMS) {
    MV_HEAD_B;
    int nb = a.n_in / 256;
    const uchar *rw = wb + a.w_off + (ulong64)row * nb * 136;
    for (int b = lane; b < nb; b += 32) {
        const uchar *blk = rw + (ulong64)b * 136;
        float d = f16f(blk);
        unsigned sh = (unsigned)blk[2] | ((unsigned)blk[3] << 8);
        const uchar *sl = blk + 4;
        const uchar *q  = blk + 8;
        ulong64 base = (ulong64)b * 256;
        for (int ib = 0; ib < 8; ib++) {
            int ls = ((sl[ib / 2] >> 4 * (ib % 2)) & 0xF) | (((sh >> 2 * ib) & 3) << 4);
            float dl = d * (ls - 32);
            for (int j = 0; j < 16; j++) {
                MV_FMA(dl * (float)kv_iq4[q[j] & 0xF], base + ib * 32 + j);
                MV_FMA(dl * (float)kv_iq4[q[j] >> 4],  base + ib * 32 + j + 16);
            }
            q += 16;
        }
    }
    MV_TAIL_B;
}

// MXFP4 (gpt-oss expert tensors): 17-byte block = one E8M0 scale byte (a
// biased power-of-two exponent, 2^(e-127), NOT an fp16) + 32 packed E2M1
// nibbles indexing a fixed signed codebook. Table and decode are 1:1 with
// dq_mxfp4 in quants.c — ldexpf keeps 2^(e-127) exact down into the
// subnormal range, where exp2f of a float could flush to zero.
static __device__ const float kv_mxfp4[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
     0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

extern "C" __global__ void k_mv_mxfp4(MV_PARAMS) {
    MV_HEAD;
    int nb = a.n_in / 32;
    const uchar *rw = wb + a.w_off + (ulong64)row * nb * 17;
    float s = 0;
    for (int b = lane; b < nb; b += 32) {
        const uchar *blk = rw + (ulong64)b * 17;
        float d = ldexpf(1.0f, (int)blk[0] - 127);
        const uchar *q = blk + 1;
        const float *xp = x + b * 32;
        float t = 0;
        for (int j = 0; j < 16; j++) {
            t += kv_mxfp4[q[j] & 0xF] * xp[j];
            t += kv_mxfp4[q[j] >> 4]  * xp[j + 16];
        }
        s += d * t;
    }
    MV_TAIL;
}

extern "C" __global__ void k_mv_mxfp4_b(MV_PARAMS) {
    MV_HEAD_B;
    int nb = a.n_in / 32;
    const uchar *rw = wb + a.w_off + (ulong64)row * nb * 17;
    for (int b = lane; b < nb; b += 32) {
        const uchar *blk = rw + (ulong64)b * 17;
        float d = ldexpf(1.0f, (int)blk[0] - 127);
        const uchar *q = blk + 1;
        ulong64 base = (ulong64)b * 32;
        for (int j = 0; j < 16; j++) {
            MV_FMA(d * kv_mxfp4[q[j] & 0xF], base + j);
            MV_FMA(d * kv_mxfp4[q[j] >> 4],  base + j + 16);
        }
    }
    MV_TAIL_B;
}

// ---------------------------------------------------------------- rope
// grid: (ceil(half_dim/32), n_heads, batch); vs = element stride per column


extern "C" __global__ void k_rope(float *v, const float *fr, rope_args a,
                                  const int *posp, int vs) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y;
    if (j >= a.half_dim || h >= a.n_heads) return;
    int pos = *posp + blockIdx.z;
    float ang = pos * fr[j];
    float c = cosf(ang) * a.mscale, s = sinf(ang) * a.mscale;
    float *p = v + (ulong64)blockIdx.z * vs + h * a.head_dim;
    int i0 = a.neox ? j : 2 * j;
    int i1 = a.neox ? j + a.half_dim : i0 + 1;
    float x0 = p[i0], x1 = p[i1];
    p[i0] = x0 * c - x1 * s;
    p[i1] = x0 * s + x1 * c;
}

// ------------------------------------------------------------- kv storage
// The cache is either fp16 (2 bytes/value) or q8_0 (32 values per 34-byte
// block: one fp16 scale + 32 int8 quants), selected at load and identical in
// layout to the CPU cache so the two paths share the same host buffer. All
// cache offsets below are therefore BYTE offsets, and the pointers are byte
// pointers: a q8_0 row is only 2-byte aligned, so no wider load is legal.
//
// q8_0 quantization here is the same arithmetic as q8_quant_row() in quants.c
// (amax/127 scale, round-half-away-from-zero, RN fp16 scale), so a row
// quantized on the GPU is bit-identical to the same row quantized on the CPU.

struct q8_blk { __half d; signed char qs[32]; };   // 34 bytes, 2-byte aligned

#define KV_ROW_BYTES(kv_dim, q8) \
    ((q8) ? (ulong64)((kv_dim) / 32) * 34 : (ulong64)(kv_dim) * 2)

__device__ __forceinline__ void kv_store_row(unsigned char *cache,
                                             const float *src, int kv_dim,
                                             int q8, int i) {
    if (q8) {
        q8_blk *b = (q8_blk *)(cache + (ulong64)i * 34);
        const float *x = src + i * 32;
        float amax = 0;
        for (int j = 0; j < 32; j++) amax = fmaxf(amax, fabsf(x[j]));
        float d  = amax / 127.0f;
        float id = d > 0 ? 1.0f / d : 0.0f;
        b->d = __float2half(d);
        for (int j = 0; j < 32; j++) b->qs[j] = (signed char)roundf(x[j] * id);
    } else {
        ((__half *)cache)[i] = __float2half(src[i]);
    }
}

// q * k for one head: paired accumulation, mirroring the fp16 path and
// vec_dot(T_Q8_0) in quants.c (per-block int sum, then scaled)
__device__ __forceinline__ float kv_dot(const unsigned char *row,
                                        const float *qh, int hd, int q8) {
    float s = 0;
    if (q8) {
        for (int b = 0; b < hd / 32; b++) {
            const q8_blk *blk = (const q8_blk *)(row + (ulong64)b * 34);
            const float *xp = qh + b * 32;
            float t = 0;
            for (int j = 0; j < 32; j += 2)
                t += xp[j] * blk->qs[j] + xp[j + 1] * blk->qs[j + 1];
            s += __half2float(blk->d) * t;
        }
    } else {
        const __half2 *k2 = (const __half2 *)row;
        for (int i = 0; i < hd / 2; i++) {
            float2 kf = __half22float2(k2[i]);
            // paired add reassociates FP vs. a sequential accumulation;
            // temp-0 gate covered it on tested models
            s += qh[2 * i] * kf.x + qh[2 * i + 1] * kf.y;
        }
    }
    return s;
}

// the value pair at element offset 2*i2 of one head's row. Element pairs never
// straddle a q8 block (32 is even), so one block lookup serves both.
__device__ __forceinline__ float2 kv_pair(const unsigned char *row,
                                          int i2, int q8) {
    if (q8) {
        const q8_blk *blk = (const q8_blk *)(row + (ulong64)(i2 / 16) * 34);
        float d = __half2float(blk->d);
        int j = (2 * i2) & 31;
        return make_float2(d * blk->qs[j], d * blk->qs[j + 1]);
    }
    return __half22float2(((const __half2 *)row)[i2]);
}

// grid.y = token column; cache rows for consecutive positions are contiguous

extern "C" __global__ void k_store_kv(const float *k, const float *v,
                                      unsigned char *kc, unsigned char *vc,
                                      int kv_dim, ulong64 l_off,
                                      const int *posp, int q8) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n = q8 ? kv_dim / 32 : kv_dim;
    if (i < n) {
        ulong64 row_b = KV_ROW_BYTES(kv_dim, q8);
        ulong64 dst = l_off + (ulong64)(*posp + blockIdx.y) * row_b;
        const float *ks = k + (ulong64)blockIdx.y * kv_dim;
        const float *vs = v + (ulong64)blockIdx.y * kv_dim;
        kv_store_row(kc + dst, ks, kv_dim, q8, i);
        kv_store_row(vc + dst, vs, kv_dim, q8, i);
    }
}

// ---------------------------------------------------------------- attention
// One block per (head, token): scores -> softmax -> weighted value sum.
// att scratch is MVB planes of [n_head][n_ctx].


// byte offset of head kvh's slice within a cache row
__device__ __forceinline__ ulong64 kv_head_off(int kvh, int hd, int q8) {
    return q8 ? (ulong64)(kvh * hd / 32) * 34 : (ulong64)(kvh * hd) * 2;
}

// sinks: gpt-oss per-head learned attention-sink logits for this layer, or
// NULL. Transcribed from softmax_sink() in model.c: the sink joins the max
// scan and the denominator but has NO value row, so the probabilities over
// real positions sum to < 1 and the head's output shrinks. The sink competes
// against ALREADY-SCALED scores; it is not itself scaled.
// Position-chunk cap for the V accumulation. nchunk*lanes never exceeds the
// block, so the partial buffer is bounded at tpg*2 floats regardless.
#define ATTN_VMAX 8

extern "C" __global__ void k_attn(const float *q, const unsigned char *kc,
                                  const unsigned char *vc, float *att, float *out,
                                  attn_args a, const int *posp,
                                  const float *sinks) {
    __shared__ float red[256];
    __shared__ float vpart[256 * 2];   // nchunk*lanes <= tpg <= 256
    int h = blockIdx.x, tid = threadIdx.x, tpg = blockDim.x;
    int tk = blockIdx.y;                 // token column in the tile
    int pos = *posp + tk;
    int hd = a.head_dim;
    int kvh = h / (a.n_head / a.n_head_kv);
    int kv_dim = a.n_head_kv * hd;
    ulong64 row_b = KV_ROW_BYTES(kv_dim, a.q8);
    ulong64 base  = a.l_off + kv_head_off(kvh, hd, a.q8);
    int t0 = 0;                          // sliding-window start
    if (a.window > 0 && pos - a.window + 1 > 0) t0 = pos - a.window + 1;
    const float *qh = q + (ulong64)tk * a.qs + h * hd;
    float *ah = att + ((ulong64)tk * a.n_head + h) * a.n_ctx;

    for (int t = t0 + tid; t <= pos; t += tpg)
        ah[t] = kv_dot(kc + base + (ulong64)t * row_b, qh, hd, a.q8) * a.scale;
    __syncthreads();

    // max
    float mx = -1e30f;
    for (int t = t0 + tid; t <= pos; t += tpg) mx = fmaxf(mx, ah[t]);
    red[tid] = mx;
    __syncthreads();
    for (int off = tpg / 2; off > 0; off >>= 1) {
        if (tid < off) red[tid] = fmaxf(red[tid], red[tid + off]);
        __syncthreads();
    }
    mx = red[0];
    if (sinks && sinks[h] > mx) mx = sinks[h];
    __syncthreads();
    // exp + sum
    float sum = 0;
    for (int t = t0 + tid; t <= pos; t += tpg) {
        float e = expf(ah[t] - mx);
        ah[t] = e;
        sum += e;
    }
    red[tid] = sum;
    __syncthreads();
    for (int off = tpg / 2; off > 0; off >>= 1) {
        if (tid < off) red[tid] += red[tid + off];
        __syncthreads();
    }
    sum = red[0];
    if (sinks) sum += expf(sinks[h] - mx);
    __syncthreads();

    // V accumulation, spread over the whole block. `lanes` float2 dims x
    // `nchunk` disjoint position strides; each thread owns one (lane, chunk)
    // pair, and the partials are summed per lane below.
    const int lanes  = hd / 2;
    int nchunk = lanes > 0 ? tpg / lanes : 1;
    if (nchunk < 1) nchunk = 1;
    if (nchunk > ATTN_VMAX) nchunk = ATTN_VMAX;
    const int vlane  = lanes > 0 ? tid % lanes : 0;
    const int vchunk = lanes > 0 ? tid / lanes : 0;

    // Parallel path only when every lane is covered by the block. With
    // lanes > tpg (head_dim > 512, which gemma4 can reach) the chunking would
    // leave the upper dims unwritten, so that case keeps the original loop.
    if (lanes > 0 && lanes <= tpg) {
    if (vchunk < nchunk && vlane < lanes) {
        float o0 = 0, o1 = 0;
        for (int t = t0 + vchunk; t <= pos; t += nchunk) {
            float2 vf = kv_pair(vc + base + (ulong64)t * row_b, vlane, a.q8);
            o0 += ah[t] * vf.x;
            o1 += ah[t] * vf.y;
        }
        vpart[(vchunk * lanes + vlane) * 2 + 0] = o0;
        vpart[(vchunk * lanes + vlane) * 2 + 1] = o1;
    }
    __syncthreads();
    for (int i2 = tid; i2 < lanes; i2 += tpg) {
        float s0 = 0, s1 = 0;
        for (int c = 0; c < nchunk; c++) {
            s0 += vpart[(c * lanes + i2) * 2 + 0];
            s1 += vpart[(c * lanes + i2) * 2 + 1];
        }
        out[(ulong64)tk * a.os + h * hd + 2 * i2]     = s0 / sum;
        out[(ulong64)tk * a.os + h * hd + 2 * i2 + 1] = s1 / sum;
    }
    } else {
        for (int i2 = tid; i2 < lanes; i2 += tpg) {
            float o0 = 0, o1 = 0;
            for (int t = t0; t <= pos; t++) {
                float2 vf = kv_pair(vc + base + (ulong64)t * row_b, i2, a.q8);
                o0 += ah[t] * vf.x;
                o1 += ah[t] * vf.y;
            }
            out[(ulong64)tk * a.os + h * hd + 2 * i2]     = o0 / sum;
            out[(ulong64)tk * a.os + h * hd + 2 * i2 + 1] = o1 / sum;
        }
    }
}

// --------------------------------------------------- flash-decoding attention
// Decode (batch==1, one query token, long KV) attention. The plain k_attn runs
// one block per (head, token): a 4B decode step is 32 blocks on a 46-SM GPU,
// each re-reading the whole fp16 KV cache serially. These two kernels split the
// KV range across ATTN_SPLITS blocks per head (fixed, compile-time constant, so
// the CUDA graph stays valid across positions) and merge the partials:
//
//   k_attn_dec  : grid (n_head, ATTN_SPLITS, tn). Each (head, split) block
//                 computes softmax over its on-device-computed KV slice with
//                 the SAME within-slice reduction structure as k_attn (paired
//                 q*k, strided-then-tree max/sum, sequential weighted-V), and
//                 writes an un-normalised partial: weighted-V + local max +
//                 local sum. Empty slices write a -inf-max sentinel.
//   k_attn_merge: grid (n_head, tn). Combines the ATTN_SPLITS partials with a
//                 global max and the standard exp(m_j - M) rescale, divides by
//                 the merged sum, writes out. Merge order is fixed (0..SPLITS)
//                 so it is deterministic across positions.
//
// The cross-slice merge reassociates the softmax sum (extra exp(m_j - M) and a
// regrouped add) relative to k_attn's single global reduction, so identity is
// not bitwise and is verified empirically by kernel-verify. Within a slice the
// order is preserved. Partials scratch layout per (tk, head, split):
//   [0..hd)  un-normalised weighted V ; [hd] local max ; [hd+1] local sum.

#define ATTN_SPLITS 8

// LAUNCH INVARIANT: blockDim.x must be a power of two and <= 128 (red[]'s
// size) for the tree reduction and shared indexing here. Host launches with 128.
extern "C" __global__ void k_attn_dec(const float *q, const unsigned char *kc,
                                      const unsigned char *vc, float *att, float *part,
                                      attn_args a, const int *posp) {
    __shared__ float red[128];
    int h = blockIdx.x, sp = blockIdx.y, tk = blockIdx.z;
    int tid = threadIdx.x, tpg = blockDim.x;
    int pos = *posp + tk;
    int hd = a.head_dim;
    int kvh = h / (a.n_head / a.n_head_kv);
    int kv_dim = a.n_head_kv * hd;
    ulong64 row_b = KV_ROW_BYTES(kv_dim, a.q8);
    ulong64 base  = a.l_off + kv_head_off(kvh, hd, a.q8);
    int t0 = 0;
    if (a.window > 0 && pos - a.window + 1 > 0) t0 = pos - a.window + 1;
    int total = pos + 1 - t0;
    int slice = (total + ATTN_SPLITS - 1) / ATTN_SPLITS;     // on-device from pos
    int s0 = t0 + sp * slice;
    int s1 = s0 + slice;
    if (s1 > pos + 1) s1 = pos + 1;
    float *P = part + (((ulong64)tk * a.n_head + h) * ATTN_SPLITS + sp) * (hd + 2);
    if (s0 >= s1) {                       // empty slice: sentinel, skipped in merge
        if (tid == 0) { P[hd] = -1e30f; P[hd + 1] = 0.f; }
        return;
    }
    const float *qh = q + (ulong64)tk * a.qs + h * hd;
    float *ah = att + ((ulong64)tk * a.n_head + h) * a.n_ctx;

    for (int t = s0 + tid; t < s1; t += tpg)
        ah[t] = kv_dot(kc + base + (ulong64)t * row_b, qh, hd, a.q8) * a.scale;
    __syncthreads();
    float mx = -1e30f;
    for (int t = s0 + tid; t < s1; t += tpg) mx = fmaxf(mx, ah[t]);
    red[tid] = mx;
    __syncthreads();
    for (int off = tpg / 2; off > 0; off >>= 1) {
        if (tid < off) red[tid] = fmaxf(red[tid], red[tid + off]);
        __syncthreads();
    }
    mx = red[0];
    __syncthreads();
    float sum = 0;
    for (int t = s0 + tid; t < s1; t += tpg) {
        float e = expf(ah[t] - mx);
        ah[t] = e;
        sum += e;
    }
    red[tid] = sum;
    __syncthreads();
    for (int off = tpg / 2; off > 0; off >>= 1) {
        if (tid < off) red[tid] += red[tid + off];
        __syncthreads();
    }
    sum = red[0];
    __syncthreads();
    for (int i2 = tid; i2 < hd / 2; i2 += tpg) {
        float o0 = 0, o1 = 0;
        for (int t = s0; t < s1; t++) {
            float2 vf = kv_pair(vc + base + (ulong64)t * row_b, i2, a.q8);
            o0 += ah[t] * vf.x;
            o1 += ah[t] * vf.y;
        }
        P[2 * i2]     = o0;               // un-normalised (merge divides by sum)
        P[2 * i2 + 1] = o1;
    }
    if (tid == 0) { P[hd] = mx; P[hd + 1] = sum; }
}

// sinks: same contract as k_attn. The sink joins at the GLOBAL reduction —
// it competes in the merged max and adds one exp term to the merged
// denominator, and since it has no value row the numerator is untouched, so
// the split partials (k_attn_dec / k_attn_dec_seq) need no sink awareness.
extern "C" __global__ void k_attn_merge(float *out, const float *part,
                                        attn_args a, const int *posp,
                                        const float *sinks) {
    int h = blockIdx.x, tk = blockIdx.y;
    int tid = threadIdx.x, tpg = blockDim.x;
    int hd = a.head_dim;
    const float *base = part + ((ulong64)tk * a.n_head + h) * ATTN_SPLITS * (hd + 2);
    float M = -1e30f;
    for (int sp = 0; sp < ATTN_SPLITS; sp++)
        M = fmaxf(M, base[sp * (hd + 2) + hd]);
    if (sinks && sinks[h] > M) M = sinks[h];
    float L = 0.f;
    for (int sp = 0; sp < ATTN_SPLITS; sp++) {
        float m = base[sp * (hd + 2) + hd];
        if (m <= -1e29f) continue;
        L += base[sp * (hd + 2) + hd + 1] * expf(m - M);
    }
    if (sinks) L += expf(sinks[h] - M);
    for (int i = tid; i < hd; i += tpg) {
        float acc = 0.f;
        for (int sp = 0; sp < ATTN_SPLITS; sp++) {
            const float *P = base + sp * (hd + 2);
            float m = P[hd];
            if (m <= -1e29f) continue;
            acc += P[i] * expf(m - M);
        }
        out[(ulong64)tk * a.os + h * hd + i] = acc / L;
    }
}

// ============================================================================
// Batched decode: one token for each of N *independent* sequences (Phase 6)
// ============================================================================
//
// The prefill tile kernels above batch N tokens of ONE sequence: consecutive
// positions, one KV cache, so a single base position and a single cache
// pointer describe the whole tile. Continuous batching needs the other shape
// — N tokens of N DIFFERENT sequences, each at its own position, each writing
// and reading its own KV region — and that is the only thing the kernels below
// change. Every column is still computed exactly as a lone token would be.
//
// Two mechanical differences from the tile kernels, and nothing else:
//
//   posp is an ARRAY indexed by the token column, not a base + column offset.
//   kcp/vcp are ARRAYS of device pointers, one KV cache per sequence, so no
//   sequence's rows are reachable from another's column.
//
// The numerical contract is the point. `k_gemv_*_b` below decode each weight
// once and FMA it into MODEL_BATCH_MAX accumulators, in the same lane mapping
// and the same warp-reduction tree as the batch-1 `k_gemv_*` they twin — so
// column t's result is BITWISE what k_gemv_* computes for that column alone.
// That is why cuda.c pairs each batched kernel with the batch-1 kernel it
// mirrors rather than reusing the prefill GEMMs, which are faster and would
// reassociate. Identity is not an accident here; it is the selection rule —
// and when a k_gemv_* body is rewritten, its twin below must be rewritten with
// it or the rule quietly stops holding (it did, for three weeks in 2026-07).
// What holds it down is tests/test_batch.c run on a QUANTIZED model: `make
// test` runs it on test-q8.gguf for exactly that reason, because the F32
// fixture takes k_mv_f32/k_mv_f32_b and never touches this family at all.

// grid: (ceil(half_dim/32), n_heads, batch); pos per column
extern "C" __global__ void k_rope_seq(float *v, const float *fr, rope_args a,
                                      const int *posp, int vs) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y;
    if (j >= a.half_dim || h >= a.n_heads) return;
    int pos = posp[blockIdx.z];
    float ang = pos * fr[j];
    float c = cosf(ang) * a.mscale, s = sinf(ang) * a.mscale;
    float *p = v + (ulong64)blockIdx.z * vs + h * a.head_dim;
    int i0 = a.neox ? j : 2 * j;
    int i1 = a.neox ? j + a.half_dim : i0 + 1;
    float x0 = p[i0], x1 = p[i1];
    p[i0] = x0 * c - x1 * s;
    p[i1] = x0 * s + x1 * c;
}

// grid.y = sequence column; each column stores into its OWN cache at its OWN
// position, so two sequences at the same position never collide
extern "C" __global__ void k_store_kv_seq(const float *k, const float *v,
                                          const ulong64 *kcp, const ulong64 *vcp,
                                          int kv_dim, ulong64 l_off,
                                          const int *posp, int q8) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n = q8 ? kv_dim / 32 : kv_dim;
    if (i < n) {
        int sq = blockIdx.y;
        ulong64 row_b = KV_ROW_BYTES(kv_dim, q8);
        ulong64 dst = l_off + (ulong64)posp[sq] * row_b;
        const float *ks = k + (ulong64)sq * kv_dim;
        const float *vs = v + (ulong64)sq * kv_dim;
        kv_store_row((unsigned char *)kcp[sq] + dst, ks, kv_dim, q8, i);
        kv_store_row((unsigned char *)vcp[sq] + dst, vs, kv_dim, q8, i);
    }
}

// Flash-decoding attention over N sequences. Body is k_attn_dec verbatim with
// pos and the cache pointers taken per column; the within-slice reduction, the
// split count and the partial layout are untouched, so k_attn_merge (which
// reads neither position nor cache) serves this path unchanged and each
// column's result is bitwise what the unbatched decode produces.
// LAUNCH INVARIANT: blockDim.x must be a power of two and <= 128 (red[]'s
// size) for the tree reduction and shared indexing here. Host launches with 128.
extern "C" __global__ void k_attn_dec_seq(const float *q, const ulong64 *kcp,
                                          const ulong64 *vcp, float *att,
                                          float *part, attn_args a,
                                          const int *posp) {
    __shared__ float red[128];
    int h = blockIdx.x, sp = blockIdx.y, tk = blockIdx.z;
    int tid = threadIdx.x, tpg = blockDim.x;
    int pos = posp[tk];
    const unsigned char *kc = (const unsigned char *)kcp[tk];
    const unsigned char *vc = (const unsigned char *)vcp[tk];
    int hd = a.head_dim;
    int kvh = h / (a.n_head / a.n_head_kv);
    int kv_dim = a.n_head_kv * hd;
    ulong64 row_b = KV_ROW_BYTES(kv_dim, a.q8);
    ulong64 base  = a.l_off + kv_head_off(kvh, hd, a.q8);
    int t0 = 0;
    if (a.window > 0 && pos - a.window + 1 > 0) t0 = pos - a.window + 1;
    int total = pos + 1 - t0;
    int slice = (total + ATTN_SPLITS - 1) / ATTN_SPLITS;
    int s0 = t0 + sp * slice;
    int s1 = s0 + slice;
    if (s1 > pos + 1) s1 = pos + 1;
    float *P = part + (((ulong64)tk * a.n_head + h) * ATTN_SPLITS + sp) * (hd + 2);
    if (s0 >= s1) {                       // empty slice: sentinel, skipped in merge
        if (tid == 0) { P[hd] = -1e30f; P[hd + 1] = 0.f; }
        return;
    }
    const float *qh = q + (ulong64)tk * a.qs + h * hd;
    float *ah = att + ((ulong64)tk * a.n_head + h) * a.n_ctx;

    for (int t = s0 + tid; t < s1; t += tpg)
        ah[t] = kv_dot(kc + base + (ulong64)t * row_b, qh, hd, a.q8) * a.scale;
    __syncthreads();
    float mx = -1e30f;
    for (int t = s0 + tid; t < s1; t += tpg) mx = fmaxf(mx, ah[t]);
    red[tid] = mx;
    __syncthreads();
    for (int off = tpg / 2; off > 0; off >>= 1) {
        if (tid < off) red[tid] = fmaxf(red[tid], red[tid + off]);
        __syncthreads();
    }
    mx = red[0];
    __syncthreads();
    float sum = 0;
    for (int t = s0 + tid; t < s1; t += tpg) {
        float e = expf(ah[t] - mx);
        ah[t] = e;
        sum += e;
    }
    red[tid] = sum;
    __syncthreads();
    for (int off = tpg / 2; off > 0; off >>= 1) {
        if (tid < off) red[tid] += red[tid + off];
        __syncthreads();
    }
    sum = red[0];
    __syncthreads();
    for (int i2 = tid; i2 < hd / 2; i2 += tpg) {
        float o0 = 0, o1 = 0;
        for (int t = s0; t < s1; t++) {
            float2 vf = kv_pair(vc + base + (ulong64)t * row_b, i2, a.q8);
            o0 += ah[t] * vf.x;
            o1 += ah[t] * vf.y;
        }
        P[2 * i2]     = o0;
        P[2 * i2 + 1] = o1;
    }
    if (tid == 0) { P[hd] = mx; P[hd + 1] = sum; }
}

// ---- multi-column twins of the decode GEMVs ----
//
// Two requirements pull in opposite directions here.
//
// IDENTITY says each column must emit the same FMA sequence, in the same
// order, with the same warp_sum tree, as the batch-1 k_gemv_* kernel it
// replaces. That fixes the arithmetic completely; the only freedom left is
// where x is read from, and reading it from shared memory changes no value.
//
// SPEED says the column loop must be unrolled — a runtime trip count costs
// more than the wasted columns it saves, measured. But an unrolled loop has a
// fixed width, so a kernel that always does eight columns costs the same for
// two sequences as for eight, and a half-full microbatch pays full price.
//
// Both are satisfied by generating each kernel at several fixed widths and
// letting cuda.c launch the narrowest one that covers the microbatch. The
// bodies below are macros instantiated per width for exactly that reason;
// NC is the compile-time column count, always <= MVB, and the buffers stay
// MVB columns wide so the strides are unchanged.
//
// Q4_K and Q5_K also have a width-8 twin already in the tree — k_gemm_q4_K and
// k_gemm_q5_K were built on this same lane geometry for prefill — but they are
// left alone and re-derived here so the prefill path keeps the kernels it was
// verified with, and so every width comes from one source.
//
// DERIVE EACH BODY FROM THE CURRENT k_gemv_* SOURCE, NEVER FROM THE COMMENT
// ABOVE IT. These macros were exact twins when they were written (d0439ea,
// 2026-07-20) and stopped being twins eight days later, when 7ef0209 rewrote
// k_gemv_q8_0/_q4_K/_q5_K/_q6_K into their v2 load shapes and left the macros
// at v1. The comments still claimed twinhood the whole time, so for three weeks
// every quantized microbatch silently returned different bits than a lone
// decode — diagnosed in docs/cuda-microbatch-identity-2026-08-18.md. The macros
// below are re-derived from the v2 bodies; what holds them down now is
// `./test-batch test-q8.gguf`, a QUANTIZED fixture that make test runs, because
// the F32 fixture alone could never have caught it.
//
// Only two things may differ from the batch-1 kernel: x is read from shared
// memory rather than global, and per-block quantities the columns share are
// hoisted. Neither changes a value. The lane->element mapping, the per-lane
// accumulation order and the warp_sum partition must match exactly.

// ---- Q8_0: k_gemv_q8_0 v2 — lane l takes elements [(l&7)*4, +4) of block
// b0+(l>>3), four blocks per trip, then v1's element-per-lane tail. x staged.
// (k_gemm_q8_0 maps a lane to a whole block instead, so it is not a twin.)
// Blocks are staged SMPAD floats apart, not 32: with a 32-float stride the
// four block-groups of a warp land on the same eight banks (4-way conflict);
// the odd stride spreads them across all 32.
#define GEMVB_Q8_0(NAME, NC)                                                   \
extern "C" __global__ void NAME(MV_PARAMS) {                                   \
    __shared__ float xsm[NC][Q8_CHUNK * SMPAD];                                \
    unsigned warp = threadIdx.x >> 5;                                          \
    unsigned lane = threadIdx.x & 31;                                          \
    unsigned row  = blockIdx.x * GEMM_WARPS + warp;                            \
    int nb = a.n_in / 32;                                                      \
    const uchar *rw = wb + a.w_off +                                           \
                      (ulong64)(row < (unsigned)a.n_out ? row : 0) * nb * 34;  \
    int bsub = (int)(lane >> 3);                                               \
    int boff = ((int)lane & 7) * 4;                                            \
    float s[NC] = {0};                                                         \
    for (int cs = 0; cs < nb; cs += Q8_CHUNK) {                                \
        int cblocks = nb - cs < Q8_CHUNK ? nb - cs : Q8_CHUNK;                 \
        int celems  = cblocks * 32;                                            \
        int base_e  = cs * 32;                                                 \
        _Pragma("unroll")                                                      \
        for (int t = 0; t < NC; t++) {                                         \
            const float *xg = x + (ulong64)t * a.xs + base_e;                  \
            for (int e = threadIdx.x; e < celems; e += blockDim.x)             \
                xsm[t][(e >> 5) * SMPAD + (e & 31)] = xg[e];                   \
        }                                                                      \
        __syncthreads();                                                       \
        if (row < (unsigned)a.n_out) {                                         \
            /* chunks are Q8_CHUNK-aligned and Q8_CHUNK is a multiple of 4, so \
               the batch-1 kernel's b4 = nb & ~3 boundary falls inside the     \
               LAST chunk at exactly cblocks & ~3 — every lane therefore sees  \
               the same four-block trips, then the same tail, in that order */ \
            int c4 = cblocks & ~3;                                             \
            for (int bi = 0; bi < c4; bi += 4) {                               \
                const uchar *blk = rw + (ulong64)(cs + bi + bsub) * 34;        \
                float d = f16f(blk);                                           \
                const uchar *qp = blk + 2 + boff;                              \
                ushort16 u0 = *(const ushort16 *)qp,                           \
                         u1 = *(const ushort16 *)(qp + 2);                     \
                float q0 = (float)(int)(signed char)(u0 & 0xFF);               \
                float q1 = (float)(int)(signed char)(u0 >> 8);                 \
                float q2 = (float)(int)(signed char)(u1 & 0xFF);               \
                float q3 = (float)(int)(signed char)(u1 >> 8);                 \
                int xo = (bi + bsub) * SMPAD + boff;                           \
                _Pragma("unroll")                                              \
                for (int t = 0; t < NC; t++) {                                 \
                    const float *xp = xsm[t] + xo;                             \
                    s[t] += d * (q0 * xp[0] + q1 * xp[1] +                     \
                                 q2 * xp[2] + q3 * xp[3]);                     \
                }                                                              \
            }                                                                  \
            for (int bi = c4; bi < cblocks; bi++) {                            \
                const uchar *blk = rw + (ulong64)(cs + bi) * 34;               \
                float d = f16f(blk);                                           \
                const signed char *q = (const signed char *)(blk + 2);         \
                float qv = (float)q[lane];                                     \
                _Pragma("unroll")                                              \
                for (int t = 0; t < NC; t++)                                   \
                    s[t] += d * (qv * xsm[t][bi * SMPAD + lane]);              \
            }                                                                  \
        }                                                                      \
        __syncthreads();                                                       \
    }                                                                          \
    if (row < (unsigned)a.n_out)                                               \
        for (int t = 0; t < a.batch && t < NC; t++) {                          \
            float r = warp_sum(s[t]);                                          \
            if (lane == 0) y[(ulong64)t * a.ys + row] =                        \
                a.has_bias ? r + bias[row] : r;                                \
        }                                                                      \
}

// ---- Q4_0: k_gemv_q4_0's twin (2026-08-19). Q4_0 had no width-classed twin
// at all, so enc_mv_batch fell through to f_mvb: a different reduction AND no
// x staging, measured at 0.11x of sequential decode. Same four-blocks-in-
// flight shape as Q8_0; a lane's two quant bytes carry elements boff, boff+1
// (low nibbles) and boff+16, boff+17 (high nibbles).
#define GEMVB_Q4_0(NAME, NC)                                                   \
extern "C" __global__ void NAME(MV_PARAMS) {                                   \
    __shared__ float xsm[NC][Q8_CHUNK * SMPAD];                                \
    unsigned warp = threadIdx.x >> 5;                                          \
    unsigned lane = threadIdx.x & 31;                                          \
    unsigned row  = blockIdx.x * GEMM_WARPS + warp;                            \
    int nb = a.n_in / 32;                                                      \
    const uchar *rw = wb + a.w_off +                                           \
                      (ulong64)(row < (unsigned)a.n_out ? row : 0) * nb * 18;  \
    int bsub = (int)(lane >> 3);                                               \
    int boff = ((int)lane & 7) * 2;                                            \
    float s[NC] = {0};                                                         \
    for (int cs = 0; cs < nb; cs += Q8_CHUNK) {                                \
        int cblocks = nb - cs < Q8_CHUNK ? nb - cs : Q8_CHUNK;                 \
        int celems  = cblocks * 32;                                            \
        int base_e  = cs * 32;                                                 \
        _Pragma("unroll")                                                      \
        for (int t = 0; t < NC; t++) {                                         \
            const float *xg = x + (ulong64)t * a.xs + base_e;                  \
            for (int e = threadIdx.x; e < celems; e += blockDim.x)             \
                xsm[t][(e >> 5) * SMPAD + (e & 31)] = xg[e];                   \
        }                                                                      \
        __syncthreads();                                                       \
        if (row < (unsigned)a.n_out) {                                         \
            int c4 = cblocks & ~3;                                             \
            for (int bi = 0; bi < c4; bi += 4) {                               \
                const uchar *blk = rw + (ulong64)(cs + bi + bsub) * 18;        \
                float d = f16f(blk);                                           \
                ushort16 u = *(const ushort16 *)(blk + 2 + boff);              \
                float q0 = (float)((int)( u        & 0xF) - 8);                \
                float q1 = (float)((int)((u >>  4) & 0xF) - 8);                \
                float q2 = (float)((int)((u >>  8) & 0xF) - 8);                \
                float q3 = (float)((int)((u >> 12) & 0xF) - 8);                \
                int xo = (bi + bsub) * SMPAD + boff;                           \
                _Pragma("unroll")                                              \
                for (int t = 0; t < NC; t++) {                                 \
                    const float *xp = xsm[t] + xo;                             \
                    s[t] += d * (q0 * xp[0] + q2 * xp[1] +                     \
                                 q1 * xp[16] + q3 * xp[17]);                   \
                }                                                              \
            }                                                                  \
            for (int bi = c4; bi < cblocks; bi++) {                            \
                const uchar *blk = rw + (ulong64)(cs + bi) * 18;               \
                float d = f16f(blk);                                           \
                const uchar *q = blk + 2;                                      \
                int j = (int)lane & 15, hi = (int)lane >> 4;                   \
                int qv = hi ? (q[j] >> 4) : (q[j] & 0xF);                      \
                float w = d * (float)(qv - 8);                                 \
                _Pragma("unroll")                                              \
                for (int t = 0; t < NC; t++)                                   \
                    s[t] += w * xsm[t][bi * SMPAD + lane];                     \
            }                                                                  \
        }                                                                      \
        __syncthreads();                                                       \
    }                                                                          \
    if (row < (unsigned)a.n_out)                                               \
        for (int t = 0; t < a.batch && t < NC; t++) {                          \
            float r = warp_sum(s[t]);                                          \
            if (lane == 0) y[(ulong64)t * a.ys + row] =                        \
                a.has_bias ? r + bias[row] : r;                                \
        }                                                                      \
}

// Q4_K/Q5_K stage x in groups of the EIGHT elements a lane owns, one padding
// float per group: with a flat 256-float row all 32 lanes read el = lane*8,
// i.e. only four distinct banks (8-way conflict). A 9-float group stride makes
// bank (9*lane + k) & 31 a bijection over the warp — no conflict, same values.
#define KG8      8              // elements one lane owns in a k-quant block
#define KG8PAD   9              // 8 + 1: makes per-lane smem reads conflict-free
#define KG8ROW   (32 * KG8PAD)  // staged floats per column per 256-element block

// ---- Q4_K: k_gemv_q4_K v2 — lane l owns elements [l*8, l*8+8), the per-group
// affine factored out as dg*Sum(nib*x) - mmg*Sum(x). The v1 form this macro
// used to carry (a separate `dg*nib - mmg` FMA per element straight into s)
// is a different expression, hence different bits.
#define GEMVB_Q4_K(NAME, NC)                                                   \
extern "C" __global__ void NAME(MV_PARAMS) {                                   \
    __shared__ float xsm[NC][KG8ROW];                                          \
    unsigned warp = threadIdx.x >> 5;                                          \
    unsigned lane = threadIdx.x & 31;                                          \
    unsigned row  = blockIdx.x * GEMM_WARPS + warp;                            \
    int nb = a.n_in / 256;                                                     \
    const uchar *rw = wb + a.w_off +                                           \
                      (ulong64)(row < (unsigned)a.n_out ? row : 0) * nb * 144; \
    float s[NC] = {0};                                                         \
    int g     = (int)(lane >> 2);                                              \
    int ji    = (int)(lane >> 3);                                              \
    int sh    = ((((int)lane >> 2) & 1) == 0) ? 0 : 4;                         \
    int bbase = ((int)lane & 3) * 8;                                           \
    int el    = (int)lane * KG8PAD;                                            \
    for (int b = 0; b < nb; b++) {                                             \
        const uchar *blk = rw + (ulong64)b * 144;                              \
        int base_e = b * 256;                                                  \
        _Pragma("unroll")                                                      \
        for (int t = 0; t < NC; t++) {                                         \
            const float *xg = x + (ulong64)t * a.xs + base_e;                  \
            for (int e = threadIdx.x; e < 256; e += blockDim.x)                \
                xsm[t][(e >> 3) * KG8PAD + (e & 7)] = xg[e];                   \
        }                                                                      \
        __syncthreads();                                                       \
        if (row < (unsigned)a.n_out) {                                         \
            float dd   = f16f(blk);                                            \
            float dmin = f16f(blk + 2);                                        \
            uchar sg, mg;                                                      \
            get_scale_min_k4(g, blk + 4, &sg, &mg);                            \
            float dg = dd * (float)sg, mmg = dmin * (float)mg;                 \
            uint2 qv = *(const uint2 *)(blk + 16 + ji * 32 + bbase);           \
            uint v0 = (qv.x >> sh) & 0x0F0F0F0Fu;                              \
            uint v1 = (qv.y >> sh) & 0x0F0F0F0Fu;                              \
            float n0 = (float)(v0 & 0xFF),        n1 = (float)((v0 >>  8) & 0xFF); \
            float n2 = (float)((v0 >> 16) & 0xFF), n3 = (float)((v0 >> 24));    \
            float n4 = (float)(v1 & 0xFF),        n5 = (float)((v1 >>  8) & 0xFF); \
            float n6 = (float)((v1 >> 16) & 0xFF), n7 = (float)((v1 >> 24));    \
            _Pragma("unroll")                                                  \
            for (int t = 0; t < NC; t++) {                                     \
                const float *xp = xsm[t] + el;                                 \
                float tt = n0 * xp[0] + n1 * xp[1] + n2 * xp[2] + n3 * xp[3]   \
                         + n4 * xp[4] + n5 * xp[5] + n6 * xp[6] + n7 * xp[7];  \
                float sx = xp[0] + xp[1] + xp[2] + xp[3]                       \
                         + xp[4] + xp[5] + xp[6] + xp[7];                      \
                s[t] += dg * tt - mmg * sx;                                    \
            }                                                                  \
        }                                                                      \
        __syncthreads();                                                       \
    }                                                                          \
    if (row < (unsigned)a.n_out)                                               \
        for (int t = 0; t < a.batch && t < NC; t++) {                          \
            float r = warp_sum(s[t]);                                          \
            if (lane == 0) y[(ulong64)t * a.ys + row] =                        \
                a.has_bias ? r + bias[row] : r;                                \
        }                                                                      \
}

// ---- Q5_K: k_gemv_q5_K v2, i.e. Q4_K's factored form with the fifth bit —
// bit g of each qh byte, added as 16 before the conversion to float.
#define GEMVB_Q5_K(NAME, NC)                                                   \
extern "C" __global__ void NAME(MV_PARAMS) {                                   \
    __shared__ float xsm[NC][KG8ROW];                                          \
    unsigned warp = threadIdx.x >> 5;                                          \
    unsigned lane = threadIdx.x & 31;                                          \
    unsigned row  = blockIdx.x * GEMM_WARPS + warp;                            \
    int nb = a.n_in / 256;                                                     \
    const uchar *rw = wb + a.w_off +                                           \
                      (ulong64)(row < (unsigned)a.n_out ? row : 0) * nb * 176; \
    float s[NC] = {0};                                                         \
    int g     = (int)(lane >> 2);                                              \
    int ji    = (int)(lane >> 3);                                              \
    int sh    = ((((int)lane >> 2) & 1) == 0) ? 0 : 4;                         \
    int bbase = ((int)lane & 3) * 8;                                           \
    int hshift = g;                                                            \
    int el    = (int)lane * KG8PAD;                                            \
    for (int b = 0; b < nb; b++) {                                             \
        const uchar *blk = rw + (ulong64)b * 176;                              \
        int base_e = b * 256;                                                  \
        _Pragma("unroll")                                                      \
        for (int t = 0; t < NC; t++) {                                         \
            const float *xg = x + (ulong64)t * a.xs + base_e;                  \
            for (int e = threadIdx.x; e < 256; e += blockDim.x)                \
                xsm[t][(e >> 3) * KG8PAD + (e & 7)] = xg[e];                   \
        }                                                                      \
        __syncthreads();                                                       \
        if (row < (unsigned)a.n_out) {                                         \
            float dd   = f16f(blk);                                            \
            float dmin = f16f(blk + 2);                                        \
            uchar sg, mg;                                                      \
            get_scale_min_k4(g, blk + 4, &sg, &mg);                            \
            float dg = dd * (float)sg, mmg = dmin * (float)mg;                 \
            uint2 qv = *(const uint2 *)(blk + 48 + ji * 32 + bbase);           \
            uint2 hv = *(const uint2 *)(blk + 16 + bbase);                     \
            uint v0 = (qv.x >> sh) & 0x0F0F0F0Fu;                              \
            uint v1 = (qv.y >> sh) & 0x0F0F0F0Fu;                              \
            uint h0 = ((hv.x >> hshift) & 0x01010101u) << 4;                   \
            uint h1 = ((hv.y >> hshift) & 0x01010101u) << 4;                   \
            v0 += h0; v1 += h1;                                                \
            float n0 = (float)(v0 & 0xFF),        n1 = (float)((v0 >>  8) & 0xFF); \
            float n2 = (float)((v0 >> 16) & 0xFF), n3 = (float)((v0 >> 24));    \
            float n4 = (float)(v1 & 0xFF),        n5 = (float)((v1 >>  8) & 0xFF); \
            float n6 = (float)((v1 >> 16) & 0xFF), n7 = (float)((v1 >> 24));    \
            _Pragma("unroll")                                                  \
            for (int t = 0; t < NC; t++) {                                     \
                const float *xp = xsm[t] + el;                                 \
                float tt = n0 * xp[0] + n1 * xp[1] + n2 * xp[2] + n3 * xp[3]   \
                         + n4 * xp[4] + n5 * xp[5] + n6 * xp[6] + n7 * xp[7];  \
                float sx = xp[0] + xp[1] + xp[2] + xp[3]                       \
                         + xp[4] + xp[5] + xp[6] + xp[7];                      \
                s[t] += dg * tt - mmg * sx;                                    \
            }                                                                  \
        }                                                                      \
        __syncthreads();                                                       \
    }                                                                          \
    if (row < (unsigned)a.n_out)                                               \
        for (int t = 0; t < a.batch && t < NC; t++) {                          \
            float r = warp_sum(s[t]);                                          \
            if (lane == 0) y[(ulong64)t * a.ys + row] =                        \
                a.has_bias ? r + bias[row] : r;                                \
        }                                                                      \
}

// ---- Q6_K: k_gemv_q6_K v2 — d factored out of the four-term group, a whole
// BLOCK reduced into its own accumulator, and those block sums split across
// TWO running accumulators (even blocks into s0, odd into s1, s0 + s1 at the
// end) because v2 keeps two blocks in flight. Folding both into one running
// sum, as this macro used to, is a different reduction tree.
// (k_gemm_q6_K premultiplies d into each weight instead, which agrees in exact
// arithmetic but not necessarily in floating point, so it is not a twin.)
#define GEMVB_Q6_K(NAME, NC)                                                   \
extern "C" __global__ void NAME(MV_PARAMS) {                                   \
    __shared__ float xsm[NC][256];                                             \
    unsigned warp = threadIdx.x >> 5;                                          \
    unsigned lane = threadIdx.x & 31;                                          \
    unsigned row  = blockIdx.x * GEMM_WARPS + warp;                            \
    int nb = a.n_in / 256;                                                     \
    const uchar *rw = wb + a.w_off +                                           \
                      (ulong64)(row < (unsigned)a.n_out ? row : 0) * nb * 210; \
    float s0[NC] = {0}, s1[NC] = {0};                                          \
    int is = (int)(lane >> 4);                                                 \
    for (int b = 0; b < nb; b++) {                                             \
        const uchar *blk = rw + (ulong64)b * 210;                              \
        int base_e = b * 256;                                                  \
        _Pragma("unroll")                                                      \
        for (int t = 0; t < NC; t++) {                                         \
            const float *xg = x + (ulong64)t * a.xs + base_e;                  \
            for (int e = threadIdx.x; e < 256; e += blockDim.x)                \
                xsm[t][e] = xg[e];                                             \
        }                                                                      \
        __syncthreads();                                                       \
        if (row < (unsigned)a.n_out) {                                         \
            float d = f16f(blk + 208);                                         \
            float acc[NC] = {0};                                               \
            _Pragma("unroll")                                                  \
            for (int half = 0; half < 2; half++) {                             \
                const uchar *ql = blk + half * 64;                             \
                const uchar *qh = blk + 128 + half * 32;                       \
                const signed char *sc =                                        \
                    (const signed char *)(blk + 192) + half * 8;               \
                int q1 = (int)((ql[lane]      & 0xF) |                         \
                               (((qh[lane] >> 0) & 3) << 4)) - 32;             \
                int q2 = (int)((ql[lane + 32] & 0xF) |                         \
                               (((qh[lane] >> 2) & 3) << 4)) - 32;             \
                int q3 = (int)((ql[lane]      >> 4)  |                         \
                               (((qh[lane] >> 4) & 3) << 4)) - 32;             \
                int q4 = (int)((ql[lane + 32] >> 4)  |                         \
                               (((qh[lane] >> 6) & 3) << 4)) - 32;             \
                float c1 = (float)(sc[0 + is] * q1);                           \
                float c2 = (float)(sc[2 + is] * q2);                           \
                float c3 = (float)(sc[4 + is] * q3);                           \
                float c4 = (float)(sc[6 + is] * q4);                           \
                int e0 = half * 128;                                           \
                _Pragma("unroll")                                              \
                for (int t = 0; t < NC; t++) {                                 \
                    const float *xp = xsm[t] + e0;                             \
                    acc[t] += d * (c1 * xp[lane]      + c2 * xp[lane + 32] +   \
                                   c3 * xp[lane + 64] + c4 * xp[lane + 96]);   \
                }                                                              \
            }                                                                  \
            _Pragma("unroll")                                                  \
            for (int t = 0; t < NC; t++) {                                     \
                if (b & 1) s1[t] += acc[t]; else s0[t] += acc[t];              \
            }                                                                  \
        }                                                                      \
        __syncthreads();                                                       \
    }                                                                          \
    if (row < (unsigned)a.n_out)                                               \
        for (int t = 0; t < a.batch && t < NC; t++) {                          \
            float r = warp_sum(s0[t] + s1[t]);                                 \
            if (lane == 0) y[(ulong64)t * a.ys + row] =                        \
                a.has_bias ? r + bias[row] : r;                                \
        }                                                                      \
}

// Widths cuda.c can pick from. Two are enough to cover 2..8 without a
// half-empty batch paying much: a microbatch of 3 runs the 4-wide kernel.
GEMVB_Q8_0(k_gemvb_q8_0_x4, 4)
GEMVB_Q8_0(k_gemvb_q8_0_x8, 8)
GEMVB_Q4_0(k_gemvb_q4_0_x4, 4)
GEMVB_Q4_0(k_gemvb_q4_0_x8, 8)
GEMVB_Q4_K(k_gemvb_q4_K_x4, 4)
GEMVB_Q4_K(k_gemvb_q4_K_x8, 8)
GEMVB_Q5_K(k_gemvb_q5_K_x4, 4)
GEMVB_Q5_K(k_gemvb_q5_K_x8, 8)
GEMVB_Q6_K(k_gemvb_q6_K_x4, 4)
GEMVB_Q6_K(k_gemvb_q6_K_x8, 8)

// ---------------------------------------------------------------- elementwise
// grid.y = token column for k_add (different x/d strides); silu operates on
// the contiguous [batch][n_ff] region in one launch

extern "C" __global__ void k_silu_mul(float *g, const float *u, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = g[i];
        g[i] = (x / (1.0f + expf(-x))) * u[i];
    }
}

extern "C" __global__ void k_gelu_mul(float *g, const float *u, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = g[i];
        float t = tanhf(0.7978845608f * (x + 0.044715f * x * x * x));
        g[i] = 0.5f * x * (1.0f + t) * u[i];
    }
}

// gpt-oss clamped alpha-sigmoid GLU, 1:1 with gated_act(ACT_SWIGLU_OAI) in
// model.c (itself transcribed from llama.cpp's swiglu_oai): gate clamped
// above only, up clamped both sides, up carries a +1 shift. The x <= -50
// early-zero mirrors the CPU guard exactly so both backends emit the same
// value there (0.0f rather than the -0.0f the division limit would give).
static __device__ __forceinline__ float swiglu_oai(float g, float u) {
    const float alpha = 1.702f, limit = 7.0f;
    float x = g < limit ? g : limit;
    float y = u < -limit ? -limit : (u > limit ? limit : u);
    float gl = x < -50.0f ? 0.0f : x / (1.0f + expf(alpha * -x));
    return gl * (y + 1.0f);
}

extern "C" __global__ void k_swiglu_oai_mul(float *g, const float *u, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) g[i] = swiglu_oai(g[i], u[i]);
}

extern "C" __global__ void k_add(float *x, const float *d, int n, int xs, int ds) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[(ulong64)blockIdx.y * xs + i] += d[(ulong64)blockIdx.y * ds + i];
}

// whole-layer output scalar (gemma4): x *= s, grid.y = token column
extern "C" __global__ void k_scale(float *x, float s, int n, int xs) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[(ulong64)blockIdx.y * xs + i] *= s;
}

// ------------------------------------------------- sparse-MoE device routing
// Device-side softmax -> top-k -> renormalize, replacing the per-token
// DtoH round-trip that forced MoE decode onto the eager (graph_bad) path.
//
// BIT-IDENTITY CONTRACT (moe-gpu-routing spec): this kernel is the device
// mirror of moe_route() in model.c, and the byte-identical CPU==GPU greedy
// cert depends on it selecting exactly as the host would given the same
// logits. It therefore runs SERIALLY, one thread per token, with the host's
// exact arithmetic: same max-scan order, one expf per element in element
// order, same summation order, division (not reciprocal-multiply), and the
// strict `>` compare that sends ties to the lowest expert index. At
// n_expert <= 256 a serial thread is microseconds per token; do not
// restructure this into a parallel reduction — reordering floats here is
// what the contract forbids.
//
// logits: [tokens][ls]; sel/selw: [tokens][used]. lg[] is a local (per-
// thread) working copy so the router logits buffer itself stays intact.
extern "C" __global__ void k_moe_route(const float *logits, int *sel,
                                       float *selw, int ne, int used,
                                       int tokens, int ls) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= tokens) return;
    float lg[256];
    const float *src = logits + (ulong64)t * ls;
    for (int e = 0; e < ne; e++) lg[e] = src[e];
    float mx = lg[0];
    for (int e = 1; e < ne; e++)
        if (lg[e] > mx) mx = lg[e];
    float ssum = 0.0f;
    for (int e = 0; e < ne; e++) {
        // Device expf differs from the host libm by ~1-2 ulp, so the fused
        // path is NOT byte-identical to the host routing — the certified
        // MoE byte-identity property is therefore defined over the EAGER
        // path (RUNNER_MOE_EAGER=1, pinned in the certification harnesses;
        // see docs/compatibility-program.md). The fused default's contract
        // is the weaker, verified class: expert selection identical, selw
        // within 1 ulp of the host reference (the reciprocal-multiply
        // mirror below keeps the divergence to expf alone). History: a
        // correctly-rounded (float)exp((double)x) here DID bit-match
        // correctly-rounded hosts (UCRT), but a fast-math/libmvec host is
        // ~4 ulp and unreachable by construction, so certification pinned
        // eager and this kernel keeps the plain fp32 expf.
        float p = expf(lg[e] - mx);
        lg[e] = p;
        ssum += p;
    }
    // Per-element division, not a hoisted reciprocal-multiply. A
    // reciprocal mirror of the host's -freciprocal-math codegen briefly
    // lived here chasing selw bit-identity; that goal was retired by the
    // eager certification pin (RUNNER_MOE_EAGER=1, bf93510), and the
    // Blackwell splice-proof (suite plan, 2026-07-29 evening: per-kernel
    // PTX hashing + splicing the old body restores 102.9 tok/s from 79.7)
    // showed the rcp.rn.f32 form JITs ~58 us/launch slower on the MIG —
    // x48 layers = 23% of MoE decode. Plain division satisfies the fused
    // contract (selection identical, selw within ~2 ulp of the host).
    for (int e = 0; e < ne; e++) lg[e] /= ssum;
    int   *ts = sel  + (ulong64)t * used;
    float *tw = selw + (ulong64)t * used;
    float denom = 0.0f;
    for (int s = 0; s < used; s++) {
        int best = 0;
        float bp = -1.0f;
        for (int e = 0; e < ne; e++)
            if (lg[e] > bp) { bp = lg[e]; best = e; }
        ts[s] = best;
        tw[s] = bp;
        denom += bp;
        lg[best] = -1.0f;
    }
    // per-element division, same rationale as the softmax normalization
    for (int s = 0; s < used; s++) tw[s] /= denom;
}

// -------------------------------------------- indirect expert matvec (MoE)
// One launch computes ALL selected experts' matvecs for one token: grid.y is
// the expert slot, the weight base is resolved in-kernel from sel[slot] and
// the fused-3D expert stride (the same arithmetic moe_expert_weight does on
// the host — moved into the kernel args so no host round-trip remains).
//
// Numerics: each body below is a verbatim copy of the kernel enc_mv would
// have launched for that quant type at batch 1 (k_gemv_* where the coalesced
// variant exists, k_mv_* otherwise) — only the weight-base computation and
// the slot-indexed x/y columns differ, so per-row results are bit-identical
// to the eager path's launches and every existing per-model cert carries
// over. Expert FFNs carry no bias in any supported arch, so there is no bias
// parameter.
//
// xs = x column stride per slot (0: all slots read the same input, the
// gate/up case; nff: per-slot hidden, the down case); ys = y column stride.


#define MOE_MV_HEAD \
    unsigned row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5); \
    unsigned lane = threadIdx.x & 31; \
    if (row >= (unsigned)a.n_out) return; \
    const uchar *wbase = wb + a.w_off + (ulong64)sel[blockIdx.y] * a.estride; \
    x += (ulong64)blockIdx.y * a.xs; \
    float s = 0;

#define MOE_MV_TAIL \
    s = warp_sum(s); \
    if (lane == 0) y[(ulong64)blockIdx.y * a.ys + row] = s;

#define MOE_MV_PARAMS \
    const uchar *wb, const float *x, float *y, moe_args a, const int *sel

// body of k_mv_f32
extern "C" __global__ void k_moe_mv_f32(MOE_MV_PARAMS) {
    MOE_MV_HEAD;
    const float *rw = (const float *)wbase + (ulong64)row * a.n_in;
    for (int i = lane; i < a.n_in; i += 32) s += rw[i] * x[i];
    MOE_MV_TAIL;
}

// body of k_mv_f16
extern "C" __global__ void k_moe_mv_f16(MOE_MV_PARAMS) {
    MOE_MV_HEAD;
    const __half *rw = (const __half *)wbase + (ulong64)row * a.n_in;
    for (int i = lane; i < a.n_in; i += 32) s += __half2float(rw[i]) * x[i];
    MOE_MV_TAIL;
}

// body of k_gemv_q8_0 (the batch-1 kernel enc_mv picks for Q8_0)
extern "C" __global__ void k_moe_mv_q8_0(MOE_MV_PARAMS) {
    MOE_MV_HEAD;
    int nb = a.n_in / 32;
    const uchar *rw = wbase + (ulong64)row * nb * 34;
    int bsub = (int)(lane >> 3);
    int boff = ((int)lane & 7) * 4;
    int b4 = nb & ~3;
    for (int b0 = 0; b0 < b4; b0 += 4) {
        const uchar *blk = rw + (ulong64)(b0 + bsub) * 34;
        float d = f16f(blk);
        const uchar *qp = blk + 2 + boff;
        ushort16 u0 = *(const ushort16 *)qp, u1 = *(const ushort16 *)(qp + 2);
        int q0 = (int)(signed char)(u0 & 0xFF), q1 = (int)(signed char)(u0 >> 8);
        int q2 = (int)(signed char)(u1 & 0xFF), q3 = (int)(signed char)(u1 >> 8);
        const float4 xv = *(const float4 *)(x + (ulong64)(b0 + bsub) * 32 + boff);
        s += d * ((float)q0 * xv.x + (float)q1 * xv.y +
                  (float)q2 * xv.z + (float)q3 * xv.w);
    }
    for (int b = b4; b < nb; b++) {
        const uchar *blk = rw + (ulong64)b * 34;
        float d = f16f(blk);
        const signed char *q = (const signed char *)(blk + 2);
        s += d * ((float)q[lane] * x[(ulong64)b * 32 + lane]);
    }
    MOE_MV_TAIL;
}

// body of k_mv_q4_0 (no coalesced GEMV exists for Q4_0 yet)
extern "C" __global__ void k_moe_mv_q4_0(MOE_MV_PARAMS) {
    MOE_MV_HEAD;
    int nb = a.n_in / 32;
    const uchar *rw = wbase + (ulong64)row * nb * 18;
    for (int b = lane; b < nb; b += 32) {
        const uchar *blk = rw + (ulong64)b * 18;
        float d = f16f(blk);
        const uchar *q = blk + 2;
        const float *xp = x + b * 32;
        float t = 0;
        for (int j = 0; j < 16; j++)
            t += ((int)(q[j] & 0xF) - 8) * xp[j] + ((int)(q[j] >> 4) - 8) * xp[j + 16];
        s += d * t;
    }
    MOE_MV_TAIL;
}

// body of k_mv_mxfp4 (gpt-oss experts; no coalesced GEMV exists for MXFP4)
extern "C" __global__ void k_moe_mv_mxfp4(MOE_MV_PARAMS) {
    MOE_MV_HEAD;
    int nb = a.n_in / 32;
    const uchar *rw = wbase + (ulong64)row * nb * 17;
    for (int b = lane; b < nb; b += 32) {
        const uchar *blk = rw + (ulong64)b * 17;
        float d = ldexpf(1.0f, (int)blk[0] - 127);
        const uchar *q = blk + 1;
        const float *xp = x + b * 32;
        float t = 0;
        for (int j = 0; j < 16; j++) {
            t += kv_mxfp4[q[j] & 0xF] * xp[j];
            t += kv_mxfp4[q[j] >> 4]  * xp[j + 16];
        }
        s += d * t;
    }
    MOE_MV_TAIL;
}

// body of k_gemv_q4_K
extern "C" __global__ void k_moe_mv_q4_K(MOE_MV_PARAMS) {
    MOE_MV_HEAD;
    int nb = a.n_in / 256;
    const uchar *rw = wbase + (ulong64)row * nb * 144;
    int g     = (int)(lane >> 2);
    int ji    = (int)(lane >> 3);
    int sh    = ((((int)lane >> 2) & 1) == 0) ? 0 : 4;
    int bbase = ((int)lane & 3) * 8;
    for (int b = 0; b < nb; b++) {
        const uchar *blk = rw + (ulong64)b * 144;
        float dd   = f16f(blk);
        float dmin = f16f(blk + 2);
        uchar sg, mg;
        get_scale_min_k4(g, blk + 4, &sg, &mg);
        float dg = dd * (float)sg, mmg = dmin * (float)mg;
        uint2 qv = *(const uint2 *)(blk + 16 + ji * 32 + bbase);
        const float *xp = x + (ulong64)b * 256 + (int)lane * 8;
        float4 x0 = *(const float4 *)xp, x1 = *(const float4 *)(xp + 4);
        uint v0 = (qv.x >> sh) & 0x0F0F0F0Fu, v1 = (qv.y >> sh) & 0x0F0F0F0Fu;
        float t  = (float)(v0 & 0xFF)         * x0.x
                 + (float)((v0 >>  8) & 0xFF) * x0.y
                 + (float)((v0 >> 16) & 0xFF) * x0.z
                 + (float)((v0 >> 24)       ) * x0.w
                 + (float)(v1 & 0xFF)         * x1.x
                 + (float)((v1 >>  8) & 0xFF) * x1.y
                 + (float)((v1 >> 16) & 0xFF) * x1.z
                 + (float)((v1 >> 24)       ) * x1.w;
        float sx = x0.x + x0.y + x0.z + x0.w + x1.x + x1.y + x1.z + x1.w;
        s += dg * t - mmg * sx;
    }
    MOE_MV_TAIL;
}

// body of k_gemv_q5_K
extern "C" __global__ void k_moe_mv_q5_K(MOE_MV_PARAMS) {
    MOE_MV_HEAD;
    int nb = a.n_in / 256;
    const uchar *rw = wbase + (ulong64)row * nb * 176;
    int g     = (int)(lane >> 2);
    int ji    = (int)(lane >> 3);
    int sh    = ((((int)lane >> 2) & 1) == 0) ? 0 : 4;
    int bbase = ((int)lane & 3) * 8;
    int hshift = g;
    for (int b = 0; b < nb; b++) {
        const uchar *blk = rw + (ulong64)b * 176;
        float dd   = f16f(blk);
        float dmin = f16f(blk + 2);
        uchar sg, mg;
        get_scale_min_k4(g, blk + 4, &sg, &mg);
        float dg = dd * (float)sg, mmg = dmin * (float)mg;
        uint2 qv = *(const uint2 *)(blk + 48 + ji * 32 + bbase);
        uint2 hv = *(const uint2 *)(blk + 16 + bbase);
        const float *xp = x + (ulong64)b * 256 + (int)lane * 8;
        float4 x0 = *(const float4 *)xp, x1 = *(const float4 *)(xp + 4);
        uint v0 = (qv.x >> sh) & 0x0F0F0F0Fu, v1 = (qv.y >> sh) & 0x0F0F0F0Fu;
        uint h0 = ((hv.x >> hshift) & 0x01010101u) << 4;
        uint h1 = ((hv.y >> hshift) & 0x01010101u) << 4;
        v0 += h0; v1 += h1;
        float t  = (float)(v0 & 0xFF)         * x0.x
                 + (float)((v0 >>  8) & 0xFF) * x0.y
                 + (float)((v0 >> 16) & 0xFF) * x0.z
                 + (float)((v0 >> 24)       ) * x0.w
                 + (float)(v1 & 0xFF)         * x1.x
                 + (float)((v1 >>  8) & 0xFF) * x1.y
                 + (float)((v1 >> 16) & 0xFF) * x1.z
                 + (float)((v1 >> 24)       ) * x1.w;
        float sx = x0.x + x0.y + x0.z + x0.w + x1.x + x1.y + x1.z + x1.w;
        s += dg * t - mmg * sx;
    }
    MOE_MV_TAIL;
}

// body of k_gemv_q6_K
extern "C" __global__ void k_moe_mv_q6_K(MOE_MV_PARAMS) {
    MOE_MV_HEAD;
    int nb = a.n_in / 256;
    const uchar *rw = wbase + (ulong64)row * nb * 210;
    int is = (int)(lane >> 4);
    float s0 = 0, s1 = 0;
    int b2 = nb & ~1;
    for (int b = 0; b < b2; b += 2) {
        s0 += q6k_block_dot(rw + (ulong64)b * 210,       x + (ulong64)b * 256,
                            (int)lane, is);
        s1 += q6k_block_dot(rw + (ulong64)(b + 1) * 210, x + (ulong64)(b + 1) * 256,
                            (int)lane, is);
    }
    if (b2 < nb)
        s0 += q6k_block_dot(rw + (ulong64)b2 * 210, x + (ulong64)b2 * 256,
                            (int)lane, is);
    s = s0 + s1;
    MOE_MV_TAIL;
}

// Gated activation + routing-weight fold for every expert slot in one launch.
// grid.y = expert slot. Per element: the exact k_silu_mul / k_gelu_mul
// arithmetic, then a separate multiply by the slot's routing weight — the
// same two rounding steps the eager path produced with its actmul + scale
// launch pair, so values are bit-identical to it. dscale is the per-expert
// down-projection scale table (gemma-4), indexed via sel; ones when the
// model has none, so the multiply is exact and harmless.
// gbuf column stride gss, ubuf column stride uss (gemma reads gate and up
// out of one fused 2*nff column: gss = uss = 2*nff, ubuf = gbuf + nff).
// act: 0 = SiLU, 1 = tanh-GELU, 2 = gpt-oss swiglu_oai — same selector values
// as ACT_* in runner.h. gb/ub are per-expert gate/up bias tables ([n_expert]
// rows of nff, gpt-oss) or NULL; the slot's expert index picks the row, and
// the bias lands BEFORE the activation, matching moe_ffn_token on the CPU.
extern "C" __global__ void k_moe_actmul(float *gbuf, const float *ubuf,
                                        int nff, int gss, int uss, int act,
                                        const float *selw, const float *dscale,
                                        const int *sel, const float *gb,
                                        const float *ub) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nff) return;
    float *gp = gbuf + (ulong64)blockIdx.y * gss;
    const float *up = ubuf + (ulong64)blockIdx.y * uss;
    float xg = gp[i], xu = up[i], v;
    if (gb) xg += gb[(ulong64)sel[blockIdx.y] * nff + i];
    if (ub) xu += ub[(ulong64)sel[blockIdx.y] * nff + i];
    if (act == 2) {
        v = swiglu_oai(xg, xu);
    } else if (act == 1) {
        float t = tanhf(0.7978845608f * (xg + 0.044715f * xg * xg * xg));
        v = 0.5f * xg * (1.0f + t) * xu;
    } else {
        v = (xg / (1.0f + expf(-xg))) * xu;
    }
    float w = selw[blockIdx.y] * dscale[sel[blockIdx.y]];
    gp[i] = v * w;
}

// Sum the per-slot down-projections into the token's FFN output, slot 0
// first then ascending — the same accumulation order as the eager path's
// write-then-add sequence, so the sum is bit-identical to it.
// db: per-expert down-bias table ([n_expert] rows of n, gpt-oss) or NULL.
// The routing weight was folded into the hidden before the down matvec, so
// each slot's eout is already w*down(h); the CPU computes w*(down(h)+db),
// and w*down(h) + selw*db is the same quantity with the fold's association.
extern "C" __global__ void k_moe_sum(float *out, const float *eout, int n,
                                     int nslots, int es, const float *db,
                                     const int *sel, const float *selw) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float s = eout[i];
    if (db) s += selw[0] * db[(ulong64)sel[0] * n + i];
    for (int sl = 1; sl < nslots; sl++) {
        s += eout[(ulong64)sl * es + i];
        if (db) s += selw[sl] * db[(ulong64)sel[sl] * n + i];
    }
    out[i] = s;
}

// ---------------------------------------- expert-grouped prefill (MoE P2)
// CUDA port of the CPU cabdad1 grouping: route the tile, then run each
// active expert ONCE over all the tokens routed to it as a batched GEMM.
// These three kernels are the glue around the existing k_gemm family:
// gather the expert's token columns, and scatter its weighted down outputs
// back into the per-token accumulator. grid.y = position in the expert's
// token list.

// dst[c][i] = src[idx[c]][i]; ss/ds = element stride between columns
extern "C" __global__ void k_moe_gather(float *dst, const float *src,
                                        const int *idx, int n, int ss,
                                        int ds) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[(ulong64)blockIdx.y * ds + i] = src[(ulong64)idx[blockIdx.y] * ss + i];
}

// out[idx[c]][i] += w[c] * src[c][i]. Launched once per expert on the
// stream, so no two active launches write the same token column (a token
// appears at most once in one expert's list) — no atomics needed.
extern "C" __global__ void k_moe_scatter_add(float *out, const float *src,
                                             const int *idx, const float *w,
                                             int n, int os, int ss) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[(ulong64)idx[blockIdx.y] * os + i] +=
        w[blockIdx.y] * src[(ulong64)blockIdx.y * ss + i];
}

// Gated activation over per-column gate/up at arbitrary column strides,
// with NO weight fold (the grouped path applies selw in the scatter, like
// the CPU grouping). Same per-element arithmetic as k_silu_mul/k_gelu_mul.
// Covers gemma's fused gate_up layout (gss = uss = 2*nff, ubuf = gbuf+nff),
// where the contiguous k_silu_mul cannot run across columns.
extern "C" __global__ void k_moe_actmul_plain(float *gbuf, const float *ubuf,
                                              int nff, int gss, int uss,
                                              int act) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nff) return;
    float *gp = gbuf + (ulong64)blockIdx.y * gss;
    const float *up = ubuf + (ulong64)blockIdx.y * uss;
    float xg = gp[i];
    if (act == 2) {
        gp[i] = swiglu_oai(xg, up[i]);
    } else if (act == 1) {
        float t = tanhf(0.7978845608f * (xg + 0.044715f * xg * xg * xg));
        gp[i] = 0.5f * xg * (1.0f + t) * up[i];
    } else {
        gp[i] = (xg / (1.0f + expf(-xg))) * up[i];
    }
}

// strided column copy: dst[c][i] = src[c][i] (accumulator -> xb columns)
extern "C" __global__ void k_moe_copy_cols(float *dst, const float *src,
                                           int n, int ds, int ss) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[(ulong64)blockIdx.y * ds + i] = src[(ulong64)blockIdx.y * ss + i];
}

// ---------------------------------------------------------- Qwen3.5 hybrid
// The generic matvec kernels above perform every learned projection.  These
// small kernels implement only the architecture-specific stateful operators,
// keeping the runtime compact and the quantized weight support in one place.

// Qwen3.5 full-attention Q is stored head-interleaved as [Q_h, gate_h].
extern "C" __global__ void k_q35_split_q(const float *packed, float *q,
                                          float *gate, int heads, int hd,
                                          int packed_stride, int q_stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int qdim = heads * hd;
    if (i >= qdim) return;
    int h = i / hd, j = i - h * hd;
    const float *src = packed + (ulong64)blockIdx.y * packed_stride;
    q[(ulong64)blockIdx.y * q_stride + i] = src[h * 2 * hd + j];
    gate[(ulong64)blockIdx.y * q_stride + i] = src[h * 2 * hd + hd + j];
}

extern "C" __global__ void k_q35_attn_gate(float *x, const float *gate,
                                            int n, int xs, int gs) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        ulong64 xo = (ulong64)blockIdx.y * xs + i;
        ulong64 go = (ulong64)blockIdx.y * gs + i;
        x[xo] *= 1.0f / (1.0f + expf(-gate[go]));
    }
}

// One launch per token is intentional: convolution history is recurrent, so
// prompt columns cannot update it concurrently. Each channel is independent.
extern "C" __global__ void k_q35_conv(const float *mix, float *cv,
                                       const float *weight, float *history,
                                       int convdim, int kernel) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= convdim) return;
    int histn = kernel - 1;
    const float *cw = weight + (ulong64)c * kernel;
    float sum = cw[histn] * mix[c];
    for (int k = 0; k < histn; k++)
        sum += cw[k] * history[(ulong64)k * convdim + c];
    cv[c] = sum / (1.0f + expf(-sum));
    for (int k = 0; k + 1 < histn; k++)
        history[(ulong64)k * convdim + c] =
            history[(ulong64)(k + 1) * convdim + c];
    if (histn) history[(ulong64)(histn - 1) * convdim + c] = mix[c];
}

// One block per value head. The model geometry guarantees hv == state/key
// width. State is [head][value-column][key-row], matching the CPU reference.
extern "C" __global__ void k_q35_delta(float *cv, const float *z,
                                        const float *beta_in,
                                        const float *alpha_in,
                                        const float *dt, const float *a,
                                        const float *norm, float *state,
                                        float *out, int state_dim, int groups,
                                        int heads, float eps) {
    __shared__ float red[256];
    int tid = threadIdx.x, h = blockIdx.x;
    if (h >= heads) return;
    int keydim = state_dim * groups;
    int group = h % groups;
    float *q = cv + group * state_dim;
    float *k = cv + keydim + group * state_dim;
    const float *v = cv + 2 * keydim + h * state_dim;
    float *st = state + (ulong64)h * state_dim * state_dim;
    float *yo = out + h * state_dim;

    float qs = 0.0f;
    for (int i = tid; i < state_dim; i += blockDim.x) qs += q[i] * q[i];
    red[tid] = qs;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (tid < off) red[tid] += red[tid + off];
        __syncthreads();
    }
    float qscale = rsqrtf(red[0] + eps);
    float ks = 0.0f;
    for (int i = tid; i < state_dim; i += blockDim.x) ks += k[i] * k[i];
    red[tid] = ks;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (tid < off) red[tid] += red[tid + off];
        __syncthreads();
    }
    float kscale = rsqrtf(red[0] + eps);
    float beta = 1.0f / (1.0f + expf(-beta_in[h]));
    float av = alpha_in[h] + dt[h];
    float softplus = av > 20.0f ? av : log1pf(expf(av));
    float decay = expf(a[h] * softplus);

    for (int j = tid; j < state_dim; j += blockDim.x) {
        float *row = st + (ulong64)j * state_dim;
        float pred = 0.0f;
        for (int i = 0; i < state_dim; i++) {
            row[i] *= decay;
            pred += row[i] * (k[i] * kscale);
        }
        float delta = (v[j] - pred) * beta;
        float y = 0.0f;
        for (int i = 0; i < state_dim; i++) {
            row[i] += delta * (k[i] * kscale);
            y += row[i] * (q[i] * qscale);
        }
        yo[j] = y * rsqrtf((float)state_dim);
    }
    __syncthreads();
    float ss = 0.0f;
    for (int j = tid; j < state_dim; j += blockDim.x) ss += yo[j] * yo[j];
    red[tid] = ss;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (tid < off) red[tid] += red[tid + off];
        __syncthreads();
    }
    float rms = rsqrtf(red[0] / state_dim + eps);
    for (int j = tid; j < state_dim; j += blockDim.x) {
        int o = h * state_dim + j;
        float zv = z[o];
        yo[j] = yo[j] * rms * norm[j] * (zv / (1.0f + expf(-zv)));
    }
}

// Apertus ungated activation, matching model.h:xielu(). `an` and `ap` are
// already the effective softplus-transformed parameters from model_load.
// Kept after the existing kernels so adding it does not renumber their PTX
// labels and obscure review of the generated header.
extern "C" __global__ void k_xielu(float *x, int n, float an, float ap,
                                    float b, float eps) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        if (v > 0.0f) {
            x[i] = ap * v * v + b * v;
        } else {
            float mn = v < eps ? v : eps;
            x[i] = (expm1f(mn) - v) * an + b * v;
        }
    }
}
// --------------------------------------------- tensor-core Q6_K prefill GEMM
//
// The 2026-08-08 profile said prefill is 42.7% k_gemm_q4_K_tc, 30.2% k_attn
// and 26.0% k_gemm_q6_K -- and q6_K had NO tensor-core variant at all, so
// `attn_v` and `ffn_down` in every Q4_K_M file (40% of GEMM calls) ran the
// scalar path.
//
// Same structure as k_gemm_q4_K_tc: the block dequantizes a 64-row x 128-K
// fp16 weight tile once, its four warps' MMAs share it, and the epilogue tile
// aliases the weight tile to stay under the 48 KB shared cap.
//
// Only the unpacking differs, and it is taken verbatim from k_gemm_q6_K so the
// arithmetic matches the scalar path element for element -- including the
// INTEGER `sc * q` before the float multiply:
//
//   ql = blk + half*64, qh = blk + 128 + half*32,
//   sc = (int8*)(blk+192) + half*8, d = f16(blk+208)
//   K = half*128 + l      : (ql[l]    & 0xF) | ((qh[l]>>0 & 3)<<4) - 32, sc[0+is]
//   K = half*128 + l + 32 : (ql[l+32] & 0xF) | ((qh[l]>>2 & 3)<<4) - 32, sc[2+is]
//   K = half*128 + l + 64 : (ql[l]     >> 4) | ((qh[l]>>4 & 3)<<4) - 32, sc[4+is]
//   K = half*128 + l + 96 : (ql[l+32]  >> 4) | ((qh[l]>>6 & 3)<<4) - 32, sc[6+is]
//   is = l >> 4
//
// A thread stages 64 consecutive K of one row, so sseg 0 covers the first two
// quarters of the 128-K half and sseg 1 the last two.
//
// Not bit-identical to k_gemm_q6_K by construction (the weight is dequantized
// before the multiply and the sum is reassociated into 8-element matrix
// steps), exactly like the q4_K tensor-core path, so it answers to the same
// tolerance gate.

extern "C" __global__ void k_gemm_q6_K_tc(MV_PARAMS) {
    using namespace nvcuda::wmma;
    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const unsigned row0 = blockIdx.x * TC_ROWS;
    __shared__ __half sh_w[TC_ROWS * TC_K];
    __shared__ __half sh_x[TC_N * TC_K];
    float *sh_c = (float *)sh_w;

    fragment<matrix_a, 16, 16, 16, __half, row_major> fa;
    fragment<matrix_b, 16, 16, 16, __half, col_major> fb;
    fragment<accumulator, 16, 16, 16, float> fc[TC_N / 16];
    #pragma unroll
    for (int n = 0; n < TC_N / 16; n++) fill_fragment(fc[n], 0.0f);

    int nb = a.n_in / 256;
    int srow = tid >> 1, sseg = tid & 1;

    for (int b = 0; b < nb; b++) {
        #pragma unroll
        for (int koff = 0; koff < 256; koff += TC_K) {
            {   // ---- stage weights: 64 rows x 128 K, dequantized once ----
                unsigned gr = row0 + srow;
                __half *dst = sh_w + srow * TC_K + sseg * 64;
                if (gr < (unsigned)a.n_out) {
                    const uchar *blk = wb + a.w_off +
                                       (ulong64)gr * nb * 210 + (ulong64)b * 210;
                    const int half_ = koff >> 7;          // 0 or 1
                    const uchar *ql = blk + half_ * 64;
                    const uchar *qh = blk + 128 + half_ * 32;
                    const signed char *sc =
                        (const signed char *)(blk + 192) + half_ * 8;
                    float d = f16f(blk + 208);
                    // sseg 0 -> quarters 0,1 (low nibbles, qh shifts 0,2)
                    // sseg 1 -> quarters 2,3 (high nibbles, qh shifts 4,6)
                    const int shA = sseg * 4 + 0, shB = sseg * 4 + 2;
                    const int scA = sseg * 4 + 0, scB = sseg * 4 + 2;
                    #pragma unroll
                    for (int l = 0; l < 32; l++) {
                        int is = l >> 4;
                        uchar a0 = ql[l], b0 = ql[l + 32], h = qh[l];
                        int qa = (int)((sseg ? (a0 >> 4) : (a0 & 0xF))
                                       | (((h >> shA) & 3) << 4)) - 32;
                        int qb = (int)((sseg ? (b0 >> 4) : (b0 & 0xF))
                                       | (((h >> shB) & 3) << 4)) - 32;
                        dst[l]      = __float2half(d * (float)(sc[scA + is] * qa));
                        dst[l + 32] = __float2half(d * (float)(sc[scB + is] * qb));
                    }
                } else {
                    #pragma unroll
                    for (int e = 0; e < 64; e++) dst[e] = __float2half(0.0f);
                }
            }
            {   // ---- stage activations: 128 K x TC_N tokens ----
                int col = tid / 2, part = tid % 2;
                __half *dst = sh_x + col * TC_K + part * 64;
                if (col < a.batch) {
                    const float *xg = x + (ulong64)col * a.xs + b * 256 + koff
                                      + part * 64;
                    #pragma unroll
                    for (int v = 0; v < 16; v++) {
                        float4 xv = *(const float4 *)(xg + v * 4);
                        dst[v * 4 + 0] = __float2half(xv.x);
                        dst[v * 4 + 1] = __float2half(xv.y);
                        dst[v * 4 + 2] = __float2half(xv.z);
                        dst[v * 4 + 3] = __float2half(xv.w);
                    }
                } else {
                    #pragma unroll
                    for (int e = 0; e < 64; e++) dst[e] = __float2half(0.0f);
                }
            }
            __syncthreads();
            const __half *wt = sh_w + warp * 16 * TC_K;
            #pragma unroll
            for (int k = 0; k < TC_K; k += 16) {
                load_matrix_sync(fa, wt + k, TC_K);
                #pragma unroll
                for (int n = 0; n < TC_N / 16; n++) {
                    load_matrix_sync(fb, sh_x + n * 16 * TC_K + k, TC_K);
                    mma_sync(fc[n], fa, fb, fc[n]);
                }
            }
            __syncthreads();
        }
    }

    __syncthreads();
    #pragma unroll
    for (int n = 0; n < TC_N / 16; n++)
        store_matrix_sync(sh_c + warp * 16 * TC_N + n * 16, fc[n], TC_N,
                          mem_row_major);
    __syncthreads();
    for (int idx = tid; idx < TC_ROWS * TC_N; idx += blockDim.x) {
        int rr = idx / TC_N, tt = idx % TC_N;
        unsigned gr = row0 + rr;
        if (gr < (unsigned)a.n_out && tt < a.batch)
            y[(ulong64)tt * a.ys + gr] =
                a.has_bias ? sh_c[idx] + bias[gr] : sh_c[idx];
    }
}
