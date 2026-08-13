// Metal compute kernels: the full single-token forward pass.
// Mirrors the CPU implementations in quants.c/model.c bit-layout for bit-layout.
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// ---------------------------------------------------------------- rmsnorm

// One threadgroup per column: a prompt batch normalizes every token in one
// dispatch instead of n. Strides are explicit because x and y scratch can be
// strided differently (n_embd vs xdim).
struct norm_args { int n, x_stride, y_stride; float eps; };

kernel void k_rmsnorm(device const float *x_all [[buffer(0)]],
                      device float       *y_all [[buffer(1)]],
                      device const float *w     [[buffer(2)]],
                      constant norm_args &a     [[buffer(3)]],
                      uint3 tid3 [[thread_position_in_threadgroup]],
                      uint3 tpg3 [[threads_per_threadgroup]],
                      uint3 tgpig [[threadgroup_position_in_grid]]) {
    threadgroup float red[256];
    uint tid = tid3.x, tpg = tpg3.x;
    int n = a.n;
    float eps = a.eps;
    device const float *x = x_all + (ulong)tgpig.x * a.x_stride;
    device float       *y = y_all + (ulong)tgpig.x * a.y_stride;
    float s = 0;
    for (int i = tid; i < n; i += tpg) s += x[i] * x[i];
    red[tid] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint off = tpg / 2; off > 0; off >>= 1) {
        if (tid < off) red[tid] += red[tid + off];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float r = rsqrt(red[0] / n + eps);
    for (int i = tid; i < n; i += tpg) y[i] = x[i] * r * w[i];
}

// per-head RMSNorm (qwen3 Q/K norm): one threadgroup per head
// grid.y is the prompt batch: threadgroup (h, col) normalizes head h of column
// col. Each (head, column) pair was already an independent reduction when this
// was encoded once per token, so taking the batch in grid.y is BIT-IDENTICAL,
// not a tolerance trade -- the per-pair arithmetic is character for character
// the same and only the encoding changes. `stride` is elements between columns
// (q_dim for Q, kv_dim for K); at n_col == 1 it is never read.
kernel void k_qknorm(device float       *v   [[buffer(0)]],
                     device const float *w   [[buffer(1)]],
                     constant int       &hd  [[buffer(2)]],
                     constant float     &eps [[buffer(3)]],
                     constant int       &stride [[buffer(4)]],
                     uint2 tgpig [[threadgroup_position_in_grid]],
                     uint2 tpitg [[thread_position_in_threadgroup]],
                     uint2 ntg   [[threads_per_threadgroup]]) {
    uint tid = tpitg.x, tpg = ntg.x;
    threadgroup float red[128];
    device float *x = v + (ulong)tgpig.y * stride + tgpig.x * hd;
    float s = 0;
    for (int i = tid; i < hd; i += tpg) s += x[i] * x[i];
    red[tid] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint off = tpg / 2; off > 0; off >>= 1) {
        if (tid < off) red[tid] += red[tid + off];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float r = rsqrt(red[0] / hd + eps);
    for (int i = tid; i < hd; i += tpg) x[i] = x[i] * r * w[i];
}

kernel void k_head_rmsnorm(device const float *src [[buffer(0)]],
                           device float       *dst [[buffer(1)]],
                           device const float *w   [[buffer(2)]],
                           constant int       &hd  [[buffer(3)]],
                           constant float     &eps [[buffer(4)]],
                           constant int       &has_weight [[buffer(5)]],
                           constant int       &stride [[buffer(6)]],
                           uint2 tgpig [[threadgroup_position_in_grid]],
                           uint2 tpitg [[thread_position_in_threadgroup]],
                           uint2 ntg   [[threads_per_threadgroup]]) {
    uint tid = tpitg.x, tpg = ntg.x;
    threadgroup float red[128];
    // Batched in grid.y for the same reason, and with the same bit-identity
    // argument, as k_qknorm above. src and dst share a stride: both the K and V
    // staging buffers are strided by kv_dim.
    ulong col = (ulong)tgpig.y * stride;
    device const float *x = src + col + tgpig.x * hd;
    device float *y = dst + col + tgpig.x * hd;
    float s = 0;
    for (int i = tid; i < hd; i += tpg) s += x[i] * x[i];
    red[tid] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint off = tpg / 2; off > 0; off >>= 1) {
        if (tid < off) red[tid] += red[tid + off];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float r = rsqrt(red[0] / hd + eps);
    for (int i = tid; i < hd; i += tpg)
        y[i] = x[i] * r * (has_weight ? w[i] : 1.0f);
}

// ---------------------------------------------------------------- matvec
// One simdgroup (32 lanes) per output row; lanes stride over blocks.
//
// n_col > 1 turns this into a matmul: the simdgroup walks its weight row once
// per column, so for a prompt batch the row is fetched from device memory on
// the first column and served from cache for the rest — prefill stops paying
// the whole weight matrix per token. The arithmetic per output element is
// character-for-character what n_col == 1 does (same lane striding, same
// simd_sum), so batched prefill stays bit-identical to per-token submits,
// which is what the CPU==GPU gate requires.
//
// col_tile trades the two off: one threadgroup per row over ALL columns gets
// maximum cache reuse but serializes the batch and starves the GPU of
// threadgroups; col_tile columns per threadgroup keeps grid.y parallelism
// while still amortizing each weight fetch col_tile ways.
//
// x and y strides are explicit because they are NOT always n_in/n_out: the
// xb scratch is strided by xdim = max(q_dim, n_embd).

struct mv_args {
    int   n_in;
    int   n_out;
    ulong w_off;      // tensor byte offset inside the weight buffer
    int   has_bias;
    int   n_col;      // columns (prompt tokens) processed by this dispatch
    int   x_stride;   // elements between consecutive columns of x
    int   y_stride;   // elements between consecutive columns of y
    int   col_tile;   // columns per threadgroup (grid.y tiles the rest)
};

// The loop bound is uniform across the simdgroup, so every lane reaches each
// simd_sum in MV_TAIL the same number of times — a divergent simd_sum would
// be undefined.
#define MV_HEAD \
    uint row = tgpig.x * (ntg.x / 32) + sgitg; \
    if (row >= (uint)a.n_out) return; \
    int col_lo = (int)tgpig.y * a.col_tile; \
    int col_hi = min(col_lo + a.col_tile, a.n_col); \
    for (int col = col_lo; col < col_hi; col++) { \
    device const float *x = x_all + (ulong)col * a.x_stride; \
    device float       *y = y_all + (ulong)col * a.y_stride;

#define MV_TAIL \
    s = simd_sum(s); \
    if (tiisg == 0) y[row] = a.has_bias ? s + bias[row] : s; \
    }

#define MV_PARAMS \
    device const uchar *wb    [[buffer(0)]], \
    device const float *x_all [[buffer(1)]], \
    device float       *y_all [[buffer(2)]], \
    constant mv_args   &a     [[buffer(3)]], \
    device const float *bias  [[buffer(4)]], \
    uint  sgitg [[simdgroup_index_in_threadgroup]], \
    uint  tiisg [[thread_index_in_simdgroup]], \
    uint3 tgpig [[threadgroup_position_in_grid]], \
    uint3 ntg   [[threads_per_threadgroup]]

kernel void k_mv_f32(MV_PARAMS) {
    MV_HEAD;
    device const float *rw = (device const float *)(wb + a.w_off) + (ulong)row * a.n_in;
    float s = 0;
    for (int i = tiisg; i < a.n_in; i += 32) s += rw[i] * x[i];
    MV_TAIL;
}

// Widened for the same reason as k_mv_bf16 above: one half per lane step left
// f16 decode slower on the GPU than on the CPU.
kernel void k_mv_f16(MV_PARAMS) {
    MV_HEAD;
    device const half *rw = (device const half *)(wb + a.w_off) + (ulong)row * a.n_in;
    float s = 0;
    if ((a.n_in & 3) == 0) {
        device const packed_half4 *w4 = (device const packed_half4 *)rw;
        device const packed_float4 *x4 = (device const packed_float4 *)x;
        int n4 = a.n_in >> 2;
        float4 acc = 0;
        for (int i = tiisg; i < n4; i += 32) acc += float4(w4[i]) * x4[i];
        s = acc.x + acc.y + acc.z + acc.w;
    } else {
        for (int i = tiisg; i < a.n_in; i += 32) s += (float)rw[i] * x[i];
    }
    MV_TAIL;
}

// Widened loads, unchanged arithmetic.
//
// The scalar forms below issued one memory instruction per BYTE of quant and
// one per float of activation: 48 loads to consume an 18-byte q4_0 block, 64
// for a 34-byte q8_0 one. Reading the same bytes as uchar4/float4 cuts that
// roughly fourfold and is the only bandwidth lever this route has — the lane
// that owns a block is pinned by the identity contract (see MV_HEAD), so the
// work cannot be redistributed.
//
// The rule that makes it safe: an accumulating EXPRESSION is copied character
// for character from the scalar form, and only the way its operands are LOADED
// changes. `t += A * xp[j] + B * xp[j + 16]` keeps that exact shape with xp[j]
// spelled as a component of an already-loaded float4, because the Metal
// compiler contracts multiply-add by expression shape. Resurfacing the same
// products through a float4 accumulator and a horizontal sum rounds
// differently — that is what the 2026-08-07 q4_K attempt did, and it was both
// slower and off-contract.
//
// What it is worth, so nobody re-derives it: on an 8-core M1 this is NEUTRAL.
// Five interleaved rounds against e2b-q40 — 2.6 GB of q4_0, the one local
// model actually bandwidth-bound at ~40 of the M1's ~68 GB/s — read 15.14
// tok/s against 15.11 for the byte-at-a-time form, +0.2%, well inside the
// run-to-run spread. SmolLM2-135M-Q8_0 cannot see it at all (107.13 vs 106.68,
// +0.4%): 145 MB at ~107 tok/s is 19 GB/s, i.e. dispatch-bound, not
// bandwidth-bound. It is kept because it is a strict reduction in issued loads
// at identical arithmetic, not because it was measured faster here.
// See docs/negative-result-metal-multirow-matvec.md for the rest of the sweep,
// including the multi-row arrangement that measured WORSE and is not here.
kernel void k_mv_q8_0(MV_PARAMS) {
    MV_HEAD;
    int nb = a.n_in / 32;
    device const uchar *rw = wb + a.w_off + (ulong)row * nb * 34;
    float s = 0;
    for (int b = tiisg; b < nb; b += 32) {
        device const uchar *blk = rw + (ulong)b * 34;
        float d = (float)*(device const half *)blk;
        // packed_char4: the quants start two bytes into a 34-byte block, so
        // nothing about their address is 4-byte aligned.
        device const packed_char4 *q = (device const packed_char4 *)(blk + 2);
        device const packed_float4 *xp = (device const packed_float4 *)(x + b * 32);
        float t = 0;
        for (int k = 0; k < 8; k++) {
            char4 qq = q[k];
            float4 xv = xp[k];
            t += (float)qq.x * xv.x;
            t += (float)qq.y * xv.y;
            t += (float)qq.z * xv.z;
            t += (float)qq.w * xv.w;
        }
        s += d * t;
    }
    MV_TAIL;
}

kernel void k_mv_q4_0(MV_PARAMS) {
    MV_HEAD;
    int nb = a.n_in / 32;
    device const uchar *rw = wb + a.w_off + (ulong)row * nb * 18;
    float s = 0;
    for (int b = tiisg; b < nb; b += 32) {
        device const uchar *blk = rw + (ulong)b * 18;
        float d = (float)*(device const half *)blk;
        device const packed_uchar4 *q = (device const packed_uchar4 *)(blk + 2);
        device const packed_float4 *xp = (device const packed_float4 *)(x + b * 32);
        float t = 0;
        for (int k = 0; k < 4; k++) {
            uchar4 qq = q[k];
            float4 xl = xp[k], xh = xp[k + 4];
            t += ((int)(qq.x & 0xF) - 8) * xl.x + ((int)(qq.x >> 4) - 8) * xh.x;
            t += ((int)(qq.y & 0xF) - 8) * xl.y + ((int)(qq.y >> 4) - 8) * xh.y;
            t += ((int)(qq.z & 0xF) - 8) * xl.z + ((int)(qq.z >> 4) - 8) * xh.z;
            t += ((int)(qq.w & 0xF) - 8) * xl.w + ((int)(qq.w >> 4) - 8) * xh.w;
        }
        s += d * t;
    }
    MV_TAIL;
}

kernel void k_mv_q4_1(MV_PARAMS) {
    MV_HEAD;
    int nb = a.n_in / 32;
    device const uchar *rw = wb + a.w_off + (ulong)row * nb * 20;
    float s = 0;
    for (int b = tiisg; b < nb; b += 32) {
        device const uchar *blk = rw + (ulong)b * 20;
        float d = (float)*(device const half *)blk;
        float mm = (float)*(device const half *)(blk + 2);
        device const uchar *q = blk + 4;
        device const float *xp = x + b * 32;
        float t = 0, sx = 0;
        for (int j = 0; j < 16; j++) {
            t += (float)(q[j] & 0xF) * xp[j] + (float)(q[j] >> 4) * xp[j + 16];
            sx += xp[j] + xp[j + 16];
        }
        s += d * t + mm * sx;
    }
    MV_TAIL;
}

kernel void k_mv_q5_0(MV_PARAMS) {
    MV_HEAD;
    int nb = a.n_in / 32;
    device const uchar *rw = wb + a.w_off + (ulong)row * nb * 22;
    float s = 0;
    for (int b = tiisg; b < nb; b += 32) {
        device const uchar *blk = rw + (ulong)b * 22;
        float d = (float)*(device const half *)blk;
        uint qh = (uint)blk[2] | ((uint)blk[3] << 8) |
                  ((uint)blk[4] << 16) | ((uint)blk[5] << 24);
        device const uchar *q = blk + 6;
        device const float *xp = x + b * 32;
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

kernel void k_mv_q5_1(MV_PARAMS) {
    MV_HEAD;
    int nb = a.n_in / 32;
    device const uchar *rw = wb + a.w_off + (ulong)row * nb * 24;
    float s = 0;
    for (int b = tiisg; b < nb; b += 32) {
        device const uchar *blk = rw + (ulong)b * 24;
        float d = (float)*(device const half *)blk;
        float mm = (float)*(device const half *)(blk + 2);
        uint qh = (uint)blk[4] | ((uint)blk[5] << 8) |
                  ((uint)blk[6] << 16) | ((uint)blk[7] << 24);
        device const uchar *q = blk + 8;
        device const float *xp = x + b * 32;
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

static inline void get_scale_min_k4(int j, device const uchar *q,
                                    thread uchar *d, thread uchar *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4)  | ((q[j    ] >> 6) << 4);
    }
}

// bf16 is the top 16 bits of an f32, so widening is a shift and a reinterpret
// with no rounding anywhere -- exactly what bf16_to_f32() in fp16.h does, which
// is why this path can be bit-identical to the CPU rather than merely close.
// Metal has no bfloat on every target this ships to, so the weights are read as
// ushort and widened by hand.
static inline float bf16_to_f32_m(ushort h) {
    return as_type<float>((uint)h << 16);
}

// Four weights per lane step, not one. The scalar form left this kernel LOSING
// to the CPU (measured 0.91x on a 135M model) while same-size quantised models
// won 1.5-2.1x on the same device -- the difference was never model size, it
// was that every quant kernel here already loads wide and these did not.
// packed_ushort4 because the weight pointer is row-offset and carries no
// 8-byte alignment guarantee; x is float4-indexed the way k_mv_q5_K does it.
kernel void k_mv_bf16(MV_PARAMS) {
    MV_HEAD;
    device const ushort *rw =
        (device const ushort *)(wb + a.w_off) + (ulong)row * a.n_in;
    float s = 0;
    if ((a.n_in & 3) == 0) {
        device const packed_ushort4 *w4 = (device const packed_ushort4 *)rw;
        device const packed_float4 *x4 = (device const packed_float4 *)x;
        int n4 = a.n_in >> 2;
        float4 acc = 0;
        for (int i = tiisg; i < n4; i += 32) {
            ushort4 h = w4[i];
            acc += float4(as_type<float>((uint)h.x << 16),
                          as_type<float>((uint)h.y << 16),
                          as_type<float>((uint)h.z << 16),
                          as_type<float>((uint)h.w << 16)) * x4[i];
        }
        s = acc.x + acc.y + acc.z + acc.w;
    } else {
        for (int i = tiisg; i < a.n_in; i += 32) s += bf16_to_f32_m(rw[i]) * x[i];
    }
    MV_TAIL;
}

// The IQ4 non-linear codebook. Both iq4_nl and iq4_xs index it with a plain
// 4-bit code, which is the whole difference from q4_0: same 16 codes per byte
// pair, but the value is looked up rather than being the code minus eight.
// Must stay identical to kvalues_iq4nl in quants.c -- these kernels are gated
// on producing bit-identical output to that path.
constant int kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
};

// iq4_nl: 18-byte blocks of 32 weights, one f16 scale, no sub-block scales.
kernel void k_mv_iq4_nl(MV_PARAMS) {
    MV_HEAD;
    int nb = a.n_in / 32;
    device const uchar *rw = wb + a.w_off + (ulong)row * nb * 18;
    float s = 0;
    for (int b = tiisg; b < nb; b += 32) {
        device const uchar *blk = rw + (ulong)b * 18;
        float d = (float)*(device const half *)blk;
        device const uchar *q = blk + 2;
        device const float *xp = x + b * 32;
        float t = 0;
        for (int j = 0; j < 16; j++)
            t += (float)kvalues_iq4nl[q[j] & 0xF] * xp[j]
               + (float)kvalues_iq4nl[q[j] >> 4]  * xp[j + 16];
        s += d * t;
    }
    MV_TAIL;
}

// iq4_xs: 136-byte superblocks of 256, worked one 32-weight sub-block per lane
// iteration. Each sub-block carries a 6-bit scale split across two places --
// four low bits packed two-per-byte in scales_l, and the top two bits in a
// 16-bit scales_h field -- biased by 32 like q3_K's. scales_h is read as two
// bytes rather than a ushort because a 136-byte block gives offset 2 no
// two-byte alignment guarantee.
kernel void k_mv_iq4_xs(MV_PARAMS) {
    MV_HEAD;
    int nb = a.n_in / 256;
    device const uchar *rw = wb + a.w_off + (ulong)row * nb * 136;
    float s = 0;
    int nq = nb * 8;
    for (int u = tiisg; u < nq; u += 32) {
        int b = u >> 3, ib = u & 7;
        device const uchar *blk = rw + (ulong)b * 136;
        float d = (float)*(device const half *)blk;
        uint sh = (uint)blk[2] | ((uint)blk[3] << 8);
        int ls = ((blk[4 + (ib >> 1)] >> (4 * (ib & 1))) & 0xF) |
                 (((sh >> (2 * ib)) & 3) << 4);
        float dl = d * (float)(ls - 32);
        device const uchar *q  = blk + 8 + ib * 16;
        device const float *xp = x + b * 256 + ib * 32;
        float t = 0;
        for (int j = 0; j < 16; j++)
            t += (float)kvalues_iq4nl[q[j] & 0xF] * xp[j]
               + (float)kvalues_iq4nl[q[j] >> 4]  * xp[j + 16];
        s += dl * t;
    }
    MV_TAIL;
}

// q3_K's 16 six-bit scales live packed in 12 bytes: the low 4 bits of each of
// the first 8 bytes, with the top 2 bits of each scale distributed through the
// last 4. This is the same shuffle the CPU path does with three uint32 words;
// the bytes are assembled by hand rather than cast because a 110-byte block
// leaves scales at blk+96 with no alignment guarantee.
static inline void q3k_scales(device const uchar *s12, thread char *out16) {
    const uint kmask1 = 0x03030303u, kmask2 = 0x0f0f0f0fu;
    uint aux[4];
    for (int i = 0; i < 3; i++)
        aux[i] = (uint)s12[i * 4 + 0]        | ((uint)s12[i * 4 + 1] << 8) |
                ((uint)s12[i * 4 + 2] << 16) | ((uint)s12[i * 4 + 3] << 24);
    uint tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = ( aux[0]       & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = ( aux[1]       & kmask2) | (((tmp >> 2) & kmask1) << 4);
    for (int i = 0; i < 4; i++)
        for (int b = 0; b < 4; b++)
            out16[i * 4 + b] = (char)((aux[i] >> (8 * b)) & 0xFF);
}

// q3_K, quarter-superblocks, same decomposition as q2_K below. Two differences
// from q2_K: the scales are the packed 6-bit set above (biased by 32, and
// signed after the bias, so a scale can be negative), and every weight carries
// a third bit in a separate 32-byte hmask that SUBTRACTS 4 when clear. That
// mask is indexed by weight position within the whole superblock, so unlike qs
// it does not advance with the half -- both halves read the same 32 bytes and
// pick different bits, one bit per (half, shift) pair.
//
// Scalar for the same measured reason as k_mv_q4_K. Do not vectorise blind.
kernel void k_mv_q3_K(MV_PARAMS) {
    MV_HEAD;
    int nb = a.n_in / 256;
    device const uchar *rw = wb + a.w_off + (ulong)row * nb * 110;
    float s = 0;
    int nq = nb * 4;                       // quarter-superblocks on this row
    for (int u = tiisg; u < nq; u += 32) {
        int b = u >> 2, jj = u & 3;
        device const uchar *blk = rw + (ulong)b * 110;
        device const uchar *hm  = blk;                        // hmask[32]
        device const uchar *q   = blk + 32 + (jj >> 1) * 32;  // qs[64]
        float d_all = (float)*(device const half *)(blk + 108);
        char sc[16];
        q3k_scales(blk + 96, sc);
        device const float *xp = x + b * 256 + jj * 64;
        int si = (jj >> 1) * 8 + (jj & 1) * 4;
        uchar sh0 = (uchar)((jj & 1) * 4), sh1 = (uchar)(sh0 + 2);
        uchar m0 = (uchar)(1u << ((jj >> 1) * 4 + (jj & 1) * 2));
        uchar m1 = (uchar)(m0 << 1);
        float d0 = d_all * (float)((int)sc[si + 0] - 32);
        float d1 = d_all * (float)((int)sc[si + 1] - 32);
        float d2 = d_all * (float)((int)sc[si + 2] - 32);
        float d3 = d_all * (float)((int)sc[si + 3] - 32);
        float t0 = 0, t1 = 0, t2 = 0, t3 = 0;
        for (int l = 0; l < 16; l++) {
            uchar qa = q[l],  qb = q[l + 16];
            uchar ha = hm[l], hb = hm[l + 16];
            t0 += (float)((int)((qa >> sh0) & 3) - ((ha & m0) ? 0 : 4)) * xp[l];
            t1 += (float)((int)((qb >> sh0) & 3) - ((hb & m0) ? 0 : 4)) * xp[l + 16];
            t2 += (float)((int)((qa >> sh1) & 3) - ((ha & m1) ? 0 : 4)) * xp[l + 32];
            t3 += (float)((int)((qb >> sh1) & 3) - ((hb & m1) ? 0 : 4)) * xp[l + 48];
        }
        s += d0 * t0 + d1 * t1 + d2 * t2 + d3 * t3;
    }
    MV_TAIL;
}

// q2_K, in quarter-superblocks for the same occupancy reason as q4_K below.
//
// The block is 84 bytes: scales[16], qs[64], then d and dmin as halves at the
// END (offsets 80 and 82), unlike q4_K/q5_K which lead with them. Its 256
// weights are two halves of 128; each half owns 32 bytes of qs and 8 scale
// bytes, and within a half the four 2-bit shifts (0,2,4,6) each cover 32
// weights as two groups of 16 (qs[0..15] and qs[16..31]). Each scale byte
// carries BOTH a scale in its low nibble and a min in its high nibble, so
// unlike q4_K there is no 6-bit unpacking helper to call.
//
// A quarter (64 weights) is therefore one half's worth of two adjacent shifts:
// h = jj>>1 picks the half, and the four scale bytes at h*8 + (jj&1)*4 are
// exactly (shift0,group0), (shift0,group1), (shift1,group0), (shift1,group1).
// The 32 qs bytes are re-read by the other quarter of the same half, which is
// the same cache-hit-for-occupancy trade k_mv_q5_K documents for its qh.
//
// Kept scalar deliberately, matching k_mv_q4_K: hand-vectorising that one was
// measured SLOWER (see its comment). Do not vectorise this without measuring.
kernel void k_mv_q2_K(MV_PARAMS) {
    MV_HEAD;
    int nb = a.n_in / 256;
    device const uchar *rw = wb + a.w_off + (ulong)row * nb * 84;
    float s = 0;
    int nq = nb * 4;                       // quarter-superblocks on this row
    for (int u = tiisg; u < nq; u += 32) {
        int b = u >> 2, jj = u & 3;
        device const uchar *blk = rw + (ulong)b * 84;
        device const uchar *sc = blk + (jj >> 1) * 8 + (jj & 1) * 4;
        device const uchar *q  = blk + 16 + (jj >> 1) * 32;
        float d    = (float)*(device const half *)(blk + 80);
        float dmin = (float)*(device const half *)(blk + 82);
        device const float *xp = x + b * 256 + jj * 64;
        uchar sh0 = (uchar)((jj & 1) * 4), sh1 = (uchar)(sh0 + 2);
        uchar c0 = sc[0], c1 = sc[1], c2 = sc[2], c3 = sc[3];
        float d0 = d * (c0 & 0xF), m0 = dmin * (c0 >> 4);
        float d1 = d * (c1 & 0xF), m1 = dmin * (c1 >> 4);
        float d2 = d * (c2 & 0xF), m2 = dmin * (c2 >> 4);
        float d3 = d * (c3 & 0xF), m3 = dmin * (c3 >> 4);
        float t0 = 0, t1 = 0, t2 = 0, t3 = 0;
        float sx0 = 0, sx1 = 0, sx2 = 0, sx3 = 0;
        for (int l = 0; l < 16; l++) {
            uchar qa = q[l], qb = q[l + 16];
            float x0 = xp[l], x1 = xp[l + 16], x2 = xp[l + 32], x3 = xp[l + 48];
            t0 += (float)((qa >> sh0) & 3) * x0; sx0 += x0;
            t1 += (float)((qb >> sh0) & 3) * x1; sx1 += x1;
            t2 += (float)((qa >> sh1) & 3) * x2; sx2 += x2;
            t3 += (float)((qb >> sh1) & 3) * x3; sx3 += x3;
        }
        s += d0 * t0 - m0 * sx0 + d1 * t1 - m1 * sx1
           + d2 * t2 - m2 * sx2 + d3 * t3 - m3 * sx3;
    }
    MV_TAIL;
}

// Work is handed out in QUARTER-superblocks (64 weights), not whole ones.
// Whole-superblock striding starved the simdgroup on every matvec whose input
// is n_embd: nb = n_in/256 is 10 for a 2560-wide model, so 10 lanes worked and
// 22 idled — 31% occupancy on q/k/v/o/gate/up, and only 62% on the wider
// ffn_down (40 blocks over 32 lanes). Idle lanes cost more than their
// arithmetic here: a lane that does nothing also issues no load, so the
// simdgroup had 10 memory requests in flight where it could have had 32, and
// decode never came close to memory peak. Quartering lifts those to 62% and
// 100%.
//
// A quarter is the smallest unit that does not re-read weights: 64 weights
// live in 32 bytes of q as low and high nibbles, so splitting finer would
// fetch the same bytes twice.
kernel void k_mv_q4_K(MV_PARAMS) {
    MV_HEAD;
    int nb = a.n_in / 256;
    device const uchar *rw = wb + a.w_off + (ulong)row * nb * 144;
    float s = 0;
    int nq = nb * 4;                       // quarter-superblocks on this row
    for (int u = tiisg; u < nq; u += 32) {
        int b = u >> 2, jj = u & 3;
        device const uchar *blk = rw + (ulong)b * 144;
        float d    = (float)*(device const half *)blk;
        float dmin = (float)*(device const half *)(blk + 2);
        device const uchar *sc = blk + 4;
        device const packed_uchar4 *q4 =
            (device const packed_uchar4 *)(blk + 16 + jj * 32);
        device const packed_float4 *xp4 =
            (device const packed_float4 *)(x + b * 256 + jj * 64);
        uchar s1, m1, s2, m2;
        get_scale_min_k4(jj * 2 + 0, sc, &s1, &m1);
        get_scale_min_k4(jj * 2 + 1, sc, &s2, &m2);
        float d1 = d * s1, mm1 = dmin * m1;
        float d2 = d * s2, mm2 = dmin * m2;
        // Hand-vectorising this loop was tried on 2026-08-07 and measured
        // SLOWER (6.58 -> 6.43 tok/s) — but that attempt moved the SUM into
        // float4 accumulators and paid for four horizontal reductions the
        // scalar form never needs. This one widens only the LOADS: the four
        // accumulators stay scalar and every `+=` keeps its original
        // expression, so the arithmetic is untouched and what disappears is 96
        // of the 128 load instructions per quarter-superblock.
        float t1 = 0, t2 = 0, sx1 = 0, sx2 = 0;
        for (int k = 0; k < 8; k++) {
            uchar4 qq = q4[k];
            float4 xl = xp4[k], xh = xp4[k + 8];
            t1 += (float)(qq.x & 0xF) * xl.x;  sx1 += xl.x;
            t2 += (float)(qq.x >> 4)  * xh.x;  sx2 += xh.x;
            t1 += (float)(qq.y & 0xF) * xl.y;  sx1 += xl.y;
            t2 += (float)(qq.y >> 4)  * xh.y;  sx2 += xh.y;
            t1 += (float)(qq.z & 0xF) * xl.z;  sx1 += xl.z;
            t2 += (float)(qq.z >> 4)  * xh.z;  sx2 += xh.z;
            t1 += (float)(qq.w & 0xF) * xl.w;  sx1 += xl.w;
            t2 += (float)(qq.w >> 4)  * xh.w;  sx2 += xh.w;
        }
        s += d1 * t1 - mm1 * sx1 + d2 * t2 - mm2 * sx2;
    }
    MV_TAIL;
}

// Quarter-superblocks per lane and a vectorised body, the same two changes
// that took q4_K and q6_K decode from behind the CPU path to ahead of it. The
// only wrinkle is qh: unlike q4_K's q pointer it does NOT advance per quarter
// — all four quarters read the same 32 bytes and pick different bit pairs out
// of them — so quartering re-reads 32 of the block's 176 bytes three extra
// times. That is a cache hit against a 22-lane occupancy win, and the
// measurement agrees.
kernel void k_mv_q5_K(MV_PARAMS) {
    MV_HEAD;
    int nb = a.n_in / 256;
    device const uchar *rw = wb + a.w_off + (ulong)row * nb * 176;
    float s = 0;
    int nq = nb * 4;                       // quarter-superblocks on this row
    for (int u = tiisg; u < nq; u += 32) {
        int b = u >> 2, jj = u & 3;
        device const uchar *blk = rw + (ulong)b * 176;
        float d    = (float)*(device const half *)blk;
        float dmin = (float)*(device const half *)(blk + 2);
        device const uchar *sc = blk + 4;
        device const uchar *qh = blk + 16;
        device const uchar *q  = blk + 48 + jj * 32;
        device const float *xp = x + b * 256 + jj * 64;
        uchar s1, m1, s2, m2;
        get_scale_min_k4(jj * 2 + 0, sc, &s1, &m1);
        get_scale_min_k4(jj * 2 + 1, sc, &s2, &m2);
        float d1 = d * s1, mm1 = dmin * m1;
        float d2 = d * s2, mm2 = dmin * m2;
        uchar u1 = (uchar)(1u << (2 * jj)), u2 = (uchar)(2u << (2 * jj));
        // packed_uchar4 for the same reason as q6_K: 176 is a multiple of 16,
        // but q sits at blk+48+jj*32 and qh at blk+16, so only packed types
        // are safe without re-deriving alignment for every offset. The
        // activations are packed too, and for a reason that had been missed:
        // x is x_all + col * x_stride and x_stride is a model dimension, so a
        // plain float4 load off it was asserting a 16-byte alignment the
        // caller never promised.
        device const packed_uchar4 *q4  = (device const packed_uchar4 *)q;
        device const packed_uchar4 *qh4 = (device const packed_uchar4 *)qh;
        device const packed_float4 *xlo = (device const packed_float4 *)xp;
        device const packed_float4 *xhi = (device const packed_float4 *)(xp + 32);
        float4 a1v = 0, a2v = 0;
        for (int k = 0; k < 8; k++) {
            uchar4 qq = q4[k], hh = qh4[k];
            float4 hi1 = select(float4(0.0f), float4(16.0f), (hh & u1) != 0);
            float4 hi2 = select(float4(0.0f), float4(16.0f), (hh & u2) != 0);
            a1v += (d1 * (float4(qq & 0xF) + hi1) - mm1) * xlo[k];
            a2v += (d2 * (float4(qq >> 4)  + hi2) - mm2) * xhi[k];
        }
        s += a1v.x + a1v.y + a1v.z + a1v.w + a2v.x + a2v.y + a2v.z + a2v.w;
    }
    MV_TAIL;
}

// Half-superblocks per lane, for the same reason k_mv_q4_K uses quarters: at
// nb = n_in/256 = 10, whole-block striding left 22 of 32 lanes idle and the
// simdgroup with a third of the memory requests it could have had in flight.
// A half (128 weights) is the natural unit here — ql[l] and ql[l+32] both feed
// one l-iteration, and qh packs four weights' high bits into one byte, so
// splitting finer would re-read those bytes. 31% -> 62% occupancy on the
// n_embd-wide matvecs, 62% -> 100% on ffn_down.
kernel void k_mv_q6_K(MV_PARAMS) {
    MV_HEAD;
    int nb = a.n_in / 256;
    device const uchar *rw = wb + a.w_off + (ulong)row * nb * 210;
    float s = 0;
    int nh = nb * 2;                       // half-superblocks on this row
    for (int u = tiisg; u < nh; u += 32) {
        int b = u >> 1, half_i = u & 1;
        device const uchar *blk = rw + (ulong)b * 210;
        device const uchar *ql = blk + half_i * 64;
        device const uchar *qh = blk + 128 + half_i * 32;
        device const char  *sc = (device const char *)(blk + 192) + half_i * 8;
        float d = (float)*(device const half *)(blk + 208);
        device const float *xp = x + b * 256 + half_i * 128;
        float t[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int is = 0; is < 2; is++) {
            // packed_uchar4, NOT uchar4: a q6_K superblock is 210 bytes, which
            // is not a multiple of 4, so ql/qh sit at odd alignments for odd
            // blocks and a uchar4 load there is undefined. It happened to
            // return correct results on an M1 and passed the tolerance gate —
            // which is exactly why it needed catching by reading the block
            // size rather than by trusting the test. packed_uchar4 has
            // 1-byte alignment and compiles to the same loads where the
            // address does happen to be aligned.
            device const packed_uchar4 *l4  = (device const packed_uchar4 *)(ql + is * 16);
            device const packed_uchar4 *l4b = (device const packed_uchar4 *)(ql + 32 + is * 16);
            device const packed_uchar4 *h4  = (device const packed_uchar4 *)(qh + is * 16);
            device const packed_float4 *x0 = (device const packed_float4 *)(xp + is * 16);
            device const packed_float4 *x1 = (device const packed_float4 *)(xp + 32 + is * 16);
            device const packed_float4 *x2 = (device const packed_float4 *)(xp + 64 + is * 16);
            device const packed_float4 *x3 = (device const packed_float4 *)(xp + 96 + is * 16);
            float4 a0 = 0, a1 = 0, a2 = 0, a3 = 0;
            for (int k = 0; k < 4; k++) {
                uchar4 lo = l4[k], hi = l4b[k], h = h4[k];   // widen on load
                a0 += (float4(lo & 0xF) + float4((h >> 0) & 3) * 16.0f - 32.0f) * x0[k];
                a1 += (float4(hi & 0xF) + float4((h >> 2) & 3) * 16.0f - 32.0f) * x1[k];
                a2 += (float4(lo >> 4)  + float4((h >> 4) & 3) * 16.0f - 32.0f) * x2[k];
                a3 += (float4(hi >> 4)  + float4((h >> 6) & 3) * 16.0f - 32.0f) * x3[k];
            }
            t[is * 4 + 0] = a0.x + a0.y + a0.z + a0.w;
            t[is * 4 + 1] = a1.x + a1.y + a1.z + a1.w;
            t[is * 4 + 2] = a2.x + a2.y + a2.z + a2.w;
            t[is * 4 + 3] = a3.x + a3.y + a3.z + a3.w;
        }
        s += d * (sc[0] * t[0] + sc[2] * t[1] + sc[4] * t[2] + sc[6] * t[3] +
                  sc[1] * t[4] + sc[3] * t[5] + sc[5] * t[6] + sc[7] * t[7]);
    }
    MV_TAIL;
}

constant float kv_mxfp4[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
     0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

kernel void k_mv_mxfp4(MV_PARAMS) {
    MV_HEAD;
    int nb = a.n_in / 32;
    device const uchar *rw = wb + a.w_off + (ulong)row * nb * 17;
    float s = 0;
    for (int b = tiisg; b < nb; b += 32) {
        device const uchar *blk = rw + (ulong)b * 17;
        float d = ldexp(1.0f, (int)blk[0] - 127);
        device const uchar *q = blk + 1;
        device const float *xp = x + b * 32;
        float t = 0;
        for (int j = 0; j < 16; j++) {
            t += kv_mxfp4[q[j] & 0xF] * xp[j];
            t += kv_mxfp4[q[j] >> 4]  * xp[j + 16];
        }
        s += d * t;
    }
    MV_TAIL;
}


// ------------------------------------------- fast (reassociating) decode matvec
// The k_mv_* family above IS the CPU<->GPU byte-identity contract, and that
// contract is what caps it. Each output row accumulates over
// `for (i = tiisg; i < n; i += 32)` into one scalar and finishes with one
// simd_sum, so which lane owns which block decides which partial sums enter
// the reduction tree. The 2026-08-12 sweep
// (docs/negative-result-metal-multirow-matvec.md) measured the two levers that
// survive that pin — wider loads (neutral) and rows-per-simdgroup (zero-sum,
// because simdgroups x rows is fixed at n_out) — and concluded that the
// roofline gap needs the transformations identity forbids, on a SECOND kernel
// promoted by teacher-forced tolerance the way the tiled GEMM below already is.
//
// This is that kernel family. It answers to tests/test_mv_tol.c through
// gpu_mv_force(); the identity family stays reachable and byte-for-byte
// unchanged behind RUNNER_METAL_MV=0.
//
// What it does that identity forbids:
//
//   * float4 accumulation. The identity form must spell each product as its
//     own scalar `+=` in the original expression order; here four lanes of a
//     float4 accumulate independently and are summed horizontally at the end.
//     That is a different reduction tree, and it is the point: it turns the
//     per-element scalar FMA chain into one vector FMA per four elements.
//   * the q4_0 zero-point factored out of the inner loop. sum (q-8)*y*d
//     becomes d*(sum q*y) - 8d*(sum y), so the per-element integer subtract
//     disappears and the nibble goes straight to float.
//
// The hypothesis being tested, stated so a later reader can check it rather
// than re-derive it: at 36% of roofline this route is ISSUE-bound, not
// traffic-bound — weight bytes are irreducible (every byte is read exactly
// once either way), so the only thing left to cut is issued instructions per
// byte consumed. If that hypothesis is wrong the gate still passes and the
// bench simply shows nothing, which is a result worth recording either way.
//
// Decode only: enc_mv_n selects this at n_col == 1. Prefill already has the
// tiled GEMM, which is a strictly better answer for a real batch.

kernel void k_mvf_q4_0(MV_PARAMS) {
    MV_HEAD;
    int nb = a.n_in / 32;
    device const uchar *rw = wb + a.w_off + (ulong)row * nb * 18;
    float4 acc = 0;      // sum over blocks of d * (q . y)
    float  corr = 0;     // sum over blocks of d * (sum y), the -8 zero-point
    for (int b = tiisg; b < nb; b += 32) {
        device const uchar *blk = rw + (ulong)b * 18;
        float d = (float)*(device const half *)blk;
        device const packed_uchar4 *q = (device const packed_uchar4 *)(blk + 2);
        device const packed_float4 *xp = (device const packed_float4 *)(x + b * 32);
        float4 t = 0, sy = 0;
        for (int k = 0; k < 4; k++) {
            uchar4 qq = q[k];
            float4 xl = xp[k], xh = xp[k + 4];
            t  += float4(qq & 0xF) * xl;
            t  += float4(qq >> 4)  * xh;
            sy += xl + xh;
        }
        acc  += d * t;
        corr += d * (sy.x + sy.y + sy.z + sy.w);
    }
    float s = (acc.x + acc.y + acc.z + acc.w) - 8.0f * corr;
    MV_TAIL;
}

// q8_0 has no zero-point to factor, so this is the float4-accumulator half of
// the change alone: 8 vector FMAs per block against 32 scalar ones.
kernel void k_mvf_q8_0(MV_PARAMS) {
    MV_HEAD;
    int nb = a.n_in / 32;
    device const uchar *rw = wb + a.w_off + (ulong)row * nb * 34;
    float4 acc = 0;
    for (int b = tiisg; b < nb; b += 32) {
        device const uchar *blk = rw + (ulong)b * 34;
        float d = (float)*(device const half *)blk;
        device const packed_char4 *q = (device const packed_char4 *)(blk + 2);
        device const packed_float4 *xp = (device const packed_float4 *)(x + b * 32);
        float4 t = 0;
        for (int k = 0; k < 8; k++) t += float4(q[k]) * xp[k];
        acc += d * t;
    }
    float s = acc.x + acc.y + acc.z + acc.w;
    MV_TAIL;
}

// ------------------------------------------------------- tiled prefill GEMM
// The matvec kernels above give one output element per simdgroup: 32 lanes
// each do one FMA and then a log-depth reduction, so almost none of the work
// is reuse. This computes an output TILE per threadgroup with Apple's
// simdgroup matrix units instead, which is where prompt processing has its
// real headroom.
//
// It is deliberately NOT bit-identical to the matvec path and cannot be: the
// weight is dequantized before the multiply (d*q)*x rather than d*(q*x), and
// the sum is reassociated into 8-element matrix steps. That is the same trade
// the CUDA tensor-core prefill makes, so it answers to the same kind of gate —
// tests/test_tc_tol.c, driven through gpu_tc_force() — rather than to an
// identity claim it cannot honour.
//
// Orientation: out[c][r] = sum_k W[r][k] * X[c][k], i.e. out = X * W^T. That
// makes the result tile row-major in the COLUMN index, which is exactly how y
// is laid out (column c at offset c*y_stride), so the store is contiguous.
// W is loaded transposed straight out of threadgroup memory.

// MM_TM/MM_TN must stay in step with the dispatch grid in metal.m
// (enc_mv_n's dispatchThreadgroups) and with tests/test_metal_kquants.m's
// `submit()` helper -- none of the three can see the other two, so a change
// here that is not mirrored there is a silent correctness bug, not a build
// error (the grid would just cover the wrong number of output tiles).
#define MM_TM 64    // output rows per threadgroup: multiple of 32 (4 simdgroups x 8)
#define MM_TN 32    // output columns (prompt tokens) per threadgroup: multiple of 8
#define MM_TK 32    // k-step: one q8_0/q4_0/mxfp4 block, and a q4_K sub-block
// Row-groups per simdgroup and column-groups per tile; every simdgroup owns a
// distinct MM_RPS-tall slice of the weight tile and all four cover the same
// MM_CG activation column-groups, for MM_RPS*MM_CG accumulators per thread.
#define MM_RPS (MM_TM / 32)
#define MM_CG  (MM_TN / 8)

struct mm_args {
    int   n_in, n_out, n_col;
    ulong w_off;
    int   has_bias, x_stride, y_stride;
};

#define MM_PARAMS \
    device const uchar *wb   [[buffer(0)]], \
    device const float *x    [[buffer(1)]], \
    device float       *y    [[buffer(2)]], \
    constant mm_args   &a    [[buffer(3)]], \
    device const float *bias [[buffer(4)]], \
    uint3 tgpig [[threadgroup_position_in_grid]], \
    uint3 tpitg [[thread_position_in_threadgroup]], \
    uint  sgitg [[simdgroup_index_in_threadgroup]]

// Shared prologue/epilogue; DEQ_CHUNK fills tg_w[r][0..MM_TK) for this
// thread's row/sub-range from the type's own block layout.
//
// tg_w/tg_x stage in HALF, not float: the simdgroup A/B operands are loaded
// as simdgroup_half8x8 and multiply-accumulated into a simdgroup_float8x8
// accumulator (MSL's mixed-precision simdgroup_multiply_accumulate, the same
// pattern llama.cpp's Metal mul_mm uses). That halves the threadgroup-memory
// traffic per tile and doubles the matrix-unit throughput on hardware that
// runs half-operand GEMM faster than float-operand GEMM, at the cost of a
// half-precision rounding step on every staged weight/activation value. That
// is exactly the trade this kernel family already answers to
// tests/test_tc_tol.c for (see the file comment above): not bit-identical to
// the scalar path, gated on teacher-forced logit deviation instead. The
// output accumulator (tg_c) and the final store stay float32 — only the
// per-tile operands round to half, not the reduction.
#define MM_BODY(...) \
    threadgroup half  tg_w[MM_TM * MM_TK]; \
    threadgroup half  tg_x[MM_TN * MM_TK]; \
    threadgroup float tg_c[MM_TN * MM_TM]; \
    const int row0 = (int)tgpig.x * MM_TM; \
    const int col0 = (int)tgpig.y * MM_TN; \
    const int tid  = (int)tpitg.x; \
    simdgroup_float8x8 acc[MM_RPS][MM_CG]; \
    for (int rg = 0; rg < MM_RPS; rg++) \
        for (int cg = 0; cg < MM_CG; cg++) acc[rg][cg] = simdgroup_float8x8(0.0f); \
    for (int k0 = 0; k0 < a.n_in; k0 += MM_TK) { \
        /* 128 threads x 8 values/pass = one 32-row x MM_TK weight slab;
           MM_RPS passes cover the full MM_TM-row tile */ \
        for (int p = 0; p < MM_RPS; p++) { \
            int r = p * 32 + (tid >> 2), sub = (tid & 3) * 8; \
            int row = row0 + r; \
            threadgroup half *dst = tg_w + r * MM_TK + sub; \
            if (row < a.n_out) { __VA_ARGS__; } \
            else { for (int j = 0; j < 8; j++) dst[j] = 0.0h; } \
        } \
        /* 128 threads x 4 values/pass = the MM_TN x MM_TK activation tile */ \
        for (int i = tid * 4; i < MM_TN * MM_TK; i += 128 * 4) { \
            for (int j = 0; j < 4; j++) { \
                int idx = i + j, cc = idx / MM_TK, kk = idx % MM_TK; \
                tg_x[idx] = (half)((col0 + cc < a.n_col) \
                    ? x[(ulong)(col0 + cc) * a.x_stride + k0 + kk] : 0.0f); \
            } \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        for (int kk = 0; kk < MM_TK / 8; kk++) { \
            simdgroup_half8x8 acols[MM_CG]; \
            for (int cg = 0; cg < MM_CG; cg++) \
                simdgroup_load(acols[cg], tg_x + cg * 8 * MM_TK + kk * 8, MM_TK); \
            for (int rg = 0; rg < MM_RPS; rg++) { \
                simdgroup_half8x8 B; \
                int wrg = (int)sgitg * MM_RPS + rg; \
                simdgroup_load(B, tg_w + wrg * 8 * MM_TK + kk * 8, MM_TK, \
                               ulong2(0, 0), true); \
                for (int cg = 0; cg < MM_CG; cg++) \
                    simdgroup_multiply_accumulate(acc[rg][cg], acols[cg], B, \
                                                  acc[rg][cg]); \
            } \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
    } \
    for (int rg = 0; rg < MM_RPS; rg++) { \
        int wrg = (int)sgitg * MM_RPS + rg; \
        for (int cg = 0; cg < MM_CG; cg++) \
            simdgroup_store(acc[rg][cg], tg_c + cg * 8 * MM_TM + wrg * 8, MM_TM); \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    for (int idx = tid; idx < MM_TN * MM_TM; idx += 128) { \
        int cc = idx / MM_TM, rr = idx % MM_TM; \
        if (col0 + cc < a.n_col && row0 + rr < a.n_out) \
            y[(ulong)(col0 + cc) * a.y_stride + row0 + rr] = \
                a.has_bias ? tg_c[idx] + bias[row0 + rr] : tg_c[idx]; \
    }

kernel void k_mm_f32(MM_PARAMS) {
    MM_BODY({
        device const float *rw = (device const float *)(wb + a.w_off)
                               + (ulong)row * a.n_in + k0 + sub;
        for (int j = 0; j < 8; j++) dst[j] = rw[j];
    })
}

kernel void k_mm_f16(MM_PARAMS) {
    MM_BODY({
        device const half *rw = (device const half *)(wb + a.w_off)
                              + (ulong)row * a.n_in + k0 + sub;
        for (int j = 0; j < 8; j++) dst[j] = (float)rw[j];
    })
}

kernel void k_mm_q8_0(MM_PARAMS) {
    MM_BODY({
        int nb = a.n_in / 32;
        device const uchar *blk = wb + a.w_off + ((ulong)row * nb + k0 / 32) * 34;
        float d = (float)*(device const half *)blk;
        device const char *q = (device const char *)(blk + 2) + sub;
        for (int j = 0; j < 8; j++) dst[j] = d * (float)q[j];
    })
}

kernel void k_mm_q4_0(MM_PARAMS) {
    MM_BODY({
        int nb = a.n_in / 32;
        device const uchar *blk = wb + a.w_off + ((ulong)row * nb + k0 / 32) * 18;
        float d = (float)*(device const half *)blk;
        device const uchar *q = blk + 2;
        /* low nibbles hold lanes 0-15, high nibbles lanes 16-31 */
        for (int j = 0; j < 8; j++) {
            int lane = sub + j;
            int v = lane < 16 ? (int)(q[lane] & 0xF) : (int)(q[lane - 16] >> 4);
            dst[j] = d * (float)(v - 8);
        }
    })
}

// Tiled-GEMM variants for the formats that only had a matvec. Without these,
// prefill fell back to the matvec path and ran SLOWER on the GPU than on the
// CPU (measured: tinyllama q2_K 36.9 -> 28.2 tok/s prompt, iq4_xs 101.5 ->
// 87.5) because the matvec re-reads every weight once per token column, while
// this reads each weight tile once for sixteen. Types that already had a
// k_mm gained 3.7-5.3x on the same measurement, which is the prize here.
//
// MM_TK is 32, and `sub` is a multiple of 8, so each chunk of 8 stays inside
// one block AND inside one nibble half -- which is what makes these as simple
// as picking the right base pointer and shift.

kernel void k_mm_bf16(MM_PARAMS) {
    MM_BODY({
        device const ushort *rw = (device const ushort *)(wb + a.w_off)
                                + (ulong)row * a.n_in + k0 + sub;
        for (int j = 0; j < 8; j++) dst[j] = bf16_to_f32_m(rw[j]);
    })
}

kernel void k_mm_iq4_nl(MM_PARAMS) {
    MM_BODY({
        int k = k0 + sub, b = k >> 5, t0 = k & 31;
        device const uchar *blk =
            wb + a.w_off + ((ulong)row * (a.n_in >> 5) + b) * 18;
        float d = (float)*(device const half *)blk;
        device const uchar *q = blk + 2;
        bool hi = t0 >= 16;
        int base = hi ? t0 - 16 : t0;
        for (int j = 0; j < 8; j++) {
            uchar by = q[base + j];
            dst[j] = d * (float)kvalues_iq4nl[hi ? (by >> 4) : (by & 0xF)];
        }
    })
}

kernel void k_mm_iq4_xs(MM_PARAMS) {
    MM_BODY({
        int k = k0 + sub, nsb = a.n_in / 256;
        int sb = k >> 8, sbk = k & 255, ib = sbk >> 5, t0 = sbk & 31;
        device const uchar *blk = wb + a.w_off + ((ulong)row * nsb + sb) * 136;
        float d = (float)*(device const half *)blk;
        uint sh = (uint)blk[2] | ((uint)blk[3] << 8);
        int ls = ((blk[4 + (ib >> 1)] >> (4 * (ib & 1))) & 0xF) |
                 (((sh >> (2 * ib)) & 3) << 4);
        float dl = d * (float)(ls - 32);
        device const uchar *q = blk + 8 + ib * 16;
        bool hi = t0 >= 16;
        int base = hi ? t0 - 16 : t0;
        for (int j = 0; j < 8; j++) {
            uchar by = q[base + j];
            dst[j] = dl * (float)kvalues_iq4nl[hi ? (by >> 4) : (by & 0xF)];
        }
    })
}

kernel void k_mm_q2_K(MM_PARAMS) {
    MM_BODY({
        int k = k0 + sub, nsb = a.n_in / 256;
        int sb = k >> 8, sbk = k & 255;
        int h = sbk >> 7, rem = sbk & 127, j32 = rem >> 5, rem2 = rem & 31;
        int g = rem2 >> 4, l0 = rem2 & 15;
        device const uchar *blk = wb + a.w_off + ((ulong)row * nsb + sb) * 84;
        float d    = (float)*(device const half *)(blk + 80);
        float dmin = (float)*(device const half *)(blk + 82);
        uchar scb  = blk[h * 8 + j32 * 2 + g];
        float dl = d * (scb & 0xF), ml = dmin * (scb >> 4);
        device const uchar *q = blk + 16 + h * 32 + g * 16;
        uchar shift = (uchar)(2 * j32);
        for (int j = 0; j < 8; j++)
            dst[j] = dl * (float)((q[l0 + j] >> shift) & 3) - ml;
    })
}

kernel void k_mm_q3_K(MM_PARAMS) {
    MM_BODY({
        int k = k0 + sub, nsb = a.n_in / 256;
        int sb = k >> 8, sbk = k & 255;
        int h = sbk >> 7, rem = sbk & 127, j32 = rem >> 5, rem2 = rem & 31;
        int g = rem2 >> 4, l0 = rem2 & 15;
        device const uchar *blk = wb + a.w_off + ((ulong)row * nsb + sb) * 110;
        float d_all = (float)*(device const half *)(blk + 108);
        char sc[16];
        q3k_scales(blk + 96, sc);
        float dl = d_all * (float)((int)sc[h * 8 + j32 * 2 + g] - 32);
        device const uchar *hm = blk + g * 16;          /* not offset by half */
        device const uchar *q  = blk + 32 + h * 32 + g * 16;
        uchar shift = (uchar)(2 * j32);
        uchar mbit  = (uchar)(1u << (h * 4 + j32));
        for (int j = 0; j < 8; j++)
            dst[j] = dl * (float)((int)((q[l0 + j] >> shift) & 3)
                                  - ((hm[l0 + j] & mbit) ? 0 : 4));
    })
}

kernel void k_mm_q4_K(MM_PARAMS) {
    MM_BODY({
        int nsb = a.n_in / 256;
        int sb  = k0 / 256;              /* superblock */
        int j32 = (k0 % 256) / 32;       /* which 32-lane sub-block */
        device const uchar *blk = wb + a.w_off + ((ulong)row * nsb + sb) * 144;
        float dall = (float)*(device const half *)blk;
        float dmin = (float)*(device const half *)(blk + 2);
        device const uchar *sc = blk + 4;
        uchar s1, m1;
        get_scale_min_k4(j32, sc, &s1, &m1);
        float d = dall * s1, mm = dmin * m1;
        device const uchar *q = blk + 16 + (j32 / 2) * 32;
        bool hi = (j32 & 1) != 0;
        for (int j = 0; j < 8; j++) {
            int lane = sub + j;
            int v = hi ? (int)(q[lane] >> 4) : (int)(q[lane] & 0xF);
            dst[j] = d * (float)v - mm;
        }
    })
}

kernel void k_mm_q6_K(MM_PARAMS) {
    MM_BODY({
        int nsb = a.n_in / 256;
        int sb  = k0 / 256;
        int off = k0 % 256;              /* 0,32,...,224 */
        device const uchar *blk = wb + a.w_off + ((ulong)row * nsb + sb) * 210;
        float dall = (float)*(device const half *)(blk + 208);
        device const char *scs = (device const char *)(blk + 192);
        /* q6_K packs 128 lanes per half: ql[0..63] low nibbles + qh 2 bits */
        int half_i = off / 128, within = off % 128;
        device const uchar *ql = blk + half_i * 64;
        device const uchar *qh = blk + 128 + half_i * 32;
        device const char  *sc = scs + half_i * 8;
        for (int j = 0; j < 8; j++) {
            int lane = within + sub + j;      /* 0..127 within this half */
            int l = lane % 32, grp = lane / 32;
            /* groups 0,1 take low nibbles and 2,3 high; the byte index cycles
               l, l+32, l, l+32 — matching the reference's four q1..q4 lanes */
            int qb = (int)ql[l + (grp & 1) * 32];
            int qlv = (grp >= 2) ? (qb >> 4) : (qb & 0xF);
            int qhv = ((int)(qh[l] >> (grp * 2)) & 3) << 4;
            int is = (l / 16) & 1;
            dst[j] = dall * (float)sc[grp * 2 + is] * (float)((qlv | qhv) - 32);
        }
    })
}

kernel void k_mm_mxfp4(MM_PARAMS) {
    MM_BODY({
        int nb = a.n_in / 32;
        device const uchar *blk = wb + a.w_off + ((ulong)row * nb + k0 / 32) * 17;
        float d = ldexp(1.0f, (int)blk[0] - 127);
        device const uchar *q = blk + 1;
        for (int j = 0; j < 8; j++) {
            int lane = sub + j;
            uchar nib = lane < 16 ? (q[lane] & 0xF) : (q[lane - 16] >> 4);
            dst[j] = d * kv_mxfp4[nib];
        }
    })
}

// ---------------------------------------------------------------- rope

// pos is the FIRST column's position; column c rotates at pos + c, so a whole
// prompt batch ropes in one dispatch (grid.z selects the column).
struct rope_args {
    int   head_dim, n_heads, half_dim, pos, neox;
    float mscale;
    int   stride;     // elements between consecutive columns
};

kernel void k_rope(device float       *v_all [[buffer(0)]],
                   device const float *fr    [[buffer(1)]],
                   constant rope_args &a     [[buffer(2)]],
                   uint3 gid [[thread_position_in_grid]]) {
    int j = gid.x, h = gid.y, col = gid.z;
    if (j >= a.half_dim || h >= a.n_heads) return;
    float ang = (a.pos + col) * fr[j];
    float c = cos(ang) * a.mscale, s = sin(ang) * a.mscale;
    device float *v = v_all + (ulong)col * a.stride;
    device float *p = v + h * a.head_dim;
    int i0 = a.neox ? j : 2 * j;
    int i1 = a.neox ? j + a.half_dim : i0 + 1;
    float x0 = p[i0], x1 = p[i1];
    p[i0] = x0 * c - x1 * s;
    p[i1] = x0 * s + x1 * c;
}

// ---------------------------------------------------------------- kv store
// The KV cache is either fp16 (2 bytes/value) or q8_0 (32 values per 34-byte
// block: one fp16 scale + 32 int8 quants). Offsets are byte offsets so the CPU
// and Metal paths share the same cache layout.

static inline ulong kv_row_bytes(int kv_dim, int q8) {
    return q8 ? (ulong)(kv_dim / 32) * 34 : (ulong)kv_dim * 2;
}

static inline ulong kv_head_off(int kvh, int hd, int q8) {
    return q8 ? (ulong)(kvh * hd / 32) * 34 : (ulong)(kvh * hd) * 2;
}

static inline void kv_store_row(device uchar *cache, device const float *src,
                                int q8, uint i) {
    if (q8) {
        device uchar *blk = cache + (ulong)i * 34;
        device half *dptr = (device half *)blk;
        device char *q = (device char *)(blk + 2);
        device const float *x = src + i * 32;
        float amax = 0;
        for (int j = 0; j < 32; j++) amax = max(amax, fabs(x[j]));
        float d = amax / 127.0f;
        float id = d > 0 ? 1.0f / d : 0.0f;
        *dptr = (half)d;
        for (int j = 0; j < 32; j++) q[j] = (char)round(x[j] * id);
    } else {
        ((device half *)cache)[i] = (half)src[i];
    }
}

static inline float kv_dot(device const uchar *row, device const float *qh,
                           int hd, int q8) {
    float s = 0;
    if (q8) {
        for (int b = 0; b < hd / 32; b++) {
            device const uchar *blk = row + (ulong)b * 34;
            float d = (float)*(device const half *)blk;
            device const char *q = (device const char *)(blk + 2);
            device const float *xp = qh + b * 32;
            float t = 0;
            for (int j = 0; j < 32; j += 2)
                t += xp[j] * (float)q[j] + xp[j + 1] * (float)q[j + 1];
            s += d * t;
        }
    } else {
        device const half *k = (device const half *)row;
        for (int i = 0; i < hd; i++) s += qh[i] * (float)k[i];
    }
    return s;
}

static inline float2 kv_pair(device const uchar *row, int i2, int q8) {
    if (q8) {
        device const uchar *blk = row + (ulong)(i2 / 16) * 34;
        float d = (float)*(device const half *)blk;
        device const char *q = (device const char *)(blk + 2);
        int j = (2 * i2) & 31;
        return float2(d * (float)q[j], d * (float)q[j + 1]);
    }
    device const half *h = (device const half *)row + 2 * i2;
    return float2((float)h[0], (float)h[1]);
}

// off is the FIRST column's byte offset; column c lands row_b bytes further on,
// so a prompt batch stores every token's K/V in one dispatch (grid.y = column).
struct store_args { int kv_dim, q8, stride; ulong off, row_b; };

kernel void k_store_kv(device const float *k_all [[buffer(0)]],
                       device const float *v_all [[buffer(1)]],
                       device uchar       *kc [[buffer(2)]],
                       device uchar       *vc [[buffer(3)]],
                       constant store_args &a [[buffer(4)]],
                       uint2 gid [[thread_position_in_grid]]) {
    int n = a.q8 ? a.kv_dim / 32 : a.kv_dim;
    uint i = gid.x, col = gid.y;
    if ((int)i < n) {
        ulong off = a.off + (ulong)col * a.row_b;
        kv_store_row(kc + off, k_all + (ulong)col * a.stride, a.q8, i);
        kv_store_row(vc + off, v_all + (ulong)col * a.stride, a.q8, i);
    }
}

// ---------------------------------------------------------------- attention
// One threadgroup per head: scores -> softmax -> weighted value sum.

// pos is the FIRST column's position; column c attends over [.., pos + c], so
// a prompt batch runs every token's attention in one dispatch (grid.y = col).
// Each column keeps its own score row and output slice, so the arithmetic per
// (head, position) is exactly what a per-token dispatch computed.
struct attn_args {
    int   head_dim, n_head, n_head_kv, n_ctx, pos;
    ulong l_off;      // this layer's byte offset into the kv cache
    float scale;
    int   q8;
    int   window;     // sliding-window size for this layer (0 = full)
    int   has_sinks;  // gpt-oss: per-head sink joins softmax denominator only
    int   q_stride, att_stride, out_stride;
};

// Cooperative-read variant of the K-score loop (RUNNER_METAL_ATTN_COOP=1).
//
// The 2026-08-15 diagnosis measured decode's KV read at 1.52 GB/s, ~18x worse
// per byte than the weight read in the same forward, and named the cause:
// k_attn gives each THREAD a whole KV row, so adjacent lanes address memory
// `row_b` apart -- 1024 B on e2b -- and one simdgroup load touches 32 distinct
// cache lines. (The V accumulation below is already coalesced: consecutive
// threads take consecutive elements of the SAME row, so only the score loop
// scatters, and only that loop changes here.)
//
// Cooperative form: one SIMDGROUP owns a row and its 32 lanes split the head
// dimension, so a load covers 32 consecutive elements instead of 32 rows, and
// the per-row dot finishes with one simd_sum.
//
// That reduction is why this is a separate kernel rather than an edit: summing
// 32 lane partials through a reduction tree is not the sequential accumulation
// k_attn performs, so the result is NOT bit-identical and the identity route
// cannot host it. It answers to tests/test_attn_tol.c the way the fast matvec
// answers to test_mv_tol.c, and ships off until a measurement promotes it.
static inline float kv_dot_coop(device const uchar *row, device const float *qh,
                                int hd, int q8, uint lane) {
    float s = 0;
    if (q8) {
        // One 32-element block per lane step: the block's scale is read once
        // and the 32 quants beneath it are contiguous.
        for (int b = (int)lane; b < hd / 32; b += 32) {
            device const uchar *blk = row + (ulong)b * 34;
            float d = (float)*(device const half *)blk;
            device const char *q = (device const char *)(blk + 2);
            device const float *xp = qh + b * 32;
            float t = 0;
            for (int j = 0; j < 32; j++) t += xp[j] * (float)q[j];
            s += d * t;
        }
    } else {
        device const half *k = (device const half *)row;
        for (int i = (int)lane; i < hd; i += 32) s += qh[i] * (float)k[i];
    }
    return simd_sum(s);
}

// Shared prologue and epilogue for the two attention kernels below. They
// differ ONLY in how the K scores are computed; keeping the rest in one place
// is what stops the gated variant from drifting from the route it is measured
// against.
#define ATTN_PROLOGUE \
    /* Decode gives this kernel only n_head threadgroups (8 on gemma-3-4b) on */ \
    /* an 8-core GPU, and its work grows linearly with context, so */ \
    /* threads-per-group is the only parallelism lever short of splitting the */ \
    /* KV range across threadgroups. 128 -> 256 is worth ~11% at 3k context; */ \
    /* 512 and 1024 measured IDENTICAL to 256 on an M1 (the pipeline allows */ \
    /* 1024), because per-thread work shrinks as fast as the reduction's */ \
    /* barrier count grows. 256 is therefore the measured size, and the extra */ \
    /* threadgroup scratch for 1024 buys nothing. The host still clamps to the */ \
    /* pipeline limit and to a power of two, which these reductions require. */ \
    threadgroup float red[256]; \
    uint tid = tid3.x, tpg = tpg3.x; \
    uint h = tgpig.x, col = tgpig.y; \
    int hd = a.head_dim; \
    int kvh = h / (a.n_head / a.n_head_kv); \
    int kv_dim = a.n_head_kv * hd; \
    ulong row_b = kv_row_bytes(kv_dim, a.q8); \
    ulong base = a.l_off + kv_head_off(kvh, hd, a.q8); \
    int pos = a.pos + (int)col; \
    device const float *q = q_all + (ulong)col * a.q_stride; \
    device float *att = att_all + (ulong)col * a.att_stride; \
    device float *out = out_all + (ulong)col * a.out_stride; \
    device const float *qh = q + h * hd; \
    device float *ah = att + (ulong)h * a.n_ctx; \
    int t0 = 0; \
    if (a.window > 0 && pos - a.window + 1 > 0) t0 = pos - a.window + 1; \


#define ATTN_EPILOGUE \
 \
    /* max */ \
    float mx = -1e30f; \
    for (int t = t0 + tid; t <= pos; t += tpg) mx = max(mx, ah[t]); \
    if (a.has_sinks && tid == 0) mx = max(mx, sinks[h]); \
    red[tid] = mx; \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    for (uint off = tpg / 2; off > 0; off >>= 1) { \
        if (tid < off) red[tid] = max(red[tid], red[tid + off]); \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
    } \
    mx = red[0]; \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    /* exp + sum */ \
    float sum = 0; \
    for (int t = t0 + tid; t <= pos; t += tpg) { \
        float e = exp(ah[t] - mx); \
        ah[t] = e; \
        sum += e; \
    } \
    if (a.has_sinks && tid == 0) sum += exp(sinks[h] - mx); \
    red[tid] = sum; \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    for (uint off = tpg / 2; off > 0; off >>= 1) { \
        if (tid < off) red[tid] += red[tid + off]; \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
    } \
    sum = red[0]; \
    threadgroup_barrier(mem_flags::mem_device); \
 \
    for (int i = tid; i < hd; i += tpg) { \
        float o = 0; \
        for (int t = t0; t <= pos; t++) \
            o += ah[t] * kv_pair(vc + base + (ulong)t * row_b, i / 2, a.q8)[i & 1]; \
        out[h * hd + i] = o / sum; \
    }

kernel void k_attn(device const float *q_all   [[buffer(0)]],
                   device const uchar *kc      [[buffer(1)]],
                   device const uchar *vc      [[buffer(2)]],
                   device float       *att_all [[buffer(3)]],
                   device float       *out_all [[buffer(4)]],
                   constant attn_args &a   [[buffer(5)]],
                   device const float *sinks [[buffer(6)]],
                   uint3 tgpig [[threadgroup_position_in_grid]],
                   uint3 tid3 [[thread_position_in_threadgroup]],
                   uint3 tpg3 [[threads_per_threadgroup]]) {
    ATTN_PROLOGUE
    for (int t = t0 + tid; t <= pos; t += tpg) {
        ah[t] = kv_dot(kc + base + (ulong)t * row_b, qh, hd, a.q8) * a.scale;
    }
    threadgroup_barrier(mem_flags::mem_device);
    ATTN_EPILOGUE
}

// Cooperative-read twin: one simdgroup owns a KV row and its lanes split
// head_dim, so a load covers 32 consecutive elements instead of 32 rows.
// Everything after the score loop is the shared epilogue.
kernel void k_attn_coop(device const float *q_all   [[buffer(0)]],
                   device const uchar *kc      [[buffer(1)]],
                   device const uchar *vc      [[buffer(2)]],
                   device float       *att_all [[buffer(3)]],
                   device float       *out_all [[buffer(4)]],
                   constant attn_args &a   [[buffer(5)]],
                   device const float *sinks [[buffer(6)]],
                   uint3 tgpig [[threadgroup_position_in_grid]],
                   uint3 tid3 [[thread_position_in_threadgroup]],
                   uint3 tpg3 [[threads_per_threadgroup]]) {
    ATTN_PROLOGUE
    {
        uint lane = tid & 31u, sg = tid >> 5, n_sg = tpg >> 5;
        for (int t = t0 + (int)sg; t <= pos; t += (int)n_sg) {
            float sc = kv_dot_coop(kc + base + (ulong)t * row_b, qh, hd,
                                   a.q8, lane) * a.scale;
            if (lane == 0) ah[t] = sc;
        }
    }
    threadgroup_barrier(mem_flags::mem_device);
    ATTN_EPILOGUE
}

// ------------------------------------------------- chunked decode attention
//
// k_attn gets n_head threadgroups and nothing else, so decode attention runs
// 8 threadgroups on an 8-core GPU while its work grows linearly with context.
// Widening threads-per-group bought ~11% at 3k and then saturated; the ceiling
// is threadgroup COUNT. These two kernels split the position range instead, so
// the grid becomes n_head * n_chunks.
//
// Standard online-softmax decomposition: each chunk reduces its own slice to
// (max, sum, unnormalised weighted V), and the combine pass rescales every
// chunk onto a common maximum. exp(s-m) never sees a positive argument, so the
// split cannot overflow where the single-pass kernel would not.
//
// The scores still land in the shared att buffer: chunks own disjoint position
// ranges, so they cannot collide.

struct attn_chunk_args {
    int   head_dim, n_head, n_head_kv, n_ctx, pos;
    ulong l_off;
    float scale;
    int   q8;
    int   window;
    int   chunk;       // positions per chunk
    int   n_chunks;
    int   q_stride, att_stride;
};

/* Shared halves of the chunked attention kernel; the two variants below
   differ only in how the K scores are read. */
#define ATTNC_PROLOGUE \
    threadgroup float red[256]; \
    uint tid = tid3.x, tpg = tpg3.x; \
    uint h = tgpig.x, z = tgpig.y; \
    int hd = a.head_dim; \
    int kvh = h / (a.n_head / a.n_head_kv); \
    int kv_dim = a.n_head_kv * hd; \
    ulong row_b = kv_row_bytes(kv_dim, a.q8); \
    ulong base = a.l_off + kv_head_off(kvh, hd, a.q8); \
    int pos = a.pos; \
    device const float *qh = q_all + h * hd; \
    device float *ah = att_all + (ulong)h * a.n_ctx; \
 \
    int t0 = 0; \
    if (a.window > 0 && pos - a.window + 1 > 0) t0 = pos - a.window + 1; \
    int lo = t0 + (int)z * a.chunk; \
    int hi = min(lo + a.chunk - 1, pos); \
 \
    device float *acc = acc_all + ((ulong)h * a.n_chunks + z) * hd; \
    device float *ms  = ms_all  + ((ulong)h * a.n_chunks + z) * 2; \
 \
    if (lo > hi) { /* empty chunk: neutral partial */ \
        for (int i = tid; i < hd; i += tpg) acc[i] = 0; \
        if (tid == 0) { ms[0] = -1e30f; ms[1] = 0; } \
        return; \
    } \


#define ATTNC_EPILOGUE \
 \
    float mx = -1e30f; \
    for (int t = lo + (int)tid; t <= hi; t += tpg) mx = max(mx, ah[t]); \
    red[tid] = mx; \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    for (uint off = tpg / 2; off > 0; off >>= 1) { \
        if (tid < off) red[tid] = max(red[tid], red[tid + off]); \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
    } \
    mx = red[0]; \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
 \
    float sum = 0; \
    for (int t = lo + (int)tid; t <= hi; t += tpg) { \
        float e = exp(ah[t] - mx); \
        ah[t] = e; \
        sum += e; \
    } \
    red[tid] = sum; \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    for (uint off = tpg / 2; off > 0; off >>= 1) { \
        if (tid < off) red[tid] += red[tid + off]; \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
    } \
    sum = red[0]; \
    threadgroup_barrier(mem_flags::mem_device); \
 \
    /* UNNORMALISED: the combine pass owns the division, because the divisor is */ \
    /* not known until every chunk's maximum is. */ \
    for (int i = tid; i < hd; i += tpg) { \
        float o = 0; \
        for (int t = lo; t <= hi; t++) \
            o += ah[t] * kv_pair(vc + base + (ulong)t * row_b, i / 2, a.q8)[i & 1]; \
        acc[i] = o; \
    } \
    if (tid == 0) { ms[0] = mx; ms[1] = sum; }

kernel void k_attn_chunk(device const float *q_all   [[buffer(0)]],
                         device const uchar *kc      [[buffer(1)]],
                         device const uchar *vc      [[buffer(2)]],
                         device float       *att_all [[buffer(3)]],
                         device float       *acc_all [[buffer(4)]],
                         device float       *ms_all  [[buffer(5)]],
                         constant attn_chunk_args &a [[buffer(6)]],
                         uint3 tgpig [[threadgroup_position_in_grid]],
                         uint3 tid3  [[thread_position_in_threadgroup]],
                         uint3 tpg3  [[threads_per_threadgroup]]) {
    ATTNC_PROLOGUE
    for (int t = lo + (int)tid; t <= hi; t += tpg)
        ah[t] = kv_dot(kc + base + (ulong)t * row_b, qh, hd, a.q8) * a.scale;
    threadgroup_barrier(mem_flags::mem_device);
    ATTNC_EPILOGUE
}

// Cooperative-read twin of the chunked kernel. This is the one that matters
// at long context: decode takes the chunked path whenever the span is worth
// splitting, so k_attn_coop above only ever runs on short spans.
kernel void k_attn_chunk_coop(device const float *q_all   [[buffer(0)]],
                         device const uchar *kc      [[buffer(1)]],
                         device const uchar *vc      [[buffer(2)]],
                         device float       *att_all [[buffer(3)]],
                         device float       *acc_all [[buffer(4)]],
                         device float       *ms_all  [[buffer(5)]],
                         constant attn_chunk_args &a [[buffer(6)]],
                         uint3 tgpig [[threadgroup_position_in_grid]],
                         uint3 tid3  [[thread_position_in_threadgroup]],
                         uint3 tpg3  [[threads_per_threadgroup]]) {
    ATTNC_PROLOGUE
    {
        uint lane = tid & 31u, sg = tid >> 5, n_sg = tpg >> 5;
        for (int t = lo + (int)sg; t <= hi; t += (int)n_sg) {
            float sc = kv_dot_coop(kc + base + (ulong)t * row_b, qh, hd,
                                   a.q8, lane) * a.scale;
            if (lane == 0) ah[t] = sc;
        }
    }
    threadgroup_barrier(mem_flags::mem_device);
    ATTNC_EPILOGUE
}

struct attn_comb_args { int head_dim, n_head, n_chunks, has_sinks, out_stride; };

kernel void k_attn_combine(device const float *acc_all [[buffer(0)]],
                           device const float *ms_all  [[buffer(1)]],
                           device float       *out_all [[buffer(2)]],
                           constant attn_comb_args &a  [[buffer(3)]],
                           device const float *sinks   [[buffer(4)]],
                           uint3 tgpig [[threadgroup_position_in_grid]],
                           uint3 tid3  [[thread_position_in_threadgroup]],
                           uint3 tpg3  [[threads_per_threadgroup]]) {
    uint tid = tid3.x, tpg = tpg3.x;
    uint h = tgpig.x;
    int hd = a.head_dim;
    device const float *ms = ms_all + (ulong)h * a.n_chunks * 2;

    float M = -1e30f;
    for (int z = 0; z < a.n_chunks; z++) M = max(M, ms[z * 2]);
    if (a.has_sinks) M = max(M, sinks[h]);

    float denom = 0;
    for (int z = 0; z < a.n_chunks; z++) denom += ms[z * 2 + 1] * exp(ms[z * 2] - M);
    // A sink joins the denominator only -- it contributes no value vector --
    // exactly as the single-pass kernel treats it.
    if (a.has_sinks) denom += exp(sinks[h] - M);

    for (int i = tid; i < hd; i += tpg) {
        float o = 0;
        for (int z = 0; z < a.n_chunks; z++)
            o += acc_all[((ulong)h * a.n_chunks + z) * hd + i] * exp(ms[z * 2] - M);
        out_all[h * hd + i] = o / denom;
    }
}

// ---------------------------------------------------------------- elementwise

// The elementwise family takes the prompt batch in grid.y: thread (i, col)
// touches element i of column col, at gs/us elements per column. Each element
// was always independent, so this is BIT-IDENTICAL -- only the encoding
// changes. It exists because these ops were 175n of the 240n per-token
// dispatches measured in docs/metal-dispatch-census-2026-08-13.md, and
// batching the 65n of qknorm/headnorm was already worth +2.1% prefill.
// At n_col == 1 the strides are never read.
kernel void k_silu_mul(device float       *g [[buffer(0)]],
                       device const float *u [[buffer(1)]],
                       constant int       &n [[buffer(2)]],
                       constant int       &gs [[buffer(3)]],
                       constant int       &us [[buffer(4)]],
                       uint2 gid [[thread_position_in_grid]]) {
    if ((int)gid.x < n) {
        device float *gp = g + (ulong)gid.y * gs;
        device const float *up = u + (ulong)gid.y * us;
        float x = gp[gid.x];
        gp[gid.x] = (x / (1.0f + exp(-x))) * up[gid.x];
    }
}

// attention output gate (afmoe / muse-glimmer): x *= sigmoid(gate), the
// same arithmetic as the CPU's 1/(1+expf(-g)) and CUDA's k_q35_attn_gate.
// exp() overflow on a large negative gate saturates to 0 exactly as libm's
// expf does, so no clamp is needed for identity with the CPU oracle.
kernel void k_sigmoid_mul(device float       *x [[buffer(0)]],
                          device const float *g [[buffer(1)]],
                          constant int       &n [[buffer(2)]],
                          constant int       &xs [[buffer(3)]],
                          constant int       &gs [[buffer(4)]],
                          uint2 gid [[thread_position_in_grid]]) {
    if ((int)gid.x < n) {
        device float *xp = x + (ulong)gid.y * xs;
        device const float *gp = g + (ulong)gid.y * gs;
        xp[gid.x] *= 1.0f / (1.0f + exp(-gp[gid.x]));
    }
}

kernel void k_gelu_mul(device float       *g [[buffer(0)]],
                       device const float *u [[buffer(1)]],
                       constant int       &n [[buffer(2)]],
                       constant int       &gs [[buffer(3)]],
                       constant int       &us [[buffer(4)]],
                       uint2 gid [[thread_position_in_grid]]) {
    if ((int)gid.x < n) {
        device float *gp = g + (ulong)gid.y * gs;
        device const float *up = u + (ulong)gid.y * us;
        float x = gp[gid.x];
        // Metal compiles with fast math, where tanh() is evaluated through
        // exp(2a): for large |a| that overflows to inf and inf/inf yields NaN,
        // while the CPU oracle's libm tanhf saturates. Gemma-class models
        // reach it — gemma-3-4b's layer-0 gate produced NaN logits here, and
        // the model emitted only token 0. Clamping to a magnitude where tanh
        // is already exactly +/-1.0f in fp32 cannot change any representable
        // result, so this guard is invisible to the identity gates. Same
        // hazard, same fix as the `g < -80` early-out in the CPU silu path.
        float a = 0.7978845608f * (x + 0.044715f * x * x * x);
        float t = tanh(clamp(a, -16.0f, 16.0f));
        gp[gid.x] = 0.5f * x * (1.0f + t) * up[gid.x];
    }
}

kernel void k_add(device float       *x [[buffer(0)]],
                  device const float *d [[buffer(1)]],
                  constant int       &n [[buffer(2)]],
                  constant int       &xs [[buffer(3)]],
                  constant int       &ds [[buffer(4)]],
                  uint2 gid [[thread_position_in_grid]]) {
    if ((int)gid.x < n)
        x[(ulong)gid.y * xs + gid.x] += d[(ulong)gid.y * ds + gid.x];
}

kernel void k_scale(device float       *x [[buffer(0)]],
                    constant float     &s [[buffer(1)]],
                    constant int       &n [[buffer(2)]],
                    constant int       &xs [[buffer(3)]],
                    uint2 gid [[thread_position_in_grid]]) {
    if ((int)gid.x < n) x[(ulong)gid.y * xs + gid.x] *= s;
}

kernel void k_head_transform(device float     *logits [[buffer(0)]],
                             device const int *suppress [[buffer(1)]],
                             constant int     &nv [[buffer(2)]],
                             constant float   &softcap [[buffer(3)]],
                             constant int     &ns [[buffer(4)]],
                             uint i [[thread_position_in_grid]]) {
    if ((int)i < nv && softcap > 0.0f)
        logits[i] = softcap * tanh(logits[i] / softcap);
    if ((int)i < ns) {
        int tok = suppress[i];
        if (tok >= 0 && tok < nv) logits[tok] = -1e30f;
    }
}

// -------------------------------------------------------------- sparse MoE
// Plain router + fused-3D expert layout. This is the Metal twin of the first
// CUDA MoE slice: softmax -> top-k -> renormalize on device, then one indirect
// expert matvec launch per projection.

kernel void k_moe_route(device const float *logits [[buffer(0)]],
                        device int         *sel    [[buffer(1)]],
                        device float       *selw   [[buffer(2)]],
                        constant int       &ne     [[buffer(3)]],
                        constant int       &used   [[buffer(4)]],
                        constant int       &tokens [[buffer(5)]],
                        constant int       &ls     [[buffer(6)]],
                        uint t [[thread_position_in_grid]]) {
    if ((int)t >= tokens) return;
    float lg[256];
    device const float *src = logits + (ulong)t * ls;
    for (int e = 0; e < ne; e++) lg[e] = src[e];

    float mx = lg[0];
    for (int e = 1; e < ne; e++)
        if (lg[e] > mx) mx = lg[e];
    float sum = 0.0f;
    for (int e = 0; e < ne; e++) {
        float p = exp(lg[e] - mx);
        lg[e] = p;
        sum += p;
    }
    for (int e = 0; e < ne; e++) lg[e] /= sum;

    device int *ts = sel + (ulong)t * used;
    device float *tw = selw + (ulong)t * used;
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
    if (denom < 6.103515625e-5f) denom = 6.103515625e-5f;
    for (int s = 0; s < used; s++) tw[s] /= denom;
}

struct moe_args {
    int   n_in;
    int   n_out;
    ulong w_off;
    ulong estride;
    int   xs;
    int   ys;
    int   has_bias;
    int   bias_stride;
};

#define MOE_MV_HEAD \
    uint row = tgpig.x * (ntg.x / 32) + sgitg; \
    if (row >= (uint)a.n_out) return; \
    device const uchar *wbase = wb + a.w_off + (ulong)sel[tgpig.y] * a.estride; \
    device const float *xp0 = x + (ulong)tgpig.y * a.xs; \
    float s = 0;

#define MOE_MV_TAIL \
    s = simd_sum(s); \
    if (tiisg == 0) { \
        if (a.has_bias) s += bias[(ulong)sel[tgpig.y] * a.bias_stride + row]; \
        y[(ulong)tgpig.y * a.ys + row] = s; \
    }

#define MOE_MV_PARAMS \
    device const uchar *wb  [[buffer(0)]], \
    device const float *x   [[buffer(1)]], \
    device float       *y   [[buffer(2)]], \
    constant moe_args  &a   [[buffer(3)]], \
    device const int   *sel [[buffer(4)]], \
    device const float *bias [[buffer(5)]], \
    uint  sgitg [[simdgroup_index_in_threadgroup]], \
    uint  tiisg [[thread_index_in_simdgroup]], \
    uint3 tgpig [[threadgroup_position_in_grid]], \
    uint3 ntg   [[threads_per_threadgroup]]

kernel void k_moe_mv_f32(MOE_MV_PARAMS) {
    MOE_MV_HEAD;
    device const float *rw = (device const float *)wbase + (ulong)row * a.n_in;
    for (int i = tiisg; i < a.n_in; i += 32) s += rw[i] * xp0[i];
    MOE_MV_TAIL;
}

kernel void k_moe_mv_f16(MOE_MV_PARAMS) {
    MOE_MV_HEAD;
    device const half *rw = (device const half *)wbase + (ulong)row * a.n_in;
    for (int i = tiisg; i < a.n_in; i += 32) s += (float)rw[i] * xp0[i];
    MOE_MV_TAIL;
}

kernel void k_moe_mv_q8_0(MOE_MV_PARAMS) {
    MOE_MV_HEAD;
    int nb = a.n_in / 32;
    device const uchar *rw = wbase + (ulong)row * nb * 34;
    for (int b = tiisg; b < nb; b += 32) {
        device const uchar *blk = rw + (ulong)b * 34;
        float d = (float)*(device const half *)blk;
        device const char *q = (device const char *)(blk + 2);
        device const float *xp = xp0 + b * 32;
        float t = 0;
        for (int j = 0; j < 32; j++) t += (float)q[j] * xp[j];
        s += d * t;
    }
    MOE_MV_TAIL;
}

kernel void k_moe_mv_q4_0(MOE_MV_PARAMS) {
    MOE_MV_HEAD;
    int nb = a.n_in / 32;
    device const uchar *rw = wbase + (ulong)row * nb * 18;
    for (int b = tiisg; b < nb; b += 32) {
        device const uchar *blk = rw + (ulong)b * 18;
        float d = (float)*(device const half *)blk;
        device const uchar *q = blk + 2;
        device const float *xp = xp0 + b * 32;
        float t = 0;
        for (int j = 0; j < 16; j++)
            t += ((int)(q[j] & 0xF) - 8) * xp[j] + ((int)(q[j] >> 4) - 8) * xp[j + 16];
        s += d * t;
    }
    MOE_MV_TAIL;
}

// The MoE kernels are twins of the dense k_mv_* family and carried the same
// occupancy defect: a whole 256-weight superblock per lane means nb =
// n_in/256 lanes work, which is 11 of 32 for a 2816-wide expert. Same fix,
// same units -- quarters for q4_K/q5_K, halves for q6_K -- chosen so no
// weight byte is read twice. The dense side measured 5.43 -> 9.22 tok/s from
// this plus vectorisation.
kernel void k_moe_mv_q4_K(MOE_MV_PARAMS) {
    MOE_MV_HEAD;
    int nb = a.n_in / 256;
    device const uchar *rw = wbase + (ulong)row * nb * 144;
    int nq = nb * 4;
    for (int u = tiisg; u < nq; u += 32) {
        int b = u >> 2, jj = u & 3;
        device const uchar *blk = rw + (ulong)b * 144;
        float d    = (float)*(device const half *)blk;
        float dmin = (float)*(device const half *)(blk + 2);
        device const uchar *sc = blk + 4;
        device const uchar *q  = blk + 16 + jj * 32;
        device const float *xp = xp0 + b * 256 + jj * 64;
        uchar s1, m1, s2, m2;
        get_scale_min_k4(jj * 2 + 0, sc, &s1, &m1);
        get_scale_min_k4(jj * 2 + 1, sc, &s2, &m2);
        float d1 = d * s1, mm1 = dmin * m1;
        float d2 = d * s2, mm2 = dmin * m2;
        float t1 = 0, t2 = 0, sx1 = 0, sx2 = 0;
        for (int l = 0; l < 32; l++) {
            t1 += (float)(q[l] & 0xF) * xp[l];      sx1 += xp[l];
            t2 += (float)(q[l] >> 4)  * xp[l + 32]; sx2 += xp[l + 32];
        }
        s += d1 * t1 - mm1 * sx1 + d2 * t2 - mm2 * sx2;
    }
    MOE_MV_TAIL;
}

kernel void k_moe_mv_q5_K(MOE_MV_PARAMS) {
    MOE_MV_HEAD;
    int nb = a.n_in / 256;
    device const uchar *rw = wbase + (ulong)row * nb * 176;
    int nq = nb * 4;
    for (int u = tiisg; u < nq; u += 32) {
        int b = u >> 2, jj = u & 3;
        device const uchar *blk = rw + (ulong)b * 176;
        float d    = (float)*(device const half *)blk;
        float dmin = (float)*(device const half *)(blk + 2);
        device const uchar *sc = blk + 4;
        device const uchar *qh = blk + 16;   // shared by all four quarters
        device const uchar *q  = blk + 48 + jj * 32;
        device const float *xp = xp0 + b * 256 + jj * 64;
        uchar s1, m1, s2, m2;
        get_scale_min_k4(jj * 2 + 0, sc, &s1, &m1);
        get_scale_min_k4(jj * 2 + 1, sc, &s2, &m2);
        float d1 = d * s1, mm1 = dmin * m1;
        float d2 = d * s2, mm2 = dmin * m2;
        uchar u1 = (uchar)(1u << (2 * jj)), u2 = (uchar)(2u << (2 * jj));
        for (int l = 0; l < 32; l++) {
            s += (d1 * (float)((q[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - mm1) * xp[l];
            s += (d2 * (float)((q[l] >> 4)  + ((qh[l] & u2) ? 16 : 0)) - mm2) * xp[l + 32];
        }
    }
    MOE_MV_TAIL;
}

kernel void k_moe_mv_q6_K(MOE_MV_PARAMS) {
    MOE_MV_HEAD;
    int nb = a.n_in / 256;
    device const uchar *rw = wbase + (ulong)row * nb * 210;
    int nh = nb * 2;
    for (int u = tiisg; u < nh; u += 32) {
        int b = u >> 1, half_i = u & 1;
        device const uchar *blk = rw + (ulong)b * 210;
        device const uchar *ql = blk + half_i * 64;
        device const uchar *qh = blk + 128 + half_i * 32;
        device const char  *sc = (device const char *)(blk + 192) + half_i * 8;
        float d = (float)*(device const half *)(blk + 208);
        device const float *xp = xp0 + b * 256 + half_i * 128;
        float t[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        // Vectorised exactly as the dense twin, which this shape is worth the
        // most on: four outputs per iteration assembled from ql and qh with
        // per-element shifts is more than the compiler folds. packed_uchar4
        // because a 210-byte superblock leaves ql/qh at odd alignments.
        for (int is = 0; is < 2; is++) {
            device const packed_uchar4 *l4  = (device const packed_uchar4 *)(ql + is * 16);
            device const packed_uchar4 *l4b = (device const packed_uchar4 *)(ql + 32 + is * 16);
            device const packed_uchar4 *h4  = (device const packed_uchar4 *)(qh + is * 16);
            device const packed_float4 *x0 = (device const packed_float4 *)(xp + is * 16);
            device const packed_float4 *x1 = (device const packed_float4 *)(xp + 32 + is * 16);
            device const packed_float4 *x2 = (device const packed_float4 *)(xp + 64 + is * 16);
            device const packed_float4 *x3 = (device const packed_float4 *)(xp + 96 + is * 16);
            float4 a0 = 0, a1 = 0, a2 = 0, a3 = 0;
            for (int k = 0; k < 4; k++) {
                uchar4 lo = l4[k], hi = l4b[k], h = h4[k];
                a0 += (float4(lo & 0xF) + float4((h >> 0) & 3) * 16.0f - 32.0f) * x0[k];
                a1 += (float4(hi & 0xF) + float4((h >> 2) & 3) * 16.0f - 32.0f) * x1[k];
                a2 += (float4(lo >> 4)  + float4((h >> 4) & 3) * 16.0f - 32.0f) * x2[k];
                a3 += (float4(hi >> 4)  + float4((h >> 6) & 3) * 16.0f - 32.0f) * x3[k];
            }
            t[is * 4 + 0] = a0.x + a0.y + a0.z + a0.w;
            t[is * 4 + 1] = a1.x + a1.y + a1.z + a1.w;
            t[is * 4 + 2] = a2.x + a2.y + a2.z + a2.w;
            t[is * 4 + 3] = a3.x + a3.y + a3.z + a3.w;
        }
        s += d * (sc[0] * t[0] + sc[2] * t[1] + sc[4] * t[2] + sc[6] * t[3] +
                  sc[1] * t[4] + sc[3] * t[5] + sc[5] * t[6] + sc[7] * t[7]);
    }
    MOE_MV_TAIL;
}

kernel void k_moe_mv_mxfp4(MOE_MV_PARAMS) {
    MOE_MV_HEAD;
    int nb = a.n_in / 32;
    device const uchar *rw = wbase + (ulong)row * nb * 17;
    for (int b = tiisg; b < nb; b += 32) {
        device const uchar *blk = rw + (ulong)b * 17;
        float d = ldexp(1.0f, (int)blk[0] - 127);
        device const uchar *q = blk + 1;
        device const float *xp = xp0 + b * 32;
        float t = 0;
        for (int j = 0; j < 16; j++) {
            t += kv_mxfp4[q[j] & 0xF] * xp[j];
            t += kv_mxfp4[q[j] >> 4]  * xp[j + 16];
        }
        s += d * t;
    }
    MOE_MV_TAIL;
}

static inline float swiglu_oai(float g, float u) {
    const float alpha = 1.702f, limit = 7.0f;
    float x = g < limit ? g : limit;
    float y = clamp(u, -limit, limit);
    float gl = x < -50.0f ? 0.0f : x / (1.0f + exp(alpha * -x));
    return gl * (y + 1.0f);
}

kernel void k_moe_actmul(device float       *gbuf [[buffer(0)]],
                         device const float *ubuf [[buffer(1)]],
                         constant int4      &args [[buffer(2)]],
                         uint2 gid [[thread_position_in_grid]]) {
    int nff = args.x, gss = args.y, uss = args.z, act = args.w;
    int i = gid.x, slot = gid.y;
    if (i >= nff) return;
    device float *g = gbuf + (ulong)slot * gss;
    device const float *u = ubuf + (ulong)slot * uss;
    float x = g[i];
    if (act == 2) {
        g[i] = swiglu_oai(x, u[i]);
    } else if (act == 1) {
        float t = tanh(0.7978845608f * (x + 0.044715f * x * x * x));
        g[i] = 0.5f * x * (1.0f + t) * u[i];
    } else {
        g[i] = (x / (1.0f + exp(-x))) * u[i];
    }
}

kernel void k_moe_sum(device float       *out    [[buffer(0)]],
                      device const float *eout   [[buffer(1)]],
                      device const float *selw   [[buffer(2)]],
                      device const float *dscale [[buffer(3)]],
                      device const int   *sel    [[buffer(4)]],
                      constant int       &n      [[buffer(5)]],
                      constant int       &nslots [[buffer(6)]],
                      constant int       &es     [[buffer(7)]],
                      constant int       &has_dscale [[buffer(8)]],
                      uint i [[thread_position_in_grid]]) {
    if ((int)i >= n) return;
    float s = 0.0f;
    for (int slot = 0; slot < nslots; slot++) {
        float w = selw[slot] * (has_dscale ? dscale[sel[slot]] : 1.0f);
        s += w * eout[(ulong)slot * es + i];
    }
    out[i] = s;
}
