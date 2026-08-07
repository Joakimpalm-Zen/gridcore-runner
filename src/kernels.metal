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
kernel void k_qknorm(device float       *v   [[buffer(0)]],
                     device const float *w   [[buffer(1)]],
                     constant int       &hd  [[buffer(2)]],
                     constant float     &eps [[buffer(3)]],
                     uint h   [[threadgroup_position_in_grid]],
                     uint tid [[thread_position_in_threadgroup]],
                     uint tpg [[threads_per_threadgroup]]) {
    threadgroup float red[128];
    device float *x = v + h * hd;
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
                           uint h   [[threadgroup_position_in_grid]],
                           uint tid [[thread_position_in_threadgroup]],
                           uint tpg [[threads_per_threadgroup]]) {
    threadgroup float red[128];
    device const float *x = src + h * hd;
    device float *y = dst + h * hd;
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

kernel void k_mv_f16(MV_PARAMS) {
    MV_HEAD;
    device const half *rw = (device const half *)(wb + a.w_off) + (ulong)row * a.n_in;
    float s = 0;
    for (int i = tiisg; i < a.n_in; i += 32) s += (float)rw[i] * x[i];
    MV_TAIL;
}

kernel void k_mv_q8_0(MV_PARAMS) {
    MV_HEAD;
    int nb = a.n_in / 32;
    device const uchar *rw = wb + a.w_off + (ulong)row * nb * 34;
    float s = 0;
    for (int b = tiisg; b < nb; b += 32) {
        device const uchar *blk = rw + (ulong)b * 34;
        float d = (float)*(device const half *)blk;
        device const char *q = (device const char *)(blk + 2);
        device const float *xp = x + b * 32;
        float t = 0;
        for (int j = 0; j < 32; j++) t += (float)q[j] * xp[j];
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
        device const uchar *q = blk + 2;
        device const float *xp = x + b * 32;
        float t = 0;
        for (int j = 0; j < 16; j++)
            t += ((int)(q[j] & 0xF) - 8) * xp[j] + ((int)(q[j] >> 4) - 8) * xp[j + 16];
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
        device const uchar *q  = blk + 16 + jj * 32;
        device const float *xp = x + b * 256 + jj * 64;
        uchar s1, m1, s2, m2;
        get_scale_min_k4(jj * 2 + 0, sc, &s1, &m1);
        get_scale_min_k4(jj * 2 + 1, sc, &s2, &m2);
        float d1 = d * s1, mm1 = dmin * m1;
        float d2 = d * s2, mm2 = dmin * m2;
        // Hand-vectorising this loop with uchar4/float4 was tried on
        // 2026-08-07 and is SLOWER: 6.58 -> 6.43 tok/s, consistent over three
        // runs. The scalar form lets the compiler pick its own vectorisation,
        // and the explicit version pays for uchar4->float4 conversions and
        // four horizontal sums that the scalar accumulators never need. Left
        // scalar deliberately; do not "optimise" it again without measuring.
        float t1 = 0, t2 = 0, sx1 = 0, sx2 = 0;
        for (int l = 0; l < 32; l++) {
            t1 += (float)(q[l] & 0xF) * xp[l];      sx1 += xp[l];
            t2 += (float)(q[l] >> 4)  * xp[l + 32]; sx2 += xp[l + 32];
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
        // are safe without re-deriving alignment for every offset.
        device const packed_uchar4 *q4  = (device const packed_uchar4 *)q;
        device const packed_uchar4 *qh4 = (device const packed_uchar4 *)qh;
        device const float4 *xlo = (device const float4 *)xp;
        device const float4 *xhi = (device const float4 *)(xp + 32);
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
            device const float4 *x0 = (device const float4 *)(xp + is * 16);
            device const float4 *x1 = (device const float4 *)(xp + 32 + is * 16);
            device const float4 *x2 = (device const float4 *)(xp + 64 + is * 16);
            device const float4 *x3 = (device const float4 *)(xp + 96 + is * 16);
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

#define MM_TM 32    // output rows per threadgroup
#define MM_TN 16    // output columns (prompt tokens) per threadgroup
#define MM_TK 32    // k-step: one q8_0/q4_0/mxfp4 block, and a q4_K sub-block

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
#define MM_BODY(...) \
    threadgroup float tg_w[MM_TM * MM_TK]; \
    threadgroup float tg_x[MM_TN * MM_TK]; \
    threadgroup float tg_c[MM_TN * MM_TM]; \
    const int row0 = (int)tgpig.x * MM_TM; \
    const int col0 = (int)tgpig.y * MM_TN; \
    const int tid  = (int)tpitg.x; \
    simdgroup_float8x8 acc0 = simdgroup_float8x8(0.0f); \
    simdgroup_float8x8 acc1 = simdgroup_float8x8(0.0f); \
    for (int k0 = 0; k0 < a.n_in; k0 += MM_TK) { \
        /* 128 threads x 8 values = the 32x32 weight tile */ \
        { \
            int r = tid >> 2, sub = (tid & 3) * 8; \
            int row = row0 + r; \
            threadgroup float *dst = tg_w + r * MM_TK + sub; \
            if (row < a.n_out) { __VA_ARGS__; } \
            else { for (int j = 0; j < 8; j++) dst[j] = 0.0f; } \
        } \
        /* 128 threads x 4 values = the 16x32 activation tile */ \
        for (int i = tid * 4; i < MM_TN * MM_TK; i += 128 * 4) { \
            for (int j = 0; j < 4; j++) { \
                int idx = i + j, cc = idx / MM_TK, kk = idx % MM_TK; \
                tg_x[idx] = (col0 + cc < a.n_col) \
                    ? x[(ulong)(col0 + cc) * a.x_stride + k0 + kk] : 0.0f; \
            } \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        for (int kk = 0; kk < MM_TK / 8; kk++) { \
            simdgroup_float8x8 A0, A1, B; \
            simdgroup_load(A0, tg_x + kk * 8, MM_TK); \
            simdgroup_load(A1, tg_x + 8 * MM_TK + kk * 8, MM_TK); \
            simdgroup_load(B, tg_w + (int)sgitg * 8 * MM_TK + kk * 8, MM_TK, \
                           ulong2(0, 0), true); \
            simdgroup_multiply_accumulate(acc0, A0, B, acc0); \
            simdgroup_multiply_accumulate(acc1, A1, B, acc1); \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
    } \
    simdgroup_store(acc0, tg_c + (int)sgitg * 8, MM_TM); \
    simdgroup_store(acc1, tg_c + 8 * MM_TM + (int)sgitg * 8, MM_TM); \
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
    // Decode gives this kernel only n_head threadgroups (8 on gemma-3-4b) on
    // an 8-core GPU, and its work grows linearly with context, so
    // threads-per-group is the only parallelism lever short of splitting the
    // KV range across threadgroups. 128 -> 256 is worth ~11% at 3k context;
    // 512 and 1024 measured IDENTICAL to 256 on an M1 (the pipeline allows
    // 1024), because per-thread work shrinks as fast as the reduction's
    // barrier count grows. 256 is therefore the measured size, and the extra
    // threadgroup scratch for 1024 buys nothing. The host still clamps to the
    // pipeline limit and to a power of two, which these reductions require.
    threadgroup float red[256];
    uint tid = tid3.x, tpg = tpg3.x;
    uint h = tgpig.x, col = tgpig.y;
    int hd = a.head_dim;
    int kvh = h / (a.n_head / a.n_head_kv);
    int kv_dim = a.n_head_kv * hd;
    ulong row_b = kv_row_bytes(kv_dim, a.q8);
    ulong base = a.l_off + kv_head_off(kvh, hd, a.q8);
    int pos = a.pos + (int)col;
    device const float *q = q_all + (ulong)col * a.q_stride;
    device float *att = att_all + (ulong)col * a.att_stride;
    device float *out = out_all + (ulong)col * a.out_stride;
    device const float *qh = q + h * hd;
    device float *ah = att + (ulong)h * a.n_ctx;
    int t0 = 0;
    if (a.window > 0 && pos - a.window + 1 > 0) t0 = pos - a.window + 1;

    for (int t = t0 + tid; t <= pos; t += tpg) {
        ah[t] = kv_dot(kc + base + (ulong)t * row_b, qh, hd, a.q8) * a.scale;
    }
    threadgroup_barrier(mem_flags::mem_device);

    // max
    float mx = -1e30f;
    for (int t = t0 + tid; t <= pos; t += tpg) mx = max(mx, ah[t]);
    if (a.has_sinks && tid == 0) mx = max(mx, sinks[h]);
    red[tid] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint off = tpg / 2; off > 0; off >>= 1) {
        if (tid < off) red[tid] = max(red[tid], red[tid + off]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    mx = red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // exp + sum
    float sum = 0;
    for (int t = t0 + tid; t <= pos; t += tpg) {
        float e = exp(ah[t] - mx);
        ah[t] = e;
        sum += e;
    }
    if (a.has_sinks && tid == 0) sum += exp(sinks[h] - mx);
    red[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint off = tpg / 2; off > 0; off >>= 1) {
        if (tid < off) red[tid] += red[tid + off];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    sum = red[0];
    threadgroup_barrier(mem_flags::mem_device);

    for (int i = tid; i < hd; i += tpg) {
        float o = 0;
        for (int t = t0; t <= pos; t++)
            o += ah[t] * kv_pair(vc + base + (ulong)t * row_b, i / 2, a.q8)[i & 1];
        out[h * hd + i] = o / sum;
    }
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
    threadgroup float red[256];
    uint tid = tid3.x, tpg = tpg3.x;
    uint h = tgpig.x, z = tgpig.y;
    int hd = a.head_dim;
    int kvh = h / (a.n_head / a.n_head_kv);
    int kv_dim = a.n_head_kv * hd;
    ulong row_b = kv_row_bytes(kv_dim, a.q8);
    ulong base = a.l_off + kv_head_off(kvh, hd, a.q8);
    int pos = a.pos;
    device const float *qh = q_all + h * hd;
    device float *ah = att_all + (ulong)h * a.n_ctx;

    int t0 = 0;
    if (a.window > 0 && pos - a.window + 1 > 0) t0 = pos - a.window + 1;
    int lo = t0 + (int)z * a.chunk;
    int hi = min(lo + a.chunk - 1, pos);

    device float *acc = acc_all + ((ulong)h * a.n_chunks + z) * hd;
    device float *ms  = ms_all  + ((ulong)h * a.n_chunks + z) * 2;

    if (lo > hi) {                       // empty chunk: neutral partial
        for (int i = tid; i < hd; i += tpg) acc[i] = 0;
        if (tid == 0) { ms[0] = -1e30f; ms[1] = 0; }
        return;
    }

    for (int t = lo + (int)tid; t <= hi; t += tpg)
        ah[t] = kv_dot(kc + base + (ulong)t * row_b, qh, hd, a.q8) * a.scale;
    threadgroup_barrier(mem_flags::mem_device);

    float mx = -1e30f;
    for (int t = lo + (int)tid; t <= hi; t += tpg) mx = max(mx, ah[t]);
    red[tid] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint off = tpg / 2; off > 0; off >>= 1) {
        if (tid < off) red[tid] = max(red[tid], red[tid + off]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    mx = red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float sum = 0;
    for (int t = lo + (int)tid; t <= hi; t += tpg) {
        float e = exp(ah[t] - mx);
        ah[t] = e;
        sum += e;
    }
    red[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint off = tpg / 2; off > 0; off >>= 1) {
        if (tid < off) red[tid] += red[tid + off];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    sum = red[0];
    threadgroup_barrier(mem_flags::mem_device);

    // UNNORMALISED: the combine pass owns the division, because the divisor is
    // not known until every chunk's maximum is.
    for (int i = tid; i < hd; i += tpg) {
        float o = 0;
        for (int t = lo; t <= hi; t++)
            o += ah[t] * kv_pair(vc + base + (ulong)t * row_b, i / 2, a.q8)[i & 1];
        acc[i] = o;
    }
    if (tid == 0) { ms[0] = mx; ms[1] = sum; }
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

kernel void k_silu_mul(device float       *g [[buffer(0)]],
                       device const float *u [[buffer(1)]],
                       constant int       &n [[buffer(2)]],
                       uint i [[thread_position_in_grid]]) {
    if ((int)i < n) {
        float x = g[i];
        g[i] = (x / (1.0f + exp(-x))) * u[i];
    }
}

kernel void k_gelu_mul(device float       *g [[buffer(0)]],
                       device const float *u [[buffer(1)]],
                       constant int       &n [[buffer(2)]],
                       uint i [[thread_position_in_grid]]) {
    if ((int)i < n) {
        float x = g[i];
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
        g[i] = 0.5f * x * (1.0f + t) * u[i];
    }
}

kernel void k_add(device float       *x [[buffer(0)]],
                  device const float *d [[buffer(1)]],
                  constant int       &n [[buffer(2)]],
                  uint i [[thread_position_in_grid]]) {
    if ((int)i < n) x[i] += d[i];
}

kernel void k_scale(device float       *x [[buffer(0)]],
                    constant float     &s [[buffer(1)]],
                    constant int       &n [[buffer(2)]],
                    uint i [[thread_position_in_grid]]) {
    if ((int)i < n) x[i] *= s;
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

kernel void k_moe_mv_q4_K(MOE_MV_PARAMS) {
    MOE_MV_HEAD;
    int nb = a.n_in / 256;
    device const uchar *rw = wbase + (ulong)row * nb * 144;
    for (int b = tiisg; b < nb; b += 32) {
        device const uchar *blk = rw + (ulong)b * 144;
        float d    = (float)*(device const half *)blk;
        float dmin = (float)*(device const half *)(blk + 2);
        device const uchar *sc = blk + 4;
        device const uchar *q  = blk + 16;
        device const float *xp = xp0 + b * 256;
        int is = 0;
        for (int j = 0; j < 256; j += 64) {
            uchar s1, m1, s2, m2;
            get_scale_min_k4(is + 0, sc, &s1, &m1);
            get_scale_min_k4(is + 1, sc, &s2, &m2);
            float d1 = d * s1, mm1 = dmin * m1;
            float d2 = d * s2, mm2 = dmin * m2;
            float t1 = 0, t2 = 0, sx1 = 0, sx2 = 0;
            for (int l = 0; l < 32; l++) {
                t1 += (float)(q[l] & 0xF) * xp[l];      sx1 += xp[l];
                t2 += (float)(q[l] >> 4)  * xp[l + 32]; sx2 += xp[l + 32];
            }
            s += d1 * t1 - mm1 * sx1 + d2 * t2 - mm2 * sx2;
            q += 32; is += 2; xp += 64;
        }
    }
    MOE_MV_TAIL;
}

kernel void k_moe_mv_q5_K(MOE_MV_PARAMS) {
    MOE_MV_HEAD;
    int nb = a.n_in / 256;
    device const uchar *rw = wbase + (ulong)row * nb * 176;
    for (int b = tiisg; b < nb; b += 32) {
        device const uchar *blk = rw + (ulong)b * 176;
        float d    = (float)*(device const half *)blk;
        float dmin = (float)*(device const half *)(blk + 2);
        device const uchar *sc = blk + 4;
        device const uchar *qh = blk + 16;
        device const uchar *q  = blk + 48;
        device const float *xp = xp0 + b * 256;
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
    MOE_MV_TAIL;
}

kernel void k_moe_mv_q6_K(MOE_MV_PARAMS) {
    MOE_MV_HEAD;
    int nb = a.n_in / 256;
    device const uchar *rw = wbase + (ulong)row * nb * 210;
    for (int b = tiisg; b < nb; b += 32) {
        device const uchar *blk = rw + (ulong)b * 210;
        device const uchar *ql = blk;
        device const uchar *qh = blk + 128;
        device const char  *sc = (device const char *)(blk + 192);
        float d = (float)*(device const half *)(blk + 208);
        device const float *xp = xp0 + b * 256;
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
