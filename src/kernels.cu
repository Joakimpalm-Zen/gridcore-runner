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

typedef unsigned char  uchar;
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

#define MVB 8   // max tokens per matvec tile (keep in sync with cuda.c)

struct mv_args {
    int     n_in;
    int     n_out;
    ulong64 w_off;    // tensor byte offset inside the weight buffer
    int     has_bias;
    int     batch;    // 1..MVB token columns (k_mv_* ignores it)
    int     xs, ys;   // element stride between x / y columns
};

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
    float s[MVB] = {0};

// apply one decoded weight w at element index idx to every token column
#define MV_FMA(w, idx) do { \
    float _w = (w); ulong64 _i = (idx); \
    _Pragma("unroll") \
    for (int t = 0; t < MVB; t++) s[t] += _w * x[(ulong64)t * a.xs + _i]; \
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
            for (int l = 0; l < 32; l++) {
                t1 += (float)(q[l] & 0xF) * xp[l];      sx1 += xp[l];
                t2 += (float)(q[l] >> 4)  * xp[l + 32]; sx2 += xp[l + 32];
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
            for (int l = 0; l < 32; l++) {
                MV_FMA(d1 * (float)(q[l] & 0xF) - mm1, base + j + l);
                MV_FMA(d2 * (float)(q[l] >> 4)  - mm2, base + j + l + 32);
            }
            q += 32; is += 2;
        }
    }
    MV_TAIL_B;
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

// ---------------------------------------------------------------- rope
// grid: (ceil(half_dim/32), n_heads, batch); vs = element stride per column

struct rope_args {
    int   head_dim, n_heads, half_dim, neox;
    float mscale;
};

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

// ---------------------------------------------------------------- kv store
// grid.y = token column; cache rows for consecutive positions are contiguous

extern "C" __global__ void k_store_kv(const float *k, const float *v,
                                      __half *kc, __half *vc,
                                      int kv_dim, ulong64 l_off,
                                      const int *posp) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < kv_dim) {
        ulong64 src = (ulong64)blockIdx.y * kv_dim + i;
        ulong64 dst = l_off + (ulong64)(*posp + blockIdx.y) * kv_dim + i;
        kc[dst] = __float2half(k[src]);
        vc[dst] = __float2half(v[src]);
    }
}

// ---------------------------------------------------------------- attention
// One block per (head, token): scores -> softmax -> weighted value sum.
// att scratch is MVB planes of [n_head][n_ctx].

struct attn_args {
    int     head_dim, n_head, n_head_kv, n_ctx;
    ulong64 l_off;    // this layer's element offset into the kv cache
    float   scale;
    int     qs, os;   // q / out element stride per token column
    int     window;   // sliding-window size for this layer (0 = full)
};

extern "C" __global__ void k_attn(const float *q, const __half *kc,
                                  const __half *vc, float *att, float *out,
                                  attn_args a, const int *posp) {
    __shared__ float red[256];
    int h = blockIdx.x, tid = threadIdx.x, tpg = blockDim.x;
    int tk = blockIdx.y;                 // token column in the tile
    int pos = *posp + tk;
    int hd = a.head_dim;
    int kvh = h / (a.n_head / a.n_head_kv);
    int kv_dim = a.n_head_kv * hd;
    int t0 = 0;                          // sliding-window start
    if (a.window > 0 && pos - a.window + 1 > 0) t0 = pos - a.window + 1;
    const float *qh = q + (ulong64)tk * a.qs + h * hd;
    float *ah = att + ((ulong64)tk * a.n_head + h) * a.n_ctx;

    for (int t = t0 + tid; t <= pos; t += tpg) {
        const __half *kt = kc + a.l_off + (ulong64)t * kv_dim + kvh * hd;
        float s = 0;
        for (int i = 0; i < hd; i++) s += qh[i] * __half2float(kt[i]);
        ah[t] = s * a.scale;
    }
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
    __syncthreads();

    for (int i = tid; i < hd; i += tpg) {
        float o = 0;
        for (int t = t0; t <= pos; t++)
            o += ah[t] * __half2float(vc[a.l_off + (ulong64)t * kv_dim + kvh * hd + i]);
        out[(ulong64)tk * a.os + h * hd + i] = o / sum;
    }
}

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

extern "C" __global__ void k_add(float *x, const float *d, int n, int xs, int ds) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[(ulong64)blockIdx.y * xs + i] += d[(ulong64)blockIdx.y * ds + i];
}

// whole-layer output scalar (gemma4): x *= s, grid.y = token column
extern "C" __global__ void k_scale(float *x, float s, int n, int xs) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[(ulong64)blockIdx.y * xs + i] *= s;
}
