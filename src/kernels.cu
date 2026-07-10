// CUDA compute kernels: the full single-token forward pass.
// 1:1 port of kernels.metal; mirrors quants.c/model.c bit layouts exactly.
// Compiled to PTX at development time (make ptx) and embedded via kernels_ptx.h;
// the driver JIT-compiles for the resident GPU. No toolkit needed at runtime.
#include <cuda_fp16.h>

typedef unsigned char  uchar;
typedef unsigned int   uint;
typedef unsigned long long ulong64;

static __device__ __forceinline__ float f16f(const uchar *p) {
    return __half2float(*(const __half *)p);
}

// ---------------------------------------------------------------- rmsnorm

extern "C" __global__ void k_rmsnorm(const float *x, float *y, const float *w,
                                     int n, float eps) {
    __shared__ float red[256];
    int tid = threadIdx.x, tpg = blockDim.x;
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

// per-head RMSNorm (qwen3 Q/K norm): one block per head
extern "C" __global__ void k_qknorm(float *v, const float *w, int hd, float eps) {
    __shared__ float red[128];
    int tid = threadIdx.x, tpg = blockDim.x;
    float *x = v + blockIdx.x * hd;
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

struct mv_args {
    int     n_in;
    int     n_out;
    ulong64 w_off;    // tensor byte offset inside the weight buffer
    int     has_bias;
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

extern "C" __global__ void k_mv_f32(const uchar *wb, const float *x, float *y,
                                    mv_args a, const float *bias) {
    MV_HEAD;
    const float *rw = (const float *)(wb + a.w_off) + (ulong64)row * a.n_in;
    float s = 0;
    for (int i = lane; i < a.n_in; i += 32) s += rw[i] * x[i];
    MV_TAIL;
}

extern "C" __global__ void k_mv_f16(const uchar *wb, const float *x, float *y,
                                    mv_args a, const float *bias) {
    MV_HEAD;
    const __half *rw = (const __half *)(wb + a.w_off) + (ulong64)row * a.n_in;
    float s = 0;
    for (int i = lane; i < a.n_in; i += 32) s += __half2float(rw[i]) * x[i];
    MV_TAIL;
}

extern "C" __global__ void k_mv_q8_0(const uchar *wb, const float *x, float *y,
                                     mv_args a, const float *bias) {
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

extern "C" __global__ void k_mv_q4_0(const uchar *wb, const float *x, float *y,
                                     mv_args a, const float *bias) {
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

extern "C" __global__ void k_mv_q4_1(const uchar *wb, const float *x, float *y,
                                     mv_args a, const float *bias) {
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

extern "C" __global__ void k_mv_q5_0(const uchar *wb, const float *x, float *y,
                                     mv_args a, const float *bias) {
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

extern "C" __global__ void k_mv_q5_1(const uchar *wb, const float *x, float *y,
                                     mv_args a, const float *bias) {
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

extern "C" __global__ void k_mv_q4_K(const uchar *wb, const float *x, float *y,
                                     mv_args a, const float *bias) {
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

extern "C" __global__ void k_mv_q5_K(const uchar *wb, const float *x, float *y,
                                     mv_args a, const float *bias) {
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

extern "C" __global__ void k_mv_q6_K(const uchar *wb, const float *x, float *y,
                                     mv_args a, const float *bias) {
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

// ---------------------------------------------------------------- rope

struct rope_args {
    int   head_dim, n_heads, half_dim, pos, neox;
    float mscale;
};

extern "C" __global__ void k_rope(float *v, const float *fr, rope_args a) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y;
    if (j >= a.half_dim || h >= a.n_heads) return;
    float ang = a.pos * fr[j];
    float c = cosf(ang) * a.mscale, s = sinf(ang) * a.mscale;
    float *p = v + h * a.head_dim;
    int i0 = a.neox ? j : 2 * j;
    int i1 = a.neox ? j + a.half_dim : i0 + 1;
    float x0 = p[i0], x1 = p[i1];
    p[i0] = x0 * c - x1 * s;
    p[i1] = x0 * s + x1 * c;
}

// ---------------------------------------------------------------- kv store

extern "C" __global__ void k_store_kv(const float *k, const float *v,
                                      __half *kc, __half *vc,
                                      int kv_dim, ulong64 off) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < kv_dim) {
        kc[off + i] = __float2half(k[i]);
        vc[off + i] = __float2half(v[i]);
    }
}

// ---------------------------------------------------------------- attention
// One block per head: scores -> softmax -> weighted value sum.

struct attn_args {
    int     head_dim, n_head, n_head_kv, n_ctx, pos;
    ulong64 l_off;    // this layer's element offset into the kv cache
    float   scale;
};

extern "C" __global__ void k_attn(const float *q, const __half *kc,
                                  const __half *vc, float *att, float *out,
                                  attn_args a) {
    __shared__ float red[256];
    int h = blockIdx.x, tid = threadIdx.x, tpg = blockDim.x;
    int hd = a.head_dim;
    int kvh = h / (a.n_head / a.n_head_kv);
    int kv_dim = a.n_head_kv * hd;
    const float *qh = q + h * hd;
    float *ah = att + (ulong64)h * a.n_ctx;

    for (int t = tid; t <= a.pos; t += tpg) {
        const __half *kt = kc + a.l_off + (ulong64)t * kv_dim + kvh * hd;
        float s = 0;
        for (int i = 0; i < hd; i++) s += qh[i] * __half2float(kt[i]);
        ah[t] = s * a.scale;
    }
    __syncthreads();

    // max
    float mx = -1e30f;
    for (int t = tid; t <= a.pos; t += tpg) mx = fmaxf(mx, ah[t]);
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
    for (int t = tid; t <= a.pos; t += tpg) {
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
        for (int t = 0; t <= a.pos; t++)
            o += ah[t] * __half2float(vc[a.l_off + (ulong64)t * kv_dim + kvh * hd + i]);
        out[h * hd + i] = o / sum;
    }
}

// ---------------------------------------------------------------- elementwise

extern "C" __global__ void k_silu_mul(float *g, const float *u, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = g[i];
        g[i] = (x / (1.0f + expf(-x))) * u[i];
    }
}

extern "C" __global__ void k_add(float *x, const float *d, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += d[i];
}
