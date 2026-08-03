// Metal compute kernels: the full single-token forward pass.
// Mirrors the CPU implementations in quants.c/model.c bit-layout for bit-layout.
#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------- rmsnorm

kernel void k_rmsnorm(device const float *x   [[buffer(0)]],
                      device float       *y   [[buffer(1)]],
                      device const float *w   [[buffer(2)]],
                      constant int       &n   [[buffer(3)]],
                      constant float     &eps [[buffer(4)]],
                      uint tid [[thread_position_in_threadgroup]],
                      uint tpg [[threads_per_threadgroup]]) {
    threadgroup float red[256];
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

struct mv_args {
    int   n_in;
    int   n_out;
    ulong w_off;      // tensor byte offset inside the weight buffer
    int   has_bias;
};

#define MV_HEAD \
    uint row = tgpig.x * (ntg.x / 32) + sgitg; \
    if (row >= (uint)a.n_out) return;

#define MV_TAIL \
    s = simd_sum(s); \
    if (tiisg == 0) y[row] = a.has_bias ? s + bias[row] : s;

#define MV_PARAMS \
    device const uchar *wb   [[buffer(0)]], \
    device const float *x    [[buffer(1)]], \
    device float       *y    [[buffer(2)]], \
    constant mv_args   &a    [[buffer(3)]], \
    device const float *bias [[buffer(4)]], \
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

kernel void k_mv_q4_K(MV_PARAMS) {
    MV_HEAD;
    int nb = a.n_in / 256;
    device const uchar *rw = wb + a.w_off + (ulong)row * nb * 144;
    float s = 0;
    for (int b = tiisg; b < nb; b += 32) {
        device const uchar *blk = rw + (ulong)b * 144;
        float d    = (float)*(device const half *)blk;
        float dmin = (float)*(device const half *)(blk + 2);
        device const uchar *sc = blk + 4;
        device const uchar *q  = blk + 16;
        device const float *xp = x + b * 256;
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

kernel void k_mv_q5_K(MV_PARAMS) {
    MV_HEAD;
    int nb = a.n_in / 256;
    device const uchar *rw = wb + a.w_off + (ulong)row * nb * 176;
    float s = 0;
    for (int b = tiisg; b < nb; b += 32) {
        device const uchar *blk = rw + (ulong)b * 176;
        float d    = (float)*(device const half *)blk;
        float dmin = (float)*(device const half *)(blk + 2);
        device const uchar *sc = blk + 4;
        device const uchar *qh = blk + 16;
        device const uchar *q  = blk + 48;
        device const float *xp = x + b * 256;
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

kernel void k_mv_q6_K(MV_PARAMS) {
    MV_HEAD;
    int nb = a.n_in / 256;
    device const uchar *rw = wb + a.w_off + (ulong)row * nb * 210;
    float s = 0;
    for (int b = tiisg; b < nb; b += 32) {
        device const uchar *blk = rw + (ulong)b * 210;
        device const uchar *ql = blk;
        device const uchar *qh = blk + 128;
        device const char  *sc = (device const char *)(blk + 192);
        float d = (float)*(device const half *)(blk + 208);
        device const float *xp = x + b * 256;
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

// ---------------------------------------------------------------- rope

struct rope_args {
    int   head_dim, n_heads, half_dim, pos, neox;
    float mscale;
};

kernel void k_rope(device float       *v  [[buffer(0)]],
                   device const float *fr [[buffer(1)]],
                   constant rope_args &a  [[buffer(2)]],
                   uint2 gid [[thread_position_in_grid]]) {
    int j = gid.x, h = gid.y;
    if (j >= a.half_dim || h >= a.n_heads) return;
    float ang = a.pos * fr[j];
    float c = cos(ang) * a.mscale, s = sin(ang) * a.mscale;
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

kernel void k_store_kv(device const float *k  [[buffer(0)]],
                       device const float *v  [[buffer(1)]],
                       device uchar       *kc [[buffer(2)]],
                       device uchar       *vc [[buffer(3)]],
                       constant int       &kv_dim [[buffer(4)]],
                       constant ulong     &off    [[buffer(5)]],
                       constant int       &q8     [[buffer(6)]],
                       uint i [[thread_position_in_grid]]) {
    int n = q8 ? kv_dim / 32 : kv_dim;
    if ((int)i < n) {
        kv_store_row(kc + off, k, q8, i);
        kv_store_row(vc + off, v, q8, i);
    }
}

// ---------------------------------------------------------------- attention
// One threadgroup per head: scores -> softmax -> weighted value sum.

struct attn_args {
    int   head_dim, n_head, n_head_kv, n_ctx, pos;
    ulong l_off;      // this layer's byte offset into the kv cache
    float scale;
    int   q8;
    int   window;     // sliding-window size for this layer (0 = full)
    int   has_sinks;  // gpt-oss: per-head sink joins softmax denominator only
};

kernel void k_attn(device const float *q   [[buffer(0)]],
                   device const uchar *kc  [[buffer(1)]],
                   device const uchar *vc  [[buffer(2)]],
                   device float       *att [[buffer(3)]],
                   device float       *out [[buffer(4)]],
                   constant attn_args &a   [[buffer(5)]],
                   device const float *sinks [[buffer(6)]],
                   uint h   [[threadgroup_position_in_grid]],
                   uint tid [[thread_position_in_threadgroup]],
                   uint tpg [[threads_per_threadgroup]]) {
    threadgroup float red[256];
    int hd = a.head_dim;
    int kvh = h / (a.n_head / a.n_head_kv);
    int kv_dim = a.n_head_kv * hd;
    ulong row_b = kv_row_bytes(kv_dim, a.q8);
    ulong base = a.l_off + kv_head_off(kvh, hd, a.q8);
    device const float *qh = q + h * hd;
    device float *ah = att + (ulong)h * a.n_ctx;
    int t0 = 0;
    if (a.window > 0 && a.pos - a.window + 1 > 0) t0 = a.pos - a.window + 1;

    for (int t = t0 + tid; t <= a.pos; t += tpg) {
        ah[t] = kv_dot(kc + base + (ulong)t * row_b, qh, hd, a.q8) * a.scale;
    }
    threadgroup_barrier(mem_flags::mem_device);

    // max
    float mx = -1e30f;
    for (int t = t0 + tid; t <= a.pos; t += tpg) mx = max(mx, ah[t]);
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
    for (int t = t0 + tid; t <= a.pos; t += tpg) {
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
        for (int t = t0; t <= a.pos; t++)
            o += ah[t] * kv_pair(vc + base + (ulong)t * row_b, i / 2, a.q8)[i & 1];
        out[h * hd + i] = o / sum;
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
        float t = tanh(0.7978845608f * (x + 0.044715f * x * x * x));
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
