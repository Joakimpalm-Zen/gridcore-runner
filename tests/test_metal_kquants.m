// Direct Metal matvec/GEMM parity against the CPU dequantizer.
// Identity columns reconstruct every weight, so an index error cannot hide in
// a dot-product sum. Additional shapes cover grid tails and superblock strides.
//
// The sweep covers q4_0, q8_0, q4_K, q6_K, f16 and f32 as well as q2_K/q3_K.
// Only the last two had a direct parity test; the rest were covered solely by
// end-to-end CPU-vs-GPU runs, which cannot say WHICH kernel drifted and cannot
// reach a shape a shipped model does not happen to have. Every n_out below is
// deliberately not a multiple of the rows one threadgroup covers, so each case
// lands on a partial last threadgroup.
#import <Metal/Metal.h>

#include "../src/fp16.h"
#include "../src/kernels_metal.h"
#include "../src/quants.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int n_in, n_out;
    uint64_t w_off;
    int has_bias, n_col, x_stride, y_stride, col_tile;
} mv_args_host;

typedef struct {
    int n_in, n_out, n_col;
    uint64_t w_off;
    int has_bias, x_stride, y_stride;
} mm_args_host;

_Static_assert(sizeof(mv_args_host) == 40, "mv_args host/Metal layout drift");
_Static_assert(sizeof(mm_args_host) == 40, "mm_args host/Metal layout drift");

// The matvec dispatch geometry src/metal.m uses, restated here because a
// Metal source embedded as a C string shares no header with its callers: one
// output row per simdgroup, 128 threads = 4 simdgroups per threadgroup. Get
// this wrong and the last threadgroup stops short of n_out, the tail rows are
// never written, and they keep the sentinel this test fills the output with.
#define MV_TG 128
#define MV_ROWS_PER_TG (MV_TG / 32)

static int failures;
static uint32_t rng_state = 0x9e3779b9u;

#define CHECK(cond, ...) do { if (!(cond)) { \
    fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
    fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); failures++; \
} } while (0)

static uint32_t rnd32(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}

static float rnd_float(void) {
    return (float)(int32_t)rnd32() / 2147483648.0f;
}

// Random payload bytes, then real scales written back over them: a random f16
// scale is as likely to be inf or NaN as anything else, and the reference sum
// would then agree with the kernel on garbage.
static void make_weights(int type, uint8_t *weights, int n_in, int n_out) {
    size_t row_size = ggml_row_size(type, n_in);
    size_t block_size = ggml_type_size(type);
    int per_block = (type == T_Q4_0 || type == T_Q8_0) ? 32 : 256;
    int n_block = n_in / per_block;
    for (int row = 0; row < n_out; row++) {
        uint8_t *row_data = weights + (size_t)row * row_size;
        if (type == T_F32) {
            float *f = (float *)row_data;
            for (int i = 0; i < n_in; i++) f[i] = rnd_float();
            continue;
        }
        if (type == T_F16) {
            f16_t *h = (f16_t *)row_data;
            for (int i = 0; i < n_in; i++) h[i] = f32_to_f16(rnd_float());
            continue;
        }
        for (size_t i = 0; i < row_size; i++) row_data[i] = (uint8_t)rnd32();
        for (int block = 0; block < n_block; block++) {
            uint8_t *b = row_data + (size_t)block * block_size;
            f16_t d = f32_to_f16(0.015625f * (1 + (row + block) % 4));
            f16_t dmin = f32_to_f16(0.0078125f * (1 + (row + 2 * block) % 4));
            switch (type) {
            case T_Q2_K:
                memcpy(b + 80, &d, sizeof d);
                memcpy(b + 82, &dmin, sizeof dmin);
                break;
            case T_Q3_K: memcpy(b + 108, &d, sizeof d); break;
            case T_Q6_K: memcpy(b + 208, &d, sizeof d); break;
            case T_Q4_K:
                memcpy(b + 0, &d, sizeof d);
                memcpy(b + 2, &dmin, sizeof dmin);
                break;
            case T_Q4_0:
            case T_Q8_0: memcpy(b + 0, &d, sizeof d); break;
            default: break;
            }
        }
    }
}

static id<MTLComputePipelineState> make_pipeline(id<MTLDevice> dev,
                                                  id<MTLLibrary> lib,
                                                  const char *name) {
    id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
    if (!fn) {
        CHECK(false, "missing Metal function %s", name);
        return nil;
    }
    NSError *err = nil;
    id<MTLComputePipelineState> pipeline =
        [dev newComputePipelineStateWithFunction:fn error:&err];
    [fn release];
    if (!pipeline)
        CHECK(false, "pipeline %s: %s", name,
              err ? err.localizedDescription.UTF8String : "unknown error");
    return pipeline;
}

static bool submit(id<MTLCommandQueue> queue,
                   id<MTLComputePipelineState> pipeline,
                   id<MTLBuffer> weights, id<MTLBuffer> x, id<MTLBuffer> y,
                   id<MTLBuffer> bias, const void *args, size_t args_size,
                   bool mm, int n_out, int n_col, int col_tile) {
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:weights offset:0 atIndex:0];
    [enc setBuffer:x offset:0 atIndex:1];
    [enc setBuffer:y offset:0 atIndex:2];
    [enc setBytes:args length:args_size atIndex:3];
    [enc setBuffer:bias offset:0 atIndex:4];
    // MM_TILE_M/MM_TILE_N mirror MM_TM/MM_TN in kernels.metal -- see the
    // matching comment in src/metal.m's enc_mv_n.
    enum { MM_TILE_M = 64, MM_TILE_N = 32 };
    MTLSize grid = mm
        // union of the two 2026-08-12 reworks: the GEMM branch's 64x32
        // tile geometry AND the matvec branch's row-group dispatch
        ? MTLSizeMake((n_out + MM_TILE_M - 1) / MM_TILE_M,
                      (n_col + MM_TILE_N - 1) / MM_TILE_N, 1)
        : MTLSizeMake((n_out + MV_ROWS_PER_TG - 1) / MV_ROWS_PER_TG,
                      (n_col + col_tile - 1) / col_tile, 1);
    [enc dispatchThreadgroups:grid
        threadsPerThreadgroup:MTLSizeMake(mm ? 128 : MV_TG, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status == MTLCommandBufferStatusError) {
        CHECK(false, "Metal dispatch failed: %s",
              cb.error ? cb.error.localizedDescription.UTF8String : "unknown error");
        return false;
    }
    return true;
}

static void run_case(id<MTLDevice> dev, id<MTLCommandQueue> queue,
                     id<MTLComputePipelineState> pipeline, int type, bool mm,
                     int n_in, int n_out, int n_col, bool identity) {
    const size_t w_off = 32;
    size_t row_size = ggml_row_size(type, n_in);
    size_t weights_size = row_size * (size_t)n_out;
    int x_stride = n_in + 3, y_stride = n_out + 5;
    size_t x_count = (size_t)n_col * x_stride;
    size_t y_count = (size_t)n_col * y_stride;
    uint8_t *weights_host = calloc(1, w_off + weights_size);
    float *x_host = calloc(x_count, sizeof(float));
    float *y_host = malloc(y_count * sizeof(float));
    float *bias_host = calloc((size_t)n_out, sizeof(float));
    float *decoded = malloc((size_t)n_in * sizeof(float));
    CHECK(weights_host && x_host && y_host && bias_host && decoded,
          "host allocation for %s", ggml_type_name(type));
    if (!weights_host || !x_host || !y_host || !bias_host || !decoded) goto done;

    make_weights(type, weights_host + w_off, n_in, n_out);
    for (size_t i = 0; i < y_count; i++) y_host[i] = 123456.0f;
    if (identity) {
        CHECK(n_col == n_in, "identity case needs n_col == n_in");
        for (int col = 0; col < n_col; col++) x_host[(size_t)col * x_stride + col] = 1.0f;
    } else {
        for (int col = 0; col < n_col; col++)
            for (int k = 0; k < n_in; k++)
                x_host[(size_t)col * x_stride + k] = rnd_float();
    }

    id<MTLBuffer> wb = [dev newBufferWithBytes:weights_host
                                        length:w_off + weights_size
                                       options:MTLResourceStorageModeShared];
    id<MTLBuffer> xb = [dev newBufferWithBytes:x_host
                                        length:x_count * sizeof(float)
                                       options:MTLResourceStorageModeShared];
    id<MTLBuffer> yb = [dev newBufferWithBytes:y_host
                                        length:y_count * sizeof(float)
                                       options:MTLResourceStorageModeShared];
    id<MTLBuffer> bb = [dev newBufferWithBytes:bias_host
                                        length:(size_t)n_out * sizeof(float)
                                       options:MTLResourceStorageModeShared];
    CHECK(wb && xb && yb && bb, "Metal buffer allocation for %s", ggml_type_name(type));
    if (!wb || !xb || !yb || !bb) {
        [wb release]; [xb release]; [yb release]; [bb release];
        goto done;
    }

    bool submitted;
    if (mm) {
        mm_args_host args = { n_in, n_out, n_col, w_off, 0, x_stride, y_stride };
        submitted = submit(queue, pipeline, wb, xb, yb, bb, &args, sizeof args,
                           true, n_out, n_col, 1);
    } else {
        int col_tile = n_col < 8 ? n_col : 8;
        mv_args_host args = { n_in, n_out, w_off, 0, n_col,
                              x_stride, y_stride, col_tile };
        submitted = submit(queue, pipeline, wb, xb, yb, bb, &args, sizeof args,
                           false, n_out, n_col, col_tile);
    }

    if (submitted) {
        const float *got = yb.contents;
        for (int row = 0; row < n_out; row++) {
            const uint8_t *row_data = weights_host + w_off + (size_t)row * row_size;
            dequant_row(type, row_data, decoded, n_in);
            for (int col = 0; col < n_col; col++) {
                double ref = 0.0, mag = 0.0;
                const float *xc = x_host + (size_t)col * x_stride;
                for (int k = 0; k < n_in; k++) {
                    double term = (double)decoded[k] * xc[k];
                    ref += term;
                    mag += fabs(term);
                }
                double actual = got[(size_t)col * y_stride + row];
                double tol = 5e-5 * mag + 2e-5;
                if (!isfinite(actual) || fabs(actual - ref) > tol) {
                    CHECK(false,
                          "%s %s n_in=%d n_out=%d n_col=%d row=%d col=%d: "
                          "got %.9g ref %.9g tol %.3g",
                          ggml_type_name(type), mm ? "mm" : "mv", n_in, n_out,
                          n_col, row, col, actual, ref, tol);
                    if (failures > 20) break;
                }
            }
            if (failures > 20) break;
        }
    }
    [wb release]; [xb release]; [yb release]; [bb release];

done:
    free(weights_host); free(x_host); free(y_host); free(bias_host); free(decoded);
}

int main(void) {
    @autoreleasepool {
        f16_init();
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) {
            printf("metal kquant kernels: skipped (no Metal device)\n");
            return 0;
        }
        NSError *err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:
                                  [NSString stringWithUTF8String:k_metal_src]
                                              options:nil error:&err];
        if (!lib) {
            fprintf(stderr, "FAIL: Metal library compile: %s\n",
                    err ? err.localizedDescription.UTF8String : "unknown error");
            [dev release];
            return 1;
        }
        id<MTLCommandQueue> queue = [dev newCommandQueue];
        // n_in granularity: superblock types index 256 weights at a time, the
        // 32-block types 32, and the raw float types nothing.
        const struct { int type; const char *suffix; int step; } kinds[] = {
            { T_Q2_K, "q2_K", 256 }, { T_Q3_K, "q3_K", 256 },
            { T_Q4_K, "q4_K", 256 }, { T_Q6_K, "q6_K", 256 },
            { T_Q4_0, "q4_0",  32 }, { T_Q8_0, "q8_0",  32 },
            { T_F16,  "f16",    1 }, { T_F32,  "f32",    1 },
        };
        for (size_t i = 0; i < sizeof kinds / sizeof *kinds; i++) {
            int type = kinds[i].type, step = kinds[i].step;
            char mv_name[32], mm_name[32];
            snprintf(mv_name, sizeof mv_name, "k_mv_%s", kinds[i].suffix);
            snprintf(mm_name, sizeof mm_name, "k_mm_%s", kinds[i].suffix);
            id<MTLComputePipelineState> mv = make_pipeline(dev, lib, mv_name);
            id<MTLComputePipelineState> mm = make_pipeline(dev, lib, mm_name);
            if (mv && mm) {
                run_case(dev, queue, mv, type, false, 512, 3, 512, true);
                run_case(dev, queue, mm, type, true, 512, 3, 512, true);
                const int shapes[][3] = {
                    { 256, 1, 2 }, { 512, 33, 7 }, { 768, 35, 17 },
                    { 288, 13, 3 },   // n_in not a whole number of superblocks
                    { 258, 21, 5 },   // and not even a whole float4
                };
                for (size_t s = 0; s < sizeof shapes / sizeof *shapes; s++) {
                    if (shapes[s][0] % step) continue;
                    run_case(dev, queue, mv, type, false,
                             shapes[s][0], shapes[s][1], shapes[s][2], false);
                    // The GEMM kernels index whole k-steps of 32.
                    if (shapes[s][0] % 32) continue;
                    run_case(dev, queue, mm, type, true,
                             shapes[s][0], shapes[s][1], shapes[s][2], false);
                }
            }
            [mv release]; [mm release];
        }
        [queue release];
        [lib release];
        [dev release];
    }
    if (failures) {
        fprintf(stderr, "metal kquant kernels: %d failures\n", failures);
        return 1;
    }
    printf("metal kernels: q2_K/q3_K/q4_K/q6_K/q4_0/q8_0/f16/f32 "
           "mv+mm shape sweep ok\n");
    return 0;
}
