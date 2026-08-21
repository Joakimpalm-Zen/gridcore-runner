// D8 slice 1: the transposed matvec on the device, byte-identical to the
// CPU trainer's chain.
//
// The deterministic-training claim extends to CUDA only if the GPU produces
// THE SAME BYTES as the CPU backward — not close, identical. This gate runs
// dx += W^T dy on real fixture tensors through both paths and memcmp's the
// float buffers: the kernel starts each accumulator from dx's incoming
// value, walks j serially per output element with fmaf, and skips zero
// dy[j] exactly as the CPU loop does, so any difference is a decode
// mismatch, not reduction order. Also pins determinism (two GPU runs
// byte-equal) and prints throughput for the record.
//
// Skips (not passes) without a CUDA device or when the type has no k_mvt_*.
//
//     ./test-mvt test.gguf         # F32 tensors
//     ./test-mvt test-q8.gguf      # Q8_0
//     ./test-mvt test-bf16.gguf    # BF16
#include "runner.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { \
    fprintf(stderr, "FAIL: "); fprintf(stderr, __VA_ARGS__); \
    fprintf(stderr, "\n"); g_fail = 1; } } while (0)

static double clk(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

static uint64_t rng = 0x9E3779B97F4A7C15ull;
static float frnd(void) {
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return (float)(int32_t)(uint32_t)(rng >> 32) / 2147483648.0f;
}

// the CPU trainer's chain, verbatim semantics (model.c matvec_t)
static void cpu_mvt(const gguf_tensor *w, const float *dy, float *dx,
                    int n_in, int n_out, int batch) {
    float *rowbuf = malloc(sizeof(float) * (size_t)n_in);
    size_t rs = ggml_row_size(w->type, n_in);
    for (int t = 0; t < batch; t++) {
        const float *dyt = dy + (size_t)t * n_out;
        float *dxt = dx + (size_t)t * n_in;
        for (int j = 0; j < n_out; j++) {
            float d = dyt[j];
            if (d == 0.0f) continue;
            dequant_row(w->type, (const uint8_t *)w->data + (size_t)j * rs,
                        rowbuf, n_in);
            for (int i = 0; i < n_in; i++)
                dxt[i] = fmaf(rowbuf[i], d, dxt[i]);
        }
    }
    free(rowbuf);
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "test.gguf";
    f16_init();
    model_params p;
    memset(&p, 0, sizeof(p));
    p.gpu_mode = GPU_AUTO;
    p.n_threads = 2;
    p.n_ctx = 64;
    model_t m;
    memset(&m, 0, sizeof(m));
    if (!model_load(&m, path, &p)) {
        fprintf(stderr, "FAIL: cannot load %s\n", path);
        return 1;
    }
    if (!m.gpu) {
        printf("mvt: no GPU backend — skipping (not passing)\n");
        model_free(&m);
        return 0;
    }
    enum { BATCH = 3 };
    int tested = 0;
    for (int l = 0; l < m.n_layer && l < 2; l++) {
        gguf_tensor *ws[] = { m.layers[l].wq, m.layers[l].w_down };
        for (size_t k = 0; k < sizeof(ws) / sizeof(*ws); k++) {
            gguf_tensor *w = ws[k];
            if (!w) continue;
            int n_in = (int)w->ne[0], n_out = (int)w->ne[1];
            float *dy = malloc(sizeof(float) * BATCH * (size_t)n_out);
            float *dx_cpu = malloc(sizeof(float) * BATCH * (size_t)n_in);
            float *dx_gpu = malloc(sizeof(float) * BATCH * (size_t)n_in);
            float *dx_gpu2 = malloc(sizeof(float) * BATCH * (size_t)n_in);
            for (int i = 0; i < BATCH * n_out; i++)
                dy[i] = (i % 7 == 0) ? 0.0f : frnd();  // exercise the skip
            for (int i = 0; i < BATCH * n_in; i++)
                dx_cpu[i] = frnd();
            memcpy(dx_gpu, dx_cpu, sizeof(float) * BATCH * (size_t)n_in);
            memcpy(dx_gpu2, dx_cpu, sizeof(float) * BATCH * (size_t)n_in);
            if (!gpu_mvt(&m, w, dy, dx_gpu, n_in, n_out, BATCH)) {
                // a backend without training kernels (Metal today) skips the
                // whole gate rather than failing it; the CUDA runs are where
                // this gate asserts
                printf("mvt: %s — no device kernel (type %d), skipping\n",
                       w->name, w->type);
                free(dy); free(dx_cpu); free(dx_gpu); free(dx_gpu2);
                continue;
            }
            cpu_mvt(w, dy, dx_cpu, n_in, n_out, BATCH);
            CHECK(memcmp(dx_cpu, dx_gpu,
                         sizeof(float) * BATCH * (size_t)n_in) == 0,
                  "%s: GPU result differs from the CPU chain", w->name);
            CHECK(gpu_mvt(&m, w, dy, dx_gpu2, n_in, n_out, BATCH) &&
                  memcmp(dx_gpu, dx_gpu2,
                         sizeof(float) * BATCH * (size_t)n_in) == 0,
                  "%s: GPU result not deterministic across runs", w->name);
            tested++;
            free(dy); free(dx_cpu); free(dx_gpu); free(dx_gpu2);
        }
    }
    if (!tested) {
        printf("mvt: no device training kernels on this backend — skipping "
               "(not passing)\n");
        model_free(&m);
        return 0;
    }

    // throughput on the largest tensor (the lm_head shape dominates the
    // real backward): printed, not asserted
    if (tested && m.output) {
        int n_in = (int)m.output->ne[0], n_out = (int)m.output->ne[1];
        float *dy = calloc((size_t)n_out, sizeof(float));
        float *dx = calloc((size_t)n_in, sizeof(float));
        for (int i = 0; i < n_out; i++) dy[i] = frnd();
        if (gpu_mvt(&m, m.output, dy, dx, n_in, n_out, 1)) {
            double t0 = clk();
            int reps = 20;
            for (int r = 0; r < reps; r++)
                gpu_mvt(&m, m.output, dy, dx, n_in, n_out, 1);
            double gpu_s = (clk() - t0) / reps;
            t0 = clk();
            cpu_mvt(m.output, dy, dx, n_in, n_out, 1);
            double cpu_s = clk() - t0;
            printf("mvt: head [%d x %d] gpu %.3f ms  cpu %.3f ms  (%.1fx)\n",
                   n_in, n_out, gpu_s * 1e3, cpu_s * 1e3, cpu_s / gpu_s);
        }
        free(dy); free(dx);
    }
    model_free(&m);
    printf(g_fail ? "mvt: FAILED\n" : "mvt: ok\n");
    return g_fail;
}
