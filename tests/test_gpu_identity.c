// CPU vs GPU agreement on the scalar route, at LOGIT precision.
//
// What the project's CPU/GPU contract actually is, stated precisely because
// this gate was first written against a stronger reading of it and failed
// everywhere, including on code nobody had touched: the scalar route is
// TOKEN-identical, and that is what `make test-metal-kquant` and friends
// verify by comparing two runs' generated text. It is NOT logit-bit-identical.
// The CPU sums a row with SIMD lanes, the GPU with a 32-lane simd_sum
// reduction tree; those round differently, and the measured residue is around
// 1e-7 of a logit on an F32 toy and 1e-3 on a real Q8_0 model -- far below
// anything argmax can see, which is exactly why token identity holds.
//
// So why add this at all? Because token identity is BLIND at fixture scale.
// tests/test_attn_knobs says so in its own docstring: on a toy random-weight
// model every variant emits byte-identical text. Scaling every Q by up to 9x
// on the llama-4 temperature fixture changes not one emitted token. A
// fixture-scale backend feature -- exactly the kind a backend grows one knob
// at a time -- could therefore be implemented WRONG on the GPU and pass every
// gate in the tree.
//
// This closes that gap by measuring the CPU/GPU logit deviation as a fraction
// of the logit range and bounding it. The bound sits far above the reduction
// residue and far below a real mistake: a backend that silently ignores the
// attention temperature lands orders of magnitude out, because it is not a
// rounding difference, it is a missing multiply.
//
//     ./test-gpu-identity test-attn/k_temp_live.gguf
//     ./test-gpu-identity models/granite-4.1-8b-Q4_0.gguf 8   # 8 GPU layers
//
// Skips (never passes quietly) when there is no GPU on the machine, or when
// the model fell back to the CPU -- comparing the CPU against itself is the
// vacuity every tolerance gate here is written to avoid.
#include "runner.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { STEPS = 24, MAX_TOK = 96, N_BATCH = 32 };

// Mean |dlogit| as a fraction of the mean logit range. Measured residue on
// this machine: 2.3e-8 (F32 toy), 1.9e-7 (the llama-4 temperature fixture),
// 5.4e-5 (SmolLM2 Q8_0 -- quantized weights dequantize slightly differently
// on the two sides, which dominates). The bound sits ~10x above the worst of
// those and ~4 orders below a real mistake: dropping the attention
// temperature entirely measures 2.05e-2, i.e. 41x over this limit
// and 5 orders of magnitude above the correct implementation's 1.9e-7.
#define GPU_DEV_FRAC 5e-4

static int g_gpu_layers = 0;

// Deliberately not the tolerance-gate corpus: this runs on toy fixtures whose
// vocabularies are tiny, so the text only has to tokenize, not to be prose.
static const char *TEXT =
    "The quick brown fox jumps over the lazy dog while the observatory "
    "published a revised catalogue listing four thousand objects in 1929.";

static float *run(const char *path, int gpu_mode, int *n_vocab_out,
                  bool *used_gpu, const int32_t *toks, int n_tok) {
    model_t m;
    memset(&m, 0, sizeof(m));
    model_params p;
    memset(&p, 0, sizeof(p));
    p.gpu_mode = gpu_mode;
    p.n_ctx    = n_tok + 8;
    p.n_batch  = N_BATCH;
    p.gpu_layers_override = gpu_mode == GPU_OFF ? 0 : g_gpu_layers;

    if (!model_load(&m, path, &p)) {
        fprintf(stderr, "cannot load %s\n", path);
        return NULL;
    }
    *used_gpu = m.gpu != NULL;
    *n_vocab_out = m.n_vocab;

    float *out = malloc(sizeof(float) * (size_t)STEPS * (size_t)m.n_vocab);
    if (!out) { model_free(&m); return NULL; }

    int prefill = n_tok - STEPS;
    float *lg = NULL;
    for (int off = 0; off < prefill; off += N_BATCH) {
        int n = prefill - off < N_BATCH ? prefill - off : N_BATCH;
        lg = model_forward_batch(&m, toks + off, n, off, off + n == prefill);
    }
    if (!lg) { model_free(&m); free(out); return NULL; }
    memcpy(out, lg, sizeof(float) * (size_t)m.n_vocab);

    for (int s = 1; s < STEPS; s++) {
        lg = model_forward(&m, toks[prefill + s - 1], prefill + s - 1);
        if (!lg) { model_free(&m); free(out); return NULL; }
        memcpy(out + (size_t)s * m.n_vocab, lg,
               sizeof(float) * (size_t)m.n_vocab);
    }
    model_free(&m);
    return out;
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "test.gguf";
    if (argc > 2) g_gpu_layers = atoi(argv[2]);

    f16_init();
    // Pin the identity route. The tiled prefill GEMM (RUNNER_METAL_MM) and the
    // reassociating decode matvec (RUNNER_METAL_MV) are deliberately NOT
    // byte-identical -- they answer to test_tc_tol.c and test_mv_tol.c. This
    // gate is about the contract they are exceptions to, so it forces both
    // off; leaving them at their defaults would make it fail by design.
    gpu_tc_force(0);
    gpu_mv_force(0);

    gguf_file gf;
    if (!gguf_open(&gf, path)) { fprintf(stderr, "cannot open %s\n", path); return 1; }
    tokenizer tk;
    if (!tokenizer_init(&tk, &gf)) {
        fprintf(stderr, "cannot init tokenizer for %s\n", path);
        gguf_close(&gf); return 1;
    }
    static int32_t toks[MAX_TOK];
    int n_tok = tok_encode(&tk, TEXT, toks, MAX_TOK, true, false);
    tokenizer_free(&tk);
    gguf_close(&gf);
    if (n_tok < 8) { fprintf(stderr, "text tokenized to %d tokens\n", n_tok); return 1; }
    for (int i = n_tok; i < MAX_TOK; i++) toks[i] = toks[i - n_tok + 1];
    n_tok = MAX_TOK;

    printf("gpu-identity: %s | %d tokens, %d teacher-forced positions\n",
           path, n_tok, STEPS);

    int nv_cpu = 0, nv_gpu = 0;
    bool cpu_used_gpu = false, gpu_used_gpu = false;
    float *cpu = run(path, GPU_OFF, &nv_cpu, &cpu_used_gpu, toks, n_tok);
    if (!cpu) { printf("gpu-identity: FAILED (cpu run)\n"); return 1; }
    float *gpu = run(path, GPU_AUTO, &nv_gpu, &gpu_used_gpu, toks, n_tok);
    if (!gpu) { free(cpu); printf("gpu-identity: FAILED (gpu run)\n"); return 1; }

    if (!gpu_used_gpu) {
        printf("  skipped: no GPU, or this model fell back to the CPU — a "
               "comparison of the CPU against itself proves nothing\n"
               "gpu-identity: ok (skipped)\n");
        free(cpu); free(gpu);
        return 0;
    }
    if (nv_cpu != nv_gpu) {
        printf("FAIL: vocab size differs (%d vs %d)\n", nv_cpu, nv_gpu);
        free(cpu); free(gpu);
        return 1;
    }

    size_t n = (size_t)STEPS * (size_t)nv_cpu;
    size_t n_diff = 0;
    double worst = 0, sum = 0;
    for (size_t i = 0; i < n; i++) {
        double d = fabs((double)cpu[i] - (double)gpu[i]);
        sum += d;
        if (d > worst) worst = d;
        if (memcmp(&cpu[i], &gpu[i], sizeof(float)) != 0) n_diff++;
    }
    double mean_dev = sum / (double)n;

    // Range over REAL logits only: some archs suppress vocabulary entries with
    // a large negative sentinel, which would make a fraction-of-range bound
    // vacuously true.
    double range_sum = 0;
    for (int s2 = 0; s2 < STEPS; s2++) {
        double lo = 1e300, hi = -1e300;
        for (int v = 0; v < nv_cpu; v++) {
            double x = (double)cpu[(size_t)s2 * nv_cpu + v];
            if (x <= -1e29) continue;
            if (x < lo) lo = x;
            if (x > hi) hi = x;
        }
        if (hi > lo) range_sum += hi - lo;
    }
    double range = range_sum / STEPS;
    double frac = range > 0 ? mean_dev / range : 1e9;

    // Anti-vacuity: all-zero logits would compare equal and prove nothing.
    double absmax = 0;
    for (size_t i = 0; i < n; i++) {
        double a = fabs((double)cpu[i]);
        if (a > absmax) absmax = a;
    }
    if (absmax < 1e-6) {
        printf("FAIL: the CPU run produced all-zero logits\n"
               "gpu-identity: FAILED\n");
        free(cpu); free(gpu);
        return 1;
    }

    printf("  %zu logits over %d positions | %zu differ, mean|dlogit| %.3g "
           "= %.3g of mean range %.3g (limit %g), worst %.3g\n",
           n, STEPS, n_diff, mean_dev, frac, range, GPU_DEV_FRAC, worst);

    if (frac > GPU_DEV_FRAC) {
        printf("FAIL: CPU and GPU disagree by more than reduction-order "
               "rounding — this is a missing or wrong operation on one side, "
               "not a rounding difference\ngpu-identity: FAILED\n");
        free(cpu); free(gpu);
        return 1;
    }
    printf("gpu-identity: ok\n");
    free(cpu); free(gpu);
    return 0;
}
