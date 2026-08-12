// The fused int8 CPU dot tolerance gate (suite plan
// "blackwell-goal-2026-08-13-opus-performance" Phase 1; the doctrine section
// settles why this is a tolerance gate and not an identity gate).
//
// vec_dot_i8 quantizes the ACTIVATIONS to int8 per 32-element block. Unlike
// the SIMD f32 kernels — which only reassociate the same fp32 sums and stay
// token-identical — this route changes the arithmetic, so a gate demanding
// bit-identity would be a gate that gets disabled. What a correct fused
// kernel genuinely has is TEACHER-FORCED agreement: feed both configurations
// the same fixed tokens at every position and the per-position logit
// difference is activation-rounding, not drift.
//
// This follows test_tc_tol.c (which follows test_kv_tol.c) with two
// differences that come from where the route lives:
//
//   * it is CPU-only and single-column: the fused route runs at n_batch == 1,
//     so the PREFILL of every configuration is bit-identical and only the
//     teacher-forced decode positions can move. That is exactly how the route
//     reaches production — decode is the only place it is used.
//   * the top-1 bar is ZERO flips, not a near-tie allowance. The promotion bar
//     the plan sets for this lever is 0/64, matching what the TC prefill path
//     had to clear before it was defaulted on. A combo that cannot hold it is
//     not promoted; it is not a combo whose bar gets widened.
//
// Configurations:
//   scalar    the reference: RUNNER_CPU_I8 forced off
//   scalar-t7 the negative control: scalar again at a different thread count.
//             Per-row partitioning makes the runner thread-count-invariant, so
//             this control is expected to read EXACTLY zero — it is here to
//             prove the harness can see a difference at all when one exists,
//             and to catch a partitioning regression for free.
//   i8        the measure: the fused route forced on
//
//     ./test-i8-tol models/Llama-3.2-3B-Instruct-Q4_K_M.gguf
//     ./test-i8-tol models/granite-4.1-8b-Q4_0/granite-4.1-8b-Q4_0.gguf 16
//
// Default model is test.gguf (F32 toy): the harness runs and self-skips,
// because F32 has no fused kernel.
#include "runner.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { STEPS = 64, MAX_TOK = 192, N_BATCH = 64 };

#define DISAGREE_MAX 0        // promotion bar: zero top-1 flips in STEPS
#define I8_DEV_FRAC  0.005    // mean|dlogit| bound, fraction of mean range

static int g_fail = 0;

static void ck(int cond, const char *what) {
    if (!cond) { fprintf(stderr, "FAIL: %s\n", what); g_fail = 1; }
}

// Same text as test_tc_tol.c / test_kv_tol.c and for the same reason: real
// text has the near-tie structure production sees; random ids make everything
// a near-tie.
static const char *TEXT =
    "The city of Lisbon sits on seven hills above the Tagus estuary, and its "
    "oldest quarter survived the 1755 earthquake largely intact because the "
    "bedrock there is firmer than the reclaimed ground downriver. Rebuilding "
    "the lower town took decades, and the grid of streets laid out afterwards "
    "was among the first in Europe designed with seismic loads in mind. "
    "def parse_header(buf, size):\n"
    "    if size < 8:\n"
    "        raise ValueError('short header')\n"
    "    magic, version = struct.unpack('<II', buf[:8])\n"
    "    return magic, version\n"
    "In 1929 the observatory published a revised catalogue listing 4218 "
    "objects, of which roughly one in nine turned out on later inspection to "
    "be a duplicate entry under a second designation. The correction was not "
    "issued until 1934, by which time three separate groups had independently "
    "noticed the discrepancy and written to the editors about it.";

typedef struct {
    const char *name;
    int         i8;          // i8_dot_force argument while this config runs
    int         n_threads;
    bool        available;
    float      *logits;      // [STEPS][n_vocab], owned
    int32_t    *top1;        // [STEPS], owned
} config;

static int argmax(const float *v, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) if (v[i] > v[best]) best = i;
    return best;
}

static float top2_gap(const float *v, int n, int best) {
    float second = -FLT_MAX;
    for (int i = 0; i < n; i++)
        if (i != best && v[i] > second) second = v[i];
    return v[best] - second;
}

// Range over REAL logits only: some archs suppress vocabulary entries with a
// large negative sentinel, which would make every fraction-of-range bound
// vacuously true.
#define SUPPRESSED_BELOW (-1e29f)

static float logit_range(const float *v, int n) {
    float lo = FLT_MAX, hi = -FLT_MAX;
    for (int i = 0; i < n; i++) {
        if (v[i] <= SUPPRESSED_BELOW) continue;
        if (v[i] < lo) lo = v[i];
        if (v[i] > hi) hi = v[i];
    }
    return hi > lo ? hi - lo : 0.0f;
}

static bool run_config(config *c, const char *path, const int32_t *toks,
                       int n_tok, int n_vocab) {
    i8_dot_force(c->i8);

    model_t m;
    memset(&m, 0, sizeof(m));
    model_params p;
    memset(&p, 0, sizeof(p));
    p.gpu_mode  = GPU_OFF;        // this is a CPU route; the GPU would hide it
    p.n_threads = c->n_threads;
    p.n_ctx     = n_tok + 8;
    p.n_batch   = N_BATCH;

    if (!model_load(&m, path, &p)) {
        fprintf(stderr, "  %-10s load failed\n", c->name);
        return false;
    }

    c->logits = malloc(sizeof(float) * (size_t)STEPS * (size_t)n_vocab);
    c->top1   = malloc(sizeof(int32_t) * STEPS);
    if (!c->logits || !c->top1) { model_free(&m); return false; }

    int prefill = n_tok - STEPS;
    float *lg = NULL;
    for (int off = 0; off < prefill; off += N_BATCH) {
        int n = prefill - off < N_BATCH ? prefill - off : N_BATCH;
        lg = model_forward_batch(&m, toks + off, n, off, off + n == prefill);
        if (off + n == prefill && !lg) { model_free(&m); return false; }
    }
    if (!lg) { model_free(&m); return false; }
    memcpy(c->logits, lg, sizeof(float) * (size_t)n_vocab);
    c->top1[0] = (int32_t)argmax(lg, n_vocab);

    for (int s = 1; s < STEPS; s++) {
        // feed the REAL token, never this configuration's own prediction
        lg = model_forward(&m, toks[prefill + s - 1], prefill + s - 1);
        if (!lg) { model_free(&m); return false; }
        memcpy(c->logits + (size_t)s * n_vocab, lg,
               sizeof(float) * (size_t)n_vocab);
        c->top1[s] = (int32_t)argmax(lg, n_vocab);
    }
    model_free(&m);

    // anti-vacuity: a run that produced no logits proves nothing
    double absmax = 0;
    for (size_t i = 0; i < (size_t)STEPS * (size_t)n_vocab; i++) {
        double a = fabs((double)c->logits[i]);
        if (a > absmax) absmax = a;
    }
    if (absmax < 1e-6) {
        fprintf(stderr, "FAIL: %s produced all-zero logits\n", c->name);
        g_fail = 1;
        return false;
    }
    c->available = true;
    return true;
}

static double mean_abs_diff(const config *a, const config *b, int n_vocab) {
    double sum = 0;
    size_t n = (size_t)STEPS * (size_t)n_vocab;
    for (size_t i = 0; i < n; i++)
        sum += fabs((double)a->logits[i] - (double)b->logits[i]);
    return sum / (double)n;
}

static double mean_range(const config *a, int n_vocab) {
    double sum = 0;
    for (int s = 0; s < STEPS; s++)
        sum += (double)logit_range(a->logits + (size_t)s * n_vocab, n_vocab);
    return sum / STEPS;
}

static void top1_stats(const config *a, const config *b, int n_vocab,
                       int *n_diff, double *worst_frac) {
    *n_diff = 0;
    *worst_frac = 0.0;
    for (int s = 0; s < STEPS; s++) {
        if (a->top1[s] == b->top1[s]) continue;
        (*n_diff)++;
        const float *row = a->logits + (size_t)s * n_vocab;
        float gap   = top2_gap(row, n_vocab, a->top1[s]);
        float range = logit_range(row, n_vocab);
        double frac = range > 0 ? (double)gap / (double)range : 0.0;
        if (frac > *worst_frac) *worst_frac = frac;
    }
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "test.gguf";
    int n_threads = argc > 2 ? atoi(argv[2]) : 8;

    f16_init();

    gguf_file gf;
    if (!gguf_open(&gf, path)) {
        fprintf(stderr, "cannot open %s\n", path);
        return 1;
    }
    // The fused kernels exist for Q4_K, Q8_0 and Q4_0. Without one of those
    // types the two configurations run the same code and the gate measures
    // nothing.
    bool has_i8_type = false;
    for (uint64_t i = 0; i < gf.n_tensors; i++)
        if (gf.tensors[i].type == T_Q4_K || gf.tensors[i].type == T_Q8_0 ||
            gf.tensors[i].type == T_Q4_0) has_i8_type = true;

    tokenizer tk;
    if (!tokenizer_init(&tk, &gf)) {
        fprintf(stderr, "cannot init tokenizer for %s\n", path);
        gguf_close(&gf);
        return 1;
    }
    static int32_t toks[MAX_TOK];
    int n_tok = tok_encode(&tk, TEXT, toks, MAX_TOK, true, false);
    tokenizer_free(&tk);
    gguf_close(&gf);

    if (n_tok < 16) {
        fprintf(stderr, "text tokenized to only %d tokens\n", n_tok);
        return 1;
    }
    for (int i = n_tok; i < MAX_TOK; i++) toks[i] = toks[i - n_tok + 1];
    n_tok = MAX_TOK;

    printf("i8-tol: %s | %d tokens, %d teacher-forced positions, %d threads\n",
           path, n_tok, STEPS, n_threads);
    if (!has_i8_type) {
        printf("  skipped: no fused-capable tensor in this model (kernels: "
               "Q4_K/Q8_0/Q4_0)\n" "i8-tol: ok (skipped)\n");
        return 0;
    }

    enum { N_CFG = 3 };
    config cfgs[N_CFG] = {
        { "scalar",    0, n_threads,     false, NULL, NULL },
        { "scalar-t7", 0, 7,             false, NULL, NULL },
        { "i8",        1, n_threads,     false, NULL, NULL },
    };

    int n_vocab = 0;
    {
        model_t probe;
        memset(&probe, 0, sizeof(probe));
        model_params p;
        memset(&p, 0, sizeof(p));
        p.gpu_mode = GPU_OFF;
        p.n_threads = n_threads;
        p.n_ctx = n_tok + 8;
        p.n_batch = N_BATCH;
        if (!model_load(&probe, path, &p)) {
            fprintf(stderr, "cannot load %s\n", path);
            return 1;
        }
        n_vocab = probe.n_vocab;
        model_free(&probe);
    }

    unsigned long before = i8_dot_dispatches();
    for (int i = 0; i < N_CFG; i++)
        run_config(&cfgs[i], path, toks, n_tok, n_vocab);
    unsigned long fired = i8_dot_dispatches() - before;
    i8_dot_force(-1);

    config *ref = &cfgs[0], *ctrl = &cfgs[1], *i8 = &cfgs[2];

    if (!ref->available || !i8->available) {
        printf("  i8 tolerance gate  : skipped (config unavailable)\n"
               "i8-tol: %s\n", g_fail ? "FAILED" : "ok (skipped)");
        return g_fail;
    }

    if (fired == 0) {
        printf("  i8 vs scalar : the fused route never dispatched — this "
               "build has no kernel for these tensors. Skipping, not "
               "passing.\ni8-tol: ok (skipped)\n");
        return g_fail;
    }

    if (ctrl->available) {
        double floor_ = mean_abs_diff(ctrl, ref, n_vocab);
        int cd; double cw;
        top1_stats(ref, ctrl, n_vocab, &cd, &cw);
        printf("  scalar-t7 vs scalar : mean|dlogit| %.9f, top1 diff %d/%d  "
               "(thread-count control; per-row partitioning makes this exact)\n",
               floor_, cd, STEPS);
        ck(floor_ == 0.0 && cd == 0,
           "the scalar route is thread-count invariant (identity contract)");
    }

    double impl = mean_abs_diff(i8, ref, n_vocab);
    double range = mean_range(ref, n_vocab);
    double frac_of_range = range > 0 ? impl / range : DBL_MAX;

    int n_diff;
    double worst;
    top1_stats(ref, i8, n_vocab, &n_diff, &worst);

    printf("  i8        vs scalar : %lu fused matvecs, mean|dlogit| %.6f = "
           "%.5f of mean logit range %.2f (limit %.3f)\n",
           fired, impl, frac_of_range, range, I8_DEV_FRAC);
    printf("  i8        vs scalar : top1 diff %d/%d (limit %d), worst flip "
           "margin %.4f of range\n", n_diff, STEPS, DISAGREE_MAX, worst);

    ck(frac_of_range <= I8_DEV_FRAC,
       "fused-int8 logits stay within the deviation bound of the scalar path");
    ck(n_diff <= DISAGREE_MAX,
       "this (format, model) combo is NOT promotable: fused int8 and scalar "
       "disagree on at least one teacher-forced token");

    for (int i = 0; i < N_CFG; i++) { free(cfgs[i].logits); free(cfgs[i].top1); }
    printf(g_fail ? "i8-tol: FAILED\n" : "i8-tol: ok\n");
    return g_fail;
}
