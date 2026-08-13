// The cooperative-KV attention read (arc B; the fix implied by
// docs/metal-decode-kv-traffic-2026-08-15.md).
//
// That measurement found decode's KV read running at 1.52 GB/s, ~18x worse per
// byte than the weight read in the same forward, because k_attn gives each
// THREAD a whole KV row: adjacent lanes address memory row_b apart -- 1024 B
// on e2b -- so one simdgroup load touches 32 distinct cache lines.
//
// k_attn_coop gives one SIMDGROUP a row and splits head_dim across its lanes,
// so a load covers 32 consecutive elements. The per-row dot then finishes with
// a simd_sum over 32 lane partials, which is NOT the sequential accumulation
// k_attn performs -- so the route is not bit-identical by construction and a
// gate demanding identity would be a gate that gets disabled.
//
// This follows test_mv_tol.c exactly, including its determinism control and
// its zero-flip promotion bar: a combo that cannot hold 0/64 is not promoted,
// and the bar is not widened to fit a lever.
//
//     ./test-attn-tol models/e2b-q40.gguf
//     ./test-attn-tol models/SmolLM2-135M-Instruct-Q8_0.gguf
//
// Default model is test.gguf (F32 toy): it runs and self-skips only if the
// route never dispatches.
#include "runner.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { STEPS = 64, MAX_TOK = 192, N_BATCH = 64 };

#define DISAGREE_MAX 0        // promotion bar: zero top-1 flips in STEPS
#define COOP_DEV_FRAC 0.005    // mean|dlogit| bound, fraction of mean range

static int g_fail = 0;
static int g_gpu_layers = 0;

static void ck(int cond, const char *what) {
    if (!cond) { fprintf(stderr, "FAIL: %s\n", what); g_fail = 1; }
}

// Same text as test_i8_tol.c / test_tc_tol.c and for the same reason: real
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
    int         mv;          // gpu_attn_coop_force argument while this config runs
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
    gpu_attn_coop_force(c->mv);

    model_t m;
    memset(&m, 0, sizeof(m));
    model_params p;
    memset(&p, 0, sizeof(p));
    p.gpu_mode = GPU_AUTO;          // this is a GPU route; the CPU would hide it
    p.n_ctx    = n_tok + 8;
    p.n_batch  = N_BATCH;
    p.gpu_layers_override = g_gpu_layers;

    if (!model_load(&m, path, &p)) {
        fprintf(stderr, "  %-12s load failed\n", c->name);
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
    if (argc > 2) g_gpu_layers = atoi(argv[2]);

    f16_init();

    gguf_file gf;
    if (!gguf_open(&gf, path)) {
        fprintf(stderr, "cannot open %s\n", path);
        return 1;
    }
    // The fast kernels exist for Q4_0 and Q8_0. Without one of those types the
    // two configurations run the same code and the gate measures nothing.
    bool has_mv_type = true;   // attention runs for every arch

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

    printf("attn-tol: %s | %d tokens, %d teacher-forced positions\n",
           path, n_tok, STEPS);
    if (!has_mv_type) {
        printf("  skipped: no fast-capable tensor in this model (kernels: "
               "Q4_0/Q8_0)\n" "attn-tol: ok (skipped)\n");
        return 0;
    }

    enum { N_CFG = 3 };
    config cfgs[N_CFG] = {
        { "scalar",       0, false, NULL, NULL },
        { "scalar-rerun", 0, false, NULL, NULL },
        { "attn-coop",    1, false, NULL, NULL },
    };

    int n_vocab = 0;
    {
        model_t probe;
        memset(&probe, 0, sizeof(probe));
        model_params p;
        memset(&p, 0, sizeof(p));
        p.gpu_mode = GPU_AUTO;
        p.n_ctx = n_tok + 8;
        p.n_batch = N_BATCH;
        p.gpu_layers_override = g_gpu_layers;
        if (!model_load(&probe, path, &p)) {
            fprintf(stderr, "cannot load %s\n", path);
            return 1;
        }
        n_vocab = probe.n_vocab;
        model_free(&probe);
    }

    unsigned long before = gpu_attn_coop_dispatches();
    for (int i = 0; i < N_CFG; i++)
        run_config(&cfgs[i], path, toks, n_tok, n_vocab);
    unsigned long fired = gpu_attn_coop_dispatches() - before;
    gpu_attn_coop_force(-1);

    config *ref = &cfgs[0], *ctrl = &cfgs[1], *fast = &cfgs[2];

    if (!ref->available || !fast->available) {
        printf("  mv tolerance gate  : skipped (config unavailable)\n"
               "attn-tol: %s\n", g_fail ? "FAILED" : "ok (skipped)");
        return g_fail;
    }

    if (fired == 0) {
        printf("  mv-fast vs scalar : the fast route never dispatched — no "
               "GPU, no offloaded tensor of a fast-capable type, or a CPU "
               "fallback. Skipping, not passing.\nattn-tol: ok (skipped)\n");
        return g_fail;
    }

    if (ctrl->available) {
        double floor_ = mean_abs_diff(ctrl, ref, n_vocab);
        int cd; double cw;
        top1_stats(ref, ctrl, n_vocab, &cd, &cw);
        printf("  scalar-rerun vs scalar : mean|dlogit| %.9f, top1 diff %d/%d  "
               "(determinism control; the GPU decode route is reproducible)\n",
               floor_, cd, STEPS);
        ck(floor_ == 0.0 && cd == 0,
           "the scalar GPU decode route is run-to-run deterministic");
    }

    double impl = mean_abs_diff(fast, ref, n_vocab);
    double range = mean_range(ref, n_vocab);
    double frac_of_range = range > 0 ? impl / range : DBL_MAX;

    int n_diff;
    double worst;
    top1_stats(ref, fast, n_vocab, &n_diff, &worst);

    printf("  attn-coop vs scalar : %lu fast matvecs, mean|dlogit| %.6f = "
           "%.5f of mean logit range %.2f (limit %.3f)\n",
           fired, impl, frac_of_range, range, COOP_DEV_FRAC);
    printf("  attn-coop vs scalar : top1 diff %d/%d (limit %d), worst flip "
           "margin %.4f of range\n", n_diff, STEPS, DISAGREE_MAX, worst);

    ck(frac_of_range <= COOP_DEV_FRAC,
       "fast-matvec logits stay within the deviation bound of the scalar path");
    ck(n_diff <= DISAGREE_MAX,
       "this (format, model) combo is NOT promotable: the fast matvec and the "
       "scalar route disagree on at least one teacher-forced token");

    for (int i = 0; i < N_CFG; i++) { free(cfgs[i].logits); free(cfgs[i].top1); }
    printf(g_fail ? "attn-tol: FAILED\n" : "attn-tol: ok\n");
    return g_fail;
}
