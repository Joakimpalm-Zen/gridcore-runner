// The fused-MoE routing tolerance gate (suite plan: "tolerance-form check on
// the shipping fused MoE decode path").
//
// Certification defines MoE byte identity over the EAGER host-routing path
// (RUNNER_MOE_EAGER=1, pinned in the compatibility harnesses since bf93510).
// The shipping default is the fused device-routing path, whose contract is
// deliberately weaker and stated in the compat doc: expert SELECTION identical,
// routing weights within ~2 ulp of the host reference. That contract has been
// checked by hand at the first routing on two boxes — a spot check, not a gate.
//
// This is the gate. Same design as test_kv_tol.c / test_tc_tol.c, because the
// same reasoning applies: two arithmetically different paths cannot be held to
// token identity without producing a gate somebody disables, but they can be
// held to TEACHER-FORCED agreement, where every difference is arithmetic
// rather than accumulated drift.
//
// Configurations (one process, switched through gpu_moe_eager_force):
//   eager   the reference: host routing, the certified path
//   fused   the measure: device routing kernel + fused indirect expert matvecs
//
// The gate:
//   1. top-1 agreement at every teacher-forced position, with the near-tie
//      escape shared with the other tolerance gates: at most DISAGREE_MAX of
//      positions may flip, and every flip must be a genuine near-tie (margin
//      <= TIE_FRAC of the logit range). A decisive-margin flip is a routing
//      bug — selection diverged, not just its weights.
//   2. mean|dlogit| bounded as a fraction of the mean logit range. Routing
//      weights entering at ~2 ulp reach the logits multiplied by expert
//      outputs, so the bound is empirical rather than derived; MOE_DEV_FRAC is
//      set well under the decisive-flip threshold and far above ulp noise.
//
// Skips (not passes) when: no GPU, no MoE model, no full GPU offload, the
// fused path is ineligible for this model, or the two paths are bit-identical.
// Bit-identity means the fused path never engaged, and calling a comparison of
// a path with itself "tolerant" is exactly the vacuity kv_tol warns about.
//
//     ./test-moe-tol models/Qwen3-30B-A3B-Q4_K_M.gguf
//     ./test-moe-tol models/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf 90
//
// Default model is test.gguf (dense F32 toy): the harness runs and self-skips.
#include "runner.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { STEPS = 32, MAX_TOK = 160, N_BATCH = 64 };

#define DISAGREE_MAX  0.05   // flips allowed, as a fraction of STEPS
#define TIE_FRAC      0.02   // a flip must be within this fraction of range
#define MOE_DEV_FRAC  0.005  // mean|dlogit| bound, fraction of mean range

static int g_fail = 0;
static int g_reserve_vram_pct = 0;

static void ck(int cond, const char *what) {
    if (!cond) { fprintf(stderr, "FAIL: %s\n", what); g_fail = 1; }
}

// Same passage as the other tolerance gates, for the same reason: real text
// carries the near-tie structure production sees, random ids make everything
// a near-tie and the escape hatch meaningless.
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
    "be a duplicate entry under a second designation.";

typedef struct {
    const char *name;
    int         eager;       // gpu_moe_eager_force argument for this config
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
    float second = -INFINITY;
    for (int i = 0; i < n; i++)
        if (i != best && v[i] > second) second = v[i];
    return v[best] - second;
}

// gemma-4 suppresses vocabulary entries with a large negative sentinel;
// including it makes the range ~1e30 and every fraction-of-range bound
// vacuously true.
#define SUPPRESSED_BELOW (-1e29f)

static float logit_range(const float *v, int n) {
    float lo = INFINITY, hi = -INFINITY;
    for (int i = 0; i < n; i++) {
        if (v[i] <= SUPPRESSED_BELOW) continue;
        if (v[i] < lo) lo = v[i];
        if (v[i] > hi) hi = v[i];
    }
    return hi > lo ? hi - lo : 0.0f;
}

static bool run_config(config *c, const char *path, const int32_t *toks,
                       int n_tok, int n_vocab) {
    gpu_moe_eager_force(c->eager);

    model_t m;
    memset(&m, 0, sizeof(m));
    model_params p;
    memset(&p, 0, sizeof(p));
    p.gpu_mode  = GPU_AUTO;
    p.n_threads = 4;
    p.n_ctx     = n_tok + 8;
    p.n_batch   = N_BATCH;
    p.reserve_vram_pct = g_reserve_vram_pct;

    if (!model_load(&m, path, &p)) {
        fprintf(stderr, "  %-8s load failed\n", c->name);
        return false;
    }
    if (m.n_expert <= 0) {
        fprintf(stderr, "  %-8s skipped (not a sparse-MoE model)\n", c->name);
        model_free(&m);
        return false;
    }
    // The comparison is between two GPU routing paths; a CPU fallback or a
    // partial split would compare the host path with itself and pass vacuously.
    if (!m.gpu || m.gpu_layers < m.n_layer) {
        fprintf(stderr, "  %-8s skipped (no full GPU offload: %d/%d layers)\n",
                c->name, m.gpu_layers, m.n_layer);
        model_free(&m);
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

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "test.gguf";
    if (argc > 2) g_reserve_vram_pct = atoi(argv[2]);

    f16_init();

    gguf_file gf;
    if (!gguf_open(&gf, path)) {
        fprintf(stderr, "cannot open %s\n", path);
        return 1;
    }
    tokenizer tok;
    if (!tokenizer_init(&tok, &gf)) {
        fprintf(stderr, "cannot init tokenizer\n");
        gguf_close(&gf);
        return 1;
    }
    static int32_t toks[MAX_TOK];
    int n_tok = tok_encode(&tok, TEXT, toks, MAX_TOK, true, false);
    tokenizer_free(&tok);
    gguf_close(&gf);
    if (n_tok < 16) {
        fprintf(stderr, "text tokenized to only %d tokens\n", n_tok);
        return 1;
    }
    // pad by repeating, exactly as the other tolerance gates do, so a
    // byte-vocab toy and a real BPE model both reach MAX_TOK positions
    for (int i = n_tok; i < MAX_TOK; i++) toks[i] = toks[i - n_tok + 1];
    n_tok = MAX_TOK;

    int n_vocab = 0;
    {
        model_t probe;
        memset(&probe, 0, sizeof(probe));
        model_params pp;
        memset(&pp, 0, sizeof(pp));
        pp.gpu_mode = GPU_OFF;
        pp.n_threads = 4;
        pp.n_ctx = n_tok + 8;
        pp.n_batch = N_BATCH;
        if (!model_load(&probe, path, &pp)) {
            fprintf(stderr, "cannot load %s\n", path);
            return 1;
        }
        n_vocab = probe.n_vocab;
        model_free(&probe);
    }

    printf("moe-tol: %s (%d tokens, %d teacher-forced positions)\n",
           path, n_tok, STEPS);

    config eager = { "eager", 1, false, NULL, NULL };
    config fused = { "fused", 0, false, NULL, NULL };
    bool ok_e = run_config(&eager, path, toks, n_tok, n_vocab);
    bool ok_f = ok_e && run_config(&fused, path, toks, n_tok, n_vocab);
    gpu_moe_eager_force(-1);

    if (!ok_e || !ok_f) {
        printf("  moe tolerance gate : skipped (GPU, MoE or offload unavailable)\n"
               "moe-tol: %s\n", g_fail ? "FAILED" : "ok (skipped)");
        free(eager.logits); free(eager.top1);
        free(fused.logits); free(fused.top1);
        return g_fail;
    }

    size_t n = (size_t)STEPS * (size_t)n_vocab;
    if (memcmp(eager.logits, fused.logits, sizeof(float) * n) == 0) {
        printf("  bit-identical to the eager path: the fused router never "
               "engaged for this model; skipping, not passing\n"
               "moe-tol: ok (skipped)\n");
        free(eager.logits); free(eager.top1);
        free(fused.logits); free(fused.top1);
        return 0;
    }

    double sum = 0;
    for (size_t i = 0; i < n; i++)
        sum += fabs((double)eager.logits[i] - (double)fused.logits[i]);
    double dev = sum / (double)n;
    double range = 0;
    for (int s = 0; s < STEPS; s++)
        range += (double)logit_range(eager.logits + (size_t)s * n_vocab, n_vocab);
    range /= STEPS;

    int n_diff = 0;
    double worst = 0.0;
    for (int s = 0; s < STEPS; s++) {
        if (eager.top1[s] == fused.top1[s]) continue;
        n_diff++;
        const float *row = eager.logits + (size_t)s * n_vocab;
        float gap = top2_gap(row, n_vocab, eager.top1[s]);
        float rng = logit_range(row, n_vocab);
        double frac = rng > 0 ? (double)gap / (double)rng : 0.0;
        if (frac > worst) worst = frac;
    }

    double dev_frac = range > 0 ? dev / range : 0.0;
    printf("  mean|dlogit|       : %.3e  (%.3e of the %.3f mean range)\n",
           dev, dev_frac, range);
    printf("  top-1 flips        : %d/%d (worst decisive margin %.4f of range)\n",
           n_diff, STEPS, worst);

    ck(n_diff <= (int)(DISAGREE_MAX * STEPS + 0.5),
       "fused routing flips too many teacher-forced top-1 tokens");
    ck(worst <= TIE_FRAC,
       "a fused-routing top-1 flip had a decisive margin (selection diverged)");
    ck(dev_frac <= MOE_DEV_FRAC,
       "fused-vs-eager mean logit deviation exceeds the bound");

    free(eager.logits); free(eager.top1);
    free(fused.logits); free(fused.top1);
    printf("moe-tol: %s\n", g_fail ? "FAILED" : "ok");
    return g_fail;
}
