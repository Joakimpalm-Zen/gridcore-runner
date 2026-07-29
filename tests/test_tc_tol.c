// The tensor-core GEMM tolerance gate (suite plan "TC promotion + coverage";
// spec docs/specs/2026-07-22-tensor-core-gemm-scope.md §constraints).
//
// The TC prefill GEMM computes with fp16 weight/activation tiles and fp32
// accumulation. That is NOT bit-identical to the scalar-fp32 kernels and never
// will be, so — exactly as with the q8 KV cache (test_kv_tol.c, whose design
// this follows) — a gate demanding free-running token identity would be a gate
// that gets disabled. What a correct TC kernel genuinely has is TEACHER-FORCED
// agreement: feed every configuration the same fixed tokens at every position
// and the per-position logit difference is purely arithmetic, not drift.
//
// Configurations:
//   scalar-b64   the reference: scalar GEMMs, prompt batch 64
//   scalar-b16   the negative control: same scalar kernels, batch 16 — prompt
//                chunking only reassociates the same fp32 sums, so this is
//                legal-reassociation noise, reported as the floor
//   tc-b64       the measure: RUNNER_CUDA_TC path forced on
//
// TC engages at batch>1 only, so the perturbation under test enters through
// the PREFILL (and flows to the gated positions through the KV cache and
// residual stream) — which is exactly how it reaches production decode.
//
// The gate:
//   1. top-1 agreement at every teacher-forced position, with the kv_tol
//      near-tie escape: at most DISAGREE_MAX of positions may flip, and every
//      flip must be a genuine near-tie (margin ≤ TIE_FRAC of the logit range).
//      A flip with a decisive margin is a bug no matter what the averages say.
//   2. mean|dlogit| bounded as a fraction of the mean logit range (TC_DEV_FRAC).
//      Unlike q8-KV, the reassociation floor is NOT the yardstick here: fp16
//      activation rounding (~2^-11 relative) sits orders of magnitude above
//      fp32 reassociation noise (~2^-23), so a ratio-to-floor gate would need
//      a slack constant so large it gated nothing. The floor is still printed
//      as context; the bound is absolute, calibrated on real Q4_K models
//      (dense measured ~1e-3 of range; a layout/scale bug lands orders of
//      magnitude out, and top-1 catches everything decisive anyway).
//
// Skips (not passes) when: no GPU, no Q4_K tensor in the model, or the TC
// logits are bit-identical to scalar — bit-identity means the TC kernel never
// engaged (unresolved symbol, older PTX), and concluding "tolerant" from a
// comparison of a path with itself would be the vacuity kv_tol warns about.
//
//     ./test-tc-tol models/Llama-3.2-3B-Instruct-Q4_K_M.gguf
//     ./test-tc-tol models/Qwen3-30B-A3B-Q4_K_M.gguf 20   # cap VRAM at 20%
//
// Default model is test.gguf (F32 toy): the harness runs and self-skips.
#include "runner.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { STEPS = 64, MAX_TOK = 192, N_BATCH = 64, N_BATCH_CTRL = 16 };

#define DISAGREE_MAX 0.05   // flips allowed, as a fraction of STEPS
#define TIE_FRAC     0.02   // a flip must be within this fraction of range
#define TC_DEV_FRAC  0.005  // mean|dlogit| bound, fraction of mean range

static int g_fail = 0;

static void ck(int cond, const char *what) {
    if (!cond) { fprintf(stderr, "FAIL: %s\n", what); g_fail = 1; }
}

// Same text as test_kv_tol.c and for the same reason: real text has the
// near-tie structure production sees; random ids make everything a near-tie.
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

static int g_reserve_vram_pct = 0;

typedef struct {
    const char *name;
    int         tc;          // gpu_tc_force argument while this config runs
    int         n_batch;
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

// Range over REAL logits only: some archs (gemma4) suppress vocabulary
// entries by writing a large negative sentinel; including it makes the range
// ~1e30 and every fraction-of-range bound vacuously true.
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
    gpu_tc_force(c->tc);

    model_t m;
    memset(&m, 0, sizeof(m));
    model_params p;
    memset(&p, 0, sizeof(p));
    p.gpu_mode  = GPU_AUTO;
    p.n_threads = 4;
    p.n_ctx     = n_tok + 8;
    p.n_batch   = c->n_batch;
    p.reserve_vram_pct = g_reserve_vram_pct;

    if (!model_load(&m, path, &p)) {
        fprintf(stderr, "  %-12s load failed\n", c->name);
        return false;
    }
    // The comparison is between GPU code paths; a CPU fallback would compare
    // the scalar CPU path with itself and pass vacuously.
    if (!m.gpu || m.gpu_layers < m.n_layer) {
        fprintf(stderr, "  %-12s skipped (no full GPU offload: %d/%d layers)\n",
                c->name, m.gpu_layers, m.n_layer);
        model_free(&m);
        return false;
    }

    c->logits = malloc(sizeof(float) * (size_t)STEPS * (size_t)n_vocab);
    c->top1   = malloc(sizeof(int32_t) * STEPS);
    if (!c->logits || !c->top1) { model_free(&m); return false; }

    int prefill = n_tok - STEPS;
    float *lg = NULL;
    for (int off = 0; off < prefill; off += c->n_batch) {
        int n = prefill - off < c->n_batch ? prefill - off : c->n_batch;
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
    if (argc > 2) g_reserve_vram_pct = atoi(argv[2]);

    f16_init();

    gguf_file gf;
    if (!gguf_open(&gf, path)) {
        fprintf(stderr, "cannot open %s\n", path);
        return 1;
    }
    // the TC kernel exists for Q4_K only; without a Q4_K tensor the TC and
    // scalar paths are the same code and the gate would measure nothing
    bool has_q4k = false;
    for (uint64_t i = 0; i < gf.n_tensors; i++)
        if (gf.tensors[i].type == T_Q4_K) has_q4k = true;

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

    printf("tc-tol: %s | %d tokens, %d teacher-forced positions\n",
           path, n_tok, STEPS);
    if (!has_q4k) {
        printf("  skipped: no Q4_K tensor in this model (TC kernel is Q4_K "
               "only)\n" "tc-tol: ok (skipped)\n");
        return 0;
    }

    enum { N_CFG = 3 };
    config cfgs[N_CFG] = {
        { "scalar-b64", 0, N_BATCH,      false, NULL, NULL },
        { "scalar-b16", 0, N_BATCH_CTRL, false, NULL, NULL },
        { "tc-b64",     1, N_BATCH,      false, NULL, NULL },
    };

    int n_vocab = 0;
    {
        model_t probe;
        memset(&probe, 0, sizeof(probe));
        model_params p;
        memset(&p, 0, sizeof(p));
        p.gpu_mode = GPU_OFF;
        p.n_threads = 4;
        p.n_ctx = n_tok + 8;
        p.n_batch = N_BATCH;
        if (!model_load(&probe, path, &p)) {
            fprintf(stderr, "cannot load %s\n", path);
            return 1;
        }
        n_vocab = probe.n_vocab;
        model_free(&probe);
    }

    for (int i = 0; i < N_CFG; i++)
        run_config(&cfgs[i], path, toks, n_tok, n_vocab);
    gpu_tc_force(-1);

    config *ref = &cfgs[0], *ctrl = &cfgs[1], *tc = &cfgs[2];

    if (!ref->available || !tc->available) {
        printf("  tc tolerance gate  : skipped (GPU or config unavailable)\n"
               "tc-tol: %s\n", g_fail ? "FAILED" : "ok (skipped)");
        return g_fail;
    }

    double impl = mean_abs_diff(tc, ref, n_vocab);

    // bit-identity means the TC kernel never launched (unresolved symbol,
    // older PTX): comparing a path with itself must not read as tolerance
    if (impl == 0.0) {
        printf("  tc-b64 vs scalar-b64 : logits BIT-IDENTICAL — the TC path "
               "did not engage; skipping, not passing\n"
               "tc-tol: ok (skipped)\n");
        return g_fail;
    }

    double range = mean_range(ref, n_vocab);
    double frac_of_range = range > 0 ? impl / range : INFINITY;
    if (ctrl->available) {
        double floor_ = mean_abs_diff(ctrl, ref, n_vocab);
        printf("  scalar-b16 vs scalar-b64 : mean|dlogit| %.6f   "
               "(fp32 reassociation floor; context, not the gate)\n", floor_);
    }

    int n_diff;
    double worst;
    top1_stats(ref, tc, n_vocab, &n_diff, &worst);
    double flip_frac = (double)n_diff / STEPS;

    printf("  tc-b64     vs scalar-b64 : mean|dlogit| %.6f = %.5f of mean "
           "logit range %.2f (limit %.3f)\n", impl, frac_of_range, range,
           TC_DEV_FRAC);
    printf("  tc-b64     vs scalar-b64 : top1 diff %d/%d (%.1f%%, limit "
           "%.0f%%), worst margin %.4f of range (limit %.3f)\n",
           n_diff, STEPS, 100.0 * flip_frac, 100.0 * DISAGREE_MAX,
           worst, TIE_FRAC);

    ck(frac_of_range <= TC_DEV_FRAC,
       "TC logits stay within the deviation bound of the scalar path");
    ck(flip_frac <= DISAGREE_MAX,
       "TC and scalar pick the same token at nearly every position");
    ck(worst <= TIE_FRAC,
       "every TC/scalar token disagreement is a near-tie, not a decision");

    for (int i = 0; i < N_CFG; i++) { free(cfgs[i].logits); free(cfgs[i].top1); }
    printf(g_fail ? "tc-tol: FAILED\n" : "tc-tol: ok\n");
    return g_fail;
}
