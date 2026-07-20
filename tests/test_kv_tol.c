// Phase 8: a SOUND correctness gate for the q8_0 KV cache on the GPU.
//
// The gate this replaces asserted that a free-running greedy generation with
// a q8 cache produced token-identical text on GPU and CPU. That gate cannot
// be made to pass, and — more importantly — it was never measuring what it
// claimed to measure:
//
//   * Free-running generation feeds its own argmax back in. One last-ulp
//     difference at one near-tie changes the next token, which changes every
//     token after it. So the metric is not "how wrong is the kernel" but
//     "how many near-ties did this prompt happen to contain", which is a
//     property of the prompt and the model, not of the code under test.
//   * q8_0's step is amax/127 per 32-value block, roughly 15x coarser than
//     fp16 near the same magnitude. Reassociation noise that fp16 absorbs
//     silently can move a q8 value by a whole quantization level. This is
//     not hypothetical: on gemma-3-4b the same divergence reproduces with no
//     GPU involved at all, CPU-only, purely by changing -b 64 to -b 1.
//
// So token identity under q8 is not a property the implementation can have,
// and a gate that demands it is a gate that will be disabled.
//
// What this test measures instead is TEACHER-FORCED agreement. Every
// configuration is fed the *same* fixed token sequence at every position, so
// no configuration is ever reacting to its own earlier divergence, and the
// difference between two logit vectors at a given position is purely numeric.
//
// The tolerance is not a magic constant, and it is not a guess about how
// close two q8 implementations "ought" to be. It is calibrated against a
// NEGATIVE CONTROL: the CPU q8 path compared against *itself*, run with a
// different prompt batch size.
//
//     reassoc = mean|q8_cpu(b=1) - q8_cpu(b=64)|    no GPU involved at all
//     impl    = mean|q8_gpu      - q8_cpu(b=64)|
//
// Changing the batch size only reassociates the same sums — it is a
// mathematically equivalent computation, so `reassoc` is pure floating-point
// noise passed through the quantizer. It is the floor: no faithful q8
// implementation can do better, because q8's step is ~15x coarser than fp16
// and last-ulp input differences therefore reach the logits amplified.
//
// The gate is that the GPU differs from the CPU by no more than a small
// multiple of what the CPU already differs from itself. That is a property a
// correct implementation genuinely has, it needs no per-model tuning, and a
// layout bug, a wrong scale, or a mis-sliced head cannot satisfy it — those
// produce differences orders of magnitude above the reassociation floor.
//
// mean|q8_cpu - f16_cpu| (the inherent cost of the format) is reported too,
// as context for how large all of these numbers are, but it is deliberately
// NOT the gate: the GPU/CPU gap is a sizeable fraction of it for exactly the
// amplification reason above, and a gate demanding otherwise would be the
// same mistake as demanding token identity, one level down.
//
// The f16 invariant is checked the strict way, unchanged: f16 GPU must be
// token-identical to f16 CPU. That one IS a property the implementation has
// and must keep.
//
//     ./test-kv-tol models/Qwen3-4B-Q4_K_M.gguf
//     ./test-kv-tol models/Qwen3-4B-Q4_K_M.gguf 20   # cap VRAM at 20%
//
// Default model is test.gguf, whose head_dim is not a multiple of 32, so the
// q8 half self-skips there and `make test` still exercises the harness.
#include "runner.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// 192 tokens keeps the toy test.gguf (256 training positions) off the YaRN
// path, so `make test` measures the plain rope. N_BATCH must divide the
// prefill the same way for every configuration, or the comparison would be
// measuring batch reassociation instead of the cache format.
enum { STEPS = 64, MAX_TOK = 192, N_BATCH = 64 };

// How many times the reassociation floor the GPU is allowed to sit at. The
// GPU reorders far more than a batch-size change does (different thread
// counts, different accumulation trees, fused multiply-adds in different
// places), so demanding 1.0x would be wrong; but a real layout or scale bug
// lands orders of magnitude out, not at 3x.
#define REASSOC_SLACK  3.0
// TIE_FRAC: a top-1 disagreement is only excusable if the two candidates were
// within this fraction of the logit range of each other — i.e. a genuine
// near-tie. A disagreement with a decisive margin is a bug no matter what the
// average error looks like.
#define TIE_FRAC       0.02
// and near-ties must stay rare even so
#define DISAGREE_MAX   0.05

static int g_fail = 0;

static void ck(int cond, const char *what) {
    if (!cond) { fprintf(stderr, "FAIL: %s\n", what); g_fail = 1; }
}

// Natural text, not random ids: the near-tie structure of a real logit
// distribution is what production actually sees, and random token ids produce
// a flat confused distribution where everything is a near-tie.
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

// Percent of total VRAM the GPU configs may use (0 = take all free VRAM,
// the normal behavior). Settable so the gate can run on a shared GPU without
// starving whatever else is on it.
static int g_reserve_vram_pct = 0;

typedef struct {
    const char *name;
    bool        kv_q8;
    int         gpu_mode;
    int         n_batch;     // prompt batch size; varying it only reassociates
    bool        available;   // config actually ran as requested
    float      *logits;      // [STEPS][n_vocab], owned
    int32_t    *top1;        // [STEPS], owned
} config;

static model_params params_for(const config *c, int n_ctx) {
    model_params p;
    memset(&p, 0, sizeof(p));
    p.gpu_mode  = c->gpu_mode;
    p.n_threads = 4;
    p.n_ctx     = n_ctx;
    p.n_batch   = c->n_batch;
    p.kv_q8     = c->kv_q8;
    p.reserve_vram_pct = g_reserve_vram_pct;
    return p;
}

static int argmax(const float *v, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) if (v[i] > v[best]) best = i;
    return best;
}

// Second-best value, used to size the margin at a disagreement.
static float top2_gap(const float *v, int n, int best) {
    float second = -INFINITY;
    for (int i = 0; i < n; i++)
        if (i != best && v[i] > second) second = v[i];
    return v[best] - second;
}

static float logit_range(const float *v, int n) {
    float lo = INFINITY, hi = -INFINITY;
    for (int i = 0; i < n; i++) {
        if (v[i] < lo) lo = v[i];
        if (v[i] > hi) hi = v[i];
    }
    return hi - lo;
}

// Teacher-forced sweep: position i always sees the real token i, never this
// configuration's own prediction. That is what makes the per-position logit
// comparison below a comparison of arithmetic rather than of drift.
static bool run_config(config *c, const char *path, const int32_t *toks,
                       int n_tok, int n_vocab) {
    model_t m;
    memset(&m, 0, sizeof(m));
    model_params p = params_for(c, n_tok + 8);
    if (!model_load(&m, path, &p)) {
        fprintf(stderr, "  %-12s load failed\n", c->name);
        return false;
    }
    // Did we get the configuration we asked for? q8 silently falls back to
    // f16 when head_dim is not block-aligned or the backend lacks kernels,
    // and asking for a GPU on a CPU-only build is not an error either.
    bool got_q8  = m.kv_q8;
    bool got_gpu = m.gpu != NULL && m.gpu_layers > 0;

    // Cross-platform safety invariant, checked on every platform including
    // the ones whose backend cannot do q8 at all. A backend that reports no
    // q8 attention kernels must never end up with a q8 cache while it is
    // running layers: those kernels would read q8_0 blocks as fp16 and return
    // fluent, plausible, wrong text — the one failure mode no downstream test
    // would catch. This is the assertion that makes metal.m's
    // gpu_kv_q8_ok()==false a guarantee rather than a stub nobody checks.
    ck(!(got_q8 && got_gpu && !gpu_kv_q8_ok()),
       "a backend without q8 attention kernels never runs a q8 KV cache");

    if (c->kv_q8 != got_q8 || (c->gpu_mode == GPU_AUTO) != got_gpu) {
        fprintf(stderr, "  %-12s skipped (asked kv_q8=%d gpu=%d, got "
                "kv_q8=%d gpu=%d/%d layers)\n", c->name, (int)c->kv_q8,
                c->gpu_mode == GPU_AUTO, (int)got_q8, m.gpu_layers, m.n_layer);
        model_free(&m);
        return false;
    }

    c->logits = malloc(sizeof(float) * (size_t)STEPS * (size_t)n_vocab);
    c->top1   = malloc(sizeof(int32_t) * STEPS);
    if (!c->logits || !c->top1) { model_free(&m); return false; }

    // Prefill in n_batch-sized chunks: model_forward_batch processes at most
    // n_batch tokens per call, and only the final chunk needs logits.
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
        // feed the REAL token at prefill+s-1, not our own argmax
        lg = model_forward(&m, toks[prefill + s - 1], prefill + s - 1);
        if (!lg) { model_free(&m); return false; }
        memcpy(c->logits + (size_t)s * n_vocab, lg,
               sizeof(float) * (size_t)n_vocab);
        c->top1[s] = (int32_t)argmax(lg, n_vocab);
    }
    model_free(&m);

    // Anti-vacuity guard. Every comparison below is a *difference* between two
    // configurations, and differences between two degenerate runs are zero —
    // which reads as a perfect pass. So refuse to draw any conclusion from a
    // configuration that did not actually produce logits.
    double absmax = 0;
    for (size_t i = 0; i < (size_t)STEPS * (size_t)n_vocab; i++) {
        double a = fabs((double)c->logits[i]);
        if (a > absmax) absmax = a;
    }
    if (absmax < 1e-6) {
        fprintf(stderr, "FAIL: %s produced all-zero logits — this "
                "configuration measured nothing\n", c->name);
        g_fail = 1;
        return false;
    }

    c->available = true;
    return true;
}

// mean absolute logit difference over every position and every vocab entry
static double mean_abs_diff(const config *a, const config *b, int n_vocab) {
    double sum = 0;
    size_t n = (size_t)STEPS * (size_t)n_vocab;
    for (size_t i = 0; i < n; i++)
        sum += fabs((double)a->logits[i] - (double)b->logits[i]);
    return sum / (double)n;
}

// how often the two configs' argmax differs, and the worst margin at which
// they disagreed (as a fraction of the logit range at that position)
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

    // Builds the fp16 -> fp32 lookup table. Without it every f16_to_f32 call
    // returns 0, so the whole CPU forward pass — embeddings, KV reads, logits
    // — silently evaluates to zeros, and any test comparing two such runs
    // passes because 0 == 0. The all-zero guard below is what catches this
    // class of mistake rather than trusting the call to stay here.
    f16_init();

    // tokenize the fixed text with the model's own tokenizer
    gguf_file gf;
    if (!gguf_open(&gf, path)) {
        fprintf(stderr, "cannot open %s\n", path);
        return 1;
    }
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
    // A byte-level toy vocab needs far more tokens for this text than a real
    // BPE vocab does; a real vocab may need fewer than MAX_TOK. Tile to a
    // fixed length either way, so every model is gated over the same number
    // of KV rows.
    for (int i = n_tok; i < MAX_TOK; i++) toks[i] = toks[i - n_tok + 1];
    n_tok = MAX_TOK;

    printf("kv-tol: %s | %d tokens, %d teacher-forced positions\n",
           path, n_tok, STEPS);

    enum { N_CFG = 5 };
    config cfgs[N_CFG] = {
        { "f16-cpu",   false, GPU_OFF,  N_BATCH, false, NULL, NULL },
        { "f16-gpu",   false, GPU_AUTO, N_BATCH, false, NULL, NULL },
        { "q8-cpu",    true,  GPU_OFF,  N_BATCH, false, NULL, NULL },
        { "q8-gpu",    true,  GPU_AUTO, N_BATCH, false, NULL, NULL },
        // the negative control: same code, same device, same cache format,
        // only the prompt batching differs, so every difference it shows is
        // pure reassociation noise amplified by the q8 quantizer
        { "q8-cpu-b1", true,  GPU_OFF,  1,       false, NULL, NULL },
    };

    // n_vocab is a property of the file; read it from the first load
    int n_vocab = 0;
    {
        model_t probe;
        memset(&probe, 0, sizeof(probe));
        model_params p = params_for(&cfgs[0], n_tok + 8);
        if (!model_load(&probe, path, &p)) {
            fprintf(stderr, "cannot load %s\n", path);
            return 1;
        }
        n_vocab = probe.n_vocab;
        model_free(&probe);
    }

    for (int i = 0; i < N_CFG; i++)
        run_config(&cfgs[i], path, toks, n_tok, n_vocab);

    config *f16c = &cfgs[0], *f16g = &cfgs[1], *q8c = &cfgs[2],
           *q8g  = &cfgs[3], *q8b1 = &cfgs[4];

    // ---------------------------------------------------------- invariant
    // fp16 GPU is token-identical to fp16 CPU. Strict, and staying strict:
    // this one the implementation genuinely has.
    if (f16c->available && f16g->available) {
        int n_diff;
        double worst;
        top1_stats(f16c, f16g, n_vocab, &n_diff, &worst);
        printf("  f16-cpu vs f16-gpu : top1 diff %d/%d\n", n_diff, STEPS);
        ck(n_diff == 0, "fp16 GPU is token-identical to fp16 CPU "
                        "(teacher-forced)");
    } else {
        printf("  f16-cpu vs f16-gpu : skipped (no GPU)\n");
    }

    // -------------------------------------------------------- q8 tolerance
    if (q8c->available && q8g->available && q8b1->available &&
        f16c->available) {
        double quant_err  = mean_abs_diff(q8c, f16c, n_vocab);  // context only
        double reassoc    = mean_abs_diff(q8b1, q8c, n_vocab);  // the floor
        double impl_err   = mean_abs_diff(q8g, q8c, n_vocab);   // the measure
        double ratio = reassoc > 0 ? impl_err / reassoc : INFINITY;

        int n_diff;
        double worst;
        top1_stats(q8c, q8g, n_vocab, &n_diff, &worst);
        double frac = (double)n_diff / STEPS;

        printf("  q8-cpu    vs f16-cpu : mean|dlogit| %.6f   (what the format "
               "costs; context, not the gate)\n", quant_err);
        printf("  q8-cpu-b1 vs q8-cpu  : mean|dlogit| %.6f   (reassociation "
               "floor, CPU only)\n", reassoc);
        printf("  q8-gpu    vs q8-cpu  : mean|dlogit| %.6f   %.2fx the floor "
               "(limit %.1fx)\n", impl_err, ratio, REASSOC_SLACK);
        printf("  q8-gpu    vs q8-cpu  : top1 diff %d/%d (%.1f%%, limit %.0f%%)"
               ", worst margin %.4f of range (limit %.3f)\n",
               n_diff, STEPS, 100.0 * frac, 100.0 * DISAGREE_MAX,
               worst, TIE_FRAC);

        ck(ratio <= REASSOC_SLACK,
           "q8 GPU differs from q8 CPU no more than q8 CPU differs from "
           "itself under legal reassociation");
        ck(frac <= DISAGREE_MAX,
           "q8 GPU and q8 CPU pick the same token at nearly every position");
        ck(worst <= TIE_FRAC,
           "every q8 GPU/CPU token disagreement is a near-tie, not a decision");
    } else {
        printf("  q8 tolerance gate  : skipped (q8 or GPU unavailable)\n");
    }

    for (int i = 0; i < N_CFG; i++) { free(cfgs[i].logits); free(cfgs[i].top1); }
    printf(g_fail ? "kv-tol: FAILED\n" : "kv-tol: ok\n");
    return g_fail;
}
