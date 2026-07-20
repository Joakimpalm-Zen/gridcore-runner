// Continuous agent batching (Phase 6): the batched decode primitive.
//
// The whole value of batching N sequences into one microbatch is throughput,
// and the whole risk is that it quietly changes the answer. So this test is
// built around one claim and measures the other:
//
//   1. IDENTITY (the gate). N sequences decoded together produce, token for
//      token, exactly what the same N sequences produce decoded one at a time.
//      Checked on the sampled greedy tokens *and* on the raw logits, because
//      argmax hides a lot: two logit vectors can differ in the last mantissa
//      bit for hundreds of steps before the tokens finally diverge, and a test
//      that only looks at tokens would call that passing. A batched decode
//      that changes the numbers is a regression, not a speedup.
//
//   2. THROUGHPUT (the point). The same N sequences, the same number of
//      tokens, timed batched and timed sequentially. Printed, not asserted:
//      the ratio is a property of the GPU and the model, and a test that fails
//      on a slow machine teaches nobody anything.
//
// Sequences are given deliberately *different* prompts and different lengths.
// Equal-length sequences would let a batch that silently shared one position
// or one KV region still pass, which is exactly the bug worth catching.
//
// Default model is the generated test.gguf so `make test` exercises the API
// and the fallback path everywhere; pass a real GGUF to exercise the CUDA
// microbatch:
//
//     ./test-batch models/Qwen3-4B-Q4_K_M.gguf
#include "runner.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// local clock: engine.c owns now_s and this test deliberately does not link it
static double clk(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

static int g_fail = 0;

static void ck(int cond, const char *what) {
    if (!cond) { fprintf(stderr, "FAIL: %s\n", what); g_fail = 1; }
}

enum { NSEQ = 8, STEPS = 24, MAX_PROMPT = 24 };

// Distinct prompts of distinct lengths: sequence i decodes at position
// PLEN[i] + step, so every sequence in a microbatch sits at a different KV
// position on every step. A batch that broadcasts one position, or that lets
// one sequence read another's cache, cannot survive that.
static const int PLEN[NSEQ] = { 5, 11, 8, 17, 6, 14, 9, 20 };

static void seed_prompt(int seq, int32_t *out, int n, int n_vocab) {
    for (int i = 0; i < n; i++)
        out[i] = (int32_t)(((seq + 1) * 7919 + i * 104729) % n_vocab);
}

static int argmax(const float *v, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) if (v[i] > v[best]) best = i;
    return best;
}

static model_params base_params(int n_ctx) {
    model_params p;
    memset(&p, 0, sizeof(p));
    p.gpu_mode  = GPU_AUTO;
    p.n_threads = 4;
    p.n_ctx     = n_ctx;
    p.n_batch   = 32;
    return p;
}

// Prefill sequence i from scratch and return the first token to decode.
// Called before each run so both runs start from an identical KV state.
static int32_t prefill(model_t *m, int seq, int *pos) {
    int32_t toks[MAX_PROMPT];
    seed_prompt(seq, toks, PLEN[seq], m->n_vocab);
    float *lg = model_forward_batch(m, toks, PLEN[seq], 0, true);
    if (!lg) return -1;
    *pos = PLEN[seq];
    return (int32_t)argmax(lg, m->n_vocab);
}

static int run_width(const char *path, int nseq) {
    // room for the longest prompt plus every decoded token, with slack
    model_params p = base_params(MAX_PROMPT + STEPS + 64);

    model_t seqs[NSEQ];
    memset(seqs, 0, sizeof(seqs));
    model_t *sp[NSEQ];
    for (int i = 0; i < nseq; i++) {
        if (!model_load(&seqs[i], path, &p)) {
            fprintf(stderr, "FAIL: load sequence %d from %s\n", i, path);
            for (int j = 0; j < i; j++) model_free(&seqs[j]);
            return 1;
        }
        sp[i] = &seqs[i];
    }
    int n_vocab = seqs[0].n_vocab;

    // ---------------------------------------------------------- sequential
    // The reference. Each sequence decodes against its own model_t exactly as
    // it does today, one model_forward per token; the interleaving is what a
    // server running --parallel N without batching already does.
    static int32_t ref_tok[NSEQ][STEPS];
    static float  *ref_logits[NSEQ][STEPS];
    int32_t cur[NSEQ];
    int     pos[NSEQ];
    for (int i = 0; i < nseq; i++) {
        cur[i] = prefill(sp[i], i, &pos[i]);
        ck(cur[i] >= 0, "sequential prefill");
    }
    double t0 = clk();
    for (int s = 0; s < STEPS; s++)
        for (int i = 0; i < nseq; i++) {
            float *lg = model_forward(sp[i], cur[i], pos[i]);
            if (!lg) { ck(0, "sequential forward"); goto done; }
            ref_logits[i][s] = malloc(sizeof(float) * (size_t)n_vocab);
            if (!ref_logits[i][s]) { ck(0, "oom"); goto done; }
            memcpy(ref_logits[i][s], lg, sizeof(float) * (size_t)n_vocab);
            cur[i] = (int32_t)argmax(lg, n_vocab);
            ref_tok[i][s] = cur[i];
            pos[i]++;
        }
    double seq_s = clk() - t0;

    // ------------------------------------------------------------- batched
    // Same sequences, same prompts, same starting KV — one microbatch per
    // step instead of NSEQ forwards.
    {
        model_batch *b = model_batch_create(sp, nseq);
        if (!b) { ck(0, "model_batch_create"); goto done; }

        for (int i = 0; i < nseq; i++) {
            cur[i] = prefill(sp[i], i, &pos[i]);
            ck(cur[i] >= 0, "batched prefill");
        }
        int idx[NSEQ];
        for (int i = 0; i < nseq; i++) idx[i] = i;

        int diverged = -1;
        double t1 = clk();
        for (int s = 0; s < STEPS; s++) {
            float *out[NSEQ] = { 0 };
            if (!model_batch_decode(b, idx, cur, pos, nseq, out)) {
                ck(0, "model_batch_decode");
                break;
            }
            for (int i = 0; i < nseq; i++) {
                if (!out[i]) { ck(0, "batch produced no logits"); goto batched_done; }
                if (diverged < 0 &&
                    memcmp(out[i], ref_logits[i][s], sizeof(float) * (size_t)n_vocab) != 0)
                    diverged = s;
                cur[i] = (int32_t)argmax(out[i], n_vocab);
                if (cur[i] != ref_tok[i][s]) {
                    fprintf(stderr, "FAIL: seq %d step %d: batched token %d, "
                            "sequential token %d\n", i, s, cur[i], ref_tok[i][s]);
                    g_fail = 1;
                }
                pos[i]++;
            }
        }
    batched_done:;
        double bat_s = clk() - t1;
        model_batch_free(b);

        ck(diverged < 0, "batched logits are bit-identical to sequential");
        if (diverged >= 0)
            fprintf(stderr, "       first bitwise difference at step %d\n", diverged);

        double toks = (double)nseq * STEPS;
        printf("  N=%d  sequential %6.1f tok/s (%5.1f ms/step)   batched %6.1f tok/s"
               " (%5.1f ms/step)   speedup %.2fx\n",
               nseq, toks / seq_s, 1000.0 * seq_s / STEPS,
               toks / bat_s, 1000.0 * bat_s / STEPS, seq_s / bat_s);
    }

done:
    for (int i = 0; i < NSEQ; i++)
        for (int s = 0; s < STEPS; s++) { free(ref_logits[i][s]); ref_logits[i][s] = NULL; }
    for (int i = 0; i < nseq; i++) model_free(&seqs[i]);
    return g_fail;
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "test.gguf";
    // Required before any CPU forward pass: it builds the fp16 -> fp32 table
    // that f16_to_f32 reads. Without it the whole CPU path evaluates to zeros
    // and the identity assertion below passes vacuously, because two all-zero
    // logit vectors are bit-identical. That is exactly what happened here
    // until it was caught: on a GPU box the full offload hid it, and on a
    // CPU-only box this test was asserting nothing at all.
    f16_init();
    printf("batch: %s | microbatch max %d\n", path, model_batch_max());
    // Sweep the width: one sequence must not regress (a batch of one is the
    // path a lightly loaded server spends most of its time on), and the win
    // has to grow with the width or the whole exercise is pointless.
    int widths[] = { 1, 2, 4, 8 };
    for (size_t w = 0; w < sizeof(widths) / sizeof(*widths); w++) {
        if (argc > 2 && atoi(argv[2]) != widths[w]) continue;
        run_width(path, widths[w]);
    }
    printf(g_fail ? "batch: FAILED\n" : "batch: ok\n");
    return g_fail;
}
