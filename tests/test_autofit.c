// Memory-placement arithmetic that no fixture can reach: the reservation
// auto-fit (`-c 0` with --reserve-ram/--reserve-vram) and the KV-evicts-weights
// trade note.
//
// This gate exists because the behaviour it covers is unreachable on any
// machine this project is developed on. The auto-fit only runs when a budget is
// tight enough to force the context down, and the per-slot activation head
// alone is 256 MB: on an 8 GB host even a 1% reservation leaves negative room,
// so the branch returns "does not fit" and nothing downstream is exercised. The
// numbers that matter belong to 7B-and-up models on 24 GB cards. Rather than
// pretend a 135M fixture can stand in for that, the arithmetic is fed those
// numbers directly.
//
// The specific regression guarded here is real and is recorded in model.c: the
// reservation is a budget for the SERVER, not for one sequence. Weights are
// uploaded once and shared; the KV cache and activation head are paid PER SLOT.
// Billing both once over-committed a multi-slot server by nearly the slot
// count.
#include <stdio.h>
#include <stdint.h>
#include "model.h"

static int g_fail = 0;
static void ck(int cond, const char *what) {
    if (!cond) { fprintf(stderr, "FAIL: %s\n", what); g_fail = 1; }
    else       { fprintf(stderr, "ok: %s\n", what); }
}

#define HEAD ((uint64_t)MODEL_AUTOFIT_HEAD)

// Whatever the auto-fit returns, actually running that many slots at that
// context must fit inside the budget. This is the whole contract.
static int fits(uint64_t budget, uint64_t weights, uint64_t kv_per_tok,
                int n_seq, long long ctx) {
    return weights + HEAD * (uint64_t)n_seq +
           (uint64_t)ctx * kv_per_tok * (uint64_t)n_seq <= budget;
}

// The case from model.c: Qwen2.5-7B, `--reserve-vram 40 --parallel 4 -c 0` on a
// 24 GiB card. 40% of the card is a 10.15 GB reservation; each slot's cache
// measured 1.88 GB at a 32768 context, which is 57,373 bytes per token. The
// weight figure is the remainder implied by the 12.47 GB that was actually
// allocated, not an independent measurement.
#define Q_BUDGET   10150000000ull
#define Q_WEIGHTS   4950000000ull
#define Q_KV_TOK          57373ull
#define Q_TRAIN            32768

static void test_multislot_is_not_billed_once(void) {
    // The bug: size the context as if one slot existed, then run four.
    long long one = model_autofit_tokens(Q_BUDGET, Q_WEIGHTS, HEAD, Q_KV_TOK, 1);
    ck(one > 0, "the single-slot case fits at all");
    long long buggy = model_autofit_clamp(one, Q_TRAIN);
    ck(!fits(Q_BUDGET, Q_WEIGHTS, Q_KV_TOK, 4, buggy),
       "billing one slot and running four overruns the reservation");

    // The fix: four slots, sized for four slots.
    long long four = model_autofit_tokens(Q_BUDGET, Q_WEIGHTS, HEAD, Q_KV_TOK, 4);
    ck(four > 0, "the four-slot case still fits something");
    ck(four < one, "more slots means a smaller window");
    ck(fits(Q_BUDGET, Q_WEIGHTS, Q_KV_TOK, 4,
            model_autofit_clamp(four, Q_TRAIN)),
       "four slots sized for four slots stay inside the reservation");
}

// The contract must hold across the whole space, not just one anecdote.
static void test_budget_is_never_exceeded(void) {
    const uint64_t budgets[] = { 4ull<<30, 10ull<<30, 24ull<<30, 80ull<<30 };
    const uint64_t weights[] = { 1ull<<30, 4ull<<30, 13ull<<30 };
    const uint64_t kvs[]     = { 16384, 57373, 262144 };
    const int      seqs[]    = { 1, 2, 4, 8 };
    int checked = 0, bad = 0;
    for (unsigned b = 0; b < sizeof budgets / sizeof *budgets; b++)
    for (unsigned w = 0; w < sizeof weights / sizeof *weights; w++)
    for (unsigned k = 0; k < sizeof kvs / sizeof *kvs; k++)
    for (unsigned s = 0; s < sizeof seqs / sizeof *seqs; s++) {
        long long fit = model_autofit_tokens(budgets[b], weights[w], HEAD,
                                             kvs[k], seqs[s]);
        if (fit <= 0) continue;            // "does not fit" is a valid answer
        checked++;
        if (!fits(budgets[b], weights[w], kvs[k], seqs[s], fit)) bad++;
    }
    ck(checked > 0, "the sweep reaches the budget-limited regime");
    ck(bad == 0, "no combination is ever sized past its budget");
}

static void test_degenerate_inputs(void) {
    // weights alone blow the budget: no context fits, and it must say so
    // rather than divide its way to a negative or wrap around
    ck(model_autofit_tokens(1ull<<30, 8ull<<30, HEAD, 57373, 1) == 0,
       "a budget smaller than the weights fits nothing");
    // a model with no KV rows of its own must not divide by zero
    ck(model_autofit_tokens(24ull<<30, 4ull<<30, HEAD, 0, 1) == 0,
       "zero KV bytes per token is not a division by zero");
    ck(model_autofit_tokens(24ull<<30, 4ull<<30, HEAD, 57373, 0) ==
       model_autofit_tokens(24ull<<30, 4ull<<30, HEAD, 57373, 1),
       "a slot count below one is treated as one");
}

static void test_clamp(void) {
    ck(model_autofit_clamp(1000000, 8192) == 8192,
       "the trained context is a ceiling");
    ck(model_autofit_clamp(4096, 8192) == 4096,
       "a fitting context is used as-is");
    ck(model_autofit_clamp(10, 8192) == 512,
       "a tiny fit is floored at a usable window");
    ck(model_autofit_clamp(100, 256) == 512,
       "the floor wins over a very small trained context");
}

// The KV-evicts-weights trade note. Same class of blindness: it fires only on a
// partial GPU split that the KV cache caused, which needs a model bigger than
// the card. It is observable in the Blackwell stress-context artifacts under
// tests/compatibility/out/ but nothing gated the threshold, so a regression
// would have shown up as the note quietly never appearing again.
static void test_kv_trade_note(void) {
    const uint64_t GB = 1000000000ull;

    // the case it exists for: 3.36 GB of KV, 7 of 32 layers displaced
    ck(model_kv_trade_note(25, 32, 3360 * (GB / 1000), 6 * GB),
       "a KV cache dominating the device earns the note");

    // a full offload traded nothing away
    ck(!model_kv_trade_note(32, 32, 3360 * (GB / 1000), 6 * GB),
       "a fully offloaded model gets no note");
    // and neither did a load that put nothing on the device
    ck(!model_kv_trade_note(0, 32, 3360 * (GB / 1000), 6 * GB),
       "a CPU-only load gets no note");

    // a split forced by the model's own size is not a trade -c can take back,
    // so a small KV against big weights must stay quiet
    ck(!model_kv_trade_note(25, 32, 1 * GB, 40 * GB),
       "a model simply too big for the card gets no note");

    // the threshold is strict: exactly a quarter is not "dominating"
    ck(!model_kv_trade_note(25, 32, 5 * GB, 20 * GB),
       "the note needs more than a quarter, not exactly a quarter");
    ck(model_kv_trade_note(25, 32, 5 * GB + 1, 20 * GB),
       "one byte past the threshold does earn it");

    // unknown sizes must not be reported as a trade
    ck(!model_kv_trade_note(25, 32, 0, 20 * GB),
       "an unmeasured KV size gets no note");
    ck(!model_kv_trade_note(25, 32, 5 * GB, 0),
       "an unmeasured weight size gets no note");
}

int main(void) {
    test_multislot_is_not_billed_once();
    test_budget_is_never_exceeded();
    test_degenerate_inputs();
    test_clamp();
    test_kv_trade_note();
    if (g_fail) { fprintf(stderr, "test-autofit FAILED\n"); return 1; }
    fprintf(stderr, "test-autofit: all checks passed\n");
    return 0;
}
