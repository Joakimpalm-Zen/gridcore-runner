// Sampler presets and the sampling contract.
//
// Everything here is model-free on purpose: preset selection reads only the
// GGUF's `general.architecture` and `general.name` strings, and sample_pick
// takes a plain logits array, so the whole surface is testable without
// loading weights.
#include "runner.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#define EQ(a, b) (fabsf((a) - (b)) < 1e-6f)

// The names below are the real `general.name` values read out of the GGUFs in
// models/ — matching is only worth testing against strings that ship.
static void test_preset_selection(void) {
    const sampler_preset *p;

    p = sampler_preset_for("qwen3", "Qwen3 4B Instruct Awq");
    assert(!strcmp(p->name, "qwen3"));
    assert(EQ(p->temp, 0.6f) && EQ(p->top_p, 0.95f) && p->top_k == 20);
    assert(EQ(p->min_p, 0.0f));

    p = sampler_preset_for("qwen2", "Qwen2.5 32B Instruct");
    assert(!strcmp(p->name, "qwen2.5"));
    assert(EQ(p->temp, 0.7f) && EQ(p->top_p, 0.8f) && p->top_k == 20);
    assert(EQ(p->repeat_penalty, 1.05f));

    p = sampler_preset_for("gemma3", "Gemma 3 4b It");
    assert(!strcmp(p->name, "gemma3"));
    assert(EQ(p->temp, 1.0f) && EQ(p->top_p, 0.95f) && p->top_k == 64);

    p = sampler_preset_for("phi3", "Phi 3.5 Mini Instruct");
    assert(!strcmp(p->name, "phi3"));
    assert(EQ(p->temp, 0.0f));

    // three families share `general.architecture: llama`, so the name is the
    // only thing that separates them
    p = sampler_preset_for("llama", "Llama 3.2 3B Instruct");
    assert(!strcmp(p->name, "llama3"));
    assert(EQ(p->temp, 0.6f) && EQ(p->top_p, 0.9f));
    assert(!strcmp(sampler_preset_for("llama", "Meta Llama 3.1 8B Instruct")->name,
                   "llama3"));

    p = sampler_preset_for("llama", "Mistral-7B-Instruct-v0.3");
    assert(!strcmp(p->name, "mistral"));
    assert(EQ(p->temp, 0.7f) && EQ(p->top_p, 1.0f));

    p = sampler_preset_for("llama", "Smollm2 1.7B 8k Mix7 Ep2 v2");
    assert(!strcmp(p->name, "smollm2"));
    assert(EQ(p->temp, 0.2f) && EQ(p->top_p, 0.9f));

    // Gridcore Syntetik declares arch "llama", name "gridcore-<size>"; the
    // suite-native contract compiler resolves to a deterministic preset, not
    // a vendor one, and greedy is the point (temp 0, no penalty).
    p = sampler_preset_for("llama", "gridcore-10m");
    assert(!strcmp(p->name, "gridcore"));
    assert(EQ(p->temp, 0.0f) && EQ(p->repeat_penalty, 1.0f) && p->top_k == 0);
    assert(!strcmp(sampler_preset_for("llama", "Syntetik 350M")->name, "gridcore"));
}

// A model nobody published numbers for must not silently borrow another
// family's: it falls back to the generic preset, which is what runner shipped
// as its fixed defaults before presets existed.
static void test_preset_fallback(void) {
    const sampler_preset *g = sampler_preset_for("llama", "tinyllama_tinyllama-1.1b-chat-v1.0");
    assert(!strcmp(g->name, "generic"));
    assert(EQ(g->temp, 0.8f) && EQ(g->top_p, 0.95f) && EQ(g->min_p, 0.05f));
    assert(g->top_k == 40 && EQ(g->repeat_penalty, 1.1f));

    // "tinyllama" must not be read as the llama-3 family
    assert(strcmp(sampler_preset_for("llama", "TinyLlama 1.1B")->name, "llama3") != 0);
    // missing or unknown metadata is not an error
    assert(!strcmp(sampler_preset_for(NULL, NULL)->name, "generic"));
    assert(!strcmp(sampler_preset_for("mamba", "")->name, "generic"));
}

static void test_preset_table_is_enumerable(void) {
    int n = 0;
    for (const sampler_preset *p = sampler_preset_at(0); p; p = sampler_preset_at(++n)) {
        assert(p->name && p->name[0]);
        assert(p->source && p->source[0]);   // every preset states where it came from
        assert(p->temp >= 0.0f && p->top_p > 0.0f && p->repeat_penalty > 0.0f);
    }
    assert(n >= 8);
    assert(sampler_preset_at(n) == NULL);
}

static void test_override_precedence(void) {
    sampler s = { .rng = 12345 };
    sampler_override none = {0};

    const sampler_preset *p = sampler_resolve(&s, "qwen3", "Qwen3 4B", &none);
    assert(!strcmp(p->name, "qwen3"));
    assert(EQ(s.temp, 0.6f) && EQ(s.top_p, 0.95f) && s.top_k == 20);
    assert(s.rng == 12345);  // resolution never disturbs sampler state

    // an explicit value always wins, and only that field moves
    sampler_override ov = { .has_temp = true, .temp = 0.05f,
                            .has_repeat_penalty = true, .repeat_penalty = 1.0f };
    p = sampler_resolve(&s, "qwen3", "Qwen3 4B", &ov);
    assert(EQ(s.temp, 0.05f) && EQ(s.repeat_penalty, 1.0f));
    assert(EQ(s.top_p, 0.95f) && s.top_k == 20);

    // overrides win over the generic preset too, and a zero override is a
    // real value rather than "unset"
    sampler_override zero = { .has_top_k = true, .top_k = 0,
                              .has_min_p = true, .min_p = 0.0f };
    p = sampler_resolve(&s, "llama", "tinyllama", &zero);
    assert(!strcmp(p->name, "generic"));
    assert(s.top_k == 0 && EQ(s.min_p, 0.0f) && EQ(s.temp, 0.8f));

    // resolving twice from the same overrides is idempotent — the server
    // re-resolves on every model swap
    sampler again = s;
    sampler_resolve(&s, "llama", "tinyllama", &zero);
    assert(!memcmp(&again, &s, sizeof s));
}

static void test_describe(void) {
    sampler s = {0};
    sampler_override none = {0};
    const sampler_preset *p = sampler_resolve(&s, "gemma3", "Gemma 3 4b It", &none);
    char buf[256];
    sampler_describe(&s, p, buf, sizeof buf);
    assert(strstr(buf, "gemma3"));
    assert(strstr(buf, "top_k 64"));
    assert(strstr(buf, "temp 1.00"));
    // a short buffer must truncate, not overrun
    char tiny[8];
    sampler_describe(&s, p, tiny, sizeof tiny);
    assert(strlen(tiny) < sizeof tiny);
}

// The contract: `temp <= 0` returns the model's argmax. The repeat penalty is
// a diversity knob and has no meaning when the caller asked for the single
// most likely token, so greedy decoding does not apply it. Before this, a
// repeated token could be pushed below its rivals and `--temp 0` returned
// something that was not the argmax at all.
static void test_greedy_ignores_repeat_penalty(void) {
    sampler s = { .temp = 0.0f, .repeat_penalty = 2.0f, .top_p = 1.0f };
    sampler_reset(&s);
    float logits[4] = { 10.0f, 9.0f, 1.0f, 0.5f };

    assert(sample_pick(&s, logits, 4, NULL, NULL) == 0);
    sampler_accept(&s, 0);   // token 0 is now in the penalty window
    float again[4] = { 10.0f, 9.0f, 1.0f, 0.5f };
    assert(sample_pick(&s, again, 4, NULL, NULL) == 0);
    // and the logits themselves are left untouched for the caller
    assert(EQ(again[0], 10.0f));
}

// ...but sampling still penalises. Same window, temp above zero: with the
// penalty halving token 0's logit it drops below token 1, and a temperature
// low enough to make sampling deterministic must therefore pick token 1.
static void test_sampling_applies_repeat_penalty(void) {
    sampler s = { .temp = 0.01f, .repeat_penalty = 2.0f, .top_p = 1.0f, .rng = 7 };
    sampler_reset(&s);
    sampler_accept(&s, 0);
    float logits[4] = { 10.0f, 9.0f, 1.0f, 0.5f };
    assert(sample_pick(&s, logits, 4, NULL, NULL) == 1);
}

// Stop tokens are exempt from the penalty: a chat template puts the turn
// terminator in the prompt, so the penalty window is seeded with it and a
// penalised model can never end its turn.
static void test_stop_tokens_stay_exempt(void) {
    sampler s = { .temp = 0.01f, .repeat_penalty = 2.0f, .top_p = 1.0f, .rng = 7,
                  .no_penalty = { 0 }, .n_no_penalty = 1 };
    sampler_reset(&s);
    sampler_accept(&s, 0);
    float logits[4] = { 10.0f, 9.0f, 1.0f, 0.5f };
    assert(sample_pick(&s, logits, 4, NULL, NULL) == 0);
}

// Greedy with a validity filter stays greedy-over-valid-tokens, and still
// without the penalty.
static bool reject_zero(void *ud, int token) { (void)ud; return token != 0; }

static void test_greedy_constrained(void) {
    sampler s = { .temp = 0.0f, .repeat_penalty = 2.0f, .top_p = 1.0f };
    sampler_reset(&s);
    sampler_accept(&s, 1);
    float logits[4] = { 10.0f, 9.0f, 1.0f, 0.5f };
    assert(sample_pick(&s, logits, 4, reject_zero, NULL) == 1);
}

int main(void) {
    test_preset_selection();
    test_preset_fallback();
    test_preset_table_is_enumerable();
    test_override_precedence();
    test_describe();
    test_greedy_ignores_repeat_penalty();
    test_sampling_applies_repeat_penalty();
    test_stop_tokens_stay_exempt();
    test_greedy_constrained();
    puts("sampler tests ok");
    return 0;
}
