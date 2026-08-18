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
#include <stdlib.h>
#include <string.h>

#define EQ(a, b) (fabsf((a) - (b)) < 1e-6f)

// The names below are the real `general.name` values read out of the GGUFs in
// models/ — matching is only worth testing against strings that ship.
static void test_preset_selection(void) {
    const sampler_preset *p;

    p = sampler_preset_for("qwen3", "Qwen3 4B Instruct Awq", -1);
    assert(!strcmp(p->name, "qwen3"));
    assert(EQ(p->temp, 0.6f) && EQ(p->top_p, 0.95f) && p->top_k == 20);
    assert(EQ(p->min_p, 0.0f));

    p = sampler_preset_for("qwen2", "Qwen2.5 32B Instruct", -1);
    assert(!strcmp(p->name, "qwen2.5"));
    assert(EQ(p->temp, 0.7f) && EQ(p->top_p, 0.8f) && p->top_k == 20);
    assert(EQ(p->repeat_penalty, 1.05f));

    p = sampler_preset_for("gemma3", "Gemma 3 4b It", -1);
    assert(!strcmp(p->name, "gemma3"));
    assert(EQ(p->temp, 1.0f) && EQ(p->top_p, 0.95f) && p->top_k == 64);

    p = sampler_preset_for("phi3", "Phi 3.5 Mini Instruct", -1);
    assert(!strcmp(p->name, "phi3"));
    assert(EQ(p->temp, 0.0f));

    // three families share `general.architecture: llama`, so the name is the
    // only thing that separates them
    p = sampler_preset_for("llama", "Llama 3.2 3B Instruct", -1);
    assert(!strcmp(p->name, "llama3"));
    assert(EQ(p->temp, 0.6f) && EQ(p->top_p, 0.9f));
    assert(!strcmp(sampler_preset_for("llama", "Meta Llama 3.1 8B Instruct", -1)->name,
                   "llama3"));

    p = sampler_preset_for("llama", "Mistral-7B-Instruct-v0.3", -1);
    assert(!strcmp(p->name, "mistral"));
    assert(EQ(p->temp, 0.7f) && EQ(p->top_p, 1.0f));

    p = sampler_preset_for("llama", "Smollm2 1.7B 8k Mix7 Ep2 v2", -1);
    assert(!strcmp(p->name, "smollm2"));
    assert(EQ(p->temp, 0.2f) && EQ(p->top_p, 0.9f));

    // Mistral Nemo departs from the rest of the Mistral family: the vendor
    // card demands temperature 0.3. Plain Mistrals must NOT land here, and
    // NVIDIA's "Nemotron" (which contains "nemo") must not either.
    p = sampler_preset_for("llama", "Mistral-Nemo-Instruct-2407", -1);
    assert(!strcmp(p->name, "mistral-nemo"));
    assert(EQ(p->temp, 0.3f));
    assert(!strcmp(sampler_preset_for("llama", "Mistral-7B-Instruct-v0.3", -1)->name,
                   "mistral"));
    assert(strcmp(sampler_preset_for("llama", "Nemotron Mini 4B", -1)->name,
                  "mistral-nemo") != 0);

    // European llama-declaring families with published settings.
    p = sampler_preset_for("llama", "Lucie-7B-Instruct", -1);
    assert(!strcmp(p->name, "lucie"));
    assert(EQ(p->temp, 0.6f) && EQ(p->top_p, 0.9f));

    p = sampler_preset_for("llama", "salamandra-7b-instruct", -1);
    assert(!strcmp(p->name, "salamandra"));
    assert(EQ(p->temp, 0.6f) && EQ(p->repeat_penalty, 1.2f));

    p = sampler_preset_for("llama", "Teuken-7B-instruct-commercial-v0.4", -1);
    assert(!strcmp(p->name, "teuken"));
    assert(EQ(p->temp, 0.7f) && EQ(p->top_p, 0.95f));

    // EuroLLM publishes no sampling settings (gated repo, no guidance in the
    // quantizer mirrors) — it must stay on generic rather than borrow one.
    assert(!strcmp(sampler_preset_for("llama", "EuroLLM-9B-Instruct", -1)->name,
                   "generic"));

    // Quantizer metadata is unreliable: a real community salamandra GGUF
    // ships general.name "snapshots". The combined identity (name + load-path
    // basename) recovers the family; call sites resolve through it.
    char ident[256];
    sampler_ident("snapshots", "models/salamandra-7b-instruct-Q4_K_M-f32.gguf",
                  ident, sizeof(ident));
    p = sampler_preset_for("llama", ident, -1);
    assert(!strcmp(p->name, "salamandra"));
    sampler_ident(NULL, "C:\\models\\Mistral-Nemo-Instruct.gguf",
                  ident, sizeof(ident));
    assert(!strcmp(sampler_preset_for("llama", ident, -1)->name, "mistral-nemo"));

    // Gridcore Syntetik declares arch "llama", name "gridcore-<size>"; the
    // suite-native contract compiler resolves to a deterministic preset, not
    // a vendor one, and greedy is the point (temp 0, no penalty).
    p = sampler_preset_for("llama", "gridcore-10m", -1);
    assert(!strcmp(p->name, "gridcore"));
    assert(EQ(p->temp, 0.0f) && EQ(p->repeat_penalty, 1.0f) && p->top_k == 0);
    assert(!strcmp(sampler_preset_for("llama", "Syntetik 350M", -1)->name, "gridcore"));
}

// A model nobody published numbers for must not silently borrow another
// family's: it falls back to the generic preset, which is what runner shipped
// as its fixed defaults before presets existed.
static void test_preset_fallback(void) {
    const sampler_preset *g = sampler_preset_for("llama", "tinyllama_tinyllama-1.1b-chat-v1.0", -1);
    assert(!strcmp(g->name, "generic"));
    assert(EQ(g->temp, 0.8f) && EQ(g->top_p, 0.95f) && EQ(g->min_p, 0.05f));
    assert(g->top_k == 40 && EQ(g->repeat_penalty, 1.1f));

    // "tinyllama" must not be read as the llama-3 family
    assert(strcmp(sampler_preset_for("llama", "TinyLlama 1.1B", -1)->name, "llama3") != 0);
    // missing or unknown metadata is not an error
    assert(!strcmp(sampler_preset_for(NULL, NULL, -1)->name, "generic"));
    assert(!strcmp(sampler_preset_for("mamba", "", -1)->name, "generic"));
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

    const sampler_preset *p = sampler_resolve(&s, "qwen3", "Qwen3 4B", -1, &none);
    assert(!strcmp(p->name, "qwen3"));
    assert(EQ(s.temp, 0.6f) && EQ(s.top_p, 0.95f) && s.top_k == 20);
    assert(s.rng == 12345);  // resolution never disturbs sampler state

    // an explicit value always wins, and only that field moves
    sampler_override ov = { .has_temp = true, .temp = 0.05f,
                            .has_repeat_penalty = true, .repeat_penalty = 1.0f };
    p = sampler_resolve(&s, "qwen3", "Qwen3 4B", -1, &ov);
    assert(EQ(s.temp, 0.05f) && EQ(s.repeat_penalty, 1.0f));
    assert(EQ(s.top_p, 0.95f) && s.top_k == 20);

    // overrides win over the generic preset too, and a zero override is a
    // real value rather than "unset"
    sampler_override zero = { .has_top_k = true, .top_k = 0,
                              .has_min_p = true, .min_p = 0.0f };
    p = sampler_resolve(&s, "llama", "tinyllama", -1, &zero);
    assert(!strcmp(p->name, "generic"));
    assert(s.top_k == 0 && EQ(s.min_p, 0.0f) && EQ(s.temp, 0.8f));

    // resolving twice from the same overrides is idempotent — the server
    // re-resolves on every model swap
    sampler again = s;
    sampler_resolve(&s, "llama", "tinyllama", -1, &zero);
    assert(!memcmp(&again, &s, sizeof s));
}

static void test_describe(void) {
    sampler s = {0};
    sampler_override none = {0};
    const sampler_preset *p = sampler_resolve(&s, "gemma3", "Gemma 3 4b It", -1, &none);
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

// The penalty window holds token ids, and nothing that fills it is bounded by
// the vocabulary the LOGITS were sized with: the tokenizer's n_vocab comes
// from tokenizer.ggml.tokens while the logits array is sized by tok_embd's
// rows, and speculative decoding accepts ids drafted from a second model's
// vocabulary. `logits[tok] /= repeat_penalty` is a read-modify-write, so an id
// outside [0, n_vocab) writes past the array — a scaled float at an index the
// model file chose. The padding either side of the logits is the canary.
static void test_penalty_window_ignores_ids_outside_the_vocabulary(void) {
    enum { PAD = 8, V = 4 };
    float *block = malloc(sizeof(float) * (PAD + V + PAD));
    assert(block);
    for (int i = 0; i < PAD + V + PAD; i++) block[i] = 100.0f + (float)i;
    float *logits = block + PAD;
    logits[0] = 10.0f; logits[1] = 9.0f; logits[2] = 1.0f; logits[3] = 0.5f;

    sampler s = { .temp = 0.01f, .repeat_penalty = 2.0f, .top_p = 1.0f, .rng = 7 };
    sampler_reset(&s);
    sampler_accept(&s, V + 3);   // past the end
    sampler_accept(&s, -2);      // and before the start
    assert(sample_pick(&s, logits, V, NULL, NULL) == 0);
    for (int i = 0; i < PAD; i++) assert(EQ(block[i], 100.0f + (float)i));
    for (int i = PAD + V; i < PAD + V + PAD; i++)
        assert(EQ(block[i], 100.0f + (float)i));
    free(block);
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

// The "no filter at all" configuration -- top-k off, top-p 1.0, min-p off --
// is now served from a head instead of a full-vocabulary sort, with the head
// sized from the roulette draw itself. Two properties have to survive that,
// and both are checked here through the public API only:
//
//   determinism -- the same seed and the same logits must give the same token,
//   because the head is sized from a draw and a bug there would show up as a
//   seed-dependent wobble;
//
//   distribution -- sampling many seeds must still track the softmax over the
//   WHOLE vocabulary. If the head were sized too small the tail would be
//   truncated and the common tokens would be over-represented.
//
// The stronger check that the token is bit-for-bit what the old full sort
// returned is an end-to-end one against the previous binary at a fixed seed;
// it is recorded in docs/ rather than run here, since it needs two builds.
static void test_no_filter_sampling_is_deterministic_and_unbiased(void) {
    enum { V = 8192, TRIALS = 4000 };
    float *logits = malloc(sizeof(float) * V);
    float *scratch = malloc(sizeof(float) * V);
    assert(logits && scratch);
    uint64_t seed = 0x9E3779B97F4A7C15ull;
    for (int i = 0; i < V; i++) {
        seed = seed * 6364136223846793005ull + 1442695040888963407ull;
        logits[i] = (float)((seed >> 40) / (double)(1 << 24)) * 4.0f - 6.0f;
    }
    // three dominant tokens carrying most of the mass
    logits[42] = 6.0f; logits[1234] = 5.4f; logits[7000] = 4.8f;

    sampler base = {0};
    base.temp = 1.0f; base.top_p = 1.0f; base.top_k = 0; base.min_p = 0.0f;
    base.repeat_penalty = 1.0f;

    // determinism: same seed twice, same token
    for (int trial = 0; trial < 50; trial++) {
        sampler a = base, b = base;
        a.rng = b.rng = 0xD1CE0000ull + (uint64_t)trial;
        memcpy(scratch, logits, sizeof(float) * V);
        int ta = sample_pick(&a, scratch, V, NULL, NULL);
        memcpy(scratch, logits, sizeof(float) * V);
        int tb = sample_pick(&b, scratch, V, NULL, NULL);
        assert(ta == tb);
    }

    // distribution: empirical frequency of the dominant tokens must match the
    // analytic softmax over the whole vocabulary
    double mx = -1e30, total = 0;
    for (int i = 0; i < V; i++) if (logits[i] > mx) mx = logits[i];
    for (int i = 0; i < V; i++) total += exp(logits[i] - mx);
    const int watch[3] = { 42, 1234, 7000 };
    int hits[3] = {0, 0, 0};
    sampler s = base;
    s.rng = 0xBEEF1234ull;
    for (int trial = 0; trial < TRIALS; trial++) {
        memcpy(scratch, logits, sizeof(float) * V);
        int tok = sample_pick(&s, scratch, V, NULL, NULL);
        for (int w = 0; w < 3; w++) if (tok == watch[w]) hits[w]++;
    }
    for (int w = 0; w < 3; w++) {
        double want = exp(logits[watch[w]] - mx) / total;
        double got = (double)hits[w] / TRIALS;
        // 4000 draws: a 4-sigma band on the rarest of the three is ~0.02
        assert(fabs(got - want) < 0.03);
    }
    free(logits); free(scratch);
    puts("ok: no-filter sampling is deterministic and matches the full softmax");
}


// The DETECTED TEMPLATE outranks the model name. Both answer "which family is
// this", but the template is read out of the checkpoint while the name is a
// label a re-quantiser can change -- and for Nemo the two presets disagree on
// the temperature its own model card calls out (0.3, not 0.7).
static void test_template_outranks_the_model_name(void) {
    // a Nemo export whose name says nothing about Nemo, or nothing about
    // Mistral: by name alone both land on the wrong preset
    assert(!strcmp(sampler_preset_for("llama", "MN-12B-Instruct", -1)->name,
                   "llama3") ||
           strcmp(sampler_preset_for("llama", "MN-12B-Instruct", -1)->name,
                  "mistral-nemo") != 0);
    assert(!strcmp(sampler_preset_for("llama", "Mistral-12B-Instruct", -1)->name,
                   "mistral"));
    // with the template known, both reach mistral-nemo
    assert(!strcmp(sampler_preset_for("llama", "MN-12B-Instruct",
                                      TMPL_MISTRAL_NEMO)->name, "mistral-nemo"));
    assert(!strcmp(sampler_preset_for("llama", "Mistral-12B-Instruct",
                                      TMPL_MISTRAL_NEMO)->name, "mistral-nemo"));
    // and the v0.1/v0.2 and v0.3 framings both mean the plain preset
    assert(!strcmp(sampler_preset_for("llama", "whatever",
                                      TMPL_MISTRAL)->name, "mistral"));
    assert(!strcmp(sampler_preset_for("llama", "whatever",
                                      TMPL_MISTRAL_V1)->name, "mistral"));
    // a suite-native model still wins over the template: it is checked first
    // and its name is the only thing that identifies it
    assert(!strcmp(sampler_preset_for("llama", "gridcore-10m",
                                      TMPL_MISTRAL_NEMO)->name, "gridcore"));
    // an unknown template does not disturb name-based selection
    assert(!strcmp(sampler_preset_for("llama", "Mistral-Nemo-Instruct-2407",
                                      -1)->name, "mistral-nemo"));
}

int main(void) {
    test_template_outranks_the_model_name();
    test_preset_selection();
    test_preset_fallback();
    test_preset_table_is_enumerable();
    test_override_precedence();
    test_describe();
    test_greedy_ignores_repeat_penalty();
    test_sampling_applies_repeat_penalty();
    test_stop_tokens_stay_exempt();
    test_penalty_window_ignores_ids_outside_the_vocabulary();
    test_greedy_constrained();
    test_no_filter_sampling_is_deterministic_and_unbiased();
    puts("sampler tests ok");
    return 0;
}
