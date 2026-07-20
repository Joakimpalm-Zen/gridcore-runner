// Tokenizer tests against a real SPM (llama) vocabulary.
//
// test.gguf's vocab is byte-fallback only (<unk>, <s>, </s>, <0x00>..<0xFF>),
// so it cannot exercise score-based piece merging at all. These run against
// committed vocabulary-only fixtures carrying the real 32000-piece TinyLlama
// vocab (see scripts/make-vocab-fixture.py). Expected ids are ground truth
// from sentencepiece, not from this implementation, so a wrong-but-
// self-consistent encoder still fails.
//
// Both fixtures must produce identical ids: vocab-spm-zeroscores.gguf carries
// all-zero scores plus a merges list, the shape many real conversions ship,
// and is the regression test for rebuilding scores from merge rank.
#include "runner.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *fixtures[] = {
    "tests/fixtures/vocab-spm.gguf",             // real sentencepiece scores
    "tests/fixtures/vocab-spm-zeroscores.gguf",  // all-zero scores + merges
};

static const char *current; // fixture under test, for failure messages

static void expect_ids(tokenizer *t, const char *text,
                       const int32_t *want, int n_want) {
    int32_t got[64];
    // add_bos and parse_special on: the engine's default path
    int n = tok_encode(t, text, got, (int)(sizeof(got) / sizeof(*got)), true, true);
    if (n != n_want || memcmp(got, want, sizeof(int32_t) * n_want) != 0) {
        fprintf(stderr, "%s: encode(\"%s\"): got [", current, text);
        for (int i = 0; i < n; i++) fprintf(stderr, "%s%d", i ? ", " : "", got[i]);
        fprintf(stderr, "], want [");
        for (int i = 0; i < n_want; i++) fprintf(stderr, "%s%d", i ? ", " : "", want[i]);
        fprintf(stderr, "]\n");
        abort();
    }
}

// Score-based merging: whole word pieces, not per-character fallback, with the
// U+2581 space prefix applied to the first segment.
static void test_spm_merges_whole_words(tokenizer *t) {
    const int32_t want[] = { 1, 15043, 3186 }; // <s> ▁Hello ▁world
    expect_ids(t, "Hello world", want, 3);
}

// Merge order must follow score/rank, not position. Left-to-right merging
// yields ▁llam + a here, which is a valid decoding but the wrong tokenization.
static void test_spm_merge_order_beats_position(tokenizer *t) {
    const int32_t want[] = { 1, 11148, 3304 }; // <s> ▁ll ama
    expect_ids(t, "llama", want, 3);
}

// A codepoint with no vocab piece decomposes to one <0xNN> token per UTF-8
// byte, and merging resumes normally afterwards.
static void test_spm_byte_fallback(tokenizer *t) {
    // <s> ▁ <0xF0> <0x9F> <0xA6> <0x99> ▁ll ama
    const int32_t want[] = { 1, 29871, 243, 162, 169, 156, 11148, 3304 };
    expect_ids(t, "\xF0\x9F\xA6\x99 llama", want, 8);
}

int main(void) {
    for (size_t i = 0; i < sizeof(fixtures) / sizeof(*fixtures); i++) {
        current = fixtures[i];

        gguf_file g;
        if (!gguf_open(&g, current)) {
            fprintf(stderr, "cannot open fixture %s (run from the repo root; "
                            "regenerate with scripts/make-vocab-fixture.py)\n", current);
            return 1;
        }

        tokenizer t;
        if (!tokenizer_init(&t, &g)) {
            fprintf(stderr, "tokenizer_init failed on %s\n", current);
            return 1;
        }
        assert(t.model == TOK_SPM);
        assert(t.n_vocab == 32000);

        test_spm_merges_whole_words(&t);
        test_spm_merge_order_beats_position(&t);
        test_spm_byte_fallback(&t);

        tokenizer_free(&t);
        gguf_close(&g);
    }
    puts("tokenizer tests ok");
    return 0;
}
