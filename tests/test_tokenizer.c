// Tokenizer tests against a real SPM (llama) vocabulary.
//
// test.gguf's vocab is byte-fallback only (<unk>, <s>, </s>, <0x00>..<0xFF>),
// so it cannot exercise score-based piece merging at all. These tests run
// against TinyLlama-1.1B-Chat (Llama-2 SPM vocab, 32000 pieces with scores);
// fetch it with scripts/get-test-vocab.sh. Expected ids are ground truth from
// the HuggingFace reference tokenizer, not from this implementation.
//
// The model is large and not in CI, so a missing file skips rather than fails.
#include "runner.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MODEL_ENV  "RUNNER_TEST_VOCAB"
#define MODEL_PATH "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

// encode with BOS and special-token parsing on (the engine's default path)
static int encode(tokenizer *t, const char *text, int32_t *out, int cap) {
    return tok_encode(t, text, out, cap, true, true);
}

static void expect_ids(tokenizer *t, const char *text,
                       const int32_t *want, int n_want) {
    int32_t got[64];
    int n = encode(t, text, got, (int)(sizeof(got) / sizeof(*got)));
    if (n != n_want || memcmp(got, want, sizeof(int32_t) * n_want) != 0) {
        fprintf(stderr, "encode(\"%s\"): got [", text);
        for (int i = 0; i < n; i++) fprintf(stderr, "%s%d", i ? ", " : "", got[i]);
        fprintf(stderr, "], want [");
        for (int i = 0; i < n_want; i++) fprintf(stderr, "%s%d", i ? ", " : "", want[i]);
        fprintf(stderr, "]\n");
        abort();
    }
}

// Score-based merging: "Hello world" is two whole pieces, not per-character
// byte fallback, and the leading U+2581 space prefix is applied.
static void test_spm_merges_whole_words(tokenizer *t) {
    const int32_t want[] = { 1, 15043, 3186 }; // <s> ▁Hello ▁world
    expect_ids(t, "Hello world", want, 3);
}

int main(void) {
    const char *path = getenv(MODEL_ENV);
    if (!path) path = MODEL_PATH;

    gguf_file g;
    if (!gguf_open(&g, path)) {
        printf("tokenizer tests skipped: no vocab model at %s "
               "(run scripts/get-test-vocab.sh, or set %s)\n", path, MODEL_ENV);
        return 0;
    }

    tokenizer t;
    if (!tokenizer_init(&t, &g)) {
        fprintf(stderr, "tokenizer_init failed on %s\n", path);
        return 1;
    }
    assert(t.model == TOK_SPM);

    test_spm_merges_whole_words(&t);

    tokenizer_free(&t);
    gguf_close(&g);
    puts("tokenizer tests ok");
    return 0;
}
