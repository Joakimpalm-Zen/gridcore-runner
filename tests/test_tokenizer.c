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

// A control token in the input text is matched as that single id only when
// parse_special is on; otherwise its characters are ordinary text.
static void test_special_token_needs_parse_special(tokenizer *t) {
    int32_t ids[16];
    int n = tok_encode(t, "</s>", ids, 16, false, true);
    assert(n == 1 && ids[0] == 2);

    n = tok_encode(t, "</s>", ids, 16, false, false);
    assert(n > 1);          // spelled out rather than matched
    assert(ids[0] != 2);
}

// Control tokens decode to nothing, so they never reach the output stream,
// but tok_raw still exposes their text for stop-sequence matching.
static void test_control_token_decodes_empty(tokenizer *t) {
    char buf[64];
    assert(tok_is_control(t, 2));
    assert(tok_decode(t, 2, buf, sizeof(buf)) == 0);
    assert(strcmp(tok_raw(t, 2), "</s>") == 0);
}

// Round-trip: decoding what was encoded returns the input, except that SPM
// prepends U+2581 to the first segment, which decodes back to a space.
static void test_spm_roundtrip_adds_space_prefix(tokenizer *t) {
    static const char *const texts[] = {
        "hello", "The quick brown fox.", "tokenization", "café", "\xF0\x9F\xA6\x99 llama",
    };
    for (size_t i = 0; i < sizeof(texts) / sizeof(*texts); i++) {
        int32_t ids[128];
        int n = tok_encode(t, texts[i], ids, 128, false, true);
        char out[512];
        int m = 0;
        for (int k = 0; k < n; k++)
            m += tok_decode(t, ids[k], out + m, (int)sizeof(out) - m);
        out[m] = 0;

        char want[512];
        snprintf(want, sizeof(want), " %s", texts[i]);
        if (strcmp(out, want) != 0) {
            fprintf(stderr, "%s: roundtrip(\"%s\") -> \"%s\", want \"%s\"\n",
                    current, texts[i], out, want);
            abort();
        }
    }
}

// ------------------------------------------------------------------ BPE

// Assert the pre-token split by naming the pieces the text must break into.
// Each piece is a single vocabulary entry in the BPE fixtures, so one id per
// piece means a misplaced boundary shows up directly.
static void expect_pieces(tokenizer *t, const char *text,
                          const char *const *pieces, int n_pieces) {
    int32_t want[32];
    for (int i = 0; i < n_pieces; i++) {
        want[i] = tok_find(t, pieces[i]);
        if (want[i] < 0) {
            fprintf(stderr, "%s: fixture lacks piece \"%s\"\n", current, pieces[i]);
            abort();
        }
    }
    int32_t got[32];
    int n = tok_encode(t, text, got, (int)(sizeof(got) / sizeof(*got)), false, true);
    if (n != n_pieces || memcmp(got, want, sizeof(int32_t) * n_pieces) != 0) {
        fprintf(stderr, "%s: encode(\"%s\"): got %d ids [", current, text, n);
        for (int i = 0; i < n; i++) fprintf(stderr, "%s%d", i ? ", " : "", got[i]);
        fprintf(stderr, "], want %d [", n_pieces);
        for (int i = 0; i < n_pieces; i++) fprintf(stderr, "%s%d", i ? ", " : "", want[i]);
        fprintf(stderr, "]\n");
        abort();
    }
}

// A single non-letter character leads a following letter run, so the boundary
// falls before "/end", not after it. The original GPT-2 rule split them apart.
static void test_bpe_punct_leads_letters(tokenizer *t) {
    static const char *const want[] = { "tokenization", "/end", "." };
    expect_pieces(t, "tokenization/end.", want, 3);
}

// \s*[\r\n]+ takes the whole newline run, so "\n\n" is one pre-token.
static void test_bpe_newline_run_is_one_token(tokenizer *t) {
    // "\xC4\x8A" is U+010A, the byte-mapped newline
    static const char *const want[] = { "a", "\xC4\x8A\xC4\x8A", "b" };
    expect_pieces(t, "a\n\nb", want, 3);
}

// Contractions split off ahead of the letter rule, and a leading space joins
// the word after them. "\xC4\xA0" is U+0120, the byte-mapped space.
static void test_bpe_contraction_and_space(tokenizer *t) {
    static const char *const want[] = { "I", "'ll", "\xC4\xA0go" };
    expect_pieces(t, "I'll go", want, 3);
}

// The digit rule is the one place llama-bpe and qwen2 genuinely differ:
// \p{N}{1,3} against \p{N}. Same vocabulary, same input, different split.
static void test_bpe_digit_grouping_llama3(tokenizer *t) {
    static const char *const want[] = { "123", "456", "789", "0" };
    expect_pieces(t, "1234567890", want, 4);
}

static void test_bpe_digit_grouping_qwen2(tokenizer *t) {
    static const char *const want[] = { "1","2","3","4","5","6","7","8","9","0" };
    expect_pieces(t, "1234567890", want, 10);
}

// BPE has no space prefix, so the round-trip is byte-exact. The byte-level
// alphabet covers any input, including the whitespace the split rules key on.
static void test_bpe_roundtrip_is_exact(tokenizer *t) {
    static const char *const texts[] = {
        "tokenization/end.", "I'll go", "a\n\nb", "1234567890", "x\ty  z",
    };
    for (size_t i = 0; i < sizeof(texts) / sizeof(*texts); i++) {
        int32_t ids[128];
        int n = tok_encode(t, texts[i], ids, 128, false, true);
        char out[512];
        int m = 0;
        for (int k = 0; k < n; k++)
            m += tok_decode(t, ids[k], out + m, (int)sizeof(out) - m);
        out[m] = 0;
        if (strcmp(out, texts[i]) != 0) {
            fprintf(stderr, "%s: roundtrip mismatch: got \"%s\"\n", current, out);
            abort();
        }
    }
}

// Under the original GPT-2 rules a leading non-letter does NOT join the word,
// so "/end" splits apart and "end" falls back to its byte pieces. This is the
// direct counterpart of test_bpe_punct_leads_letters on identical vocabulary.
static void test_bpe_punct_stays_split(tokenizer *t) {
    static const char *const want[] = { "tokenization", "/", "e", "n", "d", "." };
    expect_pieces(t, "tokenization/end.", want, 6);
}

// smollm keeps the original GPT-2 whitespace rules, so a newline run is not
// glued into one pre-token the way llama-bpe's \s*[\r\n]+ does.
static void test_bpe_newline_run_stays_split(tokenizer *t) {
    static const char *const want[] = { "a", "\xC4\x8A", "\xC4\x8A", "b" };
    expect_pieces(t, "a\n\nb", want, 4);
}

// Its Digits pass also bounds the whitespace rule: the run ends at the digit,
// so both spaces stay together where "  leading" would give one back.
static void test_bpe_digits_bound_whitespace(tokenizer *t) {
    static const char *const want[] = { "\xC4\xA0\xC4\xA0", "1", "2" };
    expect_pieces(t, "  12", want, 3);
}

static void run_bpe_fixture(const char *path, int pre, void (*digits)(tokenizer *)) {
    current = path;
    gguf_file g;
    if (!gguf_open(&g, path)) {
        fprintf(stderr, "cannot open fixture %s\n", path);
        exit(1);
    }
    tokenizer t;
    if (!tokenizer_init(&t, &g)) {
        fprintf(stderr, "tokenizer_init failed on %s\n", path);
        exit(1);
    }
    assert(t.model == TOK_BPE);
    assert(t.pre == pre);

    if (pre == TOK_PRE_SMOLLM) {
        // smollm keeps the original GPT-2 rules; the others use the newer regex
        test_bpe_punct_stays_split(&t);
        test_bpe_newline_run_stays_split(&t);
        test_bpe_digits_bound_whitespace(&t);
    } else {
        test_bpe_punct_leads_letters(&t);
        test_bpe_newline_run_is_one_token(&t);
    }
    test_bpe_contraction_and_space(&t);
    test_bpe_roundtrip_is_exact(&t);
    digits(&t);

    tokenizer_free(&t);
    gguf_close(&g);
}

// gemma4's normalizer rewrites a space to U+2581 before any merging, so a
// space becomes a piece of its own rather than a prefix on the next word.
static void test_bpe_spm_space_becomes_metaspace(tokenizer *t) {
    static const char *const want[] = { "a", "\xE2\x96\x81", "b" };
    expect_pieces(t, "a b", want, 3);
}

// Merges still run normally over raw UTF-8, with no byte->unicode mapping.
static void test_bpe_spm_merges_run(tokenizer *t) {
    static const char *const want[] = { "hello" };
    expect_pieces(t, "hello", want, 1);
}

// The real gemma-4 vocabulary has no literal CR piece, so CR must fall back to
// its <0x0D> byte token. Dropping it silently loses input: runner and the
// HuggingFace reference disagreed on 11 of the 721 corpus strings for exactly
// this reason, every one of them containing CR.
static void test_bpe_spm_byte_fallback_single(tokenizer *t) {
    static const char *const want[] = { "<0x0D>" };
    expect_pieces(t, "\r", want, 1);
}

// A multi-byte codepoint with no piece of its own decomposes to one <0xNN>
// token per UTF-8 byte, in order. U+00A0 (NBSP) is the corpus case.
static void test_bpe_spm_byte_fallback_multibyte(tokenizer *t) {
    static const char *const want[] = { "a", "<0xC2>", "<0xA0>", "b" };
    expect_pieces(t, "a\xC2\xA0" "b", want, 4);
}

static void run_bpe_spm_fixture(const char *path) {
    current = path;
    gguf_file g;
    if (!gguf_open(&g, path)) {
        fprintf(stderr, "cannot open fixture %s\n", path);
        exit(1);
    }
    tokenizer t;
    if (!tokenizer_init(&t, &g)) {
        fprintf(stderr, "tokenizer_init failed on %s\n", path);
        exit(1);
    }
    assert(t.model == TOK_BPE_SPM);

    test_bpe_spm_space_becomes_metaspace(&t);
    test_bpe_spm_merges_run(&t);
    test_bpe_spm_byte_fallback_single(&t);
    test_bpe_spm_byte_fallback_multibyte(&t);

    tokenizer_free(&t);
    gguf_close(&g);
}

// ------------------------------------------------- special-token matching
//
// tok_encode probes for a special token at every byte offset of the input, and
// the committed vocabulary fixtures carry two or three specials that all begin
// with '<' — so nothing here covered a special that starts with a tab, a
// space, a letter, or one special that is a prefix of another. Real vocabs do:
// gemma-3 ships 6,414 (HTML tags, <unusedNNN>, and whitespace runs of every
// length from 1 to 31, which begin with ' ', '\t' and '\n').
//
// The vocabulary is assembled here rather than committed, because what is
// being checked is the matching rule, not any real model's pieces.
typedef struct { unsigned char *b; size_t n, cap; } gbuf;
static void gput(gbuf *w, const void *p, size_t n) {
    if (w->n + n > w->cap) {
        w->cap = (w->n + n) * 2 + 64;
        w->b = realloc(w->b, w->cap);
        assert(w->b);
    }
    memcpy(w->b + w->n, p, n);
    w->n += n;
}
static void gu32(gbuf *w, uint32_t v) { gput(w, &v, 4); }
static void gu64(gbuf *w, uint64_t v) { gput(w, &v, 8); }
static void gstr(gbuf *w, const char *s) { gu64(w, strlen(s)); gput(w, s, strlen(s)); }
static void gkey(gbuf *w, const char *k, uint32_t type) { gstr(w, k); gu32(w, type); }

// Specials spanning six different first bytes, plus "ABC"/"ABCD" sharing one:
// the longer must win, which is the rule the ordering inside a first-byte
// group exists to preserve.
static const char *const SPECIALS[] = {
    "<|im_start|>", "<|im_end|>", "[INST]", "\t\t", "\n\n", "   ", "ABC", "ABCD",
};
#define N_SPECIALS ((int)(sizeof(SPECIALS) / sizeof(*SPECIALS)))
static const char *const PIECES[] = { "h", "o", "A", "B", "C", "D" };
#define N_PIECES ((int)(sizeof(PIECES) / sizeof(*PIECES)))

static void write_special_fixture(const char *path) {
    gbuf kv = {0};
    uint64_t nkv = 0;
    gkey(&kv, "general.architecture", GGUF_T_STR); gstr(&kv, "llama"); nkv++;
    gkey(&kv, "tokenizer.ggml.model", GGUF_T_STR); gstr(&kv, "llama"); nkv++;

    int n_tok = 1 + N_PIECES + N_SPECIALS;
    gkey(&kv, "tokenizer.ggml.tokens", GGUF_T_ARR);
    gu32(&kv, GGUF_T_STR); gu64(&kv, (uint64_t)n_tok);
    gstr(&kv, "<unk>");
    for (int i = 0; i < N_PIECES; i++) gstr(&kv, PIECES[i]);
    for (int i = 0; i < N_SPECIALS; i++) gstr(&kv, SPECIALS[i]);
    nkv++;

    gkey(&kv, "tokenizer.ggml.token_type", GGUF_T_ARR);
    gu32(&kv, GGUF_T_I32); gu64(&kv, (uint64_t)n_tok);
    gu32(&kv, 2);                                            // <unk>
    for (int i = 0; i < N_PIECES; i++) gu32(&kv, 1);         // normal
    for (int i = 0; i < N_SPECIALS; i++) gu32(&kv, 4);       // user-defined
    nkv++;

    gkey(&kv, "tokenizer.ggml.scores", GGUF_T_ARR);
    gu32(&kv, GGUF_T_F32); gu64(&kv, (uint64_t)n_tok);
    for (int i = 0; i < n_tok; i++) { float z = 0.0f; gput(&kv, &z, 4); }
    nkv++;

    gkey(&kv, "tokenizer.ggml.unknown_token_id", GGUF_T_U32); gu32(&kv, 0); nkv++;
    gkey(&kv, "tokenizer.ggml.bos_token_id", GGUF_T_U32); gu32(&kv, 0); nkv++;
    gkey(&kv, "tokenizer.ggml.eos_token_id", GGUF_T_U32); gu32(&kv, 0); nkv++;
    gkey(&kv, "tokenizer.ggml.add_bos_token", GGUF_T_BOOL);
    { unsigned char f = 0; gput(&kv, &f, 1); } nkv++;
    gkey(&kv, "tokenizer.ggml.add_space_prefix", GGUF_T_BOOL);
    { unsigned char f = 0; gput(&kv, &f, 1); } nkv++;

    gbuf w = {0};
    gu32(&w, 0x46554747); gu32(&w, 3); gu64(&w, 0); gu64(&w, nkv);
    gput(&w, kv.b, kv.n);
    while (w.n % 32) { unsigned char z = 0; gput(&w, &z, 1); }
    FILE *f = fopen(path, "wb");
    assert(f);
    assert(fwrite(w.b, 1, w.n, f) == w.n);
    fclose(f);
    free(kv.b); free(w.b);
}

static void test_special_matching_across_first_bytes(void) {
    const char *path = "tok-specials.gguf";
    write_special_fixture(path);
    current = path;
    gguf_file g;
    assert(gguf_open(&g, path));
    tokenizer t;
    assert(tokenizer_init(&t, &g));
    assert(t.n_special == N_SPECIALS);

    int32_t ids[32];
    for (int i = 0; i < N_SPECIALS; i++) {
        int want = tok_find(&t, SPECIALS[i]);
        assert(want >= 0);
        // alone
        int n = tok_encode(&t, SPECIALS[i], ids, 32, false, true);
        if (n != 1 || ids[0] != want) {
            fprintf(stderr, "%s: encode(\"%s\") gave %d ids, first %d, want 1 x %d\n",
                    current, SPECIALS[i], n, n > 0 ? ids[0] : -1, want);
            abort();
        }
        // between ordinary pieces, and twice in a row
        char text[64];
        snprintf(text, sizeof(text), "h%s%so", SPECIALS[i], SPECIALS[i]);
        n = tok_encode(&t, text, ids, 32, false, true);
        assert(n == 4);
        assert(ids[0] == tok_find(&t, "h") && ids[1] == want && ids[2] == want &&
               ids[3] == tok_find(&t, "o"));
    }

    // "ABCD" must beat "ABC": both live in the same first-byte group, and the
    // longest match wins wherever the group is scanned from.
    int n = tok_encode(&t, "ABCD", ids, 32, false, true);
    assert(n == 1 && ids[0] == tok_find(&t, "ABCD"));
    n = tok_encode(&t, "ABCB", ids, 32, false, true);
    assert(n == 2 && ids[0] == tok_find(&t, "ABC") && ids[1] == tok_find(&t, "B"));

    tokenizer_free(&t);
    gguf_close(&g);
    remove(path);
}

int main(void) {
    test_special_matching_across_first_bytes();

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
        test_special_token_needs_parse_special(&t);
        test_control_token_decodes_empty(&t);
        test_spm_roundtrip_adds_space_prefix(&t);

        tokenizer_free(&t);
        gguf_close(&g);
    }

    run_bpe_fixture("tests/fixtures/vocab-bpe-llama3.gguf",
                    TOK_PRE_LLAMA3, test_bpe_digit_grouping_llama3);
    run_bpe_fixture("tests/fixtures/vocab-bpe-gpt4o.gguf",
                    TOK_PRE_LLAMA4, test_bpe_digit_grouping_llama3);
    run_bpe_fixture("tests/fixtures/vocab-bpe-qwen2.gguf",
                    TOK_PRE_QWEN2, test_bpe_digit_grouping_qwen2);
    run_bpe_fixture("tests/fixtures/vocab-bpe-qwen35.gguf",
                    TOK_PRE_QWEN2, test_bpe_digit_grouping_qwen2);
    run_bpe_fixture("tests/fixtures/vocab-bpe-smollm.gguf",
                    TOK_PRE_SMOLLM, test_bpe_digit_grouping_qwen2);

    run_bpe_spm_fixture("tests/fixtures/vocab-bpe-spm-gemma4.gguf");

    puts("tokenizer tests ok");
    return 0;
}
