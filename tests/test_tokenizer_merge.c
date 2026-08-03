// The two BPE/SentencePiece merge implementations must produce identical ids.
//
// Both loops apply merges best-first, leftmost on a tie. One rescans every
// adjacent pair after every merge — O(n^2), which is what runner shipped — and
// one keeps a priority queue of candidates, which is O(n log n) and is what
// runs on any segment long enough for the difference to matter. Which one runs
// is purely a cost decision, so their outputs must never differ.
//
// This is the gate for that, and it is the gate because ids are load-bearing:
// every `greedy_reference` certification in the compatibility matrix is a
// claim about exact output, and a tokenizer that shifts one id invalidates all
// of them. The first working version of the queue failed here — a symbol
// absorbed as somebody's right-hand side kept its old length and `next`, so a
// stale candidate naming it still passed the liveness test and merged a symbol
// that had left the list. On EuroLLM that turned "  index." into three ids
// instead of two.
//
// Every committed vocabulary fixture is exercised, because the two families
// reach different code (spm_encode vs bpe_word) and gemma-4 reaches bpe_word
// with whole lines rather than pre-tokens. Inputs are the shared corpus plus
// shapes chosen to stress a merge loop specifically: long unbroken runs, no
// spaces at all, repeated bigrams, and text that straddles the length
// threshold in both directions.
#include "runner.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fail = 0;
static const char *g_fixture;

#define CAP 8192

static void compare(tokenizer *t, const char *label, const char *text) {
    static int32_t a[CAP], b[CAP];
    for (int bos = 0; bos < 2; bos++) {
        for (int sp = 0; sp < 2; sp++) {
            tok_merge_force(0);
            int na = tok_encode(t, text, a, CAP, bos != 0, sp != 0);
            tok_merge_force(1);
            int nb = tok_encode(t, text, b, CAP, bos != 0, sp != 0);
            tok_merge_force(-1);
            if (na == nb && (na <= 0 || memcmp(a, b, sizeof(int32_t) * (size_t)na) == 0))
                continue;
            fprintf(stderr, "FAIL %s [%s bos=%d special=%d]: rescan %d ids, "
                    "queue %d ids\n", g_fixture, label, bos, sp, na, nb);
            int lim = na < nb ? na : nb;
            for (int i = 0; i < lim; i++)
                if (a[i] != b[i]) {
                    fprintf(stderr, "  first difference at %d: %d vs %d\n", i, a[i], b[i]);
                    break;
                }
            g_fail = 1;
            return;
        }
    }
}

// Shapes a merge loop can get wrong that ordinary prose does not reach.
static void adversarial(tokenizer *t) {
    static char buf[6000];
    struct { const char *label; const char *unit; size_t len; } cases[] = {
        { "one repeated character",   "a",        4000 },
        { "repeated bigram",          "ab",       4000 },
        { "no spaces, mixed letters", "qwertyuio",4000 },
        { "digits",                   "0123456789", 4000 },
        { "spm space markers",        "\xE2\x96\x81", 3000 },
        { "unbroken prose",
          "the quick brown fox jumps over the lazy dog ", 4000 },
        // straddle the queue threshold from both sides
        { "just under the threshold", "ab", 20 },
        { "just over the threshold",  "ab", 60 },
    };
    for (size_t c = 0; c < sizeof cases / sizeof *cases; c++) {
        size_t u = strlen(cases[c].unit), n = 0;
        while (n + u < cases[c].len && n + u < sizeof(buf) - 1) {
            memcpy(buf + n, cases[c].unit, u);
            n += u;
        }
        buf[n] = 0;
        compare(t, cases[c].label, buf);
    }
    compare(t, "empty", "");
    compare(t, "single space", " ");
    compare(t, "newlines only", "\n\n\n\n");
    compare(t, "devanagari", "देवनागरी संयुक्ताक्षर परीक्षण");
    compare(t, "thai combining", "ก่อนอื่นเลย ทดสอบเครื่องหมาย");
    compare(t, "cjk", "日本語のテキストと漢字とひらがな");
    compare(t, "emoji zwj", "🏳️‍🌈👨‍👩‍👧‍👦🇸🇪");
}

static void run_fixture(const char *path) {
    gguf_file g;
    if (!gguf_open(&g, path)) {
        fprintf(stderr, "FAIL: cannot open %s\n", path);
        g_fail = 1;
        return;
    }
    tokenizer t;
    if (!tokenizer_init(&t, &g)) {
        fprintf(stderr, "FAIL: tokenizer_init on %s\n", path);
        gguf_close(&g);
        g_fail = 1;
        return;
    }
    g_fixture = path;

    int lines = 0;
    FILE *f = fopen("tests/fixtures/tokenizer-corpus.txt", "rb");
    if (f) {
        char line[1024];
        while (fgets(line, sizeof(line), f)) {
            size_t n = strlen(line);
            while (n > 0 && (line[n - 1] == '\n' || line[n - 1] == '\r')) line[--n] = 0;
            compare(&t, "corpus", line);
            lines++;
        }
        fclose(f);
    } else {
        fprintf(stderr, "FAIL: cannot read tests/fixtures/tokenizer-corpus.txt "
                        "(run from the repo root)\n");
        g_fail = 1;
    }
    adversarial(&t);
    printf("  %-44s %d corpus lines + adversarial shapes\n", path, lines);

    tokenizer_free(&t);
    gguf_close(&g);
}

int main(int argc, char **argv) {
    f16_init();
    static const char *const FIXTURES[] = {
        "tests/fixtures/vocab-spm.gguf",
        "tests/fixtures/vocab-spm-zeroscores.gguf",
        "tests/fixtures/vocab-bpe-llama3.gguf",
        "tests/fixtures/vocab-bpe-qwen2.gguf",
        "tests/fixtures/vocab-bpe-qwen35.gguf",
        "tests/fixtures/vocab-bpe-smollm.gguf",
        "tests/fixtures/vocab-bpe-spm-gemma4.gguf",
    };
    if (argc > 1) {
        for (int i = 1; i < argc; i++) run_fixture(argv[i]);
    } else {
        for (size_t i = 0; i < sizeof FIXTURES / sizeof *FIXTURES; i++)
            run_fixture(FIXTURES[i]);
    }
    if (g_fail) {
        fprintf(stderr, "merge-path equivalence FAILED\n");
        return 1;
    }
    puts("merge paths agree on every fixture");
    return 0;
}
