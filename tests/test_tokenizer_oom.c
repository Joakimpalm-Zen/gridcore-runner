// Allocation-failure tests for the tokenizer.
//
// Every buffer here is sized from GGUF input: vocabulary length, merge count,
// and the byte length of the text being encoded. Runner deliberately operates
// near memory limits (--reserve, multi-GB weights, hybrid GPU/CPU splits), so a
// failed allocation is a normal condition, not an exotic one — and a model
// loaded from an untrusted GGUF must fail cleanly rather than crash.
//
// tokenizer.c is compiled straight into this test with its allocators macro-
// substituted for instrumented ones, the same technique as test_json_oom.c.
// gguf.c/compat.c/quants.c are linked normally, so their allocations are not
// counted and gguf_open/gguf_close stay outside the failure window.
#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static long alloc_calls;   // allocations attempted since the last reset
static long alloc_live;    // outstanding blocks; must return to its baseline
static long fail_at = -1;  // index of the allocation to fail, -1 for none

static void *t_malloc(size_t n) {
    if (fail_at >= 0 && alloc_calls++ == fail_at) return NULL;
    if (fail_at < 0) alloc_calls++;
    void *p = malloc(n);
    if (p) alloc_live++;
    return p;
}

static void *t_calloc(size_t a, size_t b) {
    if (fail_at >= 0 && alloc_calls++ == fail_at) return NULL;
    if (fail_at < 0) alloc_calls++;
    void *p = calloc(a, b);
    if (p) alloc_live++;
    return p;
}

// tokenizer.c allocates only with malloc/calloc; a realloc added later would go
// uncounted, and t_free would then drive alloc_live negative — which the
// baseline checks below flag just as loudly as a leak.
static void t_free(void *p) {
    if (p) alloc_live--;
    free(p);
}

#define malloc  t_malloc
#define calloc  t_calloc
#define free    t_free
#include "../src/tokenizer.c"
#undef malloc
#undef calloc
#undef free

// The SPM fixtures carry the real 32000-piece TinyLlama vocab; the zeroscores
// one additionally drives the rebuild-scores-from-merges path. The BPE fixtures
// add the merge hashmap and the byte-level encode path.
static const char *const fixtures[] = {
    "tests/fixtures/vocab-spm.gguf",
    "tests/fixtures/vocab-spm-zeroscores.gguf",
    "tests/fixtures/vocab-bpe-llama3.gguf",
    "tests/fixtures/vocab-bpe-qwen2.gguf",
    "tests/fixtures/vocab-bpe-smollm.gguf",
};

static void open_fixture(gguf_file *g, const char *path) {
    if (!gguf_open(g, path)) {
        fprintf(stderr, "cannot open fixture %s (run from the repo root)\n", path);
        exit(1);
    }
}

// tokenizer_init already reports failure as `false` and both callers (main.c,
// server.c) already act on it, so a failed allocation has a path home. What it
// must never do is report success on a half-built tokenizer: hmap_init used to
// publish a nonzero capacity even when its calloc failed, after which the very
// next hmap_put wrote through a NULL table.
static void test_init_survives_allocation_failure(void) {
    for (size_t i = 0; i < sizeof(fixtures) / sizeof(*fixtures); i++) {
        gguf_file g;
        open_fixture(&g, fixtures[i]);

        fail_at = -1;
        alloc_calls = 0;
        alloc_live = 0;
        tokenizer t;
        assert(tokenizer_init(&t, &g));   // the fixture itself must load
        long total = alloc_calls;
        assert(total > 0);
        tokenizer_free(&t);
        assert(alloc_live == 0);          // a clean init/free must not leak

        for (long k = 0; k < total; k++) {
            fail_at = k;
            alloc_calls = 0;
            alloc_live = 0;
            tokenizer u;
            bool ok = tokenizer_init(&u, &g);
            tokenizer_free(&u);           // must be safe either way
            if (alloc_live != 0) {
                fprintf(stderr, "%s: failing allocation %ld of %ld: %ld block(s) "
                                "leaked\n", fixtures[i], k, total, alloc_live);
                abort();
            }
            if (ok) {
                fprintf(stderr, "%s: failing allocation %ld of %ld: init reported "
                                "success on a half-built tokenizer\n",
                        fixtures[i], k, total);
                abort();
            }
        }
        fail_at = -1;
        gguf_close(&g);
    }
}

// Texts chosen to reach every allocating encode path: the SPM symbol list and
// its U+2581 rewrite buffer, the BPE codepoint/offset arrays and byte-mapped
// word buffer, and bpe_word's two symbol arrays. Multi-byte and special-token
// input keeps the segment loop running more than once.
static const char *const texts[] = {
    "hello",
    "The quick brown fox jumps over the lazy dog, 1234567890 times.",
    "caf\xC3\xA9 \xF0\x9F\xA6\x99 llama",
    "</s>tokenization/end.\n\nI'll go",
    "  12",
};

// tok_encode's error channel (RNR-006): if an encode helper has to drop a text
// segment because a temporary allocation failed, the whole call returns -1
// rather than a silently-shorter token count that every caller would mistake
// for a legitimately short prompt. So under injected allocation failure the
// result must be EITHER the full clean count (the failed allocation did not
// affect the encode) OR exactly -1 — never a positive partial. It must also
// not crash or strand the buffers it did allocate.
static void test_encode_survives_allocation_failure(void) {
    for (size_t i = 0; i < sizeof(fixtures) / sizeof(*fixtures); i++) {
        gguf_file g;
        open_fixture(&g, fixtures[i]);

        fail_at = -1;
        alloc_calls = 0;
        alloc_live = 0;
        tokenizer t;
        assert(tokenizer_init(&t, &g));
        long base_live = alloc_live;   // the tokenizer's own blocks stay live

        for (size_t x = 0; x < sizeof(texts) / sizeof(*texts); x++) {
            int32_t ids[256];
            const int cap = (int)(sizeof(ids) / sizeof(*ids));

            fail_at = -1;
            alloc_calls = 0;
            int want_n = tok_encode(&t, texts[x], ids, cap, true, true);
            long total = alloc_calls;
            assert(alloc_live == base_live);   // a clean encode must not leak

            for (long k = 0; k < total; k++) {
                fail_at = k;
                alloc_calls = 0;
                int32_t got[256];
                int n = tok_encode(&t, texts[x], got, cap, true, true);
                if (alloc_live != base_live) {
                    fprintf(stderr, "%s: encode(\"%s\"), failing allocation %ld of "
                                    "%ld: %ld block(s) leaked\n", fixtures[i],
                            texts[x], k, total, alloc_live - base_live);
                    abort();
                }
                if (n != want_n && n != -1) {
                    fprintf(stderr, "%s: encode(\"%s\"), failing allocation %ld of "
                                    "%ld: produced %d tokens — a silent partial; the "
                                    "contract is the full %d or -1\n", fixtures[i],
                            texts[x], k, total, n, want_n);
                    abort();
                }
            }
            fail_at = -1;

            // and the success path is untouched: same call, same ids
            int again = tok_encode(&t, texts[x], ids, cap, true, true);
            assert(again == want_n);
        }

        tokenizer_free(&t);
        assert(alloc_live == 0);
        gguf_close(&g);
    }
}

int main(void) {
    test_init_survives_allocation_failure();
    test_encode_survives_allocation_failure();
    puts("tokenizer oom tests ok");
    return 0;
}
