// Allocation-failure tests for the JSON parser and string builder.
//
// Runner deliberately operates near memory limits (--reserve, multi-GB weights,
// hybrid GPU/CPU splits), so a failed allocation is a normal condition here, not
// an exotic one. json.c parses untrusted HTTP request bodies, which makes every
// unchecked allocation on that path a remote crash.
//
// json.c is self-contained (json.h plus libc), so it is compiled straight into
// this test with the allocators macro-substituted for instrumented ones. The
// system headers are included first so the macros rewrite only json.c's calls
// and never the declarations in <stdlib.h>.
#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static long alloc_calls;   // allocations attempted since the last reset
static long alloc_live;    // outstanding blocks; must return to 0
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

static void *t_realloc(void *p, size_t n) {
    if (fail_at >= 0 && alloc_calls++ == fail_at) return NULL;
    if (fail_at < 0) alloc_calls++;
    void *q = realloc(p, n);
    if (q && !p) alloc_live++;   // realloc(NULL, n) behaves as malloc
    return q;
}

static void t_free(void *p) {
    if (p) alloc_live--;
    free(p);
}

#define malloc  t_malloc
#define calloc  t_calloc
#define realloc t_realloc
#define free    t_free
#include "../src/json.c"
#undef malloc
#undef calloc
#undef realloc
#undef free

// Inputs chosen to drive every allocating path: string growth past the initial
// 32-byte buffer, escape decoding, surrogate pairs, array and object element
// appends, and deep nesting.
static const char *const inputs[] = {
    "\"short\"",
    "\"a string comfortably longer than the thirty-two byte initial buffer so realloc runs\"",
    "\"escapes \\n\\t\\\" \\\\ \\u00e9 \\ud83e\\udd99 done\"",
    "[1,2,3,4,5,6,7,8,9,10]",
    "{\"a\":1,\"b\":2,\"c\":3,\"d\":[1,2,3],\"e\":{\"f\":\"g\"}}",
    "[[[[[[[[1]]]]]]]]",
    "{\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"x\",\"parameters\":{}}}]}",
};

// Every allocation on the parse path is failed in turn. The parser may succeed
// or return NULL, but it must never crash and must never strand a block: the
// `p = realloc(p, n)` idiom loses the original pointer, which shows up here as
// alloc_live != 0.
static void test_parse_survives_allocation_failure(void) {
    for (size_t i = 0; i < sizeof(inputs) / sizeof(*inputs); i++) {
        const char *s = inputs[i];
        size_t n = strlen(s);

        fail_at = -1;
        alloc_calls = 0;
        alloc_live = 0;
        jv *v = json_parse(s, n);
        assert(v != NULL);          // the input itself must be valid
        long total = alloc_calls;
        jv_free(v);
        assert(alloc_live == 0);    // clean parse must not leak

        for (long k = 0; k < total; k++) {
            fail_at = k;
            alloc_calls = 0;
            alloc_live = 0;
            jv *r = json_parse(s, n);
            if (r) jv_free(r);
            if (alloc_live != 0) {
                fprintf(stderr, "input %zu, failing allocation %ld of %ld: "
                                "%ld block(s) leaked\n", i, k, total, alloc_live);
                abort();
            }
        }
        fail_at = -1;
    }
}

// The same for the builder that assembles every HTTP response body. sb_put
// returns void, so a caller cannot see failure; at minimum it must not write
// through a NULL buffer.
static void test_builder_survives_allocation_failure(void) {
    fail_at = -1;
    alloc_calls = 0;
    alloc_live = 0;
    sbuf b = { 0 };
    for (int i = 0; i < 40; i++) sb_lit(&b, "chunk of response body text ");
    sb_esc(&b, "needs \"escaping\" \n", 18);
    sb_fmt(&b, "%d %s", 42, "formatted");
    long total = alloc_calls;
    t_free(b.s);
    assert(alloc_live == 0);

    for (long k = 0; k < total; k++) {
        fail_at = k;
        alloc_calls = 0;
        alloc_live = 0;
        sbuf c = { 0 };
        for (int i = 0; i < 40; i++) sb_lit(&c, "chunk of response body text ");
        sb_esc(&c, "needs \"escaping\" \n", 18);
        sb_fmt(&c, "%d %s", 42, "formatted");
        t_free(c.s);
        if (alloc_live != 0) {
            fprintf(stderr, "builder, failing allocation %ld of %ld: "
                            "%ld block(s) leaked\n", k, total, alloc_live);
            abort();
        }
    }
    fail_at = -1;
}

int main(void) {
    test_parse_survives_allocation_failure();
    test_builder_survives_allocation_failure();
    puts("json oom tests ok");
    return 0;
}
