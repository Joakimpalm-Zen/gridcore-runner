// Allocation-failure tests for the JSON-Schema compiler.
//
// schema_compile is driven by response_format.json_schema out of an untrusted
// HTTP request body: the property counts, enum sizes, key strings and nesting
// depth are all attacker-chosen, and each of them sizes an allocation. Runner
// deliberately operates near memory limits, so a failed allocation is a normal
// condition — an unchecked one here is a remote crash.
//
// schema.c and json.c are compiled straight into this test with their
// allocators macro-substituted, the same technique as test_json_oom.c. json.c
// has to come along because the compiler serialises enum/const literals through
// jv_dump/sbuf, so a failure inside the builder is a schema.c failure path too.
// jsonmode.c is linked normally: it never allocates, and only the validator
// (not the compiler under test here) calls into it.
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

static void *t_realloc(void *p, size_t n) {
    if (fail_at >= 0 && alloc_calls++ == fail_at) return NULL;
    if (fail_at < 0) alloc_calls++;
    void *q = realloc(p, n);
    if (q && !p) alloc_live++;   // realloc(NULL, n) behaves as malloc
    return q;
}

// strdup must be instrumented too, not merely for the failure injection: its
// blocks are released through schema_free's (substituted) free, so leaving it
// alone would unbalance the counter and hide real leaks.
static char *t_strdup(const char *s) {
    size_t n = strlen(s) + 1;
    char *p = t_malloc(n);
    if (p) memcpy(p, s, n);
    return p;
}

static void t_free(void *p) {
    if (p) alloc_live--;
    free(p);
}

#define malloc  t_malloc
#define calloc  t_calloc
#define realloc t_realloc
#define strdup  t_strdup
#define free    t_free
#include "../src/json.c"
#include "../src/schema.c"
#undef malloc
#undef calloc
#undef realloc
#undef strdup
#undef free

// One schema per compile path: plain object with required keys, arrays with
// bounds, enum and const literals (which run through jv_dump), a scalar-const
// oneOf, a first-byte-dispatched union, a `type` array union, the discriminated
// tool/args shape compile_discriminated_action recognises, and deep nesting.
static const char *const schemas[] = {
    "{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"string\"},"
        "\"b\":{\"type\":\"integer\"}},\"required\":[\"a\",\"b\"]}",
    "{\"type\":\"array\",\"items\":{\"type\":\"string\",\"minLength\":2,"
        "\"maxLength\":8},\"minItems\":1,\"maxItems\":4}",
    "{\"type\":\"string\",\"enum\":[\"alpha\",\"beta\",\"gamma\"]}",
    "{\"const\":\"fixed\"}",
    "{\"oneOf\":[{\"const\":\"x\"},{\"const\":\"y\"},{\"const\":3}]}",
    "{\"anyOf\":[{\"type\":\"string\"},{\"type\":\"integer\"},{\"type\":\"null\"}]}",
    "{\"type\":[\"string\",\"null\"]}",
    "{\"oneOf\":["
        "{\"properties\":{\"tool\":{\"const\":\"read\"},"
            "\"args\":{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"}}}},"
         "\"required\":[\"tool\",\"args\"]},"
        "{\"properties\":{\"tool\":{\"const\":\"write\"},"
            "\"args\":{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"}}}},"
         "\"required\":[\"tool\",\"args\"]}]}",
    "{\"type\":\"object\",\"properties\":{\"deep\":{\"type\":\"object\","
        "\"properties\":{\"deeper\":{\"type\":\"array\",\"items\":"
        "{\"type\":\"object\",\"properties\":{\"leaf\":{\"enum\":[1,2,3]}}}}}}}}",
};

// Every allocation on the compile path is failed in turn. schema_compile may
// succeed or return NULL, but it must never crash and never strand a block —
// and when it does fail it must say so, because both callers (main.c and
// server.c) print that message and would otherwise report an empty reason.
static void test_compile_survives_allocation_failure(void) {
    for (size_t i = 0; i < sizeof(schemas) / sizeof(*schemas); i++) {
        const char *s = schemas[i];

        fail_at = -1;
        alloc_calls = 0;
        alloc_live = 0;
        jv *doc = json_parse(s, strlen(s));
        assert(doc != NULL);              // the schema text itself must be valid
        long base_live = alloc_live;      // the parsed document stays live

        char err[256];
        alloc_calls = 0;
        snode *n = schema_compile(doc, err, sizeof(err));
        if (!n) {
            fprintf(stderr, "schema %zu does not compile cleanly: %s\n", i, err);
            abort();
        }
        long total = alloc_calls;
        assert(total > 0);
        schema_free(n);
        assert(alloc_live == base_live);  // a clean compile must not leak

        for (long k = 0; k < total; k++) {
            fail_at = k;
            alloc_calls = 0;
            err[0] = 0;
            snode *r = schema_compile(doc, err, sizeof(err));
            if (!r && err[0] == 0) {
                fprintf(stderr, "schema %zu, failing allocation %ld of %ld: "
                                "rejected with an empty error message\n",
                        i, k, total);
                abort();
            }
            schema_free(r);
            if (alloc_live != base_live) {
                fprintf(stderr, "schema %zu, failing allocation %ld of %ld: "
                                "%ld block(s) leaked\n", i, k, total,
                        alloc_live - base_live);
                abort();
            }
        }
        fail_at = -1;

        jv_free(doc);
        assert(alloc_live == 0);
    }
}

int main(void) {
    test_compile_survives_allocation_failure();
    puts("schema oom tests ok");
    return 0;
}
