#include "runner.h"
#include "json.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void test_strict_bounded_numbers(void) {
    // 1e400/-1e400 overflow to inf: isfinite() catches that only when the
    // build is not -ffast-math, which the release build is
    const char *bad[] = { "01", "1.", "-.1", "1e", "-nan", "1e9999",
                          "1e400", "-1e400", "1e308000" };
    for (size_t i = 0; i < sizeof(bad) / sizeof(*bad); i++)
        assert(json_parse(bad[i], strlen(bad[i])) == NULL);

    const char bounded[] = { '1', '2' }; // deliberately not NUL-terminated
    jv *v = json_parse(bounded, 1);
    assert(v != NULL);
    assert(v->type == J_NUM && v->num == 1.0);
    jv_free(v);

    v = json_parse("-1.25e+3", 8);
    assert(v != NULL);
    assert(v->type == J_NUM && v->num == -1250.0);
    jv_free(v);
}

static void test_json_rejects_unpaired_utf16_surrogates(void) {
    const char *bad[] = {
        "\"\\uD800\"",          // lone high surrogate
        "\"\\uD800x\"",         // high surrogate followed by ordinary text
        "\"\\uD800\\u0041\"",  // high surrogate followed by a non-low escape
        "\"\\uDC00\"",          // lone low surrogate
    };
    for (size_t i = 0; i < sizeof(bad) / sizeof(*bad); i++)
        assert(json_parse(bad[i], strlen(bad[i])) == NULL);

    jv *paired = json_parse("\"\\uD83D\\uDE00\"", 14);
    assert(paired != NULL);
    assert(!strcmp(paired->str, "\xF0\x9F\x98\x80"));
    jv_free(paired);

    char out[4];
    int outn = -1;
    assert(json_unescape("\\uDC00", 6, out, &outn) == -1);
    assert(json_unescape("\\uD800x", 7, out, &outn) == -1);
    assert(json_unescape("\\uD800\\u0041", 12, out, &outn) == -1);
    assert(json_unescape("\\uD83D\\uDE00", 12, out, &outn) == 12);
    assert(outn == 4 && !memcmp(out, "\xF0\x9F\x98\x80", 4));
}

static void test_json_rejects_ill_formed_raw_utf8(void) {
    const char *bad[] = {
        "\"\x80\"",             // bare continuation
        "\"\xC0\x80\"",         // overlong NUL
        "\"\xED\xA0\x80\"",     // UTF-8 encoding of a surrogate
        "\"\xF4\x90\x80\x80\"", // beyond U+10FFFF
        "\"\xE2\x82\"",         // truncated sequence
    };
    for (size_t i = 0; i < sizeof(bad) / sizeof(*bad); i++)
        assert(json_parse(bad[i], strlen(bad[i])) == NULL);

    const char good[] = "\"caf\xC3\xA9 \xF0\x9F\x98\x80\"";
    jv *v = json_parse(good, strlen(good));
    assert(v != NULL);
    jv_free(v);
}

static void test_json_rejects_embedded_nul(void) {
    const char *bad[] = {
        "\"a\\u0000b\"",
        "{\"model\\u0000shadow\":1}",
    };
    for (size_t i = 0; i < sizeof(bad) / sizeof(*bad); i++)
        assert(json_parse(bad[i], strlen(bad[i])) == NULL);

    char out[4];
    int outn = -1;
    assert(json_unescape("\\u0000", 6, out, &outn) == -1);
}

static void test_json_rejects_duplicate_object_keys(void) {
    const char *bad[] = {
        "{\"max_tokens\":4,\"max_tokens\":4096}",
        "{\"outer\":{\"model\":\"safe\",\"model\":\"shadow\"}}",
        "{\"model\":1,\"m\\u006fdel\":2}",
    };
    for (size_t i = 0; i < sizeof(bad) / sizeof(*bad); i++)
        assert(json_parse(bad[i], strlen(bad[i])) == NULL);

    // Reusing a name in different objects is not a duplicate.
    const char *good = "[{\"id\":1},{\"id\":2}]";
    jv *v = json_parse(good, strlen(good));
    assert(v != NULL);
    jv_free(v);
}

static void test_json_close_partial_string(void) {
    jsonv v;
    jsonv_init(&v);
    assert(jsonv_feed(&v, "{\"a\":\"x", 7));
    char out[64];
    int n = jsonv_close(&v, out, sizeof(out));
    assert(n > 0);

    char full[128];
    snprintf(full, sizeof(full), "{\"a\":\"x%s", out);
    jv *parsed = json_parse(full, strlen(full));
    assert(parsed != NULL);
    jv_free(parsed);
}

static void test_schema_required_close(void) {
    const char *src =
        "{\"type\":\"object\",\"properties\":{"
        "\"a\":{\"type\":\"string\"},"
        "\"b\":{\"type\":\"integer\"}"
        "},\"required\":[\"a\",\"b\"]}";
    jv *schema_json = json_parse(src, strlen(src));
    assert(schema_json != NULL);
    char err[128];
    snode *schema = schema_compile(schema_json, err, sizeof(err));
    assert(schema != NULL);

    sval v;
    sval_init(&v, schema);
    assert(sval_feed(&v, "{\"a\":\"x\"", 8));
    char out[128];
    int n = sval_close(&v, out, sizeof(out));
    assert(n > 0);

    char full[256];
    snprintf(full, sizeof(full), "{\"a\":\"x\"%s", out);
    jv *parsed = json_parse(full, strlen(full));
    assert(parsed != NULL);
    assert(jv_get(parsed, "a") != NULL);
    assert(jv_get(parsed, "b") != NULL);

    jv_free(parsed);
    schema_free(schema);
    jv_free(schema_json);
}

// A constrained document must begin with its opening token.
//
// Leading whitespace is the livelock that burns the budget. A model that
// would rather write a preamble finds every prose token rejected, and the
// only legal bytes left are spaces and newlines — so it emits those, forever,
// and never reaches the opening brace. Measured on the stub model with
// tool_choice=required and max_tokens=32: 32 tab tokens and no document.
//
// Whitespace *inside* the document is untouched: it is how a real model
// pretty-prints, and Llama-3.2 emits `{ "a" : "b" }` spacing on every call.
// Only the run before the root value opens is banned, because that is the
// only position where burning whitespace yields a document with no content
// in it at all.
static void test_leading_whitespace_is_refused_but_interior_is_kept(void) {
    const char *src =
        "{\"type\":\"object\",\"properties\":{"
        "\"a\":{\"type\":\"string\"},"
        "\"b\":{\"type\":\"array\",\"items\":{\"type\":\"string\"}}"
        "},\"required\":[\"a\",\"b\"]}";
    jv *schema_json = json_parse(src, strlen(src));
    assert(schema_json != NULL);
    char err[128];
    snode *schema = schema_compile(schema_json, err, sizeof(err));
    assert(schema != NULL);

    sval v;
    // every flavour of leading whitespace is refused outright
    const char *ws[] = { " ", "\n", "\t", "\r", "\n\n  " };
    for (size_t i = 0; i < sizeof(ws) / sizeof(*ws); i++) {
        sval_init(&v, schema);
        assert(!sval_feed(&v, ws[i], (int)strlen(ws[i])));
    }

    // the opening token itself is of course still fine
    sval_init(&v, schema);
    assert(sval_feed(&v, "{", 1));

    // ...and interior whitespace, in every position a model uses it, is
    // accepted exactly as before: this is Llama-3.2's real output shape
    const char *pretty =
        "{\n  \"a\" : \"x\" ,\n  \"b\" : [ \"p\" , \"q\" ]\n}";
    sval_init(&v, schema);
    assert(sval_feed(&v, pretty, (int)strlen(pretty)));
    assert(v.done);

    schema_free(schema);
    jv_free(schema_json);
}

// json mode (no schema) draws the same line through the same states.
static void test_json_mode_leading_whitespace_is_refused(void) {
    jsonv v;
    jsonv_init(&v);
    assert(!jsonv_feed(&v, " ", 1));
    jsonv_init(&v);
    assert(!jsonv_feed(&v, "\n\n", 2));

    jsonv_init(&v);
    const char *pretty = "{\n  \"a\" : [ 1 , 2 ] ,\n  \"b\" : null\n}";
    assert(jsonv_feed(&v, pretty, (int)strlen(pretty)));
    assert(v.done);
}

// A close() with no generated payload behind it used to invent a complete,
// schema-valid document out of nothing: `{"progress":"","next_step":""}`.
// That is indistinguishable from an answer the model actually produced, so a
// caller could not tell that its whole token budget had gone on a reasoning
// prelude that never reached the opening brace. Measured on Qwen3-4B, this
// silently replaced Thane's compaction summaries with empty session state.
//
// jsonv_close has always declined to fabricate ("returns 0 if generation
// never started an object"); sval_close must honour the same contract, so an
// unproductive decode fails visibly instead of parsing as a real answer.
static void test_schema_close_without_payload_fabricates_nothing(void) {
    const char *src =
        "{\"type\":\"object\",\"properties\":{"
        "\"progress\":{\"type\":\"string\"},"
        "\"next_step\":{\"type\":\"string\"}"
        "},\"required\":[\"progress\",\"next_step\"]}";
    jv *schema_json = json_parse(src, strlen(src));
    assert(schema_json != NULL);
    char err[128];
    snode *schema = schema_compile(schema_json, err, sizeof(err));
    assert(schema != NULL);

    char out[256];

    // nothing fed at all
    sval v;
    sval_init(&v, schema);
    out[0] = 'x';
    assert(sval_close(&v, out, sizeof(out)) == 0);
    assert(out[0] == 0);

    // the moment the document opens, truncation must still complete it
    sval_init(&v, schema);
    assert(sval_feed(&v, "{", 1));
    int n = sval_close(&v, out, sizeof(out));
    assert(n > 0);
    char full[512];
    snprintf(full, sizeof(full), "{%s", out);
    jv *parsed = json_parse(full, strlen(full));
    assert(parsed != NULL);
    assert(jv_get(parsed, "progress") != NULL);
    assert(jv_get(parsed, "next_step") != NULL);
    jv_free(parsed);

    schema_free(schema);
    jv_free(schema_json);
}

// An empty "type" array compiled to a union with no alternatives. pick_alt
// then matched no byte, so sampling stalled and the forced-completion path
// read alts[0] out of a zero-byte allocation: a single unauthenticated
// request body segfaulted the whole server, taking every slot with it.
// Reachable nested as well, so the check belongs at the compile site.
static void test_schema_rejects_empty_type_union(void) {
    static const char *const bad[] = {
        "{\"type\":[]}",
        "{\"type\":\"array\",\"items\":{\"type\":[]},\"minItems\":1}",
        "{\"properties\":{\"a\":{\"type\":[]}},\"required\":[\"a\"]}",
    };
    for (size_t i = 0; i < sizeof(bad) / sizeof(*bad); i++) {
        jv *j = json_parse(bad[i], strlen(bad[i]));
        assert(j != NULL);
        char err[128];
        snode *schema = schema_compile(j, err, sizeof(err));
        assert(schema == NULL);
        jv_free(j);
    }
}

// minItems/minLength are bounded only by INT_MAX, and the completion path
// looped to that bound even though the output buffer had long since filled.
// Nested, this pinned a slot for minutes. The loops now stop when the buffer
// is full, so an absurd bound costs the same as a small one.
static void test_schema_huge_min_bounds_terminate(void) {
    static const char *const src[] = {
        "{\"type\":\"string\",\"minLength\":2000000000}",
        "{\"type\":\"array\",\"minItems\":2000000000,"
        "\"items\":{\"type\":\"array\",\"minItems\":2000000000,"
        "\"items\":{\"type\":\"integer\"}}}",
    };
    for (size_t i = 0; i < sizeof(src) / sizeof(*src); i++) {
        jv *j = json_parse(src[i], strlen(src[i]));
        assert(j != NULL);
        char err[128];
        snode *schema = schema_compile(j, err, sizeof(err));
        assert(schema != NULL);   // the schema itself is legal
        sval v;
        sval_init(&v, schema);
        char out[4096];
        int n = sval_close(&v, out, sizeof(out));   // must return, not spin
        assert(n >= 0 && n < (int)sizeof(out));
        schema_free(schema);
        jv_free(j);
    }
}

static void test_schema_rejects_bad_bounds(void) {
    const char *src =
        "{\"type\":\"array\",\"items\":{\"type\":\"string\"},"
        "\"minItems\":2,\"maxItems\":1}";
    jv *schema_json = json_parse(src, strlen(src));
    assert(schema_json != NULL);
    char err[128];
    snode *schema = schema_compile(schema_json, err, sizeof(err));
    assert(schema == NULL);
    assert(strstr(err, "bounds") != NULL);
    jv_free(schema_json);
}

static void test_schema_rejects_non_integer_or_huge_bounds(void) {
    const char *bad[] = {
        "{\"type\":\"array\",\"items\":{},\"minItems\":1.5}",
        "{\"type\":\"array\",\"items\":{},\"maxItems\":1e100}",
        "{\"type\":\"string\",\"minLength\":1e100}",
        "{\"type\":\"number\",\"minimum\":\"zero\"}",
        "{\"type\":\"integer\",\"exclusiveMaximum\":null}",
    };
    for (size_t i = 0; i < sizeof(bad) / sizeof(*bad); i++) {
        jv *schema_json = json_parse(bad[i], strlen(bad[i]));
        assert(schema_json != NULL);
        char err[128];
        snode *schema = schema_compile(schema_json, err, sizeof(err));
        assert(schema == NULL);
        assert(strstr(err, "bounds") != NULL);
        jv_free(schema_json);
    }
}

static void test_schema_rejects_escaped_keys(void) {
    const char *src =
        "{\"type\":\"object\",\"properties\":{\"bad\\\"key\":{\"type\":\"string\"}}}";
    jv *schema_json = json_parse(src, strlen(src));
    assert(schema_json != NULL);
    char err[128];
    snode *schema = schema_compile(schema_json, err, sizeof(err));
    assert(schema == NULL);
    assert(strstr(err, "property keys") != NULL);
    jv_free(schema_json);
}

// A keyword the compiler cannot enforce must be a compile error, not a
// silently weaker constraint.
static void test_schema_rejects_unenforceable_keywords(void) {
    static const char *const bad[] = {
        "{\"type\":\"integer\",\"multipleOf\":2}",
        "{\"allOf\":[{\"type\":\"string\"}]}",
        "{\"not\":{\"type\":\"string\"}}",
        "{\"$ref\":\"#/$defs/x\"}",
        "{\"type\":\"array\",\"uniqueItems\":true}",
        "{\"type\":\"array\",\"prefixItems\":[{\"type\":\"string\"}]}",
        "{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"string\"}},"
            "\"patternProperties\":{\"^x\":{\"type\":\"string\"}}}",
        // keyword that belongs to a different type than the one declared
        "{\"type\":\"string\",\"minItems\":2}",
        "{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"string\"}},"
            "\"maxLength\":3}",
    };
    for (size_t i = 0; i < sizeof(bad) / sizeof(*bad); i++) {
        jv *schema_json = json_parse(bad[i], strlen(bad[i]));
        assert(schema_json != NULL);
        char err[128];
        snode *schema = schema_compile(schema_json, err, sizeof(err));
        assert(schema == NULL);
        assert(strstr(err, "keyword") != NULL);
        jv_free(schema_json);
    }
}

// Bounded repetition: {n} and {n,m}. The unbounded forms (`+`, `{n,}`) only
// ever needed a floor, so nothing enforced a ceiling; these do, and the ceiling
// has to be enforced during decoding rather than at the closing quote, or the
// model is free to emit an over-long tail and only discover it is invalid once
// it is too late to take back.
static void test_schema_bounded_repetition(void) {
    struct { const char *pat; const char *ok; const char *too_short;
             const char *too_long; } cases[] = {
        // exactly six
        { "^wf_[a-z0-9-]{6}$",   "\"wf_abc123\"", "\"wf_abc12\"", "\"wf_abc1234\"" },
        // two to four
        { "^id-[0-9]{2,4}$",     "\"id-123\"",    "\"id-1\"",     "\"id-12345\"" },
        // a degenerate but legal range: exactly one
        { "^v[0-9]{1,1}$",       "\"v7\"",        "\"v\"",        "\"v77\"" },
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(*cases); i++) {
        char src[128];
        snprintf(src, sizeof(src), "{\"type\":\"string\",\"pattern\":\"%s\"}",
                 cases[i].pat);
        jv *j = json_parse(src, strlen(src));
        char err[128];
        snode *schema = schema_compile(j, err, sizeof(err));
        assert(schema != NULL);

        sval v; sval_init(&v, schema);
        assert(sval_feed(&v, cases[i].ok, (int)strlen(cases[i].ok)) && v.done);

        sval s; sval_init(&s, schema);
        assert(!sval_feed(&s, cases[i].too_short, (int)strlen(cases[i].too_short)));

        // the over-long case must be refused ON THE OFFENDING BYTE, not at the
        // closing quote -- that is the whole point of a decode-time ceiling
        sval l; sval_init(&l, schema);
        assert(!sval_feed(&l, cases[i].too_long, (int)strlen(cases[i].too_long)));

        // and a forced completion of a partial string must itself be valid
        sval p; sval_init(&p, schema);
        assert(sval_feed(&p, "\"", 1));
        char suffix[64];
        int nfix = sval_close(&p, suffix, sizeof(suffix));
        assert(nfix > 0);
        char full[128]; snprintf(full, sizeof(full), "\"%s", suffix);
        sval chk; sval_init(&chk, schema);
        assert(sval_feed(&chk, full, (int)strlen(full)) && chk.done);

        schema_free(schema); jv_free(j);
    }

    // an upper bound below the lower bound is an empty language: refuse it at
    // compile time rather than accept a schema no output can ever satisfy
    const char *empty[] = {
        "{\"type\":\"string\",\"pattern\":\"^x[a-z]{5,2}$\"}",
        // minLength that the bounded pattern can never reach
        "{\"type\":\"string\",\"pattern\":\"^x[a-z]{2}$\",\"minLength\":9}",
        // malformed quantifiers stay errors rather than being reinterpreted
        "{\"type\":\"string\",\"pattern\":\"^x[a-z]{}$\"}",
        "{\"type\":\"string\",\"pattern\":\"^x[a-z]{2,3,4}$\"}",
        "{\"type\":\"string\",\"pattern\":\"^x[a-z]{,3}$\"}",
    };
    for (size_t i = 0; i < sizeof(empty) / sizeof(*empty); i++) {
        jv *j = json_parse(empty[i], strlen(empty[i]));
        char err[128];
        err[0] = 0;
        snode *schema = schema_compile(j, err, sizeof(err));
        assert(schema == NULL);
        assert(err[0] != 0);
        jv_free(j);
    }

    // the unbounded forms must keep behaving exactly as before
    const char *unbounded = "{\"type\":\"string\",\"pattern\":\"^n[0-9]{2,}$\"}";
    jv *j = json_parse(unbounded, strlen(unbounded));
    char err[128];
    snode *schema = schema_compile(j, err, sizeof(err));
    assert(schema != NULL);
    const char *long_ok = "\"n0123456789012345\"";
    sval v; sval_init(&v, schema);
    assert(sval_feed(&v, long_ok, (int)strlen(long_ok)) && v.done);
    schema_free(schema); jv_free(j);
}

// A pattern with more than one repeated class was refused outright, so
// `^[A-Z]{3}[0-9]{4}$` -- an ordinary ticket/serial shape -- could not be
// enforced at all. Multiple segments are accepted when every segment before
// the last is FIXED-length: the position of each class is then determined by
// the byte offset alone, so the enforced language is still exactly the
// declared one. A variable-length segment in the middle stays refused,
// because two different segment splits could explain the same prefix and
// this compiler does not guess.
static void test_schema_multi_segment_pattern(void) {
    struct { const char *pat; const char *ok; const char *bad1;
             const char *bad2; } cases[] = {
        // two adjacent classes, the motivating case
        { "^[A-Z]{3}[0-9]{4}$", "\"ABC1234\"", "\"ABCD234\"", "\"ABC12345\"" },
        // a literal separator between them
        { "^[A-Z]{2}-[0-9]{3}$", "\"AB-123\"", "\"AB:123\"", "\"AB-12\"" },
        // shorthand classes, twice
        { "^\\\\d{3}-\\\\d{2}$", "\"123-45\"", "\"12a-45\"", "\"123-4\"" },
        // a trailing literal after the last class
        { "^[0-9]{2}X$", "\"12X\"", "\"12Y\"", "\"123X\"" },
        // only the LAST segment may be variable
        { "^[A-Z]{2}[0-9]{2,4}$", "\"AB1234\"", "\"AB1\"", "\"AB12345\"" },
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(*cases); i++) {
        char src[160];
        snprintf(src, sizeof(src), "{\"type\":\"string\",\"pattern\":\"%s\"}",
                 cases[i].pat);
        jv *j = json_parse(src, strlen(src));
        char err[128];
        snode *schema = schema_compile(j, err, sizeof(err));
        assert(schema != NULL);

        sval v; sval_init(&v, schema);
        assert(sval_feed(&v, cases[i].ok, (int)strlen(cases[i].ok)) && v.done);

        sval a; sval_init(&a, schema);
        assert(!sval_feed(&a, cases[i].bad1, (int)strlen(cases[i].bad1)));
        sval b; sval_init(&b, schema);
        assert(!sval_feed(&b, cases[i].bad2, (int)strlen(cases[i].bad2)));

        // a forced close mid-string must still produce a legal document
        sval p2; sval_init(&p2, schema);
        assert(sval_feed(&p2, "\"", 1));
        char suffix[80];
        int nfix = sval_close(&p2, suffix, sizeof(suffix));
        assert(nfix > 0);
        char full[160]; snprintf(full, sizeof(full), "\"%s", suffix);
        sval chk; sval_init(&chk, schema);
        assert(sval_feed(&chk, full, (int)strlen(full)) && chk.done);

        schema_free(schema); jv_free(j);
    }

    // a variable-length segment before the last one is ambiguous: refused
    const char *ambiguous[] = {
        "{\"type\":\"string\",\"pattern\":\"^[A-Z]{1,3}[0-9]{2}$\"}",
        "{\"type\":\"string\",\"pattern\":\"^[A-Z]+[0-9]{2}$\"}",
    };
    for (size_t i = 0; i < sizeof(ambiguous) / sizeof(*ambiguous); i++) {
        jv *j = json_parse(ambiguous[i], strlen(ambiguous[i]));
        char err[128];
        assert(schema_compile(j, err, sizeof(err)) == NULL);
        assert(strstr(err, "pattern") != NULL);
        jv_free(j);
    }
}

static void test_schema_agent_id_pattern_is_enforced(void) {
    const char *src =
        "{\"type\":\"string\",\"pattern\":\"^wf_[a-z0-9-]{6,}$\"}";
    jv *j = json_parse(src, strlen(src));
    char err[128];
    snode *schema = schema_compile(j, err, sizeof(err));
    assert(schema != NULL);
    const char *good[] = { "\"wf_abc-19\"", "\"wf_123456\"" };
    for (size_t i = 0; i < sizeof(good) / sizeof(*good); i++) {
        sval v; sval_init(&v, schema);
        assert(sval_feed(&v, good[i], (int)strlen(good[i])) && v.done);
    }
    const char *bad[] = { "\"xx_abc123\"", "\"wf_abc\"", "\"wf_ABC123\"" };
    for (size_t i = 0; i < sizeof(bad) / sizeof(*bad); i++) {
        sval v; sval_init(&v, schema);
        assert(!sval_feed(&v, bad[i], (int)strlen(bad[i])));
    }
    sval partial; sval_init(&partial, schema);
    assert(sval_feed(&partial, "\"wf_a", 5));
    char suffix[32];
    int n = sval_close(&partial, suffix, sizeof(suffix));
    assert(n > 0);
    char full[64]; snprintf(full, sizeof(full), "\"wf_a%s", suffix);
    sval check; sval_init(&check, schema);
    assert(sval_feed(&check, full, (int)strlen(full)) && check.done);
    schema_free(schema); jv_free(j);

    src = "{\"type\":\"string\","
          "\"pattern\":\"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$\"}";
    j = json_parse(src, strlen(src));
    schema = schema_compile(j, err, sizeof(err));
    assert(schema != NULL);
    sval protocol; sval_init(&protocol, schema);
    assert(sval_feed(&protocol, "\"proto-1\"", 9) && protocol.done);
    schema_free(schema); jv_free(j);
}

static void test_schema_pattern_shorthand_classes(void) {
    // ^\d{5}$ — postal codes, order numbers, IDs — is the most common
    // fixed-length pattern there is, and it used to be refused over a
    // spelling: the compiler demanded a bracket set, so [0-9]{5} compiled and
    // \d{5} did not. \d and \w now expand to exactly the ASCII sets their
    // bracket equivalents would, in every position the class is legal.
    struct { const char *pat; const char *good; const char *bad; } cases[] = {
        { "^\\\\d{5}$",     "\"12345\"",      "\"1234a\""      },
        { "^\\\\d{5}$",     "\"00000\"",      "\"123456\""     },
        { "^\\\\w{3}$",     "\"a_9\"",        "\"a-9\""        },
        { "^\\\\w{3}$",     "\"ABC\"",        "\"AB\""         },
        { "^ORD-\\\\d{6}$", "\"ORD-100000\"", "\"XRD-100000\"" },
        { "^\\\\d{2,4}$",   "\"123\"",        "\"12345\""      },
        { "^\\\\d+$",       "\"7\"",          "\"x\""          },
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(*cases); i++) {
        char src[160];
        snprintf(src, sizeof(src), "{\"type\":\"string\",\"pattern\":\"%s\"}",
                 cases[i].pat);
        jv *j = json_parse(src, strlen(src));
        assert(j != NULL);
        char err[128];
        snode *schema = schema_compile(j, err, sizeof(err));
        assert(schema != NULL);
        sval good; sval_init(&good, schema);
        assert(sval_feed(&good, cases[i].good, (int)strlen(cases[i].good)) &&
               good.done);
        sval bad; sval_init(&bad, schema);
        // either rejected outright, or accepted-but-not-complete; both mean
        // the declared language is being enforced rather than waved through
        assert(!(sval_feed(&bad, cases[i].bad, (int)strlen(cases[i].bad)) &&
                 bad.done));
        schema_free(schema); jv_free(j);
    }
    // \s is NOT supported, on purpose: it includes tab/newline/CR, and JSON
    // forbids raw control characters inside a string, so a grammar emitting
    // them would produce output the caller cannot parse. Narrowing it to
    // "space" would enforce a different language than the one declared, so it
    // fails closed like every other form this compiler cannot honour exactly.
    const char *ws = "{\"type\":\"string\",\"pattern\":\"^\\\\s{2}$\"}";
    jv *j = json_parse(ws, strlen(ws));
    assert(j != NULL);
    char err[128];
    assert(schema_compile(j, err, sizeof(err)) == NULL);
    assert(strstr(err, "pattern") != NULL);
    jv_free(j);
}

static void test_schema_pattern_regex_syntax_is_rejected_not_reinterpreted(void) {
    // the compiler matches prefix and class LITERALLY; regex syntax it would
    // silently reinterpret (escapes, negation, prefix metacharacters) must be
    // a compile error — otherwise the enforced language differs from the
    // declared pattern (e.g. [\d] would accept a literal backslash the
    // declared regex forbids)
    const char *bad_patterns[] = {
        "{\"type\":\"string\",\"pattern\":\"^x[\\\\d]{2,}$\"}",   // class escape
        "{\"type\":\"string\",\"pattern\":\"^x[^a]+$\"}",         // negated class
        "{\"type\":\"string\",\"pattern\":\"^a.b[a-z]+$\"}",      // prefix metachar
        "{\"type\":\"string\",\"pattern\":\"^a\\\\.b[a-z]+$\"}",  // prefix escape
    };
    for (size_t i = 0; i < sizeof(bad_patterns) / sizeof(*bad_patterns); i++) {
        jv *j = json_parse(bad_patterns[i], strlen(bad_patterns[i]));
        assert(j != NULL);
        char err[128];
        snode *schema = schema_compile(j, err, sizeof(err));
        assert(schema == NULL);
        assert(strstr(err, "pattern") != NULL);
        jv_free(j);
    }
    // literal-safe specials INSIDE the class stay supported (regex treats
    // them as literals there, so literal matching agrees with the pattern)
    const char *ok_src =
        "{\"type\":\"string\",\"pattern\":\"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$\"}";
    jv *j = json_parse(ok_src, strlen(ok_src));
    char err[128];
    snode *schema = schema_compile(j, err, sizeof(err));
    assert(schema != NULL);
    schema_free(schema); jv_free(j);
}

static void test_schema_number_bounds_reject_dead_minus_and_close_in_bounds(void) {
    // minus under a non-negative minimum is a dead prefix: no suffix can ever
    // satisfy the bound, so the first byte must be refused (the integer path
    // already does) instead of stalling with every token masked at the end
    const char *src = "{\"type\":\"number\",\"minimum\":0}";
    jv *j = json_parse(src, strlen(src));
    char err[128];
    snode *schema = schema_compile(j, err, sizeof(err));
    assert(schema != NULL);
    sval v; sval_init(&v, schema);
    assert(!sval_feed(&v, "-", 1));
    sval ok; sval_init(&ok, schema);
    assert(sval_feed(&ok, "3.5", 3));
    schema_free(schema); jv_free(j);

    // a bounded number filled in by force-close stays INSIDE its bounds —
    // both the untouched-value close path and the required-property
    // minimal-emission path
    src = "{\"type\":\"object\",\"properties\":{"
          "\"a\":{\"type\":\"number\",\"minimum\":5}},"
          "\"required\":[\"a\"]}";
    j = json_parse(src, strlen(src));
    schema = schema_compile(j, err, sizeof(err));
    assert(schema != NULL);
    const char *starts[] = { "{", "{\"a\":" };
    for (size_t i = 0; i < sizeof(starts) / sizeof(*starts); i++) {
        sval closing; sval_init(&closing, schema);
        assert(sval_feed(&closing, starts[i], (int)strlen(starts[i])));
        char suffix[64];
        int n = sval_close(&closing, suffix, sizeof(suffix));
        assert(n > 0);
        char full[128];
        snprintf(full, sizeof(full), "%s%s", starts[i], suffix);
        jv *parsed = json_parse(full, strlen(full));
        assert(parsed != NULL);
        jv *a = jv_get(parsed, "a");
        assert(a != NULL && a->type == J_NUM && a->num >= 5.0);
        jv_free(parsed);
    }
    schema_free(schema); jv_free(j);
}

static void test_schema_merges_enum_and_const_anyof(void) {
    const char *src = "{\"anyOf\":["
        "{\"type\":\"string\",\"enum\":[\"pending\",\"completed\"]},"
        "{\"type\":\"string\",\"const\":\"deleted\"}]}";
    jv *j = json_parse(src, strlen(src));
    char err[128];
    snode *schema = schema_compile(j, err, sizeof(err));
    assert(schema != NULL);
    const char *good[] = { "\"pending\"", "\"completed\"", "\"deleted\"" };
    for (size_t i = 0; i < sizeof(good) / sizeof(*good); i++) {
        sval v; sval_init(&v, schema);
        assert(sval_feed(&v, good[i], (int)strlen(good[i])) && v.done);
    }
    schema_free(schema); jv_free(j);
}

static void test_schema_integer_bounds_are_enforced(void) {
    const char *src =
        "{\"type\":\"integer\",\"minimum\":0,"
        "\"exclusiveMinimum\":3,\"maximum\":55}";
    jv *schema_json = json_parse(src, strlen(src));
    assert(schema_json != NULL);
    char err[128];
    snode *schema = schema_compile(schema_json, err, sizeof(err));
    assert(schema != NULL);

    const char *good[] = { "4", "10", "55" };
    for (size_t i = 0; i < sizeof(good) / sizeof(*good); i++) {
        sval v;
        sval_init(&v, schema);
        assert(sval_feed(&v, good[i], (int)strlen(good[i])));
        char out[32];
        int n = sval_close(&v, out, sizeof(out));
        assert(n == 0);
    }

    const char *bad[] = { "-1", "0", "3", "56", "999" };
    for (size_t i = 0; i < sizeof(bad) / sizeof(*bad); i++) {
        sval v;
        sval_init(&v, schema);
        bool prefix_ok = sval_feed(&v, bad[i], (int)strlen(bad[i]));
        // A delimiter is the observable attempt to finish the out-of-range
        // number; an impossible prefix may be refused even earlier.
        assert(!prefix_ok || !sval_feed(&v, " ", 1));
    }

    schema_free(schema);
    jv_free(schema_json);
}

static void test_schema_integer_bounds_complete_truncation(void) {
    const char *src =
        "{\"type\":\"object\",\"properties\":{"
        "\"line\":{\"type\":\"integer\",\"minimum\":50,\"maximum\":55}"
        "},\"required\":[\"line\"]}";
    jv *schema_json = json_parse(src, strlen(src));
    assert(schema_json != NULL);
    char err[128];
    snode *schema = schema_compile(schema_json, err, sizeof(err));
    assert(schema != NULL);

    sval v;
    sval_init(&v, schema);
    const char *partial = "{\"line\":5";
    assert(sval_feed(&v, partial, (int)strlen(partial)));
    char suffix[64];
    int n = sval_close(&v, suffix, sizeof(suffix));
    assert(n > 0);
    char full[128];
    snprintf(full, sizeof(full), "%s%s", partial, suffix);
    jv *parsed = json_parse(full, strlen(full));
    assert(parsed != NULL);
    jv *line = jv_get(parsed, "line");
    assert(line && line->type == J_NUM && line->num >= 50 && line->num <= 55);

    jv_free(parsed);
    schema_free(schema);
    jv_free(schema_json);
}

static void test_schema_number_bounds_are_enforced(void) {
    const char *src = "{\"type\":\"object\",\"properties\":{"
        "\"timeout\":{\"type\":\"number\",\"minimum\":0,"
        "\"maximum\":600000}},\"required\":[\"timeout\"]}";
    jv *schema_json = json_parse(src, strlen(src));
    assert(schema_json != NULL);
    char err[128];
    snode *schema = schema_compile(schema_json, err, sizeof(err));
    assert(schema != NULL);

    const char *good[] = { "{\"timeout\":0}", "{\"timeout\":1.5}",
                           "{\"timeout\":6e5}" };
    for (size_t i = 0; i < sizeof(good) / sizeof(*good); i++) {
        sval v; sval_init(&v, schema);
        assert(sval_feed(&v, good[i], (int)strlen(good[i])));
        assert(v.done);
    }
    const char *bad[] = { "{\"timeout\":-0.1}",
                          "{\"timeout\":600000.1}",
                          "{\"timeout\":7e5}" };
    for (size_t i = 0; i < sizeof(bad) / sizeof(*bad); i++) {
        sval v; sval_init(&v, schema);
        assert(!sval_feed(&v, bad[i], (int)strlen(bad[i])));
    }

    schema_free(schema);
    jv_free(schema_json);
}

// The other half of the invariant: keywords that are pure annotations carry
// no constraint, so ignoring them ignores nothing. Real OpenAI tool payloads
// are full of them and must keep compiling.
static void test_schema_accepts_annotation_keywords(void) {
    const char *src =
        "{\"type\":\"object\",\"title\":\"T\",\"description\":\"d\","
        "\"$schema\":\"https://json-schema.org/draft/2020-12/schema\","
        "\"properties\":{\"a\":{\"type\":\"string\",\"description\":\"d\","
        "\"default\":\"x\",\"examples\":[\"y\"],\"format\":\"uri\"}},"
        "\"required\":[\"a\"],"
        "\"additionalProperties\":false}";
    jv *schema_json = json_parse(src, strlen(src));
    assert(schema_json != NULL);
    char err[128];
    snode *schema = schema_compile(schema_json, err, sizeof(err));
    assert(schema != NULL);
    schema_free(schema);
    jv_free(schema_json);
}

static void test_schema_accepts_claude_open_metadata_object(void) {
    const char *src =
        "{\"type\":\"object\",\"propertyNames\":{\"type\":\"string\"},"
        "\"additionalProperties\":{}}";
    jv *schema_json = json_parse(src, strlen(src));
    assert(schema_json != NULL);
    char err[128];
    snode *schema = schema_compile(schema_json, err, sizeof(err));
    assert(schema != NULL);
    sval v;
    sval_init(&v, schema);
    const char *doc = "{\"owner\":\"agent\",\"deleted\":null}";
    assert(sval_feed(&v, doc, (int)strlen(doc)));
    assert(v.done);
    schema_free(schema);
    jv_free(schema_json);
}

static void test_schema_typed_additional_properties_map(void) {
    const char *src =
        "{\"type\":\"object\",\"additionalProperties\":{"
        "\"type\":\"object\",\"properties\":{\"n\":{\"type\":\"string\"}},"
        "\"required\":[\"n\"],\"additionalProperties\":false}}";
    jv *schema_json = json_parse(src, strlen(src));
    assert(schema_json != NULL);
    char err[128];
    snode *schema = schema_compile(schema_json, err, sizeof(err));
    assert(schema != NULL);

    const char *good = "{\"alpha\":{\"n\":\"x\"},\"beta\":{\"n\":\"y\"}}";
    sval v;
    sval_init(&v, schema);
    assert(sval_feed(&v, good, (int)strlen(good)) && v.done);

    const char *bad[] = {
        "{\"alpha\":{}}",
        "{\"alpha\":{\"n\":1}}",
    };
    for (size_t i = 0; i < sizeof(bad) / sizeof(*bad); i++) {
        sval_init(&v, schema);
        assert(!sval_feed(&v, bad[i], (int)strlen(bad[i])));
    }

    // Arbitrary keys are real string content (including escapes), and every
    // started map must still force-close to a document accepted by the same
    // schema when generation stops at a token boundary.
    const char *partial[] = {
        "{\"alpha",
        "{\"a\\\\",
        "{\"alpha\":",
        "{\"alpha\":{\"n\":\"x\"},",
    };
    for (size_t i = 0; i < sizeof(partial) / sizeof(*partial); i++) {
        sval_init(&v, schema);
        assert(sval_feed(&v, partial[i], (int)strlen(partial[i])));
        if (i == 0) assert(sval_ws_is_content(&v));
        char suffix[256];
        int n = sval_close(&v, suffix, sizeof(suffix));
        assert(n > 0);
        char full[512];
        snprintf(full, sizeof(full), "%s%s", partial[i], suffix);
        jv *parsed = json_parse(full, strlen(full));
        assert(parsed != NULL);
        jv_free(parsed);
        sval check;
        sval_init(&check, schema);
        assert(sval_feed(&check, full, (int)strlen(full)) && check.done);
    }

    schema_free(schema);
    jv_free(schema_json);
}

// The compiled object enforces a CLOSED property set. `false` asks for
// exactly that and compiles; `true` asks for the opposite and used to be
// dropped, making the output STRICTER than the schema permitted.
static void test_schema_additional_properties(void) {
    static const char *const bad[] = {
        "{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"string\"}},"
            "\"additionalProperties\":true}",
        "{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"string\"}},"
            "\"additionalProperties\":{\"type\":\"string\"}}",
    };
    for (size_t i = 0; i < sizeof(bad) / sizeof(*bad); i++) {
        jv *schema_json = json_parse(bad[i], strlen(bad[i]));
        assert(schema_json != NULL);
        char err[128];
        snode *schema = schema_compile(schema_json, err, sizeof(err));
        assert(schema == NULL);
        assert(strstr(err, "additionalProperties") != NULL);
        jv_free(schema_json);
    }
}

// An *empty* properties map with additionalProperties:false is a different
// statement from an absent one: it declares an object with no permitted keys,
// i.e. exactly `{}`. Compiling it to the open any-object machine would have
// accepted any object at all — the opposite of the request — so it used to be
// rejected instead. Real agent clients (the Codex CLI's zero-argument tools)
// send it, and it is exactly expressible, so it compiles.
static void test_schema_empty_closed_object(void) {
    const char *srcs[] = {
        "{\"type\":\"object\",\"properties\":{},\"additionalProperties\":false}",
        // Omitting `properties` says the same thing: no declared keys, and no
        // additional ones. Claude and other clients use both spellings.
        "{\"type\":\"object\",\"additionalProperties\":false}",
    };
    for (size_t i = 0; i < sizeof(srcs) / sizeof(*srcs); i++) {
        const char *src = srcs[i];
        jv *schema_json = json_parse(src, strlen(src));
        assert(schema_json != NULL);
        char err[128];
        snode *schema = schema_compile(schema_json, err, sizeof(err));
        assert(schema != NULL);

        // the empty object is accepted and closes with nothing outstanding
        sval v;
        sval_init(&v, schema);
        assert(sval_feed(&v, "{", 1));
        assert(sval_feed(&v, "}", 1));
        char out[64];
        assert(sval_close(&v, out, sizeof(out)) == 0);

        // any key at all is refused: the set really is closed
        sval_init(&v, schema);
        assert(sval_feed(&v, "{", 1));
        assert(!sval_feed(&v, "\"", 1));

        // and a truncation mid-document still closes to a legal `{}`
        sval_init(&v, schema);
        assert(sval_feed(&v, "{", 1));
        int n = sval_close(&v, out, sizeof(out));
        assert(n == 1 && out[0] == '}');

        schema_free(schema);
        jv_free(schema_json);
    }
}

// `required` with no `properties` compiled to the open any-object machine,
// which enforces no key at all — the requirement vanished silently.
static void test_schema_rejects_required_without_properties(void) {
    const char *src = "{\"type\":\"object\",\"required\":[\"a\"]}";
    jv *schema_json = json_parse(src, strlen(src));
    assert(schema_json != NULL);
    char err[128];
    snode *schema = schema_compile(schema_json, err, sizeof(err));
    assert(schema == NULL);
    assert(strstr(err, "required") != NULL);
    jv_free(schema_json);

    // an empty required list asks for nothing, so it stays legal
    const char *ok = "{\"type\":\"object\",\"required\":[]}";
    schema_json = json_parse(ok, strlen(ok));
    assert(schema_json != NULL);
    schema = schema_compile(schema_json, err, sizeof(err));
    assert(schema != NULL);
    schema_free(schema);
    jv_free(schema_json);
}

static void test_schema_oneof_const_scalars(void) {
    const char *src =
        "{\"oneOf\":[{\"const\":\"read_file\"},{\"const\":\"done\"}]}";
    jv *schema_json = json_parse(src, strlen(src));
    assert(schema_json != NULL);
    char err[128];
    snode *schema = schema_compile(schema_json, err, sizeof(err));
    assert(schema != NULL);

    sval v;
    sval_init(&v, schema);
    assert(sval_feed(&v, "\"done\"", 6));

    schema_free(schema);
    jv_free(schema_json);
}

static void test_schema_oneof_const_numeric_prefixes(void) {
    const char *src =
        "{\"oneOf\":[{\"const\":1},{\"const\":12}]}";
    jv *schema_json = json_parse(src, strlen(src));
    assert(schema_json != NULL);
    char err[128];
    snode *schema = schema_compile(schema_json, err, sizeof(err));
    assert(schema != NULL);

    sval v;
    sval_init(&v, schema);
    assert(sval_feed(&v, "12", 2));

    schema_free(schema);
    jv_free(schema_json);
}

static void test_schema_rejects_oversized_oneof_const_scalars(void) {
    char src[4096];
    int n = snprintf(src, sizeof(src), "{\"oneOf\":[");
    for (int i = 0; i < 65; i++)
        n += snprintf(src + n, sizeof(src) - (size_t)n,
                      "%s{\"const\":%d}", i ? "," : "", i);
    snprintf(src + n, sizeof(src) - (size_t)n, "]}");
    jv *schema_json = json_parse(src, strlen(src));
    assert(schema_json != NULL);
    char err[128];
    snode *schema = schema_compile(schema_json, err, sizeof(err));
    assert(schema == NULL);
    assert(strstr(err, "enum size") != NULL);
    jv_free(schema_json);
}

static void test_schema_string_length_bounds_close_and_reject(void) {
    const char *src = "{\"type\":\"string\",\"minLength\":5,\"maxLength\":8}";
    jv *schema_json = json_parse(src, strlen(src));
    assert(schema_json != NULL);
    char err[128];
    snode *schema = schema_compile(schema_json, err, sizeof(err));
    assert(schema != NULL);

    sval v;
    sval_init(&v, schema);
    assert(sval_feed(&v, "\"abc", 4));
    char out[64];
    int n = sval_close(&v, out, sizeof(out));
    assert(n > 0);
    char full[128];
    snprintf(full, sizeof(full), "\"abc%s", out);
    jv *parsed = json_parse(full, strlen(full));
    assert(parsed != NULL);
    assert(strlen(jv_str(parsed, "")) >= 5);
    jv_free(parsed);

    sval_init(&v, schema);
    assert(!sval_feed(&v, "\"123456789\"", 11));

    schema_free(schema);
    jv_free(schema_json);
}

static void test_schema_string_minlength_full_close(void) {
    const char *src =
        "{\"type\":\"object\",\"properties\":{"
        "\"name\":{\"type\":\"string\",\"minLength\":3}"
        "},\"required\":[\"name\"]}";
    jv *schema_json = json_parse(src, strlen(src));
    assert(schema_json != NULL);
    char err[128];
    snode *schema = schema_compile(schema_json, err, sizeof(err));
    assert(schema != NULL);

    sval v;
    char out[128];

    // Closing a decode that never opened the document emits nothing: there is
    // no partial value to pad out. (This used to return a fabricated
    // `{"name":"   "}`; see
    // test_schema_close_without_payload_fabricates_nothing for why that had
    // to stop.) minLength padding is exercised below, where the model really
    // did start the string.
    sval_init(&v, schema);
    assert(sval_close(&v, out, sizeof(out)) == 0);

    sval_init(&v, schema);
    assert(sval_feed(&v, "{\"name\":", 8));
    int n = sval_close(&v, out, sizeof(out));
    assert(n > 0);
    char full[256];
    snprintf(full, sizeof(full), "{\"name\":%s", out);
    jv *parsed = json_parse(full, strlen(full));
    assert(parsed != NULL);
    assert(strlen(jv_str(jv_get(parsed, "name"), "")) >= 3);

    jv_free(parsed);
    schema_free(schema);
    jv_free(schema_json);
}

static void test_schema_discriminated_action_args(void) {
    const char *src =
        "{\"oneOf\":["
        "{\"type\":\"object\",\"properties\":{"
        "\"thinking\":{\"type\":\"string\"},"
        "\"tool\":{\"const\":\"read_file\"},"
        "\"args\":{\"type\":\"object\",\"properties\":{"
        "\"path\":{\"type\":\"string\"}},\"required\":[\"path\"]}"
        "},\"required\":[\"thinking\",\"tool\",\"args\"]},"
        "{\"type\":\"object\",\"properties\":{"
        "\"thinking\":{\"type\":\"string\"},"
        "\"tool\":{\"const\":\"done\"},"
        "\"args\":{\"type\":\"object\",\"properties\":{"
        "\"summary\":{\"type\":\"string\"}},\"required\":[\"summary\"]}"
        "},\"required\":[\"thinking\",\"tool\",\"args\"]}"
        "]}";
    jv *schema_json = json_parse(src, strlen(src));
    assert(schema_json != NULL);
    char err[128];
    snode *schema = schema_compile(schema_json, err, sizeof(err));
    assert(schema != NULL);

    sval v;
    const char *valid =
        "{\"thinking\":\"x\",\"tool\":\"done\",\"args\":{\"summary\":\"ok\"}}";
    sval_init(&v, schema);
    assert(sval_feed(&v, valid, strlen(valid)));

    const char *invalid =
        "{\"thinking\":\"x\",\"tool\":\"done\",\"args\":{\"path\":\"README.md\"}}";
    sval_init(&v, schema);
    assert(!sval_feed(&v, invalid, strlen(invalid)));

    const char *partial = "{\"thinking\":\"x\",\"tool\":\"done\"";
    sval_init(&v, schema);
    assert(sval_feed(&v, partial, strlen(partial)));
    char out[128];
    int n = sval_close(&v, out, sizeof(out));
    assert(n > 0);
    char full[256];
    snprintf(full, sizeof(full), "{\"thinking\":\"x\",\"tool\":\"done\"%s", out);
    jv *parsed = json_parse(full, strlen(full));
    assert(parsed != NULL);
    jv *args = jv_get(parsed, "args");
    assert(args != NULL);
    assert(jv_get(args, "summary") != NULL);
    assert(jv_get(args, "path") == NULL);

    jv_free(parsed);
    schema_free(schema);
    jv_free(schema_json);
}

static void test_schema_union_dispatch_rules(void) {
    // ambiguous alternatives (two object shapes) must fail at compile time
    const char *dup =
        "{\"oneOf\":[{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"string\"}}},"
        "{\"type\":\"object\",\"properties\":{\"b\":{\"type\":\"string\"}}}]}";
    jv *j = json_parse(dup, strlen(dup));
    assert(j != NULL);
    char err[128];
    assert(schema_compile(j, err, sizeof(err)) == NULL);
    jv_free(j);

    // nested oneOf inside a union alternative is not dispatchable
    const char *nested =
        "{\"oneOf\":[{\"oneOf\":[{\"type\":\"string\"},{\"type\":\"integer\"}]},"
        "{\"type\":\"boolean\"}]}";
    j = json_parse(nested, strlen(nested));
    assert(j != NULL);
    assert(schema_compile(j, err, sizeof(err)) == NULL);
    jv_free(j);

    // a mixed-first-byte enum alternative must reach ALL its literals
    const char *mixed =
        "{\"oneOf\":[{\"enum\":[1,\"a\"]},{\"type\":\"boolean\"}]}";
    j = json_parse(mixed, strlen(mixed));
    assert(j != NULL);
    snode *schema = schema_compile(j, err, sizeof(err));
    assert(schema != NULL);
    sval v;
    sval_init(&v, schema);
    assert(sval_feed(&v, "\"a\"", 3));
    assert(v.done);
    sval_init(&v, schema);
    assert(sval_feed(&v, "true", 4));
    assert(v.done);
    schema_free(schema);
    jv_free(j);

    // disjoint types still compile and dispatch
    const char *ok = "{\"oneOf\":[{\"type\":\"string\"},{\"type\":\"null\"}]}";
    j = json_parse(ok, strlen(ok));
    assert(j != NULL);
    schema = schema_compile(j, err, sizeof(err));
    assert(schema != NULL);
    sval_init(&v, schema);
    assert(sval_feed(&v, "null", 4));
    assert(v.done);
    schema_free(schema);
    jv_free(j);
}

static void test_schema_discriminated_required_must_match(void) {
    const char *src =
        "{\"oneOf\":["
        "{\"type\":\"object\",\"properties\":{"
        "\"tool\":{\"const\":\"a\"},\"args\":{\"type\":\"object\","
        "\"properties\":{\"x\":{\"type\":\"string\"}}}},"
        "\"required\":[\"tool\",\"args\"]},"
        "{\"type\":\"object\",\"properties\":{"
        "\"tool\":{\"const\":\"b\"},\"args\":{\"type\":\"object\","
        "\"properties\":{\"y\":{\"type\":\"string\"}}}},"
        "\"required\":[\"tool\"]}"
        "]}";
    jv *j = json_parse(src, strlen(src));
    assert(j != NULL);
    char err[128];
    assert(schema_compile(j, err, sizeof(err)) == NULL);
    assert(strstr(err, "required") != NULL);
    jv_free(j);
}

static void test_schema_long_string_value(void) {
    // file-sized string values must not trip the length counter (a 16-bit
    // counter would wrap past 32767 and reject the closing quote)
    const char *src = "{\"type\":\"string\"}";
    jv *schema_json = json_parse(src, strlen(src));
    assert(schema_json != NULL);
    char err[128];
    snode *schema = schema_compile(schema_json, err, sizeof(err));
    assert(schema != NULL);

    sval v;
    sval_init(&v, schema);
    assert(sval_feed(&v, "\"", 1));
    char chunk[4096];
    memset(chunk, 'a', sizeof(chunk));
    for (int i = 0; i < 20; i++) // 80KB of content
        assert(sval_feed(&v, chunk, sizeof(chunk)));
    assert(sval_feed(&v, "\"", 1));
    assert(v.done);

    schema_free(schema);
    jv_free(schema_json);
}

static void test_schema_escape_at_maxlength_rejected(void) {
    // at maxLength only the closing quote may follow; starting an escape
    // there would force close() to overrun the bound to stay valid JSON
    const char *src = "{\"type\":\"string\",\"maxLength\":3}";
    jv *schema_json = json_parse(src, strlen(src));
    assert(schema_json != NULL);
    char err[128];
    snode *schema = schema_compile(schema_json, err, sizeof(err));
    assert(schema != NULL);

    sval v;
    sval_init(&v, schema);
    assert(!sval_feed(&v, "\"abc\\", 5));

    sval_init(&v, schema); // an escape below the bound still counts as one char
    assert(sval_feed(&v, "\"ab\\n\"", 6));
    assert(v.done);

    schema_free(schema);
    jv_free(schema_json);
}

static void test_schema_nested_tool_key_keeps_discriminator(void) {
    // a "tool" key inside a nested object between the discriminator and args
    // must not overwrite the outer object's chosen alternative
    const char *src =
        "{\"oneOf\":["
        "{\"type\":\"object\",\"properties\":{"
        "\"tool\":{\"const\":\"read_file\"},"
        "\"meta\":{\"type\":\"object\",\"properties\":{"
        "\"tool\":{\"enum\":[\"x\",\"y\"]}},\"required\":[\"tool\"]},"
        "\"args\":{\"type\":\"object\",\"properties\":{"
        "\"path\":{\"type\":\"string\"}},\"required\":[\"path\"]}"
        "},\"required\":[\"tool\",\"meta\",\"args\"]},"
        "{\"type\":\"object\",\"properties\":{"
        "\"tool\":{\"const\":\"done\"},"
        "\"meta\":{\"type\":\"object\",\"properties\":{"
        "\"tool\":{\"enum\":[\"x\",\"y\"]}},\"required\":[\"tool\"]},"
        "\"args\":{\"type\":\"object\",\"properties\":{"
        "\"summary\":{\"type\":\"string\"}},\"required\":[\"summary\"]}"
        "},\"required\":[\"tool\",\"meta\",\"args\"]}"
        "]}";
    jv *schema_json = json_parse(src, strlen(src));
    assert(schema_json != NULL);
    char err[128];
    snode *schema = schema_compile(schema_json, err, sizeof(err));
    assert(schema != NULL);

    sval v;
    const char *valid = "{\"tool\":\"read_file\",\"meta\":{\"tool\":\"y\"},"
                        "\"args\":{\"path\":\"a.txt\"}}";
    sval_init(&v, schema);
    assert(sval_feed(&v, valid, strlen(valid)));
    assert(v.done);

    // args from the wrong alternative must still be rejected
    const char *invalid = "{\"tool\":\"read_file\",\"meta\":{\"tool\":\"y\"},"
                          "\"args\":{\"summary\":\"no\"}}";
    sval_init(&v, schema);
    assert(!sval_feed(&v, invalid, strlen(invalid)));

    schema_free(schema);
    jv_free(schema_json);
}

static void test_schema_close_mid_discriminator_matches_args(void) {
    // aborting generation inside the second alternative's tool literal must
    // close with that alternative's args, not alternative 0's
    const char *src =
        "{\"oneOf\":["
        "{\"type\":\"object\",\"properties\":{"
        "\"tool\":{\"const\":\"alpha\"},"
        "\"args\":{\"type\":\"object\",\"properties\":{"
        "\"path\":{\"type\":\"string\"}},\"required\":[\"path\"]}"
        "},\"required\":[\"tool\",\"args\"]},"
        "{\"type\":\"object\",\"properties\":{"
        "\"tool\":{\"const\":\"zeta\"},"
        "\"args\":{\"type\":\"object\",\"properties\":{"
        "\"summary\":{\"type\":\"string\"}},\"required\":[\"summary\"]}"
        "},\"required\":[\"tool\",\"args\"]}"
        "]}";
    jv *schema_json = json_parse(src, strlen(src));
    assert(schema_json != NULL);
    char err[128];
    snode *schema = schema_compile(schema_json, err, sizeof(err));
    assert(schema != NULL);

    sval v;
    const char *partial = "{\"tool\":\"ze"; // unambiguously alternative 1
    sval_init(&v, schema);
    assert(sval_feed(&v, partial, strlen(partial)));
    char out[256];
    int n = sval_close(&v, out, sizeof(out));
    assert(n > 0);
    char full[512];
    snprintf(full, sizeof(full), "%s%s", partial, out);
    jv *parsed = json_parse(full, strlen(full));
    assert(parsed != NULL);
    assert(!strcmp(jv_str(jv_get(parsed, "tool"), ""), "zeta"));
    jv *args = jv_get(parsed, "args");
    assert(args != NULL);
    assert(jv_get(args, "summary") != NULL);
    assert(jv_get(args, "path") == NULL);

    jv_free(parsed);
    schema_free(schema);
    jv_free(schema_json);
}

static void test_schema_rejects_discriminator_after_conditional_args(void) {
    const char *src =
        "{\"oneOf\":["
        "{\"type\":\"object\",\"properties\":{"
        "\"thinking\":{\"type\":\"string\"},"
        "\"args\":{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"}},\"required\":[\"path\"]},"
        "\"tool\":{\"const\":\"read_file\"}"
        "},\"required\":[\"thinking\",\"args\",\"tool\"]},"
        "{\"type\":\"object\",\"properties\":{"
        "\"thinking\":{\"type\":\"string\"},"
        "\"args\":{\"type\":\"object\",\"properties\":{\"summary\":{\"type\":\"string\"}},\"required\":[\"summary\"]},"
        "\"tool\":{\"const\":\"done\"}"
        "},\"required\":[\"thinking\",\"args\",\"tool\"]}"
        "]}";
    jv *schema_json = json_parse(src, strlen(src));
    assert(schema_json != NULL);
    char err[128];
    snode *schema = schema_compile(schema_json, err, sizeof(err));
    assert(schema == NULL);
    assert(strstr(err, "tool") != NULL);
    jv_free(schema_json);
}

// A response body must be decodable by a strict UTF-8 JSON client. Structural
// validation is not enough: 0xC0 has a valid two-byte lead pattern and can be
// followed by a valid continuation byte, yet no conforming decoder accepts it
// (it is an overlong encoding). Emitting it produced a Messages body Python
// refused outright with "invalid start byte", which a model reaches whenever a
// byte-fallback token emits a stray byte or max_tokens cuts a character in
// half. Every rejected form must come back as U+FFFD, never as raw bytes.
static void test_escape_replaces_ill_formed_utf8(void) {
    struct { const char *in; size_t n; const char *what; } bad[] = {
        { "\xC0\x80",         2, "overlong two-byte (0xC0)" },
        { "\xC1\xBF",         2, "overlong two-byte (0xC1)" },
        { "\xE0\x80\x80",     3, "overlong three-byte" },
        { "\xED\xA0\x80",     3, "UTF-16 surrogate U+D800" },
        { "\xF0\x80\x80\x80", 4, "overlong four-byte" },
        { "\xF4\x90\x80\x80", 4, "beyond U+10FFFF" },
        { "\xF5\x80\x80\x80", 4, "lead byte 0xF5" },
        { "\x80",             1, "bare continuation byte" },
        { "\xE2\x82",         2, "truncated three-byte at end of buffer" },
    };
    for (size_t i = 0; i < sizeof(bad) / sizeof(*bad); i++) {
        char out[64];
        size_t m = json_escape(bad[i].in, bad[i].n, out, sizeof(out));
        for (size_t k = 0; k < m; k++)
            if ((unsigned char)out[k] >= 0x80) {
                // the only bytes above ASCII that may survive are U+FFFD's
                assert(((unsigned char)out[k] == 0xEF ||
                        (unsigned char)out[k] == 0xBF ||
                        (unsigned char)out[k] == 0xBD) && bad[i].what);
            }
        // and the result must itself re-validate as UTF-8
        for (size_t k = 0; k < m; ) {
            unsigned char c = (unsigned char)out[k];
            size_t len = c < 0x80 ? 1 : (c & 0xE0) == 0xC0 ? 2 :
                         (c & 0xF0) == 0xE0 ? 3 : (c & 0xF8) == 0xF0 ? 4 : 0;
            assert(len != 0 && k + len <= m && bad[i].what);
            k += len;
        }
    }
    // well-formed input must pass through untouched
    const char *ok = "caf\xC3\xA9 \xE2\x82\xAC \xF0\x9F\x98\x80";
    char out[64];
    size_t m = json_escape(ok, strlen(ok), out, sizeof(out));
    assert(m == strlen(ok) && memcmp(out, ok, m) == 0);
    puts("ok: ill-formed UTF-8 is replaced, well-formed passes through");
}

// jv_dump_tojson exists to reproduce jinja's `tojson` filter byte for byte,
// because four families declare their tools through it (chatml.jinja:11,
// chatml-think.jinja:9, ornith.jinja:50, muse.jinja:73) and the declaration
// block is prompt text the model was trained on, not presentation.
//
// Every `want` below was RECORDED from that filter -- jinja2 with the
// reference's own tojson (json.dumps(ensure_ascii=False), the settings
// scripts/template-conformance.py installs) -- not written from expectation.
static void test_tojson_dump_matches_jinja(void) {
    struct { const char *doc; const char *want; const char *what; } cases[] = {
        // the tool declaration the conformance matrix actually sends
        { "{\"type\":\"function\",\"function\":{\"name\":\"get_weather\","
          "\"description\":\"Get the current weather for a city.\","
          "\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":"
          "{\"type\":\"string\",\"description\":\"City name.\"}},"
          "\"required\":[\"city\"]}}}",
          "{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", "
          "\"description\": \"Get the current weather for a city.\", "
          "\"parameters\": {\"type\": \"object\", \"properties\": {\"city\": "
          "{\"type\": \"string\", \"description\": \"City name.\"}}, "
          "\"required\": [\"city\"]}}}",
          "the tool declaration block" },
        // empty containers take NO inner space
        { "{}", "{}", "empty object" },
        { "[]", "[]", "empty array" },
        { "{\"a\":{},\"b\":[],\"c\":[[],{}]}",
          "{\"a\": {}, \"b\": [], \"c\": [[], {}]}", "nested empty containers" },
        // insertion order, never sorted (json.dumps sort_keys defaults False)
        { "{\"z\":1,\"a\":2,\"m\":3}", "{\"z\": 1, \"a\": 2, \"m\": 3}",
          "key order is the document's, not sorted" },
        // ensure_ascii=False: non-ASCII stays raw UTF-8, never \uXXXX
        { "{\"k\":\"caf\\u00e9 \\u20ac \\ud83d\\ude00\"}",
          "{\"k\": \"caf\xC3\xA9 \xE2\x82\xAC \xF0\x9F\x98\x80\"}",
          "non-ASCII is raw UTF-8" },
        // short forms for \" \\ \n \t \r \b \f, \u00xx lowercase for the
        // rest of C0, and NEITHER '/' nor DEL is escaped
        { "{\"k\":\"a\\\"b\\\\c\\nd\\te\\rf\\bg\\fh\\u001fi/j\\u007f\"}",
          "{\"k\": \"a\\\"b\\\\c\\nd\\te\\rf\\bg\\fh\\u001fi/j\x7f\"}",
          "escaping" },
        { "{\"t\":true,\"f\":false,\"n\":null}",
          "{\"t\": true, \"f\": false, \"n\": null}", "literals" },
        { "[1,\"a\",null,true,{},[]]", "[1, \"a\", null, true, {}, []]",
          "array separators" },
        { "{\"nested\":{\"deep\":{\"x\":[1,2,3]}}}",
          "{\"nested\": {\"deep\": {\"x\": [1, 2, 3]}}}", "nesting" },
        { "\"plain string\"", "\"plain string\"", "bare string" },
        { "42", "42", "bare number" },
        { "{\"min\":0,\"max\":100,\"mult\":0.01}",
          "{\"min\": 0, \"max\": 100, \"mult\": 0.01}", "schema numerics" },
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(*cases); i++) {
        jv *v = json_parse(cases[i].doc, strlen(cases[i].doc));
        assert(v != NULL && cases[i].what);
        sbuf spaced = { 0 };
        jv_dump_tojson(v, &spaced);
        assert(!spaced.failed && spaced.s && cases[i].what);
        if (strcmp(spaced.s, cases[i].want) != 0) {
            fprintf(stderr, "tojson drift (%s)\n  want: %s\n  got : %s\n",
                    cases[i].what, cases[i].want, spaced.s);
            assert(0);
        }
        // and the compact dump the HTTP response path uses must NOT have
        // moved: this variant is additive, and a body that silently
        // reformatted would be the expensive way to find that out
        sbuf compact = { 0 };
        jv_dump(v, &compact);
        assert(!compact.failed && compact.s && cases[i].what);
        assert(strstr(compact.s, ", ") == NULL && cases[i].what);
        assert(strstr(compact.s, "\": ") == NULL && cases[i].what);
        free(spaced.s);
        free(compact.s);
        jv_free(v);
    }
    puts("ok: jv_dump_tojson reproduces jinja's tojson; jv_dump stays compact");
}

int main(void) {
    test_strict_bounded_numbers();
    test_json_rejects_unpaired_utf16_surrogates();
    test_json_rejects_ill_formed_raw_utf8();
    test_json_rejects_embedded_nul();
    test_json_rejects_duplicate_object_keys();
    test_json_close_partial_string();
    test_schema_required_close();
    test_leading_whitespace_is_refused_but_interior_is_kept();
    test_json_mode_leading_whitespace_is_refused();
    test_schema_close_without_payload_fabricates_nothing();
    test_schema_rejects_empty_type_union();
    test_schema_huge_min_bounds_terminate();
    test_schema_rejects_bad_bounds();
    test_schema_rejects_non_integer_or_huge_bounds();
    test_schema_rejects_escaped_keys();
    test_schema_rejects_unenforceable_keywords();
    test_schema_bounded_repetition();
    test_schema_multi_segment_pattern();
    test_schema_agent_id_pattern_is_enforced();
    test_schema_pattern_shorthand_classes();
    test_schema_pattern_regex_syntax_is_rejected_not_reinterpreted();
    test_schema_number_bounds_reject_dead_minus_and_close_in_bounds();
    test_schema_merges_enum_and_const_anyof();
    test_schema_integer_bounds_are_enforced();
    test_schema_integer_bounds_complete_truncation();
    test_schema_number_bounds_are_enforced();
    test_schema_additional_properties();
    test_schema_empty_closed_object();
    test_schema_rejects_required_without_properties();
    test_schema_accepts_annotation_keywords();
    test_schema_accepts_claude_open_metadata_object();
    test_schema_typed_additional_properties_map();
    test_schema_oneof_const_scalars();
    test_schema_oneof_const_numeric_prefixes();
    test_schema_rejects_oversized_oneof_const_scalars();
    test_schema_string_length_bounds_close_and_reject();
    test_schema_string_minlength_full_close();
    test_schema_discriminated_action_args();
    test_schema_union_dispatch_rules();
    test_schema_discriminated_required_must_match();
    test_schema_long_string_value();
    test_schema_escape_at_maxlength_rejected();
    test_schema_nested_tool_key_keeps_discriminator();
    test_schema_close_mid_discriminator_matches_args();
    test_schema_rejects_discriminator_after_conditional_args();
    test_escape_replaces_ill_formed_utf8();
    test_tojson_dump_matches_jinja();
    puts("json/schema tests ok");
    return 0;
}
