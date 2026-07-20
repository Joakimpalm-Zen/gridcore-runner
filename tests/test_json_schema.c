#include "runner.h"
#include "json.h"

#include <assert.h>
#include <stdio.h>
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
// silently weaker constraint. `pattern` is the headline case: dropping it
// leaves a string unconstrained in exactly the way the caller asked it not
// to be.
static void test_schema_rejects_unenforceable_keywords(void) {
    static const char *const bad[] = {
        "{\"type\":\"string\",\"pattern\":\"^[a-z]+$\"}",
        "{\"type\":\"string\",\"format\":\"date-time\"}",
        "{\"type\":\"integer\",\"minimum\":0}",
        "{\"type\":\"integer\",\"maximum\":10}",
        "{\"type\":\"integer\",\"multipleOf\":2}",
        "{\"allOf\":[{\"type\":\"string\"}]}",
        "{\"not\":{\"type\":\"string\"}}",
        "{\"$ref\":\"#/$defs/x\"}",
        "{\"type\":\"array\",\"uniqueItems\":true}",
        "{\"type\":\"array\",\"prefixItems\":[{\"type\":\"string\"}]}",
        "{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"string\"}},"
            "\"patternProperties\":{\"^x\":{\"type\":\"string\"}}}",
        // nested: the check must run at every depth, not just the root
        "{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"string\","
            "\"pattern\":\"x\"}}}",
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

// The other half of the invariant: keywords that are pure annotations carry
// no constraint, so ignoring them ignores nothing. Real OpenAI tool payloads
// are full of them and must keep compiling.
static void test_schema_accepts_annotation_keywords(void) {
    const char *src =
        "{\"type\":\"object\",\"title\":\"T\",\"description\":\"d\","
        "\"$schema\":\"https://json-schema.org/draft/2020-12/schema\","
        "\"properties\":{\"a\":{\"type\":\"string\",\"description\":\"d\","
        "\"default\":\"x\",\"examples\":[\"y\"]}},\"required\":[\"a\"],"
        "\"additionalProperties\":false}";
    jv *schema_json = json_parse(src, strlen(src));
    assert(schema_json != NULL);
    char err[128];
    snode *schema = schema_compile(schema_json, err, sizeof(err));
    assert(schema != NULL);
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
        // no properties: the generic any-object machine cannot close the set
        "{\"type\":\"object\",\"additionalProperties\":false}",
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
    sval_init(&v, schema);
    char out[128];
    int n = sval_close(&v, out, sizeof(out));
    assert(n > 0);
    jv *parsed = json_parse(out, strlen(out));
    assert(parsed != NULL);
    assert(strlen(jv_str(jv_get(parsed, "name"), "")) >= 3);
    jv_free(parsed);

    sval_init(&v, schema);
    assert(sval_feed(&v, "{\"name\":", 8));
    n = sval_close(&v, out, sizeof(out));
    assert(n > 0);
    char full[256];
    snprintf(full, sizeof(full), "{\"name\":%s", out);
    parsed = json_parse(full, strlen(full));
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

int main(void) {
    test_strict_bounded_numbers();
    test_json_close_partial_string();
    test_schema_required_close();
    test_schema_rejects_empty_type_union();
    test_schema_huge_min_bounds_terminate();
    test_schema_rejects_bad_bounds();
    test_schema_rejects_non_integer_or_huge_bounds();
    test_schema_rejects_escaped_keys();
    test_schema_rejects_unenforceable_keywords();
    test_schema_additional_properties();
    test_schema_rejects_required_without_properties();
    test_schema_accepts_annotation_keywords();
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
    puts("json/schema tests ok");
    return 0;
}
