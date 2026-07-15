#include "runner.h"
#include "json.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

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

int main(void) {
    test_json_close_partial_string();
    test_schema_required_close();
    test_schema_rejects_bad_bounds();
    test_schema_rejects_escaped_keys();
    test_schema_oneof_const_scalars();
    test_schema_oneof_const_numeric_prefixes();
    test_schema_string_length_bounds_close_and_reject();
    test_schema_discriminated_action_args();
    puts("json/schema tests ok");
    return 0;
}
