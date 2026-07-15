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

int main(void) {
    test_json_close_partial_string();
    test_schema_required_close();
    test_schema_rejects_bad_bounds();
    test_schema_rejects_escaped_keys();
    test_schema_oneof_const_scalars();
    puts("json/schema tests ok");
    return 0;
}
