// Strict tool-call envelope: OpenAI tools[] compiled into a discriminated
// union the sampler can enforce, and the generated envelope mapped back to
// OpenAI tool_calls.
//
// The point of the envelope is that the *schema engine* does the guaranteeing,
// so these tests do not stop at "the right JSON text was produced": every
// envelope is handed to schema_compile and driven through the same sval
// validator the sampler uses. A branch the validator would not enforce is a
// branch the model can escape.
#include "runner.h"
#include "json.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static jv *parse(const char *s) {
    jv *v = json_parse(s, strlen(s));
    assert(v != NULL);
    return v;
}

// compile an envelope's schema the way the server does
static snode *compile(const tool_envelope *e) {
    jv *j = parse(e->schema_src);
    char err[192];
    snode *n = schema_compile(j, err, sizeof(err));
    if (!n) fprintf(stderr, "envelope did not compile: %s\n%s\n", err,
                    e->schema_src);
    jv_free(j);
    return n;
}

// does the compiled envelope accept this complete document?
static bool accepts(const snode *root, const char *doc) {
    sval v;
    sval_init(&v, root);
    return sval_feed(&v, doc, (int)strlen(doc)) && v.done;
}

static const char *TOOLS =
    "[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\","
      "\"description\":\"look up weather\",\"parameters\":{\"type\":\"object\","
      "\"properties\":{\"city\":{\"type\":\"string\"},"
      "\"units\":{\"enum\":[\"c\",\"f\"]}},\"required\":[\"city\",\"units\"]}}},"
     "{\"type\":\"function\",\"function\":{\"name\":\"add\","
      "\"parameters\":{\"type\":\"object\","
      "\"properties\":{\"a\":{\"type\":\"integer\"},\"b\":{\"type\":\"integer\"}},"
      "\"required\":[\"a\",\"b\"]}}}]";

// The headline guarantee: the model cannot invent a tool name, cannot invent
// an argument key, and cannot get an argument's type wrong — the union is
// enforced during sampling rather than parsed hopefully afterward.
static void test_auto_envelope_constrains_names_and_arguments(void) {
    jv *tools = parse(TOOLS);
    tool_envelope e;
    char err[192];
    assert(tool_envelope_build(tools, NULL, NULL, &e, err, sizeof(err)) == 1);
    snode *root = compile(&e);
    assert(root != NULL);

    assert(accepts(root, "{\"tool\":\"get_weather\","
                         "\"args\":{\"city\":\"Oslo\",\"units\":\"c\"}}"));
    assert(accepts(root, "{\"tool\":\"add\",\"args\":{\"a\":1,\"b\":2}}"));
    // the final branch carries an ordinary assistant reply
    assert(accepts(root, "{\"tool\":\"final\",\"args\":{\"content\":\"hi\"}}"));

    // invented tool name
    assert(!accepts(root, "{\"tool\":\"rm_rf\",\"args\":{}}"));
    // right tool, wrong argument key
    assert(!accepts(root, "{\"tool\":\"add\",\"args\":{\"x\":1,\"b\":2}}"));
    // right tool, wrong argument type
    assert(!accepts(root, "{\"tool\":\"add\",\"args\":{\"a\":\"1\",\"b\":2}}"));
    // arguments belonging to the *other* branch
    assert(!accepts(root, "{\"tool\":\"add\",\"args\":{\"city\":\"Oslo\","
                          "\"units\":\"c\"}}"));
    // a required argument left out
    assert(!accepts(root, "{\"tool\":\"add\",\"args\":{\"a\":1}}"));

    schema_free(root);
    tool_envelope_free(&e);
    jv_free(tools);
}

// max_tokens can cut generation anywhere. sval_close must still produce a
// document that parses AND that the envelope itself accepts, or the caller
// receives a tool call it cannot execute.
static void test_truncated_call_stays_valid_and_executable(void) {
    jv *tools = parse(TOOLS);
    tool_envelope e;
    char err[192];
    assert(tool_envelope_build(tools, NULL, NULL, &e, err, sizeof(err)) == 1);
    snode *root = compile(&e);
    assert(root != NULL);

    const char *prefix = "{\"tool\":\"get_weather\",\"args\":{\"city\":\"Os";
    for (size_t cut = 1; cut <= strlen(prefix); cut++) {
        sval v;
        sval_init(&v, root);
        if (!sval_feed(&v, prefix, (int)cut)) continue; // not a reachable prefix
        char tail[512];
        int n = sval_close(&v, tail, sizeof(tail));
        assert(n >= 0);

        char doc[1024];
        snprintf(doc, sizeof(doc), "%.*s%s", (int)cut, prefix, tail);
        jv *parsed = json_parse(doc, strlen(doc));
        assert(parsed != NULL);          // valid JSON
        assert(accepts(root, doc));      // and still a legal envelope

        // and it maps back to something executable
        sbuf content = {0}, tc = {0};
        int rc = tool_envelope_map(&e, doc, strlen(doc), &content, &tc);
        assert(rc == 0 || rc == 1);
        if (rc == 1) assert(tc.n > 0);
        free(content.s);
        free(tc.s);
        jv_free(parsed);
    }
    schema_free(root);
    tool_envelope_free(&e);
    jv_free(tools);
}

static void test_tool_choice_required_removes_the_final_branch(void) {
    jv *tools = parse(TOOLS);
    jv *choice = parse("\"required\"");
    tool_envelope e;
    char err[192];
    assert(tool_envelope_build(tools, choice, NULL, &e, err, sizeof(err)) == 1);
    snode *root = compile(&e);
    assert(root != NULL);

    assert(accepts(root, "{\"tool\":\"add\",\"args\":{\"a\":1,\"b\":2}}"));
    // "required" means a tool call is the only legal output
    assert(!accepts(root, "{\"tool\":\"final\",\"args\":{\"content\":\"hi\"}}"));

    schema_free(root);
    tool_envelope_free(&e);
    jv_free(choice);
    jv_free(tools);
}

static void test_tool_choice_named_leaves_exactly_one_branch(void) {
    jv *tools = parse(TOOLS);
    jv *choice = parse("{\"type\":\"function\",\"function\":{\"name\":\"add\"}}");
    tool_envelope e;
    char err[192];
    assert(tool_envelope_build(tools, choice, NULL, &e, err, sizeof(err)) == 1);
    snode *root = compile(&e);
    assert(root != NULL);

    assert(accepts(root, "{\"tool\":\"add\",\"args\":{\"a\":1,\"b\":2}}"));
    assert(!accepts(root, "{\"tool\":\"get_weather\","
                          "\"args\":{\"city\":\"Oslo\",\"units\":\"c\"}}"));
    assert(!accepts(root, "{\"tool\":\"final\",\"args\":{\"content\":\"hi\"}}"));

    schema_free(root);
    tool_envelope_free(&e);
    jv_free(choice);
    jv_free(tools);
}

// "none" is not a constraint the envelope can express — it is the absence of
// tool calling — so the build declines strict mode rather than inventing one.
static void test_tool_choice_none_declines_strict_mode(void) {
    jv *tools = parse(TOOLS);
    jv *choice = parse("\"none\"");
    tool_envelope e;
    char err[192];
    assert(tool_envelope_build(tools, choice, NULL, &e, err, sizeof(err)) == 0);
    jv_free(choice);
    jv_free(tools);

    // and no tools at all is likewise not an error, just not strict
    assert(tool_envelope_build(NULL, NULL, NULL, &e, err, sizeof(err)) == 0);
}

// tools and response_format in the same request: the caller's schema becomes
// the shape of the final branch, so "answer me in this JSON, or call a tool"
// is one union and both halves are guaranteed.
static void test_response_format_schema_becomes_the_final_branch(void) {
    jv *tools = parse(TOOLS);
    jv *final = parse("{\"type\":\"object\",\"properties\":"
                      "{\"answer\":{\"type\":\"string\"}},"
                      "\"required\":[\"answer\"]}");
    tool_envelope e;
    char err[192];
    assert(tool_envelope_build(tools, NULL, final, &e, err, sizeof(err)) == 1);
    snode *root = compile(&e);
    assert(root != NULL);

    assert(accepts(root, "{\"tool\":\"final\",\"args\":{\"answer\":\"42\"}}"));
    assert(!accepts(root, "{\"tool\":\"final\",\"args\":{\"content\":\"42\"}}"));
    assert(accepts(root, "{\"tool\":\"add\",\"args\":{\"a\":1,\"b\":2}}"));

    // the final branch maps back as the caller's own JSON document, not as a
    // string field lifted out of it
    sbuf content = {0}, tc = {0};
    const char *doc = "{\"tool\":\"final\",\"args\":{\"answer\":\"42\"}}";
    assert(tool_envelope_map(&e, doc, strlen(doc), &content, &tc) == 0);
    assert(content.s && strstr(content.s, "\"answer\""));
    assert(tc.n == 0);
    free(content.s);
    free(tc.s);

    schema_free(root);
    tool_envelope_free(&e);
    jv_free(final);
    jv_free(tools);
}

static void test_map_produces_openai_tool_call_items(void) {
    jv *tools = parse(TOOLS);
    tool_envelope e;
    char err[192];
    assert(tool_envelope_build(tools, NULL, NULL, &e, err, sizeof(err)) == 1);

    sbuf content = {0}, tc = {0};
    const char *doc = "{\"tool\":\"get_weather\","
                      "\"args\":{\"city\":\"Oslo\",\"units\":\"c\"}}";
    assert(tool_envelope_map(&e, doc, strlen(doc), &content, &tc) == 1);
    assert(content.n == 0);           // no content alongside a tool call
    assert(tc.s != NULL);

    // the item must be a well-formed OpenAI tool_calls entry whose arguments
    // are a JSON *string* holding the argument document
    char wrapped[1024];
    snprintf(wrapped, sizeof(wrapped), "[%.*s]", (int)tc.n, tc.s);
    jv *arr = parse(wrapped);
    assert(arr->type == J_ARR && arr->n == 1);
    jv *fn = jv_get(arr->items[0], "function");
    assert(fn != NULL);
    assert(!strcmp(jv_str(jv_get(arr->items[0], "type"), ""), "function"));
    assert(jv_str(jv_get(arr->items[0], "id"), NULL) != NULL);
    assert(!strcmp(jv_str(jv_get(fn, "name"), ""), "get_weather"));
    const char *args = jv_str(jv_get(fn, "arguments"), NULL);
    assert(args != NULL);
    jv *parsed_args = parse(args);
    assert(!strcmp(jv_str(jv_get(parsed_args, "city"), ""), "Oslo"));

    jv_free(parsed_args);
    jv_free(arr);
    free(content.s);
    free(tc.s);

    // the final branch is content, not a call
    sbuf c2 = {0}, t2 = {0};
    const char *fdoc = "{\"tool\":\"final\",\"args\":{\"content\":\"hello\"}}";
    assert(tool_envelope_map(&e, fdoc, strlen(fdoc), &c2, &t2) == 0);
    assert(c2.s && !strncmp(c2.s, "hello", 5) && c2.n == 5);
    assert(t2.n == 0);
    free(c2.s);
    free(t2.s);

    tool_envelope_free(&e);
    jv_free(tools);
}

// Malformed declarations are rejected at request time. Accepting them would
// mean compiling a union that does not describe the tools the caller has, and
// then guaranteeing it.
static void test_malformed_tool_declarations_are_rejected(void) {
    static const char *const bad[] = {
        "{\"not\":\"an array\"}",
        "[{\"type\":\"function\"}]",                             // no function
        "[{\"type\":\"function\",\"function\":{}}]",             // no name
        "[{\"type\":\"function\",\"function\":{\"name\":\"\"}}]", // empty name
        "[{\"type\":\"retrieval\",\"function\":{\"name\":\"a\"}}]",
        "[{\"type\":\"function\",\"function\":{\"name\":\"a\",\"parameters\":7}}]",
        // duplicate names would make the discriminator ambiguous
        "[{\"type\":\"function\",\"function\":{\"name\":\"a\"}},"
         "{\"type\":\"function\",\"function\":{\"name\":\"a\"}}]",
        // "final" is the reserved discriminator for the no-call branch
        "[{\"type\":\"function\",\"function\":{\"name\":\"final\"}}]",
        // a parameters schema the compiler cannot enforce must not be
        // silently approximated
        "[{\"type\":\"function\",\"function\":{\"name\":\"a\",\"parameters\":"
         "{\"type\":\"object\",\"properties\":{\"p\":{\"type\":\"string\","
         "\"pattern\":\"^x$\"}}}}}]",
    };
    for (size_t i = 0; i < sizeof(bad) / sizeof(*bad); i++) {
        jv *tools = parse(bad[i]);
        tool_envelope e;
        char err[192];
        int rc = tool_envelope_build(tools, NULL, NULL, &e, err, sizeof(err));
        if (rc == 1) {
            // the builder accepted it; the compiler is the last line of defence
            snode *root = compile(&e);
            assert(root == NULL);
            tool_envelope_free(&e);
        } else {
            assert(rc == -1);
            assert(err[0] != 0);
        }
        jv_free(tools);
    }
}

static void test_malformed_tool_choice_is_rejected(void) {
    static const char *const bad[] = {
        "\"maybe\"", "7", "[]",
        "{\"type\":\"retrieval\",\"function\":{\"name\":\"add\"}}",
        "{\"type\":\"function\"}",
        "{\"type\":\"function\",\"function\":{\"name\":\"nope\"}}", // not declared
    };
    jv *tools = parse(TOOLS);
    for (size_t i = 0; i < sizeof(bad) / sizeof(*bad); i++) {
        jv *choice = parse(bad[i]);
        tool_envelope e;
        char err[192];
        assert(tool_envelope_build(tools, choice, NULL, &e, err,
                                   sizeof(err)) == -1);
        assert(err[0] != 0);
        jv_free(choice);
    }
    // "required" / a named tool without any tools is a contradiction, not a
    // request to answer normally
    jv *req_choice = parse("\"required\"");
    tool_envelope e;
    char err[192];
    assert(tool_envelope_build(NULL, req_choice, NULL, &e, err,
                               sizeof(err)) == -1);
    jv_free(req_choice);
    jv_free(tools);
}

// A tool with no parameters is legal and must still round-trip.
static void test_parameterless_tool(void) {
    jv *tools = parse("[{\"type\":\"function\","
                      "\"function\":{\"name\":\"ping\"}}]");
    tool_envelope e;
    char err[192];
    assert(tool_envelope_build(tools, NULL, NULL, &e, err, sizeof(err)) == 1);
    snode *root = compile(&e);
    assert(root != NULL);
    assert(accepts(root, "{\"tool\":\"ping\",\"args\":{}}"));

    sbuf content = {0}, tc = {0};
    const char *doc = "{\"tool\":\"ping\",\"args\":{}}";
    assert(tool_envelope_map(&e, doc, strlen(doc), &content, &tc) == 1);
    assert(tc.s && strstr(tc.s, "ping"));
    free(content.s);
    free(tc.s);

    schema_free(root);
    tool_envelope_free(&e);
    jv_free(tools);
}

// The system turn has to actually name the tools and the envelope, or the
// model has no way to produce what the sampler is willing to accept.
static void test_system_turn_teaches_the_envelope(void) {
    jv *tools = parse(TOOLS);
    tool_envelope e;
    char err[192];
    assert(tool_envelope_build(tools, NULL, NULL, &e, err, sizeof(err)) == 1);
    assert(e.system_turn != NULL);
    assert(strstr(e.system_turn, "get_weather"));
    assert(strstr(e.system_turn, "\"tool\""));
    assert(strstr(e.system_turn, "\"args\""));
    assert(strstr(e.system_turn, "final"));   // the no-call branch is offered
    tool_envelope_free(&e);

    jv *choice = parse("\"required\"");
    assert(tool_envelope_build(tools, choice, NULL, &e, err, sizeof(err)) == 1);
    // under "required" there is no final branch, so it must not be advertised
    assert(strstr(e.system_turn, "\"final\"") == NULL);
    tool_envelope_free(&e);
    jv_free(choice);
    jv_free(tools);
}

int main(void) {
    test_auto_envelope_constrains_names_and_arguments();
    test_truncated_call_stays_valid_and_executable();
    test_tool_choice_required_removes_the_final_branch();
    test_tool_choice_named_leaves_exactly_one_branch();
    test_tool_choice_none_declines_strict_mode();
    test_response_format_schema_becomes_the_final_branch();
    test_map_produces_openai_tool_call_items();
    test_malformed_tool_declarations_are_rejected();
    test_malformed_tool_choice_is_rejected();
    test_parameterless_tool();
    test_system_turn_teaches_the_envelope();
    puts("tool envelope tests ok");
    return 0;
}
