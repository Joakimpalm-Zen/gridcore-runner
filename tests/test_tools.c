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
    for (int i = 0; doc[i]; i++) {
        if (!sval_feed(&v, doc + i, 1)) {
            if (getenv("RUNNER_SCHEMA_TRACE"))
                fprintf(stderr, "schema rejected byte %d (%#x), depth=%d "
                        "kind=%d phase=%d idx=%d lit=%s in %s\n",
                        i, (unsigned char)doc[i], v.depth,
                        v.stack[v.depth - 1].node->kind,
                        v.stack[v.depth - 1].phase,
                        v.stack[v.depth - 1].idx,
                        v.stack[v.depth - 1].node->n_lits
                            ? v.stack[v.depth - 1].node->lits[0] : "-",
                        doc);
            return false;
        }
    }
    return v.done;
}

static void test_atem_structured_tool_automaton(void) {
    jv *tools = parse(
        "[{\"type\":\"function\",\"function\":{\"name\":\"data.store\","
        "\"parameters\":{\"type\":\"object\",\"properties\":{"
        "\"payload\":{\"type\":\"object\",\"properties\":{\"x\":{\"type\":\"integer\"}},\"required\":[\"x\"]},"
        "\"tags\":{\"type\":\"array\",\"items\":{\"type\":\"string\"}}},"
        "\"required\":[\"payload\",\"tags\"]}}},"
        "{\"type\":\"function\",\"function\":{\"name\":\"data.clear\","
        "\"parameters\":{\"type\":\"object\",\"properties\":{},\"required\":[]}}}]"
    );
    char err[192];
    snode *root = schema_compile_atem_tools(tools, err, sizeof(err));
    if (!root) fprintf(stderr, "atem tools did not compile: %s\n", err);
    assert(root != NULL);
    assert(accepts(root,
        "<atem:function_calls>\n<atem:invoke name=\"data.store\">\n"
        "<atem:parameter name=\"payload\">{\"x\":2}</atem:parameter>\n"
        "<atem:parameter name=\"tags\">[\"a\",\"b\"]</atem:parameter>\n"
        "</atem:invoke>\n</atem:function_calls>"));
    assert(accepts(root,
        "<atem:function_calls>\n<atem:invoke name=\"data.clear\">\n"
        "</atem:invoke>\n</atem:function_calls>"));
    assert(!accepts(root,
        "<atem:function_calls>\n<atem:invoke name=\"data.store\">\n"
        "<atem:parameter name=\"tags\">[]</atem:parameter>\n"
        "<atem:parameter name=\"payload\">{\"x\":2}</atem:parameter>\n"
        "</atem:invoke>\n</atem:function_calls>"));
    assert(!accepts(root,
        "<atem:function_calls>\n<atem:invoke name=\"data.clear\">\n"
        "</atem:invoke>"));
    schema_free(root);
    jv_free(tools);
}

static void test_atem_scalar_is_raw_until_parameter_close(void) {
    jv *tools = parse(
        "[{\"type\":\"function\",\"function\":{\"name\":\"notes.save\","
        "\"parameters\":{\"type\":\"object\",\"properties\":{"
        "\"text\":{\"type\":\"string\"}},\"required\":[\"text\"]}}}]"
    );
    char err[192];
    snode *root = schema_compile_atem_tools(tools, err, sizeof(err));
    if (!root) fprintf(stderr, "raw atem tool did not compile: %s\n", err);
    assert(root != NULL);
    assert(accepts(root,
        "<atem:function_calls>\n<atem:invoke name=\"notes.save\">\n"
        "<atem:parameter name=\"text\">raw \"quotes\" and \\slashes"
        "</atem:parameter>\n</atem:invoke>\n</atem:function_calls>"));
    assert(!accepts(root,
        "<atem:function_calls>\n<atem:invoke name=\"notes.save\">\n"
        "<atem:parameter name=\"text\">unterminated\n"
        "</atem:invoke>\n</atem:function_calls>"));
    schema_free(root);
    jv_free(tools);
}

static void test_atem_truncation_closes_started_call(void) {
    jv *tools = parse(
        "[{\"type\":\"function\",\"function\":{\"name\":\"notes.save\","
        "\"parameters\":{\"type\":\"object\",\"properties\":{"
        "\"text\":{\"type\":\"string\"}},\"required\":[\"text\"]}}}]"
    );
    char err[192];
    snode *root = schema_compile_atem_tools(tools, err, sizeof(err));
    assert(root != NULL);
    const char *prefix =
        "<atem:function_calls>\n<atem:invoke name=\"notes.save\">\n"
        "<atem:parameter name=\"text\">half-written <";
    sval partial;
    sval_init(&partial, root);
    assert(sval_feed(&partial, prefix, (int)strlen(prefix)));
    char tail[512];
    int tail_n = sval_close(&partial, tail, sizeof(tail));
    assert(tail_n > 0);
    char full[1024];
    snprintf(full, sizeof(full), "%s%s", prefix, tail);
    assert(accepts(root, full));
    assert(strstr(full, "</atem:parameter>\n</atem:invoke>\n"
                        "</atem:function_calls>") != NULL);
    schema_free(root);
    jv_free(tools);
}

static void test_atem_buffered_maps_reasoning_and_multiple_calls(void) {
    tool_envelope e = {0};
    e.atem = true;
    const char *doc =
        " to=self<|message|>check both cities<|eom|><|start|>assistant"
        " to=weather.get<|message|><atem:function_calls>\n"
        "<atem:invoke name=\"weather.get\">\n"
        "<atem:parameter name=\"city\">Oslo</atem:parameter>\n"
        "</atem:invoke>\n</atem:function_calls><|eom|><|start|>assistant"
        " to=weather.get<|message|><atem:function_calls>\n"
        "<atem:invoke name=\"weather.get\">\n"
        "<atem:parameter name=\"city\">Bergen</atem:parameter>\n"
        "</atem:invoke>\n</atem:function_calls>";
    sbuf content = {0}, calls = {0};
    assert(tool_envelope_map(&e, doc, strlen(doc), &content, &calls) == 2);
    assert(content.n == 0);
    assert(calls.s && strstr(calls.s, "\"id\":\"call_0\""));
    assert(strstr(calls.s, "\"id\":\"call_1\""));
    assert(strstr(calls.s, "\\\"city\\\":\\\"Oslo\\\""));
    assert(strstr(calls.s, "\\\"city\\\":\\\"Bergen\\\""));
    free(content.s);
    free(calls.s);
}

static void test_atem_header_discriminates_matching_invoke(void) {
    jv *tools = parse(
        "[{\"type\":\"function\",\"function\":{\"name\":\"data.store\","
        "\"parameters\":{\"type\":\"object\",\"properties\":{},\"required\":[]}}},"
        "{\"type\":\"function\",\"function\":{\"name\":\"data.clear\","
        "\"parameters\":{\"type\":\"object\",\"properties\":{},\"required\":[]}}}]"
    );
    char err[192];
    snode *root = schema_compile_atem_turn(
        tools, true, NULL, NULL, ATEM_TURN_DIRECT, err, sizeof(err));
    assert(root != NULL);
    assert(accepts(root, " to=data.store<|message|><atem:function_calls>\n"
        "<atem:invoke name=\"data.store\">\n</atem:invoke>\n</atem:function_calls>"));
    assert(!accepts(root, " to=data.store<|message|><atem:function_calls>\n"
        "<atem:invoke name=\"data.clear\">\n</atem:invoke>\n</atem:function_calls>"));
    assert(accepts(root, " to=user<|message|>plain answer<|eot|>"));
    schema_free(root);
    root = schema_compile_atem_turn(tools, false, "data.clear", NULL,
                                    ATEM_TURN_AFTER_REASONING,
                                    err, sizeof(err));
    assert(root != NULL);
    assert(accepts(root, "data.clear<|message|><atem:function_calls>\n"
        "<atem:invoke name=\"data.clear\">\n</atem:invoke>\n</atem:function_calls>"));
    schema_free(root);
    root = schema_compile_atem_turn(tools, false, "data.clear", NULL,
                                    ATEM_TURN_EITHER, err, sizeof(err));
    assert(root != NULL);
    assert(accepts(root, " to=data.clear<|message|><atem:function_calls>\n"
        "<atem:invoke name=\"data.clear\">\n</atem:invoke>\n</atem:function_calls>"));
    assert(accepts(root, "data.clear<|message|><atem:function_calls>\n"
        "<atem:invoke name=\"data.clear\">\n</atem:invoke>\n</atem:function_calls>"));
    schema_free(root);
    jv_free(tools);
}

// The to=user free-text answer ends at the model's own stop token, which
// decodes to no bytes — the automaton's spelled `<|eot|>` sentinel never
// arrives on the byte stream. sval_at_raw_tail is what lets the engine accept
// a stop there; without it every native plain answer burned the full token
// budget and finished "length" (measured live 2026-08-11).
static void test_atem_stop_token_is_valid_at_the_raw_answer_tail(void) {
    jv *tools = parse(
        "[{\"type\":\"function\",\"function\":{\"name\":\"weather.get\","
        "\"parameters\":{\"type\":\"object\",\"properties\":{"
        "\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}}}]"
    );
    char err[192];
    snode *root = schema_compile_atem_turn(
        tools, true, NULL, NULL, ATEM_TURN_DIRECT, err, sizeof(err));
    assert(root != NULL);

    sval v;
    sval_init(&v, root);
    const char *answer = " to=user<|message|>The answer is 391";
    assert(sval_feed(&v, answer, (int)strlen(answer)));
    assert(sval_at_raw_tail(&v));      // mid-answer: stop is a natural end

    sval_init(&v, root);
    const char *empty = " to=user<|message|>";
    assert(sval_feed(&v, empty, (int)strlen(empty)));
    assert(sval_at_raw_tail(&v));      // an empty answer is still an answer

    sval_init(&v, root);
    const char *header = " to=us";
    assert(sval_feed(&v, header, (int)strlen(header)));
    assert(!sval_at_raw_tail(&v));     // mid-header: stop would truncate it

    sval_init(&v, root);
    const char *param = " to=weather.get<|message|><atem:function_calls>\n"
        "<atem:invoke name=\"weather.get\">\n"
        "<atem:parameter name=\"city\">Osl";
    assert(sval_feed(&v, param, (int)strlen(param)));
    assert(!sval_at_raw_tail(&v));     // a parameter value is NOT the tail:
                                       // close tags must still be emitted
    schema_free(root);
    jv_free(tools);
}

static void test_atem_declared_optional_parameters_are_constrained(void) {
    jv *tools = parse(
        "[{\"type\":\"function\",\"function\":{\"name\":\"search\","
        "\"parameters\":{\"type\":\"object\",\"properties\":{"
        "\"query\":{\"type\":\"string\"},\"limit\":{\"type\":\"integer\"}},"
        "\"required\":[\"query\"]}}}]"
    );
    char err[192];
    snode *root = schema_compile_atem_turn(
        tools, true, NULL, NULL, ATEM_TURN_DIRECT, err, sizeof(err));
    assert(root != NULL);
    assert(accepts(root, " to=search<|message|><atem:function_calls>\n"
        "<atem:invoke name=\"search\">\n"
        "<atem:parameter name=\"query\">muse</atem:parameter>\n"
        "<atem:parameter name=\"limit\">5</atem:parameter>\n"
        "</atem:invoke>\n</atem:function_calls>"));
    tool_envelope e = {.atem = true, .tools = tools};
    const char *doc = " to=search<|message|><atem:function_calls>\n"
        "<atem:invoke name=\"search\">\n"
        "<atem:parameter name=\"query\">muse</atem:parameter>\n"
        "<atem:parameter name=\"limit\">5</atem:parameter>\n"
        "</atem:invoke>\n</atem:function_calls>";
    sbuf content = {0}, calls = {0};
    assert(tool_envelope_map(&e, doc, strlen(doc), &content, &calls) == 1);
    assert(calls.s && strstr(calls.s, "{\\\"query\\\":\\\"muse\\\",\\\"limit\\\":5}"));
    free(content.s); free(calls.s);
    schema_free(root);
    jv_free(tools);
}

static void test_atem_parallel_turn_constrains_two_recipient_calls(void) {
    jv *tools = parse(
        "[{\"type\":\"function\",\"function\":{\"name\":\"weather.get\","
        "\"parameters\":{\"type\":\"object\",\"properties\":{"
        "\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}}}]"
    );
    char err[192];
    snode *root = schema_compile_atem_parallel(tools, NULL, err, sizeof(err));
    assert(root != NULL);
    const char *doc =
        " to=weather.get<|message|><atem:function_calls>\n"
        "<atem:invoke name=\"weather.get\">\n"
        "<atem:parameter name=\"city\">Oslo</atem:parameter>\n"
        "</atem:invoke>\n</atem:function_calls>assistant"
        " to=weather.get<|message|><atem:function_calls>\n"
        "<atem:invoke name=\"weather.get\">\n"
        "<atem:parameter name=\"city\">Bergen</atem:parameter>\n"
        "</atem:invoke>\n</atem:function_calls>";
    assert(accepts(root, doc));
    schema_free(root); jv_free(tools);
}

static void test_muse_generic_envelope_is_constrained_behind_user_recipient(void) {
    jv *schema = parse(
        "{\"type\":\"object\",\"properties\":{\"tool\":{\"const\":\"ping\"},"
        "\"args\":{\"type\":\"object\",\"properties\":{},\"required\":[]}},"
        "\"required\":[\"tool\",\"args\"]}"
    );
    char err[192];
    snode *root = schema_compile_muse_user_payload(schema, err, sizeof(err));
    assert(root != NULL);
    assert(accepts(root,
        " to=user<|message|>{\"tool\":\"ping\",\"args\":{}}"));
    assert(accepts(root,
        "user<|message|>{\"tool\":\"ping\",\"args\":{}}"));
    assert(!accepts(root,
        " to=ping<|message|>{\"tool\":\"ping\",\"args\":{}}"));
    schema_free(root);
    jv_free(schema);

    jv *tools = parse("[{\"type\":\"function\",\"function\":{\"name\":\"ping\","
                      "\"parameters\":{\"type\":\"object\",\"properties\":{},"
                      "\"required\":[]}}}]");
    tool_envelope e;
    assert(tool_envelope_build(tools, NULL, NULL, &e, err, sizeof(err)) == 1);
    e.muse_user_header = true;
    const char *doc = " to=user<|message|>{\"tool\":\"ping\",\"args\":{}}";
    sbuf content = {0}, calls = {0};
    assert(tool_envelope_map(&e, doc, strlen(doc), &content, &calls) == 1);
    assert(calls.s && strstr(calls.s, "\"name\":\"ping\""));
    free(content.s); free(calls.s);
    tool_envelope_free(&e); jv_free(tools);
}

static void test_atem_auto_user_branch_honors_response_schema(void) {
    jv *tools = parse("[{\"type\":\"function\",\"function\":{\"name\":\"ping\","
                      "\"parameters\":{\"type\":\"object\",\"properties\":{},"
                      "\"required\":[]}}}]");
    jv *final = parse("{\"type\":\"object\",\"properties\":{"
                      "\"summary\":{\"type\":\"string\"}},"
                      "\"required\":[\"summary\"]}");
    char err[192];
    snode *root = schema_compile_atem_turn(
        tools, true, NULL, final, ATEM_TURN_EITHER, err, sizeof(err));
    assert(root != NULL);
    assert(accepts(root,
        " to=user<|message|>{\"summary\":\"ok\"}"));
    assert(!accepts(root,
        " to=user<|message|>{\"summary\":4}"));
    schema_free(root); jv_free(final); jv_free(tools);
}

static void test_atem_truncated_string_enum_recovers_closest_member(void) {
    jv *tools = parse(
        "[{\"type\":\"function\",\"function\":{\"name\":\"classify\","
        "\"parameters\":{\"type\":\"object\",\"properties\":{"
        "\"label\":{\"type\":\"string\",\"enum\":[\"alpha\",\"beta\"]}},"
        "\"required\":[\"label\"]}}}]"
    );
    tool_envelope e = {.atem = true, .tools = tools};
    const char *doc = " to=classify<|message|><atem:function_calls>\n"
        "<atem:invoke name=\"classify\">\n"
        "<atem:parameter name=\"label\">bet</atem:parameter>\n"
        "</atem:invoke>\n</atem:function_calls>";
    sbuf content = {0}, calls = {0};
    assert(tool_envelope_map(&e, doc, strlen(doc), &content, &calls) == 1);
    assert(calls.s && strstr(calls.s, "{\\\"label\\\":\\\"beta\\\"}"));
    free(content.s); free(calls.s); jv_free(tools);
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

static void test_ornith_native_tool_protocol(void) {
    jv *tools = parse(TOOLS);
    sbuf prompt = {0};
    tools_render_for(TMPL_ORNITH, tools, &prompt);
    assert(prompt.s != NULL);
    assert(strstr(prompt.s, "# Tools\n"));
    assert(strstr(prompt.s, "<tools>\n"));
    assert(strstr(prompt.s, "\"name\":\"get_weather\""));
    assert(strstr(prompt.s, "<function=example_function_name>"));
    assert(strstr(prompt.s, "<parameter=example_parameter_1>"));

    jv *history = parse(
        "[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\","
        "\"arguments\":\"{\\\"city\\\":\\\"Oslo\\\",\\\"days\\\":2}\"}}]");
    sbuf replay = {0};
    tool_history_render_for(TMPL_ORNITH, history, &replay);
    assert(!strcmp(replay.s,
        "<tool_call>\n<function=get_weather>\n"
        "<parameter=city>\nOslo\n</parameter>\n"
        "<parameter=days>\n2\n</parameter>\n"
        "</function>\n</tool_call>"));

    sbuf content = {0}, calls = {0};
    sb_lit(&content,
        "checking\n<tool_call>\n<function=get_weather>\n"
        "<parameter=city>\nOslo\n</parameter>\n"
        "<parameter=units>\n\"c\"\n</parameter>\n"
        "</function>\n</tool_call>");
    assert(tool_calls_parse_for(TMPL_ORNITH, &content, &calls) == 1);
    assert(content.n == strlen("checking\n"));

    char wrapped[1024];
    snprintf(wrapped, sizeof(wrapped), "[%.*s]", (int)calls.n, calls.s);
    jv *arr = parse(wrapped);
    jv *fn = jv_get(arr->items[0], "function");
    assert(!strcmp(jv_str(jv_get(fn, "name"), ""), "get_weather"));
    jv *args = parse(jv_str(jv_get(fn, "arguments"), ""));
    assert(!strcmp(jv_str(jv_get(args, "city"), ""), "Oslo"));
    assert(!strcmp(jv_str(jv_get(args, "units"), ""), "c"));

    jv_free(args);
    jv_free(arr);
    free(content.s);
    free(calls.s);
    free(prompt.s);
    free(replay.s);
    jv_free(history);
    jv_free(tools);
}

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

// ------------------------------------------------- streaming demultiplexer
//
// The streaming path cannot wait for the whole document, so the same mapping
// runs incrementally. These tests drive it ONE BYTE AT A TIME, which is the
// worst case a real token stream can produce and the one that catches a
// decision made on a boundary the parser happened to like.

typedef struct {
    sbuf content, args;
    char name[64];
    int  begins;
} demux_log;

static int log_content(void *ud, const char *b, int n) {
    sb_put(&((demux_log *)ud)->content, b, n);
    return 0;
}
static int log_begin(void *ud, const char *name) {
    demux_log *l = ud;
    snprintf(l->name, sizeof(l->name), "%s", name);
    l->begins++;
    return 0;
}
static int log_args(void *ud, const char *b, int n) {
    sb_put(&((demux_log *)ud)->args, b, n);
    return 0;
}

// feed `doc` in chunks of `step` bytes (0 == one byte at a time) and return
// what the sink saw
static void demux_step(const tool_envelope *e, const char *doc, size_t step,
                       demux_log *l) {
    memset(l, 0, sizeof(*l));
    tool_stream_sink sink = { l, log_content, log_begin, log_args, NULL };
    tool_stream s;
    tool_stream_init(&s, e, &sink);
    size_t len = strlen(doc);
    if (!step) step = 1;
    for (size_t i = 0; i < len; i += step) {
        size_t k = len - i < step ? len - i : step;
        assert(tool_stream_feed(&s, doc + i, (int)k) == 0);
    }
    if (tool_stream_called(&s)) assert(l->begins >= 1);
    tool_stream_free(&s);
}

static void demux(const tool_envelope *e, const char *doc, demux_log *l) {
    demux_step(e, doc, 1, l);
}

static void log_free(demux_log *l) { free(l->content.s); free(l->args.s); }

// the same property the SSE boundary matrix asserts one level up: what the
// client sees may not depend on where the token boundaries happened to fall
static void demux_every_split(const tool_envelope *e, const char *doc) {
    demux_log ref;
    demux_step(e, doc, 1, &ref);
    for (size_t step = 2; step <= strlen(doc) + 1; step++) {
        demux_log got;
        demux_step(e, doc, step, &got);
        assert(got.begins == ref.begins);
        assert(!strcmp(got.name, ref.name));
        assert(got.content.n == ref.content.n);
        assert(got.args.n == ref.args.n);
        assert(!got.content.n || !memcmp(got.content.s, ref.content.s, ref.content.n));
        assert(!got.args.n || !memcmp(got.args.s, ref.args.s, ref.args.n));
        log_free(&got);
    }
    log_free(&ref);
}

// The headline guarantee of Phase 2: a streamed call arrives as tool-call
// arguments, never as content, and never with a byte of envelope syntax.
static void test_stream_demux_never_leaks_the_envelope(void) {
    jv *tools = parse(TOOLS);
    tool_envelope e;
    char err[192];
    assert(tool_envelope_build(tools, NULL, NULL, &e, err, sizeof(err)) == 1);

    demux_log l;
    demux(&e, "{\"tool\":\"get_weather\","
              "\"args\":{\"city\":\"Oslo\",\"units\":\"c\"}}", &l);
    assert(l.begins == 1);
    assert(!strcmp(l.name, "get_weather"));
    assert(l.content.n == 0);          // no content alongside a call
    assert(l.args.s != NULL);
    // the concatenated deltas are exactly the argument document
    assert(!strcmp(l.args.s, "{\"city\":\"Oslo\",\"units\":\"c\"}"));
    free(l.content.s);
    free(l.args.s);

    // the final branch is unescaped assistant text, and no call at all
    demux(&e, "{\"tool\":\"final\",\"args\":{\"content\":\"hi\\nthere\"}}", &l);
    assert(l.begins == 0);
    assert(l.args.n == 0);
    assert(l.content.s && !strcmp(l.content.s, "hi\nthere"));
    log_free(&l);

    tool_envelope_free(&e);
    jv_free(tools);
}

// A demux that decides differently depending on how the tokenizer happened to
// split the document would leak the envelope on some requests and not others.
static void test_stream_demux_is_boundary_independent(void) {
    jv *tools = parse(TOOLS);
    tool_envelope e;
    char err[192];
    assert(tool_envelope_build(tools, NULL, NULL, &e, err, sizeof(err)) == 1);

    demux_every_split(&e, "{\"tool\":\"add\",\"args\":{\"a\":1,\"b\":-20}}");
    demux_every_split(&e, "{\"tool\":\"final\",\"args\":"
                          "{\"content\":\"a \\\"quoted\\\" \\u00e5 tail\"}}");
    // whitespace the sampler is free to emit must not reach the client
    demux_every_split(&e, "{ \"tool\" : \"add\" , \"args\" : "
                          "{ \"a\" : 1 , \"b\" : 2 } }");

    demux_log l;
    demux(&e, "{ \"tool\" : \"add\" , \"args\" : { \"a\" : 1 , \"b\" : 2 } }", &l);
    assert(!strcmp(l.args.s, "{\"a\":1,\"b\":2}"));
    log_free(&l);

    // escapes are decoded exactly as the buffered mapper decodes them
    demux(&e, "{\"tool\":\"final\",\"args\":"
              "{\"content\":\"a \\\"q\\\" \\u00e5 \\ud83d\\ude00\"}}", &l);
    assert(!strcmp(l.content.s, "a \"q\" \xc3\xa5 \xf0\x9f\x98\x80"));
    log_free(&l);

    tool_envelope_free(&e);
    jv_free(tools);
}

static void test_atem_stream_demux_is_boundary_independent(void) {
    tool_envelope e = {0};
    e.atem = true;
    const char *doc =
        " to=notes.save<|message|><atem:function_calls>\n"
        "<atem:invoke name=\"notes.save\">\n"
        "<atem:parameter name=\"text\">raw \"quote\"</atem:parameter>\n"
        "</atem:invoke>\n</atem:function_calls><|eom|><|start|>assistant"
        " to=notes.save<|message|><atem:function_calls>\n"
        "<atem:invoke name=\"notes.save\">\n"
        "<atem:parameter name=\"text\">again</atem:parameter>\n"
        "</atem:invoke>\n</atem:function_calls>";
    demux_every_split(&e, doc);
    demux_log l;
    demux(&e, doc, &l);
    assert(l.begins == 2);
    assert(!strcmp(l.name, "notes.save"));
    assert(l.content.n == 0);
    assert(l.args.s && !strcmp(l.args.s,
        "{\"text\":\"raw \\\"quote\\\"\"}{\"text\":\"again\"}"));
    log_free(&l);

    demux(&e, " to=user<|message|>plain answer<|eot|>", &l);
    assert(l.begins == 0 && l.args.n == 0);
    assert(l.content.s && !strcmp(l.content.s, "plain answer"));
    log_free(&l);
}

static void test_muse_schema_payload_stream_hides_recipient_header(void) {
    tool_envelope e = {.muse_plain_payload = true};
    const char *doc = " to=user<|message|>{\"summary\":\"ok\"}";
    demux_every_split(&e, doc);
    demux_log l;
    demux(&e, doc, &l);
    assert(l.begins == 0 && l.args.n == 0);
    assert(l.content.s && !strcmp(l.content.s, "{\"summary\":\"ok\"}"));
    log_free(&l);
}

// max_tokens can cut the envelope anywhere. sval_close completes it before
// the last bytes reach us, but a prefix that stops earlier still must not
// leak: whatever was undecided stays held back rather than becoming content.
static void test_stream_demux_holds_back_an_undecided_prefix(void) {
    jv *tools = parse(TOOLS);
    tool_envelope e;
    char err[192];
    assert(tool_envelope_build(tools, NULL, NULL, &e, err, sizeof(err)) == 1);

    static const char *const doc =
        "{\"tool\":\"get_weather\",\"args\":{\"city\":\"Oslo\",\"units\":\"c\"}}";
    char prefix[128];
    for (size_t k = 1; k < strlen(doc); k++) {
        memcpy(prefix, doc, k);
        prefix[k] = 0;
        demux_log l;
        demux(&e, prefix, &l);
        // nothing that arrives is ever envelope syntax
        assert(l.content.n == 0);
        if (l.args.n)
            assert(l.args.s[0] == '{' && !strstr(l.args.s, "\"tool\""));
        // and a call is only announced once the tool is actually known
        if (l.begins) assert(!strcmp(l.name, "get_weather"));
        log_free(&l);
    }

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
// parallel_tool_calls: the document is {"calls":[<entry>, ...]} over the same
// discriminated union, so several calls map in one turn. Driven directly here
// rather than through a model, because the point is the mapping — which ids
// are assigned, how entries are separated, and what a mixed document does —
// and a sampled model would only ever exercise whichever shape it happened
// to emit.
static void test_parallel_envelope_maps_several_calls(void) {
    jv *tools = parse(TOOLS);
    tool_envelope e;
    char err[192];
    assert(tool_envelope_build_ex(tools, NULL, NULL, true, &e,
                                  err, sizeof(err)) == 1);
    assert(e.parallel);
    assert(e.max_calls > 1);

    // the schema must accept a multi-entry array
    snode *root = compile(&e);
    assert(root != NULL);
    const char *two =
        "{\"calls\":[{\"tool\":\"get_weather\","
        "\"args\":{\"city\":\"Oslo\",\"units\":\"c\"}},"
        "{\"tool\":\"get_weather\","
        "\"args\":{\"city\":\"Bergen\",\"units\":\"f\"}}]}";
    assert(accepts(root, two));

    sbuf content = {0}, tc = {0};
    int rc = tool_envelope_map(&e, two, strlen(two), &content, &tc);
    assert(rc == 2);
    assert(tc.s != NULL);
    tc.s[tc.n] = 0;
    // two complete items, separated, with distinct ascending ids
    assert(strstr(tc.s, "\"id\":\"call_0\"") != NULL);
    assert(strstr(tc.s, "\"id\":\"call_1\"") != NULL);
    assert(strstr(tc.s, "Oslo") != NULL && strstr(tc.s, "Bergen") != NULL);
    assert(strstr(tc.s, "}},{") != NULL);   // exactly one separator between them
    free(content.s); free(tc.s);

    // a single-entry array is the ordinary one-call turn
    const char *one =
        "{\"calls\":[{\"tool\":\"get_weather\","
        "\"args\":{\"city\":\"Oslo\",\"units\":\"c\"}}]}";
    content = (sbuf){0}; tc = (sbuf){0};
    assert(tool_envelope_map(&e, one, strlen(one), &content, &tc) == 1);
    free(content.s); free(tc.s);

    // the final branch rides the same array: one entry, answering directly
    const char *final_only =
        "{\"calls\":[{\"tool\":\"final\",\"args\":{\"content\":\"hi\"}}]}";
    content = (sbuf){0}; tc = (sbuf){0};
    assert(tool_envelope_map(&e, final_only, strlen(final_only),
                             &content, &tc) == 0);
    assert(content.n == 2 && memcmp(content.s, "hi", 2) == 0);
    assert(tc.n == 0);
    free(content.s); free(tc.s);

    // a document of the WRONG shape for this envelope is refused, not guessed
    const char *single_shape =
        "{\"tool\":\"get_weather\","
        "\"args\":{\"city\":\"Oslo\",\"units\":\"c\"}}";
    content = (sbuf){0}; tc = (sbuf){0};
    assert(tool_envelope_map(&e, single_shape, strlen(single_shape),
                             &content, &tc) == -1);
    free(content.s); free(tc.s);

    schema_free(root);
    tool_envelope_free(&e);
    jv_free(tools);
}

// The default build is unchanged by the new parameter: same single-object
// document, same mapping. A regression here would break every existing client.
static void test_default_envelope_is_unchanged(void) {
    jv *tools = parse(TOOLS);
    tool_envelope a, b;
    char err[192];
    assert(tool_envelope_build(tools, NULL, NULL, &a, err, sizeof(err)) == 1);
    assert(tool_envelope_build_ex(tools, NULL, NULL, false, &b,
                                  err, sizeof(err)) == 1);
    assert(!a.parallel && !b.parallel);
    assert(strcmp(a.schema_src, b.schema_src) == 0);
    assert(strcmp(a.system_turn, b.system_turn) == 0);
    tool_envelope_free(&a);
    tool_envelope_free(&b);
    jv_free(tools);
}

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
    test_atem_structured_tool_automaton();
    test_atem_scalar_is_raw_until_parameter_close();
    test_atem_truncation_closes_started_call();
    test_atem_buffered_maps_reasoning_and_multiple_calls();
    test_atem_header_discriminates_matching_invoke();
    test_atem_stop_token_is_valid_at_the_raw_answer_tail();
    test_atem_declared_optional_parameters_are_constrained();
    test_atem_parallel_turn_constrains_two_recipient_calls();
    test_muse_generic_envelope_is_constrained_behind_user_recipient();
    test_atem_auto_user_branch_honors_response_schema();
    test_atem_truncated_string_enum_recovers_closest_member();
    test_ornith_native_tool_protocol();
    test_auto_envelope_constrains_names_and_arguments();
    test_truncated_call_stays_valid_and_executable();
    test_tool_choice_required_removes_the_final_branch();
    test_tool_choice_named_leaves_exactly_one_branch();
    test_tool_choice_none_declines_strict_mode();
    test_response_format_schema_becomes_the_final_branch();
    test_map_produces_openai_tool_call_items();
    test_stream_demux_never_leaks_the_envelope();
    test_stream_demux_is_boundary_independent();
    test_atem_stream_demux_is_boundary_independent();
    test_muse_schema_payload_stream_hides_recipient_header();
    test_stream_demux_holds_back_an_undecided_prefix();
    test_malformed_tool_declarations_are_rejected();
    test_malformed_tool_choice_is_rejected();
    test_parameterless_tool();
    test_parallel_envelope_maps_several_calls();
    test_default_envelope_is_unchanged();
    test_system_turn_teaches_the_envelope();
    puts("tool envelope tests ok");
    return 0;
}
