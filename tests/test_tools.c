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
                        "kind=%d phase=%d idx=%d pos=%d alive=%llu lit=%s in %s\n",
                        i, (unsigned char)doc[i], v.depth,
                        v.stack[v.depth - 1].node->kind,
                        v.stack[v.depth - 1].phase,
                        v.stack[v.depth - 1].idx,
                        v.stack[v.depth - 1].lit_pos,
                        (unsigned long long)v.stack[v.depth - 1].alive,
                        v.stack[v.depth - 1].node->n_lits
                            ? v.stack[v.depth - 1].node->lits[0] : "-",
                        doc);
            return false;
        }
    }
    return v.done || sval_at_raw_tail(&v);
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

// Truncation INSIDE the tool name is the interesting one: the name is the
// discriminator, and the argument block that follows is chosen by it. The
// closer finishes the name from the surviving candidate, so it also has to
// record which one it picked -- otherwise the completed turn names one tool
// and carries another's parameters, and the validator that produced it
// refuses to read it back.
static void test_atem_truncation_inside_a_tool_name_picks_its_own_args(void) {
    jv *tools = parse(
        "[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\","
        "\"parameters\":{\"type\":\"object\",\"properties\":{"
        "\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}}},"
        "{\"type\":\"function\",\"function\":{\"name\":\"add\",\"parameters\":"
        "{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"integer\"},"
        "\"b\":{\"type\":\"integer\"}},\"required\":[\"a\",\"b\"]}}}]");
    char err[192];
    snode *root = schema_compile_atem_tools(tools, err, sizeof(err));
    assert(root != NULL);
    // "a" can only become "add" -- the second declared tool, not the first
    const char *prefix = "<atem:function_calls>\n<atem:invoke name=\"a";
    sval partial;
    sval_init(&partial, root);
    assert(sval_feed(&partial, prefix, (int)strlen(prefix)));
    char tail[512];
    int tail_n = sval_close(&partial, tail, sizeof(tail));
    assert(tail_n > 0);
    char full[1024];
    snprintf(full, sizeof(full), "%s%s", prefix, tail);
    if (!accepts(root, full)) fprintf(stderr, "closed turn: %s\n", full);
    assert(accepts(root, full));
    assert(strstr(full, "name=\"add\"") != NULL);
    assert(strstr(full, "name=\"a\"") != NULL);   // the parameter block's own
    assert(strstr(full, "name=\"city\"") == NULL);
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

static void test_atem_buffered_mapper_honors_nonterminated_length(void) {
    tool_envelope e = {.atem = true};
    const char *source =
        " to=ping<|message|><atem:function_calls>\n"
        "<atem:invoke name=\"ping\">\n"
        "</atem:invoke>\n</atem:function_calls>";
    size_t n = strlen(source);
    char *doc = malloc(n);
    assert(doc != NULL);
    memcpy(doc, source, n); /* deliberately no trailing NUL */
    sbuf content = {0}, calls = {0};
    assert(tool_envelope_map(&e, doc, n, &content, &calls) == 1);
    assert(calls.s && strstr(calls.s, "\"name\":\"ping\""));
    free(doc); free(content.s); free(calls.s);
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

    size_t doc_n = strlen(doc);
    char *bounded = malloc(doc_n);
    assert(bounded != NULL);
    memcpy(bounded, doc, doc_n); /* deliberately no trailing NUL */
    content = (sbuf){0};
    calls = (sbuf){0};
    assert(tool_envelope_map(&e, bounded, doc_n, &content, &calls) == 1);
    assert(calls.s && strstr(calls.s, "\"name\":\"ping\""));
    free(bounded); free(content.s); free(calls.s);

    char *missing_header = malloc(2);
    assert(missing_header != NULL);
    memcpy(missing_header, "{}", 2); /* no marker and no trailing NUL */
    content = (sbuf){0};
    calls = (sbuf){0};
    assert(tool_envelope_map(&e, missing_header, 2, &content, &calls) == -1);
    free(missing_header); free(content.s); free(calls.s);

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

static void test_atem_truncated_integer_respects_declared_bounds(void) {
    jv *tools = parse(
        "[{\"type\":\"function\",\"function\":{\"name\":\"set_level\","
        "\"parameters\":{\"type\":\"object\",\"properties\":{"
        "\"level\":{\"type\":\"integer\",\"minimum\":5,\"maximum\":9}},"
        "\"required\":[\"level\"]}}}]"
    );
    tool_envelope e = {.atem = true, .tools = tools};
    const char *doc = " to=set_level<|message|><atem:function_calls>\n"
        "<atem:invoke name=\"set_level\">\n"
        "<atem:parameter name=\"level\"></atem:parameter>\n"
        "</atem:invoke>\n</atem:function_calls>";
    sbuf content = {0}, calls = {0};
    assert(tool_envelope_map(&e, doc, strlen(doc), &content, &calls) == 1);
    assert(calls.s && strstr(calls.s, "{\\\"level\\\":5}"));
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
    // SPACED, not compact: ornith.jinja:50 is `{{- tool | tojson }}`, and
    // jinja's tojson separates with `, ` and `: `. The compact spelling this
    // line used to assert was runner's own output, not the reference's.
    assert(strstr(prompt.s, "\"name\": \"get_weather\""));
    assert(strstr(prompt.s, "<function=example_function_name>"));
    assert(strstr(prompt.s, "<parameter=example_parameter_1>"));

    jv *history = parse(
        "[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\","
        "\"arguments\":\"{\\\"city\\\":\\\"Oslo\\\",\\\"days\\\":2}\"}}]");
    sbuf replay = {0};
    tool_history_render_for(TMPL_ORNITH, history, false, &replay);
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
    sbuf reasoning, content, args, names; // names: calls, space-separated
    char name[64];             // the LAST call_begin, for single-call tests
    int  begins, ends;
    bool called;                // tool_stream_called() once feeding is done
} demux_log;

static int log_content(void *ud, const char *b, int n) {
    sb_put(&((demux_log *)ud)->content, b, n);
    return 0;
}
static int log_reasoning(void *ud, const char *b, int n) {
    sb_put(&((demux_log *)ud)->reasoning, b, n);
    return 0;
}
static int log_begin(void *ud, const char *name) {
    demux_log *l = ud;
    snprintf(l->name, sizeof(l->name), "%s", name);
    if (l->names.n) sb_lit(&l->names, " ");
    sb_put(&l->names, name, strlen(name));
    l->begins++;
    return 0;
}
static int log_args(void *ud, const char *b, int n) {
    sb_put(&((demux_log *)ud)->args, b, n);
    return 0;
}
static int log_end(void *ud) {
    ((demux_log *)ud)->ends++;
    return 0;
}

// feed `doc` in chunks of `step` bytes (0 == one byte at a time) and return
// what the sink saw
static void demux_step(const tool_envelope *e, const char *doc, size_t step,
                       demux_log *l) {
    memset(l, 0, sizeof(*l));
    tool_stream_sink sink = { l, log_reasoning, log_content,
                              log_begin, log_args, log_end };
    tool_stream s;
    tool_stream_init(&s, e, &sink);
    size_t len = strlen(doc);
    if (!step) step = 1;
    for (size_t i = 0; i < len; i += step) {
        size_t k = len - i < step ? len - i : step;
        assert(tool_stream_feed(&s, doc + i, (int)k) == 0);
    }
    assert(tool_stream_finish(&s) == 0);
    if (tool_stream_called(&s)) assert(l->begins >= 1);
    l->called = tool_stream_called(&s);
    tool_stream_free(&s);
}

static void demux(const tool_envelope *e, const char *doc, demux_log *l) {
    demux_step(e, doc, 1, l);
}

static void log_free(demux_log *l) {
    free(l->reasoning.s);
    free(l->content.s);
    free(l->args.s);
    free(l->names.s);
}

// the same property the SSE boundary matrix asserts one level up: what the
// client sees may not depend on where the token boundaries happened to fall
static void demux_every_split(const tool_envelope *e, const char *doc) {
    demux_log ref;
    demux_step(e, doc, 1, &ref);
    for (size_t step = 2; step <= strlen(doc) + 1; step++) {
        demux_log got;
        demux_step(e, doc, step, &got);
        assert(got.begins == ref.begins);
        assert(got.ends == ref.ends);
        assert(got.called == ref.called);
        assert(!strcmp(got.name, ref.name));
        assert(got.names.n == ref.names.n);
        assert(!got.names.n || !memcmp(got.names.s, ref.names.s, ref.names.n));
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

// ------------------------------------------- parallel streaming demux
//
// The single-call TS_TOOL/TS_ARGS/TS_VALUE/TS_FINAL_STR chain already knows
// how to find a "tool"/"args" (or "content") key and forward the value that
// follows -- it does not care what wraps it, since every key is located by
// scanning forward from wherever the last entry left off. The parallel
// document only adds one new decision: what happens when a value ends. For
// the plain envelope that is the whole document. Inside {"calls":[...]} it
// is one array element, and TS_ENTRY_SEP looks for the entry's own closing
// '}' and then a ',' (another entry) or ']' (done) in whatever of the input
// is left over -- which may already be sitting in the very same feed() call.

static void test_parallel_stream_demux_emits_each_call(void) {
    jv *tools = parse(TOOLS);
    tool_envelope e;
    char err[192];
    assert(tool_envelope_build_ex(tools, NULL, NULL, true, &e,
                                  err, sizeof(err)) == 1);

    demux_log l;
    demux(&e, "{\"calls\":[{\"tool\":\"get_weather\","
              "\"args\":{\"city\":\"Oslo\",\"units\":\"c\"}},"
              "{\"tool\":\"add\",\"args\":{\"a\":1,\"b\":2}}]}", &l);
    assert(l.begins == 2 && l.ends == 2);
    assert(l.called);
    assert(!strcmp(l.names.s, "get_weather add"));
    assert(l.content.n == 0);          // no content alongside calls
    // the concatenated argument deltas are exactly the two argument
    // documents, back to back, with no separator or envelope syntax
    assert(l.args.s && !strcmp(l.args.s,
        "{\"city\":\"Oslo\",\"units\":\"c\"}{\"a\":1,\"b\":2}"));
    log_free(&l);

    // a single-entry array holding the final branch is an ordinary answer:
    // the wrapper is present, but nothing was called
    demux(&e, "{\"calls\":[{\"tool\":\"final\",\"args\":{\"content\":\"hi\"}}]}",
         &l);
    assert(l.begins == 0 && l.ends == 0);
    assert(!l.called);
    assert(l.content.s && !strcmp(l.content.s, "hi"));
    log_free(&l);

    // mixed document (the system prompt discourages this, but the schema
    // does not forbid it): a call followed by a final entry must still
    // report the call, and "called" must survive past the later non-call
    // entry rather than being reset by it
    demux(&e, "{\"calls\":[{\"tool\":\"add\",\"args\":{\"a\":1,\"b\":2}},"
              "{\"tool\":\"final\",\"args\":{\"content\":\"done\"}}]}", &l);
    assert(l.begins == 1 && l.ends == 1);
    assert(l.called);
    assert(!strcmp(l.names.s, "add"));
    assert(l.args.s && !strcmp(l.args.s, "{\"a\":1,\"b\":2}"));
    assert(l.content.s && !strcmp(l.content.s, "done"));
    log_free(&l);

    tool_envelope_free(&e);
    jv_free(tools);
}

// A demux that decides differently depending on how the tokenizer happened
// to split the array would leak the envelope, or misattribute an index, on
// some requests and not others -- exactly the failure mode the single-call
// version of this test guards against, extended across entries.
static void test_parallel_stream_demux_is_boundary_independent(void) {
    jv *tools = parse(TOOLS);
    tool_envelope e;
    char err[192];
    assert(tool_envelope_build_ex(tools, NULL, NULL, true, &e,
                                  err, sizeof(err)) == 1);

    demux_every_split(&e, "{\"calls\":[{\"tool\":\"get_weather\","
                          "\"args\":{\"city\":\"Oslo\",\"units\":\"c\"}},"
                          "{\"tool\":\"add\",\"args\":{\"a\":1,\"b\":-2}}]}");
    // whitespace around the entry separator is exactly as free as within one
    // entry, and must not leak either
    demux_every_split(&e, "{ \"calls\" : [ { \"tool\" : \"add\" , \"args\" : "
                          "{ \"a\" : 1 , \"b\" : 2 } } , { \"tool\" : \"add\" "
                          ", \"args\" : { \"a\" : 3 , \"b\" : 4 } } ] }");
    // three entries: the between-entries loop, not just the two-call case
    demux_every_split(&e,
        "{\"calls\":[{\"tool\":\"add\",\"args\":{\"a\":1,\"b\":2}},"
        "{\"tool\":\"add\",\"args\":{\"a\":3,\"b\":4}},"
        "{\"tool\":\"get_weather\",\"args\":"
        "{\"city\":\"Bergen\",\"units\":\"f\"}}]}");

    tool_envelope_free(&e);
    jv_free(tools);
}

// max_tokens can cut a parallel document mid-call exactly as it can the
// single-call envelope. sval_close completes the array to its declared
// minItems, so the closer's tail may finish a call the model itself never
// finished typing -- and that tail reaches the demux through the very same
// feed() pipeline ordinary tokens do. Every call that was ever announced
// must still be announced, closed, and carry parseable arguments.
static void test_parallel_stream_demux_survives_truncation(void) {
    jv *tools = parse(TOOLS);
    tool_envelope e;
    char err[192];
    assert(tool_envelope_build_ex(tools, NULL, NULL, true, &e,
                                  err, sizeof(err)) == 1);
    snode *root = compile(&e);
    assert(root != NULL);

    const char *prefix =
        "{\"calls\":[{\"tool\":\"get_weather\","
        "\"args\":{\"city\":\"Oslo\",\"units\":\"c\"}},"
        "{\"tool\":\"add\",\"args\":{\"a\":1,\"b\":";
    sval partial;
    sval_init(&partial, root);
    assert(sval_feed(&partial, prefix, (int)strlen(prefix)));
    char tail[512];
    int tail_n = sval_close(&partial, tail, sizeof(tail));
    assert(tail_n > 0);
    char full[1024];
    snprintf(full, sizeof(full), "%s%s", prefix, tail);
    assert(accepts(root, full));

    demux_log l;
    demux(&e, full, &l);
    // both entries were announced and closed, including the one the closer
    // finished on the model's behalf -- a client waiting on call_end for
    // that index must not be left hanging just because the budget ran out
    assert(l.begins == 2 && l.ends == 2);
    assert(l.called);
    assert(!strcmp(l.names.s, "get_weather add"));
    static const char *const first_args = "{\"city\":\"Oslo\",\"units\":\"c\"}";
    assert(l.args.s && !strncmp(l.args.s, first_args, strlen(first_args)));
    // the second call's arguments are whatever the closer completed them to
    // -- still valid JSON, still executable, just not the model's own text
    const char *second = l.args.s + strlen(first_args);
    jv *parsed = json_parse(second, strlen(second));
    assert(parsed && parsed->type == J_OBJ);
    jv_free(parsed);
    log_free(&l);

    // the closer's tail lands in one lump sum after however many bytes were
    // fed before it; the call sequence must not depend on where that was
    demux_every_split(&e, full);

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

static void test_harmony_native_turn_constrains_channels_recipients_and_args(void) {
    jv *tools = parse(TOOLS);
    char err[192];
    snode *root = schema_compile_harmony_turn(
        tools, true, NULL, NULL, true, err, sizeof(err));
    assert(root != NULL);
    assert(accepts(root,
        "<|channel|>commentary to=functions.add<|constrain|>json"
        "<|message|>{\"a\":1,\"b\":2}"));
    assert(accepts(root,
        "<|channel|>analysis<|message|>Need arithmetic."
        "<|end|><|start|>assistant<|channel|>commentary to=functions.add"
        "<|constrain|>json<|message|>{\"a\":1,\"b\":2}"));
    assert(accepts(root,
        "<|channel|>commentary<|message|>I’ll calculate that."
        "<|end|><|start|>assistant<|channel|>commentary to=functions.add"
        "<|constrain|>json<|message|>{\"a\":1,\"b\":2}"));
    assert(accepts(root, "<|channel|>final<|message|>No tool needed."));
    assert(accepts(root,
        "<|channel|>analysis<|message|>No tool needed."
        "<|end|><|start|>assistant<|channel|>final<|message|>The answer is 3."));
    assert(!accepts(root,
        "<|channel|>commentary to=functions.unknown<|constrain|>json<|message|>{}"));
    assert(!accepts(root,
        "<|channel|>commentary to=functions.add<|constrain|>json"
        "<|message|>{\"a\":\"1\",\"b\":2}"));
    // The analysis bound applies on the auto branch too. Lifting it there was
    // tried and refuted live (see the note in harmony_after_reasoning): the
    // model deliberates past the token budget instead of answering.
    char bounded[1024];
    const char *open = "<|channel|>analysis<|message|>";
    const char *close = "<|end|><|start|>assistant<|channel|>commentary "
                        "to=functions.add<|constrain|>json<|message|>"
                        "{\"a\":1,\"b\":2}";
    size_t at = 0;
    memcpy(bounded + at, open, strlen(open)); at += strlen(open);
    memset(bounded + at, 'x', 192); at += 192;
    memcpy(bounded + at, close, strlen(close) + 1);
    assert(accepts(root, bounded));
    bounded[strlen(open) + 192] = 'x';
    memcpy(bounded + strlen(open) + 193, close, strlen(close) + 1);
    assert(!accepts(root, bounded));

    // The reasoning block ends at the FIRST occurrence of its terminator, and
    // that terminator begins with `<|end|>` -- so reasoning that itself ends
    // with `<|end|>` puts the scanner nine bytes into a match that the tenth
    // byte breaks, with a live three-byte match (`<|e`) left behind it.
    // Restarting from zero instead steps over the real occurrence entirely,
    // and the validator then disagrees with every plain search for the same
    // bytes about where the reasoning stopped.
    assert(accepts(root,
        "<|channel|>analysis<|message|>Need arithmetic.<|end|>"
        "<|end|><|start|>assistant<|channel|>commentary to=functions.add"
        "<|constrain|>json<|message|>{\"a\":1,\"b\":2}"));
    schema_free(root);

    root = schema_compile_harmony_turn(
        tools, false, "add", NULL, false, err, sizeof(err));
    assert(root != NULL);
    assert(accepts(root,
        "<|channel|>commentary to=functions.add<|constrain|>json"
        "<|message|>{\"a\":1,\"b\":2}"));
    assert(!accepts(root,
        "<|channel|>commentary to=functions.get_weather<|constrain|>json"
        "<|message|>{\"city\":\"Oslo\",\"units\":\"c\"}"));
    assert(!accepts(root, "<|channel|>final<|message|>No call."));
    assert(!accepts(root,
        "<|channel|>analysis<|message|>Need call."
        "<|end|><|start|>assistant<|channel|>commentary to=functions.add"
        "<|constrain|>json<|message|>{\"a\":1,\"b\":2}"));
    schema_free(root);
    jv_free(tools);
}

// The same truncation on Harmony, where the discriminators NEST: the channel
// name selects a branch, and the tool name inside it selects another. A
// minimal completion emits the inner name as the first candidate, so the
// arguments after it belong to that one -- not to whatever the outer channel
// happened to choose.
static void test_harmony_truncation_pairs_the_name_it_emits(void) {
    jv *tools = parse(TOOLS);
    char err[192];
    snode *root = schema_compile_harmony_turn(
        tools, true, NULL, NULL, true, err, sizeof(err));
    assert(root != NULL);
    const char *cuts[] = {
        "<|channel|>",                               // no channel yet
        "<|channel|>commentar",                      // inside the CHANNEL name:
                                                     // the whole call after it
                                                     // is synthesized
        "<|channel|>analysi",
        "<|channel|>commentary to=functions.",       // no tool name yet
        "<|channel|>commentary to=functions.a",      // only "add" survives
        "<|channel|>commentary to=functions.get_weather<|constrain|>json<|message|>{",
    };
    for (size_t i = 0; i < sizeof(cuts) / sizeof(*cuts); i++) {
        sval v;
        sval_init(&v, root);
        assert(sval_feed(&v, cuts[i], (int)strlen(cuts[i])));
        char tail[512];
        int n = sval_close(&v, tail, sizeof(tail));
        assert(n > 0);
        char full[1024];
        snprintf(full, sizeof(full), "%s%s", cuts[i], tail);
        if (!accepts(root, full)) fprintf(stderr, "closed turn: %s\n", full);
        assert(accepts(root, full));
    }
    // the name the closer completes decides the arguments that follow it
    sval v;
    sval_init(&v, root);
    const char *cut = "<|channel|>commentary to=functions.a";
    assert(sval_feed(&v, cut, (int)strlen(cut)));
    char tail[512];
    assert(sval_close(&v, tail, sizeof(tail)) > 0);
    char full[1024];
    snprintf(full, sizeof(full), "%s%s", cut, tail);
    assert(strstr(full, "functions.add") != NULL);
    assert(strstr(full, "\"a\":") != NULL && strstr(full, "\"b\":") != NULL);
    assert(strstr(full, "\"city\"") == NULL);
    schema_free(root);
    jv_free(tools);
}

static void test_harmony_native_mapping_and_stream_boundaries(void) {
    jv *tools = parse(TOOLS);
    tool_envelope e;
    char err[192];
    assert(tool_envelope_build(tools, NULL, NULL, &e, err, sizeof(err)) == 1);
    e.harmony = true;
    e.tools = tools;

    sbuf reason = {0}, content = {0}, calls = {0};
    const char *doc = "<|channel|>analysis<|message|>Need arithmetic."
                      "<|end|><|start|>assistant<|channel|>commentary "
                      "to=functions.add<|constrain|>json"
                      "<|message|>{\"a\":1,\"b\":2}";
    assert(tool_envelope_map_channels(&e, doc, strlen(doc), &reason,
                                      &content, &calls) == 1);
    assert(!strcmp(reason.s, "Need arithmetic."));
    assert(content.n == 0);
    assert(strstr(calls.s, "\"name\":\"add\""));
    assert(strstr(calls.s, "{\\\"a\\\":1,\\\"b\\\":2}"));
    free(reason.s); free(content.s); free(calls.s);

    reason = (sbuf){0}; content = (sbuf){0}; calls = (sbuf){0};
    doc = "<|channel|>commentary<|message|>I’ll calculate that."
          "<|end|><|start|>assistant<|channel|>commentary "
          "to=functions.add<|constrain|>json"
          "<|message|>{\"a\":1,\"b\":2}";
    assert(tool_envelope_map_channels(&e, doc, strlen(doc), &reason,
                                      &content, &calls) == 1);
    assert(!strcmp(content.s, "I’ll calculate that."));
    free(reason.s); free(content.s); free(calls.s);

    reason = (sbuf){0}; content = (sbuf){0}; calls = (sbuf){0};
    doc = "<|channel|>analysis<|message|>Briefly."
          "<|end|><|start|>assistant<|channel|>final<|message|>"
          "The answer is 3.<|return|>";
    assert(tool_envelope_map_channels(&e, doc, strlen(doc), &reason,
                                      &content, &calls) == 0);
    assert(!strcmp(reason.s, "Briefly."));
    assert(!strcmp(content.s, "The answer is 3."));
    free(reason.s); free(content.s); free(calls.s);

    for (size_t step = 1; step <= strlen(doc); step++) {
        demux_log log;
        demux_step(&e, doc, step, &log);
        assert(!strcmp(log.reasoning.s, "Briefly."));
        assert(!strcmp(log.content.s, "The answer is 3."));
        assert(!log.called);
        log_free(&log);
    }
    doc = "<|channel|>commentary to=functions.add<|constrain|>json"
          "<|message|>{\"a\":1,\"b\":2}";
    for (size_t step = 1; step <= strlen(doc); step++) {
        demux_log log;
        demux_step(&e, doc, step, &log);
        assert(log.called && log.begins == 1 && log.ends == 1);
        assert(!strcmp(log.name, "add"));
        assert(!strcmp(log.args.s, "{\"a\":1,\"b\":2}"));
        log_free(&log);
    }
    tool_envelope_free(&e);
    jv_free(tools);
}


// ---------------------------------------------------------------- gemma4
//
// gemma4's native call is `<|tool_call>call:NAME{key:VALUE,...}<tool_call|>`,
// with strings delimited by the reserved <|"|> token and object keys left
// UNQUOTED. The point of compiling that rather than leaving the family on the
// generic JSON envelope is that `env != NULL` is what buys constrained
// decoding -- so these tests drive the same validator the sampler drives, and
// the truncation case below is the project's headline claim on this family.
static void test_gemma4_native_turn_constrains_names_and_arguments(void) {
    jv *tools = parse(TOOLS);
    char err[192];
    snode *root = schema_compile_gemma4_turn(
        tools, true, NULL, NULL, false, false, err, sizeof(err));
    if (!root) fprintf(stderr, "gemma4 turn did not compile: %s\n", err);
    assert(root != NULL);
    assert(accepts(root,
        "<|tool_call>call:get_weather{city:<|\"|>Oslo<|\"|>,units:<|\"|>c<|\"|>}"
        "<tool_call|>"));
    assert(accepts(root, "<|tool_call>call:add{a:1,b:2}<tool_call|>"));
    // prose is the union's catch-all branch, and it is not the call branch:
    // a turn that says nothing about tools is a legal answer
    assert(accepts(root, "It is -3 C in Oslo.<turn|>"));
    // an undeclared name, a misspelled enum member and a mistyped integer are
    // all unreachable rather than merely unusual
    assert(!accepts(root, "<|tool_call>call:unknown{}<tool_call|>"));
    assert(!accepts(root,
        "<|tool_call>call:get_weather{city:<|\"|>Oslo<|\"|>,units:<|\"|>k<|\"|>}"
        "<tool_call|>"));
    assert(!accepts(root, "<|tool_call>call:add{a:<|\"|>1<|\"|>,b:2}<tool_call|>"));
    // keys come in jinja dictsort order, which is what the model was trained
    // on and what template.c renders into the history
    assert(!accepts(root, "<|tool_call>call:add{b:2,a:1}<tool_call|>"));
    schema_free(root);

    // tool_choice: "required" removes the prose branch entirely -- enforcement
    // IS the envelope, and this is the property that dies if gemma4 is taken
    // off the strict path to reach its native syntax
    root = schema_compile_gemma4_turn(tools, false, NULL, NULL, false, false,
                                      err, sizeof(err));
    assert(root != NULL);
    assert(accepts(root, "<|tool_call>call:add{a:1,b:2}<tool_call|>"));
    assert(!accepts(root, "It is -3 C in Oslo.<turn|>"));
    schema_free(root);

    // tool_choice: {"name": "add"} leaves exactly one branch
    root = schema_compile_gemma4_turn(tools, false, "add", NULL, false, false,
                                      err, sizeof(err));
    assert(root != NULL);
    assert(accepts(root, "<|tool_call>call:add{a:1,b:2}<tool_call|>"));
    assert(!accepts(root,
        "<|tool_call>call:get_weather{city:<|\"|>Oslo<|\"|>,units:<|\"|>c<|\"|>}"
        "<tool_call|>"));
    schema_free(root);
    jv_free(tools);
}

static void test_gemma4_thought_block_precedes_either_branch(void) {
    jv *tools = parse(TOOLS);
    char err[192];
    snode *root = schema_compile_gemma4_turn(
        tools, true, NULL, NULL, true, false, err, sizeof(err));
    assert(root != NULL);
    assert(accepts(root,
        "<|channel>thought\nOslo needs a lookup.<channel|>"
        "<|tool_call>call:get_weather{city:<|\"|>Oslo<|\"|>,units:<|\"|>c<|\"|>}"
        "<tool_call|>"));
    assert(accepts(root,
        "<|channel>thought\nNo tool needed.<channel|>It is cold.<turn|>"));
    assert(accepts(root, "<|tool_call>call:add{a:1,b:2}<tool_call|>"));
    assert(accepts(root, "Plain answer.<turn|>"));
    schema_free(root);

    // the continuation turn whose PROMPT already opened the thought: the
    // document starts inside it, so the grammar must not expect the opener
    root = schema_compile_gemma4_turn(tools, true, NULL, NULL, true, true,
                                      err, sizeof(err));
    assert(root != NULL);
    assert(accepts(root, "Still cold.<channel|>It is -3 C.<turn|>"));
    assert(accepts(root,
        "Need the other city.<channel|>"
        "<|tool_call>call:get_weather{city:<|\"|>Bergen<|\"|>,units:<|\"|>c<|\"|>}"
        "<tool_call|>"));
    schema_free(root);
    jv_free(tools);
}

static void test_gemma4_truncated_call_still_parses(void) {
    // THE HEADLINE, on this family. A call cut off by the token budget closes
    // to the smallest schema-legal ending, and what the caller receives is
    // still JSON their client can execute.
    jv *tools = parse(TOOLS);
    char err[192];
    snode *root = schema_compile_gemma4_turn(
        tools, true, NULL, NULL, false, false, err, sizeof(err));
    assert(root != NULL);
    const char *partial = "<|tool_call>call:get_weather{city:<|\"|>Os";
    sval v;
    sval_init(&v, root);
    assert(sval_feed(&v, partial, (int)strlen(partial)));
    char tail[256];
    int n = sval_close(&v, tail, sizeof(tail));
    assert(n > 0);
    sbuf doc = {0};
    sb_put(&doc, partial, strlen(partial));
    sb_put(&doc, tail, (size_t)n);

    tool_envelope e = {0};
    assert(tool_envelope_build(tools, NULL, NULL, &e, err, sizeof(err)) == 1);
    e.gemma4 = true;
    e.tools = tools;
    sbuf content = {0}, calls = {0};
    assert(tool_envelope_map(&e, doc.s, doc.n, &content, &calls) == 1);
    assert(strstr(calls.s, "\"name\":\"get_weather\""));
    // the arguments field is a JSON string a client will json.loads(): the
    // native <|"|> spelling must not survive into it
    assert(!strstr(calls.s, "<|"));
    assert(strstr(calls.s, "\\\"city\\\":\\\"Os"));
    free(doc.s); free(content.s); free(calls.s);
    tool_envelope_free(&e);
    schema_free(root);
    jv_free(tools);
}

static void test_gemma4_native_mapping_and_stream_boundaries(void) {
    jv *tools = parse(TOOLS);
    char err[192];
    tool_envelope e;
    assert(tool_envelope_build(tools, NULL, NULL, &e, err, sizeof(err)) == 1);
    e.gemma4 = true;
    e.tools = tools;

    sbuf reason = {0}, content = {0}, calls = {0};
    const char *doc =
        "<|channel>thought\nNeeds a lookup.<channel|>"
        "<|tool_call>call:get_weather{city:<|\"|>Oslo<|\"|>,units:<|\"|>c<|\"|>}"
        "<tool_call|>";
    assert(tool_envelope_map_channels(&e, doc, strlen(doc), &reason,
                                      &content, &calls) == 1);
    assert(!strcmp(reason.s, "Needs a lookup."));
    assert(content.n == 0);
    assert(strstr(calls.s, "\"name\":\"get_weather\""));
    assert(strstr(calls.s, "{\\\"city\\\":\\\"Oslo\\\",\\\"units\\\":\\\"c\\\"}"));
    free(reason.s); free(content.s); free(calls.s);

    // two calls in one turn: gemma4 concatenates the blocks with no separator
    reason = (sbuf){0}; content = (sbuf){0}; calls = (sbuf){0};
    doc = "<|tool_call>call:add{a:1,b:2}<tool_call|>"
          "<|tool_call>call:add{a:3,b:4}<tool_call|>";
    assert(tool_envelope_map_channels(&e, doc, strlen(doc), &reason,
                                      &content, &calls) == 2);
    assert(strstr(calls.s, "\"id\":\"call_0\"") &&
           strstr(calls.s, "\"id\":\"call_1\""));
    free(reason.s); free(content.s); free(calls.s);

    // prose: the turn close is framing and must not reach the client
    reason = (sbuf){0}; content = (sbuf){0}; calls = (sbuf){0};
    doc = "It is -3 C in Oslo.<turn|>";
    assert(tool_envelope_map_channels(&e, doc, strlen(doc), &reason,
                                      &content, &calls) == 0);
    assert(!strcmp(content.s, "It is -3 C in Oslo."));
    free(reason.s); free(content.s); free(calls.s);

    // ---- the streamed path reaches the same call from the same bytes,
    // whatever the chunk boundaries are
    doc = "<|channel>thought\nNeeds a lookup.<channel|>"
          "<|tool_call>call:get_weather{city:<|\"|>Oslo<|\"|>,units:<|\"|>c<|\"|>}"
          "<tool_call|>";
    for (size_t step = 1; step <= strlen(doc); step++) {
        demux_log log;
        demux_step(&e, doc, step, &log);
        assert(log.called && log.begins == 1 && log.ends == 1);
        assert(!strcmp(log.name, "get_weather"));
        assert(!strcmp(log.args.s, "{\"city\":\"Oslo\",\"units\":\"c\"}"));
        assert(!strcmp(log.reasoning.s, "Needs a lookup."));
        assert(log.content.n == 0);
        log_free(&log);
    }
    doc = "It is -3 C in Oslo.<turn|>";
    for (size_t step = 1; step <= strlen(doc); step++) {
        demux_log log;
        demux_step(&e, doc, step, &log);
        assert(!log.called && log.begins == 0);
        // the held-back tail is the whole point: a chunk boundary inside
        // "<turn|>" must not leak a partial marker as assistant text
        assert(!strcmp(log.content.s, "It is -3 C in Oslo."));
        log_free(&log);
    }
    tool_envelope_free(&e);
    jv_free(tools);
}

static void test_gemma4_structured_arguments_round_trip(void) {
    jv *tools = parse(
        "[{\"type\":\"function\",\"function\":{\"name\":\"store\","
        "\"parameters\":{\"type\":\"object\",\"properties\":{"
        "\"payload\":{\"type\":\"object\",\"properties\":{"
            "\"x\":{\"type\":\"integer\"},\"ok\":{\"type\":\"boolean\"}}},"
        "\"tags\":{\"type\":\"array\",\"items\":{\"type\":\"string\"}}},"
        "\"required\":[\"payload\",\"tags\"]}}}]");
    char err[192];
    snode *root = schema_compile_gemma4_turn(tools, false, NULL, NULL, false,
                                             false, err, sizeof(err));
    if (!root) fprintf(stderr, "gemma4 structured turn: %s\n", err);
    assert(root != NULL);
    // nested objects keep UNQUOTED keys and dictsort order (ok before x);
    // arrays are bounded but variable-length
    const char *doc = "<|tool_call>call:store{payload:{ok:true,x:2},"
                      "tags:[<|\"|>a<|\"|>,<|\"|>b<|\"|>]}<tool_call|>";
    assert(accepts(root, doc));
    assert(accepts(root,
        "<|tool_call>call:store{payload:{ok:false,x:0},tags:[]}<tool_call|>"));
    assert(!accepts(root,
        "<|tool_call>call:store{payload:{\"ok\":true,\"x\":2},tags:[]}"
        "<tool_call|>"));
    schema_free(root);

    tool_envelope e;
    assert(tool_envelope_build(tools, NULL, NULL, &e, err, sizeof(err)) == 1);
    e.gemma4 = true;
    e.tools = tools;
    sbuf content = {0}, calls = {0};
    assert(tool_envelope_map(&e, doc, strlen(doc), &content, &calls) == 1);
    assert(strstr(calls.s,
        "{\\\"payload\\\":{\\\"ok\\\":true,\\\"x\\\":2},"
        "\\\"tags\\\":[\\\"a\\\",\\\"b\\\"]}"));
    free(content.s); free(calls.s);
    tool_envelope_free(&e);
    jv_free(tools);
}

static void test_gemma4_untyped_parameter_is_refused_not_guessed(void) {
    // The generic envelope admits an untyped parameter as free JSON. gemma4's
    // native syntax has no free-form spelling, so the choice is between an
    // error and an unconstrained call that the mapper may not be able to read
    // back. It errors, and the message names the parameter.
    jv *tools = parse(
        "[{\"type\":\"function\",\"function\":{\"name\":\"note\","
        "\"parameters\":{\"type\":\"object\",\"properties\":{"
        "\"body\":{\"description\":\"anything\"}}}}}]");
    char err[192];
    err[0] = 0;
    snode *root = schema_compile_gemma4_turn(tools, true, NULL, NULL, false,
                                             false, err, sizeof(err));
    assert(root == NULL);
    assert(strstr(err, "body") && strstr(err, "type"));
    jv_free(tools);
}

// A gemma4 array is compiled by REPLICATING the element grammar once per
// admissible position, so nesting multiplies: with the default 24-item bound
// three levels is 14k element expansions, four is 346k, and six -- which
// G4_MAX_DEPTH admits -- is past 10^8. tools[] is request data, so an
// unbounded expansion there is the client choosing how much of the server's
// memory to take. It has to be refused, and the message has to say what the
// caller can do about it.
static jv *g4_nested_array_tool(int levels) {
    char inner[512];
    snprintf(inner, sizeof(inner), "{\"type\":\"integer\"}");
    for (int i = 0; i < levels; i++) {
        char wrapped[512];
        snprintf(wrapped, sizeof(wrapped),
                 "{\"type\":\"array\",\"items\":%s}", inner);
        snprintf(inner, sizeof(inner), "%s", wrapped);
    }
    char src[1024];
    snprintf(src, sizeof(src),
             "[{\"type\":\"function\",\"function\":{\"name\":\"f\","
             "\"parameters\":{\"type\":\"object\",\"properties\":"
             "{\"x\":%s}}}}]", inner);
    return parse(src);
}

static void test_gemma4_nested_arrays_are_bounded_not_expanded(void) {
    char err[192];
    for (int levels = 4; levels <= 6; levels++) {
        jv *tools = g4_nested_array_tool(levels);
        err[0] = 0;
        snode *root = schema_compile_gemma4_turn(tools, false, NULL, NULL,
                                                 false, false,
                                                 err, sizeof(err));
        assert(root == NULL);
        // six levels trips the depth ceiling first -- the innermost value sits
        // one level past G4_MAX_DEPTH -- so only four and five reach the
        // expansion ceiling. Both are refusals; the reason differs.
        assert(strstr(err, levels == 6 ? "nests deeper" : "maxItems"));
        jv_free(tools);
    }
    // what the caller does about it: bounding the arrays makes the same shape
    // compile, and it still constrains
    jv *bounded = parse(
        "[{\"type\":\"function\",\"function\":{\"name\":\"f\","
        "\"parameters\":{\"type\":\"object\",\"properties\":{"
        "\"x\":{\"type\":\"array\",\"maxItems\":2,\"items\":"
          "{\"type\":\"array\",\"maxItems\":2,\"items\":"
            "{\"type\":\"array\",\"maxItems\":2,\"items\":"
              "{\"type\":\"array\",\"maxItems\":2,\"items\":"
                "{\"type\":\"integer\"}}}}}}}}}]");
    err[0] = 0;
    snode *root = schema_compile_gemma4_turn(bounded, false, NULL, NULL, false,
                                             false, err, sizeof(err));
    if (!root) fprintf(stderr, "bounded gemma4 nesting: %s\n", err);
    assert(root != NULL);
    assert(accepts(root, "<|tool_call>call:f{x:[[[[1,2],[3]],[[4]]]]}"
                         "<tool_call|>"));
    schema_free(root);
    jv_free(bounded);

    // and the ordinary two-level case every real tool uses stays legal
    jv *ordinary = g4_nested_array_tool(2);
    err[0] = 0;
    root = schema_compile_gemma4_turn(ordinary, false, NULL, NULL, false, false,
                                      err, sizeof(err));
    if (!root) fprintf(stderr, "two-level gemma4 nesting: %s\n", err);
    assert(root != NULL);
    assert(accepts(root, "<|tool_call>call:f{x:[[1,2],[3]]}<tool_call|>"));
    schema_free(root);
    jv_free(ordinary);
}

// The same truncation property as tests/test_json_schema.c states for plain
// schemas, over the NATIVE grammars: walk only the bytes the validator still
// admits, stop at a random one, and require the force-closed turn to be one
// the same validator reads back to completion. Hand-picked truncation points
// test what someone thought of; this reaches the ones nobody did -- it is what
// turned up the wrong-tool's-arguments pair above. Deterministic seed.
static unsigned long turn_rnd(void) {
    static unsigned long s = 0xDEADBEEF12345UL;
    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
    return s;
}

static void test_native_truncation_closes_to_a_legal_turn(void) {
    jv *tools = parse(TOOLS);
    char err[192];
    struct { const char *name; snode *root; } G[] = {
        { "atem_tools",  schema_compile_atem_tools(tools, err, sizeof(err)) },
        { "atem_turn",   schema_compile_atem_turn(tools, true, NULL, NULL,
                             ATEM_TURN_DIRECT, err, sizeof(err)) },
        { "harmony",     schema_compile_harmony_turn(tools, true, NULL, NULL,
                             true, err, sizeof(err)) },
        { "harmony_req", schema_compile_harmony_turn(tools, false, NULL, NULL,
                             false, err, sizeof(err)) },
        { "gemma4",      schema_compile_gemma4_turn(tools, true, NULL, NULL,
                             true, false, err, sizeof(err)) },
        { "gemma4_par",  schema_compile_gemma4_parallel(tools, NULL, err,
                             sizeof(err)) },
    };
    int closed = 0;
    for (size_t g = 0; g < sizeof(G) / sizeof(*G); g++) {
        if (!G[g].root) fprintf(stderr, "%s did not compile: %s\n", G[g].name, err);
        assert(G[g].root != NULL);
        for (int iter = 0; iter < 120; iter++) {
            sval v;
            sval_init(&v, G[g].root);
            char doc[4096];
            int n = 0;
            int steps = (int)(turn_rnd() % 90);
            for (int k = 0; k < steps && n < 1500; k++) {
                unsigned char legal[128];
                int nl = 0;
                for (int c = 1; c < 0x80; c++) {
                    char b = (char)c;
                    sval t;
                    if (sval_trial(&v, &t, &b, 1)) legal[nl++] = (unsigned char)c;
                }
                if (!nl) break;
                char b = (char)legal[turn_rnd() % (unsigned)nl];
                assert(sval_feed(&v, &b, 1));
                doc[n++] = b;
                if (v.done) break;
            }
            if (v.done) continue;
            char tail[2048];
            int tn = sval_close(&v, tail, sizeof(tail));
            if (tn <= 0) continue;
            assert(n + tn < (int)sizeof(doc));
            memcpy(doc + n, tail, (size_t)tn);
            n += tn;
            doc[n] = 0;
            closed++;
            if (!accepts(G[g].root, doc))
                fprintf(stderr, "%s closed turn not re-accepted:\n%s\n",
                        G[g].name, doc);
            assert(accepts(G[g].root, doc));
        }
        schema_free(G[g].root);
    }
    assert(closed > 100);
    jv_free(tools);
}

static void test_buffered_mapper_rejects_invalid_arguments(void) {
    tool_envelope e = {0};
    sbuf content = {0}, calls = {0};
    const char *doc = "{\"tool\":\"ping\",\"args\":{}}";

    assert(tool_envelope_map(NULL, doc, strlen(doc), &content, &calls) == -1);
    assert(tool_envelope_map(&e, NULL, 0, &content, &calls) == -1);
    assert(tool_envelope_map(&e, doc, strlen(doc), NULL, &calls) == -1);
    assert(tool_envelope_map(&e, doc, strlen(doc), &content, NULL) == -1);
}

int main(void) {
    test_buffered_mapper_rejects_invalid_arguments();
    test_atem_structured_tool_automaton();
    test_atem_scalar_is_raw_until_parameter_close();
    test_atem_truncation_closes_started_call();
    test_atem_truncation_inside_a_tool_name_picks_its_own_args();
    test_atem_buffered_maps_reasoning_and_multiple_calls();
    test_atem_buffered_mapper_honors_nonterminated_length();
    test_atem_header_discriminates_matching_invoke();
    test_atem_stop_token_is_valid_at_the_raw_answer_tail();
    test_atem_declared_optional_parameters_are_constrained();
    test_atem_parallel_turn_constrains_two_recipient_calls();
    test_muse_generic_envelope_is_constrained_behind_user_recipient();
    test_atem_auto_user_branch_honors_response_schema();
    test_atem_truncated_string_enum_recovers_closest_member();
    test_atem_truncated_integer_respects_declared_bounds();
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
    test_parallel_stream_demux_emits_each_call();
    test_parallel_stream_demux_is_boundary_independent();
    test_parallel_stream_demux_survives_truncation();
    test_default_envelope_is_unchanged();
    test_system_turn_teaches_the_envelope();
    test_harmony_native_turn_constrains_channels_recipients_and_args();
    test_harmony_truncation_pairs_the_name_it_emits();
    test_harmony_native_mapping_and_stream_boundaries();
    test_gemma4_native_turn_constrains_names_and_arguments();
    test_gemma4_thought_block_precedes_either_branch();
    test_gemma4_truncated_call_still_parses();
    test_gemma4_native_mapping_and_stream_boundaries();
    test_gemma4_structured_arguments_round_trip();
    test_gemma4_untyped_parameter_is_refused_not_guessed();
    test_gemma4_nested_arrays_are_bounded_not_expanded();
    test_native_truncation_closes_to_a_legal_turn();
    puts("tool envelope tests ok");
    return 0;
}
