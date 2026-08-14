// A replayed tool result must NAME the function it came from, or the request
// must be refused.
//
// Harmony (gpt-oss) writes a tool turn as an authored one:
//
//     <|start|>functions.get_weather<|channel|>commentary<|message|>...<|end|>
//
// The name comes from the tool CALL the result answers, looked up by id in the
// request history. Each of the three surfaces resolves it its own way, and
// each used to have a way of failing that produced a turn the model was never
// trained on:
//
//   * Responses and Messages resolved to NULL when the caller did not replay
//     the item that made the call -- and the renderer then falls through to
//     `<|start|>tool<|message|>`. That role string parses, but the reference
//     renderer never emits it and not one of the 23 reference goldens contains
//     it, so the model has never seen it.
//   * chat/completions resolved to the tool_call_id STRING, giving
//     `<|start|>functions.call_1`: a function name that was never declared,
//     invented from an identifier.
//
// The Responses one is the one ordinary traffic hits. This runtime is
// stateless, so a client that keeps its own history and posts only the newest
// items never sends the `function_call` back, and every turn of its tool loop
// went out off-protocol with a 200 on it.
//
// So the rule under test: when the name cannot be resolved, either it is
// DEDUCED -- exactly one tool is declared, so there is only one function the
// result can be from -- or the request is refused with a 400 that says so.
// Never a rendered turn that names nothing, and never one that names an id.
//
// handle_chat is static in src/server.c, so that TU is #included here and left
// out of the link, exactly as tests/test_prompt_budget.c does it. The prompt is
// captured from a stubbed run_completion; the refusals are read back off a real
// socketpair, so the status code and the message the caller would actually see
// are what get asserted.
#include "runner.h"
#include "json.h"
#include "http.h"
#include "server_int.h"
#include "completion.h"
#include "template.h"
#include "api.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

static int ta_fail = 0;

static void ck(int cond, const char *what) {
    if (!cond) { fprintf(stderr, "FAIL: %s\n", what); ta_fail = 1; }
    else        fprintf(stderr, "ok: %s\n", what);
}

// ------------------------------------------------------ captured generation

static char *ta_prompt;

void run_completion(slot_t *s, sock_t fd, const char *prompt, int api,
                    jv *req, const tool_envelope *env) {
    (void)s; (void)fd; (void)api; (void)req; (void)env;
    free(ta_prompt);
    ta_prompt = prompt ? strdup(prompt) : NULL;
}

// completion.c's request readers, stubbed: the fields they read are absent
// from every request below, so the defaults are the whole behaviour.
bool absent(const jv *v) { return !v || v->type == J_NULL; }

bool request_bool(jv *req, const char *key, bool dflt, bool *out) {
    jv *v = jv_get(req, key);
    if (absent(v)) { *out = dflt; return true; }
    if (v->type != J_BOOL) return false;
    *out = v->b;
    return true;
}

bool request_keep_alive(jv *req, bool *present, int *seconds) {
    (void)req; (void)seconds;
    *present = false;
    return true;
}

jv *request_schema(jv *req) { (void)req; return NULL; }

void server_record_work(int n_prompt, int n_gen, double gen_seconds) {
    (void)n_prompt; (void)n_gen; (void)gen_seconds;
}

void server_work_totals(unsigned long long *prompt_tokens,
                        unsigned long long *gen_tokens, double *gen_seconds) {
    if (prompt_tokens) *prompt_tokens = 0;
    if (gen_tokens)    *gen_tokens = 0;
    if (gen_seconds)   *gen_seconds = 0;
}

#include "../src/server.c"

// ------------------------------------------------------------------ fixture

static int  ta_status;          // HTTP status the route answered with, or 0
static char ta_error[1024];     // the error body, verbatim

// Run one request body through a route on a Harmony slot. Whatever the route
// writes to the socket is read back; whatever it hands run_completion is kept.
static void ta_run(void (*route)(slot_t *, sock_t, jv *), const char *body) {
    free(ta_prompt);
    ta_prompt = NULL;
    ta_status = 0;
    ta_error[0] = 0;

    int sv[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0) {
        ck(0, "socketpair");
        return;
    }
    jv *req = json_parse(body, strlen(body));
    assert(req);
    slot_t s = {0};
    s.tmpl = TMPL_HARMONY;
    route(&s, (sock_t)sv[0], req);
    jv_free(req);

    shutdown(sv[0], SHUT_WR);
    static char buf[8192];
    size_t got = 0;
    for (;;) {
        ssize_t r = recv(sv[1], buf + got, sizeof(buf) - 1 - got, 0);
        if (r <= 0) break;
        got += (size_t)r;
        if (got >= sizeof(buf) - 1) break;
    }
    buf[got] = 0;
    close(sv[0]);
    close(sv[1]);
    if (!strncmp(buf, "HTTP/1.1 ", 9)) ta_status = atoi(buf + 9);
    // the message is a JSON string and these messages quote the id they could
    // not resolve, so the closing quote is the first UNESCAPED one
    const char *m = strstr(buf, "\"message\":\"");
    if (m) {
        m += 11;
        size_t n = 0;
        while (*m && *m != '"' && n < sizeof(ta_error) - 1) {
            if (*m == '\\' && m[1]) m++;
            ta_error[n++] = *m++;
        }
        ta_error[n] = 0;
    }
}

// The two shapes that must never reach a gpt-oss prompt, whatever else the
// request did. Checked on EVERY run below, refused or not.
//
// Neither check reads the rest of the turn: what a correctly named tool turn
// looks like after the name is template.c's business, and this file is about
// the name.
static void ta_never_off_protocol(const char *what) {
    char msg[192];
    if (!ta_prompt) return;
    // no Harmony role is spelled "tool" -- the roles are system, developer,
    // user, assistant, and the function's own name
    snprintf(msg, sizeof msg, "%s: no nameless <|start|>tool turn", what);
    ck(strstr(ta_prompt, "<|start|>tool") == NULL, msg);
    snprintf(msg, sizeof msg, "%s: no function name invented from a call id",
             what);
    ck(strstr(ta_prompt, "functions.call_") == NULL, msg);
}

static void ta_expect_refused(const char *what) {
    char msg[192];
    snprintf(msg, sizeof msg, "%s: answered 400", what);
    ck(ta_status == 400, msg);
    snprintf(msg, sizeof msg, "%s: no prompt was built", what);
    ck(ta_prompt == NULL, msg);
    snprintf(msg, sizeof msg, "%s: the message says what could not be done",
             what);
    ck(strstr(ta_error, "attribut") != NULL, msg);
    // these messages quote the id they could not resolve, and a real id is
    // long: a buffer that fits the prose but not the id cuts the sentence
    // that tells the caller what to change
    size_t n = strlen(ta_error);
    snprintf(msg, sizeof msg, "%s: the message is not truncated", what);
    ck(n > 0 && ta_error[n - 1] == '.', msg);
    ta_never_off_protocol(what);
    if (getenv("RUNNER_ATTR_TRACE"))
        fprintf(stderr, "--- status %d error %s\n", ta_status, ta_error);
}

static void ta_expect_named(const char *what, const char *fn) {
    char msg[192], want[128];
    snprintf(msg, sizeof msg, "%s: a prompt was produced", what);
    ck(ta_prompt != NULL, msg);
    // the tool turn is the only one authored by the function itself; the
    // assistant's own call turn is `<|start|>assistant to=functions.NAME`
    snprintf(want, sizeof want, "<|start|>functions.%s", fn);
    snprintf(msg, sizeof msg, "%s: the tool turn is authored by functions.%s",
             what, fn);
    ck(ta_prompt && strstr(ta_prompt, want) != NULL, msg);
    ta_never_off_protocol(what);
    if (getenv("RUNNER_ATTR_TRACE") && ta_prompt)
        fprintf(stderr, "--- prompt: %s\n", ta_prompt);
}

// ------------------------------------------------------------ request bodies

#define TA_ONE_TOOL_RESP \
    "\"tools\":[{\"type\":\"function\",\"name\":\"get_weather\"," \
    "\"parameters\":{\"type\":\"object\",\"properties\":" \
    "{\"city\":{\"type\":\"string\"}}}}]"
#define TA_TWO_TOOLS_RESP \
    "\"tools\":[{\"type\":\"function\",\"name\":\"get_weather\"," \
    "\"parameters\":{\"type\":\"object\",\"properties\":" \
    "{\"city\":{\"type\":\"string\"}}}}," \
    "{\"type\":\"function\",\"name\":\"get_time\"," \
    "\"parameters\":{\"type\":\"object\",\"properties\":{}}}]"

#define TA_ONE_TOOL_CHAT \
    "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\"," \
    "\"parameters\":{\"type\":\"object\",\"properties\":" \
    "{\"city\":{\"type\":\"string\"}}}}}]"
#define TA_TWO_TOOLS_CHAT \
    "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\"," \
    "\"parameters\":{\"type\":\"object\",\"properties\":" \
    "{\"city\":{\"type\":\"string\"}}}}}," \
    "{\"type\":\"function\",\"function\":{\"name\":\"get_time\"," \
    "\"parameters\":{\"type\":\"object\",\"properties\":{}}}}]"

#define TA_ONE_TOOL_ANTH \
    "\"tools\":[{\"name\":\"get_weather\",\"input_schema\":" \
    "{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}}}}]"
#define TA_TWO_TOOLS_ANTH \
    "\"tools\":[{\"name\":\"get_weather\",\"input_schema\":" \
    "{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}}}}," \
    "{\"name\":\"get_time\",\"input_schema\":{\"type\":\"object\"}}]"

// ------------------------------------------------------------------- checks

// ---- /v1/responses: the client posts the OUTPUT of a call whose
// `function_call` item it never sent back. This is the reachable one.
//
// The refused cases carry a full-length id, the shape a real client sends, so
// the message that quotes it is checked at the size it really has.
#define TA_LONG_CALL_ID "call_fJ7mQz2xR9pL4vT8sN1kD6hG0bC3aY5wE7uI"

static void test_responses_orphan_output_two_tools(void) {
    ta_run(handle_responses,
           "{\"model\":\"m\",\"input\":["
           "{\"type\":\"message\",\"role\":\"user\",\"content\":\"weather?\"},"
           "{\"type\":\"function_call_output\",\"call_id\":\"" TA_LONG_CALL_ID
           "\",\"output\":\"sunny, 21C\"}],"
           TA_TWO_TOOLS_RESP "}");
    ta_expect_refused("responses, orphan function_call_output, 2 tools");
}

static void test_responses_orphan_output_one_tool(void) {
    ta_run(handle_responses,
           "{\"model\":\"m\",\"input\":["
           "{\"type\":\"message\",\"role\":\"user\",\"content\":\"weather?\"},"
           "{\"type\":\"function_call_output\",\"call_id\":\"call_1\","
           "\"output\":\"sunny, 21C\"}],"
           TA_ONE_TOOL_RESP "}");
    // one declared tool is not a guess: there is no other function the result
    // could be from, and the namespace the model is reading has only that one
    ta_expect_named("responses, orphan function_call_output, 1 tool",
                    "get_weather");
}

static void test_responses_resolvable_is_unchanged(void) {
    ta_run(handle_responses,
           "{\"model\":\"m\",\"input\":["
           "{\"type\":\"message\",\"role\":\"user\",\"content\":\"weather?\"},"
           "{\"type\":\"function_call\",\"call_id\":\"call_1\","
           "\"name\":\"get_time\",\"arguments\":\"{}\"},"
           "{\"type\":\"function_call_output\",\"call_id\":\"call_1\","
           "\"output\":\"12:00\"}],"
           TA_TWO_TOOLS_RESP "}");
    ta_expect_named("responses, resolvable call_id", "get_time");
}

// ---- /v1/messages: the same hole, reached by not replaying the tool_use

static void test_messages_orphan_result_two_tools(void) {
    ta_run(handle_messages,
           "{\"model\":\"m\",\"max_tokens\":16,\"messages\":["
           "{\"role\":\"user\",\"content\":[{\"type\":\"tool_result\","
           "\"tool_use_id\":\"toolu_01A1B2C3D4E5F6G7H8J9K0LMNPQRSTUV\","
           "\"content\":\"sunny, 21C\"}]}],"
           TA_TWO_TOOLS_ANTH "}");
    ta_expect_refused("messages, orphan tool_result, 2 tools");
}

static void test_messages_orphan_result_one_tool(void) {
    ta_run(handle_messages,
           "{\"model\":\"m\",\"max_tokens\":16,\"messages\":["
           "{\"role\":\"user\",\"content\":[{\"type\":\"tool_result\","
           "\"tool_use_id\":\"toolu_1\",\"content\":\"sunny, 21C\"}]}],"
           TA_ONE_TOOL_ANTH "}");
    ta_expect_named("messages, orphan tool_result, 1 tool", "get_weather");
}

static void test_messages_resolvable_is_unchanged(void) {
    ta_run(handle_messages,
           "{\"model\":\"m\",\"max_tokens\":16,\"messages\":["
           "{\"role\":\"user\",\"content\":\"time?\"},"
           "{\"role\":\"assistant\",\"content\":[{\"type\":\"tool_use\","
           "\"id\":\"toolu_1\",\"name\":\"get_time\",\"input\":{}}]},"
           "{\"role\":\"user\",\"content\":[{\"type\":\"tool_result\","
           "\"tool_use_id\":\"toolu_1\",\"content\":\"12:00\"}]}],"
           TA_TWO_TOOLS_ANTH "}");
    ta_expect_named("messages, resolvable tool_use_id", "get_time");
}

// ---- /v1/chat/completions: the id-as-name variant. `name` is absent and the
// tool_call_id matches nothing, so the id itself became the function name.

static void test_chat_unmatched_id_two_tools(void) {
    ta_run(handle_chat,
           "{\"model\":\"m\",\"messages\":["
           "{\"role\":\"user\",\"content\":\"weather?\"},"
           "{\"role\":\"tool\",\"tool_call_id\":\"" TA_LONG_CALL_ID "\","
           "\"content\":\"sunny, 21C\"}],"
           TA_TWO_TOOLS_CHAT "}");
    ta_expect_refused("chat, unmatched tool_call_id, 2 tools");
}

static void test_chat_unmatched_id_one_tool(void) {
    ta_run(handle_chat,
           "{\"model\":\"m\",\"messages\":["
           "{\"role\":\"user\",\"content\":\"weather?\"},"
           "{\"role\":\"tool\",\"tool_call_id\":\"call_1\","
           "\"content\":\"sunny, 21C\"}],"
           TA_ONE_TOOL_CHAT "}");
    ta_expect_named("chat, unmatched tool_call_id, 1 tool", "get_weather");
}

static void test_chat_no_id_at_all(void) {
    ta_run(handle_chat,
           "{\"model\":\"m\",\"messages\":["
           "{\"role\":\"user\",\"content\":\"weather?\"},"
           "{\"role\":\"tool\",\"content\":\"sunny, 21C\"}],"
           TA_TWO_TOOLS_CHAT "}");
    ta_expect_refused("chat, tool result with neither name nor id, 2 tools");
}

static void test_chat_resolvable_is_unchanged(void) {
    ta_run(handle_chat,
           "{\"model\":\"m\",\"messages\":["
           "{\"role\":\"user\",\"content\":\"time?\"},"
           "{\"role\":\"assistant\",\"tool_calls\":[{\"id\":\"call_1\","
           "\"type\":\"function\",\"function\":{\"name\":\"get_time\","
           "\"arguments\":\"{}\"}}]},"
           "{\"role\":\"tool\",\"tool_call_id\":\"call_1\","
           "\"content\":\"12:00\"}],"
           TA_TWO_TOOLS_CHAT "}");
    ta_expect_named("chat, resolvable tool_call_id", "get_time");
}

static void test_chat_explicit_name_is_unchanged(void) {
    ta_run(handle_chat,
           "{\"model\":\"m\",\"messages\":["
           "{\"role\":\"user\",\"content\":\"time?\"},"
           "{\"role\":\"tool\",\"name\":\"get_time\","
           "\"content\":\"12:00\"}],"
           TA_TWO_TOOLS_CHAT "}");
    ta_expect_named("chat, tool result carrying its own name", "get_time");
}

int main(void) {
    test_responses_orphan_output_two_tools();
    test_responses_orphan_output_one_tool();
    test_responses_resolvable_is_unchanged();
    test_messages_orphan_result_two_tools();
    test_messages_orphan_result_one_tool();
    test_messages_resolvable_is_unchanged();
    test_chat_unmatched_id_two_tools();
    test_chat_unmatched_id_one_tool();
    test_chat_no_id_at_all();
    test_chat_resolvable_is_unchanged();
    test_chat_explicit_name_is_unchanged();
    free(ta_prompt);
    fprintf(stderr, ta_fail ? "test-tool-attribution: FAILED\n"
                            : "test-tool-attribution: all checks passed\n");
    return ta_fail;
}
