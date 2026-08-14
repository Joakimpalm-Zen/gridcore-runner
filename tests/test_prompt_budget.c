// The per-request prompt buffer must hold the WHOLE prompt.
//
// Every surface builds its prompt the same way: estimate a size from the
// turns it is about to render, malloc that, and render into it. Nothing ever
// checked that the estimate was big enough, and template.c's emit() truncates
// silently once the cap is reached rather than overflowing. What gets cut is
// the TAIL -- the last user turn and the `<|start|>assistant` generation
// header -- so the model answers a question it never saw, with no error
// anywhere. Harmony made that reachable in ordinary traffic: the `# Tools`
// TypeScript namespace and a replayed `reasoning_content` analysis turn are
// both rendered into the prompt and neither was in any estimate.
//
// tests/test_template.c renders into a fixed char[4096] and so can never see
// this: the bug is not in the renderer, it is in the per-request budget the
// routes hand it. This file drives the real route entry points with the real
// arithmetic and asserts the rendered prompt is COMPLETE.
//
// handle_chat is static in server.c, so that translation unit is #included
// here and left out of the link. run_completion and the request-field readers
// are stubbed for the same reason test_responses_sm stubs nothing and links
// everything: here the generation path is exactly what we do not want -- the
// prompt is the artifact under test, so it is captured and generation never
// happens.
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

static int bt_fail = 0;

static void ck(int cond, const char *what) {
    if (!cond) { fprintf(stderr, "FAIL: %s\n", what); bt_fail = 1; }
    else        fprintf(stderr, "ok: %s\n", what);
}

// ------------------------------------------------------ captured generation

static char *bt_prompt;

void run_completion(slot_t *s, sock_t fd, const char *prompt, int api,
                    jv *req, const tool_envelope *env) {
    (void)s; (void)fd; (void)api; (void)req; (void)env;
    free(bt_prompt);
    bt_prompt = prompt ? strdup(prompt) : NULL;
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

// A marker the user's turn carries. It is the LAST thing rendered before the
// generation header, so it is the first casualty of a short budget.
#define BT_MARKER "OSLO-SENTINEL"

// Harmony's generation header when tools are declared (template.c: with tools
// the model picks its own channel, so the header is bare).
#define BT_HEAD "<|start|>assistant"

// n bytes of prose. Real tool descriptions are prose, and prose is what the
// TypeScript namespace copies verbatim into the prompt.
static char *bt_filler(size_t n) {
    static const char *word[] = { "the ", "tool ", "returns ", "a ", "value ",
                                  "for ", "one ", "city ", "name ", "given " };
    char *s = malloc(n + 16);
    assert(s);
    size_t i = 0;
    for (int k = 0; i < n; k++) {
        const char *w = word[k % 10];
        size_t wn = strlen(w);
        memcpy(s + i, w, wn);
        i += wn;
    }
    s[i] = 0;
    return s;
}

// Run one request body through a route on a Harmony slot and capture the
// prompt it would have generated from.
static void bt_run(void (*route)(slot_t *, sock_t, jv *), const char *body) {
    free(bt_prompt);
    bt_prompt = NULL;
    jv *req = json_parse(body, strlen(body));
    assert(req);
    slot_t s = {0};
    s.tmpl = TMPL_HARMONY;
    route(&s, (sock_t)-1, req);
    jv_free(req);
}

// The three properties a complete Harmony prompt has. Any one of them missing
// means the buffer ran out before the renderer did.
static void bt_expect_complete(const char *what) {
    char msg[192];
    snprintf(msg, sizeof msg, "%s: a prompt was produced", what);
    ck(bt_prompt != NULL, msg);
    if (!bt_prompt) return;

    snprintf(msg, sizeof msg, "%s: tool namespace is rendered", what);
    ck(strstr(bt_prompt, "# Tools") != NULL, msg);

    snprintf(msg, sizeof msg, "%s: the user's turn survived", what);
    ck(strstr(bt_prompt, BT_MARKER) != NULL, msg);

    size_t n = strlen(bt_prompt), h = strlen(BT_HEAD);
    snprintf(msg, sizeof msg, "%s: ends with the generation header", what);
    ck(n >= h && !strcmp(bt_prompt + n - h, BT_HEAD), msg);
    if (bt_fail && getenv("RUNNER_PROMPT_TRACE"))
        fprintf(stderr, "--- tail: %s\n", bt_prompt + (n > 120 ? n - 120 : 0));
}

// ------------------------------------------------------------ request bodies

static char *bt_chat_body(size_t desc, size_t reasoning) {
    char *d = bt_filler(desc), *r = bt_filler(reasoning);
    sbuf b = {0};
    sb_lit(&b, "{\"model\":\"m\",\"messages\":[");
    if (reasoning) {
        sb_lit(&b, "{\"role\":\"user\",\"content\":\"hello\"},"
                   "{\"role\":\"assistant\",\"content\":\"sure\","
                   "\"reasoning_content\":\"");
        sb_esc(&b, r, strlen(r));
        sb_lit(&b, "\"},");
    }
    sb_lit(&b, "{\"role\":\"user\",\"content\":\"What is the weather in "
               BT_MARKER "?\"}],\"tools\":[{\"type\":\"function\","
               "\"function\":{\"name\":\"get_weather\",\"description\":\"");
    sb_esc(&b, d, strlen(d));
    sb_lit(&b, "\",\"parameters\":{\"type\":\"object\",\"properties\":"
               "{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}}}]}");
    free(d); free(r);
    assert(!b.failed);
    return b.s;
}

static char *bt_responses_body(size_t desc) {
    char *d = bt_filler(desc);
    sbuf b = {0};
    sb_lit(&b, "{\"model\":\"m\",\"input\":[{\"type\":\"message\","
               "\"role\":\"user\",\"content\":\"What is the weather in "
               BT_MARKER "?\"}],\"tools\":[{\"type\":\"function\","
               "\"name\":\"get_weather\",\"description\":\"");
    sb_esc(&b, d, strlen(d));
    sb_lit(&b, "\",\"parameters\":{\"type\":\"object\",\"properties\":"
               "{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}}]}");
    free(d);
    assert(!b.failed);
    return b.s;
}

static char *bt_messages_body(size_t desc) {
    char *d = bt_filler(desc);
    sbuf b = {0};
    sb_lit(&b, "{\"model\":\"m\",\"max_tokens\":16,\"messages\":"
               "[{\"role\":\"user\",\"content\":\"What is the weather in "
               BT_MARKER "?\"}],\"tools\":[{\"name\":\"get_weather\","
               "\"description\":\"");
    sb_esc(&b, d, strlen(d));
    sb_lit(&b, "\",\"input_schema\":{\"type\":\"object\",\"properties\":"
               "{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}}]}");
    free(d);
    assert(!b.failed);
    return b.s;
}

// ------------------------------------------------------------------- checks

// A tool declaration far larger than any fixed slack a call site could have
// guessed. The namespace it renders into the prompt is roughly this size, and
// nothing in any estimate accounted for it.
enum { BT_BIG = 24000 };

static void test_chat_big_tool_declaration(void) {
    char *body = bt_chat_body(BT_BIG, 0);
    bt_run(handle_chat, body);
    bt_expect_complete("chat/completions, 24KB tool description");
    free(body);
}

static void test_chat_replayed_reasoning(void) {
    // Small declarations, large replayed analysis turn: the reasoning turn is
    // pushed into the turn list with no matching contribution to the estimate,
    // so this fails even when the tool namespace is tiny.
    char *body = bt_chat_body(32, BT_BIG);
    bt_run(handle_chat, body);
    bt_expect_complete("chat/completions, 24KB replayed reasoning_content");
    free(body);
}

static void test_responses_big_tool_declaration(void) {
    char *body = bt_responses_body(BT_BIG);
    bt_run(handle_responses, body);
    bt_expect_complete("responses, 24KB tool description");
    free(body);
}

static void test_messages_big_tool_declaration(void) {
    char *body = bt_messages_body(BT_BIG);
    bt_run(handle_messages, body);
    bt_expect_complete("messages, 24KB tool description");
    free(body);
}

int main(void) {
    test_chat_big_tool_declaration();
    test_chat_replayed_reasoning();
    test_responses_big_tool_declaration();
    test_messages_big_tool_declaration();
    free(bt_prompt);
    fprintf(stderr, bt_fail ? "test-prompt-budget: FAILED\n"
                            : "test-prompt-budget: all checks passed\n");
    return bt_fail;
}
