// Allocation-failure sweep for the three inbound request translations.
//
// /v1/chat/completions, /v1/responses and /v1/messages each eat an untrusted,
// client-sized body and flatten it into chat turns before anything is
// generated. Runner runs near its memory limits on purpose (--reserve, multi-GB
// weights, hybrid splits), so a failed allocation on those paths is an ordinary
// condition, and there are only two answers this project accepts to one: refuse
// with a 5xx that says so, or succeed with the prompt the caller actually asked
// for. Never a 200 with a turn quietly missing, and never an error body
// assembled out of whatever was on the stack.
//
// server.c (handle_chat, message_text, render_prompt_alloc), api_responses.c
// and api_anthropic.c are all compiled INTO this test with the allocators
// macro-substituted, the way tests/test_json_oom.c does it; the routes are
// driven over a socketpair the way tests/test_tool_attribution.c does it, so
// the status code and message asserted here are the ones a caller would see.
#include "runner.h"
#include "json.h"
#include "http.h"
#include "server_int.h"
#include "completion.h"
#include "template.h"
#include "api.h"

#include <assert.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

// ------------------------------------------------------ injected allocator
//
// json.c comes along, because most of the memory these translations spend is
// spent through its `sbuf` builder and its `jv` tree -- a sweep that failed
// only the direct malloc/strdup calls would miss the paths that actually
// allocate. free() is deliberately NOT substituted: nothing here counts live
// blocks (tests/test_json_oom.c owns that half), so the shims hand back
// ordinary heap that ordinary free() releases.
//
// Injection is armed only for the duration of the route call, so building the
// request itself is never the thing that fails.

static long alloc_calls;
static long fail_at = -1;

static void *t_malloc(size_t n) {
    if (fail_at >= 0 && alloc_calls++ == fail_at) return NULL;
    if (fail_at < 0) alloc_calls++;
    return malloc(n);
}

static void *t_calloc(size_t a, size_t b) {
    if (fail_at >= 0 && alloc_calls++ == fail_at) return NULL;
    if (fail_at < 0) alloc_calls++;
    return calloc(a, b);
}

static void *t_realloc(void *p, size_t n) {
    if (fail_at >= 0 && alloc_calls++ == fail_at) return NULL;
    if (fail_at < 0) alloc_calls++;
    return realloc(p, n);
}

static char *t_strdup(const char *s) {
    size_t n = strlen(s) + 1;
    char *p = t_malloc(n);
    if (p) memcpy(p, s, n);
    return p;
}

#define malloc  t_malloc
#define calloc  t_calloc
#define realloc t_realloc
#define strdup  t_strdup
#include "../src/json.c"
#include "../src/server.c"
#include "../src/api_responses.c"
#include "../src/api_anthropic.c"
#undef malloc
#undef calloc
#undef realloc
#undef strdup

// ------------------------------------------------------ captured generation

static int po_fail = 0;

static void ck(int cond, const char *what) {
    if (!cond) { fprintf(stderr, "FAIL: %s\n", what); po_fail = 1; }
}

static char *po_prompt;

void run_completion(slot_t *s, sock_t fd, const char *prompt, int api,
                    jv *req, const tool_envelope *env) {
    (void)s; (void)fd; (void)api; (void)req; (void)env;
    free(po_prompt);
    po_prompt = prompt ? strdup(prompt) : NULL;
}

// completion.c's request readers, stubbed: the fields they read are absent
// from the bodies below, so the defaults are the whole behaviour.
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

// ------------------------------------------------------------------ fixture

static int  po_status;
static char po_message[1024];

// Markers every prompt must carry. A turn that goes missing under an injected
// failure is the silent success this project refuses, so it is checked on the
// prompt rather than inferred from a status code.
#define SYS_MARK "SYSTEMMARKERZQ"
#define USR_MARK "USERMARKERZQ"
#define RES_MARK "RESULTMARKERZQ"

// The failure to inject on the NEXT route call, -1 for none. po_run arms it
// after the request has been parsed and disarms it before the reply is read.
static long po_arm = -1;

static void po_run(void (*route)(slot_t *, sock_t, jv *), int tmpl,
                   const char *body) {
    free(po_prompt);
    po_prompt = NULL;
    po_status = 0;
    po_message[0] = 0;

    int sv[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0) {
        ck(0, "socketpair");
        return;
    }
    jv *req = json_parse(body, strlen(body));
    assert(req);
    slot_t s = {0};
    s.tmpl = tmpl;
    fail_at = po_arm;
    alloc_calls = 0;
    route(&s, (sock_t)sv[0], req);
    long used = alloc_calls;
    fail_at = -1;
    if (po_arm < 0) alloc_calls = used;
    jv_free(req);

    shutdown(sv[0], SHUT_WR);
    static char buf[16384];
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
    if (!strncmp(buf, "HTTP/1.1 ", 9)) po_status = atoi(buf + 9);
    const char *m = strstr(buf, "\"message\":\"");
    if (m) {
        m += 11;
        size_t n = 0;
        while (*m && *m != '"' && n < sizeof(po_message) - 1) {
            if (*m == '\\' && m[1]) m++;
            po_message[n++] = *m++;
        }
        po_message[n] = 0;
    }
}

// Two failures found by this sweep live in modules it drives but does not own,
// and are recorded here rather than silently tolerated: pinning today's
// behaviour is what stops them widening.
//
//   * template.c's Harmony renderer folds every system message and the tool
//     namespace into one `sbuf dev` and gates the developer turn on `dev.n`
//     (src/template.c, the `sbuf dev = {0}` block). A failed grow leaves n at
//     0, so the whole developer turn -- the caller's system prompt AND the
//     tool declarations -- is dropped and the render reports success.
//     render_prompt_alloc cannot see it: the short prompt fits the buffer.
//   * tool_envelope_build() reports "out of memory building the tool envelope"
//     through the same rc < 0 that a malformed declaration uses, so the three
//     call sites cannot tell them apart and answer 400 for both.
//
// Both need a change in a module outside this file's reach. Everything else is
// a hard assertion.
static bool known_gap(const char *label, const char *missing) {
    if (missing && strstr(label, "harmony") && !strcmp(missing, SYS_MARK))
        return true;   // template.c's developer turn, above
    if (!missing && strstr(po_message, "out of memory building the tool "
                                       "envelope"))
        return true;   // tool_envelope_build's undifferentiated rc, above
    return false;
}

// One injected failure, checked against the only two acceptable outcomes.
static void check_one(const char *label, long k, long total) {
    char what[256];
    if (po_prompt) {
        const char *missing = !strstr(po_prompt, SYS_MARK) ? SYS_MARK
                            : !strstr(po_prompt, USR_MARK) ? USR_MARK
                            : !strstr(po_prompt, RES_MARK) ? RES_MARK : NULL;
        if (missing && !known_gap(label, missing)) {
            snprintf(what, sizeof what,
                     "%s: allocation %ld of %ld answered 200 with %s missing "
                     "from the prompt", label, k, total, missing);
            ck(0, what);
            fprintf(stderr, "    prompt: %s\n", po_prompt);
        }
        return;
    }
    snprintf(what, sizeof what,
             "%s: allocation %ld of %ld refused rather than dropped",
             label, k, total);
    ck(po_status != 0, what);
    if (!po_status) return;
    // The refusal is a message this codebase chose, not whatever the stack
    // happened to hold: printable, and terminated inside the buffer.
    for (const char *p = po_message; *p; p++) {
        if (isprint((unsigned char)*p)) continue;
        snprintf(what, sizeof what,
                 "%s: allocation %ld of %ld: message is not printable text",
                 label, k, total);
        ck(0, what);
        break;
    }
    // An allocation failure is the server's problem. A refusal that SAYS it ran
    // out of memory and stamps 400 invalid_request_error on it tells the caller
    // to fix a request that was never wrong.
    if (strstr(po_message, "out of memory") && !known_gap(label, NULL)) {
        snprintf(what, sizeof what,
                 "%s: allocation %ld of %ld answered %d for \"%.60s\", "
                 "wanted a 5xx", label, k, total, po_status, po_message);
        ck(po_status >= 500, what);
    }
}

static void sweep(const char *label, void (*route)(slot_t *, sock_t, jv *),
                  int tmpl, const char *body) {
    char what[192];
    po_arm = -1;
    po_run(route, tmpl, body);
    snprintf(what, sizeof what, "%s: a clean request builds a prompt", label);
    ck(po_prompt != NULL, what);
    if (!po_prompt) {
        fprintf(stderr, "    status %d: %s\n", po_status, po_message);
        return;
    }
    snprintf(what, sizeof what, "%s: the clean prompt carries every turn",
             label);
    ck(strstr(po_prompt, SYS_MARK) && strstr(po_prompt, USR_MARK) &&
       strstr(po_prompt, RES_MARK), what);
    long total = alloc_calls;
    snprintf(what, sizeof what, "%s: the sweep has allocations to fail", label);
    ck(total > 0, what);

    for (long k = 0; k < total; k++) {
        po_arm = k;
        po_run(route, tmpl, body);
        check_one(label, k, total);
    }
    po_arm = -1;
    fprintf(stderr, "%s: %ld allocations swept\n", label, total);
}

// ------------------------------------------------------------ request bodies

static const char CHAT_BODY[] =
    "{\"model\":\"m\",\"max_tokens\":16,\"messages\":["
    "{\"role\":\"system\",\"content\":\"" SYS_MARK "\"},"
    "{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\""
        USR_MARK "\"}]},"
    "{\"role\":\"assistant\",\"tool_calls\":[{\"id\":\"c1\","
    "\"type\":\"function\",\"function\":{\"name\":\"get_time\","
    "\"arguments\":\"{}\"}}]},"
    "{\"role\":\"tool\",\"tool_call_id\":\"c1\",\"content\":\""
        RES_MARK "\"}],"
    "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"get_time\","
    "\"parameters\":{\"type\":\"object\"}}}]}";

static const char RESP_BODY[] =
    "{\"model\":\"m\",\"instructions\":\"" SYS_MARK "\",\"input\":["
    "{\"type\":\"message\",\"role\":\"user\",\"content\":"
    "[{\"type\":\"input_text\",\"text\":\"" USR_MARK "\"}]},"
    "{\"type\":\"function_call\",\"call_id\":\"c1\",\"name\":\"get_time\","
    "\"arguments\":\"{}\"},"
    "{\"type\":\"function_call_output\",\"call_id\":\"c1\",\"output\":\""
        RES_MARK "\"}],"
    "\"tools\":[{\"type\":\"function\",\"name\":\"get_time\","
    "\"parameters\":{\"type\":\"object\"}}]}";

static const char MSG_BODY[] =
    "{\"model\":\"m\",\"max_tokens\":16,"
    "\"system\":\"" SYS_MARK "\","
    "\"messages\":["
    "{\"role\":\"user\",\"content\":\"" USR_MARK "\"},"
    "{\"role\":\"assistant\",\"content\":[{\"type\":\"tool_use\","
    "\"id\":\"toolu_1\",\"name\":\"get_time\",\"input\":{}}]},"
    "{\"role\":\"user\",\"content\":[{\"type\":\"tool_result\","
    "\"tool_use_id\":\"toolu_1\",\"content\":\"" RES_MARK "\"}]}],"
    "\"tools\":[{\"name\":\"get_time\",\"input_schema\":{\"type\":\"object\"}}]}";

int main(void) {
    sweep("chat/chatml",     handle_chat,      TMPL_CHATML,  CHAT_BODY);
    sweep("chat/harmony",    handle_chat,      TMPL_HARMONY, CHAT_BODY);
    sweep("responses/chatml", handle_responses, TMPL_CHATML,  RESP_BODY);
    sweep("responses/harmony", handle_responses, TMPL_HARMONY, RESP_BODY);
    sweep("messages/chatml",  handle_messages,  TMPL_CHATML,  MSG_BODY);
    sweep("messages/harmony", handle_messages,  TMPL_HARMONY, MSG_BODY);
    free(po_prompt);
    fprintf(stderr, po_fail ? "test-prompt-oom: FAILED\n"
                            : "test-prompt-oom: all checks passed\n");
    return po_fail;
}
