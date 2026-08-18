// Allocation-failure sweep for the Anthropic Messages inbound translation.
//
// /v1/messages eats an untrusted, client-sized request body and flattens it
// into chat turns before anything is generated. Runner runs near its memory
// limits on purpose (--reserve, multi-GB weights, hybrid splits), so a failed
// allocation on that path is an ordinary condition, and there are only two
// answers this project accepts to one: refuse with a 5xx that says so, or
// succeed with the prompt the caller actually asked for. Never a 200 with a
// turn quietly missing, and never an error body assembled out of whatever was
// on the stack.
//
// api_anthropic.c is compiled INTO this test with the allocators
// macro-substituted, the way tests/test_json_oom.c does it. server.c is
// #included unsubstituted (it owns the static route table and the prompt
// renderer, exactly as tests/test_tool_attribution.c arranges it) so the
// failures injected here are the ones this file is about.
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

static int mo_fail = 0;

static void ck(int cond, const char *what) {
    if (!cond) { fprintf(stderr, "FAIL: %s\n", what); mo_fail = 1; }
}

// ------------------------------------------------------ captured generation

static char *mo_prompt;

void run_completion(slot_t *s, sock_t fd, const char *prompt, int api,
                    jv *req, const tool_envelope *env) {
    (void)s; (void)fd; (void)api; (void)req; (void)env;
    free(mo_prompt);
    mo_prompt = prompt ? strdup(prompt) : NULL;
}

// completion.c's request readers, stubbed: the fields they read are absent
// from the request below, so the defaults are the whole behaviour.
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

// ------------------------------------------------------ injected allocator

static long alloc_calls;
static long fail_at = -1;

static void *t_malloc(size_t n) {
    if (fail_at >= 0 && alloc_calls++ == fail_at) return NULL;
    if (fail_at < 0) alloc_calls++;
    return malloc(n);
}

static char *t_strdup(const char *s) {
    size_t n = strlen(s) + 1;
    char *p = t_malloc(n);
    if (p) memcpy(p, s, n);
    return p;
}

#define malloc  t_malloc
#define strdup  t_strdup
#include "../src/api_anthropic.c"
#undef malloc
#undef strdup

// ------------------------------------------------------------------ fixture

static int  mo_status;
static char mo_message[1024];
static int  mo_tmpl = TMPL_CHATML;

// Markers the prompt must carry. A turn that goes missing under an injected
// failure is the silent success this project refuses, so it is checked on the
// prompt rather than inferred from a status code.
#define SYS_MARK "SYSTEMMARKERZQ"
#define USR_MARK "USERMARKERZQ"
#define RES_MARK "RESULTMARKERZQ"

static const char BODY[] =
    "{\"model\":\"m\",\"max_tokens\":16,"
    "\"system\":\"" SYS_MARK "\","
    "\"messages\":["
    "{\"role\":\"user\",\"content\":\"" USR_MARK "\"},"
    "{\"role\":\"assistant\",\"content\":[{\"type\":\"tool_use\","
    "\"id\":\"toolu_1\",\"name\":\"get_time\",\"input\":{}}]},"
    "{\"role\":\"user\",\"content\":[{\"type\":\"tool_result\","
    "\"tool_use_id\":\"toolu_1\",\"content\":\"" RES_MARK "\"}]}],"
    "\"tools\":[{\"name\":\"get_time\",\"input_schema\":{\"type\":\"object\"}}]}";

static void mo_run(void) {
    free(mo_prompt);
    mo_prompt = NULL;
    mo_status = 0;
    mo_message[0] = 0;

    int sv[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0) {
        ck(0, "socketpair");
        return;
    }
    jv *req = json_parse(BODY, strlen(BODY));
    assert(req);
    slot_t s = {0};
    s.tmpl = mo_tmpl;
    handle_messages(&s, (sock_t)sv[0], req);
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
    if (!strncmp(buf, "HTTP/1.1 ", 9)) mo_status = atoi(buf + 9);
    const char *m = strstr(buf, "\"message\":\"");
    if (m) {
        m += 11;
        size_t n = 0;
        while (*m && *m != '"' && n < sizeof(mo_message) - 1) {
            if (*m == '\\' && m[1]) m++;
            mo_message[n++] = *m++;
        }
        mo_message[n] = 0;
    }
}

// One injected failure, checked against the only two acceptable outcomes.
static void check_one(long k, long total) {
    char what[192];
    if (mo_prompt) {
        // succeeded: the prompt must be the one the request asked for
        if (!strstr(mo_prompt, SYS_MARK) || !strstr(mo_prompt, USR_MARK) ||
            !strstr(mo_prompt, RES_MARK)) {
            snprintf(what, sizeof what,
                     "allocation %ld of %ld: answered 200 with a turn missing "
                     "from the prompt", k, total);
            ck(0, what);
            fprintf(stderr, "    prompt: %s\n", mo_prompt);
        }
        return;
    }
    snprintf(what, sizeof what,
             "allocation %ld of %ld: refused rather than dropped", k, total);
    ck(mo_status != 0, what);
    if (!mo_status) return;
    // An allocation failure is the server's problem, not a malformed request:
    // 400 tells the caller to change something they did not get wrong.
    snprintf(what, sizeof what,
             "allocation %ld of %ld: answered %d, wanted a 5xx", k, total,
             mo_status);
    ck(mo_status >= 500, what);
    // the message is a fixed string this file chose, not whatever the stack
    // happened to hold: printable, terminated, and about memory
    for (const char *p = mo_message; *p; p++) {
        if (isprint((unsigned char)*p)) continue;
        snprintf(what, sizeof what,
                 "allocation %ld of %ld: error message is not printable text",
                 k, total);
        ck(0, what);
        break;
    }
    snprintf(what, sizeof what,
             "allocation %ld of %ld: the message says what went wrong (got "
             "\"%.60s\")", k, total, mo_message);
    ck(strstr(mo_message, "memory") != NULL, what);
}

static void sweep(int tmpl, const char *label) {
    mo_tmpl = tmpl;
    fail_at = -1;
    alloc_calls = 0;
    mo_run();
    ck(mo_prompt != NULL, label);
    if (!mo_prompt) return;
    ck(strstr(mo_prompt, SYS_MARK) && strstr(mo_prompt, USR_MARK) &&
       strstr(mo_prompt, RES_MARK), "the clean prompt carries every turn");
    long total = alloc_calls;
    ck(total > 0, "the sweep has allocations to fail");

    for (long k = 0; k < total; k++) {
        fail_at = k;
        alloc_calls = 0;
        mo_run();
        check_one(k, total);
    }
    fail_at = -1;
    fprintf(stderr, "%s: %ld allocations swept\n", label, total);
}

int main(void) {
    sweep(TMPL_CHATML, "chatml: a clean request builds a prompt");
    sweep(TMPL_HARMONY, "harmony: a clean request builds a prompt");
    free(mo_prompt);
    fprintf(stderr, mo_fail ? "test-messages-oom: FAILED\n"
                            : "test-messages-oom: all checks passed\n");
    return mo_fail;
}
