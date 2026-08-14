// Runner-side driver for scripts/template-conformance.py.
//
// Reads a JSON job list on stdin and writes each case's NATIVE render to
// stdout, so the python harness can diff runner's C renderer byte-for-byte
// against the model's own jinja chat template.
//
//   stdin : {"cases":[{"id":"...", "template":"chatml",
//                      "add_generation_prompt":true,
//                      "request":{<a verbatim OpenAI chat request body>}}, ...],
//            "tokenize":[{"id":"...", "gguf":"/abs/path.gguf",
//                         "text":"..."}, ...]}
//   stdout: {"results":[{"id":"...","prompt":"...","error":null} ...],
//            "tokens":[{"id":"...","ids":[1,2],"error":null} ...]}
//
// ---------------------------------------------------------------------------
// WHY THIS DRIVES handle_chat AND NOT render_messages_with_tools
//
// render_messages_with_tools() takes FLAT {role, content, name, channel}
// turns. An OpenAI request is not that shape: content may be a part array,
// assistant turns carry tool_calls, harmony explodes one assistant message
// into several turns, ornith rewrites tool turns to user turns and folds the
// system message into the tool declarations, muse names the turn after the
// function it called, and the tool declarations themselves become a leading
// system turn whose syntax depends on whether the strict envelope applies.
// ALL of that lives in src/server.c, upstream of the renderer.
//
// The first version of this harness re-implemented that flattening in python.
// It was the one part of the comparison that was a COPY of runner rather than
// runner, so it went stale silently whenever server.c changed, and it could
// not express a content part-array at all. This file replaces the copy with
// the real thing: src/server.c is #included (handle_chat is static there, the
// same technique tests/test_prompt_budget.c and tests/test_tool_attribution.c
// already use) and driven with the exact request body the python side hands
// jinja. Nothing between the request and the renderer is re-implemented.
//
// ONE substitution is made, and it is the only one:
//
//   handle_chat always renders with add_assistant = true -- the server has no
//   reason to build a prompt that does not ask for a completion. The matrix
//   needs both, because render_messages_with_tools() is a public entry point
//   whose add_assistant = false branch is reached from src/main.c's CLI path
//   and must conform too. So render_messages_with_tools is macro-renamed
//   across server.c's translation unit to conf_render_messages_with_tools
//   below, which forwards to the real function with the CASE's
//   add_generation_prompt in place of handle_chat's hard-coded true.
//
// Everything else -- flattening, tool declarations, thinking mode, the
// grow-until-it-fits buffer loop -- is runner's own code on runner's own
// path.
//
// The generation path is stubbed rather than linked: the prompt is the
// artifact under test, so it is captured out of run_completion and no model
// is ever loaded. api_responses.c / api_anthropic.c are left out of the link
// for the same reason and their three entry points are stubbed here.
#include "runner.h"
#include "json.h"
#include "http.h"
#include "server_int.h"
#include "completion.h"
#include "template.h"
#include "api.h"
#include "gguf.h"
#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

// --------------------------------------------------------- captured render

static char *g_prompt;          // the prompt handle_chat would have generated
static int   g_prompt_calls;    // how many times it got that far
static bool  g_addgen = true;   // the case's add_generation_prompt

void run_completion(slot_t *s, sock_t fd, const char *prompt, int api,
                    jv *req, const tool_envelope *env) {
    (void)s; (void)fd; (void)api; (void)req; (void)env;
    free(g_prompt);
    g_prompt = prompt ? strdup(prompt) : NULL;
    g_prompt_calls++;
}

// completion.c's request readers, stubbed. The fields they read are absent
// from every request the harness builds, so the defaults are the whole
// behaviour -- except request_bool, which server.c uses to read
// parallel_tool_calls and atem_tool_calling and which the matrix does set.
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

// The other two inbound protocol adapters. Referenced only from server.c's
// route table, which this harness never reaches.
void handle_responses(slot_t *s, sock_t fd, jv *req) {
    (void)s; (void)fd; (void)req;
}
void handle_messages(slot_t *s, sock_t fd, jv *req) {
    (void)s; (void)fd; (void)req;
}
void handle_count_tokens(slot_t *s, sock_t fd, jv *req) {
    (void)s; (void)fd; (void)req;
}

// The add_generation_prompt substitution described at the top of this file.
// template.h is already included above, so the #define below rewrites only
// server.c's CALL, not the declaration -- and the forwarding definition after
// the #undef reaches the real function in template.c.
size_t conf_render_messages_with_tools(int tmpl, const chat_msg *msgs,
                                       int n_msgs, bool add_assistant,
                                       int thinking, const jv *tools,
                                       char *out, size_t cap);

#define render_messages_with_tools conf_render_messages_with_tools
#include "../src/server.c"
#undef render_messages_with_tools

size_t conf_render_messages_with_tools(int tmpl, const chat_msg *msgs,
                                       int n_msgs, bool add_assistant,
                                       int thinking, const jv *tools,
                                       char *out, size_t cap) {
    (void)add_assistant;   // always true from handle_chat; see the header note
    return render_messages_with_tools(tmpl, msgs, n_msgs, g_addgen, thinking,
                                      tools, out, cap);
}

// ------------------------------------------------------------------- input

static char *read_all(FILE *f, size_t *out_n) {
    size_t cap = 1 << 16, n = 0;
    char *b = malloc(cap);
    if (!b) return NULL;
    for (;;) {
        if (n + 1 >= cap) {
            char *g = realloc(b, cap * 2);
            if (!g) { free(b); return NULL; }
            b = g; cap *= 2;
        }
        size_t got = fread(b + n, 1, cap - n - 1, f);
        n += got;
        if (got == 0) break;
    }
    b[n] = 0;
    *out_n = n;
    return b;
}

// ------------------------------------------------------------ one render
//
// Whatever the route writes to the socket is read back, so a request the
// server REFUSES reports its own 400 instead of looking like an empty render.
// A refusal is not a difference and not a pass: the python side turns it into
// NOT-CHECKED.

static void drain(int fd, char *buf, size_t cap) {
    size_t got = 0;
    for (;;) {
        ssize_t r = recv(fd, buf + got, cap - 1 - got, 0);
        if (r <= 0) break;
        got += (size_t)r;
        if (got >= cap - 1) break;
    }
    buf[got] = 0;
}

// The `message` field of runner's error body, unescaped enough to read.
static void error_message(const char *body, char *out, size_t cap) {
    out[0] = 0;
    const char *m = strstr(body, "\"message\":\"");
    if (!m) {
        if (!strncmp(body, "HTTP/1.1 ", 9))
            snprintf(out, cap, "route answered HTTP %.3s with no message",
                     body + 9);
        return;
    }
    m += 11;
    size_t n = 0;
    while (*m && *m != '"' && n < cap - 1) {
        if (*m == '\\' && m[1]) m++;
        out[n++] = *m++;
    }
    out[n] = 0;
}

// Returns 0 and fills *prompt (owned) on success, or -1 with *err filled.
static int render_one(int tmpl, bool addgen, jv *request,
                      char **prompt, char *err, size_t errcap) {
    g_addgen = addgen;
    free(g_prompt);
    g_prompt = NULL;
    g_prompt_calls = 0;
    err[0] = 0;

    int sv[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0) {
        snprintf(err, errcap, "socketpair failed");
        return -1;
    }
    slot_t s = {0};
    s.tmpl = tmpl;
    handle_chat(&s, (sock_t)sv[0], request);
    shutdown(sv[0], SHUT_WR);

    static char body[8192];
    drain(sv[1], body, sizeof body);
    close(sv[0]);
    close(sv[1]);

    if (g_prompt_calls == 1 && g_prompt) {
        *prompt = g_prompt;
        g_prompt = NULL;
        return 0;
    }
    if (g_prompt_calls > 1) {
        snprintf(err, errcap, "handle_chat rendered %d times for one request; "
                 "this harness assumes one", g_prompt_calls);
        return -1;
    }
    char msg[1024];
    error_message(body, msg, sizeof msg);
    // No render and no error body would mean handle_chat stopped calling
    // run_completion -- the harness would then be measuring nothing, so it
    // says so rather than reporting an empty prompt.
    snprintf(err, errcap, "%s", msg[0] ? msg :
             "handle_chat produced no prompt and no error body");
    return -1;
}

// -------------------------------------------------------------- tokenizers
//
// The tokenizer lives in the GGUF, so runner can answer "what token ids does
// this text become" for any family whose model is on the shelf. Only the
// METADATA is needed, so the header-only read path is used: it does not
// require the tensor data to be present or intact.

typedef struct {
    char      *path;
    gguf_file  g;
    tokenizer  t;
    bool       open;      // gguf mapped
    bool       ready;     // tokenizer built
    char       err[256];
} tokslot;

static tokslot *g_toks;
static int      g_n_toks;

static tokslot *tok_for(const char *path) {
    for (int i = 0; i < g_n_toks; i++)
        if (!strcmp(g_toks[i].path, path)) return &g_toks[i];
    tokslot *g = realloc(g_toks, sizeof(*g_toks) * (size_t)(g_n_toks + 1));
    if (!g) return NULL;
    g_toks = g;
    tokslot *s = &g_toks[g_n_toks++];
    memset(s, 0, sizeof *s);
    s->path = strdup(path);
    if (!s->path) { snprintf(s->err, sizeof s->err, "oom"); return s; }
    if (!gguf_open_header(&s->g, path)) {
        snprintf(s->err, sizeof s->err, "cannot read GGUF metadata from %s",
                 path);
        return s;
    }
    s->open = true;
    if (!tokenizer_init(&s->t, &s->g)) {
        snprintf(s->err, sizeof s->err,
                 "%s carries no usable tokenizer vocabulary", path);
        return s;
    }
    s->ready = true;
    return s;
}

static void toks_free(void) {
    for (int i = 0; i < g_n_toks; i++) {
        if (g_toks[i].ready) tokenizer_free(&g_toks[i].t);
        if (g_toks[i].open)  gguf_close(&g_toks[i].g);
        free(g_toks[i].path);
    }
    free(g_toks);
}

// -------------------------------------------------------------------- main

static void emit_str(sbuf *out, const char *key, const char *v) {
    sb_fmt(out, "\"%s\":", key);
    if (!v) { sb_lit(out, "null"); return; }
    sb_lit(out, "\"");
    sb_esc(out, v, strlen(v));
    sb_lit(out, "\"");
}

static int do_cases(jv *cases, sbuf *out) {
    sb_lit(out, "\"results\":[");
    for (int i = 0; i < cases->n; i++) {
        jv *c = cases->items[i];
        const char *id = jv_str(jv_get(c, "id"), "");
        const char *tname = jv_str(jv_get(c, "template"), "raw");
        int tmpl = template_from_name(tname);
        if (tmpl < 0) {
            fprintf(stderr, "unknown template %s\n", tname);
            return 2;
        }
        bool addgen = jv_bool(jv_get(c, "add_generation_prompt"), true);
        jv *request = jv_get(c, "request");
        char err[1024];
        char *prompt = NULL;
        int rc = request
            ? render_one(tmpl, addgen, request, &prompt, err, sizeof err)
            : (snprintf(err, sizeof err, "case has no `request`"), -1);
        if (i) sb_lit(out, ",");
        sb_lit(out, "{");
        emit_str(out, "id", id);
        sb_lit(out, ",");
        emit_str(out, "prompt", rc == 0 ? prompt : NULL);
        sb_lit(out, ",");
        emit_str(out, "error", rc == 0 ? NULL : err);
        sb_lit(out, "}");
        free(prompt);
    }
    sb_lit(out, "]");
    return 0;
}

static int do_tokenize(jv *jobs, sbuf *out) {
    enum { TOK_CAP = 1 << 20 };
    int32_t *ids = malloc(sizeof(int32_t) * TOK_CAP);
    if (!ids) { fprintf(stderr, "oom\n"); return 2; }
    sb_lit(out, "\"tokens\":[");
    for (int i = 0; i < jobs->n; i++) {
        jv *j = jobs->items[i];
        const char *id = jv_str(jv_get(j, "id"), "");
        const char *path = jv_str(jv_get(j, "gguf"), NULL);
        const char *text = jv_str(jv_get(j, "text"), "");
        if (i) sb_lit(out, ",");
        sb_lit(out, "{");
        emit_str(out, "id", id);
        tokslot *t = path ? tok_for(path) : NULL;
        if (!t || !t->ready) {
            sb_lit(out, ",\"ids\":null,");
            emit_str(out, "error", t && t->err[0] ? t->err
                                                  : "no tokenizer GGUF given");
            sb_lit(out, "}");
            continue;
        }
        // add_bos/parse_special exactly as the generation path tokenizes a
        // prompt (src/completion.c, src/server.c, src/api_anthropic.c all
        // pass true,true). A token comparison run under different flags
        // would answer a question production never asks.
        int n = tok_encode(&t->t, text, ids, TOK_CAP, true, true);
        if (n < 0) {
            sb_lit(out, ",\"ids\":null,");
            emit_str(out, "error", "tok_encode failed (allocation, or the "
                                   "text exceeded the harness token cap)");
            sb_lit(out, "}");
            continue;
        }
        sb_lit(out, ",\"ids\":[");
        for (int k = 0; k < n; k++)
            sb_fmt(out, "%s%d", k ? "," : "", (int)ids[k]);
        sb_lit(out, "],");
        emit_str(out, "error", NULL);
        sb_lit(out, "}");
    }
    sb_lit(out, "]");
    free(ids);
    return 0;
}

int main(void) {
    size_t n_in = 0;
    char *in = read_all(stdin, &n_in);
    if (!in) { fprintf(stderr, "oom\n"); return 2; }
    jv *req = json_parse(in, n_in);
    if (!req) { fprintf(stderr, "input is not JSON\n"); return 2; }

    sbuf out = {0};
    sb_lit(&out, "{");
    jv *cases = jv_get(req, "cases");
    if (cases && cases->type == J_ARR) {
        int rc = do_cases(cases, &out);
        if (rc) return rc;
    } else {
        sb_lit(&out, "\"results\":[]");
    }
    jv *jobs = jv_get(req, "tokenize");
    sb_lit(&out, ",");
    if (jobs && jobs->type == J_ARR) {
        int rc = do_tokenize(jobs, &out);
        if (rc) return rc;
    } else {
        sb_lit(&out, "\"tokens\":[]");
    }
    sb_lit(&out, "}");

    if (out.failed) { fprintf(stderr, "oom building results\n"); return 2; }
    fwrite(out.s, 1, out.n, stdout);
    free(out.s);
    free(in);
    jv_free(req);
    toks_free();
    free(g_prompt);
    return 0;
}
