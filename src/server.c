// HTTP server: OpenAI-compatible API with N parallel inference slots.
//
//   POST /v1/chat/completions   messages, sampling params, stream (SSE),
//                               response_format {"type":"json_object"}
//   POST /v1/completions        raw prompt completion
//   POST /v1/embeddings         mean-pooled L2-normed embeddings
//   GET  /v1/models             the loaded model
//   GET  /health                liveness
//
// Swap-mode request bodies may carry "keep_alive" (seconds of idle before
// the model unloads; 0 = unload now, negative = keep forever).
//
// Each slot owns a full inference context (KV cache + thread pool); model
// weights are shared between slots through the page cache (mmap).
#include "runner.h"
#include "json.h"
#include "compat.h"

#include <errno.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <stdatomic.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <ws2tcpip.h>
static void sock_init(void) {
    WSADATA w;
    WSAStartup(MAKEWORD(2, 2), &w);
}
static int  sock_recv(int fd, char *buf, size_t n) { return recv(fd, buf, (int)n, 0); }
static int  sock_send(int fd, const char *buf, size_t n) { return send(fd, buf, (int)n, 0); }
static void sock_close(int fd) { closesocket(fd); }
#else
#include <signal.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
static void sock_init(void) { signal(SIGPIPE, SIG_IGN); }
static int  sock_recv(int fd, char *buf, size_t n) { return (int)read(fd, buf, n); }
static int  sock_send(int fd, const char *buf, size_t n) { return (int)write(fd, buf, n); }
static void sock_close(int fd) { close(fd); }
#endif

// case-insensitive prefix compare (strncasecmp is not universal)
static int ci_ncmp(const char *a, const char *b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        int ca = a[i] >= 'A' && a[i] <= 'Z' ? a[i] + 32 : (unsigned char)a[i];
        int cb = b[i] >= 'A' && b[i] <= 'Z' ? b[i] + 32 : (unsigned char)b[i];
        if (ca != cb) return ca - cb;
        if (ca == 0) return 0;
    }
    return 0;
}

// ---------------------------------------------------------------- helpers

typedef struct { char *s; size_t n, cap; } sbuf;

static void sb_put(sbuf *b, const char *s, size_t n) {
    if (b->n + n + 1 > b->cap) {
        b->cap = (b->n + n + 1) * 2 + 256;
        b->s = realloc(b->s, b->cap);
    }
    memcpy(b->s + b->n, s, n);
    b->n += n;
    b->s[b->n] = 0;
}

#define sb_lit(b, lit) sb_put(b, lit, strlen(lit))

#if defined(__GNUC__) || defined(__clang__)
static void sb_fmt(sbuf *b, const char *fmt, ...)
    __attribute__((format(printf, 2, 3)));
#endif
static void sb_fmt(sbuf *b, const char *fmt, ...) {
    char tmp[4096];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(tmp, sizeof(tmp), fmt, ap);
    va_end(ap);
    if (n > 0) sb_put(b, tmp, n < (int)sizeof(tmp) ? (size_t)n : sizeof(tmp) - 1);
}

// escaped variant: appends s as JSON string content
static void sb_esc(sbuf *b, const char *s, size_t n) {
    char *tmp = malloc(n * 6 + 8);
    size_t m = json_escape(s, n, tmp, n * 6 + 8);
    sb_put(b, tmp, m);
    free(tmp);
}

static bool send_all(int fd, const char *s, size_t n) {
    while (n > 0) {
        int w = sock_send(fd, s, n);
        if (w <= 0) return false;
        s += w;
        n -= (size_t)w;
    }
    return true;
}

static void send_response(int fd, int code, const char *ctype, const char *body,
                          size_t blen) {
    char hdr[256];
    const char *msg = code == 200 ? "OK" : code == 400 ? "Bad Request" :
                      code == 404 ? "Not Found" : "Internal Server Error";
    int hn = snprintf(hdr, sizeof(hdr),
                      "HTTP/1.1 %d %s\r\nContent-Type: %s\r\n"
                      "Content-Length: %zu\r\nConnection: close\r\n\r\n",
                      code, msg, ctype, blen);
    if (send_all(fd, hdr, hn)) send_all(fd, body, blen);
}

static void send_error(int fd, int code, const char *message) {
    char body[512], esc[384];
    json_escape(message, strlen(message), esc, sizeof(esc));
    int n = snprintf(body, sizeof(body),
                     "{\"error\":{\"message\":\"%s\",\"type\":\"invalid_request_error\"}}",
                     esc);
    send_response(fd, code, "application/json", body, n);
}

// ---------------------------------------------------------------- slots

typedef struct {
    model_t   *m;         // slot 0 borrows the preloaded model
    tokenizer *tok;       // shared (read-only)
    sampler    smp;
    engine     e;
    int        id;
    int        tmpl;
    pthread_t  th;
} slot_t;

typedef struct {
    int  fds[512];
    int  head, tail, count;
    bool shutdown;
    pthread_mutex_t mu;
    pthread_cond_t cv;
} fdqueue;

// model registry for swap mode (-m "name=path,name2=path2"): one model
// resident at a time (llama-swap semantics), loaded on demand, unloaded
// after --ttl idle seconds
typedef struct {
    char name[64];
    char path[1024];
    int  tmpl;
} reg_entry;

static struct {
    slot_t    *slots;
    int        n_slots;
    fdqueue    q;
    const char *model_name;
    int        n_predict_cap;
    atomic_int req_counter;
    // swap mode
    reg_entry   reg[16];
    int         n_reg;
    bool        single;        // single-model serve wrapped as a 1-entry registry
    bool        borrowed;      // slot 0's model/tok containers owned by the caller
    int         resident;      // index into reg, -1 = none
    double      last_used;
    int         ttl;
    model_params mp;
    pthread_mutex_t swap_mu;
    atomic_bool busy;
} SV;

static void unload_resident(void) {
    if (SV.resident < 0) return;
    slot_t *s = &SV.slots[0];
    fprintf(stderr, "swap: unloading %s\n", SV.reg[SV.resident].name);
    tokenizer_free(s->tok);
    model_free(s->m);
    // single-model serve borrows main()'s stack containers for the first
    // residency; only heap containers from a swap_to() reload are freed
    if (!SV.borrowed) { free(s->tok); free(s->m); }
    SV.borrowed = false;
    s->m = NULL;
    s->tok = NULL;
    SV.resident = -1;
}

// resolve + load the requested model; returns entry index or -1
static int swap_to(const char *want) {
    int idx = 0; // default: first entry
    if (SV.single) want = NULL; // single-model serve answers as any name
    if (want && *want) {
        idx = -1;
        for (int i = 0; i < SV.n_reg; i++)
            if (!strcmp(SV.reg[i].name, want)) { idx = i; break; }
        // OpenAI clients often send a placeholder model name; accept it
        if (idx < 0 && (!strcmp(want, "runner") || !strcmp(want, "default")))
            idx = 0;
        if (idx < 0) return -1;
    }
    pthread_mutex_lock(&SV.swap_mu);
    if (SV.resident != idx) {
        unload_resident();
        slot_t *s = &SV.slots[0];
        fprintf(stderr, "swap: loading %s (%s)\n",
                SV.reg[idx].name, SV.reg[idx].path);
        s->m = malloc(sizeof(model_t));
        s->tok = malloc(sizeof(tokenizer));
        if (!model_load(s->m, SV.reg[idx].path, &SV.mp) ||
            !tokenizer_init(s->tok, &s->m->gf)) {
            fprintf(stderr, "swap: failed to load %s\n", SV.reg[idx].name);
            free(s->m); free(s->tok);
            s->m = NULL; s->tok = NULL;
            pthread_mutex_unlock(&SV.swap_mu);
            return -1;
        }
        s->tmpl = SV.reg[idx].tmpl =
            template_detect(gguf_get_str(&s->m->gf, "tokenizer.chat_template", NULL),
                            s->tok);
        engine_init(&s->e, s->m, s->tok, &s->smp);
        SV.resident = idx;
    }
    SV.last_used = now_s();
    SV.model_name = SV.reg[idx].name;
    pthread_mutex_unlock(&SV.swap_mu);
    return idx;
}

static void *ttl_reaper(void *arg) {
    (void)arg;
    for (;;) {
        struct timespec ts = { 5, 0 };
        nanosleep(&ts, NULL);
        if (SV.ttl <= 0 || SV.resident < 0 || atomic_load(&SV.busy)) continue;
        if (pthread_mutex_trylock(&SV.swap_mu) != 0) continue;
        if (SV.resident >= 0 && !atomic_load(&SV.busy) &&
            now_s() - SV.last_used > SV.ttl)
            unload_resident();
        pthread_mutex_unlock(&SV.swap_mu);
    }
    return NULL;
}

static void q_push(int fd) {
    pthread_mutex_lock(&SV.q.mu);
    if (SV.q.count < (int)(sizeof(SV.q.fds) / sizeof(int))) {
        SV.q.fds[SV.q.tail] = fd;
        SV.q.tail = (SV.q.tail + 1) % (int)(sizeof(SV.q.fds) / sizeof(int));
        SV.q.count++;
        pthread_cond_signal(&SV.q.cv);
    } else {
        sock_close(fd); // overloaded
    }
    pthread_mutex_unlock(&SV.q.mu);
}

static int q_pop(void) {
    pthread_mutex_lock(&SV.q.mu);
    while (SV.q.count == 0 && !SV.q.shutdown)
        pthread_cond_wait(&SV.q.cv, &SV.q.mu);
    int fd = -1;
    if (SV.q.count > 0) {
        fd = SV.q.fds[SV.q.head];
        SV.q.head = (SV.q.head + 1) % (int)(sizeof(SV.q.fds) / sizeof(int));
        SV.q.count--;
    }
    pthread_mutex_unlock(&SV.q.mu);
    return fd;
}

// ---------------------------------------------------------------- generation

typedef struct {
    sbuf  out;          // accumulated completion text
    sbuf  reason;       // accumulated reasoning text (thinking-tag models)
    int   fd;
    bool  stream;       // SSE mode
    bool  dead;         // client went away
    char  id[48];
    bool  chat;         // chat.completion vs text_completion chunk shape
    think_split ts;     // thinking-tag splitter (pass-through when untagged)
} gen_ctx;

// emit one section of split output: reasoning goes to the OpenAI-style
// reasoning_content field, everything else to content
static int gen_emit(void *ud, int reasoning, const char *bytes, int n) {
    gen_ctx *g = ud;
    sb_put(reasoning ? &g->reason : &g->out, bytes, n);
    if (!g->stream || g->dead) return g->dead ? 1 : 0;
    sbuf chunk = {0};
    sb_fmt(&chunk, "{\"id\":\"%s\",\"object\":\"%s\",\"choices\":[{\"index\":0,",
           g->id, g->chat ? "chat.completion.chunk" : "text_completion");
    if (g->chat) sb_fmt(&chunk, "\"delta\":{\"%s\":\"",
                        reasoning ? "reasoning_content" : "content");
    else         sb_lit(&chunk, "\"text\":\"");
    sb_esc(&chunk, bytes, n);
    if (g->chat) sb_lit(&chunk, "\"},\"finish_reason\":null}]}");
    else         sb_lit(&chunk, "\",\"finish_reason\":null}]}");
    sbuf sse = {0};
    sb_fmt(&sse, "data: %s\n\n", chunk.s);
    if (!send_all(g->fd, sse.s, sse.n)) g->dead = true;
    free(chunk.s);
    free(sse.s);
    return g->dead ? 1 : 0;
}

static int gen_collect(void *ud, const char *bytes, int n) {
    gen_ctx *g = ud;
    return think_feed(&g->ts, bytes, n, gen_emit, g);
}

// parse gemma4 tool-call blocks — <|tool_call>call:NAME{json}<tool_call|> —
// out of the content, appending OpenAI tool_calls items to tc. Returns the
// number of calls; content is compacted in place.
static int parse_tool_calls(sbuf *content, sbuf *tc) {
    if (!content->s) return 0;
    static const char OPEN[] = "<|tool_call>call:";
    static const char CLOSE[] = "<tool_call|>";
    int n_calls = 0;
    char *w = content->s; // write cursor for the compacted content
    const char *p = content->s, *end = content->s + content->n;
    while (p < end) {
        const char *o = strstr(p, OPEN);
        if (!o) {
            memmove(w, p, end - p);
            w += end - p;
            break;
        }
        memmove(w, p, o - p);
        w += o - p;
        const char *name = o + sizeof(OPEN) - 1;
        const char *brace = name;
        while (brace < end && *brace != '{' && *brace != '<' && brace - name < 128)
            brace++;
        if (brace >= end || *brace != '{') { p = name; continue; } // not a call
        // brace-match the args object (string- and escape-aware)
        const char *q = brace;
        int depth = 0;
        bool in_str = false;
        for (; q < end; q++) {
            if (in_str) {
                if (*q == '\\') q++;
                else if (*q == '"') in_str = false;
            } else if (*q == '"') in_str = true;
            else if (*q == '{') depth++;
            else if (*q == '}' && --depth == 0) { q++; break; }
        }
        if (depth != 0) { p = name; continue; } // truncated: leave as content
        sb_fmt(tc, "%s{\"id\":\"call_%d\",\"type\":\"function\",\"function\":"
                   "{\"name\":\"", n_calls ? "," : "", n_calls);
        sb_esc(tc, name, (int)(brace - name));
        sb_lit(tc, "\",\"arguments\":\"");
        sb_esc(tc, brace, (int)(q - brace));
        sb_lit(tc, "\"}}");
        n_calls++;
        p = q;
        if (end - p >= (int)sizeof(CLOSE) - 1 &&
            memcmp(p, CLOSE, sizeof(CLOSE) - 1) == 0)
            p += sizeof(CLOSE) - 1;
    }
    if (n_calls) content->n = (int)(w - content->s);
    return n_calls;
}

// run one completion on a slot and write the HTTP response
static void run_completion(slot_t *s, int fd, const char *prompt, bool chat,
                           jv *req) {
    model_t *m = s->m;
    engine *e = &s->e;

    // per-request sampling params
    s->smp.temp = (float)jv_num(jv_get(req, "temperature"), s->smp.temp);
    s->smp.top_p = (float)jv_num(jv_get(req, "top_p"), s->smp.top_p);
    s->smp.min_p = (float)jv_num(jv_get(req, "min_p"), s->smp.min_p);
    s->smp.top_k = (int)jv_num(jv_get(req, "top_k"), s->smp.top_k);
    double seed = jv_num(jv_get(req, "seed"), 0);
    if (seed > 0) s->smp.rng = (uint64_t)seed;
    int max_tokens = (int)jv_num(jv_get(req, "max_tokens"),
                     jv_num(jv_get(req, "max_completion_tokens"), SV.n_predict_cap));
    bool stream = jv_bool(jv_get(req, "stream"), false);
    // OpenAI logprobs (chat, buffered responses only)
    bool want_lp = chat && !stream && jv_bool(jv_get(req, "logprobs"), false);
    int lp_n = want_lp ? (int)jv_num(jv_get(req, "top_logprobs"), 0) : 0;
    if (lp_n < 0) lp_n = 0;
    if (lp_n > 20) lp_n = 20;
    jv *rf = jv_get(req, "response_format");
    e->json_mode = rf && strcmp(jv_str(jv_get(rf, "type"), ""), "json_object") == 0;
    // schema-constrained decoding: OpenAI response_format {type:"json_schema",
    // json_schema:{schema:{...}}} or an Ollama-style "format" schema object
    snode *schema = NULL;
    jv *sch = NULL;
    if (rf && strcmp(jv_str(jv_get(rf, "type"), ""), "json_schema") == 0) {
        jv *js = jv_get(rf, "json_schema");
        sch = js ? (jv_get(js, "schema") ? jv_get(js, "schema") : js) : NULL;
    }
    jv *fmt = jv_get(req, "format");
    if (!sch && fmt && fmt->type == J_OBJ) sch = fmt;
    if (sch) {
        char serr[128];
        schema = schema_compile(sch, serr, sizeof(serr));
        if (!schema) {
            char msg[192];
            snprintf(msg, sizeof(msg), "unsupported json schema: %s", serr);
            send_error(fd, 400, msg);
            return;
        }
    }
    e->schema = schema;

    size_t cap = strlen(prompt) + 16;
    int32_t *toks = malloc(sizeof(int32_t) * cap);
    int n_prompt = tok_encode(s->tok, prompt, toks, (int)cap, true, true);
    if (n_prompt == 0) { free(toks); send_error(fd, 400, "empty prompt"); return; }
    if (n_prompt >= m->n_ctx) {
        free(toks);
        send_error(fd, 400, "prompt exceeds context window");
        return;
    }

    if (want_lp && max_tokens > 0) {
        e->lp_cap    = max_tokens;
        e->lp_n      = lp_n;
        e->lp_chosen = malloc(sizeof(float) * max_tokens);
        e->lp_ids    = malloc(sizeof(int32_t) * max_tokens);
        e->lp_top    = lp_n ? malloc(sizeof(lp_alt) * (size_t)max_tokens * lp_n) : NULL;
    }

    // reuse the KV for whatever prefix this prompt shares with the last one —
    // pipeline callers repeat long system/template prefixes verbatim
    int keep = engine_rewind(e, toks, n_prompt);
    float *logits = engine_feed(e, toks + keep, n_prompt - keep);
    free(toks);
    if (!logits) {
        e->schema = NULL;
        schema_free(schema);
        free(e->lp_chosen); free(e->lp_ids); free(e->lp_top);
        e->lp_chosen = NULL; e->lp_ids = NULL; e->lp_top = NULL;
        e->lp_cap = e->lp_n = e->lp_count = 0;
        send_error(fd, 500, "context overflow");
        return;
    }

    gen_ctx g = { .out = {0}, .fd = fd, .stream = stream, .chat = chat };
    snprintf(g.id, sizeof(g.id), "%s-%d", chat ? "chatcmpl" : "cmpl",
             atomic_fetch_add(&SV.req_counter, 1));
    // split thinking channels out of chat responses; raw completions stay raw
    think_init(&g.ts, chat ? m->think_open : NULL, m->think_close);

    if (stream) {
        const char *hdr = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                          "Cache-Control: no-cache\r\nConnection: close\r\n\r\n";
        if (!send_all(fd, hdr, strlen(hdr))) return;
    }

    double gtime;
    int n_gen = engine_generate(e, logits, max_tokens, gen_collect, &g, &gtime);
    think_finish(&g.ts, gen_emit, &g);
    const char *finish = e->hit_stop ? "stop" : "length";

    if (stream) {
        if (!g.dead) {
            char fin[256];
            int fn = snprintf(fin, sizeof(fin),
                "data: {\"id\":\"%s\",\"object\":\"%s\",\"choices\":[{\"index\":0,"
                "%s,\"finish_reason\":\"%s\"}]}\n\ndata: [DONE]\n\n",
                g.id, chat ? "chat.completion.chunk" : "text_completion",
                chat ? "\"delta\":{}" : "\"text\":\"\"", finish);
            send_all(fd, fin, fn);
        }
    } else {
        sbuf tc = {0};
        int n_tc = chat ? parse_tool_calls(&g.out, &tc) : 0;
        if (n_tc) finish = "tool_calls";
        sbuf r = {0};
        sb_fmt(&r, "{\"id\":\"%s\",\"object\":\"%s\",\"created\":%ld,\"model\":\"", g.id,
               chat ? "chat.completion" : "text_completion",
               (long)time(NULL));
        sb_esc(&r, SV.model_name, strlen(SV.model_name));
        sb_lit(&r, "\",\"choices\":[{\"index\":0,");
        if (chat) sb_lit(&r, "\"message\":{\"role\":\"assistant\",\"content\":\"");
        else      sb_lit(&r, "\"text\":\"");
        sb_esc(&r, g.out.s ? g.out.s : "", g.out.n);
        sb_lit(&r, "\"");
        if (n_tc) {
            sb_lit(&r, ",\"tool_calls\":[");
            sb_put(&r, tc.s, tc.n);
            sb_lit(&r, "]");
        }
        free(tc.s);
        if (chat && g.reason.n > 0) {
            sb_lit(&r, ",\"reasoning_content\":\"");
            sb_esc(&r, g.reason.s, g.reason.n);
            sb_lit(&r, "\"");
        }
        if (chat) sb_lit(&r, "},");
        else      sb_lit(&r, ",");
        if (chat && e->lp_count > 0) {
            char tb[512];
            sb_lit(&r, "\"logprobs\":{\"content\":[");
            for (int i = 0; i < e->lp_count; i++) {
                if (i) sb_lit(&r, ",");
                int tn = tok_decode(s->tok, e->lp_ids[i], tb, sizeof(tb));
                sb_lit(&r, "{\"token\":\"");
                sb_esc(&r, tb, tn);
                sb_fmt(&r, "\",\"logprob\":%.6f,\"top_logprobs\":[", e->lp_chosen[i]);
                for (int j = 0; j < e->lp_n; j++) {
                    const lp_alt *a = &e->lp_top[(size_t)i * e->lp_n + j];
                    if (a->id < 0) break;
                    if (j) sb_lit(&r, ",");
                    tn = tok_decode(s->tok, a->id, tb, sizeof(tb));
                    sb_lit(&r, "{\"token\":\"");
                    sb_esc(&r, tb, tn);
                    sb_fmt(&r, "\",\"logprob\":%.6f}", a->lp);
                }
                sb_lit(&r, "]}");
            }
            sb_lit(&r, "]},");
        }
        sb_fmt(&r, "\"finish_reason\":\"%s\"}],"
                   "\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,"
                   "\"total_tokens\":%d}}",
               finish, n_prompt, n_gen, n_prompt + n_gen);
        send_response(fd, 200, "application/json", r.s, r.n);
        free(r.s);
    }
    fprintf(stderr, "[slot %d] %s: %d prompt (%d cached) + %d gen tok (%.1f tok/s)%s%s\n",
            s->id, g.id, n_prompt, keep, n_gen,
            n_gen / (gtime > 0 ? gtime : 1e-9),
            schema ? " [schema]" : e->json_mode ? " [json]" : "",
            g.dead ? " [client gone]" : "");
    e->schema = NULL;
    schema_free(schema);
    think_free(&g.ts);
    free(g.reason.s);
    free(g.out.s);
    free(e->lp_chosen); free(e->lp_ids); free(e->lp_top);
    e->lp_chosen = NULL; e->lp_ids = NULL; e->lp_top = NULL;
    e->lp_cap = e->lp_n = e->lp_count = 0;
}

// ---------------------------------------------------------------- routes

static void handle_chat(slot_t *s, int fd, jv *req) {
    jv *msgs = jv_get(req, "messages");
    if (!msgs || msgs->type != J_ARR || msgs->n == 0) {
        send_error(fd, 400, "missing messages");
        return;
    }
    chat_msg *cm = malloc(sizeof(chat_msg) * msgs->n);
    size_t total = 0;
    int n_cm = 0;
    for (int i = 0; i < msgs->n; i++) {
        const char *role = jv_str(jv_get(msgs->items[i], "role"), "user");
        const char *content = jv_str(jv_get(msgs->items[i], "content"), NULL);
        if (!content) continue;
        cm[n_cm++] = (chat_msg){ role, content };
        total += strlen(role) + strlen(content) + 64;
    }
    if (n_cm == 0) { free(cm); send_error(fd, 400, "no message content"); return; }
    char *prompt = malloc(total + 256);
    render_messages(s->tmpl, cm, n_cm, true, prompt, total + 256);
    run_completion(s, fd, prompt, true, req);
    free(prompt);
    free(cm);
}

static void handle_completion(slot_t *s, int fd, jv *req) {
    const char *prompt = jv_str(jv_get(req, "prompt"), NULL);
    if (!prompt) { send_error(fd, 400, "missing prompt"); return; }
    run_completion(s, fd, prompt, false, req);
}

static void handle_embeddings(slot_t *s, int fd, jv *req) {
    jv *input = jv_get(req, "input");
    const char *one = jv_str(input, NULL);
    int n_in = one ? 1 : (input && input->type == J_ARR ? input->n : 0);
    if (n_in == 0) { send_error(fd, 400, "missing input"); return; }

    model_t *m = s->m;
    float *emb = malloc(sizeof(float) * m->n_embd);
    sbuf r = {0};
    sb_lit(&r, "{\"object\":\"list\",\"data\":[");
    int total = 0;
    bool ok = true;
    for (int k = 0; k < n_in && ok; k++) {
        const char *txt = one ? one : jv_str(input->items[k], NULL);
        if (!txt) { send_error(fd, 400, "input must be strings"); ok = false; break; }
        size_t cap = strlen(txt) + 16;
        int32_t *toks = malloc(sizeof(int32_t) * cap);
        int n = tok_encode(s->tok, txt, toks, (int)cap, true, true);
        if (n == 0 || !model_embed(m, toks, n, emb)) {
            free(toks);
            send_error(fd, 400, n == 0 ? "empty input" : "input exceeds context window");
            ok = false;
            break;
        }
        free(toks);
        total += n;
        sb_fmt(&r, "%s{\"object\":\"embedding\",\"index\":%d,\"embedding\":[",
               k ? "," : "", k);
        for (int j = 0; j < m->n_embd; j++)
            sb_fmt(&r, "%s%.7g", j ? "," : "", emb[j]);
        sb_lit(&r, "]}");
    }
    // model_embed overwrote this slot's KV cache — invalidate the prefix cache
    engine_reset(&s->e);
    s->e.pos = 0;
    if (ok) {
        sb_lit(&r, "],\"model\":\"");
        sb_esc(&r, SV.model_name, strlen(SV.model_name));
        sb_fmt(&r, "\",\"usage\":{\"prompt_tokens\":%d,\"total_tokens\":%d}}",
               total, total);
        send_response(fd, 200, "application/json", r.s, r.n);
        fprintf(stderr, "[slot %d] embeddings: %d input(s), %d tok\n",
                s->id, n_in, total);
    }
    free(r.s);
    free(emb);
}

// ---------------------------------------------------------------- http

static void handle_conn(slot_t *s, int fd) {
    char hdr[16384];
    size_t got = 0;
    char *body_start = NULL;
    while (got < sizeof(hdr) - 1) {
        int r = sock_recv(fd, hdr + got, sizeof(hdr) - 1 - got);
        if (r <= 0) return;
        got += (size_t)r;
        hdr[got] = 0;
        if ((body_start = strstr(hdr, "\r\n\r\n")) != NULL) break;
    }
    if (!body_start) { send_error(fd, 400, "bad request"); return; }
    body_start += 4;

    char method[8] = {0}, path[256] = {0};
    sscanf(hdr, "%7s %255s", method, path);

    size_t content_length = 0;
    for (char *p = hdr; p < body_start; p++) {
        if (ci_ncmp(p, "Content-Length:", 15) == 0) {
            content_length = (size_t)strtoull(p + 15, NULL, 10);
            break;
        }
    }
    if (content_length > 32u * 1024 * 1024) {
        send_error(fd, 400, "body too large");
        return;
    }

    char *body = NULL;
    if (content_length > 0) {
        body = malloc(content_length + 1);
        size_t have = got - (size_t)(body_start - hdr);
        if (have > content_length) have = content_length;
        memcpy(body, body_start, have);
        while (have < content_length) {
            int r = sock_recv(fd, body + have, content_length - have);
            if (r <= 0) { free(body); return; }
            have += (size_t)r;
        }
        body[content_length] = 0;
    }

    if (!strcmp(method, "GET") && !strcmp(path, "/unload")) {
        // llama-swap-compatible: free the resident model's memory now
        if (SV.n_reg > 0) {
            pthread_mutex_lock(&SV.swap_mu);
            unload_resident();
            pthread_mutex_unlock(&SV.swap_mu);
        }
        const char *b = "{\"status\":\"ok\"}";
        send_response(fd, 200, "application/json", b, strlen(b));
    } else if (!strcmp(method, "GET") && !strcmp(path, "/health")) {
        char b[256];
        int n;
        if (SV.n_reg > 0)
            n = snprintf(b, sizeof(b), "{\"status\":\"ok\",\"resident\":%s%s%s}",
                         SV.resident >= 0 ? "\"" : "",
                         SV.resident >= 0 ? SV.reg[SV.resident].name : "null",
                         SV.resident >= 0 ? "\"" : "");
        else
            n = snprintf(b, sizeof(b), "{\"status\":\"ok\"}");
        send_response(fd, 200, "application/json", b, n);
    } else if (!strcmp(method, "GET") && !strcmp(path, "/v1/models")) {
        sbuf r = {0};
        sb_lit(&r, "{\"object\":\"list\",\"data\":[");
        if (SV.n_reg > 0) {
            for (int i = 0; i < SV.n_reg; i++)
                sb_fmt(&r, "%s{\"id\":\"%s\",\"object\":\"model\","
                           "\"owned_by\":\"runner\"}", i ? "," : "", SV.reg[i].name);
        } else {
            char esc[256];
            json_escape(SV.model_name, strlen(SV.model_name), esc, sizeof(esc));
            sb_fmt(&r, "{\"id\":\"%s\",\"object\":\"model\",\"owned_by\":\"runner\"}", esc);
        }
        sb_lit(&r, "]}");
        send_response(fd, 200, "application/json", r.s, r.n);
        free(r.s);
    } else if (!strcmp(method, "POST") &&
               (!strcmp(path, "/v1/chat/completions") ||
                !strcmp(path, "/v1/completions") ||
                !strcmp(path, "/v1/embeddings"))) {
        jv *req = body ? json_parse(body, content_length) : NULL;
        if (!req) {
            send_error(fd, 400, "invalid JSON body");
        } else {
            atomic_store(&SV.busy, true);
            bool ok = true;
            if (SV.n_reg > 0 &&
                swap_to(jv_str(jv_get(req, "model"), NULL)) < 0) {
                send_error(fd, 400, "unknown model (see /v1/models)");
                ok = false;
            }
            if (ok) {
                if (strcmp(path, "/v1/chat/completions") == 0) handle_chat(s, fd, req);
                else if (strcmp(path, "/v1/embeddings") == 0) handle_embeddings(s, fd, req);
                else handle_completion(s, fd, req);
                // Ollama-style keep_alive: seconds of idle before the model
                // unloads (swap mode) — 0 unloads now, negative pins forever
                jv *ka = jv_get(req, "keep_alive");
                if (ka && SV.n_reg > 0) {
                    double v = jv_num(ka, (double)SV.ttl);
                    pthread_mutex_lock(&SV.swap_mu);
                    if (v == 0) unload_resident();
                    else SV.ttl = v < 0 ? 0 : (int)v;
                    pthread_mutex_unlock(&SV.swap_mu);
                }
            }
            atomic_store(&SV.busy, false);
            jv_free(req);
        }
    } else {
        send_error(fd, 404, "not found");
    }
    free(body);
}

static void *slot_worker(void *arg) {
    slot_t *s = arg;
    for (;;) {
        int fd = q_pop();
        if (fd < 0) return NULL;
        handle_conn(s, fd);
        sock_close(fd);
    }
}

// ---------------------------------------------------------------- entry

int server_run(model_t *base, tokenizer *tok, const char *model_path,
               const model_params *mp, sampler defaults, int port, int parallel,
               int n_threads, int ttl) {
    sock_init();
    if (parallel < 1) parallel = 1;
    if (parallel > 16) parallel = 16;
    bool swap_mode = strchr(model_path, '=') != NULL;
    if (ttl < 0) ttl = swap_mode ? 300 : 0; // single-model default: never unload

    // "name=path,name2=path2" enables swap mode: one resident model,
    // loaded per request's "model" field, unloaded after ttl idle seconds
    if (strchr(model_path, '=')) {
        char tmp[4096];
        snprintf(tmp, sizeof(tmp), "%s", model_path);
        for (char *tk = strtok(tmp, ","); tk && SV.n_reg < 16;
             tk = strtok(NULL, ",")) {
            char *eq = strchr(tk, '=');
            if (!eq) { fprintf(stderr, "error: bad registry entry '%s'\n", tk); return 1; }
            *eq = 0;
            snprintf(SV.reg[SV.n_reg].name, sizeof(SV.reg[0].name), "%s", tk);
            snprintf(SV.reg[SV.n_reg].path, sizeof(SV.reg[0].path), "%s", eq + 1);
            if (!plat_file_readable(SV.reg[SV.n_reg].path)) {
                fprintf(stderr, "error: cannot read %s\n", SV.reg[SV.n_reg].path);
                return 1;
            }
            SV.n_reg++;
        }
        if (parallel > 1) {
            fprintf(stderr, "note: model swapping uses a single inference slot; "
                    "ignoring --parallel %d\n", parallel);
        }
        parallel = SV.n_reg > 0 ? 1 : parallel;
        SV.resident = -1;
        SV.ttl = ttl;
        SV.mp = *mp;
        SV.mp.verbose = false;
        SV.mp.n_threads = n_threads;
        pthread_mutex_init(&SV.swap_mu, NULL);
    }

    int threads_per_slot = n_threads / parallel;
    if (threads_per_slot < 1) threads_per_slot = 1;
    bool single_setup = false;

    const char *name = strrchr(model_path, '/');
    const char *bsname = strrchr(model_path, '\\'); // Windows path separator
    if (bsname && (!name || bsname > name)) name = bsname;
    SV.model_name = SV.n_reg > 0 ? SV.reg[0].name : (name ? name + 1 : model_path);
    SV.n_predict_cap = 1024;
    SV.n_slots = parallel;
    SV.slots = calloc(parallel, sizeof(slot_t));
    pthread_mutex_init(&SV.q.mu, NULL);
    pthread_cond_init(&SV.q.cv, NULL);

    if (SV.n_reg > 0) {
        // swap mode: models are loaded on demand
        slot_t *s = &SV.slots[0];
        s->id = 0;
        s->smp = defaults;
        pthread_t rt;
        pthread_create(&rt, NULL, ttl_reaper, NULL);
    } else {
        single_setup = parallel == 1;
        int tmpl = template_detect(gguf_get_str(&base->gf, "tokenizer.chat_template", NULL),
                                   tok);
        model_params slot_mp = *mp;
        slot_mp.verbose = false;
        slot_mp.n_threads = threads_per_slot;

        for (int i = 0; i < parallel; i++) {
            slot_t *s = &SV.slots[i];
            s->id = i;
            s->tok = tok;
            s->tmpl = tmpl;
            s->smp = defaults;
            s->smp.rng = defaults.rng ^ (0x9E3779B97F4A7C15ull * (unsigned)(i + 1));
            if (i == 0) {
                s->m = base;
                base->tp = tpool_create(threads_per_slot);
            } else {
                s->m = malloc(sizeof(model_t));
                if (!model_load(s->m, model_path, &slot_mp)) {
                    fprintf(stderr, "error: failed to load slot %d\n", i);
                    return 1;
                }
            }
            engine_init(&s->e, s->m, s->tok, &s->smp);
        }

        if (single_setup) {
            // join the registry machinery so GET /unload frees the resident
            // model (the next request lazily reloads it) and --ttl works.
            // slot 0's containers are the caller's; borrowed avoids freeing
            // them on the first unload
            snprintf(SV.reg[0].name, sizeof(SV.reg[0].name), "%s", SV.model_name);
            snprintf(SV.reg[0].path, sizeof(SV.reg[0].path), "%s", model_path);
            SV.reg[0].tmpl = tmpl;
            SV.model_name = SV.reg[0].name;
            SV.n_reg = 1;
            SV.single = true;
            SV.borrowed = true;
            SV.resident = 0;
            SV.last_used = now_s();
            SV.ttl = ttl;
            SV.mp = *mp;
            SV.mp.verbose = false;
            SV.mp.n_threads = threads_per_slot;
            pthread_mutex_init(&SV.swap_mu, NULL);
            pthread_t rt;
            pthread_create(&rt, NULL, ttl_reaper, NULL);
        }
    }

    int lfd = (int)socket(AF_INET, SOCK_STREAM, 0);
    int one = 1;
    setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, (const char *)&one, sizeof(one));
    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons((uint16_t)port);
    if (bind(lfd, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
        fprintf(stderr, "error: cannot bind 127.0.0.1:%d (%s)\n", port, strerror(errno));
        return 1;
    }
    listen(lfd, 64);

    for (int i = 0; i < parallel; i++)
        pthread_create(&SV.slots[i].th, NULL, slot_worker, &SV.slots[i]);

    if (SV.n_reg > 0 && !SV.single)
        fprintf(stderr,
                "server listening on http://127.0.0.1:%d — %d models, swap on demand"
                " (ttl %ds)\n"
                "  POST /v1/chat/completions | POST /v1/completions | GET /v1/models | GET /health\n",
                port, SV.n_reg, SV.ttl);
    else
        fprintf(stderr,
                "server listening on http://127.0.0.1:%d — %d slot%s x %d threads\n"
                "  POST /v1/chat/completions | POST /v1/completions | GET /v1/models | GET /health\n",
                port, parallel, parallel > 1 ? "s" : "", threads_per_slot);

    for (;;) {
        int cfd = (int)accept(lfd, NULL, NULL);
        if (cfd < 0) {
            if (errno == EINTR) continue;
            break;
        }
        q_push(cfd);
    }
    return 0;
}
