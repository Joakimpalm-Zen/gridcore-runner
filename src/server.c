// HTTP server: OpenAI-compatible API with N parallel inference slots.
//
//   POST /v1/chat/completions   messages, sampling params, stream (SSE),
//                               response_format {"type":"json_object"}
//   POST /v1/completions        raw prompt completion
//   GET  /v1/models             the loaded model
//   GET  /health                liveness
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
    free(s->tok);
    free(s->m);
    s->m = NULL;
    s->tok = NULL;
    SV.resident = -1;
}

// resolve + load the requested model; returns entry index or -1
static int swap_to(const char *want) {
    int idx = 0; // default: first entry
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
    int   fd;
    bool  stream;       // SSE mode
    bool  dead;         // client went away
    char  id[48];
    bool  chat;         // chat.completion vs text_completion chunk shape
} gen_ctx;

static int gen_collect(void *ud, const char *bytes, int n) {
    gen_ctx *g = ud;
    sb_put(&g->out, bytes, n);
    if (!g->stream || g->dead) return g->dead ? 1 : 0;
    sbuf chunk = {0};
    sb_fmt(&chunk, "{\"id\":\"%s\",\"object\":\"%s\",\"choices\":[{\"index\":0,",
           g->id, g->chat ? "chat.completion.chunk" : "text_completion");
    if (g->chat) sb_lit(&chunk, "\"delta\":{\"content\":\"");
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

// run one completion on a slot and write the HTTP response
static void run_completion(slot_t *s, int fd, const char *prompt, bool chat,
                           jv *req) {
    model_t *m = s->m;
    engine *e = &s->e;

    // per-request sampling params
    s->smp.temp = (float)jv_num(jv_get(req, "temperature"), s->smp.temp);
    s->smp.top_p = (float)jv_num(jv_get(req, "top_p"), s->smp.top_p);
    s->smp.top_k = (int)jv_num(jv_get(req, "top_k"), s->smp.top_k);
    double seed = jv_num(jv_get(req, "seed"), 0);
    if (seed > 0) s->smp.rng = (uint64_t)seed;
    int max_tokens = (int)jv_num(jv_get(req, "max_tokens"),
                     jv_num(jv_get(req, "max_completion_tokens"), SV.n_predict_cap));
    bool stream = jv_bool(jv_get(req, "stream"), false);
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

    engine_reset(e);
    float *logits = engine_feed(e, toks, n_prompt);
    free(toks);
    if (!logits) {
        e->schema = NULL;
        schema_free(schema);
        send_error(fd, 500, "context overflow");
        return;
    }

    gen_ctx g = { .out = {0}, .fd = fd, .stream = stream, .chat = chat };
    snprintf(g.id, sizeof(g.id), "%s-%d", chat ? "chatcmpl" : "cmpl",
             atomic_fetch_add(&SV.req_counter, 1));

    if (stream) {
        const char *hdr = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                          "Cache-Control: no-cache\r\nConnection: close\r\n\r\n";
        if (!send_all(fd, hdr, strlen(hdr))) return;
    }

    double gtime;
    int n_gen = engine_generate(e, logits, max_tokens, gen_collect, &g, &gtime);
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
        sbuf r = {0};
        sb_fmt(&r, "{\"id\":\"%s\",\"object\":\"%s\",\"created\":%ld,\"model\":\"", g.id,
               chat ? "chat.completion" : "text_completion",
               (long)time(NULL));
        sb_esc(&r, SV.model_name, strlen(SV.model_name));
        sb_lit(&r, "\",\"choices\":[{\"index\":0,");
        if (chat) sb_lit(&r, "\"message\":{\"role\":\"assistant\",\"content\":\"");
        else      sb_lit(&r, "\"text\":\"");
        sb_esc(&r, g.out.s ? g.out.s : "", g.out.n);
        if (chat) sb_lit(&r, "\"},");
        else      sb_lit(&r, "\",");
        sb_fmt(&r, "\"finish_reason\":\"%s\"}],"
                   "\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,"
                   "\"total_tokens\":%d}}",
               finish, n_prompt, n_gen, n_prompt + n_gen);
        send_response(fd, 200, "application/json", r.s, r.n);
        free(r.s);
    }
    fprintf(stderr, "[slot %d] %s: %d prompt + %d gen tok (%.1f tok/s)%s%s\n",
            s->id, g.id, n_prompt, n_gen,
            n_gen / (gtime > 0 ? gtime : 1e-9),
            schema ? " [schema]" : e->json_mode ? " [json]" : "",
            g.dead ? " [client gone]" : "");
    e->schema = NULL;
    schema_free(schema);
    free(g.out.s);
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
                !strcmp(path, "/v1/completions"))) {
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
                else handle_completion(s, fd, req);
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

    if (SV.n_reg > 0)
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
