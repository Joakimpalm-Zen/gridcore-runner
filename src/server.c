// HTTP server: OpenAI-compatible API with N parallel inference slots.
//
//   POST /v1/chat/completions   messages, sampling params, stream (SSE),
//                               response_format {"type":"json_object"}
//   POST /v1/completions        raw prompt completion
//   POST /v1/embeddings         mean-pooled L2-normed embeddings
//   GET  /v1/models             the loaded model
//   GET  /v1/capabilities       registry + feature discovery
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
#include <limits.h>
#include <float.h>
#include <math.h>
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
static int  sock_peek(int fd, char *buf, size_t n) { return recv(fd, buf, (int)n, MSG_PEEK); }
static void sock_close(int fd) { closesocket(fd); }
static void sock_recv_timeout(int fd, double s) {
    DWORD ms = (DWORD)(s * 1000.0);
    if (ms == 0) ms = 1; // 0 would mean "block forever" on winsock
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, (const char *)&ms, sizeof(ms));
}
static void sock_send_timeout(int fd, double s) {
    DWORD ms = (DWORD)(s * 1000.0);
    if (ms == 0) ms = 1;
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, (const char *)&ms, sizeof(ms));
}
#else
#include <signal.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/select.h>
#include <netinet/in.h>
#include <sys/socket.h>
static void sock_init(void) { signal(SIGPIPE, SIG_IGN); }
static int  sock_recv(int fd, char *buf, size_t n) { return (int)read(fd, buf, n); }
static int  sock_send(int fd, const char *buf, size_t n) { return (int)write(fd, buf, n); }
static int  sock_peek(int fd, char *buf, size_t n) { return (int)recv(fd, buf, n, MSG_PEEK); }
static void sock_close(int fd) { close(fd); }
static void sock_recv_timeout(int fd, double s) {
    struct timeval tv;
    tv.tv_sec = (time_t)s;
    tv.tv_usec = (suseconds_t)((s - (double)tv.tv_sec) * 1e6);
    if (tv.tv_sec == 0 && tv.tv_usec == 0) tv.tv_usec = 1000;
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
}
static void sock_send_timeout(int fd, double s) {
    struct timeval tv;
    tv.tv_sec = (time_t)s;
    tv.tv_usec = (suseconds_t)((s - (double)tv.tv_sec) * 1e6);
    if (tv.tv_sec == 0 && tv.tv_usec == 0) tv.tv_usec = 1000;
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
}
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
                      code == 404 ? "Not Found" : code == 408 ? "Request Timeout" :
                      "Internal Server Error";
    int hn = snprintf(hdr, sizeof(hdr),
                      "HTTP/1.1 %d %s\r\nContent-Type: %s\r\n"
                      "Content-Length: %zu\r\nConnection: close\r\n\r\n",
                      code, msg, ctype, blen);
    if (send_all(fd, hdr, hn)) send_all(fd, body, blen);
}

// Send a built JSON body, or 500 if the builder ran out of memory. A short
// body that still parses as success is worse than an error: the client cannot
// tell anything went wrong.
static void send_error(int fd, int code, const char *message);

static void send_built(int fd, sbuf *b) {
    if (b->failed) send_error(fd, 500, "out of memory building response");
    else send_response(fd, 200, "application/json", b->s, b->n);
}

static void send_error(int fd, int code, const char *message) {
    char body[512], esc[384];
    json_escape(message, strlen(message), esc, sizeof(esc));
    int n = snprintf(body, sizeof(body),
                     "{\"error\":{\"message\":\"%s\",\"type\":\"%s\"}}",
                     esc, code >= 500 ? "server_error" : "invalid_request_error");
    send_response(fd, code, "application/json", body, n);
}

// ---------------------------------------------------------------- slots

typedef struct {
    model_t   *m;         // slot 0 borrows the preloaded model
    tokenizer *tok;       // shared (read-only)
    sampler    smp;
    sampler    smp_base;  // pristine server defaults (per-request resets)
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
    atomic_int ctx_size;
    atomic_int req_counter;
    // swap mode
    reg_entry   reg[16];
    int         n_reg;
    bool        single;        // single-model serve wrapped as a 1-entry registry
    bool        borrowed;      // slot 0's model/tok containers owned by the caller
    atomic_int  resident;      // index into reg, -1 = none
    double      last_used;
    int         ttl;
    model_params mp;
    pthread_mutex_t swap_mu;
    atomic_bool busy;
    model_t    *draft;        // per-slot draft would be plural; single-model
    int         draft_k;      // serve has exactly one slot (see swap_to)
    // sampling defaults come from the served model's family preset; the CLI
    // overrides are kept so a swapped-in model can be re-resolved against them
    sampler_override ov;
    const char *preset_name;
} SV;

static int resident_load(void) {
    return atomic_load_explicit(&SV.resident, memory_order_relaxed);
}

static void resident_store(int v) {
    atomic_store_explicit(&SV.resident, v, memory_order_relaxed);
}

static int context_load(void) {
    return atomic_load_explicit(&SV.ctx_size, memory_order_relaxed);
}

static void context_store(int value) {
    atomic_store_explicit(&SV.ctx_size, value, memory_order_relaxed);
}

static void unload_resident(void) {
    int res = resident_load();
    if (res < 0) return;
    slot_t *s = &SV.slots[0];
    fprintf(stderr, "swap: unloading %s\n", SV.reg[res].name);
    tokenizer_free(s->tok);
    model_free(s->m);
    // single-model serve borrows main()'s stack containers for the first
    // residency; only heap containers from a swap_to() reload are freed
    if (!SV.borrowed) { free(s->tok); free(s->m); }
    SV.borrowed = false;
    s->m = NULL;
    s->tok = NULL;
    resident_store(-1);
}

// swap_to results below 0: the name matched no registry entry (a caller
// typo — 400) vs the entry exists but its model failed to load (a broken
// model — 5xx). Callers must tell them apart.
#define SWAP_UNKNOWN     (-1)
#define SWAP_LOAD_FAILED (-2)

// resolve + load the requested model; returns entry index or SWAP_*
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
        if (idx < 0) return SWAP_UNKNOWN;
    }
    pthread_mutex_lock(&SV.swap_mu);
    if (resident_load() != idx) {
        unload_resident();
        slot_t *s = &SV.slots[0];
        fprintf(stderr, "swap: loading %s (%s)\n",
                SV.reg[idx].name, SV.reg[idx].path);
        s->m = calloc(1, sizeof(model_t));
        s->tok = calloc(1, sizeof(tokenizer));
        bool model_ok = s->m && model_load(s->m, SV.reg[idx].path, &SV.mp);
        bool tok_ok = model_ok && s->tok && tokenizer_init(s->tok, &s->m->gf);
        if (!model_ok || !tok_ok) {
            fprintf(stderr, "swap: failed to load %s\n", SV.reg[idx].name);
            if (s->tok) tokenizer_free(s->tok);
            if (s->m) model_free(s->m);
            free(s->m); free(s->tok);
            s->m = NULL; s->tok = NULL;
            pthread_mutex_unlock(&SV.swap_mu);
            return SWAP_LOAD_FAILED;
        }
        s->tmpl = SV.reg[idx].tmpl =
            template_detect(gguf_get_str(&s->m->gf, "tokenizer.chat_template", NULL),
                            s->tok);
        // sampling defaults follow the model, so they are re-resolved on every
        // swap; rng state and the penalty exemptions carry across untouched
        const sampler_preset *sp =
            sampler_resolve(&s->smp, s->m->arch,
                            gguf_get_str(&s->m->gf, "general.name", NULL), &SV.ov);
        s->smp_base = s->smp;
        SV.preset_name = sp->name;
        char sdesc[256];
        sampler_describe(&s->smp, sp, sdesc, sizeof(sdesc));
        fprintf(stderr, "sampling: %s\n", sdesc);
        engine_init(&s->e, s->m, s->tok, &s->smp);
        context_store(s->m->n_ctx);
        if (SV.single && SV.draft) {
            // engine_init memsets the engine; the draft (own KV, own pool)
            // survives target unload/reload and is re-attached here
            s->e.dm = SV.draft;
            s->e.draft_k = SV.draft_k;
        }
        resident_store(idx);
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
        if (resident_load() < 0 || atomic_load(&SV.busy)) continue;
        if (pthread_mutex_trylock(&SV.swap_mu) != 0) continue;
        int ttl = SV.ttl;
        if (ttl > 0 && resident_load() >= 0 && !atomic_load(&SV.busy) &&
            now_s() - SV.last_used > ttl)
            unload_resident();
        pthread_mutex_unlock(&SV.swap_mu);
    }
    return NULL;
}

static bool start_reaper(void) {
    pthread_attr_t attr;
    if (pthread_attr_init(&attr) != 0) return false;
    bool ok = pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED) == 0;
    pthread_t thread;
    if (ok) ok = pthread_create(&thread, &attr, ttl_reaper, NULL) == 0;
    pthread_attr_destroy(&attr);
    if (!ok) fprintf(stderr, "error: cannot start model TTL reaper\n");
    return ok;
}

static bool init_swap_runtime(const model_params *mp, int n_threads, int ttl) {
    SV.ttl = ttl;
    SV.mp = *mp;
    SV.mp.verbose = false;
    SV.mp.n_threads = n_threads;
    if (pthread_mutex_init(&SV.swap_mu, NULL) != 0) {
        fprintf(stderr, "error: cannot initialize model swap mutex\n");
        return false;
    }
    if (!start_reaper()) {
        pthread_mutex_destroy(&SV.swap_mu);
        return false;
    }
    return true;
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
    // OpenAI "stop" sequences: matched against the content channel only
    // (reasoning text must not trigger a client's stop strings)
    const char **stop_strs; // borrowed from the request jv, valid all request
    int   n_stop;
    sbuf  hold;         // held-back tail that may still begin a stop match
    bool  stopped;      // a stop sequence matched; excluded from output
    long  created;      // stamped once: every chunk of a stream reports the
                        // same creation time, as the buffered body does
    bool  role_sent;    // "role":"assistant" is emitted on the first delta only
    // strict tool envelope, streaming: the generated document is demuxed into
    // content and tool_calls deltas as it arrives, so the envelope itself is
    // never what the client receives
    tool_stream tsx;
    bool  tsx_on;
    int   tool_index;   // OpenAI tool_calls[].index; one call per turn for now
} gen_ctx;

// common prefix of every streamed chunk. `created` and `model` are required by
// the ChatCompletionChunk schema and strictly-validating SDKs reject a chunk
// without them, so they are written here rather than per call site.
static void chunk_open(gen_ctx *g, sbuf *c) {
    sb_fmt(c, "{\"id\":\"%s\",\"object\":\"%s\",\"created\":%ld,\"model\":\"",
           g->id, g->chat ? "chat.completion.chunk" : "text_completion",
           g->created);
    sb_esc(c, SV.model_name, strlen(SV.model_name));
    sb_lit(c, "\",\"choices\":[{\"index\":0,");
}

// frame one built chunk body as an SSE event and push it; a failed send marks
// the client gone, which is what aborts generation upstream
static int chunk_send(gen_ctx *g, sbuf *c) {
    sbuf sse = {0};
    sb_fmt(&sse, "data: %s\n\n", c->s ? c->s : "");
    if (c->failed || sse.failed || !send_all(g->fd, sse.s, sse.n)) g->dead = true;
    free(c->s);
    free(sse.s);
    return g->dead ? 1 : 0;
}

static void completion_cleanup(engine *e, snode *schema, gen_ctx *g) {
    e->schema = NULL;
    schema_free(schema);
    if (g) {
        tool_stream_free(&g->tsx);
        think_free(&g->ts);
        free(g->hold.s);
        free(g->reason.s);
        free(g->out.s);
    }
    free(e->lp_chosen); free(e->lp_ids); free(e->lp_top);
    e->lp_chosen = NULL; e->lp_ids = NULL; e->lp_top = NULL;
    e->lp_cap = e->lp_n = e->lp_count = 0;
}

// emit one section of split output: reasoning goes to the OpenAI-style
// reasoning_content field, everything else to content
// one text delta on the named chat channel (or the legacy completion "text")
static int send_text_delta(gen_ctx *g, int reasoning, const char *bytes, int n) {
    if (!g->stream || g->dead) return g->dead ? 1 : 0;
    sbuf c = {0};
    chunk_open(g, &c);
    if (g->chat) {
        sb_lit(&c, "\"delta\":{");
        if (!g->role_sent) { sb_lit(&c, "\"role\":\"assistant\","); g->role_sent = true; }
        sb_fmt(&c, "\"%s\":\"", reasoning ? "reasoning_content" : "content");
    } else {
        sb_lit(&c, "\"text\":\"");
    }
    sb_esc(&c, bytes, n);
    sb_lit(&c, g->chat ? "\"},\"finish_reason\":null}]}"
                       : "\",\"finish_reason\":null}]}");
    return chunk_send(g, &c);
}

// ---- tool_stream sinks: the demuxed envelope, as OpenAI streaming events

static int sink_content(void *ud, const char *b, int n) {
    return send_text_delta(ud, 0, b, n);
}

// the opening event of a call carries everything that identifies it; the
// deltas that follow carry argument text only, keyed by the same index
static int sink_call_begin(void *ud, const char *name) {
    gen_ctx *g = ud;
    if (g->dead) return 1;
    sbuf c = {0};
    chunk_open(g, &c);
    sb_lit(&c, "\"delta\":{");
    if (!g->role_sent) { sb_lit(&c, "\"role\":\"assistant\","); g->role_sent = true; }
    sb_fmt(&c, "\"tool_calls\":[{\"index\":%d,\"id\":\"call_%d\","
               "\"type\":\"function\",\"function\":{\"name\":\"",
           g->tool_index, g->tool_index);
    sb_esc(&c, name, strlen(name));
    sb_lit(&c, "\",\"arguments\":\"\"}}]},\"finish_reason\":null}]}");
    return chunk_send(g, &c);
}

static int sink_call_args(void *ud, const char *b, int n) {
    gen_ctx *g = ud;
    if (g->dead) return 1;
    sbuf c = {0};
    chunk_open(g, &c);
    sb_fmt(&c, "\"delta\":{\"tool_calls\":[{\"index\":%d,\"function\":"
               "{\"arguments\":\"", g->tool_index);
    sb_esc(&c, b, n);
    sb_lit(&c, "\"}}]},\"finish_reason\":null}]}");
    return chunk_send(g, &c);
}

static int emit_channel(gen_ctx *g, int reasoning, const char *bytes, int n) {
    sb_put(reasoning ? &g->reason : &g->out, bytes, n);
    if (!g->stream || g->dead) return g->dead ? 1 : 0;
    // reasoning is its own channel and never part of the envelope document
    if (!reasoning && g->tsx_on) return tool_stream_feed(&g->tsx, bytes, n);
    return send_text_delta(g, reasoning, bytes, n);
}

// content-channel filter for user stop sequences: bytes are staged in
// g->hold so a stop spanning token boundaries still matches, and only the
// tail that could still begin a stop is withheld from the client
static int stop_feed(gen_ctx *g, const char *bytes, int n) {
    sb_put(&g->hold, bytes, n);
    size_t at = 0, hit_len = 0;
    for (size_t i = 0; !hit_len && i < g->hold.n; i++)
        for (int s = 0; s < g->n_stop; s++) {
            size_t len = strlen(g->stop_strs[s]);
            if (i + len <= g->hold.n &&
                memcmp(g->hold.s + i, g->stop_strs[s], len) == 0) {
                at = i;
                hit_len = len;
                break;
            }
        }
    if (hit_len) {
        if (at > 0) emit_channel(g, 0, g->hold.s, (int)at);
        g->hold.n = 0;
        g->stopped = true;
        return 1; // abort generation
    }
    size_t keep = 0;
    for (int s = 0; s < g->n_stop; s++) {
        size_t len = strlen(g->stop_strs[s]);
        size_t k = len - 1 < g->hold.n ? len - 1 : g->hold.n;
        for (; k > keep; k--)
            if (memcmp(g->hold.s + g->hold.n - k, g->stop_strs[s], k) == 0) {
                keep = k;
                break;
            }
    }
    size_t rel = g->hold.n - keep;
    int rc = 0;
    if (rel > 0) {
        rc = emit_channel(g, 0, g->hold.s, (int)rel);
        memmove(g->hold.s, g->hold.s + rel, keep);
        g->hold.n = keep;
    }
    return rc;
}

static int gen_emit(void *ud, int reasoning, const char *bytes, int n) {
    gen_ctx *g = ud;
    if (!reasoning && g->n_stop) return stop_feed(g, bytes, n);
    return emit_channel(g, reasoning, bytes, n);
}

static int gen_collect(void *ud, const char *bytes, int n) {
    gen_ctx *g = ud;
    return think_feed(&g->ts, bytes, n, gen_emit, g);
}

// A request field the server cannot use must be an error, never a silent
// fallback to the default: the caller then gets a response generated with
// settings it did not ask for and no way to detect it. Stringified numbers
// are the common shape of the mistake — several HTTP layers produce them
// from form or env-derived config.
//
// `null` is the deliberate exception and reads as absent. Every mainstream
// OpenAI SDK serialises an unset optional field as null rather than omitting
// it, so treating null as a wrong type would 400 on ordinary traffic from an
// unmodified client.
static bool absent(const jv *v) { return !v || v->type == J_NULL; }

static bool request_number(jv *req, const char *key, double dflt,
                           double min, double max, double *out) {
    jv *v = jv_get(req, key);
    if (!absent(v) && v->type != J_NUM) return false;
    double n = absent(v) ? dflt : v->num;
    if (!isfinite(n) || n < min || n > max) return false;
    *out = n;
    return true;
}

// negative sentinels: MT_UNLIMITED clamps to the context window later,
// the other two are request errors with distinct messages
enum { MT_BAD_TYPE = -3, MT_NON_FINITE = -2, MT_UNLIMITED = -1 };

static int request_max_tokens(jv *req, int dflt) {
    jv *v = jv_get(req, "max_tokens");
    if (absent(v)) v = jv_get(req, "max_completion_tokens");
    if (absent(v)) return dflt;
    if (v->type != J_NUM) return MT_BAD_TYPE;
    if (!isfinite(v->num)) return MT_NON_FINITE;
    if (v->num < 0) return MT_UNLIMITED;
    if (v->num > INT_MAX) return INT_MAX;
    return (int)v->num;
}

static bool request_keep_alive(jv *req, bool *present, int *seconds) {
    jv *v = jv_get(req, "keep_alive");
    if (!absent(v) && v->type != J_NUM) return false;
    *present = !absent(v);
    if (!*present) return true;
    if (!isfinite(v->num) || v->num > INT_MAX) return false;
    *seconds = v->num < 0 ? -1 : (int)v->num;
    return true;
}

// a boolean request flag: absent takes the default, a non-boolean is an
// error rather than a silent `false`
static bool request_bool(jv *req, const char *key, bool dflt, bool *out) {
    jv *v = jv_get(req, key);
    if (absent(v)) { *out = dflt; return true; }
    if (v->type != J_BOOL) return false;
    *out = v->b;
    return true;
}

// the caller's schema-constrained-output request: OpenAI response_format
// {type:"json_schema", json_schema:{schema:{...}}} or an Ollama-style
// "format" object. NULL when the request asked for no schema.
static jv *request_schema(jv *req) {
    jv *rf = jv_get(req, "response_format");
    jv *sch = NULL;
    if (rf && strcmp(jv_str(jv_get(rf, "type"), ""), "json_schema") == 0) {
        jv *js = jv_get(rf, "json_schema");
        sch = js ? (jv_get(js, "schema") ? jv_get(js, "schema") : js) : NULL;
    }
    jv *fmt = jv_get(req, "format");
    if (!sch && fmt && fmt->type == J_OBJ) sch = fmt;
    return sch;
}

// run one completion on a slot and write the HTTP response.
// `env` is the strict tool-call envelope when the request opted into one; it
// replaces the response_format schema, having already absorbed it as the
// shape of its `final` branch.
static void run_completion(slot_t *s, int fd, const char *prompt, bool chat,
                           jv *req, const tool_envelope *env) {
    model_t *m = s->m;
    engine *e = &s->e;

    // per-request sampling params start from the server defaults every time —
    // one request's overrides must not leak into the next on this slot; only
    // the rng STATE carries across so sampling sequences stay diverse
    double temp, top_p, min_p, top_k, seed, repeat_penalty;
    if (!request_number(req, "temperature", s->smp_base.temp, 0, FLT_MAX, &temp) ||
        !request_number(req, "top_p", s->smp_base.top_p, 0, 1, &top_p) ||
        !request_number(req, "min_p", s->smp_base.min_p, 0, 1, &min_p) ||
        !request_number(req, "top_k", s->smp_base.top_k, 0, INT_MAX, &top_k) ||
        // the model's family preset decides the default; a client that wants
        // no penalty at all asks for 1. Zero is rejected rather than treated
        // as "off": the penalty divides by it.
        !request_number(req, "repeat_penalty", s->smp_base.repeat_penalty,
                        FLT_MIN, FLT_MAX, &repeat_penalty) ||
        !request_number(req, "seed", 0, 0, 18446744073709549568.0, &seed)) {
        send_error(fd, 400, "numeric sampling parameter out of range");
        return;
    }
    uint64_t rng_state = s->smp.rng;
    s->smp = s->smp_base;
    s->smp.rng = rng_state;
    s->smp.temp = (float)temp;
    s->smp.top_p = (float)top_p;
    s->smp.min_p = (float)min_p;
    s->smp.top_k = (int)top_k;
    s->smp.repeat_penalty = (float)repeat_penalty;
    if (seed > 0) s->smp.rng = (uint64_t)seed;
    int max_tokens = request_max_tokens(req, SV.n_predict_cap);
    if (max_tokens == MT_NON_FINITE) {
        send_error(fd, 400, "max_tokens out of range");
        return;
    }
    if (max_tokens == MT_BAD_TYPE) {
        send_error(fd, 400, "max_tokens must be a number");
        return;
    }
    // "stream":"true" used to read as false and answer with a buffered
    // body, leaving a client that expected SSE waiting on events that
    // would never arrive
    bool stream = false;
    if (!request_bool(req, "stream", false, &stream)) {
        send_error(fd, 400, "stream must be a boolean");
        return;
    }
    // OpenAI logprobs (chat, buffered responses only)
    // chat only: on /v1/completions OpenAI defines logprobs as an integer
    // count, so a number there is not a type error
    bool lp_on = false;
    if (chat && !request_bool(req, "logprobs", false, &lp_on)) {
        send_error(fd, 400, "logprobs must be a boolean");
        return;
    }
    bool want_lp = chat && !stream && lp_on;
    double lp_num = 0;
    if (want_lp && !request_number(req, "top_logprobs", 0, 0, 20, &lp_num)) {
        send_error(fd, 400, "top_logprobs out of range");
        return;
    }
    int lp_n = (int)lp_num;
    if (lp_n < 0) lp_n = 0;
    if (lp_n > 20) lp_n = 20;
    // OpenAI "stop": a string or an array of up to 4 non-empty strings.
    // Pointers borrow from req, which outlives the whole request.
    const char *stops[4];
    int n_stops = 0;
    jv *stopv = jv_get(req, "stop");
    if (stopv && stopv->type != J_NULL) {
        bool bad = false;
        if (stopv->type == J_STR) {
            bad = stopv->str[0] == 0;
            if (!bad) stops[n_stops++] = stopv->str;
        } else if (stopv->type == J_ARR && stopv->n <= 4) {
            for (int i = 0; i < stopv->n && !bad; i++) {
                jv *it = stopv->items[i];
                if (!it || it->type != J_STR || it->str[0] == 0) bad = true;
                else stops[n_stops++] = it->str;
            }
        } else {
            bad = true;
        }
        if (bad) {
            send_error(fd, 400,
                       "stop must be a string or an array of up to 4 non-empty strings");
            return;
        }
    }
    jv *rf = jv_get(req, "response_format");
    if (rf) {
        // An unrecognised or malformed response_format used to fall through to
        // unconstrained decoding while still answering 200, so a caller asking
        // for guaranteed structure silently got none.
        if (rf->type != J_OBJ) {
            send_error(fd, 400, "response_format must be an object");
            return;
        }
        const char *rft = jv_str(jv_get(rf, "type"), "");
        if (strcmp(rft, "json_object") != 0 && strcmp(rft, "json_schema") != 0 &&
            strcmp(rft, "text") != 0) {
            send_error(fd, 400,
                       "response_format.type must be text, json_object or json_schema");
            return;
        }
        if (strcmp(rft, "json_schema") == 0) {
            jv *js = jv_get(rf, "json_schema");
            if (!js || js->type != J_OBJ) {
                send_error(fd, 400,
                           "response_format.json_schema must be an object");
                return;
            }
            jv *inner = jv_get(js, "schema");
            if (inner && inner->type != J_OBJ) {
                send_error(fd, 400,
                           "response_format.json_schema.schema must be an object");
                return;
            }
        }
    }
    e->json_mode = rf && strcmp(jv_str(jv_get(rf, "type"), ""), "json_object") == 0;
    // Constrained decoding. The tool envelope wins when present: it already
    // contains the caller's response_format schema as its `final` branch, so
    // compiling that separately would drop the tool branches.
    snode *schema = NULL;
    jv *sch = env ? NULL : request_schema(req);
    if (env) {
        char serr[128] = "envelope did not parse";
        jv *ej = json_parse(env->schema_src, strlen(env->schema_src));
        schema = ej ? schema_compile(ej, serr, sizeof(serr)) : NULL;
        jv_free(ej);
        if (!schema) {
            char msg[256];
            snprintf(msg, sizeof(msg), "unsupported tool schema: %s", serr);
            send_error(fd, 400, msg);
            return;
        }
    } else if (sch) {
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
    if (n_prompt == 0 || n_prompt >= m->n_ctx) {
        free(toks);
        completion_cleanup(e, schema, NULL);
        send_error(fd, 400, n_prompt == 0 ? "empty prompt"
                                          : "prompt exceeds context window");
        return;
    }
    int remaining_ctx = m->n_ctx - n_prompt;
    if (max_tokens < 0 || max_tokens > remaining_ctx) max_tokens = remaining_ctx;

    if (want_lp && max_tokens > 0) {
        e->lp_cap    = max_tokens;
        e->lp_n      = lp_n;
        e->lp_chosen = malloc(sizeof(float) * max_tokens);
        e->lp_ids    = malloc(sizeof(int32_t) * max_tokens);
        e->lp_top    = lp_n ? malloc(sizeof(lp_alt) * (size_t)max_tokens * lp_n) : NULL;
    }

    // reuse the KV for whatever prefix this prompt shares with the last one —
    // pipeline callers repeat long system/template prefixes verbatim. Callers
    // can disable reuse per request when they need exact fresh-prompt telemetry
    // or want to isolate a pathological prompt from the previous cache state.
    bool cache_prompt = jv_bool(jv_get(req, "cache_prompt"), true);
    int keep = 0;
    if (cache_prompt) keep = engine_rewind(e, toks, n_prompt);
    else              engine_reset(e);
    float *logits = engine_feed(e, toks + keep, n_prompt - keep);
    free(toks);
    if (!logits) {
        completion_cleanup(e, schema, NULL);
        send_error(fd, 500, "context overflow");
        return;
    }

    gen_ctx g = { .out = {0}, .fd = fd, .stream = stream, .chat = chat,
                  .stop_strs = stops, .n_stop = n_stops,
                  .created = (long)time(NULL) };
    snprintf(g.id, sizeof(g.id), "%s-%d", chat ? "chatcmpl" : "cmpl",
             atomic_fetch_add(&SV.req_counter, 1));
    // split thinking channels out of chat responses; raw completions stay raw
    think_init(&g.ts, chat ? m->think_open : NULL, m->think_close);
    if (stream && env) {
        tool_stream_sink sink = { &g, sink_content, sink_call_begin,
                                  sink_call_args };
        tool_stream_init(&g.tsx, env, &sink);
        g.tsx_on = true;
    }

    if (stream) {
        const char *hdr = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                          "Cache-Control: no-cache\r\nConnection: close\r\n\r\n";
        if (!send_all(fd, hdr, strlen(hdr))) {
            completion_cleanup(e, schema, &g);
            return;
        }
        // OpenAI opens a chat stream with a role-only delta. Emitting it up
        // front rather than folding the role into whatever the first text
        // delta happens to be keeps the contract independent of what the
        // model generates — including generating nothing at all.
        if (chat) {
            sbuf c = {0};
            chunk_open(&g, &c);
            sb_lit(&c, "\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}");
            g.role_sent = true;
            chunk_send(&g, &c);
        }
    }

    double gtime;
    int n_gen = engine_generate(e, logits, max_tokens, gen_collect, &g, &gtime);
    think_finish(&g.ts, gen_emit, &g);
    // generation ended without a stop match: the withheld partial-match
    // tail was ordinary output after all
    if (!g.stopped && g.hold.n > 0) {
        emit_channel(&g, 0, g.hold.s, (int)g.hold.n);
        g.hold.n = 0;
    }
    const char *finish = g.stopped || e->hit_stop ? "stop" : "length";
    // a streamed call reports the same terminal reason a buffered one does
    if (g.tsx_on && tool_stream_called(&g.tsx)) finish = "tool_calls";

    if (stream) {
        if (!g.dead) {
            sbuf c = {0};
            chunk_open(&g, &c);
            sb_fmt(&c, "%s,\"finish_reason\":\"%s\"}]}",
                   chat ? "\"delta\":{}" : "\"text\":\"\"", finish);
            bool ok = chunk_send(&g, &c) == 0;
            // OpenAI stream_options {"include_usage": true}: one extra chunk
            // with empty choices and the request's token accounting — AI-SDK
            // clients (Cline et al.) request this on every stream
            if (ok && jv_bool(jv_get(jv_get(req, "stream_options"),
                                     "include_usage"), false)) {
                sbuf u = {0};
                sb_fmt(&u, "{\"id\":\"%s\",\"object\":\"%s\",\"created\":%ld,"
                           "\"model\":\"", g.id,
                       chat ? "chat.completion.chunk" : "text_completion",
                       g.created);
                sb_esc(&u, SV.model_name, strlen(SV.model_name));
                sb_fmt(&u, "\",\"choices\":[],"
                           "\"usage\":{\"prompt_tokens\":%d,"
                           "\"completion_tokens\":%d,\"total_tokens\":%d}}",
                       n_prompt, n_gen, n_prompt + n_gen);
                ok = chunk_send(&g, &u) == 0;
            }
            if (ok) send_all(fd, "data: [DONE]\n\n", 14);
        }
    } else {
        sbuf tc = {0};
        int n_tc = 0;
        if (env) {
            // Strict mode: the whole response IS the envelope, guaranteed by
            // the schema rather than fished out of free text. A truncated
            // call was closed to a legal document by sval_close, so it is
            // still executable and still reports "tool_calls".
            sbuf mapped = {0};
            int rc = tool_envelope_map(env, g.out.s ? g.out.s : "", g.out.n,
                                       &mapped, &tc);
            if (rc == 1) {
                n_tc = 1;
                finish = "tool_calls";
                g.out.n = 0;
                free(mapped.s);
            } else if (rc == 0) {
                free(g.out.s);
                g.out = mapped;      // the final branch's payload is the reply
            } else {
                free(mapped.s);      // unparseable: hand back what was generated
            }
        } else if (chat) {
            n_tc = tool_calls_parse(&g.out, &tc);
            if (n_tc) {
                finish = "tool_calls";
                g.out.n = 0; // OpenAI convention: no content alongside
                             // tool_calls (whatever followed was the model
                             // faking a result)
            }
        }
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
            if (tc.failed) r.failed = true;
            else sb_put(&r, tc.s, tc.n);
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
                   "\"total_tokens\":%d},"
                   "\"runner_telemetry\":{\"prompt_cached_tokens\":%d,"
                   "\"prompt_eval_tokens\":%d,\"generation_seconds\":%.6f,"
                   "\"generation_tok_s\":%.3f,\"json_mode\":%s,"
                   "\"schema\":%s,\"speculative\":%s}}",
               finish, n_prompt, n_gen, n_prompt + n_gen,
               keep, n_prompt - keep, gtime,
               n_gen / (gtime > 0 ? gtime : 1e-9),
               e->json_mode ? "true" : "false",
               schema ? "true" : "false",
               e->dm ? "true" : "false");
        send_built(fd, &r);
        free(r.s);
    }
    fprintf(stderr, "[slot %d] %s: %d prompt (%d cached) + %d gen tok (%.1f tok/s)%s%s\n",
            s->id, g.id, n_prompt, keep, n_gen,
            n_gen / (gtime > 0 ? gtime : 1e-9),
            schema ? " [schema]" : e->json_mode ? " [json]" : e->dm ? " [spec]" : "",
            g.dead ? " [client gone]" : "");
    completion_cleanup(e, schema, &g);
}

// ---------------------------------------------------------------- routes

// flatten one OpenAI message to plain text: string content passes through;
// the AI-SDK part-array form (Cline et al.) concatenates its text parts;
// assistant tool_calls render in runner's own call syntax so replayed
// history reads like what the model actually emitted. Returns a heap
// string, or NULL when the message carries nothing usable.
static char *message_text(jv *msg) {
    jv *content = jv_get(msg, "content");
    sbuf b = {0};
    if (content && content->type == J_STR) {
        sb_put(&b, content->str, strlen(content->str));
    } else if (content && content->type == J_ARR) {
        for (int i = 0; i < content->n; i++) {
            const char *type = jv_str(jv_get(content->items[i], "type"), "");
            const char *text = jv_str(jv_get(content->items[i], "text"), NULL);
            if (strcmp(type, "text") != 0 || !text) continue; // images etc.
            if (b.n) sb_lit(&b, "\n");
            sb_put(&b, text, strlen(text));
        }
    }
    jv *calls = jv_get(msg, "tool_calls");
    if (calls && calls->type == J_ARR) {
        for (int i = 0; i < calls->n; i++) {
            jv *fn = jv_get(calls->items[i], "function");
            const char *name = jv_str(jv_get(fn, "name"), NULL);
            const char *args = jv_str(jv_get(fn, "arguments"), "{}");
            if (!name) continue;
            sb_fmt(&b, "<|tool_call>call:%s%s<tool_call|>", name, args);
        }
    }
    return b.s;
}

static void handle_chat(slot_t *s, int fd, jv *req) {
    jv *msgs = jv_get(req, "messages");
    if (!msgs || msgs->type != J_ARR || msgs->n == 0) {
        send_error(fd, 400, "missing messages");
        return;
    }
    // OpenAI "tools" become a leading system turn (template.c owns the syntax).
    //
    // Strict mode compiles them into a discriminated union that constrains
    // sampling, so the model cannot name an undeclared tool or malform its
    // arguments. It applies to streamed requests too: the envelope is
    // demultiplexed as it is generated (tool_stream) rather than parsed
    // afterward, so both paths reach the same call from the same guarantee.
    jv *tools = jv_get(req, "tools");
    tool_envelope env = {0};
    char terr[224];
    int rc = tool_envelope_build(tools, jv_get(req, "tool_choice"),
                                 request_schema(req), &env,
                                 terr, sizeof(terr));
    if (rc < 0) { send_error(fd, 400, terr); return; }
    bool strict = rc == 1;
    if (strict) {
        // one call per turn for now; silently ignoring a request for
        // several would leave the caller expecting calls it never gets
        bool parallel = false;
        if (!request_bool(req, "parallel_tool_calls", false, &parallel)) {
            tool_envelope_free(&env);
            send_error(fd, 400, "parallel_tool_calls must be a boolean");
            return;
        }
        if (parallel) {
            tool_envelope_free(&env);
            send_error(fd, 400,
                       "parallel_tool_calls:true is not supported yet; "
                       "one call per turn");
            return;
        }
    }
    sbuf ts = {0};
    if (strict) sb_put(&ts, env.system_turn, strlen(env.system_turn));
    else        tools_render(tools, &ts);
    chat_msg *cm = malloc(sizeof(chat_msg) * (msgs->n + 1));
    char **owned = malloc(sizeof(char *) * msgs->n);
    size_t total = ts.n + 64;
    int n_cm = 0, n_own = 0;
    if (ts.n) cm[n_cm++] = (chat_msg){ "system", ts.s };
    for (int i = 0; i < msgs->n; i++) {
        const char *role = jv_str(jv_get(msgs->items[i], "role"), "user");
        char *content = message_text(msgs->items[i]);
        if (!content) continue;
        owned[n_own++] = content;
        cm[n_cm++] = (chat_msg){ role, content };
        total += strlen(role) + strlen(content) + 64;
    }
    if (n_cm == 0) {
        free(owned);
        free(cm);
        free(ts.s);
        tool_envelope_free(&env);
        send_error(fd, 400, "no message content");
        return;
    }
    char *prompt = malloc(total + 256);
    render_messages(s->tmpl, cm, n_cm, true, prompt, total + 256);
    run_completion(s, fd, prompt, true, req, strict ? &env : NULL);
    free(prompt);
    for (int i = 0; i < n_own; i++) free(owned[i]);
    free(owned);
    free(cm);
    free(ts.s);
    tool_envelope_free(&env);
}

static void handle_completion(slot_t *s, int fd, jv *req) {
    const char *prompt = jv_str(jv_get(req, "prompt"), NULL);
    if (!prompt) { send_error(fd, 400, "missing prompt"); return; }
    run_completion(s, fd, prompt, false, req, NULL);
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
        send_built(fd, &r);
        fprintf(stderr, "[slot %d] embeddings: %d input(s), %d tok\n",
                s->id, n_in, total);
    }
    free(r.s);
    free(emb);
}

// ---------------------------------------------------------------- http

// /health and /v1/models read only startup-immutable strings plus an atomic
// resident snapshot, so they are safe to answer from the accept thread with no lock
static void send_health(int fd) {
    char b[384];
    int n, res = resident_load();
    if (SV.n_reg > 0 && res >= 0) {
        char esc[192];
        json_escape(SV.reg[res].name, strlen(SV.reg[res].name), esc, sizeof(esc));
        n = snprintf(b, sizeof(b), "{\"status\":\"ok\",\"resident\":\"%s\"}", esc);
    } else if (SV.n_reg > 0) {
        n = snprintf(b, sizeof(b), "{\"status\":\"ok\",\"resident\":null}");
    } else {
        n = snprintf(b, sizeof(b), "{\"status\":\"ok\"}");
    }
    send_response(fd, 200, "application/json", b, n);
}

static void send_models(int fd) {
    sbuf r = {0};
    sb_lit(&r, "{\"object\":\"list\",\"data\":[");
    if (SV.n_reg > 0) {
        for (int i = 0; i < SV.n_reg; i++) {
            char esc[192];
            json_escape(SV.reg[i].name, strlen(SV.reg[i].name), esc, sizeof(esc));
            sb_fmt(&r, "%s{\"id\":\"%s\",\"object\":\"model\","
                       "\"owned_by\":\"runner\"}", i ? "," : "", esc);
        }
    } else {
        char esc[256];
        json_escape(SV.model_name, strlen(SV.model_name), esc, sizeof(esc));
        sb_fmt(&r, "{\"id\":\"%s\",\"object\":\"model\",\"owned_by\":\"runner\"}", esc);
    }
    sb_lit(&r, "]}");
    send_built(fd, &r);
    free(r.s);
}

static void send_capabilities(int fd) {
    sbuf r = {0};
    int res = resident_load();
    sb_lit(&r, "{\"object\":\"runner.capabilities\",\"swap\":");
    sb_lit(&r, SV.n_reg > 0 && !SV.single ? "true" : "false");
    sb_lit(&r, ",\"resident\":");
    if (SV.n_reg > 0 && res >= 0) {
        sb_lit(&r, "\"");
        sb_esc(&r, SV.reg[res].name, strlen(SV.reg[res].name));
        sb_lit(&r, "\"");
    } else {
        sb_lit(&r, "null");
    }
    sb_fmt(&r, ",\"context\":%d,\"models\":[", context_load());
    if (SV.n_reg > 0) {
        for (int i = 0; i < SV.n_reg; i++) {
            if (i) sb_lit(&r, ",");
            sb_lit(&r, "{\"id\":\"");
            sb_esc(&r, SV.reg[i].name, strlen(SV.reg[i].name));
            sb_fmt(&r, "\",\"resident\":%s}", i == res ? "true" : "false");
        }
    } else {
        sb_lit(&r, "{\"id\":\"");
        sb_esc(&r, SV.model_name, strlen(SV.model_name));
        sb_lit(&r, "\",\"resident\":true}");
    }
    sb_lit(&r, "],\"sampling\":{\"preset\":");
    if (SV.preset_name) {
        sb_lit(&r, "\"");
        sb_esc(&r, SV.preset_name, strlen(SV.preset_name));
        sb_lit(&r, "\"");
    } else {
        sb_lit(&r, "null");  // swap mode before the first model is resident
    }
    {
        const sampler *d = &SV.slots[0].smp_base;
        sb_fmt(&r, ",\"temperature\":%.2f,\"top_p\":%.2f,\"top_k\":%d,"
                   "\"min_p\":%.2f,\"repeat_penalty\":%.2f}",
               (double)d->temp, (double)d->top_p, d->top_k,
               (double)d->min_p, (double)d->repeat_penalty);
    }
    sb_lit(&r, ",\"features\":{"
               "\"json_object\":true,"
               "\"json_schema\":true,"
               "\"stop_sequences\":true,"
               "\"schema_conditionals\":true,"
               "\"schema_string_bounds\":true,"
               "\"request_telemetry\":true,"
               "\"prefix_cache\":true,"
               "\"prefix_cache_controls\":true,"
               "\"repeat_penalty\":true,"
               "\"family_sampling_presets\":true}}");
    send_built(fd, &r);
    free(r.s);
}

static void handle_conn(slot_t *s, int fd) {
    // a stalled or dead client must not pin an inference slot: the whole
    // request (header + body) has to arrive within this budget. Generation
    // time stays unbounded — the deadline only covers reading the request.
    //
    // The write side needs its own bound. A client that sends a valid request
    // and then stops reading fills the socket buffer, and an unbounded
    // blocking write parks the slot forever: the read deadline is already
    // satisfied and never fires again, so the slot is never returned.
    sock_send_timeout(fd, 30.0);
    double deadline = now_s() + 10.0;
    char hdr[16384];
    size_t got = 0;
    char *body_start = NULL;
    while (got < sizeof(hdr) - 1) {
        double remaining = deadline - now_s();
        if (remaining <= 0) { send_error(fd, 408, "request read timed out"); return; }
        sock_recv_timeout(fd, remaining);
        int r = sock_recv(fd, hdr + got, sizeof(hdr) - 1 - got);
        if (r <= 0) {
            // r == 0: orderly close, client is gone. r < 0: timeout (or a
            // socket error, where the 408 write fails harmlessly).
            if (r < 0) send_error(fd, 408, "request read timed out");
            return;
        }
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
        if (!body) { send_error(fd, 500, "cannot allocate request body"); return; }
        size_t have = got - (size_t)(body_start - hdr);
        if (have > content_length) have = content_length;
        memcpy(body, body_start, have);
        while (have < content_length) {
            double remaining = deadline - now_s();
            if (remaining <= 0) {
                free(body);
                send_error(fd, 408, "request read timed out");
                return;
            }
            sock_recv_timeout(fd, remaining);
            int r = sock_recv(fd, body + have, content_length - have);
            if (r <= 0) {
                free(body);
                if (r < 0) send_error(fd, 408, "request read timed out");
                return;
            }
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
        send_health(fd);
    } else if (!strcmp(method, "GET") && !strcmp(path, "/v1/models")) {
        send_models(fd);
    } else if (!strcmp(method, "GET") && !strcmp(path, "/v1/capabilities")) {
        send_capabilities(fd);
    } else if (!strcmp(method, "POST") &&
               (!strcmp(path, "/v1/chat/completions") ||
                !strcmp(path, "/v1/completions") ||
                !strcmp(path, "/v1/embeddings"))) {
        jv *req = body ? json_parse(body, content_length) : NULL;
        if (!req) {
            send_error(fd, 400, "invalid JSON body");
        } else {
            bool has_keep_alive = false;
            int keep_alive = 0;
            if (!request_keep_alive(req, &has_keep_alive, &keep_alive)) {
                send_error(fd, 400, "keep_alive out of range");
                jv_free(req);
                free(body);
                return;
            }
            atomic_store(&SV.busy, true);
            bool ok = true;
            if (SV.n_reg > 0) {
                int sw = swap_to(jv_str(jv_get(req, "model"), NULL));
                if (sw == SWAP_LOAD_FAILED) {
                    send_error(fd, 500,
                               "model failed to load (registered but broken; see server log)");
                    ok = false;
                } else if (sw < 0) {
                    send_error(fd, 400, "unknown model (see /v1/models)");
                    ok = false;
                }
            }
            if (ok) {
                if (strcmp(path, "/v1/chat/completions") == 0) handle_chat(s, fd, req);
                else if (strcmp(path, "/v1/embeddings") == 0) handle_embeddings(s, fd, req);
                else handle_completion(s, fd, req);
                // Ollama-style keep_alive: seconds of idle before the model
                // unloads (swap mode) — 0 unloads now, negative pins forever
                if (has_keep_alive && SV.n_reg > 0) {
                    pthread_mutex_lock(&SV.swap_mu);
                    if (keep_alive == 0) unload_resident();
                    else SV.ttl = keep_alive < 0 ? 0 : keep_alive;
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

// answer tiny read-only GETs from the accept loop: single-slot serving means
// one long generation used to block /health until the gridcore watchdog
// declared a live runner "unhealthy: timed out". POSTs (and /unload, which
// frees the resident model and must stay serialized with generation) are
// handed to a slot untouched.
static bool accept_fastpath(int fd) {
#ifndef _WIN32
    // POSIX fd_set is a fixed-size bitmask indexed by fd value; FD_SET on an
    // fd >= FD_SETSIZE is undefined behavior (out-of-bounds write). Windows
    // fd_set is a count-based array instead, so it isn't affected.
    if (fd >= FD_SETSIZE) return false;
#endif
    fd_set rs;
    struct timeval tv = { 0, 250000 }; // loopback data lands in <1ms
    FD_ZERO(&rs);
    FD_SET(fd, &rs);
    if (select(fd + 1, &rs, NULL, NULL, &tv) != 1) return false;
    char hdr[2048];
    int n = sock_peek(fd, hdr, 64);
    if (n <= 0) { sock_close(fd); return true; } // died before speaking
    hdr[n] = 0;
    // match the path plus the space HTTP/1.x always puts before the version,
    // so "GET /healthzzz" falls through to the slot path instead of being
    // misrouted here. Bare HTTP/0.9 "GET /health\r\n" (no version) won't
    // match, but neither curl nor the gridcore watchdog send that, so it's
    // not worth the extra branch.
    bool health = !strncmp(hdr, "GET /health ", 12);
    bool models = !strncmp(hdr, "GET /v1/models ", 15);
    bool caps = !strncmp(hdr, "GET /v1/capabilities ", 21);
    if (!health && !models && !caps) return false;
    // Drain the request before replying: closing with unread bytes can RST
    // the connection and discard our response. But the accept thread must
    // never block indefinitely on a single connection — it's the only thing
    // calling accept(), so a stalled client here would queue every later
    // connection behind it, reproducing the watchdog-timeout bug this
    // fastpath exists to fix. Re-select before each recv with a short
    // timeout and cap the total drain time; if a client dribbles bytes too
    // slowly to finish the header in the budget, answer anyway (these GETs
    // are tiny and read-only, so a reply is always correct) and move on.
    double deadline = now_s() + 0.5;
    size_t got = 0;
    while (got < sizeof(hdr) - 1 && !strstr(hdr, "\r\n\r\n")) {
        double remaining = deadline - now_s();
        if (remaining <= 0) break;
        struct timeval dtv;
        dtv.tv_sec = 0;
        dtv.tv_usec = (long)(remaining * 1e6);
        if (dtv.tv_usec > 100000) dtv.tv_usec = 100000; // poll in <=100ms steps
        FD_ZERO(&rs);
        FD_SET(fd, &rs);
        if (select(fd + 1, &rs, NULL, NULL, &dtv) != 1) break;
        int r = sock_recv(fd, hdr + got, sizeof(hdr) - 1 - got);
        if (r <= 0) break;
        got += (size_t)r;
        hdr[got] = 0;
    }
    if (health) send_health(fd);
    else if (models) send_models(fd);
    else        send_capabilities(fd);
    sock_close(fd);
    return true;
}

// ---------------------------------------------------------------- entry

int server_run(model_t *base, tokenizer *tok, const char *model_path,
               const model_params *mp, sampler defaults,
               const sampler_override *ov, int port, int parallel,
               int n_threads, int ttl, const char *draft_path, int draft_k) {
    sock_init();
    if (ov) SV.ov = *ov;
    // `defaults` arrives already resolved against the preloaded model; in swap
    // mode there is no model yet and swap_to resolves per load
    SV.preset_name = base ? sampler_preset_for(base->arch,
                        gguf_get_str(&base->gf, "general.name", NULL))->name
                          : NULL;
    if (parallel < 1) parallel = 1;
    if (parallel > 16) parallel = 16;
    bool swap_mode = strchr(model_path, '=') != NULL;
    if (ttl < 0) ttl = swap_mode ? 300 : 0; // single-model default: never unload
    if (draft_path && swap_mode) {
        fprintf(stderr, "note: --draft needs a single served model — "
                "ignoring it in swap mode\n");
        draft_path = NULL;
    }

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
        resident_store(-1);
    }

    int threads_per_slot = n_threads / parallel;
    if (threads_per_slot < 1) threads_per_slot = 1;

    const char *name = strrchr(model_path, '/');
    const char *bsname = strrchr(model_path, '\\'); // Windows path separator
    if (bsname && (!name || bsname > name)) name = bsname;
    SV.model_name = SV.n_reg > 0 ? SV.reg[0].name : (name ? name + 1 : model_path);
    SV.n_predict_cap = 1024;
    context_store(base ? base->n_ctx : mp->n_ctx);
    SV.n_slots = parallel;
    SV.slots = calloc(parallel, sizeof(slot_t));
    if (!SV.slots) {
        fprintf(stderr, "error: cannot allocate server slots\n");
        return 1;
    }
    if (pthread_mutex_init(&SV.q.mu, NULL) != 0) {
        fprintf(stderr, "error: cannot initialize server queue mutex\n");
        return 1;
    }
    if (pthread_cond_init(&SV.q.cv, NULL) != 0) {
        fprintf(stderr, "error: cannot initialize server queue condition\n");
        pthread_mutex_destroy(&SV.q.mu);
        return 1;
    }

    if (SV.n_reg > 0) {
        // swap mode: models are loaded on demand
        slot_t *s = &SV.slots[0];
        s->id = 0;
        s->smp = defaults;
        s->smp_base = defaults;
        if (!init_swap_runtime(mp, n_threads, ttl)) return 1;
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
            s->smp_base = s->smp;
            if (i == 0) {
                s->m = base;
                tpool *replacement = tpool_create(threads_per_slot);
                if (!replacement) {
                    fprintf(stderr, "error: cannot create pool for slot 0\n");
                    return 1;
                }
                tpool_destroy(base->tp); // replace the single-thread load pool
                base->tp = replacement;
            } else {
                s->m = calloc(1, sizeof(model_t));
                if (!s->m) {
                    fprintf(stderr, "error: cannot allocate slot %d model\n", i);
                    return 1;
                }
                if (!model_load(s->m, model_path, &slot_mp)) {
                    fprintf(stderr, "error: failed to load slot %d\n", i);
                    return 1;
                }
            }
            engine_init(&s->e, s->m, s->tok, &s->smp);
            if (draft_path) {
                // per-slot draft context: each slot owns a full draft KV;
                // weights dedupe through the page cache like slot models
                s->e.dm = spec_draft_load(draft_path, s->m, &slot_mp);
                if (s->e.dm) s->e.draft_k = draft_k;
            }
        }

        if (parallel == 1) {
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
            resident_store(0);
            SV.last_used = now_s();
            SV.draft = SV.slots[0].e.dm;
            SV.draft_k = draft_k;
            if (!init_swap_runtime(mp, threads_per_slot, ttl)) return 1;
        }
    }

    int lfd = (int)socket(AF_INET, SOCK_STREAM, 0);
    if (lfd < 0) {
        fprintf(stderr, "error: cannot create server socket\n");
        return 1;
    }
    int one = 1;
    setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, (const char *)&one, sizeof(one));
    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons((uint16_t)port);
    if (bind(lfd, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
        fprintf(stderr, "error: cannot bind 127.0.0.1:%d (%s)\n", port, strerror(errno));
        sock_close(lfd);
        return 1;
    }
    if (listen(lfd, 64) != 0) {
        fprintf(stderr, "error: cannot listen on 127.0.0.1:%d\n", port);
        sock_close(lfd);
        return 1;
    }

    for (int i = 0; i < parallel; i++) {
        if (pthread_create(&SV.slots[i].th, NULL, slot_worker, &SV.slots[i]) != 0) {
            fprintf(stderr, "error: cannot start server slot %d\n", i);
            sock_close(lfd);
            return 1;
        }
    }

    if (SV.n_reg > 0 && !SV.single)
        fprintf(stderr,
                "server listening on http://127.0.0.1:%d — %d models, swap on demand"
                " (ttl %ds)\n"
                "  POST /v1/chat/completions | POST /v1/completions | GET /v1/models | GET /v1/capabilities | GET /health\n",
                port, SV.n_reg, SV.ttl);
    else
        fprintf(stderr,
                "server listening on http://127.0.0.1:%d — %d slot%s x %d threads\n"
                "  POST /v1/chat/completions | POST /v1/completions | GET /v1/models | GET /v1/capabilities | GET /health\n",
                port, parallel, parallel > 1 ? "s" : "", threads_per_slot);

    for (;;) {
        int cfd = (int)accept(lfd, NULL, NULL);
        if (cfd < 0) {
            if (errno == EINTR) continue;
            break;
        }
        if (!accept_fastpath(cfd)) q_push(cfd);
    }
    return 0;
}
