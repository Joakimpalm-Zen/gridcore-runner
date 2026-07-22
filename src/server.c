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

// Parse the two HTTP fields that determine request framing. Header names are
// compared as complete, line-anchored fields: text inside another field's
// value must never influence framing. Runner does not implement transfer
// codings, and multiple Content-Length fields are rejected even when equal so
// every accepted request has one unambiguous interpretation.
static bool parse_request_framing(char *first_header, char *header_end,
                                  size_t *content_length) {
    bool saw_length = false;
    *content_length = 0;
    for (char *line = first_header; line < header_end;) {
        char *end = strstr(line, "\r\n");
        if (!end || end > header_end) return false;
        char *colon = memchr(line, ':', (size_t)(end - line));
        if (colon) {
            size_t name_n = (size_t)(colon - line);
            bool is_length = name_n == 14 &&
                             ci_ncmp(line, "Content-Length", 14) == 0;
            bool is_transfer = name_n == 17 &&
                               ci_ncmp(line, "Transfer-Encoding", 17) == 0;
            if (is_transfer) return false;
            if (is_length) {
                if (saw_length) return false;
                saw_length = true;
                char *p = colon + 1;
                while (p < end && (*p == ' ' || *p == '\t')) p++;
                char *digits = p;
                size_t value = 0;
                while (p < end && *p >= '0' && *p <= '9') {
                    size_t digit = (size_t)(*p - '0');
                    if (value > ((size_t)-1 - digit) / 10) return false;
                    value = value * 10 + digit;
                    p++;
                }
                if (p == digits) return false;
                while (p < end && (*p == ' ' || *p == '\t')) p++;
                if (p != end) return false;
                *content_length = value;
            }
        }
        line = end + 2;
    }
    return true;
}

static bool parse_request_line(char *hdr, char method[8], char path[256],
                               char **first_header) {
    char *end = strstr(hdr, "\r\n");
    if (!end) return false;
    *end = 0;
    char *sp1 = strchr(hdr, ' ');
    char *sp2 = sp1 ? strchr(sp1 + 1, ' ') : NULL;
    bool shape = sp1 && sp1 != hdr && sp2 && sp2 != sp1 + 1 &&
                 sp2[1] != 0 && strchr(sp2 + 1, ' ') == NULL;
    size_t method_n = shape ? (size_t)(sp1 - hdr) : 0;
    size_t path_n = shape ? (size_t)(sp2 - sp1 - 1) : 0;
    bool valid = shape && method_n < 8 && path_n < 256 &&
                 (!strcmp(sp2 + 1, "HTTP/1.1") ||
                  !strcmp(sp2 + 1, "HTTP/1.0"));
    if (valid) {
        memcpy(method, hdr, method_n);
        method[method_n] = 0;
        memcpy(path, sp1 + 1, path_n);
        path[path_n] = 0;
    }
    *end = '\r';
    *first_header = end + 2;
    return valid;
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
    int  head, tail, count, limit;
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
    atomic_int  active_requests;
    // Loading state machine: swap_to releases swap_mu for the duration of the
    // actual model load (minutes for big files, up to --wait-for-vram longer),
    // so /unload and shutdown are never blocked behind a load. `loading` and
    // `pending_unload` are protected by swap_mu; `load_cancel` is a plain
    // flag the vram wait polls without any lock.
    bool         loading;         // a swap_to load is in flight, lock released
    bool         pending_unload;  // /unload arrived while busy; honoured at the next safe point
    volatile int load_cancel;     // aborts a --wait-for-vram queue wait
    atomic_bool shutdown;
    pthread_t   reaper_th;
    bool        reaper_started;
    model_t    *draft;        // per-slot draft would be plural; single-model
    int         draft_k;      // serve has exactly one slot (see swap_to)
    // the --draft argv string, kept only when its startup load succeeded, so
    // a swap_to() after /unload can bring the draft back with the target
    const char *draft_path;
    // sampling defaults come from the served model's family preset; the CLI
    // overrides are kept so a swapped-in model can be re-resolved against them
    sampler_override ov;
    const char *preset_name;
    // default wall-clock ceiling per generation, 0 = none (RUNNER_REQUEST_TIMEOUT)
    double      req_timeout;
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
    s->e.m = NULL;
    s->e.tok = NULL;
    context_store(0);
    resident_store(-1);
}

static void unload_draft(void) {
    if (!SV.draft) return;
    model_free(SV.draft);
    free(SV.draft);
    SV.draft = NULL;
    for (int i = 0; i < SV.n_slots; i++) SV.slots[i].e.dm = NULL;
}

// Answer GET /unload from any thread. The memory is given back immediately
// when the server is idle; when a load or a generation is in flight the wish
// is recorded and honoured at the next safe boundary instead of blocking the
// caller — /unload exists so an operator can reclaim memory, and "scheduled"
// answered now beats "done" answered after a 300s --wait-for-vram queue. An
// in-flight load is additionally cancelled at its next wait poll.
static void handle_unload(int fd) {
    bool deferred = false;
    if (SV.n_reg > 0) {
        pthread_mutex_lock(&SV.swap_mu);
        if (SV.loading) {
            SV.pending_unload = true;
            SV.load_cancel = 1;   // a queued --wait-for-vram load gives up now
            deferred = true;
        } else if (atomic_load(&SV.active_requests) > 0) {
            SV.pending_unload = true;   // freed when the last request finishes
            deferred = true;
        } else {
            unload_draft();
            unload_resident();
        }
        pthread_mutex_unlock(&SV.swap_mu);
    }
    // /unload means "give the memory back now", so the snapshots go too.
    // A model *swap* deliberately keeps them: surviving a swap is the
    // whole point of snapshotting a prefix instead of holding a slot.
    prefix_cache_clear();
    const char *b = deferred ? "{\"status\":\"ok\",\"deferred\":true}"
                             : "{\"status\":\"ok\"}";
    send_response(fd, 200, "application/json", b, strlen(b));
}

// swap_to results below 0: the name matched no registry entry (a caller
// typo — 400) vs the entry exists but its model failed to load (a broken
// model — 5xx) vs the load was discarded because /unload or shutdown arrived
// while it ran (nobody's fault — 503). Callers must tell them apart.
#define SWAP_UNKNOWN     (-1)
#define SWAP_LOAD_FAILED (-2)
#define SWAP_ABORTED     (-3)

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
    // A request supersedes any unload that was still pending: it is about to
    // (re)load a model on purpose.
    SV.pending_unload = false;
    if (resident_load() != idx) {
        unload_resident();
        // The load itself runs WITHOUT swap_mu: a multi-GB mmap+upload — or a
        // --wait-for-vram queue wait of minutes — must never hold the lock
        // /unload and shutdown need. While `loading` is set, the resident is
        // -1 and slot 0 has no model, so nothing else touches engine state;
        // the load builds into locals and installs under the lock at the end.
        SV.loading = true;
        SV.load_cancel = 0;
        pthread_mutex_unlock(&SV.swap_mu);

        fprintf(stderr, "swap: loading %s (%s)\n",
                SV.reg[idx].name, SV.reg[idx].path);
        model_t *m = calloc(1, sizeof(model_t));
        tokenizer *tok = calloc(1, sizeof(tokenizer));
        bool model_ok = m && model_load(m, SV.reg[idx].path, &SV.mp);
        bool tok_ok = model_ok && tok && tokenizer_init(tok, &m->gf);

        pthread_mutex_lock(&SV.swap_mu);
        SV.loading = false;
        bool discard = SV.pending_unload || SV.load_cancel;
        SV.pending_unload = false;
        if (!model_ok || !tok_ok || discard) {
            if (discard)
                fprintf(stderr, "swap: load of %s discarded (%s)\n",
                        SV.reg[idx].name, model_ok ? "unloaded while loading"
                                                   : "wait cancelled");
            else
                fprintf(stderr, "swap: failed to load %s\n", SV.reg[idx].name);
            if (tok_ok) tokenizer_free(tok);
            if (model_ok) model_free(m);
            free(m); free(tok);
            pthread_mutex_unlock(&SV.swap_mu);
            return discard ? SWAP_ABORTED : SWAP_LOAD_FAILED;
        }
        slot_t *s = &SV.slots[0];
        s->m = m;
        s->tok = tok;
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
        if (SV.single && !SV.draft && SV.draft_path) {
            // /unload freed the draft with the target; a draft configured at
            // startup comes back with the reload rather than staying silently
            // disabled. spec_draft_load re-runs the same gates and VRAM
            // claim the startup load did (model_free released that claim).
            SV.draft = spec_draft_load(SV.draft_path, s->m, &SV.mp);
        }
        if (SV.single && SV.draft) {
            // engine_init memsets the engine; the draft (own KV, own pool)
            // survives target unload/reload and is re-attached here
            s->e.dm = SV.draft;
            s->e.draft_k = SV.draft_k;
        }
        resident_store(idx);
    }
    SV.model_name = SV.reg[idx].name;
    pthread_mutex_unlock(&SV.swap_mu);
    return idx;
}

static void *ttl_reaper(void *arg) {
    (void)arg;
    while (!atomic_load(&SV.shutdown)) {
        struct timespec ts = { 5, 0 };
        nanosleep(&ts, NULL);
        if (resident_load() < 0 || atomic_load(&SV.active_requests)) continue;
        if (pthread_mutex_trylock(&SV.swap_mu) != 0) continue;
        int ttl = SV.ttl;
        if (ttl > 0 && resident_load() >= 0 && !atomic_load(&SV.active_requests) &&
            now_s() - SV.last_used > ttl)
            unload_resident();
        pthread_mutex_unlock(&SV.swap_mu);
    }
    return NULL;
}

static bool start_reaper(void) {
    bool ok = pthread_create(&SV.reaper_th, NULL, ttl_reaper, NULL) == 0;
    SV.reaper_started = ok;
    if (!ok) fprintf(stderr, "error: cannot start model TTL reaper\n");
    return ok;
}

static bool init_swap_runtime(const model_params *mp, int n_threads, int ttl) {
    SV.ttl = ttl;
    SV.mp = *mp;
    SV.mp.verbose = false;
    SV.mp.n_threads = n_threads;
    // swap loads run without swap_mu; /unload and shutdown cancel a queued
    // --wait-for-vram wait through this flag
    SV.mp.load_cancel = &SV.load_cancel;
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

// Admission. The queue is the server's backpressure: past its limit a client
// is told it was shed rather than having its connection dropped silently,
// because "503, retry" is something an agent runtime can act on and an
// unexplained EOF is not.
static void q_push(int fd) {
    pthread_mutex_lock(&SV.q.mu);
    bool room = SV.q.count < SV.q.limit;
    if (room) {
        SV.q.fds[SV.q.tail] = fd;
        SV.q.tail = (SV.q.tail + 1) % (int)(sizeof(SV.q.fds) / sizeof(int));
        SV.q.count++;
        pthread_cond_signal(&SV.q.cv);
    }
    pthread_mutex_unlock(&SV.q.mu);
    if (!room) {
        send_error(fd, 503, "server queue full");
        sock_close(fd);
    }
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

// ------------------------------------------------- continuous batching
//
// Throughput problem: a decode step is weight-bandwidth bound, so N slots
// decoding independently read the same 2.5 GB of weights N times to produce N
// tokens. model_batch_decode reads them once for all N. The scheduler's whole
// job is to arrange for as many sequences as possible to be at the same point
// — "needs one token" — at the same instant, and hand them over together.
//
// The shape is forced by one constraint: a microbatch borrows its lead
// sequence's activation scratch and CUDA stream, so no slot may run its own
// model_forward while a batch containing it is in flight. Hence ONE decode
// thread issuing one step at a time, rather than a thread per slot.
//
// What stays on the slot threads is everything else: HTTP, prompt building,
// prefill, and — critically — the per-token sampler / schema / stop /
// streaming work, which is pure CPU and touches no shared model state. Slot
// threads run that concurrently while the decode thread is already gathering
// the next microbatch, and a slow client blocking on a socket write therefore
// stalls its own request instead of everyone's.
//
// A sequence is named per step, never enrolled (see model_batch_decode), so
// admitting, finishing, cancelling or timing one out is only ever a change to
// which slots are in SEQ_WAIT when the next batch is cut. Nothing is torn down.

enum {
    SEQ_OFF = 0,   // slot is not generating
    SEQ_WAIT,      // has a token to forward, waiting to be picked up
    SEQ_INFLIGHT,  // in the microbatch the decode thread is running now
    SEQ_READY      // its logits have been copied back; slot thread's turn
};

typedef struct {
    int      state;
    int32_t  tok;
    int      pos;
    float   *logits;   // slot-owned copy of this sequence's last logit row
    bool     ok;       // last forward succeeded
} seq_t;

// How long the decode thread will wait for a straggler before cutting the
// batch anyway. The cost model says a step is flat within a width class, so
// waiting for a sequence that is *already generating* is nearly free — but
// waiting for one that does not exist is pure added latency, which is why the
// wait condition below is "every active sequence is here", not a fixed timer.
// This is only the safety net for a slot thread stuck on a slow socket write.
#define SCHED_GATHER_S 0.002

static struct {
    model_batch    *batch;
    seq_t          *seq;        // [n_slots]
    int             n_wait;     // sequences in SEQ_WAIT
    int             n_active;   // sequences inside a generate loop
    bool            stop, running;
    pthread_mutex_t mu;
    pthread_cond_t  dec_cv;     // decode thread waits for work
    pthread_cond_t  slot_cv;    // slot threads wait for their logits
    pthread_t       th;
    // The device turn. Prefill is scheduled separately from decode — different
    // kernel shapes — but the two must INTERLEAVE, not overlap: a microbatch
    // captures a CUDA graph on its lead sequence's stream, and graph capture
    // fails if any other kernel is launched in the context while it is open.
    // A slot prefilling on its own stream is exactly such a launch, and the
    // backend then reports "batch graph capture failed" and degrades to plain
    // launches or a CPU fallback. So prefill and decode take turns here.
    //
    // Serializing them costs nothing real anyway: both saturate the device,
    // so overlapping them only trades one's latency for the other's.
    pthread_mutex_t dev_mu;
    // metrics
    atomic_long steps, seqs_batched;
} SCH;

static bool sched_on(void) { return SCH.running; }

// Absolute deadline for pthread_cond_timedwait, which measures against
// CLOCK_REALTIME — deliberately not now_s(), which is monotonic.
static void ts_in(struct timespec *ts, double secs) {
    timespec_get(ts, TIME_UTC);
    long ns = ts->tv_nsec + (long)(secs * 1e9);
    ts->tv_sec  += ns / 1000000000L;
    ts->tv_nsec  = ns % 1000000000L;
}

static void *decode_worker(void *unused) {
    (void)unused;
    int n = SV.n_slots;
    int     *idx = malloc(sizeof(int) * (size_t)n);
    int32_t *tk  = malloc(sizeof(int32_t) * (size_t)n);
    int     *ps  = malloc(sizeof(int) * (size_t)n);
    float  **out = malloc(sizeof(float *) * (size_t)n);
    if (!idx || !tk || !ps || !out) { free(idx); free(tk); free(ps); free(out); return NULL; }
    int n_vocab = SV.slots[0].m->n_vocab;

    pthread_mutex_lock(&SCH.mu);
    for (;;) {
        while (!SCH.stop && SCH.n_wait == 0)
            pthread_cond_wait(&SCH.dec_cv, &SCH.mu);
        if (SCH.stop) break;

        // Fill the microbatch before cutting it, but only ever wait on
        // sequences that actually exist: if everyone active is already here,
        // this loop does not run at all and a lone request pays nothing.
        int cap = model_batch_max();
        for (;;) {
            int want = SCH.n_active < cap ? SCH.n_active : cap;
            if (SCH.stop || SCH.n_wait >= want) break;
            struct timespec ts;
            ts_in(&ts, SCHED_GATHER_S);
            if (pthread_cond_timedwait(&SCH.dec_cv, &SCH.mu, &ts) == ETIMEDOUT)
                break;
        }

        int nb = 0;
        for (int i = 0; i < n; i++) {
            if (SCH.seq[i].state != SEQ_WAIT) continue;
            SCH.seq[i].state = SEQ_INFLIGHT;
            idx[nb] = i;
            tk[nb]  = SCH.seq[i].tok;
            ps[nb]  = SCH.seq[i].pos;
            nb++;
        }
        SCH.n_wait -= nb;
        if (nb == 0) continue;
        pthread_mutex_unlock(&SCH.mu);

        // Take the device turn. SCH.mu is deliberately released first: these
        // two locks are never held together, in either order.
        pthread_mutex_lock(&SCH.dev_mu);
        bool ok = model_batch_decode(SCH.batch, idx, tk, ps, nb, out);
        // out[i] is only valid until the next decode, and the slot threads
        // read it after this one returns — so each row is copied home first.
        if (ok)
            for (int i = 0; i < nb; i++)
                memcpy(SCH.seq[idx[i]].logits, out[i],
                       sizeof(float) * (size_t)n_vocab);
        pthread_mutex_unlock(&SCH.dev_mu);
        atomic_fetch_add(&SCH.steps, 1);
        atomic_fetch_add(&SCH.seqs_batched, nb);

        pthread_mutex_lock(&SCH.mu);
        for (int i = 0; i < nb; i++) {
            SCH.seq[idx[i]].ok    = ok;
            SCH.seq[idx[i]].state = SEQ_READY;
        }
        pthread_cond_broadcast(&SCH.slot_cv);
    }
    pthread_mutex_unlock(&SCH.mu);
    free(idx); free(tk); free(ps); free(out);
    return NULL;
}

// Batching is enabled only for plain multi-slot serving. Swap mode reloads
// slot 0's model underneath the batch's borrowed pointers, and a single slot
// has nothing to batch with, so both keep the untouched solo path.
static bool sched_start(void) {
    if (SV.n_slots < 2 || SV.n_reg > 0) return false;
    model_t **ms = malloc(sizeof(model_t *) * (size_t)SV.n_slots);
    SCH.seq = calloc((size_t)SV.n_slots, sizeof(seq_t));
    if (!ms || !SCH.seq) { free(ms); free(SCH.seq); SCH.seq = NULL; return false; }
    for (int i = 0; i < SV.n_slots; i++) ms[i] = SV.slots[i].m;
    int n_vocab = SV.slots[0].m->n_vocab;
    for (int i = 0; i < SV.n_slots; i++) {
        SCH.seq[i].logits = malloc(sizeof(float) * (size_t)n_vocab);
        if (!SCH.seq[i].logits) { free(ms); return false; }
    }
    SCH.batch = model_batch_create(ms, SV.n_slots);
    free(ms);
    if (!SCH.batch) return false;
    if (pthread_mutex_init(&SCH.mu, NULL) != 0) return false;
    if (pthread_mutex_init(&SCH.dev_mu, NULL) != 0) return false;
    if (pthread_cond_init(&SCH.dec_cv, NULL) != 0) return false;
    if (pthread_cond_init(&SCH.slot_cv, NULL) != 0) return false;
    SCH.running = true;
    if (pthread_create(&SCH.th, NULL, decode_worker, NULL) != 0) {
        SCH.running = false;
        return false;
    }
    return true;
}

// Prefill runs per sequence on model_forward_batch, interleaved between decode
// steps rather than inside them: a prefill tile and a decode microbatch are
// different kernel shapes. The sequence is simply absent from the batch while
// this is held, which costs nothing — it is named per step, not enrolled.
static void sched_prefill_begin(void) {
    if (sched_on()) pthread_mutex_lock(&SCH.dev_mu);
}

static void sched_prefill_end(void) {
    if (sched_on()) pthread_mutex_unlock(&SCH.dev_mu);
}

// Hand one forward to the decode thread and wait for this sequence's logits.
// Returns NULL if the sequence should stop: the batch failed, the server is
// shutting down, or this request ran out its deadline while queued.
static const float *sched_forward(int id, int32_t tok, int pos, double deadline) {
    seq_t *q = &SCH.seq[id];
    pthread_mutex_lock(&SCH.mu);
    q->tok   = tok;
    q->pos   = pos;
    q->state = SEQ_WAIT;
    SCH.n_wait++;
    pthread_cond_signal(&SCH.dec_cv);
    while (q->state != SEQ_READY && !SCH.stop) {
        if (deadline > 0 && q->state == SEQ_WAIT && now_s() >= deadline) {
            // Still only queued, so dropping it is free: take it back out of
            // the pending count and let the caller close its document. A
            // sequence already INFLIGHT is waited out instead — abandoning a
            // buffer the decode thread is about to write is not free.
            q->state = SEQ_OFF;
            SCH.n_wait--;
            pthread_mutex_unlock(&SCH.mu);
            return NULL;
        }
        struct timespec ts;
        ts_in(&ts, deadline > 0 ? SCHED_GATHER_S : 0.5);
        pthread_cond_timedwait(&SCH.slot_cv, &SCH.mu, &ts);
    }
    bool ok = q->state == SEQ_READY && q->ok;
    q->state = SEQ_OFF;
    pthread_mutex_unlock(&SCH.mu);
    return ok ? q->logits : NULL;
}

// engine_generate's contract, served out of shared microbatches.
//
// The per-token work is the engine's own — the same engine_gen_step the solo
// path runs — so a request's tokens cannot depend on how busy the server was.
// All this adds is where the forward happens.
static int sched_generate(slot_t *s, float *logits, int max_new,
                          gen_cb cb, void *ud, double *gen_time,
                          double deadline) {
    engine *e = &s->e;
    // Speculative decoding drives its own batched forwards for one sequence,
    // which is the tile shape and not the microbatch shape; it stays solo.
    if (!sched_on() || e->dm) {
        // A solo generation on a batching server still has to take the device
        // turn: its model_forward launches would break a concurrent
        // microbatch's graph capture just as a prefill would.
        if (!sched_on())
            return engine_generate(e, logits, max_new, cb, ud, gen_time);
        pthread_mutex_lock(&SCH.dev_mu);
        int n = engine_generate(e, logits, max_new, cb, ud, gen_time);
        pthread_mutex_unlock(&SCH.dev_mu);
        return n;
    }

    pthread_mutex_lock(&SCH.mu);
    SCH.n_active++;
    pthread_mutex_unlock(&SCH.mu);

    const float *lg = logits;
    engine_gen_begin(e, max_new);
    int32_t tok; int pos;
    while (engine_gen_step(e, lg, cb, ud, &tok, &pos) == ENGINE_STEP_MORE) {
        if (deadline > 0 && now_s() >= deadline) break;
        if (!(lg = sched_forward(s->id, tok, pos, deadline))) break;
    }
    int n = engine_gen_end(e, cb, ud, gen_time);

    pthread_mutex_lock(&SCH.mu);
    SCH.n_active--;
    // one fewer sequence to wait for: whoever is gathering should stop holding
    pthread_cond_signal(&SCH.dec_cv);
    pthread_mutex_unlock(&SCH.mu);
    return n;
}

// ---------------------------------------------------------------- generation

// Which wire dialect this request is answered in. All four run the *same*
// generation path; they differ only in how the result is framed, which is why
// the Responses and Messages surfaces are translation layers rather than
// second engines.
enum { API_TEXT, API_CHAT, API_RESPONSES, API_MESSAGES };

typedef struct {
    sbuf  out;          // accumulated completion text
    sbuf  reason;       // accumulated reasoning text (thinking-tag models)
    int   fd;
    bool  stream;       // SSE mode
    bool  dead;         // client went away
    char  id[48];
    int   api;          // API_* — the dialect this response is framed in
    think_split ts;     // thinking-tag splitter (pass-through when untagged)
    // OpenAI "stop" sequences: matched against the content channel only
    // (reasoning text must not trigger a client's stop strings)
    const char **stop_strs; // borrowed from the request jv, valid all request
    int   n_stop;
    sbuf  hold;         // held-back tail that may still begin a stop match
    bool  stopped;      // a stop sequence matched; excluded from output
    const char *stop_hit; // which one matched (Anthropic reports it back)
    long  created;      // stamped once: every chunk of a stream reports the
                        // same creation time, as the buffered body does
    bool  role_sent;    // "role":"assistant" is emitted on the first delta only
    // strict tool envelope, streaming: the generated document is demuxed into
    // content and tool_calls deltas as it arrives, so the envelope itself is
    // never what the client receives
    tool_stream tsx;
    bool  tsx_on;
    int   tool_index;   // OpenAI tool_calls[].index; one call per turn for now
    // Responses streaming: the typed events are ordered and each one carries a
    // monotonic sequence_number, so the emitter is a small state machine over
    // "which item / content part is currently open" rather than a formatter.
    long  seq;          // next sequence_number
    int   output_index; // index of the item being streamed
    bool  item_open;    // an output_item.added has no matching .done yet
    bool  part_open;    // likewise for content_part.added
    char  item_id[48];  // id of the open item (msg_/fc_)
    sbuf  item_text;    // its text so far, replayed in the .done events
    const char *item_kind; // "message" or "function_call"
    char *call_name;    // function_call name, once known (owned)
    // items are accumulated as they complete so the terminal event can report
    // the same `output[]` a buffered request would have returned; a client
    // that only reads response.completed must not see a different turn than
    // one that followed the deltas
    sbuf  out_items;
    sbuf  out_text;     // the `output_text` aggregate (assistant text only)
    // status to close the *last* item with. An item that ends because another
    // one starts did finish; one that ends because generation was cut short
    // did not, and must say so — a client rendering output[] without reading
    // the response status would otherwise show a truncated message as whole.
    const char *close_status;
} gen_ctx;

// common prefix of every streamed chunk. `created` and `model` are required by
// the ChatCompletionChunk schema and strictly-validating SDKs reject a chunk
// without them, so they are written here rather than per call site.
static void chunk_open(gen_ctx *g, sbuf *c) {
    sb_fmt(c, "{\"id\":\"%s\",\"object\":\"%s\",\"created\":%ld,\"model\":\"",
           g->id, g->api == API_CHAT ? "chat.completion.chunk" : "text_completion",
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
    e->emit_think_prelude = false;
    schema_free(schema);
    if (g) {
        tool_stream_free(&g->tsx);
        think_free(&g->ts);
        free(g->hold.s);
        free(g->reason.s);
        free(g->out.s);
        free(g->item_text.s);
        free(g->call_name);
        free(g->out_items.s);
        free(g->out_text.s);
    }
    free(e->lp_chosen); free(e->lp_ids); free(e->lp_top);
    e->lp_chosen = NULL; e->lp_ids = NULL; e->lp_top = NULL;
    e->lp_cap = e->lp_n = e->lp_count = 0;
}

// emit one section of split output: reasoning goes to the OpenAI-style
// reasoning_content field, everything else to content
// one text delta on the named chat channel (or the legacy completion "text")
static int responses_text_delta(gen_ctx *g, int reasoning, const char *bytes,
                                int n);
static int anth_delta(gen_ctx *g, const char *kind, const char *bytes, int n);

static int send_text_delta(gen_ctx *g, int reasoning, const char *bytes, int n) {
    if (!g->stream || g->dead) return g->dead ? 1 : 0;
    if (g->api == API_RESPONSES) return responses_text_delta(g, reasoning, bytes, n);
    // Anthropic separates reasoning into a `thinking` content block rather
    // than a field on the message, so the channel selects the block kind
    if (g->api == API_MESSAGES)
        return anth_delta(g, reasoning ? "thinking" : "text", bytes, n);
    sbuf c = {0};
    chunk_open(g, &c);
    if (g->api == API_CHAT) {
        sb_lit(&c, "\"delta\":{");
        if (!g->role_sent) { sb_lit(&c, "\"role\":\"assistant\","); g->role_sent = true; }
        sb_fmt(&c, "\"%s\":\"", reasoning ? "reasoning_content" : "content");
    } else {
        sb_lit(&c, "\"text\":\"");
    }
    sb_esc(&c, bytes, n);
    sb_lit(&c, g->api == API_CHAT ? "\"},\"finish_reason\":null}]}"
                                  : "\",\"finish_reason\":null}]}");
    return chunk_send(g, &c);
}

// ---- tool_stream sinks: the demuxed envelope, as OpenAI streaming events

static int sink_content(void *ud, const char *b, int n) {
    return send_text_delta(ud, 0, b, n);
}

// the opening event of a call carries everything that identifies it; the
// deltas that follow carry argument text only, keyed by the same index
static int resp_open_item(gen_ctx *g, const char *kind);
static int resp_delta(gen_ctx *g, const char *kind, const char *bytes, int n);

static int anth_open_block(gen_ctx *g, const char *kind);

static int sink_call_begin(void *ud, const char *name) {
    gen_ctx *g = ud;
    if (g->dead) return 1;
    if (g->api == API_RESPONSES || g->api == API_MESSAGES) {
        // the name identifies the item/block, so it must be known before that
        // is announced — which is exactly when tool_stream calls this
        free(g->call_name);
        g->call_name = strdup(name);
        if (!g->call_name) { g->dead = true; return 1; }
        return g->api == API_MESSAGES ? anth_open_block(g, "tool_use")
                                      : resp_open_item(g, "function_call");
    }
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
    if (g->api == API_RESPONSES) return resp_delta(g, "function_call", b, n);
    if (g->api == API_MESSAGES) return anth_delta(g, "tool_use", b, n);
    sbuf c = {0};
    chunk_open(g, &c);
    sb_fmt(&c, "\"delta\":{\"tool_calls\":[{\"index\":%d,\"function\":"
               "{\"arguments\":\"", g->tool_index);
    sb_esc(&c, b, n);
    sb_lit(&c, "\"}}]},\"finish_reason\":null}]}");
    return chunk_send(g, &c);
}

// ------------------------------------------------- Responses API framing
//
// The Responses surface is a second *vocabulary* for the one generation path,
// not a second engine. Everything below is framing: the same bytes the chat
// dialect sends as ChatCompletionChunk deltas are sent here as ordered typed
// events, and the same buffered result is rendered as an `output[]` of items.
//
// Two properties are the actual contract, and both are what SDK clients
// validate:
//
//   1. Order. An item is announced (`output_item.added`) before any of its
//      deltas and closed (`output_item.done`) after them, with a content part
//      opened and closed inside it; `response.created` opens the stream and
//      `response.completed` closes it.
//   2. Naming. Every event names itself twice — in the SSE `event:` field and
//      in `data.type` — and typed clients dispatch on the first while
//      validating the second, so the two must always agree. Framing them in
//      one place is what guarantees that.
//
// A monotonic `sequence_number` is stamped on every event by the framer for
// the same reason: it cannot drift from the send order if nothing else can
// assign it.

// item kinds share one state machine and differ only in these names
typedef struct {
    const char *kind;       // the item's "type"
    const char *id_prefix;
    const char *part_added, *delta, *text_done, *part_done;
} resp_shape;

static const resp_shape RESP_MESSAGE = {
    "message", "msg", "response.content_part.added",
    "response.output_text.delta", "response.output_text.done",
    "response.content_part.done" };
static const resp_shape RESP_REASONING = {
    "reasoning", "rs", "response.reasoning_summary_part.added",
    "response.reasoning_summary_text.delta",
    "response.reasoning_summary_text.done",
    "response.reasoning_summary_part.done" };
static const resp_shape RESP_CALL = {
    "function_call", "fc", NULL,
    "response.function_call_arguments.delta",
    "response.function_call_arguments.done", NULL };

static const resp_shape *resp_shape_of(const char *kind) {
    if (!kind) return &RESP_MESSAGE;
    if (!strcmp(kind, "reasoning")) return &RESP_REASONING;
    if (!strcmp(kind, "function_call")) return &RESP_CALL;
    return &RESP_MESSAGE;
}

// frame one typed SSE event. `fields` holds the event-specific members already
// written as `,"key":value` pairs; it is consumed (freed) here.
//
// Both typed surfaces name every event twice — in the SSE `event:` field and
// in `data.type` — and both are validated by their SDKs, so the two names are
// written from one argument here and cannot drift apart on either. The only
// difference is `sequence_number`, which the Responses vocabulary stamps on
// every event and the Anthropic one does not have at all.
static int sse_send(gen_ctx *g, const char *type, sbuf *fields, bool seq) {
    sbuf e = {0};
    sb_fmt(&e, "event: %s\ndata: {\"type\":\"%s\"", type, type);
    if (seq) sb_fmt(&e, ",\"sequence_number\":%ld", g->seq++);
    if (fields->s) sb_put(&e, fields->s, fields->n);
    sb_lit(&e, "}\n\n");
    if (fields->failed || e.failed || !send_all(g->fd, e.s, e.n)) g->dead = true;
    free(fields->s);
    free(e.s);
    return g->dead ? 1 : 0;
}

static int resp_send(gen_ctx *g, const char *type, sbuf *fields) {
    return sse_send(g, type, fields, true);
}

static int anth_send(gen_ctx *g, const char *type, sbuf *fields) {
    return sse_send(g, type, fields, false);
}

// the open item, rendered as a Responses output item
static void resp_item_json(sbuf *b, gen_ctx *g, const char *status, bool filled) {
    const resp_shape *sh = resp_shape_of(g->item_kind);
    const char *text = g->item_text.s ? g->item_text.s : "";
    size_t text_n = g->item_text.n;
    sb_fmt(b, "{\"id\":\"%s\",\"type\":\"%s\",\"status\":\"%s\"",
           g->item_id, sh->kind, status);
    if (sh == &RESP_CALL) {
        sb_fmt(b, ",\"call_id\":\"call_%d\",\"name\":\"", g->tool_index);
        sb_esc(b, g->call_name ? g->call_name : "", strlen(g->call_name ? g->call_name : ""));
        // arguments are already a JSON *string* on the wire, so the accumulated
        // argument text is escaped into it exactly as the chat dialect does
        sb_lit(b, "\",\"arguments\":\"");
        if (filled) sb_esc(b, text, text_n);
        sb_lit(b, "\"}");
        return;
    }
    if (sh == &RESP_REASONING) {
        sb_lit(b, ",\"summary\":[");
        if (filled) {
            sb_lit(b, "{\"type\":\"summary_text\",\"text\":\"");
            sb_esc(b, text, text_n);
            sb_lit(b, "\"}");
        }
        sb_lit(b, "]}");
        return;
    }
    sb_lit(b, ",\"role\":\"assistant\",\"content\":[");
    if (filled) {
        sb_lit(b, "{\"type\":\"output_text\",\"text\":\"");
        sb_esc(b, text, text_n);
        sb_lit(b, "\",\"annotations\":[]}");
    }
    sb_lit(b, "]}");
}

// close whatever item is open: part done (when the kind has parts), then the
// item itself. Nothing is emitted when no item is open, so this is safe to
// call on every path that ends an item — including the terminal one.
static int resp_close_item(gen_ctx *g) {
    if (!g->item_open || g->dead) return g->dead ? 1 : 0;
    const resp_shape *sh = resp_shape_of(g->item_kind);
    const char *text = g->item_text.s ? g->item_text.s : "";
    size_t text_n = g->item_text.n;
    if (g->part_open) {
        sbuf f = {0};
        sb_fmt(&f, ",\"item_id\":\"%s\",\"output_index\":%d,\"content_index\":0,"
                   "\"%s\":\"", g->item_id, g->output_index,
               sh == &RESP_CALL ? "arguments" : "text");
        sb_esc(&f, text, text_n);
        sb_lit(&f, "\"");
        if (sh != &RESP_CALL) sb_lit(&f, ",\"logprobs\":[]");
        if (resp_send(g, sh->text_done, &f)) return 1;
        if (sh->part_done) {
            sbuf p = {0};
            sb_fmt(&p, ",\"item_id\":\"%s\",\"output_index\":%d,"
                       "\"content_index\":0,\"part\":", g->item_id,
                   g->output_index);
            if (sh == &RESP_REASONING) {
                sb_lit(&p, "{\"type\":\"summary_text\",\"text\":\"");
                sb_esc(&p, text, text_n);
                sb_lit(&p, "\"}");
            } else {
                sb_lit(&p, "{\"type\":\"output_text\",\"text\":\"");
                sb_esc(&p, text, text_n);
                sb_lit(&p, "\",\"annotations\":[]}");
            }
            if (resp_send(g, sh->part_done, &p)) return 1;
        }
        g->part_open = false;
    }
    const char *status = g->close_status ? g->close_status : "completed";
    sbuf d = {0};
    sb_fmt(&d, ",\"output_index\":%d,\"item\":", g->output_index);
    resp_item_json(&d, g, status, true);
    // keep the completed item for the terminal response object
    if (g->out_items.n) sb_lit(&g->out_items, ",");
    resp_item_json(&g->out_items, g, status, true);
    if (resp_shape_of(g->item_kind) == &RESP_MESSAGE)
        sb_put(&g->out_text, g->item_text.s ? g->item_text.s : "",
               g->item_text.n);
    int rc = resp_send(g, "response.output_item.done", &d);
    g->item_open = false;
    g->output_index++;
    g->item_text.n = 0;
    return rc;
}

// open an item of `kind`, closing any item already open. The id is derived
// from the kind and the output index so it is stable and collision-free.
static int resp_open_item(gen_ctx *g, const char *kind) {
    if (g->item_open && g->item_kind && !strcmp(g->item_kind, kind)) return 0;
    if (resp_close_item(g)) return 1;
    const resp_shape *sh = resp_shape_of(kind);
    g->item_kind = sh->kind;
    snprintf(g->item_id, sizeof(g->item_id), "%s_%d", sh->id_prefix,
             g->output_index);
    sbuf a = {0};
    sb_fmt(&a, ",\"output_index\":%d,\"item\":", g->output_index);
    resp_item_json(&a, g, "in_progress", false);
    if (resp_send(g, "response.output_item.added", &a)) return 1;
    g->item_open = true;
    // a function_call carries its arguments directly on the item, so it has no
    // content part; the other kinds open one before their first delta
    if (sh->part_added) {
        sbuf p = {0};
        sb_fmt(&p, ",\"item_id\":\"%s\",\"output_index\":%d,\"content_index\":0,"
                   "\"part\":", g->item_id, g->output_index);
        if (sh == &RESP_REASONING)
            sb_lit(&p, "{\"type\":\"summary_text\",\"text\":\"\"}");
        else
            sb_lit(&p, "{\"type\":\"output_text\",\"text\":\"\",\"annotations\":[]}");
        if (resp_send(g, sh->part_added, &p)) return 1;
        g->part_open = true;
    } else {
        g->part_open = true; // the arguments "part" is the item itself
    }
    return 0;
}

static int resp_delta(gen_ctx *g, const char *kind, const char *bytes, int n) {
    if (g->dead) return 1;
    if (resp_open_item(g, kind)) return 1;
    sb_put(&g->item_text, bytes, n);
    const resp_shape *sh = resp_shape_of(kind);
    sbuf f = {0};
    sb_fmt(&f, ",\"item_id\":\"%s\",\"output_index\":%d", g->item_id,
           g->output_index);
    if (sh != &RESP_CALL) sb_lit(&f, ",\"content_index\":0");
    sb_lit(&f, ",\"delta\":\"");
    sb_esc(&f, bytes, n);
    sb_lit(&f, "\"");
    if (sh == &RESP_MESSAGE) sb_lit(&f, ",\"logprobs\":[]");
    return resp_send(g, sh->delta, &f);
}

static int responses_text_delta(gen_ctx *g, int reasoning, const char *bytes,
                                int n) {
    return resp_delta(g, reasoning ? "reasoning" : "message", bytes, n);
}

// Everything the response object reports about one finished (or just-started)
// turn. Passing it as one value is what lets the buffered body and the
// terminal `response.completed` event be the *same* document: a client that
// switches `stream` on and off sees one shape, not two that drifted.
typedef struct {
    const char *status;      // in_progress | completed | incomplete
    const char *incomplete;  // reason, when status is incomplete
    // the same two facts in the Anthropic vocabulary. Both surfaces derive
    // them from the one `finish` string, so they cannot describe the turn
    // differently.
    const char *stop_reason; // end_turn | max_tokens | stop_sequence | tool_use
    const char *stop_seq;    // the matched stop sequence, when that is the reason
    bool        with_output;
    // A streamed turn already rendered its items one by one, so it hands them
    // over verbatim rather than rebuilding them from different inputs — which
    // is what keeps the streamed and buffered documents identical by
    // construction instead of by review.
    const char *output_json; size_t output_n;
    const char *output_text; size_t output_text_n;
    const char *call_name;   // non-NULL when this turn was a tool call
    const char *call_args;
    const char *text;   size_t text_n;
    const char *reason; size_t reason_n;
    bool        with_usage;
    int         n_prompt, n_gen, cached;
    // of `cached`, how many rows were forked out of the shared prefix cache
    // rather than left over in this slot, and what that saved
    int         forked;
    double      saved_s;
    double      gtime;
    bool        schema, json_mode, spec;
    jv         *req;         // echoed request fields
} resp_doc;

// One rendering of runner_telemetry for every surface that carries it — chat,
// completions, responses (streamed and buffered) and messages. They report the
// same facts, so they share the one place those facts are spelled rather than
// four format strings that drift.
static void telemetry_json(sbuf *r, const resp_doc *d) {
    sb_fmt(r, "\"runner_telemetry\":{\"prompt_cached_tokens\":%d,"
              "\"prompt_forked_tokens\":%d,\"prompt_eval_tokens\":%d,"
              "\"prefix_cache_saved_seconds\":%.6f,"
              "\"generation_seconds\":%.6f,"
              "\"generation_tok_s\":%.3f,\"json_mode\":%s,"
              "\"schema\":%s,\"speculative\":%s}",
           d->cached, d->forked, d->n_prompt - d->cached, d->saved_s, d->gtime,
           d->n_gen / (d->gtime > 0 ? d->gtime : 1e-9),
           d->json_mode ? "true" : "false", d->schema ? "true" : "false",
           d->spec ? "true" : "false");
}

static void resp_echo(sbuf *r, jv *req, const char *key, const char *dflt) {
    jv *v = jv_get(req, key);
    sb_fmt(r, ",\"%s\":", key);
    if (!v || v->type == J_NULL) sb_lit(r, dflt);
    else jv_dump(v, r);
}

static void responses_body(sbuf *r, gen_ctx *g, const resp_doc *d) {
    sb_fmt(r, "{\"id\":\"%s\",\"object\":\"response\",\"created_at\":%ld,"
              "\"status\":\"%s\",\"error\":null,\"incomplete_details\":",
           g->id, g->created, d->status);
    if (d->incomplete) sb_fmt(r, "{\"reason\":\"%s\"}", d->incomplete);
    else               sb_lit(r, "null");
    sb_lit(r, ",\"model\":\"");
    sb_esc(r, SV.model_name, strlen(SV.model_name));
    sb_lit(r, "\",\"output\":[");
    if (d->output_json) {
        sb_put(r, d->output_json, d->output_n);
    } else if (d->with_output) {
        int idx = 0;
        if (d->reason_n) {
            sb_fmt(r, "{\"id\":\"rs_%d\",\"type\":\"reasoning\","
                      "\"status\":\"completed\",\"summary\":"
                      "[{\"type\":\"summary_text\",\"text\":\"", idx);
            sb_esc(r, d->reason, d->reason_n);
            sb_lit(r, "\"}]}");
            idx++;
        }
        if (d->call_name) {
            // a tool call replaces the assistant message rather than
            // accompanying it, matching finish_reason "tool_calls"
            if (idx) sb_lit(r, ",");
            sb_fmt(r, "{\"id\":\"fc_%d\",\"type\":\"function_call\","
                      "\"status\":\"completed\",\"call_id\":\"call_0\","
                      "\"name\":\"", idx);
            sb_esc(r, d->call_name, strlen(d->call_name));
            sb_lit(r, "\",\"arguments\":\"");
            sb_esc(r, d->call_args ? d->call_args : "{}",
                   strlen(d->call_args ? d->call_args : "{}"));
            sb_lit(r, "\"}");
        } else {
            if (idx) sb_lit(r, ",");
            sb_fmt(r, "{\"id\":\"msg_%d\",\"type\":\"message\","
                      "\"status\":\"%s\",\"role\":\"assistant\",\"content\":"
                      "[{\"type\":\"output_text\",\"text\":\"", idx,
                   d->incomplete ? "incomplete" : "completed");
            sb_esc(r, d->text ? d->text : "", d->text_n);
            sb_lit(r, "\",\"annotations\":[]}]}");
        }
    }
    sb_lit(r, "],\"output_text\":\"");
    // the SDK's `response.output_text` convenience aggregate: the assistant
    // text only, empty when the turn produced a call instead
    if (d->output_json) sb_esc(r, d->output_text ? d->output_text : "",
                               d->output_text_n);
    else if (d->with_output && !d->call_name)
        sb_esc(r, d->text ? d->text : "", d->text_n);
    sb_lit(r, "\"");
    // request echo: a Responses client reads these back off the object rather
    // than remembering what it sent
    resp_echo(r, d->req, "instructions", "null");
    resp_echo(r, d->req, "metadata", "null");
    resp_echo(r, d->req, "temperature", "null");
    resp_echo(r, d->req, "top_p", "null");
    resp_echo(r, d->req, "max_output_tokens", "null");
    resp_echo(r, d->req, "reasoning", "null");
    resp_echo(r, d->req, "text", "{\"format\":{\"type\":\"text\"}}");
    resp_echo(r, d->req, "tools", "[]");
    resp_echo(r, d->req, "tool_choice", "\"auto\"");
    resp_echo(r, d->req, "parallel_tool_calls", "false");
    sb_lit(r, ",\"previous_response_id\":null,\"store\":false,"
              "\"truncation\":\"disabled\",\"user\":null,\"usage\":");
    if (d->with_usage) {
        sb_fmt(r, "{\"input_tokens\":%d,"
                  "\"input_tokens_details\":{\"cached_tokens\":%d},"
                  "\"output_tokens\":%d,"
                  "\"output_tokens_details\":{\"reasoning_tokens\":0},"
                  "\"total_tokens\":%d}",
               d->n_prompt, d->cached, d->n_gen, d->n_prompt + d->n_gen);
        sb_lit(r, ",");
        telemetry_json(r, d);
    } else {
        sb_lit(r, "null");
    }
    sb_lit(r, "}");
}

// ----------------------------------------------- Anthropic Messages framing
//
// The third vocabulary for the one generation path. Where the Responses
// surface renders the turn as an `output[]` of items, Anthropic renders it as
// a `content[]` of *blocks*, and the streaming form is the same state machine
// one level flatter: a block is opened, filled with deltas, and closed, with no
// content-part nesting inside it.
//
// The six event names and their order are the contract the Anthropic SDKs
// validate:
//
//   message_start
//     content_block_start / content_block_delta* / content_block_stop   (xN)
//   message_delta        — the terminal stop_reason and the output token count
//   message_stop
//
// There is no `data: [DONE]` sentinel: message_stop is the terminator.

// The stop_reason vocabulary, derived from the same `finish` string the chat
// dialect reports, so the two surfaces cannot disagree about how a turn ended.
static const char *anth_stop_reason(const char *finish, bool stop_hit) {
    if (!strcmp(finish, "tool_calls")) return "tool_use";
    if (!strcmp(finish, "length"))     return "max_tokens";
    // a user stop sequence is its own terminal reason in this vocabulary, and
    // the matched string is reported alongside it
    return stop_hit ? "stop_sequence" : "end_turn";
}

static int anth_close_block(gen_ctx *g) {
    if (!g->item_open || g->dead) return g->dead ? 1 : 0;
    sbuf f = {0};
    sb_fmt(&f, ",\"index\":%d", g->output_index);
    int rc = anth_send(g, "content_block_stop", &f);
    g->item_open = false;
    g->output_index++;
    g->item_text.n = 0;
    return rc;
}

// open a block of `kind`, closing any block already open. Re-opening the kind
// that is already open is a no-op, which is what makes a run of deltas on one
// channel land in one block.
static int anth_open_block(gen_ctx *g, const char *kind) {
    if (g->item_open && g->item_kind && !strcmp(g->item_kind, kind)) return 0;
    if (anth_close_block(g)) return 1;
    g->item_kind = kind;
    sbuf f = {0};
    sb_fmt(&f, ",\"index\":%d,\"content_block\":", g->output_index);
    if (!strcmp(kind, "tool_use")) {
        sb_fmt(&f, "{\"type\":\"tool_use\",\"id\":\"toolu_%d\",\"name\":\"",
               g->tool_index);
        sb_esc(&f, g->call_name ? g->call_name : "",
               strlen(g->call_name ? g->call_name : ""));
        // the arguments arrive as input_json_delta text; the block opens with
        // the empty object an SDK accumulator starts from
        sb_lit(&f, "\",\"input\":{}}");
    } else if (!strcmp(kind, "thinking")) {
        sb_lit(&f, "{\"type\":\"thinking\",\"thinking\":\"\"}");
    } else {
        sb_lit(&f, "{\"type\":\"text\",\"text\":\"\"}");
    }
    if (anth_send(g, "content_block_start", &f)) return 1;
    g->item_open = true;
    return 0;
}

static int anth_delta(gen_ctx *g, const char *kind, const char *bytes, int n) {
    if (g->dead) return 1;
    if (anth_open_block(g, kind)) return 1;
    sb_put(&g->item_text, bytes, n);
    sbuf f = {0};
    sb_fmt(&f, ",\"index\":%d,\"delta\":{\"type\":\"", g->output_index);
    // each block kind carries its text under its own delta type and key; an
    // SDK dispatches on the first and reads the second
    if (!strcmp(kind, "tool_use"))
        sb_lit(&f, "input_json_delta\",\"partial_json\":\"");
    else if (!strcmp(kind, "thinking"))
        sb_lit(&f, "thinking_delta\",\"thinking\":\"");
    else
        sb_lit(&f, "text_delta\",\"text\":\"");
    sb_esc(&f, bytes, n);
    sb_lit(&f, "\"}");
    return anth_send(g, "content_block_delta", &f);
}

// The Message object, in the one shape both the buffered body and the
// streamed message_start/message_delta pair are built from. `content_json`,
// when set, is the already-rendered block list a streamed turn accumulated,
// for the same reason the Responses surface hands its items over verbatim:
// the streamed and buffered documents stay identical by construction.
static void anth_body(sbuf *r, gen_ctx *g, const resp_doc *d) {
    sb_fmt(r, "{\"id\":\"%s\",\"type\":\"message\",\"role\":\"assistant\","
              "\"model\":\"", g->id);
    sb_esc(r, SV.model_name, strlen(SV.model_name));
    sb_lit(r, "\",\"content\":[");
    if (d->with_output) {
        int idx = 0;
        if (d->reason_n) {
            // reasoning is a block of its own here rather than a field on the
            // message, and it always precedes the answer
            sb_lit(r, "{\"type\":\"thinking\",\"thinking\":\"");
            sb_esc(r, d->reason, d->reason_n);
            sb_lit(r, "\",\"signature\":\"\"}");
            idx++;
        }
        if (d->call_name) {
            if (idx++) sb_lit(r, ",");
            sb_fmt(r, "{\"type\":\"tool_use\",\"id\":\"toolu_%d\",\"name\":\"",
                   g->tool_index);
            sb_esc(r, d->call_name, strlen(d->call_name));
            // Anthropic carries the arguments as a JSON *object*, where OpenAI
            // carries the same document as a string. That difference is
            // load-bearing: a string can hold anything, an inlined object
            // cannot. Under the strict envelope the document is guaranteed to
            // parse, but a call recovered from free text is only
            // brace-matched, so it can be balanced and still invalid —
            // inlining that verbatim would emit a body no client can read.
            // Re-dumping through the parser is what makes this total.
            sb_lit(r, "\",\"input\":");
            const char *args = d->call_args ? d->call_args : "{}";
            jv *parsed = json_parse(args, strlen(args));
            if (parsed && parsed->type == J_OBJ) jv_dump(parsed, r);
            else                                 sb_lit(r, "{}");
            jv_free(parsed);
            sb_lit(r, "}");
        } else if (d->text_n || !idx) {
            // an empty text block is still emitted when it is the only thing
            // the turn produced: content[] must never be empty
            if (idx++) sb_lit(r, ",");
            sb_lit(r, "{\"type\":\"text\",\"text\":\"");
            sb_esc(r, d->text ? d->text : "", d->text_n);
            sb_lit(r, "\"}");
        }
    }
    sb_lit(r, "],\"stop_reason\":");
    if (d->stop_reason) sb_fmt(r, "\"%s\"", d->stop_reason);
    else                sb_lit(r, "null");
    sb_lit(r, ",\"stop_sequence\":");
    if (d->stop_seq) {
        sb_lit(r, "\"");
        sb_esc(r, d->stop_seq, strlen(d->stop_seq));
        sb_lit(r, "\"");
    } else {
        sb_lit(r, "null");
    }
    // Anthropic's cache_* usage fields describe *its* prompt-caching product
    // and are deliberately not claimed here; runner's prefix-cache figure is
    // reported in runner_telemetry, as it is on every other surface.
    sb_fmt(r, ",\"usage\":{\"input_tokens\":%d,\"output_tokens\":%d}",
           d->n_prompt, d->n_gen);
    if (d->with_usage) {
        sb_lit(r, ",");
        telemetry_json(r, d);
    }
    sb_lit(r, "}");
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
                g->stop_hit = g->stop_strs[s];
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
    // the Responses API's name for the same cap
    if (absent(v)) v = jv_get(req, "max_output_tokens");
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
    // the Responses spelling of the same request. Resolved here rather than at
    // the route so the constrained-decoding path has exactly one entry point
    // regardless of which surface asked for it.
    if (!sch) {
        jv *tf = jv_get(jv_get(req, "text"), "format");
        if (tf && strcmp(jv_str(jv_get(tf, "type"), ""), "json_schema") == 0) {
            jv *inner = jv_get(tf, "schema");
            if (inner && inner->type == J_OBJ) sch = inner;
        }
    }
    return sch;
}

// did the caller ask for free-form JSON (rather than a schema)? Both dialects.
static bool request_json_mode(jv *req) {
    jv *rf = jv_get(req, "response_format");
    if (rf && strcmp(jv_str(jv_get(rf, "type"), ""), "json_object") == 0)
        return true;
    jv *tf = jv_get(jv_get(req, "text"), "format");
    return tf && strcmp(jv_str(jv_get(tf, "type"), ""), "json_object") == 0;
}

// run one completion on a slot and write the HTTP response.
// `env` is the strict tool-call envelope when the request opted into one; it
// replaces the response_format schema, having already absorbed it as the
// shape of its `final` branch.
static void run_completion(slot_t *s, int fd, const char *prompt, int api,
                           jv *req, const tool_envelope *env) {
    bool chat = api != API_TEXT; // chat-shaped: thinking channels, tools
    model_t *m = s->m;
    engine *e = &s->e;

    // Per-request deadline. A wall-clock bound is what a batching server owes
    // its clients that a serial one does not: your latency now depends on who
    // else is on the box, so there has to be a point past which you get your
    // (still valid, still schema-conforming) answer regardless. Expiry is a
    // truncation, not an error — it ends the generation and lets the normal
    // close path run, so finish_reason is "length" and constrained output is
    // closed to a legal document exactly as a token-ceiling hit would be.
    double req_timeout = SV.req_timeout;
    double rt;
    if (!request_number(req, "timeout", req_timeout, 0, 86400.0, &rt)) {
        send_error(fd, 400, "timeout out of range");
        return;
    }
    req_timeout = rt;
    double req_deadline = req_timeout > 0 ? now_s() + req_timeout : 0;

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
    if (api == API_CHAT && !request_bool(req, "logprobs", false, &lp_on)) {
        send_error(fd, 400, "logprobs must be a boolean");
        return;
    }
    bool want_lp = api == API_CHAT && !stream && lp_on;
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
    // "stop_sequences" is the Anthropic spelling of the same field. Resolved
    // here, as max_tokens already resolves its three names, so the stop filter
    // has one implementation regardless of which surface asked for it.
    if (absent(stopv)) stopv = jv_get(req, "stop_sequences");
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
    e->json_mode = request_json_mode(req);
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
    // chat responses split a thinking prelude into the reasoning channel;
    // constrained generation forwards it only when asked (raw completions
    // keep the payload-only contract)
    e->emit_think_prelude = chat && m->think_open != NULL;

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

    // Reuse the KV for whatever prefix this prompt shares with one already
    // computed — pipeline callers repeat long system/template prefixes
    // verbatim. Two tiers. The slot's own KV is free — it is already in
    // the cache this sequence decodes against. The shared snapshot store is
    // not free (a memcpy, and on CUDA a device resync), but it reaches across
    // slots, across requests and across model swaps, so it is what turns a
    // repeated system/tool/schema block into work nobody does twice.
    bool cache_prompt = jv_bool(jv_get(req, "cache_prompt"), true);
    // Opt out of the *shared* tier only: a caller isolating a pathological
    // prompt still wants its own slot's cache, and one that wants fresh-prompt
    // telemetry wants neither. cache_prompt:false remains the full opt-out.
    bool share_prefix = jv_bool(jv_get(req, "prefix_cache"), true);
    prefix_reuse reuse = { 0, 0, 0.0 };
    // Prefill is scheduled apart from decode: different kernel shape, and it
    // is this slot's own model_forward_batch, so it must not overlap a
    // microbatch containing this sequence. It cannot — the sequence only joins
    // a batch below, after prefill has returned. Forking a prefix is prefill
    // work (it touches this slot's KV, and on CUDA it issues a forward), so it
    // belongs inside the same device turn.
    sched_prefill_begin();
    if (cache_prompt && share_prefix)
        reuse = engine_prefix_reuse(e, toks, n_prompt);
    else if (cache_prompt)
        reuse.keep = engine_rewind(e, toks, n_prompt);
    else
        engine_reset(e);
    int keep = reuse.keep;
    double prefill_t0 = now_s();
    float *logits = engine_feed(e, toks + keep, n_prompt - keep);
    double prefill_s = now_s() - prefill_t0;
    sched_prefill_end();
    // Publish outside the device turn: it is a host-memory copy, and this
    // sequence is the only writer of its own KV between prefill and decode.
    if (logits && cache_prompt && share_prefix)
        engine_prefix_publish(e, toks, n_prompt, n_prompt - keep, prefill_s);
    free(toks);
    if (!logits) {
        completion_cleanup(e, schema, NULL);
        send_error(fd, 500, "context overflow");
        return;
    }
    if (chat && s->tmpl == TMPL_ORNITH) engine_think_started(e);

    gen_ctx g = { .out = {0}, .fd = fd, .stream = stream, .api = api,
                  .stop_strs = stops, .n_stop = n_stops,
                  .created = (long)time(NULL) };
    snprintf(g.id, sizeof(g.id), "%s%d",
             api == API_RESPONSES ? "resp_" : api == API_MESSAGES ? "msg_"
                                            : api == API_CHAT ? "chatcmpl-"
                                                              : "cmpl-",
             atomic_fetch_add(&SV.req_counter, 1));
    // split thinking channels out of chat responses; raw completions stay raw
    if (chat && s->tmpl == TMPL_ORNITH)
        think_init_reasoning(&g.ts, m->think_open, m->think_close);
    else
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
        if (api == API_CHAT) {
            sbuf c = {0};
            chunk_open(&g, &c);
            sb_lit(&c, "\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}");
            g.role_sent = true;
            chunk_send(&g, &c);
        } else if (api == API_RESPONSES) {
            // A Responses stream opens with the response object twice: created,
            // then in_progress. Both carry an empty output — the items are
            // announced as they start — so they are emitted before any token
            // exists, which is exactly what makes a client able to render the
            // turn's identity immediately.
            resp_doc d = { .status = "in_progress", .req = req };
            for (int i = 0; i < 2 && !g.dead; i++) {
                sbuf f = {0};
                sb_lit(&f, ",\"response\":");
                responses_body(&f, &g, &d);
                resp_send(&g, i == 0 ? "response.created"
                                     : "response.in_progress", &f);
            }
        } else if (api == API_MESSAGES) {
            // An Anthropic stream opens with the whole Message object minus
            // its content: the id, model and input token count are known
            // before a token exists, and a client renders the turn's identity
            // from them immediately. stop_reason is null until message_delta.
            resp_doc d = { .n_prompt = n_prompt, .req = req };
            sbuf f = {0};
            sb_lit(&f, ",\"message\":");
            anth_body(&f, &g, &d);
            anth_send(&g, "message_start", &f);
        }
    }

    double gtime;
    int n_gen = sched_generate(s, logits, max_tokens, gen_collect, &g, &gtime,
                               req_deadline);
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

    if (stream && api == API_RESPONSES) {
        // whatever item was still streaming is closed first: an item announced
        // with output_item.added must always reach output_item.done, including
        // when generation stopped mid-item
        bool cut = strcmp(finish, "length") == 0;
        // a tool call that was truncated is still an executable call — the
        // envelope schema closed it to a legal document — so only a message
        // is reported as unfinished
        if (cut && resp_shape_of(g.item_kind) == &RESP_MESSAGE)
            g.close_status = "incomplete";
        resp_close_item(&g);
        if (!g.dead) {
            bool truncated = cut;
            resp_doc d = { .status = truncated ? "incomplete" : "completed",
                           .incomplete = truncated ? "max_output_tokens" : NULL,
                           .output_json = g.out_items.s ? g.out_items.s : "",
                           .output_n = g.out_items.n,
                           .output_text = g.out_text.s,
                           .output_text_n = g.out_text.n,
                           .with_usage = true,
                           .n_prompt = n_prompt, .n_gen = n_gen, .cached = keep,
                           .forked = reuse.forked, .saved_s = reuse.saved_s,
                           .gtime = gtime, .schema = schema != NULL,
                           .json_mode = e->json_mode, .spec = e->dm != NULL,
                           .req = req };
            sbuf f = {0};
            sb_lit(&f, ",\"response\":");
            responses_body(&f, &g, &d);
            resp_send(&g, truncated ? "response.incomplete"
                                    : "response.completed", &f);
        }
    } else if (stream && api == API_MESSAGES) {
        // A turn that generated nothing at all still has to describe itself
        // the way the buffered body would, which always carries at least one
        // block. Opening an empty text block here is what keeps "the streamed
        // and buffered turns agree" true even in the degenerate case.
        if (!g.item_open && g.output_index == 0) anth_open_block(&g, "text");
        // whatever block was still streaming is closed first: a block
        // announced with content_block_start must always reach
        // content_block_stop, including when generation stopped mid-block
        anth_close_block(&g);
        if (!g.dead) {
            sbuf f = {0};
            sb_fmt(&f, ",\"delta\":{\"stop_reason\":\"%s\",\"stop_sequence\":",
                   anth_stop_reason(finish, g.stop_hit != NULL));
            if (g.stop_hit) {
                sb_lit(&f, "\"");
                sb_esc(&f, g.stop_hit, strlen(g.stop_hit));
                sb_lit(&f, "\"");
            } else {
                sb_lit(&f, "null");
            }
            // message_delta carries the *cumulative* output token count, which
            // is the only place a streamed turn reports it
            sb_fmt(&f, "},\"usage\":{\"input_tokens\":%d,\"output_tokens\":%d}",
                   n_prompt, n_gen);
            if (!anth_send(&g, "message_delta", &f)) {
                sbuf s2 = {0};
                anth_send(&g, "message_stop", &s2);
            }
        }
    } else if (stream) {
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
            n_tc = tool_calls_parse_for(s->tmpl, &g.out, &tc);
            if (n_tc) {
                finish = "tool_calls";
                g.out.n = 0; // OpenAI convention: no content alongside
                             // tool_calls (whatever followed was the model
                             // faking a result)
            }
        }
        if (api == API_MESSAGES) {
            // Same canonical mapping the Responses branch below uses: the
            // chat dialect's tool_calls item is where a call is extracted
            // once, and each surface only renders it in its own vocabulary.
            jv *call = n_tc ? json_parse(tc.s, tc.n) : NULL;
            jv *fn = jv_get(call, "function");
            resp_doc d = { .with_output = true,
                           .stop_reason = anth_stop_reason(finish,
                                                           g.stop_hit != NULL),
                           .stop_seq = g.stop_hit,
                           .call_name = jv_str(jv_get(fn, "name"), NULL),
                           .call_args = jv_str(jv_get(fn, "arguments"), "{}"),
                           .text = g.out.s, .text_n = g.out.n,
                           .reason = g.reason.s, .reason_n = g.reason.n,
                           .with_usage = true,
                           .n_prompt = n_prompt, .n_gen = n_gen, .cached = keep,
                           .forked = reuse.forked, .saved_s = reuse.saved_s,
                           .gtime = gtime, .schema = schema != NULL,
                           .json_mode = e->json_mode, .spec = e->dm != NULL,
                           .req = req };
            sbuf r = {0};
            anth_body(&r, &g, &d);
            send_built(fd, &r);
            free(r.s);
            jv_free(call);
            free(tc.s);
            goto done;
        }
        if (api == API_RESPONSES) {
            // The chat dialect's tool_calls item is the canonical mapping, so
            // the Responses item is derived from it rather than re-extracted
            // from the envelope: one mapping, two renderings.
            jv *call = n_tc ? json_parse(tc.s, tc.n) : NULL;
            jv *fn = jv_get(call, "function");
            bool truncated = strcmp(finish, "length") == 0;
            resp_doc d = { .status = truncated ? "incomplete" : "completed",
                           .incomplete = truncated ? "max_output_tokens" : NULL,
                           .with_output = true,
                           .call_name = jv_str(jv_get(fn, "name"), NULL),
                           .call_args = jv_str(jv_get(fn, "arguments"), "{}"),
                           .text = g.out.s, .text_n = g.out.n,
                           .reason = g.reason.s, .reason_n = g.reason.n,
                           .with_usage = true,
                           .n_prompt = n_prompt, .n_gen = n_gen, .cached = keep,
                           .forked = reuse.forked, .saved_s = reuse.saved_s,
                           .gtime = gtime, .schema = schema != NULL,
                           .json_mode = e->json_mode, .spec = e->dm != NULL,
                           .req = req };
            sbuf r = {0};
            responses_body(&r, &g, &d);
            send_built(fd, &r);
            free(r.s);
            jv_free(call);
            free(tc.s);
            goto done;
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
                   "\"total_tokens\":%d},",
               finish, n_prompt, n_gen, n_prompt + n_gen);
        resp_doc td = { .n_prompt = n_prompt, .n_gen = n_gen, .cached = keep,
                        .forked = reuse.forked, .saved_s = reuse.saved_s,
                        .gtime = gtime, .schema = schema != NULL,
                        .json_mode = e->json_mode, .spec = e->dm != NULL };
        telemetry_json(&r, &td);
        sb_lit(&r, "}");
        send_built(fd, &r);
        free(r.s);
    }
done:
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
static char *message_text(jv *msg, int tmpl) {
    jv *content = jv_get(msg, "content");
    sbuf b = {0};
    const char *role = jv_str(jv_get(msg, "role"), "user");
    const char *reason = jv_str(jv_get(msg, "reasoning_content"), NULL);
    if (tmpl == TMPL_ORNITH && !strcmp(role, "assistant")) {
        sb_lit(&b, "<think>\n");
        if (reason) sb_put(&b, reason, strlen(reason));
        sb_lit(&b, "\n</think>\n\n");
    }
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
    tool_history_render_for(tmpl, calls, &b);
    if (tmpl == TMPL_ORNITH && !strcmp(role, "tool")) {
        sbuf wrapped = {0};
        sb_lit(&wrapped, "<tool_response>\n");
        if (b.s) sb_put(&wrapped, b.s, b.n);
        sb_lit(&wrapped, "\n</tool_response>");
        free(b.s);
        b = wrapped;
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
    // Ornith is specifically trained on qwen3_xml. Keep its native protocol
    // instead of forcing the model into runner's generic JSON envelope.
    bool strict = rc == 1 && s->tmpl != TMPL_ORNITH;
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
    else        tools_render_for(s->tmpl, tools, &ts);
    bool ornith_merged_system = false;
    if (s->tmpl == TMPL_ORNITH && ts.n && msgs->n > 0 &&
        !strcmp(jv_str(jv_get(msgs->items[0], "role"), ""), "system")) {
        char *system = message_text(msgs->items[0], s->tmpl);
        if (system && system[0]) {
            sb_lit(&ts, "\n\n");
            sb_put(&ts, system, strlen(system));
        }
        free(system);
        ornith_merged_system = true;
    }
    chat_msg *cm = malloc(sizeof(chat_msg) * (msgs->n + 1));
    char **owned = malloc(sizeof(char *) * msgs->n);
    size_t total = ts.n + 64;
    int n_cm = 0, n_own = 0;
    if (ts.n) cm[n_cm++] = (chat_msg){ "system", ts.s };
    for (int i = 0; i < msgs->n; i++) {
        if (i == 0 && ornith_merged_system) continue;
        const char *role = jv_str(jv_get(msgs->items[i], "role"), "user");
        char *content = message_text(msgs->items[i], s->tmpl);
        if (!content) continue;
        owned[n_own++] = content;
        if (s->tmpl == TMPL_ORNITH && !strcmp(role, "tool")) role = "user";
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
    run_completion(s, fd, prompt, API_CHAT, req, strict ? &env : NULL);
    free(prompt);
    for (int i = 0; i < n_own; i++) free(owned[i]);
    free(owned);
    free(cm);
    free(ts.s);
    tool_envelope_free(&env);
}

// ------------------------------------------------ Responses request → chat
//
// The inbound half of the translation. A Responses request says the same
// things a chat request does in a different vocabulary, so it is rewritten
// into that vocabulary once, here, and everything downstream is the path
// /v1/chat/completions already takes. Nothing below generates or samples; if
// it did, there would be two engines to keep honest instead of one.

// Responses declares a tool flat — {"type":"function","name":...,"parameters":
// ...} — where chat nests it under "function". Rather than teach the envelope
// compiler a second shape (and risk the chat path with it), the flat form is
// re-serialised into the nested one and re-parsed. Returns an owned jv the
// caller frees, or NULL with err set.
static jv *responses_tools(jv *tools, char *err, int errcap) {
    if (!tools || tools->type == J_NULL) return NULL;
    if (tools->type != J_ARR) {
        snprintf(err, errcap, "tools must be an array");
        return NULL;
    }
    sbuf b = {0};
    sb_lit(&b, "[");
    int emitted = 0;
    // one level of `namespace` nesting is flattened, so the loop walks the
    // outer list and, for a namespace, its inner list
    for (int i = 0; i < tools->n; i++) {
        jv *outer = tools->items[i];
        if (!outer || outer->type != J_OBJ) {
            snprintf(err, errcap, "each tools[] entry must be an object");
            free(b.s);
            return NULL;
        }
        const char *otype = jv_str(jv_get(outer, "type"), "function");
        jv *group = NULL;
        if (!strcmp(otype, "namespace")) {
            // Codex groups related function tools under a namespace entry. A
            // namespace is a container, not a tool, so it is flattened rather
            // than refused: every leaf is a local function after all.
            group = jv_get(outer, "tools");
            if (!group || group->type != J_ARR) {
                snprintf(err, errcap,
                         "tools[].type \"namespace\" must carry a tools array");
                free(b.s);
                return NULL;
            }
        }
        int inner_n = group ? group->n : 1;
        for (int k = 0; k < inner_n; k++) {
            jv *t = group ? group->items[k] : outer;
            if (!t || t->type != J_OBJ) {
                snprintf(err, errcap, "each tools[] entry must be an object");
                free(b.s);
                return NULL;
            }
            const char *type = jv_str(jv_get(t, "type"), "function");
            if (strcmp(type, "function") != 0) {
                // A hosted tool the client itself marked unavailable is not a
                // request for anything, so dropping it misleads nobody. One
                // that is actually asked for is a capability this runtime does
                // not have, and saying so beats leaving the caller waiting for
                // a call that can never come.
                jv *web = jv_get(t, "external_web_access");
                if (web && web->type == J_BOOL && !web->b) continue;
                snprintf(err, errcap,
                         "tools[].type \"%.40s\" is not supported; "
                         "only \"function\" tools can run locally", type);
                free(b.s);
                return NULL;
            }
            // already nested (a client reusing its chat tool definitions)
            jv *nested = jv_get(t, "function");
            if (emitted++) sb_lit(&b, ",");
            sb_lit(&b, "{\"type\":\"function\",\"function\":");
            if (nested && nested->type == J_OBJ) {
                jv_dump(nested, &b);
            } else {
                sb_lit(&b, "{\"name\":");
                jv *nm = jv_get(t, "name");
                if (nm) jv_dump(nm, &b); else sb_lit(&b, "null");
                jv *desc = jv_get(t, "description");
                if (desc) { sb_lit(&b, ",\"description\":"); jv_dump(desc, &b); }
                jv *params = jv_get(t, "parameters");
                if (params) { sb_lit(&b, ",\"parameters\":"); jv_dump(params, &b); }
                sb_lit(&b, "}");
            }
            sb_lit(&b, "}");
        }
    }
    sb_lit(&b, "]");
    if (b.failed || !b.s) {
        snprintf(err, errcap, "out of memory translating tools");
        free(b.s);
        return NULL;
    }
    jv *out = json_parse(b.s, b.n);
    free(b.s);
    if (!out) snprintf(err, errcap, "tools did not translate to a valid shape");
    return out;
}

// tool_choice, likewise: the named form is flat here and nested in chat.
static jv *responses_tool_choice(jv *tc, char *err, int errcap) {
    if (!tc || tc->type != J_OBJ) return NULL; // strings pass through unchanged
    const char *name = jv_str(jv_get(tc, "name"), NULL);
    if (!name) {
        snprintf(err, errcap,
                 "tool_choice object must be {\"type\":\"function\",\"name\":...}");
        return NULL;
    }
    sbuf b = {0};
    sb_lit(&b, "{\"type\":\"function\",\"function\":{\"name\":\"");
    sb_esc(&b, name, strlen(name));
    sb_lit(&b, "\"}}");
    jv *out = b.failed || !b.s ? NULL : json_parse(b.s, b.n);
    free(b.s);
    if (!out) snprintf(err, errcap, "out of memory translating tool_choice");
    return out;
}

// `text.format` is the Responses spelling of `response_format`. Returns the
// schema to constrain to, or NULL; *bad is set when the field is malformed.
static jv *responses_schema(jv *req, bool *bad, char *err, int errcap) {
    *bad = false;
    jv *text = jv_get(req, "text");
    if (!text || text->type == J_NULL) return NULL;
    if (text->type != J_OBJ) {
        snprintf(err, errcap, "text must be an object");
        *bad = true;
        return NULL;
    }
    jv *fmt = jv_get(text, "format");
    if (!fmt || fmt->type == J_NULL) return NULL;
    if (fmt->type != J_OBJ) {
        snprintf(err, errcap, "text.format must be an object");
        *bad = true;
        return NULL;
    }
    const char *type = jv_str(jv_get(fmt, "type"), "");
    if (!strcmp(type, "text")) return NULL;
    if (!strcmp(type, "json_object")) return NULL; // handled as json mode
    if (strcmp(type, "json_schema") != 0) {
        snprintf(err, errcap,
                 "text.format.type must be text, json_object or json_schema");
        *bad = true;
        return NULL;
    }
    // Responses puts the schema directly on the format object rather than
    // under a json_schema wrapper
    jv *sch = jv_get(fmt, "schema");
    if (!sch || sch->type != J_OBJ) {
        snprintf(err, errcap, "text.format.schema must be an object");
        *bad = true;
        return NULL;
    }
    return sch;
}

// Flatten one `input` item to prompt text, appending it as a chat turn.
// Returns the role to file it under, or NULL when the item carries nothing.
static char *responses_item_text(jv *item, const char **role) {
    const char *type = jv_str(jv_get(item, "type"), NULL);
    sbuf b = {0};
    // a tool result the caller is feeding back: this is the tool loop
    if (type && !strcmp(type, "function_call_output")) {
        *role = "tool";
        jv *out = jv_get(item, "output");
        if (out && out->type == J_STR) sb_put(&b, out->str, strlen(out->str));
        else if (out) jv_dump(out, &b);
        return b.s ? b.s : strdup("");
    }
    // the assistant's own earlier call, replayed: rendered in runner's call
    // syntax so the history reads like what the model actually emitted
    if (type && !strcmp(type, "function_call")) {
        *role = "assistant";
        const char *name = jv_str(jv_get(item, "name"), NULL);
        const char *args = jv_str(jv_get(item, "arguments"), "{}");
        if (!name) { free(b.s); return NULL; }
        sb_fmt(&b, "<|tool_call>call:%s%s<tool_call|>", name, args);
        return b.s;
    }
    *role = jv_str(jv_get(item, "role"), "user");
    // "developer" is the Responses spelling of a system turn; chat templates
    // know the latter
    if (!strcmp(*role, "developer")) *role = "system";
    jv *content = jv_get(item, "content");
    if (content && content->type == J_STR) {
        sb_put(&b, content->str, strlen(content->str));
    } else if (content && content->type == J_ARR) {
        for (int i = 0; i < content->n; i++) {
            jv *part = content->items[i];
            const char *pt = jv_str(jv_get(part, "type"), "");
            // input_text / output_text are the Responses spellings; "text" is
            // accepted too because clients reusing chat parts send it
            if (strcmp(pt, "input_text") && strcmp(pt, "output_text") &&
                strcmp(pt, "text"))
                continue; // images and files have no local renderer
            const char *txt = jv_str(jv_get(part, "text"), NULL);
            if (!txt) continue;
            if (b.n) sb_lit(&b, "\n");
            sb_put(&b, txt, strlen(txt));
        }
    }
    return b.s;
}

// Stateful Responses features this runtime has no store behind. Refusing them
// is the project invariant: a client that asked the server to remember a turn
// and got a 200 would believe it did.
static bool responses_reject_stateful(int fd, jv *req) {
    jv *v = jv_get(req, "previous_response_id");
    if (v && v->type != J_NULL) {
        send_error(fd, 400,
                   "previous_response_id is not supported: this runtime is "
                   "stateless and stores no conversation. Send the prior turns "
                   "in `input` instead.");
        return true;
    }
    v = jv_get(req, "store");
    if (v && v->type != J_NULL) {
        if (v->type != J_BOOL) {
            send_error(fd, 400, "store must be a boolean");
            return true;
        }
        if (v->b) {
            send_error(fd, 400,
                       "store:true is not supported: this runtime is stateless "
                       "and cannot retrieve a stored response. Use store:false.");
            return true;
        }
    }
    v = jv_get(req, "background");
    if (v && v->type != J_NULL) {
        if (v->type != J_BOOL) {
            send_error(fd, 400, "background must be a boolean");
            return true;
        }
        if (v->b) {
            send_error(fd, 400,
                       "background:true is not supported: there is no response "
                       "store to poll. Use a streaming or buffered request.");
            return true;
        }
    }
    v = jv_get(req, "conversation");
    if (v && v->type != J_NULL) {
        send_error(fd, 400,
                   "conversation is not supported: this runtime is stateless "
                   "and stores no conversation.");
        return true;
    }
    // "truncation":"auto" asks the server to silently drop history to fit; a
    // caller told 200 would never learn its context had been edited
    const char *tr = jv_str(jv_get(req, "truncation"), NULL);
    if (tr && strcmp(tr, "disabled") != 0) {
        send_error(fd, 400,
                   "truncation:\"auto\" is not supported; a prompt that exceeds "
                   "the context window is rejected rather than silently cut");
        return true;
    }
    // `include` asks for extra output payloads (logprobs, image URLs, encrypted
    // reasoning) none of which this runtime can produce
    v = jv_get(req, "include");
    if (v && v->type == J_ARR && v->n > 0) {
        send_error(fd, 400,
                   "include[] is not supported; no additional output payloads "
                   "are available from this runtime");
        return true;
    }
    return false;
}

static void handle_responses(slot_t *s, int fd, jv *req) {
    if (responses_reject_stateful(fd, req)) return;

    jv *input = jv_get(req, "input");
    if (!input || input->type == J_NULL) {
        send_error(fd, 400, "missing input");
        return;
    }
    if (input->type != J_STR && input->type != J_ARR) {
        send_error(fd, 400, "input must be a string or an array of items");
        return;
    }
    // reasoning is accepted and echoed back rather than rejected: `effort` and
    // `summary` are hints about how much thinking to do, not guarantees about
    // the response document, and a local model's thinking channel is already
    // reported as a reasoning item. A malformed one is still an error.
    jv *reasoning = jv_get(req, "reasoning");
    if (reasoning && reasoning->type != J_NULL && reasoning->type != J_OBJ) {
        send_error(fd, 400, "reasoning must be an object");
        return;
    }

    char terr[224];
    bool bad_fmt = false;
    jv *final_schema = responses_schema(req, &bad_fmt, terr, sizeof(terr));
    if (bad_fmt) { send_error(fd, 400, terr); return; }

    jv *tools = responses_tools(jv_get(req, "tools"), terr, sizeof(terr));
    if (jv_get(req, "tools") && jv_get(req, "tools")->type != J_NULL && !tools) {
        send_error(fd, 400, terr);
        return;
    }
    jv *choice_raw = jv_get(req, "tool_choice");
    jv *choice_owned = NULL;
    if (choice_raw && choice_raw->type == J_OBJ) {
        choice_owned = responses_tool_choice(choice_raw, terr, sizeof(terr));
        if (!choice_owned) { jv_free(tools); send_error(fd, 400, terr); return; }
    }

    tool_envelope env = {0};
    int rc = tool_envelope_build(tools, choice_owned ? choice_owned : choice_raw,
                                 final_schema, &env, terr, sizeof(terr));
    if (rc < 0) {
        jv_free(tools);
        jv_free(choice_owned);
        send_error(fd, 400, terr);
        return;
    }
    bool strict = rc == 1;
    if (strict) {
        bool parallel = false;
        if (!request_bool(req, "parallel_tool_calls", false, &parallel)) {
            tool_envelope_free(&env);
            jv_free(tools);
            jv_free(choice_owned);
            send_error(fd, 400, "parallel_tool_calls must be a boolean");
            return;
        }
        if (parallel) {
            tool_envelope_free(&env);
            jv_free(tools);
            jv_free(choice_owned);
            send_error(fd, 400,
                       "parallel_tool_calls:true is not supported yet; "
                       "one call per turn");
            return;
        }
    }

    // assemble the turns: tool system turn, then `instructions` as a system
    // message, then the input items in order
    int n_items = input->type == J_ARR ? input->n : 1;
    sbuf ts = {0};
    if (strict) sb_put(&ts, env.system_turn, strlen(env.system_turn));
    else        tools_render(tools, &ts);
    chat_msg *cm = malloc(sizeof(chat_msg) * (size_t)(n_items + 2));
    char **owned = malloc(sizeof(char *) * (size_t)n_items);
    size_t total = ts.n + 128;
    int n_cm = 0, n_own = 0;
    if (ts.n) cm[n_cm++] = (chat_msg){ "system", ts.s };
    const char *instructions = jv_str(jv_get(req, "instructions"), NULL);
    if (instructions && instructions[0]) {
        cm[n_cm++] = (chat_msg){ "system", instructions };
        total += strlen(instructions) + 64;
    }
    if (input->type == J_STR) {
        cm[n_cm++] = (chat_msg){ "user", input->str };
        total += strlen(input->str) + 64;
    } else {
        for (int i = 0; i < input->n; i++) {
            const char *role = "user";
            char *text = responses_item_text(input->items[i], &role);
            if (!text) continue;
            owned[n_own++] = text;
            cm[n_cm++] = (chat_msg){ role, text };
            total += strlen(role) + strlen(text) + 64;
        }
    }
    if (n_cm == 0) {
        free(owned); free(cm); free(ts.s);
        tool_envelope_free(&env);
        jv_free(tools);
        jv_free(choice_owned);
        send_error(fd, 400, "no input content");
        return;
    }
    char *prompt = malloc(total + 256);
    render_messages(s->tmpl, cm, n_cm, true, prompt, total + 256);
    run_completion(s, fd, prompt, API_RESPONSES, req, strict ? &env : NULL);
    free(prompt);
    for (int i = 0; i < n_own; i++) free(owned[i]);
    free(owned);
    free(cm);
    free(ts.s);
    tool_envelope_free(&env);
    jv_free(tools);
    jv_free(choice_owned);
}

// ----------------------------------------- Anthropic Messages request → chat
//
// The inbound half of the third translation, and the same move
// handle_responses makes: an Anthropic request says the same things a chat
// request does in a different vocabulary, so it is rewritten into that
// vocabulary once, here, and everything downstream is the path
// /v1/chat/completions already takes. Nothing below generates or samples.

// A growable list of rendered chat turns.
//
// An Anthropic message does not map one-to-one onto a chat turn: a single user
// message can carry several tool_result blocks *and* text, and the chat
// vocabulary the templates speak files a tool result as its own turn. So turns
// are appended as they are produced rather than indexed by message.
typedef struct {
    chat_msg *cm;
    char    **owned;     // heap turns to free; borrowed ones are not listed
    int       n, n_own, cap;
    size_t    total;     // rendered-size estimate for the prompt buffer
    bool      failed;
} turnbuf;

static void turn_add_borrowed(turnbuf *t, const char *role, const char *text) {
    if (t->n >= t->cap) { t->failed = true; return; }
    t->cm[t->n++] = (chat_msg){ role, text };
    t->total += strlen(role) + strlen(text) + 64;
}

// takes ownership of `text` on every path, including failure
static void turn_add(turnbuf *t, const char *role, char *text) {
    if (!text) { t->failed = true; return; }
    if (t->n >= t->cap) { free(text); t->failed = true; return; }
    t->owned[t->n_own++] = text;
    turn_add_borrowed(t, role, text);
}

static void turnbuf_free(turnbuf *t) {
    for (int i = 0; i < t->n_own; i++) free(t->owned[i]);
    free(t->owned);
    free(t->cm);
}

// The text of one tool_result block. Its `content` is a string or a block
// list, and `is_error` is the one bit of the result the model needs to see
// that plain text would otherwise lose.
static char *anth_tool_result_text(jv *b) {
    sbuf r = {0};
    if (jv_bool(jv_get(b, "is_error"), false)) sb_lit(&r, "error: ");
    jv *c = jv_get(b, "content");
    if (c && c->type == J_STR) {
        sb_put(&r, c->str, strlen(c->str));
    } else if (c && c->type == J_ARR) {
        for (int i = 0; i < c->n; i++) {
            const char *txt = jv_str(jv_get(c->items[i], "text"), NULL);
            if (!txt) continue;
            if (r.n) sb_lit(&r, "\n");
            sb_put(&r, txt, strlen(txt));
        }
    } else if (c && c->type != J_NULL) {
        jv_dump(c, &r);
    }
    return r.s ? r.s : strdup("");
}

// Flatten one message's content into turns. Returns false with err set when it
// carries something this runtime cannot render: dropping an image would answer
// a question about content the model never saw, which is exactly the silent
// success this project refuses.
static bool anth_blocks(jv *msg, const char *role, turnbuf *t,
                        char *err, int errcap) {
    jv *content = jv_get(msg, "content");
    if (content && content->type == J_STR) {
        turn_add(t, role, strdup(content->str));
        return true;
    }
    if (!content || content->type != J_ARR) {
        snprintf(err, errcap,
                 "messages[].content must be a string or an array of blocks");
        return false;
    }
    sbuf body = {0};
    for (int i = 0; i < content->n; i++) {
        jv *b = content->items[i];
        if (!b || b->type != J_OBJ) {
            snprintf(err, errcap,
                     "each messages[].content block must be an object");
            free(body.s);
            return false;
        }
        const char *bt = jv_str(jv_get(b, "type"), "");
        if (!strcmp(bt, "text")) {
            const char *txt = jv_str(jv_get(b, "text"), NULL);
            if (!txt) {
                snprintf(err, errcap, "a text block must carry a text string");
                free(body.s);
                return false;
            }
            if (body.n) sb_lit(&body, "\n");
            sb_put(&body, txt, strlen(txt));
        } else if (!strcmp(bt, "tool_use")) {
            // the assistant's own earlier call, replayed: rendered in runner's
            // call syntax so the history reads like what the model emitted
            const char *name = jv_str(jv_get(b, "name"), NULL);
            if (!name) {
                snprintf(err, errcap, "a tool_use block must carry a name");
                free(body.s);
                return false;
            }
            jv *input = jv_get(b, "input");
            sb_fmt(&body, "<|tool_call>call:%s", name);
            if (input && input->type != J_NULL) jv_dump(input, &body);
            else                                sb_lit(&body, "{}");
            sb_lit(&body, "<tool_call|>");
        } else if (!strcmp(bt, "tool_result")) {
            // the tool loop closing: a result is its own turn in the chat
            // vocabulary, so it is emitted ahead of whatever text accompanies
            // it in the same Anthropic message
            turn_add(t, "tool", anth_tool_result_text(b));
        } else if (!strcmp(bt, "thinking") || !strcmp(bt, "redacted_thinking")) {
            // Replayed reasoning. Anthropic wants it back so *it* can verify a
            // signature; there is nothing to verify locally, and it is the
            // model's own scratch work rather than anything the user said, so
            // it is not put back into the prompt.
            continue;
        } else {
            snprintf(err, errcap,
                     "messages[].content block type \"%.40s\" is not supported; "
                     "this runtime renders text, tool_use and tool_result "
                     "blocks only", bt);
            free(body.s);
            return false;
        }
    }
    if (body.n) turn_add(t, role, body.s);
    else        free(body.s);
    return true;
}

// `system` is a string or a list of text blocks. Returns an owned string, or
// NULL when there is no system content (which is not an error).
static char *anth_system_text(jv *system, char *err, int errcap) {
    if (!system || system->type == J_NULL) return NULL;
    if (system->type == J_STR)
        return system->str[0] ? strdup(system->str) : NULL;
    if (system->type != J_ARR) {
        snprintf(err, errcap,
                 "system must be a string or an array of text blocks");
        return NULL;
    }
    sbuf b = {0};
    for (int i = 0; i < system->n; i++) {
        const char *txt = jv_str(jv_get(system->items[i], "text"), NULL);
        if (!txt) continue;
        if (b.n) sb_lit(&b, "\n");
        sb_put(&b, txt, strlen(txt));
    }
    return b.s;
}

// Anthropic declares a tool as {name, description, input_schema} where chat
// nests it under "function" and calls the schema "parameters". Rather than
// teach the envelope compiler a third shape (and risk the two paths already
// using it), the Anthropic form is re-serialised into the nested one and
// re-parsed — exactly what responses_tools does. Returns an owned jv.
static jv *anth_tools(jv *tools, char *err, int errcap) {
    if (!tools || tools->type == J_NULL) return NULL;
    if (tools->type != J_ARR) {
        snprintf(err, errcap, "tools must be an array");
        return NULL;
    }
    sbuf b = {0};
    sb_lit(&b, "[");
    for (int i = 0; i < tools->n; i++) {
        jv *t = tools->items[i];
        if (!t || t->type != J_OBJ) {
            snprintf(err, errcap, "each tools[] entry must be an object");
            free(b.s);
            return NULL;
        }
        // A server tool (web_search_*, computer_*, bash_*, text_editor_*) is
        // named by its `type`; a client function tool has no type at all.
        // Saying so beats leaving the caller waiting for a call that can never
        // come from a runtime with no such capability.
        const char *type = jv_str(jv_get(t, "type"), NULL);
        if (type && strcmp(type, "custom") != 0 && strcmp(type, "function") != 0) {
            snprintf(err, errcap,
                     "tools[].type \"%.40s\" is a server-side tool and is not "
                     "supported; only client tools can run locally", type);
            free(b.s);
            return NULL;
        }
        if (i) sb_lit(&b, ",");
        sb_lit(&b, "{\"type\":\"function\",\"function\":{\"name\":");
        jv *nm = jv_get(t, "name");
        if (nm) jv_dump(nm, &b); else sb_lit(&b, "null");
        jv *desc = jv_get(t, "description");
        if (desc) { sb_lit(&b, ",\"description\":"); jv_dump(desc, &b); }
        jv *params = jv_get(t, "input_schema");
        if (params) { sb_lit(&b, ",\"parameters\":"); jv_dump(params, &b); }
        sb_lit(&b, "}}");
    }
    sb_lit(&b, "]");
    if (b.failed || !b.s) {
        snprintf(err, errcap, "out of memory translating tools");
        free(b.s);
        return NULL;
    }
    jv *out = json_parse(b.s, b.n);
    free(b.s);
    if (!out) snprintf(err, errcap, "tools did not translate to a valid shape");
    return out;
}

// tool_choice is an object in every Anthropic form; chat spells three of the
// four as bare strings. Returns an owned jv, or NULL with err set.
static jv *anth_tool_choice(jv *tc, char *err, int errcap) {
    if (!tc || tc->type == J_NULL) return NULL;
    if (tc->type != J_OBJ) {
        snprintf(err, errcap, "tool_choice must be an object");
        return NULL;
    }
    // "don't disable parallel use" is a request to allow several calls in one
    // turn. The envelope is one call per turn on every surface, so this is
    // refused here exactly as parallel_tool_calls:true is on the other two,
    // rather than answered with a single call the caller cannot distinguish
    // from a considered choice.
    jv *par = jv_get(tc, "disable_parallel_tool_use");
    if (par && par->type != J_NULL) {
        if (par->type != J_BOOL) {
            snprintf(err, errcap,
                     "tool_choice.disable_parallel_tool_use must be a boolean");
            return NULL;
        }
        if (!par->b) {
            snprintf(err, errcap,
                     "tool_choice.disable_parallel_tool_use:false is not "
                     "supported yet; this runtime emits one tool call per turn. "
                     "Omit it or send true.");
            return NULL;
        }
    }
    const char *type = jv_str(jv_get(tc, "type"), NULL);
    const char *mapped = NULL;
    if (!type) {
        snprintf(err, errcap, "tool_choice.type is required");
        return NULL;
    }
    if (!strcmp(type, "auto")) mapped = "\"auto\"";
    else if (!strcmp(type, "none")) mapped = "\"none\"";
    else if (!strcmp(type, "any")) mapped = "\"required\"";  // any one tool
    else if (strcmp(type, "tool") != 0) {
        snprintf(err, errcap,
                 "tool_choice.type must be \"auto\", \"any\", \"tool\" or \"none\"");
        return NULL;
    }
    sbuf b = {0};
    if (mapped) {
        sb_lit(&b, mapped);
    } else {
        const char *name = jv_str(jv_get(tc, "name"), NULL);
        if (!name || !name[0]) {
            snprintf(err, errcap,
                     "tool_choice.type \"tool\" requires a name");
            return NULL;
        }
        sb_lit(&b, "{\"type\":\"function\",\"function\":{\"name\":\"");
        sb_esc(&b, name, strlen(name));
        sb_lit(&b, "\"}}");
    }
    jv *out = b.failed || !b.s ? NULL : json_parse(b.s, b.n);
    free(b.s);
    if (!out) snprintf(err, errcap, "out of memory translating tool_choice");
    return out;
}

// Features with no local implementation. Refusing them is the project
// invariant: a client told 200 believes the thing it asked for happened.
static bool anth_reject_unsupported(slot_t *s, int fd, jv *req) {
    jv *v = jv_get(req, "mcp_servers");
    if (v && v->type == J_ARR && v->n > 0) {
        send_error(fd, 400,
                   "mcp_servers is not supported: this runtime cannot reach "
                   "remote MCP servers on your behalf. Run the tools locally "
                   "and declare them in tools[].");
        return true;
    }
    v = jv_get(req, "container");
    if (v && v->type != J_NULL) {
        send_error(fd, 400,
                   "container is not supported: there is no code-execution "
                   "container behind this runtime.");
        return true;
    }
    v = jv_get(req, "metadata");
    if (v && v->type != J_NULL && v->type != J_OBJ) {
        send_error(fd, 400, "metadata must be an object");
        return true;
    }
    // `thinking` promises the turn will carry thinking blocks. Whether it can
    // is a property of the resident model — a thinking-tagged one separates
    // its reasoning channel already — so it is answered honestly per model
    // rather than accepted and quietly not done.
    v = jv_get(req, "thinking");
    if (v && v->type != J_NULL) {
        if (v->type != J_OBJ) {
            send_error(fd, 400, "thinking must be an object");
            return true;
        }
        const char *type = jv_str(jv_get(v, "type"), NULL);
        if (!type || (strcmp(type, "enabled") && strcmp(type, "disabled") &&
                      strcmp(type, "adaptive"))) {
            send_error(fd, 400,
                       "thinking.type must be \"enabled\", \"disabled\" or "
                       "\"adaptive\"");
            return true;
        }
        if (!strcmp(type, "enabled") && !s->m->think_open) {
            send_error(fd, 400,
                       "thinking:enabled is not supported by the resident "
                       "model: it has no reasoning channel to separate, so no "
                       "thinking block could be returned.");
            return true;
        }
    }
    return false;
}

// Build the prompt one Messages request asks for. Shared by /v1/messages and
// /v1/messages/count_tokens so the count is necessarily the count of the
// prompt the real request would have run — the two cannot drift.
//
// Returns a heap prompt on success (caller frees) with *strict/*env set, or
// NULL having already answered `fd` with the error.
static char *messages_prompt(slot_t *s, int fd, jv *req, tool_envelope *env,
                             bool *strict) {
    char terr[224];
    *strict = false;
    memset(env, 0, sizeof(*env));

    jv *msgs = jv_get(req, "messages");
    if (!msgs || msgs->type != J_ARR || msgs->n == 0) {
        send_error(fd, 400, "missing messages");
        return NULL;
    }

    jv *tools = anth_tools(jv_get(req, "tools"), terr, sizeof(terr));
    jv *raw_tools = jv_get(req, "tools");
    if (raw_tools && raw_tools->type != J_NULL && !tools) {
        send_error(fd, 400, terr);
        return NULL;
    }
    jv *choice = anth_tool_choice(jv_get(req, "tool_choice"), terr, sizeof(terr));
    jv *raw_choice = jv_get(req, "tool_choice");
    if (raw_choice && raw_choice->type != J_NULL && !choice) {
        jv_free(tools);
        send_error(fd, 400, terr);
        return NULL;
    }

    // the same envelope compiler, from the same declarations, as both OpenAI
    // surfaces: this is what makes an Anthropic tool call and a chat tool call
    // the same internal agent action
    int rc = tool_envelope_build(tools, choice, NULL, env, terr, sizeof(terr));
    if (rc < 0) {
        jv_free(tools);
        jv_free(choice);
        send_error(fd, 400, terr);
        return NULL;
    }
    *strict = rc == 1;

    sbuf ts = {0};
    if (*strict) sb_put(&ts, env->system_turn, strlen(env->system_turn));
    else         tools_render(tools, &ts);

    // upper bound on turns: the tool system turn, the system turn, and for
    // each message its own turn plus one per tool_result block it carries
    int cap = 2 + msgs->n;
    for (int i = 0; i < msgs->n; i++) {
        jv *c = jv_get(msgs->items[i], "content");
        if (c && c->type == J_ARR) cap += c->n;
    }
    turnbuf t = { .cm = malloc(sizeof(chat_msg) * (size_t)cap),
                  .owned = malloc(sizeof(char *) * (size_t)cap),
                  .cap = cap, .total = ts.n + 128 };
    if (!t.cm || !t.owned) t.failed = true;

    char *sys = NULL;
    if (!t.failed) {
        if (ts.n) turn_add_borrowed(&t, "system", ts.s);
        terr[0] = 0;
        sys = anth_system_text(jv_get(req, "system"), terr, sizeof(terr));
        if (terr[0]) {
            turnbuf_free(&t);
            free(ts.s);
            tool_envelope_free(env);
            jv_free(tools);
            jv_free(choice);
            send_error(fd, 400, terr);
            return NULL;
        }
        if (sys) turn_add(&t, "system", sys);
    }

    bool ok = !t.failed;
    for (int i = 0; ok && i < msgs->n; i++) {
        jv *msg = msgs->items[i];
        const char *role = jv_str(jv_get(msg, "role"), NULL);
        if (!role || (strcmp(role, "user") && strcmp(role, "assistant") &&
                      strcmp(role, "system"))) {
            snprintf(terr, sizeof(terr),
                     "messages[].role must be \"user\", \"assistant\" or "
                     "\"system\"");
            ok = false;
            break;
        }
        ok = anth_blocks(msg, role, &t, terr, sizeof(terr));
    }
    if (ok && t.failed) {
        snprintf(terr, sizeof(terr), "out of memory building the prompt");
        ok = false;
    }
    if (ok && t.n == 0) {
        snprintf(terr, sizeof(terr), "no message content");
        ok = false;
    }
    char *prompt = NULL;
    if (ok) {
        prompt = malloc(t.total + 256);
        if (prompt) render_messages(s->tmpl, t.cm, t.n, true, prompt,
                                    t.total + 256);
        else ok = false;
    }
    turnbuf_free(&t);
    free(ts.s);
    jv_free(tools);
    jv_free(choice);
    if (!ok) {
        tool_envelope_free(env);
        free(prompt);
        send_error(fd, 400, terr[0] ? terr : "cannot build prompt");
        return NULL;
    }
    return prompt;
}

static void handle_messages(slot_t *s, int fd, jv *req) {
    if (anth_reject_unsupported(s, fd, req)) return;
    // max_tokens is required on this surface, unlike the OpenAI ones where it
    // defaults. A caller that forgot it wants a cap, not the server's.
    jv *mt = jv_get(req, "max_tokens");
    if (!mt || mt->type == J_NULL) {
        send_error(fd, 400, "max_tokens is required");
        return;
    }
    tool_envelope env;
    bool strict = false;
    char *prompt = messages_prompt(s, fd, req, &env, &strict);
    if (!prompt) return;
    run_completion(s, fd, prompt, API_MESSAGES, req, strict ? &env : NULL);
    free(prompt);
    tool_envelope_free(&env);
}

// POST /v1/messages/count_tokens: how many input tokens this exact request
// would have cost. It runs the whole inbound translation and stops before
// generation, so the answer is the prompt the request would really have used.
static void handle_count_tokens(slot_t *s, int fd, jv *req) {
    if (anth_reject_unsupported(s, fd, req)) return;
    tool_envelope env;
    bool strict = false;
    char *prompt = messages_prompt(s, fd, req, &env, &strict);
    if (!prompt) return;
    size_t cap = strlen(prompt) + 16;
    int32_t *toks = malloc(sizeof(int32_t) * cap);
    int n = toks ? tok_encode(s->tok, prompt, toks, (int)cap, true, true) : 0;
    free(toks);
    free(prompt);
    tool_envelope_free(&env);
    if (!n) { send_error(fd, 400, "empty prompt"); return; }
    sbuf r = {0};
    sb_fmt(&r, "{\"input_tokens\":%d}", n);
    send_built(fd, &r);
    free(r.s);
}

static void handle_completion(slot_t *s, int fd, jv *req) {
    const char *prompt = jv_str(jv_get(req, "prompt"), NULL);
    if (!prompt) { send_error(fd, 400, "missing prompt"); return; }
    run_completion(s, fd, prompt, API_TEXT, req, NULL);
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

// Server-wide prefix-cache telemetry. Per-request telemetry says what one
// request saved; this says whether the cache is earning its memory —
// hit rate, resident bytes against the budget, and the prefill time it has
// avoided so far.
static void send_prefix_cache(int fd) {
    prefix_cache_stats st;
    prefix_cache_stats_get(&st);
    char b[640];
    int n = snprintf(b, sizeof(b),
        "{\"object\":\"runner.prefix_cache\","
        "\"enabled\":%s,\"entries\":%d,\"bytes\":%llu,\"budget_bytes\":%llu,"
        "\"ttl_seconds\":%.1f,\"hits\":%llu,\"misses\":%llu,\"stores\":%llu,"
        "\"evictions\":%llu,\"tokens_reused\":%llu,"
        "\"saved_prefill_seconds\":%.6f,\"prefill_seconds_per_token\":%.9f}",
        st.budget ? "true" : "false", st.entries,
        (unsigned long long)st.bytes, (unsigned long long)st.budget, st.ttl,
        (unsigned long long)st.hits, (unsigned long long)st.misses,
        (unsigned long long)st.stores, (unsigned long long)st.evictions,
        (unsigned long long)st.tokens_reused,
        st.saved_prefill_s, st.cost_per_token_s);
    send_response(fd, 200, "application/json", b, n);
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
               "\"responses_api\":true,"
               "\"messages_api\":true,"
               "\"json_object\":true,"
               "\"json_schema\":true,"
               "\"stop_sequences\":true,"
               "\"schema_conditionals\":true,"
               "\"schema_string_bounds\":true,"
               "\"schema_integer_bounds\":true,"
               "\"request_telemetry\":true,"
               "\"prefix_cache\":true,"
               "\"prefix_cache_controls\":true,"
               "\"shared_prefix_cache\":true,"
               "\"forkable_prefixes\":true,"
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
    char *header_end = body_start;
    body_start += 4;

    char method[8] = {0}, path[256] = {0};
    char *first_header = NULL;
    if (!parse_request_line(hdr, method, path, &first_header)) {
        send_error(fd, 400, "malformed request line");
        return;
    }
    // Route on the path component. SDKs use query parameters for protocol
    // feature selection — Claude Code, for example, sends
    // `/v1/messages?beta=true`. The query does not rename the resource and is
    // deliberately not interpreted by this stateless inference boundary.
    char *query = strchr(path, '?');
    if (query) *query = 0;

    size_t content_length = 0;
    if (!parse_request_framing(first_header, header_end, &content_length)) {
        send_error(fd, 400, "invalid request framing");
        return;
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
        // llama-swap-compatible: free the resident model's memory. Normally
        // answered straight from the accept loop (see accept_fastpath); this
        // path still serves a request that slipped past it.
        handle_unload(fd);
    } else if (!strcmp(method, "GET") &&
               !strcmp(path, "/v1/runner/prefix-cache")) {
        send_prefix_cache(fd);
    } else if (!strcmp(method, "POST") &&
               !strcmp(path, "/v1/runner/prefix-cache/clear")) {
        // Explicit release, for benchmarks that need a cold cache and for
        // operators reclaiming the memory without unloading the model.
        prefix_cache_clear();
        send_prefix_cache(fd);
    } else if (!strcmp(method, "GET") && !strcmp(path, "/health")) {
        send_health(fd);
    } else if (!strcmp(method, "GET") && !strcmp(path, "/v1/models")) {
        send_models(fd);
    } else if (!strcmp(method, "GET") && !strcmp(path, "/v1/capabilities")) {
        send_capabilities(fd);
    } else if (!strcmp(method, "POST") &&
               (!strcmp(path, "/v1/chat/completions") ||
                !strcmp(path, "/v1/responses") ||
                !strcmp(path, "/v1/messages") ||
                !strcmp(path, "/v1/messages/count_tokens") ||
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
            atomic_fetch_add(&SV.active_requests, 1);
            bool ok = true;
            if (SV.n_reg > 0) {
                int sw = swap_to(jv_str(jv_get(req, "model"), NULL));
                if (sw == SWAP_LOAD_FAILED) {
                    send_error(fd, 500,
                               "model failed to load (registered but broken; see server log)");
                    ok = false;
                } else if (sw == SWAP_ABORTED) {
                    send_error(fd, 503,
                               "model load abandoned (unload or shutdown requested; retry)");
                    ok = false;
                } else if (sw < 0) {
                    send_error(fd, 400, "unknown model (see /v1/models)");
                    ok = false;
                }
            }
            if (ok) {
                if (strcmp(path, "/v1/chat/completions") == 0) handle_chat(s, fd, req);
                else if (strcmp(path, "/v1/responses") == 0) handle_responses(s, fd, req);
                else if (strcmp(path, "/v1/messages") == 0) handle_messages(s, fd, req);
                else if (strcmp(path, "/v1/messages/count_tokens") == 0)
                    handle_count_tokens(s, fd, req);
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
            // Drop the request from the count BEFORE the bookkeeping lock:
            // an /unload that saw this request active left pending_unload for
            // us, and the count must already be zero when we honour it.
            atomic_fetch_sub(&SV.active_requests, 1);
            if (SV.n_reg > 0) {
                bool unloaded = false;
                pthread_mutex_lock(&SV.swap_mu);
                SV.last_used = now_s();
                if (SV.pending_unload && !SV.loading &&
                    !atomic_load(&SV.active_requests)) {
                    unload_draft();
                    unload_resident();
                    SV.pending_unload = false;
                    unloaded = true;
                }
                pthread_mutex_unlock(&SV.swap_mu);
                if (unloaded) prefix_cache_clear();
            }
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

// answer tiny GETs from the accept loop: single-slot serving means one long
// generation used to block /health until the gridcore watchdog declared a
// live runner "unhealthy: timed out". /unload is answered here too — it never
// frees anything a slot is using (handle_unload defers under an active load
// or generation), and an operator reclaiming memory must not queue behind
// the very work that holds it. POSTs are handed to a slot untouched.
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
    bool unload = !strncmp(hdr, "GET /unload ", 12);
    if (!health && !models && !caps && !unload) return false;
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
    char *header_end = strstr(hdr, "\r\n\r\n");
    if (!header_end) {
        send_error(fd, 400, "bad request");
        sock_close(fd);
        return true;
    }
    char method[8] = {0}, path[256] = {0};
    char *first_header = NULL;
    size_t content_length = 0;
    if (!parse_request_line(hdr, method, path, &first_header) ||
        !parse_request_framing(first_header, header_end, &content_length) ||
        content_length > 32u * 1024 * 1024) {
        send_error(fd, 400, "invalid request framing");
        sock_close(fd);
        return true;
    }
    if (health) send_health(fd);
    else if (models) send_models(fd);
    else if (unload) handle_unload(fd);
    else        send_capabilities(fd);
    sock_close(fd);
    return true;
}

// ---------------------------------------------------------------- entry

#ifndef _WIN32
static volatile sig_atomic_t stop_requested;
static volatile sig_atomic_t listener_fd = -1;

static void stop_handler(int sig) {
    (void)sig;
    // A second signal is the operator overruling the drain: exit now. _exit is
    // async-signal-safe (128+SIGINT — the shell's convention for a Ctrl-C kill),
    // where the alternative on a pinned drain was reaching for SIGKILL.
    if (stop_requested) _exit(130);
    stop_requested = 1;
    int fd = (int)listener_fd;
    if (fd >= 0) {
        listener_fd = -1;
        close(fd); // async-signal-safe; wakes accept()
    }
}

static void install_stop_handlers(void) {
    struct sigaction sa = {0};
    sa.sa_handler = stop_handler;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);
}

// Between the handlers installing and the listener publishing there is nothing
// a signal can close, so startup polls this at its long stops (model loads)
// and abandons the launch instead of serving a request nobody wants anymore.
static bool stop_was_requested(void) { return stop_requested != 0; }
#else
// Windows port of the same design: stop flag + listener close + drain +
// second-signal escalation, via SetConsoleCtrlHandler. The handler runs on
// its own thread (not an async-signal context), so plain volatile stores and
// closesocket() are safe here.
static volatile LONG win_stop_requested;
static volatile SOCKET win_listener_socket = INVALID_SOCKET;

static BOOL WINAPI win_stop_handler(DWORD ctrl_type) {
    switch (ctrl_type) {
    case CTRL_C_EVENT:
    case CTRL_BREAK_EVENT:
    case CTRL_CLOSE_EVENT:
    case CTRL_SHUTDOWN_EVENT: {
        // A second Ctrl-C is the operator overruling the drain: exit now,
        // with the shell's 128+SIGINT convention for parity with POSIX.
        if (InterlockedExchange(&win_stop_requested, 1)) _exit(130);
        SOCKET s = win_listener_socket;
        if (s != INVALID_SOCKET) {
            win_listener_socket = INVALID_SOCKET;
            closesocket(s); // wakes accept()
        }
        if (ctrl_type == CTRL_CLOSE_EVENT || ctrl_type == CTRL_SHUTDOWN_EVENT) {
            // Returning from these lets Windows terminate the process
            // immediately; hold the handler thread briefly so the drain in
            // main gets its window (Windows grants ~5s on close).
            Sleep(4000);
        }
        return TRUE; // handled: keep running so the drain can finish
    }
    default:
        return FALSE;
    }
}

static void install_stop_handlers(void) {
    win_stop_requested = 0;
    win_listener_socket = INVALID_SOCKET;
    SetConsoleCtrlHandler(win_stop_handler, TRUE);
}

static bool stop_was_requested(void) { return win_stop_requested != 0; }
#endif

static void queue_shutdown(void) {
    int pending[512], n = 0;
    pthread_mutex_lock(&SV.q.mu);
    SV.q.shutdown = true;
    while (SV.q.count > 0) {
        pending[n++] = SV.q.fds[SV.q.head];
        SV.q.head = (SV.q.head + 1) % (int)(sizeof(SV.q.fds) / sizeof(int));
        SV.q.count--;
    }
    pthread_cond_broadcast(&SV.q.cv);
    pthread_mutex_unlock(&SV.q.mu);
    for (int i = 0; i < n; i++) {
        send_error(pending[i], 503, "server shutting down");
        sock_close(pending[i]);
    }
}

static void sched_shutdown(void) {
    if (!SCH.running) return;
    pthread_mutex_lock(&SCH.mu);
    SCH.stop = true;
    pthread_cond_broadcast(&SCH.dec_cv);
    pthread_cond_broadcast(&SCH.slot_cv);
    pthread_mutex_unlock(&SCH.mu);
    pthread_join(SCH.th, NULL);
    model_batch_free(SCH.batch);
    for (int i = 0; i < SV.n_slots; i++) free(SCH.seq[i].logits);
    free(SCH.seq);
    pthread_cond_destroy(&SCH.slot_cv);
    pthread_cond_destroy(&SCH.dec_cv);
    pthread_mutex_destroy(&SCH.dev_mu);
    pthread_mutex_destroy(&SCH.mu);
    memset(&SCH, 0, sizeof(SCH));
}

int server_run(model_t *base, tokenizer *tok, const char *model_path,
               const model_params *mp, sampler defaults,
               const sampler_override *ov, int port, int parallel,
               int n_threads, int ttl, const char *draft_path, int draft_k) {
    sock_init();
#ifndef _WIN32
    stop_requested = 0;
    listener_fd = -1;
#endif
    install_stop_handlers(); // resets the stop flag + listener on both platforms
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
    SV.q.limit = (int)(sizeof(SV.q.fds) / sizeof(int));
    if (getenv("RUNNER_MAX_QUEUE")) {
        int lim = atoi(getenv("RUNNER_MAX_QUEUE"));
        if (lim > 0 && lim < SV.q.limit) SV.q.limit = lim;
    }
    if (getenv("RUNNER_REQUEST_TIMEOUT"))
        SV.req_timeout = atof(getenv("RUNNER_REQUEST_TIMEOUT"));
    // Shared, forkable prompt prefixes. Sized in host RAM rather than as a
    // fraction of anything, because it is the one cache whose useful size is
    // set by the *traffic* (how many distinct system/tool/schema blocks the
    // agents on this box use) and not by the model.
    {
        const char *mb = getenv("RUNNER_PREFIX_CACHE_MB");
        const char *tl = getenv("RUNNER_PREFIX_CACHE_TTL");
        prefix_cache_configure((size_t)(mb ? strtoull(mb, NULL, 10) : 512)
                                   * 1024 * 1024,
                               tl ? atof(tl) : 600.0);
    }
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
        // each slot gets its share of the CPU-forced fallback cap too
        slot_mp.cpu_fallback_threads = mp->cpu_fallback_threads / parallel;

        for (int i = 0; i < parallel; i++) {
            // a Ctrl-C during a multi-slot load means "don't start": honour it
            // between loads rather than serving with the flag silently set
            if (stop_was_requested()) {
                fprintf(stderr, "shutdown requested during startup — exiting\n");
                return 0;
            }
            slot_t *s = &SV.slots[i];
            s->id = i;
            s->tok = tok;
            s->tmpl = tmpl;
            s->smp = defaults;
            s->smp.rng = defaults.rng ^ (0x9E3779B97F4A7C15ull * (unsigned)(i + 1));
            s->smp_base = s->smp;
            if (i == 0) {
                s->m = base;
                // mirror model_load's CPU-forced bump: this replacement pool
                // would otherwise silently undo it for slot 0
                int slot_threads = threads_per_slot;
                if (base->qwen35 &&
                    slot_mp.cpu_fallback_threads > slot_threads)
                    slot_threads = slot_mp.cpu_fallback_threads;
                tpool *replacement = tpool_create(slot_threads);
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
            // a draft the gates rejected at startup stays rejected: only a
            // draft that actually served is worth reloading after /unload
            if (SV.draft) SV.draft_path = draft_path;
            if (!init_swap_runtime(mp, threads_per_slot, ttl)) return 1;
        }
    }

    // last long stop is behind us; a signal from here on is either caught
    // right now or by the published listener below
    if (stop_was_requested()) {
        fprintf(stderr, "shutdown requested during startup — exiting\n");
        return 0;
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
#ifndef _WIN32
    listener_fd = lfd;
#else
    win_listener_socket = (SOCKET)lfd;
#endif

    // Every slot's model exists now, which is the only precondition the batch
    // has. Declining is not an error: sched_generate then runs the untouched
    // solo path, which is what a single slot, swap mode, or a backend without
    // batched kernels all get.
    bool batched = sched_start();

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
                "  POST /v1/chat/completions | POST /v1/responses | POST /v1/completions\n"
                "  GET /v1/models | GET /v1/capabilities | GET /health\n",
                port, SV.n_reg, SV.ttl);
    else
        fprintf(stderr,
                "server listening on http://127.0.0.1:%d — %d slot%s x %d threads%s\n"
                "  POST /v1/chat/completions | POST /v1/responses | POST /v1/completions\n"
                "  GET /v1/models | GET /v1/capabilities | GET /health\n",
                port, parallel, parallel > 1 ? "s" : "", threads_per_slot,
                batched ? ", continuous batching" : "");

    for (;;) {
        // covers the race where the signal landed after the socket-creation
        // check but before listener_fd published: the handler had no fd to
        // close, so accept() would block forever with the flag already set.
        // A signal landing after this check finds listener_fd published and
        // closes it, so accept() fails; no window remains.
        if (stop_was_requested()) break;
        int cfd = (int)accept(lfd, NULL, NULL);
        if (cfd < 0) {
#ifndef _WIN32
            if (stop_requested) break;
#else
            if (win_stop_requested) break;
#endif
            if (errno == EINTR) continue;
            break;
        }
        if (!accept_fastpath(cfd)) q_push(cfd);
    }
    // Stop admission first, fail work that has not started, then allow active
    // requests to finish before dismantling the scheduler and model state.
#ifndef _WIN32
    if (listener_fd >= 0) {
        listener_fd = -1;
        sock_close(lfd);
    }
#else
    if (win_listener_socket != INVALID_SOCKET) {
        win_listener_socket = INVALID_SOCKET;
        sock_close(lfd);
    }
#endif
    queue_shutdown();
    // A worker parked in a --wait-for-vram queue would pin the joins below
    // for up to the full wait; the load gives up at its next poll instead.
    SV.load_cancel = 1;
    for (int i = 0; i < parallel; i++) pthread_join(SV.slots[i].th, NULL);
    sched_shutdown();
    atomic_store(&SV.shutdown, true);
    if (SV.reaper_started) pthread_join(SV.reaper_th, NULL);

    for (int i = 0; i < parallel; i++) free(SV.slots[i].e.hist);
    if (SV.n_reg > 0) {
        pthread_mutex_lock(&SV.swap_mu);
        unload_draft();
        unload_resident();
        pthread_mutex_unlock(&SV.swap_mu);
        pthread_mutex_destroy(&SV.swap_mu);
    } else {
        tokenizer_free(tok);
        for (int i = 0; i < parallel; i++) {
            model_t *draft = SV.slots[i].e.dm;
            if (draft) { model_free(draft); free(draft); }
            model_free(SV.slots[i].m);
            if (i > 0) free(SV.slots[i].m);
        }
    }
    prefix_cache_clear();
    pthread_cond_destroy(&SV.q.cv);
    pthread_mutex_destroy(&SV.q.mu);
    free(SV.slots);
    return 0;
}
