// Model residency and connection admission. Lifted out of server.c (RNR-019).
//
// Residency is the -m swap registry: at most one model resident at a time,
// loaded on demand, unloaded after --ttl idle seconds or on POST /unload. The
// load itself deliberately runs without SV.swap_mu held -- a multi-GB mmap or
// a --wait-for-vram queue wait must never block /unload or shutdown.
//
// Admission is the connection queue, which is the server's backpressure.
#include "server_int.h"
#include "json.h"
#include "compat.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

server_state SV;

int resident_load(void) {
    return atomic_load_explicit(&SV.resident, memory_order_relaxed);
}

void resident_store(int v) {
    atomic_store_explicit(&SV.resident, v, memory_order_relaxed);
}

int context_load(void) {
    return atomic_load_explicit(&SV.ctx_size, memory_order_relaxed);
}

void context_store(int value) {
    atomic_store_explicit(&SV.ctx_size, value, memory_order_relaxed);
}

void unload_resident(void) {
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

void unload_draft(void) {
    if (!SV.draft) return;
    model_free(SV.draft);
    free(SV.draft);
    SV.draft = NULL;
    for (int i = 0; i < SV.n_slots; i++) SV.slots[i].e.dm = NULL;
}

// Answer POST /unload from any thread. The memory is given back immediately
// when the server is idle; when a load or a generation is in flight the wish
// is recorded and honoured at the next safe boundary instead of blocking the
// caller — /unload exists so an operator can reclaim memory, and "scheduled"
// answered now beats "done" answered after a 300s --wait-for-vram queue. An
// in-flight load is additionally cancelled at its next wait poll.
void handle_unload(sock_t fd) {
    bool deferred = false;
    if (SV.n_reg > 0) {
        pthread_mutex_lock(&SV.swap_mu);
        if (SV.loading) {
            SV.pending_unload = true;
            atomic_store(&SV.load_cancel, 1);   // a queued --wait-for-vram load gives up now
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
static bool request_model_matches(const char *want, const char *served) {
    if (!want || !*want) return true;
    if (served && !strcmp(want, served)) return true;
    // Local clients often address an attached single-model runner by a stable
    // engine tag rather than the GGUF basename. Keep those aliases explicit so
    // "gpt-4o" or any other arbitrary provider name is not silently accepted.
    static const char *const aliases[] = { "runner", "default", "local", "test", "clu" };
    for (size_t i = 0; i < sizeof aliases / sizeof *aliases; i++)
        if (!strcmp(want, aliases[i])) return true;
    return false;
}

// resolve + load the requested model; returns entry index or SWAP_*
int swap_to(const char *want) {
    int idx = 0; // default: first entry
    if (want && *want) {
        idx = -1;
        for (int i = 0; i < SV.n_reg; i++)
            if (!strcmp(SV.reg[i].name, want)) { idx = i; break; }
        if (idx < 0 && request_model_matches(want, SV.reg[0].name))
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
        atomic_store(&SV.load_cancel, 0);
        pthread_mutex_unlock(&SV.swap_mu);

        fprintf(stderr, "swap: loading %s (%s)\n",
                SV.reg[idx].name, SV.reg[idx].path);
        model_t *m = calloc(1, sizeof(model_t));
        tokenizer *tok = calloc(1, sizeof(tokenizer));
        bool model_ok = m && model_load(m, SV.reg[idx].path, &SV.mp);
        bool tok_ok = model_ok && tok && tokenizer_init(tok, &m->gf);

        pthread_mutex_lock(&SV.swap_mu);
        SV.loading = false;
        bool discard = SV.pending_unload || atomic_load(&SV.load_cancel);
        SV.pending_unload = false;
        if (!model_ok || !tok_ok || discard) {
            if (discard)
                fprintf(stderr, "swap: load of %s discarded (%s)\n",
                        SV.reg[idx].name, model_ok ? "unloaded while loading"
                                                   : "wait cancelled");
            else
                fprintf(stderr, "swap: failed to load %s\n", SV.reg[idx].name);
            // tokenizer_init may have allocated buffers before failing (tok_ok
            // false); tokenizer_free is safe on a calloc'd/partial tokenizer and
            // frees them. Guard only against a NULL tok (calloc failure).
            if (tok) tokenizer_free(tok);
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
        char ident[256];
        sampler_ident(gguf_get_str(&s->m->gf, "general.name", NULL),
                      s->m->path, ident, sizeof(ident));
        const sampler_preset *sp =
            sampler_resolve(&s->smp, s->m->arch, ident, &SV.ov);
        s->smp_base = s->smp;
        SV.preset_name = sp->name;
        char sdesc[256];
        sampler_describe(&s->smp, sp, sdesc, sizeof(sdesc));
        fprintf(stderr, "sampling: %s\n", sdesc);
        if (!engine_init(&s->e, s->m, s->tok, &s->smp)) {
            // mirror the failed-load cleanup: leave the slot empty and the
            // resident unchanged rather than committing a half-built engine
            fprintf(stderr, "swap: out of memory initializing engine for %s\n",
                    SV.reg[idx].name);
            model_free(s->m); free(s->m); s->m = NULL;
            tokenizer_free(s->tok); free(s->tok); s->tok = NULL;
            pthread_mutex_unlock(&SV.swap_mu);
            return SWAP_LOAD_FAILED;
        }
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

bool validate_single_model_request(sock_t fd, jv *req) {
    const char *want = jv_str(jv_get(req, "model"), NULL);
    if (request_model_matches(want, SV.model_name)) return true;
    send_error_detail(fd, 404, "unknown model (see /v1/models)",
                      "model", "model_not_found");
    return false;
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

bool init_swap_runtime(const model_params *mp, int n_threads, int ttl) {
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
void q_push(sock_t fd) {
    bool room = false;
    // A worker awakened for the previous connection may not have run before
    // accept() returns the next one. Give that handoff a few scheduler ticks;
    // otherwise an idle slot's first request transiently fills the queue and
    // the following request is shed even though it should be the one queued.
    for (int retry = 0; retry < 10; retry++) {
        pthread_mutex_lock(&SV.q.mu);
        room = SV.q.count < SV.q.limit;
        if (room) break;
        pthread_mutex_unlock(&SV.q.mu);
        plat_sleep_ms(1);
    }
    if (!room) pthread_mutex_lock(&SV.q.mu);
    if (room) {
        SV.q.fds[SV.q.tail] = fd;
        SV.q.tail = (SV.q.tail + 1) % (int)(sizeof(SV.q.fds) / sizeof(sock_t));
        SV.q.count++;
        pthread_cond_signal(&SV.q.cv);
    }
    pthread_mutex_unlock(&SV.q.mu);
    if (!room) {
        send_error(fd, 503, "server queue full");
        sock_close(fd);
    }
}

sock_t q_pop(void) {
    pthread_mutex_lock(&SV.q.mu);
    while (SV.q.count == 0 && !SV.q.shutdown)
        pthread_cond_wait(&SV.q.cv, &SV.q.mu);
    sock_t fd = SOCK_INVALID;
    if (SV.q.count > 0) {
        fd = SV.q.fds[SV.q.head];
        SV.q.head = (SV.q.head + 1) % (int)(sizeof(SV.q.fds) / sizeof(sock_t));
        SV.q.count--;
    }
    pthread_mutex_unlock(&SV.q.mu);
    return fd;
}

void queue_shutdown(void) {
    sock_t pending[512];
    int n = 0;
    pthread_mutex_lock(&SV.q.mu);
    SV.q.shutdown = true;
    while (SV.q.count > 0) {
        pending[n++] = SV.q.fds[SV.q.head];
        SV.q.head = (SV.q.head + 1) % (int)(sizeof(SV.q.fds) / sizeof(sock_t));
        SV.q.count--;
    }
    pthread_cond_broadcast(&SV.q.cv);
    pthread_mutex_unlock(&SV.q.mu);
    for (int i = 0; i < n; i++) {
        send_error(pending[i], 503, "server shutting down");
        sock_close(pending[i]);
    }
}
