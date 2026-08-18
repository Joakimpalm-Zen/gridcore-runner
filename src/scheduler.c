// Continuous-batching scheduler. Lifted out of server.c (RNR-019); see sched.h.
#include "scheduler.h"
#include "compat.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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
    pthread_cond_t  dev_cv;
    unsigned        dev_next, dev_serving;   // FIFO tickets, see dev_take
    // Whether the device turn is needed at all. Its entire justification is
    // CUDA graph capture: a microbatch captures a graph on its lead sequence's
    // stream, and any other launch in that context breaks the capture. A CPU
    // build has no capture and no shared device context -- every model_t owns
    // its activation scratch and its thread pool, and the weights are read-only
    // (Phase 5 made that explicit) -- so serializing there buys nothing and
    // costs a lot. Measured on Qwen2.5-7B CPU, --parallel 4: a plain request
    // arriving during a grammar-fast-forward generation waited for the WHOLE
    // generation, 10.3 s against 4.8 s alone.
    bool device_turn;
    // decode worker's microbatch scratch [n_slots], owned by sched_start so the
    // worker never allocates and can never fail startup (RNR-005)
    int    *w_idx;
    int32_t *w_tk;
    int    *w_ps;
    float  **w_out;
    // metrics
    atomic_long steps, seqs_batched;
} SCH;

// The device turn is a FIFO turnstile, not a mutex.
//
// It has to be. A prefill yields the turn between chunks so a request that
// arrives mid-prompt does not wait for the whole thing, and that is only a
// hand-off if the queue is ordered: a pthread mutex lets the releasing thread
// -- already on-CPU, with its whole threadpool hot -- re-acquire before the
// woken waiter is scheduled. Measured on Qwen2.5-7B, 2891 prompt tokens:
// release-then-reacquire lost 44 of 45 races and the short request still
// waited 25.3 s of the 26.4 s prefill. sched_yield() did not change it.
//
// Tickets make the order explicit, so giving the turn back means giving it up.
static void dev_take(void) {
    pthread_mutex_lock(&SCH.dev_mu);
    unsigned mine = SCH.dev_next++;
    while (mine != SCH.dev_serving)
        pthread_cond_wait(&SCH.dev_cv, &SCH.dev_mu);
    pthread_mutex_unlock(&SCH.dev_mu);
}

static void dev_give(void) {
    pthread_mutex_lock(&SCH.dev_mu);
    SCH.dev_serving++;
    pthread_cond_broadcast(&SCH.dev_cv);
    pthread_mutex_unlock(&SCH.dev_mu);
}

bool sched_on(void) { return SCH.running; }

void sched_batch_totals(unsigned long long *steps, unsigned long long *seqs) {
    // relaxed: these are counters for a dashboard, not a synchronisation edge,
    // and the two are read one after the other by definition
    if (steps)
        *steps = (unsigned long long)atomic_load_explicit(&SCH.steps,
                                                          memory_order_relaxed);
    if (seqs)
        *seqs = (unsigned long long)atomic_load_explicit(&SCH.seqs_batched,
                                                         memory_order_relaxed);
}

// Absolute deadline for pthread_cond_timedwait, which measures against
// CLOCK_REALTIME — deliberately not now_s(), which is monotonic.
// long long, not long: `long` is 32 bits on the Windows build, and this sum
// starts at up to 1e9 from tv_nsec, so a caller asking for ~1.15 s or more
// wrapped to a negative nanosecond count and produced a deadline in the PAST —
// a timedwait that returns instantly, forever. No caller asks for that today.
static void ts_in(struct timespec *ts, double secs) {
    timespec_get(ts, TIME_UTC);
    long long ns = (long long)ts->tv_nsec + (long long)(secs * 1e9);
    ts->tv_sec  += (time_t)(ns / 1000000000LL);
    ts->tv_nsec  = (long)(ns % 1000000000LL);
}

static void *decode_worker(void *unused) {
    (void)unused;
    int n = SV.n_slots;
    // scratch is preallocated and owned by sched_start (see RNR-005): the
    // worker cannot fail here, so it can never exit leaving slot threads
    // waiting forever for logits it was going to produce
    int     *idx = SCH.w_idx;
    int32_t *tk  = SCH.w_tk;
    int     *ps  = SCH.w_ps;
    float  **out = SCH.w_out;
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
        if (SCH.device_turn) dev_take();
        bool ok = model_batch_decode(SCH.batch, idx, tk, ps, nb, out);
        // out[i] is only valid until the next decode, and the slot threads
        // read it after this one returns — so each row is copied home first.
        if (ok)
            for (int i = 0; i < nb; i++)
                memcpy(SCH.seq[idx[i]].logits, out[i],
                       sizeof(float) * (size_t)n_vocab);
        if (SCH.device_turn) dev_give();
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
    return NULL;   // scratch is owned and freed by sched_shutdown
}

// Batching is enabled only for plain multi-slot serving. Swap mode reloads
// slot 0's model underneath the batch's borrowed pointers, and a single slot
// has nothing to batch with, so both keep the untouched solo path.
// Free every memory buffer SCH may hold. Idempotent and safe on partial state:
// unset pointers are NULL. Does not touch sync primitives — those have their
// own staged teardown, since destroying an uninitialized one is undefined.
static void sched_free_buffers(void) {
    if (SCH.seq)
        for (int i = 0; i < SV.n_slots; i++) free(SCH.seq[i].logits);
    free(SCH.seq);   SCH.seq   = NULL;
    model_batch_free(SCH.batch); SCH.batch = NULL;
    free(SCH.w_idx); SCH.w_idx = NULL;
    free(SCH.w_tk);  SCH.w_tk  = NULL;
    free(SCH.w_ps);  SCH.w_ps  = NULL;
    free(SCH.w_out); SCH.w_out = NULL;
}

bool sched_start(void) {
    if (SV.n_slots < 2 || SV.n_reg > 0) return false;
    int n = SV.n_slots;
    model_t **ms = malloc(sizeof(model_t *) * (size_t)n);
    SCH.seq   = calloc((size_t)n, sizeof(seq_t));
    // the decode worker's microbatch scratch lives here, not in the worker, so
    // it is allocated (and checked) before "running" is ever published
    SCH.w_idx = malloc(sizeof(int)     * (size_t)n);
    SCH.w_tk  = malloc(sizeof(int32_t) * (size_t)n);
    SCH.w_ps  = malloc(sizeof(int)     * (size_t)n);
    SCH.w_out = malloc(sizeof(float *) * (size_t)n);
    if (!ms || !SCH.seq || !SCH.w_idx || !SCH.w_tk || !SCH.w_ps || !SCH.w_out) {
        free(ms); sched_free_buffers(); return false;
    }
    for (int i = 0; i < n; i++) ms[i] = SV.slots[i].m;
    int n_vocab = SV.slots[0].m->n_vocab;
    for (int i = 0; i < n; i++) {
        SCH.seq[i].logits = malloc(sizeof(float) * (size_t)n_vocab);
        if (!SCH.seq[i].logits) { free(ms); sched_free_buffers(); return false; }
    }
    // Only a GPU-backed model needs the device turn; see the field's comment.
    SCH.device_turn = SV.slots[0].m->gpu != NULL;
    SCH.batch = model_batch_create(ms, n);
    free(ms);
    if (!SCH.batch) { sched_free_buffers(); return false; }

    // sync primitives — torn down in reverse on any subsequent failure
    if (pthread_mutex_init(&SCH.mu, NULL) != 0) { sched_free_buffers(); return false; }
    SCH.dev_next = SCH.dev_serving = 0;
    if (pthread_cond_init(&SCH.dev_cv, NULL) != 0) {
        pthread_mutex_destroy(&SCH.mu);
        sched_free_buffers(); return false;
    }
    if (pthread_mutex_init(&SCH.dev_mu, NULL) != 0) {
        pthread_cond_destroy(&SCH.dev_cv); pthread_mutex_destroy(&SCH.mu);
        sched_free_buffers(); return false;
    }
    if (pthread_cond_init(&SCH.dec_cv, NULL) != 0) {
        pthread_cond_destroy(&SCH.dev_cv);
        pthread_mutex_destroy(&SCH.dev_mu); pthread_mutex_destroy(&SCH.mu);
        sched_free_buffers(); return false;
    }
    if (pthread_cond_init(&SCH.slot_cv, NULL) != 0) {
        pthread_cond_destroy(&SCH.dec_cv);
        pthread_cond_destroy(&SCH.dev_cv);
    pthread_mutex_destroy(&SCH.dev_mu); pthread_mutex_destroy(&SCH.mu);
        sched_free_buffers(); return false;
    }

    // Everything the worker needs now exists — only here is it safe to publish
    // running and launch the worker. Because the worker allocates nothing, it
    // cannot fail startup and strand slot threads on logits that never arrive.
    SCH.running = true;
    if (pthread_create(&SCH.th, NULL, decode_worker, NULL) != 0) {
        SCH.running = false;
        pthread_cond_destroy(&SCH.slot_cv); pthread_cond_destroy(&SCH.dec_cv);
        pthread_cond_destroy(&SCH.dev_cv);
    pthread_mutex_destroy(&SCH.dev_mu); pthread_mutex_destroy(&SCH.mu);
        sched_free_buffers();
        return false;
    }
    return true;
}

// Prefill runs per sequence on model_forward_batch, interleaved between decode
// steps rather than inside them: a prefill tile and a decode microbatch are
// different kernel shapes. The sequence is simply absent from the batch while
// this is held, which costs nothing — it is named per step, not enrolled.
void sched_prefill_begin(void) {
    if (sched_on() && SCH.device_turn) dev_take();
}

void sched_prefill_end(void) {
    if (sched_on() && SCH.device_turn) dev_give();
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
int sched_generate(slot_t *s, float *logits, int max_new,
                          gen_cb cb, void *ud, double *gen_time,
                          double deadline) {
    engine *e = &s->e;
    // Speculative decoding drives its own batched forwards for one sequence,
    // which is the tile shape and not the microbatch shape; it stays solo.
    // engine_wants_spec covers grammar fast-forward without a draft model.
    if (!sched_on() || engine_wants_spec(e)) {
        // A solo generation on a batching server still has to take the device
        // turn: its model_forward launches would break a concurrent
        // microbatch's graph capture just as a prefill would.
        if (!sched_on() || !SCH.device_turn)
            return engine_generate(e, logits, max_new, cb, ud, gen_time);
        // NOTE: this holds the turn for the WHOLE generation, not per forward,
        // so on a GPU one speculative request still stalls every other slot for
        // its full length. Taking the turn per forward needs engine_generate to
        // call a hook around each one; filed, not done.
        dev_take();
        int n = engine_generate(e, logits, max_new, cb, ud, gen_time);
        dev_give();
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

void sched_shutdown(void) {
    if (!SCH.running) return;
    pthread_mutex_lock(&SCH.mu);
    SCH.stop = true;
    pthread_cond_broadcast(&SCH.dec_cv);
    pthread_cond_broadcast(&SCH.slot_cv);
    pthread_mutex_unlock(&SCH.mu);
    pthread_join(SCH.th, NULL);
    pthread_cond_destroy(&SCH.slot_cv);
    pthread_cond_destroy(&SCH.dec_cv);
    pthread_cond_destroy(&SCH.dev_cv);
    pthread_mutex_destroy(&SCH.dev_mu);
    pthread_mutex_destroy(&SCH.mu);
    sched_free_buffers();
    memset(&SCH, 0, sizeof(SCH));
}
