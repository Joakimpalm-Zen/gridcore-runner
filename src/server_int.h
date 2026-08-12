// Shared server state: what the slot workers, the scheduler, the residency
// layer and the routes all touch.
//
// This header does not make SV less of a global -- it makes it a NAMED global
// with a declared type, so the modules split out of server.c under RNR-019 can
// be separate translation units at all. Turning SV into a context passed down
// from server_run is the remaining half of that finding and a real behavioural
// change, not a move; it is deliberately not attempted here.
#ifndef RUNNER_SERVER_INT_H
#define RUNNER_SERVER_INT_H

#include <pthread.h>
#include <stdatomic.h>

#include "runner.h"
#include "http.h"

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
    sock_t fds[512];
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

typedef struct {
    slot_t    *slots;
    int        n_slots;
    fdqueue    q;
    const char *model_name;
    int        n_predict_cap;
    atomic_int ctx_size;
    atomic_int req_counter;
    // swap mode
    reg_entry   reg[RUNNER_MAX_MODELS];
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
    // `pending_unload` are protected by swap_mu; `load_cancel` is a lock-free
    // atomic flag the vram wait polls (volatile is not cross-thread
    // synchronization — RNR-008).
    bool         loading;         // a swap_to load is in flight, lock released
    bool         pending_unload;  // /unload arrived while busy; honoured at the next safe point
    atomic_int   load_cancel;     // aborts a --wait-for-vram queue wait
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
    bool        ignore_eos;   // process-level --ignore-eos, including reloads
} server_state;

extern server_state SV;

// Cumulative work done by this process, for a capacity dashboard. The state
// lives in completion.c, beside the single place every surface's generation
// passes through -- not in server_state, because the tools that link the
// completion path without the server (test-responses-sm) must still resolve
// the recording call.
//
// Raw monotonic totals on purpose: a tok/s field would bake in an averaging
// window the runner has no business choosing, and would be wrong for every
// consumer whose window differs. The engine measures; the suite presents.
void server_record_work(int n_prompt, int n_gen, double gen_seconds);
void server_work_totals(unsigned long long *prompt_tokens,
                        unsigned long long *gen_tokens, double *gen_seconds);

// swap_to results below 0: the name matched no registry entry (a caller
// typo -- 400) vs the entry exists but its model failed to load (a broken
// model -- 5xx) vs the load was discarded because /unload or shutdown arrived
// while it ran (nobody's fault -- 503). Callers must tell them apart.
#define SWAP_UNKNOWN     (-1)
#define SWAP_LOAD_FAILED (-2)
#define SWAP_ABORTED     (-3)

// ---- residency and admission (registry.c) ----

int  resident_load(void);
void resident_store(int v);
int  context_load(void);
void context_store(int value);
// Both assume SV.swap_mu is held by the caller.
void unload_resident(void);
void unload_draft(void);
void handle_unload(sock_t fd);
// Resolve + load the requested model; returns the registry index or a SWAP_*.
int  swap_to(const char *want);
bool validate_single_model_request(sock_t fd, jv *req);
bool init_swap_runtime(const model_params *mp, int n_threads, int ttl);
// Admission: q_push sheds with 503 when the queue is full, q_pop blocks until
// a connection arrives or the queue is shut down (SOCK_INVALID then).
void   q_push(sock_t fd);
sock_t q_pop(void);
// Drain the queue at shutdown: every waiting connection is told 503
// rather than dropped, and the workers blocked in q_pop are woken.
void   queue_shutdown(void);

#endif // RUNNER_SERVER_INT_H
