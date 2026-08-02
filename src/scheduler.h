// Continuous batching: the scheduler that decides which sequences decode
// together. See sched.c for why there is one decode thread rather than one
// per slot.
//
// The interface is deliberately six functions wide. SCH -- the batch, the
// per-sequence states, the device turn, the worker scratch -- is private to
// sched.c and nothing outside it names any of that.
#ifndef RUNNER_SCHEDULER_H
#define RUNNER_SCHEDULER_H

#include "server_int.h"

// Is batching running at all? A single-slot server never starts it.
bool sched_on(void);
// Start / tear down the decode thread. sched_start allocates every buffer the
// worker will need, so the worker itself can never fail (RNR-005).
bool sched_start(void);
void sched_shutdown(void);
// Prefill and decode must interleave rather than overlap -- a microbatch
// captures a CUDA graph, and any other launch in the context breaks capture.
// A slot takes the device turn around its own prefill with these.
void sched_prefill_begin(void);
void sched_prefill_end(void);
// engine_generate's contract, served out of shared microbatches. The per-token
// work is the engine's own, so a request's tokens do not depend on how busy
// the server was; all this changes is where the forward happens.
int  sched_generate(slot_t *s, float *logits, int max_new,
                    gen_cb cb, void *ud, double *gen_time, double deadline);

#endif // RUNNER_SCHEDULER_H
