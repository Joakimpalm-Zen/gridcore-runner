// Server lifecycle: run one HTTP server instance and request its shutdown.
#ifndef RUNNER_SERVER_H
#define RUNNER_SERVER_H

#include "runner.h"

// `tmpl_override` is --chat-template resolved to a TMPL_* value, or -1 to
// detect it from the model. It is a single model's template, so the caller
// must not pass one with a swap set: main.c refuses that combination rather
// than picking a member to be right about. Applies to the initial load AND to
// every reload of the same model after /unload or a --ttl expiry, which is
// where detection would otherwise quietly take the override back.
int server_run(model_t *base, tokenizer *tok, const char *model_path,
               const model_params *mp, sampler defaults,
               const sampler_override *ov, int port, int parallel,
               int n_threads, int ttl, const char *draft_path, int draft_k,
               bool ignore_eos, int tmpl_override, bool force_uncertified);

// Request the same graceful stop as the platform's first SIGINT / console
// control event. This closes the listener so a server_run blocked in accept()
// can drain and return. A second request keeps the operator-facing escalation
// behavior and exits immediately.
void server_request_stop(void);

#endif // RUNNER_SERVER_H
