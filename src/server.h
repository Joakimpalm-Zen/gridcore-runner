// Server lifecycle: run one HTTP server instance and request its shutdown.
#ifndef RUNNER_SERVER_H
#define RUNNER_SERVER_H

#include "runner.h"

int server_run(model_t *base, tokenizer *tok, const char *model_path,
               const model_params *mp, sampler defaults,
               const sampler_override *ov, int port, int parallel,
               int n_threads, int ttl, const char *draft_path, int draft_k,
               bool ignore_eos);

// Request the same graceful stop as the platform's first SIGINT / console
// control event. This closes the listener so a server_run blocked in accept()
// can drain and return. A second request keeps the operator-facing escalation
// behavior and exits immediately.
void server_request_stop(void);

#endif // RUNNER_SERVER_H
