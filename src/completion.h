// One generation, framed in whichever wire dialect the request arrived in.
//
// The four API surfaces are not four engines. They run the SAME generation
// path and differ only in how the result is written back -- ChatCompletion
// chunks, Responses typed events, Anthropic content blocks, or a raw text
// completion. That is why this is one module and one entry point rather than
// a file per surface: the framing has to stay next to the loop that feeds it,
// or the surfaces drift.
//
// Everything else in here -- gen_ctx, resp_doc, the SSE emitters, the stop
// matcher, the UTF-8 hold-back, the logprob writers -- is private. The routes
// reach it through run_completion and the four request readers below.
#ifndef RUNNER_COMPLETION_H
#define RUNNER_COMPLETION_H

#include "json.h"
#include "runner.h"
#include "server_int.h"

// Which wire dialect this request is answered in.
enum { API_TEXT, API_CHAT, API_RESPONSES, API_MESSAGES };

// Generate an answer to `prompt` and write it back on fd in the `api` dialect,
// buffered or streamed as the request asked. `env` is the strict tool-call
// envelope when the request opted into one, else NULL.
void run_completion(slot_t *s, sock_t fd, const char *prompt, int api,
                    jv *req, const tool_envelope *env);

// Request-field readers the routes share with the completion path. `absent`
// distinguishes a missing key from an explicit JSON null, which several
// fields treat differently.
bool absent(const jv *v);
bool request_bool(jv *req, const char *key, bool dflt, bool *out);
// "keep_alive": swap-mode idle seconds. Returns false when malformed.
bool request_keep_alive(jv *req, bool *present, int *seconds);
// The response_format / text.format schema, or NULL when unconstrained.
jv  *request_schema(jv *req);

#endif // RUNNER_COMPLETION_H
