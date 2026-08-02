// The two inbound protocol adapters.
//
// runner answers on four API surfaces and runs one generation path. Two of
// those surfaces arrive in their own request vocabulary -- OpenAI's Responses
// API and Anthropic's Messages API -- and each is translated into the chat
// shape before it reaches run_completion. That is the whole of what these
// modules do: read a foreign request, reject what runner does not implement
// with a message in that API's own error shape, and produce a prompt.
//
// Keeping the translation OUT of the generation path is what stops the
// surfaces becoming three engines. The outbound half -- framing the answer
// back into typed events or content blocks -- is in completion.c for the
// mirror-image reason: it has to stay next to the loop that feeds it.
#ifndef RUNNER_API_H
#define RUNNER_API_H

#include "json.h"
#include "runner.h"
#include "completion.h"
#include "http.h"
#include "server_int.h"

// POST /v1/responses
void handle_responses(slot_t *s, sock_t fd, jv *req);
// POST /v1/messages and POST /v1/messages/count_tokens
void handle_messages(slot_t *s, sock_t fd, jv *req);
void handle_count_tokens(slot_t *s, sock_t fd, jv *req);

#endif // RUNNER_API_H
