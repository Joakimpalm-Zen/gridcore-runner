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

// Render a prompt into a buffer guaranteed to hold ALL of it (server.c).
//
// render_messages_with_tools() truncates silently at `cap`, and it truncates
// the TAIL: the newest user turn and the generation header. A short buffer
// therefore does not fail the request, it changes the question the model is
// asked. Every surface used to size that buffer from a hand-rolled estimate
// summed over the turns, and every such estimate is one unaccounted addition
// away from doing exactly that -- Harmony added two in one commit (the
// `# Tools` namespace and the replayed `reasoning_content` analysis turn),
// which is the argument against fixing it with a bigger constant.
//
// So the size is measured rather than guessed: render, and if the buffer came
// back full, grow and render again. `hint` is a first guess only; too small
// costs one extra render pass, never a truncated prompt.
//
// Returns a heap prompt the caller frees, or NULL when allocation failed.
char *render_prompt_alloc(int tmpl, const chat_msg *msgs, int n_msgs,
                          bool add_assistant, int thinking, const jv *tools,
                          size_t hint);

// POST /v1/responses
void handle_responses(slot_t *s, sock_t fd, jv *req);
// POST /v1/messages and POST /v1/messages/count_tokens
void handle_messages(slot_t *s, sock_t fd, jv *req);
void handle_count_tokens(slot_t *s, sock_t fd, jv *req);

#endif // RUNNER_API_H
