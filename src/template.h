// Chat templates, tool-call envelopes, thinking-tag splitting.
#ifndef RUNNER_TEMPLATE_H
#define RUNNER_TEMPLATE_H

#include <stddef.h>
#include <stdbool.h>
#include "tokenizer.h"

// MISTRAL is LLAMA2's [INST] framing without the <<SYS>> block: Mistral's own
// template rejects a system role outright, so the text is folded into the
// first user turn rather than wrapped in markers it never saw in training.
enum { TMPL_CHATML, TMPL_LLAMA2, TMPL_LLAMA3, TMPL_ZEPHYR, TMPL_GEMMA,
       TMPL_GEMMA4, TMPL_MISTRAL, TMPL_PHI3, TMPL_APERTUS, TMPL_ORNITH,
       TMPL_RAW };
typedef struct { const char *role, *content; } chat_msg;
int         template_detect(const char *meta_tmpl, tokenizer *tok);
int         template_from_name(const char *name); // -1 if unknown
const char *template_name(int tmpl);
// render messages; add_assistant appends the assistant generation prefix.
// returns bytes written (excl. NUL)
//
// enable_thinking currently affects TMPL_GEMMA4 only, and it defaults to FALSE
// because that is what the reference template defaults to:
//
//     {{- '<|turn>model\n' -}}
//     {%- if not enable_thinking | default(false) -%}
//         {{- '<|channel>thought\n<channel|>' -}}
//     {%- endif -%}
//
// (llama.cpp models/templates/google-gemma-4-31B-it.jinja). Passing true
// suppresses the empty-thought pre-seed and lets the model open its own
// thought block — upstream's opt-in branch, not the default.
//
// Until 2026-08-08 runner had no parameter here and always rendered the
// thinking-enabled shape, so it disagreed with every reference-following
// engine on every gemma-4 turn. That is the likely cause of tool prompts that
// opened a thought block and never closed it, returning one byte from a
// hundred generated tokens.
size_t render_messages(int tmpl, const chat_msg *msgs, int n_msgs,
                       bool add_assistant, bool enable_thinking,
                       char *out, size_t cap);

// chat tool-call convention (template.c; sbuf/jv live in json.h)
struct sbuf;
struct jv;
// read the per-request thinking opt-in (top level or chat_template_kwargs);
// false when absent, which is the reference template's own default
bool req_enable_thinking(struct jv *req);
// render OpenAI "tools" declarations as a system turn (no-op when absent)
void tools_render(const struct jv *tools, struct sbuf *out);
void tools_render_for(int tmpl, const struct jv *tools, struct sbuf *out);
void tool_history_render_for(int tmpl, const struct jv *calls, struct sbuf *out);
// parse tool-call blocks out of content into OpenAI tool_calls items;
// returns the call count, content is compacted in place
int  tool_calls_parse(struct sbuf *content, struct sbuf *tc);
int  tool_calls_parse_for(int tmpl, struct sbuf *content, struct sbuf *tc);
// ------------------------------------------- strict tool-call envelope
//
// OpenAI tools[] compiled into a discriminated union — one branch per tool
// plus a `final` branch for an ordinary or schema-constrained answer:
//
//   {"tool":"get_weather","args":{"city":"Oslo"}}
//   {"tool":"final","args":{"content":"it is cold"}}
//
// Constraining sampling to that union is what makes the guarantee: the model
// cannot name a tool that was not declared, cannot invent an argument key or
// get its type wrong, and a max_tokens truncation closes to a document that
// is still a legal call. That replaces parsing hopeful output afterward,
// which is what tool_calls_parse above does and remains as the fallback for
// requests that do not opt in.
enum { TCH_AUTO, TCH_REQUIRED, TCH_NONE, TCH_NAMED };

typedef struct {
    int   kind;           // TCH_*
    char *schema_src;     // envelope JSON schema, for schema_compile (owned)
    char *system_turn;    // system message teaching the envelope (owned)
    bool  final_is_text;  // final branch is {"content": "..."} rather than
                          // the caller's own response_format schema
    // parallel_tool_calls: the document becomes {"calls":[<entry>, ...]} —
    // one uniform array whose items are the same discriminated union, so a
    // direct answer is just a one-element array holding the final branch.
    // Bounded by construction: an unbounded array under a token budget is a
    // truncation waiting to happen.
    bool  parallel;
    int   max_calls;
} tool_envelope;

// Build the envelope for one request. `final_schema` is the caller's
// response_format schema, or NULL for a plain text answer. Returns
//    1  strict mode applies; free with tool_envelope_free
//    0  strict mode does not apply (no tools, or tool_choice "none")
//   -1  malformed tools / tool_choice; err holds the client-facing reason
int  tool_envelope_build(struct jv *tools, struct jv *tool_choice,
                         struct jv *final_schema, tool_envelope *out,
                         char *err, int errcap);
// Same, but `parallel` opts into the bounded multi-call form. Buffered
// surfaces only: the streaming demultiplexer still tracks one call per turn.
int  tool_envelope_build_ex(struct jv *tools, struct jv *tool_choice,
                            struct jv *final_schema, bool parallel,
                            tool_envelope *out, char *err, int errcap);
void tool_envelope_free(tool_envelope *e);

// Map a generated envelope document back to the OpenAI response shape.
// Returns the NUMBER of tool_calls[] items appended to tc (1 for the ordinary
// single-call envelope, 0..max_calls for the parallel form), 0 for the final
// branch with assistant content appended instead, and -1 when doc is not a
// well-formed envelope. A parallel document may mix the two: any final
// branches contribute content, any tool branches contribute calls.
int  tool_envelope_map(const tool_envelope *e, const char *doc, size_t n,
                       struct sbuf *content, struct sbuf *tc);

// Streaming counterpart of tool_envelope_map.
//
// tool_envelope_map needs the whole document, which is exactly what a stream
// does not have: the envelope is decided one token at a time, and the client
// must see the decision as it happens rather than after the fact. This is the
// same mapping run incrementally — bytes in, demultiplexed events out — so a
// streamed request reaches the same call as a buffered one.
//
// Nothing is emitted until the branch is known, which is what keeps envelope
// syntax out of the client's `content`: the discriminator is buffered, and by
// the time anything is forwarded it is already known to be either assistant
// text or tool arguments. Argument text is forwarded raw (insignificant
// whitespace removed) so it stays a JSON *string* the caller can execute;
// `final` text is unescaped, matching what the buffered path hands back.
//
// Every callback returns non-zero to stop generation (the client went away);
// that result propagates out of tool_stream_feed unchanged.
typedef struct {
    void *ud;
    int (*content)(void *ud, const char *bytes, int n);
    int (*call_begin)(void *ud, const char *name);
    int (*call_args)(void *ud, const char *bytes, int n);
} tool_stream_sink;

typedef struct {
    const tool_envelope *env;
    tool_stream_sink     sink;
    int   state;
    char  *head;          // undecided prefix, held back from the client
    size_t head_n, head_cap;
    char *name;           // selected branch, once known (owned)
    bool  called;         // a tool branch, rather than `final`, was selected
    int   depth;          // nesting inside the value being forwarded
    bool  started;        // the forwarded value has produced its first byte
    bool  in_str, esc;    // JSON string state within that value
    char  pend[16];       // partial escape sequence awaiting more bytes
    int   n_pend;
} tool_stream;

void tool_stream_init(tool_stream *s, const tool_envelope *e,
                      const tool_stream_sink *sink);
int  tool_stream_feed(tool_stream *s, const char *bytes, int n);
// true once a tool branch was selected, i.e. finish_reason is "tool_calls"
bool tool_stream_called(const tool_stream *s);
void tool_stream_free(tool_stream *s);

// streaming splitter for thinking-tag models: bytes between
// open and close tags, including architectures that interleave them with
// plain text — reach the callback as reasoning (reasoning=1), the rest as
// content (reasoning=0). Partial tags at chunk boundaries are held back until
// they resolve. With open == NULL every byte passes straight through.
typedef int (*think_cb)(void *ud, int reasoning, const char *bytes, int n);
typedef struct {
    const char *open, *close;
    int   state;
    char *buf;
    int   n, cap;
} think_split;
void think_init(think_split *t, const char *open, const char *close);
void think_init_reasoning(think_split *t, const char *open, const char *close);
int  think_feed(think_split *t, const char *bytes, int n, think_cb cb, void *ud);
int  think_finish(think_split *t, think_cb cb, void *ud); // flush held bytes
void think_free(think_split *t);

#endif // RUNNER_TEMPLATE_H
