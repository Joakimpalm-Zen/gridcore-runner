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
       TMPL_MUSE, TMPL_GRANITE,
       // gpt-oss / OpenAI Harmony. Channel-structured: the assistant writes
       // an `analysis` message before its `final` one, and both ride the
       // same <|start|>role<|channel|>name<|message|> framing.
       TMPL_HARMONY,
       TMPL_RAW,
       // ChatML whose own template declares <think>: Qwen3 and relatives.
       // Split from TMPL_CHATML only so the thinking control below has
       // somewhere to attach -- rendering is otherwise identical, and every
       // tmpl-dependent branch elsewhere keys on TMPL_ORNITH, so this falls
       // into the same generic path TMPL_CHATML does.
       TMPL_CHATML_THINK };

// How the generation prompt should treat a thinking model.
//
// THINK_DEFAULT means "whatever this model family's reference template does",
// and that is deliberately NOT one answer for all families:
//   gemma-4  reference defaults enable_thinking FALSE -> pre-seed an empty
//            thought block, suppressing reasoning
//   Qwen3    reference defaults enable_thinking TRUE  -> emit nothing, the
//            model opens its own <think>
// A single boolean cannot express that, which is why this is tri-state: the
// absence of a request field has to mean "match the reference", not "false".
enum { THINK_DEFAULT = 0, THINK_ON, THINK_OFF };
struct jv;
typedef struct {
    const char *role, *content;
    // Native templates whose tool-result turn names the invoked function use
    // this optional field. Ordinary messages leave it NULL.
    const char *name;
} chat_msg;
int         template_detect(const char *meta_tmpl, tokenizer *tok);
int         template_from_name(const char *name); // -1 if unknown
const char *template_name(int tmpl);
// render messages; add_assistant appends the assistant generation prefix.
// returns bytes written (excl. NUL)
//
// `thinking` is THINK_DEFAULT / THINK_ON / THINK_OFF.
//
//   TMPL_CHATML_THINK  Qwen/Qwen3-* tokenizer_config.json
//     enable_thinking defaults TRUE; the false branch appends
//     '<think>\n\n</think>\n\n' after the assistant header. So DEFAULT and ON
//     emit nothing extra and OFF appends the closed block.
//
//   TMPL_GEMMA4        gemma-4 tokenizer.chat_template (read from the GGUF)
//     The generation prompt is '<|turn>model\n' and nothing else, in EITHER
//     mode -- there is no pre-seeded thought block. After a tool response it
//     emits nothing at all, or an OPEN '<|channel>thought\n' when thinking.
//     Selecting thinking for this family happens elsewhere: the template sets
//     enable_thinking | default(false) and, when true, injects '<|think|>\n'
//     into the FIRST SYSTEM TURN. That is not implemented here, so THINK_ON is
//     accepted and ignored rather than approximated.
//
// THINK_DEFAULT means "render what this family's own template renders", which
// is not the same answer for both -- hence tri-state rather than a bool.
//
// History worth keeping: on 2026-08-08 this file claimed gemma-4's reference
// pre-seeds an empty thought block when thinking is off, and the renderer was
// changed to match. That came from a web summary of llama.cpp's copy of the
// template, not the model's own, and it was wrong. gemma-4-E2B's planning
// score fell 0.575 -> 0.300 and it began emitting reasoning prose as its
// visible answer. Read the artifact, not a description of it.
size_t render_messages(int tmpl, const chat_msg *msgs, int n_msgs,
                       bool add_assistant, int thinking,
                       char *out, size_t cap);
// Structured-tools variant used by native templates. The legacy entry above
// is exactly this call with tools == NULL.
size_t render_messages_with_tools(int tmpl, const chat_msg *msgs, int n_msgs,
                                  bool add_assistant, int thinking,
                                  const struct jv *tools,
                                  char *out, size_t cap);

// chat tool-call convention (template.c; sbuf/jv live in json.h)
struct sbuf;
struct jv;
// read the per-request thinking control (top level or chat_template_kwargs);
// THINK_DEFAULT when absent, so a silent request renders what the model's own
// reference template would render
int req_thinking_mode(struct jv *req);
// render OpenAI "tools" declarations as a system turn (no-op when absent)
void tools_render(const struct jv *tools, struct sbuf *out);
void tools_render_for(int tmpl, const struct jv *tools, struct sbuf *out);
void tool_history_render_for(int tmpl, const struct jv *calls, struct sbuf *out);
// Resolve a tool result's native turn name from message.name or from its
// tool_call_id and a preceding assistant tool_calls entry. Borrowed pointer.
const char *tool_result_name(const struct jv *messages, int message_index);
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
    bool  atem;           // Muse native recipient + <atem:invoke> protocol
    bool  muse_user_header; // generic JSON override follows to=user header
    bool  muse_plain_payload; // stream a schema payload after to=user header
    struct jv *tools;     // borrowed request declarations for native compiler
    char *named;          // owned named-tool choice, when kind == TCH_NAMED
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
bool muse_user_payload_strip(struct sbuf *payload);

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
    int (*call_end)(void *ud); // a native turn or a parallel JSON document
                               // may carry another call after this one
} tool_stream_sink;

typedef struct {
    const tool_envelope *env;
    tool_stream_sink     sink;
    int   state;
    char  *head;          // undecided prefix, held back from the client
    size_t head_n, head_cap;
    char *name;           // selected branch, once known (owned)
    bool  called;         // the CURRENT entry selected a tool branch, not
                          // `final` -- reset between entries of a parallel
                          // document, unlike any_called below
    bool  any_called;     // a tool branch was selected by ANY entry so far;
                          // this is what finish_reason "tool_calls" reads
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
// Harmony's analysis channel as a splitter open/close pair — in DECODED form.
// The splitter sees detokenized text, and Harmony's control tokens
// (<|channel|>, <|message|>, <|end|>, <|start|>) all decode to nothing, so the
// bytes that actually arrive around the channel handoff are the bare words:
//
//   <|channel|>analysis<|message|>            -> "analysis"
//   <|end|><|start|>assistant<|channel|>final<|message|> -> "assistantfinal"
//
// Measured live on gpt-oss-20b at temp 0, whose raw stream reads
// "analysisWe have a conversation...assistantfinal2 + 2 equals **4**."
// This is the same trap muse hit: its markers are the decoded " to=self" /
// "assistant to=user", not the control-token spellings (see
// test_muse_plain_thinking_close_leaves_no_recipient_residue). Writing the
// control-token forms here compiles, renders, and silently never matches.
//
// The close must span the WHOLE handoff. Narrowing it would leak "assistant"
// or "final" into content, which is the residue bug muse recorded.
#define HARMONY_THINK_OPEN  "analysis"
#define HARMONY_THINK_CLOSE "assistantfinal"

void think_init(think_split *t, const char *open, const char *close);
void think_init_reasoning(think_split *t, const char *open, const char *close);
int  think_feed(think_split *t, const char *bytes, int n, think_cb cb, void *ud);
int  think_finish(think_split *t, think_cb cb, void *ud); // flush held bytes
void think_free(think_split *t);

#endif // RUNNER_TEMPLATE_H
