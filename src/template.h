// Chat templates, tool-call envelopes, thinking-tag splitting.
#ifndef RUNNER_TEMPLATE_H
#define RUNNER_TEMPLATE_H

#include <stddef.h>
#include <stdbool.h>
#include "tokenizer.h"

// MISTRAL is [INST] framing without llama-2's <<SYS>> block: Mistral's own
// template rejects a system role outright, so the text is folded into a user
// turn rather than wrapped in markers it never saw in training. It is NOT
// llama-2's renderer with that block swapped out, which is what it was until
// 2026-08-14 -- see TMPL_MISTRAL_V1 below for what that cost.
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
       TMPL_CHATML_THINK,
       // The other two Mistral instruction framings in circulation. TMPL_MISTRAL
       // above is the v0.3 form -- shared byte-for-byte with
       // Mistral-Small-Instruct-2409 and the derivative population that copied
       // it -- and is what `--chat-template mistral` and any Mistral-shaped
       // template this build cannot place select.
       //
       //   TMPL_MISTRAL       "[INST] "  ... "[/INST]"    v0.3 / Small-2409
       //   TMPL_MISTRAL_V1    " [INST] " ... " [/INST]"   v0.1 / v0.2
       //   TMPL_MISTRAL_NEMO  "[INST]"   ... "[/INST]"    Nemo-Instruct-2407
       //
       // Whitespace, and not cosmetic whitespace: this is SentencePiece, where
       // `[INST] What is 2+2? [/INST]` is 11 tokens and
       // `[INST] What is 2+2?[/INST]` is 10 -- measured on
       // Mistral-7B-Instruct-v0.3 on hardware. Every divergent space is a token
       // in a position the model never trained on.
       //
       // They also disagree about something bigger than spacing: v0.1/v0.2 fold
       // the system text into the FIRST user turn, v0.3 and Nemo into the LAST.
       //
       // Until 2026-08-14 runner rendered all Mistral checkpoints through
       // llama-2's case with only the <<SYS>> block swapped out, producing a
       // FOURTH framing (`[INST] ` ... ` [/INST]`, system into the first turn)
       // that matches no Mistral checkpoint at all. It was never trying to be
       // Mistral.
       TMPL_MISTRAL_V1, TMPL_MISTRAL_NEMO,
       // gemma-4 ships THREE chat-template revisions and they are NOT one
       // renderer. The mainline template (12B, 26B-A4B, 31B; sha ae53464b)
       // pre-seeds an empty CLOSED thought block '<|channel>thought\n
       // <channel|>' on the thinking-OFF generation prompt; the E-series
       // (E2B sha 0a2c8073, E4B sha 241c50d8) does NOT. Pre-seeding it is a
       // measured NO-OP on mainline (byte-identical output) but CATASTROPHIC
       // on the E-series (E2B planning 0.650 -> 0.250, reasoning-leak), so the
       // two must render that one site differently. TMPL_GEMMA4 stays the
       // E-series id and its every behaviour; TMPL_GEMMA4_MAINLINE differs
       // from it ONLY at that generation-prompt site. Every other tmpl branch
       // (tool protocol, sampler, parse) treats them the same via is_gemma4().
       TMPL_GEMMA4_MAINLINE,
       // What template_detect returns when NOTHING matched. It renders
       // llama-2 markup, because changing what unrecognised models render is
       // a behavioural decision and not this constant's job -- but it is a
       // DISTINCT id, so "we recognised Llama-2" and "we recognised nothing"
       // stop being the same answer. They were the same answer until
       // 2026-08-15, and that conflation is why per-turn BOS could not be
       // added: the literal <s> would have reached every unrecognised model,
       // and in a vocabulary without <s> as a special it tokenizes as three
       // characters, which is worse than the token merely being absent.
       // Not reachable through --chat-template: you cannot ask for it.
       TMPL_LLAMA2_FALLBACK };

// True for EITHER gemma-4 template family. The mainline and the E-series
// differ at exactly one generation-prompt site (see TMPL_GEMMA4_MAINLINE
// above); every other tmpl-dependent decision -- the native tool protocol,
// tool-call replay, tool parsing, the primed-think probe -- is identical, so
// those sites ask this rather than naming one id and silently breaking the
// other. Added when the mainline id split off TMPL_GEMMA4 (2026-08-19): a
// missed site would leave mainline tool calls speaking the generic protocol.
static inline bool is_gemma4(int tmpl) {
    return tmpl == TMPL_GEMMA4 || tmpl == TMPL_GEMMA4_MAINLINE;
}

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
    // Harmony history may carry separate analysis/commentary assistant
    // messages before a final answer or recipient-bearing call.
    const char *channel;
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
//   TMPL_GEMMA4        gemma-4 E-series tokenizer.chat_template (from the GGUF)
//     The generation prompt is '<|turn>model\n' and nothing else, in EITHER
//     mode -- there is no pre-seeded thought block. After a tool response it
//     emits nothing at all, or an OPEN '<|channel>thought\n' when thinking.
//     TMPL_GEMMA4_MAINLINE is byte-identical EXCEPT that its thinking-OFF
//     generation prompt appends the empty CLOSED block '<|channel>thought\n
//     <channel|>' after '<|turn>model\n'. That is the mainline template's own
//     `{%- if not enable_thinking -%}` branch (enable_thinking defaults
//     false), a MEASURED no-op there and catastrophic on the E-series, which
//     is why the two ids exist.
//     Thinking for this family is selected in the FIRST SYSTEM TURN, not at
//     the generation prompt: THINK_ON injects '<|think|>\n' at the top of it,
//     opening that turn even when the caller sent no system message, exactly
//     as the template's own condition does.
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
// Replay an assistant turn's tool_calls in the family's own syntax.
//
// `turn_has_text` is whether THAT TURN carried visible content, and it cannot
// be recovered from `out`: by the time this runs, `out` may already hold
// framing the caller added (ornith's <think> block), so an empty-buffer test
// answers a different question. ornith is the family that needs it -- its
// reference opens the first call with "\n\n" after text and with nothing
// without it (ornith.jinja:106-109).
void tool_history_render_for(int tmpl, const struct jv *calls,
                             bool turn_has_text, struct sbuf *out);
// Build a one-element OpenAI tool_calls array from a name and its arguments
// (a JSON string). Owned jv or NULL on OOM. The adapter the typed surfaces use
// to reach tool_history_render_for's serializer from their own call vocabulary.
struct jv *tool_call_synth(const char *name, const char *args_json);
// Assemble an assistant turn's flattened content in the family's native tool
// protocol -- the same bytes handle_chat produces for a live chat turn.
// `text` is the turn's visible text (may be NULL), `calls` an OpenAI-shaped
// tool_calls array; `out` must be empty (or hold only caller framing). Sets
// *turn_name to the name the assistant chat_msg must carry (muse) or NULL.
void assistant_calls_render(int tmpl, const char *text, const struct jv *calls,
                            struct sbuf *out, const char **turn_name);
// Wrap a replayed tool RESULT in the family's native protocol; returns the role
// ("user" for ornith's <tool_response> user turn, "tool" otherwise) and writes
// the wrapped content to `out`.
const char *tool_result_wrap(int tmpl, const char *result, struct sbuf *out);
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

// The native tool protocol a request uses, or TP_GENERIC for the strict JSON
// envelope. Exactly one holds per envelope: a single field makes the
// mutually-exclusive families unrepresentable as a boolean combination (the
// old five flags permitted states like atem+harmony that cannot exist).
enum tool_proto {
    TP_GENERIC = 0,   // strict JSON envelope, or no native protocol
    TP_ATEM,          // Muse native recipient + <atem:invoke>
    TP_MUSE_USER,     // Muse: generic JSON override after a to=user header
    TP_HARMONY,       // gpt-oss native channels + functions recipient
    TP_GEMMA4,        // gemma4 native <|tool_call>call:NAME{...}
    TP_MUSE_PLAIN,    // Muse: stream a schema payload after a to=user header
};

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
    int   proto;          // enum tool_proto — the native protocol, or TP_GENERIC
    struct jv *tools;     // borrowed request declarations for native compiler
    bool  owns_tools;     // Anthropic translation survives prompt construction
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

// Select the model-native tool-DECLARATION protocol for a slot's template
// family on `env`, so /v1/chat/completions, /v1/responses and /v1/messages all
// teach declared tools in the format the family was trained on -- gemma4's
// `<|tool>declaration:...`, muse's atem schema block, Harmony's `functions`
// namespace -- rather than the generic block. Sets env's native flags
// (atem/muse_user_header/harmony/gemma4) and its borrowed `tools` pointer for
// the families that render their own declarations, and returns the declarations
// to hand render_prompt_alloc (NULL when the family uses the generic block).
// *skip_generic is set true when the caller must NOT emit its generic/strict
// declaration turn because the family renders its own.
//
// `strict` is whether the strict envelope applies (tool_envelope_build returned
// 1 and, for the chat surface, the slot is not ornith, whose native qwen3_xml
// declaration tools_render_for already handles). `atem_tool_calling` selects
// muse's native atem protocol over the generic to=user override; pass true for
// the native default. The returned pointer aliases `tools` -- the caller keeps
// ownership (env->owns_tools is not touched here).
const struct jv *tool_decl_native(int tmpl, bool strict, bool atem_tool_calling,
                                  struct jv *tools, tool_envelope *env,
                                  bool *skip_generic);

// Map a generated envelope document back to the OpenAI response shape.
// Returns the NUMBER of tool_calls[] items appended to tc (1 for the ordinary
// single-call envelope, 0..max_calls for the parallel form), 0 for the final
// branch with assistant content appended instead, and -1 when doc is not a
// well-formed envelope. A parallel document may mix the two: any final
// branches contribute content, any tool branches contribute calls.
int  tool_envelope_map(const tool_envelope *e, const char *doc, size_t n,
                       struct sbuf *content, struct sbuf *tc);
int  tool_envelope_map_channels(const tool_envelope *e, const char *doc,
                                size_t n, struct sbuf *reasoning,
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
    int (*reasoning)(void *ud, const char *bytes, int n);
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
int  tool_stream_finish(tool_stream *s);
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
