// The Responses-API framing state machine, tested directly (Phase 3 item
// "C-level Responses state-machine test").
//
// tests/conformance/test_responses.py already validates this surface against
// a live server, but only as far as a real generation happens to exercise it.
// The two properties the framer actually promises are structural, and they are
// cheapest to pin here, with no model and no HTTP:
//
//   1. ORDER. An item is announced (`response.output_item.added`) before any
//      of its deltas and closed (`.done`) after them, with a content part
//      opened and closed inside it. Nothing may appear between a part's
//      `.done` and its item's `.done`, and an item can never close twice.
//   2. NAMING. Every event names itself twice — in the SSE `event:` field and
//      in `data.type` — because typed SDK clients dispatch on the first while
//      validating the second. They are written from one argument precisely so
//      they cannot drift; this test fails if that is ever refactored apart.
//
// Plus the invariant the framer alone can guarantee: `sequence_number` is
// monotonic from 0 with no gaps, because nothing else assigns it.
//
// The state machine is `static` inside src/completion.c (it was server.c
// before RNR-019 split generation and wire framing out), so this file
// includes that translation unit and links the rest of the engine around it.
// Events are captured through a socketpair — the real send path, not a stub,
// so a framing bug that only shows up in the bytes still fails here.
#include "runner.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#include "../src/completion.c"

static int g_fail = 0;

static void ck(int cond, const char *what) {
    if (!cond) { fprintf(stderr, "FAIL: %s\n", what); g_fail = 1; }
    else        fprintf(stderr, "ok: %s\n", what);
}

// one captured SSE event: the `event:` name and the `data.type` value
typedef struct { char ev[64]; char type[64]; long seq; } event_t;

enum { MAX_EV = 64 };
static event_t g_ev[MAX_EV];
static int g_n_ev;

// Parse the raw SSE stream the framer wrote into the socket. Deliberately
// literal: if the wire shape changes, this notices.
static void parse_events(const char *buf) {
    g_n_ev = 0;
    for (const char *p = buf; (p = strstr(p, "event: ")) && g_n_ev < MAX_EV; ) {
        p += 7;
        const char *nl = strchr(p, '\n');
        if (!nl) break;
        event_t *e = &g_ev[g_n_ev];
        size_t n = (size_t)(nl - p);
        if (n >= sizeof(e->ev)) n = sizeof(e->ev) - 1;
        memcpy(e->ev, p, n);
        e->ev[n] = 0;

        const char *t = strstr(nl, "\"type\":\"");
        e->type[0] = 0;
        if (t) {
            t += 8;
            const char *q = strchr(t, '"');
            if (q) {
                size_t m = (size_t)(q - t);
                if (m >= sizeof(e->type)) m = sizeof(e->type) - 1;
                memcpy(e->type, t, m);
                e->type[m] = 0;
            }
        }
        const char *s = strstr(nl, "\"sequence_number\":");
        e->seq = s ? strtol(s + 18, NULL, 10) : -1;
        g_n_ev++;
        p = nl;
    }
}

static int index_of(const char *type, int from) {
    for (int i = from; i < g_n_ev; i++)
        if (!strcmp(g_ev[i].type, type)) return i;
    return -1;
}

// Drive the framer over a socketpair and capture what it wrote.
static void run_sequence(const char *kind, bool close_it) {
    int sv[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0) {
        ck(0, "socketpair");
        return;
    }
    gen_ctx g;
    memset(&g, 0, sizeof(g));
    g.fd = sv[0];
    g.close_status = "completed";

    if (resp_open_item(&g, kind)) ck(0, "resp_open_item wrote");
    resp_delta(&g, kind, "he", 2);
    resp_delta(&g, kind, "llo", 3);
    if (close_it) resp_close_item(&g);

    shutdown(sv[0], SHUT_WR);
    static char buf[65536];
    size_t got = 0;
    for (;;) {
        ssize_t r = recv(sv[1], buf + got, sizeof(buf) - 1 - got, 0);
        if (r <= 0) break;
        got += (size_t)r;
        if (got >= sizeof(buf) - 1) break;
    }
    buf[got] = 0;
    close(sv[0]);
    close(sv[1]);
    parse_events(buf);
    free(g.item_text.s);
    free(g.out_items.s);
    free(g.out_text.s);
    free(g.call_name);
}

static void test_message_item_order(void) {
    run_sequence("message", true);
    ck(g_n_ev >= 4, "a message item emits at least added/part/delta/done");

    int added = index_of("response.output_item.added", 0);
    int part  = index_of("response.content_part.added", 0);
    int delta = index_of("response.output_text.delta", 0);
    int tdone = index_of("response.output_text.done", 0);
    int pdone = index_of("response.content_part.done", 0);
    int idone = index_of("response.output_item.done", 0);

    ck(added == 0, "the item is announced first");
    ck(part  > added, "the content part opens inside the item");
    ck(delta > part,  "deltas come after the part opens");
    ck(tdone > delta, "the text is closed after its deltas");
    ck(pdone > tdone, "the part closes after the text");
    ck(idone > pdone, "the item closes last");
    ck(index_of("response.output_item.done", idone + 1) < 0,
       "an item never closes twice");
}

static void test_event_names_agree(void) {
    run_sequence("message", true);
    bool agree = g_n_ev > 0;
    for (int i = 0; i < g_n_ev; i++)
        if (strcmp(g_ev[i].ev, g_ev[i].type) != 0) agree = false;
    ck(agree, "every event's SSE name matches its data.type");
}

static void test_sequence_numbers_are_dense(void) {
    run_sequence("message", true);
    bool dense = g_n_ev > 0;
    for (int i = 0; i < g_n_ev; i++)
        if (g_ev[i].seq != i) dense = false;
    ck(dense, "sequence_number runs 0..n-1 with no gaps or repeats");
}

static void test_function_call_item_has_no_content_part(void) {
    // A function_call item streams argument deltas directly: RESP_CALL
    // declares no part_added/part_done, and emitting them would break typed
    // clients that only expect parts on message items.
    run_sequence("function_call", true);
    ck(index_of("response.output_item.added", 0) == 0,
       "the function_call item is announced first");
    ck(index_of("response.function_call_arguments.delta", 0) > 0,
       "argument deltas are emitted");
    ck(index_of("response.content_part.added", 0) < 0,
       "a function_call item opens no content part");
    ck(index_of("response.output_item.done", 0) > 0,
       "the function_call item is closed");
}

static void test_unclosed_item_emits_no_done(void) {
    // Closing is the caller's decision; the framer must not invent it.
    run_sequence("message", false);
    ck(index_of("response.output_item.added", 0) == 0, "item opened");
    ck(index_of("response.output_item.done", 0) < 0,
       "an item left open emits no .done");
}

// A character can span two generated tokens, and each streamed delta is JSON
// escaped on its own — so without holding the unfinished tail back, each half
// is an ill-formed sequence, becomes U+FFFD, and the streamed turn says
// something different from the buffered one. Observed on a real model:
// buffered rendered a CJK character where the stream rendered three U+FFFD.
// This pins the boundary arithmetic that holds it together.
static void test_utf8_tail_is_held_until_the_character_completes(void) {
    struct { const char *buf; int n; int want; const char *what; } cases[] = {
        { "abc",                 3, 0, "ascii is never held" },
        { "caf\xC3\xA9",         5, 0, "complete two-byte" },
        { "caf\xC3",             4, 1, "two-byte missing its continuation" },
        { "\xE6\x84\x9A",        3, 0, "complete three-byte" },
        { "\xE6\x84",            2, 2, "three-byte missing one" },
        { "\xE6",                1, 1, "three-byte missing two" },
        { "\xF0\x9F\x98\x80",    4, 0, "complete four-byte (emoji)" },
        { "\xF0\x9F\x98",        3, 3, "four-byte missing one" },
        { "\x80\x80",            2, 0, "stray continuations can never complete" },
        { "",                    0, 0, "empty" },
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(*cases); i++)
        ck(u8_incomplete_tail(cases[i].buf, cases[i].n) == cases[i].want,
           cases[i].what);
}

// ------------------------------------------- native-envelope failure paths
//
// A Harmony (gpt-oss) turn produces NOTHING through tool_stream_feed: the whole
// protocol document is buffered and demultiplexed at the end, in
// tool_stream_finish. So the mapping either succeeds and the client gets every
// delta at once, or it fails and the client gets nothing at all — and "nothing
// at all" used to mean an SSE stream with no finish_reason, no `data: [DONE]`
// and no terminator of any kind, because a mapping failure was recorded as
// `g.dead` (the flag that means the CLIENT went away) and every terminal block
// is guarded by `if (!g.dead)`.
//
// These drive the real entry point — run_completion over a socketpair, with the
// byte-vocabulary test model — because reachability is the bug: an emitter
// tested in isolation would pass while the code that calls it is skipped.
static const char *g_model_path = "test.gguf";

typedef struct { model_t m; tokenizer tok; slot_t s; } harness;

static bool harness_open(harness *h) {
    memset(h, 0, sizeof(*h));
    model_params p;
    memset(&p, 0, sizeof(p));
    p.gpu_mode  = GPU_AUTO;
    p.n_threads = 1;
    p.n_ctx     = 128;
    p.n_batch   = 8;
    if (!model_load(&h->m, g_model_path, &p)) return false;
    if (!tokenizer_init(&h->tok, &h->m.gf)) { model_free(&h->m); return false; }
    h->s.m   = &h->m;
    h->s.tok = &h->tok;
    h->s.tmpl = TMPL_HARMONY;
    h->s.smp_base.temp = 0;              // greedy: the turn must be repeatable
    h->s.smp_base.top_p = 1;
    h->s.smp_base.repeat_penalty = 1.0f;
    h->s.smp_base.rng = 1;
    h->s.smp = h->s.smp_base;
    engine_init(&h->s.e, &h->m, &h->tok, &h->s.smp);
    SV.model_name    = "test-harmony";
    SV.n_predict_cap = 64;
    return true;
}

static void harness_close(harness *h) {
    tokenizer_free(&h->tok);
    model_free(&h->m);
}

static const char *HARNESS_TOOLS =
    "[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\","
    "\"parameters\":{\"type\":\"object\",\"properties\":"
    "{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}}}]";

// One request through run_completion, captured off the wire.
//
// run_completion writes synchronously and nothing drains the far end until it
// returns, so a case whose response exceeds the socketpair's buffer would block
// forever. Every case here is a handful of small events; keep it that way, or
// drain from a second thread.
static void run_request(harness *h, const char *req_json,
                        const tool_envelope *env, int api,
                        char *out, size_t cap) {
    out[0] = 0;
    int sv[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0) { ck(0, "socketpair"); return; }
    jv *req = json_parse(req_json, strlen(req_json));
    if (!req) { ck(0, "request parsed"); close(sv[0]); close(sv[1]); return; }
    run_completion(&h->s, sv[0], "hello", api, req, env);
    jv_free(req);
    shutdown(sv[0], SHUT_WR);
    size_t got = 0;
    for (;;) {
        ssize_t r = recv(sv[1], out + got, cap - 1 - got, 0);
        if (r <= 0) break;
        got += (size_t)r;
        if (got >= cap - 1) break;
    }
    out[got] = 0;
    close(sv[0]);
    close(sv[1]);
}

static char g_wire[262144];

// DEFECT A. However a stream ends, it must END. A Harmony document that cannot
// be mapped is a generation fault, not a dead client: the client is still
// there, still reading, and owed a terminal chunk it can stop on.
static void test_unmappable_harmony_stream_still_terminates(void) {
    harness h;
    if (!harness_open(&h)) { ck(0, "test model loaded"); return; }
    jv *tools = json_parse(HARNESS_TOOLS, strlen(HARNESS_TOOLS));
    tool_envelope env = {0};
    char err[224];
    ck(tool_envelope_build(tools, NULL, NULL, &env, err, sizeof(err)) == 1,
       "harmony envelope built");
    env.harmony = true;
    env.tools   = tools;

    // max_tokens 0 is a legal request that generates no document at all, so
    // harmony_map has nothing to map and returns -1 — the same terminal state
    // a stop-corrupted document reaches, with no model randomness in it.
    run_request(&h, "{\"stream\":true,\"max_tokens\":0}", &env, API_CHAT,
                g_wire, sizeof(g_wire));
    ck(strstr(g_wire, "\"role\":\"assistant\"") != NULL,
       "the stream opened with the role delta");
    ck(strstr(g_wire, "\"finish_reason\":\"") != NULL,
       "an unmappable harmony stream still reports a finish_reason");
    ck(strstr(g_wire, "data: [DONE]") != NULL,
       "an unmappable harmony stream still sends data: [DONE]");
    ck(strstr(g_wire, "<|channel|>") == NULL,
       "no harmony protocol text reaches the client as content");
    // and it says which fault it was: "stop" over an empty message would tell
    // the caller the model chose to say nothing
    ck(strstr(g_wire, "\"finish_reason\":\"error\"") != NULL,
       "the terminal chunk reports a generation fault, not a clean stop");
    ck(strstr(g_wire, "\"finish_detail\":\"envelope_unmapped\"") != NULL,
       "runner_telemetry keeps which fault it was");

    // The same promise on the other two streaming surfaces: a Responses turn
    // must reach a terminal response event, and an Anthropic turn must reach
    // message_stop. Neither has a `data: [DONE]` to fall back on.
    run_request(&h, "{\"stream\":true,\"max_tokens\":0}", &env, API_RESPONSES,
                g_wire, sizeof(g_wire));
    ck(strstr(g_wire, "event: response.incomplete") != NULL,
       "an unmappable harmony Responses stream reaches a terminal event");
    ck(strstr(g_wire, "\"reason\":\"envelope_unmapped\"") != NULL,
       "and names the reason rather than claiming a token-budget truncation");
    ck(strstr(g_wire, "<|channel|>") == NULL,
       "no harmony protocol text reaches a Responses client");

    // The Anthropic stream terminates too, but not with message_delta /
    // message_stop: those assert a completed Message, and this turn produced
    // none. Its terminator is the documented `error` event — see
    // test_anthropic_streamed_fault_is_an_error_event below.

    tool_envelope_free(&env);
    jv_free(tools);
    harness_close(&h);
}

// DEFECT C, streaming. The headers went out with the 200 before a token
// existed, so there is no status code left to set — and Anthropic's SSE
// protocol has a terminator for exactly this, documented alongside
// overloaded_error:
//
//   event: error
//   data: {"type": "error", "error": {"type": "overloaded_error", ...}}
//
// So the stream ends on `error`, and specifically NOT on message_delta +
// message_stop: message_delta would have to carry one of the seven
// stop_reasons (end_turn, before this), and message_stop asserts a Message
// that completed. Terminating is not optional — leaving the stream hanging is
// the failure 6c4dfc2 fixed on the OpenAI side and must not come back here.
static void test_anthropic_streamed_fault_is_an_error_event(void) {
    harness h;
    if (!harness_open(&h)) { ck(0, "test model loaded"); return; }
    jv *tools = json_parse(HARNESS_TOOLS, strlen(HARNESS_TOOLS));
    tool_envelope env = {0};
    char err[224];
    ck(tool_envelope_build(tools, NULL, NULL, &env, err, sizeof(err)) == 1,
       "harmony envelope built");
    env.harmony = true;
    env.tools   = tools;

    run_request(&h, "{\"stream\":true,\"max_tokens\":0}", &env, API_MESSAGES,
                g_wire, sizeof(g_wire));
    ck(strstr(g_wire, "event: message_start") != NULL,
       "the Anthropic stream opened with message_start");
    ck(strstr(g_wire, "event: error") != NULL,
       "an unmappable harmony Anthropic stream terminates on an error event");
    ck(strstr(g_wire, "\"type\":\"error\"") != NULL,
       "the event names itself in data.type as well as in event:");
    ck(strstr(g_wire, "\"error\":{\"type\":\"api_error\"") != NULL,
       "carrying Anthropic's error object, typed api_error");
    ck(strstr(g_wire, "\"finish_detail\":\"envelope_unmapped\"") != NULL,
       "and runner_telemetry keeps which fault it was");
    ck(strstr(g_wire, "event: message_delta") == NULL,
       "no message_delta: there is no stop_reason that means \"faulted\"");
    ck(strstr(g_wire, "\"stop_reason\":\"end_turn\"") == NULL,
       "and nothing claims the model finished its turn normally");
    ck(strstr(g_wire, "event: message_stop") == NULL,
       "no message_stop: it would assert a Message that completed");
    ck(strstr(g_wire, "event: content_block_start") == NULL,
       "and no empty text block is invented to stand in for the answer");
    ck(strstr(g_wire, "<|channel|>") == NULL,
       "no harmony protocol text reaches an Anthropic client");

    // Narrow again: an ordinary stream still ends the ordinary way.
    run_request(&h, "{\"stream\":true,\"max_tokens\":4}", NULL, API_MESSAGES,
                g_wire, sizeof(g_wire));
    ck(strstr(g_wire, "event: message_delta") != NULL,
       "an ordinary Anthropic stream still reports a stop_reason");
    ck(strstr(g_wire, "event: message_stop") != NULL,
       "and still reaches message_stop");
    ck(strstr(g_wire, "event: error") == NULL,
       "with no error event");

    tool_envelope_free(&env);
    jv_free(tools);
    harness_close(&h);
}

// DEFECT C, buffered. Anthropic defines exactly seven `stop_reason` values —
// end_turn, max_tokens, stop_sequence, tool_use, pause_turn, refusal and
// model_context_window_exceeded — and NONE of them means "generation faulted".
// Its own docs draw the line: "Unlike errors, which indicate failures in
// processing your request, `stop_reason` tells you why Claude completed its
// response generation." A fault is not a completion, so it is an error object
// with a 4xx/5xx, not a 200 Message carrying the least-wrong enum member.
//
// Before this, a fault landed on `end_turn` — the surface telling the caller
// the model finished normally over an empty content[], which is the one
// reading a client cannot recover from.
static void test_anthropic_buffered_fault_is_an_error_object(void) {
    harness h;
    if (!harness_open(&h)) { ck(0, "test model loaded"); return; }
    jv *tools = json_parse(HARNESS_TOOLS, strlen(HARNESS_TOOLS));
    tool_envelope env = {0};
    char err[224];
    ck(tool_envelope_build(tools, NULL, NULL, &env, err, sizeof(err)) == 1,
       "harmony envelope built");
    env.harmony = true;
    env.tools   = tools;

    // same deterministic fault the streaming test uses: no document at all is
    // generated, so there is nothing for the demultiplexer to map
    run_request(&h, "{\"max_tokens\":0}", &env, API_MESSAGES,
                g_wire, sizeof(g_wire));
    ck(!strncmp(g_wire, "HTTP/1.1 500", 12),
       "a buffered Anthropic fault answers 500, not 200");
    ck(strstr(g_wire, "\"type\":\"error\"") != NULL,
       "the body is Anthropic's top-level error object");
    ck(strstr(g_wire, "\"type\":\"api_error\"") != NULL,
       "typed as api_error: the fault is the server's, and it is retryable");
    ck(strstr(g_wire, "\"message\":\"") != NULL,
       "the error object carries a message");
    ck(strstr(g_wire, "\"stop_reason\"") == NULL,
       "no stop_reason is reported: the turn did not complete");
    ck(strstr(g_wire, "\"type\":\"message\"") == NULL,
       "and no Message object is served for a turn that produced none");
    ck(strstr(g_wire, "<|channel|>") == NULL,
       "no harmony protocol text reaches the client");

    // The refusal is narrow. A turn that ends any ordinary way is still a 200
    // Message with a stop_reason from the documented seven.
    run_request(&h, "{\"max_tokens\":4}", NULL, API_MESSAGES,
                g_wire, sizeof(g_wire));
    ck(!strncmp(g_wire, "HTTP/1.1 200", 12),
       "an ordinary Anthropic turn is still a 200");
    ck(strstr(g_wire, "\"type\":\"message\"") != NULL,
       "and still a Message object");
    ck(strstr(g_wire, "\"stop_reason\":\"") != NULL,
       "carrying a stop_reason");

    tool_envelope_free(&env);
    jv_free(tools);
    harness_close(&h);
}

// DEFECT A, the trigger. A `stop` sequence is a rule about the model's VISIBLE
// text. Under a strict tool envelope the generated document is protocol —
// Harmony channel markers, a recipient header, or the JSON envelope's own
// syntax — and the caller never sees a byte of it, so matching their strings
// against it fires on framing they did not write.
//
// It also corrupts the turn, which is what made this a hang rather than a
// surprise: stop_feed drops the matched bytes from the emitted document while
// the constraint validator has ALREADY consumed them, so the closer
// constraint_close computes no longer continues the text the client received.
// Measured on this fixture — generic envelope, buffered, stop `"` — the client
// got `"content":"{tool"`, raw envelope syntax served as assistant text. On the
// Harmony envelope the mapping fails outright and the stream never terminates.
//
// Honouring it properly is not available on this path: the Harmony
// demultiplexer does not know which bytes are visible text until the document
// is finished, by which point there is nothing left to stop. So the only
// choices are to refuse or to ignore, and this server does not ignore what a
// caller asked for.
static void test_stop_with_a_strict_envelope_is_refused(void) {
    harness h;
    if (!harness_open(&h)) { ck(0, "test model loaded"); return; }
    jv *tools = json_parse(HARNESS_TOOLS, strlen(HARNESS_TOOLS));
    char err[224];

    tool_envelope harmony = {0};
    ck(tool_envelope_build(tools, NULL, NULL, &harmony, err, sizeof(err)) == 1,
       "harmony envelope built");
    harmony.harmony = true;
    harmony.tools   = tools;
    run_request(&h, "{\"stream\":true,\"stop\":[\"\\n\\n\"]}", &harmony,
                API_CHAT, g_wire, sizeof(g_wire));
    ck(!strncmp(g_wire, "HTTP/1.1 400", 12),
       "stop alongside native harmony tool calling is refused, not accepted");
    ck(strstr(g_wire, "stop") != NULL, "the refusal names the field");
    ck(strstr(g_wire, "data: [DONE]") == NULL,
       "the refusal is an HTTP error, not a half-opened SSE stream");
    tool_envelope_free(&harmony);

    // the same refusal for the generic JSON envelope, reached through the
    // Anthropic spelling of the field: one rule, however it was asked for
    h.s.tmpl = TMPL_CHATML;
    tool_envelope generic = {0};
    ck(tool_envelope_build(tools, NULL, NULL, &generic, err, sizeof(err)) == 1,
       "generic envelope built");
    run_request(&h, "{\"stop_sequences\":[\"\\\"\"]}", &generic, API_MESSAGES,
                g_wire, sizeof(g_wire));
    ck(!strncmp(g_wire, "HTTP/1.1 400", 12),
       "stop_sequences alongside the generic tool envelope is refused too");
    ck(strstr(g_wire, "{tool") == NULL,
       "and no envelope syntax is served as assistant content");
    tool_envelope_free(&generic);

    // The refusal is narrow: without a strict envelope there is no protocol
    // document to confuse, and `stop` keeps working exactly as documented.
    run_request(&h, "{\"max_tokens\":4,\"stop\":[\"z\"]}", NULL, API_CHAT,
                g_wire, sizeof(g_wire));
    ck(!strncmp(g_wire, "HTTP/1.1 200", 12),
       "stop without tools is still honoured");

    jv_free(tools);
    harness_close(&h);
}

// DEFECT B. The buffered path degrades by handing back whatever was generated
// when the envelope will not map. For a strict envelope that is not a degraded
// answer, it is the protocol itself: Harmony's control-token spellings reach
// g.out verbatim (engine.c forwards tok_raw for decoded-empty control tokens),
// and the generic envelope's raw JSON syntax reaches it the same way. Both were
// served to clients as assistant `content` — the Harmony case verbatim down to
// `to=functions.unknown`, the generic case measured on this fixture as `{tool`.
//
// A document nobody can read is not content. The client gets an empty answer
// and a finish reason that says a fault happened, which it can act on.
static void test_unmappable_envelope_is_not_served_as_content(void) {
    jv *tools = json_parse(HARNESS_TOOLS, strlen(HARNESS_TOOLS));
    char err[224];

    // the exact document a stop sequence used to produce: the framing is
    // intact enough to look like Harmony and broken enough not to map
    static const char HARMONY_BROKEN[] =
        "<|channel|>analysis<|message|>Let me check the weather.<|end|>"
        "<|start|>assistant<|channel|>commentary to=functions.unknown"
        "<|constrain|>json<|message|>{\"city\":\"Oslo\"}";
    tool_envelope harmony = {0};
    ck(tool_envelope_build(tools, NULL, NULL, &harmony, err, sizeof(err)) == 1,
       "harmony envelope built");
    harmony.harmony = true;
    harmony.tools   = tools;

    gen_ctx g;
    memset(&g, 0, sizeof(g));
    sb_put(&g.out, HARMONY_BROKEN, sizeof(HARMONY_BROKEN) - 1);
    sbuf tc = {0};
    int rc = envelope_map_buffered(&harmony, &g, &tc);
    ck(rc == -1, "the broken harmony document does not map");
    ck(g.out.n == 0,
       "an unmappable harmony document is not handed back as content");
    ck(!g.out.n || !strstr(g.out.s, "<|channel|>"),
       "no harmony control-token spelling survives into content");
    free(g.out.s); free(g.reason.s); free(tc.s);
    tool_envelope_free(&harmony);

    // and the same for the generic JSON envelope, whose raw document is
    // envelope syntax rather than anything the caller asked to read
    tool_envelope generic = {0};
    ck(tool_envelope_build(tools, NULL, NULL, &generic, err, sizeof(err)) == 1,
       "generic envelope built");
    memset(&g, 0, sizeof(g));
    sb_put(&g.out, "{tool", 5);
    tc = (sbuf){0};
    rc = envelope_map_buffered(&generic, &g, &tc);
    ck(rc == -1, "the broken generic document does not map");
    ck(g.out.n == 0,
       "an unmappable generic envelope is not handed back as content");
    free(g.out.s); free(g.reason.s); free(tc.s);
    tool_envelope_free(&generic);

    jv_free(tools);
}

// A turn can call SEVERAL tools. The chat dialect splices them into one
// `tool_calls` array and tool_calls_parse emits one entry per call it finds in
// the model's output, so a two-call turn is ordinary traffic rather than an
// exotic mode — and the Responses and Anthropic bodies render that SAME list in
// their own vocabulary.
//
// Deriving it from a parse of the comma-separated entries alone lost the whole
// turn: json_parse refuses trailing garbage, so anything past the first entry
// made it return NULL, and both surfaces then served a turn with a
// tool_calls/tool_use terminal reason and not one call in it — the client is
// told a tool was called and given nothing to execute.
static void test_every_tool_call_reaches_the_typed_surfaces(void) {
    static const char TWO[] =
        "{\"id\":\"call_0\",\"type\":\"function\",\"function\":"
        "{\"name\":\"get_weather\",\"arguments\":\"{\\\"city\\\":\\\"Oslo\\\"}\"}},"
        "{\"id\":\"call_1\",\"type\":\"function\",\"function\":"
        "{\"name\":\"get_time\",\"arguments\":\"{\\\"tz\\\":\\\"UTC\\\"}\"}}";
    sbuf tc = {0};
    sb_put(&tc, TWO, sizeof(TWO) - 1);
    jv *calls = tool_calls_array(&tc, 2);
    ck(calls && calls->type == J_ARR && calls->n == 2,
       "both calls of a two-call turn are extracted");

    SV.model_name = "test-multi-call";
    gen_ctx g;
    memset(&g, 0, sizeof(g));
    snprintf(g.id, sizeof(g.id), "resp_multi");
    resp_doc d = { .status = "completed", .with_output = true, .calls = calls };

    sbuf r = {0};
    responses_body(&r, &g, &d);
    ck(r.s && strstr(r.s, "\"name\":\"get_weather\"") &&
       strstr(r.s, "\"name\":\"get_time\""),
       "a Responses body carries every function_call item");
    ck(r.s && strstr(r.s, "\"call_id\":\"call_1\""),
       "and the second call keeps its own call_id");
    ck(r.s && !strstr(r.s, "\"type\":\"message\""),
       "with no assistant message invented alongside them");
    free(r.s);

    sbuf a = {0};
    anth_body(&a, &g, &d);
    ck(a.s && strstr(a.s, "\"name\":\"get_weather\"") &&
       strstr(a.s, "\"name\":\"get_time\""),
       "an Anthropic body carries every tool_use block");
    ck(a.s && strstr(a.s, "\"id\":\"toolu_1\""),
       "and the second block keeps its own id");
    ck(a.s && strstr(a.s, "\"input\":{\"city\":\"Oslo\"}"),
       "arguments are re-dumped as the object this surface carries");
    free(a.s);

    // a one-call turn is unchanged: the ids it always emitted are index 0
    jv_free(calls);
    tc.n = 0;
    sb_put(&tc, TWO, (size_t)(strstr(TWO, "},{") - TWO) + 1);
    calls = tool_calls_array(&tc, 1);
    ck(calls && calls->n == 1, "a one-call turn still extracts one call");
    d.calls = calls;
    r = (sbuf){0};
    responses_body(&r, &g, &d);
    ck(r.s && strstr(r.s, "\"id\":\"fc_0\"") &&
       strstr(r.s, "\"call_id\":\"call_0\""),
       "a single function_call keeps the ids it always had");
    free(r.s);
    jv_free(calls);
    free(tc.s);
}

int main(int argc, char **argv) {
    if (argc > 1) g_model_path = argv[1];
    test_every_tool_call_reaches_the_typed_surfaces();
    test_utf8_tail_is_held_until_the_character_completes();
    test_message_item_order();
    test_event_names_agree();
    test_sequence_numbers_are_dense();
    test_function_call_item_has_no_content_part();
    test_unclosed_item_emits_no_done();
    test_unmappable_harmony_stream_still_terminates();
    test_anthropic_buffered_fault_is_an_error_object();
    test_anthropic_streamed_fault_is_an_error_event();
    test_stop_with_a_strict_envelope_is_refused();
    test_unmappable_envelope_is_not_served_as_content();
    if (!g_fail) fprintf(stderr, "all responses state-machine tests passed\n");
    return g_fail;
}
