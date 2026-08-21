// Generation and wire framing. Lifted out of server.c (RNR-019); see completion.h.
#include "completion.h"
#include "compat.h"

#include "http.h"
#include "scheduler.h"

#include <errno.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// ---------------------------------------------------------------- generation

typedef struct {
    sbuf  out;          // accumulated completion text
    sbuf  reason;       // accumulated reasoning text (thinking-tag models)
    int   fd;
    bool  stream;       // SSE mode
    bool  dead;         // client went away
    char  id[48];
    int   api;          // API_* — the dialect this response is framed in
    think_split ts;     // thinking-tag splitter (pass-through when untagged)
    // OpenAI "stop" sequences: matched against the content channel only
    // (reasoning text must not trigger a client's stop strings)
    const char **stop_strs; // borrowed from the request jv, valid all request
    int   n_stop;
    sbuf  hold;         // held-back tail that may still begin a stop match
    bool  stopped;      // a stop sequence matched; excluded from output
    engine *eng;        // the generating engine, so a stop match can truncate
                        // its constrained document to what was delivered
    const char *stop_hit; // which one matched (Anthropic reports it back)
    long  created;      // stamped once: every chunk of a stream reports the
                        // same creation time, as the buffered body does
    bool  role_sent;    // "role":"assistant" is emitted on the first delta only
    // strict tool envelope, streaming: the generated document is demuxed into
    // content and tool_calls deltas as it arrives, so the envelope itself is
    // never what the client receives
    tool_stream tsx;
    bool  tsx_on;
    int   tool_index;   // OpenAI tool_calls[].index; one call per turn for now
    // Responses streaming: the typed events are ordered and each one carries a
    // monotonic sequence_number, so the emitter is a small state machine over
    // "which item / content part is currently open" rather than a formatter.
    long  seq;          // next sequence_number
    int   output_index; // index of the item being streamed
    bool  item_open;    // an output_item.added has no matching .done yet
    bool  part_open;    // likewise for content_part.added
    char  item_id[48];  // id of the open item (msg_/fc_)
    sbuf  item_text;    // its text so far, replayed in the .done events
    const char *item_kind; // "message" or "function_call"
    char *call_name;    // function_call name, once known (owned)
    // items are accumulated as they complete so the terminal event can report
    // the same `output[]` a buffered request would have returned; a client
    // that only reads response.completed must not see a different turn than
    // one that followed the deltas
    sbuf  out_items;
    sbuf  out_text;     // the `output_text` aggregate (assistant text only)
    // status to close the *last* item with. An item that ends because another
    // one starts did finish; one that ends because generation was cut short
    // did not, and must say so — a client rendering output[] without reading
    // the response status would otherwise show a truncated message as whole.
    const char *close_status;
    // A multi-byte character can span two generated tokens, and a delta is
    // escaped on its own — so each half would be an ill-formed sequence and
    // get replaced with U+FFFD, destroying a character the buffered path
    // renders correctly. Hold an incomplete trailing sequence back here and
    // prepend it to the next delta. Indexed by channel (0 = text,
    // 1 = reasoning) so the two streams cannot splice each other's bytes.
    // At most 3 bytes can ever be pending: a 4-byte sequence missing one.
    char  u8_pend[2][4];
    int   u8_pend_n[2];
} gen_ctx;

// common prefix of every streamed chunk. `created` and `model` are required by
// the ChatCompletionChunk schema and strictly-validating SDKs reject a chunk
// without them, so they are written here rather than per call site.
static void chunk_open(gen_ctx *g, sbuf *c) {
    sb_fmt(c, "{\"id\":\"%s\",\"object\":\"%s\",\"created\":%ld,\"model\":\"",
           g->id, g->api == API_CHAT ? "chat.completion.chunk" : "text_completion",
           g->created);
    sb_esc(c, SV.model_name, strlen(SV.model_name));
    sb_lit(c, "\",\"choices\":[{\"index\":0,");
}

// frame one built chunk body as an SSE event and push it; a failed send marks
// the client gone, which is what aborts generation upstream
static int chunk_send(gen_ctx *g, sbuf *c) {
    sbuf sse = {0};
    sb_fmt(&sse, "data: %s\n\n", c->s ? c->s : "");
    if (c->failed || sse.failed || !send_all(g->fd, sse.s, sse.n)) g->dead = true;
    free(c->s);
    free(sse.s);
    return g->dead ? 1 : 0;
}

static void append_chat_logprobs(sbuf *r, slot_t *s, engine *e) {
    char tb[512];
    sb_lit(r, "\"logprobs\":{\"content\":[");
    for (int i = 0; i < e->lp_count; i++) {
        if (i) sb_lit(r, ",");
        int tn = tok_decode(s->tok, e->lp_ids[i], tb, sizeof(tb));
        sb_lit(r, "{\"token\":\"");
        sb_esc(r, tb, tn);
        sb_fmt(r, "\",\"logprob\":%.6f,\"top_logprobs\":[", e->lp_chosen[i]);
        for (int j = 0; j < e->lp_n; j++) {
            const lp_alt *a = &e->lp_top[(size_t)i * e->lp_n + j];
            if (a->id < 0) break;
            if (j) sb_lit(r, ",");
            tn = tok_decode(s->tok, a->id, tb, sizeof(tb));
            sb_lit(r, "{\"token\":\"");
            sb_esc(r, tb, tn);
            sb_fmt(r, "\",\"logprob\":%.6f}", a->lp);
        }
        sb_lit(r, "]}");
    }
    sb_lit(r, "]}");
}

static void append_text_logprobs(sbuf *r, slot_t *s, engine *e) {
    char tb[512];
    int offset = 0;
    sb_lit(r, "\"logprobs\":{\"tokens\":[");
    for (int i = 0; i < e->lp_count; i++) {
        if (i) sb_lit(r, ",");
        int tn = tok_decode(s->tok, e->lp_ids[i], tb, sizeof(tb));
        sb_lit(r, "\""); sb_esc(r, tb, tn); sb_lit(r, "\"");
    }
    // Ids alongside the rendered pieces. Two distinct ids can decode to the
    // same string (and control tokens render differently across engines), so
    // anything comparing our output to another runtime's needs the id to tell
    // "same token" from "same text" — scripts/token_divergence.py does.
    sb_lit(r, "],\"token_ids\":[");
    for (int i = 0; i < e->lp_count; i++) {
        if (i) sb_lit(r, ",");
        sb_fmt(r, "%d", e->lp_ids[i]);
    }
    sb_lit(r, "],\"token_logprobs\":[");
    for (int i = 0; i < e->lp_count; i++) {
        if (i) sb_lit(r, ",");
        sb_fmt(r, "%.6f", e->lp_chosen[i]);
    }
    sb_lit(r, "],\"top_logprobs\":[");
    for (int i = 0; i < e->lp_count; i++) {
        if (i) sb_lit(r, ",");
        sb_lit(r, "{");
        for (int j = 0; j < e->lp_n; j++) {
            const lp_alt *a = &e->lp_top[(size_t)i * e->lp_n + j];
            if (a->id < 0) break;
            if (j) sb_lit(r, ",");
            int tn = tok_decode(s->tok, a->id, tb, sizeof(tb));
            sb_lit(r, "\""); sb_esc(r, tb, tn);
            sb_fmt(r, "\":%.6f", a->lp);
        }
        sb_lit(r, "}");
    }
    // The OpenAI top_logprobs shape is a string->float map with nowhere to put
    // an id, so the ids ride in a parallel array in the same order.
    sb_lit(r, "],\"top_token_ids\":[");
    for (int i = 0; i < e->lp_count; i++) {
        if (i) sb_lit(r, ",");
        sb_lit(r, "[");
        for (int j = 0; j < e->lp_n; j++) {
            const lp_alt *a = &e->lp_top[(size_t)i * e->lp_n + j];
            if (a->id < 0) break;
            if (j) sb_lit(r, ",");
            sb_fmt(r, "%d", a->id);
        }
        sb_lit(r, "]");
    }
    sb_lit(r, "],\"text_offset\":[");
    for (int i = 0; i < e->lp_count; i++) {
        if (i) sb_lit(r, ",");
        sb_fmt(r, "%d", offset);
        offset += tok_decode(s->tok, e->lp_ids[i], tb, sizeof(tb));
    }
    sb_lit(r, "]}");
}

static void completion_cleanup(engine *e, snode *schema, gen_ctx *g) {
    e->schema = NULL;
    e->emit_think_prelude = false;
    e->constraint_includes_prelude = false;
    schema_free(schema);
    if (g) {
        tool_stream_free(&g->tsx);
        think_free(&g->ts);
        free(g->hold.s);
        free(g->reason.s);
        free(g->out.s);
        free(g->item_text.s);
        free(g->call_name);
        free(g->out_items.s);
        free(g->out_text.s);
    }
    free(e->lp_chosen); free(e->lp_ids); free(e->lp_top);
    e->lp_chosen = NULL; e->lp_ids = NULL; e->lp_top = NULL;
    e->lp_cap = e->lp_n = e->lp_count = 0;
    free(e->cl_recs);
    e->cl_recs = NULL;
    e->cl_cap = e->cl_count = e->cl_probe = 0;
}

// emit one section of split output: reasoning goes to the OpenAI-style
// reasoning_content field, everything else to content
// one text delta on the named chat channel (or the legacy completion "text")
static int responses_text_delta(gen_ctx *g, int reasoning, const char *bytes,
                                int n);
static int anth_delta(gen_ctx *g, const char *kind, const char *bytes, int n);

// How many trailing bytes of `s` begin a UTF-8 sequence that is not finished
// yet — i.e. how much must be held back until the next token arrives. 0 when
// the buffer already ends on a character boundary.
//
// This reads the byte SHAPE only; it deliberately does not judge validity.
// A stray continuation byte returns 0 and goes out immediately, because
// nothing can complete it. An invalid lead such as 0xC0 has a two-byte shape
// and IS held for one token — harmless, because whatever it becomes is then
// rejected by json_escape and rendered U+FFFD, and the flush at end of
// generation stops it being held forever. Validity belongs in one place, and
// that place is the escaper.
static int u8_incomplete_tail(const char *s, int n) {
    for (int back = 1; back <= 3 && back <= n; back++) {
        unsigned char c = (unsigned char)s[n - back];
        if ((c & 0xC0) == 0x80) continue;          // continuation: keep looking
        int len = (c & 0xE0) == 0xC0 ? 2 : (c & 0xF0) == 0xE0 ? 3 :
                  (c & 0xF8) == 0xF0 ? 4 : 0;
        // A lead byte whose sequence runs past the end of what we have is the
        // only case worth waiting on.
        return (len > back) ? back : 0;
    }
    return 0;
}

static int send_text_delta_raw(gen_ctx *g, int reasoning, const char *bytes, int n);

// Splice in whatever was held back last time, and hold back an unfinished
// tail this time. Callers hand over whole tokens; characters do not respect
// token boundaries.
static int send_text_delta(gen_ctx *g, int reasoning, const char *bytes, int n) {
    if (!g->stream || g->dead) return g->dead ? 1 : 0;
    int ch = reasoning ? 1 : 0;
    int held = g->u8_pend_n[ch];
    if (!held && u8_incomplete_tail(bytes, n) == 0)
        return send_text_delta_raw(g, reasoning, bytes, n);   // the common path

    int total = held + n;
    char *joined = malloc((size_t)total + 1);
    if (!joined) {
        // An OOM must not REORDER the stream. Sending the new bytes and
        // leaving the hold in place did not drop the held half — it delivered
        // it later, prepended to a subsequent delta or flushed at the end, so
        // the first half of a character arrived after words the model wrote
        // after it. Emitting the hold first splits one character across two
        // deltas (each half escapes to U+FFFD), which is exactly what
        // flush_text_delta already does at end of generation, and is the only
        // outcome here that keeps the model's own order.
        int rc = send_text_delta_raw(g, reasoning, g->u8_pend[ch], held);
        g->u8_pend_n[ch] = 0;
        if (rc) return rc;
        return send_text_delta_raw(g, reasoning, bytes, n);
    }
    memcpy(joined, g->u8_pend[ch], (size_t)held);
    memcpy(joined + held, bytes, (size_t)n);
    g->u8_pend_n[ch] = 0;

    int tail = u8_incomplete_tail(joined, total);
    int emit = total - tail;
    if (tail) {
        memcpy(g->u8_pend[ch], joined + emit, (size_t)tail);
        g->u8_pend_n[ch] = tail;
    }
    int rc = emit > 0 ? send_text_delta_raw(g, reasoning, joined, emit) : 0;
    free(joined);
    return rc;
}

// Emit whatever is still held back. A sequence unfinished when generation ends
// is genuinely truncated, so it goes out and is replaced with U+FFFD — the one
// honest rendering of bytes the model never completed.
static int flush_text_delta(gen_ctx *g, int reasoning) {
    int ch = reasoning ? 1 : 0;
    int held = g->u8_pend_n[ch];
    if (!held) return 0;
    g->u8_pend_n[ch] = 0;
    return send_text_delta_raw(g, reasoning, g->u8_pend[ch], held);
}

static int send_text_delta_raw(gen_ctx *g, int reasoning, const char *bytes, int n) {
    if (!g->stream || g->dead) return g->dead ? 1 : 0;
    if (g->api == API_RESPONSES) return responses_text_delta(g, reasoning, bytes, n);
    // Anthropic separates reasoning into a `thinking` content block rather
    // than a field on the message, so the channel selects the block kind
    if (g->api == API_MESSAGES)
        return anth_delta(g, reasoning ? "thinking" : "text", bytes, n);
    sbuf c = {0};
    chunk_open(g, &c);
    if (g->api == API_CHAT) {
        sb_lit(&c, "\"delta\":{");
        if (!g->role_sent) { sb_lit(&c, "\"role\":\"assistant\","); g->role_sent = true; }
        sb_fmt(&c, "\"%s\":\"", reasoning ? "reasoning_content" : "content");
    } else {
        sb_lit(&c, "\"text\":\"");
    }
    sb_esc(&c, bytes, n);
    sb_lit(&c, g->api == API_CHAT ? "\"},\"finish_reason\":null}]}"
                                  : "\",\"finish_reason\":null}]}");
    return chunk_send(g, &c);
}

// ---- tool_stream sinks: the demuxed envelope, as OpenAI streaming events

static int sink_content(void *ud, const char *b, int n) {
    return send_text_delta(ud, 0, b, n);
}

static int sink_reasoning(void *ud, const char *b, int n) {
    gen_ctx *g = ud;
    sb_put(&g->reason, b, (size_t)n);
    return send_text_delta(g, 1, b, n);
}

// the opening event of a call carries everything that identifies it; the
// deltas that follow carry argument text only, keyed by the same index
static int resp_open_item(gen_ctx *g, const char *kind);
static int resp_delta(gen_ctx *g, const char *kind, const char *bytes, int n);
static int resp_close_item(gen_ctx *g);

static int anth_open_block(gen_ctx *g, const char *kind);
static int anth_close_block(gen_ctx *g);

static int sink_call_begin(void *ud, const char *name) {
    gen_ctx *g = ud;
    if (g->dead) return 1;
    if (g->api == API_RESPONSES || g->api == API_MESSAGES) {
        // the name identifies the item/block, so it must be known before that
        // is announced — which is exactly when tool_stream calls this
        free(g->call_name);
        g->call_name = strdup(name);
        if (!g->call_name) { g->dead = true; return 1; }
        return g->api == API_MESSAGES ? anth_open_block(g, "tool_use")
                                      : resp_open_item(g, "function_call");
    }
    sbuf c = {0};
    chunk_open(g, &c);
    sb_lit(&c, "\"delta\":{");
    if (!g->role_sent) { sb_lit(&c, "\"role\":\"assistant\","); g->role_sent = true; }
    sb_fmt(&c, "\"tool_calls\":[{\"index\":%d,\"id\":\"call_%d\","
               "\"type\":\"function\",\"function\":{\"name\":\"",
           g->tool_index, g->tool_index);
    sb_esc(&c, name, strlen(name));
    sb_lit(&c, "\",\"arguments\":\"\"}}]},\"finish_reason\":null}]}");
    return chunk_send(g, &c);
}

static int sink_call_args(void *ud, const char *b, int n) {
    gen_ctx *g = ud;
    if (g->dead) return 1;
    if (g->api == API_RESPONSES) return resp_delta(g, "function_call", b, n);
    if (g->api == API_MESSAGES) return anth_delta(g, "tool_use", b, n);
    sbuf c = {0};
    chunk_open(g, &c);
    sb_fmt(&c, "\"delta\":{\"tool_calls\":[{\"index\":%d,\"function\":"
               "{\"arguments\":\"", g->tool_index);
    sb_esc(&c, b, n);
    sb_lit(&c, "\"}}]},\"finish_reason\":null}]}");
    return chunk_send(g, &c);
}

static int sink_call_end(void *ud) {
    gen_ctx *g = ud;
    int rc = 0;
    if (g->api == API_RESPONSES) rc = resp_close_item(g);
    else if (g->api == API_MESSAGES) rc = anth_close_block(g);
    if (!rc) g->tool_index++;
    return rc;
}

// ------------------------------------------------- Responses API framing
//
// The Responses surface is a second *vocabulary* for the one generation path,
// not a second engine. Everything below is framing: the same bytes the chat
// dialect sends as ChatCompletionChunk deltas are sent here as ordered typed
// events, and the same buffered result is rendered as an `output[]` of items.
//
// Two properties are the actual contract, and both are what SDK clients
// validate:
//
//   1. Order. An item is announced (`output_item.added`) before any of its
//      deltas and closed (`output_item.done`) after them, with a content part
//      opened and closed inside it; `response.created` opens the stream and
//      `response.completed` closes it.
//   2. Naming. Every event names itself twice — in the SSE `event:` field and
//      in `data.type` — and typed clients dispatch on the first while
//      validating the second, so the two must always agree. Framing them in
//      one place is what guarantees that.
//
// A monotonic `sequence_number` is stamped on every event by the framer for
// the same reason: it cannot drift from the send order if nothing else can
// assign it.

// item kinds share one state machine and differ only in these names
typedef struct {
    const char *kind;       // the item's "type"
    const char *id_prefix;
    const char *part_added, *delta, *text_done, *part_done;
} resp_shape;

static const resp_shape RESP_MESSAGE = {
    "message", "msg", "response.content_part.added",
    "response.output_text.delta", "response.output_text.done",
    "response.content_part.done" };
static const resp_shape RESP_REASONING = {
    "reasoning", "rs", "response.reasoning_summary_part.added",
    "response.reasoning_summary_text.delta",
    "response.reasoning_summary_text.done",
    "response.reasoning_summary_part.done" };
static const resp_shape RESP_CALL = {
    "function_call", "fc", NULL,
    "response.function_call_arguments.delta",
    "response.function_call_arguments.done", NULL };

static const resp_shape *resp_shape_of(const char *kind) {
    if (!kind) return &RESP_MESSAGE;
    if (!strcmp(kind, "reasoning")) return &RESP_REASONING;
    if (!strcmp(kind, "function_call")) return &RESP_CALL;
    return &RESP_MESSAGE;
}

// frame one typed SSE event. `fields` holds the event-specific members already
// written as `,"key":value` pairs; it is consumed (freed) here.
//
// Both typed surfaces name every event twice — in the SSE `event:` field and
// in `data.type` — and both are validated by their SDKs, so the two names are
// written from one argument here and cannot drift apart on either. The only
// difference is `sequence_number`, which the Responses vocabulary stamps on
// every event and the Anthropic one does not have at all.
static int sse_send(gen_ctx *g, const char *type, sbuf *fields, bool seq) {
    sbuf e = {0};
    sb_fmt(&e, "event: %s\ndata: {\"type\":\"%s\"", type, type);
    if (seq) sb_fmt(&e, ",\"sequence_number\":%ld", g->seq++);
    if (fields->s) sb_put(&e, fields->s, fields->n);
    sb_lit(&e, "}\n\n");
    if (fields->failed || e.failed || !send_all(g->fd, e.s, e.n)) g->dead = true;
    free(fields->s);
    free(e.s);
    return g->dead ? 1 : 0;
}

static int resp_send(gen_ctx *g, const char *type, sbuf *fields) {
    return sse_send(g, type, fields, true);
}

static int anth_send(gen_ctx *g, const char *type, sbuf *fields) {
    return sse_send(g, type, fields, false);
}

// the open item, rendered as a Responses output item
static void resp_item_json(sbuf *b, gen_ctx *g, const char *status, bool filled) {
    const resp_shape *sh = resp_shape_of(g->item_kind);
    const char *text = g->item_text.s ? g->item_text.s : "";
    size_t text_n = g->item_text.n;
    sb_fmt(b, "{\"id\":\"%s\",\"type\":\"%s\",\"status\":\"%s\"",
           g->item_id, sh->kind, status);
    if (sh == &RESP_CALL) {
        sb_fmt(b, ",\"call_id\":\"call_%d\",\"name\":\"", g->tool_index);
        sb_esc(b, g->call_name ? g->call_name : "", strlen(g->call_name ? g->call_name : ""));
        // arguments are already a JSON *string* on the wire, so the accumulated
        // argument text is escaped into it exactly as the chat dialect does
        sb_lit(b, "\",\"arguments\":\"");
        if (filled) sb_esc(b, text, text_n);
        sb_lit(b, "\"}");
        return;
    }
    if (sh == &RESP_REASONING) {
        sb_lit(b, ",\"summary\":[");
        if (filled) {
            sb_lit(b, "{\"type\":\"summary_text\",\"text\":\"");
            sb_esc(b, text, text_n);
            sb_lit(b, "\"}");
        }
        sb_lit(b, "]}");
        return;
    }
    sb_lit(b, ",\"role\":\"assistant\",\"content\":[");
    if (filled) {
        sb_lit(b, "{\"type\":\"output_text\",\"text\":\"");
        sb_esc(b, text, text_n);
        sb_lit(b, "\",\"annotations\":[]}");
    }
    sb_lit(b, "]}");
}

// close whatever item is open: part done (when the kind has parts), then the
// item itself. Nothing is emitted when no item is open, so this is safe to
// call on every path that ends an item — including the terminal one.
static int resp_close_item(gen_ctx *g) {
    if (!g->item_open || g->dead) return g->dead ? 1 : 0;
    const resp_shape *sh = resp_shape_of(g->item_kind);
    const char *text = g->item_text.s ? g->item_text.s : "";
    size_t text_n = g->item_text.n;
    if (g->part_open) {
        sbuf f = {0};
        sb_fmt(&f, ",\"item_id\":\"%s\",\"output_index\":%d,\"content_index\":0,"
                   "\"%s\":\"", g->item_id, g->output_index,
               sh == &RESP_CALL ? "arguments" : "text");
        sb_esc(&f, text, text_n);
        sb_lit(&f, "\"");
        if (sh != &RESP_CALL) sb_lit(&f, ",\"logprobs\":[]");
        if (resp_send(g, sh->text_done, &f)) return 1;
        if (sh->part_done) {
            sbuf p = {0};
            sb_fmt(&p, ",\"item_id\":\"%s\",\"output_index\":%d,"
                       "\"content_index\":0,\"part\":", g->item_id,
                   g->output_index);
            if (sh == &RESP_REASONING) {
                sb_lit(&p, "{\"type\":\"summary_text\",\"text\":\"");
                sb_esc(&p, text, text_n);
                sb_lit(&p, "\"}");
            } else {
                sb_lit(&p, "{\"type\":\"output_text\",\"text\":\"");
                sb_esc(&p, text, text_n);
                sb_lit(&p, "\",\"annotations\":[]}");
            }
            if (resp_send(g, sh->part_done, &p)) return 1;
        }
        g->part_open = false;
    }
    const char *status = g->close_status ? g->close_status : "completed";
    sbuf d = {0};
    sb_fmt(&d, ",\"output_index\":%d,\"item\":", g->output_index);
    resp_item_json(&d, g, status, true);
    // keep the completed item for the terminal response object
    if (g->out_items.n) sb_lit(&g->out_items, ",");
    resp_item_json(&g->out_items, g, status, true);
    if (resp_shape_of(g->item_kind) == &RESP_MESSAGE)
        sb_put(&g->out_text, g->item_text.s ? g->item_text.s : "",
               g->item_text.n);
    int rc = resp_send(g, "response.output_item.done", &d);
    g->item_open = false;
    g->output_index++;
    g->item_text.n = 0;
    return rc;
}

// open an item of `kind`, closing any item already open. The id is derived
// from the kind and the output index so it is stable and collision-free.
static int resp_open_item(gen_ctx *g, const char *kind) {
    if (g->item_open && g->item_kind && !strcmp(g->item_kind, kind)) return 0;
    if (resp_close_item(g)) return 1;
    const resp_shape *sh = resp_shape_of(kind);
    g->item_kind = sh->kind;
    snprintf(g->item_id, sizeof(g->item_id), "%s_%d", sh->id_prefix,
             g->output_index);
    sbuf a = {0};
    sb_fmt(&a, ",\"output_index\":%d,\"item\":", g->output_index);
    resp_item_json(&a, g, "in_progress", false);
    if (resp_send(g, "response.output_item.added", &a)) return 1;
    g->item_open = true;
    // a function_call carries its arguments directly on the item, so it has no
    // content part; the other kinds open one before their first delta
    if (sh->part_added) {
        sbuf p = {0};
        sb_fmt(&p, ",\"item_id\":\"%s\",\"output_index\":%d,\"content_index\":0,"
                   "\"part\":", g->item_id, g->output_index);
        if (sh == &RESP_REASONING)
            sb_lit(&p, "{\"type\":\"summary_text\",\"text\":\"\"}");
        else
            sb_lit(&p, "{\"type\":\"output_text\",\"text\":\"\",\"annotations\":[]}");
        if (resp_send(g, sh->part_added, &p)) return 1;
        g->part_open = true;
    } else {
        g->part_open = true; // the arguments "part" is the item itself
    }
    return 0;
}

static int resp_delta(gen_ctx *g, const char *kind, const char *bytes, int n) {
    if (g->dead) return 1;
    if (resp_open_item(g, kind)) return 1;
    sb_put(&g->item_text, bytes, n);
    const resp_shape *sh = resp_shape_of(kind);
    sbuf f = {0};
    sb_fmt(&f, ",\"item_id\":\"%s\",\"output_index\":%d", g->item_id,
           g->output_index);
    if (sh != &RESP_CALL) sb_lit(&f, ",\"content_index\":0");
    sb_lit(&f, ",\"delta\":\"");
    sb_esc(&f, bytes, n);
    sb_lit(&f, "\"");
    if (sh == &RESP_MESSAGE) sb_lit(&f, ",\"logprobs\":[]");
    return resp_send(g, sh->delta, &f);
}

static int responses_text_delta(gen_ctx *g, int reasoning, const char *bytes,
                                int n) {
    return resp_delta(g, reasoning ? "reasoning" : "message", bytes, n);
}

// Everything the response object reports about one finished (or just-started)
// turn. Passing it as one value is what lets the buffered body and the
// terminal `response.completed` event be the *same* document: a client that
// switches `stream` on and off sees one shape, not two that drifted.
typedef struct {
    const char *status;      // in_progress | completed | incomplete
    const char *incomplete;  // reason, when status is incomplete
    // the same two facts in the Anthropic vocabulary. Both surfaces derive
    // them from the one `finish` string, so they cannot describe the turn
    // differently.
    const char *stop_reason; // end_turn | max_tokens | stop_sequence | tool_use
    const char *stop_seq;    // the matched stop sequence, when that is the reason
    bool        with_output;
    // A streamed turn already rendered its items one by one, so it hands them
    // over verbatim rather than rebuilding them from different inputs — which
    // is what keeps the streamed and buffered documents identical by
    // construction instead of by review.
    const char *output_json; size_t output_n;
    const char *output_text; size_t output_text_n;
    // The turn's tool calls, in the chat dialect's canonical shape (a J_ARR of
    // {id,type,function:{name,arguments}}). One extraction, three renderings:
    // the chat body splices the same list into `tool_calls`, and a turn that
    // called several tools must not lose the extras on the way to a surface
    // whose vocabulary happens to spell them differently.
    jv         *calls;
    const char *text;   size_t text_n;
    const char *reason; size_t reason_n;
    bool        with_usage;
    int         n_prompt, n_gen, cached;
    // of `cached`, how many rows were forked out of the shared prefix cache
    // rather than left over in this slot, and what that saved
    int         forked;
    double      saved_s;
    double      gtime;
    bool        schema, json_mode, spec;
    // Set when the OpenAI wire value was widened to a standard one (see
    // openai_finish): carries the reason we would otherwise have emitted.
    const char *finish_detail;
    // Major page faults taken while serving this request. Nonzero means the
    // time went to disk, not to arithmetic.
    uint64_t    major_faults;
    jv         *req;         // echoed request fields
} resp_doc;

// Cumulative work counters (declared in server_int.h). Microseconds rather
// than a double because there is no portable atomic double, and this only
// needs summing and one division at read time.
static atomic_ullong g_total_prompt_tokens;
static atomic_ullong g_total_gen_tokens;
static atomic_ullong g_total_gen_micros;

void server_record_work(int n_prompt, int n_gen, double gen_seconds) {
    if (n_prompt > 0)
        atomic_fetch_add(&g_total_prompt_tokens, (unsigned long long)n_prompt);
    if (n_gen > 0)
        atomic_fetch_add(&g_total_gen_tokens, (unsigned long long)n_gen);
    if (gen_seconds > 0)
        atomic_fetch_add(&g_total_gen_micros,
                         (unsigned long long)(gen_seconds * 1e6));
}

void server_work_totals(unsigned long long *prompt_tokens,
                        unsigned long long *gen_tokens, double *gen_seconds) {
    *prompt_tokens = atomic_load(&g_total_prompt_tokens);
    *gen_tokens    = atomic_load(&g_total_gen_tokens);
    *gen_seconds   = (double)atomic_load(&g_total_gen_micros) / 1e6;
}

// One rendering of runner_telemetry for every surface that carries it — chat,
// completions, responses (streamed and buffered) and messages. They report the
// same facts, so they share the one place those facts are spelled rather than
// four format strings that drift.
static void telemetry_json(sbuf *r, const resp_doc *d) {
    sb_fmt(r, "\"runner_telemetry\":{\"prompt_cached_tokens\":%d,"
              "\"prompt_forked_tokens\":%d,\"prompt_eval_tokens\":%d,"
              "\"prefix_cache_saved_seconds\":%.6f,"
              "\"generation_seconds\":%.6f,"
              "\"generation_tok_s\":%.3f,\"major_page_faults\":%llu,"
              "\"json_mode\":%s,"
              "\"schema\":%s,\"speculative\":%s",
           d->cached, d->forked, d->n_prompt - d->cached, d->saved_s, d->gtime,
           d->n_gen / (d->gtime > 0 ? d->gtime : 1e-9),
           (unsigned long long)d->major_faults,
           d->json_mode ? "true" : "false", d->schema ? "true" : "false",
           d->spec ? "true" : "false");
    // Only present when the standard finish_reason lost a distinction, so
    // ordinary turns are byte-for-byte what they were before.
    if (d->finish_detail) sb_fmt(r, ",\"finish_detail\":\"%s\"", d->finish_detail);
    sb_lit(r, "}");
}

static void resp_echo(sbuf *r, jv *req, const char *key, const char *dflt) {
    jv *v = jv_get(req, key);
    sb_fmt(r, ",\"%s\":", key);
    if (!v || v->type == J_NULL) sb_lit(r, dflt);
    else jv_dump(v, r);
}

// The chat dialect's `tool_calls` array is where a turn's calls are extracted
// once; the Responses and Anthropic bodies render that same list in their own
// vocabulary. `tc` holds the entries comma-separated, because that is how the
// chat body splices them into its array -- so the parse must be over the ARRAY
// they form. Parsing `tc` on its own stops after the first entry and json_parse
// then refuses the rest as trailing garbage, which returned NULL for every
// multi-call turn and dropped the whole thing.
static jv *tool_calls_array(const sbuf *tc, int n_tc) {
    if (n_tc <= 0 || !tc->s || tc->n == 0) return NULL;
    sbuf a = {0};
    sb_lit(&a, "[");
    sb_put(&a, tc->s, tc->n);
    sb_lit(&a, "]");
    jv *v = a.failed ? NULL : json_parse(a.s, a.n);
    free(a.s);
    if (v && v->type != J_ARR) { jv_free(v); v = NULL; }
    return v;
}

static const char *call_field(const jv *calls, int i, const char *key,
                              const char *dflt) {
    return jv_str(jv_get(jv_get(calls->items[i], "function"), key), dflt);
}

static void responses_body(sbuf *r, gen_ctx *g, const resp_doc *d) {
    int n_calls = d->calls ? d->calls->n : 0;
    sb_fmt(r, "{\"id\":\"%s\",\"object\":\"response\",\"created_at\":%ld,"
              "\"status\":\"%s\",\"error\":null,\"incomplete_details\":",
           g->id, g->created, d->status);
    if (d->incomplete) sb_fmt(r, "{\"reason\":\"%s\"}", d->incomplete);
    else               sb_lit(r, "null");
    sb_lit(r, ",\"model\":\"");
    sb_esc(r, SV.model_name, strlen(SV.model_name));
    sb_lit(r, "\",\"output\":[");
    if (d->output_json) {
        sb_put(r, d->output_json, d->output_n);
    } else if (d->with_output) {
        int idx = 0;
        if (d->reason_n) {
            sb_fmt(r, "{\"id\":\"rs_%d\",\"type\":\"reasoning\","
                      "\"status\":\"completed\",\"summary\":"
                      "[{\"type\":\"summary_text\",\"text\":\"", idx);
            sb_esc(r, d->reason, d->reason_n);
            sb_lit(r, "\"}]}");
            idx++;
        }
        if (n_calls) {
            // calls replace the assistant message rather than accompanying it,
            // matching finish_reason "tool_calls"
            for (int i = 0; i < n_calls; i++) {
                const char *name = call_field(d->calls, i, "name", "");
                const char *args = call_field(d->calls, i, "arguments", "{}");
                if (idx) sb_lit(r, ",");
                sb_fmt(r, "{\"id\":\"fc_%d\",\"type\":\"function_call\","
                          "\"status\":\"completed\",\"call_id\":\"call_%d\","
                          "\"name\":\"", idx, i);
                sb_esc(r, name, strlen(name));
                sb_lit(r, "\",\"arguments\":\"");
                sb_esc(r, args, strlen(args));
                sb_lit(r, "\"}");
                idx++;
            }
        } else {
            if (idx) sb_lit(r, ",");
            sb_fmt(r, "{\"id\":\"msg_%d\",\"type\":\"message\","
                      "\"status\":\"%s\",\"role\":\"assistant\",\"content\":"
                      "[{\"type\":\"output_text\",\"text\":\"", idx,
                   d->incomplete ? "incomplete" : "completed");
            sb_esc(r, d->text ? d->text : "", d->text_n);
            sb_lit(r, "\",\"annotations\":[]}]}");
        }
    }
    sb_lit(r, "],\"output_text\":\"");
    // the SDK's `response.output_text` convenience aggregate: the assistant
    // text only, empty when the turn produced a call instead
    if (d->output_json) sb_esc(r, d->output_text ? d->output_text : "",
                               d->output_text_n);
    else if (d->with_output && !n_calls)
        sb_esc(r, d->text ? d->text : "", d->text_n);
    sb_lit(r, "\"");
    // request echo: a Responses client reads these back off the object rather
    // than remembering what it sent
    resp_echo(r, d->req, "instructions", "null");
    resp_echo(r, d->req, "metadata", "null");
    resp_echo(r, d->req, "temperature", "null");
    resp_echo(r, d->req, "top_p", "null");
    resp_echo(r, d->req, "max_output_tokens", "null");
    resp_echo(r, d->req, "reasoning", "null");
    resp_echo(r, d->req, "text", "{\"format\":{\"type\":\"text\"}}");
    resp_echo(r, d->req, "tools", "[]");
    resp_echo(r, d->req, "tool_choice", "\"auto\"");
    resp_echo(r, d->req, "parallel_tool_calls", "false");
    sb_lit(r, ",\"previous_response_id\":null,\"store\":false,"
              "\"truncation\":\"disabled\",\"user\":null,\"usage\":");
    if (d->with_usage) {
        sb_fmt(r, "{\"input_tokens\":%d,"
                  "\"input_tokens_details\":{\"cached_tokens\":%d},"
                  "\"output_tokens\":%d,"
                  "\"output_tokens_details\":{\"reasoning_tokens\":0},"
                  "\"total_tokens\":%d}",
               d->n_prompt, d->cached, d->n_gen, d->n_prompt + d->n_gen);
        sb_lit(r, ",");
        telemetry_json(r, d);
    } else {
        sb_lit(r, "null");
    }
    sb_lit(r, "}");
}

// ----------------------------------------------- Anthropic Messages framing
//
// The third vocabulary for the one generation path. Where the Responses
// surface renders the turn as an `output[]` of items, Anthropic renders it as
// a `content[]` of *blocks*, and the streaming form is the same state machine
// one level flatter: a block is opened, filled with deltas, and closed, with no
// content-part nesting inside it.
//
// The six event names and their order are the contract the Anthropic SDKs
// validate:
//
//   message_start
//     content_block_start / content_block_delta* / content_block_stop   (xN)
//   message_delta        — the terminal stop_reason and the output token count
//   message_stop
//
// There is no `data: [DONE]` sentinel: message_stop is the terminator.

// The stop_reason vocabulary, derived from the same `finish` string the chat
// dialect reports, so the two surfaces cannot disagree about how a turn ended.
// The OpenAI `finish_reason` vocabulary is a CLOSED set — stop, length,
// tool_calls, content_filter, function_call. `reasoning_limit` is ours, and
// emitting it raw on a surface whose whole selling point is drop-in
// compatibility breaks any typed client whose finish_reason is a closed union.
// The official OpenAI SDK happens to deserialise it, which is what let this sit
// unnoticed; a strictly-typed one has no case for it.
//
// `length` is not a euphemism here: the turn really did stop because it ran out
// of token budget. The only thing lost is WHERE the budget went, and that
// belongs in an extension field rather than smuggled into a standard enum —
// which is exactly what the Responses (`max_output_tokens`) and Anthropic
// (`max_tokens`) surfaces already do. Chat was the odd one out.
//
// The distinction is preserved as `finish_detail` in runner_telemetry, so no
// information is lost: a client that cares can still tell a prelude exhaustion
// from an ordinary truncation.
//
// KNOWN GAP: streamed chat/completions never carried runner_telemetry at all
// (its only extra terminal chunk is the opt-in include_usage one), so a
// STREAMED turn reports "length" with no detail available anywhere. Buffered
// turns, Responses and Anthropic all keep the full distinction. Widening the
// streamed terminal chunk is a separate wire change and is deliberately not
// bundled here.
//
// Owner decision 2026-08-08. Field evidence: a reasoning model truncated inside
// its <think> block returns an EMPTY answer, and a harness that saw the
// non-standard reason recorded incapability instead of truncation — every
// agent case scored 0.00 for every model in the Syntetik-MoE baseline.
//
// "envelope_error" is widened the same way and for the same reason: it is a
// generation fault, which this surface already spells "error" (see e->oom), and
// inventing a second non-standard value would only give typed clients a second
// case they have never heard of. Which fault it was survives as finish_detail.
static const char *openai_finish(const char *finish) {
    if (!strcmp(finish, "reasoning_limit")) return "length";
    if (!strcmp(finish, "envelope_error"))  return "error";
    return finish;
}

// Non-NULL only when the wire value above lost a distinction worth keeping.
static const char *finish_detail_of(const char *finish) {
    if (!strcmp(finish, "reasoning_limit")) return "reasoning_limit";
    if (!strcmp(finish, "envelope_error"))  return "envelope_unmapped";
    return NULL;
}

// A GENERATION FAULT: the turn produced nothing anybody can read, and the
// reason is the server's rather than the model's or the caller's. Two causes
// reach this state and both are reported the same way on every surface —
// e->oom (an allocation failure mid-generation) and "envelope_error" (a
// constrained document the demultiplexer could not map back).
//
// It is deliberately NOT the same class as "length" or "stop": those are
// completions with a boundary, and a client can use what it got. A fault has
// no usable turn behind it at all.
static bool generation_faulted(const char *finish) {
    return !strcmp(finish, "error") || !strcmp(finish, "envelope_error");
}

// Anthropic's vocabulary has no member for a generation fault, and it is not
// an oversight: the API draws the line the other way round from OpenAI's, in
// its own words — "Unlike errors, which indicate failures in processing your
// request, `stop_reason` tells you why Claude completed its response
// generation." All seven values (end_turn, max_tokens, stop_sequence,
// tool_use, pause_turn, refusal, model_context_window_exceeded) describe a
// turn that COMPLETED. A fault is an error object, not a stop_reason, so a
// faulted turn never reaches this function at all — both callers test
// generation_faulted() and divert to anth_error_json() first. Nothing here
// understates a fault any more.
static const char *anth_stop_reason(const char *finish, bool stop_hit) {
    if (!strcmp(finish, "tool_calls")) return "tool_use";
    if (!strcmp(finish, "length") ||
        !strcmp(finish, "reasoning_limit")) return "max_tokens";
    // a user stop sequence is its own terminal reason in this vocabulary, and
    // the matched string is reported alongside it
    return stop_hit ? "stop_sequence" : "end_turn";
}

// The error object's members, written once so the buffered body and the
// streamed `error` event cannot describe the same fault differently — the same
// reason sse_send names every event from one argument.
//
// `api_error` with a 500 is a deliberate pick out of Anthropic's typed set,
// and each alternative in that set says something false here:
//
//   invalid_request_error (400)  blames the caller for a request that was
//                                accepted and well-formed. Nothing they can
//                                edit changes the outcome, and a 4xx is the
//                                one class SDKs do NOT retry.
//   overloaded_error (529)       means capacity. Clients shed load or fail
//                                over on it; nothing here is loaded.
//   timeout_error (504)          nothing timed out.
//
// `api_error` is documented as "an unexpected error ... internal to
// Anthropic's systems. Retry the request with exponential backoff" — which is
// what both faults are, and what a client should do about them. An unmappable
// envelope is sampling-dependent, so a retry can genuinely succeed; an
// allocation failure can clear too. 500 is also the classification under which
// every official SDK already retries with backoff unprompted.
//
// Which fault it was survives in runner_telemetry, exactly as it does on the
// chat and Responses surfaces, rather than being smuggled into `error.type`
// where a typed client has no case for it.
static void anth_error_json(sbuf *r, const char *finish) {
    const char *msg = !strcmp(finish, "envelope_error")
        ? "the model's constrained tool-call document could not be mapped "
          "back to a response, so this turn produced no readable content. "
          "The request was valid; retry it."
        : "generation ran out of memory before producing a response. "
          "The request was valid; retry it.";
    sb_lit(r, "\"error\":{\"type\":\"api_error\",\"message\":\"");
    sb_esc(r, msg, strlen(msg));
    sb_lit(r, "\"}");
    const char *detail = finish_detail_of(finish);
    if (detail)
        sb_fmt(r, ",\"runner_telemetry\":{\"finish_detail\":\"%s\"}", detail);
}

static int anth_close_block(gen_ctx *g) {
    if (!g->item_open || g->dead) return g->dead ? 1 : 0;
    sbuf f = {0};
    sb_fmt(&f, ",\"index\":%d", g->output_index);
    int rc = anth_send(g, "content_block_stop", &f);
    g->item_open = false;
    g->output_index++;
    g->item_text.n = 0;
    return rc;
}

// open a block of `kind`, closing any block already open. Re-opening the kind
// that is already open is a no-op, which is what makes a run of deltas on one
// channel land in one block.
static int anth_open_block(gen_ctx *g, const char *kind) {
    if (g->item_open && g->item_kind && !strcmp(g->item_kind, kind)) return 0;
    if (anth_close_block(g)) return 1;
    g->item_kind = kind;
    sbuf f = {0};
    sb_fmt(&f, ",\"index\":%d,\"content_block\":", g->output_index);
    if (!strcmp(kind, "tool_use")) {
        sb_fmt(&f, "{\"type\":\"tool_use\",\"id\":\"toolu_%d\",\"name\":\"",
               g->tool_index);
        sb_esc(&f, g->call_name ? g->call_name : "",
               strlen(g->call_name ? g->call_name : ""));
        // the arguments arrive as input_json_delta text; the block opens with
        // the empty object an SDK accumulator starts from
        sb_lit(&f, "\",\"input\":{}}");
    } else if (!strcmp(kind, "thinking")) {
        sb_lit(&f, "{\"type\":\"thinking\",\"thinking\":\"\"}");
    } else {
        sb_lit(&f, "{\"type\":\"text\",\"text\":\"\"}");
    }
    if (anth_send(g, "content_block_start", &f)) return 1;
    g->item_open = true;
    return 0;
}

static int anth_delta(gen_ctx *g, const char *kind, const char *bytes, int n) {
    if (g->dead) return 1;
    if (anth_open_block(g, kind)) return 1;
    sb_put(&g->item_text, bytes, n);
    sbuf f = {0};
    sb_fmt(&f, ",\"index\":%d,\"delta\":{\"type\":\"", g->output_index);
    // each block kind carries its text under its own delta type and key; an
    // SDK dispatches on the first and reads the second
    if (!strcmp(kind, "tool_use"))
        sb_lit(&f, "input_json_delta\",\"partial_json\":\"");
    else if (!strcmp(kind, "thinking"))
        sb_lit(&f, "thinking_delta\",\"thinking\":\"");
    else
        sb_lit(&f, "text_delta\",\"text\":\"");
    sb_esc(&f, bytes, n);
    sb_lit(&f, "\"}");
    return anth_send(g, "content_block_delta", &f);
}

// The Message object, in the one shape both the buffered body and the
// streamed message_start/message_delta pair are built from. `content_json`,
// when set, is the already-rendered block list a streamed turn accumulated,
// for the same reason the Responses surface hands its items over verbatim:
// the streamed and buffered documents stay identical by construction.
static void anth_body(sbuf *r, gen_ctx *g, const resp_doc *d) {
    int n_calls = d->calls ? d->calls->n : 0;
    sb_fmt(r, "{\"id\":\"%s\",\"type\":\"message\",\"role\":\"assistant\","
              "\"model\":\"", g->id);
    sb_esc(r, SV.model_name, strlen(SV.model_name));
    sb_lit(r, "\",\"content\":[");
    if (d->with_output) {
        int idx = 0;
        if (d->reason_n) {
            // reasoning is a block of its own here rather than a field on the
            // message, and it always precedes the answer
            sb_lit(r, "{\"type\":\"thinking\",\"thinking\":\"");
            sb_esc(r, d->reason, d->reason_n);
            sb_lit(r, "\",\"signature\":\"\"}");
            idx++;
        }
        if (n_calls) {
            for (int i = 0; i < n_calls; i++) {
                const char *name = call_field(d->calls, i, "name", "");
                const char *args = call_field(d->calls, i, "arguments", "{}");
                if (idx++) sb_lit(r, ",");
                sb_fmt(r, "{\"type\":\"tool_use\",\"id\":\"toolu_%d\","
                          "\"name\":\"", i);
                sb_esc(r, name, strlen(name));
                // Anthropic carries the arguments as a JSON *object*, where
                // OpenAI carries the same document as a string. That difference
                // is load-bearing: a string can hold anything, an inlined
                // object cannot. Under the strict envelope the document is
                // guaranteed to parse, but a call recovered from free text is
                // only brace-matched, so it can be balanced and still invalid —
                // inlining that verbatim would emit a body no client can read.
                // Re-dumping through the parser is what makes this total.
                sb_lit(r, "\",\"input\":");
                jv *parsed = json_parse(args, strlen(args));
                if (parsed && parsed->type == J_OBJ) jv_dump(parsed, r);
                else                                 sb_lit(r, "{}");
                jv_free(parsed);
                sb_lit(r, "}");
            }
        } else if (d->text_n || !idx) {
            // an empty text block is still emitted when it is the only thing
            // the turn produced: content[] must never be empty
            if (idx++) sb_lit(r, ",");
            sb_lit(r, "{\"type\":\"text\",\"text\":\"");
            sb_esc(r, d->text ? d->text : "", d->text_n);
            sb_lit(r, "\"}");
        }
    }
    sb_lit(r, "],\"stop_reason\":");
    if (d->stop_reason) sb_fmt(r, "\"%s\"", d->stop_reason);
    else                sb_lit(r, "null");
    sb_lit(r, ",\"stop_sequence\":");
    if (d->stop_seq) {
        sb_lit(r, "\"");
        sb_esc(r, d->stop_seq, strlen(d->stop_seq));
        sb_lit(r, "\"");
    } else {
        sb_lit(r, "null");
    }
    // Anthropic's cache_* usage fields describe *its* prompt-caching product
    // and are deliberately not claimed here; runner's prefix-cache figure is
    // reported in runner_telemetry, as it is on every other surface.
    sb_fmt(r, ",\"usage\":{\"input_tokens\":%d,\"output_tokens\":%d}",
           d->n_prompt, d->n_gen);
    if (d->with_usage) {
        sb_lit(r, ",");
        telemetry_json(r, d);
    }
    sb_lit(r, "}");
}

static int emit_channel(gen_ctx *g, int reasoning, const char *bytes, int n) {
    sb_put(reasoning ? &g->reason : &g->out, bytes, n);
    if (!g->stream || g->dead) return g->dead ? 1 : 0;
    // reasoning is its own channel and never part of the envelope document
    if (!reasoning && g->tsx_on) return tool_stream_feed(&g->tsx, bytes, n);
    return send_text_delta(g, reasoning, bytes, n);
}

// content-channel filter for user stop sequences: bytes are staged in
// g->hold so a stop spanning token boundaries still matches, and only the
// tail that could still begin a stop is withheld from the client
static int stop_feed(gen_ctx *g, const char *bytes, int n) {
    // A stop sequence is a rule about the MODEL's visible text. The tail the
    // engine synthesizes to close a truncated constrained document is not
    // that — it is the server making the client's copy legal — and filtering
    // it eats exactly the bytes that do the work: closing `{"a":"xx` yields
    // `","b":""}`, and `"}"` or `"\n\n"` are ordinary stops that match a
    // closer even when the model never produced one.
    if (g->eng && g->eng->constraint_closing)
        return emit_channel(g, 0, bytes, n);
    sb_put(&g->hold, bytes, n);
    size_t at = 0, hit_len = 0;
    for (size_t i = 0; !hit_len && i < g->hold.n; i++)
        for (int s = 0; s < g->n_stop; s++) {
            size_t len = strlen(g->stop_strs[s]);
            if (i + len <= g->hold.n &&
                memcmp(g->hold.s + i, g->stop_strs[s], len) == 0) {
                at = i;
                hit_len = len;
                g->stop_hit = g->stop_strs[s];
                break;
            }
        }
    if (hit_len) {
        if (at > 0) emit_channel(g, 0, g->hold.s, (int)at);
        // The constraint validator consumed these bytes before this sink ever
        // ran, so it is now ahead of the client's copy by everything dropped
        // here. Tell it, or the tail it synthesizes at the end of generation
        // continues a document the client does not have.
        if (g->eng) engine_constraint_truncate(g->eng, (int)(g->hold.n - at));
        g->hold.n = 0;
        g->stopped = true;
        return 1; // abort generation
    }
    size_t keep = 0;
    for (int s = 0; s < g->n_stop; s++) {
        size_t len = strlen(g->stop_strs[s]);
        size_t k = len - 1 < g->hold.n ? len - 1 : g->hold.n;
        for (; k > keep; k--)
            if (memcmp(g->hold.s + g->hold.n - k, g->stop_strs[s], k) == 0) {
                keep = k;
                break;
            }
    }
    size_t rel = g->hold.n - keep;
    int rc = 0;
    if (rel > 0) {
        rc = emit_channel(g, 0, g->hold.s, (int)rel);
        memmove(g->hold.s, g->hold.s + rel, keep);
        g->hold.n = keep;
    }
    return rc;
}

// The buffered counterpart of tool_stream_finish: one finished strict-envelope
// document, demultiplexed into the channels the client actually receives.
//
// Strict mode means the whole response IS the envelope, guaranteed by the
// schema rather than fished out of free text. A truncated call was closed to a
// legal document by sval_close, so it still PARSES — but only a cleanly
// finished document reports "tool_calls"; a truncated one keeps "length" so the
// caller knows the arguments are minimal closures, not the model's intent
// (finding A, 2026-08-11 evaluation).
//
// Consumes g->out and replaces it with whatever the client should see; appends
// any calls to tc. Returns the number of tool calls (parallel turns map several
// at once), 0 for a plain answer, or -1 when the document could not be mapped.
static int envelope_map_buffered(const tool_envelope *env, gen_ctx *g, sbuf *tc) {
    sbuf mapped = {0};
    sbuf mapped_reason = {0};
    int rc = tool_envelope_map_channels(
        env, g->out.s ? g->out.s : "", g->out.n, &mapped_reason, &mapped, tc);
    if (rc >= 0 && mapped_reason.n) {
        free(g->reason.s);
        g->reason = mapped_reason;
    } else free(mapped_reason.s);
    if (rc >= 1) {
        if (env->proto == TP_HARMONY && mapped.n) {
            // Harmony's commentary message is intentionally visible before its
            // recipient-bearing call. OpenAI-shaped responses permit content
            // and tool_calls together, and the streaming demux already emits
            // both in that order.
            free(g->out.s);
            g->out = mapped;
        } else {
            g->out.n = 0;
            free(mapped.s);
        }
    } else if (rc == 0) {
        free(g->out.s);
        g->out = mapped;         // the final branch's payload is the reply
    } else {
        // Unparseable. The old fallback here handed g->out back verbatim, on
        // the reading that a degraded answer beats none — but under a strict
        // envelope g->out is not an answer at all, it is the protocol. Harmony
        // control-token spellings reach it verbatim (engine.c forwards tok_raw
        // for decoded-empty control tokens), so clients were served
        // `<|channel|>analysis<|message|>...to=functions.unknown...` as
        // assistant content; the generic envelope leaks its own JSON syntax the
        // same way, measured as `{tool`. Neither is text anybody asked for, and
        // a client cannot tell either from a real answer.
        //
        // So the document is dropped and the turn reports the fault instead.
        // An empty answer with an honest finish reason is something a caller
        // can retry on; protocol framing dressed as content is not.
        free(mapped.s);
        free(g->out.s);
        g->out = (sbuf){0};
    }
    return rc;
}

static int gen_emit(void *ud, int reasoning, const char *bytes, int n) {
    gen_ctx *g = ud;
    if (!reasoning && g->n_stop) return stop_feed(g, bytes, n);
    return emit_channel(g, reasoning, bytes, n);
}

static int gen_collect(void *ud, const char *bytes, int n) {
    gen_ctx *g = ud;
    return think_feed(&g->ts, bytes, n, gen_emit, g);
}

// A request field the server cannot use must be an error, never a silent
// fallback to the default: the caller then gets a response generated with
// settings it did not ask for and no way to detect it. Stringified numbers
// are the common shape of the mistake — several HTTP layers produce them
// from form or env-derived config.
//
// `null` is the deliberate exception and reads as absent. Every mainstream
// OpenAI SDK serialises an unset optional field as null rather than omitting
// it, so treating null as a wrong type would 400 on ordinary traffic from an
// unmodified client.
bool absent(const jv *v) { return !v || v->type == J_NULL; }

static bool request_number(jv *req, const char *key, double dflt,
                           double min, double max, double *out) {
    jv *v = jv_get(req, key);
    if (!absent(v) && v->type != J_NUM) return false;
    double n = absent(v) ? dflt : v->num;
    if (!isfinite(n) || n < min || n > max) return false;
    *out = n;
    return true;
}

static bool whole_number(double n) {
    return n == floor(n);
}

// Largest double strictly below 2^64, i.e. 2^64 - 2048: the next representable
// double IS 2^64, and converting that to uint64_t is undefined. Naming it lets
// the assertion below guarantee the cast rather than a runtime check that can
// never fire -- the range check already rejects anything above this, so a
// second `seed >= 2^64` test downstream would be dead code. If anyone widens
// this bound, the build stops instead of the behaviour going quietly undefined.
#define SEED_MAX 18446744073709549568.0
_Static_assert(SEED_MAX < 18446744073709551616.0,
               "seed bound must stay below 2^64 so the uint64_t cast is defined");

// negative sentinels: MT_UNLIMITED clamps to the context window later,
// the other sentinels are request errors with distinct messages
enum { MT_FRACTIONAL = -5, MT_NEGATIVE = -4, MT_BAD_TYPE = -3,
       MT_NON_FINITE = -2, MT_UNLIMITED = -1 };

static int request_max_tokens(jv *req, int dflt) {
    jv *v = jv_get(req, "max_tokens");
    if (absent(v)) v = jv_get(req, "max_completion_tokens");
    // the Responses API's name for the same cap
    if (absent(v)) v = jv_get(req, "max_output_tokens");
    if (absent(v)) return dflt;
    if (v->type != J_NUM) return MT_BAD_TYPE;
    if (!isfinite(v->num)) return MT_NON_FINITE;
    if (v->num < 0) return MT_NEGATIVE;
    if (v->num > INT_MAX) return INT_MAX;
    if (!whole_number(v->num)) return MT_FRACTIONAL;
    return (int)v->num;
}

bool request_keep_alive(jv *req, bool *present, int *seconds) {
    jv *v = jv_get(req, "keep_alive");
    if (!absent(v) && v->type != J_NUM) return false;
    *present = !absent(v);
    if (!*present) return true;
    if (!isfinite(v->num) || v->num > INT_MAX || !whole_number(v->num))
        return false;
    *seconds = v->num < 0 ? -1 : (int)v->num;
    return true;
}

// SDKs routinely serialize neutral values for features this single-choice
// engine does not implement. Accept only the forms whose semantics are exactly
// a no-op; reject every value that would otherwise be silently ignored.
static const char *unsupported_completion_field(jv *req) {
    jv *v = jv_get(req, "n");
    if (!absent(v) && (v->type != J_NUM || !isfinite(v->num) || v->num != 1))
        return "n";
    v = jv_get(req, "frequency_penalty");
    if (!absent(v) && (v->type != J_NUM || !isfinite(v->num) || v->num != 0))
        return "frequency_penalty";
    v = jv_get(req, "presence_penalty");
    if (!absent(v) && (v->type != J_NUM || !isfinite(v->num) || v->num != 0))
        return "presence_penalty";
    v = jv_get(req, "logit_bias");
    if (!absent(v) && (v->type != J_OBJ || v->n != 0)) return "logit_bias";
    // `user` is advisory rather than an inference control, but recognizing it
    // still means rejecting malformed values instead of accepting any JSON.
    v = jv_get(req, "user");
    if (!absent(v) && v->type != J_STR) return "user";
    return NULL;
}

// a boolean request flag: absent takes the default, a non-boolean is an
// error rather than a silent `false`
bool request_bool(jv *req, const char *key, bool dflt, bool *out) {
    jv *v = jv_get(req, key);
    if (absent(v)) { *out = dflt; return true; }
    if (v->type != J_BOOL) return false;
    *out = v->b;
    return true;
}

// the caller's schema-constrained-output request: OpenAI response_format
// {type:"json_schema", json_schema:{schema:{...}}} or an Ollama-style
// "format" object. NULL when the request asked for no schema.
jv *request_schema(jv *req) {
    jv *rf = jv_get(req, "response_format");
    jv *sch = NULL;
    if (rf && strcmp(jv_str(jv_get(rf, "type"), ""), "json_schema") == 0) {
        jv *js = jv_get(rf, "json_schema");
        sch = js ? (jv_get(js, "schema") ? jv_get(js, "schema") : js) : NULL;
    }
    jv *fmt = jv_get(req, "format");
    if (!sch && fmt && fmt->type == J_OBJ) sch = fmt;
    // the Responses spelling of the same request. Resolved here rather than at
    // the route so the constrained-decoding path has exactly one entry point
    // regardless of which surface asked for it.
    if (!sch) {
        jv *tf = jv_get(jv_get(req, "text"), "format");
        if (tf && strcmp(jv_str(jv_get(tf, "type"), ""), "json_schema") == 0) {
            jv *inner = jv_get(tf, "schema");
            if (inner && inner->type == J_OBJ) sch = inner;
        }
    }
    return sch;
}

// did the caller ask for free-form JSON (rather than a schema)? Both dialects.
static bool request_json_mode(jv *req) {
    jv *rf = jv_get(req, "response_format");
    if (rf && strcmp(jv_str(jv_get(rf, "type"), ""), "json_object") == 0)
        return true;
    jv *tf = jv_get(jv_get(req, "text"), "format");
    return tf && strcmp(jv_str(jv_get(tf, "type"), ""), "json_object") == 0;
}

// run one completion on a slot and write the HTTP response.
// `env` is the strict tool-call envelope when the request opted into one; it
// replaces the response_format schema, having already absorbed it as the
// shape of its `final` branch.
// Drop the device turn and immediately ask for it back. A waiter that has been
// blocked gets the mutex ahead of us on any fair-ish implementation, and on an
// unfair one this still bounds the hold to one chunk rather than one prompt.
static void prefill_yield_turn(void *ud) {
    (void)ud;
    // end() then begin() puts this prefill at the BACK of the device queue,
    // so a request that arrived mid-prefill runs at the next chunk boundary
    // instead of after the whole prompt. That only works because the turn is
    // FIFO (see sched_prefill_begin): with a plain mutex the releasing thread
    // barges straight back in -- measured, the waiter lost 44 of 45 races and
    // still waited 25.3 s of a 26.4 s prefill.
    sched_prefill_end();
    sched_prefill_begin();
}

void run_completion(slot_t *s, sock_t fd, const char *prompt, int api,
                           jv *req, const tool_envelope *env) {
    bool chat = api != API_TEXT; // chat-shaped: thinking channels, tools
    model_t *m = s->m;
    engine *e = &s->e;
    bool muse_forced_think = chat && s->tmpl == TMPL_MUSE &&
                             req_thinking_mode(req) == THINK_ON;
    // Harmony primes the analysis channel in the prompt for every mode except
    // THINK_OFF (see the generation-prompt comment in template.c), so the
    // stream starts INSIDE reasoning and the splitter must too — the same
    // reason ornith always does and muse does when forced. Getting this wrong
    // is silent: the split simply never fires and the model's analysis is
    // served as content, which is what the first live run did.
    bool harmony_primed_think = chat && s->tmpl == TMPL_HARMONY &&
                                !(env && env->proto == TP_HARMONY) &&
                                req_thinking_mode(req) != THINK_OFF;
    // gemma4's thought block is opened BY THE PROMPT on a tool-result
    // continuation with thinking on (template.c's g4_prev == 2 branch), so the
    // grammar must start inside it. Asked of the prompt itself rather than
    // re-derived from the message list: the renderer owns that decision, and a
    // second copy of the condition is a second thing to keep in step.
    size_t prompt_n = prompt ? strlen(prompt) : 0;
    static const char G4_THOUGHT_OPEN[] = "<|channel>thought\n";
    bool gemma4_primed_think =
        chat && is_gemma4(s->tmpl) && prompt_n >= sizeof(G4_THOUGHT_OPEN) - 1 &&
        !memcmp(prompt + prompt_n - (sizeof(G4_THOUGHT_OPEN) - 1),
                G4_THOUGHT_OPEN, sizeof(G4_THOUGHT_OPEN) - 1);
    // Major faults are page-ins that went to disk, so the delta across a
    // request IS the paging stall rather than an inference from wall-clock: a
    // slow request that took no major faults was slow for some other reason.
    // Without this, a machine whose weights have been evicted reports a
    // cheerful tok/s while a five-token reply takes a minute.
    uint64_t faults_at_start = plat_major_faults();
    // Allocated here rather than beside the response struct, so the start line
    // below and the completion line share one name.
    char req_id[48];
    snprintf(req_id, sizeof(req_id), "%s%d",
             api == API_RESPONSES ? "resp_" : api == API_MESSAGES ? "msg_"
                                            : api == API_CHAT ? "chatcmpl-"
                                                              : "cmpl-",
             atomic_fetch_add(&SV.req_counter, 1));

    const char *unsupported = unsupported_completion_field(req);
    if (unsupported) {
        char msg[128];
        snprintf(msg, sizeof(msg), "%s has unsupported semantics", unsupported);
        send_error(fd, 400, msg);
        return;
    }

    // Per-request deadline. A wall-clock bound is what a batching server owes
    // its clients that a serial one does not: your latency now depends on who
    // else is on the box, so there has to be a point past which you get your
    // (still valid, still schema-conforming) answer regardless. Expiry is a
    // truncation, not an error — it ends the generation and lets the normal
    // close path run, so finish_reason is "length" and constrained output is
    // closed to a legal document exactly as a token-ceiling hit would be.
    double req_timeout = SV.req_timeout;
    double rt;
    if (!request_number(req, "timeout", req_timeout, 0, 86400.0, &rt)) {
        send_error(fd, 400, "timeout out of range");
        return;
    }
    req_timeout = rt;
    double req_deadline = req_timeout > 0 ? now_s() + req_timeout : 0;

    // per-request sampling params start from the server defaults every time —
    // one request's overrides must not leak into the next on this slot; only
    // the rng STATE carries across so sampling sequences stay diverse
    double temp, top_p, min_p, top_k, seed, repeat_penalty;
    if (!request_number(req, "temperature", s->smp_base.temp, 0, FLT_MAX, &temp) ||
        !request_number(req, "top_p", s->smp_base.top_p, 0, 1, &top_p) ||
        !request_number(req, "min_p", s->smp_base.min_p, 0, 1, &min_p) ||
        !request_number(req, "top_k", s->smp_base.top_k, 0, INT_MAX, &top_k) ||
        // the model's family preset decides the default; a client that wants
        // no penalty at all asks for 1. Zero is rejected rather than treated
        // as "off": the penalty divides by it.
        !request_number(req, "repeat_penalty", s->smp_base.repeat_penalty,
                        FLT_MIN, FLT_MAX, &repeat_penalty) ||
        !request_number(req, "seed", 0, 0, SEED_MAX, &seed)) {
        send_error(fd, 400, "numeric sampling parameter out of range");
        return;
    }
    // top_k and seed are held as integer engine state, so accepting 2.5 and
    // truncating it to 2 would run the request with settings the caller did
    // not ask for.
    if (!whole_number(top_k) || !whole_number(seed)) {
        // Not "out of range": 2.5 is inside every bound top_k has, and saying
        // otherwise sends a caller hunting for a limit they never exceeded.
        send_error(fd, 400, "top_k and seed must be whole numbers");
        return;
    }
    uint64_t rng_state = s->smp.rng;
    s->smp = s->smp_base;
    s->smp.rng = rng_state;
    s->smp.temp = (float)temp;
    s->smp.top_p = (float)top_p;
    s->smp.min_p = (float)min_p;
    s->smp.top_k = (int)top_k;
    s->smp.repeat_penalty = (float)repeat_penalty;
    if (seed > 0) s->smp.rng = (uint64_t)seed;
    int max_tokens = request_max_tokens(req, SV.n_predict_cap);
    if (max_tokens == MT_NON_FINITE) {
        send_error(fd, 400, "max_tokens out of range");
        return;
    }
    if (max_tokens == MT_BAD_TYPE) {
        send_error(fd, 400, "max_tokens must be a number");
        return;
    }
    // Distinct from MT_BAD_TYPE on purpose: 1.5 *is* a number, so telling a
    // caller it is not sends them looking for a type bug they do not have.
    // Name the actual rule.
    if (max_tokens == MT_FRACTIONAL) {
        send_error(fd, 400, "max_tokens must be a whole number");
        return;
    }
    if (max_tokens == MT_NEGATIVE) {
        send_error(fd, 400, "max_tokens must be non-negative");
        return;
    }
    // "stream":"true" used to read as false and answer with a buffered
    // body, leaving a client that expected SSE waiting on events that
    // would never arrive
    bool stream = false;
    if (!request_bool(req, "stream", false, &stream)) {
        send_error(fd, 400, "stream must be a boolean");
        return;
    }
    // OpenAI logprobs. Chat uses a boolean plus top_logprobs; legacy text
    // completions use logprobs itself as the requested alternative count.
    // The engine stores one request-sized diagnostic table. Buffered replies
    // serialize it with the choice; streams send it in one diagnostic chunk
    // immediately before the terminal finish chunk.
    bool lp_on = false;
    if (api == API_CHAT && !request_bool(req, "logprobs", false, &lp_on)) {
        send_error(fd, 400, "logprobs must be a boolean");
        return;
    }
    double lp_num = 0;
    if (api == API_TEXT &&
        (!request_number(req, "logprobs", 0, 0, 20, &lp_num) ||
         !whole_number(lp_num))) {
        send_error(fd, 400, "logprobs out of range");
        return;
    }
    bool want_lp = (api == API_CHAT && lp_on) ||
                   (api == API_TEXT && lp_num > 0);
    if (api == API_CHAT && want_lp &&
        (!request_number(req, "top_logprobs", 0, 0, 20, &lp_num) ||
         !whole_number(lp_num))) {
        send_error(fd, 400, "top_logprobs out of range");
        return;
    }
    int lp_n = (int)lp_num;
    if (lp_n < 0) lp_n = 0;
    if (lp_n > 20) lp_n = 20;
    // JC-R1 constrained-choice posteriors (suite judgment-coprocessor plan):
    // buffered-only like logprobs, and only meaningful when a schema/JSON
    // constraint gives the sampler a legal set to choose from — both are
    // validated below once the request's constraint is known.
    bool cl_on = false;
    if (!request_bool(req, "choice_logprobs", false, &cl_on)) {
        send_error(fd, 400, "choice_logprobs must be a boolean");
        return;
    }
    double cl_probe_d = 32;
    if (cl_on && (!request_number(req, "choice_logprobs_probe", 32, 8, 64,
                                  &cl_probe_d) ||
                  !whole_number(cl_probe_d))) {
        send_error(fd, 400, "choice_logprobs_probe out of range (8..64)");
        return;
    }
    if (cl_on && stream) {
        send_error(fd, 400, "choice_logprobs is buffered-only; "
                            "set stream to false");
        return;
    }
    // OpenAI "stop": a string or an array of up to 4 non-empty strings.
    // Pointers borrow from req, which outlives the whole request.
    const char *stops[4];
    int n_stops = 0;
    jv *stopv = jv_get(req, "stop");
    // "stop_sequences" is the Anthropic spelling of the same field. Resolved
    // here, as max_tokens already resolves its three names, so the stop filter
    // has one implementation regardless of which surface asked for it.
    if (absent(stopv)) stopv = jv_get(req, "stop_sequences");
    if (stopv && stopv->type != J_NULL) {
        bool bad = false;
        if (stopv->type == J_STR) {
            bad = stopv->str[0] == 0;
            if (!bad) stops[n_stops++] = stopv->str;
        } else if (stopv->type == J_ARR && stopv->n <= 4) {
            for (int i = 0; i < stopv->n && !bad; i++) {
                jv *it = stopv->items[i];
                if (!it || it->type != J_STR || it->str[0] == 0) bad = true;
                else stops[n_stops++] = it->str;
            }
        } else {
            bad = true;
        }
        if (bad) {
            send_error(fd, 400,
                       "stop must be a string or an array of up to 4 non-empty strings");
            return;
        }
    }
    // A stop sequence is a rule about the model's VISIBLE text — that is the
    // only thing the caller can see to write a rule about, and it is why the
    // filter already excludes the reasoning channel. Under the strict tool
    // envelope the generated document is not text at all: it is protocol
    // (Harmony channel markers and recipient headers, Muse's <atem:invoke>,
    // gemma4's <|tool_call> blocks, or the generic envelope's own JSON
    // syntax), and the client receives only the
    // demultiplexed result. Matching their strings against the raw document
    // fires on framing they never wrote and never see — ["\n\n"], ["}"] and
    // ["<|"] are ordinary agent defaults and every one of them hits.
    //
    // It also corrupts the turn rather than merely cutting it short. stop_feed
    // drops the matched bytes from the emitted document, but the constraint
    // validator consumed them before the sink ever ran, so the completion
    // constraint_close synthesizes continues a state the client's copy never
    // reached. Measured: the generic envelope with stop `"` returns
    // `"content":"{tool"` — envelope syntax served as assistant text; the
    // Harmony envelope fails to map at all.
    //
    // Honouring the request properly is not available on this path. The
    // Harmony demultiplexer cannot tell visible text from framing until the
    // document is complete, and by then there is nothing left to stop. So the
    // choice is between refusing and ignoring, and a field this server cannot
    // use is an error rather than a silent downgrade — the same call
    // parallel_tool_calls makes in server.c, for the same reason: a caller who
    // is quietly ignored has no way to detect it.
    if (n_stops > 0 && env) {
        send_error(fd, 400,
                   "stop is not supported with tool calling: the sequences "
                   "would be matched against the tool-call protocol the model "
                   "generates, not the text you receive");
        return;
    }
    jv *rf = jv_get(req, "response_format");
    if (rf) {
        // An unrecognised or malformed response_format used to fall through to
        // unconstrained decoding while still answering 200, so a caller asking
        // for guaranteed structure silently got none.
        if (rf->type != J_OBJ) {
            send_error(fd, 400, "response_format must be an object");
            return;
        }
        const char *rft = jv_str(jv_get(rf, "type"), "");
        if (strcmp(rft, "json_object") != 0 && strcmp(rft, "json_schema") != 0 &&
            strcmp(rft, "text") != 0) {
            send_error(fd, 400,
                       "response_format.type must be text, json_object or json_schema");
            return;
        }
        if (strcmp(rft, "json_schema") == 0) {
            jv *js = jv_get(rf, "json_schema");
            if (!js || js->type != J_OBJ) {
                send_error(fd, 400,
                           "response_format.json_schema must be an object");
                return;
            }
            jv *inner = jv_get(js, "schema");
            if (inner && inner->type != J_OBJ) {
                send_error(fd, 400,
                           "response_format.json_schema.schema must be an object");
                return;
            }
        }
    }
    e->json_mode = request_json_mode(req);
    // Constrained decoding. The tool envelope wins when present: it already
    // contains the caller's response_format schema as its `final` branch, so
    // compiling that separately would drop the tool branches.
    snode *schema = NULL;
    jv *sch = env ? NULL : request_schema(req);
    if (env) {
        char serr[128] = "envelope did not parse";
        if (env->proto == TP_HARMONY) {
            const char *only = env->kind == TCH_NAMED ? env->named : NULL;
            schema = schema_compile_harmony_turn(
                env->tools, env->kind == TCH_AUTO, only,
                env->kind == TCH_AUTO ? request_schema(req) : NULL,
                req_thinking_mode(req) != THINK_OFF, serr, sizeof(serr));
        } else if (env->proto == TP_GEMMA4) {
            const char *only = env->kind == TCH_NAMED ? env->named : NULL;
            schema = env->parallel && env->kind != TCH_AUTO
                       ? schema_compile_gemma4_parallel(env->tools, only,
                                                        serr, sizeof(serr))
                       : schema_compile_gemma4_turn(
                             env->tools, env->kind == TCH_AUTO, only,
                             env->kind == TCH_AUTO ? request_schema(req) : NULL,
                             req_thinking_mode(req) != THINK_OFF,
                             gemma4_primed_think, serr, sizeof(serr));
        } else if (env->proto == TP_ATEM) {
            const char *only = env->kind == TCH_NAMED ? env->named : NULL;
            jv *final = env->kind == TCH_AUTO ? request_schema(req) : NULL;
            schema = final
                       ? schema_compile_atem_turn(
                             env->tools, true, only, final, ATEM_TURN_EITHER,
                             serr, sizeof(serr))
                       : muse_forced_think
                       ? schema_compile_atem_turn(
                             env->tools, env->kind == TCH_AUTO, only, NULL,
                             ATEM_TURN_AFTER_REASONING,
                             serr, sizeof(serr))
                       : env->parallel && env->kind != TCH_AUTO
                       ? schema_compile_atem_parallel(env->tools, only,
                                                     serr, sizeof(serr))
                       : schema_compile_atem_turn(
                             env->tools, env->kind == TCH_AUTO, only, NULL,
                             ATEM_TURN_EITHER,
                             serr, sizeof(serr));
        } else {
            jv *ej = json_parse(env->schema_src, strlen(env->schema_src));
            schema = ej ? (env->proto == TP_MUSE_USER
                           ? schema_compile_muse_user_payload(ej, serr,
                                                              sizeof(serr))
                           : schema_compile(ej, serr, sizeof(serr))) : NULL;
            jv_free(ej);
        }
        if (!schema) {
            char msg[256];
            snprintf(msg, sizeof(msg), "unsupported tool schema: %s", serr);
            send_error(fd, 400, msg);
            return;
        }
    } else if (sch) {
        char serr[128];
        schema = chat && s->tmpl == TMPL_MUSE
                   ? schema_compile_muse_user_payload(sch, serr, sizeof(serr))
                   : schema_compile(sch, serr, sizeof(serr));
        if (!schema) {
            char msg[192];
            snprintf(msg, sizeof(msg), "unsupported json schema: %s", serr);
            send_error(fd, 400, msg);
            return;
        }
    }
    e->schema = schema;
    e->constraint_includes_prelude = env && env->proto == TP_HARMONY;
    // chat responses split a thinking prelude into the reasoning channel;
    // constrained generation forwards it only when asked (raw completions
    // keep the payload-only contract)
    e->emit_think_prelude = chat && m->think_open != NULL &&
                            !(env && env->proto == TP_HARMONY);

    size_t cap = strlen(prompt) + 16;
    int32_t *toks = malloc(sizeof(int32_t) * cap);
    // -1 from tok_encode is an allocation failure mid-encode (a dropped
    // segment); a NULL toks is the same class. Both are 500s, never a prompt
    // silently short by the missing piece.
    int n_prompt = toks ? tok_encode(s->tok, prompt, toks, (int)cap, true, true) : -1;
    if (n_prompt < 0) {
        free(toks);
        completion_cleanup(e, schema, NULL);
        send_error(fd, 500, "out of memory tokenizing prompt");
        return;
    }
    if (n_prompt == 0 || n_prompt >= m->n_ctx) {
        free(toks);
        completion_cleanup(e, schema, NULL);
        if (n_prompt == 0) send_error(fd, 400, "empty prompt");
        else send_error_detail(fd, 400, "prompt exceeds context window",
                               api == API_TEXT ? "prompt" :
                               api == API_RESPONSES ? "input" : "messages",
                               "context_length_exceeded");
        return;
    }
    int remaining_ctx = m->n_ctx - n_prompt;
    if (max_tokens < 0 || max_tokens > remaining_ctx) max_tokens = remaining_ctx;

    if (cl_on) {
        if (!schema && !e->json_mode) {
            free(toks);
            completion_cleanup(e, schema, NULL);
            send_error(fd, 400, "choice_logprobs requires a json_schema "
                                "response_format, tool schema, or JSON mode");
            return;
        }
        if (e->dm) {
            free(toks);
            completion_cleanup(e, schema, NULL);
            send_error(fd, 400, "choice_logprobs is not supported with "
                                "speculative decoding");
            return;
        }
        if (max_tokens > 0) {
            e->cl_recs  = malloc(sizeof(cl_rec) * (size_t)max_tokens);
            if (!e->cl_recs) {
                free(toks);
                completion_cleanup(e, schema, NULL);
                send_error(fd, 500,
                           "out of memory allocating choice_logprobs buffer");
                return;
            }
            e->cl_cap   = max_tokens;
            e->cl_count = 0;
            e->cl_probe = (int)cl_probe_d;
        }
    }
    if (want_lp && max_tokens > 0) {
        e->lp_cap    = max_tokens;
        e->lp_n      = lp_n;
        e->lp_chosen = malloc(sizeof(float) * max_tokens);
        e->lp_ids    = malloc(sizeof(int32_t) * max_tokens);
        e->lp_top    = lp_n ? malloc(sizeof(lp_alt) * (size_t)max_tokens * lp_n) : NULL;
        // These scale with max_tokens * top_logprobs (tens of MB): a NULL here
        // would be dereferenced while collecting logprobs. Fail the request
        // cleanly instead. completion_cleanup frees whatever did allocate.
        if (!e->lp_chosen || !e->lp_ids || (lp_n && !e->lp_top)) {
            free(toks);
            completion_cleanup(e, schema, NULL);
            send_error(fd, 500, "out of memory allocating logprobs buffers");
            return;
        }
    }

    // Reuse the KV for whatever prefix this prompt shares with one already
    // computed — pipeline callers repeat long system/template prefixes
    // verbatim. Two tiers. The slot's own KV is free — it is already in
    // the cache this sequence decodes against. The shared snapshot store is
    // not free (a memcpy, and on CUDA a device resync), but it reaches across
    // slots, across requests and across model swaps, so it is what turns a
    // repeated system/tool/schema block into work nobody does twice.
    bool cache_prompt = jv_bool(jv_get(req, "cache_prompt"), true);
    // Opt out of the *shared* tier only: a caller isolating a pathological
    // prompt still wants its own slot's cache, and one that wants fresh-prompt
    // telemetry wants neither. cache_prompt:false remains the full opt-out.
    bool share_prefix = jv_bool(jv_get(req, "prefix_cache"), true);
    prefix_reuse reuse = { 0, 0, 0.0 };
    // Prefill is scheduled apart from decode: different kernel shape, and it
    // is this slot's own model_forward_batch, so it must not overlap a
    // microbatch containing this sequence. It cannot — the sequence only joins
    // a batch below, after prefill has returned. Forking a prefix is prefill
    // work (it touches this slot's KV, and on CUDA it issues a forward), so it
    // belongs inside the same device turn.
    // Yield the device turn between prefill chunks. Without it a long prompt
    // holds it for the whole prefill and every other slot waits: measured at
    // 26.2 s for a short request arriving during a 2,300-token prefill.
    engine_set_prefill_yield(e, prefill_yield_turn, NULL);
    sched_prefill_begin();
    if (cache_prompt && share_prefix)
        reuse = engine_prefix_reuse(e, toks, n_prompt);
    else if (cache_prompt)
        reuse.keep = engine_rewind(e, toks, n_prompt);
    else
        engine_reset(e);
    int keep = reuse.keep;
    double prefill_t0 = now_s();
    // Name the request BEFORE any work, and say it started.
    //
    // The completion line below is printed on completion only, so a request
    // that never finishes leaves no trace at all — reported from a Mac where a
    // 1,200-token prompt returned nothing in 300 s and the server log was
    // empty for the whole five minutes. A start line costs one stderr write
    // per request and makes that case visible; it also carries the prompt
    // length and how much of it the cache covered, which is what says whether
    // a long silence is a cold prefill or something worse.
    fprintf(stderr, "[slot %d] %s: start, %d prompt (%d cached)\n",
            s->id, req_id, n_prompt, keep);

    float *logits = engine_feed(e, toks + keep, n_prompt - keep);
    double prefill_s = now_s() - prefill_t0;
    sched_prefill_end();
    engine_set_prefill_yield(e, NULL, NULL);
    // Publish outside the device turn: it is a host-memory copy, and this
    // sequence is the only writer of its own KV between prefill and decode.
    if (logits && cache_prompt && share_prefix)
        engine_prefix_publish(e, toks, n_prompt, n_prompt - keep, prefill_s);
    free(toks);
    if (!logits) {
        completion_cleanup(e, schema, NULL);
        send_error_detail(fd, 400, "context overflow",
                          api == API_TEXT ? "prompt" :
                          api == API_RESPONSES ? "input" : "messages",
                          "context_length_exceeded");
        return;
    }
    if ((chat && s->tmpl == TMPL_ORNITH) || muse_forced_think ||
        harmony_primed_think)
        engine_think_started(e);

    gen_ctx g = { .out = {0}, .fd = fd, .stream = stream, .api = api,
                  .stop_strs = stops, .n_stop = n_stops, .eng = e,
                  .created = (long)time(NULL) };
    tool_envelope muse_plain_env = {.proto = TP_MUSE_PLAIN};
    snprintf(g.id, sizeof(g.id), "%s", req_id);
    // split thinking channels out of chat responses; raw completions stay raw
    if (env && (env->proto == TP_HARMONY || env->proto == TP_GEMMA4))
        think_init(&g.ts, NULL, NULL);
    else if ((chat && s->tmpl == TMPL_ORNITH) || muse_forced_think ||
        harmony_primed_think)
        think_init_reasoning(&g.ts, m->think_open, m->think_close);
    else
        think_init(&g.ts, chat ? m->think_open : NULL, m->think_close);
    const tool_envelope *stream_env = env ? env
        : stream && chat && s->tmpl == TMPL_MUSE && schema
        ? &muse_plain_env : NULL;
    if (stream && stream_env) {
        tool_stream_sink sink = { &g, sink_reasoning, sink_content, sink_call_begin,
                                  sink_call_args, sink_call_end };
        tool_stream_init(&g.tsx, stream_env, &sink);
        g.tsx_on = true;
    }

    if (stream) {
        const char *hdr = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                          "Cache-Control: no-cache\r\nConnection: close\r\n\r\n";
        if (!send_all(fd, hdr, strlen(hdr))) {
            completion_cleanup(e, schema, &g);
            return;
        }
        // OpenAI opens a chat stream with a role-only delta. Emitting it up
        // front rather than folding the role into whatever the first text
        // delta happens to be keeps the contract independent of what the
        // model generates — including generating nothing at all.
        if (api == API_CHAT) {
            sbuf c = {0};
            chunk_open(&g, &c);
            sb_lit(&c, "\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}");
            g.role_sent = true;
            chunk_send(&g, &c);
        } else if (api == API_RESPONSES) {
            // A Responses stream opens with the response object twice: created,
            // then in_progress. Both carry an empty output — the items are
            // announced as they start — so they are emitted before any token
            // exists, which is exactly what makes a client able to render the
            // turn's identity immediately.
            resp_doc d = { .status = "in_progress", .req = req };
            for (int i = 0; i < 2 && !g.dead; i++) {
                sbuf f = {0};
                sb_lit(&f, ",\"response\":");
                responses_body(&f, &g, &d);
                resp_send(&g, i == 0 ? "response.created"
                                     : "response.in_progress", &f);
            }
        } else if (api == API_MESSAGES) {
            // An Anthropic stream opens with the whole Message object minus
            // its content: the id, model and input token count are known
            // before a token exists, and a client renders the turn's identity
            // from them immediately. stop_reason is null until message_delta.
            resp_doc d = { .n_prompt = n_prompt, .req = req };
            sbuf f = {0};
            sb_lit(&f, ",\"message\":");
            anth_body(&f, &g, &d);
            anth_send(&g, "message_start", &f);
        }
    }

    // A configured draft is not proof that this request speculated. Logprob
    // and choice-logprob capture deliberately select the solo walk, so record
    // the execution decision while those request flags are still installed.
    bool spec_used = engine_wants_spec(e);
    double gtime;
    int n_gen = sched_generate(s, logits, max_tokens, gen_collect, &g, &gtime,
                               req_deadline);
    // The one place every surface's generation passes through, so /health's
    // cumulative counters see chat, completions, responses and messages alike.
    server_record_work(n_prompt, n_gen, gtime);
    think_finish(&g.ts, gen_emit, &g);
    // generation ended without a stop match: the withheld partial-match
    // tail was ordinary output after all
    if (!g.stopped && g.hold.n > 0) {
        emit_channel(&g, 0, g.hold.s, (int)g.hold.n);
        g.hold.n = 0;
    }
    // A native-envelope document the demultiplexer cannot map is a GENERATION
    // fault, not a dead client — and the difference decides whether the stream
    // is terminated. Recording it as g.dead skipped all three terminal blocks
    // below, so the client got the opening events and then nothing: no deltas,
    // no finish_reason, no `data: [DONE]`, a bare FIN, and a server log that
    // blamed the client for leaving. It is still there, still reading, and owed
    // a terminal chunk it can stop on.
    //
    // tool_stream_finish returns -1 for exactly that failure; any other
    // non-zero value came from a sink refusing to write, which IS the client
    // going away.
    bool unmapped = false;
    if (g.tsx_on) {
        int fin = tool_stream_finish(&g.tsx);
        if (fin == -1)     unmapped = true;
        else if (fin != 0) g.dead = true;
    }
    // Anything still held for a multi-byte character the model never finished
    // goes out now rather than disappearing. It is genuinely truncated, so it
    // renders as U+FFFD — but silently dropping bytes would make the streamed
    // turn shorter than the buffered one, which is the invariant that matters.
    flush_text_delta(&g, 0);
    flush_text_delta(&g, 1);
    // e->oom: generation hit an allocation failure — report it truthfully as
    // "error", never let it masquerade as a clean "stop". An unmappable
    // envelope is the same class of truth: the turn produced no answer anyone
    // can read, and reporting "stop" over an empty message would tell the
    // caller the model chose to say nothing.
    const char *finish = e->oom ? "error"
                       : unmapped ? "envelope_error"
                       : e->prelude_exhausted ? "reasoning_limit"
                       : g.stopped || e->hit_stop ? "stop" : "length";
    // a streamed call reports the same terminal reason a buffered one does.
    // Only a CLEANLY FINISHED document may claim "tool_calls": a budget
    // truncation still parses (the closer guarantees that) but the envelope
    // must keep saying "length", or the caller executes a half-generated
    // call as if it were complete — finding A of the 2026-08-11 external
    // evaluation, which found the Responses incomplete path dead whenever a
    // tool call was present.
    if (g.tsx_on && tool_stream_called(&g.tsx) && !strcmp(finish, "stop"))
        finish = "tool_calls";

    if (stream && api == API_RESPONSES) {
        // whatever item was still streaming is closed first: an item announced
        // with output_item.added must always reach output_item.done, including
        // when generation stopped mid-item
        bool cut = strcmp(finish, "length") == 0 ||
                   strcmp(finish, "reasoning_limit") == 0;
        // The turn produced a document nothing could read, so it did not
        // complete — and the reason is neither of the two standard ones.
        // "max_output_tokens" would be the closest lie, and a lie a client can
        // act on: it retries with a larger budget forever. An unrecognised
        // reason string is inert by comparison, and true.
        bool failed = strcmp(finish, "envelope_error") == 0;
        // a tool call that was truncated is still an executable call — the
        // envelope schema closed it to a legal document — so only a message
        // is reported as unfinished
        if ((cut || failed) && resp_shape_of(g.item_kind) == &RESP_MESSAGE)
            g.close_status = "incomplete";
        resp_close_item(&g);
        if (!g.dead) {
            bool truncated = cut || failed;
            resp_doc d = { .status = truncated ? "incomplete" : "completed",
                           .incomplete = failed ? "envelope_unmapped"
                                       : cut   ? "max_output_tokens" : NULL,
                           .output_json = g.out_items.s ? g.out_items.s : "",
                           .output_n = g.out_items.n,
                           .output_text = g.out_text.s,
                           .output_text_n = g.out_text.n,
                           .with_usage = true,
                           .n_prompt = n_prompt, .n_gen = n_gen, .cached = keep,
                           .forked = reuse.forked, .saved_s = reuse.saved_s,
                           .gtime = gtime, .major_faults = plat_major_faults() - faults_at_start,
                           .schema = schema != NULL,
                           .json_mode = e->json_mode, .spec = spec_used,
                           .req = req };
            sbuf f = {0};
            sb_lit(&f, ",\"response\":");
            responses_body(&f, &g, &d);
            resp_send(&g, truncated ? "response.incomplete"
                                    : "response.completed", &f);
        }
    } else if (stream && api == API_MESSAGES && generation_faulted(finish)) {
        // The 200 and the headers went out before a token existed, so there is
        // no status code left to correct — and Anthropic's SSE protocol has a
        // terminator for exactly this case, documented alongside
        // overloaded_error. The stream ends on `error`.
        //
        // Deliberately NOT message_delta + message_stop: message_delta must
        // carry one of the seven stop_reasons, all of which describe a turn
        // that completed, and message_stop asserts a Message that finished.
        // Sending either would be the buffered end_turn lie in streamed form.
        // Terminating at all is not optional though — a stream left hanging is
        // the failure fixed on the OpenAI surfaces in 6c4dfc2, and this is the
        // Anthropic spelling of the same terminal chunk.
        //
        // Nothing synthesises a closing content_block_stop for a block the
        // fault interrupted. An error may arrive at any point in this protocol
        // by construction, an SDK accumulator discards the message on it, and
        // closing a block would be the one gesture that suggested the content
        // before it was a usable partial answer.
        if (!g.dead) {
            sbuf f = {0};
            sb_lit(&f, ",");
            anth_error_json(&f, finish);
            anth_send(&g, "error", &f);
        }
    } else if (stream && api == API_MESSAGES) {
        // A turn that generated nothing at all still has to describe itself
        // the way the buffered body would, which always carries at least one
        // block. Opening an empty text block here is what keeps "the streamed
        // and buffered turns agree" true even in the degenerate case.
        if (!g.item_open && g.output_index == 0) anth_open_block(&g, "text");
        // whatever block was still streaming is closed first: a block
        // announced with content_block_start must always reach
        // content_block_stop, including when generation stopped mid-block
        anth_close_block(&g);
        if (!g.dead) {
            sbuf f = {0};
            sb_fmt(&f, ",\"delta\":{\"stop_reason\":\"%s\",\"stop_sequence\":",
                   anth_stop_reason(finish, g.stop_hit != NULL));
            if (g.stop_hit) {
                sb_lit(&f, "\"");
                sb_esc(&f, g.stop_hit, strlen(g.stop_hit));
                sb_lit(&f, "\"");
            } else {
                sb_lit(&f, "null");
            }
            // message_delta carries the *cumulative* output token count, which
            // is the only place a streamed turn reports it
            sb_fmt(&f, "},\"usage\":{\"input_tokens\":%d,\"output_tokens\":%d}",
                   n_prompt, n_gen);
            if (!anth_send(&g, "message_delta", &f)) {
                sbuf s2 = {0};
                anth_send(&g, "message_stop", &s2);
            }
        }
    } else if (stream) {
        if (!g.dead) {
            if (chat && e->lp_count > 0) {
                sbuf lp = {0};
                chunk_open(&g, &lp);
                sb_lit(&lp, "\"delta\":{},");
                append_chat_logprobs(&lp, s, e);
                sb_lit(&lp, ",\"finish_reason\":null}]}");
                chunk_send(&g, &lp);
            } else if (!chat && e->lp_count > 0) {
                sbuf lp = {0};
                chunk_open(&g, &lp);
                sb_lit(&lp, "\"text\":\"\",");
                append_text_logprobs(&lp, s, e);
                sb_lit(&lp, ",\"finish_reason\":null}]}");
                chunk_send(&g, &lp);
            }
            sbuf c = {0};
            chunk_open(&g, &c);
            sb_fmt(&c, "%s,\"finish_reason\":\"%s\"}]",
                   chat ? "\"delta\":{}" : "\"text\":\"\"",
                   openai_finish(finish));
            // A streamed turn carries no runner_telemetry of its own, so
            // without this the reason widened away by openai_finish() would be
            // recoverable on buffered turns and nowhere at all on streamed
            // ones. Emitted only when there IS a distinction to keep, so an
            // ordinary stream stays byte-for-byte what it was.
            const char *sdet = finish_detail_of(finish);
            if (sdet)
                sb_fmt(&c, ",\"runner_telemetry\":{\"finish_detail\":\"%s\"}",
                       sdet);
            sb_lit(&c, "}");
            bool ok = chunk_send(&g, &c) == 0;
            // OpenAI stream_options {"include_usage": true}: one extra chunk
            // with empty choices and the request's token accounting — AI-SDK
            // clients (Cline et al.) request this on every stream
            if (ok && jv_bool(jv_get(jv_get(req, "stream_options"),
                                     "include_usage"), false)) {
                sbuf u = {0};
                sb_fmt(&u, "{\"id\":\"%s\",\"object\":\"%s\",\"created\":%ld,"
                           "\"model\":\"", g.id,
                       chat ? "chat.completion.chunk" : "text_completion",
                       g.created);
                sb_esc(&u, SV.model_name, strlen(SV.model_name));
                sb_fmt(&u, "\",\"choices\":[],"
                           "\"usage\":{\"prompt_tokens\":%d,"
                           "\"completion_tokens\":%d,\"total_tokens\":%d}}",
                       n_prompt, n_gen, n_prompt + n_gen);
                ok = chunk_send(&g, &u) == 0;
            }
            if (ok) send_all(fd, "data: [DONE]\n\n", 14);
        }
    } else {
        sbuf tc = {0};
        int n_tc = 0;
        if (env) {
            n_tc = envelope_map_buffered(env, &g, &tc);
            if (n_tc >= 1 && !strcmp(finish, "stop")) finish = "tool_calls";
            // The document was dropped, so the turn has no answer to report.
            // Saying "stop" over the resulting empty content would claim the
            // model chose to say nothing; the streamed path names the same
            // fault the same way.
            if (n_tc < 0) { n_tc = 0; finish = "envelope_error"; }
        } else if (chat) {
            if (s->tmpl == TMPL_MUSE && schema)
                muse_user_payload_strip(&g.out);
            n_tc = tool_calls_parse_for(s->tmpl, &g.out, &tc);
            if (n_tc) {
                if (!strcmp(finish, "stop")) finish = "tool_calls";
                g.out.n = 0; // OpenAI convention: no content alongside
                             // tool_calls (whatever followed was the model
                             // faking a result)
            }
        }
        if (api == API_MESSAGES) {
            // A fault is an error object here, not a Message. Anthropic's
            // seven stop_reason values all describe a turn that COMPLETED, so
            // there is no member to report this with — end_turn, which this
            // used to send, told the caller the model finished normally over
            // an empty content[].
            //
            // The partial content question, answered deliberately: a turn that
            // faulted midway may already have produced text, and the analogous
            // truncation case (stop_reason "max_tokens" WITH the partial in
            // the body) is not available to borrow. Truncation is a completion
            // with a known boundary — the budget ran out, and every byte before
            // it is the model's finished intent up to that point. A mapping
            // failure has no boundary and no intent: under a strict envelope
            // the buffer is not a partial answer at all, it is protocol
            // framing, and envelope_map_buffered has already dropped it for
            // that reason. So there is nothing to preserve on this path even in
            // principle. For the oom fault there IS real text, and it is still
            // dropped: shipping it would require a stop_reason, and every one
            // of the seven would be a false statement about why generation
            // stopped. It survives where it belongs — the server log, and the
            // OpenAI surfaces, which have an "error" finish_reason to carry it
            // honestly alongside the content.
            if (generation_faulted(finish)) {
                sbuf r = {0};
                sb_lit(&r, "{\"type\":\"error\",");
                anth_error_json(&r, finish);
                sb_lit(&r, "}");
                if (r.failed) send_error(fd, 500, "out of memory building "
                                                  "response");
                else send_response(fd, 500, "application/json", r.s, r.n);
                free(r.s);
                free(tc.s);
                goto done;
            }
            // Same canonical mapping the Responses branch below uses: the
            // chat dialect's tool_calls array is where a turn's calls are
            // extracted once, and each surface only renders them in its own
            // vocabulary.
            jv *call = tool_calls_array(&tc, n_tc);
            resp_doc d = { .with_output = true,
                           .stop_reason = anth_stop_reason(finish,
                                                           g.stop_hit != NULL),
                           .stop_seq = g.stop_hit,
                           .calls = call,
                           .text = g.out.s, .text_n = g.out.n,
                           .reason = g.reason.s, .reason_n = g.reason.n,
                           .with_usage = true,
                           .n_prompt = n_prompt, .n_gen = n_gen, .cached = keep,
                           .forked = reuse.forked, .saved_s = reuse.saved_s,
                           .gtime = gtime, .major_faults = plat_major_faults() - faults_at_start,
                           .schema = schema != NULL,
                           .json_mode = e->json_mode, .spec = spec_used,
                           .req = req };
            sbuf r = {0};
            anth_body(&r, &g, &d);
            send_built(fd, &r);
            free(r.s);
            jv_free(call);
            free(tc.s);
            goto done;
        }
        if (api == API_RESPONSES) {
            // The chat dialect's tool_calls array is the canonical mapping, so
            // the Responses items are derived from it rather than re-extracted
            // from the envelope: one mapping, two renderings.
            jv *call = tool_calls_array(&tc, n_tc);
            bool cut    = strcmp(finish, "length") == 0;
            // the streamed branch above reports the same fault the same way:
            // a turn whose document could not be mapped did not complete, and
            // "max_output_tokens" would send the caller retrying with a bigger
            // budget forever
            bool failed = strcmp(finish, "envelope_error") == 0;
            bool truncated = cut || failed;
            resp_doc d = { .status = truncated ? "incomplete" : "completed",
                           .incomplete = failed ? "envelope_unmapped"
                                       : cut   ? "max_output_tokens" : NULL,
                           .with_output = true,
                           .calls = call,
                           .text = g.out.s, .text_n = g.out.n,
                           .reason = g.reason.s, .reason_n = g.reason.n,
                           .with_usage = true,
                           .n_prompt = n_prompt, .n_gen = n_gen, .cached = keep,
                           .forked = reuse.forked, .saved_s = reuse.saved_s,
                           .gtime = gtime, .major_faults = plat_major_faults() - faults_at_start,
                           .schema = schema != NULL,
                           .json_mode = e->json_mode, .spec = spec_used,
                           .req = req };
            sbuf r = {0};
            responses_body(&r, &g, &d);
            send_built(fd, &r);
            free(r.s);
            jv_free(call);
            free(tc.s);
            goto done;
        }
        sbuf r = {0};
        sb_fmt(&r, "{\"id\":\"%s\",\"object\":\"%s\",\"created\":%ld,\"model\":\"", g.id,
               chat ? "chat.completion" : "text_completion",
               (long)time(NULL));
        sb_esc(&r, SV.model_name, strlen(SV.model_name));
        sb_lit(&r, "\",\"choices\":[{\"index\":0,");
        if (chat) sb_lit(&r, "\"message\":{\"role\":\"assistant\",\"content\":\"");
        else      sb_lit(&r, "\"text\":\"");
        sb_esc(&r, g.out.s ? g.out.s : "", g.out.n);
        sb_lit(&r, "\"");
        if (n_tc) {
            sb_lit(&r, ",\"tool_calls\":[");
            if (tc.failed) r.failed = true;
            else sb_put(&r, tc.s, tc.n);
            sb_lit(&r, "]");
        }
        free(tc.s);
        if (chat && g.reason.n > 0) {
            sb_lit(&r, ",\"reasoning_content\":\"");
            sb_esc(&r, g.reason.s, g.reason.n);
            sb_lit(&r, "\"");
        }
        if (chat) sb_lit(&r, "},");
        else      sb_lit(&r, ",");
        if (chat && e->lp_count > 0) {
            char tb[512];
            sb_lit(&r, "\"logprobs\":{\"content\":[");
            for (int i = 0; i < e->lp_count; i++) {
                if (i) sb_lit(&r, ",");
                int tn = tok_decode(s->tok, e->lp_ids[i], tb, sizeof(tb));
                sb_lit(&r, "{\"token\":\"");
                sb_esc(&r, tb, tn);
                sb_fmt(&r, "\",\"logprob\":%.6f,\"top_logprobs\":[", e->lp_chosen[i]);
                for (int j = 0; j < e->lp_n; j++) {
                    const lp_alt *a = &e->lp_top[(size_t)i * e->lp_n + j];
                    if (a->id < 0) break;
                    if (j) sb_lit(&r, ",");
                    tn = tok_decode(s->tok, a->id, tb, sizeof(tb));
                    sb_lit(&r, "{\"token\":\"");
                    sb_esc(&r, tb, tn);
                    sb_fmt(&r, "\",\"logprob\":%.6f}", a->lp);
                }
                sb_lit(&r, "]}");
            }
            sb_lit(&r, "]},");
        } else if (!chat && e->lp_count > 0) {
            // same body as the non-streaming path, ids included
            append_text_logprobs(&r, s, e);
            sb_lit(&r, ",");
        }
        // JC-R1 constrained-choice posteriors: one entry per decision point
        // (a constrained step where >= 2 probed candidates were legal),
        // shared shape across the chat and text surfaces.
        if (e->cl_count > 0) {
            char tb[512];
            sb_lit(&r, "\"choice_logprobs\":[");
            for (int i = 0; i < e->cl_count; i++) {
                const cl_rec *c = &e->cl_recs[i];
                if (i) sb_lit(&r, ",");
                sb_fmt(&r, "{\"index\":%d,\"n_legal\":%d,"
                           "\"coverage\":%.6f,\"alternatives\":[",
                       c->pos, c->n_legal, c->coverage);
                int stored = c->n_legal < CL_MAX_ALT ? c->n_legal : CL_MAX_ALT;
                for (int j = 0; j < stored; j++) {
                    if (j) sb_lit(&r, ",");
                    int tn = tok_decode(s->tok, c->ids[j], tb, sizeof(tb));
                    sb_lit(&r, "{\"token\":\"");
                    sb_esc(&r, tb, tn);
                    sb_fmt(&r, "\",\"id\":%d,\"prob\":%.6f,"
                               "\"logprob\":%.6f}",
                           c->ids[j], c->prob[j], c->raw_lp[j]);
                }
                sb_lit(&r, "]}");
            }
            sb_lit(&r, "],");
        }
        sb_fmt(&r, "\"finish_reason\":\"%s\"}],"
                   "\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,"
                   "\"total_tokens\":%d},",
               openai_finish(finish), n_prompt, n_gen, n_prompt + n_gen);
        resp_doc td = { .n_prompt = n_prompt, .n_gen = n_gen, .cached = keep,
                        .forked = reuse.forked, .saved_s = reuse.saved_s,
                        .gtime = gtime, .major_faults = plat_major_faults() - faults_at_start,
                           .schema = schema != NULL,
                        .finish_detail = finish_detail_of(finish),
                        .json_mode = e->json_mode, .spec = spec_used };
        telemetry_json(&r, &td);
        sb_lit(&r, "}");
        send_built(fd, &r);
        free(r.s);
    }
done: ;
    // A paging note only when there was paging. Silence is the normal case and
    // a per-request "0 page-ins" would be noise, but when the weights have been
    // evicted this line is the only thing that says the time went to the disk
    // rather than to the model.
    uint64_t faults = plat_major_faults() - faults_at_start;
    char paging[80];
    paging[0] = '\0';
    if (faults)
        snprintf(paging, sizeof(paging),
                 " [%llu page-ins — weights not resident]",
                 (unsigned long long)faults);
    fprintf(stderr, "[slot %d] %s: %d prompt (%d cached) + %d gen tok (%.1f tok/s)%s%s%s\n",
            s->id, g.id, n_prompt, keep, n_gen,
            n_gen / (gtime > 0 ? gtime : 1e-9),
            schema ? " [schema]" : e->json_mode ? " [json]" : spec_used ? " [spec]" : "",
            // "[client gone]" is an accusation, and it used to be made about a
            // client that never left: an unmappable envelope set the same flag.
            // The two are named apart so a log line means what it says.
            g.dead ? " [client gone]"
                   : unmapped ? " [envelope unmapped]" : "", paging);
    completion_cleanup(e, schema, &g);
}
