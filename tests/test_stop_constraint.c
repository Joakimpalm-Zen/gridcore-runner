// A client `stop` sequence, under constrained output, must still deliver a
// document the client can parse.
//
// The two halves of the turn are owned by different modules and used to
// disagree about what the client received:
//
//   - the engine feeds every decoded byte to the schema validator BEFORE the
//     sink sees it (constraint_accept), and synthesizes the closing tail from
//     the validator's state when generation ends (constraint_close);
//   - the sink (stop_feed in completion.c) drops the matched stop bytes, and
//     everything held behind them, from what it emits.
//
// So the validator was AHEAD of the client's copy by the matched span, and the
// closer it produced continued a document nobody had. With a strict tool
// envelope that showed up as a document that would not map; with a plain
// `response_format: json_schema` and no tools it showed up as INVALID JSON —
// ordinary traffic, silently corrupted rather than failed.
//
// This drives the REAL sink (src/completion.c is included, as
// tests/test_responses_sm.c does, because gen_ctx and stop_feed are static
// there) against the REAL engine, so nothing about the stop filter is
// re-implemented here and cannot drift.
//
// ---------------------------------------------------------------------------
// DETERMINISM: every arm's bytes are FORCED, none are sampled
//
// A gate that is green on one architecture and red on two is worse than no
// gate. Free generation is not portable — different CPUs take different dot
// kernels, so the rounding differs, so the argmax differs, so the bytes
// differ. An earlier version of this file had a json_object arm that relied on
// what the fixture happened to sample; it emitted `\n\n` on macOS/ARM and tabs
// on x86, and its stop never fired on x86. Nothing here may depend on that.
//
// Two independent ways of forcing bytes are used, and each arm says which:
//
//   1. GRAMMAR-FORCED (the schema arms). Both properties are required and each
//      value is a ONE-element enum, so the grammar admits exactly one document.
//      The stop is `aa` — bytes INSIDE an enum string literal, where the
//      grammar cannot admit whitespace, so no tokenization and no argmax can
//      separate them or insert anything between them. These arms run the real
//      model_forward, which is worth keeping: they prove the whole path.
//
//   2. LOGIT-FORCED (the json_object arms). engine_gen_begin/step/end is the
//      public API for a caller that owns the forward, so the test supplies the
//      logits itself: one exactly-representable 1.0f against zeros, chosen so
//      the greedy argmax is the token the test picked. `logits[i] >
//      logits[best]` over those values is an integer-exact comparison on every
//      architecture. Each step also asserts the engine returned the token that
//      was asked for, so a constraint veto is a failure rather than a silent
//      substitution. No model arithmetic runs at all.
#include "runner.h"
#include "json.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/completion.c"

static const char *g_path = "test.gguf";
static int g_fail = 0;

static void ck(int cond, const char *what) {
    if (!cond) { fprintf(stderr, "FAIL: %s\n", what); g_fail = 1; }
    else        fprintf(stderr, "ok: %s\n", what);
}

enum { CTX = 256 };

// one required property per value, each value a one-element enum: the grammar
// admits exactly `{"first_field":"aa","second_field":"cc"}` and nothing else
static const char *SCHEMA_SRC =
    "{\"type\":\"object\",\"properties\":{"
    "\"first_field\":{\"type\":\"string\",\"enum\":[\"aa\"]},"
    "\"second_field\":{\"type\":\"string\",\"enum\":[\"cc\"]}"
    "},\"required\":[\"first_field\",\"second_field\"]}";

// Inside an enum string literal. The grammar cannot put whitespace there and
// cannot choose different bytes, so these two bytes are adjacent in the
// generated stream whatever the vocabulary does with them — and the legal
// closure for the truncated state (`aa","second_field":"cc"}`) begins with
// them, which is the second way this path used to eat its own output.
static const char *SCHEMA_STOP = "aa";

typedef struct { model_t m; tokenizer tok; sampler smp; engine e; } slot;

static bool slot_open(slot *s) {
    model_params p;
    memset(&p, 0, sizeof(p));
    p.gpu_mode  = GPU_OFF;   // keep the test hermetic
    p.n_threads = 1;
    p.n_ctx     = CTX;
    p.n_batch   = 8;
    memset(s, 0, sizeof(*s));
    if (!model_load(&s->m, g_path, &p)) return false;
    if (!tokenizer_init(&s->tok, &s->m.gf)) { model_free(&s->m); return false; }
    s->smp.temp = 0;
    s->smp.repeat_penalty = 1.0f;
    s->smp.rng = 1;
    return engine_init(&s->e, &s->m, &s->tok, &s->smp);
}

static void slot_close(slot *s) {
    free(s->e.hist);
    tokenizer_free(&s->tok);
    model_free(&s->m);
}

// The server's own generation sink, set up the way handle_completion sets it
// up for a buffered request carrying `stop`.
static void sink_open(gen_ctx *g, engine *e, const char **stops, int n_stops) {
    memset(g, 0, sizeof(*g));
    g->fd = -1;
    g->stream = false;
    g->api = API_CHAT;
    g->stop_strs = stops;
    g->n_stop = n_stops;
    g->eng = e;
    think_init(&g->ts, NULL, NULL);
}

// the two post-generation steps handle_completion performs, then hand back the
// content the client would have received
static void sink_close(gen_ctx *g, char *out, int cap, bool *stopped,
                       const char **hit) {
    think_finish(&g->ts, gen_emit, g);
    if (!g->stopped && g->hold.n > 0) {
        emit_channel(g, 0, g->hold.s, (int)g->hold.n);
        g->hold.n = 0;
    }
    int n = (int)g->out.n;
    if (n > cap - 1) n = cap - 1;
    if (n > 0) memcpy(out, g->out.s, (size_t)n);
    out[n] = 0;
    if (stopped) *stopped = g->stopped;
    if (hit) *hit = g->stop_hit;
    free(g->out.s);
    free(g->reason.s);
    free(g->hold.s);
}

// GRAMMAR-FORCED: one constrained generation from a fixed prompt, sampled by
// the real model against a schema that admits exactly one document.
static bool run_turn(slot *s, snode *schema, bool spec,
                     const char **stops, int n_stops,
                     char *out, int cap, const char **hit) {
    s->e.schema = schema;
    s->e.json_mode = false;
    s->e.gram_ff = spec;
    engine_reset(&s->e);

    gen_ctx g;
    sink_open(&g, &s->e, stops, n_stops);

    int32_t prompt[4] = { 1, 20, 30, 40 };
    float *logits = engine_feed(&s->e, prompt, 4);
    if (!logits) { out[0] = 0; return false; }
    engine_generate(&s->e, logits, 64, gen_collect, &g, NULL);

    bool stopped = false;
    sink_close(&g, out, cap, &stopped, hit);
    return stopped;
}

// LOGIT-FORCED: drive the engine one token at a time over engine_gen_step,
// choosing every token by handing it logits, so `text` is generated exactly.
// Returns false if the run could not be scripted (the vocabulary could not
// spell the text, or the constraint vetoed a token) — never silently degrades
// to something else.
static bool run_scripted(slot *s, const char *text,
                         const char **stops, int n_stops,
                         char *out, int cap, bool *stopped, const char **hit) {
    s->e.schema = NULL;
    s->e.json_mode = true;
    s->e.gram_ff = false;
    engine_reset(&s->e);

    enum { MAX_IDS = 64 };
    int32_t ids[MAX_IDS];
    int n_ids = tok_encode_raw(&s->tok, text, (int)strlen(text), ids, MAX_IDS);
    if (n_ids <= 0) return false;
    for (int i = 0; i < n_ids; i++)
        if (ids[i] < 0 || ids[i] >= s->m.n_vocab) return false;

    float *logits = calloc((size_t)s->m.n_vocab, sizeof(float));
    if (!logits) return false;

    gen_ctx g;
    sink_open(&g, &s->e, stops, n_stops);

    bool scripted = true;
    engine_gen_begin(&s->e, n_ids);
    for (int i = 0; i < n_ids; i++) {
        logits[ids[i]] = 1.0f;   // exactly representable; the rest are +0.0f
        int32_t tok = -1; int pos = 0;
        int rc = engine_gen_step(&s->e, logits, gen_collect, &g, &tok, &pos);
        logits[ids[i]] = 0.0f;
        if (rc == ENGINE_STEP_DONE) break;   // the stop matched, or the doc closed
        if (tok != ids[i]) { scripted = false; break; }
        // no model_forward: the next step's logits are the test's, not the
        // model's, which is the whole point
    }
    engine_gen_end(&s->e, gen_collect, &g, NULL);
    free(logits);

    sink_close(&g, out, cap, stopped, hit);
    return scripted;
}

// The document a client would have to accept: it must parse, and it must be a
// legal instance of the schema it asked for.
static bool document_is_legal(const char *doc, const snode *schema) {
    int n = (int)strlen(doc);
    if (n == 0) return false;
    jv *j = json_parse(doc, n);
    if (!j) return false;
    jv_free(j);
    sval v;
    sval_init(&v, schema);
    if (!sval_feed(&v, doc, n)) return false;
    return v.done;
}

// The json_object equivalent. Both checks, for the same reason document_is_legal
// runs both: json_parse is an INDEPENDENT parser, so it cannot share a bug with
// the jsonv machine that produced the closer, while jsonv's `.done` is the one
// that states the property directly — the document is complete, not merely
// well-formed so far.
static bool json_document_is_complete(const char *doc) {
    int n = (int)strlen(doc);
    if (n == 0) return false;
    jv *j = json_parse(doc, n);
    if (!j) return false;
    jv_free(j);
    jsonv v;
    jsonv_init(&v);
    if (!jsonv_feed(&v, doc, n)) return false;
    return v.done;
}

static snode *compile_schema(void) {
    jv *sj = json_parse(SCHEMA_SRC, (int)strlen(SCHEMA_SRC));
    char err[128];
    snode *schema = sj ? schema_compile(sj, err, sizeof(err)) : NULL;
    jv_free(sj);
    return schema;
}

static void test_stop_under_schema(void) {
    snode *schema = compile_schema();
    if (!schema) { ck(0, "schema compiles"); return; }

    slot s;
    if (!slot_open(&s)) { ck(0, "fixture model loads"); schema_free(schema); return; }

    // control: the same constrained turn with no stop sequence
    char whole[512];
    run_turn(&s, schema, false, NULL, 0, whole, sizeof(whole), NULL);
    fprintf(stderr, "   unstopped document: %s\n", whole);
    ck(document_is_legal(whole, schema),
       "constrained turn without a stop sequence is a legal document");

    const char *stops[1] = { SCHEMA_STOP };
    const char *hit = NULL;
    char doc[512];
    bool stopped = run_turn(&s, schema, false, stops, 1, doc, sizeof(doc), &hit);
    fprintf(stderr, "   stopped document:   %s\n", doc);

    // a passing validity check on a turn where the stop never fired would be
    // vacuous, so pin the match itself
    ck(stopped && hit && !strcmp(hit, SCHEMA_STOP),
       "the stop sequence actually matched");
    ck(document_is_legal(doc, schema),
       "a stop match still delivers a legal instance of the schema");

    slot_close(&s);
    schema_free(schema);
}

// The speculative walk (grammar fast-forward, or a draft model) is a second
// generation loop, and it leaves by different doors than engine_gen_step does.
// One of them is a callback that aborted — which is exactly what a stop match
// is — so the same request must not depend on which loop served it.
static void test_stop_under_schema_speculative(void) {
    snode *schema = compile_schema();
    if (!schema) { ck(0, "schema compiles"); return; }

    slot s;
    if (!slot_open(&s)) { ck(0, "fixture model loads"); schema_free(schema); return; }

    const char *stops[1] = { SCHEMA_STOP };
    const char *hit = NULL;
    char doc[512];
    bool stopped = run_turn(&s, schema, true, stops, 1, doc, sizeof(doc), &hit);
    fprintf(stderr, "   speculative stopped document: %s\n", doc);
    ck(s.e.spec_st.gr_drafted > 0, "the speculative walk actually ran");
    ck(stopped && hit, "the stop sequence matched on the speculative walk");
    ck(document_is_legal(doc, schema),
       "a stop match on the speculative walk still delivers a legal document");

    slot_close(&s);
    schema_free(schema);
}

// `response_format: json_object` runs the same desync through a DIFFERENT
// validator — jsonv and jsonv_close, not sval and sval_close — and
// engine_constraint_truncate re-seats it down its own branch, so it gets its
// own arms rather than riding on the schema case.
//
// There is no schema to pin the bytes here, so the tokens are logit-forced.
static void test_stop_under_json_mode(void) {
    slot s;
    if (!slot_open(&s)) { ck(0, "fixture model loads"); return; }

    // `":` lands between the key and its value: the truncated state is inside
    // the key string, which jsonv_close finishes as `":null}` — a closer that
    // begins with the stop that fired.
    static const char *stops[1] = { "\":" };
    const char *hit = NULL;
    char doc[512];
    bool stopped = false;
    bool scripted = run_scripted(&s, "{\"a\":\"bb\"}", stops, 1,
                                 doc, sizeof(doc), &stopped, &hit);
    fprintf(stderr, "   json_object stopped document: %s\n", doc);
    ck(scripted, "the json_object turn generated exactly the scripted tokens");
    ck(stopped && hit && !strcmp(hit, "\":"),
       "the stop sequence matched in json_object mode");
    ck(json_document_is_complete(doc),
       "a stop match under json_object still delivers a complete JSON document");

    slot_close(&s);
}

// The closer is not model output, and a client's stop sequence must not touch
// it EVEN WHEN NO STOP EVER MATCHED. `stop:["}"]` with a json_object request is
// the case that broke: the model here never produces a `}` at all, so the only
// one in the turn is the brace jsonv_close adds — and filtering it left the
// client an object that never closed.
static void test_closer_survives_an_unmatched_stop(void) {
    slot s;
    if (!slot_open(&s)) { ck(0, "fixture model loads"); return; }

    static const char *stops[1] = { "}" };
    const char *hit = NULL;
    char doc[512];
    bool stopped = true;
    // deliberately unterminated: the string and the object are both left open
    bool scripted = run_scripted(&s, "{\"a\":\"bb", stops, 1,
                                 doc, sizeof(doc), &stopped, &hit);
    fprintf(stderr, "   unmatched-stop document: %s\n", doc);
    ck(scripted, "the turn generated exactly the scripted tokens");
    ck(!stopped && !hit, "the model never produced the stop sequence");
    ck(json_document_is_complete(doc),
       "the synthesized closer survives a stop sequence that matches it");

    slot_close(&s);
}

int main(int argc, char **argv) {
    if (argc > 1) g_path = argv[1];
    test_stop_under_schema();
    test_stop_under_json_mode();
    test_closer_survives_an_unmatched_stop();
    test_stop_under_schema_speculative();
    if (!g_fail) fprintf(stderr, "all stop/constraint tests passed\n");
    return g_fail;
}
