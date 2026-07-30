// JC-R2 grammar fast-forward (see the spec walk in engine.c).
//
// The feature's whole value is skipping forwards on grammar-pinned spans; its
// whole risk is that a fast-forwarded generation is not the generation the
// model would have produced on its own. So the gate here is *identity*:
//
//   a constrained generation with fast-forward on must produce, byte for
//   byte, the output of the plain solo path — with and without a draft
//   model in the loop — and must actually have fast-forwarded (a passing
//   identity check that never drafted anything would be vacuous).
//
// The negative cases pin the exclusions: logprob/choice-logprob capture and
// unconstrained requests must never take the speculative walk for grammar's
// sake.
#include "runner.h"
#include "json.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *g_path = "test.gguf";
static int g_fail = 0;

static void ck(int cond, const char *what) {
    if (!cond) { fprintf(stderr, "FAIL: %s\n", what); g_fail = 1; }
    else        fprintf(stderr, "ok: %s\n", what);
}

enum { CTX = 256 };

static model_params base_params(void) {
    model_params p;
    memset(&p, 0, sizeof(p));
    p.gpu_mode  = GPU_OFF;   // the verify path is CPU; keep the test hermetic
    p.n_threads = 1;
    p.n_ctx     = CTX;
    p.n_batch   = 8;
    return p;
}

typedef struct { model_t m; tokenizer tok; sampler smp; engine e; } slot;

static bool slot_open(slot *s, const model_params *p) {
    memset(s, 0, sizeof(*s));
    if (!model_load(&s->m, g_path, p)) return false;
    if (!tokenizer_init(&s->tok, &s->m.gf)) { model_free(&s->m); return false; }
    s->smp.temp = 0;
    s->smp.repeat_penalty = 1.0f;
    s->smp.rng = 1;
    engine_init(&s->e, &s->m, &s->tok, &s->smp);
    return true;
}

static void slot_close(slot *s) {
    free(s->e.hist);
    tokenizer_free(&s->tok);
    model_free(&s->m);
}

typedef struct { char buf[4096]; int n; } sink;

static int collect(void *ud, const char *bytes, int n) {
    sink *o = ud;
    if (o->n + n < (int)sizeof(o->buf)) { memcpy(o->buf + o->n, bytes, n); o->n += n; }
    return 0;
}

// an object schema whose required-property skeleton pins long byte runs
static const char *SCHEMA_SRC =
    "{\"type\":\"object\",\"properties\":{"
    "\"first_field\":{\"type\":\"string\",\"enum\":[\"aa\",\"ab\"]},"
    "\"second_field\":{\"type\":\"integer\"}"
    "},\"required\":[\"first_field\",\"second_field\"]}";

static snode *compile_schema(void) {
    jv *j = json_parse(SCHEMA_SRC, (int)strlen(SCHEMA_SRC));
    if (!j) return NULL;
    char err[128];
    snode *sc = schema_compile(j, err, sizeof(err));
    jv_free(j);
    return sc;
}

// run one constrained generation from a fixed prompt; returns bytes generated
static int gen(slot *s, snode *schema, bool json_mode, bool ff, model_t *dm,
               sink *out) {
    s->e.schema = schema;
    s->e.json_mode = json_mode;
    s->e.gram_ff = ff;
    s->e.dm = dm;
    s->e.draft_k = dm ? 4 : 0;
    engine_reset(&s->e);
    int32_t prompt[4] = { 1, 20, 30, 40 };
    float *logits = engine_feed(&s->e, prompt, 4);
    if (!logits) return -1;
    out->n = 0;
    return engine_generate(&s->e, logits, 64, collect, out, NULL);
}

static void test_identity_and_engagement(void) {
    snode *schema = compile_schema();
    if (!schema) { ck(0, "schema compiles"); return; }
    slot a, b;
    model_params p = base_params();
    if (!slot_open(&a, &p) || !slot_open(&b, &p)) { ck(0, "load model"); return; }

    sink plain, fast;
    ck(gen(&a, schema, false, false, NULL, &plain) > 0, "plain path generates");
    ck(gen(&b, schema, false, true,  NULL, &fast)  > 0, "fast-forward path generates");
    ck(plain.n == fast.n && memcmp(plain.buf, fast.buf, plain.n) == 0,
       "schema output is byte-identical with fast-forward on");
    ck(b.e.spec_st.gr_drafted > 0, "the grammar actually drafted");
    ck(b.e.spec_st.gr_accepted > 0, "grammar drafts were accepted");
    fprintf(stderr, "   (schema: %d rounds, grammar %d/%d over %d bytes)\n",
            b.e.spec_st.rounds, b.e.spec_st.gr_accepted,
            b.e.spec_st.gr_drafted, fast.n);

    // JSON mode pins less (no property skeleton) but must stay identical too
    sink jp, jf;
    ck(gen(&a, NULL, true, false, NULL, &jp) > 0, "plain json_mode generates");
    ck(gen(&b, NULL, true, true,  NULL, &jf) > 0, "ff json_mode generates");
    ck(jp.n == jf.n && memcmp(jp.buf, jf.buf, jp.n) == 0,
       "json_mode output is byte-identical with fast-forward on");

    slot_close(&a);
    slot_close(&b);
    schema_free(schema);
}

static void test_identity_with_draft_model(void) {
    snode *schema = compile_schema();
    if (!schema) { ck(0, "schema compiles"); return; }
    slot a, b;
    model_t dm;
    model_params p = base_params();
    if (!slot_open(&a, &p) || !slot_open(&b, &p) || !model_load(&dm, g_path, &p)) {
        ck(0, "load models"); return;
    }

    sink plain, fast;
    ck(gen(&a, schema, false, false, NULL, &plain) > 0, "plain path generates");
    ck(gen(&b, schema, false, true, &dm,  &fast)  > 0, "ff+draft path generates");
    ck(plain.n == fast.n && memcmp(plain.buf, fast.buf, plain.n) == 0,
       "output is byte-identical with fast-forward composed with --draft");
    ck(b.e.spec_st.gr_drafted > 0, "grammar drafts engage alongside the draft model");

    model_free(&dm);
    slot_close(&a);
    slot_close(&b);
    schema_free(schema);
}

static void test_exclusions(void) {
    slot s;
    model_params p = base_params();
    if (!slot_open(&s, &p)) { ck(0, "load model"); return; }
    snode *schema = compile_schema();

    s.e.schema = schema; s.e.gram_ff = true;
    ck(engine_wants_spec(&s.e), "constrained + ff wants the spec walk");
    s.e.cl_cap = 1;
    ck(!engine_wants_spec(&s.e), "choice_logprobs capture excludes the spec walk");
    s.e.cl_cap = 0; s.e.lp_cap = 1;
    ck(!engine_wants_spec(&s.e), "logprobs capture excludes the spec walk");
    s.e.lp_cap = 0; s.e.schema = NULL;
    ck(!engine_wants_spec(&s.e), "unconstrained never fast-forwards");
    s.e.gram_ff = false; s.e.schema = schema;
    ck(!engine_wants_spec(&s.e), "disabled fast-forward never engages");

    schema_free(schema);
    slot_close(&s);
}

int main(int argc, char **argv) {
    if (argc > 1) g_path = argv[1];
    test_identity_and_engagement();
    test_identity_with_draft_model();
    test_exclusions();
    if (!g_fail) fprintf(stderr, "all grammar fast-forward tests passed\n");
    return g_fail;
}
