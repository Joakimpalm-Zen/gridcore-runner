// Generation engine shared by the CLI and the server: prompt feeding,
// stop-token handling, optional JSON-constrained sampling.
#include "runner.h"

#include "compat.h"

#include <stdio.h>
#include <string.h>

double now_s(void) { return plat_now(); }

void engine_init(engine *e, model_t *m, tokenizer *tok, sampler *smp) {
    memset(e, 0, sizeof(*e));
    e->m = m;
    e->tok = tok;
    e->smp = smp;
    e->stop_ids[e->n_stop++] = tok->eos_id;
    static const char *stops[] = { "<|im_end|>", "<|eot_id|>", "<|end_of_text|>",
                                   "<|endoftext|>", "</s>" };
    for (size_t i = 0; i < sizeof(stops) / sizeof(*stops); i++) {
        int id = tok_find(tok, stops[i]);
        if (id < 0 || e->n_stop >= 8) continue;
        bool dup = false;
        for (int j = 0; j < e->n_stop; j++) if (e->stop_ids[j] == id) dup = true;
        if (!dup) e->stop_ids[e->n_stop++] = id;
    }
    jsonv_init(&e->jv);
}

void engine_reset(engine *e) {
    e->pos = 0;
    e->hit_stop = false;
    sampler_reset(e->smp);
    jsonv_init(&e->jv);
    if (e->schema) sval_init(&e->sv, e->schema);
}

static bool is_stop(engine *e, int id) {
    for (int i = 0; i < e->n_stop; i++) if (e->stop_ids[i] == id) return true;
    return false;
}

float *engine_feed(engine *e, const int32_t *toks, int n) {
    float *logits = NULL;
    model_t *m = e->m;
    for (int i = 0; i < n; ) {
        int chunk = n - i < m->n_batch ? n - i : m->n_batch;
        if (e->pos + chunk > m->n_ctx) return NULL;
        bool last = (i + chunk == n);
        logits = model_forward_batch(m, toks + i, chunk, e->pos, last);
        e->pos += chunk;
        i += chunk;
        if (e->progress && n > 512 && (i % 512 < m->n_batch || last))
            fprintf(stderr, "\rprompt: %d/%d tokens%s", i, n, last ? "\n" : "");
    }
    for (int i = 0; i < n; i++) sampler_accept(e->smp, toks[i]);
    return logits;
}

// validity filter for JSON mode: token must keep output a valid JSON prefix;
// stop/control tokens are allowed only once the object is complete
static bool schema_ok(void *ud, int id) {
    engine *e = ud;
    if (is_stop(e, id) || tok_is_control(e->tok, id)) return e->sv.done;
    char buf[512];
    int n = tok_decode(e->tok, id, buf, sizeof(buf));
    if (n == 0) return e->sv.done;
    sval tmp = e->sv;
    return sval_feed(&tmp, buf, n);
}

static bool json_ok(void *ud, int id) {
    engine *e = ud;
    if (is_stop(e, id) || tok_is_control(e->tok, id)) return e->jv.done;
    char buf[512];
    int n = tok_decode(e->tok, id, buf, sizeof(buf));
    if (n == 0) return e->jv.done;
    jsonv tmp = e->jv;
    return jsonv_feed(&tmp, buf, n);
}

int engine_generate(engine *e, float *logits, int max_new,
                    gen_cb cb, void *ud, double *gen_time) {
    char buf[512];
    int n_gen = 0;
    e->hit_stop = false;
    double t0 = now_s();
    while ((max_new < 0 || n_gen < max_new) && e->pos < e->m->n_ctx) {
        int tok = sample_pick(e->smp, logits, e->m->n_vocab,
                              e->schema ? schema_ok :
                              e->json_mode ? json_ok : NULL, e);
        if (tok < 0) { e->hit_stop = true; break; } // no valid continuation
        sampler_accept(e->smp, tok);
        if (is_stop(e, tok) && !e->ignore_eos) { e->hit_stop = true; break; }
        int n = tok_decode(e->tok, tok, buf, sizeof(buf));
        if (e->schema && n > 0) sval_feed(&e->sv, buf, n);
        else if (e->json_mode && n > 0) jsonv_feed(&e->jv, buf, n);
        if (cb && n > 0 && cb(ud, buf, n) != 0) { n_gen++; break; }
        n_gen++;
        if ((e->schema && e->sv.done) || (!e->schema && e->json_mode && e->jv.done)) {
            e->hit_stop = true;
            break;
        }
        logits = model_forward(e->m, tok, e->pos++);
    }
    if (e->schema && !e->sv.done) {
        // budget expired mid-document: complete it per the schema
        char cbuf[4096];
        int cn = sval_close(&e->sv, cbuf, sizeof(cbuf));
        if (cn > 0 && cb) cb(ud, cbuf, cn); // finish_reason stays "length"
    } else if (!e->schema && e->json_mode && !e->jv.done) {
        // budget expired mid-object: emit a minimal valid completion
        char cbuf[600];
        int cn = jsonv_close(&e->jv, cbuf, sizeof(cbuf));
        if (cn > 0 && cb) cb(ud, cbuf, cn); // finish_reason stays "length"
    }
    if (gen_time) *gen_time = now_s() - t0;
    return n_gen;
}
