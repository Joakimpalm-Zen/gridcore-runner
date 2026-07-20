// Generation engine shared by the CLI and the server: prompt feeding,
// stop-token handling, optional JSON-constrained sampling.
#include "runner.h"

#include "compat.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

double now_s(void) { return plat_now(); }

enum { CP_PROBE, CP_THINK, CP_OUTPUT };

static void constraint_reset(engine *e) {
    e->constraint_phase = e->m && e->m->think_open && e->m->think_close
                            ? CP_PROBE : CP_OUTPUT;
    e->constraint_tag_possible = e->constraint_phase == CP_PROBE;
    e->constraint_tag_match = 0;
    e->constraint_close_match = 0;
}

void engine_init(engine *e, model_t *m, tokenizer *tok, sampler *smp) {
    free(e->hist); // slot engines are re-inited on model swap; e must be zeroed
    memset(e, 0, sizeof(*e));
    e->m = m;
    e->tok = tok;
    e->smp = smp;
    e->stop_ids[e->n_stop++] = tok->eos_id;
    static const char *stops[] = { "<|im_end|>", "<|eot_id|>", "<|end_of_text|>",
                                   "<|endoftext|>", "</s>",
                                   // gemma turn terminators (gemma1-3 / gemma4)
                                   "<end_of_turn>", "<turn|>" };
    for (size_t i = 0; i < sizeof(stops) / sizeof(*stops); i++) {
        int id = tok_find(tok, stops[i]);
        if (id < 0 || e->n_stop >= 8) continue;
        bool dup = false;
        for (int j = 0; j < e->n_stop; j++) if (e->stop_ids[j] == id) dup = true;
        if (!dup) e->stop_ids[e->n_stop++] = id;
    }
    jsonv_init(&e->jv);
    constraint_reset(e);
    e->hist = malloc(sizeof(int32_t) * m->n_ctx);
}

void engine_reset(engine *e) {
    e->pos = 0;
    e->hit_stop = false;
    sampler_reset(e->smp);
    jsonv_init(&e->jv);
    if (e->schema) sval_init(&e->sv, e->schema);
    constraint_reset(e);
    e->dpos = 0;
}

int engine_rewind(engine *e, const int32_t *toks, int n) {
    int keep = 0;
    if (e->hist)
        while (keep < e->pos && keep < n - 1 && e->hist[keep] == toks[keep])
            keep++; // n - 1: always feed at least one token to get logits
    e->pos = keep;
    // the draft's KV beyond the kept prefix was computed from the previous
    // request's tokens; the catch-up loop re-feeds hist[dpos..pos)
    if (e->dpos > keep) e->dpos = keep;
    e->hit_stop = false;
    sampler_reset(e->smp);
    // the kept prefix still counts toward the repeat-penalty window
    for (int i = 0; i < keep; i++) sampler_accept(e->smp, toks[i]);
    jsonv_init(&e->jv);
    if (e->schema) sval_init(&e->sv, e->schema);
    constraint_reset(e);
    return keep;
}

// load a draft model for speculative decoding, with the same gates in CLI
// and server mode: the target must keep a CPU verify path, and the vocabs
// must match modulo family padding. NULL (with a stderr note) = run plain.
model_t *spec_draft_load(const char *path, const model_t *target,
                         const model_params *mp) {
    if (target->gpu && target->gpu_layers >= target->n_layer) {
        fprintf(stderr, "draft: target is fully GPU-offloaded — speculative "
                "decoding needs the CPU verify path, ignoring --draft\n");
        return NULL;
    }
    model_params dmp = *mp;
    dmp.n_ctx = target->n_ctx; // draft must cover the target's positions
    dmp.kv_q8 = false;
    dmp.gpu_mode = GPU_AUTO;   // a small draft usually fits VRAM whole
    model_t *dm = malloc(sizeof(model_t));
    if (!model_load(dm, path, &dmp)) {
        // model_load memsets the struct on entry and may allocate before failing
        // (late load failures must free partial buffers to avoid leaks)
        model_free(dm);
        free(dm);
        return NULL;
    }
    if (abs(dm->n_vocab - target->n_vocab) > 512) {
        // model families pad the vocab differently per size; small
        // differences are padding ids the draft never emits
        fprintf(stderr, "draft: vocab mismatch (%d vs %d) — ignoring --draft\n",
                dm->n_vocab, target->n_vocab);
        model_free(dm);
        free(dm);
        return NULL;
    }
    return dm;
}

static bool is_stop(engine *e, int id) {
    for (int i = 0; i < e->n_stop; i++) if (e->stop_ids[i] == id) return true;
    return false;
}

float *engine_feed(engine *e, const int32_t *toks, int n) {
    float *logits = NULL;
    model_t *m = e->m;
    if (e->hist && e->pos + n <= m->n_ctx)
        memcpy(e->hist + e->pos, toks, sizeof(int32_t) * n);
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

// Advance a tag-prefix match by one byte, retaining the longest suffix that
// can still begin the tag. Tags are tiny, so the direct overlap check keeps
// this state self-contained without another allocation or public parser API.
static int tag_advance(const char *tag, int match, char c) {
    int tl = (int)strlen(tag);
    int max = match + 1 < tl ? match + 1 : tl;
    for (int k = max; k > 0; k--) {
        int start = match + 1 - k;
        bool same = true;
        for (int j = 0; j < k; j++) {
            int at = start + j;
            char got = at < match ? tag[at] : c;
            if (got != tag[j]) { same = false; break; }
        }
        if (same) return k;
    }
    return 0;
}

static bool constraint_payload_feed(engine *e, bool schema,
                                    const char *bytes, int n) {
    if (schema) {
        sval tmp = e->sv;
        if (!sval_feed(&tmp, bytes, n)) return false;
        e->sv = tmp;
    } else {
        jsonv tmp = e->jv;
        if (!jsonv_feed(&tmp, bytes, n)) return false;
        e->jv = tmp;
    }
    return true;
}

static void constraint_payload_reset(engine *e, bool schema) {
    if (schema) sval_init(&e->sv, e->schema);
    else        jsonv_init(&e->jv);
}

// Feed one decoded token through the optional thinking prelude and then the
// JSON/schema validator. On success, *visible is the first byte belonging to
// the constrained payload, or -1 when this token is entirely prelude/tag.
// The function mutates only its engine argument, so sample lookahead calls it
// on a shallow engine copy and acceptance calls it on the real engine.
static bool constraint_feed(engine *e, bool schema, const char *bytes, int n,
                            int *visible) {
    *visible = -1;
    if (e->constraint_phase == CP_OUTPUT) {
        if (!constraint_payload_feed(e, schema, bytes, n)) return false;
        *visible = 0;
        return true;
    }

    const char *open = e->m->think_open;
    const char *close = e->m->think_close;
    if (e->constraint_phase == CP_THINK) {
        int cl = (int)strlen(close);
        for (int i = 0; i < n; i++) {
            e->constraint_close_match = tag_advance(
                close, e->constraint_close_match, bytes[i]);
            if (e->constraint_close_match != cl) continue;
            e->constraint_phase = CP_OUTPUT;
            e->constraint_close_match = 0;
            constraint_payload_reset(e, schema);
            int at = i + 1;
            if (at < n && !constraint_payload_feed(e, schema, bytes + at, n - at))
                return false;
            if (at < n) *visible = at;
            return true;
        }
        return true;
    }

    // CP_PROBE keeps both safe starts alive: a direct constrained payload, or
    // optional leading whitespace followed by the declared opening tag.
    bool payload_ok;
    sval sv = e->sv;
    jsonv jv = e->jv;
    if (schema) payload_ok = sval_feed(&sv, bytes, n);
    else        payload_ok = jsonv_feed(&jv, bytes, n);

    int ol = (int)strlen(open);
    bool tag_ok = e->constraint_tag_possible;
    int match = e->constraint_tag_match;
    for (int i = 0; tag_ok && i < n; i++) {
        char c = bytes[i];
        if (match == 0 && (c == ' ' || c == '\n' || c == '\r' || c == '\t'))
            continue;
        if (c != open[match]) { tag_ok = false; break; }
        match++;
        if (match != ol) continue;

        e->constraint_phase = CP_THINK;
        e->constraint_tag_possible = false;
        e->constraint_tag_match = 0;
        e->constraint_close_match = 0;
        constraint_payload_reset(e, schema);
        int sub = -1;
        if (i + 1 < n &&
            !constraint_feed(e, schema, bytes + i + 1, n - i - 1, &sub))
            return false;
        if (sub >= 0) *visible = i + 1 + sub;
        return true;
    }

    if (!payload_ok && !tag_ok) return false;
    if (payload_ok) {
        if (schema) e->sv = sv;
        else        e->jv = jv;
    }
    e->constraint_tag_possible = tag_ok;
    e->constraint_tag_match = tag_ok ? match : 0;
    if (!tag_ok) {
        e->constraint_phase = CP_OUTPUT;
        *visible = 0;
    }
    // Ambiguous leading whitespace is deliberately held back. It is optional
    // JSON whitespace if the payload wins, and prelude whitespace otherwise.
    return true;
}

static bool constraint_done(const engine *e, bool schema) {
    return e->constraint_phase == CP_OUTPUT &&
           (schema ? e->sv.done : e->jv.done);
}

// validity filter for constrained mode. Before a declared thinking block
// closes, ordinary tokens are allowed; stop/control tokens remain valid only
// inside that prelude or after the constrained payload is complete.
static bool constraint_token_ok(engine *e, int id, bool schema) {
    if (is_stop(e, id) || tok_is_control(e->tok, id))
        return e->constraint_phase == CP_THINK || constraint_done(e, schema);
    char buf[512];
    int n = tok_decode(e->tok, id, buf, sizeof(buf));
    if (n == 0)
        return e->constraint_phase == CP_THINK || constraint_done(e, schema);
    engine tmp = *e;
    int visible;
    return constraint_feed(&tmp, schema, buf, n, &visible);
}

static bool schema_ok(void *ud, int id) {
    return constraint_token_ok(ud, id, true);
}

static bool json_ok(void *ud, int id) {
    return constraint_token_ok(ud, id, false);
}

static int constraint_accept(engine *e, bool schema, const char *bytes, int n,
                             gen_cb cb, void *ud) {
    int visible;
    if (!constraint_feed(e, schema, bytes, n, &visible)) return 1;
    return cb && visible >= 0 && visible < n
             ? cb(ud, bytes + visible, n - visible) : 0;
}

// logprob capture: raw-logit log-softmax stats taken BEFORE sample_pick
// mutates the recent tokens' logits with the repeat penalty
typedef struct {
    float lse;                 // log sum exp of the raw logits
    float snap[256];           // raw values of the penalty-window tokens
    int   snap_ids[256], n_snap;
} lp_pre;

static void lp_capture_pre(engine *e, const float *logits, lp_pre *p) {
    int V = e->m->n_vocab;
    float mx = logits[0];
    for (int i = 1; i < V; i++) if (logits[i] > mx) mx = logits[i];
    double sum = 0;
    for (int i = 0; i < V; i++) sum += expf(logits[i] - mx);
    p->lse = mx + logf((float)sum);
    p->n_snap = 0;
    sampler *s = e->smp;
    for (int i = 0; i < s->n_recent; i++) {
        p->snap_ids[p->n_snap] = s->recent[i];
        p->snap[p->n_snap++]   = logits[s->recent[i]];
    }
    // top-N alternatives for this position (insertion into a small list)
    if (e->lp_n <= 0) return;
    lp_alt *top = e->lp_top + (size_t)e->lp_count * e->lp_n;
    int filled = 0;
    for (int i = 0; i < V; i++) {
        float lp = logits[i] - p->lse;
        if (filled == e->lp_n && lp <= top[filled - 1].lp) continue;
        int j = filled < e->lp_n ? filled++ : e->lp_n - 1;
        while (j > 0 && top[j - 1].lp < lp) { top[j] = top[j - 1]; j--; }
        top[j].id = i; top[j].lp = lp;
    }
    for (int j = filled; j < e->lp_n; j++) { top[j].id = -1; top[j].lp = 0; }
}

// speculative decoding: sampler-equality verification (llama.cpp-style) —
// sample each position from the TARGET's logits with the full sampler chain;
// a draft is accepted when the sampled token equals it, so output follows
// exactly the same distribution as the non-speculative path
// budget expired mid-document: complete it so constrained output stays valid
// JSON (the caller's finish_reason stays "length")
static void constraint_close(engine *e, gen_cb cb, void *ud) {
    // A hard token ceiling may land inside the reasoning prelude. Preserve the
    // constrained-output contract by synthesizing the same minimal valid tail
    // used for an ordinary mid-document ceiling; reasoning tokens still count
    // toward max_new and no extra model sampling occurs here.
    if ((e->schema || e->json_mode) && e->constraint_phase != CP_OUTPUT) {
        constraint_payload_reset(e, e->schema != NULL);
        e->constraint_phase = CP_OUTPUT;
    }
    if (e->schema && !e->sv.done) {
        char cbuf[4096];
        int cn = sval_close(&e->sv, cbuf, sizeof(cbuf));
        if (cn > 0 && cb) cb(ud, cbuf, cn);
    } else if (!e->schema && e->json_mode && !e->jv.done) {
        char cbuf[600];
        int cn = jsonv_close(&e->jv, cbuf, sizeof(cbuf));
        if (cn > 0 && cb) cb(ud, cbuf, cn);
    }
}

static int engine_generate_spec(engine *e, float *logits, int max_new,
                                gen_cb cb, void *ud, double *gen_time) {
    char buf[512];
    int n_gen = 0;
    e->hit_stop = false;
    e->lp_count = 0;
    double t0 = now_s();
    model_t *m = e->m, *dm = e->dm;
    int K = e->draft_k;
    if (K < 1) K = 1;
    if (K > m->spec_batch - 1) K = m->spec_batch - 1;
    int32_t d[16];
    float *dl = NULL; // draft logits for position dpos
    // Even under JSON/schema constraints, speculation stays target-exact:
    // the draft proposes, but only target-sampled tokens feed the validator.
    sample_ok_fn ok = e->schema ? schema_ok : e->json_mode ? json_ok : NULL;
    int st_rounds = 0, st_drafted = 0, st_accepted = 0;
    #define SPEC_STATS() fprintf(stderr, \
        "spec: %d rounds, %d drafted, %d accepted (%.2f tok/round)\n", \
        st_rounds, st_drafted, st_accepted, \
        st_rounds ? (double)n_gen / st_rounds : 0)

    while ((max_new < 0 || n_gen < max_new) && e->pos < m->n_ctx) {
        st_rounds++;
        // catch the draft up on tokens accepted since its last position
        while (e->dpos < e->pos) {
            int chunk = e->pos - e->dpos < dm->n_batch ? e->pos - e->dpos
                                                       : dm->n_batch;
            if (e->dpos + chunk > dm->n_ctx) { dl = NULL; break; }
            dl = model_forward_batch(dm, e->hist + e->dpos, chunk, e->dpos,
                                     e->dpos + chunk == e->pos);
            e->dpos += chunk;
        }
        // draft up to K tokens greedily (nd == 0 degrades to plain decoding)
        int nd = 0;
        while (dl && nd < K && e->pos + nd + 1 < m->n_ctx &&
               e->dpos + 1 < dm->n_ctx) {
            int best = 0;
            for (int i = 1; i < dm->n_vocab; i++)
                if (dl[i] > dl[best]) best = i;
            if (best >= m->n_vocab) break; // padding id the target can't embed
            d[nd++] = best;
            st_drafted++;
            dl = model_forward(dm, best, e->dpos++);
        }
        // one batched target forward computes every draft position's hidden
        // state; row logits are pulled lazily as the walk reaches them
        if (nd && !model_forward_batch_keep(m, d, nd, e->pos))
            nd = 0; // verify unavailable: plain decoding

        // walk the drafts (i < nd) plus one bonus position (i == nd)
        int i = 0;
        for (; i <= nd; i++) {
            if (max_new >= 0 && n_gen >= max_new) goto rewind;
            float *ti = i == 0 ? logits : model_spec_row_logits(m, i - 1);
            int tok = sample_pick(e->smp, ti, m->n_vocab, ok, e);
            if (tok < 0) {
                e->hit_stop = true;
                e->pos += i; // keep the accepted drafts' KV
                if (e->dpos > e->pos) e->dpos = e->pos;
                goto done;
            }
            sampler_accept(e->smp, tok);
            if (getenv("RUNNER_DEBUG_TOKENS")) fprintf(stderr, " %d", tok);
            if (is_stop(e, tok) && !e->ignore_eos) {
                e->hit_stop = true;
                e->pos += i; // keep the accepted drafts' KV
                if (e->dpos > e->pos) e->dpos = e->pos;
                goto done;
            }
            int n = tok_decode(e->tok, tok, buf, sizeof(buf));
            int rc = e->schema && n > 0
                       ? constraint_accept(e, true, buf, n, cb, ud)
                   : e->json_mode && n > 0
                       ? constraint_accept(e, false, buf, n, cb, ud)
                   : cb && n > 0 ? cb(ud, buf, n) : 0;
            n_gen++;
            bool constrained_done = e->schema
                                      ? constraint_done(e, true)
                                      : e->json_mode && constraint_done(e, false);
            if (i < nd && tok == d[i] && rc == 0) {
                e->hist[e->pos + i] = tok; // accepted: its KV is already right
                st_accepted++;
                if (constrained_done) {
                    e->hit_stop = true;
                    e->pos += i + 1;
                    if (e->dpos > e->pos) e->dpos = e->pos;
                    if (gen_time) *gen_time = now_s() - t0;
                    SPEC_STATS();
                    return n_gen;
                }
                continue;
            }
            // mismatch, bonus position, or aborted: forward the real token
            e->pos += i;
            if (constrained_done) {
                e->hit_stop = true;
                if (e->dpos > e->pos) e->dpos = e->pos;
                if (gen_time) *gen_time = now_s() - t0;
                SPEC_STATS();
                return n_gen;
            }
            if (e->hist && e->pos < m->n_ctx) e->hist[e->pos] = tok;
            logits = model_forward(m, tok, e->pos);
            e->pos++;
            // rewind the draft to just before this token: the next catch-up
            // refeeds it, fixing the draft's KV for the rejected position AND
            // refreshing dl (clamping to pos left dl stale from the abandoned
            // round — acceptance collapsed to ~zero)
            if (e->dpos > e->pos - 1) e->dpos = e->pos - 1;
            if (rc) {
                if (gen_time) *gen_time = now_s() - t0;
                SPEC_STATS();
                return n_gen;
            }
            i = -1; // signal: already advanced
            break;
        }
        if (i >= 0) {
rewind:
            e->pos += i > nd ? nd : i; // budget hit mid-walk: keep accepted
            if (e->dpos > e->pos) e->dpos = e->pos;
            if (max_new >= 0 && n_gen >= max_new) break;
        }
    }
done:
    constraint_close(e, cb, ud);
    if (gen_time) *gen_time = now_s() - t0;
    SPEC_STATS();
    #undef SPEC_STATS
    return n_gen;
}

int engine_generate(engine *e, float *logits, int max_new,
                    gen_cb cb, void *ud, double *gen_time) {
    char buf[512];
    int n_gen = 0;
    if (e->dm && e->lp_cap == 0)
        return engine_generate_spec(e, logits, max_new, cb, ud, gen_time);
    e->hit_stop = false;
    e->lp_count = 0;
    double t0 = now_s();
    while ((max_new < 0 || n_gen < max_new) && e->pos < e->m->n_ctx) {
        lp_pre pre;
        pre.lse = 0; pre.n_snap = 0;
        bool want_lp = e->lp_cap && e->lp_count < e->lp_cap;
        if (want_lp) lp_capture_pre(e, logits, &pre);
        int tok = sample_pick(e->smp, logits, e->m->n_vocab,
                              e->schema ? schema_ok :
                              e->json_mode ? json_ok : NULL, e);
        if (tok < 0) { e->hit_stop = true; break; } // no valid continuation
        sampler_accept(e->smp, tok);
        if (getenv("RUNNER_DEBUG_TOKENS")) fprintf(stderr, " %d", tok);
        if (is_stop(e, tok) && !e->ignore_eos) { e->hit_stop = true; break; }
        if (want_lp) {
            float raw = logits[tok]; // unmutated unless in the penalty window
            for (int i = 0; i < pre.n_snap; i++)
                if (pre.snap_ids[i] == tok) { raw = pre.snap[i]; break; }
            e->lp_ids[e->lp_count]    = tok;
            e->lp_chosen[e->lp_count] = raw - pre.lse;
            e->lp_count++;
        }
        int n = tok_decode(e->tok, tok, buf, sizeof(buf));
        int rc = e->schema && n > 0
                   ? constraint_accept(e, true, buf, n, cb, ud)
               : e->json_mode && n > 0
                   ? constraint_accept(e, false, buf, n, cb, ud)
               : cb && n > 0 ? cb(ud, buf, n) : 0;
        if (rc != 0) { n_gen++; break; }
        n_gen++;
        if ((e->schema && constraint_done(e, true)) ||
            (!e->schema && e->json_mode && constraint_done(e, false))) {
            e->hit_stop = true;
            break;
        }
        if (e->hist && e->pos < e->m->n_ctx) e->hist[e->pos] = tok;
        logits = model_forward(e->m, tok, e->pos++);
    }
    constraint_close(e, cb, ud);
    if (gen_time) *gen_time = now_s() - t0;
    return n_gen;
}
