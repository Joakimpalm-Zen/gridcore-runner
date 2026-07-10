// Token sampling: temperature, top-k, top-p, repeat penalty, and optional
// validity-constrained selection (used for JSON mode).
#include "runner.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static uint64_t rng_next(uint64_t *s) {
    uint64_t x = *s;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    return *s = x;
}
static float rng_f32(uint64_t *s) {
    return (rng_next(s) >> 40) / 16777216.0f;
}

void sampler_reset(sampler *s) {
    s->n_recent = 0;
    s->recent_head = 0;
}

void sampler_accept(sampler *s, int tok) {
    s->recent[s->recent_head] = tok;
    s->recent_head = (s->recent_head + 1) % 256;
    if (s->n_recent < 256) s->n_recent++;
}

typedef struct { float p; int id; } cand_t;
static int cand_cmp(const void *a, const void *b) {
    float d = ((const cand_t *)b)->p - ((const cand_t *)a)->p;
    return d > 0 ? 1 : d < 0 ? -1 : 0;
}

int sample_pick(sampler *s, float *logits, int n_vocab, sample_ok_fn ok, void *ud) {
    if (s->repeat_penalty != 1.0f) {
        for (int i = 0; i < s->n_recent; i++) {
            int tok = s->recent[i];
            if (logits[tok] > 0) logits[tok] /= s->repeat_penalty;
            else                 logits[tok] *= s->repeat_penalty;
        }
    }

    // fast paths that avoid sorting the whole vocabulary
    if (s->temp <= 0) {
        int best = 0;
        for (int i = 1; i < n_vocab; i++) if (logits[i] > logits[best]) best = i;
        if (!ok || ok(ud, best)) return best;
    } else if (ok) {
        // constrained + sampled: quick check whether the unconstrained flow
        // would need filtering at all is not worth it; fall through
    }

    cand_t *c = malloc(sizeof(cand_t) * n_vocab);
    float temp = s->temp > 0 ? s->temp : 1.0f;
    for (int i = 0; i < n_vocab; i++) c[i] = (cand_t){ logits[i] / temp, i };
    qsort(c, n_vocab, sizeof(cand_t), cand_cmp);

    if (s->temp <= 0) {
        // greedy constrained: first valid candidate in probability order
        for (int i = 0; i < n_vocab; i++) {
            if (ok(ud, c[i].id)) { int r = c[i].id; free(c); return r; }
        }
        free(c);
        return -1;
    }

    int want = (s->top_k > 0 && s->top_k < n_vocab) ? s->top_k : n_vocab;
    int k = 0;
    if (ok) {
        // keep the `want` most likely *valid* candidates, in order
        for (int i = 0; i < n_vocab && k < want; i++)
            if (ok(ud, c[i].id)) c[k++] = c[i];
        if (k == 0) { free(c); return -1; }
    } else {
        k = want;
    }

    // softmax over survivors
    float mx = c[0].p, sum = 0;
    for (int i = 0; i < k; i++) { c[i].p = expf(c[i].p - mx); sum += c[i].p; }
    for (int i = 0; i < k; i++) c[i].p /= sum;
    // top-p
    if (s->top_p < 1.0f) {
        float cum = 0;
        int cut = k;
        for (int i = 0; i < k; i++) {
            cum += c[i].p;
            if (cum >= s->top_p) { cut = i + 1; break; }
        }
        k = cut;
        cum = 0;
        for (int i = 0; i < k; i++) cum += c[i].p;
        for (int i = 0; i < k; i++) c[i].p /= cum;
    }
    float r = rng_f32(&s->rng), cum = 0;
    int pick = c[k - 1].id;
    for (int i = 0; i < k; i++) {
        cum += c[i].p;
        if (r < cum) { pick = c[i].id; break; }
    }
    free(c);
    return pick;
}
