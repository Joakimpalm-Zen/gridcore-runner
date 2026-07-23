// Token sampling: temperature, top-k, top-p, repeat penalty, optional
// validity-constrained selection (used for JSON mode), and the per-family
// defaults that seed all of the above.
#include "runner.h"

#include <ctype.h>
#include <math.h>
#include <stdio.h>
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
    const cand_t *x = a, *y = b;
    if (x->p != y->p) return y->p > x->p ? 1 : -1;
    return x->id - y->id;   // deterministic order among exact ties
}

// Sample from `k` candidates carrying temperature-scaled logits in descending
// order: softmax, top-p, min-p, then the roulette pick. `norm_sum` > 0 uses a
// caller-computed softmax denominator (the head fast path passes the whole
// vocabulary's mass so truncation does not change any candidate's share);
// 0 computes it over the k candidates, the historical behavior.
static int pick_scaled(sampler *s, cand_t *c, int k, float norm_sum) {
    float mx = c[0].p, sum = 0;
    for (int i = 0; i < k; i++) { c[i].p = expf(c[i].p - mx); sum += c[i].p; }
    if (norm_sum > 0) sum = norm_sum;
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
    // min-p: drop candidates far less likely than the best one
    if (s->min_p > 0.0f) {
        float floor_p = s->min_p * c[0].p;
        int cut = k;
        for (int i = 1; i < k; i++)
            if (c[i].p < floor_p) { cut = i; break; }
        if (cut < k) {
            k = cut;
            float cum = 0;
            for (int i = 0; i < k; i++) cum += c[i].p;
            for (int i = 0; i < k; i++) c[i].p /= cum;
        }
    }
    float r = rng_f32(&s->rng), cum = 0;
    int pick = c[k - 1].id;
    for (int i = 0; i < k; i++) {
        cum += c[i].p;
        if (r < cum) { pick = c[i].id; break; }
    }
    return pick;
}

// Head cap for the large-vocab fast path: distributions whose top-k/top-p
// survivors exceed this fall back to the exact full sort.
#define HEAD_CAP 4096

int sample_pick(sampler *s, float *logits, int n_vocab, sample_ok_fn ok, void *ud) {
    // Greedy is a determinism request: return the model's argmax, unmodified.
    // The repeat penalty exists to add variety to *sampled* output and has no
    // meaning when the caller asked for the single most likely token, so it
    // does not run here. (It used to, which meant `--temp 0` could return a
    // token that was not the argmax — and made an exempt-list necessary just
    // to let a penalised model emit its own stop token.)
    if (s->temp > 0 && s->repeat_penalty != 1.0f) {
        for (int i = 0; i < s->n_recent; i++) {
            int tok = s->recent[i];
            bool exempt = false;
            for (int k = 0; k < s->n_no_penalty; k++)
                if (s->no_penalty[k] == tok) { exempt = true; break; }
            if (exempt) continue;
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

    float temp = s->temp > 0 ? s->temp : 1.0f;
    int want_k = (s->top_k > 0 && s->top_k < n_vocab) ? s->top_k : n_vocab;

    // Large-vocab fast path: sorting a 128k-entry vocabulary costs ~12ms per
    // token — it halved Llama-3.2's measured decode rate. Everything top-k /
    // top-p can keep lives in a small head of the distribution, so find that
    // head with counting passes and sort only it. Exactness: the head holds
    // every token above a logit threshold; it is used only once it provably
    // contains the whole surviving set (>= top_k candidates, or at least
    // top_p of the total mass, whose cutoff prefix then lies inside). The
    // whole-vocabulary softmax denominator is passed through, so each
    // candidate's probability is what the full sort would have given it.
    if (!ok && s->temp > 0 && n_vocab >= 4096 &&
        (want_k < n_vocab || s->top_p < 1.0f)) {
        float mx = logits[0] / temp;
        for (int i = 1; i < n_vocab; i++) {
            float v = logits[i] / temp;
            if (v > mx) mx = v;
        }
        double total = 0;
        for (int i = 0; i < n_vocab; i++)
            total += expf(logits[i] / temp - mx);
        cand_t *h = malloc(sizeof(cand_t) * HEAD_CAP);
        if (h) {
            float t_log = logf(1.0f / 1024.0f);   // head: p >= p_max/1024
            for (int loosen = 0; loosen < 6; loosen++, t_log *= 4) {
                int m = 0;
                bool overflow = false;
                double head_mass = 0;
                for (int i = 0; i < n_vocab; i++) {
                    float v = logits[i] / temp;
                    if (v - mx >= t_log) {
                        if (m == HEAD_CAP) { overflow = true; break; }
                        h[m++] = (cand_t){ v, i };
                        head_mass += expf(v - mx);
                    }
                }
                if (overflow) break;   // head too broad: full sort below
                bool enough = (want_k < n_vocab)
                                  ? m >= want_k
                                  : head_mass >= (double)s->top_p * total;
                if (!enough) continue; // loosen the threshold and retry
                qsort(h, m, sizeof(cand_t), cand_cmp);
                int k = m < want_k ? m : want_k;
                // top_k active: softmax over the k survivors (historical
                // semantics); top_k off: normalize by the whole vocabulary
                int pick = pick_scaled(s, h, k,
                                       want_k < n_vocab ? 0 : (float)total);
                free(h);
                return pick;
            }
            free(h);
        }
    }

    cand_t *c = malloc(sizeof(cand_t) * n_vocab);
    if (!c) return -2;  // allocation failure — an error, not a stop
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

    int k = 0;
    if (ok) {
        // keep the `want_k` most likely *valid* candidates, in order
        for (int i = 0; i < n_vocab && k < want_k; i++)
            if (ok(ud, c[i].id)) c[k++] = c[i];
        if (k == 0) { free(c); return -1; }
    } else {
        k = want_k;
    }

    int pick = pick_scaled(s, c, k, 0);
    free(c);
    return pick;
}

// ------------------------------------------------------ per-family presets
//
// Each entry uses the family's own published recommendation where one exists;
// `source` records which one, and --caps prints it. Where a family publishes
// nothing for a knob, the generic value is kept rather than invented — with
// one deliberate exception, phi3's repeat penalty, explained below.
//
// About repeat_penalty: it divides the raw logit, so the size of the nudge it
// gives is proportional to logit magnitude. Measured over models/, Llama-3.2's
// logits top out near +20 while Phi-3.5's reach about +65, which makes the
// same 1.1 setting roughly three times stronger on Phi-3.5. Penalty p shifts a
// logit x by x*(1 - 1/p); matching 1.1's effect at x=20 (a shift of ~1.8) at
// x=65 needs p ~= 1.03. That is where phi3's number comes from, and it is the
// only value in this table that is calibration rather than citation.
static const sampler_preset PRESETS[] = {
    // runner's historical fixed defaults; the fallback for families that
    // publish nothing
    { "generic", "runner defaults (no vendor recommendation for this model)",
      0.80f, 0.95f, 0.05f, 1.10f, 40 },

    // Qwen3 model card, "Best Practices" — thinking-mode settings, since
    // runner surfaces qwen3's thinking channel. (Non-thinking is 0.7/0.8/20.)
    // The card also states repetition_penalty 1.0 and warns against greedy.
    { "qwen3", "Qwen3 model card best practices (thinking mode)",
      0.60f, 0.95f, 0.00f, 1.00f, 20 },

    // Qwen2.5-Instruct generation_config.json
    { "qwen2.5", "Qwen2.5-Instruct generation_config.json",
      0.70f, 0.80f, 0.00f, 1.05f, 20 },

    // Meta's Llama-3.x-Instruct generation_config.json ships temperature 0.6
    // and top_p 0.9 and no top_k, so top-k filtering is off here.
    { "llama3", "Llama-3.x-Instruct generation_config.json (Meta)",
      0.60f, 0.90f, 0.00f, 1.10f, 0 },

    // Mistral publish no sampling params in the v0.3 generation_config; these
    // are the documented Mistral API defaults.
    { "mistral", "Mistral AI API defaults (no params in v0.3 generation_config)",
      0.70f, 1.00f, 0.00f, 1.10f, 0 },

    // Gemma team's stated optimum for Gemma 3 inference (min_p 0.0, with 0.01
    // called out as optional — the stated optimum is used).
    { "gemma3", "Gemma 3 inference settings published by the Gemma team",
      1.00f, 0.95f, 0.00f, 1.10f, 64 },

    // Phi-3.5-mini-instruct model card sample inference code, which runs
    // temperature 0.0 / do_sample False. Greedy by default is unusual but it
    // is what Microsoft publish; a caller wanting variety overrides --temp,
    // and the calibrated penalty above is waiting for them when they do.
    { "phi3", "Phi-3.5-mini-instruct model card sample inference args",
      0.00f, 1.00f, 0.00f, 1.03f, 0 },

    // SmolLM2-1.7B-Instruct model card: "We suggest to use temperature=0.2,
    // top_p=0.9".
    { "smollm2", "SmolLM2-Instruct model card suggestion",
      0.20f, 0.90f, 0.00f, 1.10f, 0 },

    // Gridcore Syntetik (gridcore-model): the suite's own from-scratch
    // decoder that compiles requests into auditable execution contracts.
    // A contract compiler wants deterministic, reproducible output under
    // schema enforcement — not creative sampling — so the preset is greedy
    // with no repeat penalty (a penalty on a constrained JSON grammar only
    // distorts a distribution the schema already pins). PROVISIONAL: aligned
    // to Syntetik's stated "auditable/deterministic" design; the model team
    // (gridcore-model) should confirm or override once training settles.
    { "gridcore", "Gridcore Syntetik contract compiler (deterministic)",
      0.00f, 1.00f, 0.00f, 1.00f, 0 },
};
#define N_PRESETS ((int)(sizeof(PRESETS) / sizeof(PRESETS[0])))

const sampler_preset *sampler_preset_at(int i) {
    return i >= 0 && i < N_PRESETS ? &PRESETS[i] : NULL;
}

static const sampler_preset *by_name(const char *name) {
    for (int i = 0; i < N_PRESETS; i++)
        if (!strcmp(PRESETS[i].name, name)) return &PRESETS[i];
    return &PRESETS[0];
}

// case-insensitive substring search over a bounded lowercase copy
static bool has(const char *hay, const char *needle) {
    return hay && strstr(hay, needle) != NULL;
}

const sampler_preset *sampler_preset_for(const char *arch, const char *name) {
    char lname[128];
    size_t n = name ? strlen(name) : 0;
    if (n >= sizeof(lname)) n = sizeof(lname) - 1;
    for (size_t i = 0; i < n; i++) lname[i] = (char)tolower((unsigned char)name[i]);
    lname[n] = 0;

    if (!arch) arch = "";
    // Gridcore Syntetik declares general.architecture "llama" (it is a
    // llama-shaped decoder) and general.name "gridcore-<size>", so the name
    // is what identifies it. Checked first: a suite-native model should never
    // fall through to a vendor preset. "syntetik" is accepted too in case the
    // published name changes to the product name.
    if (has(lname, "gridcore-") || has(lname, "syntetik"))
        return by_name("gridcore");

    // Architectures that name exactly one family.
    if (!strcmp(arch, "qwen3"))  return by_name("qwen3");
    if (!strcmp(arch, "qwen2"))  return by_name("qwen2.5");
    if (!strcmp(arch, "phi3"))   return by_name("phi3");
    if (!strcmp(arch, "gemma3") || !strcmp(arch, "gemma4"))
        return by_name("gemma3");

    // Llama, Mistral and SmolLM2 GGUFs all declare `llama`, so only the model
    // name separates them. Checked before the llama-3 test because "smollm2"
    // and "mistral" never contain it, but a stray "llama" in a merge name
    // would otherwise win.
    if (has(lname, "mistral"))  return by_name("mistral");
    if (has(lname, "smollm"))   return by_name("smollm2");
    if (has(lname, "qwen3"))    return by_name("qwen3");
    if (has(lname, "qwen2"))    return by_name("qwen2.5");
    if (has(lname, "gemma-3") || has(lname, "gemma 3")) return by_name("gemma3");
    if (has(lname, "phi-3") || has(lname, "phi 3"))     return by_name("phi3");
    // "llama 3", "llama-3", "llama3" — and not "tinyllama-1.1b", which is a
    // different model with no published settings of its own
    if (has(lname, "llama 3") || has(lname, "llama-3") || has(lname, "llama3"))
        return by_name("llama3");

    return &PRESETS[0];
}

const sampler_preset *sampler_resolve(sampler *s, const char *arch,
                                      const char *name,
                                      const sampler_override *ov) {
    const sampler_preset *p = sampler_preset_for(arch, name);
    s->temp           = p->temp;
    s->top_p          = p->top_p;
    s->min_p          = p->min_p;
    s->repeat_penalty = p->repeat_penalty;
    s->top_k          = p->top_k;
    if (ov) {
        if (ov->has_temp)           s->temp           = ov->temp;
        if (ov->has_top_p)          s->top_p          = ov->top_p;
        if (ov->has_min_p)          s->min_p          = ov->min_p;
        if (ov->has_top_k)          s->top_k          = ov->top_k;
        if (ov->has_repeat_penalty) s->repeat_penalty = ov->repeat_penalty;
    }
    return p;
}

void sampler_describe(const sampler *s, const sampler_preset *p,
                      char *buf, size_t cap) {
    snprintf(buf, cap,
             "%s (temp %.2f, top_p %.2f, top_k %d, min_p %.2f, repeat_penalty %.2f)",
             p ? p->name : "custom", (double)s->temp, (double)s->top_p,
             s->top_k, (double)s->min_p, (double)s->repeat_penalty);
}
