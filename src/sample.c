// Token sampling: temperature, top-k, top-p, repeat penalty, optional
// validity-constrained selection (used for JSON mode), and the per-family
// defaults that seed all of the above.
#include "sample.h"
#include "template.h"   // TMPL_* — the detected template outranks the name

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

// True top-k by selection instead of by sorting. After this call c[0..k-1]
// holds the k largest under cand_cmp, in arbitrary order among themselves.
//
// This is exact, not approximate: cand_cmp is a TOTAL order (logit descending,
// then id ascending, and ids are unique), so the k-largest set is unique and
// sorting just those k reproduces the first k of a full sort element for
// element. That is what lets top-k skip the O(n log n) sort of a 262k-entry
// vocabulary for an O(n) partition plus an O(k log k) sort of the survivors.
//
// Median-of-three pivots. Real logit vectors are not adversarial, but the
// quadratic case is cheap to avoid and this runs on every sampled token.
static void select_topk(cand_t *c, int n, int k) {
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        // order lo/mid/hi so c[mid] is the median, then park it at lo as pivot
        if (cand_cmp(&c[mid], &c[lo]) < 0) { cand_t t = c[lo]; c[lo] = c[mid]; c[mid] = t; }
        if (cand_cmp(&c[hi], &c[lo]) < 0)  { cand_t t = c[lo]; c[lo] = c[hi]; c[hi] = t; }
        if (cand_cmp(&c[mid], &c[hi]) < 0) { cand_t t = c[mid]; c[mid] = c[hi]; c[hi] = t; }
        cand_t pivot = c[lo];
        int i = lo, j = hi;
        while (i < j) {
            while (i < j && cand_cmp(&c[j], &pivot) >= 0) j--;
            c[i] = c[j];
            while (i < j && cand_cmp(&c[i], &pivot) <= 0) i++;
            c[j] = c[i];
        }
        c[i] = pivot;
        // the pivot is now final at i; recurse only into the side holding k
        if (i >= k) hi = i - 1;
        else        lo = i + 1;
    }
}

// Sample from `k` candidates carrying temperature-scaled logits in descending
// order: softmax, top-p, min-p, then the roulette pick. `norm_sum` > 0 uses a
// caller-computed softmax denominator (the head fast path passes the whole
// vocabulary's mass so truncation does not change any candidate's share);
// 0 computes it over the k candidates, the historical behavior.
// `r_pre` >= 0 supplies the roulette draw instead of taking one from the RNG.
// The no-filter fast path below has to know the draw before it can size its
// head, and the draw must happen exactly ONCE per sample_pick call whichever
// path ends up running, or a seeded run stops reproducing. `fell_off` reports
// that the cumulative walk never reached `r` — only possible when a caller
// hands in a candidate set that does not carry the whole mass, which is
// precisely the case the fast path has to detect and retreat from.
static int pick_scaled(sampler *s, cand_t *c, int k, float norm_sum,
                       float r_pre, bool *fell_off) {
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
    float r = r_pre >= 0.0f ? r_pre : rng_f32(&s->rng), cum = 0;
    int pick = c[k - 1].id;
    bool hit = false;
    for (int i = 0; i < k; i++) {
        cum += c[i].p;
        if (r < cum) { pick = c[i].id; hit = true; break; }
    }
    if (fell_off) *fell_off = !hit;
    return pick;
}

// RUNNER_SAMPLE_STATS=1 traces why the head fast path was or was not usable.
// Cached: this sits in the per-token path and getenv is not free.
static bool sample_stats(void) {
    static int on = -1;
    if (on < 0) { const char *e = getenv("RUNNER_SAMPLE_STATS"); on = e && *e && strcmp(e, "0") != 0; }
    return on != 0;
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
            // The window is filled from the token stream, whose ids are bounded
            // by the TOKENIZER's vocabulary (or a draft model's) rather than by
            // the length of this logits array. The penalty below writes through
            // `tok`, so an id the logits cannot hold is dropped here — this is
            // the only place recent[] is used as an index, which is why the
            // bound lives at the indexing site instead of in sampler_accept,
            // where n_vocab is not known.
            if (tok < 0 || tok >= n_vocab) continue;
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
    // "No filter at all" -- top-k off, top-p 1.0, min-p off -- used to be the
    // ONE configuration excluded from the fast path, so the case asking for
    // the least work paid a full-vocabulary sort every token. That is what the
    // shipped mistral and gpt-oss presets request, and it cost them 38-48% of
    // decode. It can be served from a head too: the roulette walk stops at the
    // first token whose cumulative mass reaches `r`, so any head carrying more
    // than `r` of the total mass provably contains that token. Draw `r` up
    // front to size the head with; it is passed to whichever path runs, so the
    // RNG still advances exactly once per call and a seeded run is unchanged.
    bool no_filter = want_k >= n_vocab && s->top_p >= 1.0f && s->min_p <= 0.0f;
    float r_pre = -1.0f;
    if (no_filter && !ok && s->temp > 0 && n_vocab >= 4096)
        r_pre = rng_f32(&s->rng);

    // Every combination now has a head criterion, so the fast path is always
    // worth attempting; it still retreats to the full sort when the head
    // cannot be shown to contain the survivors.
    // top-k is served by selection, not by a threshold head. The head could
    // only satisfy `m >= want_k` by widening until it held k entries, and the
    // loosening schedule multiplies a NEGATIVE log-threshold by 4 — one step
    // takes it from p_max/1024 to p_max/e^27, which admits most of the
    // vocabulary and overflows HEAD_CAP. Measured on Qwen3-8B and
    // Ministral-8B: the first head carries 99% of the mass in ~11 entries,
    // fails `m >= 40`, and the next step overflows. Relaxing the criterion is
    // not available either, because pick_scaled renormalises over exactly the
    // k it is given, so serving 11 where 40 were asked changes every
    // probability. Selection gives the true k in O(n) and is exact.
    if (!ok && s->temp > 0 && want_k < n_vocab) {
        cand_t *c = malloc(sizeof(cand_t) * n_vocab);
        if (c) {
            for (int i = 0; i < n_vocab; i++)
                c[i] = (cand_t){ logits[i] / temp, i };
            select_topk(c, n_vocab, want_k);
            qsort(c, want_k, sizeof(cand_t), cand_cmp);
            int pick = pick_scaled(s, c, want_k, 0, r_pre, NULL);
            if (sample_stats()) fprintf(stderr, "[smp select k=%d]", want_k);
            free(c);
            return pick;
        }
        // allocation failed: fall through to the full-sort path below
    }
    if (!ok && s->temp > 0 && n_vocab >= 4096) {
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
                if (overflow) { if (sample_stats()) fprintf(stderr, "[smp overflow m=%d]", m); break; }   // head too broad: full sort below
                // The head must provably contain the whole surviving set:
                // top_k candidates, or top_p of the mass, or -- with no filter
                // at all -- more mass than the draw `r` needs to land in.
                // Strictly greater, because the walk takes the first token
                // with cum > r and equality would leave it just past the end.
                bool enough;
                // want_k < n_vocab is handled by select_topk above and never
                // reaches here.
                if (s->top_p < 1.0f)
                    // the top-p cutoff prefix lies inside a head carrying
                    // top_p of the mass
                    enough = head_mass >= (double)s->top_p * total;
                else if (s->min_p > 0.0f)
                    // min-p keeps everything within min_p of the best token,
                    // which is a LOGIT threshold — so a head cut at or below
                    // log(min_p) provably contains the whole surviving set,
                    // whatever mass it happens to carry
                    enough = t_log <= logf(s->min_p);
                else
                    // nothing filters: the walk stops at the first token whose
                    // cumulative mass reaches r, so a head carrying more than
                    // r of the total provably contains it
                    enough = head_mass > (double)r_pre * total;
                if (!enough) { if (sample_stats()) fprintf(stderr, "[smp short m=%d mass=%.4f need=%.4f]", m, head_mass/total, no_filter ? (double)r_pre : (double)s->top_p); continue; } // loosen the threshold and retry
                qsort(h, m, sizeof(cand_t), cand_cmp);
                int k = m < want_k ? m : want_k;
                // top_k active: softmax over the k survivors (historical
                // semantics); top_k off: normalize by the whole vocabulary
                bool fell_off = false;
                int pick = pick_scaled(s, h, k,
                                       want_k < n_vocab ? 0 : (float)total,
                                       r_pre, &fell_off);
                // The mass check is done in double while the walk accumulates
                // in float, so a draw sitting within an ulp of the head's edge
                // can still run off it. Retreat to the exact full sort rather
                // than return the head's last token, which is what the walk
                // falls back to and would be silently wrong.
                if (fell_off) { if (sample_stats()) fprintf(stderr, "[smp felloff]"); free(h); break; }
                if (sample_stats()) fprintf(stderr, "[smp fast m=%d]", m);
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

    int pick = pick_scaled(s, c, k, 0, r_pre, NULL);
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

    // OpenAI's gpt-oss model card: temperature 1.0 and top_p 1.0, and it
    // publishes no repetition penalty — which matters beyond taste here,
    // because the generic fallback's 1.10 silently diverges this model from
    // every reference implementation that defaults to 1.0.
    { "gpt-oss", "OpenAI gpt-oss model card (temperature 1.0, top_p 1.0)",
      1.00f, 1.00f, 0.00f, 1.00f, 0 },

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

    // Mistral's Nemo card is explicit that this family departs from other
    // Mistral models: "Unlike previous Mistral models, Mistral Nemo requires
    // smaller temperatures. We recommend to use a temperature of 0.3." (their
    // sample code runs 0.35). Everything else inherits the mistral preset.
    { "mistral-nemo", "Mistral-Nemo-Instruct-2407 model card (temperature 0.3)",
      0.30f, 1.00f, 0.00f, 1.10f, 0 },

    // OpenLLM-France Lucie-7B-Instruct generation_config.json:
    // temperature 0.6, top_p 0.9 (do_sample true; nothing else stated).
    { "lucie", "Lucie-7B-Instruct generation_config.json (OpenLLM-France)",
      0.60f, 0.90f, 0.00f, 1.10f, 0 },

    // BSC salamandra-7b-instruct generation_config.json: temperature 0.6,
    // repetition_penalty 1.2 (no top_p/top_k stated, so both are off).
    { "salamandra", "salamandra-7b-instruct generation_config.json (BSC-LT)",
      0.60f, 1.00f, 0.00f, 1.20f, 0 },

    // OpenGPT-X Teuken-7B-instruct model card usage example:
    // temperature 0.7, top_p 0.95. An example rather than a stated best
    // practice — the weakest citation grade here, but it is what the vendor
    // publishes.
    { "teuken", "Teuken-7B-instruct model card usage example (OpenGPT-X)",
      0.70f, 0.95f, 0.00f, 1.10f, 0 },

    // Gridcore Syntetik (repo xyntetik-shade, model project Shade — the
    // preset name is a published-artifact contract and stays "gridcore"):
    // the suite's own from-scratch
    // decoder that compiles requests into auditable execution contracts.
    // A contract compiler wants deterministic, reproducible output under
    // schema enforcement — not creative sampling — so the preset is greedy
    // with no repeat penalty (a penalty on a constrained JSON grammar only
    // distorts a distribution the schema already pins). PROVISIONAL: aligned
    // to Syntetik's stated "auditable/deterministic" design; the model team
    // (xyntetik-shade) should confirm or override once training settles.
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

// Combined preset-matching identity: general.name plus the load path's
// basename. Quantizer metadata is unreliable — a real community conversion
// shipped general.name "snapshots" (the converter's HF cache directory) —
// and the filename usually still carries the family name. Substring matching
// over both is strictly more informative and no less safe.
void sampler_ident(const char *name, const char *path, char *buf, size_t n) {
    const char *base = NULL;
    if (path) {
        const char *slash = strrchr(path, '/');
        const char *bslash = strrchr(path, '\\');
        base = bslash > slash ? bslash + 1 : slash ? slash + 1 : path;
    }
    snprintf(buf, n, "%s %s", name ? name : "", base ? base : "");
}

const sampler_preset *sampler_preset_for(const char *arch, const char *name,
                                        int tmpl) {
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
    if (!strcmp(arch, "gpt-oss")) return by_name("gpt-oss");
    if (!strcmp(arch, "qwen3"))  return by_name("qwen3");
    if (!strcmp(arch, "qwen2"))  return by_name("qwen2.5");
    if (!strcmp(arch, "phi3"))   return by_name("phi3");
    if (!strcmp(arch, "gemma3") || !strcmp(arch, "gemma4"))
        return by_name("gemma3");

    // The DETECTED TEMPLATE, where there is one, outranks every name test
    // below. Both signals answer "which family is this", but the template is
    // read out of the checkpoint while the name is a label a re-quantiser can
    // change; template.c already distinguishes the three Mistral framings by
    // their own text, and sample.c used to re-decide the same question from a
    // substring and disagree. Nemo is the case that bites: its model card
    // calls out temperature 0.3 as a departure from other Mistral models, so a
    // Nemo export whose name lost either "mistral" or "nemo" got the plain
    // Mistral preset at 0.7 while being rendered with the Nemo template.
    if (tmpl == TMPL_MISTRAL_NEMO) return by_name("mistral-nemo");
    if (tmpl == TMPL_MISTRAL || tmpl == TMPL_MISTRAL_V1)
        return by_name("mistral");

    // Llama, Mistral and SmolLM2 GGUFs all declare `llama`, so only the model
    // name separates them. Checked before the llama-3 test because "smollm2"
    // and "mistral" never contain it, but a stray "llama" in a merge name
    // would otherwise win.
    // European families that declare `llama` and are separated by name only.
    // Nemo needs BOTH tokens: "nemotron" (NVIDIA) contains "nemo" and must
    // not land on Mistral's temperature recommendation.
    if (has(lname, "mistral") && has(lname, "nemo"))
        return by_name("mistral-nemo");
    if (has(lname, "lucie"))      return by_name("lucie");
    if (has(lname, "salamandra")) return by_name("salamandra");
    if (has(lname, "teuken"))     return by_name("teuken");
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
                                      const char *name, int tmpl,
                                      const sampler_override *ov) {
    const sampler_preset *p = sampler_preset_for(arch, name, tmpl);
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
    // At temp <= 0 every shaping knob is bypassed (greedy argmax; see
    // sample_pick) — say so, or the banner reads as if repeat_penalty
    // applies and a reviewer comparing greedy output against another
    // engine burns time ruling it out. That happened; this line is the fix.
    if (s->temp <= 0) {
        snprintf(buf, cap, "%s (temp %.2f — greedy argmax; top_p/top_k/"
                 "min_p/repeat_penalty inactive)",
                 p ? p->name : "custom", (double)s->temp);
        return;
    }
    snprintf(buf, cap,
             "%s (temp %.2f, top_p %.2f, top_k %d, min_p %.2f, repeat_penalty %.2f)",
             p ? p->name : "custom", (double)s->temp, (double)s->top_p,
             s->top_k, (double)s->min_p, (double)s->repeat_penalty);
}
