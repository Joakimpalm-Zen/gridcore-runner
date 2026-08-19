// Generalized MoE router: every knob, checked against the dense oracle in
// LOGIT space.
//
// scripts/make-test-moe.py builds one fixture per router knob, each arranged
// so the routed FFN is mathematically identical to the dense model's FFN — a
// knob that is ignored, applied twice, or applied to the wrong quantity breaks
// that identity. tests/test_moe.py already compares the generated TEXT, which
// is the right gate for knobs that change *which* expert is selected.
//
// It is the wrong gate for knobs that change expert WEIGHTS. Greedy text is an
// argmax, and on a small random fixture a 2x change in the FFN contribution
// often does not move it, so a weight bug passes a text comparison while being
// plainly wrong. That is not hypothetical: expert_weights_norm and all three
// alternative gating functions were verified to pass the text oracle on a
// binary that does not implement them at all.
//
// So this gate compares logits, exactly as test_kv_tol / test_tc_tol /
// test_moe_tol do for the same reason. The fixtures are exact by construction
// (all weights F32, scalings are powers of two), so the bound here is EQUALITY
// rather than a tolerance — anything else means a knob moved the arithmetic.

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "runner.h"

#define PROMPT_LEN 6
#define MAX_VOCAB  (1 << 20)

// The knob fixtures, by the suffix make-test-moe.py writes.
static const char *VARIANTS[] = {
    "gsigmoid",   // expert_gating_func = 2  (sigmoid)
    "gsqrtsp",    // expert_gating_func = 4  (sqrt softplus)
    "gsmw",       // expert_gating_func = 3  (softmax over selected weights)
    "gnonorm",    // expert_weights_norm = false
    "gscale",     // expert_weights_scale = 2
    "gprobsb",    // exp_probs_b steers selection only
    "ggroup",     // group-limited top-k
};
#define N_VARIANTS ((int)(sizeof(VARIANTS) / sizeof(*VARIANTS)))

// Feed a fixed prompt and return the final-position logits. The caller owns
// the buffer. n_vocab is written on success; router_fp, when non-NULL, receives
// a fingerprint of every layer's ffn_gate_inp weights (see the sensitivity
// section for why that has to be checked).
static float *run_logits(const char *path, int *n_vocab_out,
                         uint64_t *router_fp) {
    model_t m;
    model_params p = {0};
    p.n_ctx = 64;
    p.gpu_mode = GPU_OFF;          // this gate is about host routing arithmetic
    if (!model_load(&m, path, &p)) {
        fprintf(stderr, "moe-router: cannot load %s\n", path);
        return NULL;
    }
    if (router_fp) {
        uint64_t h = 1469598103934665603ULL;              // FNV-1a
        for (int l = 0; l < m.n_layer; l++) {
            const gguf_tensor *r = m.layers[l].ffn_gate_inp;
            if (!r || !r->data) continue;
            const uint8_t *b = (const uint8_t *)r->data;
            for (uint64_t i = 0; i < r->nbytes; i++)
                h = (h ^ b[i]) * 1099511628211ULL;
        }
        *router_fp = h;
    }
    int32_t toks[PROMPT_LEN];
    for (int i = 0; i < PROMPT_LEN; i++) toks[i] = 3 + i;   // any stable ids
    float *lg = model_forward_batch(&m, toks, PROMPT_LEN, 0, true);
    if (!lg || m.n_vocab < 1 || m.n_vocab > MAX_VOCAB) {
        fprintf(stderr, "moe-router: no logits from %s\n", path);
        model_free(&m);
        return NULL;
    }
    float *out = malloc(sizeof(float) * (size_t)m.n_vocab);
    if (out) memcpy(out, lg, sizeof(float) * (size_t)m.n_vocab);
    *n_vocab_out = m.n_vocab;
    model_free(&m);
    return out;
}

int main(int argc, char **argv) {
    const char *stem = argc > 1 ? argv[1] : "test-moe";
    char path[512];

    snprintf(path, sizeof(path), "%s.dense.gguf", stem);
    int nv_dense = 0;
    float *dense = run_logits(path, &nv_dense, NULL);
    if (!dense) return 1;

    int failures = 0;
    for (int v = 0; v < N_VARIANTS; v++) {
        snprintf(path, sizeof(path), "%s.%s.gguf", stem, VARIANTS[v]);
        int nv = 0;
        float *got = run_logits(path, &nv, NULL);
        if (!got) { failures++; continue; }
        if (nv != nv_dense) {
            fprintf(stderr, "moe-router: %s vocab %d != dense %d\n",
                    VARIANTS[v], nv, nv_dense);
            free(got); failures++; continue;
        }
        double worst = 0.0;
        int worst_i = -1;
        for (int i = 0; i < nv; i++) {
            double d = fabs((double)got[i] - (double)dense[i]);
            if (d > worst) { worst = d; worst_i = i; }
        }
        // Exact: the fixtures differ from dense only by arithmetic the knob is
        // supposed to cancel out. A non-zero worst here IS the bug.
        if (worst != 0.0) {
            fprintf(stderr, "moe-router: %-9s FAIL  max|dlogit|=%.9g at %d "
                    "(dense %.9g, got %.9g)\n", VARIANTS[v], worst, worst_i,
                    (double)dense[worst_i], (double)got[worst_i]);
            failures++;
        } else {
            printf("  %-9s logits identical to dense\n", VARIANTS[v]);
        }
        free(got);
    }
    free(dense);

    // --- gating sensitivity. The oracle fixtures above cannot tell one gating
    // function from another (a zero router scores every expert equally, and
    // renormalization then hides the difference), so each alternative is also
    // run on the ROUTED fixture and required to disagree with the softmax
    // fixture an unimplemented gating function would fall through to.
    //
    // That comparison only means something if the two files differ ONLY in the
    // gating metadata. They did not: make-test-moe.py drew a fresh random
    // router per fixture, so the logits differed because the ROUTERS differed,
    // and a binary with the whole gating switch mutated to softmax passed this
    // section (measured 2026-08-19, before the fixtures were rebuilt). The
    // router fingerprint below is what stops that from being recreated: if the
    // fixtures stop sharing a router, this fails loudly instead of passing
    // vacuously.
    //
    // rsmw's baseline is rsoftnn, not rsoft, and both declare
    // expert_weights_norm = false. Gating 3 is "softmax over the SELECTED
    // weights", which under the default weight normalization is the SAME
    // arithmetic as plain softmax followed by that normalization — the knob is
    // undetectable there, by construction rather than by bug. With the norm off
    // gating 3 still normalizes and softmax does not, so the fixture pins the
    // knob where it has an observable effect.
    static const struct { const char *tag, *base; } GATED[] = {
        { "rsigmoid", "rsoft"   },   // expert_gating_func = 2
        { "rsqrtsp",  "rsoft"   },   // expert_gating_func = 4
        { "rsmw",     "rsoftnn" },   // expert_gating_func = 3, norm off
    };
    for (int v = 0; v < (int)(sizeof(GATED) / sizeof(*GATED)); v++) {
        uint64_t fp_base = 0, fp_got = 0;
        int nv_base = 0, nv = 0;
        snprintf(path, sizeof(path), "%s.%s.gguf", stem, GATED[v].base);
        float *base = run_logits(path, &nv_base, &fp_base);
        if (!base) { failures++; continue; }
        snprintf(path, sizeof(path), "%s.%s.gguf", stem, GATED[v].tag);
        float *got = run_logits(path, &nv, &fp_got);
        if (!got || nv != nv_base) { failures++; free(got); free(base); continue; }
        if (fp_got != fp_base) {
            fprintf(stderr, "moe-router: %-9s FAIL  its router differs from "
                    "%s's — this comparison cannot attribute anything to the "
                    "gating function\n", GATED[v].tag, GATED[v].base);
            failures++;
        } else {
            double worst = 0.0;
            for (int i = 0; i < nv; i++) {
                double d = fabs((double)got[i] - (double)base[i]);
                if (d > worst) worst = d;
            }
            if (worst == 0.0) {
                fprintf(stderr, "moe-router: %-9s FAIL  identical to %s on the "
                        "same router — the gating function is not being "
                        "applied\n", GATED[v].tag, GATED[v].base);
                failures++;
            } else {
                printf("  %-9s differs from %-7s on one shared router "
                       "(max|dlogit|=%.6g)\n", GATED[v].tag, GATED[v].base,
                       worst);
            }
        }
        free(got);
        free(base);
    }

    if (failures) {
        fprintf(stderr, "moe-router: %d checks FAILED (%d dense-oracle "
                "knobs + %d gating comparisons)\n", failures, N_VARIANTS,
                (int)(sizeof(GATED) / sizeof(*GATED)));
        return 1;
    }
    printf("moe-router: ok (%d knobs, exact against the dense oracle)\n",
           N_VARIANTS);
    return 0;
}
