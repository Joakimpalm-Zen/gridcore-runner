// Adaptation D3 gate: the backward pass against finite differences.
//
// The RED-first equivalent for gradients: for every adapted projection slot
// (attention q/k/v/output + ffn gate/up/down, both layers of the fixture,
// A and B sides), perturb sampled coordinates and compare the analytic
// gradient against the central difference of the ACTUAL loss the engine
// computes. A backward pass that used the wrong input, the wrong side of a
// residual, a wrong softmax jacobian or a wrong rope adjoint lands orders of
// magnitude outside the float-forward FD noise this tolerates.
//
// Also pins gradient DETERMINISM — two identical backward calls produce
// byte-identical gradient buffers and bit-equal losses — the property the
// whole "same data + same seed -> byte-identical adapter" claim (D5) stands
// on.
//
//     ./test-lora-grad test.gguf test-lora.full.gguf
#include "runner.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { \
    fprintf(stderr, "FAIL: "); fprintf(stderr, __VA_ARGS__); \
    fprintf(stderr, "\n"); g_fail = 1; } } while (0)

enum { T = 14 };

static double run_backward(model_t *m, const int32_t *toks) {
    double loss = 0;
    model_lora_grad_zero(m);
    if (!model_lora_backward(m, toks, T, &loss)) {
        CHECK(0, "model_lora_backward failed");
        return 0;
    }
    return loss;
}

int main(int argc, char **argv) {
    const char *base = argc > 1 ? argv[1] : "test.gguf";
    const char *adapter = argc > 2 ? argv[2] : "test-lora.full.gguf";
    f16_init();
    model_params p;
    memset(&p, 0, sizeof(p));
    p.gpu_mode = GPU_OFF;
    p.n_threads = 2;
    p.n_ctx = 64;
    p.n_batch = 32;
    model_t m;
    memset(&m, 0, sizeof(m));
    if (!model_load(&m, base, &p)) {
        fprintf(stderr, "FAIL: cannot load %s\n", base);
        return 1;
    }
    if (!model_lora_load(&m, adapter, 1.0f)) {
        fprintf(stderr, "FAIL: cannot load adapter %s\n", adapter);
        return 1;
    }
    int32_t toks[T];
    for (int i = 0; i < T; i++) toks[i] = 3 + (i * 7) % (m.n_vocab - 4);

    // ---- determinism: two runs, byte-identical grads and bit-equal loss
    double l1 = run_backward(&m, toks);
    // snapshot every grad buffer
    enum { MAX_BUF = 64 };
    float *snap[MAX_BUF]; int snap_n[MAX_BUF]; int nsnap = 0;
    int lay[MAX_BUF], slo[MAX_BUF], whi[MAX_BUF];
    for (int l = 0; l < m.n_layer; l++)
        for (int s = 0; s < 7; s++)
            for (int w = 0; w < 2; w++) {
                int cnt = 0;
                float *g = model_lora_gradbuf(&m, l, s, w, &cnt);
                if (!g || nsnap >= MAX_BUF) continue;
                snap[nsnap] = malloc(sizeof(float) * (size_t)cnt);
                memcpy(snap[nsnap], g, sizeof(float) * (size_t)cnt);
                snap_n[nsnap] = cnt;
                lay[nsnap] = l; slo[nsnap] = s; whi[nsnap] = w;
                nsnap++;
            }
    CHECK(nsnap >= 14, "expected adapters on every slot, saw %d buffers", nsnap);
    double l2 = run_backward(&m, toks);
    CHECK(l1 == l2, "loss not deterministic: %.17g vs %.17g", l1, l2);
    for (int i = 0; i < nsnap; i++) {
        int cnt = 0;
        float *g = model_lora_gradbuf(&m, lay[i], slo[i], whi[i], &cnt);
        CHECK(g && cnt == snap_n[i] &&
              memcmp(g, snap[i], sizeof(float) * (size_t)cnt) == 0,
              "grad buffer %d not byte-deterministic", i);
    }
    printf("ok: loss %.6f, %d grad buffers byte-deterministic across runs\n",
           l1, nsnap);

    // ---- finite differences on sampled coordinates of every buffer
    double worst = 0, cos_num = 0, cos_a = 0, cos_f = 0;
    int checked = 0;
    for (int i = 0; i < nsnap; i++) {
        int cnt = 0;
        float *theta = model_lora_param(&m, lay[i], slo[i], whi[i], &cnt);
        CHECK(theta != NULL, "param buffer missing for grad buffer %d", i);
        if (!theta) continue;
        int coords[3] = { 0, cnt / 2, cnt - 1 };
        for (int c = 0; c < 3; c++) {
            int k = coords[c];
            float save = theta[k];
            // eps large enough to average across the f16 rounding of the
            // cached K/V rows the loss actually flows through (the forward
            // quantizes v/k to ~5e-4 relative; a step well above that sees
            // the smooth loss, a smaller one sees the staircase)
            float eps = 1e-2f;
            theta[k] = save + eps;
            double lp = run_backward(&m, toks);
            theta[k] = save - eps;
            double lm = run_backward(&m, toks);
            theta[k] = save;
            double fd = (lp - lm) / (2.0 * (double)eps);
            double an = (double)snap[i][k];
            // two-scale escape: the f16-KV rounding staircases the finite
            // difference non-monotonically in eps (probed: fd oscillates
            // AROUND the analytic value and converges to it as eps grows —
            // a real jacobian bias would persist instead). A coordinate that
            // misses at the small step gets one retry at 8x, where the
            // staircase averages out.
            if (fabs(fd - an) > 1e-2 &&
                fabs(fd - an) > 0.05 * (fabs(an) > fabs(fd) ? fabs(an)
                                                            : fabs(fd))) {
                float e2 = 8e-2f;
                theta[k] = save + e2;
                lp = run_backward(&m, toks);
                theta[k] = save - e2;
                lm = run_backward(&m, toks);
                theta[k] = save;
                fd = (lp - lm) / (2.0 * (double)e2);
            }
            double mag = fabs(fd) > fabs(an) ? fabs(fd) : fabs(an);
            double rel = mag > 0.02 ? fabs(fd - an) / mag : fabs(fd - an);
            if (rel > worst && fabs(fd - an) > 1e-2) worst = rel;
            // two-sided bound: 5% relative for well-resolved coordinates,
            // or inside the measured FD noise floor for tiny gradients. The
            // floor is REAL and was probed, not assumed: every layer-0
            // parameter's loss path crosses the f16 rounding of layer-1's
            // cached K/V, which staircases the finite difference by up to
            // ~7e-3 absolute at eps=1e-2 — and doubling eps halves the
            // discrepancy while the analytic value stands still, the noise
            // signature. A wrong jacobian term produces O(1) relative error
            // on the LARGE coordinates, which the relative arm still
            // catches; the cosine below guards the aggregate.
            CHECK(rel <= 0.05 || fabs(fd - an) <= 1e-2,
                  "layer %d slot %d %s[%d]: fd %.6g vs analytic %.6g "
                  "(rel %.3g)", lay[i], slo[i], whi[i] ? "B" : "A", k, fd,
                  an, rel);
            cos_num += fd * an; cos_a += an * an; cos_f += fd * fd;
            checked++;
        }
    }
    double cosine = cos_num / (sqrt(cos_a) * sqrt(cos_f) + 1e-30);
    CHECK(cosine > 0.999, "fd/analytic cosine %.6f over %d coords", cosine,
          checked);
    printf("ok: %d FD coordinates across %d buffers, worst rel err %.4g, "
           "cosine %.6f\n", checked, nsnap, worst, cosine);

    for (int i = 0; i < nsnap; i++) free(snap[i]);
    model_free(&m);
    printf(g_fail ? "lora-grad: FAILED\n" : "lora-grad: ok\n");
    return g_fail;
}
