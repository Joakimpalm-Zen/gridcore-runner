// Shared model weights (Phase 5).
//
// Two model_t values loaded from the same file are two *sequences* over one set
// of weights. This test pins the three properties that makes that safe to rely
// on, through the public model API only:
//
//   1. they agree — same CPU/GPU split, same logits for the same input, so a
//      request cannot get a different answer depending on which slot took it;
//   2. they are isolated — interleaving two sequences leaves each one's output
//      exactly what it would have been running alone, which is the property a
//      shared KV cache would destroy;
//   3. they free exactly once, in any order, including a load that comes after
//      the last release (run under ASan, which is where a refcount that frees
//      early or twice actually shows up).
//
// The default model is the generated test.gguf, which is small but does go
// through the real CUDA backend when one is present.
#include "runner.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *g_path = "test.gguf";
static int g_fail = 0;

static void ck(int cond, const char *what) {
    if (!cond) { fprintf(stderr, "FAIL: %s\n", what); g_fail = 1; }
}

static model_params base_params(void) {
    model_params p;
    memset(&p, 0, sizeof(p));
    p.gpu_mode  = GPU_AUTO;
    p.n_threads = 1;
    p.n_ctx     = 128;
    p.n_batch   = 8;
    return p;
}

// SEQ is short enough to stay inside the smallest supported context and long
// enough that a clobbered KV cache changes the answer rather than hiding in
// one token of attention.
enum { SEQ = 6 };
static const int32_t TOKENS[SEQ] = { 1, 2, 3, 4, 5, 6 };

// One load/free cycle is two instances sharing one upload, freed in the order
// that outlives its owner: x pays for the weights, y keeps decoding after x is
// gone, and only y's release may destroy them.
static void cycles(const model_params *p, int n) {
    for (int i = 0; i < n; i++) {
        model_t x, y;
        if (!model_load(&x, g_path, p)) { ck(0, "cycle load x"); return; }
        if (!model_load(&y, g_path, p)) { ck(0, "cycle load y"); model_free(&x); return; }
        ck(model_forward(&x, TOKENS[0], 0) != NULL, "cycle forward x");
        ck(model_forward(&y, TOKENS[0], 0) != NULL, "cycle forward y");
        model_free(&x);
        ck(model_forward(&y, TOKENS[1], 1) != NULL,
           "survivor forwards after the instance that uploaded the weights is freed");
        model_free(&y);
    }
}

static float *copy_logits(const float *src, int n) {
    float *dst = malloc(sizeof(float) * (size_t)n);
    if (dst) memcpy(dst, src, sizeof(float) * (size_t)n);
    return dst;
}

int main(int argc, char **argv) {
    if (argc > 1) g_path = argv[1];
    f16_init();
    model_params p = base_params();

    // Free VRAM with nothing of ours loaded. The last release of the shared
    // weights has to give this back: a registry entry that is reused forever
    // and never destroyed looks perfectly healthy from inside a run — every
    // load hits the cache — and only shows up as VRAM that never comes back
    // after the final unload, which is exactly what breaks model swap.
    size_t vram_start = 0, vram_total = 0;
    bool have_vram = gpu_mem_info(&vram_start, &vram_total);

    // --- reference: one model, one sequence, nothing else running ---
    model_t ref;
    if (!model_load(&ref, g_path, &p)) {
        fprintf(stderr, "FAIL: cannot load %s\n", g_path);
        return 1;
    }
    int n_vocab = ref.n_vocab;
    int ref_gpu_layers = ref.gpu_layers;
    bool ref_on_gpu = ref.gpu != NULL;
    float *want[SEQ] = { 0 };
    for (int t = 0; t < SEQ; t++) {
        float *lg = model_forward(&ref, TOKENS[t], t);
        ck(lg != NULL, "reference forward returned logits");
        if (!lg) return 1;
        want[t] = copy_logits(lg, n_vocab);
        ck(want[t] != NULL, "reference logits copied");
        if (!want[t]) return 1;
    }
    model_free(&ref);

    // --- two instances of the same file, decoded interleaved ---
    model_t a, b;
    ck(model_load(&a, g_path, &p), "second instance loads");
    ck(model_load(&b, g_path, &p), "third instance loads");

    // A second loader adopts the first one's split rather than re-deciding
    // against a VRAM figure the first one has already reduced. Slots that
    // disagree here would run different numbers of layers on the GPU and could
    // answer the same request differently.
    ck(a.gpu_layers == ref_gpu_layers && b.gpu_layers == ref_gpu_layers,
       "every instance uses the same CPU/GPU split");
    ck((a.gpu != NULL) == ref_on_gpu && (b.gpu != NULL) == ref_on_gpu,
       "every instance reaches the same backend");

    for (int t = 0; t < SEQ; t++) {
        // step a, then b, at the same position: if they shared KV rows or
        // activation scratch, the second write would poison the first's history
        float *la = model_forward(&a, TOKENS[t], t);
        ck(la != NULL, "interleaved forward (a) returned logits");
        if (!la) return 1;
        float *la_copy = copy_logits(la, n_vocab);
        float *lb = model_forward(&b, TOKENS[t], t);
        ck(lb != NULL, "interleaved forward (b) returned logits");
        if (!lb || !la_copy) return 1;

        int bad_a = 0, bad_b = 0;
        for (int i = 0; i < n_vocab; i++) {
            if (la_copy[i] != want[t][i]) bad_a++;
            if (lb[i] != want[t][i]) bad_b++;
        }
        if (bad_a || bad_b) {
            fprintf(stderr, "FAIL: step %d diverged from the solo reference "
                    "(a: %d/%d logits, b: %d/%d)\n",
                    t, bad_a, n_vocab, bad_b, n_vocab);
            g_fail = 1;
        }
        free(la_copy);
    }

    // --- teardown in the order that exercises the refcount ---
    // b first (a still holds the shared weights), then a (last reference), then
    // a fresh load that has to rebuild what the last release destroyed.
    model_free(&b);
    float *after = model_forward(&a, TOKENS[0], 0);
    ck(after != NULL, "surviving instance still decodes after its peer is freed");
    model_free(&a);

    model_t c;
    ck(model_load(&c, g_path, &p), "reload after the last instance was freed");
    float *lc = model_forward(&c, TOKENS[0], 0);
    ck(lc != NULL, "reloaded instance decodes");
    if (lc) {
        int bad = 0;
        for (int i = 0; i < n_vocab; i++) if (lc[i] != want[0][i]) bad++;
        ck(bad == 0, "reloaded instance matches the original logits");
    }
    model_free(&c);

    // --- device memory is returned, exactly once, every cycle ---
    //
    // ASan cannot see this: it instruments the host heap, and on this box the
    // CUDA driver will not even enumerate devices under ASan, so the sanitized
    // run of this test is a CPU run. The device-side equivalent of a leak check
    // is the driver's own accounting — load and unload repeatedly and require
    // free VRAM to come back. A missing MemFree makes this drift down; a double
    // free would already have aborted the run.
    size_t f0 = 0, f1 = 0, f2 = 0, total = 0;
    if (have_vram && gpu_mem_info(&f0, &total)) {
        cycles(&p, 3);
        (void)gpu_mem_info(&f1, &total);
        cycles(&p, 3);
        (void)gpu_mem_info(&f2, &total);
        // free VRAM is a device-wide figure, so another process starting up
        // mid-test can move it. A leak cannot be confused with that: it repeats,
        // in the same direction, in every window. Two consecutive windows both
        // losing memory is the signal; one is noise.
        long long d1 = (long long)f0 - (long long)f1;
        long long d2 = (long long)f1 - (long long)f2;
        const long long TOL = 16 * 1024 * 1024;
        if (d1 > TOL && d2 > TOL) {
            fprintf(stderr, "FAIL: VRAM falls every cycle window (%.1f then "
                    "%.1f MB over 3 load/free cycles each) — an allocation is "
                    "not being freed\n", d1 / 1e6, d2 / 1e6);
            g_fail = 1;
        } else {
            printf("vram stable across load/free cycles (%.1f / %.1f MB)\n",
                   d1 / 1e6, d2 / 1e6);
        }
    }

    for (int t = 0; t < SEQ; t++) free(want[t]);

    // everything this test loaded is now freed: VRAM must be back where it
    // started, or the last release did not actually destroy the shared upload
    size_t vram_end = 0;
    if (have_vram && gpu_mem_info(&vram_end, &vram_total)) {
        long long held = (long long)vram_start - (long long)vram_end;
        const long long TOL = 64 * 1024 * 1024;   // driver context bookkeeping
        if (held > TOL) {
            fprintf(stderr, "FAIL: %.1f MB of VRAM still held after every model "
                    "was freed (free %.1f -> %.1f MB). Either an allocation is "
                    "leaked or the shared weights are never destroyed. (If "
                    "another process started on this GPU mid-run, re-run.)\n",
                    held / 1e6, vram_start / 1e6, vram_end / 1e6);
            g_fail = 1;
        } else {
            printf("vram fully returned after unload (%.1f MB outstanding)\n",
                   held / 1e6);
        }
    }

    if (g_fail) return 1;
    printf("shared weight tests ok (%s, %s)\n", g_path,
           ref_on_gpu ? "gpu backend" : "cpu only");
    return 0;
}
