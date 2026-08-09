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
static bool g_gpu_off = false;

static void ck(int cond, const char *what) {
    if (!cond) { fprintf(stderr, "FAIL: %s\n", what); g_fail = 1; }
}

static model_params base_params(void) {
    model_params p;
    memset(&p, 0, sizeof(p));
    p.gpu_mode  = g_gpu_off ? GPU_OFF : GPU_AUTO;
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
    g_gpu_off = getenv("RUNNER_TEST_GPU_OFF") != NULL;
    f16_init();
    model_params p = base_params();

    // Free VRAM with nothing of ours loaded. The last release of the shared
    // weights has to give this back: a registry entry that is reused forever
    // and never destroyed looks perfectly healthy from inside a run — every
    // load hits the cache — and only shows up as VRAM that never comes back
    // after the final unload, which is exactly what breaks model swap.
    size_t vram_start = 0, vram_total = 0;
    bool have_vram = !g_gpu_off && gpu_mem_info(&vram_start, &vram_total);
    // Say when the device-side leak check is NOT running, and why. It is
    // legitimately CUDA-only — Metal's gpu_mem_info() declines by design,
    // because unified memory has no separate VRAM pool to leak — but a silent
    // skip makes "ok" read identically whether the check ran or not, which is
    // the failure mode this suite has already been bitten by twice.
    if (!have_vram)
        printf("note: device VRAM leak check SKIPPED (no separate VRAM pool "
               "reported — unified memory, CPU-only, or --gpu off)\n");
    // Fixture scale governs whether this check CAN fail: it compares free VRAM
    // across windows of 3 load/free cycles against a 16 MB tolerance, so a
    // leak of the whole model must exceed that. At the 370 KB test.gguf that
    // was 1.1 MB per window — mathematically incapable of firing. Pass
    // ASAN_MODEL (or argv[1]) a model of at least ~64 MB for this to mean
    // anything; the Makefile default now does.
    if (have_vram && vram_total)
        printf("vram leak check armed (free %.1f MB of %.1f MB at start)\n",
               vram_start / 1e6, vram_total / 1e6);

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

    // The HOST half is shared too, not only the device upload: a and b must be
    // two sequences over one parse. Everything else in this file still passes
    // on a build that re-reads and re-converts the whole file per instance —
    // which is what runner did until this change, at 29.7 MB of touched host
    // memory per extra slot on a 7B model, most of it a tokenizer vocabulary
    // the slots never read. So the aliasing is asserted directly.
    ck(a.layers == b.layers, "instances share one layer array");
    ck(a.gf.map == b.gf.map, "instances share one mapping of the file");
    ck(a.layers[0].attn_norm_w == b.layers[0].attn_norm_w,
       "instances share the f32 norm conversions");
    ck(a.out_norm_w == b.out_norm_w, "instances share the output norm");

    // ...and only when the parameters the bind phase reads agree. gpu_mode
    // decides whether a q8 KV cache is accepted, which is settled before the
    // seam, so a differing request must get its own record rather than buffers
    // built under another answer. The sanitizer target already runs GPU_OFF,
    // so it changes the other bind-time key (the requested KV type) instead.
    // A key that ignored either would hand one instance the other's decision.
    {
        model_params distinct = p;
        if (g_gpu_off) distinct.kv_q8 = !p.kv_q8;
        else           distinct.gpu_mode = GPU_OFF;
        model_t d;
        ck(model_load(&d, g_path, &distinct),
           "instance with a different weight-side parameter loads");
        ck(d.layers != a.layers,
           "a differing weight-side parameter gets its own parse");
        model_free(&d);
        ck(model_forward(&a, TOKENS[0], 0) != NULL,
           "freeing an unrelated record leaves the shared one intact");
    }

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
    // ASan cannot see this: it instruments the host heap. The sanitized target
    // explicitly forces CPU because current NVIDIA drivers retain unattributed
    // process-lifetime allocations that LSan cannot suppress narrowly. The
    // ordinary target runs this device-side equivalent of a leak check: load
    // and unload repeatedly and require free VRAM to come back.
    //
    // THE PIN IS LOAD-BEARING, not tidiness. shared_destroy() ends with
    // cuPrimaryCtxRelease(), and releasing a CUDA primary context reclaims
    // EVERY allocation inside it — so when the last model unloads, a missed
    // cuMemFree is handed back by the driver anyway and this check sees
    // nothing. Measured 2026-08-09 on an RTX 3070: deliberately leaking
    // w->weights (4.1 GB, six times, on an 8.59 GB card) produced 0.0 MB of
    // drift and a GREEN result. The gate was structurally incapable of
    // detecting the exact bug it exists for.
    //
    // Holding a second, unrelated model open across both windows keeps the
    // context alive, so the cycled model's allocations are genuinely freed
    // individually and a missed free shows up as drift. test.gguf is tiny and
    // always built, and being a different path it gets its own shared entry —
    // pinning with g_path instead would keep the cycled entry's refcount above
    // zero and destroy nothing at all.
    size_t f0 = 0, f1 = 0, f2 = 0, total = 0;
    model_t pin;
    bool pinned = have_vram && model_load(&pin, "test.gguf", &p);
    if (have_vram && !pinned)
        printf("note: no context pin (test.gguf did not load) — a missed "
               "cuMemFree may be masked by context teardown\n");
    if (have_vram && gpu_mem_info(&f0, &total)) {
        cycles(&p, 3);
        (void)gpu_mem_info(&f1, &total);
        cycles(&p, 3);
        (void)gpu_mem_info(&f2, &total);
        if (pinned) model_free(&pin);
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
