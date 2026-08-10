// The split guard (cuda.c) must actually fire. A load whose file identity is
// unavailable builds privately and re-decides its own CPU/GPU split; when that
// split disagrees with a resident instance of the same path, the guard has to
// say so loudly on stderr — that disagreement is two server slots answering
// one request differently, which is the user-visible half of the 2026-08-04
// defect. This harness manufactures the disagreement deterministically:
// instance A loads normally, instance B loads with the identity injected away
// (RUNNER_TEST_NO_FILE_ID) and --gpu-layers forced to a different split. The
// make target test-split-guard asserts the report appears; a build where the
// guard is deleted runs this harness identically and goes red there — that is
// what keeps the gate falsifiable.
#include "runner.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *g_path = "test.gguf";

#ifdef RUNNER_GPU_CUDA
static model_params base_params(void) {
    model_params p;
    memset(&p, 0, sizeof(p));
    p.gpu_mode  = GPU_AUTO;
    p.n_threads = 1;
    p.n_ctx     = 128;
    p.n_batch   = 8;
    return p;
}
#endif

int main(int argc, char **argv) {
    if (argc > 1) g_path = argv[1];
    if (getenv("RUNNER_TEST_GPU_OFF")) {
        printf("skip: test-split-guard (RUNNER_TEST_GPU_OFF; the guard lives "
               "in the GPU registry)\n");
        return 0;
    }
#ifndef RUNNER_GPU_CUDA
    // The guard under test is `split_guard()` in src/cuda.c — CUDA only. The
    // old skip condition said "no GPU backend", which is not the same thing:
    // Metal IS a GPU backend, so on a Mac this harness ran, manufactured a
    // genuine split disagreement, and then failed looking for a message no
    // CUDA-less build can emit. It never surfaced because a Makefile target
    // collision meant this binary was never built at all (fixed 2026-08-08).
    //
    // Metal having no equivalent guard is a real gap, recorded in the suite
    // plan rather than papered over here: two Metal slots of one server can
    // still re-decide the split differently without a file identity, and
    // nothing says so.
    printf("skip: test-split-guard (guard is CUDA-only; this build has no "
           "CUDA backend)\n");
    return 0;
#else
    f16_init();
    model_params p = base_params();

    model_t a;
    if (!model_load(&a, g_path, &p)) {
        fprintf(stderr, "FAIL: load a\n");
        return 1;
    }
    if (a.gpu_layers < 2) {
        printf("skip: test-split-guard (no GPU backend, or fewer than 2 "
               "offloaded layers to disagree over)\n");
        model_free(&a);
        return 0;
    }

    // B: identity lost, split forced away from A's. The guard must notice.
#ifdef _WIN32
    _putenv("RUNNER_TEST_NO_FILE_ID=1");
#else
    setenv("RUNNER_TEST_NO_FILE_ID", "1", 1);
#endif
    model_params q = base_params();
    q.gpu_layers_override = a.gpu_layers - 1;
    model_t b;
    if (!model_load(&b, g_path, &q)) {
        fprintf(stderr, "FAIL: load b\n");
        model_free(&a);
        return 1;
    }
    fprintf(stderr, "split-guard harness: a=%d b=%d of %d layers\n",
            a.gpu_layers, b.gpu_layers, a.n_layer);
    int fail = 0;
    if (b.gpu_layers == a.gpu_layers) {
        // the override did not take: the guard had nothing to detect, so a
        // green grep downstream would be vacuous — fail instead of skip
        fprintf(stderr, "FAIL: could not force a split disagreement\n");
        fail = 1;
    }
    model_free(&b);
    model_free(&a);
    if (!fail) printf("split disagreement manufactured; see stderr\n");
    return fail;
#endif // RUNNER_GPU_CUDA
}
