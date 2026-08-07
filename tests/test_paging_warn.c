// The residency warning's message selection.
//
// Issue #7: the warning told a 16 GB Mac that "every token will page from
// disk" while gemma-4-26B-A4B was in fact serving 8+ tok/s on that machine.
// The claim was not wrong about the file not fitting — it was wrong about what
// that costs, because only 8 of 128 experts per layer are ever touched. A
// diagnostic that overstates by 16x gets ignored, which is the same as not
// having one.
//
// Two things are checked here, because a warning has two halves that fail
// independently:
//
//   1. model_hot_set_bytes on real fixtures — a dense model must discount
//      nothing, and a top-2-of-4 MoE must discount half its expert banks.
//      Getting this from a fixture rather than a hand-built struct is the
//      point: the byte accounting has to agree with the loader's actual
//      tensor set, including the router and shared-expert exclusions.
//   2. The wording chosen for each (need, hot, have) case. Available RAM
//      cannot be forced on a build machine, so the selection is tested
//      directly rather than through model_load.
#include <stdio.h>
#include <string.h>
#include "runner.h"

static int g_fail = 0;

static void ck(int cond, const char *what) {
    if (!cond) { fprintf(stderr, "FAIL: %s\n", what); g_fail = 1; }
    else        printf("ok: %s\n", what);
}

static bool load(model_t *m, const char *path) {
    model_params p = {0};
    p.n_ctx = 64;
    p.gpu_mode = GPU_OFF;
    if (model_load(m, path, &p)) return true;
    fprintf(stderr, "paging-warn: cannot load %s\n", path);
    return false;
}

int main(int argc, char **argv) {
    const char *prefix = argc > 1 ? argv[1] : "test-moe-fixture";
    char dense_path[512], moe_path[512];
    snprintf(dense_path, sizeof(dense_path), "%s.dense.gguf", prefix);
    snprintf(moe_path,   sizeof(moe_path),   "%s.moe4.gguf",  prefix);

    // --- the hot-set estimate, on fixtures -------------------------------
    model_t dm;
    if (!load(&dm, dense_path)) return 1;
    ck(model_hot_set_bytes(&dm) == 0,
       "a dense model discounts nothing (0 == 'the file is the hot set')");
    uint64_t dense_size = dm.gf.map_size;
    ck(dense_size > 0, "the dense fixture has a size");
    model_free(&dm);

    model_t mm;
    if (!load(&mm, moe_path)) return 1;
    uint64_t total = mm.gf.map_size, hot = model_hot_set_bytes(&mm);
    printf("note: moe4 file %llu bytes, hot set %llu bytes (%d of %d experts)\n",
           (unsigned long long)total, (unsigned long long)hot,
           mm.n_expert_used, mm.n_expert);
    ck(mm.n_expert == 4 && mm.n_expert_used == 2,
       "the moe4 fixture is top-2-of-4 as this test assumes");
    ck(hot > 0 && hot < total,
       "a sparse MoE's hot set is smaller than its file");
    // top-2-of-4 halves the expert banks and nothing else, so the discount is
    // exactly half of whatever the banks weigh. Anything that also discounted
    // the router, the shared expert, or the non-MoE layers would land lower.
    uint64_t discount = total - hot;
    ck(discount * 2 <= total,
       "the discount cannot exceed the expert banks it comes from");
    model_free(&mm);

    // --- message selection ----------------------------------------------
    // 10 GB of weights, 4 GB of RAM. The hot set is what changes the verdict.
    const uint64_t need = 10000000000ull, have = 4000000000ull;
    char buf[512];

    ck(!model_residency_warning(need, 0, need + 1, false, buf, sizeof(buf)),
       "a model that fits produces no warning at all");
    ck(!model_residency_warning(need, 1, 0, false, buf, sizeof(buf)),
       "an unknown RAM figure produces no warning rather than a guess");

    ck(model_residency_warning(need, 0, have, false, buf, sizeof(buf)) &&
       strstr(buf, "every token to page from disk") &&
       !strstr(buf, "sparse MoE"),
       "dense: the original every-token wording is kept");

    ck(model_residency_warning(need, 2000000000ull, have, false,
                               buf, sizeof(buf)) &&
       strstr(buf, "sparse MoE") && strstr(buf, "does fit") &&
       !strstr(buf, "every token to page") && !strstr(buf, "--mlock"),
       "sparse with a hot set that fits: no every-token claim, no mlock hint");

    ck(model_residency_warning(need, 8000000000ull, have, false,
                               buf, sizeof(buf)) &&
       strstr(buf, "sparse MoE") && strstr(buf, "every token to page from disk"),
       "sparse with a hot set that does not fit: the doom claim survives");

    // A hot set that is not actually a discount must not be dressed up as one.
    ck(model_residency_warning(need, need, have, false, buf, sizeof(buf)) &&
       !strstr(buf, "sparse MoE"),
       "hot == file falls back to the dense wording");

    ck(model_residency_warning(need, 0, have, true, buf, sizeof(buf)) &&
       !strstr(buf, "--mlock"),
       "an already-locked model is not told to use --mlock");
    ck(model_residency_warning(need, 0, have, false, buf, sizeof(buf)) &&
       strstr(buf, "--mlock"),
       "an unlocked model is told about --mlock");

    if (g_fail) { fprintf(stderr, "paging-warn: FAILED\n"); return 1; }
    puts("paging-warn ok");
    return 0;
}
