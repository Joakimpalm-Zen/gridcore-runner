// RNR-013: a VRAM claim that is admitted into the registry but then cannot
// allocate its lease handle must roll the exact (pid, seq) entry back. The
// owning process stays alive, so dead-PID reaping would never clear it — a
// leftover entry would refuse future runners against a reservation that never
// became real.
//
// vramreg.c is compiled in with calloc macro-substituted (same technique as
// test_tokenizer_oom.c); its only calloc is the lease handle, so failing calloc
// fails exactly the post-admission allocation. compat.c is linked normally.
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define GB (1024ull * 1024ull * 1024ull)

static int g_fail_calloc = 0;
static void *rb_calloc(size_t a, size_t b) {
    if (g_fail_calloc) return NULL;   // real calloc — macro is defined below
    return calloc(a, b);
}
#define calloc rb_calloc
#include "../src/vramreg.c"
#undef calloc

static uint64_t fixed_free(void *ud) { return *(uint64_t *)ud; }

// one registry line per admitted entry
static int count_entries(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    int lines = 0, c;
    while ((c = fgetc(f)) != EOF)
        if (c == '\n') lines++;
    fclose(f);
    return lines;
}

int main(void) {
    char dir[] = "/tmp/vramrb-XXXXXX";
    assert(mkdtemp(dir));
    setenv("RUNNER_VRAM_REGISTRY_DIR", dir, 1);

    const char *gpu = "rollback-test";
    char path[512];
    registry_path(gpu, path, sizeof(path));
    uint64_t free_amt = 24 * GB;

    // baseline: a normal claim leaves exactly one entry, release clears it
    vram_lease *a = vram_claim(gpu, "/models/m.gguf", 1 * GB,
                               fixed_free, &free_amt, 0, NULL, NULL, NULL, 0);
    assert(a && "an idle GPU must admit the first claim");
    assert(count_entries(path) == 1);
    vram_release(a);
    assert(count_entries(path) == 0);

    // now the lease allocation fails after admission
    g_fail_calloc = 1;
    char err[256] = {0};
    vram_lease *b = vram_claim(gpu, "/models/m.gguf", 1 * GB,
                               fixed_free, &free_amt, 0, NULL, NULL, err, sizeof(err));
    g_fail_calloc = 0;
    assert(!b && "a lease-allocation failure must return NULL");
    if (count_entries(path) != 0) {
        fprintf(stderr, "phantom reservation left behind: %d entrie(s) (RNR-013)\n",
                count_entries(path));
        abort();
    }

    // the registry is still fully usable afterward
    vram_lease *c2 = vram_claim(gpu, "/models/m.gguf", 1 * GB,
                                fixed_free, &free_amt, 0, NULL, NULL, NULL, 0);
    assert(c2 && "registry must remain usable after a rolled-back claim");
    assert(count_entries(path) == 1);
    vram_release(c2);
    assert(count_entries(path) == 0);

    remove(path);
    rmdir(dir);
    printf("vram rollback test ok\n");
    return 0;
}
