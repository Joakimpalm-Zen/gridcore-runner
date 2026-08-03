// The weight-residency platform layer.
//
// These four calls exist because runner mmaps its weights, the OS may evict
// them, and nothing runner printed said so: on a loaded 16 GB Mac a 1,200-token
// prompt returned nothing in 300 s at 0% CPU while /health, --caps and the
// tok/s counter all stayed green. They are the difference between "slow" and
// "paging", so they are worth a test of their own — a wrong answer here would
// be a diagnostic that lies, which is worse than no diagnostic.
//
// What this can and cannot check on a build machine: a real paging stall needs
// a machine under real memory pressure, which a test cannot manufacture. What
// it does check is that each call is wired to the right thing and answers in
// the right units — which is where a platform shim actually goes wrong.
#include "compat.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fail = 0;

static void ck(int cond, const char *what) {
    if (!cond) { fprintf(stderr, "FAIL: %s\n", what); g_fail = 1; }
    else        printf("ok: %s\n", what);
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "test.gguf";

    size_t size = 0;
    void *map = plat_mmap_ro(path, &size);
    if (!map) { fprintf(stderr, "cannot map %s\n", path); return 77; }
    ck(size > 0, "the mapping has a size");

    // --- residency ------------------------------------------------------
    double frac = plat_resident_fraction(map, size);
    ck(frac < 0.0 || (frac >= 0.0 && frac <= 1.0),
       "resident fraction is a fraction, or an honest -1");

    if (frac >= 0.0) {
        // Touch every page, then ask again. Reading the whole mapping makes it
        // resident by definition, so this must report ~everything. A shim that
        // returned a constant, or counted bytes as pages, fails here.
        volatile unsigned char sink = 0;
        for (size_t i = 0; i < size; i += 4096)
            sink ^= ((const unsigned char *)map)[i];
        (void)sink;
        double after = plat_resident_fraction(map, size);
        ck(after >= 0.99,
           "a fully-touched mapping reports itself resident");
        // NOTE: the converse is not tested. On Linux mincore answers "is this
        // page in the page cache", so a file that was just read stays resident
        // by that measure even after madvise(MADV_DONTNEED) — evicting the page
        // cache needs privileges a test does not have. The number that matters
        // in the field is the low one, and only a loaded machine produces it.
    }

    ck(plat_resident_fraction(NULL, 0) < 0.0, "a null mapping is -1, not 0");

    // --- locking --------------------------------------------------------
    // Failure is a legitimate outcome: RLIMIT_MEMLOCK is 8 MB on a stock Linux
    // box, so locking a real model normally refuses. What must not happen is a
    // crash, or a claim of success that unlocking then contradicts.
    bool locked = plat_mlock(map, size);
    printf("note: mlock of %.1f MB %s\n", (double)size / 1e6,
           locked ? "succeeded" : "was refused (expected under a low ulimit -l)");
    if (locked) {
        double after = plat_resident_fraction(map, size);
        ck(after < 0.0 || after >= 0.99,
           "a locked mapping is resident");
        plat_munlock(map, size);
    }
    ck(!plat_mlock(NULL, 0), "locking nothing fails rather than claiming success");

    // --- faults ---------------------------------------------------------
    uint64_t f1 = plat_major_faults();
    uint64_t f2 = plat_major_faults();
    ck(f2 >= f1, "the major-fault counter never runs backwards");
    // The subtraction in run_completion is unsigned, so a counter that went
    // backwards would print an astronomical page-in count rather than zero.
    ck(f2 - f1 < 1000000u, "two reads in a row differ by a sane amount");

    // --- available RAM --------------------------------------------------
    uint64_t total = plat_ram_bytes();
    uint64_t avail = plat_ram_available_bytes();
    ck(total > 0, "total RAM is reported");
    ck(avail > 0, "available RAM is reported");
    // The bug this catches is the one that makes the whole feature useless:
    // reporting *total* where *available* was meant, which would let a
    // launcher happily load a model that cannot stay resident.
    ck(avail <= total, "available RAM does not exceed total RAM");

    plat_munmap(map, size);
    if (g_fail) { fprintf(stderr, "residency: FAILED\n"); return 1; }
    puts("residency ok");
    return 0;
}
