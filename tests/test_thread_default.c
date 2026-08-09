#include "compat.h"
#include "tpool.h"

#include <stdio.h>

#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

static int g_fail = 0;

static void ck(int cond, const char *what) {
    if (!cond) { fprintf(stderr, "FAIL: %s\n", what); g_fail = 1; }
    else        printf("ok: %s\n", what);
}

int main(void) {
    int logical = plat_cpu_count();
    int def = plat_default_thread_count();
    ck(logical >= 1, "logical CPU count is positive");
    ck(def >= 1, "default thread count is positive");
    ck(def <= logical, "default thread count does not exceed logical CPUs");
    ck(def <= 64, "default thread count is capped");

#ifdef __APPLE__
    int perf = 0;
    size_t len = sizeof(perf);
    if (sysctlbyname("hw.perflevel0.physicalcpu", &perf, &len, NULL, 0) == 0 &&
        len == sizeof(perf) && perf > 0) {
        int want = perf > 64 ? 64 : perf;
        ck(def == want, "Apple asymmetric default uses performance cores");
    }
#endif

    // An over-large request must be CLAMPED AND SAID SO. Silently discarding
    // an explicit -t value is the failure mode the Syntetik-MoE run lost time
    // to: `-t 128` behaved exactly like `-t 64` with nothing on stderr.
    // tpool_create() caps at TP_MAX; here we pin that the cap holds and that
    // the pool reports the clamped size rather than the requested one.
    tpool *big = tpool_create(1024);
    ck(big != NULL, "tpool_create survives an over-large request");
    if (big) {
        ck(tpool_size(big) <= 64, "over-large thread request is clamped");
        ck(tpool_size(big) == 64, "clamp lands on TP_MAX, not something else");
        tpool_destroy(big);
    }

    if (g_fail) {
        fprintf(stderr, "thread defaults: FAILED\n");
        return 1;
    }
    puts("thread defaults ok");
    return 0;
}
