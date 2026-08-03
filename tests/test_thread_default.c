#include "compat.h"

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

    if (g_fail) {
        fprintf(stderr, "thread defaults: FAILED\n");
        return 1;
    }
    puts("thread defaults ok");
    return 0;
}
