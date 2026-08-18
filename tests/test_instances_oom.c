// Allocation-failure tests for the instance discovery registry.
//
// instances_list() is read by the tray on every menu open and by any
// controller enumerating live runners, and its declared contract is narrow:
// "returns a malloc'd array, count in *n". A NULL with a non-zero count is not
// a value any caller checks for -- src/tray.c walks recs[i] straight after --
// so breaking that contract is a crash in the one process that is supposed to
// be watching the others.
//
// instances.c is compiled straight into this test with the allocators
// macro-substituted, the same way tests/test_json_oom.c does it. The system
// headers come first so the macros rewrite only instances.c's own calls. json.c
// stays a separate translation unit on the real allocator: this sweep is about
// the registry's own bookkeeping, not the parser's (which test_json_oom owns).
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#ifndef _WIN32
#include <unistd.h>
#endif

static long alloc_calls;   // allocations attempted since the last reset
static long alloc_live;    // outstanding blocks; must return to 0
static long fail_at = -1;  // index of the allocation to fail, -1 for none

static void *t_malloc(size_t n) {
    if (fail_at >= 0 && alloc_calls++ == fail_at) return NULL;
    if (fail_at < 0) alloc_calls++;
    void *p = malloc(n);
    if (p) alloc_live++;
    return p;
}

static void *t_calloc(size_t a, size_t b) {
    if (fail_at >= 0 && alloc_calls++ == fail_at) return NULL;
    if (fail_at < 0) alloc_calls++;
    void *p = calloc(a, b);
    if (p) alloc_live++;
    return p;
}

static void *t_realloc(void *p, size_t n) {
    if (fail_at >= 0 && alloc_calls++ == fail_at) return NULL;
    if (fail_at < 0) alloc_calls++;
    void *q = realloc(p, n);
    if (q && !p) alloc_live++;   // realloc(NULL, n) behaves as malloc
    return q;
}

static char *t_strdup(const char *s) {
    size_t n = strlen(s) + 1;
    char *p = t_malloc(n);
    if (p) memcpy(p, s, n);
    return p;
}

static void t_free(void *p) {
    if (p) alloc_live--;
    free(p);
}

#define malloc  t_malloc
#define calloc  t_calloc
#define realloc t_realloc
#define strdup  t_strdup
#define free    t_free
#include "../src/instances.c"
#undef malloc
#undef calloc
#undef realloc
#undef strdup
#undef free

#ifdef _WIN32
#include <process.h>
#include <direct.h>
#define getpid _getpid
#define setenv_compat(k, v) _putenv_s(k, v)
#else
#define setenv_compat(k, v) setenv(k, v, 1)
#endif

static int fails = 0;
#define CHECK(cond, msg) do { if (!(cond)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, msg); fails++; } \
} while (0)

// Enough records to make the array grow twice (8 -> 16 -> 32), because the
// growth step is where the contract breaks. Each one claims OUR pid so the
// liveness check keeps it and nothing is swept mid-sweep.
#define N_FAKE 20

static void write_fake(const char *dir, long name, const char *model) {
    char path[1300];
    snprintf(path, sizeof path, "%s/%ld.json", dir, name);
    FILE *f = fopen(path, "wb");
    if (!f) return;
    fprintf(f, "{\"pid\": %ld, \"started\": %lld, \"mode\": \"serve\","
               " \"port\": 8123, \"version\": \"test\","
               " \"models\": [{\"name\": \"%s\", \"path\": \"/x/%s\"}]}\n",
            (long)getpid(), (long long)time(NULL), model, model);
    fclose(f);
}

// Every returned record is read, exactly as src/tray.c reads them: a contract
// violation that is only observed by checking the pointer is a contract the
// callers do not have.
static void walk(instance_rec *r, int n) {
    for (int i = 0; i < n; i++) {
        volatile long pid = r[i].pid;
        (void)pid;
        for (int m = 0; m < r[i].n_models; m++) {
            volatile size_t k = r[i].model_names[m]
                              ? strlen(r[i].model_names[m]) : 0;
            (void)k;
        }
    }
}

int main(void) {
    char home[512];
#ifdef _WIN32
    snprintf(home, sizeof home, "%s\\xyntetik-oom-%ld", getenv("TEMP"),
             (long)getpid());
    _mkdir(home);
    setenv_compat("APPDATA", home);
#else
    snprintf(home, sizeof home, "/tmp/xyntetik-oom-%ld", (long)getpid());
    mkdir(home, 0755);
    setenv_compat("HOME", home);
#endif
    const char *dir = instances_dir();
    CHECK(dir != NULL, "instances_dir resolves");
    if (!dir) return 1;

    for (int i = 0; i < N_FAKE; i++) {
        char model[64];
        snprintf(model, sizeof model, "m%d.gguf", i);
        write_fake(dir, (long)getpid() + 1 + i, model);
    }

    // baseline: no injected failure. Everything is listed and nothing leaks.
    fail_at = -1;
    alloc_calls = 0;
    alloc_live = 0;
    int n = 0;
    instance_rec *r = instances_list(&n);
    CHECK(n == N_FAKE, "baseline lists every record");
    CHECK(r != NULL, "baseline returns an array");
    if (r) walk(r, n);
    long total = alloc_calls;
    instances_list_free(r, n);
    CHECK(alloc_live == 0, "baseline frees every block");
    CHECK(total > 0, "the sweep has allocations to fail");

    // Fail each allocation in turn. instances_list may return fewer records --
    // a partial list is the honest answer when the array cannot grow -- but it
    // must never hand back a NULL with a count, and it must never strand a
    // block: `arr = realloc(arr, n)` loses the original pointer, which shows up
    // here as alloc_live != 0.
    for (long k = 0; k < total; k++) {
        fail_at = k;
        alloc_calls = 0;
        alloc_live = 0;
        int m = 0;
        instance_rec *p = instances_list(&m);
        if (!p && m != 0) {
            fprintf(stderr, "FAIL: allocation %ld of %ld: NULL array with "
                            "count %d -- every caller indexes that\n",
                    k, total, m);
            fails++;
            fail_at = -1;
            break;
        }
        if (p) walk(p, m);
        instances_list_free(p, m);
        if (alloc_live != 0) {
            fprintf(stderr, "FAIL: allocation %ld of %ld: %ld block(s) "
                            "leaked\n", k, total, alloc_live);
            fails++;
            fail_at = -1;
            break;
        }
    }
    fail_at = -1;

    // the fixture files are still there: an allocation failure is not a reason
    // to sweep a live process's record away
    alloc_calls = 0;
    alloc_live = 0;
    r = instances_list(&n);
    CHECK(n == N_FAKE, "no record was swept by the failure sweep");
    instances_list_free(r, n);

    for (int i = 0; i < N_FAKE; i++) {
        char path[1300];
        snprintf(path, sizeof path, "%s/%ld.json", dir, (long)getpid() + 1 + i);
        remove(path);
    }
    if (fails) {
        fprintf(stderr, "test_instances_oom: %d FAILURES\n", fails);
        return 1;
    }
    printf("instances allocation-failure sweep ok (%ld allocations)\n", total);
    return 0;
}
