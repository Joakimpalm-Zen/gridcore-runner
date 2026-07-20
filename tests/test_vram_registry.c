// Cross-process VRAM reservation registry.
//
// The behaviour under test is the one that was missing the night six orphaned
// `runner --serve` processes sat on this box's 24GB MIG slice — one of them for
// 4h39m — and nothing in runner knew the others existed. A shared-weights VRAM
// check failed and then passed on re-run; a benchmark ran under contention.
//
// Everything here is GPU-free on purpose. The registry takes the free-VRAM
// figure through a callback, so the whole surface is drivable with synthetic
// numbers and runs identically on a CI box with no GPU at all.
#include "runner.h"
#include "compat.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <direct.h>
#include <process.h>
#else
#include <signal.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#define GB (1024ull * 1024ull * 1024ull)

// A fixed free-VRAM reading. Real callers hand in gpu_mem_info.
static uint64_t fixed_free(void *ud) { return *(uint64_t *)ud; }

// Point the registry at a scratch directory so a test run never touches the
// real one in $XDG_RUNTIME_DIR, and so each test starts from empty.
static const char *scratch_dir(void) {
    static char dir[512];
    snprintf(dir, sizeof(dir), "vramreg-test-%ld", (long)getpid());
#ifdef _WIN32
    _mkdir(dir);
#else
    mkdir(dir, 0700);
#endif
    setenv("RUNNER_VRAM_REGISTRY_DIR", dir, 1);
    return dir;
}

// THE test. A second runner must refuse, and the refusal must name the holder:
// pid, model, bytes, uptime. A bare "out of memory" is the failure this whole
// module exists to stop.
static void test_second_runner_refuses_naming_the_holder(void) {
    scratch_dir();
    const char *gpu = "MIG-399aa5c7-bb47-5ccb-b688-54fefec06647";

    // first runner: takes 5.2GB and keeps it
    uint64_t free_before = 24 * GB;
    vram_lease *first = vram_claim(gpu, "/models/Qwen3-4B-Q4_K_M.gguf",
                                   5200000000ull /* 5.2GB */,
                                   fixed_free, &free_before, 0, NULL, NULL, 0);
    assert(first && "the first runner on an idle GPU must be admitted");
    vram_commit(first, 5200000000ull);

    // second runner: wants more than what is left
    uint64_t free_now = 24 * GB - 5200000000ull;   // what the driver now reports
    char err[1024] = {0};
    vram_lease *second = vram_claim(gpu, "/models/gemma-4-12B-it-Q4_K_M.gguf",
                                    20 * GB, fixed_free, &free_now, 0,
                                    NULL, err, sizeof(err));
    assert(!second && "a request that does not fit must be refused, not queued");

    // the refusal has to be actionable: who, what, how much, how long
    char pidstr[32];
    snprintf(pidstr, sizeof(pidstr), "pid %ld", (long)getpid());
    assert(strstr(err, pidstr) && "refusal must name the holding pid");
    assert(strstr(err, "Qwen3-4B-Q4_K_M") && "refusal must name the held model");
    assert(strstr(err, "5.2GB") && "refusal must state the held bytes");
    assert(strstr(err, "up ") && "refusal must state how long it has been held");

    vram_release(first);
}

#ifndef _WIN32
// The orphan case, reproduced. A runner that dies without deregistering — a
// SIGKILL, a crash, an OOM kill — must not leave its reservation behind. This
// is exactly how six runners came to be holding a 24GB slice with nothing
// running that wanted it.
static void test_dead_pid_reservation_is_reclaimed(void) {
    scratch_dir();
    const char *gpu = "MIG-dead-pid-test";
    uint64_t free_all = 24 * GB;

    // A child claims 20GB and is killed before it can commit or release.
    int ready[2];
    assert(pipe(ready) == 0);
    pid_t child = fork();
    assert(child >= 0);
    if (child == 0) {
        close(ready[0]);
        vram_lease *l = vram_claim(gpu, "/models/orphan-Q4_K_M.gguf", 20 * GB,
                                   fixed_free, &free_all, 0, NULL, NULL, 0);
        char ok = l ? 'y' : 'n';
        ssize_t w = write(ready[1], &ok, 1);
        (void)w;
        for (;;) pause();          // holds the reservation until killed
    }
    close(ready[1]);
    char ok = 0;
    assert(read(ready[0], &ok, 1) == 1 && ok == 'y');
    close(ready[0]);

    // While it lives, its 20GB is in flight and a second 20GB does not fit —
    // and the refusal names it.
    char err[1024] = {0};
    assert(!vram_claim(gpu, "/models/mine.gguf", 20 * GB, fixed_free, &free_all,
                       0, NULL, err, sizeof(err)));
    char pidstr[32];
    snprintf(pidstr, sizeof(pidstr), "pid %ld", (long)child);
    assert(strstr(err, pidstr) && strstr(err, "orphan-Q4_K_M"));

    kill(child, SIGKILL);
    int st = 0;
    waitpid(child, &st, 0);

    // Now that the owner is gone the reservation must evaporate, with no
    // sweeper, no timeout and no reboot: the next claim reaps it.
    err[0] = 0;
    vram_lease *mine = vram_claim(gpu, "/models/mine.gguf", 20 * GB,
                                  fixed_free, &free_all, 0, NULL, err, sizeof(err));
    assert(mine && "a dead process's reservation must not poison the GPU");
    vram_release(mine);
}

// Nothing about a live holder may be lost in the reaping: reap() walks the same
// array it rewrites, so an entry that dies next to a live one is the case where
// an off-by-one would quietly delete the survivor.
static void test_reaping_keeps_live_neighbours(void) {
    scratch_dir();
    const char *gpu = "MIG-mixed-test";
    uint64_t free_all = 24 * GB;

    int ready[2];
    assert(pipe(ready) == 0);
    pid_t child = fork();
    assert(child >= 0);
    if (child == 0) {
        close(ready[0]);
        vram_claim(gpu, "/models/doomed.gguf", 1 * GB, fixed_free, &free_all,
                   0, NULL, NULL, 0);
        char c = 'y';
        ssize_t w = write(ready[1], &c, 1);
        (void)w;
        for (;;) pause();
    }
    close(ready[1]);
    char c = 0;
    assert(read(ready[0], &c, 1) == 1);
    close(ready[0]);

    vram_lease *survivor = vram_claim(gpu, "/models/survivor.gguf", 2 * GB,
                                      fixed_free, &free_all, 0, NULL, NULL, 0);
    assert(survivor);
    vram_commit(survivor, 2 * GB);
    free_all -= 2 * GB;   // the allocation really happened: the driver sees it

    kill(child, SIGKILL);
    int st = 0;
    waitpid(child, &st, 0);

    // 22GB free with the survivor's 2GB already inside that figure, so a 23GB
    // ask fails — and the message must name the survivor and NOT the reaped
    // child.
    char err[1024] = {0};
    assert(!vram_claim(gpu, "/models/huge.gguf", 23 * GB, fixed_free, &free_all,
                       0, NULL, err, sizeof(err)));
    assert(strstr(err, "survivor") && "the live holder must survive reaping");
    assert(!strstr(err, "doomed") && "the dead holder must be gone");

    vram_release(survivor);
}

// --wait-for-vram is the opt-in alternative to refusing: queue behind whoever
// is holding the GPU and start when they let go. A long benchmark that would
// rather wait 90 seconds than fail is the case this exists for.
static void test_wait_for_vram_queues_then_proceeds(void) {
    scratch_dir();
    const char *gpu = "MIG-wait-test";
    uint64_t free_all = 24 * GB;

    int ready[2];
    assert(pipe(ready) == 0);
    pid_t child = fork();
    assert(child >= 0);
    if (child == 0) {
        close(ready[0]);
        vram_lease *l = vram_claim(gpu, "/models/holder.gguf", 20 * GB,
                                   fixed_free, &free_all, 0, NULL, NULL, 0);
        char c = l ? 'y' : 'n';
        ssize_t w = write(ready[1], &c, 1);
        (void)w;
        sleep(2);
        vram_release(l);           // clean exit hands the reservation back
        _exit(0);
    }
    close(ready[1]);
    char c = 0;
    assert(read(ready[0], &c, 1) == 1 && c == 'y');
    close(ready[0]);

    // without waiting this is an immediate refusal
    assert(!vram_claim(gpu, "/models/queued.gguf", 20 * GB, fixed_free,
                       &free_all, 0, NULL, NULL, 0));

    // with waiting it blocks until the holder releases, then proceeds
    double t0 = plat_now();
    vram_lease *queued = vram_claim(gpu, "/models/queued.gguf", 20 * GB,
                                    fixed_free, &free_all, 30, NULL, NULL, 0);
    double waited = plat_now() - t0;
    assert(queued && "--wait-for-vram must queue, not fail");
    assert(waited >= 1.0 && "it must actually have waited for the holder");
    assert(waited < 25.0 && "it must proceed as soon as the holder releases");

    int st = 0;
    waitpid(child, &st, 0);
    vram_release(queued);
}

// A wait that runs out still has to explain itself: the timeout path produces
// the same named-holder message the immediate refusal does, not a bare timeout.
static void test_wait_timeout_still_names_the_holder(void) {
    scratch_dir();
    const char *gpu = "MIG-wait-timeout-test";
    uint64_t free_all = 24 * GB;

    vram_lease *hog = vram_claim(gpu, "/models/stubborn-Q4_K_M.gguf", 20 * GB,
                                 fixed_free, &free_all, 0, NULL, NULL, 0);
    assert(hog);

    char err[1024] = {0};
    assert(!vram_claim(gpu, "/models/queued.gguf", 20 * GB, fixed_free,
                       &free_all, 1, NULL, err, sizeof(err)));
    assert(strstr(err, "stubborn-Q4_K_M") && "a timed-out wait must still name the holder");

    vram_release(hog);
}
#endif

int main(void) {
    test_second_runner_refuses_naming_the_holder();
#ifndef _WIN32
    test_dead_pid_reservation_is_reclaimed();
    test_reaping_keeps_live_neighbours();
    test_wait_for_vram_queues_then_proceeds();
    test_wait_timeout_still_names_the_holder();
#else
    puts("vram registry: cross-process reaping tests need fork(); skipped on Windows");
#endif
    puts("vram registry tests ok");
    return 0;
}
