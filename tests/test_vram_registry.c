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

#include <stdatomic.h>

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <direct.h>
#include <io.h>
#include <process.h>
#else
#include <dirent.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#define GB (1024ull * 1024ull * 1024ull)

// A fixed free-VRAM reading. Real callers hand in gpu_mem_info.
static uint64_t fixed_free(void *ud) { return *(uint64_t *)ud; }

// Registry file for a gpu id inside the scratch dir, matching registry_path()
// in vramreg.c. Tests that pre-seed or inspect the ledger need the real path.
static void reg_file(const char *dir, const char *gpu_id, char *out, size_t cap) {
    snprintf(out, cap, "%s/gridcore-vram-%s.reg", dir, gpu_id);
}

static long file_size(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fclose(f);
    return n;
}

// Remove the scratch directory at process exit so runs stop littering the
// repo root with vramreg-test-<pid>/ debris (26 had accumulated). Registered
// via atexit from the PARENT only; forked children either _exit() or die by
// signal, neither of which runs atexit handlers, and the pid guard makes the
// handler a no-op anywhere else regardless.
static long scratch_owner_pid = 0;
static char scratch_path[512];

static void scratch_cleanup(void) {
    if ((long)getpid() != scratch_owner_pid || !scratch_path[0]) return;
#ifdef _WIN32
    char pattern[600];
    struct _finddata_t fd;
    snprintf(pattern, sizeof(pattern), "%s\\*", scratch_path);
    intptr_t h = _findfirst(pattern, &fd);
    if (h != -1) {
        do {
            if (strcmp(fd.name, ".") && strcmp(fd.name, "..")) {
                char p[1200];
                snprintf(p, sizeof(p), "%s\\%s", scratch_path, fd.name);
                remove(p);
            }
        } while (_findnext(h, &fd) == 0);
        _findclose(h);
    }
    _rmdir(scratch_path);
#else
    DIR *d = opendir(scratch_path);
    if (d) {
        struct dirent *e;
        while ((e = readdir(d)) != NULL) {
            if (strcmp(e->d_name, ".") && strcmp(e->d_name, "..")) {
                char p[1200];
                snprintf(p, sizeof(p), "%s/%s", scratch_path, e->d_name);
                remove(p);
            }
        }
        closedir(d);
    }
    rmdir(scratch_path);
#endif
}

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
    if (scratch_owner_pid == 0) {
        scratch_owner_pid = (long)getpid();
        snprintf(scratch_path, sizeof(scratch_path), "%s", dir);
        atexit(scratch_cleanup);
    }
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
                                   fixed_free, &free_before, 0, NULL, NULL, NULL, 0);
    assert(first && "the first runner on an idle GPU must be admitted");
    vram_commit(first, 5200000000ull);

    // second runner: wants more than what is left
    uint64_t free_now = 24 * GB - 5200000000ull;   // what the driver now reports
    char err[1024] = {0};
    vram_lease *second = vram_claim(gpu, "/models/gemma-4-12B-it-Q4_K_M.gguf",
                                    20 * GB, fixed_free, &free_now, 0, NULL,
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

// A registry the rmw cannot read back — here, one grown past the 1MB read cap
// — must fail the whole rmw, not run against an empty view. Proceeding used to
// truncate the file on write-back, destroying every other process's entries.
// Accounting is best-effort, so the claim itself still goes through.
static void test_unreadable_registry_is_not_truncated(void) {
    const char *dir = scratch_dir();
    char path[600];
    reg_file(dir, "bigfile-test", path, sizeof(path));

    FILE *f = fopen(path, "w");
    assert(f);
    fprintf(f, "%ld\t1\t0\t%llu\tC\t5200000000\t/models/other-holder.gguf\n",
            (long)getpid(), (unsigned long long)time(NULL));
    for (int i = 0; i < (1 << 20) / 32 + 8; i++)
        fprintf(f, "# padding line, 32 bytes long .\n");
    fclose(f);
    long before = file_size(path);
    assert(before >= (1 << 20));

    uint64_t free_all = 24 * GB;
    vram_lease *l = vram_claim("bigfile-test", "/models/mine.gguf", 1 * GB,
                               fixed_free, &free_all, 0, NULL, NULL, NULL, 0);
    assert(l && "an unreadable registry must not stop a runner (best-effort)");
    assert(file_size(path) == before &&
           "a claim that could not read the registry must not truncate it");

    vram_release(l);
    assert(file_size(path) == before &&
           "a release that could not read the registry must not truncate it");
    remove(path);
}

#ifndef _WIN32
// The registry can fall back to a world-writable directory (/tmp), where its
// path is predictable: another local user can plant a symlink there and a
// naive O_CREAT open would write registry content wherever it points. The
// open must refuse to follow; the claim still proceeds unaccounted.
static void test_symlinked_registry_is_refused(void) {
    const char *dir = scratch_dir();
    char victim[600], link_path[600];
    snprintf(victim, sizeof(victim), "%s/victim", dir);
    FILE *f = fopen(victim, "w");
    assert(f);
    fputs("precious\n", f);
    fclose(f);
    reg_file(dir, "symlink-test", link_path, sizeof(link_path));
    remove(link_path);
    assert(symlink("victim", link_path) == 0);

    uint64_t free_all = 24 * GB;
    vram_lease *l = vram_claim("symlink-test", "/models/mine.gguf", 1 * GB,
                               fixed_free, &free_all, 0, NULL, NULL, NULL, 0);
    assert(l && "a hijacked registry path must not stop a runner (best-effort)");
    vram_release(l);

    char buf[64] = {0};
    f = fopen(victim, "r");
    assert(f);
    assert(fgets(buf, sizeof(buf), f) && !strcmp(buf, "precious\n") &&
           "the registry open must not follow a planted symlink");
    fclose(f);
}

// Several claims from one process are told apart by seq alone; two concurrent
// claims minting the same seq would make vram_release remove the wrong entry.
// Hammer vram_claim from several threads and demand every ledger line carries
// a distinct seq.
#define SEQ_THREADS 8
#define SEQ_CLAIMS  8
static vram_lease *seq_leases[SEQ_THREADS][SEQ_CLAIMS];

static void *seq_claimer(void *arg) {
    long t = (long)(intptr_t)arg;
    uint64_t free_all = 24 * GB;
    for (int i = 0; i < SEQ_CLAIMS; i++)
        seq_leases[t][i] = vram_claim("seq-test", "/models/tiny.gguf",
                                      1024 * 1024, fixed_free, &free_all, 0, NULL,
                                      NULL, NULL, 0);
    return NULL;
}

static void test_concurrent_claims_mint_distinct_seqs(void) {
    const char *dir = scratch_dir();
    pthread_t th[SEQ_THREADS];
    for (long t = 0; t < SEQ_THREADS; t++)
        assert(pthread_create(&th[t], NULL, seq_claimer, (void *)(intptr_t)t) == 0);
    for (int t = 0; t < SEQ_THREADS; t++) pthread_join(th[t], NULL);

    char path[600];
    reg_file(dir, "seq-test", path, sizeof(path));
    FILE *f = fopen(path, "r");
    assert(f);
    long seqs[SEQ_THREADS * SEQ_CLAIMS];
    int n = 0;
    char line[1024];
    while (fgets(line, sizeof(line), f) && n < SEQ_THREADS * SEQ_CLAIMS) {
        long pid = 0, seq = -1;
        assert(sscanf(line, "%ld %ld", &pid, &seq) == 2);
        for (int i = 0; i < n; i++)
            assert(seqs[i] != seq && "two concurrent claims minted one seq");
        seqs[n++] = seq;
    }
    fclose(f);
    assert(n == SEQ_THREADS * SEQ_CLAIMS && "every claim must be in the ledger");

    for (int t = 0; t < SEQ_THREADS; t++)
        for (int i = 0; i < SEQ_CLAIMS; i++) {
            assert(seq_leases[t][i]);
            vram_release(seq_leases[t][i]);
        }
    assert(file_size(path) == 0 && "releasing every claim must empty the ledger");
}

// Where the platform cannot report process start times (procstart 0 in the
// ledger), the pid-reuse guard is blind: a recycled pid adopts a dead runner's
// 'P' entry and its phantom bytes pin the GPU forever. The mitigation is age:
// no real load is still pending after an hour, so reap() drops it. A fresh
// guardless 'P' entry must still count — it is somebody's live load.
static void test_stale_guardless_pending_is_reaped(void) {
    const char *dir = scratch_dir();
    char path[600];
    reg_file(dir, "stale-pending-test", path, sizeof(path));
    uint64_t now = (uint64_t)time(NULL);
    FILE *f = fopen(path, "w");
    assert(f);
    fprintf(f, "%ld\t7\t0\t%llu\tP\t%llu\t/models/stale-loader.gguf\n",
            (long)getpid(), (unsigned long long)(now - 2 * 3600),
            (unsigned long long)(20 * GB));
    fprintf(f, "%ld\t8\t0\t%llu\tP\t%llu\t/models/fresh-loader.gguf\n",
            (long)getpid(), (unsigned long long)now,
            (unsigned long long)(3 * GB));
    fclose(f);

    // the fresh 3GB pending still counts, so 22GB of 24GB must be refused —
    // and the refusal names the live loader, not the reaped orphan
    uint64_t free_all = 24 * GB;
    char err[1024] = {0};
    assert(!vram_claim("stale-pending-test", "/models/mine.gguf", 22 * GB,
                       fixed_free, &free_all, 0, NULL, NULL, err, sizeof(err)));
    assert(strstr(err, "fresh-loader") && "a fresh guardless entry must survive");
    assert(!strstr(err, "stale-loader") && "the stale guardless entry must be gone");

    // with the stale 20GB dropped, a 20GB ask fits
    vram_lease *l = vram_claim("stale-pending-test", "/models/mine.gguf",
                               20 * GB, fixed_free, &free_all, 0, NULL, NULL, NULL, 0);
    assert(l && "a stale guardless pending entry must not pin phantom bytes");
    vram_release(l);
}

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
                                   fixed_free, &free_all, 0, NULL, NULL, NULL, 0);
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
                       0, NULL, NULL, err, sizeof(err)));
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
                                  fixed_free, &free_all, 0, NULL, NULL, err, sizeof(err));
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
                   0, NULL, NULL, NULL, 0);
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
                                      fixed_free, &free_all, 0, NULL, NULL, NULL, 0);
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
                       0, NULL, NULL, err, sizeof(err)));
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
                                   fixed_free, &free_all, 0, NULL, NULL, NULL, 0);
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
                       &free_all, 0, NULL, NULL, NULL, 0));

    // with waiting it blocks until the holder releases, then proceeds
    double t0 = plat_now();
    vram_lease *queued = vram_claim(gpu, "/models/queued.gguf", 20 * GB,
                                    fixed_free, &free_all, 30, NULL, NULL, NULL, 0);
    double waited = plat_now() - t0;
    assert(queued && "--wait-for-vram must queue, not fail");
    assert(waited >= 1.0 && "it must actually have waited for the holder");
    assert(waited < 25.0 && "it must proceed as soon as the holder releases");

    int st = 0;
    waitpid(child, &st, 0);
    vram_release(queued);
}

// A queued wait must be interruptible: /unload and shutdown point the claim's
// cancel flag at their intent, and a wait that ignored it would pin the swap
// path (and a joining shutdown) for the full --wait-for-vram budget.
static atomic_int cancel_flag;

static void *cancel_setter(void *arg) {
    (void)arg;
    plat_sleep_ms(300);
    atomic_store(&cancel_flag, 1);
    return NULL;
}

static void test_cancelled_wait_gives_up_promptly(void) {
    scratch_dir();
    const char *gpu = "MIG-wait-cancel-test";
    uint64_t free_all = 24 * GB;

    vram_lease *hog = vram_claim(gpu, "/models/holder.gguf", 20 * GB,
                                 fixed_free, &free_all, 0, NULL, NULL, NULL, 0);
    assert(hog);

    atomic_store(&cancel_flag, 0);
    pthread_t th;
    assert(pthread_create(&th, NULL, cancel_setter, NULL) == 0);
    double t0 = plat_now();
    char err[1024] = {0};
    vram_lease *queued = vram_claim(gpu, "/models/queued.gguf", 20 * GB,
                                    fixed_free, &free_all, 30, &cancel_flag,
                                    NULL, err, sizeof(err));
    double waited = plat_now() - t0;
    pthread_join(th, NULL);
    assert(!queued && "a cancelled wait must fail the claim, not admit it");
    assert(waited < 5.0 && "cancellation must cut a 30s wait short");
    assert(strstr(err, "cancel") && "the error must say the wait was cancelled");

    vram_release(hog);
}

// A wait that runs out still has to explain itself: the timeout path produces
// the same named-holder message the immediate refusal does, not a bare timeout.
static void test_wait_timeout_still_names_the_holder(void) {
    scratch_dir();
    const char *gpu = "MIG-wait-timeout-test";
    uint64_t free_all = 24 * GB;

    vram_lease *hog = vram_claim(gpu, "/models/stubborn-Q4_K_M.gguf", 20 * GB,
                                 fixed_free, &free_all, 0, NULL, NULL, NULL, 0);
    assert(hog);

    char err[1024] = {0};
    assert(!vram_claim(gpu, "/models/queued.gguf", 20 * GB, fixed_free,
                       &free_all, 1, NULL, NULL, err, sizeof(err)));
    assert(strstr(err, "stubborn-Q4_K_M") && "a timed-out wait must still name the holder");

    vram_release(hog);
}
#endif

int main(void) {
    test_second_runner_refuses_naming_the_holder();
    test_unreadable_registry_is_not_truncated();
#ifndef _WIN32
    test_symlinked_registry_is_refused();
    test_concurrent_claims_mint_distinct_seqs();
    test_stale_guardless_pending_is_reaped();
    test_dead_pid_reservation_is_reclaimed();
    test_reaping_keeps_live_neighbours();
    test_wait_for_vram_queues_then_proceeds();
    test_cancelled_wait_gives_up_promptly();
    test_wait_timeout_still_names_the_holder();
#else
    puts("vram registry: cross-process reaping tests need fork(); skipped on Windows");
#endif
    puts("vram registry tests ok");
    return 0;
}
