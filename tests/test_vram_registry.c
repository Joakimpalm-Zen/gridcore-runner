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
#include <sys/select.h>
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
    snprintf(out, cap, "%s/xyntetik-vram-%s.reg", dir, gpu_id);
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
    assert(_putenv_s("RUNNER_VRAM_REGISTRY_DIR", dir) == 0);
#else
    mkdir(dir, 0700);
    assert(setenv("RUNNER_VRAM_REGISTRY_DIR", dir, 1) == 0);
#endif
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
                                   0, fixed_free, &free_before, 0, NULL, NULL, NULL, 0);
    assert(first && "the first runner on an idle GPU must be admitted");
    vram_commit(first, 5200000000ull);

    // second runner: wants more than what is left
    uint64_t free_now = 24 * GB - 5200000000ull;   // what the driver now reports
    char err[1024] = {0};
    vram_lease *second = vram_claim(gpu, "/models/gemma-4-12B-it-Q4_K_M.gguf",
                                    20 * GB, 0, fixed_free, &free_now, 0, NULL,
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

// The priority tag is advisory metadata (multi-tenant scheduler arc, ENGINE
// half): recorded on the claim, round-tripped through the on-disk ledger, and
// printed in the refusal listing next to the fields the previous test locks.
static void test_priority_tag_is_recorded_in_refusal(void) {
    scratch_dir();
    const char *gpu = "MIG-priority-tag-test";

    uint64_t free_before = 24 * GB;
    vram_lease *first = vram_claim(gpu, "/models/holder.gguf", 5 * GB,
                                   3 /* priority */, fixed_free, &free_before,
                                   0, NULL, NULL, NULL, 0);
    assert(first && "an idle GPU must admit the first claim");

    uint64_t free_now = 24 * GB - 5 * GB;
    char err[1024] = {0};
    vram_lease *second = vram_claim(gpu, "/models/too-big.gguf", 22 * GB, 0,
                                    fixed_free, &free_now, 0, NULL, NULL,
                                    err, sizeof(err));
    assert(!second);
    assert(strstr(err, "priority 3") &&
           "refusal listing must show the holder's priority");

    vram_release(first);
}

// Rule: no behavior change for a caller that passes no new flags. A
// default-priority (0) claim must produce EXACTLY the fields the original
// refusal test asserts — pid, model, bytes, uptime — plus the one new field.
static void test_default_priority_matches_legacy_fields_plus_new_field(void) {
    scratch_dir();
    const char *gpu = "MIG-default-priority-test";

    uint64_t free_before = 24 * GB;
    vram_lease *first = vram_claim(gpu, "/models/Qwen3-4B-Q4_K_M.gguf",
                                   5200000000ull, 0 /* default priority */,
                                   fixed_free, &free_before, 0, NULL, NULL,
                                   NULL, 0);
    assert(first);
    vram_commit(first, 5200000000ull);

    uint64_t free_now = 24 * GB - 5200000000ull;
    char err[1024] = {0};
    vram_lease *second = vram_claim(gpu, "/models/gemma-4-12B-it-Q4_K_M.gguf",
                                    20 * GB, 0, fixed_free, &free_now, 0, NULL,
                                    NULL, err, sizeof(err));
    assert(!second);

    char pidstr[32];
    snprintf(pidstr, sizeof(pidstr), "pid %ld", (long)getpid());
    assert(strstr(err, pidstr) && "refusal must still name the holding pid");
    assert(strstr(err, "Qwen3-4B-Q4_K_M") && "refusal must still name the held model");
    assert(strstr(err, "5.2GB") && "refusal must still state the held bytes");
    assert(strstr(err, "up ") && "refusal must still state how long it has been held");
    assert(strstr(err, "priority 0") &&
           "the only addition for a default-priority claim is the new field");

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
                               0, fixed_free, &free_all, 0, NULL, NULL, NULL, 0);
    assert(l && "an unreadable registry must not stop a runner (best-effort)");
    assert(file_size(path) == before &&
           "a claim that could not read the registry must not truncate it");

    vram_release(l);
    assert(file_size(path) == before &&
           "a release that could not read the registry must not truncate it");
    remove(path);
}

// Backward compatibility, decided and locked here: a ledger line written by a
// binary that predates the priority field has exactly 7 tab-separated fields
// (no trailing priority). It must still parse, and it must be read as
// priority 0 — not dropped, not misparsed, not treated as corrupt.
static void test_legacy_record_without_priority_reads_as_zero(void) {
    const char *dir = scratch_dir();
    char path[600];
    reg_file(dir, "legacy-format-test", path, sizeof(path));
    uint64_t now = (uint64_t)time(NULL);

    FILE *f = fopen(path, "w");
    assert(f);
    // exactly 7 fields, no priority column — the pre-existing on-disk format.
    // 'P' (pending), not 'C', because this fixed-free-VRAM test drives the fit
    // arithmetic directly: a 'C' entry is already inside the driver's free
    // figure and would not subtract, so it would never produce a refusal here.
    fprintf(f, "%ld\t1\t0\t%llu\tP\t5000000000\t/models/old-binary-holder.gguf\n",
            (long)getpid(), (unsigned long long)now);
    fclose(f);

    uint64_t free_all = 24 * GB;
    char err[1024] = {0};
    assert(!vram_claim("legacy-format-test", "/models/mine.gguf", 22 * GB, 0,
                       fixed_free, &free_all, 0, NULL, NULL, err, sizeof(err)));
    assert(strstr(err, "old-binary-holder") &&
           "a legacy 7-field line must still be read as a live holder");
    assert(strstr(err, "priority 0") &&
           "a legacy record with no priority column must read as priority 0");
}

// Exact refusal-line format, priority included. Built by hand (like the
// stale-guardless test above) so the holder's uptime is a fixed number
// instead of racing the wall clock — this pins the literal string, not just
// its pieces.
static void test_refusal_line_priority_format_is_exact(void) {
    const char *dir = scratch_dir();
    char path[600];
    reg_file(dir, "priority-format-test", path, sizeof(path));
    uint64_t now = (uint64_t)time(NULL);

    FILE *f = fopen(path, "w");
    assert(f);
    // 'P' (pending) so the fixed free-VRAM reading below actually gets
    // subtracted — see the comment in the legacy-record test above.
    fprintf(f, "%ld\t1\t0\t%llu\tP\t5000000000\t/models/format-holder.gguf\t7\n",
            (long)getpid(), (unsigned long long)(now - 90));
    fclose(f);

    uint64_t free_all = 24 * GB;
    char err[1024] = {0};
    assert(!vram_claim("priority-format-test", "/models/mine.gguf", 22 * GB, 0,
                       fixed_free, &free_all, 0, NULL, NULL, err, sizeof(err)));

    char want[160];
    snprintf(want, sizeof(want),
             "pid %ld holding 5.0GB for format-holder, up 1m, priority 7 (loading)",
             (long)getpid());
    assert(strstr(err, want) && "refusal line format must match exactly");
}

// The refusal must not contradict itself in consecutive lines.
//
// `holders` counts OTHER runners by design, but the listing prints every
// non-waiter entry, this process's own included. So a runner blocked by its
// own earlier claim was told "No other runner is registered on this GPU — the
// memory is held by something outside runner's accounting", directly above a
// line naming its own pid as the holder of exactly that memory. Both cannot be
// true, and it is the sentence that is wrong: the memory is inside runner's
// accounting; it is simply ours. The sentence belongs to the case it was
// written for — a shortfall with nobody at all in the ledger.
static void test_refusal_does_not_blame_an_outsider_it_just_named(void) {
    scratch_dir();
    uint64_t free_all = 24 * GB;

    // (a) our own pending claim is what does not leave room
    const char *gpu = "self-holder-test";
    vram_lease *first = vram_claim(gpu, "/models/first.gguf", 20 * GB, 0,
                                   fixed_free, &free_all, 0, NULL, NULL, NULL, 0);
    assert(first);

    char err[1024] = {0};
    vram_status st = {0};
    assert(!vram_claim(gpu, "/models/second.gguf", 10 * GB, 0, fixed_free,
                       &free_all, 0, NULL, &st, err, sizeof(err)));
    assert(st.holders == 0 && "our own claim is not another runner");
    char pidstr[40];
    snprintf(pidstr, sizeof(pidstr), "pid %ld holding", (long)getpid());
    assert(strstr(err, pidstr) && "the listing still names who holds it");
    assert(!strstr(err, "outside runner's accounting") &&
           "a holder the same message names is not an outsider");
    vram_release(first);

    // (b) the sentence's real case is untouched: an empty ledger and a
    // shortfall the registry genuinely cannot explain
    uint64_t little = 1 * GB;
    char err2[1024] = {0};
    assert(!vram_claim("empty-ledger-test", "/models/big.gguf", 20 * GB, 0,
                       fixed_free, &little, 0, NULL, NULL, err2, sizeof(err2)));
    assert(strstr(err2, "outside runner's accounting") &&
           "an unattributable shortfall must still say so");
}

// err_cap is the caller's choice, and the module has to survive every value of
// it. The refusal message was assembled with unguarded `off += snprintf(...)`
// steps: snprintf returns what it WOULD have written, so a buffer too small
// for the first sentence leaves off past err_cap, and the next step writes at
// err + off with a length of err_cap - off — an unsigned subtraction that
// underflows to a huge size. Nothing in the message is bounded by the module
// either: the gpu id and the model path both come from outside it.
//
// Padding rather than a bare exact-size buffer, so this is red without needing
// a sanitizer build: the stray write lands ~79 bytes in, and any byte of the
// pad that changes is a write the caller never authorised.
static void test_small_error_buffer_is_not_overrun(void) {
    scratch_dir();
    const char *gpu = "small-err-buffer-test";

    // deliberately NOT committed: a pending claim is what puts the "claimed
    // but not yet allocated" sentence — the first unguarded step — on the path
    uint64_t free_all = 24 * GB;
    vram_lease *first = vram_claim(gpu, "/models/holder.gguf", 5 * GB, 0,
                                   fixed_free, &free_all, 0, NULL, NULL,
                                   NULL, 0);
    assert(first);

    enum { PAD = 512 };
    for (size_t cap = 1; cap <= 200; cap++) {
        char *err = malloc(cap + PAD);
        assert(err);
        memset(err, 0x5A, cap + PAD);
        assert(!vram_claim(gpu, "/models/too-big.gguf", 22 * GB, 0, fixed_free,
                           &free_all, 0, NULL, NULL, err, cap));
        for (size_t i = cap; i < cap + PAD; i++)
            assert(err[i] == 0x5A &&
                   "the refusal message must stay inside err_cap");
        assert(strlen(err) < cap && "and stay NUL-terminated inside it");
        free(err);
    }
    vram_release(first);
}

// The yield sentinel's lifecycle, end to end through the public API: unset
// until requested, set once vram_request_yield names this (gpu, pid), and
// clear again after vram_yield_clear. Also: a request aimed at a DIFFERENT
// pid must never be visible to our own lease — the primitive is scoped, not
// a broadcast "everyone on this GPU yield".
static void test_yield_flag_lifecycle(void) {
    scratch_dir();
    const char *gpu = "MIG-yield-lifecycle-test";
    uint64_t free_all = 24 * GB;

    vram_lease *l = vram_claim(gpu, "/models/holder.gguf", 1 * GB, 0,
                               fixed_free, &free_all, 0, NULL, NULL, NULL, 0);
    assert(l);

    assert(!vram_yield_requested(l) &&
           "nothing has asked this holder to yield yet");

    long other_pid = (long)getpid() + 1; // never actually this process
    assert(vram_request_yield(gpu, other_pid));
    assert(!vram_yield_requested(l) &&
           "a yield request for a different pid must not affect this lease");

    assert(vram_request_yield(gpu, (long)getpid()));
    assert(vram_yield_requested(l) &&
           "vram_request_yield for our own pid must be visible to our lease");
    // idempotent: checking again does not consume or clear it
    assert(vram_yield_requested(l));

    vram_yield_clear(l);
    assert(!vram_yield_requested(l) &&
           "vram_yield_clear must clear the request");

    vram_release(l);
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
                               0, fixed_free, &free_all, 0, NULL, NULL, NULL, 0);
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
                                      1024 * 1024, 0, fixed_free, &free_all, 0, NULL,
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
                       0, fixed_free, &free_all, 0, NULL, NULL, err, sizeof(err)));
    assert(strstr(err, "fresh-loader") && "a fresh guardless entry must survive");
    assert(!strstr(err, "stale-loader") && "the stale guardless entry must be gone");

    // with the stale 20GB dropped, a 20GB ask fits
    vram_lease *l = vram_claim("stale-pending-test", "/models/mine.gguf",
                               20 * GB, 0, fixed_free, &free_all, 0, NULL, NULL, NULL, 0);
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
                                   0, fixed_free, &free_all, 0, NULL, NULL, NULL, 0);
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
    assert(!vram_claim(gpu, "/models/mine.gguf", 20 * GB, 0, fixed_free, &free_all,
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
                                  0, fixed_free, &free_all, 0, NULL, NULL, err, sizeof(err));
    assert(mine && "a dead process's reservation must not poison the GPU");
    vram_release(mine);
}

// The same orphan one step earlier: the owner is dead, but nobody has reaped
// it yet.
//
// A zombie is a process-table entry, not a process. It answers kill(pid, 0),
// so plat_pid_alive called it alive and reap() kept its reservation — while it
// holds no VRAM, no weights and no context, and no signal can move it further.
// Only its parent's wait() can, and that may never come: a supervisor that
// does not reap, or a container whose PID 1 is not an init, leaves the entry
// standing indefinitely. The test above reaps the child before asking, which
// is why it never saw this.
//
// instances.c's instance_pid_alive already excludes zombies, for the tray's
// version of the same bug ("a permanent ghost row that Stop can never clear").
// The platform layer the VRAM ledger reaps through did not.
static void test_unreaped_dead_pid_reservation_is_reclaimed(void) {
    scratch_dir();
    const char *gpu = "MIG-zombie-pid-test";
    uint64_t free_all = 24 * GB;

    int ready[2];
    assert(pipe(ready) == 0);
    pid_t child = fork();
    assert(child >= 0);
    if (child == 0) {
        close(ready[0]);
        vram_lease *l = vram_claim(gpu, "/models/zombie-Q4_K_M.gguf", 20 * GB,
                                   0, fixed_free, &free_all, 0, NULL, NULL,
                                   NULL, 0);
        char ok = l ? 'y' : 'n';
        ssize_t w = write(ready[1], &ok, 1);
        (void)w;
        for (;;) pause();
    }
    close(ready[1]);
    char ok = 0;
    assert(read(ready[0], &ok, 1) == 1 && ok == 'y');
    close(ready[0]);

    kill(child, SIGKILL);
    // Wait for it to actually die WITHOUT reaping it: WNOWAIT leaves the
    // zombie in the table, which is the state a non-reaping supervisor sees.
    siginfo_t si;
    memset(&si, 0, sizeof si);
    assert(waitid(P_PID, (id_t)child, &si, WEXITED | WNOWAIT) == 0);
    assert(kill(child, 0) == 0 && "the zombie still answers kill(pid, 0)");

    char err[1024] = {0};
    vram_lease *mine = vram_claim(gpu, "/models/mine.gguf", 20 * GB, 0,
                                  fixed_free, &free_all, 0, NULL, NULL,
                                  err, sizeof(err));
    assert(mine && "an unreaped dead owner must not hold the GPU either");
    vram_release(mine);

    int st = 0;
    waitpid(child, &st, 0);
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
        vram_claim(gpu, "/models/doomed.gguf", 1 * GB, 0, fixed_free, &free_all,
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
                                      0, fixed_free, &free_all, 0, NULL, NULL, NULL, 0);
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
    assert(!vram_claim(gpu, "/models/huge.gguf", 23 * GB, 0, fixed_free, &free_all,
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
                                   0, fixed_free, &free_all, 0, NULL, NULL, NULL, 0);
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
    assert(!vram_claim(gpu, "/models/queued.gguf", 20 * GB, 0, fixed_free,
                       &free_all, 0, NULL, NULL, NULL, 0));

    // with waiting it blocks until the holder releases, then proceeds
    double t0 = plat_now();
    vram_lease *queued = vram_claim(gpu, "/models/queued.gguf", 20 * GB,
                                    0, fixed_free, &free_all, 30, NULL, NULL, NULL, 0);
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
                                 0, fixed_free, &free_all, 0, NULL, NULL, NULL, 0);
    assert(hog);

    atomic_store(&cancel_flag, 0);
    pthread_t th;
    assert(pthread_create(&th, NULL, cancel_setter, NULL) == 0);
    double t0 = plat_now();
    char err[1024] = {0};
    vram_lease *queued = vram_claim(gpu, "/models/queued.gguf", 20 * GB,
                                    0, fixed_free, &free_all, 30, &cancel_flag,
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
                                 0, fixed_free, &free_all, 0, NULL, NULL, NULL, 0);
    assert(hog);

    char err[1024] = {0};
    assert(!vram_claim(gpu, "/models/queued.gguf", 20 * GB, 0, fixed_free,
                       &free_all, 1, NULL, NULL, err, sizeof(err)));
    assert(strstr(err, "stubborn-Q4_K_M") && "a timed-out wait must still name the holder");

    vram_release(hog);
}

// A timed-out --wait-for-vram queue must not leave a phantom 'W' waiter
// marker behind: registry_rollback runs on both the cancel and the deadline
// exit, same as it always has for a post-admission allocation failure
// (RNR-013) — otherwise the ledger would keep naming a waiter that gave up.
static void test_wait_timeout_leaves_no_phantom_waiter(void) {
    const char *dir = scratch_dir();
    const char *gpu = "wait-timeout-phantom-test";
    char path[600];
    reg_file(dir, gpu, path, sizeof(path));
    uint64_t free_all = 24 * GB;

    vram_lease *hog = vram_claim(gpu, "/models/stubborn.gguf", 20 * GB, 0,
                                 fixed_free, &free_all, 0, NULL, NULL, NULL, 0);
    assert(hog);
    assert(!vram_claim(gpu, "/models/queued.gguf", 20 * GB, 5, fixed_free,
                       &free_all, 1, NULL, NULL, NULL, 0));

    vram_release(hog);
    assert(file_size(path) == 0 &&
           "a timed-out wait must not leave a phantom 'W' marker in the ledger");
}

// Advisory ordering among --wait-for-vram waiters: when two processes are
// both queued for the same space, the higher-priority one is admitted first
// once it frees. Not a hard guarantee — see vram_claim's header comment — but
// deterministic in this exact shape: both waiters ask for the full 20GB the
// holder frees, so only one can be admitted per round, and it must be the
// higher-priority one, not whichever happened to poll first.
static void test_higher_priority_waiter_acquires_first(void) {
    scratch_dir();
    const char *gpu = "MIG-priority-order-test";
    uint64_t free_all = 24 * GB;

    // holder takes 20 of 24GB, leaving 4GB — not enough for either waiter
    vram_lease *holder = vram_claim(gpu, "/models/holder.gguf", 20 * GB, 0,
                                    fixed_free, &free_all, 0, NULL, NULL, NULL, 0);
    assert(holder);

    int hi_ready[2], lo_ready[2], hi_done[2], lo_done[2];
    assert(pipe(hi_ready) == 0 && pipe(lo_ready) == 0);
    assert(pipe(hi_done) == 0 && pipe(lo_done) == 0);

    pid_t hi = fork();
    assert(hi >= 0);
    if (hi == 0) {
        close(hi_ready[0]); close(lo_ready[0]); close(lo_ready[1]);
        close(hi_done[0]); close(lo_done[0]); close(lo_done[1]);
        char rb = 'r';
        ssize_t w = write(hi_ready[1], &rb, 1); (void)w;
        vram_lease *l = vram_claim(gpu, "/models/hi.gguf", 20 * GB, 9 /* high */,
                                   fixed_free, &free_all, 10, NULL, NULL, NULL, 0);
        char c = l ? 'y' : 'n';
        w = write(hi_done[1], &c, 1); (void)w;
        if (l) { sleep(2); vram_release(l); }
        _exit(0);
    }
    close(hi_ready[1]);

    pid_t lo = fork();
    assert(lo >= 0);
    if (lo == 0) {
        close(lo_ready[0]); close(hi_ready[0]); close(hi_ready[1]);
        close(hi_done[0]); close(hi_done[1]); close(lo_done[0]);
        char rb = 'r';
        ssize_t w = write(lo_ready[1], &rb, 1); (void)w;
        vram_lease *l = vram_claim(gpu, "/models/lo.gguf", 20 * GB, 1 /* low */,
                                   fixed_free, &free_all, 10, NULL, NULL, NULL, 0);
        char c = l ? 'y' : 'n';
        w = write(lo_done[1], &c, 1); (void)w;
        if (l) vram_release(l);
        _exit(0);
    }
    close(lo_ready[1]);

    char rb;
    assert(read(hi_ready[0], &rb, 1) == 1);
    assert(read(lo_ready[0], &rb, 1) == 1);
    close(hi_ready[0]); close(lo_ready[0]);

    // give both children several poll cycles to register as 'W' waiters
    // before space frees
    plat_sleep_ms(1500);
    vram_release(holder);

    char hi_c = 0;
    assert(read(hi_done[0], &hi_c, 1) == 1);
    assert(hi_c == 'y' &&
           "the higher-priority waiter must acquire once space frees");
    close(hi_done[0]);

    // right after hi's win, lo must NOT have won too — hi took the only space
    // that freed, so lo's own poll around the same moment must still be
    // blocked (it has up to 10s left to eventually succeed once hi releases,
    // but that is not what this assertion is about)
    fd_set rs;
    struct timeval tv = {0, 0};
    FD_ZERO(&rs);
    FD_SET(lo_done[0], &rs);
    int has_data = select(lo_done[0] + 1, &rs, NULL, NULL, &tv);
    assert(has_data == 0 &&
           "the lower-priority waiter must not acquire ahead of the higher one");
    close(lo_done[0]);

    int st = 0;
    waitpid(hi, &st, 0);
    waitpid(lo, &st, 0);
}
#endif

int main(void) {
    test_second_runner_refuses_naming_the_holder();
    test_priority_tag_is_recorded_in_refusal();
    test_default_priority_matches_legacy_fields_plus_new_field();
    test_unreadable_registry_is_not_truncated();
    test_legacy_record_without_priority_reads_as_zero();
    test_refusal_line_priority_format_is_exact();
    test_refusal_does_not_blame_an_outsider_it_just_named();
    test_small_error_buffer_is_not_overrun();
    test_yield_flag_lifecycle();
#ifndef _WIN32
    test_symlinked_registry_is_refused();
    test_concurrent_claims_mint_distinct_seqs();
    test_stale_guardless_pending_is_reaped();
    test_dead_pid_reservation_is_reclaimed();
    test_unreaped_dead_pid_reservation_is_reclaimed();
    test_reaping_keeps_live_neighbours();
    test_wait_for_vram_queues_then_proceeds();
    test_cancelled_wait_gives_up_promptly();
    test_wait_timeout_still_names_the_holder();
    test_wait_timeout_leaves_no_phantom_waiter();
    test_higher_priority_waiter_acquires_first();
#else
    puts("vram registry: cross-process reaping tests need fork(); skipped on Windows");
#endif
    puts("vram registry tests ok");
    return 0;
}
