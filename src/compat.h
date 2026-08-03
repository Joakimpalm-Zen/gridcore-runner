// Platform abstraction: memory-mapped files, CPU count, paths, clocks.
// POSIX (Linux/macOS/BSD) and Windows (MinGW-w64 / MSYS2).
#ifndef RUNNER_COMPAT_H
#define RUNNER_COMPAT_H

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>
#include <limits.h>

static inline bool checked_u64_add(uint64_t a, uint64_t b, uint64_t *out) {
    if (b > UINT64_MAX - a) return false;
    *out = a + b;
    return true;
}

static inline bool checked_u64_mul(uint64_t a, uint64_t b, uint64_t *out) {
    if (a != 0 && b > UINT64_MAX / a) return false;
    *out = a * b;
    return true;
}

// map a regular file read-only; returns NULL on failure (missing, empty,
// directory, ...). The mapping outlives any internal handles.
void  *plat_mmap_ro(const char *path, size_t *size);
void   plat_munmap(void *p, size_t size);

int         plat_cpu_count(void);
int         plat_default_thread_count(void);
uint64_t    plat_ram_bytes(void);

// --- weight residency -----------------------------------------------------
//
// runner mmaps its weights and the OS is free to evict them. On a machine
// under real memory pressure that turns every token into SSD paging, and it is
// invisible: tok/s, /health and --caps all stay green while a five-token reply
// takes a minute. Reported from a 16 GB M2 Pro driving an IDE, where a 1.2k
// prompt produced zero tokens in 300 s at 0% CPU. These four are what let
// runner say so.

// Wire the mapping into RAM. OPT-IN and expected to fail: locking 5 GB on a
// 16 GB machine can create the pressure it was meant to avoid, so a false
// return is a normal outcome to report, not an error to abort on.
bool        plat_mlock(void *p, size_t size);
void        plat_munlock(void *p, size_t size);

// Fraction of the mapping currently in physical memory, or -1.0 where the
// platform will not say. This is the direct answer to "is the model resident",
// which model *file size* cannot give: residency is a property of the moment.
double      plat_resident_fraction(const void *p, size_t size);

// Major faults this process has taken — page-ins that went to disk. The delta
// across a request is the exact mechanism behind a paging stall, which beats
// inferring one from wall-clock: a slow request that took no major faults was
// slow for some other reason. 0 where unavailable.
uint64_t    plat_major_faults(void);

// Physically available RAM: free plus what the OS would reclaim on demand. NOT
// total RAM, because "model fits in RAM" is the test that passes right before
// the machine starts thrashing. 0 where unavailable.
uint64_t    plat_ram_available_bytes(void);
bool        plat_file_readable(const char *path);
double      plat_now(void);        // monotonic seconds

// exit(0) this process when pid dies — supervisors pass their own pid so a
// SIGKILLed gridcore-clu cannot leave an orphaned runner. pid <= 0 = no-op.
void        plat_parent_watch(long pid);

void        plat_sleep_ms(int ms);
long        plat_pid_self(void);

// Is `pid` a process that still exists? Used to reap registry entries left by
// runners that died without deregistering (SIGKILL, crash, OOM killer).
bool        plat_pid_alive(long pid);

// Opaque process-creation stamp, false when the platform cannot report one.
// The unit is per-platform (Linux: seconds since boot; Windows/macOS: seconds
// since the epoch): values are only ever compared for equality against stamps
// recorded on the same machine, so files that store them must not outlive a
// boot. Pids are recycled, so an entry whose pid is alive but whose stamp has
// moved belongs to an unrelated process and is still dead for reaping purposes.
bool        plat_pid_start_time(long pid, uint64_t *out);

// Durable-but-transient scratch directory: $XDG_RUNTIME_DIR, else the platform
// temp directory, else /tmp. Never NULL.
const char *plat_runtime_dir(void);

// Read-modify-write `path` under an exclusive cross-process lock.
//
// The whole file is read, handed to `fn`, and whatever `fn` returns (a malloc'd
// NUL-terminated string, or NULL to leave the file untouched) is written back;
// the lock is held across all of it, so concurrent callers serialise. `in` is
// "" for a file that does not exist yet.
//
// Returns false only when the file could not be opened at all. When the
// platform has no working lock the read-modify-write still happens unlocked —
// an unlockable filesystem must not stop a runner from starting.
typedef char *(*plat_rmw_fn)(const char *in, size_t in_len, void *ud);
bool        plat_file_rmw(const char *path, plat_rmw_fn fn, void *ud);

// Strict numeric parsing shared by CLI flags and environment overrides.
// parse_* return false (leaving *out untouched) on empty/garbage/overflow/sign/
// non-finite input. env_* read getenv(name): absent keeps `cur`, invalid warns
// to stderr and keeps `cur` — never a silent zero.
bool      parse_i64(const char *s, long long lo, long long hi, long long *out);
bool      parse_u64(const char *s, uint64_t lo, uint64_t hi, uint64_t *out);
bool      parse_f64(const char *s, double lo, double hi, double *out);
long long env_i64(const char *name, long long lo, long long hi, long long cur);
uint64_t  env_u64(const char *name, uint64_t lo, uint64_t hi, uint64_t cur);
double    env_f64(const char *name, double lo, double hi, double cur);

#endif
