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
uint64_t    plat_ram_bytes(void);
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

// Process creation time in seconds since the epoch, false when the platform
// cannot report it. Pids are recycled, so an entry whose pid is alive but whose
// creation time has moved belongs to an unrelated process and is still dead for
// reaping purposes.
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

#endif
