// Platform abstraction: memory-mapped files, CPU count, paths, clocks.
// POSIX (Linux/macOS/BSD) and Windows (MinGW-w64 / MSYS2).
#ifndef RUNNER_COMPAT_H
#define RUNNER_COMPAT_H

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

// map a regular file read-only; returns NULL on failure (missing, empty,
// directory, ...). The mapping outlives any internal handles.
void  *plat_mmap_ro(const char *path, size_t *size);
void   plat_munmap(void *p, size_t size);

int         plat_cpu_count(void);
uint64_t    plat_ram_bytes(void);
bool        plat_file_readable(const char *path);
double      plat_now(void);        // monotonic seconds

#endif
