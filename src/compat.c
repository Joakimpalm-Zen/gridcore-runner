#include "compat.h"

#include <stdio.h>
#include <stdlib.h>

#ifdef _WIN32

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <io.h>

void *plat_mmap_ro(const char *path, size_t *size) {
    HANDLE f = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
                           OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (f == INVALID_HANDLE_VALUE) return NULL;
    LARGE_INTEGER sz;
    if (!GetFileSizeEx(f, &sz) || sz.QuadPart <= 0) { CloseHandle(f); return NULL; }
    HANDLE m = CreateFileMappingA(f, NULL, PAGE_READONLY, 0, 0, NULL);
    CloseHandle(f); // the mapping keeps the file alive
    if (!m) return NULL;
    void *p = MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0);
    CloseHandle(m); // the view keeps the mapping alive
    if (!p) return NULL;
    *size = (size_t)sz.QuadPart;
    return p;
}

void plat_munmap(void *p, size_t size) {
    (void)size;
    if (p) UnmapViewOfFile(p);
}

int plat_cpu_count(void) {
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return si.dwNumberOfProcessors > 0 ? (int)si.dwNumberOfProcessors : 1;
}

uint64_t plat_ram_bytes(void) {
    MEMORYSTATUSEX ms = { .dwLength = sizeof(ms) };
    GlobalMemoryStatusEx(&ms);
    return ms.ullTotalPhys;
}

bool plat_file_readable(const char *path) {
    return _access(path, 4) == 0;
}

double plat_now(void) {
    static LARGE_INTEGER freq;
    if (freq.QuadPart == 0) QueryPerformanceFrequency(&freq);
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart / (double)freq.QuadPart;
}

static DWORD WINAPI parent_wait(LPVOID h) {
    WaitForSingleObject((HANDLE)h, INFINITE);
    fprintf(stderr, "parent process exited — shutting down\n");
    // _exit(0) maps to ExitProcess, which runs DLL_PROCESS_DETACH after
    // killing peer threads at arbitrary points; if a slot thread died
    // mid-CUDA-call, nvcuda's DllMain can deadlock on it, leaving a zombie
    // process holding VRAM. TerminateProcess skips DLL rundown entirely.
    TerminateProcess(GetCurrentProcess(), 0);
    return 0; // unreachable: the process is gone by the time we'd get here
}

void plat_parent_watch(long pid) {
    if (pid <= 0) return;
    HANDLE h = OpenProcess(SYNCHRONIZE, FALSE, (DWORD)pid);
    if (!h) {
        // unobservable parent == already-dead parent: refusing to run
        // unwatched is the whole point of the flag. This runs before model
        // load (no CUDA threads exist yet), so a plain _exit is fine here.
        fprintf(stderr, "error: --parent-pid %ld is not observable — exiting\n", pid);
        _exit(0);
    }
    HANDLE th = CreateThread(NULL, 0, parent_wait, h, 0, NULL);
    if (th) {
        CloseHandle(th);
    } else {
        // same rationale as the unobservable-pid branch above: this also
        // runs pre-model-load, so no CUDA threads exist yet and a plain
        // _exit is fine — but continuing unwatched would silently break
        // the flag's contract, so fail hard instead
        fprintf(stderr, "error: --parent-pid watcher thread failed to start — exiting\n");
        _exit(0);
    }
}

#else // POSIX

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <time.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

#ifdef __linux__
#include <sys/prctl.h>
#endif

void *plat_mmap_ro(const char *path, size_t *size) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return NULL;
    struct stat st;
    if (fstat(fd, &st) != 0 || !S_ISREG(st.st_mode) || st.st_size <= 0) {
        close(fd);
        return NULL;
    }
    void *p = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd); // the mapping persists after close
    if (p == MAP_FAILED) return NULL;
    *size = (size_t)st.st_size;
    return p;
}

void plat_munmap(void *p, size_t size) {
    if (p) munmap(p, size);
}

int plat_cpu_count(void) {
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? (int)n : 1;
}

uint64_t plat_ram_bytes(void) {
    long pages = sysconf(_SC_PHYS_PAGES);
    long psz   = sysconf(_SC_PAGE_SIZE);
    return pages > 0 && psz > 0 ? (uint64_t)pages * (uint64_t)psz : 0;
}

bool plat_file_readable(const char *path) {
    return access(path, R_OK) == 0;
}

double plat_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void *parent_poll(void *arg) {
    long pid = (long)(intptr_t)arg;
    for (;;) {
        struct timespec ts = { 2, 0 };
        nanosleep(&ts, NULL);
        if (kill((pid_t)pid, 0) != 0 && errno == ESRCH) {
            fprintf(stderr, "parent %ld exited — shutting down\n", pid);
            _exit(0);
        }
    }
    return NULL;
}

void plat_parent_watch(long pid) {
    if (pid <= 0) return;
#ifdef __linux__
    // instant path when the watched pid is the direct parent; the poll
    // below still covers grandparent supervisors and the pre-prctl race
    prctl(PR_SET_PDEATHSIG, SIGTERM);
#endif
    pthread_t th;
    pthread_attr_t at;
    pthread_attr_init(&at);
    pthread_attr_setdetachstate(&at, PTHREAD_CREATE_DETACHED);
    if (pthread_create(&th, &at, parent_poll, (void *)(intptr_t)pid) != 0) {
        // this runs pre-model-load (no CUDA threads exist yet), so a plain
        // _exit is fine — but continuing unwatched would silently break
        // the flag's contract, so fail hard instead
        fprintf(stderr, "error: --parent-pid watcher thread failed to start — exiting\n");
        pthread_attr_destroy(&at);
        _exit(0);
    }
    pthread_attr_destroy(&at);
}

#endif
