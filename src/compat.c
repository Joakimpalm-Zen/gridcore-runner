#include "compat.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

void plat_sleep_ms(int ms) {
    if (ms > 0) Sleep((DWORD)ms);
}

long plat_pid_self(void) {
    return (long)GetCurrentProcessId();
}

bool plat_pid_alive(long pid) {
    if (pid <= 0) return false;
    HANDLE h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, (DWORD)pid);
    // ERROR_ACCESS_DENIED means the process exists but belongs to someone else:
    // still alive, and its reservation is still real
    if (!h) return GetLastError() == ERROR_ACCESS_DENIED;
    DWORD code = 0;
    bool alive = GetExitCodeProcess(h, &code) && code == STILL_ACTIVE;
    CloseHandle(h);
    return alive;
}

bool plat_pid_start_time(long pid, uint64_t *out) {
    if (pid <= 0) return false;
    HANDLE h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, (DWORD)pid);
    if (!h) return false;
    FILETIME create, exit_t, kern, user;
    bool ok = GetProcessTimes(h, &create, &exit_t, &kern, &user);
    CloseHandle(h);
    if (!ok) return false;
    // FILETIME is 100ns ticks since 1601-01-01; 11644473600 seconds to Unix
    uint64_t ticks = ((uint64_t)create.dwHighDateTime << 32) | create.dwLowDateTime;
    *out = ticks / 10000000ull - 11644473600ull;
    return true;
}

const char *plat_runtime_dir(void) {
    const char *d = getenv("XDG_RUNTIME_DIR");
    if (d && *d) return d;
    if ((d = getenv("TEMP")) && *d) return d;
    if ((d = getenv("TMP")) && *d) return d;
    return ".";
}

bool plat_file_rmw(const char *path, plat_rmw_fn fn, void *ud) {
    HANDLE f = CreateFileA(path, GENERIC_READ | GENERIC_WRITE,
                           FILE_SHARE_READ | FILE_SHARE_WRITE, NULL,
                           OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    if (f == INVALID_HANDLE_VALUE) return false;

    OVERLAPPED ov = {0};
    // LockFileEx is advisory-by-convention here: every writer of this file goes
    // through plat_file_rmw. A failure is not fatal — see the header note.
    bool locked = LockFileEx(f, LOCKFILE_EXCLUSIVE_LOCK, 0, MAXDWORD, MAXDWORD, &ov);

    char *buf = NULL;
    size_t len = 0;
    LARGE_INTEGER sz;
    if (GetFileSizeEx(f, &sz) && sz.QuadPart > 0 && sz.QuadPart < (1 << 20)) {
        len = (size_t)sz.QuadPart;
        buf = malloc(len + 1);
        DWORD got = 0;
        if (!buf || !ReadFile(f, buf, (DWORD)len, &got, NULL)) { free(buf); buf = NULL; len = 0; }
        else { len = got; buf[len] = 0; }
    }

    char *out = fn(buf ? buf : "", len, ud);
    free(buf);

    if (out) {
        LARGE_INTEGER zero = {0};
        SetFilePointerEx(f, zero, NULL, FILE_BEGIN);
        SetEndOfFile(f);
        DWORD wrote = 0;
        WriteFile(f, out, (DWORD)strlen(out), &wrote, NULL);
        free(out);
    }

    if (locked) {
        OVERLAPPED uo = {0};
        UnlockFileEx(f, 0, MAXDWORD, MAXDWORD, &uo);
    }
    CloseHandle(f);
    return true;
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
#include <sys/file.h>
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

void plat_sleep_ms(int ms) {
    if (ms <= 0) return;
    struct timespec ts = { ms / 1000, (long)(ms % 1000) * 1000000L };
    nanosleep(&ts, NULL);
}

long plat_pid_self(void) {
    return (long)getpid();
}

bool plat_pid_alive(long pid) {
    if (pid <= 0) return false;
    // EPERM means the process exists but is owned by another user: alive, and
    // its reservation is still real. Only ESRCH proves it is gone.
    return kill((pid_t)pid, 0) == 0 || errno != ESRCH;
}

bool plat_pid_start_time(long pid, uint64_t *out) {
#ifdef __linux__
    if (pid <= 0) return false;
    char path[64];
    snprintf(path, sizeof(path), "/proc/%ld/stat", pid);
    FILE *f = fopen(path, "r");
    if (!f) return false;
    char buf[2048];
    size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    fclose(f);
    if (n == 0) return false;
    buf[n] = 0;
    // field 22 is starttime in clock ticks since boot. comm (field 2) is
    // parenthesised and may itself contain spaces, so scan from the LAST ')'.
    char *p = strrchr(buf, ')');
    if (!p) return false;
    p++;
    for (int field = 3; field <= 21; field++) {
        while (*p == ' ') p++;
        while (*p && *p != ' ') p++;
        if (!*p) return false;
    }
    unsigned long long ticks = strtoull(p, NULL, 10);
    long hz = sysconf(_SC_CLK_TCK);
    if (hz <= 0) return false;
    // relative to boot is enough: entries only ever compare against themselves
    *out = (uint64_t)(ticks / (unsigned long long)hz);
    return true;
#else
    (void)pid; (void)out;
    return false;   // macOS/BSD: pid liveness alone (see plat_pid_alive)
#endif
}

const char *plat_runtime_dir(void) {
    const char *d = getenv("XDG_RUNTIME_DIR");
    if (d && *d) return d;
    if ((d = getenv("TMPDIR")) && *d) return d;
    return "/tmp";
}

bool plat_file_rmw(const char *path, plat_rmw_fn fn, void *ud) {
    int fd = open(path, O_RDWR | O_CREAT, 0600);
    if (fd < 0) return false;

    // flock is advisory-by-convention here: every writer of this file goes
    // through plat_file_rmw. Some network filesystems refuse it; a failure is
    // not fatal, because an unlockable filesystem must not stop a runner.
    bool locked = flock(fd, LOCK_EX) == 0;

    char *buf = NULL;
    size_t len = 0;
    struct stat st;
    if (fstat(fd, &st) == 0 && st.st_size > 0 && st.st_size < (1 << 20)) {
        len = (size_t)st.st_size;
        buf = malloc(len + 1);
        ssize_t got = buf ? pread(fd, buf, len, 0) : -1;
        if (got < 0) { free(buf); buf = NULL; len = 0; }
        else { len = (size_t)got; buf[len] = 0; }
    }

    char *out = fn(buf ? buf : "", len, ud);
    free(buf);

    if (out) {
        size_t n = strlen(out);
        if (ftruncate(fd, 0) == 0) {
            ssize_t w = pwrite(fd, out, n, 0);
            (void)w;
        }
        free(out);
    }

    if (locked) flock(fd, LOCK_UN);
    close(fd);
    return true;
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
    bool attr_ok = pthread_attr_init(&at) == 0;
    if (!attr_ok || pthread_attr_setdetachstate(&at, PTHREAD_CREATE_DETACHED) != 0 ||
        pthread_create(&th, &at, parent_poll, (void *)(intptr_t)pid) != 0) {
        // this runs pre-model-load (no CUDA threads exist yet), so a plain
        // _exit is fine — but continuing unwatched would silently break
        // the flag's contract, so fail hard instead
        fprintf(stderr, "error: --parent-pid watcher thread failed to start — exiting\n");
        if (attr_ok) pthread_attr_destroy(&at);
        _exit(0);
    }
    pthread_attr_destroy(&at);
}

#endif
