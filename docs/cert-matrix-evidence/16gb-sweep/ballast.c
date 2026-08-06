// Memory ballast for 16GiB-envelope live verification.
// Adaptively grows anonymous memory, filled with non-zero pseudo-random
// bytes, while polling the real cgroup memory.current, stopping once the
// cgroup-wide resident footprint approaches (memory.max - reserve_gib).
// Freezes at that point (no further touching/growing/shrinking) and holds
// until SIGTERM, at which point it exits and the OS reclaims everything.
//
// Hard rules followed (see docs/cert-matrix-goal-2026-08-05.md):
//  - regulate against cgroup memory.max, not /proc/meminfo (host lies)
//  - fill with random bytes, not zero (zero pages are lazy/deduped)
//  - pace the last GiB slower than swap-out/OOM-killer
//  - never touch oom_score_adj
//  - freeze size once at target; anonymous (not file-backed) memory can't
//    be displaced by page cache the way a file mapping could
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <sys/mman.h>
#include <time.h>

static volatile sig_atomic_t g_stop = 0;
static void on_term(int sig) { (void)sig; g_stop = 1; }

static long long read_u64_file(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    long long v = -1;
    if (fscanf(f, "%lld", &v) != 1) v = -1;
    fclose(f);
    return v;
}

// fast xorshift64 fill -- not cryptographic, just non-zero/non-repeating
// enough to defeat zero-page and same-page dedup shortcuts.
static uint64_t g_state;
static uint64_t xorshift64(void) {
    uint64_t x = g_state;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    return g_state = x;
}
static void fill_chunk(unsigned char *p, size_t n) {
    uint64_t *w = (uint64_t *)p;
    size_t words = n / 8;
    for (size_t i = 0; i < words; i++) w[i] = xorshift64();
}

#define GIB (1024ULL * 1024 * 1024)

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s reserve_gib [safety_margin_gib]\n", argv[0]);
        return 2;
    }
    double reserve_gib = atof(argv[1]);
    double safety_gib = argc > 2 ? atof(argv[2]) : 2.0;
    g_state = 0x9E3779B97F4A7C15ULL ^ (uint64_t)time(NULL);

    signal(SIGTERM, on_term);
    signal(SIGINT, on_term);

    long long mem_max = read_u64_file("/sys/fs/cgroup/memory.max");
    if (mem_max <= 0) {
        fprintf(stderr, "error: could not read /sys/fs/cgroup/memory.max\n");
        return 1;
    }
    long long reserve_bytes = (long long)(reserve_gib * GIB);
    long long safety_bytes = (long long)(safety_gib * GIB);
    long long target_ceiling = mem_max - reserve_bytes - safety_bytes;
    fprintf(stderr, "ballast: cgroup memory.max=%.2f GiB, reserving %.2f GiB for model "
                    "+ %.2f GiB safety margin, ballast target ceiling=%.2f GiB\n",
            mem_max / (double)GIB, reserve_gib, safety_gib, target_ceiling / (double)GIB);

    size_t held_bytes = 0;
    size_t chunk = 2ULL * GIB;
    int chunks_held = 0;
    void *chunks[4096];

    while (!g_stop) {
        long long cur = read_u64_file("/sys/fs/cgroup/memory.current");
        if (cur < 0) { fprintf(stderr, "error: could not read memory.current\n"); break; }
        if (cur + (long long)chunk >= target_ceiling) {
            // shrink chunk size as we approach the ceiling for finer control
            if (chunk > 256ULL * 1024 * 1024) { chunk /= 4; continue; }
            if (cur >= target_ceiling) break;
        }
        void *m = mmap(NULL, chunk, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (m == MAP_FAILED) { fprintf(stderr, "mmap failed at held=%.2f GiB, stopping\n",
                                        held_bytes / (double)GIB); break; }
        fill_chunk((unsigned char *)m, chunk);
        chunks[chunks_held++] = m;
        held_bytes += chunk;
        fprintf(stderr, "ballast: holding %.2f GiB (cgroup current=%.2f GiB, target ceiling=%.2f GiB)\n",
                held_bytes / (double)GIB, cur / (double)GIB, target_ceiling / (double)GIB);
        // pace: slower as chunks shrink (i.e. as we approach the target)
        struct timespec ts = { 0, (chunk <= 256ULL*1024*1024) ? 300000000L : 100000000L };
        nanosleep(&ts, NULL);
        if (chunks_held >= 4096) break;
    }

    fprintf(stderr, "ballast: FROZEN at %.2f GiB held, %d chunks. Sleeping until SIGTERM.\n",
            held_bytes / (double)GIB, chunks_held);
    fflush(stderr);

    while (!g_stop) {
        sleep(5);
        long long cur = read_u64_file("/sys/fs/cgroup/memory.current");
        fprintf(stderr, "ballast: alive, holding %.2f GiB, cgroup current=%.2f GiB\n",
                held_bytes / (double)GIB, cur / (double)GIB);
        fflush(stderr);
    }
    fprintf(stderr, "ballast: SIGTERM received, releasing %.2f GiB and exiting.\n",
            held_bytes / (double)GIB);
    return 0;
}
