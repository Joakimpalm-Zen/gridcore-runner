// The device turn must be FIFO, not just mutually exclusive.
//
// A prefill gives the turn back between chunks so a request arriving mid-prompt
// runs at the next chunk boundary. That is only a hand-off if the queue is
// ordered. With a plain pthread mutex it is not: the releasing thread is already
// on-CPU with its threadpool hot, and re-acquires before the woken waiter is
// scheduled. Measured on Qwen2.5-7B with a 2891-token prompt, the waiter lost 44
// of 45 release/reacquire races and a short request still waited 25.3 s of the
// 26.4 s prefill. sched_yield() before reacquiring did not change it.
//
// So this test does not check mutual exclusion — a mutex passes that. It
// reproduces the barge: one thread holds the turn and releases-then-reacquires
// in a loop, exactly as the prefill yield does, while another waits for it once.
// The question is which cycle the waiter gets in on. FIFO says the next one.
//
// scheduler.c is #included rather than linked: the turnstile is static there,
// and widening the header for a test would make the seam look like an API.
#include "scheduler.c"

#include <stdio.h>

static _Atomic int g_cycle   = 0;   // release/reacquire cycles completed
static _Atomic int g_arrived = -1;  // cycle the waiter asked for the turn on
static _Atomic int g_entered = -1;  // cycle it actually got it on

enum { CYCLES = 200, PATIENCE = 5 };

// Stand-in for a prefill chunk: real work while the turn is held, so the
// releasing thread is genuinely running when it re-acquires.
static void chunk_work(void) {
    volatile double x = 1.0;
    for (int i = 0; i < 400000; i++) x = x * 1.0000001 + 1e-9;
    (void)x;
}

static void *waiter(void *unused) {
    (void)unused;
    // Arrival is recorded by the waiter itself, so the test measures how long
    // it was starved rather than how fast it started. Nothing here may touch
    // dev_mu: the main thread holds the turn, and probing the lock to find out
    // whether we are queued on it would deadlock a build where the turn IS the
    // lock -- which is exactly the build this test has to fail against.
    atomic_store(&g_arrived, atomic_load(&g_cycle));
    dev_take();
    atomic_store(&g_entered, atomic_load(&g_cycle));
    dev_give();
    return NULL;
}

// ts_in builds the absolute CLOCK_REALTIME deadline every timedwait in this
// file uses. It accumulated the nanoseconds in a `long`, which is 32 bits on
// the Windows build the CI matrix covers: 2.147e9 ns of headroom against a sum
// that already starts at up to 1e9 from tv_nsec. A caller asking for ~1.15 s
// or more wrapped to a NEGATIVE nanosecond count, and the deadline landed in
// the past — a "wait" that returns instantly, every time, forever.
//
// No caller does that today (SCHED_GATHER_S is 2 ms and the other is 0.5 s),
// which is exactly why this is worth pinning: nothing would report it, and the
// next timeout somebody adds is the one that finds out.
static int test_ts_in_deadlines(void) {
    static const double waits[] = { 0.002, 0.5, 1.147, 3.0, 60.0, 3600.0 };
    for (size_t i = 0; i < sizeof waits / sizeof *waits; i++) {
        struct timespec now, ts;
        timespec_get(&now, TIME_UTC);
        ts_in(&ts, waits[i]);
        if (ts.tv_nsec < 0 || ts.tv_nsec >= 1000000000L) {
            fprintf(stderr, "FAIL: ts_in(%.3f) produced tv_nsec %ld\n",
                    waits[i], (long)ts.tv_nsec);
            return 1;
        }
        double delta = (double)(ts.tv_sec - now.tv_sec) +
                       (double)(ts.tv_nsec - now.tv_nsec) * 1e-9;
        // generous: the point is "in the future by about what was asked for",
        // not a clock-accuracy measurement
        if (delta < waits[i] - 0.5 || delta > waits[i] + 0.5) {
            fprintf(stderr, "FAIL: ts_in(%.3f) is %.3f s away\n",
                    waits[i], delta);
            return 1;
        }
    }
    puts("ok: ts_in deadlines land in the future for every wait length");
    return 0;
}

int main(void) {
    if (test_ts_in_deadlines()) return 1;
    if (pthread_mutex_init(&SCH.dev_mu, NULL) != 0) return 77;
    if (pthread_cond_init(&SCH.dev_cv, NULL) != 0) return 77;
    SCH.dev_next = SCH.dev_serving = 0;

    dev_take();   // the "prefill" owns the turn before the waiter shows up

    pthread_t th;
    if (pthread_create(&th, NULL, waiter, NULL) != 0) return 77;
    while (atomic_load(&g_arrived) < 0) { /* spin until it has asked */ }

    for (int c = 0; c < CYCLES; c++) {
        chunk_work();
        atomic_fetch_add(&g_cycle, 1);
        if (atomic_load(&g_entered) >= 0) break;
        dev_give();
        dev_take();
    }
    dev_give();
    pthread_join(th, NULL);

    int entered = atomic_load(&g_entered), arrived = atomic_load(&g_arrived);
    if (entered < 0) {
        fprintf(stderr, "FAIL: the waiter never got the turn in %d cycles\n", CYCLES);
        return 1;
    }
    int starved = entered - arrived;
    if (starved > PATIENCE) {
        fprintf(stderr,
                "FAIL: the waiter was starved for %d release/reacquire cycles "
                "(FIFO owes it the next one, allowing %d)\n", starved, PATIENCE);
        return 1;
    }
    printf("ok: the waiter arrived on cycle %d and took the turn on %d\n",
           arrived, entered);
    puts("device turn ordering ok");
    return 0;
}
