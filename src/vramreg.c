// Cross-process VRAM reservation registry.
//
// Multiple runners on one GPU are legitimate and required — the conformance
// harness and several tests spawn their own servers. What was missing was
// *accounting*: six orphaned `runner --serve` processes once sat on a 24GB MIG
// slice, one of them for 4h39m, and nothing in runner knew the others existed.
// A VRAM check failed and then passed on re-run; a benchmark ran under
// contention and its numbers had to be thrown away.
//
// So this is not a lock. It is a ledger, keyed by GPU device identity, that
// answers one question: *who is holding what right now*. When a claim does not
// fit, the refusal names every holder — pid, model, bytes, uptime — instead of
// reporting a bare out-of-memory.
//
// Two states matter, and conflating them is the easy mistake:
//
//   pending    registered, not yet allocated. The driver's free-VRAM figure
//              does NOT know about these, so they must be subtracted.
//   committed  allocated. The driver's free-VRAM figure ALREADY reflects them,
//              so subtracting them again would double-count and refuse claims
//              that in fact fit.
//
// Committed entries therefore contribute nothing to the arithmetic and
// everything to the message. That is the whole point: the scarcity is measured
// from the device, the blame is read from the ledger.
//
// A third, non-claiming state exists for --wait-for-vram: a live 'W' marker
// records that some pid is queued and at what priority, so other waiters can
// see it, but it holds zero bytes and never enters the fit arithmetic.
//
// This module is deliberately small: it tags claims with an advisory priority
// and lets a holder be asked (never forced) to yield. Fair-share, priority
// lanes, and actual preemption are policy, and policy lives above this file —
// see vram_claim's header comment for exactly where the line sits.
#include "vramreg.h"
#include "compat.h"

#include <stdarg.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define REG_MAX_ENTRIES 256
#define REG_LINE_MAX    1024

typedef struct {
    long     pid;
    long     seq;          // distinguishes several claims from one process
    uint64_t procstart;    // pid-reuse guard; 0 when unavailable
    uint64_t since;        // wall-clock seconds, for "up 4h39m"
    char     state;        // 'P' pending, 'C' committed, 'W' waiting (advisory)
    uint64_t bytes;        // claimed/committed bytes; for 'W', the requested amount
    char     model[256];
    int      priority;     // advisory tag, default 0. Records from a binary
                           // older than this field are read as priority 0 (see
                           // parse(): a 7-field legacy line has no priority at
                           // all, and the zero-initialised reg_entry supplies it).
} reg_entry;

struct vram_lease {
    char     path[512];    // registry file this lease lives in
    long     pid;
    long     seq;
    uint64_t bytes;
};

// ---------------------------------------------------------------- formatting

// Deliberately the same 1e9-based "GB" the gpu-split line already prints, so
// two numbers about the same device never disagree in the same log.
static void fmt_bytes(uint64_t b, char *out, size_t cap) {
    if (b >= 1000000000ull)   snprintf(out, cap, "%.1fGB", b / 1e9);
    else if (b >= 1000000ull) snprintf(out, cap, "%.0fMB", b / 1e6);
    else                      snprintf(out, cap, "%lluB", (unsigned long long)b);
}

// Append to the refusal message, returning the new offset — saturating at
// `cap` rather than at what the text WOULD have been.
//
// snprintf reports the length it would have written, so `off += snprintf(...)`
// walks past a buffer too small to hold the sentence, and the next append then
// writes at err + off with a length of cap - off: an unsigned subtraction that
// underflows to a huge size and hands snprintf a licence to write anywhere.
// err_cap belongs to the caller and nothing in the message is bounded by this
// module — both the gpu id and the model path arrive from outside it — so the
// guard belongs here once, not at each append.
static size_t err_add(char *err, size_t cap, size_t off, const char *fmt, ...) {
    if (off >= cap) return cap;
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(err + off, cap - off, fmt, ap);
    va_end(ap);
    if (n < 0) return off;
    return off + (size_t)n > cap ? cap : off + (size_t)n;
}

// "4h39m" is the figure that made tonight's orphans obvious at a glance; a raw
// second count is not.
static void fmt_uptime(uint64_t secs, char *out, size_t cap) {
    if (secs >= 3600)    snprintf(out, cap, "%lluh%02llum",
                                  (unsigned long long)(secs / 3600),
                                  (unsigned long long)((secs % 3600) / 60));
    else if (secs >= 60) snprintf(out, cap, "%llum", (unsigned long long)(secs / 60));
    else                 snprintf(out, cap, "%llus", (unsigned long long)secs);
}

// The model as a human names it: basename, minus the .gguf.
static void model_label(const char *path, char *out, size_t cap) {
    if (!path || !*path) { snprintf(out, cap, "(unnamed)"); return; }
    const char *base = path;
    for (const char *p = path; *p; p++)
        if (*p == '/' || *p == '\\') base = p + 1;
    snprintf(out, cap, "%s", base);
    size_t n = strlen(out);
    if (n > 5 && !strcmp(out + n - 5, ".gguf")) out[n - 5] = 0;
}

// ---------------------------------------------------------------- file layout

// One file per GPU, so a MIG slice is a different ledger from its parent card
// and from a second card. The identity string goes in the filename, sanitised.
static void registry_path(const char *gpu_id, char *out, size_t cap) {
    const char *dir = getenv("RUNNER_VRAM_REGISTRY_DIR");
    if (!dir || !*dir) dir = plat_runtime_dir();
    char id[128];
    size_t n = 0;
    for (const char *p = gpu_id && *gpu_id ? gpu_id : "unknown";
         *p && n < sizeof(id) - 1; p++)
        id[n++] = (*p >= 'A' && *p <= 'Z') || (*p >= 'a' && *p <= 'z') ||
                  (*p >= '0' && *p <= '9') || *p == '.' || *p == '-' || *p == '_'
                  ? *p : '_';
    id[n] = 0;
    snprintf(out, cap, "%s/xyntetik-vram-%s.reg", dir, id);
}

// Tab-separated so a model path with spaces survives, one entry per line:
//   pid \t seq \t procstart \t since \t state \t bytes \t model [\t priority]
//
// The trailing priority field is new; a line written by a binary that
// predates it has exactly 7 fields (model is the last one, no trailing tab)
// and reads as priority 0 via the zero-initialised `e`. A line written by
// this binary always has 8 (model is followed by a tab, since the sanitised
// model text itself never contains one — see the tab-to-space scrub below).
// Both field counts are accepted; anything else is a corrupt line and dropped,
// same as today.
static int parse(const char *in, reg_entry *out, int cap) {
    int n = 0;
    for (const char *p = in; *p && n < cap; ) {
        const char *eol = strchr(p, '\n');
        size_t len = eol ? (size_t)(eol - p) : strlen(p);
        if (len > 0 && len < REG_LINE_MAX && *p != '#') {
            char line[REG_LINE_MAX];
            memcpy(line, p, len);
            line[len] = 0;
            reg_entry e = {0};
            char *save = line, *f[8];
            int nf = 0;
            for (; nf < 8; nf++) {
                f[nf] = save;
                char *t = strchr(save, '\t');
                if (!t) { nf++; break; }
                *t = 0;
                save = t + 1;
            }
            if (nf == 7 || nf == 8) {
                e.pid       = strtol(f[0], NULL, 10);
                e.seq       = strtol(f[1], NULL, 10);
                e.procstart = strtoull(f[2], NULL, 10);
                e.since     = strtoull(f[3], NULL, 10);
                e.state     = f[4][0] == 'C' ? 'C' : f[4][0] == 'W' ? 'W' : 'P';
                e.bytes     = strtoull(f[5], NULL, 10);
                snprintf(e.model, sizeof(e.model), "%s", f[6]);
                if (nf == 8) e.priority = (int)strtol(f[7], NULL, 10);
                if (e.pid > 0) out[n++] = e;
            }
        }
        if (!eol) break;
        p = eol + 1;
    }
    return n;
}

// Drop entries whose owner is gone. This is the line that fixes the orphan
// case: a SIGKILLed runner never deregisters, so its reservation would poison
// the machine until reboot if nobody reaped it. Reaping happens on every claim,
// under the lock, before any arithmetic.
static int reap(reg_entry *e, int n) {
    uint64_t now = (uint64_t)time(NULL);
    int keep = 0;
    for (int i = 0; i < n; i++) {
        if (!plat_pid_alive(e[i].pid)) continue;
        // The pid is alive, but pids get recycled. When the platform can report
        // process creation time, an owner that started later than the entry
        // claims is an unrelated process wearing a dead runner's pid.
        uint64_t start = 0;
        if (e[i].procstart && plat_pid_start_time(e[i].pid, &start) &&
            start != e[i].procstart) continue;
        // procstart 0 means no pid-reuse guard at all, so a recycled pid can
        // adopt a dead runner's pending entry and pin phantom bytes forever.
        // Age is the mitigation: no load — even one queued behind
        // --wait-for-vram — is still pending after an hour. A 'W' waiter
        // marker gets the same mitigation: a guardless one that stopped being
        // refreshed (its owner died) would otherwise block higher-priority
        // acquisition forever instead of just failing to help its own owner.
        if (e[i].procstart == 0 && (e[i].state == 'P' || e[i].state == 'W') &&
            now > e[i].since && now - e[i].since > 3600) continue;
        e[keep++] = e[i];
    }
    return keep;
}

static char *serialise(const reg_entry *e, int n) {
    size_t cap = (size_t)(n + 1) * REG_LINE_MAX;
    char *out = malloc(cap);
    if (!out) return NULL;
    size_t off = 0;
    for (int i = 0; i < n && off + REG_LINE_MAX < cap; i++)
        off += (size_t)snprintf(out + off, cap - off,
                                "%ld\t%ld\t%llu\t%llu\t%c\t%llu\t%s\t%d\n",
                                e[i].pid, e[i].seq,
                                (unsigned long long)e[i].procstart,
                                (unsigned long long)e[i].since,
                                e[i].state,
                                (unsigned long long)e[i].bytes, e[i].model,
                                e[i].priority);
    out[off] = 0;
    return out;
}

// ---------------------------------------------------------------- claim

typedef struct {
    // in
    const char  *gpu_id;
    const char  *model;
    uint64_t     need;
    int          priority;
    bool         waiting;   // true iff this call queues (wait_secs > 0):
                            // only then does a 'W' marker get written/removed
                            // and only then does the priority gate apply.
    vram_free_fn free_fn;
    void        *free_ud;
    long         pid;
    long         seq;
    uint64_t     procstart;
    // out
    bool         admitted;
    vram_status *st;
    char        *err;
    size_t       err_cap;
} claim_ctx;

static char *claim_rmw(const char *in, size_t in_len, void *ud) {
    (void)in_len;
    claim_ctx *c = ud;
    reg_entry e[REG_MAX_ENTRIES];
    int n = reap(e, parse(in, e, REG_MAX_ENTRIES));

    // A queueing call owns at most one 'W' marker, identified by (pid, seq) —
    // stable for the whole wait (vram_claim mints the seq once, before the
    // retry loop). Drop the stale copy now: this poll either replaces it with
    // a fresh one or promotes it straight to an admission.
    if (c->waiting) {
        int keep = 0;
        for (int i = 0; i < n; i++) {
            if (e[i].state == 'W' && e[i].pid == c->pid && e[i].seq == c->seq) continue;
            e[keep++] = e[i];
        }
        n = keep;
    }

    // Pending claims are in flight and invisible to the driver, so they come
    // off the free figure. Committed ones are already inside it. A 'W' marker
    // reserves nothing — it is a waiter's intent, not a claim on any bytes.
    uint64_t pending = 0;
    for (int i = 0; i < n; i++)
        if (e[i].state == 'P') pending += e[i].bytes;

    uint64_t freeb = c->free_fn ? c->free_fn(c->free_ud) : 0;
    uint64_t avail = freeb > pending ? freeb - pending : 0;

    if (c->st) {
        c->st->holders = 0;
        c->st->held_bytes = 0;
        c->st->available = avail;
        for (int i = 0; i < n; i++) {
            if (e[i].state == 'W') continue;    // a waiter holds nothing yet
            if (e[i].pid == c->pid) continue;   // our own other models
            c->st->holders++;
            c->st->held_bytes += e[i].bytes;
        }
    }

    // Advisory ordering among --wait-for-vram waiters (see vram_claim's
    // header comment for the exact rule): a waiter this poll would otherwise
    // admit still defers when another LIVE waiter outranks it AND that
    // higher-priority request also currently fits. A higher-priority request
    // that does not fit never blocks a smaller one that does — this is not a
    // reservation, so nothing here can starve a waiter whose ask is simply
    // larger than anyone will ever free in one step.
    bool outranked = false;
    if (c->waiting && c->need <= avail) {
        for (int i = 0; i < n; i++) {
            if (e[i].state != 'W') continue;
            if (e[i].priority > c->priority && e[i].bytes <= avail) {
                outranked = true;
                break;
            }
        }
    }

    if (c->need <= avail && !outranked) {
        if (n < REG_MAX_ENTRIES) {
            reg_entry me = { .pid = c->pid, .seq = c->seq,
                             .procstart = c->procstart,
                             .since = (uint64_t)time(NULL),
                             .state = 'P', .bytes = c->need,
                             .priority = c->priority };
            snprintf(me.model, sizeof(me.model), "%s", c->model ? c->model : "");
            for (char *p = me.model; *p; p++) if (*p == '\t' || *p == '\n') *p = ' ';
            e[n++] = me;
        }
        c->admitted = true;
        return serialise(e, n);
    }

    // Not admitted this round. A queueing caller leaves a fresh waiter marker
    // so others polling right now can see our priority and need.
    if (c->waiting && n < REG_MAX_ENTRIES) {
        reg_entry w = { .pid = c->pid, .seq = c->seq,
                        .procstart = c->procstart,
                        .since = (uint64_t)time(NULL),
                        .state = 'W', .bytes = c->need,
                        .priority = c->priority };
        snprintf(w.model, sizeof(w.model), "%s", c->model ? c->model : "");
        for (char *p = w.model; *p; p++) if (*p == '\t' || *p == '\n') *p = ' ';
        e[n++] = w;
    }

    // Refused. The message is the deliverable: a bare "out of memory" is what
    // sent someone chasing a phantom concurrent GPU user for an evening.
    if (c->err && c->err_cap) {
        char want[32], have[32];
        fmt_bytes(c->need, want, sizeof(want));
        fmt_bytes(avail, have, sizeof(have));
        size_t off = err_add(c->err, c->err_cap, 0,
            "%s of VRAM requested on %s, but only %s is available",
            want, c->gpu_id ? c->gpu_id : "the GPU", have);
        if (pending) {
            char p[32];
            fmt_bytes(pending, p, sizeof(p));
            off = err_add(c->err, c->err_cap, off,
                          " (%s free, %s claimed but not yet allocated)",
                          have, p);
        }
        // Keyed on what the listing below will actually print, NOT on
        // holder_count. holder_count excludes THIS process's own claims by
        // design (they are not "another runner"), while the listing prints
        // every non-waiter entry — so a runner blocked by its own earlier
        // claim was told the memory was held by something outside runner's
        // accounting, one line above its own pid being named as holding it.
        int listed = 0;
        for (int i = 0; i < n; i++)
            if (e[i].state != 'W') listed++;
        if (listed == 0)
            off = err_add(c->err, c->err_cap, off,
                ".\n  No other runner is registered on this GPU — the memory is "
                "held by something outside runner's accounting.");
        uint64_t now = (uint64_t)time(NULL);
        for (int i = 0; i < n && off + 128 < c->err_cap; i++) {
            if (e[i].state == 'W') continue;   // a waiter is not a holder
            char b[32], up[32], label[256];
            fmt_bytes(e[i].bytes, b, sizeof(b));
            fmt_uptime(now > e[i].since ? now - e[i].since : 0, up, sizeof(up));
            model_label(e[i].model, label, sizeof(label));
            off = err_add(c->err, c->err_cap, off,
                          "\n  pid %ld holding %s for %s, up %s, priority %d%s",
                          e[i].pid, b, label, up, e[i].priority,
                          e[i].state == 'P' ? " (loading)" : "");
        }
        if (off + 96 < c->err_cap)
            err_add(c->err, c->err_cap, off,
                    "\n  pass --wait-for-vram [SECONDS] to queue instead of failing");
    }
    // still write back: reaping dead owners is worth persisting even on refusal
    return serialise(e, n);
}

// Remove an already-committed (pid, seq) entry — used to roll a claim back when
// the lease handle cannot be allocated after admission (RNR-013). Defined via
// the shared edit_rmw in the commit/release section below.
static void registry_rollback(const char *path, long pid, long seq);

vram_lease *vram_claim(const char *gpu_id, const char *model_path,
                       uint64_t need_bytes, int priority,
                       vram_free_fn free_fn, void *free_ud,
                       int wait_secs, const _Atomic int *cancel,
                       vram_status *st, char *err, size_t err_cap) {
    // atomic: two slots claiming concurrently must not mint one seq, or
    // vram_release later removes the wrong entry
    static atomic_long next_seq = 1;
    if (err && err_cap) err[0] = 0;

    bool waiting = wait_secs > 0;
    claim_ctx c = { .gpu_id = gpu_id, .model = model_path, .need = need_bytes,
                    .priority = priority, .waiting = waiting,
                    .free_fn = free_fn, .free_ud = free_ud,
                    .pid = plat_pid_self(),
                    // One seq for the whole wait, not one per poll: a queueing
                    // call's 'W' marker is identified by (pid, seq), and it
                    // has to stay the same entry across retries for claim_rmw
                    // to find and refresh (or promote) it instead of leaving
                    // one orphaned waiter marker per poll.
                    .seq = atomic_fetch_add_explicit(&next_seq, 1,
                                                     memory_order_relaxed),
                    .st = st, .err = err, .err_cap = err_cap };
    if (st) { st->holders = 0; st->held_bytes = 0; st->available = 0; }
    if (!plat_pid_start_time(c.pid, &c.procstart)) c.procstart = 0;

    char path[512];
    registry_path(gpu_id, path, sizeof(path));

    // wait_secs > 0 turns refusal into a queue: retry until the holders leave.
    // The deadline is checked before the first attempt's result is honoured, so
    // --wait-for-vram 0 is exactly the refusing behaviour.
    double deadline = plat_now() + (wait_secs > 0 ? wait_secs : 0);
    for (;;) {
        c.admitted = false;
        if (!plat_file_rmw(path, claim_rmw, &c)) {
            // No registry available at all (read-only /tmp, exotic filesystem).
            // Accounting is best-effort infrastructure: never let its absence
            // stop a runner that would otherwise have started.
            c.admitted = true;
            if (err && err_cap) err[0] = 0;
        }
        if (c.admitted) break;
        if (cancel && atomic_load(cancel)) {
            if (waiting) registry_rollback(path, c.pid, c.seq);   // drop our 'W' marker
            if (err && err_cap) snprintf(err, err_cap, "vram wait cancelled");
            return NULL;
        }
        if (plat_now() >= deadline) {
            if (waiting) registry_rollback(path, c.pid, c.seq);   // drop our 'W' marker
            return NULL;
        }
        // sleep in short slices so a cancellation (an /unload or a shutdown
        // that wants this wait gone) is honoured within ~100ms, not a second
        for (int i = 0; i < 10 && !(cancel && atomic_load(cancel)); i++) plat_sleep_ms(100);
    }

    vram_lease *l = calloc(1, sizeof(*l));
    if (!l) {
        // The (pid, seq) entry is already committed but we have no handle to
        // release it later, and this live process will never be dead-PID
        // reaped — so roll the exact entry back now, or it would refuse future
        // runners against a reservation that never became real (RNR-013).
        registry_rollback(path, c.pid, c.seq);
        if (err && err_cap) snprintf(err, err_cap, "out of memory creating vram lease");
        return NULL;
    }
    snprintf(l->path, sizeof(l->path), "%s", path);
    l->pid = c.pid;
    l->seq = c.seq;
    l->bytes = need_bytes;
    return l;
}

// ---------------------------------------------------------------- commit/release

typedef struct {
    long     pid;
    long     seq;
    uint64_t bytes;
    bool     remove;
} edit_ctx;

static char *edit_rmw(const char *in, size_t in_len, void *ud) {
    (void)in_len;
    edit_ctx *x = ud;
    reg_entry e[REG_MAX_ENTRIES];
    int n = reap(e, parse(in, e, REG_MAX_ENTRIES));
    int keep = 0;
    for (int i = 0; i < n; i++) {
        bool mine = e[i].pid == x->pid && e[i].seq == x->seq;
        if (mine && x->remove) continue;
        if (mine) { e[i].state = 'C'; e[i].bytes = x->bytes; }
        e[keep++] = e[i];
    }
    return serialise(e, keep);
}

void vram_commit(vram_lease *l, uint64_t actual_bytes) {
    if (!l) return;
    // A backend that ended up using less than it asked for must say so, or the
    // slack it gave back stays invisible to the next runner. Growing is allowed
    // too — the number that matters is what the device actually holds.
    edit_ctx x = { .pid = l->pid, .seq = l->seq, .bytes = actual_bytes };
    plat_file_rmw(l->path, edit_rmw, &x);
    l->bytes = actual_bytes;
}

void vram_release(vram_lease *l) {
    if (!l) return;
    edit_ctx x = { .pid = l->pid, .seq = l->seq, .remove = true };
    plat_file_rmw(l->path, edit_rmw, &x);
    free(l);
}

static void registry_rollback(const char *path, long pid, long seq) {
    edit_ctx x = { .pid = pid, .seq = seq, .remove = true };
    plat_file_rmw(path, edit_rmw, &x);
}

// ---------------------------------------------------------------- cooperative yield
//
// A sentinel file next to the registry rather than a registry field: the
// requester is a different process from the holder and has no lease handle
// for it, only (gpu_id, pid) — exactly what a refusal listing already prints.
// Existence is the entire signal; content is never read, so no locking or
// read-modify-write is needed to check it, only to create or clear it.
static void yield_sentinel_path(const char *reg_path, long pid,
                                char *out, size_t cap) {
    char base[512];
    snprintf(base, sizeof(base), "%s", reg_path);
    size_t n = strlen(base);
    if (n > 4 && !strcmp(base + n - 4, ".reg")) base[n - 4] = 0;
    snprintf(out, cap, "%s.yield.%ld", base, pid);
}

// Content is irrelevant; a fixed non-empty payload just gives plat_file_rmw
// something to write through its existing O_NOFOLLOW-guarded open, the same
// symlink-planting defence the registry file itself relies on (see
// test_symlinked_registry_is_refused) — a raw fopen(path, "w") here would
// reopen that hole for a file whose whole path is guessable in advance.
static char *touch_rmw(const char *in, size_t in_len, void *ud) {
    (void)in; (void)in_len; (void)ud;
    char *s = malloc(2);
    if (s) { s[0] = '1'; s[1] = 0; }
    return s;
}

bool vram_request_yield(const char *gpu_id, long pid) {
    char reg[512], path[560];
    registry_path(gpu_id, reg, sizeof(reg));
    yield_sentinel_path(reg, pid, path, sizeof(path));
    return plat_file_rmw(path, touch_rmw, NULL);
}

bool vram_yield_requested(const vram_lease *l) {
    if (!l) return false;
    char path[560];
    yield_sentinel_path(l->path, l->pid, path, sizeof(path));
    return plat_file_readable(path);
}

void vram_yield_clear(const vram_lease *l) {
    if (!l) return;
    char path[560];
    yield_sentinel_path(l->path, l->pid, path, sizeof(path));
    remove(path);
}
