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
#include "runner.h"
#include "compat.h"

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
    char     state;        // 'P' pending, 'C' committed
    uint64_t bytes;
    char     model[256];
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
    snprintf(out, cap, "%s/gridcore-vram-%s.reg", dir, id);
}

// Tab-separated so a model path with spaces survives, one entry per line:
//   pid \t seq \t procstart \t since \t state \t bytes \t model
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
            char *save = line, *f[7];
            int nf = 0;
            for (; nf < 7; nf++) {
                f[nf] = save;
                char *t = strchr(save, '\t');
                if (!t) { nf++; break; }
                *t = 0;
                save = t + 1;
            }
            if (nf == 7) {
                e.pid       = strtol(f[0], NULL, 10);
                e.seq       = strtol(f[1], NULL, 10);
                e.procstart = strtoull(f[2], NULL, 10);
                e.since     = strtoull(f[3], NULL, 10);
                e.state     = f[4][0] == 'C' ? 'C' : 'P';
                e.bytes     = strtoull(f[5], NULL, 10);
                snprintf(e.model, sizeof(e.model), "%s", f[6]);
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
        // --wait-for-vram — is still pending after an hour.
        if (e[i].procstart == 0 && e[i].state == 'P' &&
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
                                "%ld\t%ld\t%llu\t%llu\t%c\t%llu\t%s\n",
                                e[i].pid, e[i].seq,
                                (unsigned long long)e[i].procstart,
                                (unsigned long long)e[i].since,
                                e[i].state,
                                (unsigned long long)e[i].bytes, e[i].model);
    out[off] = 0;
    return out;
}

// ---------------------------------------------------------------- claim

typedef struct {
    // in
    const char  *gpu_id;
    const char  *model;
    uint64_t     need;
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

    // Pending claims are in flight and invisible to the driver, so they come
    // off the free figure. Committed ones are already inside it.
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
            if (e[i].pid == c->pid) continue;   // our own other models
            c->st->holders++;
            c->st->held_bytes += e[i].bytes;
        }
    }

    if (c->need <= avail) {
        if (n < REG_MAX_ENTRIES) {
            reg_entry me = { .pid = c->pid, .seq = c->seq,
                             .procstart = c->procstart,
                             .since = (uint64_t)time(NULL),
                             .state = 'P', .bytes = c->need };
            snprintf(me.model, sizeof(me.model), "%s", c->model ? c->model : "");
            for (char *p = me.model; *p; p++) if (*p == '\t' || *p == '\n') *p = ' ';
            e[n++] = me;
        }
        c->admitted = true;
        return serialise(e, n);
    }

    // Refused. The message is the deliverable: a bare "out of memory" is what
    // sent someone chasing a phantom concurrent GPU user for an evening.
    if (c->err && c->err_cap) {
        char want[32], have[32];
        fmt_bytes(c->need, want, sizeof(want));
        fmt_bytes(avail, have, sizeof(have));
        size_t off = (size_t)snprintf(c->err, c->err_cap,
            "%s of VRAM requested on %s, but only %s is available",
            want, c->gpu_id ? c->gpu_id : "the GPU", have);
        if (pending) {
            char p[32];
            fmt_bytes(pending, p, sizeof(p));
            off += (size_t)snprintf(c->err + off, c->err_cap - off,
                                    " (%s free, %s claimed but not yet allocated)",
                                    have, p);
        }
        if (n == 0)
            off += (size_t)snprintf(c->err + off, c->err_cap - off,
                ".\n  No other runner is registered on this GPU — the memory is "
                "held by something outside runner's accounting.");
        uint64_t now = (uint64_t)time(NULL);
        for (int i = 0; i < n && off + 128 < c->err_cap; i++) {
            char b[32], up[32], label[256];
            fmt_bytes(e[i].bytes, b, sizeof(b));
            fmt_uptime(now > e[i].since ? now - e[i].since : 0, up, sizeof(up));
            model_label(e[i].model, label, sizeof(label));
            off += (size_t)snprintf(c->err + off, c->err_cap - off,
                                    "\n  pid %ld holding %s for %s, up %s%s",
                                    e[i].pid, b, label, up,
                                    e[i].state == 'P' ? " (loading)" : "");
        }
        if (off + 96 < c->err_cap)
            snprintf(c->err + off, c->err_cap - off,
                     "\n  pass --wait-for-vram [SECONDS] to queue instead of failing");
    }
    // still write back: reaping dead owners is worth persisting even on refusal
    return serialise(e, n);
}

vram_lease *vram_claim(const char *gpu_id, const char *model_path,
                       uint64_t need_bytes, vram_free_fn free_fn, void *free_ud,
                       int wait_secs, vram_status *st, char *err, size_t err_cap) {
    // atomic: two slots claiming concurrently must not mint one seq, or
    // vram_release later removes the wrong entry
    static atomic_long next_seq = 1;
    if (err && err_cap) err[0] = 0;

    claim_ctx c = { .gpu_id = gpu_id, .model = model_path, .need = need_bytes,
                    .free_fn = free_fn, .free_ud = free_ud,
                    .pid = plat_pid_self(),
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
        if (plat_now() >= deadline) return NULL;
        plat_sleep_ms(1000);
        c.seq = atomic_fetch_add_explicit(&next_seq, 1, memory_order_relaxed);
    }

    vram_lease *l = calloc(1, sizeof(*l));
    if (!l) return NULL;
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
