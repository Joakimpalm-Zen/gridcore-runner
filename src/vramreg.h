// Cross-process VRAM reservation registry.
#ifndef RUNNER_VRAMREG_H
#define RUNNER_VRAMREG_H

#include <stdint.h>
#include <stddef.h>

//
// A cross-process ledger of who is holding GPU memory, keyed by GPU device
// identity, living in $XDG_RUNTIME_DIR (else the temp dir). It is NOT a
// single-instance lock: several runners on one GPU are legitimate and the test
// suite depends on it. It exists so that when memory does run out, the refusal
// can name the holders instead of reporting a bare out-of-memory — and so that
// a runner killed with SIGKILL cannot leave a reservation poisoning the box.
//
// Entries owned by dead pids are reaped on every claim. CPU-only runs never
// claim, and so are never refused.
typedef struct vram_lease vram_lease;

// how much the device reports free, right now; the registry re-reads it while
// waiting. Callers on a real GPU pass a gpu_mem_info wrapper.
typedef uint64_t (*vram_free_fn)(void *ud);

// What the ledger looked like at the moment of the decision. `holders` is the
// count of *other* live runners registered on this GPU; a shortfall with zero
// holders is memory held by something outside runner's accounting, which is not
// a reason to refuse — there would be nobody to name and no orphan to blame.
typedef struct {
    int      holders;      // other live runners registered on this GPU
    uint64_t held_bytes;   // what they hold between them
    uint64_t available;    // free VRAM, less claims not yet allocated
} vram_status;

// Claim `need_bytes` on the GPU named by `gpu_id` (a UUID where the backend can
// supply one — a MIG slice must not share a ledger with its parent card).
//
// Returns a lease, or NULL when the request does not fit, in which case `err`
// holds a message naming every live holder: pid, model, bytes, uptime.
// wait_secs > 0 queues and retries until the deadline instead of refusing.
// `cancel` (nullable) aborts the queue wait early when *cancel goes nonzero.
// `st` (nullable) reports the ledger state behind the decision.
vram_lease *vram_claim(const char *gpu_id, const char *model_path,
                       uint64_t need_bytes, vram_free_fn free_fn, void *free_ud,
                       int wait_secs, const _Atomic int *cancel,
                       vram_status *st, char *err, size_t err_cap);

// The allocation happened: `actual_bytes` is now real device memory, visible to
// the driver's free figure, and stops being subtracted from it.
void        vram_commit(vram_lease *l, uint64_t actual_bytes);
// Clean exit. Unclean exits are covered by dead-pid reaping instead.
void        vram_release(vram_lease *l);

#endif // RUNNER_VRAMREG_H
