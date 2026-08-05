// Backend contract: CUDA / Metal offload, and the batched decode half.
#ifndef RUNNER_GPU_H
#define RUNNER_GPU_H

#include "model.h"


enum { GPU_AUTO = 0, GPU_OFF = 1 };
bool   gpu_available(char *name, int name_cap);
// Stable identity of the GPU this backend would use, preferring a UUID: on a
// MIG box the parent card's UUID is the same for every slice, so an identity
// that cannot tell slices apart is worse than useless for VRAM accounting.
// Falls back to a bus id, then to "cuda:0". false when there is no GPU.
bool   gpu_device_id(char *id, int id_cap);
// dedicated GPU memory in bytes; false when there is no discrete GPU
// (Metal's unified memory is governed by the RAM reservation instead)
bool   gpu_mem_info(size_t *free_bytes, size_t *total_bytes);
// The GPU-side fit ceiling, which is NOT the same number as free RAM: on
// Metal a single resource allocation larger than the device working-set
// limit (~2/3 of unified memory) fails even when the file would fit in RAM.
// A scheduler that only reads ram_available_bytes sees capability where
// there is a capacity cliff. false when the backend has no such limit
// distinct from what vram_bytes already reports (CUDA), or no GPU.
bool   gpu_max_working_set(size_t *bytes);
// does this backend's attention read a q8_0 KV cache? The CPU path always
// can; a backend that cannot forces the cache back to f16 rather than
// handing q8_0 blocks to kernels that would read them as fp16.
bool   gpu_kv_q8_ok(void);
// Does this backend run at least one sparse-MoE family on the device? Per-model
// gpu_init() guards still decide the exact router/layout/activation variant.
bool   gpu_moe_ok(void);
// test hook for the TC tolerance gate: force the tensor-core GEMM opt-in on
// (1) or off (0) regardless of RUNNER_CUDA_TC; -1 returns to the env default.
// A no-op on backends without a TC path (Metal, CPU-only builds).
void   gpu_tc_force(int on);
// test hook: force (1) or forbid (0) the eager MoE routing path; -1 = env
void   gpu_moe_eager_force(int on);
bool   gpu_init(model_t *m);                     // false = unsupported, use CPU
// process n tokens starting at pos (prompt batches); on success returns true
// and sets *logits to the last token's logits when want_logits (else NULL).
// false = failed, use CPU.
bool   gpu_forward_batch(model_t *m, const int32_t *tokens, int n, int pos,
                         bool want_logits, float **logits);
void   gpu_free(model_t *m); // releases GPU buffers; KV pointers become invalid
// Stop using the GPU for this model after a runtime failure, releasing
// whatever the backend can release while leaving the model usable on the CPU.
// Distinct from gpu_free: a backend whose KV cache lives in GPU-owned memory
// (Metal's unified buffers) cannot free it here, because the CPU path is about
// to read those very rows. CUDA can and does — the host KV copy is
// authoritative there — so a slot that falls back mid-run hands its VRAM, and
// its reference to the shared weights, straight back.
void   gpu_disable(model_t *m);

// ------------------------------------------- batched decode (backend half)
//
// One decode step for several independent sequences in a single microbatch.
// Every sequence keeps its own KV cache, its own position and its own logits;
// what is shared is the *work*, because a decode step is weight-bandwidth
// bound and reading the weights once for N tokens costs barely more than
// reading them once for one.
//
// The contract that makes this usable is numerical, not structural: a batched
// step must produce, for each sequence, the bits a lone step would have
// produced. Nothing here reduces across sequences, and the matvecs pick the
// multi-column twin of whatever kernel the batch-1 path would have used, so
// that holds by construction rather than by luck. See gpu_batch_decode.
typedef struct gpu_batch gpu_batch;
// Group n sequences loaded from one file into a reusable microbatch context.
// NULL = this backend/model cannot batch, and the caller decodes one by one.
gpu_batch *gpu_batch_create(model_t **seqs, int n);
void       gpu_batch_free(gpu_batch *b);
// Evaluate one token for each of the n sequences named by idx (indices into
// the create() array). tok[i]/pos[i] are that sequence's token and KV write
// position; out[i] receives its logits, owned by the batch and valid until
// the next call. false = nothing was evaluated, decode these sequences singly.
bool       gpu_batch_decode(gpu_batch *b, const int *idx, const int32_t *tok,
                            const int *pos, int n, float **out);

#endif // RUNNER_GPU_H
