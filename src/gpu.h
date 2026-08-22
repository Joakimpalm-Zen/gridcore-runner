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
// SHA-256 of the shader source compiled INTO this binary, or NULL for a
// backend with no embedded shader source. Published in --caps so a gate can
// prove the running binary carries the current kernels: the header-vs-source
// drift check cannot see a stale binary, and twice on 2026-08-07 a kernel
// change was measured against a build that did not contain it.
const char *gpu_shader_source_sha(void);
// does this backend's attention read a q8_0 KV cache? The CPU path always
// can; a backend that cannot forces the cache back to f16 rather than
// handing q8_0 blocks to kernels that would read them as fp16.
bool   gpu_kv_q8_ok(void);
// Does this backend have kernels for a tensor format? Answered by the backend
// itself so `--caps` cannot drift from what the loader will actually admit.
// The two used to be a hand-kept literal in main.c and a switch in the backend
// carrying "keep this in sync with gpu_type_ok()" — and they did drift, on
// Metal, in the direction that made --caps advertise formats it could not run.
// A published surface an advisor consumes must not be able to lie by omission
// of a manual edit, so there is now one statement of the fact per backend.
bool   gpu_quant_ok(int type);
// Does this backend run at least one sparse-MoE family on the device? Per-model
// gpu_init() guards still decide the exact router/layout/activation variant.
bool   gpu_moe_ok(void);
// Does this backend run the gemma-4 E-series per-layer-embedding path on the
// device? Same shape of claim as gpu_moe_ok: a scheduler asking "can this box
// serve an E-series model on its GPU" needs an answer before it hands one over,
// and until 0.1.11 the honest answer on Metal was no. Per-model gpu_init()
// guards still decide the rest (E2B's per-layer FFN widths, for one).
bool   gpu_eseries_ok(void);
// test hook for the TC tolerance gate: force the tensor-core GEMM opt-in on
// (1) or off (0) regardless of RUNNER_CUDA_TC; -1 returns to the env default.
// A no-op on backends without a TC path (Metal, CPU-only builds).
void   gpu_tc_force(int on);
// How many times the tensor-core GEMM has actually been dispatched. The TC
// tolerance gate used to infer engagement from "the logits differ", which
// cannot tell an unused kernel apart from an exactly-matching one — and Q8_0
// turns out to be the second case. Counting the dispatch removes the guess.
unsigned long gpu_tc_dispatches(void);
// test hook for the fast-matvec tolerance gate: force the reassociating decode
// matvec on (1) or off (0) regardless of RUNNER_METAL_MV; -1 returns to the
// env default. A no-op on backends without one (CUDA, CPU-only builds).
//
// Why a second kernel rather than a faster first one: the shipped matvec is
// the CPU<->GPU byte-identity contract, which pins which lane owns which
// block and therefore forbids every reassociating transformation worth having
// (see docs/negative-result-metal-multirow-matvec.md). The fast route answers
// to tests/test_mv_tol.c the way the tiled prefill GEMM answers to
// test_tc_tol.c, and the identity route stays reachable and unchanged.
void   gpu_mv_force(int on);
// How many times the fast matvec has actually been dispatched. Same reason the
// TC path counts: "the logits differ" cannot tell an unused kernel apart from
// an exactly-matching one, and a gate that cannot see its own subject passes
// vacuously.
unsigned long gpu_mv_dispatches(void);
// test hook for the cooperative-KV attention gate: force the reassociating
// score read on (1) or off (0) regardless of RUNNER_METAL_ATTN_COOP; -1
// returns to the env default. A no-op on backends without one.
void   gpu_attn_coop_force(int on);
unsigned long gpu_attn_coop_dispatches(void);
// test hook: force (1) or forbid (0) the eager MoE routing path; -1 = env
void   gpu_moe_eager_force(int on);
bool   gpu_init(model_t *m);                     // false = unsupported, use CPU
// process n tokens starting at pos (prompt batches); on success returns true
// and sets *logits to the last token's logits when want_logits (else NULL).
// false = failed, use CPU.
// transposed matvec on the device (training backward, D8 slice 1): dx += W^T
// dy with the CPU trainer's exact accumulation chain; false = no backend /
// no kernel for the type, the caller stays on the CPU path.
bool   gpu_mvt(model_t *m, const gguf_tensor *w, const float *dy, float *dx,
               int n_in, int n_out, int batch);

// D8 slice 2: the standalone training context. --train keeps the model
// CPU-resident (m->gpu stays NULL); this context owns its own CUDA state,
// uploads each weight tensor ONCE on first use, and runs the backward's
// dx += W^T dy on the device — the kernel proven bit-identical to the CPU
// chain in slice 1, so a call that returns false (no CUDA, no kernel for
// the type, VRAM budget exhausted) just leaves that tensor on the CPU path
// with the SAME bytes as the result. Opt-in from --train via
// RUNNER_TRAIN_GPU=1. All three are no-ops / false without CUDA.
bool   gpu_train_init(model_t *m);
void   gpu_train_free(model_t *m);
bool   gpu_train_mvt(model_t *m, const gguf_tensor *w, const float *dy,
                     float *dx, int n_in, int n_out, int nb);

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
