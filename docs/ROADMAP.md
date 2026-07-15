# Roadmap — known remaining work

Each item is self-contained enough to pick up from a fresh clone. Ordered
roughly by value to the Grid (gridcore-interpreter pipelines + gridcore-clu
agent turns).

## Metal parity (best done on actual Apple hardware)

- **Verify the `gpu_layers` fix live.** `metal.m` never set `m->gpu_layers`,
  so since partial offload landed the dispatcher re-ran the whole model on
  CPU after the GPU pass — Metal was slower than CPU-only. The one-line fix
  is in `gpu_init` (`m->gpu_layers = m->n_layer`), committed untested: no
  mac was available. Confirm output correctness + a real speedup vs
  `--gpu off` on a small model.
- **Port sliding-window + gemma3/gemma4 to Metal.** `metal.m` gates any
  model with `swa_window`, scaled embeddings, per-layer KV (`l_head_kv`),
  weightless V norm, or logit softcap back to CPU. The CUDA port
  (`cuda.c` history: "gemma4 on the CUDA backend") is the map: per-layer
  dims as launch args, V-reuses-K as zero+add, `ones` weight for the
  weightless V norm, `k_scale` kernel for the per-layer output scalar,
  softcap host-side. kernels.metal needs the same small additions
  kernels.cu got.

## Speculative decoding follow-ups

State: `--draft` works, output verified distribution-identical (self-draft
token-identical at temp 0; CI smoke covers it), telemetry line
`spec: N rounds, N drafted, N accepted`. Economics on an 8-core/3070 box:
break-even-plus for 7B+1.5B (4.07 vs 4.02 tok/s plain, K=4,
--repeat-penalty 1.0 — the penalty distorts target sampling away from the
penalty-less draft and kills acceptance).

- **gemma4 MTP draft head** — the natural gemma4 draft (no same-vocab small
  gemma exists). Separate ~254 MB GGUF (`mtp-gemma-4-12B-it.gguf` in the
  unsloth qat repo) consuming the target's post-final-norm hidden state
  recurrently. Needs its own loader + forward + verification against
  llama.cpp's implementation (they expose `t_h_nextn` for it). All
  prerequisites landed: spec engine, lazy verify logits
  (`model_spec_row_logits`), gemma4 on CUDA.
- **Server-side speculation** — landed: single-model `--serve` gives each
  slot its own draft `model_t` (weights dedupe through the page cache,
  same as slot targets), re-attached across `/unload` + lazy reload.
  Multi-model swap mode still refuses `--draft` (a registry entry can't
  guarantee a shared vocab). JSON/schema-constrained requests now use the
  same target-sampler equality verifier as unconstrained requests: the draft
  proposes tokens, but the target sampler plus the live constraint validator
  still decide every emitted token.

## Quants / kernels

- **IQ2/IQ3 codebook quants** — deliberately skipped so far; unlocks the
  smallest UD-quants for 8 GB nodes. Big kernel effort (grid codebooks).
- **CUDA kernel tuning headroom** — measurement-driven pass landed three
  experiments, each gated on median decode tok/s improving with
  byte-identical temp-0 output (RTX 3070, gated via
  `scripts/bench.sh`): (A) CUDA graphs for the single-token decode
  loop — `pos` moved into a device int so one stream capture replays for
  every decode token, RUNNER_CUDA_GRAPH_OFF=1 for A/B testing without a
  rebuild; Qwen3-4B-Q8_0 (full offload) 18.12 → 18.47 tok/s n=256
  (+1.9%). (B) 16-byte `uint4` vectorized quant loads in
  `k_mv_q4_K`/`k_mv_q4_K_b`; gemma-4-12B Q4_K_M (partial offload) 3.77 →
  3.90 tok/s n=256 (+3.4%). (C) `__half2` K/V loads in `k_attn`'s score
  and value loops (aligned 4-byte reads, half the load instructions);
  Qwen3-4B-Q8_0 n=512 decode 18.04 → 18.09 tok/s (+0.3%) — a smaller but
  real, reproducible gain, kept. Both partial-offload models (Llama-8B
  Q5_K_M, gemma-4-12B) were unaffected by A (graph path requires full
  offload); B improved gemma-4 (3.77 → 3.90, its decision target) without
  regressing Llama-8B, and C didn't regress either partial-offload model.
  Remaining headroom is smaller now that launch overhead and the hottest
  matvec/attention loads are addressed; further gains would need
  profiling to find the next bottleneck rather than obvious per-kernel
  wins.

## Architectures

- **gemma4 E2B/E4B (shared-KV / per-layer-embedding) variants** — hard-
  blocked at load with a clear error. Needs per-layer token embeddings,
  KV-sharing across trailing layers (llama.cpp `n_layer_kv_from_start`),
  and the altup/laurel-style extras in their reference.
- MoE and hybrid-SSM remain out of scope by design.
