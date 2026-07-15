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
- **Server-side speculation** — currently CLI-only because slots would
  share one draft KV cache; per-slot draft contexts would enable it for
  gridcore-clu's server calls (which are schema-constrained though — spec
  currently disables itself under schema/JSON constraints; lifting THAT
  needs constraint-state rewind per rejected draft).

## Server

- **Job-object / process-group cleanup story** for supervisors that die
  (gridcore-clu leaves orphaned runners on SIGKILL).

## Quants / kernels

- **IQ2/IQ3 codebook quants** — deliberately skipped so far; unlocks the
  smallest UD-quants for 8 GB nodes. Big kernel effort (grid codebooks).
- **CUDA kernel tuning headroom** — kernels are correctness-first PTX;
  CUDA graphs / coalesced loads were noted as "big headroom left" when the
  backend landed.

## Architectures

- **gemma4 E2B/E4B (shared-KV / per-layer-embedding) variants** — hard-
  blocked at load with a clear error. Needs per-layer token embeddings,
  KV-sharing across trailing layers (llama.cpp `n_layer_kv_from_start`),
  and the altup/laurel-style extras in their reference.
- MoE and hybrid-SSM remain out of scope by design.
