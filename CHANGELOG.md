# Changelog

All notable changes to gridcore-runner. This project is in **alpha**; the HTTP
protocol and CLI may still change between alpha releases.

## Unreleased

- **`choice_logprobs` — constrained-choice posteriors (JC-R1).** Constrained
  requests (JSON mode / `json_schema` / tool schemas) can set
  `"choice_logprobs": true` to get, per decision point (a step where ≥ 2 of
  the probed top-`M` candidates were grammar-legal; `choice_logprobs_probe`
  8–64, default 32), the legal alternatives with a posterior renormalized
  over the legal probed set, raw full-vocab logprobs, and probed coverage.
  Captured from raw logits before the repeat penalty, payload phase only
  (thinking preludes have no decision points), legality decided by the same
  validator trial the sampler uses. Buffered responses only; rejected with
  spec-decode. New `scripts/cl-calibration.py` turns labeled decision
  records into accuracy/Brier/ECE + a reliability table and can gate via
  `--max-ece`. Conformance: `tests/conformance/test_choice_logprobs.py`.
- Published benchmark MoE rows updated after the device-routing work:
  Qwen3-30B-A3B decode 102.2 tok/s (67% of llama.cpp, was 48% at
  v0.1.4) and prefill 194.0 (6.0%, was 3.3%) on the Blackwell MIG;
  gemma-4-26B 24.7 / 23.6. docs/benchmarks.md and the shareable page
  carry the same rows.
- `runner` now depends on `src/kernels_ptx.h` in the Makefile: a pull
  that changed only the regenerated PTX header rebuilt nothing, and a
  publication run measured yesterday's kernels for half an hour before
  the stale binary was caught. Same class as the certification footguns
  this week — the build must never silently serve old code.

- MoE routing normalizations reverted to per-element division — the
  k_moe_route PTX section is byte-identical to the 4719de6 body again
  (verified by section hash), which is exactly what the Blackwell
  splice-proof restored to 102.9 tok/s decode (the reciprocal-multiply
  mirror's rcp.rn.f32 codegen JITed ~58 µs/launch slower on that MIG,
  ×48 layers = 23% of MoE decode; the mirror's bit-identity purpose was
  already retired by the eager certification pin). Fused-path selw bound
  restated in the compat doc: within ~2 ulp of the host reference (two
  independent 1-ulp sources), observed 1 ulp at the first routing on
  both cert boxes. Gates on the 3070: make test green; eager-pinned
  CPU==GPU identity byte-green on both MoE models; bench md5 unchanged.

- MoE routing exp reverted to fp32 device expf (keeping the
  reciprocal-multiply mirror), now that certification pins the eager
  path (`RUNNER_MOE_EAGER=1` in the harnesses since bf93510): the
  correctly-rounded double-exp's only purpose was bit-matching
  correctly-rounded hosts, a property void on the fast-math cert box.
  NOTE the property downgrade this trades away: the fused default is no
  longer byte-identical to the host routing even on correctly-rounded
  (UCRT-class) hosts — its contract is now the verified weaker class,
  expert selection identical + selw within 1 ulp of the host reference
  (re-verified on the 3070 with RUNNER_DEBUG_MOE after the revert;
  docs/compatibility-program.md updated to match). Gates on the 3070:
  make test green; certified (eager-pinned) CPU==GPU identity green on
  both MoE models over 128 greedy tokens; bench md5 unchanged.

- Fixed the full-offload CPU==GPU byte-identity regression the Blackwell
  box found in the P1 MoE path (near-tie flip at ~token 60 on
  Qwen3-30B). Isolated with a new routing-bits discriminator: expert
  SELECTION was identical, but selw differed in the last mantissa bit
  from two compounding 1-ulp sources in k_moe_route — device expf vs the
  host libm, and exact IEEE division vs the -freciprocal-math
  reciprocal-multiply the -ffast-math host build actually emits. The
  kernel now computes the softmax exponential as
  (float)exp((double)x) — the correctly-rounded float exp, which
  bit-matches a correctly-rounded host expf (verified against UCRT on
  20M sampled inputs; residual double-rounding probability ~2^-28 per
  call) — and mirrors the reciprocal-multiply normalization. Verified on
  the 3070: fused output byte-identical to the eager (v0.1.4-certified)
  path over 128 greedy tokens on BOTH MoE models at ngl 17 and 19, and
  CPU==GPU identical; layer-1 routing bits equal, which transitively
  certifies every P1 kernel in layer 0's pipeline. Caveat for the
  full-offload re-cert: if the Blackwell CPU build auto-vectorizes the
  host softmax through libmvec's ~4-ulp expf, its CPU routing bits are
  unmatchable from device code — the new RUNNER_DEBUG_MOE dump (hex
  sel/selw from both paths) + RUNNER_MOE_EAGER (force the v0.1.4
  host-routing path) discriminate that case in two runs.
- (Q4_0, gemma4-moe) tc-tol failure investigated: the tail-safe K loop
  is NOT the cause — a synthetic n_ff=704 Q4_0 model (the exact
  X.5x128-step class gemma exercises) passes at 0.00004 of range with
  0/64 flips, and a 10x-weight variant scales the deviation smoothly
  (0.00022, still 0 flips), pointing at gemma-4's activation magnitudes
  under fp16 tile staging rather than a kernel defect. The gate is doing
  its job; the row stays unpromoted. If that row should ever pass:
  per-column activation absmax scaling in the TC tile is the identified
  follow-up.

- Tensor-core GEMM twins for Q8_0 and Q4_0 (moe-gpu-routing spec P3),
  same MMQ-style structure as the Q4_K TC kernel — block-cooperative
  fp16 weight-tile dequant with per-element values matching the scalar
  kernels exactly, fp32-accumulated m16n16k16 MMAs — plus a tail-safe K
  loop (gemma-4-MoE's n_ff_exp=704 is not a 128-multiple; elements past
  n_in stage as zeros). Both OPT-IN behind RUNNER_CUDA_TC and the
  per-(type, arch) test-tc-tol gate; tc_promoted() is unchanged, so no
  default path moved. test-tc-tol now recognizes all TC-capable types
  (was: hard-skip without Q4_K). Fresh gate rows measured on the 3070
  (full offload, 64 teacher-forced positions, 0 top-1 flips each):
  (Q8_0, qwen3) 0.00005 of range, (Q8_0, phi3) 0.00002, (Q4_0, qwen3
  requantized) 0.00003 — all far under the 0.005 bound. Promotion
  remains the owner's decision with the Blackwell rows.

- MoE GPU prefill: expert-grouped GEMM (moe-gpu-routing spec P2) — the
  CUDA port of the CPU `cabdad1` grouping. A prefill tile is routed on
  device, its routing read back once (prefill is never graph-captured),
  and each active expert then runs ONCE over all its routed tokens with
  the batched k_gemm/k_mv_b kernels — expert weights stream through the
  SMs once per tile instead of once per (token, slot); the per-token
  sync+DtoH per MoE layer collapses to one per layer per tile.
  Accumulation mirrors the CPU grouped path (ascending expert index,
  routing weight at the scatter). gemma-4's dense shared branch now also
  batches across the tile instead of looping per token.
  Follow-up measured on the 3070: the fixed-width GEMM kernels compute
  all 16 tile columns whatever the batch, and a 16-token tile routes only
  ~1-2 tokens per expert, so naive grouping LOST GPU time (gemma Q4_0
  prefill −19%). Expert matmuls now pick the narrowest kernel that
  covers the token count (batch-1 GEMV at 1, the width-classed f_gemvb
  twins to 8, the full GEMM beyond — TC whenever promoted/forced), and
  grouping engages only for expert types with width-classed kernels
  (Q8_0/Q4_K/Q5_K/Q6_K); gemma-4 Q4_0 keeps the per-token fused prefill
  until a batched Q4_0 kernel exists. 3070 result: gemma regression
  erased (14.8 tok/s prefill, above the fused path), qwen at end-to-end
  parity with GPU-busy still 1.65× the fused path's — the grouped win at
  this tile size has to come from TC on the expert GEMMs; the Blackwell
  box should A/B fused-vs-grouped prefill when re-measuring.
  Certified: CPU==GPU
  greedy byte-identical (short 128-tok and 510-tok-prefill configs on
  Qwen3-30B; short config on gemma-4-26B), pinned-b10076 text unchanged,
  bench md5s unchanged, make test green. Noted: gemma-4-26B long-prefill
  CPU-vs-GPU greedy divergence on a pathological repetitive prompt
  pre-exists in v0.1.4-alpha (P2's GPU output is byte-identical to the
  eager path's there — no regression; tracked in the suite plan).

- MoE GPU decode: device-side routing + fused indirect expert matvecs
  (moe-gpu-routing spec P1). Softmax → top-k → renormalize now runs in a
  serial-per-token device kernel that mirrors the host reference bit for
  bit (same scan/summation order, ties to lowest index), and one indirect
  launch per projection covers all top-k experts, reading `sel[]` on
  device — the per-token `cuStreamSynchronize` + DtoH round-trips are
  gone (~48/token on Qwen3-30B) and launches per decoded token fall
  ~4× on the MoE fixture (52.0 → 1.3 with the graph, see below). With no
  host-dependent branching left, fused-layout MoE no longer forces
  `graph_bad`: full-offload MoE decode is CUDA-graph captured. The
  legacy split-expert layout and expert quant types without an indirect
  kernel (outside F32/F16/Q8_0/Q4_0/Q4_K/Q5_K/Q6_K) keep the eager path
  unchanged. gemma-4's dual-branch routed experts use the same device
  routing (per-expert down scales uploaded per layer). Certified on this
  box (RTX 3070, partial offload): CPU==GPU greedy byte-identical over
  128 tokens on Qwen3-30B-A3B-Q4_K_M and gemma-4-26B-A4B-it-Q4_0;
  Qwen3-30B greedy text unchanged vs the pinned b10076 comparison;
  `make test` + `test-moe` green; bench.sh md5s unchanged.

## v0.1.4-alpha — 2026-07-29

### Headline: tensor-core prefill by default, published benchmarks, and the European roster

The tensor-core prefill GEMM is now the **default** on seven
tolerance-gated dense (Q4_K, arch) combos (+47–77% prefill, decode
unchanged), backed by a new teacher-forced tolerance gate
(`make test-tc-tol`); the decode GEMV bandwidth pass lifts dense decode to
73–79% of llama.cpp on the reference box, and the first head-to-head
benchmark is published (`docs/benchmarks.md`) with the losing rows
included. Six European models join the SHA-pinned compatibility manifest
under the new Europe & US model-scope policy (`docs/model-scope.md`) —
whose evidence runs found and fixed three real defects (a silent MoE
GPU→CPU fallback, a fast-math expf-overflow UB in CPU silu, and two
option footguns) and reported a GGUF conversion bug upstream to
OpenLLM-France. Full detail below.

- Fixed CPU decode corruption on models with extreme FFN gate values
  (found by TildeOpen-30b's certification run): silu computes expf(-g),
  fp32 expf overflows past ~88, and the -ffast-math build treats that
  overflow as UB — the auto-vectorized libmvec expf returned garbage for
  TildeOpen's last-layer gates (|g| up to ~2.7e3), degrading every
  pure-CPU decode step into <unk> emissions while GPU (CUDA expf
  saturates properly) was unaffected. gated_act now short-circuits silu
  to 0 below g = -80, where |silu| < 1.5e-33, so expf never sees an
  overflowing argument. Verified: TildeOpen CPU==GPU byte-identical over
  128 greedy tokens, greedy_reference 4/5 vs pinned b10076 (was 1/5),
  and lucie/eurollm CPU outputs bit-identical pre/post fix (the guard is
  a no-op for in-range models). RUNNER_DEBUG_ACT=N now dumps the N-th
  forward pass (was: first only) — the instrument this debug needed.
- `--gpu-layers 0` now means what it says — no GPU (same as `--gpu off`).
  It used to be the auto-fit sentinel and silently ran FULL GPU, a
  documented footgun that bit its own certification run: the roster's
  first cpu_cuda evidence used it as the CPU side and compared GPU with
  GPU. Re-verified with a true `--gpu off` CPU side: all five EU models
  and Qwen3-30B-A3B are byte-identical CPU vs GPU over 128 greedy tokens.
  Omit the flag for auto-fit.
- TildeOpen-30b added to the compatibility manifest (SHA-pinned): loads
  and generates coherently on GPU (llama arch, 60 layers, full 19.4 GB
  offload) — and its evidence run exposed an OPEN ENGINE DEFECT: the
  pure-CPU path emits <unk> for content tokens from the second generated
  token onward (batched CPU prefill is sane per the activation dump; the
  failure is single-token CPU decode, deterministic, independent of -b/-t).
  TildeOpen is the first model to trip it; its unique geometry — GQA 48:8
  (ratio 6), vocab 131072, n_embd 6144 — is the suspect surface. cpu_cuda
  and greedy_reference are recorded FAILED for TildeOpen until the defect
  is fixed; GPU serving is unaffected.
- Certified the European roster into the compatibility program: EuroLLM-9B,
  Lucie-7B, Mistral-Nemo-12B, Teuken-7B and salamandra-7b are SHA-pinned in
  `tests/compatibility/models.json` with a recorded evidence run — all five
  pass load, cpu_cuda (128-token greedy byte-identical, scalar path), chat
  and tool, plus the 8-token greedy_reference sweep against pinned llama.cpp
  b10076 (salamandra 5/5 exact; the others show the same divergence class as
  the long-certified models). Honest gaps recorded rather than skipped:
  Lucie's tokenizer FAILS the 721-string differential (259 divergences) —
  root-caused to the GGUF, not the engine: the conversion exports Lucie's
  BPE tokenizer as SentencePiece with all 65,024 merge ranks flattened to
  -1000, so the reference tokenization is unreproducible from the file by
  any engine (Runner and llama.cpp b10076 are token-identical on it, and
  the vendor's own official GGUF shares the defect — an upstream
  conversion bug affecting every GGUF consumer of Lucie);
  EuroLLM's reference repo is gated and Teuken's has no tokenizer.json, so
  their tokenizer checks are not_executed; long_context was not run for the
  roster. `reference_compare.py` fixed en route: Runner rejects unknown
  model names now, so the harness asks each server for its served model id.
- Vendor sampling presets for four European families (each cites its
  source): `mistral-nemo` — Mistral's card is explicit that Nemo "requires
  smaller temperatures. We recommend to use a temperature of 0.3", so the
  0.7 `mistral` preset was actively wrong for it; `lucie` (temp 0.6 /
  top_p 0.9, generation_config.json) and `salamandra` (temp 0.6 /
  repetition_penalty 1.2, generation_config.json); `teuken` (temp 0.7 /
  top_p 0.95, model card usage example — the weakest citation grade, and
  marked as such). EuroLLM and TildeOpen publish nothing verifiable and
  deliberately stay on `generic`. Name matching requires BOTH "mistral"
  and "nemo" so NVIDIA's Nemotron cannot land on Mistral's temperature.
- Preset matching now runs over `general.name` PLUS the load path's
  basename (`sampler_ident`): quantizer metadata is unreliable — a real
  community salamandra GGUF ships `general.name` "snapshots" (the
  converter's HF cache directory) — and the filename still carries the
  family. All three resolution sites use the combined identity.
- **Promoted the tensor-core prefill GEMM to the default for gated dense
  (Q4_K, arch) combos** (owner decision on the tolerance-gate numbers):
  `llama`, `phi3`, `gemma4`, `qwen3`, `mistral`, `gemma3`, `smollm` — every
  row measured by `test_tc_tol` on real weights with 0/64 teacher-forced
  top-1 flips and ≤0.012% mean logit deviation. Measured effect on the
  Blackwell MIG: dense Q4_K prefill +47–77% with decode unchanged
  (llama-3.2-3b 263→438 tok/s, qwen3-4b 212→352). `qwen3moe` passed its
  gate too (0.216%, one near-tie) but stays opt-in: MoE routing amplifies
  fp16 noise ~86× over dense, and this promotion covers the dense family.
  Unmeasured archs (`qwen2`, `qwen35`, `stablelm`) remain scalar.
  `RUNNER_CUDA_TC=1` still forces the path on everywhere (how a gate
  candidate is measured), `=0`/`off` forces it off; unset now means "per
  the promotion table" instead of "off". `test_kv_tol` pins the GEMM path
  scalar (`gpu_tc_force(0)`) so its strict f16 CPU==GPU invariant keeps
  measuring the KV cache format, not the GEMM. NOTE for certification:
  on promoted combos, byte-identical CPU==GPU comparisons now compare TC
  against scalar — certify the scalar path with `RUNNER_CUDA_TC=0` or use
  the tolerance form (`test_tc_tol`); free-running greedy output was
  byte-identical TC-vs-scalar on all four models checked (512+128), but
  near-tie flips are possible in principle on other prompts.
- Added the TC tolerance gate (`tests/test_tc_tol.c`, `make test-tc-tol`),
  the promotion instrument the tensor-core plan required: teacher-forced
  logits over 64 positions, gated on top-1 agreement (near-tie escape as in
  the q8-KV gate) and a bounded mean logit deviation (≤0.5% of the mean
  logit range, computed over real logits — suppression sentinels excluded).
  Skips rather than passes when the TC kernel cannot engage (no GPU, no
  Q4_K tensor, or bit-identical logits meaning the kernel never launched).
  First measurements on the Blackwell MIG: llama 0.003%, phi3 0.004%,
  gemma4 0.012% of range with 0/64 flips; qwen3moe 0.216% with 1/64 — a
  near-tie at 0.001 of range. All four pass; the qwen3moe free-running
  divergence is thereby classified as near-tie amplification (the q8-KV
  class), not decisive error. Adds `gpu_tc_force()` so one process can
  compare both paths; `RUNNER_CUDA_TC` env behavior is unchanged.
- Fixed a silent MoE GPU→CPU fallback introduced by the `--cpu-moe` binding
  layer (active even without the flag): `binding_find` bounds-checks
  `t->nbytes` on every dispatch, but the per-expert slice descriptors built
  by `moe_expert_weight` and the gemma-4 fused `gate_up` slice kept the full
  multi-expert tensor size, so expert `e >= 1` failed the check, `enc_mv`
  returned false, and the whole forward silently ran on the CPU while the
  load banner still reported a full GPU split. Clamping the slice `nbytes`
  restores the GPU path. Measured on the Blackwell MIG 1g.24gb:
  Qwen3-30B-A3B-Q4_K_M decode 4.5 → 76.3 tok/s (above the 56.5 pre-regression
  rate — expert matvecs now also use the new decode GEMVs);
  gemma-4-26B-A4B-Q4_0 6.9 → 23.5 tok/s (restored). Greedy output verified
  token-identical between CPU and GPU on both models. Note the CPU/GPU
  identity gates could not catch this class of defect: the fallback *is* the
  CPU path, so outputs matched while decode ran up to 12× slow.
- CUDA decode matvec pass (the suite plan's P1 decode lever): the Q4_K and
  Q5_K decode GEMVs now use aligned 8-byte quant loads, `float4` activation
  loads and a factored per-group affine; Q8_0 covers four blocks per warp
  iteration; Q6_K unrolls two blocks for load-level parallelism. Measured on
  an RTX 3070: Qwen3-8B-Q4_K_M decode 31.7 → 53.3 tok/s (+68%),
  Llama-3.1-8B-Q5_K_M 31.0 → 54.0 tok/s (+74%), Qwen3-4B-Q8_0 58.4 → 60.5
  tok/s. GPU output remains token-identical to the CPU path on all verified
  models.
- Prefill matvec tiles widened from 8 to 16 tokens (MVB 16), halving the
  per-tile weight passes. The Q8_0 prefill GEMM keeps its proven 8-column
  tile and runs wide tiles as two launches. Known cost on the 3070:
  Qwen3-4B-Q8_0 prefill ~-4% (113.5 → 108.7 tok/s) from extra attention-score
  L2 pressure at 16 columns; Q4_K prefill is unchanged on the default path.
- Rebuilt the opt-in tensor-core prefill GEMM (`RUNNER_CUDA_TC=1`) as an
  MMQ-style kernel: the block dequantizes a 64-row × 128-K fp16 weight tile
  to shared memory once — 8-byte quant loads, two threads per row — and four
  warps' m16n16k16 MMAs reuse it against a 16-token fp16 activation tile with
  fp32 register accumulation. The previous per-warp variant measured 6-7×
  slower than the scalar GEMM; this one measures Qwen3-8B-Q4_K_M prefill
  96.4 → 138.2 tok/s (+43%) on the RTX 3070, and its greedy output matched
  the CPU path token-for-token on the verification prompts. It remains
  opt-in behind the tolerance-gate promotion decision.
- Added sparse MoE tensor-role placement with `--cpu-moe`. CUDA retains
  attention/dense tensors and KV while expert FFNs execute from system RAM;
  packed uploads omit the expert bank instead of reserving GGUF-sized holes.
- `--caps` now advertises `tensor_placement.cpu_moe` for schedulers.
- Current Qwen3.5 GGUFs that include declared NextN/MTP blocks in
  `block_count` now load only the autoregressive backbone.
- Added native Qwen3.5/Ornith CUDA execution for recurrent Gated DeltaNet and
  full-attention blocks, including causal convolution/state kernels, gated
  attention, partial offload, pre-forward state snapshots for correct runtime
  CPU fallback, and compatibility with both `ssm_dt` tensor spellings.

## v0.1.3-alpha — 2026-07-24

### Headline: sparse Mixture-of-Experts (MoE) support

The runner now runs real sparse **mixture-of-experts** models — the class the
field is converging on for modest-VRAM hardware — on CPU, fully on the GPU, and
with **partial CPU offload for cards smaller than the model**.

- **Architectures:** Mixtral-style `llama`-with-experts and `qwen3moe`
  (Qwen3-MoE). Both the modern fused 3D expert tensors and the legacy
  split-per-expert layout (older Mixtral GGUFs) are supported by one accessor —
  no forward-path branch on layout.
- **Qwen3-30B-A3B (Q4_K_M, 128 experts, top-8)** loads in **18.6 GB**, fits a
  **24 GB MIG slice on an NVIDIA RTX PRO 6000 Blackwell** with ~6 GB free, and
  decodes at **~55 tok/s on that hardware**. This is not presented as a
  representative result for every 24 GB consumer GPU. Greedy GPU output is
  **token-identical to Runner's CPU path on the same quantized GGUF**.
- **Partial CPU offload (8–16 GB cards):** the runner fits as many leading
  layers on the GPU as the VRAM budget allows and runs the rest on the CPU.
  Every configuration tested is token-identical to Runner's CPU path on the
  same quantized GGUF, or to the full-GPU quantized run where the model fits.
- **Q3_K GPU kernel (new):** Q3_K MoE now runs on the GPU. **Mixtral-8x7B
  Q3_K_M (20.4 GB) is fully GPU-resident on the Blackwell 24 GB MIG slice**,
  with GPU output token-identical to CPU.
- **Prefill throughput:** MoE prefill groups tokens by shared expert (batched
  per-expert matmul instead of one token at a time), ~5.6× the per-token CPU
  prefill rate. Decode is unchanged and bit-identical.
- **MXFP4 read support (gpt-oss format):** the OCP microscaling FP4 quant type
  (E8M0 × E2M1) is read and dequantized; validated against the real
  `gpt-oss-20b-MXFP4.gguf` (all 72 expert tensors read; a real row dequantizes
  to spec). *(gpt-oss as a whole needs architecture support to actually run;
  the MXFP4 tensors read correctly today.)*
- **Runnable == validated:** the loader refuses at load — rather than
  miscompute — shared-expert MoE (Qwen2-MoE / DeepSeek) and non-gemma GELU
  sparse MoE until each is validated on its own. Gemma-4's GELU dual-branch MoE
  is implemented and validated separately.

Correctness is checked with synthetic MoE configurations constructed to equal a
dense FFN (asserted token-identical in CI) and CPU/GPU agreement on real
quantized models. CPU/GPU agreement is an internal consistency check; independent
Runner-vs-llama.cpp comparison is handled by
`scripts/compare_llamacpp.py` when the same GGUF, hardware, and llama.cpp build
are available. See [`docs/moe-support.md`](docs/moe-support.md).

### Reliability & security hardening (July 2026 code review, RNR-###)

The release gate from the July code review is cleared, with the remaining
hardware-only Metal validation documented separately:

- Metal runtime fallback preserves the backend resource owner after
  `gpu_disable()`, so CPU fallback can keep using unified-memory KV buffers and
  `model_free()` never frees `MTLBuffer.contents` (RNR-001).
- Quantizer honors `general.alignment` and writes atomically (RNR-002/015).
- Load/scheduler lifecycle, an OOM tranche, and VRAM rollback; the
  OOM-as-truncated-prompt semantic bug is fixed (RNR-003/005/006/013).
- Architecture admission allowlist; unknown archs are experimental behind
  `RUNNER_ALLOW_UNKNOWN_ARCH=1` (RNR-004).
- GGUF typed getters validate type/sign/range/finiteness (RNR-010); one strict
  numeric parser for CLI + env (RNR-021); bounded CLI file reads (RNR-011).
- `load_cancel` is a C atomic (RNR-008); startup lease compares process
  start-time, not just PID (RNR-017); a drift gate guards the committed
  generated GPU headers (RNR-020).
- Python client: streamed `tool_calls` assembly + preserved `finish_reason`
  (RNR-016).
- `make test-moe` runs the synthetic MoE correctness suite in Linux/macOS CI;
  release packaging now checks tag, binary and Python versions, current release
  docs, changelog, and the generated `BUILD-INFO.txt` tag/commit before creating
  archives.
- CUDA compatibility is documented as NVIDIA Turing / compute capability 7.5 or
  newer, matching the embedded `sm_75` PTX target; older NVIDIA GPUs fall back
  to CPU.
- Windows `make test` now builds the prefix-cache and VRAM-registry tests as
  distinct `.exe` targets. Native file IDs and 100 ns last-write timestamps
  prevent an in-place GGUF edit from reusing a stale prefix within the same
  second; the VRAM and output tests are portable across Windows/POSIX.
- GPU header embedding is explicitly UTF-8/LF, so the generated-header drift
  gate is deterministic across Windows, Linux, and macOS.

### Agent conformance

- New agent-torture family, **schema-constrained selection from a large enum**
  (~50 labels) — the structured-labeling task small models fail by emitting a
  plausible near-miss; schema-constrained decoding forces an exact member.

### Notes

- `--gpu-layers N` forces N leading layers on the GPU; `--reserve-vram PCT`
  caps usage. Runner still binds loopback-only by default.

## v0.1.2-alpha — 2026-07-22

- Compatibility evidence: real OpenAI/Anthropic SDKs, LiteLLM, LangChain, and a
  llama.cpp reference matrix. Earlier phases: strict tool-call schema engine,
  streaming agent events, Responses + Messages APIs, shared weights, continuous
  batching, prefix caching, q8 KV cache.

## v0.1.1-alpha — 2026-07-19
## v0.1.0-alpha — 2026-07-17

- Initial public alpha: dependency-free C inference server for GGUF models
  (CPU/CUDA/Metal), OpenAI-compatible HTTP API, sampler-level JSON-schema
  enforcement.
