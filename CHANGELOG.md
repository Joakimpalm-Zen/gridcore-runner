# Changelog

All notable changes to gridcore-runner. This project is in **alpha**; the HTTP
protocol and CLI may still change between alpha releases.

## Unreleased

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
