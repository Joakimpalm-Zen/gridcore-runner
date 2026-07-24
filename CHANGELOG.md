# Changelog

All notable changes to gridcore-runner. This project is in **alpha**; the HTTP
protocol and CLI may still change between alpha releases.

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
  24 GB card with ~6 GB free, and decodes at **~55 tok/s on the GPU** — and its
  greedy output is **token-identical to the CPU reference**.
- **Partial CPU offload (8–16 GB cards):** the runner fits as many leading
  layers on the GPU as the VRAM budget allows and runs the rest on the CPU.
  Every configuration tested is token-identical to the full-precision reference.
- **Q3_K GPU kernel (new):** Q3_K MoE now runs on the GPU. **Mixtral-8x7B
  Q3_K_M (20.4 GB) is fully GPU-resident on a 24 GB card**, GPU output
  token-identical to CPU — it was CPU-only before.
- **Prefill throughput:** MoE prefill groups tokens by shared expert (batched
  per-expert matmul instead of one token at a time), ~5.6× the per-token CPU
  prefill rate. Decode is unchanged and bit-identical.
- **MXFP4 read support (gpt-oss format):** the OCP microscaling FP4 quant type
  (E8M0 × E2M1) is read and dequantized; validated against the real
  `gpt-oss-20b-MXFP4.gguf` (all 72 expert tensors read; a real row dequantizes
  to spec). *(gpt-oss as a whole needs architecture support to actually run;
  the MXFP4 tensors read correctly today.)*
- **Runnable == validated:** the loader refuses at load — rather than
  miscompute — shared-expert MoE (Qwen2-MoE / DeepSeek) and GELU-gated (gemma)
  MoE, until each is validated on its own.

Correctness is checked two ways with no external reference engine: synthetic MoE
configurations constructed to equal a dense FFN (asserted token-identical in
CI), and CPU/GPU agreement on the real Qwen3-30B-A3B. See
[`docs/moe-support.md`](docs/moe-support.md).

### Reliability & security hardening (July 2026 code review, RNR-###)

The release gate from the July code review is cleared (Mac/Windows platform
items aside):

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
