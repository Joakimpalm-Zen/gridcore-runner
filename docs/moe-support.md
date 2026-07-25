# Sparse-MoE support — implementation and test report

Date: 2026-07-24
Runner: v0.1.3-alpha (core support plus the follow-ups at the end of this doc)
Hardware: NVIDIA RTX PRO 6000 Blackwell, **24 GB MIG slice** (`MIG 1g.24gb`),
CPU fallback on the same host (64 threads).

## Summary

The runner runs real sparse **mixture-of-experts** models — the class the field
converges on for modest-VRAM hardware — on CPU, fully on the GPU, and with
**partial CPU offload for cards smaller than the model** (8–16 GB). Headline
results: **Qwen3-30B-A3B (Q4_K_M, 128 experts, top-8) loads in 18.6 GB, fits a
24 GB card with 6 GB free, and generates at ~55 tok/s on the GPU**, token-
identical to the CPU reference; on simulated 8/12/16 GB cards it partially
offloads (19/29/39 of 48 layers on GPU) with identical output. Both supported
MoE families (qwen3moe fused, Mixtral/llama split) and both expert layouts are
covered.

## What is supported

The Mixtral / Qwen3-MoE convention: a per-token router (softmax over **all**
experts), top-k selection, weights renormalized to sum to 1, per-expert SwiGLU,
weighted sum. Concretely:

- **Architectures:** `llama` with experts (Mixtral), `qwen3moe`
  (Qwen3-MoE = qwen3 attention — qk-norm, GQA, NeoX rope — with an MoE FFN), and
  `gemma4-moe` (gemma-4's **GELU dual-branch MoE**, described below).
- **Expert tensor layouts:** both the modern **fused 3D** tensors
  (`ffn_gate_exps` / `ffn_up_exps` / `ffn_down_exps`) and the **legacy split
  per-expert 2D** tensors (`ffn_gate.{e}.weight`, older Mixtral GGUFs). One
  shared `moe_expert_weight()` accessor serves both; no forward code branches
  on the layout.
- **Execution:** CPU and GPU. On the GPU the whole model file is uploaded as
  one buffer and each expert's slice offset is used directly by the matvec
  kernel; routing runs on the host from router logits read back per token, and
  the expert SwiGLU matmuls run on the GPU. MoE layers use the eager path
  (host-dependent routing cannot be CUDA-graph-captured).

### gemma-4 GELU dual-branch MoE (`gemma4-moe`)

gemma-4's MoE is not a plain Mixtral-style sparse FFN — **every MoE layer runs
two branches and sums them**:

- a **dense shared GELU FFN** (its own `ffn_gate`/`ffn_up`/`ffn_down` +
  `post_ffw_norm_1`), always active, plus
- a **routed expert set** with a **fused `ffn_gate_up_exps`** tensor
  (`{n_embd, 2·n_ff_exp, n_expert}` — gate and up concatenated per expert),
  **per-expert `ffn_down_exps.scale`**, and a `pre_ffw_norm_2` / `post_ffw_norm_2`
  sandwich. The router runs on a **separate** weightless RMSNorm of the
  attention residual scaled by `gate_inp_scale · 1/√n_embd`, then the usual
  softmax-over-all → top-k → renormalize.

Both branches read the un-normed post-attention residual directly (they do their
own norms), and the summed result feeds the outer `post_ffn_norm` + residual.
The activation is the tanh-GELU approximation, shared with the dense gemma FFN
via one `gated_act()` so the GELU path cannot silently diverge from SiLU MoE.
**Execution: CPU and CUDA** — the GPU kernel (`gpu_gemma_moe_ffn`) mirrors the
CPU forward token-for-token (dense written straight into the layer output;
`selw · down_scale` folded into one pre-down `enc_scale`; the router's
`1/√n_embd` folded into the uploaded `gate_inp_scale`), verified token-identical
to llama.cpp b10076 and GPU/CPU-identical on **gemma-4-26B-A4B-it** (128 experts,
top-8; ~23 tok/s full-offload in the 24 GB slice). Like the other MoE families
it uses the eager path (router readback).

### Deliberately refused (no silent wrong output)

To keep runnable == validated, the loader refuses at load rather than
miscompute:

- **shared-expert MoE** (Qwen2-MoE / DeepSeek — `expert_shared_count > 0` or a
  `ffn_gate_inp_shexp` tensor): the shared expert would be silently ignored.
  (gemma-4's dense shared branch above is a *different* mechanism — a full
  always-on FFN, not an `expert_shared_count` shared expert — and is handled.)
- **GELU-gated sparse MoE outside gemma-4** — a non-gemma arch presenting GELU
  experts stays behind the architecture allowlist; only gemma-4's validated
  dual-branch layout is admitted.
- **Other MoE architectures** (`qwen2moe` etc.) stay behind the architecture
  admission allowlist until validated.

## Test results

### Qwen3-30B-A3B — Q4_K_M, `qwen3moe`, fused layout, GPU

- Source: `Qwen/Qwen3-30B-A3B-GGUF` → `Qwen3-30B-A3B-Q4_K_M.gguf` (18.56 GB).
- Geometry: 48 layers, n_embd 2048, 128 experts, top-8, expert FFN 768,
  head_dim 128 (decoupled), GQA 32/4.
- **VRAM:** 18.6 GB weights + 0.40 GB KV (ctx 4096) → **6.16 GB free of
  25.37 GB**. Comfortable on a 24 GB card.
- **Correctness:** greedy (`--temp 0`) GPU output is **token-identical to the
  CPU reference** (validated with a shared prompt). Example completions:
  - `The capital of France is` → ` Paris. The capital of Italy is Rome. The
    capital of Spain is Madrid.`
  - `def is_prime(n):` → a correct implementation (`if n < 2: return False`,
    `for i in range(2, int(n**0.5)+1): if n % i == 0: return ...`).
  - `If a train travels 60 km in 45 minutes, what is its speed in km/h? A:` →
    `80 km/h` (correct).
  - `Huvudstaden i Sverige är` → ` Stockholm. Det är också en av de största …`
    (correct, grammatical Swedish).
- **Performance (GPU, temp 0):**

  | Phase | Tokens | Throughput |
  |---|---|---|
  | Prefill | 257 | **78.7 tok/s** |
  | Decode | 128 | **55.3 tok/s** |

  Decode is the interactive number; it is stable across runs (55–56 tok/s).

### Mixtral-8x7B-Instruct — `llama`, split layout

- Source: `TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF` (Q4_K_M 26.44 GB, and
  Q3_K_M 20.36 GB).
- Geometry: 32 layers, 8 experts, top-2, **legacy split per-expert tensors**.
- **Correctness:** correct output, e.g. `The three primary colors are` →
  ` red, yellow, and blue. These colors are considered primary because they …`.
- Q4_K_M (26 GB) exceeds a 24 GB card, so on this hardware it always runs with
  **partial CPU offload** (see below). Q3_K_M (20.4 GB) fits VRAM; once the Q3_K
  GPU kernel landed (see Follow-ups, below) it runs **fully on the GPU**,
  token-identical to CPU — it was CPU-only (~4 tok/s) when this section was
  first written.

## Partial CPU offload (8–16 GB cards)

MoE models larger than the card run with the leading layers on the GPU and the
rest on the CPU. This needed a fix: the gpu-split accounted only the dense FFN
tensors, which are NULL on a MoE layer, so it undercounted each MoE layer by
its experts (~all of its weight) and never offloaded. The split now accounts
the full per-layer weight (attention + router + every expert, fused or split).

VRAM budgets were simulated on the 24 GB slice with `--reserve-vram PCT` (caps
usage to PCT% of total). **Every partial-offload configuration is token-
identical to the full-precision reference** (full-GPU for Qwen3; CPU for the
26 GB Mixtral that cannot fully fit), so offload is transparent to output.

| Model | ~8 GB | ~12 GB | ~16 GB | full |
|---|---|---|---|---|
| **Qwen3-30B-A3B** Q4_K (fused, 48 layers) | 19/48 layers, 6.7 tok/s | 29/48, 9.6 | 39/48, 16.4 | 48/48, 55.3 |
| **Mixtral-8x7B** Q4_K (split, 32 layers) | 9/32 layers, 11.6 tok/s | 13/32, 12.7 | 18/32, 14.4 | — (26 GB, never full on 24 GB) |

Decode throughput scales with the fraction of layers on the GPU. Both MoE
families (qwen3moe fused, llama split) and both expert layouts are covered.
Nothing special is required to use it — the runner fits as many leading layers
as the available (or `--reserve`-capped) VRAM allows and runs the rest on CPU.

### Synthetic equivalence tests (`tests/test_moe.py`, in CI)

Reference-free correctness: `make-test-moe.py` emits a dense model plus MoE
variants each **mathematically identical** to the dense FFN, so the runner's
already-trusted dense path is the oracle (no separate reference engine):

- `moe1` — fused, top-1, one expert zeroed → identical to dense.
- `moe2` — fused, top-2 with a zero router (0.5/0.5) → identical to dense.
- `moe3` — **split** layout, top-1 → identical to dense.

All assert byte-identical greedy output; the FFN is scaled so a broken MoE
produces different tokens (verified during development).

## Methodology

Two independent correctness checks, neither needing an external reference
engine:

1. **Dense-oracle equivalence** (synthetic, CI): MoE configurations constructed
   to equal a dense FFN, asserted token-identical.
2. **CPU/GPU agreement on real models**: the CPU forward is the runner's
   long-validated path; the GPU MoE output is asserted to match it token-for-
   token on the real Qwen3-30B-A3B.

## Follow-ups completed (2026-07-24)

- **Prefill throughput — DONE.** MoE prefill now groups tokens by shared
  expert: route the whole batch, then run each expert once over all its routed
  tokens as a batched matmul (weight rows dequantize once and stream across
  every token). Decode is untouched and bit-identical; prefill stays token-
  identical (F32 dense-oracle byte-identical, and real Qwen3-30B CPU==GPU
  preserved). Measured CPU prefill 21.6 tok/s vs 3.8 tok/s decode on a 128-token
  prompt (~5.6x the per-token rate).
- **Q3_K GPU kernel — DONE.** `k_mv_q3_K` / `k_mv_q3_K_b` (warp-per-row, dequant
  fused into the dot). **Mixtral-8x7B-Instruct Q3_K_M (20.4 GB) now loads fully
  on a 24 GB card and runs on the GPU**, token-identical to the CPU reference —
  previously it refused GPU offload and ran CPU-only.
- **MXFP4 read — DONE.** OCP microscaling FP4 (GGML type 39) — the gpt-oss
  expert-tensor format — is read and dequantized (E8M0 block scale × E2M1 code),
  admitted at load and usable through the CPU forward. Unit-tested against the
  OCP spec, and **validated on the real `gpt-oss-20b-MXFP4.gguf`**: the loader
  identifies all 72 MXFP4 expert tensors and a real `ffn_down_exps` row
  dequantizes to finite, sane weights (2527/2880 nonzero, ±0.09375 = 6·2⁻⁶,
  mean ≈ 0). No MXFP4 **GPU** kernel yet, and gpt-oss's *architecture* (sliding-
  window attention, attention sinks, its own MoE gating and tensor layout) is a
  separate task — so gpt-oss does not yet *run*, but its MXFP4 tensors read
  correctly.
- **Advisor / runner-control — DONE.** The advisor scores MoE throughput by
  *active* params (a MoE decodes at the speed of its active experts, not its
  full resident weight) while VRAM fit stays by total size; it surfaces expert
  residency, and the catalog gained Qwen3-30B-A3B and Mixtral-8x7B entries.

## Known limitations / future work

- **gpt-oss architecture** unsupported (distinct from MXFP4 read): the runner
  refuses arch `gpt-oss`. Making gpt-oss actually run needs its attention
  (sliding-window + sinks), MoE gating, and tensor-layout support — plus an
  MXFP4 **GPU** kernel for VRAM speed. `gpt-oss-20b-MXFP4.gguf` is now on disk
  for that work.
- **GELU dual-branch MoE** (gemma-4) is now **implemented** (CPU + CUDA) — see
  the `gemma4-moe` section above. `expert_shared_count`-style shared-expert MoE
  (Qwen2-MoE / DeepSeek) remains refused, behind its own validation.
- **MoE GPU decode** still forces the eager path (host-side routing readback per
  token), so MoE GPU throughput trails a dense model of the same active size.
