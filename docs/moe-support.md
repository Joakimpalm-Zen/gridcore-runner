# Sparse-MoE support — implementation and test report

Date: 2026-07-24
Runner: commit `c610550`
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

- **Architectures:** `llama` with experts (Mixtral) and `qwen3moe`
  (Qwen3-MoE = qwen3 attention — qk-norm, GQA, NeoX rope — with an MoE FFN).
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

### Deliberately refused (no silent wrong output)

To keep runnable == validated, the loader refuses at load rather than
miscompute:

- **shared-expert MoE** (Qwen2-MoE / DeepSeek — `expert_shared_count > 0` or a
  `ffn_gate_inp_shexp` tensor): the shared expert would be silently ignored.
- **GELU-gated MoE** (gemma family): the expert FFN path is SiLU-only.
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
  **partial CPU offload** (see below); it is the GPU-kernel-supported quant.
  Q3_K_M fits VRAM but Q3_K has no GPU kernel, so it runs on CPU (~4 tok/s).

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

## Known limitations / future work

- **Prefill throughput** (~79 tok/s) is lower than a dense model of comparable
  active size: the MoE FFN processes prefill tokens one at a time per expert
  (routing differs per token). Grouping tokens by shared expert would speed
  prefill; decode (the interactive path) is unaffected.
- **Q3_K GPU kernel** absent → Q3_K MoE runs on CPU. Adding it would let
  larger-but-lower-bit MoE models use the GPU.
- **MXFP4** (gpt-oss family) not read.
- **Shared-expert / GELU MoE** refused (see above) — enabling them needs the
  shared-expert path and a GELU expert FFN, each behind its own validation.
- **Advisor / runner-control**: fit-by-active-params and catalog entries for
  MoE are follow-ups; the sniffer already admits top-k MoE.
