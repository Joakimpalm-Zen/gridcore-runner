# Sparse-MoE support — implementation and test report

Date: 2026-07-24
Runner: commit `c610550`
Hardware: NVIDIA RTX PRO 6000 Blackwell, **24 GB MIG slice** (`MIG 1g.24gb`),
CPU fallback on the same host (64 threads).

## Summary

The runner runs real sparse **mixture-of-experts** models — the class the field
converges on for modest-VRAM hardware — both on CPU and on the GPU, within a
24 GB budget. The headline result: **Qwen3-30B-A3B (Q4_K_M, 128 experts,
top-8) loads in 18.6 GB, fits a 24 GB card with 6 GB free, produces correct
output, and generates at ~55 tok/s on the GPU**, token-identical to the CPU
reference.

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

### Mixtral-8x7B-Instruct — Q3_K_M, `llama`, split layout, CPU

- Source: `TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF` →
  `mixtral-8x7b-instruct-v0.1.Q3_K_M.gguf` (20.36 GB).
- Geometry: 32 layers, 8 experts, top-2, **legacy split per-expert tensors**.
- **Correctness:** loads and produces correct output, e.g.
  `The capital of France is` → ` a city that is known for its beauty, culture,
  and history. Paris is …`.
- **Why CPU, not GPU:** this GGUF is Q3_K, and the runner's GPU kernels cover
  Q8_0 / Q4_K, not Q3_K → it runs on CPU (~4 tok/s). A Q4_K Mixtral would use
  the GPU kernels but is ~26 GB, over the 24 GB budget; Q3_K fits VRAM but has
  no GPU kernel. **Mixtral-8x7B therefore does not fit a 24 GB card at a
  GPU-supported quant** — Qwen3-30B-A3B (Q4_K, 18 GB) is the ≤24 GB GPU MoE.
  This validates the split-layout loader and the CPU MoE forward on a real
  model.

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
