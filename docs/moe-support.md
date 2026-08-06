# Sparse-MoE support — implementation and test report

Date: 2026-07-24
Runner: v0.1.11-alpha (core support plus the follow-ups at the end of this doc)
Hardware: NVIDIA RTX PRO 6000 Blackwell, **24 GB MIG slice** (`MIG 1g.24gb`),
CPU fallback on the same host (64 threads).

## Summary

The runner runs real sparse **mixture-of-experts** models — the class the field
converges on for modest-VRAM hardware — on CPU, fully on the GPU, and with
**partial CPU offload for cards smaller than the model** (8–16 GB). Headline
results: **Qwen3-30B-A3B (Q4_K_M, 128 experts, top-8) loads in 18.6 GB, fits a
24 GB MIG slice with 6 GB free, and generates at ~55 tok/s on an NVIDIA RTX PRO
6000 Blackwell**, token-identical to Runner's CPU path on the same quantized
GGUF. That number is a hardware-specific measurement, not a representative
claim for every 24 GB consumer GPU. On simulated 8/12/16 GB budgets it
partially offloads (19/29/39 of 48 layers on GPU) with identical output. Both
supported MoE families are covered on CPU and CUDA. Metal now has gated decode
coverage for fused plain MoE, gpt-oss MXFP4/bias-bearing experts, and Gemma-4's
dual-branch MoE; split expert layout and shared-expert MoE remain refused there.

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
- **Execution:** CPU, CUDA, and Metal for the fused layouts. CUDA also covers
  legacy split experts and partial expert placement; Metal currently covers the
  fused device-routing path for plain/gpt-oss/Gemma-4 MoE and refuses split or
  shared-expert layouts rather than silently falling back to wrong math.

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
**Execution: CPU, CUDA, and Metal.** CUDA is verified GPU/CPU-identical on
**gemma-4-26B-A4B-it** (128 experts, top-8; ~23 tok/s full-offload in the
24 GB slice). Metal is gated on the Mac synthetic `gemma4-moe` fixture
(`make test-metal-gemma4-moe`), covering scaled embeddings, GELU, fused
`gate_up` experts, branch norms, down scales, post norms, layer scale, V
RMSNorm and final logit softcap; large real-model Metal validation is still
hardware-capacity dependent.

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
  25.37 GB** in the MIG slice.
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
- Q4_K_M (26 GB) exceeds this 24 GB MIG slice, so on this hardware it always
  runs with **partial CPU offload** (see below). Q3_K_M (20.4 GB) fits VRAM; once the Q3_K
  GPU kernel landed (see Follow-ups, below) it runs **fully on the GPU**,
  token-identical to CPU.

## Expert-tensor CPU placement (8 GB cards)

`--cpu-moe` keeps attention, norms, output and non-MoE dense tensors on CUDA
while the complete sparse expert FFN executes directly from the mmap in system
RAM. Only the post-attention activation tile crosses the PCIe boundary per
layer. The CUDA upload is packed by tensor role, so skipped expert tensors do
not consume address-space-sized holes in VRAM. `runner --caps` advertises this
as `tensor_placement.cpu_moe` for controllers and advisors.

This mode covers fused Qwen3-MoE and legacy split Mixtral expert layouts. Its
synthetic top-1/top-2 correctness gate is byte-identical to the dense CPU
oracle. If even the retained attention/KV/output footprint does not fit, the
existing leading-layer split still applies to that smaller device set.

```sh
./runner -m Qwen3-Coder-30B-A3B-Q4_K_XL.gguf --cpu-moe -p "hello"
```

## Whole-layer partial CPU offload (8–16 GB cards)

MoE models larger than the card run with the leading layers on the GPU and the
rest on the CPU. This needed a fix: the gpu-split accounted only the dense FFN
tensors, which are NULL on a MoE layer, so it undercounted each MoE layer by
its experts (~all of its weight) and never offloaded. The split now accounts
the full per-layer weight (attention + router + every expert, fused or split).

VRAM budgets were simulated on the 24 GB slice with `--reserve-vram PCT` (caps
usage to PCT% of total). **Every partial-offload configuration is token-
identical to Runner's CPU path on the same quantized GGUF**, or to the
full-GPU quantized run where the model fits, so offload is transparent to
output.

| Model | ~8 GB | ~12 GB | ~16 GB | full |
|---|---|---|---|---|
| **Qwen3-30B-A3B** Q4_K (fused, 48 layers) | 19/48 layers, 6.7 tok/s | 29/48, 9.6 | 39/48, 16.4 | 48/48, 55.3 |
| **Mixtral-8x7B** Q4_K (split, 32 layers) | 9/32 layers, 11.6 tok/s | 13/32, 12.7 | 18/32, 14.4 | — (26 GB, never full on 24 GB) |

Decode throughput scales with the fraction of layers on the GPU. Both MoE
families (qwen3moe fused, llama split) and both expert layouts are covered.
Nothing special is required to use it — the runner fits as many leading layers
as the available (or `--reserve`-capped) VRAM allows and runs the rest on CPU.

### Synthetic equivalence tests (`make test-moe`, in CI)

Reference-free correctness: `make-test-moe.py` emits a dense model plus MoE
variants each **mathematically identical** to the dense FFN, so the runner's
already-trusted dense path is the oracle (no separate reference engine):

- `moe1` — fused, top-1, one expert zeroed → identical to dense.
- `moe2` — fused, top-2 with a zero router (0.5/0.5) → identical to dense.
- `moe3` — **split** layout, top-1 → identical to dense.
- partial-offload/fallback path — `--gpu-layers 1` exercises the split when a
  GPU is present and falls back cleanly to CPU on synthetic CI hosts.
- expert-tensor placement — `--cpu-moe --gpu-layers 2` keeps the fixture's
  attention layers on CUDA, runs fused experts on the host, and must remain
  byte-identical to the dense oracle; `--caps` must advertise the feature.

All assert byte-identical greedy output; the FFN is scaled so a broken MoE
produces different tokens (verified during development).

## Methodology

Two correctness checks run today:

1. **Dense-oracle equivalence** (synthetic, CI): MoE configurations constructed
   to equal a dense FFN, asserted token-identical.
2. **CPU/GPU agreement on real quantized models**: the CPU forward is Runner's
   long-validated path over the same GGUF; the GPU MoE output is asserted to
   match it token-for-token on the real Qwen3-30B-A3B. This is an internal
   consistency check, not independent validation.

Independent comparison against llama.cpp is now reproducible through
`scripts/compare_llamacpp.py`, which records the model hash, Runner commit,
llama.cpp version/commit, commands, hardware, driver, context, quantization,
prompt throughput, decode throughput, time to first token, VRAM snapshots,
generated tokens, raw responses, and a numeric common-token top-logprob
comparison when both endpoints expose it. The release gate requires at least
32 shared greedy tokens and a maximum absolute common-token logprob delta of
2.0 over that shared history. It deliberately stops comparing logits at the
first divergent token because subsequent logits condition on different text.
Runner CPU/GPU identity remains a separate exact 128-token gate. The throughput request uses the same
raw prompt and greedy settings; the auxiliary top-k check sends the same chat
payload to both runtimes. TTFT is a separate warmed streaming request. It emits
both JSON and Markdown:

```sh
python3 scripts/compare_llamacpp.py \
  --runner ./runner \
  --llamacpp /path/to/llama.cpp/build/bin/llama-server \
  --llamacpp-commit <llama.cpp git commit> \
  --model /models/Qwen3-30B-A3B-Q4_K_M.gguf \
  --quantization Q4_K_M \
  --ctx 4096 \
  --tokens 128 \
  --prompt "The capital of France is" \
  --out-dir tests/compatibility/out/qwen3-30b-a3b-runner-vs-llamacpp
```

The real comparison was run on 2026-07-28 with model SHA256
`0d003f6662faee786ed5da3e31b29c978de5ae5d275c8794c606a7f3c01aa8f5`.
Runner CPU and Blackwell GPU output remained byte-identical for the full
128-token prompt. Independent greedy output is not required to remain byte-identical:
both pinned llama.cpp `b10076` (`305ba519a`, CPU reference) and a newer
`91d2fc3` GPU build diverged late in the continuation after an identical
prefix. The pinned reference shares 55 tokens and its maximum common-token
logprob delta over that shared history is 1.523, passing the explicit 32/2.0
semantic gate. Strict-math and release Runner builds produced the same Runner
output, so `-ffast-math` is not the cause. Layer-by-layer tracing found gradual
floating-point accumulation drift rather than a discrete MoE routing or tensor
formula error; requiring cross-engine byte identity would therefore encode an
implementation-specific reduction order rather than model correctness.

The committed raw evidence is in
`tests/compatibility/out/qwen3-30b-a3b-runner-vs-b10076-cpu-reference/` and
`tests/compatibility/out/qwen3-30b-a3b-runner-vs-91d2fc3-gpu/`. The latter is
the comparable Blackwell performance run; the pinned CPU-reference run is for
correctness only and its throughput numbers must not be compared to Runner's
GPU numbers. CI continues to exercise the harness with fixtures.

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
  on the Blackwell 24 GB MIG slice and runs on the GPU**, token-identical to
  the CPU reference.
- **MXFP4 read — DONE.** OCP microscaling FP4 (GGML type 39) — the gpt-oss
  expert-tensor format — is read and dequantized (E8M0 block scale × E2M1 code),
  admitted at load and usable through the CPU forward. Unit-tested against the
  OCP spec, and **validated on the real `gpt-oss-20b-MXFP4.gguf`**: the loader
  identifies all 72 MXFP4 expert tensors and a real `ffn_down_exps` row
  dequantizes to finite, sane weights (2527/2880 nonzero, ±0.09375 = 6·2⁻⁶,
  mean ≈ 0). Since 2026-08-01 MXFP4 also has **GPU** kernels (`k_mv_mxfp4`,
  `k_mv_mxfp4_b`, `k_moe_mv_mxfp4` — same E8M0 `ldexpf` decode and codebook as
  the CPU, so the two backends compute the same dequant), and the gpt-oss
  architecture itself runs on both backends — see the CHANGELOG entries for
  the CPU path (2026-07-31), the SWA-rope fix and the CUDA support.
- **Advisor / runner-control — DONE.** The advisor scores MoE throughput by
  *active* params (a MoE decodes at the speed of its active experts, not its
  full resident weight) while VRAM fit stays by total size; it surfaces expert
  residency, and the catalog gained Qwen3-30B-A3B and Mixtral-8x7B entries.

## Known limitations / future work

- **gpt-oss architecture — now supported on CPU, CUDA and Metal.** The Metal
  path is smoke-gated by `make test-metal-gptoss-moe`, which exercises sink-aware
  attention, MXFP4 expert tensors, OAI SwiGLU, router bias, per-expert biases
  and SWA-pattern metadata on a small Mac-runnable fixture.
  **Memory floor for the real model (measured, 2026-08-05):** the 12.1 GB
  `gpt-oss-20b-MXFP4.gguf` weight wrap exceeds the Metal working-set limit
  on a 16 GB Mac (~2/3 of unified memory, ~10.7 GB on that class), so Metal
  init fails and the load falls back to the CPU paging path — 2.33 tok/s
  decode on an M2 Pro / 16 GB under realistic desktop load, which fails the
  "configuration worth running" bar. Full-model Metal validation needs
  **≥ 24 GB unified memory**; 16 GB machines should run the pruned
  [`gpt-oss-20b-keep30`](https://huggingface.co/Joakimpalm-Zen/gpt-oss-20b-keep30-MXFP4-GGUF)
  artifact instead (see the 0.1.6/0.1.7 notes). The
  loader now prints requested-vs-limit bytes when this allocation fails,
  and `--caps` publishes the ceiling as `gpu.max_working_set_bytes`.
- **GELU dual-branch MoE** (gemma-4) is implemented on CPU, CUDA and Metal; see
  the `gemma4-moe` section above. `expert_shared_count`-style shared-expert MoE
  (Qwen2-MoE / DeepSeek) remains refused, behind its own validation.
- **MoE GPU decode** *(resolved 2026-07-29)*: earlier versions forced the eager
  path (host-side routing readback per token). Device-side routing with fused
  indirect expert matvecs is now the default; the eager path remains pinned for
  byte-identity certification runs (see `docs/compatibility-program.md`) and
  the updated MoE rows in `docs/benchmarks.md` supersede the throughput
  figures above.
