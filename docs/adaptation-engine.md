# The adaptation engine: scoring, adapters, and training in the serving binary

*2026-08-21. Status: D1–D6 built and gated; CPU reference; CUDA training is
future work. Everything below with a number attached was measured, and the
gates named here run in `make test`.*

The runner can now score, adapt, and train — narrowly scoped as **LoRA
adaptation of a frozen quantized base, in the same binary, on the same
kernels** that serve the model. Not pretraining, not a framework: a GGUF
that learns locally, reproducibly.

## Why inside the inference engine

The systems argument: when the trainer's forward pass is literally the
inference forward pass — same kernels, same bits — the policy you sample is
the policy you train, **by construction**. Train/infer numerical mismatch
(the thing that silently breaks on-policy RL and motivated the industry's
determinism work) cannot occur between two codepaths that are one codepath.
The second claim is **deterministic training**: same data + same seed →
byte-identical adapter file, which extends the runner's reproducibility
discipline from "prove what it said" to "prove what it learned from."

Both claims are gated, not asserted (see D5 below).

## D1 — `--score`: teacher-forced logprobs

Per-token log P(token | prefix) over raw text, plus NLL and perplexity, as
versioned JSON (`xyntetik.runner.score.v1`). The eval/reward primitive
everything else builds on.

Design finding, measured during its red-first gate: **the CPU batched
forward is not bit-identical to solo forwards** (max |Δlogprob| ~1e-6 on the
fixtures). Sampling-time logprobs come from solo decode forwards — so the
scorer *defaults to the solo path*, and the ~10×-faster chunked pass is an
opt-in (`RUNNER_SCORE_CHUNKED=1`) whose deviation envelope is pinned by
test. Semantic sanity: a natural sentence scores ppl 15.3 where its
word-shuffled anagram scores 1818 (SmolLM2-135M Q8_0).

## D2 — `--lora`: adapters at inference

llama.cpp adapter-GGUF naming (`blk.N.<proj>.weight.lora_a/_b`,
`adapter.lora.alpha`), f32 deltas applied as `y += scale·B(Ax)` beside the
untouched base matvecs on the CPU dense projections (attention
q/k/v/output, FFN gate/up/down). Fails closed by name on shape/rank
mismatches, unknown targets, unsupported architectures, and GPU-resident
models. Gated through `--score`: a zero adapter is byte-identical to the
bare base; a real adapter matches the merged-weights reference model within
the float summation-order envelope (5e-4). The adapter id joins the engine's
model identity, so a cached prefix can never serve across an adapter
boundary.

## D3 — the backward pass

Activation gradients flow through the whole network in reverse — attention
including the cross-position dK/dV paths, the rope adjoint, the softmax
jacobian, rmsnorm backward — while weight gradients exist only for the
adapters. Three properties carry the design:

1. **The forward half is the inference forward.** Solo forwards tape each
   layer's residual-stream input; the backward recomputes internals with the
   same primitives (`rmsnorm`, `matvec_b`, the production attention worker)
   and reads K/V from the actual cache, so the values differentiated are the
   f16-rounded values inference attended over.
2. **The transposed quantized matvec** (`dx = Wᵀ·dy` through frozen
   quantized rows, per-row dequant, serial fmaf) carries the frozen-base
   activation gradients — for any weight type the engine can decode, and
   FD-verified through real Q8_0 rows, not just F32.
3. **Byte-determinism**: fixed sequential fmaf accumulation everywhere.

Gate: finite differences on 84 coordinates across all 28 adapter buffers
(every slot × both layers × A and B × F32 **and** Q8_0 bases), aggregate
cosine 0.99997, gradients byte-identical run to run. The gate's design
itself records a measurement: the f16 KV cache **staircases** finite
differences (probed: fd oscillates around the analytic value and converges
to it as eps grows), so the check is two-scale rather than
single-tolerance. Teeth proven by sabotage: a deliberately broken rope
adjoint fails at relative error 1.36.

Declared scope, fail-closed by property: CPU, dense SiLU transformers, f16
KV. Recurrent architectures, MoE, sliding windows, per-head norms and head
transforms refuse with a named reason.

## D4 — `--train`: the loop

Fresh rank-R adapters on every hooked projection (A seeded, B zero — an
exact no-op start), AdamW (0.9/0.999, wd 0.01) in fixed elementwise order,
per-step JSONL loss logging, checkpoints via an adapter-GGUF writer that
emits exactly the format `--lora` reads. Two data modes:

- plain text: tokenize, fixed windows, cycle;
- `.jsonl` lines `{"prompt", "completion", "weight"}`: prompt transitions
  masked to weight 0, completion transitions carry the example weight —
  `model_lora_backward_w`'s per-position weights are the policy-gradient
  hook.

Determinism is the default: fixed init seed unless `-s` is given, fixed
data order, the D3 compute path underneath.

## D5 — the gates (`make test`)

- **Deterministic training**: two identical runs → byte-identical adapter
  files and identical loss trajectories.
- Loss falls on an overfit corpus; the saved adapter loads back through
  `--lora` and improves `--score` on the trained text (fixture: nll
  5.58 → 4.59).
- JSONL mode trains with prompt masking; a different seed produces a
  different adapter (the determinism is seeded, not vacuous).
- Plus the D3 FD gate on both base types.

## Measured at scale (Blackwell, 128-thread CPU path)

**Llama-3.2-3B-Instruct Q4_K_M** (a frozen 4-bit base learning through the
transposed-quantized-matvec path), rank-8 adapters on every projection,
corpus = this repository's README introduction (164 words), 30 AdamW steps
at ctx 96, lr 2e-4, 32 threads:

| measure | value |
|---|---|
| step wall time | ~33 s (992 s for 30 steps) |
| training loss | 5.11 → 0.14 |
| exact corpus text, nll/token | **5.00 → 0.31** (ppl 148 → 1.4) |
| unrelated control sentence | 1.63 → 1.67 (barely moves) |
| off-corpus paraphrase | 6.20 → 6.65 (overfit, as expected) |

The paraphrase row is kept deliberately: 30 steps at lr 2e-4 on 164 words is
memorization, and an honest overfit report shows the specificity — the
adapter learned exactly the trained text, left the control alone, and got
slightly worse at near-miss phrasings. Generalization tuning is a recipe
question, not an engine question.

## D6 — GRPO-lite: reinforcement with zero train/infer mismatch

`scripts/train-grpo-lite.py`: sample K completions per prompt from the
runner itself (seeded — the sampling replays), score with a task reward
(built-in task: emit exactly one valid tool-call JSON object), convert
group-relative advantages (r − mean r) into weighted examples, take one
weighted `--train` pass, repeat. REINFORCE at example granularity —
deliberately the simplest correct member of the GRPO family, because the
demonstration is the mechanism: the sampler and the trainer are the same
binary, so the improvement loop has no numerical seam at all.

**Measured (SmolLM2-1.7B-Instruct Q4_K_M, 4 tool-call prompts × K=8, 6
rounds, lr 2e-5, Blackwell CPU path).** Two runs, both reported because the
failed one taught more:

- **Raw advantages (r − mean), lr 1e-4 — the policy COLLAPSED**: valid-call
  rate 0.91 → 0.78 → 0.94 → 0.88 → 0.59 → **0.09** → 0.38. On a
  high-baseline task the rare failures carry advantage ≈ −0.9 while the many
  successes carry ≈ +0.1; the asymmetric negative mass unlearns the shared
  structure. The textbook naive-REINFORCE failure, reproduced in an
  afternoon on a laptop-class stack — which is rather the point of having
  the loop this cheap to run.
- **Group-normalized, clipped advantages ((r − mean)/std, ±1), lr 2e-5 —
  stable and improving**: 0.906 → 0.906 → 0.875 → 0.938 → **0.969** →
  0.938 → 0.938 (mean reward 0.906 → 0.947). Modest headroom — the base
  model is already decent at the task — but the loop climbs instead of
  detonating, and the whole run replays from its config (seeded sampling +
  deterministic training).

Verdict against the item's own kill-switch ("the RL demo underwhelms →
supervised LoRA remains the product"): the mechanism is demonstrated — the
sampler and trainer share every bit, the loop improves a real quantized
model, and the stabilization it needed is the standard one, arrived at by
measurement. A product-grade recipe (KL anchoring, bigger prompt sets,
harder rewards) is future work, not engine work.

## D8 (begun) — CUDA training, slice 1: the transposed matvec on device

The deterministic-training claim extends to the GPU only if the GPU produces
the same BYTES as the CPU backward. Slice 1 delivers that for the backward's
dominant primitive: `k_mvt_{f32,f16,bf16,q8_0,q4_0,q4_K,q6_K}` compute
`dx += Wᵀ·dy` with the CPU trainer's exact accumulation chain (accumulator
starts from dx, serial j per output element, fmaf, zero-dy skip), and the
gate (`test-mvt`, in `make test`, self-skipping without CUDA) byte-compares
the float buffers against the CPU path on real fixture and real-model
tensors. Measured on the RTX 3070 (CUDA 13.3): **bit-identical on every
type**, first try for five of seven. Throughput, honestly: with per-call
PCIe transfers the f16 head wins 1.9× over one CPU thread, while the naive
per-element q6_K decode LOSES (0.6×) — the primitive's win needs persistent
device buffers and a tiled decode, which is what the remaining D8 slices
(integrated training-side GPU context, batched taped forward) are for. What
slice 1 establishes is the hard part: determinism does not have to be traded
away to move training onto CUDA.

## D9 — `--merge-lora`: folding the adapter into the base

Serving `base + --lora` is the exact form: the frozen base plus an F32
delta, every base identity gate still valid, provenance = base sha +
adapter sha. But an adapter only helps runner users. `--merge-lora OUT`
produces the portable form — a standalone GGUF with
`W' = W + (alpha/r)·B·A` folded into each adapted projection, runnable in
any GGUF runtime:

```sh
runner -m base-Q4_K_M.gguf --lora adapter.gguf --merge-lora merged.gguf
runner -m base-Q4_K_M.gguf --lora adapter.gguf --merge-lora merged-f16.gguf --quant f16
```

Mechanics, and where the honesty lives:

- Each adapted tensor's rows are dequantized, the delta is folded in a
  fixed-order fmaf chain (the same discipline as `lora_apply`), and the row
  is requantized — to its own source type by default, or to `--quant T`.
  **Merging into a quantized type rounds the delta through that type's
  grid.** The merged file is NOT numerically the served base+adapter; how
  much of the learned behavior survives the rounding is a *measurement*,
  per target type, not a given. The exact form stays `--lora`.
- Untouched tensors (and zero-delta pairs) are copied byte-verbatim — no
  gratuitous dequant/requant churn of weights the adapter never touched.
  Gated: merging the all-zero adapter writes a file byte-identical to a
  keep-type requant of the base.
- The merge is deterministic: same base + adapter + flags → byte-identical
  merged file, and on an F32 base the merged floats are gated *byte-exactly*
  against the documented fmaf chain (`test-quantize`).
- Validation mirrors `model_lora_load` (hostile-GGUF discipline): unknown
  projections, shape/rank mismatches, half pairs, and architecture
  mismatches refuse the whole merge — a silently skipped tensor would emit
  a merged model that is not base+adapter.
- Provenance extends D7 to the standalone artifact: `OUT.merge.json`
  carries base/adapter/merged sha256s plus the scale and target, so the
  merged blob remains auditable back to what it was made from.

Requantizing into K-quant bases needed writers the requantizer didn't have:
faithful ports of ggml's `Q4_K` and `Q6_K` quantizers (and a
round-to-nearest-even `BF16`) now sit beside the existing `q8_0/q4_0/q3_K`,
gated on round-trip error through the production dequant readers and on
byte determinism. They are general: `--quantize` and `--type-plan` accept
`q3_k`, `q4_k`, `q6_k`, and `bf16` targets now too.

## Honest limits

- CPU-only v1 (CUDA training is future work); the M1-class floor is ~2B
  models per the T0 memory audit, larger bases want a many-core box.
- Ecosystem-adapter interop (adapters trained elsewhere) is unverified
  until a real third-party adapter is on the shelf — the format matches
  llama.cpp's by construction, but unmeasured is not claimed.
- The GRPO-lite demo optimizes a narrow synthetic reward; it demonstrates
  the mechanism, not a product-grade RL recipe.
