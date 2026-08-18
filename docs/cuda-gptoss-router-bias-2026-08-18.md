# The CUDA fused MoE path routed gpt-oss without its router bias

Found and fixed 2026-08-18 during the CUDA backend review. The code change
landed in `e50621b` (it was swept into that commit by a concurrent agent's
`commit -a`, so its own message never made it into the log — this note is the
record).

## The bug

`moe_route()` in `model.c` adds `ffn_gate_inp.bias` to the router **logits**
before gating. gpt-oss is the architecture that ships one (`blk.N.ffn_gate_inp.bias`,
required at load). `metal.m` passes it, and `cuda.c`'s own eager and grouped
MoE paths passed it.

`gpu_moe_ffn_fused()` passed `0`.

That path is the **default** for every gpt-oss model on CUDA: MXFP4 expert
tensors are covered by the indirect kernels, so `moe_fused_eligible()` says yes
and `k_moe_route` runs on device — softmaxing raw logits. The router then
weighted, and at realistic top-k *selected*, experts the CPU would not have.

## Evidence

CPU vs CUDA `mean|dlogit|` as a fraction of the mean logit range
(`test-gpu-identity`, Blackwell RTX PRO 6000 MIG 1g.24gb, CUDA 13.0):

| fixture | with the bug | fixed | bound |
|---|---:|---:|---:|
| `test-moe-fixture.gptoss-top1` (top-1 of 2) | **0.0044 FAIL** | 3.76e-08 ok | 2e-3 |
| `test-moe-fixture.gptoss-mxfp4` (top-2 of 2) | 9.18e-04 ok | 4.30e-08 ok | 2e-3 |
| `test-moe-fixture.moe4` (no router bias) | 3.86e-08 ok | 3.86e-08 ok | 2e-3 |

The fixed number lands on the same noise floor as the bias-free fixture, which
is what says the bias is now applied rather than merely applied differently.

`RUNNER_DEBUG_MOE=1` shows the mechanism directly. First routing, fused vs
eager routing weights as hex floats:

    before   FUSED 3f13e07e 3ed83f04   EAGER 3f39474b 3e8d716b
    after    FUSED 3f39474b 3e8d716b   EAGER 3f39474b 3e8d716b

## Why every existing gate missed it

- The bias-free MoE fixtures have no bias to drop.
- `gptoss-mxfp4` routes **top-2 of 2**: every expert runs whatever the router
  says, so the bias can only reweight two outputs. `make test-moe-tol` measured
  9.08e-04 against its 5e-3 bound and passed; free-running greedy text was
  byte-identical to the CPU at `-b 1`, `8`, `32`, `64` and at `-n` up to 128.
- No MoE fixture was ever handed to `test-gpu-identity` at all.

The new `gptoss-top1` fixture is the same tensors with
`expert_used_count = 1`, so the bias decides *which* expert runs — the regime
the shipping models are in (top-4 of 32 or 128). `make test` now runs
`test-gpu-identity` on it.

## What this does NOT fix

On the real `gpt-oss-120b-MXFP4` at 4 GPU layers, `test-gpu-identity` goes from
0.00946 to 0.00245 of logit range. The bound is 2e-3, so **gpt-oss on CUDA
still carries at least one further CPU/GPU divergence.** This was not the whole
of it. Candidates not yet separated: MXFP4 dequantization, the attention-sink
softmax, and the per-expert gate/up/down bias fold (`k_moe_actmul` /
`k_moe_sum` apply `w*down(h) + selw*db` where the CPU computes
`w*(down(h)+db)` — same quantity, different association).

No identity claim is made for any real gpt-oss file on CUDA on the strength of
this fix. README's recorded gpt-oss-20b CPU/CUDA row stands as measured; it was
taken on a binary carrying this bug, and re-measuring it needs the 20B file,
which is not on the measurement box.
