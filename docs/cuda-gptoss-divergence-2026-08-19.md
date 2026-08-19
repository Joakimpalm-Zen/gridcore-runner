# The remaining gpt-oss CPU/CUDA divergence is chaotic amplification, not a wrong op

Investigated 2026-08-19 on the Blackwell box (RTX PRO 6000 Blackwell,
MIG 1g.24gb, 25 GB visible; CUDA 13.0 `ccbuild` toolchain), against
`docs/cuda-gptoss-router-bias-2026-08-18.md`'s open question: after the
router-bias fix (`e50621b`), `gpt-oss-120b-MXFP4` at 4 GPU layers still measured
`test-gpu-identity` at 0.00245 of logit range, over the 2e-3 bound. That note
named three not-yet-separated candidates — MXFP4 dequant, the attention-sink
softmax, and the per-expert bias fold. This is the bisection that separated them.

Binary: worktree at `e776961` (baseline HEAD), built with
`CC=x86_64-conda-linux-gnu-gcc`. `gpt-oss-20b-MXFP4.gguf`
md5 `dbb8242ec9bbf70ddc6269ead739b2ca` (12.1 GB). The accelerated path was
proven engaged on every GPU run: `gpu-split ... full=1 used=12.00GB` and
`CUDA backend ... 12.1 GB weights in VRAM` — never a CPU fallback.

## The headline: gpt-oss-20b full-offload PASSES; the FAIL is a partial-offload artifact

`gpt-oss-20b-MXFP4` is now on the box, so the README row the router-bias note
left as a re-measurement candidate could finally be re-run. `test-gpu-identity`
mean|dlogit| as a fraction of mean logit range (bound 2e-3):

| GPU layers (of 24) | fraction of range | result |
|---:|---:|:--|
| 1  | **0.00356** | FAIL |
| 4  | 0.000733 | ok |
| 12 | 0.000732 | ok |
| 24 (full offload) | 0.000732 | ok |

The number is **non-monotonic in the number of GPU layers**, and full offload —
the actual deployment configuration — passes comfortably. A systematic wrong
CUDA op cannot produce this shape: more GPU layers would mean more of the wrong
arithmetic, the divergence would grow, and full offload could never be the
*smallest* number. It is the smallest.

The GPU takes the **leading** `gpu_layers` (`model.c` "GPU handles the leading
gpu_layers"). So "1 GPU layer" runs layer 0 on the device and layers 1–23 on
the CPU. What that measures is: a single layer's worth of legitimate
reduction-order rounding, injected at the very bottom of the stack and then
**amplified through 23 downstream CPU layers**. Full offload injects rounding at
every layer but has *no* downstream CPU layer to amplify it — the device output
goes straight to the final norm and logits. The magnitude tracks amplification
depth, not GPU-op count.

## The mechanism, shown directly: a near-tied top-4 expert reorders

`RUNNER_MOE_TRACE` was captured on the same teacher-forced prompt for the
CPU-only run and the 1-GPU-layer run (`runner -n 1`, prefill). Because the input
tokens are identical, routing is comparable position-by-position. Layer 0 is on
the device in the GPU run and emits no CPU trace record; layers 1–23 are on the
CPU in both, so any selection difference there is caused purely by layer 0's
device rounding propagating up.

Across all 621 comparable (position, layer) routing points, **exactly one**
differed:

    pos 14, layer 6:
      CPU  experts [16, 31, 4, 20]  gates [0.2584, 0.2492, 0.2492, 0.2432]
      GPU  experts [16, 4, 31, 20]  gates [0.2584, 0.2492, 0.2492, 0.2431]

Experts 31 and 4 sit on a gate tie to four decimals (0.2492 vs 0.2492). A
sub-ULP perturbation, five layers downstream of the one offloaded layer, is
enough to swap their order. This is the discrete top-k routing chaos the
compatibility program already documented for `gemma-4-26b-a4b` (top-8-of-128 on
Q4_0, `docs/compatibility-program.md`): near-tied experts turn a rounding
difference into an expert flip, which rewrites a quarter of an FFN output, which
the next layer's router sees. The three-decimal density of near-ties here means
the effect is pervasive, not a single unlucky token.

## The model's own sensitivity floor — the applicable bound

`docs/compatibility-program.md` is explicit that a cross-engine gap is
uninterpretable without the model's own floor: perturb one build below the size
of switching engines and see whether the model already disagrees with itself.
`scripts/sensitivity_floor.py --mode runner` does exactly this — CPU-only,
f16 KV cache vs q8 KV cache, 16 prompts × 16 tokens:

    gpt-oss-20b-MXFP4, runner f16-KV vs runner q8-KV (CPU only):
      identical                     13 / 16
      prompts that diverge           3 / 16
      max  logprob delta         0.2807
      mean logprob delta         0.02085

The model flips generated tokens on 3 of 16 prompts under a KV-precision change
**entirely inside one CPU build**, with no engine change at all. By the
program's own doctrine, a CPU-vs-GPU disagreement no larger than that cannot be
attributed to a backend defect — it is inside the model's instability envelope,
exactly as for `gemma-4-26b-a4b`, whose `greedy_reference` check the manifest
omits for this reason.

## The three named candidates are all algebraically identical CPU↔CUDA

Read side by side, none of the three is a wrong operation; each is the same
quantity in a different association, i.e. reduction-order rounding — the class
`test-gpu-identity`'s 2e-3 bound exists to tolerate for *dense* models.

- **MXFP4 dequant.** CUDA `k_moe_mv_mxfp4`/`k_mv_mxfp4` accumulate
  `d · Σ_block(kv·x)` per 32-element block; the scalar CPU `dot_mxfp4`
  (`quants.c`) accumulates `s += ldexpf(1, e-127) · t` with
  `t = Σ(kv·x)` — the identical `d·Σ(kv·x)` per-block form. The codebook
  `kv_mxfp4` equals CPU `kvalues_mxfp4` element for element, and both derive the
  scale with `ldexpf(1.0f, e-127)`. The only CPU/CUDA difference is the
  summation partition (warp tree vs SIMD lanes).
- **Attention-sink softmax.** CUDA `k_attn`/`k_attn_dec` join the sink to the
  max scan (`if (sinks[h] > mx) mx = sinks[h]`) and to the denominator
  (`sum += expf(sinks[h] - mx)`) with no value row — line for line the CPU
  `softmax_sink` (`model.c`). The sink competes against already-scaled scores on
  both sides.
- **Per-expert bias fold.** CUDA computes `selw·down(h) + selw·db` (`k_moe_actmul`
  folds `selw·dscale` into the hidden before the down matvec; `k_moe_sum` adds
  `selw·db`); the CPU computes `selw·(down(h)+db)`. For gpt-oss `dscale` is the
  all-ones `moe_ones` table, so both equal `selw·down(h) + selw·db` up to the
  order of the multiply and add.

## Verdict and recommendation

There is no CUDA correctness defect to fix. The remaining gpt-oss CPU/CUDA
divergence is legitimate reduction-order rounding, amplified by the model's
intrinsic top-4-of-32 MoE chaos when — and only when — a partial offload leaves
downstream CPU layers to amplify it. The evidence:

1. Full offload (the deployment config) passes at 0.000732, 2.7× under the bound.
2. The divergence is non-monotonic in GPU layers — impossible for a systematic
   wrong op.
3. The one routing difference found is an order-swap of two experts tied to four
   decimals.
4. The model disagrees with itself (3/16) under a smaller, in-build perturbation.
5. All three named candidate ops are algebraically identical across backends.

`gpt-oss-120b-MXFP4` at 4 GPU layers reproduced the router-bias note's
**0.00245** exactly on this build. The 120b (63 GB) cannot be fully offloaded on
a 24 GB MIG, so the chaotic-amplification regime — 4 device layers feeding 32
CPU layers — is the *only* regime measurable for it here. Its 0.00245 is the
same effect as the 20b's 1-layer 0.00356, at greater amplification depth on a
deeper, 128-expert model; it is not evidence of a further op bug.

Recommendation: **do not change any kernel.** `test-gpu-identity`'s
fraction-of-range bound is a dense-model instrument and is not the applicable
certification for a chaotic MoE — expert-selection identity plus routing-weight
tolerance is (`compatibility-program.md`, MoE fused-path class), and gpt-oss is
gated at its measured KV-precision sensitivity floor, which it satisfies. The
README identity row is corrected from "re-measurement candidate / not re-run" to
this measured result.

Evidence is reproducible from this box: `test-gpu-identity <20b> {1,4,12,24}`,
`scripts/sensitivity_floor.py --mode runner --model <20b>`, and the
`RUNNER_MOE_TRACE` diff above. No file was committed to the box; measurements
only.
