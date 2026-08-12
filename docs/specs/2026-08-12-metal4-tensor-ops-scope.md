# Metal 4 tensor-op GEMM: scope & go/no-go design (design-only)

Goal: decide, on evidence rather than improvisation, whether adopting Apple's
Metal 4 tensor API / GPU Neural Accelerators for the prefill GEMM is worth
building. This is suite plan P3. **This document is design-only.** No
M5-class Mac exists in this project's hardware set — the CUDA measurement
box (Blackwell) is x86, and Apple-silicon testing to date has been limited
to M1-class hardware (see `CHANGELOG.md`'s Metal MM promotion entry and
`docs/compatibility-program.md`'s cert matrix, both measured on M1). Nothing
here is implemented or benchmarked on the actual target hardware — it is
scoped so that whoever next holds an M5 Mac can run a short, pre-defined
measurement and get a verdict instead of starting from zero.

The discipline mirrors `docs/specs/2026-07-22-tensor-core-gemm-scope.md`
(CUDA tensor cores): state the API options and the recommendation, identify
the prerequisite the naive path will trip on, define the tolerance gate and
promotion bar *before* writing the kernel, and write an honest, falsifiable
prediction of the ceiling so a negative result re-scopes the work instead of
getting rationalized away. That spec was scoped, measured, **initially lost**
(WMMA at the runner's N=8 batch width was 7x *slower*, not faster — a
"fundamental, not a tuning miss" mechanism), and only won after the
identified prerequisite (batch-tile widening to N>=16) landed. Treat a
negative Metal 4 result with the same seriousness: it is a valid, useful
outcome of this plan, not a failure of it.

## The trigger

An external evaluation on an M5 Max measured llama.cpp prefill at
**~7,500 tok/s** on a 4.65B-class model. The runner's current Metal prefill
GEMM (`k_mm_*`, `simdgroup_float8x8` tiles, promoted default since
0.1.11 — see `CHANGELOG.md` and `docs/compatibility-program.md`) reached
**~575 tok/s** in the same comparison. That ~13x gap is large enough that
"tune the existing kernel harder" is not a credible explanation by itself —
the arithmetic implies llama.cpp is reaching the M5's dedicated GPU Neural
Accelerator hardware, a unit the runner's kernel cannot touch because it was
never written against the API that drives it.

This is a **different regime from the CUDA TC spec's finding.** There, the
scalar kernel was already near a real bandwidth/compute floor and TC only
paid off after a batch-width rewrite, with a bounded ~3x prefill ceiling.
Here, if the ~7,500 vs ~575 gap is real and hardware-driven, it implies an
order of magnitude, not 3x — but it is also **entirely conditional on
silicon this project does not own.** Sections below separate what is
architecture (true on any M5) from what is a specific model/prompt/build
data point (not yet reproduced here).

## What Apple exposes (researched 2026-08-12)

Apple's own description: "Metal 4 introduces the tensor resource and the
Metal Performance Primitives (MPP) framework, enabling efficient development
of machine learning kernels that leverage GPU neural accelerators in the
Apple M5 chip." ([Apple Machine Learning Research, "Exploring LLMs with MLX
and the Neural Accelerators in the M5 GPU"](https://machinelearning.apple.com/research/exploring-llms-mlx-m5);
[WWDC25 "Combine Metal 4 machine learning and graphics"](https://developer.apple.com/videos/play/wwdc2025/262/))

### The hardware claim, and the finding that undercuts a naive reading

The Neural Accelerator is described as a **real, dedicated per-GPU-core
matrix unit** on M5: "each one specified to perform 1,024 FP16 fused
multiply-accumulate operations per cycle" (search-aggregated from Apple's
M5 materials; not independently re-derived here — flagged as **secondary,
not primary-sourced** in this pass). Apple reports up to **4x prefill /
time-to-first-token speedup vs M4**, and a roughly **2x-over-fp16 throughput
gain for int8 matmuls**; decode gets only 19-27% (attributed to the M5's
higher memory bandwidth, 153 vs 120 GB/s — decode stays bandwidth-bound, not
compute-bound, exactly the same shape as the CUDA TC finding that decode is
TC-immune).

The important counter-evidence is **Rigel** (["Reverse-Engineering the Metal
4.1 Tensor Compute Path on the Apple M4 Max GPU", arXiv:2606.12765](https://arxiv.org/abs/2606.12765)),
which reverse-engineered the *same* Metal 4.1 tensor API — `matmul2d` over
`cooperative_tensor` fragments — running on an **M4 Max**, i.e. one
generation before the Neural Accelerator exists. Their finding: on M4 Max,
"the matmul2d operation executes entirely on the GPU shader cores with no
dedicated matrix datapath" and "no evidence of Apple Neural Engine routing"
— it is a **software path**, not hardware acceleration; their fp8 (E4M3)
variant, despite half the operand bandwidth of fp16, sustained only "0.94x
the throughput of fp16," which is the signature of emulation, not a
dedicated unit. This is strong, cited, third-party evidence that the Metal 4
tensor *API* is portable back to at least M4, but the **hardware speedup is
M5-specific** — the API existing on older silicon is not itself the win.

This is corroborated by the closest thing to a real measurement this
research pass found: **llama.cpp's own Metal 4 tensor-API patch**
([`ggml-org/llama.cpp` PR #16634](https://github.com/ggml-org/llama.cpp/pull/16634),
merged 2025-11-06). It reworks the mat-mat kernel to use the tensor API when
available, and **explicitly disables it on M4 and earlier** — the PR
discussion states testing on an M2 Ultra showed the tensor path was *slower*
than the existing `simdgroup_matrix` kernel there. On M5 hardware the same
patch reports real gains: **~23%** on an iPhone 17 Pro Max (Mistral-7B,
13.66 vs 11.08 tok/s — a decode number, consistent with Apple's 19-27%
decode figure) and **~2x on some models on a MacBook Pro M5**. This is the
single most directly comparable data point to this project's own kernel and
the strongest evidence available without owning the hardware — but note it
is prefill/decode-mixed across two very different pieces of silicon (phone
vs laptop-class M5), not the isolated prefill-only number the runner's own
gate would need.

One more real-world data point, useful for the fallback-ladder design
below: **LM Studio's bug tracker** ([lmstudio-ai/lmstudio-bug-tracker#2040](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/2040))
recorded a shipped llama.cpp runtime whose Metal 4 tensor-API *self-test*
failed on real M5 Macs (an SDK/deployment-target mismatch, not missing
hardware), silently falling back to the slow path and costing users a
**2-3x prefill regression** with no error, just a log line. This is exactly
the failure mode `gpu_available()` in `src/metal.m` was written to prevent
for the shader-library case ("a device that exists is not a backend that
works... reporting a usable backend in that state would be a lie that costs
a whole run") — any Metal 4 tensor admission test here must fail the same
way: loud, at load, defaulting to the known-good path, never a silent
downgrade discovered only in aggregate throughput.

### API options compared

| API | Portability | Min silicon for HW accel | Min OS | Precision | Maturity | Verdict |
|---|---|---|---|---|---|---|
| **Metal Performance Primitives (MPP) `tensor_ops::matmul2d` over `cooperative_tensor`** | Compiles/runs back to ~M4 (Rigel); **only M5+ engages the Neural Accelerator** | M5 (per-core Neural Accelerator) | macOS 26 for the API; **26.2+** for functional bf16 in MPP (see below) | fp16 confirmed; bf16 needs 26.2+ (a macOS 26.0 MPP bug excluded it — `ollama/ollama` issues [#13460](https://github.com/ollama/ollama/issues/13460), [#15862](https://github.com/ollama/ollama/issues/15862)); int8 matmul reported ~2x fp16 throughput on M5 | New (shipped WWDC25, llama.cpp landed it 2025-11-06); still shaking out OS-version and precision bugs in the wild as of this research pass | **Recommended target**, gated behind admission testing — it is the only option that reaches the Neural Accelerator at all, and it is the API the one comparable open-source measurement (llama.cpp) uses |
| **MPSGraph** | Runs on any Metal device back to ~2020; higher-level graph compiler decides internally whether to route through MPP | Presumably auto-routes to the Neural Accelerator on M5 when available (uncertain — not directly confirmed by this research pass) | Older / broader than raw MPP | Framework-managed; not directly controlled per-op | Mature, but is a whole-graph API — mismatched to a single-kernel GEMM seam like `enc_mv_n` | Not recommended as the integration point: it wants to own the compute graph, and the runner's design (bespoke kernels dispatched per-projection from `src/metal.m`) is the opposite of that. Worth re-checking only if MPP's low-level control proves not to pay off |
| **Current baseline: `simdgroup_float8x8` (`k_mm_*`)** | Portable to any Metal GPU with simdgroup-matrix support (M1+, promoted default since 0.1.11) | None needed — software path on every generation | Whatever the current Metal minimum is | fp32 threadgroup staging, fp32 accumulate (see "Where it plugs in" below — this is itself a possible near-term win independent of Metal 4) | Shipped, gated, measured (768 tok/s SmolLM2-135M Q8_0 on M1; see `CHANGELOG.md` 0.1.11 entry) | Stays the fallback / M1-M4 default regardless of this plan's outcome |

**Recommendation:** target MPP `tensor_ops::matmul2d` directly, not MPSGraph.
It is the only path that can reach the Neural Accelerator, it is a
per-kernel primitive that fits the existing `enc_mv_n` dispatch seam
(see "Where it plugs in" below) rather than requiring a graph-compiler
integration, and it is the API the one external, reproducible comparison
(llama.cpp) already validated as a net win on real M5 hardware. The
uncertainty is not *which API* — it is *whether the gain reproduces in this
codebase's kernel shape and quant formats*, which is exactly what the
"Honest prediction" section and the hardware-gated list after it are for.

## Where it plugs in

`src/metal.m`'s `enc_mv_n` (the seam this project already extended once, for
the CUDA-TC-style tolerance-gated Metal MM promotion) is the dispatch point:

```c
if (n_col > 1 && metal_mm_on() && g->p_mm[w->type] &&
    n_in % 32 == 0 && !(<K-quant 256-alignment guard>)) {
    // k_mm_* : simdgroup_float8x8 tiled GEMM (current baseline)
    ...
} else {
    // k_mv_* : matvec, one output element per simdgroup
    ...
}
```

A Metal 4 tensor path adds a **third rung above this**, not a replacement:
a `metal_tensor_on()` check and a `g->p_tensor[w->type]` pipeline table,
tried before the `metal_mm_on()` branch, falling through to `k_mm_*` and
then `k_mv_*` exactly as today. Concretely:

- **What stays:** the outer per-projection call sites in the forward pass
  (`enc_mv`/`enc_mv_n` callers) do not change — every weight matmul in the
  model already funnels through this one function. The matvec kernels
  (`k_mv_*`), the `RUNNER_METAL_MM` gate and its promoted-default logic, the
  `mm_args` struct, and the tile geometry constants (`MM_TM=32`, `MM_TN=16`,
  `MM_TK=32`) for the existing `simdgroup_float8x8` kernels are untouched —
  they remain the M1-M4 (and Neural-Accelerator-detection-failure) path.
  Weight loading, the `gguf_tensor`/`model_t` layer stays identical; the
  tensor path reads the same on-disk quant blocks, just dequantizes them
  into a different staging buffer shape.
- **What changes:** the `k_mm_*` kernels currently stage dequantized weight
  and activation tiles into **`float` threadgroup memory**
  (`threadgroup float tg_w[...]`, `tg_x[...]`, `kernels.metal:726-728`) before
  feeding `simdgroup_float8x8`. MPP's `matmul2d` operates over
  `cooperative_tensor` fragments and the confirmed-working precision is
  **half** (`tensor<device half, ...>` in Apple's own sample; bf16 is
  version-gated, see above) — so a tensor-path kernel restages into `half`
  (or a `cooperative_tensor` equivalent) rather than `float`, changing both
  the staging buffer's byte size (2x smaller for the same tile) and the
  accumulate precision contract, which is exactly the kind of change that
  makes bit-identity impossible and a tolerance gate mandatory (see "Gate
  design" below). Tile
  shape is also an open question: MPP is issued at simdgroup/threadgroup
  scope with its own preferred `matmul2d_descriptor` M/N/K, which is **not
  known to match `MM_TM`/`MM_TN`/`MM_TK` today** — this needs to be read
  from Apple's MPP header/docs and tuned on real hardware, not assumed.
- **New surface, small:** one pipeline table (`g->p_tensor[T_*]`), one env
  gate (`RUNNER_METAL_TENSOR`, mirroring `RUNNER_METAL_MM`/`RUNNER_CUDA_TC`),
  one admission test at `gpu_available()`/`gpu_init()` time (see "Fallback
  ladder" below), and kernel
  bodies for the same type set `k_mm_*` already covers (F32, F16, Q8_0,
  Q4_0, Q4_K, Q6_K, MXFP4 — dequant-then-multiply logic is reusable from the
  existing `k_mm_*` bodies almost verbatim; only the staging type and the
  matmul primitive itself change).

## Gate design

Same posture as the CUDA TC and Metal MM promotions — **not bit-identical by
construction**, so identity is never the bar:

- **Not bit-identical.** Half-precision staging and MPP's (currently
  undocumented, to this research pass) accumulator precision will not match
  either the scalar matvec path or the existing fp32-staged `k_mm_*` path
  bit-for-bit. This is the same class of change `test_tc_tol.c` was built
  for.
- **Promotion bar, reusing the existing instrument.** `tests/test_tc_tol.c`
  already gates Metal MM promotion (teacher-forced 64 positions, top-1
  agreement with a near-tie escape of <=5% flips each within 2% of the
  logit range, plus mean|dlogit| <=0.5% of the logit range) — extend it to
  drive `RUNNER_METAL_TENSOR` the same way `gpu_tc_force()` already drives
  `RUNNER_METAL_MM`. The promotion bar for this plan specifically:
  - **Per-combo tolerance gate** (quant type x arch), same thresholds as
    the existing table, run against the tensor path vs the *scalar* matvec
    path (not vs `k_mm_*` — two already-approximate paths comparing to each
    other would mask a regression either one introduced).
  - **0/64 teacher-forced top-1 flips** on the measured dense set before
    considering promotion (qwen3moe-class MoE archs should expect to be the
    tightest row again, per the existing gate's finding that MoE amplifies
    fp16 noise ~86x over dense — budget for that, don't be surprised by it).
  - **Decode unchanged.** The tensor path is prefill-only by design (Apple's
    own numbers put decode's gain at 19-27%, bandwidth-attributed, not
    compute) — the gate must assert decode logits/tokens are byte-identical
    to the pre-change build, i.e. `RUNNER_METAL_TENSOR` touches only the
    `n_col > 1` branch and never engages on batch-1 decode.
- **The pin.** `RUNNER_METAL_TENSOR=0` (style-matched to `RUNNER_METAL_MM`
  and `RUNNER_CUDA_TC`) forces the fallback ladder's next rung down for
  identity investigations and certification, exactly as `RUNNER_METAL_MM=0`
  and `RUNNER_CUDA_TC=0` already do — `docs/compatibility-program.md`'s
  scalar-path pinning section is the precedent to extend, not replace.
  `=1` should force the tensor path wherever a kernel+admission test exist
  (how the gate itself gets run); unset should be per-combo promoted
  default once (and only once) gated, mirroring `tc_promoted()` in
  `src/cuda.c`.

## Fallback ladder

```
tensor path (Metal 4 MPP, M5+ only, half-staged)
   -> simdgroup GEMM (k_mm_*, current default, fp32-staged, M1+)
      -> matvec (k_mv_*, batch-1 / narrow-batch / unsupported-type path)
```

**Detection at load, fail closed to the proven rung.** The LM Studio
incident (above) is the cautionary precedent: a Metal 4 tensor-API
self-test that fails silently and falls back to the *slow* path with only a
log line cost real users 2-3x prefill with no error surfaced anywhere a
scheduler could see. This project's own `gpu_available()` comment states the
standing principle: "A device that exists is not a backend that works... a
scheduler can place work BEFORE dispatching, so reporting a usable backend
in that state would be a lie that costs a whole run." The tensor-path
admission test must follow the same shape already used for the shader
library: attempt to compile/link the MPP kernel and (ideally) run one
`matmul2d` dispatch against a known input at `gpu_available()`/`gpu_init()`
time; anything short of full success reports the API as absent — never
"present but degraded." `k_mm_*` (the current default) is a fully-correct,
already-gated fallback, so failing closed to it costs the M5-tensor upside
on that boot, not correctness or a silent regression to the memory-bound
matvec path.

**What `--caps` should report.** Today `--caps`'s `"gpu"` object
(`src/main.c`, the `caps` branch) reports `moe`, `eseries`, `kv_q8`,
`shader_source_sha256` and `max_working_set_bytes` for the Metal backend,
but **does not currently report the `k_mm_*`/matvec promotion state at
all** — `gpu_tc_dispatches()` exists as an instrumentation counter but
nothing in `--caps` surfaces whether tiled-GEMM is engaged or promoted for
the loaded model. That is a pre-existing gap this plan should close either
way: add a `"tensor_gemm"` (or similarly named) boolean/tri-state to the
`--caps` `"gpu"` object reporting {absent (pre-M5 or admission test failed),
present-but-not-promoted (opt-in only), promoted (default for this
type/arch)} — the same three states `RUNNER_METAL_MM`'s design implies but
never exposed externally. A scheduler placing prefill-heavy work should be
able to read this before dispatch, the same argument `gpu_available()`'s own
comment already makes for the backend-presence bit.

## Prerequisites

- **Half staging.** `k_mm_*`'s threadgroup tiles are `float` today
  (`kernels.metal:726-728`). Whether the tensor path needs a preceding,
  independent half-staging change to the *existing* `simdgroup_float8x8`
  kernel (smaller threadgroup footprint, possibly a free win on M1-M4 on its
  own) or whether it is only ever needed newly, inside the tensor kernel
  itself, is open — flagged per the task brief as **sibling work that may
  land first**; this plan does not assume its outcome, only that if it lands
  first the tensor kernel reuses its staging helpers rather than
  duplicating them.
- **Batch/tile width.** The CUDA TC spec's decisive prerequisite was
  widening MVB from 8 to >=16 because WMMA's fp16 tiles were half-utilized
  at N=8. Metal's existing `k_mm_*` already tiles at `MM_TN=16` columns per
  threadgroup and dispatches over the model's real prefill batch width
  (`n_col`, not a fixed small constant) — so the *same* narrow-batch failure
  mode the CUDA spec hit is **not obviously present** here. What is
  genuinely unknown is whether MPP's `matmul2d_descriptor` wants a different
  (possibly larger) M/N/K than 32/16/32 to reach good Neural Accelerator
  occupancy — this is an MPP-header/hardware question, not something this
  research pass could resolve without the API's concrete minimum-tile
  guidance or a real device to probe.
- **macOS 26.2+.** Confirmed needed for functional bf16 in MPP
  (`ollama/ollama` #13460, #15862); fp16-only may work on 26.0, per
  llama.cpp's PR discussion, but bf16 is excluded pre-26.2 by a documented
  MPP bug, not a design choice — the admission test must check the
  functional path, not just "API symbols exist."

## Honest prediction (falsifiable)

- **M1-M4: no ceiling change expected — none.** Rigel's finding (matmul2d on
  M4 Max runs entirely on shader cores, no dedicated matrix datapath, fp8
  measured at 0.94x fp16 throughput — the signature of software emulation)
  and llama.cpp's own choice to disable the tensor path on M4-and-earlier
  after measuring it *slower* than `simdgroup_matrix` there (their M2 Ultra
  test) both point the same direction: **the Metal 4 tensor API is not a
  lever on any silicon this project can currently test on.** The existing
  `k_mm_*` `simdgroup_float8x8` path should remain the measured-best default
  on M1-M4 regardless of this plan's outcome. If a real M4-class Mac ever
  becomes available to gate on, the falsifiable prediction is: tensor path
  <= simdgroup path there, and any tensor-path result *faster* than
  `k_mm_*` on pre-M5 silicon should be treated as a measurement bug before a
  win.
- **M5-class: real gain expected, magnitude uncertain, ceiling well below
  the trigger's raw ratio.** Apple's own reported range (~2x-4x
  prefill/TTFT vs M4, decode +19-27%) and llama.cpp's measured ~23%-2x
  range on real M5 hardware are the credible priors — call it a **1.2x-4x
  prefill-only band**, prefill-only, decode essentially unaffected. The
  trigger's ~13x ratio (7,500 vs 575 tok/s) almost certainly reflects more
  than the tensor unit alone: different model (4.65B vs whatever this
  project measures), different quantization, and possibly a different
  starting gap that predates any tensor-unit involvement at all —
  `docs/benchmarks.md`'s CUDA comparison already shows this runner's prefill
  trailing llama.cpp for reasons unrelated to any specific accelerator
  (kernel maturity, tiling, fusion), so a large runner-vs-llama.cpp gap is
  not on its own evidence that a hardware unit explains all of it; the
  Metal side has no equivalent published comparison to check that against.
  **The go/no-go bar this plan sets: if a gated M5
  measurement lands materially below ~1.2x prefill over the current
  `k_mm_*` path, that is a negative result on par with the CUDA TC spec's
  Phase 1 (WMMA at N=8) — expected, useful, and grounds to hold the feature
  exactly as WGMMA was held pending the batch-width prerequisite.**

## What cannot be validated without M5 hardware

Listed explicitly, per the task brief, so nobody mistakes desk research for
measurement:

1. **Whether MPP's `matmul2d` engages the Neural Accelerator for this
   project's actual quant dequant-then-multiply kernel shapes** (Q4_K,
   Q6_K, Q8_0, MXFP4) rather than the dense fp16/bf16 shapes Apple's and
   llama.cpp's public numbers were measured on.
2. **The real prefill tok/s delta on this codebase's models and prompts**
   — every number in the "Honest prediction" section above is either
   Apple's own marketing/research figure or
   a different project's (llama.cpp's) measurement on different weights;
   none of it is a substitute for running `RUNNER_METAL_TENSOR=1` on this
   runner on an actual M5.
3. **MPP's exact accumulator precision and rounding behavior**, needed to
   set the tolerance gate's numeric thresholds correctly (Rigel notes the
   spec "deliberately obscures" accumulator precision and execution
   location) — the gate can be written now, but its pass/fail thresholds
   for this specific kernel should be sanity-checked against a first real
   measurement, the same way the CUDA TC gate's constants were tuned from
   the q8-KV precedent and then validated against real weights.
4. **Whether `MM_TM=32`/`MM_TN=16`/`MM_TK=32` is even close to right for
   `matmul2d_descriptor`**, or whether the tensor kernel needs a materially
   different tile shape to reach good occupancy on the Neural Accelerator.
5. **The macOS 26.0 vs 26.2+ functional gap** (bf16 support, and whatever
   else the `ollama/ollama` #13460/#15862 reports surfaced) cannot be
   confirmed to be fully resolved, or to be the complete list of
   version-gated behavior, without running on both.
6. **Detection-test false-negative/false-positive behavior** — the LM
   Studio incident shows a real shipped admission test failed in the wild;
   this plan's own admission test design (the "Fallback ladder" section
   above) is reasoned from that
   incident and from `gpu_available()`'s existing pattern, but has not been
   run against a real macOS 26.x + M5 environment to confirm it fails
   closed the way it is meant to.

## Recommendation

**Design-only, as scoped.** Do not implement until M5-class hardware is
available to this project. When it is, the first action is not a full
kernel — it is a single `matmul2d` admission-test spike against one quant
type (Q4_K, since it is the dense-arch promoted default on both CUDA TC and
Metal MM already) with `RUNNER_METAL_TENSOR` wired to `gpu_tc_force()`-style
plumbing and `test_tc_tol.c` extended to drive it, mirroring the CUDA TC
spec's own Phase 1 ("one kernel, portable, measured"). That spike either
lands near the ~1.2x-4x prefill band predicted above — worth extending to
the rest of the type table — or it does not, in which case this plan's job
is done: the negative result is committed, `k_mm_*` remains correct on
every generation including M5, and the decision was made on evidence
instead of the trigger's raw (and likely apples-to-oranges) 13x ratio.

## Sources

- [Apple Machine Learning Research — "Exploring LLMs with MLX and the Neural Accelerators in the M5 GPU"](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)
- [WWDC25 — "Combine Metal 4 machine learning and graphics"](https://developer.apple.com/videos/play/wwdc2025/262/)
- [Rigel: Reverse-Engineering the Metal 4.1 Tensor Compute Path on the Apple M4 Max GPU, arXiv:2606.12765](https://arxiv.org/abs/2606.12765)
- [ggml-org/llama.cpp PR #16634 — "metal: initial Metal4 tensor API support"](https://github.com/ggml-org/llama.cpp/pull/16634)
- [lmstudio-ai/lmstudio-bug-tracker #2040 — Metal tensor-API detection failure on M5](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/2040)
- [ollama/ollama #13460 — Metal library compile error, bfloat/half mismatch, macOS 26.2](https://github.com/ollama/ollama/issues/13460)
- [ollama/ollama #15862 — MPPTensorOpsMatMul2dImpl static_assert on Apple M5](https://github.com/ollama/ollama/issues/15862)
- [antirez/ds4 issue #14 — Metal 4 Tensor API feature request](https://github.com/antirez/ds4/issues/14)
- In-repo: `docs/specs/2026-07-22-tensor-core-gemm-scope.md` (the CUDA TC
  precedent this plan mirrors), `src/metal.m` (`enc_mv_n`, `metal_mm_on`,
  `gpu_available`), `src/kernels.metal` (`k_mm_*`, `MM_BODY`),
  `tests/test_tc_tol.c` (the tolerance gate to extend), `CHANGELOG.md`
  0.1.11 entry (Metal MM promotion numbers), `docs/compatibility-program.md`
  (scalar-path pinning precedent).
