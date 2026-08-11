# granite (IBM Granite 4.1 dense) certification — 2026-08-11

Environment: runner `1ce9103` (the granite arch commit), llama.cpp reference
**b10353** (static build), Blackwell box. Two artifacts under test, both
official IBM GGUFs (SHAs in `granite-evidence/shas.txt`):
`granite-4.1-8b-Q4_0.gguf` (5.06 GB — the M1-tier artifact) and
`granite-4.1-30b-Q3_K_S.gguf` (12.57 GB — the 16 GB-tier candidate), with
`granite-4.1-30b-Q8_0.gguf` as the 30B's well-conditioned identity and KLD
baseline. Raw gate outputs in `docs/granite-evidence/`.

## Verdicts

**granite-4.1-8b Q4_0: CERTIFIED.** The strongest identity row in the
matrix to date: tokenizer differential **0/721** (the new `dbrx` → llama3
split-rule mapping is exact), greedy identity vs llama.cpp b10353 **6/6
byte-identical** including both 256-token runs, chat clean (`"4"`, clean
stop through the new granite template), CPU↔Metal byte-identical at fixture
scale. The one blemish: cpu_cuda scored 11/12 — a single near-tie flip on
the Swedish prompt at 64 tokens, investigated to the floor rather than
recorded raw: fixture differentials (f32 AND q4_0-quantized, with llama and
muse controls) are byte-identical on CUDA, the model's own f16→q8-KV floor
is 14/16, and token-divergence vs the reference classifies **0 of 5
divergences as real** (all at ties, max Δlogprob 0.19). The flip is the
model's own numerical floor showing through, not an engine defect.

**granite-4.1-30b Q8_0: CERTIFIED-WITH-CAVEAT.** Tokenizer 0/721, chat
clean, cpu_cuda 6/6 on the Q3_K_S sibling. Greedy identity 3/6 — the same
class as the certified muse row (3/6) and gemma-26B precedent: the 64-layer
depth amplifies cross-engine accumulation differences, cross-engine
disagreement (9/16 in tie-classified runs on the 3-bit sibling) exceeds the
self-floor (2/16) but every divergence magnitude stays ≤ 0.19 nats — an
order of magnitude below afmoe's failing 1.87 — and both tails stay fluent.
Numerical tie-breaking at depth, not wrong math; the 8B's perfect score on
the identical code is the strongest evidence for that reading.

**granite-4.1-30b Q3_K_S: FAILS the 16 GB-tier quality bar.** KLD vs the
Q8_0 baseline over 400 positions: top-1 **71.0%**, mean KLD **0.314**,
top-8 overlap 0.753 (zero-point self-check exact: 0.0 / 100%). This is the
sub-4-bit wall measured on a THIRD model family, and worse than muse's
3-bit tier (81%/0.13–0.15). Engine correctness on the artifact is fine
(cpu_cuda 6/6, divergences floor-consistent); the artifact itself is too
damaged to certify for quality. **The 16 GB tier's answer for Granite is
the same as the plan predicted: the 8B IS the pruned 30B, done by IBM with
training compute — certify the 8B, skip 30B-at-3-bit.**

## Gate table

| gate | 8b Q4_0 | 30b Q3_K_S | 30b Q8_0 |
|---|---|---|---|
| tokenizer differential | 0/721 | 0/721 | (same tokenizer) |
| greedy vs b10353 | **6/6** | 1/6 (3-bit near-tie soup; see KLD) | 3/6 |
| cpu_cuda (12 rows both models) | 11/12, flip at floor | 6/6 | — |
| chat smoke | `"4"`, stop | `"4"`, stop | — |
| self-floor (f16→q8 KV) | 14/16 | 14/16 | — |
| tokdiv vs reference (tie bar 0.25) | 0 real / 5 ties | 3 real / 6 ties, max Δ 0.185 | — |
| KLD vs Q8_0 | — | **71.0% / 0.314 — FAIL** | baseline |

## Notes

- The M1-tier claim was additionally spot-checked on a real 8 GB M1: correct
  output ("The capital of France is Paris. The capital of Germany is
  Berlin…"); the 5.06 GB file fits the ~5.5 GB envelope on paper, but the
  test machine had ~3 GB available at the time, so Metal correctly declined
  and no steady-state tok/s is claimed — that number needs an idle M1.
- The cpu_cuda sweep initially ran without `RUNNER_CUDA_TC=0`; the failing
  row was re-run pinned to the scalar path before any conclusion was drawn
  (it still flipped, which is what triggered the floor protocol).
- Granite tool calling (the Hermes-style `<tool_call>` JSON in the model's
  template) is not rendered — unimplemented, not approximated. The
  JSON-schema constrained tool path works as with every arch.
