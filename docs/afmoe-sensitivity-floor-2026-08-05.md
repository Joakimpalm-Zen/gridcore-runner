# afmoe (Trinity-Nano) sensitivity-floor run — 2026-08-05

Addendum to `docs/afmoe-cert-report-2026-08-05.md`. This is the measurement
that decides whether gate 3's byte-identity failure is *unachievable* (the
model is numerically chaotic, so token identity is not a sound gate — the
Apertus / gemma-4-26B / gpt-oss precedent) or *real* (the gap exceeds what the
model's own numerics do to it).

## Verdict

**Trinity-Nano does NOT sit at its sensitivity floor. The floor argument does
not license omitting `greedy_reference` here, and it does not explain gate 3.**

Cross-engine disagreement exceeds the model's own numerical instability on
both axes the program uses, and half the divergences are not near-ties:

| measurement | prompts diverging (of 16) | max logprob delta | mean logprob delta |
|---|---:|---:|---:|
| **runner vs itself** (f16 KV vs q8_0 KV) — the floor | 5 | 0.5833 | 0.02428 |
| **llama.cpp vs itself** (cold vs warm prefix cache) | 0 | 0.1026 | 0.00477 |
| **runner vs llama.cpp b10280** — the cross-engine gap | **6** (3 tie, **3 real**) | **1.8668** | — |

All three at 16 prompts x 16 tokens, Q8_0, CPU, so the rows are directly
comparable. Evidence:
`tests/compatibility/out/sensitivity-afmoe-trinity-2026-08-05.json`,
`…-reference-2026-08-05.json`, `divergence-afmoe-trinity-2026-08-05.json`.

### Why this is the opposite of the precedents

The rule the program applies (`scripts/sensitivity_floor.py` docstring, and
`docs/compatibility-program.md`): *if a model disagrees with itself on N
prompts, no cross-engine comparison can be expected to do better than N.*
Models are excused from `greedy_reference` when the cross-engine number is at
or below their own floor:

| model | own floor | cross-engine | at floor? |
|---|---|---|---|
| Apertus 8B Q4_K_M | 0.4596 nats max | 0.4148 nats max | yes → omitted |
| gemma-4-26B-A4B Q4_0 | 11/16 prompts | 9/16 prompts | yes → omitted |
| **Trinity-Nano Q8_0** | **5/16, 0.5833 nats** | **6/16, 1.8668 nats** | **no** |

Trinity is the first of these where the cross-engine gap is *larger* than the
self-perturbation gap — more prompts (6 vs 5) and 3.2x the peak logprob
movement (1.8668 vs 0.5833). The reference is also very stable on this model
(0/16 under its own perturbation), so the instability is not coming from
llama.cpp's side.

### The three "real" divergences

`token_divergence.py` calls a divergence a tie only when **both** engines'
gaps between the two contender tokens are ≤ 0.25 nats; otherwise at least one
engine was reasonably confident and it counts as real. Three of six qualify:

| prompt | runner gap | ref gap | runner picked | ref picked |
|---|---:|---:|---|---|
| `The three laws of thermodynamics are` | 0.0225 | **0.3369** | `' Zer'` | `' The'` |
| `The quick brown fox` | **0.4012** | 0.0668 | `'.'` | `'."'` |
| `for i in range(10):\n    print(` | 0.0190 | **0.4012** | `'10'` | `'2'` |

Two of the three share a shape worth noting: the **runner is nearly tied
(0.019–0.023 nats) exactly where llama.cpp is comfortably decided
(0.34–0.40 nats)**. That is not symmetric numerical noise — it is the runner's
distribution being flatter at positions the reference finds easy. The third
case is the reverse. This asymmetry is the most specific lead the run produced.

### A separate signal: token agreement hides the largest numeric gap

The single largest logprob delta in the whole cross-engine run, **1.8668
nats**, is on `"Water boils at"` — a prompt where **both engines produced
identical tokens for all 16 positions**. Every actually-divergent prompt had a
max delta of ≤ 0.184. So token identity and numeric agreement are coming apart
in both directions here, which is a further argument against reading gate 3's
text comparison as the whole story.

## This revises the cert report's hypothesis

`docs/afmoe-cert-report-2026-08-05.md` concluded from two hand-inspected
divergence points (margins 0.124 and 0.216 nats, both near-ties) that the
failure "looks like small numerical differences tipping argmax at genuinely
close calls, not a structurally wrong implementation," while recording the
FAIL. **With 16 prompts instead of 2 sample points, that hypothesis does not
hold up**: half the divergences are not ties, and the cross-engine gap is
larger than the model's own floor. The cert report's verdict (FAIL) stands
unchanged; its *explanation* was too generous and should be read as superseded
by this addendum. The two points it inspected happened to be the near-tie kind.

## What this means for the merge decision

The floor run does **not** clear afmoe for merge-with-caveat on the
"identity is unachievable" basis — that basis is measurably absent. Options,
for the owner:

1. **Triage the 3 real divergences first.** The flatter-runner-distribution
   asymmetry above is a concrete lead, and the afmoe-specific math (gated
   attention, sigmoid routing with selection bias) is where to look. A
   logit-level dump at those three positions would settle it quickly.
2. **Merge with a narrower, honest caveat** that does *not* invoke the
   sensitivity floor: afmoe carries `load` and `tokenizer` (both measured:
   0/721 tokenizer, model loads and generates fluently), omits
   `greedy_reference` because it *fails* rather than because it is
   unachievable, and omits `chat`/`cpu_cuda` which were never run (gates 4-5
   stopped). That is a defensible manifest, but it is a different claim from
   the Apertus/gemma-4 one and should not reuse their wording.
3. **Do not** add a `greedy_reference` claim, and do not describe afmoe in the
   README as "at its sensitivity floor" — the measurement says it is not.

Note the manifest currently has no Trinity/afmoe entry at all, so nothing is
presently over-claimed; this is about what the entry should say when added.

## Reproduction

```sh
LS=~/workspace/Gridcore/lcpp-bin/llama-b10280/llama-server
M=~/workspace/models/trinity-nano/Trinity-Nano-Preview-Q8_0.gguf

python3 scripts/sensitivity_floor.py --model $M --mode runner    --runner ./runner --tokens 16
python3 scripts/sensitivity_floor.py --model $M --mode reference --reference $LS   --tokens 16
python3 scripts/token_divergence.py  --model $M --reference $LS --runner ./runner --tokens 16
```

`token_divergence.py` exits non-zero when it finds real divergences; that is
the gate verdict, not a crash (stderr was empty).

Environment identical to the cert report: runner `0.1.7-alpha` at branch head,
llama.cpp **b10280 (61881b1f7)**, Q8_0 sha256
`5fcc2428c325e8b2035d280ff28c4c659ea240ce317a9536f230acfc352b79d1`, CPU only.

One incidental correction to the cert report: it flagged llama-server's prompt
cache as having made the reference unstable at 64/256 tokens. Under this run's
own perturbation mode the reference was perfectly stable (0/16 at 16 tokens),
and both house tools already pass `cache_prompt: false`. The instability I hit
during gate 3 came from the cert doc's hand-rolled protocol, which does not —
the house tooling was already correct on that point.
