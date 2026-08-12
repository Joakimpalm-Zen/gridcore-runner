# Quant-vs-tool-call-fidelity harness

Suite plan item P2 ("quant-vs-tool-call-fidelity"). Measures how
tool-calling / structured-output fidelity degrades across the quantization
ladder of **one** model, producing a per-quant table meant to become HF
model-card content: schema conformance rate, correct tool selection rate,
argument-content agreement against the highest-precision variant, and a KLD
distribution-divergence summary.

The open question this harness exists to answer honestly: constrained
decoding may make tool calling **more** quant-robust than free-form prose —
grammar-forced output cannot become syntactically invalid the way prose can
degrade — or argument *content* may decay with bits even while the JSON
stays syntactically perfect. The harness is built to show either result. It
does not pre-commit to "this sells Q8" or to "quants are basically free" —
see the suite-wide plan's explicit warning against publishing only the
convenient story.

## What it measures, per variant

For each GGUF variant, Runner is spawned (one server at a time — variants
are never loaded concurrently) and the agent-torture request matrix
(`scripts/agent-torture.py:build_cases`, reused, not copied) runs at
temperature 0. Every case is scored against the same case run on a pinned
**reference** variant (normally the highest-precision quant in the ladder):

| Axis | What it means | Applies to |
|---|---|---|
| **Schema conformance** | Does the variant's own output validate against the tool/`response_format` JSON schema? An absolute measure — it does not need the reference to answer. | tool, tool-stream, and structured-final categories |
| **Tool selection** | Did the variant call the *same function* the reference called, for the identical request? | tool and tool-stream categories |
| **Argument agreement** | Do the parsed argument/content documents match the reference's *exactly* (`==` on the decoded JSON)? Requires tool selection to already be correct. | tool, tool-stream, and structured-final categories |

`stream_normalization` (plain prose, no schema) is still executed — it
keeps the streaming transport path exercised — but never enters these three
denominators; there is no schema or tool call to score there.

Alongside the tool matrix, the harness runs a KLD / top-1 / top-8
next-token-distribution comparison against the reference, reusing
`scripts/kld-compare-raw.py`'s raw-`/v1/completions` protocol (sound across
quants of the same model — no chat template is applied on either side, so
what's measured is the weights, not template rendering).

## Zero-point self-validation

Before measuring any variant, the **reference model is run through the
whole pipeline twice** — two independent server spawns of the same file —
and the second run is scored against the first with the *exact same
comparator* used for every other variant. Two greedy (temperature 0) runs
of identical weights must:

- agree on tool selection and argument content for every case (rate 1.0),
- score `mean_kld == 0.0` (within a tight epsilon) and `top1_agreement_pct
  == 100.0` on the KLD lane,
- leave no case or KLD position unexecuted on either run.

If any of that is not exact, the harness refuses to measure a single
variant and exits with status 2. This is not a formality: a broken
comparator, a nondeterministic decode path, or a flaky spawn would make
every downstream "quant X disagrees with the reference at rate Y" number
meaningless, and the whole point of this harness is that the numbers are
trustworthy enough to publish.

When a variant in `--variant` resolves to the same file as `--reference`
(by path, not label), the harness reuses the zero-point run's second pass
as that variant's measurement instead of loading the model a third time —
useful when the reference is deliberately kept in its own ladder as a
sanity row, and required by the smoke test below, which only has one file.

## Honest incompleteness

Every scenario that cannot run or cannot be scored is recorded, never
silently dropped — the same RNR-007 rule `scripts/compat_matrix.py` follows:

- A transport-level failure on a case (connection refused, timeout, a dead
  server) marks that case `executed: false` with the reason, and it appears
  in the variant's `tool_fidelity_not_executed` list.
- A case that ran fine on the variant but whose reference case failed is
  marked `not comparable` with an explicit reason — it is not silently
  treated as a pass or dropped from the denominator.
- A KLD corpus position that fails to query is recorded in
  `kld_not_executed` with its position index and reason, not folded into a
  bare failure count.
- A variant whose model file is missing gets a `status: "not_executed"` row
  with a reason — it still appears in `variants[]` and in the rendered
  table.

`report["complete"]` is `true` only when the zero-point pass passed **and**
every variant executed with zero not-executed entries on both lanes. A
report can still be useful and non-`complete` (e.g. one quant's file
wasn't available on this box yet) — completeness is a separate, checkable
field, never implied by the presence of numbers.

## Run it

```bash
# zero-point only: reference and sole variant are the same file
python3 scripts/quant-fidelity.py \
    --reference q4_0=models/granite-4.1-8b-Q4_0.gguf \
    --variant   q4_0=models/granite-4.1-8b-Q4_0.gguf \
    --cases 16 --kld-positions 20 \
    --out tests/quant-fidelity/out/smoke

# a real ladder: one reference, several quants (repeat --variant)
python3 scripts/quant-fidelity.py \
    --reference q8_0=models/granite-4.1-8b-Q8_0.gguf \
    --variant q8_0=models/granite-4.1-8b-Q8_0.gguf \
    --variant q4_k_m=models/granite-4.1-8b-Q4_K_M.gguf \
    --variant q4_0=models/granite-4.1-8b-Q4_0.gguf \
    --variant q2_k=models/granite-4.1-8b-Q2_K.gguf \
    --cases 120 --kld-positions 400 \
    --out tests/quant-fidelity/out/granite-ladder
```

`--reference` and `--variant` both take `LABEL=PATH`. `--cases N` is the
tool-scenario request budget per pass (agent-torture's `build_cases(N)`,
round-robin across its eight categories — pick a multiple of 8 to keep
categories balanced). `--kld-positions N` / `--kld-stride N` control the
raw-completion KLD lane over `--kld-corpus` (default:
`tests/fixtures/mixed-corpus.txt`).

Output under `--out`: `report.json` (schema `xyntetik.quant-fidelity.v1`)
and a rendered `report.md` table. Evidence is append-only — a second run to
the same `--out` refuses unless `--force`.

### Shared-box etiquette

Before every server spawn the harness waits, printing status every 15s,
until `pgrep -f "runner -m"` shows no *other* runner process — a
memory-constrained shared box (e.g. an 8 GB M1) cannot hold two model loads
at once, so the harness never contends with a concurrent job for RAM.
Disable with `--skip-idle-wait` on a box where that is not a concern.

## Cost shape for a full ladder

Each pass (one reference gold run, one reference check run for zero-point,
then one run per non-reference variant) is a full model load plus `--cases`
tool requests plus `--kld-positions` raw completions, entirely sequential.
For the suite plan's first target — granite-4.1-8b, 15 official quants in
one IBM repo — that is 16 total passes (2 for zero-point + 14 additional
variants, since the reference's own row reuses the zero-point check run) at
whatever `--cases`/`--kld-positions` the run budget allows. This is sized
for the Blackwell box, not the shared 8 GB M1: the smoke run in this commit
intentionally used small budgets (16 cases, 20 KLD positions) to fit the
memory-constrained box it validated on.

## What the smoke run proved, what a full run still needs

The committed smoke run (`--reference` and sole `--variant` both
`granite-4.1-8b-Q4_0.gguf`) exercises the zero-point path end to end
against a real spawned server and a real 8B model — it does not exercise
quant-to-quant divergence (there is only one file). A full ladder run
needs, beyond what is already built:

- the other 14 granite-4.1-8b quants downloaded onto the measurement box,
- a larger `--cases`/`--kld-positions` budget than the smoke defaults (the
  suite plan's statistics-honesty note: 15 requests/category is thin for
  small deltas),
- the Blackwell box's toolchain/measurement conventions (this harness
  builds and runs Runner the same way `kld-compare-raw.py` and
  `compat_matrix.py` already do — no new build step is required),
- a decision on how many `--variant` entries to run in one invocation vs.
  one report per quant plus a separate aggregation pass — the harness
  supports either (repeat `--variant`, or run it N times to N `--out`
  directories and diff `report.json` files).

## First measured ladder — granite-4.1-3b, 2026-08-14

Reference Q8_0, 16 agent-torture requests at temperature 0, zero-point
self-check PASSED (two independent reference spawns agreed on every tool
call at 0 KLD before anything was measured).

| variant | schema conformance | tool selection | argument agreement | mean KLD |
|---|---:|---:|---:|---:|
| q8_0 (reference) | 100.0% | 100.0% | 100.0% | 0 |
| q6_k | 100.0% | 100.0% | 64.3% | 0.0129 |
| q5_k_m | 100.0% | 100.0% | 57.1% | 0.0327 |
| q4_k_m | 100.0% | 100.0% | 57.1% | 0.1271 |
| q4_0 | 100.0% | 100.0% | 50.0% | 0.1398 |

The finding this table carries: **constrained decoding guarantees the
shape of a tool call at any quantization, not its contents.** Schema
conformance and tool selection hold at 100% down to Q4_0; argument
agreement decays monotonically to 50%. For an agent loop that is the
difference between a crash and a wrong action — an improvement, and not a
substitute for bits. Raw report:
`xyntetik-suite/docs/plans/decisions-evidence-2026-08-14/`.

## 2026-08-15: reproduced on a second family, and a size threshold appears

Hermes-4-14B (bartowski's ladder, all hashes verified against the source repo),
same harness, zero point PASSED:

| variant | schema conformance | tool selection | argument agreement | mean KLD |
|---|---|---|---|---|
| q8_0 (reference) | 100.0% | 100.0% | 100.0% | 0 |
| q6_k | **100.0%** | **100.0%** | 78.6% | 0.0155 |
| q5_k_m | **100.0%** | **100.0%** | 78.6% | 0.0131 |
| q4_k_m | **100.0%** | **100.0%** | 64.3% | 0.0205 |
| q4_0 | **100.0%** | **100.0%** | 50.0% | 0.2775 |

The split holds on a second model family and gets sharper at 14B: shape and
function choice survive every rung including legacy Q4_0, argument content
decays to half. The strongest single row is Hermes q4_0, whose whole-model
fidelity fails hard (mean KLD 0.1881 over 400 positions) while every tool call
it emits is still schema-valid and still routed to the right function.

### Whole-model fidelity has a size threshold, and it is not where "4-bit is
### unshippable" suggested

Same bar (margin-qualified top-1 >= 97% AND mean KLD <= 0.05), same protocol,
400 positions, each variant against its own Q8_0:

| model at Q4_K_M | params | mean KLD | verdict |
|---|---|---|---|
| gemma-4-E2B | ~5B (MoE, 2B active) | 0.3624 | FAIL |
| granite-4.1-3b | 3B | 0.1376 | FAIL |
| Phi-4-mini | 3.8B | 0.1258 | FAIL |
| **Qwen3-8B** | **8B** | **0.0419** | **PASS** |
| **Hermes-4-14B** | **14B** | **0.0279** | **PASS** |

Monotone in parameter count, with the crossover between roughly 4B and 8B. The
usable rule is therefore not "never ship 4-bit" but **4-bit k-quants are viable
from about 8B upward and are not viable below it** on everything measured here.
The legacy-Q4_0 rule is unchanged and gets sharper: on the same 14B model, Q4_0
scores 0.1881 against Q4_K_M's 0.0279, a 6.7x worse divergence to save 0.46 GB.

### A harness result worth keeping

Qwen3-8B could not be measured by this harness at all: its **zero point failed
twice**, on a loaded box and again on an idle one with a 600 s timeout, with two
greedy runs of the identical reference weights disagreeing on 10 of 16 cases.
The harness refused to measure any variant rather than publish numbers derived
from a nondeterministic reference. A plain tool call at temp 0 reproduces fine
across three runs; the disagreement is confined to the stress categories
(forced truncation, stream normalization, large enums, reasoning-then-tool).
Cause undetermined, recorded as open.
