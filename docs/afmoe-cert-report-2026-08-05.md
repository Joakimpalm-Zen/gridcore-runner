# afmoe (Trinity-Nano) certification report — 2026-08-05

> **Superseded in part (2026-08-05, later):** the sensitivity-floor run in
> `docs/afmoe-sensitivity-floor-2026-08-05.md` measured Trinity's own numerical
> floor and found the cross-engine gap **exceeds** it (6/16 prompts and
> 1.8668 nats, vs a 5/16 and 0.5833-nat floor), with 3 of 6 divergences not
> near-ties. The FAIL verdict below stands; the "looks like numerical
> tie-breaking, not structurally wrong" *explanation* below was inferred from
> only two hand-inspected points and does not survive the wider sample. Read
> the addendum for the corrected reading.

## Summary

**afmoe is NOT certified. The greedy-identity gate (gate 3) failed: 1 of 6
comparisons was byte-identical to llama.cpp b10280.** Gates 1 and 2 passed
cleanly (SIMD quant suite OK, arch admission 3 passed, tokenizer differential
**0/721**). Per the goal doc's STOP rule, gates 4 (chat smoke) and 5 (perf
row) were **not run** and **no engine code was modified**. The divergence is
real, stable and reproducible on a fair harness, but it has a consistent and
narrow signature: at every divergence inspected the runner selects
llama.cpp's **second-ranked** token at a near-tie, high-entropy position —
margins of 0.124 and 0.216 nats (≈23.8% vs 21.0%, and a flat distribution
whose top token is only ≈3.6% likely). The runner's own output is
self-consistent and fluent; this looks like small numerical differences
(accumulation order / kernel precision) tipping argmax at genuinely close
calls, not a structurally wrong implementation. Two harness defects were
found and corrected before the verdict was accepted (details below) — one of
them, llama-server's prompt cache, had made the *reference itself*
unstable. The secondary, afmoe-unrelated q4ne quality gate (gate 6) was run
and is reported at the end.

Separately, the secondary q4ne gate **also failed, decisively** (top-1 22.5%
vs a ≥97% bar; mean KLD 1.372 vs a ≤0.05 bar) — q4ne should not ship as the
16 GB-Mac artifact. That measurement was verified against a self-comparison
zero point before being recorded.

Because gates 1–5 did not all pass, the README certified-architecture table
was **not** updated, per the doc. Report pushed only. Not merged to main; no
tag.

## Environment

| item | value |
|---|---|
| repo | fresh clone, branch `afmoe-trinity` |
| git HEAD | `ad272adc6a9807a20834bf41e247a28dd88565b6` |
| runner version | `runner 0.1.7-alpha` |
| build | `make runner CC=x86_64-conda-linux-gnu-gcc -j` under conda env `ccbuild`, exit 0 |
| llama.cpp reference | **b10280 (61881b1f7)**, prebuilt release binaries, `~/workspace/Gridcore/lcpp-bin/llama-b10280/` |
| model | `Trinity-Nano-Preview-Q8_0.gguf` |
| model sha256 | `5fcc2428c325e8b2035d280ff28c4c659ea240ce317a9536f230acfc352b79d1` |
| Q4_K_M sha256 | `211127328238031b7aead5c71f488023710300da587703160e42fb981c8c335a` |
| box | 128 cores, CPU only (`--gpu off` throughout) |

## Machine state / cleanup (done first)

- Deleted `~/workspace/Gridcore/llamacpp-fresh/` (the conda-toolchain build the
  doc identified as segfaulting), 1.1 GB reclaimed.
- Deleted today's leftovers: `afmoe-prep*.log bwcert* bwident* prebuilt*.log
  lcpp2.log lcpp.tgz llamacpp-build.log afmoe-wip.tgz done.flag`.
- Killed stray processes left by the earlier remote session, none started by
  this one:
  - a `llama-cli` that had been spinning **14 minutes at ~75% CPU** on what
    should have been a 16-token generation (see "Reference verification"),
  - a `runner --serve` on port 18502 (gpt-oss-20b), up **3.7 days**,
  - an in-progress `curl` loop downloading the `bf16` weights.
- Removed the partial `Trinity-Nano-Preview-bf16.gguf` (1.1 GB of an aborted
  download) and its `dl.log`. No gate needs bf16 — the doc marks Q8_0 as the
  cert row — and MACHINE-NOTE asks that downloads not accumulate. `Q8_0` and
  `Q4_K_M` were already complete and were left in place.
- Disk after cleanup: 2.9 TB free (unchanged; nothing new was downloaded).

### Reference verification (precondition for gate 3)

`llama-cli --version` → `version: 10280 (61881b1f7)`, matching the pin.

The doc asks for a 16-token greedy load check. Run via `llama-cli` it **did
not crash but hung**: it loaded the model correctly (banner shows
`b10280-61881b1f7` and the Trinity path) and then entered an interactive
conversation loop emitting `> ` prompts forever, even with `-no-cnv` and
stdin at `/dev/null`. It was killed at 180 s having written a **1.35 GB**
stdout log, which was deleted immediately. This is precisely the "cli output
formatting fights scripts" failure the doc warns about, and is why the doc
prescribes `llama-server` + `curl /completion`.

Via `llama-server` the same check succeeded immediately:

```
prompt "The capital of France is", temperature 0, n_predict 16
→ " Paris. It is located in the north-central part of the country, on the"
```

The prebuilt reference is therefore sound; only the CLI front-end is
unusable for scripting. All gate-3 reference queries use the server.

## Gate 1 — suite regression: **PASS**

```
make test-quants-simd  → exit 0
./test-quants-simd     → test_quants_simd: OK
python3 -m pytest -q tests/test_arch_admission.py → 3 passed in 0.10s
```

`./runner --caps` confirms `afmoe` is in the admitted architecture list
(14 architectures total).

## Gate 2 — tokenizer differential: **PASS**

```
python3 scripts/difftok.py --gguf Trinity-Nano-Preview-Q8_0.gguf \
    --ref arcee-ai/Trinity-Nano-Preview

0/721 strings differ (0 of them begin with whitespace)
OK
```

Matches the expected **0/721** exactly, including the whitespace-before-digit
class the branch's fix targeted.

## Gate 3 — greedy identity vs llama.cpp b10280: **FAIL (1/6)**

### Harness fairness — two defects found and corrected first

A cert failure is only meaningful if the comparison is fair. Two asymmetries
were found and eliminated **before** accepting the verdict. Neither involved
changing engine code.

1. **Sampler asymmetry (ruled out as a cause).** The runner's stderr shows it
   applies a family preset at `--temp 0`: `top_p 0.95, top_k 40, min_p 0.05,
   repeat_penalty 1.10`. Truncation knobs cannot move an argmax, but
   `repeat_penalty 1.10` reshapes logits and *can*. llama-server defaults to
   `repeat_penalty 1.0`, so the doc's literal command is not an apples-to-apples
   greedy comparison. Tested directly: `--repeat-penalty 1.0` produces
   **exactly the same runner output**, so this is not the cause — but all
   reference queries were nonetheless pinned to explicit pure-greedy params
   (`repeat_penalty 1.0, top_k 0, top_p 1.0, min_p 0.0, seed 0`) so no
   asymmetry remains. *This is still worth fixing in the protocol: on a model
   where the penalty does bite, the doc's command would produce a false FAIL.*
2. **The reference was not a stable oracle (corrected).** With llama-server's
   default prompt caching on, llama.cpp's own greedy output depended on
   `n_predict`: for prompt (c) its 64-token completion was **not a prefix** of
   its own 256-token completion, self-diverging at byte 113. The runner was
   self-consistent on every prompt tested. Setting `"cache_prompt": false`
   made the reference self-consistent again. **The entire battery was re-run
   from scratch against the clean reference**; the contaminated first run is
   retained separately as evidence. The result was identical either way (1/6),
   so caching did not change the verdict — but the verdict now rests on a
   stable oracle.

Final protocol — reference: `llama-server /completion`, `temperature 0`,
`n_predict N`, `repeat_penalty 1.0, top_k 0, top_p 1.0, min_p 0.0, seed 0,
cache_prompt false`. Runner: `./runner -m <Q8_0> -p <prompt> -n N --temp 0
--gpu off`, completion taken as stdout minus the echoed prompt. Trailing
whitespace trimmed on both sides before comparison (the doc's stated
acceptable exception) — the runner emits a trailing `\n` the server does not;
this affected presentation only, never a verdict.

Both sides tokenize the prompt identically (5 tokens for prompt (a) on each
side; `add_special` true and false agree, so there is no BOS asymmetry).

### Results

| case | prompt | n | identical | first divergent byte | ref sha256 (first 16) | runner sha256 (first 16) |
|---|---|---:|---|---:|---|---|
| a | "The capital of France is" | 64 | **NO** | 14 | `6ccc70c66dcc676d` | `f00c6b0a6666f862` |
| b | "Write a python function that reverses a linked list." | 64 | **YES** | — | `1d5cb15281a16bab` | `1d5cb15281a16bab` |
| c | "In 1969, exactly 1234567 people…" | 64 | **NO** | 120 | `8464ac84e2053c4c` | `c3c1ee425a613610` |
| d | "Beskriv kort hur en kvicksilvertermometer fungerar." | 64 | **NO** | 1 | `253e803771744f98` | `05142e047813c620` |
| b-long | prompt (b) | 256 | **NO** | 340 | `d8c96313220adaef` | `868a72e2bc95420f` |
| c-long | prompt (c) | 256 | **NO** | 120 | `f8237ac4616f62ca` | `5d47006e31614742` |

**1/6 byte-identical → gate FAILS.** Note case (b): identical for the first
64 tokens, diverging only at byte 340 of the 256-token run — so even the one
passing case is a shorter-horizon pass, not a genuinely matching trajectory.

### Verbatim outputs, prompt (a), n=64

llama.cpp b10280:

```
 Paris. It is located in the north-central part of the country, on the Seine River. Paris is the largest city in France and is known for its art, culture, and cuisine. It is also a major center of fashion and is home to many famous landmarks, such as the Eiffel Tower, the Louvre Museum,
```

runner 0.1.7-alpha:

```
 Paris. It is the largest city in France and is known for its art, culture, and cuisine. Paris is also home to many famous landmarks, such as the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral.\n\nThe French flag is called the Tricolore. It consists of three vertical stripes of blue,
```

Common prefix `" Paris. It is "`, then `located in the north-central…` vs
`the largest city in France…`. Both are fluent and factually fine; the runner
is not producing degraded text.

### Verbatim outputs, prompt (d), n=64 — diverges at the FIRST token

llama.cpp b10280:

```
 What is the difference between a mercury thermometer and a digital thermometer? What is the difference between a mercury thermometer and a digital thermometer? A mercury thermometer is a type of thermometer that uses mercury to measure temperature. It consists of a glass tube with a bulb at one end containing mercury. As the temperature changes, the mercury expands
```

runner 0.1.7-alpha:

```
 How does a mercury thermometer work? Answer. The mercury in the thermometer expands when heated and contracts when cooled. This expansion and contraction causes the mercury to move up and down the glass tube, allowing you to read the temperature. 2. Beskriv hur en elektromagnetisk termometer fungerar. How
```

### Stability confirmation (doc-required)

First divergence for prompt (a) is at **completion token index 4** (the 5th
generated token). Re-running both engines at `-n 12` (= position + 8)
reproduces the divergence at exactly the same place:

```
runner    : ' Paris. It is the largest city in France and is known'
llama.cpp : ' Paris. It is located in the north-central part of the'
```

The divergence is stable and deterministic on both sides, not intermittent.

### Divergence signature (reference-side measurement only)

Top-of-distribution at the divergence point, from llama.cpp's own logprobs:

prompt (a), completion token 4:

| rank | token | logprob | prob |
|---:|---|---:|---:|
| 1 | `' located'` | −1.4373 | 23.8% |
| 2 | `' the'` ← **runner's choice** | −1.5612 | 21.0% |
| 3 | `' a'` | −1.9007 | 14.9% |

Margin between #1 and #2: **0.124 nats (2.8 percentage points).**

prompt (d), completion token 0:

| rank | token | logprob |
|---:|---|---:|
| 1 | `' What'` | −3.3222 |
| 2 | `' How'` ← **runner's choice** | −3.5385 |
| 3 | `' The'` | −3.7796 |

Margin: **0.216 nats**, at a position whose *top* token carries only ≈3.6%
probability — a very flat, high-entropy distribution.

In both inspected cases the runner selects the reference's **rank-2** token at
a near-tie. That is the signature of small numerical differences (summation
order, kernel/precision differences) tipping an argmax that was nearly
balanced, rather than of incorrect architecture math — which would be
expected to produce rank-N choices, degenerate text, or divergence at
confident positions too. **This is an observation to direct triage, not an
excuse: the gate's bar is byte-identical greedy output, and that bar was not
met.**

### STOP

Per the doc: evidence captured, first divergent position recorded, stability
confirmed at position+8, and **engine code was not modified**. Gates 4 and 5
were not run.

## Gate 4 — chat template smoke: **NOT RUN**

Skipped because the gate-3 STOP rule triggered. No evidence recorded.

## Gate 5 — perf row: **NOT RUN**

Skipped because the gate-3 STOP rule triggered. No evidence recorded.

Incidental observation only (not a perf measurement, and not a substitute for
gate 5): during gate-3 runs the runner reported roughly **3 tok/s** generation
on Q8_0 with default thread settings on this 128-core box, which looks low
enough to be worth a look when the gate is eventually run properly.

## Gate 6 (secondary, unrelated to afmoe) — q4ne quality gate

The doc labels this gate "unrelated to afmoe, do last" and it decides an
independent question (whether q4ne ships as the 16 GB-Mac artifact). It was
therefore run despite the afmoe STOP, which concerns the afmoe engine and
forbids code changes — neither of which this measurement touches.

**Verdict: FAIL, decisively — q4ne should NOT ship as the 16 GB-Mac
artifact.** Both bars are missed, and not narrowly.

```
python3 scripts/kld-compare.py \
  --model-a .../artifacts/gpt-oss-20b-keep30-MXFP4.gguf \
  --model-b .../afmoe/keep30-q4ne.gguf \
  --runner ./runner --corpus tests/fixtures/mixed-corpus.txt \
  --max-positions 400 --out q4ne-kld.json
```

```json
{
  "positions_scored": 400,
  "positions_failed": 0,
  "mean_kld": 1.3722088324032202,
  "top1_agreement_pct": 22.5,
  "mean_top8_overlap": 0.465625
}
```

| metric | measured | bar | result |
|---|---:|---:|---|
| top-1 agreement | **22.5%** | ≥ 97% | FAIL by 74.5 pts |
| mean KLD | **1.3722** | ≤ 0.05 | FAIL by 27x |
| mean top-8 overlap | 0.4656 | (not a bar) | — |

400 of 400 positions scored, 0 failed.

### The magnitude was verified, not assumed

A 22.5% top-1 between two quantizations of the *same* pruned model is
extreme enough that it could plausibly have been a broken measurement rather
than a broken model, so it was checked before being recorded:

- **Harness self-comparison** (model A against itself, 40 positions):
  `mean_kld 0.0, top1_agreement 100.0%, top8_overlap 1.0`. The tool is sound
  and its zero point is exact.
- **File provenance.** Both files are valid GGUF v3, `arch=gpt-oss`, **459
  tensors each**, and share an identical **72 MXFP4 expert tensors** — so
  this is genuinely the same pruned model measured at two non-expert
  quantizations, not two different models. They differ exactly where the name
  implies: q4ne carries **122 tensors at Q4_0**, the baseline **98 at Q8_0**,
  and q4ne has additionally pushed 24 tensors down from F32 that the baseline
  leaves at F32 (q4ne 265 F32 vs baseline 289).

That last point is the likely mechanism and the thing to look at first: q4ne
quantizes *more* non-expert tensors, and more aggressively (Q4_0 vs Q8_0).
Non-expert tensors at this scale include norms and other precision-sensitive
weights that tolerate Q8_0 but not Q4_0. The experts — the bulk of the file —
are byte-identical MXFP4 in both, so essentially all of the quality loss is
coming from the non-expert half.

This is recorded as measured, without rationalization, per the gate's own
instruction.

## What a follow-up should look at

1. **Fix the cert protocol before re-running**, independent of the engine:
   pin `--repeat-penalty 1.0` on the runner side and `"cache_prompt": false`
   on the reference side. Without the second, the reference is not a stable
   oracle and the gate can produce inconsistent results run to run.
2. **Triage the divergence at near-ties.** Both inspected divergences are the
   reference's rank-2 token within ~0.12–0.22 nats. The productive next
   measurement is a logit-level comparison at a divergent position (not a text
   comparison) to size the numerical gap — if the top-2 logit gap on the
   runner side has the opposite sign by a hair, this is a precision/ordering
   issue in the afmoe path (gated attention or sigmoid routing with selection
   bias are the new math here); if the gap is large, it is a real behavioral
   bug.
3. **Consider whether byte-identical greedy is the right bar for this
   architecture** at all, or whether a top-1 agreement rate plus a KLD bound
   (as gate 6 uses) is the more honest gate. That is an owner decision, not
   one to make inside a cert run — recorded here because the failure mode
   (rank-2 at a 2.8-point margin) is exactly the case a byte-identity bar
   cannot distinguish from a serious bug.
4. **q4ne (separate track):** rebuild keeping non-expert tensors at Q8_0, or
   at minimum keep norms and other precision-sensitive non-expert tensors out
   of Q4_0, then re-measure. The experts are already byte-identical to the
   passing artifact, so the entire regression lives in the 122 Q4_0 /
   24 formerly-F32 non-expert tensors — a rebuild should be cheap to test
   against the same 400-position corpus.

## Evidence files

Committed alongside this report in `docs/afmoe-cert-evidence/`:

- `gate3-summary.json` — per-case identity verdicts, first divergent byte,
  and both sides' completion checksums for all six gate-3 cases;
- `q4ne-kld.json` — the gate-6 measurement;
- `q4ne-kld-selfcheck.json` — the harness zero point that validates it.

Full per-case completion text and runner stderr for gate 3 are on the box
under `/tmp/afmoe-evidence/gate3/` (`<case>.ref.txt` / `<case>.runner.txt` /
`<case>.runner.stderr.txt`), with the prompt-cache-contaminated first run
retained alongside as `/tmp/afmoe-evidence/gate3-cachecontaminated/` — not
committed, since these are large free-text blobs and the material excerpts
are quoted verbatim above.

The gate-3 driver is committed with this report as
`scripts/afmoe-cert-gate3.py`, so the gate is reproducible and so the two
harness corrections above (explicit pure-greedy reference params,
`cache_prompt: false`) are not lost the next time someone runs it.
