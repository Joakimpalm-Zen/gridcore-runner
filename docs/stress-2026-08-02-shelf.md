# Model-shelf stress pass — RTX 3070 box, 2026-08-02

Every GGUF resident on this machine (19 files, ~150 GB) run against Runner
`4748830` by `scripts/stress-models.py`. Per-model JSON is committed under
`tests/compatibility/out/stress/`.

Machine: Intel i7-7700K (4c/8t), 15.9 GB RAM, RTX 3070 8 GB (driver 596.36),
Windows 11. Context 1024, greedy, `--bench-json` for throughput.

**These numbers are from the second pass**, run on an otherwise idle machine
after the `--cpu-moe auto` fix. The first pass ran while other work competed for
the box, and several of its figures were wrong by up to 2.5× (Qwen3.5-9B read
3.0 tok/s there against 7.5 here). Where the two disagree, this one is the
measurement, and the conclusions the first pass supported have been re-derived
rather than carried over.

## Result: no engine faults

| | |
|---|---|
| Models that loaded and generated | 18 / 19 |
| Refused (correctly) | 1 — `ibm-granite` (`unsupported architecture`) |
| GPU fallbacks, kernel-launch failures, crashes, timeouts | **0** |
| Models where greedy CUDA output ≠ host output | **0** |
| Configurations swept | 30 (default, `--kv q8`, both `--cpu-moe` modes) |

Granite's refusal is the no-footguns contract working, not a fault: the
architecture would load under llama-style math and produce plausible wrong
output, so the engine declines it by name.

**Every model that ran was byte-identical between `--gpu auto` and `--gpu off`**,
including all three MoE families and both partial- and full-offload splits.
That is the strongest single result here — it covers a set far wider than the
pinned compatibility matrix, at splits the matrix never exercises.

`--cpu-moe auto`, which faulted on all three MoE models in the first pass, is
clean on all three here. The fix is described at the end.

## Optimal settings for this machine

### 1. The VRAM cliff at ~6 GB of weights

Anything that fully offloads runs 24–77 tok/s; anything that does not runs
1–11. On this box that boundary — not parameter count — decides whether a model
is usable interactively.

| Fully offloaded | decode tok/s | | Partially offloaded | decode tok/s |
|---|---:|---|---|---:|
| Phi-4-mini Q8_0 (4.1 GB) | 76.5 | | gemma-4-26B-A4B (14.4 GB) | 10.8 |
| Llama-3.1-8B Q4_K_M (4.9 GB) | 62.4 | | Qwen3-30B-A3B (18.6 GB) | 9.1 |
| Llama-3.1-8B Q5_K_M (5.7 GB) | 56.8 | | Qwen3.5-9B Q8_0 (9.8 GB) | 7.6 |
| Ministral-8B Q5_K_M (5.7 GB) | 35.1 | | gemma-3-12B Q4_K_M (7.3 GB) | 7.5 |
| Qwen3-4B Q8_0 (4.3 GB) | 34.4 | | gemma-4-12B Q4_K_M (7.4 GB) | 6.3 |
| Qwen3-8B Q4_K_M (5.0 GB) | 31.8 | | Qwen3-14B Q4_K_M (9.0 GB) | 5.2 |
| gemma-4-E4B Q4_K_M (5.0 GB) | 26.8 | | Devstral-Small (14.3 GB) | 1.6 |

The two MoE models at the top of the right-hand column are the interesting
result: a 14.4 GB and an 18.6 GB model now decode faster than several 7–9 GB
dense ones, because only their active experts have to move.

### 2. `--cpu-moe auto` for every MoE model

The largest single lever on the shelf, and only usable at all since the fix
below.

| Model | default | `--cpu-moe auto` | bare `--cpu-moe` |
|---|---:|---:|---:|
| gemma-4-26B-A4B Q4_0 | 5.9 | **10.8** (1.8×) | 5.2 |
| Qwen3-30B-A3B Q4_K_M | 5.6 | **9.1** (1.6×) | 3.8 |
| gpt-oss-20b MXFP4 | 0.37 | 0.38 | 0.20 |

gpt-oss does not benefit: at 12.1 GB the fit keeps 13 of 24 expert banks either
way. Bare `--cpu-moe` is the worst option in every case — the flag is only
worth using in its `auto` form.

### 3. `--kv q8` is model-specific, not a rule

The first pass suggested a clean rule ("take it whenever the model does not
fully fit"). On idle hardware the effect is smaller and does not generalise:

| Model | f16 | q8 | change |
|---|---:|---:|---:|
| gemma-3-12B Q4_K_M | 5.5 | 7.5 | **+36%** |
| gemma-4-12B Q4_K_M | 5.4 | 6.3 | +17% |
| gemma-4-E4B Q4_K_M | 23.9 | 26.8 | +12% |
| Qwen3-14B Q4_K_M | 4.8 | 5.2 | +8% |
| Qwen3.5-9B Q8_0 | 7.5 | 7.6 | +1% |
| Qwen2.5-Coder-14B | 4.9 | 4.9 | 0% |
| gemma-4-12B QAT XL | 5.6 | 5.4 | −4% |
| Llama-3.1-8B Q4_K_M (fits) | 62.4 | 59.9 | −4% |

Worth trying on a partially-offloaded model, worth measuring rather than
assuming, and a small loss on one that already fits. The gemma family gains
most, which is consistent with its large per-layer KV.

The first pass also reported Qwen3.5-9B as an exception where q8 KV was
*slower* (3.0 → 2.6) and blamed the hybrid recurrent architecture. **That was a
contention artifact** — on an idle box it is 7.5 → 7.6. The architectural
explanation was reasoning from a bad number, and is withdrawn.

## Recommended launch line per model

| Model | GB | Offload | Recommended | decode tok/s |
|---|---:|---|---|---:|
| `microsoft_Phi-4-mini-instruct-Q8_0` | 4.1 | full | `runner -m <model>` | 76.5 |
| `Meta-Llama-3.1-8B-Instruct-Q4_K_M` | 4.9 | full | `runner -m <model>` | 62.4 |
| `Meta-Llama-3.1-8B-Instruct-Q5_K_M` | 5.7 | full | `runner -m <model>` | 56.8 |
| `Ministral-8B-Instruct-2410-Q5_K_M` | 5.7 | full | `runner -m <model>` | 35.1 |
| `Qwen_Qwen3-4B-Q8_0` | 4.3 | full | `runner -m <model>` | 34.4 |
| `Qwen3-8B-Q4_K_M` | 5.0 | full | `runner -m <model>` | 31.8 |
| `gemma-4-E4B-it-Q4_K_M` | 5.0 | full | `runner -m <model> --kv q8` | 26.8 |
| `gemma-4-26B-A4B-it-Q4_0` | 14.4 | partial | `runner -m <model> --cpu-moe auto` | **10.8** |
| `Qwen3-30B-A3B-Q4_K_M` | 18.6 | partial | `runner -m <model> --cpu-moe auto` | **9.1** |
| `Qwen_Qwen3.5-9B-Q8_0` | 9.8 | partial | `runner -m <model>` | 7.6 |
| `google_gemma-3-12b-it-Q4_K_M` | 7.3 | partial | `runner -m <model> --kv q8` | 7.5 |
| `gemma-4-12B-it-Q4_K_M` | 7.4 | partial | `runner -m <model> --kv q8` | 6.3 |
| `gemma-4-12B-it-qat-UD-Q4_K_XL` | 6.7 | partial | `runner -m <model>` | 5.6 |
| `Qwen3-14B-Q4_K_M` | 9.0 | partial | `runner -m <model> --kv q8` | 5.2 |
| `Qwen2.5-Coder-14B-Instruct-Q4_K_M` | 9.0 | partial | `runner -m <model>` | 4.9 |
| `mistralai_Devstral-Small-2507-Q4_K_M` | 14.3 | partial | `runner -m <model>` | 1.6 |
| `Qwen3.6-27B-Q4_K_M` | 16.8 | partial | `runner -m <model>` | 1.2 |
| `gpt-oss-20b-MXFP4` | 12.1 | partial | `runner -m <model> --cpu-moe auto` | 0.38 |
| `ibm-granite_granite-3.3-8b-instruct-Q8_0` | 8.7 | — | refused by design | — |

## Anomalies — two of three RESOLVED, and they were not the engine

- ~~**Llama-3.1-8B decodes twice as fast as Qwen3-8B at the same size**
  (62.4 vs 31.8).~~ **Explained: the sampler, not the model.** At `--temp 0`
  they are 62.9 and 60.2 — the same, within noise. Qwen3's preset carries
  `top_p 0.95`/`top_k 20`, which on this model's flat distribution lands in the
  full-vocabulary sort; Llama-3.1's `top_p 0.9` does not. My speculation about
  per-head QK norms and vocabulary size was reasoning from a confounded number.
- ~~**Ministral-8B Q5_K_M (35.1) against Llama-3.1-8B Q5_K_M (56.8)**, same
  quant, same size, same code path.~~ **Same cause.** At `--temp 0` they are
  56.4 and 57.1. Mistral's preset ships `top_p 1.0`, the slowest configuration
  (see the sampler section below).
- **Still open — Devstral-Small (1.6) and Qwen3.6-27B (1.2) sit far below the
  other partially-offloaded models**, including MoE models twice their size.
  This one is not the sampler: both are dense and large, so every layer's full
  weight moves each token, where the MoE models move only their active experts.

The lesson generalises past these two rows: **a throughput table built with
each model's own preset is comparing samplers as well as models.** Anything
cross-model in this document should be re-read with that in mind.

## Context and KV edges (`scripts/stress-context.py`)

Probed on the five models that fully offload, since the edges are the same code
path for all of them.

- `-c 0` (auto-fit) resolves to 4096 on every model and states the KV cost.
- `-c 1000000` is **refused with the number that made it impossible** —
  `cannot allocate buffers (ctx 1000000 needs 131072.0 MB KV cache)` — and
  exits non-zero. The suite plan's open item on large-context KV budgeting
  describes this as "failing opaquely"; on this build it does not.
- ~~One real usability gap: **`-c 32768` is accepted silently and costs ~8× decode**
  (Llama-3.1-8B: 66.5 → 8.6 tok/s), because a 4.3 GB KV cache evicts the weights
  that were fitting in VRAM. The engine prints the KV size but never connects it
  to the slowdown.~~ **Closed the next day by `2633d0b`**, which prints a note
  naming the KV size, how many layers it displaced and two remedies. The
  measurement above still stands; only the silence is gone. Re-verified in
  [`stress-2026-08-04-f27e7bb.md`](stress-2026-08-04-f27e7bb.md).

## The fix this pass validates

The first pass found `--cpu-moe auto` faulting on all three MoE models. The
symptom looked like a VRAM over-commit; it was not one. Adding the missing
diagnostic made the engine say so in one line:

```
gpu: tensor 'output.weight' is not resident on the device (no binding covers it)
```

Two decisions disagreed about who owns the output projection. `full` — whether
`output.weight` is uploaded — is `G == n_layer` **and** the output tensor still
fitting the budget. `partial` — whether the device computes logits — is the
layer count **alone**. The auto fit places attention for every layer, then
spends whatever budget survives on expert banks; when the last of it went to a
bank, `output.weight` was dropped while `G` still equalled `n_layer`, so the
forward asked the device for logits and looked up a tensor that was not there.
`--cpu-moe 20` "worked" only because placing fewer banks left room for it by
accident, which is also why it was slower.

The fit now holds back what a full split still owes before placing banks, with
an invariant guard that hands the last layer to the CPU if that reserve cannot
be met. Reserving is the better trade on merit too: the output projection runs
every token over the whole vocabulary, an expert bank only when its layer is
routed to.

## Caveat on the absolute numbers — RESOLVED

An earlier version of this document recorded a 1.7x disagreement between
Runner's two measurement paths as unexplained. It is explained, and it was not
a measurement bug: **the two runs used different samplers.**

`--bench-json` with no `--temp` uses the model's vendor sampling preset. The
cross-engine benchmarks and the `--serve` measurement sent `temperature: 0`.
With the sampler matched, the two paths agree to within noise:

| Ministral-8B Q5_K_M, same 625-token prompt, `-c 1024`, `-n 64` | decode tok/s |
|---|---:|
| `--serve`, greedy, derived from the stream | 55.9 |
| `--serve`, greedy, first-to-last-token window | 56.8 |
| `--bench-json --temp 0` | **56.7** |
| `--bench-json`, default preset | 34.2 |

So both numbers were right about what they measured, and the earlier "warmup"
hypothesis was correctly rejected — it just was not the alternative either.

**The tables above therefore include per-model sampler cost**, because they
were produced with each model's own preset. That is a defensible thing to
measure (it is what a user gets by default), but it is not a like-for-like
engine comparison across models, and two entries in the anomalies section were
wrong because of it. Cross-engine figures in
`docs/bench-2026-08-01-3070.md` are greedy on both sides and unaffected.

## The sampler finding this turned up

Chasing the above produced a real optimisation target. Decode with the default
preset against decode at `--temp 0`, same prompt and context:

| Model | preset | greedy | preset cost |
|---|---:|---:|---:|
| Qwen3-8B Q4_K_M | 31.4 | 60.2 | **−48%** |
| Ministral-8B Q5_K_M | 34.7 | 56.4 | **−38%** |
| Llama-3.1-8B Q5_K_M | 55.9 | 57.1 | −2% |
| Llama-3.1-8B Q4_K_M | 61.3 | 62.9 | −3% |
| Phi-4-mini Q8_0 | 76.9 | 77.2 | 0% |

**Correction — the mechanism I first published was wrong for these models.**
The isolation run that produced it left `min_p` at the preset's value, and the
models measured do not use the presets I named. The accurate picture, from
instrumenting the sampler (`RUNNER_SAMPLE_STATS=1`):

There are **two** ways the large-vocabulary fast path is missed, and they hit
different configurations.

1. **No filter at all** — `top_k 0`, `top_p 1.0`. `sample.c` guarded the fast
   path with `(want_k < n_vocab || top_p < 1.0f)`, so the case asking for the
   least work was the one case excluded, and full-sorted the vocabulary every
   token. **Fixed** (see below): 30.6 → 59.3 tok/s, a 1.94x speedup, output
   identical at fixed seed. This is the shape the `mistral` and `gpt-oss`
   presets request.

2. **`top_k` set, but the head is smaller than `k`** — still open. With
   `top_k 40`, the first threshold (`p >= p_max/1024`) yields only ~11
   candidates *carrying 99% of the mass*, which fails the `m >= want_k` test.
   The loosening schedule then jumps by `e^-27` and overflows the 4096-entry
   head cap, so it full-sorts anyway. Trace from Ministral-8B:
   `[smp short m=11 mass=0.9908 need=0.9500][smp overflow m=4096]`.
   This cannot be fixed by relaxing the criterion — `pick_scaled` renormalises
   over exactly the top `k`, so dropping to 11 would change every probability.
   It needs a selection algorithm (quickselect for the true top-k) rather than
   threshold probing.

**Case 2 is the one most models hit**, because `generic` (top_k 40), `qwen3`
(20) and `gemma3` (64) all set `top_k`. That is what the 38-48% preset-vs-greedy
gap in the table above actually measures — not `top_p 1.0`, which those models
never request. Ministral-8B resolves to the `generic` preset, not `mistral`,
which is itself worth a look.

### The fix for case 1, and why it is exact

The head is now sized from the roulette draw itself. `pick_scaled` walks the
candidates in descending probability and stops at the first whose cumulative
mass reaches `r`, so **any head carrying more than `r` of the total provably
contains that token** — no filtering knob required. `r` is drawn up front and
handed to whichever path runs, so the RNG still advances exactly once per call
and a seeded run is unchanged. If the double-precision mass check and the
float walk disagree at the very edge, the walk reports it and the code retreats
to the exact full sort rather than returning the head's last token.

`min_p` gets a head criterion too, and a simpler one: min-p keeps everything
within a fixed ratio of the best token, which is a *logit* threshold, so a head
cut at or below `log(min_p)` contains the whole surviving set whatever mass it
carries.

Verified two ways. `tests/test_sampler.c` pins determinism (same seed, same
token) and that the sampled distribution still tracks the full-vocabulary
softmax — if the head were sized too small the tail would be truncated and
common tokens over-represented. And end-to-end against the pre-change binary at
fixed seeds 1/7/42/1234/99999 on Ministral-8B, Qwen3-8B and Llama-3.1-8B, both
in the accelerated configurations and in the ones that must not change:
byte-identical everywhere.
