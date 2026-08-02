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

## Anomalies worth a follow-up

Not faults — every one of these models is correct — but performance shapes that
do not follow from size alone. All reproduced across both passes.

- **Llama-3.1-8B and Qwen3-8B are the same 4.9/5.0 GB at full offload, and
  Llama decodes twice as fast** (62.4 vs 31.8 tok/s). Qwen3 adds per-head QK
  norms and a larger vocabulary; neither obviously accounts for 2×.
- **Ministral-8B Q5_K_M (35.1) against Llama-3.1-8B Q5_K_M (56.8)** — same
  quant, same file size, same `llama` code path, 1.6× apart.
- **Devstral-Small (1.6) and Qwen3.6-27B (1.2) sit far below the other
  partially-offloaded models**, including MoE models twice their size. Both are
  dense and large, so every layer's full weight moves each token; the MoE models
  move only their active experts.

## Context and KV edges (`scripts/stress-context.py`)

Probed on the five models that fully offload, since the edges are the same code
path for all of them.

- `-c 0` (auto-fit) resolves to 4096 on every model and states the KV cost.
- `-c 1000000` is **refused with the number that made it impossible** —
  `cannot allocate buffers (ctx 1000000 needs 131072.0 MB KV cache)` — and
  exits non-zero. The suite plan's open item on large-context KV budgeting
  describes this as "failing opaquely"; on this build it does not.
- One real usability gap: **`-c 32768` is accepted silently and costs ~8× decode**
  (Llama-3.1-8B: 66.5 → 8.6 tok/s), because a 4.3 GB KV cache evicts the weights
  that were fitting in VRAM. The engine prints the KV size but never connects it
  to the slowdown.

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

## Caveat on the absolute numbers

Comparisons within this document are sound: one tool, one settings set, one
idle machine. Two warnings on the absolute figures.

Runner's two measurement paths disagree. On Ministral-8B, `--bench-json`
reports ~35 tok/s decode while the same binary driven through `--serve` and
measured off the streaming response reported 55.6. The obvious explanation —
that a short `-n` amortises warmup over too few tokens — was tested and is
**wrong** (`-n 8` gives 31.0, `-n 64` gives 32.5–36.0, nowhere near 1.7×). It is
recorded unexplained rather than explained away. Treat `--bench-json` as an
internally consistent ranking, not a quotable throughput, and do not compare it
against another engine; `docs/bench-2026-08-01-3070.md` has cross-engine figures
measured through the streaming path for both engines.

And the first pass of this document is the standing reminder that a contended
box produces numbers that look perfectly plausible and are simply wrong.
