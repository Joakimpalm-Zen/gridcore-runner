# Model-shelf stress pass — RTX 3070 box, 2026-08-02

Every GGUF resident on this machine (19 files, ~150 GB) run against Runner
`7555e84` by `scripts/stress-models.py`. Per-model JSON is committed under
`tests/compatibility/out/stress/`.

Machine: Intel i7-7700K (4c/8t), 15.9 GB RAM, RTX 3070 8 GB (driver 596.36),
Windows 11. Context 1024, greedy, `--bench-json` for throughput.

## Result: no engine faults

| | |
|---|---|
| Models that loaded and generated | 18 / 19 |
| Refused (correctly) | 1 — `ibm-granite` (`unsupported architecture`) |
| GPU fallbacks, kernel-launch failures, crashes, timeouts | **0** |
| Models where greedy CUDA output ≠ host output | **0** |
| Degenerate/empty output on the GPU path | 0 |

Granite's refusal is the no-footguns contract working, not a fault: the
architecture would load under llama-style math and produce plausible wrong
output, so the engine declines it by name.

**Every model that ran was byte-identical between `--gpu auto` and `--gpu off`**,
including all three MoE families and both partial- and full-offload splits.
That is the strongest single result here — it covers a set far wider than the
pinned compatibility matrix, at splits the matrix never exercises.

## One bug, and it is bigger than first filed

**`--cpu-moe auto` fails on every MoE model on this machine — 3 of 3.** Each
run reports two faults (`kernel launch failed`, then `forward failed at
runtime — releasing the backend`) and continues on the host:

| Model | `auto` placed | Result |
|---|---|---|
| gpt-oss-20b MXFP4 | 14/24 expert banks | **fault**, fell back to CPU |
| gemma-4-26B-A4B Q4_0 | 13/30 expert banks | **fault**, fell back to CPU |
| Qwen3-30B-A3B Q4_K_M | 17/48 expert banks | **fault**, fell back to CPU |

Bare `--cpu-moe` (all experts on the host) is clean on all three — 0 faults —
so the defect is specifically the *auto fit choosing how many banks to place*,
not the device expert path and not the new MXFP4 kernels. It was filed on
2026-08-01 against gpt-oss alone; this pass shows it is architecture-independent
and deterministic, which makes it a bug in the fit's headroom calculation
rather than anything model-specific.

The user-visible damage is worse than the fault count suggests: `auto` is the
mode the split banner *recommends* when a bare `--cpu-moe` leaves VRAM unused,
so following the engine's own advice is what triggers it. And the fallback is
near-silent — output still appears, just at host speed.

Measured cost when it happens, decode tok/s:

| Model | `--cpu-moe auto` | bare `--cpu-moe` | plain `--gpu auto` |
|---|---:|---:|---:|
| gpt-oss-20b | 0.194 (faulted) | 0.196 | 0.362 |
| gemma-4-26B-A4B | 5.63 (faulted) | 4.82 | 4.66 |
| Qwen3-30B-A3B | 5.24 (faulted) | 5.24 | 5.54 |

## Optimal settings for this machine

Two rules cover the whole shelf, and both follow from the 8 GB card.

**1. There is a hard cliff at ~6 GB of weights.** Anything that fully offloads
runs 25–78 tok/s; anything that does not runs 1–7 tok/s. On this box that
boundary — not parameter count — is what decides whether a model is usable
interactively.

| Fully offloaded (`full=1`) | decode tok/s | | Partially offloaded (`full=0`) | decode tok/s |
|---|---:|---|---|---:|
| Phi-4-mini Q8_0 (4.1 GB) | 77.6 | | gemma-4-12B QAT XL (6.7 GB) | 5.7 |
| Llama-3.1-8B Q4_K_M (4.9 GB) | 62.4 | | gemma-3-12B Q4_K_M (7.3 GB) | 7.2 |
| Llama-3.1-8B Q5_K_M (5.7 GB) | 56.2 | | gemma-4-12B Q4_K_M (7.4 GB) | 6.6 |
| Ministral-8B Q5_K_M (5.7 GB) | 34.3 | | Qwen2.5-Coder-14B (9.0 GB) | 5.5 |
| Qwen3-4B Q8_0 (4.3 GB) | 34.5 | | Qwen3-14B Q4_K_M (9.0 GB) | 5.2 |
| Qwen3-8B Q4_K_M (5.0 GB) | 31.7 | | Qwen3-30B-A3B (18.6 GB) | 5.5 |
| gemma-4-E4B Q4_K_M (5.0 GB) | 25.4 | | Devstral-Small (14.3 GB) | 1.6 |

**2. `--kv q8` iff the model does not fully fit.** It buys VRAM back, which
buys offloaded layers, which is worth far more than the KV precision costs:

| Model | f16 | q8 | change |
|---|---:|---:|---:|
| gemma-3-12B Q4_K_M | 4.9 | 7.2 | **+49%** |
| gemma-4-12B Q4_K_M | 4.9 | 6.6 | **+35%** |
| Qwen2.5-Coder-14B | 4.5 | 5.5 | +21% |
| Qwen3-14B | 4.8 | 5.2 | +10% |
| Phi-4-mini (fits) | 77.6 | 75.4 | −3% |
| Llama-3.1-8B Q4_K_M (fits) | 62.4 | 61.0 | −2% |

For a model that already fits, q8 KV costs a couple of percent and buys only
context headroom — so it is the right default only when a long context is
actually wanted.

One exception worth naming: **Qwen3.5-9B Q8_0 gets *slower* with q8 KV**
(3.0 → 2.6). It is the hybrid recurrent (`qwen35`) architecture, whose
recurrent state is not KV cache, so q8 shrinks a smaller share of its
footprint while still paying the quantisation cost on the attention layers.

## Recommended launch line per model

Fastest configuration measured that neither faulted nor deviated. Where two
configurations were within a couple of percent the simpler one is preferred, so
`--kv q8` appears only where it actually earned it.

| Model | GB | Offload | Recommended | decode tok/s |
|---|---:|---|---|---:|
| `microsoft_Phi-4-mini-instruct-Q8_0` | 4.1 | full | `runner -m <model>` | 77.6 |
| `Meta-Llama-3.1-8B-Instruct-Q4_K_M` | 4.9 | full | `runner -m <model>` | 62.4 |
| `Meta-Llama-3.1-8B-Instruct-Q5_K_M` | 5.7 | full | `runner -m <model>` | 56.2 |
| `Ministral-8B-Instruct-2410-Q5_K_M` | 5.7 | full | `runner -m <model>` | 34.3 |
| `Qwen_Qwen3-4B-Q8_0` | 4.3 | full | `runner -m <model>` | 34.1 |
| `Qwen3-8B-Q4_K_M` | 5.0 | full | `runner -m <model>` | 31.7 |
| `gemma-4-E4B-it-Q4_K_M` | 5.0 | full | `runner -m <model>` | 25.1 |
| `google_gemma-3-12b-it-Q4_K_M` | 7.3 | partial | `runner -m <model> --kv q8` | 7.2 |
| `gemma-4-12B-it-Q4_K_M` | 7.4 | partial | `runner -m <model> --kv q8` | 6.6 |
| `gemma-4-12B-it-qat-UD-Q4_K_XL` | 6.7 | partial | `runner -m <model> --kv q8` | 5.7 |
| `Qwen2.5-Coder-14B-Instruct-Q4_K_M` | 9.0 | partial | `runner -m <model> --kv q8` | 5.5 |
| `Qwen3-30B-A3B-Q4_K_M` | 18.6 | partial | `runner -m <model>` (**not** `--cpu-moe auto`) | 5.5 |
| `Qwen3-14B-Q4_K_M` | 9.0 | partial | `runner -m <model> --kv q8` | 5.2 |
| `gemma-4-26B-A4B-it-Q4_0` | 14.4 | partial | `runner -m <model>` (**not** `--cpu-moe auto`) | 4.7 |
| `Qwen_Qwen3.5-9B-Q8_0` | 9.8 | partial | `runner -m <model>` (q8 KV is slower here) | 3.0 |
| `mistralai_Devstral-Small-2507-Q4_K_M` | 14.3 | partial | `runner -m <model>` | 1.6 |
| `Qwen3.6-27B-Q4_K_M` | 16.8 | partial | `runner -m <model>` | 1.2 |
| `gpt-oss-20b-MXFP4` | 12.1 | partial | `runner -m <model>` (**not** `--cpu-moe auto`) | 0.4 |
| `ibm-granite_granite-3.3-8b-instruct-Q8_0` | 8.7 | — | refused by design | — |

For the three MoE models the plain layer split beats every `--cpu-moe`
variant that runs, and `auto` must be avoided entirely until the fit bug above
is fixed. That inverts the advice the split banner currently prints.

## Anomalies worth a follow-up

These are not faults — every one of these models is correct — but they are
performance shapes that do not follow from size alone.

- **Llama-3.1-8B and Qwen3-8B are the same 4.9/5.0 GB at full offload, and
  Llama decodes twice as fast** (62.4 vs 31.7 tok/s). Qwen3 adds per-head QK
  norms and a larger vocabulary, but neither obviously accounts for 2×.
- **Ministral-8B Q5_K_M (34.3) against Llama-3.1-8B Q5_K_M (56.2)** — same
  quant, same file size, same `llama` code path, 1.6× apart.
- **gemma-4-E4B is the slowest of the fully-offloaded models** (25.4 tok/s)
  despite being one of the smallest. Its per-layer-embedding pre-pass runs on
  the host once per forward by design; this quantifies what that costs.

## Context and KV edges (`scripts/stress-context.py`)

Probed on the five models that fully offload, since the edges are the same code
path for all of them.

- `-c 0` (auto-fit) resolves to 4096 on every model and states the KV cost.
- `-c 1000000` is **refused with the number that made it impossible** —
  `cannot allocate buffers (ctx 1000000 needs 131072.0 MB KV cache)` — and
  exits non-zero. The suite plan's open item on large-context KV budgeting
  describes this as "failing opaquely"; on this build it does not. What it does
  is refuse rather than auto-cap, which is the correct half of that item's ask,
  so the remaining work is narrower than the item currently reads.
- One real usability gap: **`-c 32768` is accepted silently and costs ~8× decode**
  (Llama-3.1-8B: 66.5 → 8.6 tok/s), because a 4.3 GB KV cache evicts the weights
  that were fitting in VRAM. The engine prints the KV size but never connects it
  to the slowdown, so the operator sees an inexplicably slow model and no
  suggestion that the context request caused it. Naming that trade at load —
  the split banner already reports layer placement — would cost one line.

## Caveat on the absolute numbers — and an unresolved discrepancy

Comparisons *within* this table are sound: every row was measured the same way,
same tool, same settings. The absolute figures need one warning.

Runner's two measurement paths disagree with each other. On Ministral-8B,
`--bench-json` reports **36.0 tok/s** decode (64 tokens in 1.78 s, from its own
`gen_s`), while the same binary driven through `--serve` and measured off the
streaming response reported **55.6 tok/s** yesterday on the same card and file.

The obvious explanation — that a short `-n` amortises graph-capture warmup over
too few tokens — was tested and is **wrong**: `-n 8` gives 31.0 tok/s and
`-n 64` gives 32.5–36.0, a spread of a few percent, nowhere near 1.7×. So the
gap is something else, and it is recorded here unexplained rather than
explained away. Candidates not yet separated: the streaming derivation's window
(`(n-1)/(total − ttft)`) may exclude time that `gen_s` includes, or the two
paths may differ in how a decode step is timed.

Until that is resolved, treat `--bench-json` numbers as an internally
consistent ranking rather than a quotable throughput, and do not compare them
against another engine. The cross-engine figures in
`docs/bench-2026-08-01-3070.md` were measured through the streaming path for
both engines, so they are self-consistent for that purpose.
