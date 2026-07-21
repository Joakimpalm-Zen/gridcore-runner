# Agent torture: Runner vs llama.cpp vs Ollama — Llama-3.2-3B, CPU

First real cross-runtime run of the agent torture suite (2026-07-21). Same
model, same quant, same request matrix, same box, all on CPU.

## Setup

- **Model:** Llama-3.2-3B-Instruct-Q4_K_M (the *same* GGUF loaded into all
  three runtimes).
- **Hardware:** one box, CPU only (an NVIDIA MIG slice was present but
  disabled for every runtime, so this is an apples-to-apples CPU comparison).
- **Matrix:** `agent-torture.py --cases 12` — 3 per category, deterministic.
- **Runtimes:** Runner (loopback serve, `--gpu off`), llama.cpp `llama-server`
  b10076 (`-ngl 0 --jinja`), Ollama 0.32.1 (OpenAI `/v1` surface, CPU).
- Reproduce: `agent-torture.py --endpoint 127.0.0.1:PORT --runtime NAME
  --model-name M --cases 12 --out results/<runtime>` against each server.
  The full request and raw response for every case are in each
  `<runtime>/raw.jsonl`.

## Result

| category | Runner | llama.cpp | Ollama |
|---|---|---|---|
| `nested_arguments` (deep object/array/enum args) | **3/3** | 0/3 | 0/3 |
| `tool_selection` (right tool under forced choice) | **3/3** | 2/3 | 2/3 |
| `forced_truncation` (budget dies mid-call) | **3/3** | 0/3 | 0/3 |
| `stream_normalization` (transport-invariant SSE) | 3/3 | 3/3 | 3/3 |
| **total valid** | **12/12** | 5/12 | 5/12 |

## What it shows

The split is exactly where Runner's thesis lives, and nowhere else:

- **`nested_arguments` — 3/3 vs 0/3.** With a deep nested tool-argument schema,
  Runner's schema-driven sampling produces arguments that satisfy the schema
  every time. On the same model, llama.cpp and Ollama emit arguments that
  don't (`schema` failures) — the model, left to free-generate, mangles the
  nested shape.
- **`forced_truncation` — 3/3 vs 0/3.** When the token budget dies *inside* the
  tool call, Runner minimally completes valid JSON (its always-parses-on-length
  contract); the others return truncated, unparseable JSON (`protocol`
  failures).
- **`tool_selection` and `stream_normalization`** are close or identical — no
  runtime has an edge where schema enforcement isn't the deciding factor. That
  is the honest boundary of the claim: Runner is not "better at everything," it
  is *structurally correct* where free generation is structurally fragile.

## The honest caveat: raw speed

Runner is **slower on CPU here** — 111s for the 12 cases versus llama.cpp's
5.7s and Ollama's 8.2s. Two things inflate that gap and one is real:

- Runner ran at its **default 8 threads**, untuned for this 128-core box, while
  llama.cpp defaults to physical-core parallelism. Much of the gap is that
  config difference, not engine speed.
- Runner also *did more work*: it completed 12 valid structured generations;
  llama.cpp/Ollama failed 7 of theirs fast (a truncated/invalid call returns
  quickly). "valid structured tasks/second" divides passes by wall time, so it
  is not a like-for-like latency number.
- The residual is real and expected: **llama.cpp wins on raw CPU throughput** —
  Runner's own README says so. This suite measures the *other* axis:
  conformance. On that axis, at this model size, Runner is the only runtime
  that produces valid structured output for the adversarial cases.

A GPU run and a thread-matched CPU run are the obvious next comparisons
(runner Phase 10).
