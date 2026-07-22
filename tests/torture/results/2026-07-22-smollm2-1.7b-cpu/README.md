# Agent torture: Runner vs llama.cpp — SmolLM2-1.7B, CPU

Second cross-runtime run of the agent torture suite (2026-07-22), on a
deliberately small model to test where the wedge goes as the model weakens.
Same model, same quant, same request matrix, same box, all on CPU.

## Setup

- **Model:** SmolLM2-1.7B-Instruct-Q4_K_M (the *same* GGUF loaded into both
  runtimes).
- **Hardware:** one box, CPU only (GPU present but off for both runtimes —
  apples-to-apples CPU).
- **Matrix:** `agent-torture.py --cases 12` — 3 per category, deterministic.
- **Runtimes:** Runner (loopback serve, `--gpu off`), llama.cpp `llama-server`
  b10076 (`-ngl 0 --jinja`).
- Reproduce: `agent-torture.py --endpoint 127.0.0.1:PORT --runtime NAME
  --model SmolLM2-1.7B-Q4_K_M --cases 12 --out results/<runtime>` against each
  server. Every request and raw response is in each `<runtime>/raw.jsonl`.

## Result

| category | Runner | llama.cpp |
|---|---|---|
| `nested_arguments` (deep object/array/enum args) | **3/3** | 0/3 |
| `tool_selection` (right tool under forced choice) | **3/3** | 0/3 |
| `forced_truncation` (budget dies mid-call) | **3/3** | 0/3 |
| `stream_normalization` (transport-invariant SSE) | 3/3 | 3/3 |
| **total valid** | **12/12** | 3/12 |

All 9 llama.cpp failures are the same kind: **`protocol` — "expected exactly one
tool call, got None."** The model emitted no parseable tool call at all.

## What it shows — and the honest mechanism

This is a **wider** split than the [Llama-3.2-3B run](../2026-07-21-llama-3.2-3b-cpu/README.md)
(there llama.cpp scored 5/12 and *did* get `tool_selection` 2/3). The reason is
the important part, and it is not "Runner is smarter":

- **llama.cpp's tool calling runs through the model's Jinja chat template.**
  SmolLM2-1.7B, free-generating under that path, did not produce a tool call the
  OpenAI surface could parse for *any* of the three tool-call categories — it
  returned no `tool_calls`. A small model without a strong tool-call template has
  nothing for the template path to lean on.
- **Runner's tool calling is schema-constrained sampling, not template-driven.**
  It forces a call that satisfies the declared schema regardless of whether the
  model has a tool template at all. So the same 1.7B model produces 12/12 valid
  structured calls.

So the honest claim is **template-independence**: Runner makes schema-correct
tool calling work on models whose chat template a Jinja tool path cannot drive.
That is a real capability difference, and it grows exactly as the model gets
smaller/weaker — but it is a *mechanism* difference (constrained sampling vs
template + free generation), not a claim that Runner's model reasons better.
`stream_normalization` is 3/3 for both: where schema enforcement is not the
deciding factor, the runtimes agree.

## The honest caveat: raw speed

Runner is **much slower on CPU here** — 100.5s for the 12 cases versus
llama.cpp's 3.9s. The same two things as the Llama run inflate the gap:

- Runner ran at its **default 8 threads**, untuned for this many-core box, while
  llama.cpp defaults to physical-core parallelism. Much of the gap is config.
- Runner also *did more work*: it completed 12 valid structured generations;
  llama.cpp returned no tool call fast (nothing to generate returns quickly), so
  its "valid tasks/second" (0.761 vs Runner's 0.119) divides passes by wall time
  and is not a like-for-like latency number.

llama.cpp wins on raw CPU throughput — a different axis, stated plainly. The
point of this run is the tool-call correctness column, and there the small model
makes the boundary sharper than the 3B did.
