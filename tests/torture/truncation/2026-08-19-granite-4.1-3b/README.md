# Truncation recovery — Runner vs vLLM vs llama.cpp vs Ollama vs TensorRT-LLM vs SGLang, granite-4.1-3b, 2026-08-19

Same box, same model, same tool schema, same prompt, same budgets. The
head-to-head for the "tool calls survive the token limit" headline. Method,
recipe, and the full serve flags are in
[`docs/truncation-benchmark.md`](../../../../docs/truncation-benchmark.md).

| runtime | version | artifact | backend | truncated rungs (1..16) | rung 64 (control) |
|---|---|---|---|---|---|
| Runner   | 0.1.19-alpha (`c7d6c3d`) | granite-4.1-3b **Q4_K_M GGUF** | CPU | parseable `tool_calls`, `finish_reason: length`, no leak | completes (`tool_calls`) |
| vLLM     | 0.27.1 + xgrammar 0.2.3 | granite-4.1-3b **HF safetensors** | CUDA (sm_120 MIG) | **no `tool_calls`** at every rung; hermes framing leaks into `content` | completes (`tool_calls`) |
| llama.cpp | b10488 (`9d77fa172`) | granite-4.1-3b **Q4_K_M GGUF** | CPU | **no `tool_calls`** at 1–8 (framing leaks into `content`); at 16 a `tool_calls` object with **unparseable** truncated `arguments` (`{"city": "`) | completes (`tool_calls`) |
| Ollama   | 0.32.14 | granite-4.1-3b **Q4_K_M GGUF** (imported) | GPU (CUDA MIG) | **no `tool_calls`** at 1–8 (empty `content`, no leak); at 16 the server returns **HTTP 500** (`invalid tool call arguments … unexpected end of JSON input`) | completes (`tool_calls`) |
| TensorRT-LLM `†` | 1.2.1 | **Qwen3-1.7B** HF safetensors (`†` substitute — see below) | CUDA (sm_120 MIG) | **no `tool_calls`** at any rung; bare `<tool_call>` tag leaks into `content` at rung 1, empty `content` at 2–16 (HTTP 200, no 500) | completes (`tool_calls`) |
| SGLang `†` | 0.5.17 + xgrammar 0.2.1 | **Qwen3-1.7B** HF safetensors (`†` substitute — see below) | CUDA (RTX 3070 sm_86, **WSL2**) | **no `tool_calls`** at any rung; bare `<tool_call>` tag leaks into `content` at rung 1, empty `content` at 2–16 (HTTP 200, no 500) | completes (`tool_calls`) |

`report.json` in each subdirectory is the raw probe output: per-rung
`finish_reason`, `tool_calls` presence, JSON-parseability, `content` leakage,
and the base64 of every HTTP response body. SGLang was blocked on the Blackwell
MIG slice (like the note below) and so was measured on a *different box* — the
Windows RTX 3070 under WSL2 — on the same Qwen3-1.7B substitute; its
`sglang/report.json` is that run.

## `†` TensorRT-LLM and SGLang: measured on a substitute model

TensorRT-LLM 1.2.1 **cannot serve granite-4.1-3b** — `GraniteForCausalLM` is not
in its PyTorch-backend model registry (`Unknown architecture for
AutoModelForCausalLM: GraniteForCausalLM`). Rather than leave the cell empty, the
same probe was run against a TRT-LLM-supported architecture, **Qwen3-1.7B**
(`Qwen3ForCausalLM`, matching `--tool_parser qwen3`), because the truncation
property is an *engine* guarantee, not a model-quality one. This is a
truncation-behaviour result for the engine, **not** an apples-to-apples quality
compare with the granite columns. Two further engine-level substitutions, both
disclosed: TRT-LLM 1.2.1's OpenAI server **rejects `tool_choice: "required"`
(HTTP 400)**, so `"auto"` is used (which forces the call at every budget here);
and Qwen3 is a reasoning model, so `enable_thinking: false` is set so it emits
the tool call directly (with thinking on it never reaches the JSON even at 64).
Full rationale in [`docs/truncation-benchmark.md`](../../../../docs/truncation-benchmark.md)
and in `tensorrt-llm/report.json`'s `notes` field.

**SGLang shares the same Qwen3-1.7B substitute**, for a related-but-different
reason: granite-4.1-3b is an internal checkpoint whose safetensors were not
obtainable on the box SGLang ran on, so it was pointed at the same Qwen3-1.7B the
TRT-LLM column uses — which is why the two columns are byte-identical (same model,
same Hermes-style `qwen` parser family). SGLang also needs `enable_thinking:
false`, but — unlike TRT-LLM — it **accepts `tool_choice: "required"` (HTTP 200)**,
so its column is measured with `required`, matching the granite/vLLM/Runner
columns; the `"auto"` run gives byte-identical per-rung verdicts. Full rationale
and the two-attempt history (Blackwell MIG blocked → RTX 3070/WSL2 measured) are
in [`docs/truncation-benchmark.md`](../../../../docs/truncation-benchmark.md) and
in `sglang/report.json`'s `notes` field.

## The verdict, rung by rung

At every budget below the control, exactly one engine hands the client an
**executable** tool call: Runner. Each competitor fails, and the distinct failure
modes are worth distinguishing because they rank differently for a caller
(SGLang and TensorRT-LLM share a mode — same substitute model, same parser):

- **vLLM** — no `tool_calls` object at any truncated rung, and the raw hermes
  framing (`<tool_call>\n{"name": "get_weather…`) arrives in `content` as if the
  model had said it. A naive client renders protocol as prose.
- **llama.cpp** — same framing leak into `content` at 1–8; at 16 it does emit a
  `tool_calls` object, but the `arguments` are the truncated prefix `{"city": "`
  and do **not** parse as JSON. A client that trusts `tool_calls[0].arguments`
  gets a `JSONDecodeError`.
- **Ollama** — hides the framing (empty `content`, no leak) but also produces no
  call at 1–8; at 16 its server-side hermes parser hits the unterminated JSON
  and returns **HTTP 500**, so the client gets a hard error, not a call.
- **TensorRT-LLM** `†` — no `tool_calls` object at any truncated rung; the bare
  opening `<tool_call>` tag leaks into `content` at rung 1, and `content` is empty
  at 2–16 (the `qwen3` parser suppresses the incomplete framing). Unlike Ollama it
  returns HTTP 200 with an empty callless message rather than a 500. (Measured on
  Qwen3-1.7B — see the `†` note above.)
- **SGLang** `†` — identical to the TensorRT-LLM column (same Qwen3-1.7B
  substitute, same Hermes-style `qwen` parser): no `tool_calls` at any truncated
  rung; bare `<tool_call>` tag leaks into `content` at rung 1, empty `content` at
  2–16, HTTP 200 (no 500). Measured with `tool_choice: "required"` (SGLang accepts
  it), on the RTX 3070 / WSL2 — see the `†` note above.

Ranked by how a caller experiences it: executable call (Runner) > detectable
empty/error response (vLLM empty list, Ollama 500, TRT-LLM/SGLang empty message) >
protocol rendered as prose (vLLM/llama.cpp `content` leak) > a `tool_calls`
object whose arguments silently fail to parse (llama.cpp at 16). None of the five
competitors closes the document; Runner is the only engine whose truncated rungs
are all parseable calls.

## Read before quoting

- **Ladder:** `max_tokens` ∈ {1, 2, 3, 5, 8, 16, 64}. 64 is the control rung;
  every measured engine completes there, which is what proves the smaller rungs
  measure truncation and not a misconfiguration that would fail at every budget.
- **Not the same backend.** Runner and llama.cpp ran on CPU; vLLM and Ollama on
  the GPU. The verdict is tool-call survival, not throughput, so the backend
  should not change it — but it is a difference and it is stated rather than
  buried.
- **Not the same weights format for vLLM.** vLLM served the bf16 HF checkpoint;
  Runner, llama.cpp, and Ollama served/imported the **same** Q4_K_M GGUF
  (identical bytes; Ollama imported it via `ollama create -f`). Quantisation
  could move an argument *value*; it does not decide whether a parseable call is
  produced.
- **Every competitor is correctly configured for tool calls.** vLLM ran
  `--enable-auto-tool-choice --tool-call-parser hermes`; llama.cpp ran `--jinja`
  (default) so tool calls come from Granite's own chat template; Ollama imported
  the GGUF with its autodetected template and reports the `tools` capability.
  All four were driven with `tool_choice: "required"`. These are working
  tool-calling servers, not crippled ones.
- **TensorRT-LLM, measured on a substitute model with disclosed config
  differences.** TRT-LLM 1.2.1 does not implement `GraniteForCausalLM`, so it was
  measured on Qwen3-1.7B with `--tool_parser qwen3` (the `†` note explains why
  this still measures the engine-level property). It also rejects
  `tool_choice: "required"` (HTTP 400), so `"auto"` was used — which forces the
  call at every budget here because Qwen3 emits `<tool_call>` from the first
  token; and `enable_thinking: false` was set (Qwen3 is a reasoning model and
  otherwise never reaches the JSON even at 64). These are engine-level facts about
  TRT-LLM, not a crippled configuration — but they are more caveats than the four
  granite columns carry, so the TRT-LLM cell is read as "measured, same Outcome A,
  on a substitute model" rather than as an identical granite run.
- **SGLang, measured on a second box after the Blackwell MIG blocked it.** SGLang
  0.5.17 could not be kept alive on the Blackwell MIG 1g.24gb slice the other GPU
  engines used: torch's caching allocator NVML-asserts (`NVML_SUCCESS == r INTERNAL
  ASSERT FAILED`, `avail mem=2.83 GB` on a 24 GB idle slice) and
  `PYTORCH_NO_CUDA_MEMORY_CACHING=1` clears the assert but OOMs, across
  `--mem-fraction-static` 0.45/0.55/0.70, `--context-length` 2048,
  `expandable_segments:True`, and `SGLANG_DISABLE_NVML=1`. That is a MIG-partition
  allocator bug, not a capacity limit — so it was measured on a **full** GPU with a
  normal allocator instead: the Windows RTX 3070 (sm_86, 8 GB) under WSL2, where it
  stood up and served the ladder. It is therefore on a different box and a
  different GPU than the granite columns (and on the Qwen3-1.7B substitute) — all
  stated, none buried — but it *is* a measured Outcome-A result now, not an empty
  cell.
- Deterministic (`temperature: 0`): re-running reproduces these verdicts.
- **One redaction in `llamacpp/report.json`.** llama.cpp echoes the served
  model's on-disk path in each response's `model` field; that path was replaced
  with the model name `granite-4.1-3b` in the committed base64 bodies so no host
  filesystem layout is published. Nothing else in the bodies was touched, and
  the truncation verdict does not depend on the `model` string.

Refresh the vLLM/llama.cpp/Ollama/TensorRT-LLM/SGLang columns with the recipes in
the doc; refresh the Runner column, or gate a build, with `make test-truncation`.
(`tensorrt-llm/report.json` and `sglang/report.json` needed no redaction — their
`model` field is the basename `Qwen3-1.7B` and no host path appears in any body.)
