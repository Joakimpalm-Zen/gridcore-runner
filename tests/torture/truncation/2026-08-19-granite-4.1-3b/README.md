# Truncation recovery — Runner vs vLLM vs llama.cpp vs Ollama, granite-4.1-3b, 2026-08-19

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
| SGLang   | 0.5.17 | — | — | *environment-blocked, unmeasured* — see below | — |

`report.json` in each subdirectory is the raw probe output: per-rung
`finish_reason`, `tool_calls` presence, JSON-parseability, `content` leakage,
and the base64 of every HTTP response body. (No `sglang/` directory: it was
never kept alive on this box — the cell is deliberately empty, not a failure
verdict.)

## The verdict, rung by rung

At every budget below the control, exactly one engine hands the client an
**executable** tool call: Runner. Each competitor fails, in its own way, and the
three ways are worth distinguishing because they rank differently for a caller:

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

Ranked by how a caller experiences it: executable call (Runner) > detectable
empty/error response (vLLM empty list, Ollama 500) > protocol rendered as prose
(vLLM/llama.cpp `content` leak) > a `tool_calls` object whose arguments silently
fail to parse (llama.cpp at 16). None of the three competitors closes the
document; Runner is the only engine whose truncated rungs are all parseable
calls.

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
- **SGLang, unmeasured (environment-blocked).** SGLang 0.5.17 could not be kept
  alive on this box's MIG 1g.24gb slice. torch's caching allocator NVML-asserts
  (`NVML_SUCCESS == r INTERNAL ASSERT FAILED` in `CUDACachingAllocator.cpp`) and
  reports `avail mem=2.83 GB` on a 24 GB idle slice; disabling the caching
  allocator (`PYTORCH_NO_CUDA_MEMORY_CACHING=1`) clears the assert but OOMs
  during weight allocation. This was retried across `--mem-fraction-static`
  0.45/0.55/0.70, `--context-length` 2048, `expandable_segments:True`, and
  `SGLANG_DISABLE_NVML=1`; all die before the server accepts a request. This is
  an environment limitation, **not** a truncation-test result — its behaviour
  is unknown, hence the empty cell.
- Deterministic (`temperature: 0`): re-running reproduces these verdicts.
- **One redaction in `llamacpp/report.json`.** llama.cpp echoes the served
  model's on-disk path in each response's `model` field; that path was replaced
  with the model name `granite-4.1-3b` in the committed base64 bodies so no host
  filesystem layout is published. Nothing else in the bodies was touched, and
  the truncation verdict does not depend on the `model` string.

Refresh the vLLM/llama.cpp/Ollama columns with the recipes in the doc; refresh
the Runner column, or gate a build, with `make test-truncation`.
