# Truncation recovery — Runner vs vLLM, granite-4.1-3b, 2026-08-19

Same box, same model, same tool schema, same prompt, same budgets. The
head-to-head for the "tool calls survive the token limit" headline. Method,
recipe, and the full serve flags are in
[`docs/truncation-benchmark.md`](../../../../docs/truncation-benchmark.md).

| runtime | version | artifact | backend | truncated rungs (1..16) | rung 64 (control) |
|---|---|---|---|---|---|
| Runner | 0.1.19-alpha (`c7d6c3d`) | granite-4.1-3b **Q4_K_M GGUF** | CPU | parseable `tool_calls`, `finish_reason: length`, no leak | completes (`tool_calls`) |
| vLLM   | 0.27.1 + xgrammar 0.2.3 | granite-4.1-3b **HF safetensors** | CUDA (sm_120 MIG) | **no `tool_calls`**, `finish_reason: length`, hermes framing leaks into `content` | completes (`tool_calls`) |

`report.json` in each subdirectory is the raw probe output: per-rung
`finish_reason`, `tool_calls` presence, JSON-parseability, `content` leakage,
and the base64 of every HTTP response body.

## Read before quoting

- **Ladder:** `max_tokens` ∈ {1, 2, 3, 5, 8, 16, 64}. 64 is the control rung;
  both engines complete there, which is what proves the smaller rungs measure
  truncation and not a misconfiguration that would fail at every budget.
- **Not the same backend.** Runner ran on CPU and vLLM on the GPU, because vLLM
  is GPU-first. The verdict is tool-call survival, not throughput, so the
  backend should not change it — but it is a difference and it is stated rather
  than buried.
- **Not the same weights format.** vLLM served the fp16 HF checkpoint; Runner
  served the Q4_K_M GGUF of the same model. Quantisation could move an argument
  *value*; it does not decide whether a parseable call is produced.
- **What the vLLM cell measures.** vLLM was run with
  `--enable-auto-tool-choice --tool-call-parser hermes` and `tool_choice:
  "required"` — a correctly configured tool-calling server, not a crippled one.
  The hermes parser only yields a `tool_calls` object once the closing framing
  has been generated, so a budget that ends first leaves the partial call in
  `content`. That is the failure the headline is about, reproduced.
- **SGLang** is not included; its column is intentionally absent, not "failed".
- Deterministic (`temperature: 0`): re-running reproduces these verdicts.

Refresh the vLLM column with the recipe in the doc; refresh the Runner column,
or gate a build, with `make test-truncation`.
