# The agent torture suite

A standalone, adversarial agent-conformance suite. It runs one repeatable
request matrix that stresses the things real agent harnesses break on — nested
tool arguments, forced truncation mid-call, tool selection under pressure, and
transport-invariant streaming — and reports a verdict per request with enough
preserved evidence to audit every one.

It is **runtime-agnostic**: the same matrix runs against Runner or any other
OpenAI-compatible server (llama.cpp's server, Ollama, vLLM), so results
compare directly on identical hardware. The bar is not "did it answer" — it is
"did it emit *exactly one* valid tool call, select the *right* tool, produce
arguments that satisfy the schema, and stream in a way that does not depend on
how the bytes were chunked."

## What it tests

| Category | What breaks without it |
|---|---|
| `nested_arguments` | Deep object/array/enum argument schemas — the shape most tool-call parsers mangle. |
| `tool_selection` | The right tool chosen from several, under a forced `tool_choice`. |
| `forced_truncation` | A hard token ceiling landing *inside* a tool call — the arguments must still be valid JSON. |
| `stream_normalization` | SSE that normalizes to the same result regardless of TCP segmentation, ends with `[DONE]`, and carries a real finish reason. |

## Run it

```bash
# Runner (spawned locally on the CPU, deterministic):
python3 scripts/agent-torture.py --model path/to/model.gguf --cases 100

# Any OpenAI-compatible server already listening on localhost:
python3 scripts/agent-torture.py --endpoint 127.0.0.1:8080 \
    --runtime llama.cpp --runtime-version b3200 --model qwen2.5-7b --cases 100
python3 scripts/agent-torture.py --endpoint 127.0.0.1:11434 \
    --runtime ollama --model qwen2.5 --cases 100
python3 scripts/agent-torture.py --endpoint 127.0.0.1:8000 \
    --runtime vllm --model Qwen/Qwen2.5-7B-Instruct --cases 100
```

`--endpoint` is loopback-only on purpose: the runtimes under comparison run on
the same box (identical hardware is the whole point). Tunnel a remote runtime
to a local port first (`ssh -L 8080:127.0.0.1:8080 host`).

### Starting the other runtimes

```bash
# llama.cpp
llama-server -m model.gguf --host 127.0.0.1 --port 8080 --jinja

# Ollama (its OpenAI-compatible surface is on /v1)
ollama serve                # then: ollama pull qwen2.5 ; it serves on 11434

# vLLM
vllm serve Qwen/Qwen2.5-7B-Instruct --host 127.0.0.1 --port 8000
```

## What you get

Two files under `--out` (default `tests/torture/out/`):

- `report.json` — runtime name + version, model, per-category pass/fail
  totals, `valid_structured_tasks_per_second`, elapsed, and peak RSS (Runner
  only — a foreign process's RSS is not read). Labeled with the runtime, so
  two runtimes' reports diff directly.
- `raw.jsonl` — one line per request: the exact request body, the raw response
  (base64, so nothing is lost or reinterpreted), the normalized stream where
  applicable, and the failure category on a miss. Every verdict is auditable
  from this file alone.

## Reproduce and compare

The matrix is deterministic: `build_cases(N)` returns the same requests every
run, round-robin across categories, so a diff between two `report.json` files
is a diff between two runtimes — not two random samples. To publish a
comparison, run the same `--cases N` against each runtime on the same machine
and commit all four `out/` directories side by side.

Ollama routes on the OpenAI `model` field, so pass `--model-name <name>` (the
name you gave `ollama create`); Runner and llama.cpp ignore it. The flag is
recorded in `raw.jsonl` so the request stays reproducible.

## Published comparisons

- [2026-07-21 — Runner vs llama.cpp vs Ollama, Llama-3.2-3B, CPU](../tests/torture/results/2026-07-21-llama-3.2-3b-cpu/README.md):
  Runner 12/12, llama.cpp 5/12, Ollama 5/12. The split is exactly the schema
  cases — Runner wins `nested_arguments` and `forced_truncation` 3/3 vs 0/3;
  the rest are close. (llama.cpp is far faster on raw CPU throughput — that is
  the other axis, and the readout is honest about it.)

## Submit a result

Bring your nastiest tool schema and your hardware. Open an
[agent-torture result](../.github/ISSUE_TEMPLATE/agent-torture-result.yml)
issue with the runtime, model, quant, hardware, your `report.json` totals, and
— if something failed — the offending `raw.jsonl` line. Verified submissions
are published in the comparison table.

## Copy-ready client examples

The suite speaks the OpenAI tool-call contract, so these clients target Runner
(or any runtime) unchanged — point the base URL at the server.

**OpenAI Python SDK**

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="not-needed")
resp = client.chat.completions.create(
    model="local",
    messages=[{"role": "user", "content": "dispatch a fast job for /a and /b"}],
    tools=[{"type": "function", "function": {
        "name": "dispatch_job",
        "parameters": {"type": "object", "additionalProperties": False,
            "required": ["job"], "properties": {"job": {"type": "object",
                "additionalProperties": False, "required": ["mode", "targets"],
                "properties": {"mode": {"enum": ["fast", "safe"]},
                    "targets": {"type": "array", "items": {"type": "object",
                        "additionalProperties": False,
                        "required": ["path", "retries"],
                        "properties": {"path": {"type": "string"},
                                       "retries": {"type": "integer"}}}}}}}}}}],
    tool_choice={"type": "function", "function": {"name": "dispatch_job"}},
    temperature=0)
print(resp.choices[0].message.tool_calls[0].function.arguments)
```

**Vercel AI SDK (TypeScript)**

```ts
import { createOpenAI } from "@ai-sdk/openai";
import { generateText, tool } from "ai";
import { z } from "zod";

const runtime = createOpenAI({ baseURL: "http://127.0.0.1:8080/v1", apiKey: "x" });
const { toolCalls } = await generateText({
  model: runtime("local"),
  temperature: 0,
  tools: { dispatch_job: tool({
    parameters: z.object({ job: z.object({
      mode: z.enum(["fast", "safe"]),
      targets: z.array(z.object({ path: z.string(), retries: z.number().int() })),
    }) }) }) },
  toolChoice: { type: "tool", toolName: "dispatch_job" },
  prompt: "dispatch a fast job for /a and /b",
});
console.log(toolCalls[0].args);
```

**curl**

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "local", "temperature": 0,
  "messages": [{"role": "user", "content": "dispatch a fast job for /a and /b"}],
  "tools": [{"type": "function", "function": {"name": "dispatch_job",
    "parameters": {"type": "object", "required": ["job"], "additionalProperties": false,
      "properties": {"job": {"type": "object"}}}}}],
  "tool_choice": {"type": "function", "function": {"name": "dispatch_job"}}
}'
```

Codex, Cline, OpenCode, and Claude-compatible clients all take a base URL and
an OpenAI-shaped tool schema — the same request works; only the base URL
changes.
