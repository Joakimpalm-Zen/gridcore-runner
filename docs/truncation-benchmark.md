# Truncation-recovery benchmark

**Claim under test:** *tool calls survive the token limit.* When a tool call is
constrained by a JSON-schema grammar and the generation budget runs out
mid-object, Runner's closer emits the smallest schema-legal ending, so the
client still receives a **parseable `tool_calls` entry** with
`finish_reason: "length"`. The alternative — what an unconstrained emitter does
— is to leave a half-written string, or leak the tool-call framing into
`content` with no `tool_calls` object at all. An agent that receives no usable
call cannot proceed; it retries from scratch, burning tokens, time, and context.

This benchmark pins that property to an executable, someone-else-runnable
artifact and to a per-release regression gate. It drives a fixed tool schema and
prompt across a token ladder and records, per rung, exactly what the client
sees.

- Probe: [`scripts/truncation-benchmark.py`](../scripts/truncation-benchmark.py)
- Gate: `make test-truncation` (spawns Runner on the CPU fixture and asserts the
  property; see [the gate](#the-regression-gate) below)
- Committed raw results:
  [`tests/torture/truncation/2026-08-19-granite-4.1-3b/`](../tests/torture/truncation/2026-08-19-granite-4.1-3b/)

## What is pinned

Everything that could move a verdict is fixed as data in the probe, so every
engine answers the same question:

| Knob | Value |
|---|---|
| Token ladder (`max_tokens`) | 1, 2, 3, 5, 8, 16, 64 |
| Control rung | 64 (must complete, proving the small rungs measure truncation, not misconfiguration) |
| Tool | `get_weather(city: string, units: enum[celsius,fahrenheit])`, both `required`, `additionalProperties: false` |
| Prompt | `What is the weather in Paris? Use fahrenheit.` |
| `tool_choice` | `required` (forces a call at every budget) |
| `temperature` | 0 (deterministic) |
| Model | `granite-4.1-3b` — Q4_K_M GGUF for Runner, llama.cpp and Ollama; the HF safetensors of the same model for vLLM |

The property is an **engine guarantee** (grammar + closer), not model quality:
it holds identically for the random 2-layer CI fixture and for granite-4.1-3b,
which is why the gate can run on any CPU with no GPU and no competitor.

## The property, rung by rung

For every truncated rung (1, 2, 3, 5, 8, 16):

- a tool call is present, and its `arguments` parse as JSON;
- `finish_reason` is `"length"`;
- nothing leaks into `content`.

For the control rung (64): a tool call is present and parses, and it completed
(`finish_reason: "tool_calls"`).

## Results — granite-4.1-3b, one box, 2026-08-19

Same box, same model, same schema/prompt/budgets, four engines run one at a time
on the loopback port. Runner and llama.cpp served the Q4_K_M GGUF on CPU; Ollama
imported that same GGUF and served it on the GPU; vLLM served the HF safetensors
on the GPU. The verdict is tool-call survival, not throughput, so the backend
split does not move it — but it is a difference and is stated rather than buried,
as the other cross-runtime results in this tree are.

Runner at `c7d6c3d` (0.1.19-alpha), re-confirmed today building at `153cefa`;
vLLM 0.27.1 + xgrammar 0.2.3; llama.cpp b10488 (`9d77fa172`); Ollama 0.32.14.

**Per-rung, what the client receives — is there a parseable, executable
`tool_calls` entry?**

| `max_tokens` | Runner | vLLM | llama.cpp | Ollama |
|---:|:--|:--|:--|:--|
| 1  | **parses** | none; leaks `<tool_call>` into `content` | none; leaks `<tool_call>` into `content` | none; empty `content` |
| 2  | **parses** | none; leaks `<tool_call>\n` | none; leaks `<tool_call>\n` | none; empty `content` |
| 3  | **parses** | none; leaks `<tool_call>\n{"` | none; leaks `<tool_call>\n{"` | none; empty `content` |
| 5  | **parses** | none; leaks `…{"name":` | none; leaks `…{"name":` | none; empty `content` |
| 8  | **parses** | none; leaks `…{"name": "get_weather` | none; leaks `…{"name": "get_weather` | none; empty `content` |
| 16 | **parses** | none; leaks `…{"city": "` | `tool_calls` present but `arguments` = `{"city": "` **do not parse** | **HTTP 500** (`invalid tool call arguments … unexpected end of JSON input`) |
| 64 (control) | parses, `tool_calls` | parses, `tool_calls` | parses, `tool_calls` | parses, `tool_calls` |

Every measured engine completes normally at 64 (the control). At every smaller
budget **only Runner returns a parseable `tool_calls` entry.** The three
competitors each fail differently, and the differences rank differently for a
caller (executable call > detectable empty/error response > protocol rendered as
prose > a `tool_calls` object whose arguments silently fail to parse):

- **vLLM** — empty `tool_calls` list at every truncated rung; the raw hermes
  framing leaks into `content` as assistant prose.
- **llama.cpp** — same framing leak at 1–8; at 16 it emits a `tool_calls` object
  whose truncated `arguments` (`{"city": "`) are not valid JSON.
- **Ollama** — hides the framing (empty `content`, no leak) but returns no call
  at 1–8, and at 16 its server-side hermes parser fails the unterminated JSON
  with an HTTP 500.

None of the three closes the document. The full per-rung records, including the
base64 of every HTTP body, are in `vllm/report.json`, `llamacpp/report.json`,
`ollama/report.json`, and `runner-cpu/report.json` beside this note. (The Runner
results subdirectory is `runner-cpu/`, not `runner/`, because the repo's
`.gitignore` excludes the `runner` binary and would swallow a directory of that
name.)

The vLLM and Ollama cells measure their `hermes` tool-call parser path, and
llama.cpp's cell measures its `--jinja` template path: each only yields (or tries
to yield) a `tool_calls` object once the closing framing has been generated, so a
budget that ends before that leaves the call partial. That is the mechanism the
headline is about.

> SGLang is not measured here; its cell is intentionally empty. It could not be
> kept alive on this box's MIG 1g.24gb slice — torch's caching allocator
> NVML-asserts (`NVML_SUCCESS == r INTERNAL ASSERT FAILED`, `avail mem=2.83 GB`
> on a 24 GB idle slice) and `PYTORCH_NO_CUDA_MEMORY_CACHING=1` clears the assert
> but OOMs, across `--mem-fraction-static` 0.45/0.55/0.70, `--context-length`
> 2048, `expandable_segments:True` and `SGLANG_DISABLE_NVML=1`. This is an
> environment limitation, not a truncation result: its behaviour is unknown, so
> the cell stays empty rather than guessed.

## Serve recipes

### Runner (Q4_K_M GGUF)

The probe spawns Runner itself, so no separate serve step is needed:

```sh
python3 scripts/truncation-benchmark.py \
    --runner ./runner \
    --model /path/to/granite-4.1-3b-Q4_K_M.gguf \
    --runtime runner \
    --out tests/torture/truncation/<date>-granite-4.1-3b/runner --assert
```

Equivalent manual serve (what the probe does under the hood): `./runner -m
granite-4.1-3b-Q4_K_M.gguf --serve --port <p> -c 1024 --parallel 1 --gpu off`.

### vLLM (HF safetensors of the same model)

vLLM 0.27.1 in a conda env with xgrammar 0.2.3. On an sm_120 (Blackwell) MIG
slice the working flags were:

```sh
export CUDA_VISIBLE_DEVICES=<your GPU or MIG device>
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=TRITON_ATTN
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/granite-4.1-3b \
    --served-model-name granite \
    --port 8000 --max-model-len 4096 --enforce-eager \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --gpu-memory-utilization 0.85
```

`--enforce-eager` and the Triton attention backend are what make vLLM start on
this MIG slice without a full CUDA toolkit for its FlashInfer JIT path; they do
not affect the tool-call verdict. Then point the probe at it (loopback only —
tunnel a remote host to a local port first):

```sh
python3 scripts/truncation-benchmark.py \
    --endpoint 127.0.0.1:8000 --runtime vllm --runtime-version 0.27.1 \
    --model-name granite \
    --out tests/torture/truncation/<date>-granite-4.1-3b/vllm
```

`--assert` is refused against a competitor `--endpoint`: the gate enforces
*Runner's* guarantee, and asserting it against another runtime would only encode
that runtime's behaviour as a requirement.

### llama.cpp (same Q4_K_M GGUF)

The official `b10488` `ubuntu-x64` release binary (upstream publishes no Linux
CUDA asset for this tag, so it ran CPU). Tool calls come from the model's own
chat template via `--jinja`, which is on by default; no separate tool-call
parser flag exists or is needed:

```sh
llama-server -m /path/to/granite-4.1-3b-Q4_K_M.gguf \
    --host 127.0.0.1 --port 8080 --jinja -c 4096
python3 scripts/truncation-benchmark.py \
    --endpoint 127.0.0.1:8080 --runtime llama.cpp --runtime-version b10488 \
    --model-name granite-4.1-3b \
    --out tests/torture/truncation/<date>-granite-4.1-3b/llamacpp
```

### Ollama (same Q4_K_M GGUF, imported)

Ollama 0.32.14. Import the *same* GGUF so the weights are identical
(`ollama create` from a one-line `Modelfile`); the imported model auto-detects
its template and reports the `tools` capability:

```sh
printf 'FROM %s\n' /path/to/granite-4.1-3b-Q4_K_M.gguf > Modelfile
OLLAMA_HOST=127.0.0.1:11435 ollama create granite-4.1-3b -f Modelfile
python3 scripts/truncation-benchmark.py \
    --endpoint 127.0.0.1:11435 --runtime ollama --runtime-version 0.32.14 \
    --model-name granite-4.1-3b \
    --out tests/torture/truncation/<date>-granite-4.1-3b/ollama
```

## The regression gate

`make test-truncation` builds Runner, spawns it on the committed CPU fixture
(`test.gguf`), drives the same ladder, and fails red on any violation:

```
truncation-recovery gate: PASS (6 truncated rungs closed to parseable tool
calls, control rung completed)
```

The same check runs through pytest in
[`tests/test_truncation_benchmark.py`](../tests/test_truncation_benchmark.py),
which additionally proves the checker is not vacuous: it feeds synthetic ladders
that break each way the property can fail — unparseable arguments, a missing
call, framing leaked into `content`, a wrong `finish_reason`, a control rung
that truncates — and asserts each is caught.

### How a regression would turn it red

The gate asserts the observable the headline names, so a build that loses the
property fails it:

- a closer that stopped force-closing the constrained JSON would return
  unparseable `arguments` at the truncated rungs → `rung N: tool-call arguments
  do not parse as JSON`;
- a tool path that dropped the call under truncation (vLLM's observed behaviour)
  → `rung N: no tool_calls`;
- a renderer that let framing reach `content` → `rung N: framing leaked into
  content`;
- a finish-reason regression → `rung N: expected finish_reason 'length'`.

Because the property is engine-level, the mutation that would flip the gate is a
change to the closer / grammar-completion path, not to any model. The synthetic
negative controls in the pytest file stand in for that mutation without needing
a deliberately broken build in the tree.

## Reproduction notes

- Deterministic (`temperature: 0`); re-running yields identical verdicts. The
  exact argument *values* at fixture scale are meaningless (the random fixture
  emits empty/`celsius` defaults) — the property is the *shape* surviving, which
  is what the checker asserts.
- vLLM served the bf16 HF checkpoint; Runner, llama.cpp and Ollama served/
  imported the same Q4_K_M GGUF of the same model. Quantisation could in
  principle move an argument value, but not whether a parseable call is produced.
- The runner column was first produced by a runner built at `c7d6c3d` on the box
  and re-confirmed here building at `153cefa` (both 0.1.19-alpha; identical
  per-rung verdicts), in an isolated worktree on the same box as the competitor
  runs. The competitor rows were each measured one at a time on the loopback
  port, since the engines contend for the GPU and the port.
