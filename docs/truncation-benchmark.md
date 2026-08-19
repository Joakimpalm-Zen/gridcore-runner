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

Everything that could move a verdict is fixed as data in the probe, so both
engines answer the same question:

| Knob | Value |
|---|---|
| Token ladder (`max_tokens`) | 1, 2, 3, 5, 8, 16, 64 |
| Control rung | 64 (must complete, proving the small rungs measure truncation, not misconfiguration) |
| Tool | `get_weather(city: string, units: enum[celsius,fahrenheit])`, both `required`, `additionalProperties: false` |
| Prompt | `What is the weather in Paris? Use fahrenheit.` |
| `tool_choice` | `required` (forces a call at every budget) |
| `temperature` | 0 (deterministic) |
| Model | `granite-4.1-3b` — Q4_K_M GGUF for Runner, the HF safetensors of the same model for vLLM |

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

Same box, same model, same schema/prompt/budgets. Runner served the Q4_K_M GGUF
on CPU; vLLM served the HF safetensors on the GPU. The verdict is tool-call
survival, not throughput, so the backend split does not move it — but it is a
difference and is stated rather than buried, as the other cross-runtime results
in this tree are.

Runner at `c7d6c3d` (0.1.19-alpha); vLLM 0.27.1 + xgrammar 0.2.3.

| `max_tokens` | Runner `finish_reason` | Runner tool call | Runner parseable | vLLM `finish_reason` | vLLM tool call | vLLM parseable | vLLM `content` leak |
|---:|:--|:--|:--|:--|:--|:--|:--|
| 1  | length | yes | yes | length | **none** | no | `<tool_call>` |
| 2  | length | yes | yes | length | **none** | no | `<tool_call>\n` |
| 3  | length | yes | yes | length | **none** | no | `<tool_call>\n{"` |
| 5  | length | yes | yes | length | **none** | no | `<tool_call>\n{"name":` |
| 8  | length | yes | yes | length | **none** | no | `<tool_call>\n{"name": "get_weather` |
| 16 | length | yes | yes | length | **none** | no | `<tool_call>\n{"name": "get_weather", "arguments": {"city": "` |
| 64 | tool_calls | yes | yes | tool_calls | yes | yes | — |

Both engines complete normally at 64 (the control). At every smaller budget
Runner returns a parseable `tool_calls` entry while vLLM returns an empty
`tool_calls` list and leaks the raw hermes framing into `content`. The full
per-rung records, including the base64 of every HTTP body, are in
`vllm/report.json` and `runner-cpu/report.json` beside this note. (The results
subdirectory is `runner-cpu/`, not `runner/`, because the repo's `.gitignore`
excludes the `runner` binary and would swallow a directory of that name.)

The competitor cell measures vLLM's `--tool-call-parser hermes` path
specifically: the parser only produces a `tool_calls` object once the closing
framing has been generated, so a budget that ends before that leaves the partial
call in `content`. That is the mechanism the headline is about.

> SGLang is not measured here; its cell is intentionally empty rather than
> guessed.

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
- vLLM served the fp16 HF checkpoint and Runner the Q4_K_M GGUF of the same
  model; quantisation could in principle move an argument value, but not whether
  a parseable call is produced.
- The runner column was produced by a runner built at HEAD (`c7d6c3d`) in an
  isolated worktree on the same box as the vLLM run.
