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
on the GPU. A fifth engine, **TensorRT-LLM**, is measured on a *substitute
model* (`†` below) because granite-4.1-3b is not in its model registry — see the
dedicated note after the table. The verdict is tool-call survival, not
throughput, so the backend split does not move it — but it is a difference and
is stated rather than buried, as the other cross-runtime results in this tree
are.

Runner at `c7d6c3d` (0.1.19-alpha), re-confirmed today building at `153cefa`;
vLLM 0.27.1 + xgrammar 0.2.3; llama.cpp b10488 (`9d77fa172`); Ollama 0.32.14;
TensorRT-LLM 1.2.1 (`†` on Qwen3-1.7B, `--tool_parser qwen3`, sm_120 MIG).

**Per-rung, what the client receives — is there a parseable, executable
`tool_calls` entry?**

| `max_tokens` | Runner | vLLM | llama.cpp | Ollama | TensorRT-LLM `†` |
|---:|:--|:--|:--|:--|:--|
| 1  | **parses** | none; leaks `<tool_call>` into `content` | none; leaks `<tool_call>` into `content` | none; empty `content` | none; leaks bare `<tool_call>` tag into `content` |
| 2  | **parses** | none; leaks `<tool_call>\n` | none; leaks `<tool_call>\n` | none; empty `content` | none; empty `content` |
| 3  | **parses** | none; leaks `<tool_call>\n{"` | none; leaks `<tool_call>\n{"` | none; empty `content` | none; empty `content` |
| 5  | **parses** | none; leaks `…{"name":` | none; leaks `…{"name":` | none; empty `content` | none; empty `content` |
| 8  | **parses** | none; leaks `…{"name": "get_weather` | none; leaks `…{"name": "get_weather` | none; empty `content` | none; empty `content` |
| 16 | **parses** | none; leaks `…{"city": "` | `tool_calls` present but `arguments` = `{"city": "` **do not parse** | **HTTP 500** (`invalid tool call arguments … unexpected end of JSON input`) | none; empty `content` |
| 64 (control) | parses, `tool_calls` | parses, `tool_calls` | parses, `tool_calls` | parses, `tool_calls` | parses, `tool_calls` |

Every measured engine completes normally at 64 (the control). At every smaller
budget **only Runner returns a parseable `tool_calls` entry.** The four
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
- **TensorRT-LLM** `†` — no `tool_calls` at any truncated rung: at rung 1 the
  bare opening `<tool_call>` tag leaks into `content`, and at 2–16 `content` is
  empty (the `qwen3` parser suppresses the incomplete framing but yields no
  call). Unlike Ollama it does **not** 500 at 16 — it returns HTTP 200 with an
  empty, callless message. Measured on Qwen3-1.7B, not granite (see note).

None of the four closes the document.

### `†` TensorRT-LLM: measured on a substitute model, and why

TensorRT-LLM 1.2.1 was stood up on the same Blackwell sm_120 MIG 1g.24gb slice as
vLLM (pip `tensorrt-llm==1.2.1`; version pinned). It could **not** serve
granite-4.1-3b: `GraniteForCausalLM` is absent from its PyTorch-backend model
registry (`Unknown architecture for AutoModelForCausalLM: GraniteForCausalLM`).
Rather than leave the cell empty, the *same probe* was run against a
TRT-LLM-supported architecture — **Qwen3-1.7B** (`Qwen3ForCausalLM`), which has a
matching `--tool_parser qwen3` — because the truncation property is an **engine**
guarantee, not a model-quality one. This is a truncation-behaviour result for
the engine, **not** a quality cross-compare with the granite columns. Three
substitutions, each engine-level and disclosed:

1. **Model** — Qwen3-1.7B (HF safetensors) in place of granite-4.1-3b
   (unsupported by the TRT-LLM zoo).
2. **`tool_choice`** — TRT-LLM 1.2.1's OpenAI server **rejects
   `tool_choice: "required"` with HTTP 400** (it accepts only `none`/`auto`/a
   named tool). The measurement uses `"auto"`, which with this prompt and tool
   forces the call at every budget (Qwen3 emits the `<tool_call>` framing from
   the first token). A *named* tool choice does force the call but bypasses the
   `qwen3` parser and dumps the raw `<tool_call>…` framing into `arguments`
   unparseably at **every** rung including the control, so it is not a valid
   truncation measurement.
3. **`enable_thinking: false`** — Qwen3 is a reasoning model; with thinking on it
   spends the whole budget inside `<think>…` and never reaches the JSON even at
   64 (the control would fail). Thinking is disabled so it behaves as a direct
   tool-caller comparable to the non-reasoning granite the other engines served.

The verdict is unchanged by the substitution: TensorRT-LLM leaves the caller with
no executable tool call at every truncated rung. Raw per-rung records (with the
substitution notes embedded in the report's `notes` field) are in
`tensorrt-llm/report.json`. The full per-rung records, including the
base64 of every HTTP body, are in `vllm/report.json`, `llamacpp/report.json`,
`ollama/report.json`, and `runner-cpu/report.json` beside this note. (The Runner
results subdirectory is `runner-cpu/`, not `runner/`, because the repo's
`.gitignore` excludes the `runner` binary and would swallow a directory of that
name.)

The vLLM and Ollama cells measure their `hermes` tool-call parser path,
llama.cpp's cell measures its `--jinja` template path, and TensorRT-LLM's cell
measures its `--tool_parser qwen3` path: each only yields (or tries to yield) a
`tool_calls` object once the closing framing has been generated, so a budget that
ends before that leaves the call partial. That is the mechanism the headline is
about.

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

### TensorRT-LLM (`†` Qwen3-1.7B HF safetensors — substitute model)

TensorRT-LLM 1.2.1 on the Blackwell sm_120 MIG 1g.24gb slice (`pip install
tensorrt-llm==1.2.1 --extra-index-url https://pypi.nvidia.com`). Standing it up
needed three environment fixes, recorded so it reproduces:

- **MPI** — the wheel needs a native MPI runtime: `conda install -c conda-forge
  openmpi` into the env (otherwise `import tensorrt_llm` dies on `libmpi.so`).
- **CUDA 13 libs** — the wheel is built against CUDA 13; put a CUDA-13 toolkit on
  the loader path (`LD_LIBRARY_PATH=<cuda13>/lib`) or `libcublasLt.so.13` is
  missing at import.
- **flashinfer sm_120 JIT** — TRT-LLM JIT-compiles flashinfer's fused RMSNorm for
  `sm_120a` at startup. Point it at a real CUDA 13 toolchain so the compile+link
  succeeds: `CUDA_HOME=<cuda13>` (its `bin/nvcc`), a host `c++`/`gcc` on `PATH`
  (nvcc 13 wants gcc ≤ 14), and the CUDA stubs on the linker path
  (`LIBRARY_PATH=<cuda13>/lib/stubs`) so `-lcuda` resolves. Clear
  `~/.cache/flashinfer` after changing `CUDA_HOME` — a stale cached `build.ninja`
  keeps the old (wrong) `/usr/local/cuda` path. This is the TRT-LLM analogue of
  vLLM's "flashinfer sm_120 JIT fails on this slice" workaround.

Serve (guided decoding via xgrammar; `--tool_parser qwen3`):

```sh
# llm_opts.yaml:  guided_decoding_backend: xgrammar
export CUDA_HOME=/path/to/cuda13 CUDA_PATH=$CUDA_HOME
export PATH=/path/to/hostcc:$CUDA_HOME/bin:$PATH        # provides c++/gcc + nvcc
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$CUDA_HOME/lib/stubs:$CUDA_HOME/lib:$LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=<your MIG device UUID>
trtllm-serve serve /path/to/Qwen3-1.7B \
    --host 127.0.0.1 --port 8000 --backend pytorch \
    --tool_parser qwen3 --max_batch_size 1 --max_seq_len 2048 \
    --free_gpu_memory_fraction 0.5 --extra_llm_api_options llm_opts.yaml
```

The committed `scripts/truncation-benchmark.py` cannot drive TRT-LLM **unchanged**
— it hardcodes `tool_choice: "required"` (HTTP 400 on this server) and sets no
`chat_template_kwargs`. The `tensorrt-llm/report.json` here was produced by a
standalone probe using the *identical* tool schema, prompt, ladder and
`observe()` logic, but with `tool_choice: "auto"` and
`chat_template_kwargs: {"enable_thinking": false}` (see the report's `notes`
field and the `†` note above for why). No redaction was needed: the server's
`model` field is the basename `Qwen3-1.7B`, and no host path appears in any body.

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
