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
dedicated note after the table. A sixth, **SGLang** (`†` too), is measured on a
*different box* — the Windows RTX 3070 (sm_86 Ampere, 8 GB) under WSL2 — on the
same Qwen3-1.7B substitute, because it could not be stood up on the Blackwell MIG
slice the other GPU engines used (that first attempt, and why the 3070 was the
fallback, are in the note after the table). The verdict is tool-call survival,
not throughput, so the backend and box splits do not move it — but they are
differences and are stated rather than buried, as the other cross-runtime results
in this tree are.

Runner at `c7d6c3d` (0.1.19-alpha), re-confirmed today building at `153cefa`;
vLLM 0.27.1 + xgrammar 0.2.3; llama.cpp b10488 (`9d77fa172`); Ollama 0.32.14;
TensorRT-LLM 1.2.1 (`†` on Qwen3-1.7B, `--tool_parser qwen3`, sm_120 MIG);
SGLang 0.5.17 + xgrammar 0.2.1 (`†` on Qwen3-1.7B, `--tool-call-parser qwen`,
RTX 3070 sm_86 under WSL2).

**Per-rung, what the client receives — is there a parseable, executable
`tool_calls` entry?**

| `max_tokens` | Runner | vLLM | llama.cpp | Ollama | TensorRT-LLM `†` | SGLang `†` |
|---:|:--|:--|:--|:--|:--|:--|
| 1  | **parses** | none; leaks `<tool_call>` into `content` | none; leaks `<tool_call>` into `content` | none; empty `content` | none; leaks bare `<tool_call>` tag into `content` | none; leaks bare `<tool_call>` tag into `content` |
| 2  | **parses** | none; leaks `<tool_call>\n` | none; leaks `<tool_call>\n` | none; empty `content` | none; empty `content` | none; empty `content` |
| 3  | **parses** | none; leaks `<tool_call>\n{"` | none; leaks `<tool_call>\n{"` | none; empty `content` | none; empty `content` | none; empty `content` |
| 5  | **parses** | none; leaks `…{"name":` | none; leaks `…{"name":` | none; empty `content` | none; empty `content` | none; empty `content` |
| 8  | **parses** | none; leaks `…{"name": "get_weather` | none; leaks `…{"name": "get_weather` | none; empty `content` | none; empty `content` | none; empty `content` |
| 16 | **parses** | none; leaks `…{"city": "` | `tool_calls` present but `arguments` = `{"city": "` **do not parse** | **HTTP 500** (`invalid tool call arguments … unexpected end of JSON input`) | none; empty `content` | none; empty `content` |
| 64 (control) | parses, `tool_calls` | parses, `tool_calls` | parses, `tool_calls` | parses, `tool_calls` | parses, `tool_calls` | parses, `tool_calls` |

Every measured engine completes normally at 64 (the control). At every smaller
budget **only Runner returns a parseable `tool_calls` entry.** The five
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
- **SGLang** `†` — behaviourally identical to the TensorRT-LLM column, which is
  expected: it is the *same* Qwen3-1.7B substitute driven through the same
  Hermes-style `qwen` tool-call parser. No `tool_calls` at any truncated rung: the
  bare opening `<tool_call>` tag leaks into `content` at rung 1, `content` is empty
  at 2–16 (HTTP 200, no 500). Unlike TensorRT-LLM, SGLang **accepts**
  `tool_choice: "required"` (HTTP 200), so this column is measured with `required`
  — matching the granite/vLLM/Runner columns rather than falling back to `"auto"`;
  the `"auto"` run gives byte-identical per-rung verdicts. Measured on Qwen3-1.7B
  on the RTX 3070 / WSL2, not granite (see note).

None of the five closes the document.

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
llama.cpp's cell measures its `--jinja` template path, TensorRT-LLM's cell
measures its `--tool_parser qwen3` path, and SGLang's cell measures its
`--tool-call-parser qwen` path (with the `xgrammar` grammar backend): each only
yields (or tries to yield) a `tool_calls` object once the closing framing has been
generated, so a budget that ends before that leaves the call partial. That is the
mechanism the headline is about.

### `†` SGLang: measured on the RTX 3070 / WSL2, after the Blackwell MIG blocked it

SGLang's cell was empty in the first pass because it **could not be kept alive on
the Blackwell MIG 1g.24gb slice** the other GPU engines used — torch's caching
allocator NVML-asserts (`NVML_SUCCESS == r INTERNAL ASSERT FAILED`,
`avail mem=2.83 GB` on a 24 GB idle slice), and `PYTORCH_NO_CUDA_MEMORY_CACHING=1`
clears the assert but OOMs, across `--mem-fraction-static` 0.45/0.55/0.70,
`--context-length` 2048, `expandable_segments:True`, and `SGLANG_DISABLE_NVML=1`.
That is a MIG-partition allocator bug, not a capacity problem, so the fallback was
a **full** GPU with a normal allocator: the Windows RTX 3070 (sm_86 Ampere, 8 GB,
driver 596.36) under WSL2. There SGLang 0.5.17 stood up and served the ladder.

Because granite-4.1-3b (an internal checkpoint whose safetensors were not
obtainable on the 3070 box) is not served here either, SGLang was run on the
**same Qwen3-1.7B substitute the TensorRT-LLM column uses** — so its result is,
as expected, byte-identical to that column: the truncation property is an engine
guarantee, and both engines drive the same model through the same Hermes-style
`qwen` tool-call parser. Two disclosures specific to SGLang:

1. **`enable_thinking: false`** — as for TensorRT-LLM, Qwen3 is a reasoning model
   and otherwise spends the whole budget in `<think>` (the control at 64 would
   fail), so thinking is disabled via `chat_template_kwargs` to make it a direct
   tool-caller comparable to the non-reasoning granite the other engines served.
2. **`tool_choice: "required"` is accepted** — unlike TensorRT-LLM 1.2.1 (which
   rejects it HTTP 400 and forced the fallback to `"auto"`), SGLang serves
   `required` with HTTP 200, so this column is measured with `required`, matching
   the granite/vLLM/Runner columns. The `"auto"` run was also taken and gives
   byte-identical per-rung verdicts.

Raw per-rung records (with these notes embedded in the report's `notes` field) are
in `sglang/report.json`. No redaction was needed: the server's `model` field is
the basename `Qwen3-1.7B`, and no host path appears in any body.

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

### SGLang (`†` Qwen3-1.7B HF safetensors — substitute model)

SGLang 0.5.17 on the Windows **RTX 3070 (sm_86 Ampere, 8 GB, driver 596.36) under
WSL2**, in a Python 3.12 env (`pip install "sglang[all]"`, which pulls
torch 2.11.0+cu130, flashinfer 0.6.15, xgrammar 0.2.1). Three environment fixes
were needed and are recorded so it reproduces on a fresh box:

- **JIT linker paths** — SGLang JIT-compiles a few kernels (QK-norm, fused-RoPE)
  at first request via `nvcc` from the pip `nvidia-*-cu13` wheels. Those wheels put
  the CUDA libs in `.../nvidia/cu13/lib` with only versioned `.so.13` names, while
  the build links `-lcudart` from a `lib64` dir. Point `CUDA_HOME` at that pip
  toolkit, add `cu13/lib` (and `/usr/lib/wsl/lib` for `libcuda`) to `LIBRARY_PATH`,
  and create the unversioned `libcudart.so`→`libcudart.so.13` symlink (plus a
  `lib64`→`lib` link) so `-lcudart`/`-lcuda` resolve. Otherwise the scheduler dies
  with `collect2: ld returned 1 exit status` / `cannot find -lcudart`.
- **CUDA graph** — capture needs `nvcc`/`CUDA_HOME` set the same way; simplest on a
  fresh box is `--disable-cuda-graph` (a perf knob only; it does not touch the
  tool-call verdict).
- **`enable_thinking: false`** — supplied per-request via `chat_template_kwargs`
  (the model's template exposes the toggle; SGLang auto-detects it).

Serve (guided decoding via xgrammar; `--tool-call-parser qwen`):

```sh
SP=<env>/lib/python3.12/site-packages
export CUDA_HOME=$SP/nvidia/cu13 CUDA_PATH=$CUDA_HOME
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:/usr/lib/wsl/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$CUDA_HOME/lib:/usr/lib/wsl/lib:$LIBRARY_PATH
( cd $CUDA_HOME/lib && ln -sf libcudart.so.13 libcudart.so && ln -sf lib ../cu13/lib64 )
python -m sglang.launch_server \
    --model-path /path/to/Qwen3-1.7B --served-model-name Qwen3-1.7B \
    --host 127.0.0.1 --port 30000 \
    --tool-call-parser qwen --grammar-backend xgrammar \
    --attention-backend triton --context-length 4096 \
    --mem-fraction-static 0.75 --max-running-requests 1 --disable-cuda-graph
```

As with TensorRT-LLM, the committed `scripts/truncation-benchmark.py` cannot drive
SGLang **unchanged** because it sets no `chat_template_kwargs` (Qwen3 needs
`enable_thinking: false`). The `sglang/report.json` here was produced by a
standalone probe importing the *identical* tool schema, prompt, ladder and
`observe()` logic, adding only `chat_template_kwargs: {"enable_thinking": false}`;
`tool_choice` was kept at `"required"` (SGLang accepts it), so the column matches
the granite/vLLM/Runner columns. No redaction was needed: the server's `model`
field is the basename `Qwen3-1.7B`, and no host path appears in any body.

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
