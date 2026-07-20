# Gridcore Runner

> **Public alpha** (`0.1.1-alpha`). Runner is the inference engine of
> Gridcore, a larger local-agent project whose other layers are not public
> yet. The engine itself is complete, CI-tested on Linux/macOS/Windows, and
> daily-driven by the rest of the stack — but it has met few machines other
> than ours, which is exactly what an alpha is for. Run your GGUF models on
> your hardware and [open an issue](../../issues) for anything that crashes,
> misbehaves, or underperforms; `runner --version` and `runner --caps`
> output make a bug report actionable. The threat model and security
> policy live in [SECURITY.md](SECURITY.md); the correctness gates every
> change must hold are in [CONTRIBUTING.md](CONTRIBUTING.md).

The inference engine of the Gridcore suite (repo: `gridcore-runner`,
binary: `runner`). A compact local LLM inference engine, written from scratch in plain C
(~9,700 lines of hand-written C, no dependencies beyond libc/pthreads). It loads standard **GGUF**
model files — the de-facto format for local models — and runs them on CPU
(AVX2-accelerated) or GPU (CUDA, Metal), with a particular focus on squeezing
**large contexts out of small models**.

```
./runner -m models/SmolLM2-135M-Instruct-Q8_0.gguf -i          # interactive chat
./runner -m model.gguf -f big-document.txt -c 8192 -n 200      # 4x the training context
./runner -m model.gguf --serve --parallel 2                    # OpenAI-compatible API server
./runner -m model.gguf -p "..." --json                         # guaranteed-valid JSON output
./runner -m big.gguf --draft small.gguf -p "..."               # speculative decoding
```

## Why not llama.cpp?

llama.cpp is the reference implementation — broader architecture coverage,
more quant formats, faster kernels, a huge community. runner exists because
this stack needs something llama.cpp structurally cannot be: an engine small
enough to own outright, whose serving contracts the projects above it (the
Gridcore agent and task-interpreter layers, not yet public) can build
against *exactly*.

**The whole engine bends in an afternoon.** ~9,700 lines of hand-written C, one
`make`, no ggml split, no CMake, no submodules — one person holds all of it
in their head. When the grid's watchdog kept declaring busy-but-healthy
runners dead, `/health` moved into the accept loop the same day. When
SIGKILLed supervisors leaked orphaned runners holding VRAM, `--parent-pid`
landed the same day. Speculative decoding under `--serve`, CUDA graphs on
the decode loop: hours each. Against a 300k-line upstream that moves daily,
every one of those is a feature request and a wait — or a fork you now
maintain anyway, without the small-codebase payoff.

**Schema conformance is the product, not a plugin.** runner compiles a JSON
Schema into a streaming validator that *drives sampling*: properties emit in
declared order, unknown keys are impossible, and when the token budget dies
mid-document the output is minimally completed so it still parses. Those
exact semantics are load-bearing upstream — clu streams its `thinking` field
live *because* declared-order emission is guaranteed, and the interpreter's
planner/worker handoffs assume every call returns conformant JSON, so a weak
model's format failures are structurally impossible and only content quality
remains. llama.cpp's GBNF grammars can express JSON, but this contract —
declared order, always-parses-on-length, OpenAI `response_format` wiring —
would then live in someone else's engine, free to drift under two projects
that depend on its details.

**Deployment is one static file.** CUDA goes through the driver API with
embedded PTX: no CUDA toolkit at build or run time, no cuBLAS, no DLLs.
Copy the binary to any node with an NVIDIA driver and it offloads; without
one, it runs CPU. Fleet nodes don't get a build matrix.

**Fleet primitives are built in, not bolted on.** Multi-model swap with
per-request selection and idle TTL, `--caps` machine reports for the
scheduler, `--reserve` budgeting with auto-fit context, `/unload`,
`--parent-pid` supervisor lifetime. The equivalent llama.cpp deployment is
llama-server + llama-swap + supervision scripts; this grid schedules whole
machines, so the whole story lives in the one binary it already ships.

**There is no `--host` flag to get wrong.** runner binds `127.0.0.1` and gives
you no way to change it — no flag, no environment variable, no config key. The
honest framing is not that runner is more secure: the *defaults* are identical,
since llama-server and Ollama both bind loopback out of the box. The difference
is that they kept the override and runner removed it. In January 2026
SentinelLabs and Censys found ~175,000 exposed Ollama instances across 130
countries, no auth, no firewall, and nearly half with tool calling enabled —
which turns an open inference port into an open shell. Nobody set out to
publish those; they set `0.0.0.0` for one afternoon's convenience and never got
back to the firewall. runner has tool calling on three API surfaces too, so the
blast radius would be the same; it just has no line to write. Reaching a runner
from another machine is a reverse proxy, an SSH tunnel, or Tailscale, which is
where authentication and TLS belong rather than hand-rolled inside an inference
engine. And because it is a promise rather than an accident, it is a gate:
`tests/test_bind.c` fails the build if the constant or the CLI surface moves,
and `tests/conformance/test_loopback_bind.py` fails if the running server ever
answers on a non-loopback address.

The trade is explicit and deliberate: llama.cpp wins on raw speed, exotic
quants, and new-architecture coverage (runner deliberately skips MoE/SSM
architectures, IQ2/IQ3 quants, and Vulkan). runner wins when the engine is a
load-bearing component of a larger system that has to trust it, extend it,
and debug it to the last line. Correctness is held to the reference: GPU
output is verified token-identical to the CPU path, and gemma4 is verified
token-identical to llama.cpp itself.

## Build

```
make          # produces ./runner
make debug    # ASan/UBSan build for development
```

## Platform support

Plain C with a small platform layer (`src/compat.c`) — no dependencies beyond
libc and pthreads. CI builds and smoke-tests every push on:

| Platform | Toolchain | GPU |
|---|---|---|
| Linux (x86_64) | gcc | CUDA (NVIDIA driver only, no toolkit needed) |
| macOS (arm64) | Apple clang | Metal |
| Windows (x86_64) | MinGW-w64 via MSYS2 (`pacman -S make mingw-w64-ucrt-x86_64-gcc`, then `make`) | CUDA (NVIDIA driver only, no toolkit needed) |

The fp16 kernels use ARM hardware half-floats when available and fall back to
portable table lookups elsewhere. Little-endian hosts only (GGUF is
little-endian; every mainstream x86/ARM/RISC-V system qualifies).

## GPU

Two backends implement the same small interface (`src/gpu_none.c` documents
it); `--gpu auto` (the default) uses one whenever the model's quant formats
have kernels (F32, F16, Q8_0, Q4_0/1, Q5_0/1, Q4_K, Q5_K, Q6_K, IQ4_NL,
IQ4_XS); anything
else falls back to CPU with a message, as does any GPU runtime failure or a
model that does not fit. GPU output is verified token-identical to the CPU
path across every supported quant on both backends.

**Metal (Apple Silicon):** model weights are wrapped **zero-copy** from the
mmap (no extra RAM), the KV cache lives in unified memory shared with the
CPU, and each generated token is a single GPU command buffer. On
unified-memory Macs single-token generation is memory-bandwidth-bound, so
the GPU gives modest speedups (~15–20% on a 1.1B) — its real value is
freeing the CPU cores and growing headroom on bigger GPUs.

**CUDA (NVIDIA, Linux/Windows):** the driver API is loaded dynamically
(`nvcuda.dll` / `libcuda.so.1`) and kernels ship as embedded PTX, so neither
building nor running needs the CUDA toolkit — a machine without an NVIDIA
driver just uses the CPU. Weights are copied to VRAM once, the KV cache (fp16
or q8_0, see `--kv`) lives in VRAM with the
host copy kept authoritative, and prompt batches run as 8-token tiles that
decode each weight once for all tokens. A model too large for VRAM is
**partially offloaded** — as many leading layers as fit run on the GPU and
the CPU finishes the rest, so oversized models still get a speedup instead
of falling all the way back to CPU (the `--reserve-vram` cap sets how deep
the split goes). Measured on an RTX 3070: 6–36 tok/s generation
across 1.5B–8B quantized models (5–8× the same box's CPU) and 2–3× CPU
prompt evaluation. Regenerate the PTX header after kernel changes with
`make ptx` (needs a CUDA toolkit at development time only).

Vulkan (AMD/Intel) is not written yet — those machines run the CPU path.

## Resource reservations

```
./runner --serve -m model.gguf -c 0 --reserve 50
```

`--reserve P` caps runner at P% of **total** VRAM and RAM (override each with
`--reserve-vram` / `--reserve-ram`; `--reserve-cpu P` sizes the thread count
as P% of cores). With `-c 0`, the context window is auto-fit to whatever the
reservation leaves after the weights — a small model grows its window into
the reserved room (capped at its training context), a big one gets what fits,
and one that cannot fit at all falls back per the normal rules.

`GET /unload` frees the resident model's memory (single-model serve included)
so the machine can be reclaimed without stopping the server; the next request
reloads it transparently. `--ttl N` unloads automatically after N idle
seconds (default: 300 in swap mode, never in single-model mode).

## Fitting models to machines (requantizer)

```
./runner -m model-f16.gguf --quantize model-q4.gguf --quant q4_0
```

Rewrites a GGUF with its weight matrices converted to `q8_0`, `q4_0`, or
`f16` — one downloaded model can be re-packed to fit each node's RAM/VRAM
(258 MB f16 → 138 MB q8_0 → 74 MB q4_0 for a 135M model). Norms, biases and
rope factors stay f32; tensors already smaller than the target are kept;
metadata is copied verbatim. Output verified against reference quantizations
of the same model.

## Serving multiple models (swap mode)

```
./runner -m "clu=qwen3-14b.gguf,bit=qwen3-4b.gguf" --serve --ttl 300
```

Swap semantics built in: the server advertises every registered model
on `/v1/models`, keeps **one** resident at a time, loads the one named in
each request's `"model"` field on demand, and unloads after `--ttl` idle
seconds (0 = never) to free RAM/VRAM for whatever runs next. `/health`
reports the resident model. Swap mode uses a single inference slot
(matching one-model-per-GPU scheduling); use `--parallel` with a single
model when you want concurrent slots instead.

## Machine capability report

`runner --caps` prints what a scheduler needs to place work on a node:

```json
{"os":"macos","arch":"arm64","cpu_cores":8,"ram_bytes":8589934592,
 "gpu":{"backend":"metal","name":"Apple M1","unified_memory":true},
 "quants":[...],"gpu_quants":[...]}
```

## Get a model

Two ways:

1. **`./download-model.sh`** fetches a small test model.
2. **Any GGUF from Hugging Face**, e.g.:
   ```
   curl -L -O "https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q8_0.gguf"
   ```

Note: safetensors checkpoints must be converted to GGUF first — runner runs
the converted GGUF.

## Large contexts on small models

This is runner's specialty. Three pieces work together:

- **Automatic context extension.** Ask for more context than the model was
  trained on (`-c 8192` on a 2k model) and runner applies YaRN rope scaling
  automatically — no flags needed. Verified: TinyLlama (trained at 2,048)
  retrieves a fact from the start of a 4,285-token prompt at 4x extension.
  Models with rope-scaling metadata (linear/YaRN) or llama-3.x frequency
  factors (`rope_freqs.weight`) get their native scaling applied; manual
  control via `--rope-scale` and `--rope-base`.
- **fp16 or q8_0 KV cache** (`--kv f16|q8`) — fp16 is half the memory per
  context token of fp32, so twice the context fits. A 1.1B model at 32k context
  needs ~740 MB of fp16 cache; the verbose flag (`-v`) prints the exact number
  before committing. `--kv q8` packs the cache as `q8_0` blocks (34 bytes per
  32 values), cutting it to ~53% of fp16 and roughly doubling the context that
  fits a given budget. It works on both the CPU path and the CUDA backend, and
  it is included in the `--reserve` auto-fit and the GPU layer-split
  calculation. It requires every layer's `head_dim` to be a multiple of 32
  (checked at load; the cache silently stays fp16 otherwise). q8 KV is
  **lossy** — it does not reproduce fp16 output token-for-token — so fp16
  remains the default.
- **Batched prompt processing.** Long prompts are evaluated in batches
  (default 64 tokens): each weight row is dequantized once and reused across
  the whole batch, and logits are skipped for all but the last token.
  ~8x faster prompt ingestion than token-at-a-time (TinyLlama Q4_K_M:
  5 → 40 tok/s; Qwen2.5-0.5B: ~97 tok/s), with live progress on stderr.

## HTTP server (OpenAI-compatible)

```
./runner -m model.gguf --serve --port 8080 --parallel 2
```

Endpoints: `POST /v1/chat/completions`, `POST /v1/responses`,
`POST /v1/completions`,
`POST /v1/embeddings` (mean-pooled, L2-normalized), `GET /v1/models`,
`GET /v1/capabilities` (which reports the active family sampling preset),
`GET /health`, `GET /unload`. Chat completions understand `logprobs` /
`top_logprobs`, `min_p`, `repeat_penalty` (send `1` for none), `stop` (a
string or up to 4 strings, matched across token boundaries and excluded from
output), OpenAI `tools` (declared
in the prompt, parsed back into `tool_calls`), and swap-mode `keep_alive`.
Agent clients speaking the AI-SDK dialect (Cline, OpenCode, …) work as-is:
part-array message content is flattened, assistant `tool_calls` history and
`role:"tool"` results render into the conversation, and
`stream_options.include_usage` gets its usage chunk. Thinking-tuned models
(gemma4) get their reasoning channels split into `reasoning_content` instead
of leaking channel tags into content. Works with any OpenAI client:

Buffered completion responses include `runner_telemetry` with cached prompt
tokens, prompt tokens evaluated this request, generation timing, and whether
JSON/schema/speculative decoding was active.
Set `"cache_prompt": false` on a request to bypass prefix KV reuse and force
the full prompt to be evaluated.

```python
import openai
client = openai.OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="none")
r = client.chat.completions.create(
    model="runner",
    messages=[{"role": "user", "content": "Hello!"}],
    response_format={"type": "json_object"},   # optional: forced-valid JSON
    stream=True,                                # optional: SSE streaming
)
```

`--parallel N` creates N independent inference slots (each with its own KV
cache and thread pool, splitting `-t` threads between them); requests are
served concurrently, and model weights are shared between slots through the
mmap page cache, so memory grows only by KV cache per slot. The server binds
to 127.0.0.1 only. Streaming clients that disconnect stop generation
immediately. Multiple *processes* also share weights the same way — running
several `runner` instances against one GGUF costs the file size once.

## Responses API (`POST /v1/responses`)

The OpenAI Responses surface, so Codex-style agent clients and the OpenAI
SDK's `client.responses` work against a local GGUF with no translation proxy.

It is a translation layer, not a second engine: a Responses request is
reshaped into the same prompt and the same strict tool envelope that
`/v1/chat/completions` builds, so both surfaces produce identical calls with
identical guarantees, and `stream=True` stays a transport choice.

```python
import openai
client = openai.OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="none")
r = client.responses.create(
    model="runner",
    instructions="You are terse.",
    input="What is the weather in Oslo?",
    tools=[{"type": "function", "name": "get_weather",
            "parameters": {"type": "object",
                           "properties": {"city": {"type": "string"}},
                           "required": ["city"]}}],
)
for item in r.output:
    if item.type == "function_call":
        print(item.name, item.arguments)   # guaranteed to parse and conform
```

Supported: `instructions`; `input` as a string or an item array (including
`function_call` and `function_call_output` items, which is the tool loop);
function `tools` in both the flat Responses shape and the nested chat shape;
`tool_choice` auto/none/required/named; `text.format` text/`json_object`/
`json_schema`; `max_output_tokens`; the usual sampling parameters;
`reasoning` (accepted and echoed — a local model's thinking channel comes back
as a `reasoning` output item); and `store:false`.

Streaming emits the ordered typed events SDKs validate: `response.created`,
`response.in_progress`, `response.output_item.added`,
`response.content_part.added`, `response.output_text.delta` (or
`response.function_call_arguments.delta`), the matching `.done` events,
`response.output_item.done`, and `response.completed` — or
`response.incomplete` when `max_output_tokens` cut the turn short. Every event
carries a monotonic `sequence_number`, and usage, cached-token counts and
`runner_telemetry` ride on the terminal event exactly as on a buffered body.

**Not supported, and refused rather than ignored** (this runtime is stateless,
so answering 200 would tell a caller its turn was persisted when it was not):
`store:true`, `previous_response_id`, `background:true`, `conversation`,
`truncation:"auto"`, `include[]`, hosted tools (`web_search`, `file_search`,
…), and `parallel_tool_calls:true` (one call per turn for now). Each returns
400 with a message naming the field and why.

### Codex CLI custom provider

Point Codex at runner as an OpenAI-compatible Responses provider. In
`~/.codex/config.toml`:

```toml
model = "runner"
model_provider = "runner"

[model_providers.runner]
name = "Gridcore Runner"
base_url = "http://127.0.0.1:8080/v1"
wire_api = "responses"
# runner ignores the key, but the client insists on sending one
env_key = "RUNNER_API_KEY"
# this runtime has no response store, so history must be sent every turn
# rather than referenced by previous_response_id
```

```
export RUNNER_API_KEY=none
./runner -m model.gguf --serve --port 8080 -c 16384
codex "list the files here"
```

Verified against `codex-cli` 0.144.6 driving Qwen3-4B: Codex emits an
`exec_command` function call, runs it, feeds the `function_call_output` back,
and runner answers the follow-up turn.

Notes from running it for real:

- **Give it context.** Codex's system prompt plus its tool declarations is
  around 10k tokens before your question, so `-c 16384` or more. The prefix
  cache then does the heavy lifting — the second turn of the loop above reused
  9943 of 9962 prompt tokens.
- **Run Codex in its normal sandbox** (the default, or `--sandbox
  workspace-write`). In those modes Codex declares `web_search` with
  `external_web_access: false`, which runner drops as a disabled capability.
  Under `--dangerously-bypass-approvals-and-sandbox` Codex *enables*
  `web_search`, and runner refuses the request rather than pretend to offer a
  tool it cannot run.
- Use a model instruction-tuned for tool use — the strict envelope guarantees
  a *well-formed* call, not a well-*chosen* one.
- `--parallel 1` is fine; Codex issues one request at a time.

## Anthropic Messages API (`POST /v1/messages`)

The Anthropic surface, so the `anthropic` SDK and Anthropic-compatible agent
clients work against a local GGUF with no translation proxy. `POST
/v1/messages/count_tokens` answers the matching pre-flight question.

Like `/v1/responses`, it is a translation layer rather than a second engine: an
Anthropic request is reshaped into the same prompt and the same strict tool
envelope `/v1/chat/completions` builds, so a tool call made through this
surface and one made through the OpenAI surfaces are the same internal action
with the same guarantees.

```python
import anthropic
client = anthropic.Anthropic(base_url="http://127.0.0.1:8080", api_key="none")

m = client.messages.create(
    model="runner", max_tokens=256,
    system="You are terse.",
    tools=[{"name": "get_weather",
            "description": "Look up the current weather for a city.",
            "input_schema": {"type": "object",
                             "properties": {"city": {"type": "string"}},
                             "required": ["city"]}}],
    messages=[{"role": "user", "content": "What is the weather in Oslo?"}])

use = next(b for b in m.content if b.type == "tool_use")
follow = client.messages.create(
    model="runner", max_tokens=256,
    tools=[...],
    messages=[{"role": "user", "content": "What is the weather in Oslo?"},
              {"role": "assistant", "content": m.content},
              {"role": "user", "content": [{"type": "tool_result",
                                            "tool_use_id": use.id,
                                            "content": "-3C and snowing"}]}])
```

Verified against the real `anthropic` 0.117.0 Python SDK driving Qwen3-4B, not
only asserted on the wire: the loop above returns `stop_reason: "tool_use"`
with a `ToolUseBlock`, and the follow-up turn answers *"The weather in Oslo is
-3°C and snowing."* `client.messages.stream` accumulates every event into its
typed class and `get_final_message()` returns the parsed turn, including its
tool call and, on a thinking-tagged model, its `ThinkingBlock`.

Supported: `system` as a string or block list, `content` as a string or a
block list, `tool_use` / `tool_result` blocks, all four `tool_choice` forms,
`stop_sequences` (reported back by name in `stop_sequence`), `temperature`,
`top_p`, `top_k`, `metadata`, and the full SSE event sequence —
`message_start`, `content_block_start`, `content_block_delta`,
`content_block_stop`, `message_delta`, `message_stop`, with no `[DONE]`
sentinel. Reasoning is separated into `thinking` blocks on a model that has a
reasoning channel.

Refused rather than silently ignored, per this project's invariant:
`mcp_servers`, `container`, server-side tools (`web_search_*`, `computer_*`,
…), `image` and `document` content blocks, `tool_choice.
disable_parallel_tool_use: false` (the envelope is one call per turn on every
surface), and `thinking: {"type": "enabled"}` on a model with no reasoning
channel to separate. `max_tokens` is required, as it is upstream.

## Structured output (JSON and JSON Schema)

Two levels of guarantee:

- **Any-JSON**: `--json` on the CLI or `"response_format": {"type":
  "json_object"}` over the API — output is always exactly one syntactically
  valid JSON object.
- **Schema-conformant**: `--json-schema file.json` on the CLI, or OpenAI-style
  `"response_format": {"type": "json_schema", "json_schema": {"schema":
  {...}}}` (a top-level `"format": {...}` schema object is also
  accepted). The schema is compiled into a streaming validator that drives
  sampling, so output *conforms*: object properties are emitted **in declared
  order** (required ones always present, optional ones skippable, no unknown
  keys), enums and `const` are enforced literally, type unions like
  `["string","null"]` resolve correctly, `oneOf`/`anyOf` scalar `const`
  alternatives become enum constraints, strings honor `minLength` /
  `maxLength`, arrays honor `items` and `min/maxItems`, and open `{}` values
  accept any JSON. Runner also supports the discriminated action-object shape
  used by Clu: a top-level `oneOf` of same-ordered objects where a `tool`
  `const` selects that branch's `args` object schema, so cross-tool argument
  keys can be rejected during sampling. Supported subset:
  object/array/string/number/integer/boolean/null, enum, const, type unions,
  scalar-const `oneOf`/`anyOf`, and same-shape `tool`-discriminated object
  alternatives; unsupported constructs are rejected at request time with a
  clear error.

Both modes: if the token budget expires mid-document, runner completes it
minimally (per the schema when there is one) so the result always parses, and
reports `finish_reason: "length"`. Syntax and structure are guaranteed;
semantic quality is still the model's job.

## Getting reliable answers out of small models

Small models fail in ways that look like model stupidity but are often
configuration. The classic: `-p "One plus one is"` on a 135M instruct model
answers "10" — the same model answers "two" when asked properly. Measured on
the same model, same question:

| Setup | Answer |
|---|---|
| raw completion (`-p`), default sampling | "10" ✗ |
| chat mode (`-i`), default sampling | "two, but…" (rambles) |
| chat mode + `--temp 0` | "One plus one equals two." ✓ |

Rules of thumb, in order of impact:

1. **Model size is the ceiling.** 135M is a toy; 0.5B handles simple
   extraction; 1.5B–3B is the reliability sweet spot on 8 GB machines. No
   decoding trick substitutes for parameters.
2. **Always use the chat format** (`-i` or the HTTP API) for questions —
   instruct models are only calibrated inside their template. Raw `-p` is for
   text continuation.
3. **For anything with a right answer**: `--temp 0`. Greedy returns the
   model's argmax and applies no repeat penalty at all, so the answer is
   reproducible. (The penalty distorts short factual answers by punishing
   reuse of tokens from the question; it only earns its keep on long
   free-form generation, where `--temp` is above zero anyway.)
4. **Use `--json` / `response_format`** when output feeds a program — it
   eliminates format failures so only content errors remain.
5. **Extended context ≠ extended reasoning.** YaRN retrieval works at 2–4x,
   but a small model can't *reason over* thousands of tokens at once. Past
   that, chunk the work (map-reduce) or retrieve only relevant passages
   instead of stuffing the window.

## Usage

```
runner -m model [options]

  -m PATH        GGUF model file
  -p TEXT        prompt (one-shot completion; \n etc. are unescaped)
  -f FILE        read prompt from file (appended after -p text)
  -i             interactive chat mode
  --serve        HTTP server mode (OpenAI-compatible API)
  --port N       server port (default 8080)
  --parallel N   parallel inference slots in server mode (default 1)
  --json         constrain output to a single valid JSON object
  --json-schema F constrain output to the JSON Schema in file F
  --quantize OUT rewrite the model to OUT.gguf (see --quant) and exit
  --quant T      quantize target: q8_0 | q4_0 | f16 (default q4_0)
  --ttl N        swap mode: unload an idle model after N seconds (default 300)
  -n N           max new tokens (default 256, -1 = until EOS)
  -c N           context length (default: min(model max, 4096));
                 beyond the training context, YaRN is applied automatically
  -b N           prompt batch size (default 64)
  -t N           threads (default: min(8, cpus))
  -s N           RNG seed
  --temp F       temperature (0 = greedy: the model's argmax, with no
                 repeat penalty applied)
  --top-k N      top-k sampling (0 = off)
  --top-p F      nucleus sampling
  --min-p F      min-p vs the top candidate (0 = off)
  --repeat-penalty F   penalty on recently emitted tokens (1 = off)
                 The five options above default to the served model family's
                 published recommended settings, chosen from the GGUF's
                 architecture and name and logged at load. `runner --caps`
                 prints the whole preset table with a source for each entry;
                 an option given explicitly always overrides its preset.
  --rope-scale F force linear rope position scaling
  --rope-base F  override rope frequency base
  --system TEXT  system prompt for chat mode
  --chat-template chatml|llama2|llama3|mistral|zephyr|phi3|gemma|gemma4|raw
                 (default: auto)
  --no-bos       don't prepend BOS
  --ignore-eos   keep generating past end-of-text tokens
  --gpu auto|off GPU offload if a backend is available (default auto)
  --kv f16|q8    KV cache storage (default f16); q8 halves it, CPU and CUDA
  --draft PATH   small same-vocab GGUF for speculative decoding
  --draft-k N    draft tokens per round (default 4)
  --bench-json   run a small decode benchmark and print JSON metrics
  --caps         print machine capabilities as JSON and exit
  --version      print the runner version and exit
  -v             print model hyperparameters and memory use
```

Chat mode keeps the KV cache across turns (no re-processing of history) and
auto-detects the chat template (ChatML, Llama-2/3, Mistral, Zephyr, Phi-3,
Gemma, Gemma-4) from the model's metadata and vocabulary. Mistral and Llama-2
both frame turns with `[INST]`, and Phi-3 and Zephyr both use `<|role|>`, so
detection keys on the terminator each one actually uses — a Mistral model gets
no `<<SYS>>` block, which its own template rejects. Thinking-tuned models show their
reasoning between `[thinking]` markers. The server additionally reuses the
KV cache for the longest shared prompt prefix across requests, so repeated
system/template prefixes skip prompt evaluation entirely.

## What's implemented

| Area | Support |
|---|---|
| File format | GGUF v2/v3, memory-mapped (weights are never copied) |
| Architectures | `llama` (Llama 2/3, Mistral, TinyLlama, SmolLM2, …), `qwen2` (QKV biases), `qwen3` (per-head QK norms), `phi3` (fused QKV and gate/up tensors, LongRoPE short/long factors), `gemma3` (QAT and regular: sandwich norms, sliding-window attention with dual rope bases, scaled embeddings), `gemma4` (heterogeneous per-layer KV, V-less global layers, thinking channels, tool calls; verified token-identical to llama.cpp) — all CPU + CUDA |
| Tokenizers | SPM (score-based merging, byte fallback, merge-rank reconstruction when a conversion writes all-zero scores) and byte-level BPE, with per-family pre-tokenizer rules selected from `tokenizer.ggml.pre`: `llama-bpe`, `qwen2`, `smollm`, and the original GPT-2 regex as the default |
| Tensor types | F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL, IQ4_XS — every commonly served quant |
| Long context | fp16 KV cache, batched prompt eval, YaRN / linear / llama-3 freq-factor rope scaling with auto-extension |
| Tokenizers | SentencePiece (llama) with byte fallback; byte-level BPE (gpt2) with merges, special-token parsing |
| Transformer | RMSNorm, RoPE (adjacent-pair and NeoX), grouped-query attention, SwiGLU, tied embeddings |
| Sampling | temperature, top-k, top-p, min-p, repeat penalty, greedy; suppress-token bias; JSON and JSON-Schema constrained decoding; speculative decoding with a draft model |
| Server | OpenAI-compatible HTTP API, SSE streaming, N parallel slots, multi-model swap with idle TTL + keep_alive, prompt-prefix KV reuse, embeddings, logprobs, tool calls |
| GPU | CUDA (NVIDIA): full + partial (layer-split) offload; Metal (Apple Silicon): full forward pass, zero-copy weights — both CPU-identical output |
| CPU | AVX2/FMA dot kernels for every hot quant format (~3x scalar generation) |
| Threading | persistent pthread pool; matmul rows and attention heads run in parallel |

Verified end-to-end with: SmolLM2-135M (Q8_0, Q4_K_M, Q3_K_M/IQ4_NL),
TinyLlama-1.1B (Q4_K_M, Q2_K), Qwen2.5-0.5B-Instruct (Q4_K_M), including a
needle-retrieval test at 2x and 4x training context and a 3,600-token
needle test on Qwen2.5.

Tokenizer output is checked against the HuggingFace reference tokenizer over a
721-string corpus — exact for Llama-3.1-8B, Llama-3.2-3B, Qwen2.5-32B,
Qwen3-4B, gemma-3-4b, SmolLM2-1.7B and Phi-3.5-mini. Mistral-7B-v0.3 differs on
2 of 721, all inputs *beginning* with whitespace: its `Metaspace
prepend_scheme=first` replaces a leading space with the U+2581 prefix where
Llama-2 adds one on top, and no GGUF key distinguishes the two. Greedy
generation at temperature 0 is token-identical between CUDA and CPU for every
model above.

Not implemented (by design, to stay small): Vulkan (AMD/Intel run on CPU),
MoE and hybrid-SSM architectures (Mamba/Jamba/Qwen3.5-style), gemma4's
shared-KV E2B/E4B variants and MTP draft head, IQ2/IQ3 codebook quants, full
GBNF grammar sampling (JSON mode only), TLS/auth on the server (bind it
behind a reverse proxy if you need those).

## How it works

```
src/runner.h     shared types
src/gguf.c       GGUF parser — mmaps the file, reads metadata KVs and tensor table
src/quants.c     dequantization + fused dot-product kernels per quant format
                 (bit-exact with the reference GGUF block layouts), fp16 LUT,
                 threadpool
src/tokenizer.c  SPM (score-based bigram merging) and BPE (rank-based merging,
                 GPT-2 byte↔unicode mapping), hash maps for vocab/merges
src/model.c      weight wiring by tensor name, rope scaling setup, and the
                 batched forward pass; fp16 KV cache
src/sample.c     temperature/top-k/top-p sampling with optional validity
                 constraint
src/jsonmode.c   incremental JSON-prefix validator + auto-close
src/schema.c     JSON-Schema compiler + streaming conformance validator
src/quantize.c   GGUF requantizer (q8_0 / q4_0 / f16)
src/template.c   chat templates, thinking-channel splitter, tool-call syntax
src/engine.c     prompt feeding + generation loop (incl. speculative decoding)
                 shared by CLI and server, prompt-prefix KV reuse
src/json.c       JSON parser/escaper/serializer + string builder for the API
src/server.c     HTTP server, OpenAI-compatible routes, parallel slots, swap
src/metal.m      Metal GPU backend (kernels.metal: the forward pass in MSL)
src/cuda.c       CUDA GPU backend via the driver API (kernels.cu -> embedded
                 PTX; full and partial layer-split offload)
src/compat.c     platform layer (mmap, clocks, cpu/ram detection)
src/main.c       CLI, --caps
python/          supported Python endpoint + child-process client for Runner consumers
```

Weights stay quantized in the mmap'd file; matmuls dequantize on the fly, so
memory use is roughly file size + KV cache + a few MB of activations.

## License

[Apache 2.0](LICENSE).
