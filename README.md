# Xyntetik Runner

Local LLM inference that behaves like infrastructure: agent tool calls that
still parse when the token budget runs out, a shared GPU that queues instead of
first-come-first-crash, and a query that tells you what fits before you load
anything.

It is one executable written from scratch in plain C — no third-party runtime
dependency or ggml — reading standard GGUF on portable CPU code, x86 AVX2/FMA,
ARM NEON, CUDA, or Metal. Its scope is deliberately explicit: supported
architectures are named, unknown ones are refused, and backend claims are tied
to executable gates and pinned model evidence.

Runner includes interactive chat, speculative decoding, constrained JSON
generation, a desktop controller, and loopback-only OpenAI- and
Anthropic-compatible HTTP APIs.

For release history and benchmark narratives, see [CHANGELOG.md](CHANGELOG.md)
and [docs/benchmarks.md](docs/benchmarks.md). Keeping that material there makes
this file a current operator reference rather than a second changelog.

## Quick start

Download a prebuilt binary from the [latest release](../../releases/latest)
for Linux, macOS, or Windows, or build from source:

```sh
git clone https://github.com/Joakimpalm-Zen/xyntetik-runner
cd xyntetik-runner
make
./runner --version   # -> runner 0.1.15-alpha
```

CUDA builds and releases need only an NVIDIA driver at runtime. The CUDA
toolkit is needed only by developers regenerating the embedded PTX.

Release archives name the binary for their platform — `runner-macos-arm64`,
`runner-linux-x86_64`, `runner-windows-x86_64.exe` — so either rename it to
`runner` or substitute that name in the commands below. A source build produces
`runner` directly.

If you have no GGUF handy, this one is 2.63 GB and runs on an 8 GB machine:

```sh
curl -L -o model.gguf \
  https://huggingface.co/Joakimpalm-Zen/gemma-4-E2B-it-Q4_0-GGUF/resolve/main/gemma-4-E2B-it-Q4_K_M-Q4_0-mix.gguf
```

Run a GGUF:

```sh
./runner -m model.gguf -i
./runner -m model.gguf -p "Explain prefix caching" --temp 0
./runner -m model.gguf --serve --parallel 2
./runner -m model.gguf -p "Return a status object" --json
./runner -m model.gguf -f big-document.txt -c 8192 -n 200
./runner -m big.gguf --draft small.gguf -p "Continue this code"
```

> **Public alpha (`0.1.15-alpha`).** CI builds and smoke-tests Linux, macOS,
> and Windows, but the project still has limited hardware coverage. Include
> `runner --version`, `runner --caps`, the model's exact filename, and the load
> log in issue reports. Read [SECURITY.md](SECURITY.md) for the threat model and
> [CONTRIBUTING.md](CONTRIBUTING.md) for the required correctness gates.

## Why runner

Choose Runner when local inference needs to behave like dependable
infrastructure: easy to deploy, bounded by the machine, and explicit about what
it can prove. The list below is ordered by how much difference each one makes in
practice, most first.

- **Tool calls survive the token limit.** An agent that receives broken JSON
  cannot proceed; it retries from scratch, burning tokens, time, and context
  window. The mechanism here is not ordinary JSON Schema support, it is
  forced-truncation recovery: once a document starts, Runner emits the smallest
  schema-legal ending when the token budget expires, so the arguments still
  parse. On local models, where context is tight and generation is slow, that is
  the difference between an agent loop that finishes and one that crashes. The
  committed [agent-torture gate](docs/agent-torture.md) tests this exact failure
  mode.
- **A shared GPU stops being first-come, first-crash.** Run a coding agent
  beside an embeddings model beside a draft model and the usual outcome is that
  one load kills another. Runner processes on the same GPU share a VRAM
  registry: a refused load names every live holder by PID, model, bytes, and
  uptime, `--wait-for-vram` turns that refusal into a bounded queue, and records
  left by dead processes are reaped. It makes a GPU something you can schedule
  rather than something you hope fits.
- **You can ask what fits before loading anything.** The usual way to find out
  whether a model fits is to load it and wait for the failure. `--caps` needs no
  model file and returns one JSON document containing live RAM/VRAM, backend and
  GPU limits, CPU and GPU quant lists, admitted architectures, placement modes,
  and model-count limits. A supervisor, tray controller, or CI job can reject an
  incompatible placement before dispatch, which removes a whole class of
  load-wait-fail-retry loops.
- **Constrained decisions come with a confidence signal.** `choice_logprobs`
  records each JSON-schema branch as legal alternatives, a posterior
  renormalized over them, and the probed probability mass — how confident the
  model was choosing one branch over another, which is what routing and
  calibrated classification actually need. The included calibration tool turns
  labeled decisions into accuracy, Brier-score, and ECE gates. This is a
  decision record rather than ordinary token logprobs, and a power-user feature:
  most workloads will never reach for it.
- **A hardware switch has a correctness contract.** If you move a workload
  between backends and the output quietly changes, that is a bug, not a tuning
  artifact. CPU/GPU identity here belongs to an exact SHA-256-pinned model and
  execution path, and faster kernels that reassociate floating-point sums must
  pass numerical tolerance gates rather than inherit a correctness claim from the
  backend name. Most users never compare outputs across backends; this is
  documented because the project treats correctness as a gate, not because it is
  a headline.

The full compatibility method is in
[docs/compatibility-program.md](docs/compatibility-program.md), performance
measurements are in [docs/performance.md](docs/performance.md), and rejected
MoE caching work is in
[docs/negative-result-expert-cache.md](docs/negative-result-expert-cache.md).

## Build and platforms

```sh
make          # release build: ./runner or runner.exe
make debug    # ASan/UBSan development build where supported
make test     # unit, fixture, generated-source, and backend gates
```

Runner uses ordinary platform C, math, threading, mmap/file-mapping, and
dynamic-loader libraries. GGUF is little-endian, so little-endian hosts are
required.

| Platform | Toolchain | Accelerated path |
|---|---|---|
| Linux x86_64 | GCC | AVX2/FMA; CUDA on NVIDIA Turing / compute capability 7.5 or newer |
| macOS arm64 | Apple Clang | ARM NEON; Metal on Apple Silicon |
| Windows x86_64 | MinGW-w64 via MSYS2 | AVX2/FMA; CUDA on NVIDIA Turing / compute capability 7.5 or newer |

On Windows, install `make` and `mingw-w64-ucrt-x86_64-gcc` from an MSYS2 UCRT64
shell, then run `make`.

## Models and conversion

Runner accepts GGUF v2/v3. Safetensors checkpoints must be converted to GGUF
first. Multi-part GGUF is not implemented; merge the parts with
`llama-gguf-split --merge` before loading. The loader detects split metadata
and names the required fix.

Fetch the small test model with:

```sh
./download-model.sh
```

For manual downloads, verify both the command exit status and resulting byte
size. A partially downloaded GGUF can otherwise look like a model failure.

### Requantization and expert pruning

Repack weight matrices to `q8_0`, `q4_0`, or `f16`:

```sh
./runner -m model-f16.gguf --quantize model-q4.gguf --quant q4_0
```

Norms, biases, and rope factors stay f32; tensors already smaller than the
target are retained. Metadata is copied.

`--prune-experts` rewrites stacked-layout MoE tensors using an explicit JSON
plan. It is a mechanism, not a quality claim: pruning needs a model-specific
evaluation against the unpruned parent.

```json
{"layer_0":[0,3,7],"layer_1":[1,2,5]}
```

```sh
# Prune only; surviving tensors keep their current quant type.
./runner -m model.gguf --quantize pruned.gguf --prune-experts keep.json

# Prune and requantize the survivors.
./runner -m model.gguf --quantize pruned-q4.gguf \
  --prune-experts keep.json --quant q4_0
```

A layer omitted from the plan keeps all experts. Invalid keys, empty lists,
out-of-range IDs, and unsupported tensor layouts fail instead of silently
producing a different model. `scripts/moe-prune-plan.py` can build a plan from
calibration data.

### Published artifacts

Artifacts produced by this project are published only after their stated gate
against the named parent. Read each repository's provenance before treating a
derivative as equivalent to an original checkpoint.

- [gpt-oss-20b-keep30-MXFP4](https://huggingface.co/Joakimpalm-Zen/gpt-oss-20b-keep30-MXFP4-GGUF)
  is an 11.5 GB, 32-to-30-expert derivative intended for a 16 GB envelope.
- [gemma-4-E2B-it Q4_K_M/Q4_0 mix](https://huggingface.co/Joakimpalm-Zen/gemma-4-E2B-it-Q4_0-GGUF) — renamed 2026-08-11 to say what it is (mixed retention, not a straight Q4_0; bytes unchanged)
  is a 2.63 GB requantization of a Q4_K_M parent. Its published quality gate
  is against that parent, not the original bf16 checkpoint.

## Command-line reference

`runner --help` remains authoritative for the binary being executed. This
grouped reference makes the complete interface discoverable without mixing
flags into unrelated feature sections.

### Modes and input

| Option | Purpose |
|---|---|
| `-m PATH` | GGUF path. In serve mode, `name=path,name2=path2` enables multi-model swap mode. |
| `-p TEXT` | One-shot prompt; escaped sequences such as `\n` are unescaped. |
| `-f FILE` | Append file contents to the prompt. |
| `-i` | Stateful interactive chat. |
| `--serve` | Start the HTTP server. |
| `--tray` | Be the macOS/Windows tray controller instead of running a model. Required where there is no terminal. See [Desktop tray](#desktop-tray). |
| `--no-tray` | Opt out of the tray everywhere, including the one that otherwise follows `--serve` and `-i`. |
| `--port N` | Server port, default `8080`. |
| `--parallel N` | Independent inference slots for a single-model server, default `1`. |
| `--ttl N` | Swap-mode idle unload timeout, default `300`; `0` disables it. |
| `--json` | Constrain output to one valid JSON object. |
| `--json-schema FILE` | Constrain output to the schema in `FILE`. |

### Generation and context

| Option | Purpose |
|---|---|
| `-n N` | Maximum generated tokens, default `256`; `-1` runs until EOS. |
| `-c N` | Context length; default is the smaller of model maximum and 4096. `0` auto-fits with a reservation. |
| `-b N` | Prompt batch size, default `64`. |
| `-t N` | Worker threads; defaults to physical cores and is capped at `64`. |
| `-s N` | RNG seed; default is time-based. |
| `--think` / `--no-think` | Request the model family's thinking or non-thinking prompt shape. With neither flag, Runner renders whatever that family's own reference template renders, which is not the same answer for every family. Families without a distinct thinking prompt accept the flag and ignore it rather than approximate one. |
| `--temp F` | Temperature; `0` is greedy and disables repeat penalty. |
| `--top-k N` | Top-k sampling; `0` disables it. |
| `--top-p F` | Nucleus sampling threshold. |
| `--min-p F` | Probability floor relative to the top candidate; `0` disables it. |
| `--repeat-penalty F` | Recent-token penalty; `1` disables it. |
| `--rope-scale F` | Force linear rope position scaling. |
| `--rope-base F` | Override the rope frequency base. |
| `--system TEXT` | System prompt in interactive chat. |
| `--chat-template NAME` | Force `chatml`, `llama2`, `llama3`, `mistral`, `zephyr`, `phi3`, `gemma`, `gemma4`, `apertus`, `ornith`, `muse`, `granite`, or `raw`; default is auto-detection. |
| `--no-bos` | Do not add the beginning-of-sequence token. |
| `--ignore-eos` | Continue generation past end-of-text tokens. |

### Placement and memory

| Option | Purpose |
|---|---|
| `--gpu auto\|off` | Auto-detect offload, or force CPU. |
| `--gpu-layers N` | Force the first `N` layers onto the GPU; `0` means no GPU. Omit for auto-fit. |
| `--cpu-moe [N\|auto]` | CUDA hybrid placement: keep all, the deepest `N`, or an auto-fit set of expert FFNs in system RAM. |
| `--wait-for-vram [S]` | Wait for another registered runner to release VRAM, default `300` seconds, instead of failing immediately. |
| `--reserve P` | Limit this process to `P` percent of total RAM and VRAM. |
| `--reserve-vram P` | Override only the VRAM budget. |
| `--reserve-ram P` | Override only the RAM budget. |
| `--reserve-cpu P` | Size the default thread count as a percentage of cores. |
| `--kv f16\|q8` | KV storage; f16 is default, q8 uses about 53% as much memory and is lossy. |
| `--mlock` | Ask the OS to wire mapped weights into RAM; failure is non-fatal. |
| `--moe-prefetch on\|off\|auto` | Prefetch routed expert blocks. Auto enables it only for measured oversubscribed Apple Silicon cases. |
| `--draft PATH` | Same-vocabulary draft GGUF for speculative decoding in one-shot, chat, or single-model serve mode. |
| `--draft-k N` | Draft tokens per speculative round, default `4`. |

### Conversion, diagnostics, and integration

| Option | Purpose |
|---|---|
| `--quantize OUT` | Rewrite the loaded model to `OUT` and exit. |
| `--quant q8_0\|q4_0\|f16` | Requantization target; default `q4_0`, or keep per-tensor types when pruning alone. |
| `--prune-experts FILE` | Apply a per-layer MoE expert keep-list while rewriting. |
| `--bench-json` | Run the built-in prompt/decode benchmark and print JSON metrics. |
| `--caps` | Print machine, backend, quant, architecture, placement, and sampling capabilities as JSON. |
| `--version` | Print the version and exit. |
| `--parent-pid N` | Exit when process `N` dies; intended for supervisor cleanup. |
| `-v` | Print verbose model and memory information. |

### Usage behavior

Use chat mode or an API chat surface to judge an instruction-tuned model.
Raw `-p` completion deliberately bypasses chat framing and is primarily useful
for benchmarks and deterministic comparison gates.

Sampling defaults come from a per-family preset selected from model metadata
and filename. The chosen preset is logged at load, `--caps` publishes the full
preset table, and explicit sampling flags always win. At `--temp 0`, runner
returns the model argmax without applying repeat penalty.

Interactive chat keeps its KV state across turns and auto-detects the template
from metadata and vocabulary. Thinking channels are displayed separately.
The server additionally reuses the longest shared prompt prefix across
requests.

On macOS and Windows, a session you sit with — a bare invocation, `--serve`, or
`-i` — also raises the desktop tray, which is left running afterwards. One-shot
`-p` runs, tooling modes, pipes, scripts, CI, and Linux keep text-mode
behavior, and `--no-tray` opts out everywhere. See [Desktop tray](#desktop-tray).

## Runtime and hardware

### CPU and GPU backends

CPU execution has portable scalar kernels plus AVX2/FMA and ARM NEON paths.
`--gpu auto` selects a usable backend and falls back with a reason when a model
layout, tensor type, runtime, or capacity is unsupported.

| Backend | Tensor formats |
|---|---|
| CPU | F32, F16, BF16, Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL, IQ4_XS, MXFP4 |
| Metal | The full CPU list |
| CUDA | The CPU list except BF16 and Q2_K |

`runner --caps` is the live source of truth for a particular executable and
machine. Architecture and MoE layout checks still happen at model load; a
listed tensor kernel does not imply that every architecture using that tensor
is implemented on that backend.

**Metal:** Apple Silicon uses zero-copy mapped weights and unified-memory KV.
Metal supports f16 and q8 KV, dense and selected MoE layouts, and tiled prefill
GEMMs. Metal is all-or-nothing rather than layer-split; a file above
`gpu.max_working_set_bytes` in `--caps` falls back to CPU. The embedded shader
gate compiles the library and verifies all 52 backend-referenced kernels.

**CUDA:** Linux and Windows use the dynamically loaded driver API and embedded
`sm_75` PTX. Full and partial layer offload are supported. Sparse MoE can keep
expert FFNs in RAM with `--cpu-moe` while attention and dense tensors remain on
the GPU. `make ptx` regenerates the embedded header and requires a CUDA toolkit
only for that development step.

Scalar-path CPU/GPU identity is an evidence result, not a property inferred
from a backend name. CUDA tensor-core and Metal tiled prefill kernels
reassociate floating-point sums, so they are promoted by teacher-forced
tolerance tests. `RUNNER_CUDA_TC=0` and `RUNNER_METAL_MM=0` pin the scalar
matvec paths for identity investigations.

Vulkan is not implemented; AMD and Intel GPUs use the CPU path.

### Long contexts

- A requested context above the training length applies model metadata for
  linear/YaRN/llama-3 rope scaling, or automatic YaRN extension when metadata
  does not supply a native scheme. `--rope-scale` and `--rope-base` override
  that behavior.
- `--kv q8` stores q8_0 blocks when every layer's head dimension is divisible
  by 32. It works on CPU, CUDA, and Metal, participates in capacity auto-fit,
  and is intentionally not token-identical to f16 KV. An incompatible head
  dimension is reported at load and keeps the cache in f16.
- Prompt evaluation is batched; `-b` controls the batch and `-v` prints the KV
  allocation before inference.

### Resource control

`--reserve` and its RAM, VRAM, and CPU variants let runner coexist with other
workloads. With `-c 0`, the context grows into the remaining reservation up to
the model's training context. A cross-process registry prevents a second
runner from blindly consuming occupied VRAM; `--wait-for-vram` turns that
refusal into a bounded queue.

`--mlock` can prevent mapped weights from being evicted, but should not be used
to force a model larger than available RAM to stay resident. Sparse MoE load
logs distinguish total file size from the smaller per-token hot set.

On high-core-count hosts, sparse MoE decode can be memory-bandwidth bound well
before the 64-thread cap. Measure `-t 12` to `-t 16` as well as the default;
the project recorded 17.0 tok/s at 12-16 threads versus 7.8 tok/s at 64 on one
128-core gemma-4-26B-A4B run. This is workload evidence, not a universal
thread-count rule.

## Serving and APIs

Start a single-model server:

```sh
./runner -m model.gguf --serve --port 8080 --parallel 2
```

The server is HTTP on loopback only, with no TLS or authentication. Binding to
`127.0.0.1` is an invariant rather than a default: there is no host flag,
environment variable, config key, or local-network toggle that can expose it.
Put it behind an authenticated reverse proxy or tunnel when remote access is
needed; do not forward the port directly. Host and Origin validation rejects
non-loopback authorities.

### Endpoints

| Method and path | Purpose |
|---|---|
| `POST /v1/chat/completions` | OpenAI Chat Completions, including SSE, tools, structured output, logprobs, and stop strings. |
| `POST /v1/responses` | OpenAI Responses translation over the same engine and tool envelope. |
| `POST /v1/completions` | Legacy raw prompt completions. |
| `POST /v1/embeddings` | Mean-pooled, L2-normalized embeddings. |
| `POST /v1/messages` | Anthropic Messages translation. |
| `POST /v1/messages/count_tokens` | Token count for the matching Messages request. |
| `GET /v1/models` | Registered models and current residency. |
| `GET /v1/capabilities` | Active model, sampling preset, and optional Xyntetik agent profile. |
| `GET /v1/runner/prefix-cache` | Prefix-cache size, limits, and counters. |
| `POST /v1/runner/prefix-cache/clear` | Release cached prefixes without unloading the model. |
| `GET /health` | Server and resident-model health. |
| `POST /unload` | Release resident model and draft memory; the next request reloads on demand. |

`GET /unload` is deliberately refused with `405`; unloading is a state change.

Buffered generation responses include `runner_telemetry` with prompt tokens
reused/evaluated, generation timing, paging counters, and structured or
speculative mode flags. Set request field `"cache_prompt": false` to bypass
prefix reuse. Streaming clients that disconnect cancel generation.

`--parallel N` creates independent KV caches and thread pools while sharing
mapped weights. Threads are divided across slots. Multi-model swap mode uses
one slot because only one model is resident at a time, and accepts up to 16
registered models:

```sh
./runner -m "code=qwen3-14b.gguf,fast=qwen3-4b.gguf" \
  --serve --ttl 300
```

Each request selects the registered name in its `model` field. `keep_alive`
can override swap residency per request. `POST /unload` works in both single-
and multi-model modes.

### Server environment

These environment variables are operator controls rather than hidden feature
switches:

| Variable | Default | Purpose |
|---|---:|---|
| `RUNNER_MAX_QUEUE` | `512` | Lower the fixed admission queue capacity. |
| `RUNNER_REQUEST_TIMEOUT` | `0` | Default generation wall-clock limit in seconds; `0` disables it. |
| `RUNNER_PREFIX_CACHE_MB` | `512` | Host-RAM budget for shared prompt prefixes; `0` disables storage. |
| `RUNNER_PREFIX_CACHE_TTL` | `600` | Prefix idle lifetime in seconds. |
| `RUNNER_MOE_PREFETCH` | per-machine auto | Compatibility fallback for `--moe-prefetch`; the CLI flag has precedence. `0`/`off` disables it and other non-empty values enable it. |

GGUF exports may opt into the versioned `gridcore.agent.*` profile. Runner
validates its protocol/tokenizer versions, schema identity, digest, and
required runtime features before allocating model state; unknown requirements
fail closed. `GET /v1/capabilities` returns the admitted profile. See
[docs/agent-profile.md](docs/agent-profile.md).

### OpenAI Chat Completions

Chat supports buffered and SSE responses, part-array content, assistant
`tool_calls` history, `role:"tool"` results, `stream_options.include_usage`,
`logprobs`/`top_logprobs`, `min_p`, `repeat_penalty`, up to four stop strings,
and `keep_alive` in swap mode. Tool declarations are rendered into the model
prompt and constrained back into well-formed `tool_calls`.

`enable_thinking`, either at the top level or inside `chat_template_kwargs`,
is the request-level form of `--think`/`--no-think`. Omitting it is not the
same as sending `false`: an absent field renders whatever the model family's
own reference template renders, and that default differs per family, so
collapsing "unspecified" onto one of them would misrender the other.

```python
import openai

client = openai.OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="none",
)
response = client.chat.completions.create(
    model="runner",
    messages=[{"role": "user", "content": "Return a status object"}],
    response_format={"type": "json_object"},
)
print(response.choices[0].message.content)
```

Constrained buffered requests can set `choice_logprobs:true`. Decision points
then include legal alternatives, posterior probability over the probed legal
set, raw logprobs, and coverage mass. `choice_logprobs_probe` defaults to 32
and is capped at 64; `scripts/cl-calibration.py` turns labeled records into an
ECE report.

### OpenAI Responses

Responses requests are translated to the same prompt, sampler, and one tool
per turn envelope as Chat Completions. Supported input includes strings and
item arrays, `function_call`/`function_call_output` loops, flat or nested
function tools, tool choice, `text.format` for text/JSON/schema, ordinary
sampling controls, `max_output_tokens`, `reasoning`, and `store:false`.

Streaming emits ordered typed lifecycle, text-delta, function-argument-delta,
done, and terminal events with monotonic `sequence_number` values. The
terminal event contains usage and runner telemetry.

Runner is stateless and refuses persistence or hosted-service fields rather
than accepting them without effect: `store:true`, `previous_response_id`,
`background:true`, `conversation`, `truncation:"auto"`, `include[]`, hosted
tools, and `parallel_tool_calls:true`.

### Anthropic Messages

Messages uses the same internal engine and constrained tool envelope. It
supports string or block-list system/content values, `tool_use`/`tool_result`,
all tool-choice forms compatible with one call per turn, stop sequences,
sampling controls, metadata, thinking-channel blocks, and Anthropic SSE event
ordering. `max_tokens` is required.

Runner refuses hosted tools, MCP/container execution, image/document blocks,
parallel tool use, and forced thinking on a model with no reasoning channel.
It implements protocol translation only; it never executes a tool.

### Coding-agent evidence

Client compatibility is a dated executable observation, not something inferred
from an API name. The 2026-08-03/04 sweep recorded complete local file-read
loops for OpenCode 1.18.4, Cline CLI 3.0.46, pi 0.81.1, Continue CLI 1.5.47,
Claude Code 2.1.220, and lean-tool-set Codex CLI 0.144.6. Aider 0.86.2 passed
transport/inference under `--dry-run` but still needs a matching model edit
profile.

Codex and other feature-rich agents can declare more than runner's 59-tool
constrained envelope. Disable unused app, multi-agent, and hosted-search tools
for a local-model session. Exact request shapes and test scope are recorded in
[docs/agent-compatibility.md](docs/agent-compatibility.md) and
[docs/compatibility-program.md](docs/compatibility-program.md).

For Codex CLI, configure a stateless Responses provider:

```toml
model = "runner"
model_provider = "runner"

[model_providers.runner]
name = "Xyntetik Runner"
base_url = "http://127.0.0.1:8080/v1"
wire_api = "responses"
env_key = "RUNNER_API_KEY"
```

```sh
export RUNNER_API_KEY=none
./runner -m model.gguf --serve -c 16384
codex "list the files here"
```

Codex's system prompt and tools can consume roughly 10k input tokens before
the user request, so use at least a 16k context for that workflow. Runner does
not implement a response store; clients must send history each turn rather
than use `previous_response_id`.

## Desktop tray

macOS and Windows ship a menu-bar / notification-area controller. It lists
every runner instance live on the machine — however it was started — with the
models each has loaded, and lets you stop any of them, pick a GGUF, and start
a desktop-managed server. Linux has no tray; `--tray` there prints an honest
error.

### When it appears

The tray follows a session you sit with, and is left running afterwards so the
next model can be loaded from it.

| Invocation | Tray |
|---|---|
| `runner` with no arguments at a terminal, or a double-click | yes |
| `runner -m model.gguf --serve` | yes |
| `runner -m model.gguf -i` | yes |
| `runner -m model.gguf -p "..."` | no |
| `--caps`, `--quantize`, `--bench-json`, `--version` | no |
| anything with `--no-tray` | no |
| pipes, scripts, CI, Linux | no |

A terminal on **either** stdin or stdout is what counts as "a person launched
this", so `runner --serve > server.log` still raises one while CI, which
usually has neither, does not. A one-shot `-p` run raises nothing on purpose:
a two-second process should not leave a menu-bar icon behind it.

`--tray` means *be* the tray rather than run a model. It is required wherever
there is no terminal — launchd, Task Scheduler, a service wrapper — because
every launch in the table above needs one. `--no-tray` opts out everywhere.

One tray runs per machine; a second exits naming the pid that owns the icon.
The tray is spawned detached with its own session, so stopping a server with
Ctrl-C leaves the menu bar alone, and it outlives the run that raised it.

### Icon states

A rounded-square core with a signal motif around it. On macOS it is a template
image, so it follows light and dark menu bars.

| State | Glyph | Meaning |
|---|---|---|
| Idle | hollow core, two opposing sweeps | No runner registered. |
| Model loaded | solid core, two opposing sweeps | A runner is up with a model resident, nothing in flight. |
| Running | solid core, four-segment ring | Inference is in flight. |

The ring is segmented rather than closed because a menu-bar template image
cannot animate: four gaps read as motion where a circle reads as a badge.

"Loaded" and "running" are told apart by `active_requests` from `/health`,
polled on the same 5-second timer that refreshes the icon — so a request
shorter than the tick can pass unseen. It is an indicator, not telemetry. When
the count cannot be read the icon shows "model loaded", because a server that
is up but unreachable still has a model resident.

Configuration, the instance registry, autostart, uninstall, and the headless
validation seams are documented in
[docs/tray-controller.md](docs/tray-controller.md).

## Structured output

Runner provides two sampler-level guarantees:

- `--json` or OpenAI `response_format.type=json_object` emits one valid JSON
  object.
- `--json-schema FILE`, OpenAI `json_schema`, Responses `text.format`, and tool
  parameter schemas compile to a streaming conformance validator.

The supported schema subset covers objects, arrays, strings, numbers,
integers, booleans, null, enums, const, type unions, integer bounds, string
lengths and supported anchored patterns, array item/count constraints,
scalar-const `oneOf`/`anyOf`, and the tool-discriminated object union used by
agent clients. Required properties are present, unknown properties are blocked
for closed objects, and tool arguments are generated against the selected
tool's schema.

Unsupported or ambiguous constraints fail at compile/request time. In
particular, general overlapping `oneOf` branches are not tracked in parallel;
branches must diverge at a supported discriminator. This is a subset of JSON
Schema 2020-12, not full JSON Schema or GBNF.

If the budget ends after a document starts, runner emits the minimal legal
suffix and reports a length finish — on the tool-call path too: a truncated
call is still returned as a parseable `tool_calls` entry, but the envelope
keeps the truncation signal (`finish_reason: "length"`, Responses
`status: "incomplete"` with `max_output_tokens`, Anthropic
`stop_reason: "max_tokens"`) so a caller knows the arguments are minimal
closures rather than the model's completed intent. If the model never starts
the document, runner returns empty content rather than inventing required
values. Syntax and schema shape are guaranteed; semantic correctness and tool
selection remain the model's responsibility.

## Support matrix

`runner --caps` publishes the architecture IDs admitted by the current binary:

| GGUF `general.architecture` | Notes |
|---|---|
| `llama`, `mistral`, `smollm`, `stablelm` | Llama-style dense families with family tokenizers/templates. |
| `qwen2`, `qwen3` | QKV-bias and per-head-QK-norm variants. |
| `qwen35` | Dense Qwen3.5/Ornith Gated DeltaNet plus full attention; CPU and CUDA. |
| `qwen3moe` | Fused and legacy split sparse-MoE layouts on CPU/CUDA; supported fused layouts on Metal. |
| `gemma3` | Regular and QAT layouts, sliding-window attention, sandwich norms. |
| `gemma4` | Heterogeneous attention, thinking channels, E-series, and supported dense/MoE layouts. |
| `phi3` | Fused QKV and gate/up tensors, LongRoPE factors. |
| `gpt-oss` | Attention sinks, alpha-sigmoid GLU, expert biases, MXFP4 experts. |
| `apertus` | xIELU FFN; CPU and CUDA. |
| `afmoe` | Arcee Trinity sparse MoE; CPU only, with CUDA/Metal refusal. |
| `muse-glimmer` | Meta Muse Glimmer 30B, text path: gated attention, QK and sandwich norms, SWA with NoPE globals, softcapped logits. CPU, CUDA and Metal. Certified; evidence in `docs/muse-glimmer-cert-2026-08-11.md`. No vision encoder or atem tool syntax. |
| `granite` | IBM Granite dense (3.x/4.1): the four muP scalars (embedding, fixed attention, residual, divided logit). CPU, CUDA and Metal. Certified; evidence in `docs/granite-cert-2026-08-11.md`. granitemoe and granitehybrid are separate arch ids and not admitted. |

Admission remains layout-specific. Shared-expert MoE, unsupported split expert
layouts, or architecture-specific tensor arrangements can still be refused
even when the architecture ID is listed.

| Area | Current support |
|---|---|
| File format | GGUF v2/v3, mmap/file-mapped host weights, single-file models only. |
| Tokenizers | SPM and byte-level BPE with llama, qwen2, smollm, tekken, llama4/o200k, Gemma, and GPT-2-family pre-tokenization rules. |
| Quantizations | `--caps` lists the admitted tensor formats: the k-quant and legacy families plus MXFP4 and the codebook i-quants (IQ1_S/M, IQ2_XXS/XS/S, IQ3_XXS/S, IQ4_NL/XS). The IQ1, IQ2 and IQ3 families are CPU-only with NEON/AVX2 dequant kernels; the CUDA and Metal backends refuse those files loudly instead of computing wrong. |
| Transformer | RMSNorm, adjacent-pair and NeoX RoPE, grouped-query attention, SwiGLU/GELU/xIELU family paths, tied embeddings, dense and selected sparse MoE. |
| Sampling | Greedy, temperature, top-k, top-p, min-p, repeat penalty, stop strings, JSON/schema constraints, speculative decoding. |
| Context | Batched prefill, f16/q8 KV, linear/YaRN/llama-3 scaling, automatic extension. |
| Serving | Chat Completions, Responses, legacy completions, embeddings, Anthropic Messages, SSE, parallel slots, model swap, prefix reuse. |
| Desktop | macOS menu bar and Windows notification-area controller. |

Not implemented: Vulkan; TLS/auth; remote bind; multi-part GGUF loading;
Qwen2-MoE/DeepSeek/Kimi shared-expert or MLA layouts; Mamba/Jamba; Gemma-4 MTP
draft heads; IQ2/IQ3 codebook quants; full GBNF; image/document inputs; hosted
tools; response persistence; or parallel tool calls.

## Compatibility evidence

The machine-readable manifest is
[`tests/compatibility/models.json`](tests/compatibility/models.json). It pins
files by SHA-256 and declares checks independently:

| Check | Meaning |
|---|---|
| `load` | The pinned file hashes and loads. |
| `tokenizer` | The committed 721-string corpus is compared with the model's Hugging Face tokenizer. |
| `greedy_reference` | Greedy tokens are compared with a pinned llama.cpp revision. |
| `cpu_cuda` | CPU and CUDA scalar-path greedy output are compared. |
| `chat` | A real Chat Completions request answers through the model template. |
| `tool` | A function call round-trips as schema-conformant tool output. |
| `long_context` | A needle is retrieved from an extended context. |

Being present in the manifest does not mean every check passed. Read each
entry's declared checks and notes. Current high-signal caveats include:

- Qwen3-4B's 2026-08-03 scalar CPU/CUDA recheck passed 4 of 5 prompts at 128
  tokens, so there is no blanket identity claim for that file.
- Canonical gpt-oss-20b passed an earlier 5-of-5, 16-token partial-offload test
  on an RTX 3070, but failed CPU/CUDA identity and chat/tokenizer gates on the
  later Blackwell full-offload matrix. Hardware and test-contract scope matter.
- Gemma-4-26B-A4B QAT's old 16-token CPU/CUDA result is not a substitute for
  the manifest's pending 128-token re-verification.
- Numerically sensitive models may use a measured self-sensitivity floor
  instead of claiming cross-engine token identity.

The full 2026-08-05 pass/fail/refusal matrix, including failed derivatives, is
in [docs/cert-matrix-status.md](docs/cert-matrix-status.md). Architecture and
model-family additions must update the manifest and executable gates, not only
this README.

## How it works

```text
src/gguf.c            GGUF metadata and tensor-table parser
src/tokenizer.c       SPM/BPE tokenization and family pre-tokenizers
src/quants.c          scalar, AVX2/FMA, and NEON quantized dot kernels
src/model.c           tensor admission, weight wiring, and forward pass
src/sample.c          sampling filters and token selection
src/jsonmode.c        incremental JSON-prefix validation
src/schema.c          JSON-Schema compiler and streaming validator
src/template.c        chat templates, thinking channels, and tool syntax
src/engine.c          prompt feeding, prefix cache, and speculative decode
src/quantize.c        requantization and stacked-MoE expert pruning
src/scheduler.c       persistent worker scheduling
src/cuda.c            CUDA driver backend; kernels.cu becomes embedded PTX
src/metal.m           Metal backend; kernels.metal is embedded at build time
src/server.c          loopback HTTP server, slots, routing, and lifecycle
src/completion.c      shared completion request/response path
src/api_responses.c   OpenAI Responses translation
src/api_anthropic.c   Anthropic Messages translation
src/registry.c        model swap and unload lifecycle
src/vramreg.c         cross-process VRAM ownership and bounded waiting
src/tray*.c           macOS/Windows desktop controller
src/compat.c          platform process, memory, mmap, clock, and socket helpers
src/main.c            CLI parsing, utility modes, and --caps
python/               supported Python client and subprocess integration
```

Host weights remain quantized in the mapped GGUF and are dequantized while
computing. CPU memory is approximately mapped weights plus KV and scratch.
CUDA copies selected weights and compute/KV buffers to VRAM; Metal wraps mapped
weights in unified memory. The load log and `--caps` are the sizing sources for
an exact model/machine combination.

## License

[Apache 2.0](LICENSE)
