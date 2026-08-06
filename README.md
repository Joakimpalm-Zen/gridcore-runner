# Gridcore Runner

A compact **local LLM inference engine written from scratch in plain C** — no
third-party runtime dependencies or ggml, one `make`, one binary. It uses only
the platform C/math/threading/dynamic-loader libraries, loads
standard **GGUF** models and runs them on **CPU (AVX2), CUDA, or Metal**, with
an OpenAI-compatible server and sampler-level JSON-schema enforcement.

**In 0.1.10 — the Gemma families stop falling back to CPU on Apple Silicon.**
Metal refused every heterogeneous-attention Gemma-4 and every GELU model,
guarding plumbing the per-token path had implemented all along. Retiring
that refusal exposed a defect worth the whole exercise: Metal compiles
kernels with fast math, where `tanh()` runs through `exp(2a)` — GELU's
argument grows as x³, overflows to `inf`, and `inf/inf` is NaN. Real
gemma-3-4b weights hit it on layer 0 and the model emitted nothing but
token 0. CUDA never showed it (nvcc isn't built with `-use_fast_math`), so
the architecture certified cleanly while the Metal path was quietly broken.
Fixed, pinned by a fixture that fails without it, and validated on real
weights: **gemma-3-4b-it Q4_K_M is byte-identical CPU vs Metal**. New
`RUNNER_METAL_NAN_TRACE=1` is the GPU-side counterpart to
`RUNNER_DEBUG_ACT` that found it.

Earlier, in 0.1.9 — the desktop release: a tray / menu-bar controller on
macOS and Windows (`--tray`), gemma-4 **E2B** loading (ARRAY-typed
per-layer FFN widths), a 19-model derivative certification campaign
(`docs/cert-matrix-status.md`), and the **[Recommended models by machine
RAM](#recommended-models-by-machine-ram)** section — measured
recommendations per machine class, and a standing refusal to recommend any
model larger than the machine's RAM.

Earlier, in 0.1.8: the Arcee Trinity MoE family (`afmoe`) — Trinity-Nano at
**13.25 tok/s, CPU-only, fully resident on an 8 GB Apple Silicon Mac** —
certified through an honest failure: greedy token-identity vs llama.cpp
failed 1/6, the mechanism was measured rather than shipped around, and the
caveat carries a falsifiable retirement test
(`docs/afmoe-divergence-triage-2026-08-05.md`). In 0.1.4: the tensor-core prefill GEMM became the default on
tolerance-gated dense Q4_K models (**+47–77% prefill**, decode unchanged),
dense decode reached **73–79% of llama.cpp** on the reference box with the
head-to-head numbers published, losing rows included
([docs/benchmarks.md](docs/benchmarks.md)), and six European models joined the
pinned compatibility manifest under the [Europe & US model
scope](docs/model-scope.md). See [CHANGELOG.md](CHANGELOG.md).

## Quick start

Download a prebuilt binary from the [latest release](../../releases/latest)
(Linux / macOS / Windows), **or** build from source — CUDA needs only the
NVIDIA driver, no toolkit; offload requires NVIDIA Turing / compute capability
7.5 or newer:

```
git clone https://github.com/Joakimpalm-Zen/gridcore-runner && cd gridcore-runner
make                 # produces ./runner (GPU auto-detected at runtime)
./runner --version   # -> runner 0.1.10-alpha
```

Then point it at any GGUF model:

```
./runner -m model.gguf -i                                   # interactive chat
./runner -m Qwen3-30B-A3B-Q4_K_M.gguf -p "..." --gpu auto   # sparse MoE on the GPU
./runner -m model.gguf --serve --parallel 2                 # OpenAI-compatible API server
./runner -m model.gguf -p "..." --json                      # guaranteed-valid JSON output
./runner -m model.gguf -f big-document.txt -c 8192 -n 200   # 4x the training context
./runner -m big.gguf --draft small.gguf -p "..."            # speculative decoding
```

> **Public alpha (`0.1.10-alpha`).** CI-tested on Linux/macOS/Windows and
> daily-driven by the rest of the Gridcore stack, but it has met few machines
> other than ours — which is what an alpha is for. Run your GGUF models and
> [open an issue](../../issues) for anything that crashes, misbehaves, or
> underperforms (`runner --version` and `runner --caps` make a report
> actionable). Threat model: [SECURITY.md](SECURITY.md); the correctness gates
> every change must hold: [CONTRIBUTING.md](CONTRIBUTING.md).

## Why runner

**Tool calls that still parse when the budget runs out.** Constrained decoding
is not novel — llama.cpp compiles JSON Schema to GBNF behind `response_format`,
Ollama takes a schema in `format`. The difference is where it sits: runner's
validator drives sampling on the path a tool call actually takes, so properties
emit in declared order, unknown keys are impossible, and **a call truncated
mid-emission still parses**. On the [agent-torture suite](docs/agent-torture.md)
— same model, same box, each runtime a `--runtime` target — that is **12/12
valid tool calls against 5/12 for llama.cpp and Ollama**, and the gap is
entirely the hard cases: deep nesting and truncation. On a model small enough
that llama.cpp's template path lands *no* parseable call (3/12), runner still
returns 12/12. Bring your nastiest schema; the suite exists to be contested.

**Small enough to own outright.** One C codebase, one `make`, no ggml split, no
CMake, no submodules — readable in a sitting, changeable the same afternoon.
`/health` in the accept loop, `--parent-pid` supervisor lifetime, speculative
decoding under `--serve`: each was an afternoon here and would be a feature
request against a 300k-line upstream. llama.cpp is broader and faster; runner is
for when the engine is load-bearing and you need to read it to the last line.

**One file to ship, and enough information to place it.** llamafile reached
driver-only GPU first by bundling tinyBLAS beside the model; runner embeds the
**PTX in the executable** instead — no toolkit, no cuBLAS, no DLLs to version
alongside. Copy it to a node with a Turing-or-newer NVIDIA GPU and a driver and
it offloads; anything else runs the CPU path. Then the half nobody ships:
`--caps` reports cores, RAM, GPU, compute capability and the quant lists CPU and
GPU each support, so a scheduler can decide *before* dispatching; `--reserve P`
caps the process at a percentage of total VRAM and RAM with the context auto-fit
to the remainder; `--parent-pid` ties its life to its supervisor. (Swapping
models is table stakes — Ollama does it natively and better. Placement is the
harder half.)

**No `--host` flag to get wrong.** runner binds `127.0.0.1` with no override —
no flag, no environment variable, no config key. llama-server and Ollama default
to loopback too; they just kept the escape hatch. [SentinelLABS and Censys
reported](https://www.sentinelone.com/labs/silent-brothers-ollama-hosts-form-anonymous-ai-network-beyond-platform-guardrails/)
**175,108 unique internet-reachable Ollama hosts** across 130 countries from
their 293-day scan ending in January 2026, nearly half with tool-calling
capability. Exposed stock Ollama APIs have no authentication. One afternoon's
`0.0.0.0` at a time. Here it is a gate, not a
hope: `tests/test_bind.c` and `tests/conformance/test_loopback_bind.py` fail the
build if the bind ever moves. Remote access belongs behind a reverse proxy, an
SSH tunnel or Tailscale, where auth and TLS already live.

**The trade, and how it is checked.** llama.cpp wins on raw speed, exotic quants
(IQ2/IQ3, Vulkan) and architecture breadth: runner does Mixtral/Qwen3 top-k MoE,
gemma-4's dual-branch GELU MoE and gpt-oss's MXFP4 experts, but skips
shared-expert MoE and most SSMs. What it does support answers to gates rather
than adjectives — GPU output verified token-identical to CPU wherever tensor
cores are not promoted, the promoted prefill path held to a measured tolerance
gate (`test-tc-tol`, run by `make test`) instead of an identity claim it cannot
have, certified architectures carrying pinned llama.cpp reference runs. And
where a model is too numerically unstable for token identity to mean anything —
a KV-precision change inside one build moving its output further than switching
engines does — that gets measured and stated, not waved at
(`scripts/sensitivity_floor.py`).

## Build and platform support

```
make          # produces ./runner
make debug    # ASan/UBSan build for development
```

Plain C with a small platform layer (`src/compat.c`) and no third-party runtime
libraries. The executable uses ordinary platform libraries (including C, math,
threading and dynamic loading where applicable). CI builds and smoke-tests every
push on:

| Platform | Toolchain | GPU |
|---|---|---|
| Linux (x86_64) | gcc | CUDA (NVIDIA Turing / compute capability 7.5 or newer; driver only, no toolkit needed) |
| macOS (arm64) | Apple clang | Metal |
| Windows (x86_64) | MinGW-w64 via MSYS2 (`pacman -S make mingw-w64-ucrt-x86_64-gcc`, then `make`) | CUDA (NVIDIA Turing / compute capability 7.5 or newer; driver only, no toolkit needed) |

The fp16 kernels use ARM hardware half-floats when available and fall back to
portable table lookups elsewhere. Little-endian hosts only (GGUF is
little-endian; every mainstream x86/ARM/RISC-V system qualifies).

## Get and fit a model

Two ways:

1. **`./download-model.sh`** fetches a small test model.
2. **Any GGUF from Hugging Face**, e.g.:
   ```
   curl -L -O "https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q8_0.gguf"
   ```
   **Check the byte size afterwards.** `download-model.sh` does; a hand-rolled
   wrapper usually does not. A reported 8B download was cut off at 290 MB by an
   HF-CDN `Connection reset by peer` — `curl` exited 56, but the surrounding
   compound command still returned 0, so nothing noticed until the load failed.
   `$?` of a pipeline is not a download check.

Note: safetensors checkpoints must be converted to GGUF first — runner runs
the converted GGUF.

### Published artifacts

Models this project built and certified are published on Hugging Face —
only artifacts that passed the quality gate (top-1 ≥ 97%, KLD ≤ 0.05 vs
their parent) get uploaded; failed experiments ship as negative-result
docs instead:

- [**gpt-oss-20b-keep30-MXFP4**](https://huggingface.co/Joakimpalm-Zen/gpt-oss-20b-keep30-MXFP4-GGUF)
  (11.5 GB) — gpt-oss-20b with the expert roster pruned 32 → 30 per layer,
  expert FFNs still native MXFP4. The 16 GB-Mac artifact: the full model's
  12.1 GB weight wrap exceeds the Metal working-set ceiling on that class
  (see [docs/moe-support.md](docs/moe-support.md)); this one fits after
  `sudo sysctl iogpu.wired_limit_mb=13312`.

### Requantizing to fit a machine

```
./runner -m model-f16.gguf --quantize model-q4.gguf --quant q4_0
```

Rewrites a GGUF with its weight matrices converted to `q8_0`, `q4_0`, or
`f16` — one downloaded model can be re-packed to fit each node's RAM/VRAM
(258 MB f16 → 138 MB q8_0 → 74 MB q4_0 for a 135M model). Norms, biases and
rope factors stay f32; tensors already smaller than the target are kept;
metadata is copied verbatim. Output verified against reference quantizations
of the same model.

## Usage

`runner --help` is the complete, authoritative option list — it ships with the
binary you are actually running, so it cannot drift from it. What follows is
the behaviour that a one-line flag description cannot carry.

**Sampling defaults come from the model.** `--temp`, `--top-k`, `--top-p`,
`--min-p` and `--repeat-penalty` default to the served family's published
recommended settings, chosen from the GGUF's architecture and name and logged
at load. `runner --caps` prints the whole preset table with a source for each
entry; passing an option explicitly always overrides its preset.

**`--gpu-layers` is a manual override, not the auto-fit knob.** Omit it and
runner fits as many layers as the budget allows; `--gpu-layers 0` means *no
GPU*, exactly like `--gpu off`.

**Chat mode is stateful and auto-templated.** It keeps the KV cache across
turns (no re-processing of history) and auto-detects the chat template (ChatML, Llama-2/3, Mistral, Zephyr, Phi-3,
Gemma, Gemma-4) from the model's metadata and vocabulary. Mistral and Llama-2
both frame turns with `[INST]`, and Phi-3 and Zephyr both use `<|role|>`, so
detection keys on the terminator each one actually uses — a Mistral model gets
no `<<SYS>>` block, which its own template rejects. Thinking-tuned models show their
reasoning between `[thinking]` markers. The server additionally reuses the
KV cache for the longest shared prompt prefix across requests, so repeated
system/template prefixes skip prompt evaluation entirely.

**`--tray` puts a controller in the menu bar / notification area** (macOS and
Windows). The grid icon lists every live runner instance on the machine with
its loaded models — swap-mode servers are asked live over `GET /v1/models` —
lets you stop any of them, and starts one pre-configured desktop-managed
server with explicit lifecycle feedback (starting → running → exited-with-log,
plus login autostart). Discovery works through per-process registry records
under `~/.gridcore/runner/` (`%APPDATA%\gridcore\runner\` on Windows) that
every run-mode invocation writes best-effort and readers self-heal, so the
CLI and serve paths are untouched when the tray is not running. Details,
config format and uninstall notes: [docs/tray-controller.md](docs/tray-controller.md).

## GPU

Two backends implement the same small interface (`src/gpu_none.c` documents
it); `--gpu auto` (the default) uses one whenever the model's quant formats
have kernels (F32, F16, Q8_0, Q4_0/1, Q5_0/1, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL,
IQ4_XS); anything
else falls back to CPU with a message, as does any GPU runtime failure or a
model that does not fit. GPU output is verified token-identical to the CPU
path across every supported quant on both backends — on the scalar kernels.
Since 2026-07-29 the tensor-core prefill GEMM is the *default* on gated dense
(Q4_K, arch) combos (+47–77% prefill): that path is fp16-tile arithmetic,
covered by a teacher-forced tolerance gate (0/64 top-1 flips on every
promoted row; the `test-tc-tol` gate in `make test`) instead of byte identity. Set
`RUNNER_CUDA_TC=0` to pin the byte-identical scalar path everywhere.

**Metal (Apple Silicon):** model weights are wrapped **zero-copy** from the
mmap (no extra RAM), the KV cache lives in unified memory shared with the
CPU, and each generated token is a single GPU command buffer. On
unified-memory Macs single-token generation is memory-bandwidth-bound, so
the GPU gives modest speedups (~15–20% on a 1.1B) — its real value is
freeing the CPU cores and growing headroom on bigger GPUs. Runtime failure
fallback keeps Metal-owned KV buffers alive for CPU recovery; the lifecycle and
remaining hardware-only validation are documented in
[docs/metal-fallback.md](docs/metal-fallback.md).

**CUDA (NVIDIA, Linux/Windows):** the driver API is loaded dynamically
(`nvcuda.dll` / `libcuda.so.1`) and kernels ship as embedded PTX, so neither
building nor running needs the CUDA toolkit — a machine without an NVIDIA
driver just uses the CPU. The embedded PTX target is `sm_75`, so GPU offload is
supported on NVIDIA Turing / compute capability 7.5 or newer; older or
unsupported NVIDIA GPUs fall back to CPU. Weights are copied to VRAM once, the
KV cache (fp16 or q8_0, see `--kv`) lives in VRAM with the
host copy kept authoritative, and prompt batches run as 16-token tiles that
decode each weight once for all tokens. A model too large for VRAM is
**partially offloaded** — as many leading layers as fit run on the GPU and
the CPU finishes the rest, so oversized models still get a speedup instead
of falling all the way back to CPU (the `--reserve-vram` cap sets how deep
the split goes). Sparse MoE models also support `--cpu-moe`: attention and
other retained dense tensors stay on CUDA while only the expert FFNs execute
from system RAM. This avoids uploading the inactive expert bank and is the
recommended placement for Qwen3-30B-A3B-class models on 8 GB cards.
The placement is per-layer: `--cpu-moe auto` fills whatever VRAM the
attention split leaves with whole expert banks and hosts only the remainder
(the split line reports `experts N/M layers on GPU`), and `--cpu-moe N`
pins exactly the deepest N expert layers to the host. A bare `--cpu-moe`
keeps every bank on the host, as before. Partial placement matters on
cards with headroom: all-or-nothing left 8.8 GB of a 12 GB card idle, and
moving just 3 of 48 banks onto a heavily-occupied 24 GB slice measured
+7% prompt and +7% decode on Qwen3-Coder-30B. Measured
on an RTX 3070: 6–36 tok/s generation
across 1.5B–8B quantized models (5–8× the same box's CPU) and 2–3× CPU
prompt evaluation. Regenerate the PTX header after kernel changes with
`make ptx` (needs a CUDA toolkit at development time only).

Vulkan (AMD/Intel) is not written yet — those machines run the CPU path.

## Sharing a machine

Three pieces cover the case where runner is not the only thing on the box:
a VRAM/RAM budget, one server that swaps between models, and a report a
scheduler can read.

### Resource reservations

```
./runner --serve -m model.gguf -c 0 --reserve 50
```

`--reserve P` caps runner at P% of **total** VRAM and RAM (override each with
`--reserve-vram` / `--reserve-ram`; `--reserve-cpu P` sizes the thread count
as P% of cores). With `-c 0`, the context window is auto-fit to whatever the
reservation leaves after the weights — a small model grows its window into
the reserved room (capped at its training context), a big one gets what fits,
and one that cannot fit at all falls back per the normal rules.

`POST /unload` frees the resident model's memory (single-model serve included)
so the machine can be reclaimed without stopping the server; the next request
reloads it transparently. **It was a `GET` in alphas before 0.1.5**, which made it
reachable from any page a user was visiting — `<img src="http://127.0.0.1:PORT/
unload">` needs no preflight and no DNS rebinding, because binding to loopback
does not stop a browser. `GET` now answers 405 naming the method rather than
404, so an existing script says what changed. `--ttl N` unloads automatically after N idle
seconds (default: 300 in swap mode, never in single-model mode).

### Serving multiple models (swap mode)

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

### Machine capability report

`runner --caps` prints what a scheduler needs to place work on a node:

```json
{"os":"macos","arch":"arm64","cpu_cores":8,"ram_bytes":8589934592,
 "gpu":{"backend":"metal","name":"Apple M1","unified_memory":true},
 "quants":[...],"gpu_quants":[...]}
```

CUDA capability reports include `"min_compute_capability":"7.5"` and
`"ptx_target":"sm_75"` so schedulers can avoid placing offload work on older
NVIDIA GPUs that will fall back to CPU.

## Large contexts

Running a context far larger than a model was trained for is runner's
specialty. Three pieces work together:

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
`GET /health`, `POST /unload`. Chat completions understand `logprobs` /
`top_logprobs`, `min_p`, `repeat_penalty` (send `1` for none), `stop` (a
string or up to 4 strings, matched across token boundaries and excluded from
output), OpenAI `tools` (declared
in the prompt, parsed back into `tool_calls`), and swap-mode `keep_alive`.
Constrained requests (JSON mode, `json_schema`, tool schemas) additionally
accept `"choice_logprobs": true` (buffered only): the response's choice gains
a `choice_logprobs` array with one record per **decision point** — a
generation step where the grammar left ≥ 2 of the probed top-`M` candidates
legal (`choice_logprobs_probe`, default 32, max 64) — carrying the legal
alternatives with a posterior renormalized over the legal probed set, their
raw full-vocabulary logprobs, and the probed coverage mass. That posterior is
the calibration surface the judgment-co-processor work builds on;
`scripts/cl-calibration.py` turns labeled decisions into a reliability/ECE
report.
Agent clients speaking the AI-SDK dialect (Cline, OpenCode, …) work as-is:
part-array message content is flattened, assistant `tool_calls` history and
`role:"tool"` results render into the conversation, and
`stream_options.include_usage` gets its usage chunk. Thinking-tuned models
(gemma4) get their reasoning channels split into `reasoning_content` instead
of leaking channel tags into content. It works with any OpenAI client (see the
example below).

The schema compiler enforces integer `minimum`, `maximum`,
`exclusiveMinimum`, and `exclusiveMaximum` bounds while sampling, including
when a truncated document has to be completed. These are not compatibility
annotations: out-of-range prefixes are refused. The support was added from
captured OpenCode 1.18.4 and Cline CLI 3.0.46 requests, whose ordinary `read`
and `bash` tools use these bounds. Both real clients' full built-in tool
declarations now pass Runner's schema compiler; end-to-end client status is
listed separately below so schema acceptance is not overstated as a completed
agent task.

OpenCode 1.18.4 is verified end to end with its AI-SDK
`@ai-sdk/openai-compatible` provider. Its real plan agent selected the built-in
Read tool, Runner streamed the constrained call, and OpenCode executed it and
returned `ORANGE-7319` on the second turn. The configured model limits for the
Qwen3-4B validation were 16,384 context and 2,048 output tokens.

Cline CLI 3.0.46 is verified end to end with its `openai-compatible` provider,
not only by replaying the captured declaration. Against Qwen3-4B it selected
the built-in `read_files` tool, consumed the tool result on turn two, and
finished with the exact fixture sentinel. The test used Cline's real streaming
AI-SDK request and its complete normal tool set.

Buffered completion responses include `runner_telemetry` with cached prompt
tokens, prompt tokens evaluated this request, generation timing, and whether
JSON/schema/speculative decoding was active.

### Verified coding-agent compatibility

These are executable client results, not claims inferred from a shared API
name. Each PASS used the published client binary and Qwen3-4B against a local
fixture; schema replay and full agent loops are called out separately where
they prove different things.

| Client tested | Result | Verified behavior |
|---|---|---|
| OpenCode 1.18.4 | PASS | Normal Plan agent, complete built-in Read call and tool-result turn over streaming Chat Completions |
| Cline CLI 3.0.46 | PASS | Normal 24-tool Plan request, `read_files` call, tool result and final answer over streaming Chat Completions |
| pi 0.81.1 | PASS | Complete Read loop on each of Chat Completions, Responses and Anthropic Messages |
| Continue CLI 1.5.47 | PASS | Read-only CLI, ten declared tools, tool-result history and final answer over Chat Completions |
| Claude Code 2.1.220 | PASS | Complete two-turn built-in Read loop over Anthropic Messages; its separately captured full built-in schema set also compiles |
| Aider 0.86.2 | PASS, model profile required | OpenAI-compatible inference and fixture result under `--dry-run`; unknown local models fall back to Aider's `whole` edit format, which is model/edit-protocol behavior rather than HTTP tool calling |
| Codex CLI 0.144.6 | CONDITIONAL | A lean Responses tool set completes a real `exec_command` loop. Feature-rich installations can expand tool namespaces beyond Runner's current 59-tool envelope; disable unused apps/multi-agent tools or provide a smaller tool set |

The installed-client sweep also considered Roo Code, but did not label it
verified: its supported surface is an editor extension and this repository's
headless fixture does not exercise the editor host. The selected terminal
clients correspond to their maintained public surfaces: [OpenCode](https://opencode.ai/docs/),
[pi](https://pi.dev/docs/latest), [Cline](https://www.npmjs.com/package/cline),
[Continue CLI](https://continue-docs.mintlify.app/cli/quickstart),
[Aider](https://aider.chat/docs/), [Claude Code](https://docs.anthropic.com/en/docs/claude-code/cli-usage),
and Codex CLI.

Exact versions, configuration-sensitive base URLs, test scope and the
client-derived regression inventory are recorded in
[`docs/agent-compatibility.md`](docs/agent-compatibility.md). The pinned
real-model, SDK, gateway and framework gates—and their machine-readable
evidence—are documented in
[`docs/compatibility-program.md`](docs/compatibility-program.md).
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
and runner answers the follow-up turn. That verification uses a lean local
tool set. The same CLI version with apps and multi-agent namespaces enabled
can flatten to more than Runner's 59-tool constrained envelope and is refused
explicitly; it is not currently an as-is PASS for every Codex installation.

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
- If the installation injects large app or multi-agent namespaces, start a
  local-model session with unused features disabled (for example `--disable
  apps --disable multi_agent`) and `web_search = "disabled"`. This is a
  current compatibility limit, not a context-window tuning problem.
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

Claude Code is also verified end to end, rather than inferred from SDK
compatibility, and the check is a script rather than a shell history:
`scripts/claude-code-e2e.sh` starts a server, writes a fixture containing a
sentinel generated for that run, points Claude Code at Runner with
`ANTHROPIC_BASE_URL`, and requires the sentinel back. Re-run against **Claude
Code 2.1.220** on 2026-08-03 with Qwen3-4B: PASS. The task is trivial and tools
are restricted to `Read` on purpose, so a failure means a protocol problem
rather than a small model wandering off.

Two things that run needs, both learned by getting them wrong: `--allowedTools`
governs *permission*, not what is declared, so Claude Code sends its whole
built-in tool set on the first request — 22.9k prompt tokens, which does not
fit in a 16k context. And the model must be named explicitly, or the CLI keeps
whatever model the developer's own session uses and dies before it makes a
single request. Separately, a captured request
containing Claude Code's full built-in tool declaration set compiles without
schema weakening. Runner accepts its `/v1/messages?beta=true` target,
`thinking.type: "adaptive"`, client-inserted system turn, open metadata object,
bounded `number`, enum-plus-const `anyOf`, and the anchored ASCII identifier
patterns used by its Workflow and Monitor tools. `format` remains annotation
behavior under the request's declared JSON Schema 2020-12 dialect.

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

Both modes: if the token budget expires after the document has begun, runner
completes it minimally (per the schema when there is one) so the result parses,
and reports `finish_reason: "length"`. If the model spends the entire budget on
a prelude and never starts the document, runner returns empty content rather
than inventing values. Syntax and structure are guaranteed for documents the
model starts; semantic quality is still the model's job.

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

## What's implemented

| Area | Support |
|---|---|
| File format | GGUF v2/v3; host weights are memory-mapped. CUDA copies selected weights into VRAM, while Metal uses zero-copy mapped weights |
| Architectures | `llama` (Llama 2/3, Mistral, TinyLlama, SmolLM2, …), `qwen2` (QKV biases), `qwen3` (per-head QK norms), dense `qwen35` (Qwen3.5/Ornith hybrid Gated DeltaNet + full attention; CPU + CUDA), `phi3` (fused QKV and gate/up tensors, LongRoPE short/long factors), `apertus` (ungated xIELU FFN; CPU + CUDA), `gemma3` (QAT and regular: sandwich norms, sliding-window attention with dual rope bases, scaled embeddings), `gemma4` (heterogeneous per-layer KV, V-less global layers, thinking channels, tool calls; verified token-identical to llama.cpp, and CPU/GPU-identical on gemma-4-12B-it) including the **E-series** (E2B/E4B: per-layer embeddings folded into every layer's residual, plus a tail of layers that own no KV cache and read an earlier layer's — CPU + CUDA, byte-identical to each other at full and every partial offload; agreement with llama.cpp is measured at the quantisation noise floor rather than as token identity, see `scripts/token_divergence.py` and `scripts/sensitivity_floor.py`), `gpt-oss` (per-head attention sinks that join the softmax max and denominator with no value row, clamped alpha-sigmoid GLU, router and per-expert biases, MXFP4 experts; CPU + CUDA, and Metal-smoke-gated with a synthetic MXFP4 fixture), `qwen3moe` and Mixtral-style sparse **MoE** (top-k router, renormalized weights, per-expert SwiGLU; fused and legacy-split expert layouts on CPU + CUDA, fused layout on Metal; Qwen3-30B-A3B measured at ~72 tok/s on an RTX PRO 6000 Blackwell 24 GB MIG slice — see docs/moe-support.md and docs/benchmarks.md), plus gemma-4's GELU **dual-branch MoE** (a dense shared GELU FFN summed with routed fused-`gate_up` experts, per-expert down scales, pre/post sandwich norms; CPU + CUDA, and Metal-smoke-gated with a synthetic Gemma-4 MoE fixture; CPU/GPU-identical on gemma-4-26B-A4B-it for CUDA. **Not** gated on token identity against llama.cpp: that model is numerically chaotic — a KV-cache precision change *inside one runner build* moves its greedy output on more prompts than switching engines does — so exact-text agreement is not achievable for it on any engine pair. See `scripts/sensitivity_floor.py` and `tests/compatibility/out/divergence-study-gemma4-moe-2026-08-01.json`). |
| Tokenizers | SPM (score-based merging, byte fallback, merge-rank reconstruction when a conversion writes all-zero scores) and byte-level BPE, with per-family pre-tokenizer rules selected from `tokenizer.ggml.pre`: `llama-bpe`, `qwen2`, `smollm`, `tekken` (Mistral Nemo/Small and Apertus: case-split letter runs, single digits), and the original GPT-2 regex as the default. gemma4 adds an SPM-style BPE: spaces normalize to U+2581 and merges run over raw UTF-8, with `<0xNN>` byte fallback for characters the vocabulary has no piece for |
| Tensor types | F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL, IQ4_XS, MXFP4 (gpt-oss; CPU + CUDA + Metal expert matvecs) — every commonly served quant. `--caps` prints the live list, plus a separate `gpu_quants`, because a few are CPU-only |
| Long context | fp16 KV cache, batched prompt eval, YaRN / linear / llama-3 freq-factor rope scaling with auto-extension |
| Transformer | RMSNorm, RoPE (adjacent-pair and NeoX), grouped-query attention, SwiGLU, tied embeddings |
| Sampling | temperature, top-k, top-p, min-p, repeat penalty, greedy; suppress-token bias; JSON and JSON-Schema constrained decoding; speculative decoding with a draft model |
| Server | OpenAI-compatible HTTP API, SSE streaming, N parallel slots, multi-model swap with idle TTL + keep_alive, prompt-prefix KV reuse, embeddings, logprobs, tool calls |
| GPU | CUDA (NVIDIA Turing / compute capability 7.5 or newer): full + partial (layer-split) offload, with scalar-path CPU identity recorded per model; Metal (Apple Silicon): full forward pass, zero-copy weights, q8 KV, plain/gpt-oss/Gemma-4 fused MoE smoke gates, with large-model hardware parity validation still open |
| CPU | AVX2/FMA dot kernels for every hot quant format (measured 1.7x scalar end-to-end on a 3B Q4 at 64 threads; see docs/performance.md) |
| Threading | persistent pthread pool; matmul rows and attention heads run in parallel |

Verified end-to-end with: SmolLM2-135M (Q8_0, Q4_K_M, Q3_K_M/IQ4_NL),
TinyLlama-1.1B (Q4_K_M, Q2_K), Qwen2.5-0.5B-Instruct (Q4_K_M), including a
needle-retrieval test at 2x and 4x training context and a 3,600-token
needle test on Qwen2.5.

Tokenizer output is checked against each model's own HuggingFace reference
tokenizer over the committed 721-string corpus in
`tests/fixtures/tokenizer-corpus.txt` (regenerate with
`scripts/tokenizer-corpus.py`, run with `scripts/difftok.py`). Exact for
SmolLM2-1.7B, Qwen2.5-32B, Qwen3-4B, Ornith-1.0-9B, gemma-4-12B,
Mistral-Nemo and — since 2026-08-03 — Apertus-8B-Instruct; 1 of 721 for
Llama-3.2-3B and gemma-3-4b, 2 of 721 for Phi-3.5-mini, and 3 of 721 for
TinyLlama-1.1B, whose three are special-token literal/adjacency cases.
Apertus's three were combining-mark sequences (Devanagari and Thai) on the
`tekken` path and are fixed: that regex is the only supported one carrying
`\p{M}` in its letter classes, so a virama or a Thai vowel sign has to stay
inside a letter run rather than end it. Llama-3.2's residual case is `Tiếng
Việt`; gemma-3's is a literal doubled U+2581 marker; Phi-3.5's two are
special-token adjacency cases. The compatibility manifest records the counts.

Mistral-7B-v0.3 differs on 44 of 721, all one known and accepted cause: its
`Metaspace prepend_scheme=first` replaces a leading space with the U+2581 prefix
where Llama-2 adds one on top, and no GGUF key distinguishes the two, so runner
keeps the Llama-2 rule rather than breaking that family. Forty of the 44 begin
with whitespace and the rest are literal U+2581 inputs, which is the same case
after normalization.

Greedy generation at temperature 0 is compared between CUDA and CPU per model
on the scalar GEMM path (`RUNNER_CUDA_TC=0`, and the default wherever tensor
cores are not promoted); identity is an evidence result, not a blanket backend
property. The 2026-08-03 128-token rerun passed 5/5 for Ornith but only 4/5 for
Qwen3-4B, whose story prompt diverged late in generation. On the promoted
dense (Q4_K, arch) combos the default prefill path is the tensor-core GEMM,
whose guarantee is the tolerance gate — 0/64 teacher-forced top-1 flips and
≤0.012% mean logit deviation on every promoted row; in free-running checks
to date its greedy output has matched the scalar path exactly, but that is
measured behavior, not a by-construction identity.

## Certified models

Runner runs a formal **compatibility program**, not just "it loaded": every
claimed architecture is certified against a real, SHA-256-pinned GGUF
(`tests/compatibility/models.json`, run by `scripts/compat_matrix.py`; full
write-up in [docs/compatibility-program.md](docs/compatibility-program.md)).
Certification is **per architecture, not per model or brand**, each check is
recorded independently (`pass` / `fail` / `not_executed`), and unknown
architectures are **refused**, not run through llama-style math — a clear
refusal beats plausible, silently-wrong output.

| Check | What it proves |
|---|---|
| **load** | the pinned file's SHA matches and the model loads (a filename is never taken as evidence) |
| **tokenizer** | runner's tokenizer matches the model's own HuggingFace reference over the committed 721-string corpus |
| **greedy_reference** | temperature-0 generation is compared token-for-token against a pinned **llama.cpp** revision |
| **cpu_cuda** | GPU and CPU produce identical greedy output (scalar path) |
| **chat** | the model answers through the real `/v1/chat/completions` surface |
| **tool** | a tool/function call round-trips as a well-formed `tool_calls` payload |
| **long_context** | a needle placed mid-document is retrieved at length (not executed for the European roster) |

Where a row says a check is *not claimed*, that is a measured decision, not a
gap: some models are numerically too sensitive for exact-text gating against a
second engine, and the honest gate is their own measured sensitivity floor
(`scripts/sensitivity_floor.py`; details per model in the manifest notes and
the linked docs).

| Architecture | Pinned model | Notes | AA† |
|---|---|---|---|
| `gemma4-moe` | gemma-4-26B-A4B-it | dual-branch dense+routed GELU MoE; CPU + CUDA, GPU/CPU-identical; token identity vs llama.cpp not claimable for this model on any engine pair ([docs/moe-support.md](docs/moe-support.md)) | 20.1 |
| `gpt-oss` | gpt-oss-20b-MXFP4 | attention sinks, MXFP4 experts; CUDA GPU/CPU byte-identical; llama.cpp agreement sits at the model's own sensitivity floor, so `greedy_reference` is not claimed | 14.9 |
| `gemma4` | gemma-4-12B-it | token-identical to llama.cpp; all checks pass. CPU + CUDA + **Metal** (0.1.10) | 13.2 |
| `gemma4` E-series | gemma-4-E4B-it | per-layer embeddings + shared-KV layers; CPU + CUDA byte-identical. E2B variants load since 2026-08-06 (per-layer FFN widths, CPU) | 8.9 |
| `qwen3moe` | Qwen3-30B-A3B | greedy-identical to llama.cpp; 128 experts / 8 active | 6.8 |
| `qwen3` | Qwen3-4B | `cpu_cuda` recheck 4/5 at 128 tokens — recorded failure | 6.8 |
| `llama` | Llama-3.2-3B, Mistral-7B-v0.3 | the reference path; `mistral`, `smollm` and `stablelm` ride it | 4.2 / 2.1 |
| `gemma3` | gemma-3-4b-it | QAT and regular; CPU + CUDA + **Metal** (0.1.10, byte-identical CPU vs Metal on the pinned file) | 1.1 |
| `qwen2` | Qwen2.5-32B-Instruct | | — |
| `qwen35` | Ornith-1.0-9B | hybrid Gated DeltaNet; CPU + CUDA 5/5 at 128 tokens | — |
| `phi3` | Phi-3.5-mini-instruct | fused QKV, LongRoPE | — |
| `apertus` | Apertus-8B-Instruct-2509 | xIELU FFN; CPU + CUDA full-graph identical; cross-engine delta below the model's own sensitivity floor | — |
| `afmoe` | Trinity-Nano-Preview-Q8_0 | **CPU only** (CUDA/Metal refuse loudly); tokenizer 0/721; `greedy_reference` not claimed with the mechanism measured, not presumed ([docs/afmoe-divergence-triage-2026-08-05.md](docs/afmoe-divergence-triage-2026-08-05.md)) | — |
| `llama` (EU roster) | EuroLLM-9B · Lucie-7B · Mistral-Nemo-12B · Teuken-7B · salamandra-7b · TildeOpen-30b | added 2026-07-29 per [docs/model-scope.md](docs/model-scope.md); each SHA-pinned with a recorded evidence run — load, 128-token `cpu_cuda`, chat, tool all pass (TildeOpen is a base model: chat/tool not claimed). Known roster caveats live in the manifest notes, incl. Lucie's tokenizer defect being in the GGUF itself | — |

† Artificial Analysis Intelligence Index (artificialanalysis.ai), fetched
2026-08-05 — context for model quality, NOT part of the certification
evidence. Non-reasoning scores where AA lists variants (gpt-oss shown at
reasoning-high, the only form AA lists); "—" = model not in AA's catalog.

New model-support work focuses on **European and US model families**
([docs/model-scope.md](docs/model-scope.md)); everything already certified
stays certified and maintained.

### Recommended models by machine RAM

Standing recommendations as of **2026-08-06**, each backed by a measured run
on real hardware (or a RAM-capped equivalent) — not extrapolation:

| Machine | Run this | Measured |
|---|---|---|
| **8 GB Apple Silicon** | **Trinity-Nano-Preview Q4_K_M** (arcee-ai) | 13.25 tok/s decode, CPU, fully resident — the standing 8 GB recommendation |
| **16 GB Apple Silicon** | [**gpt-oss-20b-keep30-MXFP4**](https://huggingface.co/Joakimpalm-Zen/gpt-oss-20b-keep30-MXFP4-GGUF) (11.5 GB) | 13.2–13.3 tok/s under a real 16 GiB cap; Metal fit after `sudo sysctl iogpu.wired_limit_mb=13312` |
| **16 GB Apple Silicon** (larger model) | gemma-4-26B-A4B-it QAT Q4_0 with `--kv q8`, served via the chat endpoints (14.4 GB) | validated 2026-08-06 on a corporate-loaded M2 Pro 16 GB: 8.1–8.2 tok/s decode, chat + JSON-schema output correct at 5.5–5.9 tok/s, greedy-deterministic. **CPU in practice**: gemma-4 runs on Metal since 0.1.10, but Metal is all-or-nothing (no partial offload) and 14.4 GB of weights exceeds the ~11.8 GiB Metal working-set ceiling on a 16 GB Mac, so it falls back with a numeric reason. Serve it — raw one-shot `-p` completions on this model can land on its documented near-tie degeneracy on any ISA |
| **Gemma on Metal** (any Mac) | any Gemma-3 / dense Gemma-4 whose weights fit the device working set (`--caps` → `gpu.max_working_set_bytes`) | byte-identical CPU vs Metal since 0.1.10. Note Metal is **all-or-nothing**: a model above the ceiling falls back to CPU entirely rather than offloading part of it |
| **24 GB GPU** | Qwen3-30B-A3B | ~72 tok/s on an RTX PRO 6000 24 GB MIG slice |

Models *larger* than a machine's RAM are not recommended in any
configuration: the paging/streaming regimes were measured exhaustively and
rejected ([docs/negative-result-expert-cache.md](docs/negative-result-expert-cache.md)).

Not implemented (by design, to stay small): Vulkan (AMD/Intel run on CPU),
`expert_shared_count`-style shared-expert MoE / MLA (Qwen2-MoE, DeepSeek/Kimi),
other hybrid-SSM architectures (Mamba/Jamba; Qwen3.5 Gated DeltaNet is supported),
gemma-4's MTP draft head, IQ2/IQ3 codebook
quants, full GBNF grammar sampling (JSON mode only), TLS/auth on the server
(bind it behind a reverse proxy if you need those).

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
src/vramreg.c    cross-process VRAM registry — a second runner on the same GPU
                 is refused (or queues with --wait-for-vram) with a message
                 naming who holds what, instead of a bare CUDA OOM
src/metal.m      Metal GPU backend (kernels.metal: the forward pass in MSL)
src/cuda.c       CUDA GPU backend via the driver API (kernels.cu -> embedded
                 PTX; full and partial layer-split offload)
src/compat.c     platform layer (mmap, clocks, cpu/ram detection)
src/main.c       CLI, --caps
python/          supported Python endpoint + child-process client for Runner consumers
```

Weights stay quantized in the mmap'd file; matmuls dequantize on the fly, so
CPU-only memory use is roughly the mapped file + KV cache + runtime scratch.
CUDA additionally copies offloaded weights and allocates compute/KV buffers in
VRAM; `--caps` and the startup placement log are the reliable sizing evidence
for a particular model and machine.

## License

[Apache 2.0](LICENSE).
