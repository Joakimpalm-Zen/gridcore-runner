# Gridcore Runner: Agent Runtime Build Plan

The working roadmap. `HANDOVER.md` records unfinished work from earlier sessions
and is not superseded by this file — where the two overlap, this file says *when*
and `HANDOVER.md` says *how*.

Sections marked **[carried over]** were not part of the original plan text; they
are open work folded in from earlier sessions so nothing is lost between them.

---

## Product Direction

Runner will remain a compact, dependency-free GGUF runtime specialized for
deterministic structured-output agents.

Runner will not try to match llama.cpp feature-for-feature. Its competitive
position will be:

> Guaranteed, streaming, schema-conformant agent execution in one small binary.

## Non-Negotiable Invariants

- Preserve declared-order JSON generation.
- Truncated structured output must still parse and conform.
- GPU output must remain token-identical to the CPU reference.
- Remain dependency-free beyond platform APIs and libc/pthreads.
- Preserve CPU, CUDA, and Metal support.
- Reject unsupported API/schema features explicitly; never silently ignore them.
- Preserve speculative decoding, prompt caching, model swap, reservations, and
  existing OpenAI Chat Completions compatibility.
- Add focused tests before each behavior change.
- Do not prioritize broad architecture, multimodal, UI, or model-store expansion
  until the specialized agent runtime is proven.

---

## The Thesis: Shared-Weight Structured Batching

The biggest single win is a shared-weight, continuously batched CUDA server
designed specifically for schema-constrained agent requests.

Today `--parallel N` creates separate model instances, and CUDA uploads the
weights for each slot. Replace that with:

```
One loaded model / one GPU weight allocation
        |
        +-- context 1: KV + sampler + schema state
        +-- context 2: KV + sampler + schema state
        +-- context 3: KV + sampler + schema state
```

Then combine the next token from every active request into one GPU microbatch.
Each request retains its own schema validator and sampling state, so Runner's
guarantees remain intact.

This produces several wins together:

- Eliminates duplicated CUDA weight memory.
- Raises concurrent throughput without adding model architectures.
- Makes `--parallel` genuinely useful on 8 GB cards.
- Moves Runner toward llama.cpp/vLLM serving performance.
- Preserves declared-order JSON and completion-on-truncation.
- Gives Gridcore more work per machine without changing its scheduler.
- Creates a clear external pitch: continuous batching with deterministic
  structured output.

Implemented in three stages, which map to Phases 5–7 below:

1. **Separate weights from contexts.** Refactor `model_t` into immutable shared
   model/backend data and mutable per-sequence state. One CUDA allocation owns
   weights; each slot owns KV, activations, sampler, and schema state.
2. **Cross-slot decode batching.** The scheduler gathers one token from each
   ready slot every millisecond or so. CUDA receives arrays of sequence IDs and
   positions and evaluates them together. Streaming responses continue
   independently.
3. **Forkable prefix caches.** Cache the KV state for repeated system prompts,
   tool declarations, and schema instructions. Agent traffic repeats enormous
   prefixes, so skipping that prefill can matter more than winning a synthetic
   token benchmark.

Success is measured in completed valid agent calls per second, VRAM consumption,
time to first token, and schema success under truncation — framing the
competition around Runner's actual job rather than raw token generation.

Explicitly **not** prioritized yet: AMD, multimodal, MoE, a UI, or dozens of
architectures. Those make Runner resemble an under-resourced llama.cpp.
Shared-weight structured batching makes it a stronger version of itself.

---

# Phase 0: Baseline and Conformance Harness

## Goal

Create measurable compatibility and correctness gates before extending the API.

## Work

- Add a reusable agent-protocol integration test harness.
- Capture requests and normalized streaming events as JSON fixtures.
- Add tests for `tool_choice`, `parallel_tool_calls`, tool schemas, reasoning,
  structured final responses, truncation, disconnects, and malformed requests.
- Add tests that split streaming output at every possible token boundary.
- Add real-model test scripts for Qwen2.5 1.5B/3B, Qwen3 4B, and dense Gemma.
- Record latency, prompt speed, generation speed, peak RAM, and peak VRAM.
- Add Python OpenAI SDK and Vercel AI SDK compatibility smoke tests.

## Exit Criteria

- Existing behavior has reproducible fixtures.
- Failures distinguish protocol, schema, model-quality, and transport errors.
- Linux, macOS, and Windows CI remain green.

## [carried over] Existing gates to build on, not replace

- `scripts/kernel-verify.py --baseline <exe> --candidate <exe> --model <gguf>`
  demands **token-identical** greedy output across 5 prompts. A faster binary
  that changes tokens is a regression, not a win. This already encodes the
  GPU-equals-CPU invariant; the new harness should call it rather than reinvent it.
- `scripts/kernel-bench.py` reports prefill/decode tok/s as JSON. Note its
  synthetic `item0000 item0001 …` prompt makes some instruct models emit EOS
  immediately, reporting decode as 0 tok/s — give it a natural prompt.
- `RUNNER_CUDA_PROFILE=1`, `RUNNER_DEBUG_ACT=1`, `RUNNER_DEBUG_TOKENS=1`,
  `RUNNER_FORCE_REQUANT=1` / `RUNNER_REQUANT_ONLY=<substr>`.
- `tests/fixtures/*.gguf` are committed vocabulary-only GGUFs (~5 KB for BPE,
  ~700 KB for SPM). The same trick — a fixture carrying only the metadata under
  test, with no weights — is the cheapest way to make new conformance fixtures
  runnable in CI.

## [carried over] Choosing a reference implementation

Tokenizer and numerical correctness were established this session by differential
testing against the HuggingFace reference tokenizer over a varied corpus. Two
lessons worth keeping:

- **Passing on one model proves very little.** Qwen2.5 was exact while
  Llama-3.1 was 67% wrong, because Qwen's vocabulary happened to lack the pieces
  a wrong split would have produced. Always test several families.
- **llama.cpp is not ground truth for quantized matmuls.** It quantizes
  activations to Q8_K first. On phi3 layer 0 it was 2% off on Q and 37% off on a
  small V value, while Runner matched an exact f64 dequantized computation.
  Arbitrate numerics against exact math (`gguf.quants.dequantize` + numpy f64);
  use llama.cpp only for *behavioral* comparison — does it stop, is the text sane.

---

# Phase 1: Strict Tool-Call Schema Engine

## Goal

Guarantee valid tool names and arguments during sampling rather than parsing
hopeful model output afterward.

## Work

- Compile OpenAI `tools[].function.parameters` into Runner schemas.
- Synthesize a discriminated union containing one branch per tool.
- Add a `final` branch for ordinary or schema-constrained final responses.
- Support `tool_choice`: `auto`, `required`, `none`, and named function.
- Reject unknown tools and arguments during token sampling.
- Guarantee valid arguments when `max_tokens` truncates generation.
- Map the generated envelope back into standard OpenAI `tool_calls`.
- Initially support one call with `parallel_tool_calls:false`.
- Add bounded multi-call arrays for `parallel_tool_calls:true`.
- Remove dependence on post-generation tag parsing for strict requests while
  retaining the existing parser as a compatibility fallback.

## Exit Criteria

- Tool arguments always conform to their declared JSON Schema.
- The model cannot invent a tool name.
- `finish_reason:"tool_calls"` is emitted correctly.
- Truncated tool calls remain valid and executable.
- Tools and `response_format` work in the same request.

## Release

`v0.2.0-alpha`: Guaranteed tool calls for local GGUF models.

---

# Phase 2: Correct Streaming Agent Events

## Goal

Make streaming tools as reliable as buffered tools.

## Work

- Detect the selected tool branch while generation is still running.
- Stream standards-compatible `tool_calls[].function.arguments` deltas.
- Never leak internal tool tags into `content`.
- Preserve separate `reasoning_content`, content, and tool-call channels.
- Emit stable call IDs, indexes, function types, and terminal events.
- Support multiple sequential or parallel calls without index drift.
- Handle client disconnects without leaving incomplete slot state.
- Verify structured output with speculative decoding enabled.

## Exit Criteria

- Buffered and streamed requests produce equivalent normalized results.
- OpenAI Python, TypeScript, and Vercel AI SDK clients accept every event.
- Every possible token/chunk split passes the streaming test matrix.

---

# Phase 3: OpenAI Responses API

## Goal

Make Runner directly usable by modern agent clients, including Codex-style
clients, without a translation proxy.

## Work

- Add `POST /v1/responses`.
- Support `instructions`, `input`, function tools, `tool_choice`,
  `parallel_tool_calls`, `text.format`, reasoning fields, and `store:false`.
- Emit ordered Responses events: created, output-item added, content-part added,
  text/tool deltas, item completion, and response completion.
- Support function-call outputs in subsequent requests.
- Preserve usage, cached-token, finish-reason, and Runner telemetry.
- Add a `/v1/responses` conformance test suite.
- Document a working Codex custom-provider configuration.

## Exit Criteria

- Codex can complete a local tool-call loop through Runner.
- OpenAI SDK Responses clients work in buffered and streaming modes.
- Unsupported stateful features return clear errors rather than being ignored.

## Release

`v0.3.0-alpha`: Open Responses-compatible local agent runtime.

---

# Phase 4: Anthropic Messages Compatibility

## Goal

Open a second tester funnel through Claude Code and Anthropic-compatible clients.

## Work

- Add `POST /v1/messages` and `/v1/messages/count_tokens`.
- Support system content, content blocks, tool use/results, tool choice, stop
  sequences, reasoning separation, and Anthropic SSE event ordering.
- Translate Anthropic requests into the same internal strict-agent schema engine.
- Do not create a separate generation implementation.

## Exit Criteria

- A basic Claude Code-compatible tool loop runs through Runner.
- OpenAI and Anthropic surfaces produce equivalent internal agent actions.

---

# Phase 5: Shared Model Weights

## Goal

Fix CUDA parallel-slot memory duplication and establish the foundation for
efficient concurrent serving.

## Work

- Split `model_t` into immutable shared model/backend weights and mutable
  sequence contexts.
- Allocate and upload CUDA weights once per resident model.
- Give every sequence its own KV cache, positions, activations, sampler,
  speculative state, and schema-validator state.
- Share compiled schemas and tokenized static prefixes where immutable.
- Update memory accounting, unloading, model swap, and failure cleanup.
- Correct documentation around CPU, Metal, and CUDA sharing behavior.

## Exit Criteria

- `--parallel N` uses one CUDA weight allocation.
- VRAM grows primarily by per-sequence KV and working state.
- One failed sequence cannot corrupt another.
- Unloading frees every shared and per-sequence allocation exactly once.

## Release

`v0.4.0-alpha`: Shared-weight concurrent agent serving.

## [carried over] Notes for the `model_t` split

- `model_t` gained a `fused_splits` field (owned array of sliced `gguf_tensor`
  descriptors pointing into the mmapped weights) for architectures that fuse
  QKV or gate/up. It is immutable weight-side data and belongs with the shared
  half, not the per-sequence half.
- Per-layer geometry arrays (`l_head_kv`, `l_head_dim`, `l_rope_dim`,
  `l_is_swa`, `kv_off`) and the rope tables (`rope_inv_freq`,
  `rope_inv_freq_local`) are likewise immutable and shareable.
- Clearly mutable and per-sequence today: `kcache`/`vcache`, `x`, `xb`, `xb2`,
  `q`, `k_tmp`, `v_tmp`, `hb`, `hb2`, `att`, `logits`, `all_logits`.
- `engine` already holds `stop_ids`/`n_stop` per instance; the stop list is
  derived from the tokenizer and is a candidate for sharing.

---

# Phase 6: Continuous Agent Batching

## Goal

Increase completed agent tasks per second without weakening schema guarantees.

## Work

- Add a scheduler that gathers ready sequences into decode microbatches.
- Pass sequence IDs and positions through batched CUDA kernels.
- Preserve independent schema, sampler, stop, and streaming state per sequence.
- Add prompt-prefill scheduling separately from decode scheduling.
- Add fairness, cancellation, queue limits, and per-request deadlines.
- Keep single-request latency as a first-class metric.

## Exit Criteria

- Concurrent throughput materially exceeds independent-slot execution.
- No additional model-weight copies are created.
- Batched and unbatched greedy output remain identical.
- Every sequence retains valid structured output under cancellation/truncation.

## Release

`v0.5.0-alpha`: Continuously batched structured-agent serving.

---

# Phase 7: Persistent and Forkable KV Prefixes

## Goal

Turn repeated agent instructions, tools, and schemas into a performance advantage.

## Work

- Key reusable prefixes by model, template, system prompt, tools, and schema.
- Snapshot completed KV prefixes in host memory.
- Fork one cached prefix into multiple sequence contexts.
- Add optional disk persistence across unloads and restarts.
- Validate snapshots against model hash, context geometry, KV type, and version.
- Add TTL and memory-budget eviction.
- Expose cache hits, bytes, and saved prefill time through telemetry.

## Exit Criteria

- Repeated agent requests skip static system/tool/schema prefill.
- Multiple users can fork the same prefix without interfering.
- Invalid or stale snapshots are rejected safely.
- Model swapping no longer necessarily destroys expensive prompt work.

## [carried over] Cache-key hazard

A prefix key must include everything that changes tokenization, not just the
text. `tokenizer.ggml.pre` now selects between distinct split rules
(`llama-bpe`, `qwen2`, `smollm`, GPT-2 default), and SPM scores may be rebuilt
from merge ranks at load. Two models with identical system prompts and different
`pre` values must not share a prefix. Key on the model hash, which covers this.

---

# Phase 8: GPU-Quantized KV Cache

## Goal

Strengthen Runner's large-context-on-small-hardware specialization.

## Work

- Port Q8 KV storage and attention reads to CUDA and Metal.
- Support configurable K/V cache types.
- Measure quality against fp16 using retrieval and structured-agent workloads.
- Include quantized KV in reservation and auto-fit calculations.
- Investigate mixed/per-phase precision only after Q8 is stable.

## Exit Criteria

- Q8 KV substantially increases usable context on limited VRAM.
- Retrieval and tool-selection regressions are measured and documented.
- GPU and CPU implementations pass deterministic correctness gates.

## Release

`v0.6.0-alpha`: Long-context structured agents on constrained GPUs.

---

# Phase 9: Public Agent Torture Suite and Traction

## Goal

Convert technical differentiation into external testers and trust.

## Work

- Publish a standalone adversarial agent-conformance suite.
- Run at least 100 requests per model/runtime/configuration.
- Compare Runner, llama.cpp, Ollama, and vLLM on identical hardware.
- Test nested arguments, enums, arrays, optional fields, multiple tools,
  reasoning, structured finals, speculative decoding, and forced truncation.
- Report valid calls, correct tool selection, schema conformance, streaming
  compliance, latency, tasks/second, RAM, and VRAM.
- Publish all prompts, requests, outputs, commands, and raw result files.
- Add copy-ready examples for OpenAI SDK, Vercel AI SDK, Codex, Cline,
  OpenCode, and Claude-compatible clients.
- Create a pinned GitHub discussion: "Bring your nastiest tool schema."
- Add an issue template requesting model, quant, request body, `--version`,
  `--caps`, and expected tool result.
- Invite external model/hardware submissions and publish verified results.

## Success Measures

- External binary downloads and GitHub stars.
- Independent bug reports and hardware results.
- Number of tested model/quant/hardware combinations.
- Agent-conformance success rate.
- Valid structured tasks per second.
- Projects using Runner without depending on the rest of Gridcore.

---

# Priority Order

1. Conformance harness.
2. Strict tool schemas and structured final responses.
3. Correct streaming tool calls.
4. `/v1/responses`.
5. Anthropic Messages compatibility.
6. Shared CUDA weights.
7. Continuous batching.
8. Persistent/forkable KV prefixes.
9. GPU Q8 KV cache.
10. Public comparative torture suite and ongoing compatibility releases.

The first public traction milestone is Phase 3. The largest serving-performance
milestone is Phase 6. The strongest long-term specialization is the combination
of strict tools, forkable agent prefixes, and quantized long-context KV.

---

# [carried over] Work Outside the Phase Plan

These predate this roadmap and are not scheduled by it. Listed so they are not
lost; slot them in deliberately.

## Recommended before or alongside Phase 0 — OOM crash bugs

`HANDOVER.md` §1 documents **three confirmed crash-on-OOM bugs in `src/json.c`**,
all reachable from untrusted HTTP request bodies. Runner deliberately runs near
memory limits (`--reserve`, multi-GB weights, hybrid GPU/CPU splits), so
allocation failure is a normal operating condition here.

Recommendation: land these ahead of the API expansion in Phases 1–4, since every
one of those phases *increases* the amount of untrusted, attacker-shaped JSON
reaching this parser. Two have clean local fixes. The third (`sb_put`, which
builds every HTTP response body) needs an interface decision first: it returns
`void`, so there is nowhere to report failure. `HANDOVER.md` recommends an `ok`
flag with callers answering 500, and is emphatic that a truncated-but-successful
response is worse than a crash.

`HANDOVER.md` §1.4 also flags ~70 further allocation sites from a heuristic scan
that are **not** verified and should be triaged, not mass-patched.

## Fits naturally into Phase 0 — fuzz and property tests

`HANDOVER.md` §2 contains a ready design: six targets in priority order with
entry points, a `tests/fuzz/` layout, a bounded `make fuzz`, and the rule that a
real parser crash should be **reported with its reproducing input, not quietly
patched**. Every parser in Runner is hand-written and eats untrusted input, which
makes this the highest-value hardening work available.

Two notes: the design predates this session, so re-read `src/tokenizer.c` before
writing a `tok_encode` harness — it changed substantially. And the three
`json.c` bugs above should reproduce immediately; if a harness does not find
them, it is not reaching the code.

## Unfinished — phi3 architecture support

Written but **deliberately not committed**; the diff is preserved outside the
repo. Fused `attn_qkv` → Q/K/V and fused `ffn_up` → gate/up row slicing
(zero-copy), plus LongRoPE short/long factor selection.

Proven: layer-0 Q and V match an exact f64 dequantized matmul to 0.01%, so the
splits and orderings are right; loads and generates fluent text; CUDA and CPU
token-identical. Unresolved: it never emits `<|end|>` and rambles where llama.cpp
answers and stops, so something downstream of layer 0 is wrong. Not shipped
because `model.c` deliberately refuses `granite`/`gemma2` rather than generating
plausible gibberish, and fluent non-terminating output is that failure mode.

Next step: compute the full forward in numpy/f64 from dequantized weights for a
short prompt and binary-search the first layer that diverges from Runner.

The phi3 *chat template* and the `<|end|>` stop token were verified separately
and are already on `main`.

## Unstarted — per-family sampling defaults

Default temp/top_p/repeat-penalty presets per model family. This was intended as
the second half of a correctness-then-quality pass; the correctness half
(tokenizers, chat templates) is done.

## Verified baseline as of 2026-07-20

Tokenizer output matches the HuggingFace reference over a 721-string corpus for
Llama-3.1-8B, Llama-3.2-3B, Qwen2.5-32B, Qwen3-4B, gemma-3-4b, SmolLM2-1.7B and
Phi-3.5-mini. Greedy generation at temp 0 is token-identical between CUDA and CPU
for every model that loads. This is the baseline any Phase 0 harness should
reproduce before it is trusted.

## Divergences accepted on purpose — do not "fix"

- **Mistral-v0.3**, 2/721 strings that *begin* with whitespace. Its
  `Metaspace prepend_scheme=first` replaces a leading space with the U+2581
  prefix; Llama-2 adds one on top. No GGUF key distinguishes them, so Runner
  keeps the Llama-2 rule rather than breaking that family.
- **Llama-3.2's** official template injects a "Cutting Knowledge Date / Today
  Date" header that Runner omits, as llama.cpp's built-in llama3 template also
  does. Replicating it would mean embedding a live date in the prompt.
