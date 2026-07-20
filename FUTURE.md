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

# Phase 0: Baseline and Conformance Harness — MOSTLY DONE

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

## [carried over] Gaps a multi-angle review found in the existing gates

Each of these is a real defect class that current CI would not have caught:

- **No allocation-failure testing existed.** `tests/test_json_oom.c` now covers
  `json.c` by compiling it with instrumented allocators and failing each
  allocation in turn. The same technique applies to `schema.c`, `tokenizer.c`
  and the request path in `server.c`, all of which still have unchecked
  allocations sized directly from request or GGUF input.
- **No adversarial schema testing existed.** Two single-request denial-of-service
  bugs were found by reading rather than by any gate: `{"type":[]}` crashed the
  process, and `minItems` up to INT_MAX pinned a slot for minutes. Both are now
  fixed with regression tests, but the fuzz work in HANDOVER.md §2 is what would
  have found them, and `schema_compile` should be its first target.
- **The write side of the socket has no test.** The existing "stalled request is
  reaped" CI step stalls only the *read* side. A client that sends a valid
  request and then stops reading was able to pin a slot forever; a send timeout
  now bounds it, and a matching CI case belongs alongside the read-side one.
- **DONE** ~~**Streaming has no token-boundary matrix.**~~ Phase 0 already calls
  for splitting streaming output at every possible boundary; ~~note that
  `tool_calls` are currently unreachable in streaming mode at all, so that
  matrix will fail until Phase 2.~~ Phase 2 made `tool_calls` reachable on a
  stream and put a tool-call stream through the same matrix.

## [carried over] Python client defects

The managed client is the reference for how an SDK is expected to behave, so
these matter more than their size suggests:

- **`ManagedRunner.start()` leaves a failed child running.** On health-deadline
  timeout it returns False without terminating the process, so a caller that
  correctly handles a failed start still owns an orphaned runner holding VRAM.
  Either stop it on timeout or make the ownership transfer explicit.
- **The streaming client silently swallows malformed SSE.** Bad JSON, missing
  `choices`, and unexpected chunk shapes are all caught and skipped, turning
  protocol corruption into output that looks complete but is not — and any later
  `finish_reason` then marks the stream finished. Malformed `data:` should raise,
  carrying the partial response.
- **The "stall" timeout is a socket-read timeout, not an inactivity timer.** The
  message says "runner produced no bytes for N seconds", but nothing tracks the
  time of the last received event. A real stall watchdog needs explicit
  last-event timing.

## [carried over] HTTP framing strictness

`Content-Length` is matched anywhere in the header block rather than at a line
start, `strtoull` runs without checking that digits were consumed or that the
value did not overflow, duplicate and conflicting lengths are accepted, and
`Transfer-Encoding: chunked` is ignored rather than rejected. The request line's
`sscanf` return is also discarded, so a malformed request line becomes a 404
instead of a 400. Severity is limited today by one-request-per-connection and a
loopback bind, but it becomes a desync primitive behind a keep-alive proxy —
which is the documented llama-swap deployment. A line-based parser that requires
exactly one valid length is the fix.

---

# Phase 1: Strict Tool-Call Schema Engine

## Goal

Guarantee valid tool names and arguments during sampling rather than parsing
hopeful model output afterward.

## Work

- **DONE** Compile OpenAI `tools[].function.parameters` into Runner schemas.
- **DONE** Synthesize a discriminated union containing one branch per tool.
- **DONE** Add a `final` branch for ordinary or schema-constrained final
  responses. A `response_format` schema in the same request becomes the shape
  of that branch, so tools and structured output compile to one union.
- **DONE** Support `tool_choice`: `auto`, `required`, `none`, and named
  function. `none` declines strict mode rather than inventing a constraint.
- **DONE** Reject unknown tools and arguments during token sampling.
- **DONE** Guarantee valid arguments when `max_tokens` truncates generation
  (`sval_close` completes the envelope to the schema's minimum).
- **DONE** Map the generated envelope back into standard OpenAI `tool_calls`.
- **DONE** Initially support one call with `parallel_tool_calls:false`;
  `parallel_tool_calls:true` is rejected rather than silently ignored.
- Add bounded multi-call arrays for `parallel_tool_calls:true`.
- **DONE** Remove dependence on post-generation tag parsing for strict
  requests while retaining the existing parser as a compatibility fallback.
  ~~Done on the buffered path. Streaming requests still take the legacy
  declare-and-parse path, because emitting the envelope as `tool_calls`
  deltas is Phase 2 work; until then a stream would send the envelope to the
  client as raw content.~~ Phase 2 closed the streaming half: both paths now
  compile the same envelope, and `tool_calls_parse` remains only as the
  fallback for requests that declare no tools.

## Exit Criteria

- **DONE (buffered)** Tool arguments always conform to their declared JSON
  Schema.
- **DONE (buffered)** The model cannot invent a tool name.
- **DONE (buffered)** `finish_reason:"tool_calls"` is emitted correctly.
- **DONE (buffered)** Truncated tool calls remain valid and executable.
- **DONE (buffered)** Tools and `response_format` work in the same request.

All five hold on the buffered path and are covered by
`tests/test_tools.c` (envelope compiled and driven through the real `sval`
validator) and `tests/conformance/test_tool_calls.py` (the same guarantees
asserted over HTTP). ~~The streaming path is Phase 2 and is unchanged.~~
Phase 2 extended all five to the streaming path, which now runs the same
envelope; the "(buffered)" qualifiers above no longer limit them.

## [carried over] Conformance holes to close here

A review of the current surface against the "never silently ignore" invariant
found these. The two response_format holes and the non-boolean `stream` were
already fixed; the two marked DONE below closed with this phase; the rest
remain:

- **DONE — ~20 JSON Schema keywords are accepted and ignored**, including
  `pattern`, `minimum`/`maximum`, `additionalProperties`, `allOf`, `$ref` and
  `format`. Two were actively wrong rather than merely absent: `pattern` was
  dropped, so a constrained string was unconstrained in exactly the way the
  caller asked it not to be; and `additionalProperties: true` was dropped
  while the compiled object enforces a *closed* property set, making output
  stricter than the schema permits. An object schema with `required` but no
  `properties` silently dropped `required` entirely.
  `schema.c` now checks every node against a type-aware allow-list and
  rejects anything it cannot enforce. Pure annotations (`title`,
  `description`, `default`, `examples`, ...) stay legal because they carry no
  constraint. `additionalProperties: false` alongside declared properties is
  accepted, being exactly what the compiled machine already does.
- **DONE — Wrong-typed request values fall back to defaults instead of
  erroring.** `request_number` could not distinguish absent from wrong-typed,
  so `"temperature": "0.7"` silently became the server default. It, plus
  `request_max_tokens`, `request_keep_alive` and the `stream` / `logprobs`
  flags, now reject a value of the wrong type. `null` deliberately still
  reads as absent, because mainstream OpenAI SDKs serialise unset optional
  fields that way.
- **Silently ignored fields**: `n`, `frequency_penalty`, `presence_penalty`,
  `logit_bias`, `user`, and on embeddings `encoding_format` and `dimensions`.
  (`repeat_penalty` was on this list and is now an honoured request field.
  `tool_choice` and `parallel_tool_calls` are now honoured on the buffered
  path — `parallel_tool_calls:true` is rejected rather than ignored — ~~but are
  still ignored on the streaming path, which keeps the legacy tool handling
  until Phase 2.~~ Phase 2: both are honoured on the streaming path too.)
- **`logprobs` is honoured only on non-streaming chat**, silently dropped on
  streaming and on `/v1/completions`, and `top_logprobs` is not range-checked
  unless `logprobs` is set.
- **Negative `max_tokens` is accepted as "unlimited"** rather than rejected.
- **Error objects omit `param` and `code`**, so clients cannot branch on
  `context_length_exceeded`. Unknown model returns 400 where OpenAI returns 404;
  context overflow returns 500 for what is a request-caused error; a full
  request queue closes the connection with no HTTP response at all.

## Release

`v0.2.0-alpha`: Guaranteed tool calls for local GGUF models.

---

# Phase 2: Correct Streaming Agent Events

## Goal

Make streaming tools as reliable as buffered tools.

## Work

- **DONE** Detect the selected tool branch while generation is still running.
  `tool_stream` (template.c) is the streaming counterpart of
  `tool_envelope_map`: it holds bytes back until the `tool` discriminator
  resolves, then forwards everything after it to the channel that branch
  selected. The call is announced as soon as the name is known.
- **DONE** Stream standards-compatible `tool_calls[].function.arguments`
  deltas. Argument text is forwarded raw with insignificant whitespace
  dropped, so the concatenated deltas are the same document the buffered path
  re-serializes.
- **DONE** Never leak internal tool tags into `content`. Holding the undecided
  prefix back is what guarantees this: by the time any byte is forwarded it is
  already known to be assistant text or argument text.
- **DONE** Preserve separate `reasoning_content`, content, and tool-call
  channels. The thinking splitter runs upstream of the demux, so reasoning
  never enters the envelope document.
- **DONE** Emit stable call IDs, indexes, function types, and terminal events.
  Identity (`index`, `id`, `type`, `function.name`) is sent once in an opening
  delta; later deltas carry argument text keyed by the same index. A streamed
  call reaches `finish_reason:"tool_calls"` and reuses the buffered path's
  `call_N` id scheme, so both paths report the same id.
- **DONE** `created` and `model` on every chunk, and `"role":"assistant"` on
  the first delta (sent as its own opening chunk, so the contract holds even
  when the model generates nothing).
- Support multiple sequential or parallel calls without index drift.
  *(Not started. `parallel_tool_calls:true` is still rejected rather than
  ignored, and the envelope is one call per turn, so `index` is always 0. The
  streaming events are already index-keyed, which is the shape a multi-call
  envelope will need.)*
- **DONE** Handle client disconnects without leaving incomplete slot state. A
  sink returning non-zero propagates out of `tool_stream_feed` and aborts
  generation exactly as a dead content channel already did;
  `completion_cleanup` frees the demux on every exit path.
- Verify structured output with speculative decoding enabled. *(Not started.)*

## [carried over] Known streaming defects

- **DONE** ~~`tool_calls` are unreachable when streaming.~~ `tool_calls_parse`
  was called only on the buffered path, so a streaming request that triggered a
  tool call streamed the raw internal marker to the client as ordinary content
  and reported `finish_reason: "stop"`. Streams now compile the same strict
  envelope as buffered requests and demultiplex it as it is generated;
  `"tool_calls"` is reachable on a stream.
- **DONE** ~~Chunks omit `created` and `model`~~, which the ChatCompletionChunk
  schema requires and the non-streaming path does emit; ~~the first delta also
  omits `"role":"assistant"`~~. All three are now emitted.

## Exit Criteria

- **DONE** Buffered and streamed requests produce equivalent normalized
  results. Asserted directly by
  `test_streamed_and_buffered_calls_are_equivalent` and
  `test_streamed_final_branch_matches_the_buffered_answer` in
  `tests/conformance/test_tool_calls.py` — same finish_reason, same tool, same
  call id, same arguments document, same final-branch text.
- OpenAI Python, TypeScript, and Vercel AI SDK clients accept every event.
  *(Not verified against the real SDKs; the wire shape they require is
  asserted by the conformance suite, but no SDK is exercised in CI.)*
- **DONE** Every possible token/chunk split passes the streaming test matrix.
  `test_stream_tool_call_survives_every_split_point` runs the byte-boundary
  matrix over a stream that carries a tool call, and `test_tools.c` runs the
  same property one level down, feeding the demux every chunking of the
  envelope and requiring identical output.

---

# Phase 3: OpenAI Responses API

## Goal

Make Runner directly usable by modern agent clients, including Codex-style
clients, without a translation proxy.

## Work

- **DONE** Add `POST /v1/responses`. It is a translation layer over the
  existing engine, not a second one: `handle_responses` (server.c) reshapes
  the request into the same prompt and the same `tool_envelope_build` union
  `/v1/chat/completions` builds, and `run_completion`'s `bool chat` became an
  `API_*` dialect so every line that decides *what* is generated is shared.
  Phase 4 can hang Anthropic off the same seam.
- **DONE** Support `instructions`, `input`, function tools, `tool_choice`,
  `parallel_tool_calls`, `text.format`, reasoning fields, and `store:false`.
  `instructions` becomes a system turn; `input` accepts a string or an item
  array; tools are accepted in the flat Responses shape, the nested chat
  shape, and inside a `namespace` group (which Codex sends); `text.format`
  resolves through the same `request_schema` entry point as
  `response_format`. `parallel_tool_calls:true` is still refused rather than
  ignored — the envelope is one call per turn, as on the chat surface.
  `reasoning` is accepted and echoed rather than refused: `effort` is a hint
  about how much thinking to do, not a promise about the response document,
  and a thinking-tag model's channel already comes back as a `reasoning`
  output item. A malformed one is still a 400.
- **DONE** Emit ordered Responses events: created, output-item added,
  content-part added, text/tool deltas, item completion, and response
  completion. Framed in one place (`resp_send`) so the SSE `event:` name and
  `data.type` cannot drift apart and `sequence_number` cannot drift from send
  order. `response.incomplete` replaces `response.completed` when
  `max_output_tokens` cut the turn short, and the item is then marked
  `incomplete` too — a client rendering `output[]` without reading the
  response status would otherwise show a truncated message as a finished one.
- **DONE** Support function-call outputs in subsequent requests.
  `function_call` and `function_call_output` input items render into the
  conversation the same way replayed chat `tool_calls` history does.
  `test_function_call_output_is_accepted_in_a_later_request` asserts the
  result actually reaches the prompt (via input-token growth) rather than
  merely being accepted.
- **DONE** Preserve usage, cached-token, finish-reason, and Runner telemetry.
  `usage.input_tokens_details.cached_tokens` and `runner_telemetry` ride on
  both the buffered body and the terminal stream event.
- **DONE** Add a `/v1/responses` conformance test suite.
  `tests/conformance/test_responses.py`, 37 tests. The suite grew from 134 to
  171.
- **DONE** Document a working Codex custom-provider configuration. README,
  "Responses API" section — verified against the real `codex-cli` 0.144.6, not
  written from the spec.

## Exit Criteria

- **DONE** Codex can complete a local tool-call loop through Runner. Verified
  with the real `codex-cli` 0.144.6 against Qwen3-4B: Codex emitted an
  `exec_command` function call, ran it, fed the `function_call_output` back,
  and Runner answered the follow-up turn — three requests, with the prefix
  cache reusing 9943 of 9962 prompt tokens on the second.
  Getting there needed three fixes that only a real client exposed, all of
  them shapes a from-the-spec reading does not predict:
  - Codex groups function tools under a `{"type":"namespace","tools":[...]}`
    entry. A namespace is a container, not a tool, so it is flattened.
  - Codex declares `web_search` with `external_web_access:false` even when web
    access is off. That is the client stating a capability is *disabled*, not
    asking for it, so it is dropped; an enabled hosted tool is still refused.
  - Codex's zero-argument tools declare
    `{"properties":{},"additionalProperties":false}`. schema.c rejected this,
    because an *absent* properties map with `additionalProperties:false`
    genuinely cannot be compiled — but an empty one is exactly expressible
    (an object with no permitted keys, i.e. `{}`), and now compiles to a
    zero-property `SN_OBJ`. See `test_schema_empty_closed_object`.
- **DONE** OpenAI SDK Responses clients work in buffered and streaming modes.
  Verified against the real `openai` 2.46.0 Python SDK, not only asserted on
  the wire: `client.responses.create` parses into a `Response`, and
  `client.responses.stream` deserialises every event into its typed class
  (`ResponseCreatedEvent`, `ResponseOutputItemAddedEvent`,
  `ResponseTextDeltaEvent`, `ResponseFunctionCallArgumentsDeltaEvent`,
  `ResponseCompletedEvent`, …) with `get_final_response()` returning the
  parsed turn including its function call. Note that the SDK's
  `get_final_response()` raises on a truncated stream because it has no
  `response.incomplete` branch — that is SDK behaviour against real OpenAI
  too, not a Runner defect.
- **DONE** Unsupported stateful features return clear errors rather than being
  ignored. `store:true`, `previous_response_id`, `background:true`,
  `conversation`, `truncation:"auto"`, `include[]` and enabled hosted tools
  each return a 400 naming the field and why.

## Not done in this phase

- Multiple parallel tool calls in one turn. `parallel_tool_calls:true` is
  refused on both surfaces; the envelope is still one call per turn, and the
  Responses items are already `output_index`-keyed, which is the shape a
  multi-call envelope will need.
- No C-level unit test covers the Responses event state machine. It is static
  in server.c, which no test binary links, so it is asserted through the HTTP
  surface instead (per AGENTS.md "test through public interfaces"). The
  ordering assertions were mutation-checked: deleting the terminal
  `resp_close_item` call fails four conformance tests.

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

## [carried over] Server lifecycle defects to fix with this refactor

These all stem from state that is global today and becomes per-sequence here:

- **`SV.busy` is a single global flag** written by every slot. It is safe only
  because swap mode implies one slot, and nothing encodes that dependency. With
  a registry and `parallel > 1`, one slot finishing clears `busy` while another
  is still generating, and the TTL reaper can then unload the model underneath
  it. A per-sequence refcount is what this actually needs.
- **`last_used` is stamped at request start**, so any generation longer than the
  TTL is followed by an immediate unload of a model that just finished working.
- **No graceful shutdown exists.** `SV.q.shutdown` is read but never written, so
  `q_pop` never returns -1 and the slot-worker exit path is unreachable; there
  is no SIGINT/SIGTERM handler, and the accept loop returns without closing the
  listener, joining slot threads, or answering queued connections.
- **`swap_to` holds `swap_mu` across a full model load**, blocking `/unload` for
  seconds.
- **`/unload` does not release the draft model**, and `context_load()` keeps
  reporting the previous model's geometry after an unload.

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

- **DONE (CUDA) / NOT DONE (Metal)** — Port Q8 KV storage and attention reads
  to CUDA and Metal. CUDA is done: `k_store_kv` quantizes rows on device and
  `k_attn` / `k_attn_dec` read q8_0 blocks (`src/kernels.cu`, regenerated into
  `src/kernels_ptx.h`). All KV offsets in `cuda.c` are now byte offsets from
  `model_kv_row_bytes` / `model_kv_byte_off`, so host and device share one
  layout and the hybrid CPU/GPU split works unchanged. Metal is **not** ported:
  `metal.m` gained only `gpu_kv_q8_ok()` returning false, which makes `--kv q8`
  fall back to f16 (with a stderr note) rather than hand q8_0 blocks to an fp16
  reader. This was not compiled or run — no macOS here.
- **DONE** — Support configurable K/V cache types. `--kv f16|q8` now applies to
  the CUDA backend too (it previously required `--gpu off`), and `--caps`
  publishes `"kv_types":["f16","q8"]` with `"kv_type_default":"f16"`. Guarded
  by a `make smoke` assertion so the default cannot drift silently.
- **PARTIAL** — Measure quality against fp16. Measured with the committed
  deterministic gates (`scripts/kernel-verify.py`, extended with
  `--baseline-args` / `--candidate-args` so one binary can be compared against
  itself in two configurations) over 7 models x 20 prompts x 48 greedy tokens.
  Retrieval and structured-agent workloads were **not** run — see Remaining.
- **DONE** — Include quantized KV in reservation and auto-fit calculations.
  The `-c 0 --reserve*` auto-fit and the CUDA layer-split budget both size the
  cache with `model_kv_row_bytes`, so q8 buys context and offloaded layers.
- **NOT DONE** — Investigate mixed/per-phase precision. Not started; Q8 is only
  just stable.

## Measured results (24 GB MIG slice of an RTX PRO 6000, CUDA)

Context gained where VRAM is the binding constraint (`-c 0 --reserve-vram P`).
q8_0 stores 32 values in 34 bytes = 53.1% of fp16, so the ceiling is 1.88x:

| model | reserve | f16 ctx | q8 ctx | gain |
|---|---|---|---|---|
| Llama-3.2-3B | 20% | 24288 | 45719 | 1.88x |
| Llama-3.2-3B | 30% | 46407 | 87354 | 1.88x |
| Llama-3.1-8B | 40% | 37824 | 71198 | 1.88x |
| Llama-3.1-8B | 60% | 76531 | 131072 (train cap) | >=1.71x |
| Mistral-7B | 30% | 22650 | 32768 (train cap) | >=1.45x |

Cache bytes at a fixed context are 53.1% of fp16 in every case measured
(e.g. Llama-3.1-8B at 131072: 17179.9 MB -> 9126.8 MB).

Throughput: prefill is consistently 2-4% slower with q8 (Llama-3.2-3B
145.4 -> 142.4 tok/s, Mistral-7B 81.5 -> 79.2, three runs each). Decode could
**not** be separated from run-to-run noise on this shared MIG slice: repeated
identical f16 runs ranged 26.8-65.8 tok/s. Decode cost of q8 KV is therefore
UNMEASURED here and needs an uncontended GPU.

## Quality delta (honest)

Divergence over 20 prompts x 48 greedy tokens, per model:

| model | f16 CPU vs f16 GPU | q8 CPU vs q8 GPU | f16 GPU vs q8 GPU |
|---|---|---|---|
| TinyLlama-1.1B | 0/20 | 0/20 | 4/20 |
| SmolLM2-1.7B | 0/20 | 0/20 | 1/20 |
| Llama-3.2-3B | 0/20 | 1/20 | 3/20 |
| Qwen3-4B | 1/20 (pre-existing) | 4/20 | 4/20 |
| gemma-3-4b | 0/20 | 1/20 | 3/20 |
| Phi-3.5-mini | 0/20 | 4/20 | 2/20 |
| Mistral-7B | 0/20 | 0/20 | 0/20 |

Readings:

- **The project invariant holds.** fp16 GPU is still token-identical to fp16
  CPU everywhere it was before. The single Qwen3-4B exception reproduces
  identically on the pre-change binary, so it is pre-existing, not a
  regression — but it means the fp16 invariant is already only gate-tight at
  the committed 5-prompt x 48-token gate, not universally.
- **Q8-KV GPU vs Q8-KV CPU is not a stable invariant, and cannot be made one.**
  Every divergence is a single late branch point (typically 25+ tokens in) with
  both continuations fluent — a greedy near-tie, not a broken kernel. Three
  independent lines of evidence:
  1. On gemma-3-4b the exact same divergence is reproducible with **no GPU
     involved at all**: CPU-only, `-b 64` vs `-b 1` (a mathematically
     equivalent reassociation) flips the same prompt under `--kv q8` and does
     not under `--kv f16`.
  2. A 6/34-layer hybrid split under q8 produces the *identical* alternative
     continuation as a full GPU offload. A layout or scale bug in 6 GPU layers
     could not yield a fluent alternative through 28 CPU layers.
  3. Divergent positions largely correlate across columns — Llama-3.2-3B
     offset 157, gemma-3-4b offset 73, Qwen3-4B offsets 154/23 and Phi-3.5
     offset 119 each appear in both the CPU-vs-GPU and the f16-vs-q8 column.
     The same positions are fragile under any perturbation. The correlation is
     not total (Phi-3.5 offsets 105/144/70 appear only in the CPU-vs-GPU
     column), which is expected: the two perturbations are different, so they
     tip different subsets of the near-ties, and Phi-3.5 is the one model whose
     CPU-vs-GPU rate (4/20) exceeds its f16-vs-q8 rate (2/20). Phi-3.5 is
     therefore the weakest case in this evidence set and the first place to
     look if a real q8 kernel bug is ever suspected.
  Root cause: q8_0's quantization step is ~1/127 of each 32-value block's
  absmax, roughly 15x coarser than fp16 near the same magnitude. Last-ulp FP
  differences that fp16 absorbs can flip a whole quant level under q8, so
  ordinary reassociation noise reaches the logits. The device-side
  quantization arithmetic itself mirrors `q8_quant_row()` exactly (amax/127,
  `roundf`, RN fp16 scale), so identical inputs give bit-identical blocks.

## Exit Criteria

- **MET** — Q8 KV substantially increases usable context on limited VRAM:
  1.88x measured on CUDA, and it is wired into both auto-fit and the layer
  split.
- **NOT MET** — Retrieval and tool-selection regressions are measured and
  documented. Only greedy token divergence was measured. No needle-in-haystack
  retrieval run and no tool-selection accuracy run; the numbers above say how
  *often* output changes, not whether it gets *worse*.
- **PARTIALLY MET** — GPU and CPU implementations pass deterministic
  correctness gates. fp16 GPU == fp16 CPU still passes everywhere. Q8 GPU ==
  Q8 CPU passes on 5 of 7 models at the committed 5-prompt gate but is not a
  sound gate in principle (see above); it needs a tolerance-based gate rather
  than a token-identity one.

## Remaining for Phase 8

1. Metal port of q8 KV storage and attention (`metal.m`, `kernels_metal.h`).
   Currently stubbed off, untested, no hardware here.
2. A quality gate appropriate to a lossy cache: retrieval (needle-in-haystack
   at 32k+) and tool-selection accuracy, f16 vs q8, reporting task success
   rather than token identity.
3. Uncontended decode benchmark for the q8 attention read path.
4. Mixed/per-phase precision (e.g. q8 K with f16 V), untouched.
5. `spec_draft_load()` in `engine.c` still forces `kv_q8 = false` for the draft
   model. That was required when q8 implied CPU-only; it is now merely
   conservative and could be revisited.

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

Status is tracked here; nothing is removed when finished, only marked.

1. ~~Conformance harness.~~ **DONE** — `tests/conformance/` (127 tests) plus its
   own CI job. Remaining from Phase 0: real-model scripts, SDK compatibility
   smokes, and the peak-VRAM figure.
2. ~~Strict tool schemas and structured final responses.~~ **DONE** — unknown
   schema keywords rejected, wrong-typed scalars 400, tools compile to a
   discriminated union. Buffered path only; streaming is item 3.
3. ~~Correct streaming tool calls.~~ **DONE** — tool_calls stream as
   incremental argument deltas, finish_reason "tool_calls" reachable, chunks
   carry created/model/role, no envelope leaks into content. Single call per
   turn; multi-call and real SDK acceptance testing remain.
4. `/v1/responses`. **NEXT** — first public traction milestone.
5. Anthropic Messages compatibility.
6. Shared CUDA weights.
7. Continuous batching.
8. Persistent/forkable KV prefixes.
9. GPU Q8 KV cache. **PARTIAL** — Q8 KV works on CUDA, 1.88x context where
   VRAM binds, fp16 still the default. Metal is a fallback stub (unverified),
   and the retrieval/tool-selection quality gate is NOT met: we measured how
   often output changes, not whether it gets worse.
10. Public comparative torture suite and ongoing compatibility releases.

The first public traction milestone is Phase 3. The largest serving-performance
milestone is Phase 6. The strongest long-term specialization is the combination
of strict tools, forkable agent prefixes, and quantized long-context KV.

---

# [carried over] Work Outside the Phase Plan

These predate this roadmap and are not scheduled by it. Listed so they are not
lost; slot them in deliberately.

## DONE — OOM crash bugs

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

## DONE — fuzz and property tests

`HANDOVER.md` §2 contains a ready design: six targets in priority order with
entry points, a `tests/fuzz/` layout, a bounded `make fuzz`, and the rule that a
real parser crash should be **reported with its reproducing input, not quietly
patched**. Every parser in Runner is hand-written and eats untrusted input, which
makes this the highest-value hardening work available.

Two notes: the design predates this session, so re-read `src/tokenizer.c` before
writing a `tok_encode` harness — it changed substantially. And the three
`json.c` bugs above should reproduce immediately; if a harness does not find
them, it is not reaching the code.

## Done — phi3 architecture support

Shipped. The weight wiring was correct throughout; what blocked it was the
repeat penalty suppressing `<|end|>`, since the chat template puts the turn
terminator in the prompt and the prompt seeds the penalty window. Stop tokens
are now exempt from the penalty. The earlier note here — "something downstream
of layer 0 is wrong" — was mistaken and is kept only as a caution: layer-0
numerics being exact did not imply the forward pass was the problem, and the
bug was two modules away in `sample.c`.

## Done — per-family sampling defaults

Shipped. `sample.c` owns a preset table keyed off the GGUF's
`general.architecture` and `general.name` (three families all declare `llama`,
so the name is what separates Llama-3.x from Mistral from SmolLM2). Each entry
cites its source in a comment and in `--caps`. Presets are visible rather than
magic: logged at load, listed by `runner --caps`, and reported per served model
by `GET /v1/capabilities`. An explicit CLI flag or request field always wins,
and the CLI overrides survive a model swap.

Both review findings were addressed:

- **The greedy contract changed.** `temp <= 0` now returns the model's argmax
  with no repeat penalty applied. Greedy is a determinism request, and a
  penalty that can move the answer off the argmax defeats it. The stop-token
  exemption stays for the sampling path, where it is still needed.
- **The penalty's logit-magnitude sensitivity is handled in the table.** A
  penalty p shifts a logit x by x*(1 - 1/p), so 1.1 at Llama's ~+20 peak is a
  ~1.8 shift, and matching that at Phi-3.5's ~+65 needs ~1.03 — which is phi3's
  preset value. It is the one number in the table that is calibration rather
  than citation, and it is commented as such.

`repeat_penalty` also became a request-level field, closing the hole noted
below: a client can now ask for no penalty by sending `1`. Zero is rejected
rather than read as "off", since the penalty divides by it.

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
