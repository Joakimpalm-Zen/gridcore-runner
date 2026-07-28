# Gridcore Runner full code review — July 2026

Review date: 2026-07-23  
Reviewed revision: `e3cc496815897f9c163f514df7385ccfc3539b08` (`main`)  
Scope: the complete `gridcore-runner` repository at that revision  
Mode: review only; no product-code fixes were made

## Executive summary

Runner has a notably strong conformance suite, careful HTTP framing, useful synthetic model fixtures, and a compact deployment story. The CPU/CUDA happy paths are much better tested than is usual for a small native inference engine. The review nevertheless found release-blocking defects in ownership, backend fallback, model admission, quantization, and allocation-failure handling.

The highest-risk issue is Metal runtime fallback: detaching the backend loses the owner of the Metal KV buffers, after which `model_free` can call `free()` on pointers returned by `MTLBuffer.contents`. The quantizer can also emit structurally corrupt GGUFs when the source declares an alignment other than 32. In long-lived swap serving, failed model/tokenizer loads leak partial state. An unknown architecture is attempted as Llama rather than refused, which creates a silent-wrong-inference failure mode. The continuous-batching worker can disappear on startup OOM while callers continue waiting.

I would not treat the current revision as ready for another public alpha until findings RNR-001 through RNR-006 are resolved and covered by failure-path tests. The current green tests do establish substantial correctness on their covered paths; they do not exercise the failure modes below.

### Finding count

| Severity | Count |
|---|---:|
| Critical | 1 |
| High | 6 |
| Medium | 10 |
| Low | 5 |
| **Total** | **22** |

Severity describes likely impact, not implementation effort. “Critical” means memory corruption or similarly unsafe behavior in a supported configuration. “High” includes silent incorrect inference, persistent resource exhaustion, deadlock, or corrupt model output. “Medium” includes bounded reliability, portability, protocol, and release-integrity failures. “Low” covers maintainability and diagnostics that should be tracked but do not presently threaten normal inference.

## Release blockers

### RNR-001 — Critical — Metal runtime fallback loses KV ownership and can invalid-free at shutdown

Locations: `src/metal.m:336-344`, `src/model.c:688-723`, `src/model.c:1067-1074`

Metal replaces the model’s malloc-owned KV pointers with `g->kc.contents` and `g->vc.contents`. On a runtime GPU error, `gpu_disable()` only assigns `m->gpu = NULL`. That makes the `gpu_t` object and all retained Objective-C resources unreachable. Later, `model_free()` calls `gpu_free(m)`, which now does nothing, and then calls ordinary `free(m->kcache)` and `free(m->vcache)` on the Metal buffer-content pointers.

Impact:

- the runtime failure leaks every Metal resource associated with the model;
- CPU fallback continues through borrowed buffer memory whose owner is lost;
- normal shutdown can invalid-free non-malloc pointers, causing a crash or allocator corruption.

The comments promise that the buffers are freed by `model_free`, but clearing the only backend pointer makes that impossible. The backend needs an explicit ownership state that preserves the Metal resource owner through CPU fallback and makes final destruction unambiguous. Add an injected command-buffer failure test on real Apple hardware under AddressSanitizer or Guard Malloc.

### RNR-002 — High — The quantizer ignores `general.alignment` and can produce corrupt GGUF files

Locations: `src/quantize.c:197-249`, `src/gguf.c:153-159`

The quantizer copies all source metadata verbatim, including `general.alignment`, but hard-codes every output tensor offset and the output data-section start to 32-byte alignment. If the input declares 64, 128, or another valid alignment, the output still advertises that alignment while placing data at 32-byte boundaries. Runner’s loader honors the advertised value and therefore computes a different `data_start` from the writer.

Impact: a successful quantization command can create a file that is misparsed or rejected by Runner and other GGUF consumers. This is deterministic for a non-32-aligned source, not merely an OOM edge case.

The writer must either preserve the declared alignment physically or rewrite the metadata consistently. Add round-trip tests with at least 32-, 64-, and 128-byte input alignment and validate the result with an independent GGUF reader.

### RNR-003 — High — Partial model and tokenizer failures leak in long-lived swap serving

Locations: `src/model.c:238-685`, `src/tokenizer.c:161-263`, `src/server.c:380-413`, `src/server.c:3915-3923`, `src/main.c:386-392`

`model_load()` and `tokenizer_init()` allocate progressively and return `false` at many points without rolling back. This is workable only if every caller always calls the corresponding free function on a partially initialized object. Callers do not follow that contract:

- swap mode calls `model_free(m)` only when `model_ok` is true and `tokenizer_free(tok)` only when `tok_ok` is true;
- parallel-slot startup returns immediately after a failed `model_load()` without freeing the partial object;
- CLI startup returns on either failure without cleanup.

In the CLI the process exits and the OS reclaims memory. In swap mode the server remains alive, so repeated requests for a malformed or unsupported registered model can accumulate mappings, heap allocations, thread pools, and reservation state.

Define and enforce one ownership contract: either loaders are transactional, or their API explicitly guarantees that the matching destructor is safe and mandatory after every call. Add late-failure fixtures (missing late-layer tensors, tokenizer failure after allocations, GPU initialization failure) and assert stable mappings, heap, threads, and VRAM-registry state across repeated swap attempts.

### RNR-004 — High — Unknown architectures are allowed to execute as Llama

Location: `src/model.c:249-267`

The loader explicitly rejects three known-incompatible architectures, but any other unrecognized `general.architecture` only produces a warning and proceeds through Llama-style wiring and math. Architecture differences routinely include Q/K layouts, normalization, scaling, position encoding, attention variants, activation functions, and output transforms. Tensor-name compatibility is not proof of mathematical compatibility.

Impact: a future or misspelled architecture can load and produce plausible-looking but incorrect output. Silent wrong inference is more dangerous than a clear unsupported-model error, especially for an engine used by agents.

Admission should be an explicit allowlist of implementations with architecture-specific validation. Experimental fallback, if retained, should require an opt-in flag and must not be represented as supported.

### RNR-005 — High — Continuous-batching startup failures can leak or leave requests waiting forever

Locations: `src/server.c:612-619`, `src/server.c:683-706`, scheduler wait paths following `sched_forward`

If the decode worker cannot allocate its four temporary arrays, it simply returns. `SCH.running` remains true; no stop/error state is recorded and no condition variable is broadcast. Requests can continue entering the scheduler and wait forever when no request deadline is configured (the default).

`sched_start()` also leaks previously allocated per-sequence logits and the batch object on later allocation or synchronization-primitive failures. `sched_shutdown()` is keyed on `running`, so it cannot reliably clean partially initialized state. Thread-creation failure similarly leaves the built scheduler behind.

Startup must publish “running” only after the worker has acknowledged successful initialization, and all partial states need one idempotent destructor. The worker needs a terminal failure state that wakes every waiter. Add deterministic allocator and `pthread_create` fault-injection tests.

### RNR-006 — High — Unchecked allocations can crash inference or silently change the prompt

Locations include `src/sample.c:172-174`, `src/model.c:16-28`, `src/model.c:73-147`, `src/model.c:287-299`, `src/model.c:367-388`, `src/model.c:580-628`, `src/model.c:1405-1416`, `src/engine.c:62`, and tokenizer temporary allocations in `src/tokenizer.c:285-286`, `338-339`, `627-628`, `694-698`, `737-738`

Concrete cases:

- the sampler’s full-candidate allocation is immediately written and sorted without a null check; GCC `-fanalyzer` independently reported the null dereference;
- tensor conversion allocates and immediately copies/dequantizes into the result;
- RoPE and per-layer geometry arrays are allocated and immediately indexed;
- `model_embed()` uses its temporary vector without checking it and ignores forward failures;
- KV offset storage is written before the later aggregate allocation check;
- tokenization helpers sometimes return the previous output count after temporary-allocation failure, which can make a request continue with omitted text instead of reporting failure;
- `engine_init()` permits a null history buffer, but speculative decoding later assumes history exists.

Impact ranges from process termination under memory pressure to successful inference over a truncated prompt. The latter is a semantic integrity failure. Public operations need explicit failure returns; allocations and size products must be checked before use; tokenization must never represent OOM as valid partial encoding.

### RNR-007 — High — Published compatibility claims exceed and contradict current evidence

Locations: `README.md:127-128`, `README.md:694`, `src/model.c:337-350`, `docs/compatibility-program.md:30,44,85`, `tests/compatibility/models.json`, `tests/compatibility/out/2026-07-22-v0.1.2-alpha.json`, `tests/compatibility/out/chat-tool-long-context-2026-07-23.json`

The README and runtime log state that Gemma 4 is token-identical to llama.cpp. The current committed reference-comparison evidence reports 0/5 for Gemma 4 at the newer reference revision, while comments cite an older b9964 result. Current reference outcomes also include Llama 3 at 3/5, Qwen 3 and Phi 3 at 4/5, and current chat/tool/long-context evidence has Qwen 3 at 0/3 and Ornith at 2/3.

Evidence provenance is also fragmented:

- the compatibility manifest declares load, tokenizer, greedy-reference, CPU/CUDA, chat, tool, and long-context checks, but `compat_matrix.py` implements only hash verification and load execution;
- other checks are supplied by independent scripts and manual aggregation, so the manifest is not an executable gate;
- a report documented as historical/immutable was updated with later results while retaining earlier Runner version/commit identity;
- the chat/tool/long-context artifact contains no schema version, exact timestamp, Runner commit/version, model hashes, hardware, command, or configuration;
- that focused run was produced by in-process mutation of script case lists rather than a committed CLI configuration, so it is not directly reproducible from the artifact.

This does not prove every divergence is an engine defect—some are model behavior, token budgets, or reference-version changes—but it does mean the broad release claims are not supported by the current evidence. Versioned reports should be append-only, self-identifying, reproducible, and generated by one gate that refuses to mark unexecuted checks complete.

## Other correctness, reliability, and portability findings

### RNR-008 — Medium — `load_cancel` is a cross-thread C data race

Locations: `src/server.c:237-263`, `src/server.c:329-340`, `src/server.c:388-400`, `src/server.c:4052-4056`, `src/vramreg.c:285-329`

`SV.load_cancel` is `volatile int`, written by request/shutdown paths and read by a loading thread through `const volatile int *`. `volatile` does not make cross-thread access atomic or establish a happens-before relationship. Concurrent access is undefined behavior and can miss or delay unload/shutdown cancellation. Use a C atomic or protect every access with a shared lock; update the API so it does not advertise volatile as synchronization.

### RNR-009 — Medium — WinSock handles are truncated to `int`

Locations: socket and queue declarations throughout `src/server.c`, especially socket creation near `3964` and `accept()` at `4027`

Windows `SOCKET` is `UINT_PTR`, but the server stores listener/client sockets and queue entries as `int`, casting the result of `socket()` and `accept()`. On 64-bit Windows this can truncate a valid handle. The same abstraction also checks `errno` in places where WinSock requires `WSAGetLastError`, degrading diagnostics.

Use a platform socket type (`SOCKET` on Windows, `int` on POSIX) throughout the queue and request functions, with platform-specific invalid/error helpers. Add a real Windows lifecycle/concurrency test rather than compile-only confidence.

### RNR-010 — Medium — GGUF typed getters accept incompatible scalar types without range validation

Locations: `src/gguf.c:214-239`

`gguf_get_u32()` accepts almost every scalar type, casts floating-point values directly, and reads signed integer values through the unsigned union member. `gguf_get_f32()` similarly interprets signed values as unsigned. `gguf_get_bool()` returns the bool union member for any metadata type. Negative, out-of-range, NaN, or type-confused geometry can therefore become huge or implementation-dependent values before downstream checks.

Typed getters should accept an explicit compatible-type set and validate sign, range, and finiteness. Duplicate metadata keys and duplicate tensor names should also be rejected rather than silently resolving to the first entry.

### RNR-011 — Medium — CLI file input and token-buffer arithmetic are not robust

Locations: `src/main.c:329-339`, `src/main.c:423-445`, `src/main.c:455-456`, `src/main.c:56-75`

Prompt/schema loading does not check `fseek`, `ftell`, negative sizes, allocation failure, or short/error reads. Signed `long` sizes are mixed into unsigned allocation expressions. Token capacity adds prompt bytes, model context, and a constant without overflow checks, allocates without checking, then narrows capacity to `int`. `unescape()` also writes through an unchecked allocation.

These paths are local rather than remotely exposed, but malformed special files, oversized inputs, or memory pressure can crash the process or invoke undefined behavior. Centralize bounded file reading and checked-size arithmetic, and carry token capacities as `size_t` until a validated API boundary.

### RNR-012 — Medium — Successful CLI paths intentionally leak all model/engine state

Locations: early returns throughout `src/main.c:386-565`

One-shot, benchmark, interactive, schema-error, and many other exits return without destroying the engine, draft model, schema, tokenizer, token/prompt buffers, or model. This is masked by process exit and even called out in `src/cuda.c`’s profiling comment.

An AddressSanitizer/LeakSanitizer CPU smoke on the tiny fixture completed inference but exited nonzero with **252,819 bytes in 307 allocations** still reachable/leaked. This is not a long-lived production leak today, but it obscures real leak testing, makes the control flow brittle, and prevents safe reuse of the CLI orchestration in an embedding. A single cleanup epilogue with idempotent destructors would also fix many failure-path leaks.

### RNR-013 — Medium — VRAM admission can leave an unreleasable live claim on allocator failure

Location: `src/vramreg.c:285-333`

The registry record is written before the `vram_lease` handle is allocated. If the final `calloc` fails, the function returns `NULL` but has no handle with which to remove the admitted entry. Because the owning process remains alive, dead-PID reaping will not clear it. Later Runner processes can be refused based on a reservation that never became real.

Allocate the handle before committing the registry entry, or roll the exact `(pid, sequence)` entry back on post-admission failure. Add allocator-failure coverage around the commit boundary.

### RNR-014 — Medium — Registry configuration is silently truncated and capped

Locations: `src/server.c:3803-3835` and registry field declarations near `src/server.c:237`

The complete `-m name=path,...` string is copied into a 4096-byte stack buffer; model names and paths are copied into fixed 64/1024-byte fields; parsing stops at 16 entries. Truncation and excess entries do not consistently produce an error. Truncated names can collide, and truncated paths can select the wrong existing file or fail with a misleading message.

Treat limits as validated public constraints and reject the full configuration atomically with an exact explanation. The Python registry builder currently has no matching length/count validation.

### RNR-015 — Medium — Quantization writes directly to the destination and leaves corrupt partial files

Locations: `src/quantize.c:189-195`, `src/quantize.c:294-300`

The output path is truncated immediately with `fopen(..., "wb")`. Any later I/O, conversion, or allocation failure returns an error but leaves the partial file at the requested destination. A prior valid model at that path is irreversibly destroyed.

Write beside the target, flush and close successfully, validate the completed file, and atomically rename. Remove the temporary file on every failure. This is separate from the alignment corruption in RNR-002.

### RNR-016 — Medium — Python streaming omits tool-call deltas and only implements a subset of SSE framing

Locations: `python/src/gridcore_runner/endpoint.py:85-188`, especially `137-157`

`RunnerEndpoint.stream_chat()` retains only `content` and `reasoning_content`. It ignores `delta.tool_calls`, so the supported Python consumer cannot represent a major Runner feature even though the server exposes tool streaming. Its parser treats every physical `data:` line as a complete JSON frame rather than joining multiple `data:` lines into one SSE event. Runner currently emits one-line JSON, so the latter interoperates with Runner itself but not the full SSE contract.

Define typed streamed tool-call assembly and preserve finish reason/usage. Either implement event framing or explicitly describe the client as Runner-wire-specific. Add streamed tool-call and fragmented/multiline event tests.

### RNR-017 — Medium — Python startup leases are vulnerable to PID reuse

Locations: `python/src/gridcore_runner/lease.py:14-121`

The lease records only an owner PID and considers any live process with that PID to be the owner. After an unclean exit and PID reuse, an unrelated process can keep the lease permanently “live,” blocking Runner startup. The native VRAM registry already solves this class of bug by recording and comparing process start time.

Record a platform process-start identity with the PID and compare both. Preserve compatibility with legacy records as a bounded migration behavior rather than indefinitely weakening new leases.

## Architecture and maintainability findings

### RNR-018 — Low — The core header is an internal dependency hub, not a deep module boundary

Location: `src/runner.h` (953 lines, approximately 166 declarations/types)

One header exposes GGUF internals, tokenizer maps, the complete mutable model/engine layout, backend contracts, VRAM registry, schemas, templates, and server-adjacent types to almost every translation unit and test. The header itself acknowledges that immutable model weights and per-sequence mutable state remain fused in one `model_t`.

This increases rebuild coupling and makes ownership changes—such as the Metal issue—hard to express safely. Split public opaque interfaces from module-private headers, and split immutable model data from sequence state. Keep tests at public seams except for deliberately isolated numerical kernels.

### RNR-019 — Low — `server.c` has accumulated too many independent responsibilities

Location: `src/server.c` (4,082 lines)

The file combines socket portability, HTTP parsing, routing, request-schema validation, OpenAI/Anthropic response translation, SSE streaming, model registry/swap lifecycle, thread queues, continuous batching, prefix-cache telemetry, and shutdown. Global `SV`, queue, scheduler, and signal state make partial initialization and teardown difficult—the scheduler and swap leaks are symptoms of this structure.

Separate transport/framing, protocol adapters, model residency, and scheduler lifecycle behind narrow interfaces. Preserve current loopback and one-request-per-connection behavior with black-box conformance tests during extraction.

### RNR-020 — Low — Generated GPU sources have no drift gate

Locations: `src/kernels.cu`, `src/kernels.ptx`, `src/kernels_ptx.h`, `src/kernels.metal`, `src/kernels_metal.h`, `scripts/embed-ptx.py`, `scripts/embed-metal.py`, `Makefile`

Normal builds consume committed generated headers. PTX regeneration is manual/toolchain-dependent and Metal embedding is explicitly manual. CI does not verify that the embedded header matches its source artifact, so a source edit can be reviewed and merged while the shipped binary still embeds older code.

Add deterministic “generate to temporary output and compare” checks. Record the CUDA compiler/toolchain version for PTX and make the Metal source-to-header byte embedding independently reproducible even when Metal compilation itself is unavailable.

### RNR-021 — Low — Environment parsing is inconsistent with CLI validation

Locations: prefix-cache environment parsing in `src/engine.c:292-306`, request/prefix settings in `src/server.c`

CLI numeric parsing checks complete consumption, range, and finiteness. Environment settings use unchecked `strtoull`, `strtod`, or `atof`, accepting trailing garbage, overflow, negatives, NaN, and infinity. Depending on the setting, this can disable deadlines, create pathological eviction behavior, or overflow byte calculations.

Reuse one strict parser for CLI and environment inputs and report invalid settings at startup.

### RNR-022 — Low — Response reason phrases and several failure diagnostics are inaccurate

Locations: status mapping near `src/server.c:178-180`, Windows error branches in `src/server.c`

Some non-500 statuses (including 503) are emitted with `Internal Server Error` as the reason phrase, and Windows socket failures frequently report `errno` rather than `WSAGetLastError`. Clients should key on the numeric status, so this is not a protocol break, but it complicates operations and conformance debugging.

Use a complete status mapping and platform error formatter.

## Test, CI, and release assessment

### What is strong

- The native tests cover JSON parsing, schema compilation/validation, tokenizer families and OOM paths, templates, tool envelopes, sampler presets, shared CUDA weights, batched decode, prefix-cache correctness, loopback binding, and the VRAM registry.
- The protocol harness is unusually broad: HTTP framing, lifecycle, streaming, OpenAI Chat/Completions/Responses, Anthropic Messages, tools, structured output, prefix caching, continuous batching, and Clu-shaped traffic.
- The server rejects transfer encoding, duplicate/conflicting content lengths, oversized headers/bodies, invalid numeric request values, and non-loopback binding. It closes each connection, limiting request-smuggling scope.
- GGUF parsing caps counts, checks tensor extents, and has a fuzz target with a useful malformed corpus.
- CUDA has shared-weight lifecycle and batch equivalence coverage on actual hardware in this review environment.
- Compatibility dependencies and model artifacts are pinned, and release artifacts include checksums and build provenance.

### Material coverage gaps

1. **Metal failure lifecycle:** no injected runtime failure, ownership, or shutdown test catches RNR-001. Normal macOS CI smoke is insufficient.
2. **Loader rollback:** no systematic fault-injection across `model_load`, tokenizer initialization, GPU initialization, swap reload, or parallel-slot setup.
3. **Quantizer format matrix:** no non-32 alignment round trip, independent-reader validation, destination-preservation test, or write-failure injection.
4. **Architecture admission:** no test that every unknown architecture is refused; the current behavior intentionally permits it.
5. **Tokenizer fuzzing:** `Makefile` explicitly lists `tok_encode` as TODO even though tokenizer input is remotely influenced through HTTP prompts.
6. **Windows runtime:** Windows CI compiles and smokes but does not match Unix coverage for shared weights, batched decode, prefix cache, KV tolerance, VRAM registry, and full server lifecycle. This leaves RNR-009 weakly covered.
7. **Real-model reference gates:** CPU/CUDA identity proves backend agreement, not model correctness. Several claimed architectures lack a passing current independent-reference matrix.
8. **Python tools:** no streamed tool-call assembly test because the API cannot currently represent it.
9. **Code generation:** no CI comparison between kernel source artifacts and embedded headers.
10. **Local `make test`:** Python tests are skipped with a successful exit when pytest is missing. CI installs pytest, but a developer’s green local run can silently omit them.

### CI/release architecture

The release workflow triggers independently on any `v*` tag, performs one synthetic smoke per platform, and creates a draft release. It does not depend on a successful full CI run for the tagged SHA and does not execute the conformance or real-model compatibility gates itself. Draft status reduces but does not eliminate the risk: a tag can produce apparently release-ready artifacts from a revision whose full CI failed or never ran.

The release job should verify the tagged commit’s required checks (or invoke a reusable full test workflow) before building artifacts. Publication should additionally require an immutable compatibility report whose Runner commit exactly matches the tag.

CI also duplicates many long compile/run command blocks instead of invoking shared Make targets. The Windows and Unix paths have already diverged in coverage. Prefer reusable targets/workflows so “the release gate” is one executable definition.

## Validation performed

Commands were run from the repository root at the reviewed revision.

| Validation | Result |
|---|---|
| `make clean && make -j8 && make test` | Passed; all native targets passed, Python client 21/21 passed, additional Python tests 7/7 passed |
| `scripts/conformance.sh` | Passed: **274 passed, 5 skipped** |
| Python compilation (`py_compile` for every `scripts/*.py`; `compileall` for Python client and conformance/test modules) | Passed |
| GCC `-fanalyzer -Wall -Wextra` full native build | Built; confirmed unchecked sampler allocation; two additional reported paths were inspected and judged analyzer false positives because of surrounding state invariants |
| ASan/UBSan debug build and CPU one-shot smoke | Inference completed; LeakSanitizer reported **252,819 bytes in 307 allocations**, confirming CLI cleanup omissions |
| CUDA native shared-weight, batch, prefix, and CPU/GPU comparison paths | Exercised by `make test` on an NVIDIA RTX PRO 6000 Blackwell Max-Q MIG slice; passed |

The q8 KV tolerance subcases skipped because the tiny fixture’s head dimension is not divisible by 32; the test explicitly reported the skip. That run therefore does not validate q8 KV numerical tolerance.

## Review scope and limitations

The sweep included:

- every production C/Objective-C/CUDA module under `src/`;
- the public/internal header and CPU quantized kernels;
- GGUF loading and writing, tokenizer, model wiring/forward paths, sampling, generation, speculative decoding, JSON/schema constraints, templates/tools, VRAM registry, server/scheduler, portability layer, CUDA and Metal backends;
- the complete Python client/process/lease package;
- all scripts, Make targets, GitHub Actions workflows, compatibility manifests/reports, primary documentation, and release instructions;
- all native, Python, conformance, fuzz-harness, compatibility-harness, and torture-harness test sources and fixtures.

Generated PTX and embedded string headers were reviewed at their generation/source boundary and for lifecycle/API use; the 31,000-line generated PTX header was not manually audited instruction by instruction. CUDA was exercised on available hardware. Metal was statically reviewed but could not be executed because the review host is Linux. Windows behavior was statically reviewed and inferred from WinSock types; no Windows runtime was available. Large real-model evidence was audited from committed reports rather than rerunning every multi-gigabyte model during this code-review pass.

These limitations are reflected in the recommendations and are not treated as proof that unexecuted platform paths are correct.

## Suggested remediation order

1. Fix and test Metal ownership/fallback (RNR-001).
2. Make model/tokenizer/scheduler lifecycle transactional and fault-injected (RNR-003, RNR-005, RNR-006, RNR-012, RNR-013).
3. Correct quantizer alignment and atomic output, then independently round-trip it (RNR-002, RNR-015).
4. Enforce an architecture allowlist and reconcile current reference evidence with public claims (RNR-004, RNR-007).
5. Replace volatile cancellation and pointer-truncating socket types (RNR-008, RNR-009).
6. Harden typed metadata, file-size arithmetic, registry/env parsing, and Python tool streaming (RNR-010, RNR-011, RNR-014, RNR-016, RNR-017, RNR-021).
7. Decompose internal boundaries and unify build/release gates without changing wire behavior (RNR-018 through RNR-020, RNR-022).

No fixes are included in this review.
