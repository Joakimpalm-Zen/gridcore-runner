# scripts/ — what each tool is for

Development and verification tooling. Nothing here is needed to *run* the
runner; several are load-bearing gates for changing it.

## Verification gates (use these before trusting a kernel/perf change)

- **`kernel-verify.py`** — demands **token-identical** greedy output between a
  baseline and a candidate binary on 5 prompts. A faster binary that changes
  tokens is a regression, not a win.
- **`kernel-bench.py`** — prefill/decode tok/s as JSON for one binary+model.
- **`template-conformance.py`** — byte-exact comparison of runner's native C
  chat renderer against each family's upstream jinja template, the gate whose
  absence let five families drift unnoticed. `make template-conformance`.
  Intentional deviations live in `template-conformance-allowlist.json`, each
  carrying a source citation the gate re-verifies and fails on when it rots;
  known differences awaiting fixes live in
  `template-conformance-baseline.json`, which should only ever shrink. Exit 2
  means NOT CHECKED (no jinja2, no network, no oracle) and is never a pass.
- **`compat_matrix.py`** — verify pinned real-model hashes and emit architecture
  load/inference evidence without committing the GGUFs.
- **`consumer_compat.py`** — run pinned OpenAI/Anthropic SDK, LiteLLM and
  LangChain clients against one live Runner and emit a JSON report.
- **`reference_compare.py`** — compare Runner and llama.cpp exact greedy text
  through equivalent raw OpenAI Completions requests.
- **`compare_llamacpp.py`** — reproducible Runner-vs-llama.cpp evidence
  harness for the same GGUF, prompt, context and sampling settings; emits JSON
  plus a provenance-complete Markdown summary, quantifies common-token
  top-logprob deltas where supported, and marks real results pending when run
  in fixture mode. `--endpoint label=url=model` additionally measures an
  already-running OpenAI-compatible daemon (Ollama, LM Studio) on the same
  prompt and budget — throughput only, and labelled as such, because the
  correctness gate is defined against the pinned llama.cpp reference and those
  runtimes do not expose comparable completion logprobs.
- **`stress-models.py`** — run every GGUF on a machine's shelf against this
  build: load, generate, CPU-vs-CUDA identity, fault detection (fallbacks,
  kernel-launch failures, refusals, timeouts) and a settings sweep, with one
  resumable JSON per model. Identity is a PREFIX test, not string equality —
  the legs may use different token budgets, and comparing truncated character
  slices reports a mismatch on the trailing newline alone.
- **`stress-context.py`** — the context/KV edges: auto-fit, a context the
  machine cannot hold, and whether a refusal says why.
- **`cpu_cuda_check.py`** — the compatibility program's `cpu_cuda` check for
  one model: greedy CPU output must be byte-identical to greedy CUDA output
  over several prompts, with the eager router pinned (MoE byte identity is
  defined over that path).
- **`kv-quality.py`** — KV-cache quality gate: compares q8 KV against f16 on
  teacher-forced logits (the deeper version of `tests/test_kv_tol.c`'s gate).
- **`verify-gguf.py`** — structural sanity check of a GGUF file (metadata,
  tensor table, offsets) without loading it into the engine.
- **`check-release.py`** — release artifact consistency gate: binary
  `--version` against the tag, the README's release strings, a CHANGELOG
  section, BUILD-INFO's tag/commit lines, the Python package version, and the
  compat report the tag must publish. Unit-tested by
  `tests/test_release_check.py`; run by the release workflow.
- **`check-generated.py`** — drift gate for the committed generated GPU
  headers (`src/kernels_metal.h`, `src/kernels_ptx.h`). Re-embeds each from its
  committed source and compares, so a `kernels.metal` edit cannot ship while
  the binary still embeds the old bytes. Pure Python: no Metal or CUDA
  toolchain, runs in well under a second. Run it after touching a shader.
- **`cert-greedy-identity.py`** — six-prompt greedy-identity gate against a
  pinned llama.cpp reference (four domains at 64 tokens, two repeated at 256,
  pure-greedy sampler and `cache_prompt:false` on the reference side).
- **`afmoe-cert-gate3.py`** — the same identity protocol as originally written
  for the afmoe certification against llama.cpp b10280, kept replayable.
- **`claude-code-e2e.sh`** — the real Claude Code binary pointed at Runner with
  `ANTHROPIC_BASE_URL`, completing a built-in `Read` tool loop. Turns the
  README's Claude Code compatibility claim into something that can be re-run.
- **`competitor-freshness.py`** — checks the runtime versions in published
  torture results against upstream releases. Metadata-only registry requests;
  a network failure is a skip, never a red scheduled run.
- **`template-conformance-harmony.py`** — the Harmony oracle used by
  `template-conformance.py`: gpt-oss is compared against `openai-harmony`
  rather than the GGUF's jinja, which is a reimplementation with known gaps.
  Its committed goldens live in `template-conformance-harmony-oracle.json`.
- **`template-conformance-render.c`** — the runner-side driver for that
  harness: reads a JSON job list, writes each case's native C render, so the
  Python side can diff byte-for-byte.

## Benchmarks

- **`batch-bench.py`** — concurrent-serving throughput + single-request latency
  against a running server (the numbers behind the batching phase work).
- **`bench.sh`** — thin wrapper for repeated `--bench-json` runs.
- **`agent-torture.py` / `torture-compare.py`** — the adversarial tool-call
  matrix and its cross-runtime report comparator (see `docs/agent-torture.md`).
- **`weight-io-bench.py`** — how this machine's storage serves large weight
  reads: mmap page faults against explicit reads, which is a property of the
  storage and the OS rather than of the model, and can change a design call.
- **`write-stall.py`** — the real write-side socket stall, with a large model
  and context rather than the tiny fixture: enough SSE to fill the loopback
  buffers against a client that never reads. A local experiment, not a CI test.

## Measurement and analysis

Nothing here gates a merge; these are how a claim gets a number behind it.

- **`quant-fidelity.py`** — how tool-calling and structured-output fidelity
  degrade across one model's quantization ladder: schema conformance, tool
  selection and argument agreement against a pinned reference variant, one
  server at a time.
- **`kld-compare.py`** — approximate KLD between two runner-served models'
  next-token distributions over a pinned corpus, via chat-completions
  `top_logprobs` (exact 0 for identical distributions, monotonic with real
  divergence; not the full-vocab KLD).
- **`kld-compare-raw.py`** — the same comparison over RAW completions, which is
  the sound one ACROSS engines: a chat-endpoint comparison of gpt-oss finds the
  two engines' templates disagreeing rather than their weights.
- **`token_divergence.py`** — where a greedy run first leaves llama.cpp's, and
  by how much: the logprob gap at the first differing argmax separates a
  near-tie from a real arithmetic bug, which a text diff cannot.
- **`sensitivity_floor.py`** — how much a given model amplifies a small numeric
  change, measured against itself (f16 vs q8 KV). Measure the floor before
  calling a cross-engine gap a bug.
- **`gen-quality-metrics.py`** — collapse detection over generated text:
  n-gram repetition runs, blank fraction, length distribution. Stdlib only.
- **`cl-calibration.py`** — calibration report for `choice_logprobs` decision
  records: when a constrained decision point says 0.8, is it right 80% of the
  time?
- **`classify-grammar-trace.py`** — classifies grammar-draft rejection causes
  from a `RUNNER_GRAMMAR_TRACE` JSONL (tail-straddle, coarse-merge, fine-split,
  seam).
- **`moe-prune-plan.py`** — turns `RUNNER_MOE_TRACE` JSONL into a
  `--prune-experts` plan (and a hot/cold precision plan), ranking experts by
  gate mass or gate x activation-norm saliency, uniform or coverage-based.

## GGUF tools

- **`gguf-inspect.py`** — a file's real per-tensor-class quant types rather
  than what its filename claims (an MXFP4 expert bank inside a "Q4_K_M" file is
  the normal case). Reads only the header and tensor directory.
- **`gguf-split.py`** — split a GGUF into llama.cpp-compatible shards, for
  testing multi-part loads; layout follows upstream `gguf-split` verbatim.
- **`gguf-depth-slice.py`** — drop whole transformer layers and keep everything
  else: surviving layers keep their on-disk bytes, `blk.N` is renumbered
  densely, and per-layer metadata arrays are filtered to the survivors.
- **`make-bf16-fixture.py`** — rewrite a GGUF's F16 tensors as BF16 in place to
  get a valid BF16 fixture; both types are 2 bytes, so every offset stays put.

## Fixtures and codegen

- **`make-test-model.py`** — builds `test.gguf`, the tiny CI fixture (plus
  malformed variants for the rejection tests).
- **`make-test-moe.py`** — the dense + sparse-MoE GGUF trio for the MoE
  equivalence test (`make test-moe`; see `docs/moe-support.md`).
- **`make-test-ornith.py`** — tiny Qwen3.5/Ornith hybrid fixture
  (via `tests/test_ornith_cpu.py`).
- **`make-vocab-fixture.py`** — tokenizer vocab fixtures in `tests/fixtures/`.
- **`tokenizer-corpus.py` / `difftok.py`** — regenerate / diff the 721-string
  tokenizer conformance corpus against Hugging Face reference tokenizers;
  revision-bound `--capture` files make the reference side replayable offline.
- **`ornith-reference.py`** — reference layout contract for Ornith GGUFs
  (unit-tested by `tests/test_ornith_reference.py`).
- **`embed-ptx.py`** — embeds `src/kernels.ptx` into `src/kernels_ptx.h`
  (invoked by `make ptx`).
- **`embed-metal.py`** — same embedding step for the Metal shader source into
  `src/kernels_metal.h` (run manually on a Mac when `kernels.metal` changes;
  there is deliberately no Makefile target on non-Mac hosts).
- **`conformance.sh`** — drives the agent-protocol conformance suite in CI.
