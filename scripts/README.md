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

## Benchmarks

- **`batch-bench.py`** — concurrent-serving throughput + single-request latency
  against a running server (the numbers behind the batching phase work).
- **`bench.sh`** — thin wrapper for repeated `--bench-json` runs.
- **`agent-torture.py` / `torture-compare.py`** — the adversarial tool-call
  matrix and its cross-runtime report comparator (see `docs/agent-torture.md`).

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
