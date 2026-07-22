# scripts/ — what each tool is for

Development and verification tooling. Nothing here is needed to *run* the
runner; several are load-bearing gates for changing it.

## Verification gates (use these before trusting a kernel/perf change)

- **`kernel-verify.py`** — demands **token-identical** greedy output between a
  baseline and a candidate binary on 5 prompts. A faster binary that changes
  tokens is a regression, not a win.
- **`kernel-bench.py`** — prefill/decode tok/s as JSON for one binary+model.
- **`compat_matrix.py`** — verify pinned real-model hashes and emit architecture
  load/inference evidence without committing the GGUFs.
- **`consumer_compat.py`** — run pinned OpenAI/Anthropic SDK, LiteLLM and
  LangChain clients against one live Runner and emit a JSON report.
- **`reference_compare.py`** — compare Runner and llama.cpp exact greedy text
  through equivalent raw OpenAI Completions requests.
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
- **`make-test-ornith.py`** — tiny Qwen3.5/Ornith hybrid fixture
  (via `tests/test_ornith_cpu.py`).
- **`make-vocab-fixture.py`** — tokenizer vocab fixtures in `tests/fixtures/`.
- **`tokenizer-corpus.py` / `difftok.py`** — regenerate / diff the 721-string
  tokenizer conformance corpus against HuggingFace reference tokenizers.
- **`ornith-reference.py`** — reference layout contract for Ornith GGUFs
  (unit-tested by `tests/test_ornith_reference.py`).
- **`embed-ptx.py`** — embeds `src/kernels.ptx` into `src/kernels_ptx.h`
  (invoked by `make ptx`).
- **`embed-metal.py`** — same embedding step for the Metal shader source into
  `src/kernels_metal.h` (run manually on a Mac when `kernels.metal` changes;
  there is deliberately no Makefile target on non-Mac hosts).
- **`conformance.sh`** — drives the agent-protocol conformance suite in CI.
