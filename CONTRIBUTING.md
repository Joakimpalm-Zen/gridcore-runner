# Contributing

Runner is in public alpha; bug reports with `runner --version` and
`runner --caps` output are the most valuable contribution.

When bumping the release, update `RUNNER_VERSION` in `src/runner.h` first;
the versions in `README.md` and `SECURITY.md` must match it verbatim, and
`python/pyproject.toml` must carry the same version in PEP 440 form (pre-release
suffixes translate: `0.1.3-alpha` → `0.1.3a0`). The Python client remains a
separate distributable with no build-time coupling to the C header.

## Correctness gates (non-negotiable)

Every change must hold these invariants, in CI and locally:

1. **GPU output is token-identical to the CPU path.** Any kernel or
   offload change must produce byte-identical temp-0 output vs `--gpu off`.
2. **gemma4 stays token-identical to llama.cpp** (the architecture was
   verified against the reference; don't drift).
3. **The CI matrix passes on Linux, macOS, and Windows** — including the
   sanitizer build (`make debug`, ASan/UBSan) and every smoke test.
4. **Schema guarantees are load-bearing.** Keys emit in declared order and
   truncated output still parses; the suite above runner depends on both.

## Building

    make            # or: make OS=Windows_NT CC=gcc under MSYS2 UCRT64
    make debug      # ASan/UBSan build

## Tests

Use `make test` for the fast schema and Python client correctness checks, and
`make smoke` for a short CPU-only end-to-end run. CI runs the full smoke matrix
on Linux, macOS, and Windows; new behavior lands with a smoke there (TDD: watch
it fail first).

## Architecture scope (the lean-engine boundary)

Runner stays a compact engine (~19K LOC) on purpose. Breadth is the failure
mode, so architecture support is admitted by decision, not accumulation:

- **Composable knobs over subsystems.** New model families should land as
  small orthogonal switches on the existing MoE/attention code (router
  options, attention sinks, NoPE, temperature scaling) — not as parallel
  forward paths.
- **Tier B stays declined in mainline.** MLA/DeepSeek-style latent attention,
  MTP heads, and linear/recurrent attention are not knobs; they change the
  attention path, the KV format, or both. They do not enter `model.c`.
- **The gated exception is Syntetik profiles.** Feasible model/runtime
  co-design for the suite's own model (MTP verifier, isolated MLA, shared
  experts, hybrid KDA) may land only behind an explicit versioned
  agent-profile admission: isolated behind a narrow attention/cache seam or
  build flag, tested with tiny generated GGUF fixtures, each with a measured
  trigger recorded before work starts. **No profile may leak complexity into
  the dense Llama path**, and a checkpoint requiring an unadmitted feature
  must fail closed at load rather than degrade silently.
- Everything above is still subject to the correctness gates: CPU==GPU
  token-identical, certified vs llama.cpp where a reference exists.

## Style

Plain C11 (gnu11), zero dependencies beyond libc/pthreads. Comments state
constraints the code can't show — not narration.
