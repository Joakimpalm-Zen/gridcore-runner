# Contributing

Runner is in public alpha; bug reports with `runner --version` and
`runner --caps` output are the most valuable contribution.

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

    python3 scripts/make-test-model.py test.gguf
    cc -std=gnu11 -Wall -Wextra -I src tests/test_json_schema.c \
       src/json.c src/jsonmode.c src/schema.c -o test-json-schema -lm
    ./test-json-schema

The full smoke matrix lives in `.github/workflows/ci.yml`; new behavior
lands with a smoke there (TDD: watch it fail first).

## Style

Plain C11 (gnu11), zero dependencies beyond libc/pthreads. Comments state
constraints the code can't show — not narration.
