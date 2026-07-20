#!/bin/sh
# Phase 0 agent-protocol conformance harness — one command, no arguments.
#
#   scripts/conformance.sh                 run everything
#   scripts/conformance.sh -k streaming    pass any extra args through to pytest
#
# Builds ./runner and ./test.gguf if they are missing, starts a server on a
# free port, runs tests/conformance, and writes tests/conformance/out/report.json
# with latency, prompt/generation speed and peak RSS.
#
# Environment:
#   CC                       compiler for the build (passed to make)
#   PYTHON                   interpreter to use (default: python3)
#   RUNNER_EXE               use an already-built binary instead of building
#   RUNNER_TEST_MODEL        model to serve (default: ./test.gguf)
#   RUNNER_CONFORMANCE_OUT   report/artifact directory
#   RUNNER_CONFORMANCE_UPDATE=1
#                            re-record the committed fixtures in
#                            tests/conformance/fixtures instead of comparing.
#                            Review the diff: a changed fixture is a changed
#                            wire format.
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT"

PYTHON=${PYTHON:-python3}
command -v "$PYTHON" >/dev/null 2>&1 || PYTHON=python

if ! "$PYTHON" -c "import pytest" >/dev/null 2>&1; then
    echo "conformance: pytest is required; install it with '$PYTHON -m pip install pytest'" >&2
    exit 1
fi

EXE=runner
case "$(uname -s 2>/dev/null || echo unknown)" in
    MINGW*|MSYS*|CYGWIN*) EXE=runner.exe ;;
esac

if [ -z "${RUNNER_EXE:-}" ] && [ ! -x "./$EXE" ]; then
    echo "conformance: building ./$EXE"
    make ${CC:+CC="$CC"} "$EXE" >/dev/null || make ${CC:+CC="$CC"} runner
fi

MODEL=${RUNNER_TEST_MODEL:-test.gguf}
if [ ! -f "$MODEL" ]; then
    echo "conformance: building $MODEL"
    "$PYTHON" scripts/make-test-model.py "$MODEL"
fi

exec "$PYTHON" -m pytest tests/conformance -q "$@"
