#!/bin/sh
# Claude Code against Runner, end to end.
#
# The README claims Claude Code compatibility. Until this script existed that
# claim rested on a validation someone ran by hand once, against one version,
# with the commands in their shell history — which is the kind of claim that
# stays in a README long after it stops being true. This makes it repeatable.
#
# What it proves: the real Claude Code binary, pointed at Runner with
# ANTHROPIC_BASE_URL, completes a built-in `Read` tool loop and returns a
# sentinel that exists nowhere but the fixture file. That is a full round trip
# through /v1/messages -- system turn, tool declarations, a tool_use block, a
# tool_result replayed in the next turn -- driven by a client nobody here
# wrote.
#
# What it does not prove: model quality. The task is deliberately trivial and
# the tool set is restricted to Read, so a failure means a protocol problem
# rather than a small model wandering off.
#
#   scripts/claude-code-e2e.sh [model.gguf]
#
# Environment:
#   RUNNER_EXE   an already-built binary (default: ./runner)
#   PORT         port to serve on (default: 8123)
#   CTX          server context length (default: 32768)
#   DEADLINE     seconds to allow Claude Code (default: 1800)
#
# Exit 0 pass, 1 fail, 77 skip (no claude binary).
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
MODEL=${1:-$ROOT/models/Qwen3-4B-Q4_K_M.gguf}
RUNNER_EXE=${RUNNER_EXE:-$ROOT/runner}
PORT=${PORT:-8123}
# Claude Code sends its whole system prompt and every built-in tool
# declaration on the first request, whatever --allowedTools permits -- that
# flag governs permission, not what is declared. Measured against 2.1.220 it
# does not fit in 16k; 32k does, with room for the tool loop.
CTX=${CTX:-32768}
# A thinking model spends hundreds of tokens per turn and the loop is several
# turns; 600 s was not enough for Qwen3-4B on one MIG slice.
DEADLINE=${DEADLINE:-1800}

if ! command -v claude >/dev/null 2>&1; then
    echo "skip: the claude CLI is not installed"
    exit 77
fi
if [ ! -f "$MODEL" ]; then
    echo "skip: model $MODEL not found"
    exit 77
fi

VERSION=$(claude --version 2>/dev/null || echo unknown)
WORK=$(mktemp -d)
# The sentinel is generated per run, so a pass cannot come from anything the
# model already knew or from a stale answer cached anywhere in the stack.
SENTINEL="GRIDCORE-$(od -An -N4 -tx1 /dev/urandom | tr -d ' \n' | tr 'a-f' 'A-F')"
printf 'The gridcore fixture sentinel is %s\n' "$SENTINEL" > "$WORK/fixture.txt"

cleanup() {
    [ -n "${SRV:-}" ] && kill "$SRV" 2>/dev/null || true
    rm -rf "$WORK"
}
trap cleanup EXIT INT TERM

"$RUNNER_EXE" -m "$MODEL" --serve --port "$PORT" -c "$CTX" --parallel 2 \
    > "$WORK/runner.log" 2>&1 &
SRV=$!
i=0
while [ "$i" -lt 180 ]; do
    grep -q listening "$WORK/runner.log" 2>/dev/null && break
    sleep 1
    i=$((i + 1))
done
if ! grep -q listening "$WORK/runner.log" 2>/dev/null; then
    echo "FAIL: runner did not start"
    tail -5 "$WORK/runner.log"
    exit 1
fi

echo "claude:  $VERSION"
echo "runner:  $("$RUNNER_EXE" --version 2>/dev/null || echo unknown)"
echo "model:   $(basename "$MODEL")"

# The served id, asked for rather than assumed. Without --model the CLI keeps
# whatever model the developer's own session is configured with (an Anthropic
# model name Runner has never heard of), and the run dies before a single
# request is made -- which is exactly how this failed the first time.
SERVED=$(curl -s "http://127.0.0.1:$PORT/v1/models" \
         | sed -n 's/.*"id":"\([^"]*\)".*/\1/p' | head -1)
[ -n "$SERVED" ] || { echo "FAIL: /v1/models named no model"; exit 1; }
echo "served:  $SERVED"

# CLAUDE_CONFIG_DIR is redirected so this never reads or writes the developer's
# own Claude Code configuration, and ANTHROPIC_API_KEY is set because the CLI
# insists on one being present -- Runner ignores it.
OUT="$WORK/claude.out"
set +e
ANTHROPIC_BASE_URL="http://127.0.0.1:$PORT" \
ANTHROPIC_API_KEY="not-used" \
ANTHROPIC_MODEL="$SERVED" \
CLAUDE_CONFIG_DIR="$WORK/config" \
timeout "$DEADLINE" claude -p --model "$SERVED" \
    "Read the file $WORK/fixture.txt and reply with the sentinel string it contains, and nothing else." \
    --allowedTools Read \
    > "$OUT" 2>"$WORK/claude.err"
RC=$?
set -e

if grep -q "$SENTINEL" "$OUT" 2>/dev/null; then
    echo "PASS: Claude Code completed the Read loop and returned the sentinel"
    exit 0
fi

echo "FAIL: the sentinel did not come back (claude exit $RC)"
echo "--- claude stdout ---"; tail -20 "$OUT" 2>/dev/null
echo "--- claude stderr ---"; tail -20 "$WORK/claude.err" 2>/dev/null
echo "--- runner log ---";    tail -20 "$WORK/runner.log"
exit 1
