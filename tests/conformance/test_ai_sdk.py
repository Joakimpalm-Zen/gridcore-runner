"""The Vercel AI SDK against the same server, via Node.

Why a second SDK at all: `test_openai_sdk.py` proves the official Python client
works, and that client is not what most local-model tooling uses. Cline,
Continue and anything built on Next.js reach for `ai` + `@ai-sdk/openai`, which
has its own request builder and its own stream parser and sends field
combinations the official SDK never does. Two independent clients is the point;
one of them agreeing with our own tests proves less than it looks like.

The cases live in `tests/aisdk/smoke.mjs` because they have to run under Node.
This module is the pytest side: it locates the smoke, runs it against the
suite's server, and reports each case as a subtest-style assertion so a failure
names the case rather than "node exited 3".

Skipped unless `tests/aisdk/node_modules` is present. That is deliberate and
matches the rule `test_messages.py` records for `anthropic`: the C suite must
not gain an npm-install step in CI. To enable it locally:

    cd tests/aisdk && npm install
"""

import json
import os
import shutil
import subprocess

import pytest

from _errors import ProtocolError

HERE = os.path.dirname(os.path.abspath(__file__))
AISDK = os.path.normpath(os.path.join(HERE, "..", "aisdk"))
SMOKE = os.path.join(AISDK, "smoke.mjs")

_node = shutil.which("node")
_installed = os.path.isdir(os.path.join(AISDK, "node_modules"))

needs_node = pytest.mark.skipif(
    _node is None or not _installed or not os.path.exists(SMOKE),
    reason="node or tests/aisdk/node_modules is absent (cd tests/aisdk && npm install)",
)


@pytest.fixture(scope="module")
def smoke(server):
    """Run the whole smoke once and return its report.

    One process for all cases: node startup plus SDK import costs more than the
    requests do, and running it per-test would multiply that by nine.
    """
    proc = subprocess.run(
        [_node, SMOKE, f"http://127.0.0.1:{server.port}"],
        cwd=AISDK, capture_output=True, text=True, timeout=600, check=False,
    )
    try:
        return json.loads(proc.stdout)
    except ValueError as exc:
        # A crash before the report is written is itself a finding, and the
        # stderr tail is the only thing that says what happened.
        raise ProtocolError("the AI SDK smoke produced no report",
                            returncode=proc.returncode,
                            stderr=proc.stderr[-800:],
                            stdout=proc.stdout[-400:]) from exc


@needs_node
def test_ai_sdk_cases_all_pass(smoke):
    if smoke["fail"]:
        raise ProtocolError("the Vercel AI SDK rejected runner's responses",
                            failures=smoke["fail"], passed=smoke["ok"])


@needs_node
def test_ai_sdk_actually_ran_its_cases(smoke):
    """A smoke that silently ran nothing would pass the test above.

    The count is asserted rather than the names, so adding a case to smoke.mjs
    does not need an edit here — but removing them all does.
    """
    total = len(smoke["ok"]) + len(smoke["fail"])
    if total < 8:
        raise ProtocolError("the AI SDK smoke ran fewer cases than it defines",
                            ran=total, ok=smoke["ok"])
