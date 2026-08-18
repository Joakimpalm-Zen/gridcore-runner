"""scripts/cert-greedy-identity.py: the six-prompt greedy-identity gate.

Its output is a certification record, so a row that says "identical: false,
first_divergent_byte: 187" has to mean the two engines disagreed — not that
one of them died.
"""

import importlib.util
import os
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "cert_greedy_identity", ROOT / "scripts/cert-greedy-identity.py")
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)

PROMPT = "The capital of France is"


def stub_runner(tmp_path, completion, exit_code, stderr=""):
    """A binary that echoes the prompt the way src/main.c does, then exits."""
    script = tmp_path / "stub.py"
    script.write_text(
        "import sys\n"
        "prompt = sys.argv[sys.argv.index('-p') + 1]\n"
        f"sys.stdout.write(prompt + {completion!r})\n"
        f"sys.stderr.write({stderr!r})\n"
        f"sys.exit({exit_code})\n")
    if os.name == "nt":
        launcher = tmp_path / "runner.cmd"
        launcher.write_text(f'@"{sys.executable}" "{script}" %*\r\n')
    else:
        launcher = tmp_path / "runner"
        launcher.write_text(
            f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n')
        launcher.chmod(0o755)
    return str(launcher)


def test_a_healthy_run_returns_the_completion(tmp_path):
    runner = stub_runner(tmp_path, " Paris.", 0)

    assert MOD.runner_completion(runner, "m.gguf", PROMPT, 64) == " Paris."


def test_a_runner_that_died_mid_generation_is_not_a_divergence(tmp_path):
    """src/main.c echoes the prompt to stdout BEFORE engine_generate, so a
    death after that point still leaves a well-formed prefix. The prompt-echo
    guard passed it, the truncated completion was compared against the
    reference's full one, and the record published a first_divergent_byte —
    attributing a killed process to the model."""
    runner = stub_runner(tmp_path, " Par", -9 if os.name != "nt" else 1,
                         stderr="Killed\n")

    with pytest.raises(SystemExit) as caught:
        MOD.runner_completion(runner, "m.gguf", PROMPT, 64)

    assert "Killed" in str(caught.value)
