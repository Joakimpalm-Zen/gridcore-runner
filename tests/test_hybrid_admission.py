"""Mamba-2 hybrid admission and the granitehybrid decode step (tracer 2).

`granitehybrid` (Granite-4 h-series) now RUNS: the runner interleaves GQA
attention with a Mamba-2 selective-SSD recurrence, applies the granite muP
scaling and treats the attention layers as NoPE, gated token-identically
against llama.cpp b10353 on the real granite-4.0-h-small (a scripted check;
this file is the unit level). `nemotron_h_moe` (Nemotron-3.5 Lightning) shares
the tensor set but adds a grouped scan this tracer has not certified, so it is
still RECOGNIZED and refused with a specific "forward not yet implemented"
message — never the generic unknown-architecture refusal, which would read as a
typo. Either way, a missing required SSM tensor must FAIL CLOSED naming it (the
hostile-GGUF discipline). No download: the fixtures are generated and mirror a
real granite-4.0-h / Nemotron GGUF (sparse-MoE, NoPE, inner == 2*embd).
"""
import os
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    return exe


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    d = tmp_path_factory.mktemp("hybrid")
    out = {}
    for arch in ("granitehybrid", "nemotron_h_moe"):
        pfx = d / arch
        subprocess.run(
            [sys.executable, ROOT / "scripts/make-test-hybrid.py", str(pfx),
             "--arch", arch], check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
        out[arch] = (pfx.with_suffix(".gguf"),
                     d / (arch + ".missing-ssm_d.gguf"))
    return out


def _run(runner_bin, model, n="1"):
    return subprocess.run(
        [runner_bin, "-m", str(model), "-p", "hi", "-n", n, "--temp", "0",
         "--gpu", "off"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)


def test_granitehybrid_loads_and_decodes(runner_bin, models):
    """The forward is implemented: granitehybrid must LOAD (not refuse) and
    decode. A build that regressed the loader back to a refusal, or that could
    not run the Mamba-2 step at all, fails here."""
    good, _ = models["granitehybrid"]
    proc = _run(runner_bin, good, n="8")
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    err = proc.stderr.decode(errors="replace")
    # the load announces the hybrid layout, not a refusal
    assert "granitehybrid" in err
    assert "forward not yet implemented" not in err


def test_granitehybrid_decode_is_deterministic(runner_bin, models):
    """Greedy decode must be reproducible run to run (the property the real
    token-identity gate builds on)."""
    good, _ = models["granitehybrid"]
    a = _run(runner_bin, good, n="8")
    b = _run(runner_bin, good, n="8")
    assert a.returncode == 0 and b.returncode == 0
    assert a.stdout == b.stdout


def test_nemotron_still_refuses_specifically(runner_bin, models):
    """Nemotron's grouped scan is not certified yet: recognized, refused with
    its own reason, and NOT conflated with an unknown-arch typo."""
    good, _ = models["nemotron_h_moe"]
    proc = _run(runner_bin, good)
    assert proc.returncode != 0, "the forward is not certified; it must refuse"
    err = proc.stderr.decode(errors="replace")
    assert "nemotron_h_moe" in err
    assert "forward not yet implemented" in err
    assert "refusing to run it through llama-style math" not in err


@pytest.mark.parametrize("arch", ["granitehybrid", "nemotron_h_moe"])
def test_missing_ssm_tensor_names_the_tensor(runner_bin, models, arch):
    _, broken = models[arch]
    proc = _run(runner_bin, broken)
    assert proc.returncode != 0, "a missing required SSM tensor must fail closed"
    err = proc.stderr.decode(errors="replace")
    assert "ssm_d" in err, "the failure must name the absent tensor"
    assert arch in err, "still recognized as the hybrid, just malformed"


def test_opt_in_does_not_smuggle_nemotron_into_llama_math(runner_bin, models):
    """RUNNER_ALLOW_UNKNOWN_ARCH lets an UNKNOWN arch try llama math; a
    recognized-but-uncertified hybrid is a different case — the opt-in must not
    turn its specific refusal into a silent-wrong run."""
    good, _ = models["nemotron_h_moe"]
    proc = subprocess.run(
        [runner_bin, "-m", str(good), "-p", "hi", "-n", "1", "--gpu", "off"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30,
        env={**os.environ, "RUNNER_ALLOW_UNKNOWN_ARCH": "1"})
    assert proc.returncode != 0
    assert "forward not yet implemented" in proc.stderr.decode(errors="replace")
