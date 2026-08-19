"""Mamba-2 hybrid families are RECOGNIZED before the forward exists.

The runner interleaves attention and recurrent layers for qwen35 (Gated
DeltaNet) already, but the Mamba-2 hybrids — `granitehybrid` (Granite-4 h) and
`nemotron_h_moe` (Nemotron-3.5 Lightning) — have a different recurrence whose
forward is not yet implemented. Admission must distinguish "we know exactly
what this is, the forward is pending" from the generic "we recognize nothing"
refusal (which would read as an unknown-architecture typo), and it must FAIL
CLOSED with a NAMED tensor when a required SSM tensor is absent from an
otherwise-recognized hybrid (the hostile-GGUF discipline). No download: the
fixtures are generated and mirror a real granite-4.0-h / Nemotron GGUF.
"""
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


def _run(runner_bin, model):
    return subprocess.run(
        [runner_bin, "-m", str(model), "-p", "hi", "-n", "1", "--gpu", "off"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)


@pytest.mark.parametrize("arch", ["granitehybrid", "nemotron_h_moe"])
def test_recognized_hybrid_refuses_specifically(runner_bin, models, arch):
    good, _ = models[arch]
    proc = _run(runner_bin, good)
    assert proc.returncode != 0, "the forward is not implemented; it must refuse"
    err = proc.stderr.decode(errors="replace")
    # recognized, not the generic unknown-arch refusal
    assert arch in err
    assert "forward not yet implemented" in err
    assert "refusing to run it through llama-style math" not in err, (
        "a recognized hybrid must not fall back to the generic unknown-arch "
        "refusal — that conflates 'known but pending' with 'unknown'")


@pytest.mark.parametrize("arch", ["granitehybrid", "nemotron_h_moe"])
def test_missing_ssm_tensor_names_the_tensor(runner_bin, models, arch):
    _, broken = models[arch]
    proc = _run(runner_bin, broken)
    assert proc.returncode != 0, "a missing required SSM tensor must fail closed"
    err = proc.stderr.decode(errors="replace")
    assert "ssm_d" in err, "the failure must name the absent tensor"
    # still recognized as the hybrid, just malformed
    assert arch in err


@pytest.mark.parametrize("arch", ["granitehybrid", "nemotron_h_moe"])
def test_opt_in_does_not_smuggle_a_hybrid_into_llama_math(runner_bin, models, arch):
    """RUNNER_ALLOW_UNKNOWN_ARCH lets an UNKNOWN arch try llama math; a
    recognized hybrid is a different case — its recurrence is wrong under llama
    math, so the opt-in must not turn the refusal into a silent-wrong run."""
    good, _ = models[arch]
    proc = subprocess.run(
        [runner_bin, "-m", str(good), "-p", "hi", "-n", "1", "--gpu", "off"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30,
        env={**__import__("os").environ, "RUNNER_ALLOW_UNKNOWN_ARCH": "1"})
    assert proc.returncode != 0
    err = proc.stderr.decode(errors="replace")
    assert "forward not yet implemented" in err
