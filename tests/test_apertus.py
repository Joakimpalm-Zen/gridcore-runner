"""Apertus: ungated MLP plus the xIELU activation.

Apertus has no `ffn_gate` — its FFN is up -> xielu -> down — which every other
dense architecture here does have, so the tensor is only optional when the
activation is the ungated one.

xIELU (ggml's op_xielu):
    x >  0 : alpha_p * x^2 + beta * x
    x <= 0 : (expm1(min(x, eps)) - x) * alpha_n + beta * x

The fixture comes in two flavours. `IDENT` sets alpha_p = alpha_n = 0 and
beta = 1, which makes xIELU the identity; `REAL` uses non-trivial parameters.
Comparing them is what shows the parameters are actually read — a build that
ignored them would produce the same output for both.
"""
import json
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


def _make(tmp_path, flavour):
    path = tmp_path / f"ap_{flavour}.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py",
                    "--apertus", flavour, str(path)],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    return path


def _gen(runner_bin, model):
    proc = subprocess.run(
        [runner_bin, "-m", str(model), "-p", "hello world", "-n", "12",
         "--temp", "0", "--gpu", "off"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    return proc.stdout


def test_ungated_mlp_loads_and_decodes(runner_bin, tmp_path):
    """No ffn_gate tensor anywhere, so a gated loader cannot read this file."""
    assert _gen(runner_bin, _make(tmp_path, "IDENT"))


def test_xielu_parameters_are_applied(runner_bin, tmp_path):
    """Identity parameters and real ones must not produce the same output."""
    ident = _gen(runner_bin, _make(tmp_path, "IDENT"))
    real = _gen(runner_bin, _make(tmp_path, "REAL"))
    assert ident != real, "xIELU parameters are being ignored"


def test_cuda_xielu_matches_cpu(runner_bin, tmp_path):
    caps = json.loads(subprocess.run(
        [runner_bin, "--caps"], cwd=ROOT, stdout=subprocess.PIPE,
        check=True, text=True).stdout)
    if (caps.get("gpu") or {}).get("backend") != "cuda":
        pytest.skip("CUDA device not available")

    model = _make(tmp_path, "REAL")
    base = [runner_bin, "-m", str(model), "-p", "hello world", "-n", "12",
            "--temp", "0"]
    cpu = subprocess.run(
        [*base, "--gpu", "off"], cwd=ROOT, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, timeout=120)
    cuda = subprocess.run(
        [*base, "--gpu", "auto"], cwd=ROOT, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, timeout=120)
    assert cpu.returncode == 0, cpu.stderr.decode(errors="replace")
    assert cuda.returncode == 0, cuda.stderr.decode(errors="replace")
    assert b"CUDA backend" in cuda.stderr
    assert b"xIELU / ungated MLP has no device kernel" not in cuda.stderr
    assert cuda.stdout == cpu.stdout
