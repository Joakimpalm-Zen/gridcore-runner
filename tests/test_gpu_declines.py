"""Public behavior gates for GPU admission fallbacks.

Every unsupported model must name the feature that caused the backend to
decline.  A generic quantization message is actively misleading when all
tensors use a supported type, and a CPU-vs-auto comparison alone would pass
vacuously after an unexplained fallback.
"""
import pathlib
import subprocess
import sys

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
PROMPT = "diagnostic gate"


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    return exe


@pytest.fixture(scope="module")
def moe_fixtures(tmp_path_factory):
    prefix = tmp_path_factory.mktemp("gpu-declines") / "f"
    subprocess.run(
        [sys.executable, ROOT / "scripts/make-test-moe.py", str(prefix)],
        check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    return prefix


def _run(runner_bin, model, mode):
    return subprocess.run(
        [runner_bin, "-m", str(model), "-p", PROMPT, "-n", "4", "--temp",
         "0", "--gpu", mode],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)


def test_gated_sparse_moe_declines_loudly(runner_bin, moe_fixtures):
    model = f"{moe_fixtures}.afmoe-plain.gguf"
    cpu = _run(runner_bin, model, "off")
    auto = _run(runner_bin, model, "auto")
    assert cpu.returncode == 0, cpu.stderr.decode(errors="replace")
    assert auto.returncode == 0, auto.stderr.decode(errors="replace")
    assert cpu.stdout, "CPU produced no output; fallback comparison is vacuous"
    assert cpu.stdout == auto.stdout

    err = auto.stderr.decode(errors="replace")
    assert "cuda backend on" not in err.lower(), err
    assert "metal backend on" not in err.lower(), err
    assert "gated attention with sparse MoE (afmoe)" in err, err
    assert "quant type without a" not in err, err


@pytest.mark.parametrize(
    "flags,tensor",
    [
        (["--granite"], "output.weight"),
        ([], "blk.0.attn_q.weight"),
    ],
    ids=["output", "layer-weight"],
)
def test_unsupported_tensor_names_itself(runner_bin, tmp_path, flags, tensor):
    model = tmp_path / (tensor.replace(".", "-") + ".gguf")
    subprocess.run(
        [sys.executable, ROOT / "scripts/make-test-model.py", "--wide", *flags,
         "--gpu-unsupported", tensor, str(model)],
        check=True, cwd=ROOT, stdout=subprocess.DEVNULL)

    cpu = _run(runner_bin, model, "off")
    auto = _run(runner_bin, model, "auto")
    assert cpu.returncode == 0, cpu.stderr.decode(errors="replace")
    assert auto.returncode == 0, auto.stderr.decode(errors="replace")
    assert cpu.stdout, "CPU produced no output; fallback comparison is vacuous"
    assert cpu.stdout == auto.stdout

    err = auto.stderr.decode(errors="replace")
    assert tensor in err, err
    assert "IQ2_XXS" in err, err
    assert "using CPU" in err, err
