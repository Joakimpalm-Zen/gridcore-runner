"""granite (IBM Granite dense): the four muP scalars are the whole family.

The arch sat on the wrong-math refusal list because it loads llama-style
but computes wrong without embedding_scale, a FIXED attention scale,
residual_scale on both branch outputs, and a divided logit scale. The
fixtures pin the two behaviors a green build cannot fake: admission
(granite must load, not refuse) and that residual_scale actually
multiplies the branches (two fixtures differing only in that scalar must
diverge). Real-model math is gated against llama.cpp separately.
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


def _make(tmp_path, name, *extra):
    path = tmp_path / f"{name}.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py",
                    "--granite", *extra, str(path)],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    return path


def _gen(runner_bin, model):
    proc = subprocess.run(
        [runner_bin, "-m", str(model), "-p", "hello world", "-n", "12",
         "--temp", "0", "--gpu", "off"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    return proc.stdout


def test_granite_is_admitted_and_deterministic(runner_bin, tmp_path):
    """The arch must load (it was on the refusal list) and decode stably."""
    model = _make(tmp_path, "granite")
    assert _gen(runner_bin, model) == _gen(runner_bin, model)


def test_residual_scale_is_applied(runner_bin, tmp_path):
    """Same weights, different residual_scale: the outputs must differ. A
    build that reads the key but never multiplies the branches passes
    admission and determinism and fails exactly here."""
    a = _gen(runner_bin, _make(tmp_path, "g_r22", "--granite-resid", "0.22"))
    b = _gen(runner_bin, _make(tmp_path, "g_r90", "--granite-resid", "0.90"))
    assert a != b
