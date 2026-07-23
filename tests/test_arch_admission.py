"""Architecture admission is an allowlist (RNR-004).

Tensor-name compatibility is not proof of mathematical compatibility, so an
unknown `general.architecture` must be refused rather than run through
llama-style math (which would emit plausible but silently wrong output). The
only escape is an explicit, clearly-labelled opt-in.
"""
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _make_model(path, arch):
    subprocess.run(
        [sys.executable, ROOT / "scripts/make-test-model.py", "--arch", arch, str(path)],
        check=True, cwd=ROOT,
    )


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / "runner"
    if not exe.exists():
        pytest.skip("runner binary not built")
    return exe


def test_unknown_arch_is_refused(runner_bin, tmp_path):
    model = tmp_path / "mystery.gguf"
    _make_model(model, "mystery")
    proc = subprocess.run(
        [runner_bin, "-m", model, "-p", "hi", "-n", "1", "--gpu", "off"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20,
    )
    assert proc.returncode != 0, "an unknown architecture must not load by default"
    err = proc.stderr.decode(errors="replace")
    assert "unsupported architecture 'mystery'" in err
    assert "refusing" in err


def test_opt_in_bypasses_admission_with_a_loud_warning(runner_bin, tmp_path):
    model = tmp_path / "mystery.gguf"
    _make_model(model, "mystery")
    proc = subprocess.run(
        [runner_bin, "-m", model, "-p", "hi", "-n", "1", "--gpu", "off"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20,
        env={**__import__("os").environ, "RUNNER_ALLOW_UNKNOWN_ARCH": "1"},
    )
    # The opt-in gets past admission (the "mystery.*" geometry keys exist, so it
    # actually loads and runs); what matters is the refusal is gone and the
    # warning is unmistakable.
    err = proc.stderr.decode(errors="replace")
    assert "is UNSUPPORTED" in err
    assert "silently wrong" in err
    assert "refusing" not in err


def test_known_arch_still_loads(runner_bin, tmp_path):
    model = tmp_path / "llama.gguf"
    _make_model(model, "llama")
    proc = subprocess.run(
        [runner_bin, "-m", model, "-p", "hi", "-n", "1", "--gpu", "off", "--temp", "0"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20,
    )
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
