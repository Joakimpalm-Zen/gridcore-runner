"""muse-glimmer (Meta Muse Glimmer 30B): dense, afmoe-style gated attention,
QK norms, sandwich norms at a fixed 1e-8 eps, SWA with a bool pattern array
whose full-attention layers are NoPE, a scaled+softcapped logit head, and an
UNTIED output projection.

The fixtures scale the real geometry down but keep its shape: head_dim is
deliberately decoupled from n_embd/n_head (the real model is 32x128 = 4096
against n_embd 6656), the sliding pattern is the real 3-local:1-global array,
and logit_scale/softcapping are declared. Real-model math is gated separately
against llama.cpp b10353 (greedy identity); these tests pin the structural
behaviors a green build cannot fake: the file loads through admission, decoding
is deterministic, and the gate/pattern tensors actually participate.
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
                    "--muse-glimmer", *extra, str(path)],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    return path


def _gen(runner_bin, model):
    proc = subprocess.run(
        [runner_bin, "-m", str(model), "-p", "hello world", "-n", "12",
         "--temp", "0", "--gpu", "off"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    return proc.stdout


def test_loads_and_decodes_deterministically(runner_bin, tmp_path):
    """Admission plus a stable greedy decode: same file, same output."""
    model = _make(tmp_path, "muse")
    assert _gen(runner_bin, model) == _gen(runner_bin, model)


def test_attention_gate_participates(runner_bin, tmp_path):
    """A zeroed attn_gate (sigmoid -> flat 0.5) must change the output; a
    build that loads the tensor but never multiplies it in would pass the
    determinism test and fail here."""
    gated = _gen(runner_bin, _make(tmp_path, "muse_gate"))
    flat = _gen(runner_bin, _make(tmp_path, "muse_flat", "--muse-gate-flat"))
    assert gated != flat


def test_swa_pattern_array_is_read(runner_bin, tmp_path):
    """The default fixture's 4th layer is full-attention AND NoPE (the muse
    rule: rope only on sliding layers). An all-sliding pattern array ropes
    every layer, so ignoring the array collapses the two fixtures."""
    patterned = _gen(runner_bin, _make(tmp_path, "muse_pat"))
    all_swa = _gen(runner_bin, _make(tmp_path, "muse_allswa", "--muse-all-swa"))
    assert patterned != all_swa
