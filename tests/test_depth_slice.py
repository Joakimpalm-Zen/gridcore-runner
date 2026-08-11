"""gguf-depth-slice.py: drop whole layers from a GGUF, keep semantics.

The muse fixture is the hard case on purpose: its per-layer bool
sliding_window_pattern array must be filtered to the survivors (a slice
that kept the old array would misassign SWA/NoPE roles to every layer
after the first cut), and its tensors must renumber densely or the
loader's need_tensor sweep fails.
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


def _slice(src, dst, drop):
    subprocess.run([sys.executable, ROOT / "scripts/gguf-depth-slice.py",
                    str(src), str(dst), drop],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)


def _gen(runner_bin, model):
    proc = subprocess.run(
        [runner_bin, "-m", str(model), "-p", "hello world", "-n", "12",
         "--temp", "0", "--gpu", "off"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    return proc.stdout


def test_sliced_model_loads_and_decodes(runner_bin, tmp_path):
    base = tmp_path / "muse.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py",
                    "--muse-glimmer", str(base)],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    sliced = tmp_path / "muse-keep6.gguf"
    # drop one sliding layer and one full/NoPE layer: the surviving pattern
    # array is irregular, which is exactly what the filter must preserve
    _slice(base, sliced, "5,7")
    out_base = _gen(runner_bin, base)
    out_sliced = _gen(runner_bin, sliced)
    assert out_sliced  # loads, decodes
    assert out_base != out_sliced  # the cut changed the function
    assert out_sliced == _gen(runner_bin, sliced)  # deterministically


def test_slice_refuses_mtp_models(tmp_path):
    base = tmp_path / "mtp.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py",
                    "--mtp-layers", "1", str(base)],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    proc = subprocess.run([sys.executable, ROOT / "scripts/gguf-depth-slice.py",
                           str(base), str(tmp_path / "out.gguf"), "0"],
                          cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.returncode != 0
    assert b"NextN/MTP" in proc.stderr + proc.stdout
