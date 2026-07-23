"""CLI file inputs are read robustly (RNR-011).

A prompt/schema file that cannot be read must fail with a clear error, never a
crash from an unchecked ftell/fread; a readable prompt file is used.
"""
import pathlib
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / "runner"
    if not exe.exists():
        pytest.skip("runner binary not built")
    return exe


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    m = tmp_path_factory.mktemp("m") / "test.gguf"
    subprocess.run(["python3", ROOT / "scripts/make-test-model.py", str(m)],
                   check=True, cwd=ROOT)
    return m


def _run(runner_bin, model, *args):
    return subprocess.run([runner_bin, "-m", model, "--gpu", "off", *args],
                          cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          timeout=20)


def test_missing_prompt_file_is_a_clean_error(runner_bin, model):
    proc = _run(runner_bin, model, "-f", "/nonexistent/prompt.txt", "-n", "1")
    assert proc.returncode != 0
    assert b"cannot read" in proc.stderr


def test_missing_schema_file_is_a_clean_error(runner_bin, model):
    proc = _run(runner_bin, model, "-p", "hi", "--json-schema",
                "/nonexistent/schema.json", "-n", "1")
    assert proc.returncode != 0
    assert b"cannot read" in proc.stderr


def test_readable_prompt_file_loads(runner_bin, model, tmp_path):
    pf = tmp_path / "prompt.txt"
    pf.write_text("Hello from a file")
    proc = _run(runner_bin, model, "-f", str(pf), "-n", "1", "--temp", "0")
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
