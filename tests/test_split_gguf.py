"""Native multi-part GGUF loading through the public runner surface."""
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
N_SHARDS = 3


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    return exe


@pytest.fixture(scope="module")
def whole(tmp_path_factory):
    model = tmp_path_factory.mktemp("split") / "whole.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", str(model)],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    return model


@pytest.fixture(scope="module")
def shards(whole):
    subprocess.run(
        [sys.executable, ROOT / "scripts/gguf-split.py", str(whole),
         str(whole.parent / "part"), str(N_SHARDS)],
        check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    found = sorted(whole.parent.glob("part-*-of-*.gguf"))
    assert len(found) == N_SHARDS, found
    return found


def _run(runner_bin, model):
    return subprocess.run(
        [runner_bin, "-m", str(model), "-p", "hi", "-n", "4", "--temp", "0",
         "--gpu", "off"], cwd=ROOT, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, timeout=120)


@pytest.mark.parametrize("which", range(N_SHARDS))
def test_every_part_loads_the_complete_model(runner_bin, whole, shards, which):
    expected = _run(runner_bin, whole)
    actual = _run(runner_bin, shards[which])
    assert expected.returncode == 0, expected.stderr.decode(errors="replace")
    assert actual.returncode == 0, actual.stderr.decode(errors="replace")
    assert actual.stdout == expected.stdout


def test_missing_part_is_refused_loudly(runner_bin, shards):
    missing = shards[1]
    parked = missing.with_suffix(".parked")
    missing.rename(parked)
    try:
        result = _run(runner_bin, shards[0])
    finally:
        parked.rename(missing)
    error = result.stderr.decode(errors="replace")
    assert result.returncode != 0
    assert "missing or unreadable split GGUF part 2/3" in error
    assert str(missing) in error


def test_parts_are_real_ggufs(shards):
    for shard in shards:
        assert shard.read_bytes()[:4] == b"GGUF"
