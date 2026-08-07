"""Split (multi-part) GGUF files are refused by name, not by symptom.

runner reads exactly one file's tensor table. Handed part 1 of a 3-part split
it used to bind whatever that part held and then print a wall of

    error: missing tensor blk.0.ffn_norm.weight
    error: missing tensor blk.0.ffn_gate.weight
    ...

— every line true, none of them the reason, and `split.count` sitting in
metadata the loader had already parsed.

The fixture is a real split, produced by scripts/gguf-split.py, which follows
llama.cpp's layout verbatim (`split.no`, `split.count`,
`split.tensors.count`, and `<prefix>-%05d-of-%05d.gguf`). Testing against a
file merely *named* like a shard would prove nothing: the refusal keys on the
metadata, which is the only thing that is actually authoritative.

Loading a whole split model is a separate, larger piece of work — see
goal-runner-cleanup-results-2026-08-07.md §6 in the suite for why the single
mmap base makes it more than a loader change.
"""
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
    m = tmp_path_factory.mktemp("split") / "whole.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", str(m)],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    return m


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
        [runner_bin, "-m", str(model), "-p", "hi", "-n", "4", "--gpu", "off"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)


@pytest.mark.parametrize("which", range(N_SHARDS))
def test_every_shard_is_refused_by_name(runner_bin, shards, which):
    """Not just part 1 — a user who hands over part 3 gets the same answer."""
    p = _run(runner_bin, shards[which])
    err = p.stderr.decode(errors="replace")
    assert p.returncode != 0, err
    assert f"part {which + 1} of a {N_SHARDS}-part split model" in err, err
    # the old symptom must not be what the user reads first
    assert "missing tensor" not in err, err


def test_the_refusal_names_the_whole_set_and_the_fix(runner_bin, shards):
    err = _run(runner_bin, shards[0]).stderr.decode(errors="replace")
    for i in range(1, N_SHARDS + 1):
        assert f"part-{i:05d}-of-{N_SHARDS:05d}.gguf" in err, err
    assert "--merge" in err, err


def test_an_ordinary_single_file_model_is_untouched(runner_bin, whole):
    """The guard keys on split.count > 1, so a normal model must not notice
    it exists. This is the assertion that catches a guard that fires on
    everything."""
    p = _run(runner_bin, whole)
    err = p.stderr.decode(errors="replace")
    assert p.returncode == 0, err
    # match the refusal's own words, not the bare word "split" — the tmp path
    # this fixture lives in contains it, which is how this assertion first
    # failed for a reason that had nothing to do with the code
    assert "split model" not in err, err


def test_the_shards_are_real_ggufs(shards):
    """If the splitter emitted garbage, the refusal above would be testing
    nothing — a malformed file is rejected long before the split check."""
    for s in shards:
        with open(s, "rb") as f:
            assert f.read(4) == b"GGUF", s
