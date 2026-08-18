"""scripts/kld-compare.py: cross-model next-token-distribution comparison.

Gates: self-comparison must land at (approximately) zero KLD / 100% top-1
agreement — the same model, greedy, given the same prefix, always predicts
the same distribution; a real quant-vs-quant comparison must give sane,
nonzero divergence (quantization measurably perturbs the logits, even for a
tiny model, even at a fairly high bit width) without erroring out.
"""
import json
import os
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "kld-compare.py"
CORPUS_TEXT = (
    "The quick brown fox jumps over the lazy dog near the riverbank while "
    "the morning sun rises slowly above the distant mountains and a gentle "
    "breeze moves through the tall grass beside the old wooden fence."
)


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if os.name == "nt" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    return exe


@pytest.fixture(scope="module")
def small_model():
    m = ROOT / "models" / "SmolLM2-135M-Instruct-Q8_0.gguf"
    if not m.exists():
        pytest.skip(f"{m} not present on this box")
    return m


@pytest.fixture(scope="module")
def requant_model(runner_bin, small_model, tmp_path_factory):
    out = tmp_path_factory.mktemp("kld") / "smol-q4_0.gguf"
    proc = subprocess.run(
        [str(runner_bin), "-m", str(small_model), "--quantize", str(out), "--quant", "q4_0"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    return out


@pytest.fixture()
def corpus(tmp_path):
    p = tmp_path / "corpus.txt"
    p.write_text(CORPUS_TEXT)
    return p


def _run(*args):
    proc = subprocess.run([sys.executable, str(SCRIPT), *args],
                           cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                           timeout=180)
    return proc


def test_self_comparison_is_near_zero_kld_and_full_agreement(runner_bin, small_model, corpus):
    proc = _run("--model-a", str(small_model), "--model-b", str(small_model),
                "--runner", str(runner_bin), "--corpus", str(corpus),
                "--max-positions", "12", "--port-a", "58701", "--port-b", "58702")
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout)
    assert result["positions_scored"] >= 8, result
    assert result["mean_kld"] < 1e-4, (
        f"a model compared with itself must have ~0 KLD, got {result['mean_kld']}")
    assert result["top1_agreement_pct"] == 100.0, (
        f"a model compared with itself must agree on top-1 everywhere, got {result}")
    assert result["mean_top8_overlap"] == 1.0


def test_different_quants_give_sane_nonzero_divergence(runner_bin, small_model, requant_model, corpus):
    proc = _run("--model-a", str(small_model), "--model-b", str(requant_model),
                "--runner", str(runner_bin), "--corpus", str(corpus),
                "--max-positions", "12", "--port-a", "58703", "--port-b", "58704")
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout)
    assert result["positions_scored"] >= 8, result
    # "sane nonzero": some real divergence, but a Q8_0 vs Q4_0 requant of a
    # tiny model should still mostly agree — not so different it looks like
    # comparing two unrelated models.
    assert result["mean_kld"] > 1e-6, "Q8_0 vs Q4_0 of the same model should not be bit-identical"
    assert result["mean_kld"] < 5.0, f"divergence implausibly large for a same-model requant: {result}"
    assert 0.0 < result["top1_agreement_pct"] <= 100.0
    assert 0.0 <= result["mean_top8_overlap"] <= 1.0


def test_endpoint_mode_reuses_an_already_running_server(runner_bin, small_model, corpus):
    # kld-compare.py must also work against servers it didn't start itself
    # (Task 5a's pruned-vs-unpruned comparison will likely run this way).
    proc = subprocess.Popen(
        [str(runner_bin), "-m", str(small_model), "--serve", "--port", "58705", "--gpu", "off"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        import time
        import urllib.request
        for _ in range(200):
            try:
                urllib.request.urlopen("http://127.0.0.1:58705/v1/models", timeout=1)
                break
            except Exception:
                time.sleep(0.25)
        else:
            pytest.fail("test server did not come up")

        proc2 = _run("--endpoint-a", "http://127.0.0.1:58705",
                     "--model-name-a", small_model.name,
                     "--endpoint-b", "http://127.0.0.1:58705",
                     "--model-name-b", small_model.name,
                     "--corpus", str(corpus), "--max-positions", "6")
        assert proc2.returncode == 0, proc2.stderr
        result = json.loads(proc2.stdout)
        assert result["top1_agreement_pct"] == 100.0
    finally:
        proc.kill()
        proc.wait()


# kld-compare.py refuses a run in which nothing was scored ("error: no
# positions scored", exit 1). kld-compare-raw.py -- the v2 that publishes the
# ADOPTED margin-qualified criterion -- did not, so a dead endpoint, a wrong
# --model-name (the server 404s those) or a reference that reports no logprobs
# produced an all-null report and a clean exit code.

RAW_SCRIPT = ROOT / "scripts" / "kld-compare-raw.py"


def test_kld_compare_raw_refuses_a_run_that_scored_no_positions(corpus):
    proc = subprocess.run(
        [sys.executable, str(RAW_SCRIPT),
         "--endpoint-a", "http://127.0.0.1:1", "--model-name-a", "m",
         "--endpoint-b", "http://127.0.0.1:2", "--model-name-b", "m",
         "--corpus", str(corpus), "--max-positions", "4"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        timeout=180)

    result = json.loads(proc.stdout)
    assert result["positions_scored"] == 0
    assert result["positions_failed"] > 0
    # the evidence is still emitted, so --out records the failed run honestly
    assert proc.returncode == 1, proc.stdout
