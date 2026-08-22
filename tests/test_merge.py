"""--merge-lora at the CLI: the merged file serves what base + --lora serves.

The byte-level gates (exact fmaf chain, verbatim copies, hostile-adapter
refusals, requant grid bounds) live in test_quantize.c. What only an
end-to-end run can show is that the merged GGUF *behaves*: on the F32
fixture, scoring through the merged file matches scoring base+--lora to
float noise (same math, different summation order), and the provenance
record beside the artifact carries shas that match the actual files.
"""
import hashlib
import json
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SENT = "the merged model and the adapted model say the same thing."


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    return exe


@pytest.fixture(scope="module")
def fixtures():
    base, adapter = ROOT / "test.gguf", ROOT / "test-lora.adapter.gguf"
    if not base.exists() or not adapter.exists():
        pytest.skip("run `make test-lora.full.gguf` first")
    return base, adapter


def _merge(runner_bin, base, adapter, out, extra=()):
    return subprocess.run(
        [runner_bin, "-m", str(base), "--lora", str(adapter),
         "--merge-lora", str(out), *extra],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)


def _score(runner_bin, model, lora=None):
    cmd = [runner_bin, "-m", str(model), "--score", "-p", SENT, "-t", "2",
           "--gpu", "off"]
    if lora:
        cmd += ["--lora", str(lora)]
    p = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, timeout=120)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    return json.loads(p.stdout)["nll_mean"]


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_merged_scores_like_base_plus_lora(runner_bin, fixtures, tmp_path):
    base, adapter = fixtures
    merged = tmp_path / "merged.gguf"
    p = _merge(runner_bin, base, adapter, merged)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    served = _score(runner_bin, base, lora=adapter)
    folded = _score(runner_bin, merged)
    plain = _score(runner_bin, base)
    # same math up to summation order on the F32 fixture...
    assert abs(folded - served) < 1e-4, (folded, served)
    # ...and it is actually the adapted model, not the base
    assert abs(folded - plain) > 1e-3, (folded, plain)


def test_provenance_record_matches_files(runner_bin, fixtures, tmp_path):
    base, adapter = fixtures
    merged = tmp_path / "m.gguf"
    p = _merge(runner_bin, base, adapter, merged)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    rec = json.loads((tmp_path / "m.gguf.merge.json").read_text())
    assert rec["base"]["sha256"] == _sha(base)
    assert rec["adapter"]["sha256"] == _sha(adapter)
    assert rec["merged"]["sha256"] == _sha(merged)
    assert rec["target"] == "keep"


def test_quant_target_writes_that_type(runner_bin, fixtures, tmp_path):
    base, adapter = fixtures
    merged = tmp_path / "m8.gguf"
    p = _merge(runner_bin, base, adapter, merged, extra=("--quant", "q8_0"))
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    # the merged-quantized file loads and still behaves adapted (q8 grid
    # noise on top of the f32 delta, so the bound is looser)
    served = _score(runner_bin, base, lora=adapter)
    folded = _score(runner_bin, merged)
    assert abs(folded - served) < 5e-2, (folded, served)


def test_merge_lora_requires_lora(runner_bin, fixtures, tmp_path):
    base, _ = fixtures
    p = subprocess.run(
        [runner_bin, "-m", str(base), "--merge-lora",
         str(tmp_path / "x.gguf")],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
    assert p.returncode != 0
    assert b"--lora" in p.stderr
