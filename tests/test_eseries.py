"""Gemma-4 E-series: per-layer embeddings and shared-KV layers.

Both mechanisms are structural rather than numeric, so a tiny generated model
exercises them end to end without the 5 GB real file. What matters here:

* the shared-KV map is llama.cpp's rule exactly — a layer at or past
  ``n_layer - shared_kv_layers`` reads the last KV-owning layer *of its own
  sliding/full type*, which is ``kv_from_start - 2`` sliding and
  ``kv_from_start - 1`` full. An off-by-one here still produces fluent text,
  so nothing but an explicit assertion catches it.
* shared layers reserve no cache rows, so the KV allocation shrinks with them.
* the two mechanisms are independent: either alone must load and run.
* the CUDA path agrees with the host bit for bit, including when a partial
  offload puts a shared-KV layer on a different device from the layer whose
  rows it reads.

Token-level agreement with llama.cpp is measured separately, against the real
weights, by scripts/token_divergence.py.
"""
import os
import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]

# The fixture declares 6 layers with every third one a full-attention layer,
# so the sliding/full pattern is [S, S, F, S, S, F].
N_LAYER = 6
FULL_LAYERS = {2, 5}


def _make(path, shared_kv, ple):
    subprocess.run(
        [sys.executable, ROOT / "scripts/make-test-model.py",
         "--eseries", f"{shared_kv},{ple}", str(path)],
        check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    return path


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    return exe


def _run(runner_bin, model, *extra, env=None, tokens=4, prompt="hi"):
    return subprocess.run(
        [runner_bin, "-m", str(model), "-p", prompt, "-n", str(tokens),
         "--gpu", "off", *extra],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120,
        env={**os.environ, **(env or {})})


def _expected_map(shared_kv):
    """llama-model.cpp's reuse callback, written out independently."""
    kv_from_start = N_LAYER - shared_kv
    out = {}
    for layer in range(kv_from_start, N_LAYER):
        is_full = layer in FULL_LAYERS
        out[layer] = kv_from_start - (1 if is_full else 2)
    return out


def test_shared_kv_map_matches_the_reference_rule(runner_bin, tmp_path):
    model = _make(tmp_path / "shared.gguf", shared_kv=3, ple=16)
    proc = _run(runner_bin, model, env={"RUNNER_DEBUG_ACT": "1"}, tokens=1)
    found = {int(a): int(b) for a, b in
             re.findall(r"ACT L(\d+)\s+shared-kv src=(\d+)",
                        proc.stderr.decode(errors="replace"))}
    assert found == _expected_map(3)
    # every source must itself own a cache, or the alias reads unwritten rows
    assert all(src < N_LAYER - 3 for src in found.values())


def test_shared_layers_reserve_no_cache_rows(runner_bin, tmp_path):
    def kv_mb(shared_kv):
        model = _make(tmp_path / f"kv{shared_kv}.gguf", shared_kv, ple=16)
        out = _run(runner_bin, model, "-v", tokens=1).stderr.decode(errors="replace")
        match = re.search(r"kv cache\s+([\d.]+) MB", out)
        assert match, out
        return float(match.group(1))

    full, shared = kv_mb(0), kv_mb(3)
    # 3 of 6 layers stop owning rows, and every layer here has the same KV
    # geometry, so the allocation should halve
    assert shared == pytest.approx(full / 2, rel=0.05)


@pytest.mark.parametrize("shared_kv,ple", [(3, 16), (3, 0), (0, 16)])
def test_each_mechanism_loads_and_decodes_on_its_own(runner_bin, tmp_path,
                                                     shared_kv, ple):
    model = _make(tmp_path / f"e{shared_kv}_{ple}.gguf", shared_kv, ple)
    proc = _run(runner_bin, model)
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    assert b"tok/s" in proc.stdout + proc.stderr


def test_decoding_is_deterministic(runner_bin, tmp_path):
    model = _make(tmp_path / "det.gguf", shared_kv=3, ple=16)
    outs = {_run(runner_bin, model, "--temp", "0", tokens=8).stdout
            for _ in range(3)}
    assert len(outs) == 1, "greedy decoding must not vary run to run"


# --- CUDA ---------------------------------------------------------------
# Both mechanisms have a device implementation; the gate is that it agrees with
# the host bit for bit. A silent GPU fallback would make these pass vacuously,
# so each one asserts the backend actually engaged.


GPU_PROMPT = "hello world"


def _gpu_run(runner_bin, model, *extra, tokens=12):
    return subprocess.run(
        [runner_bin, "-m", str(model), "-p", GPU_PROMPT, "-n", str(tokens),
         "--temp", "0", "--gpu", "auto", *extra],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)


def _requires_gpu(runner_bin, tmp_path):
    model = _make(tmp_path / "probe.gguf", shared_kv=3, ple=16)
    proc = _gpu_run(runner_bin, model, tokens=1)
    if b"CUDA backend" not in proc.stderr and b"Metal" not in proc.stderr:
        pytest.skip("no GPU backend available")
    return model


@pytest.mark.parametrize("shared_kv,ple", [(3, 16), (3, 0), (0, 16)])
def test_gpu_matches_cpu_bit_for_bit(runner_bin, tmp_path, shared_kv, ple):
    _requires_gpu(runner_bin, tmp_path)
    model = _make(tmp_path / f"g{shared_kv}_{ple}.gguf", shared_kv, ple)
    cpu = _run(runner_bin, model, "--temp", "0", tokens=12, prompt=GPU_PROMPT)
    gpu = _gpu_run(runner_bin, model)
    assert b"CUDA backend" in gpu.stderr or b"Metal" in gpu.stderr, \
        "GPU silently fell back; this comparison would be vacuous"
    assert gpu.stdout == cpu.stdout


@pytest.mark.parametrize("gpu_layers", [1, 2, 3, 4, 5])
def test_partial_offload_matches_cpu_across_the_shared_kv_boundary(
        runner_bin, tmp_path, gpu_layers):
    """The split can land before, on, or after the first shared-KV layer.

    With 6 layers and shared_kv=3 the tail starts at layer 3, so a layer that
    owns no cache can end up on a different device from the layer whose rows it
    reads. That boundary is where an aliasing mistake would show.
    """
    model = _requires_gpu(runner_bin, tmp_path)
    cpu = _run(runner_bin, model, "--temp", "0", tokens=12, prompt=GPU_PROMPT)
    gpu = _gpu_run(runner_bin, model, "--gpu-layers", str(gpu_layers))
    assert gpu.stdout == cpu.stdout
