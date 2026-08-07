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
import json
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


def _caps(runner_bin):
    out = subprocess.run([runner_bin, "--caps"], cwd=ROOT,
                         stdout=subprocess.PIPE, check=True).stdout
    return json.loads(out)


def test_caps_publishes_the_eseries_boolean(runner_bin):
    """A scheduler has to know whether this box can serve an E-series model on
    its GPU *before* it hands one over. The kernels shipped in 0.1.11; the
    visibility did not, so a scheduler had no way to tell a machine that runs
    E-series on Metal from one that silently falls back to CPU (issue #3)."""
    gpu = _caps(runner_bin)["gpu"]
    if gpu is None:
        pytest.skip("no GPU backend in this build")
    assert isinstance(gpu["eseries"], bool), gpu
    # Sits next to the other capability booleans, and means the same kind of
    # thing: a backend-wide claim, not a per-model promise.
    assert isinstance(gpu["moe"], bool) and isinstance(gpu["kv_q8"], bool)


def test_the_eseries_boolean_is_not_a_lie(runner_bin, tmp_path):
    """The failure mode a hardcoded literal has is drifting away from the code
    it describes. So make the claim falsifiable: if this backend says true, an
    E-series model must actually run on it rather than fall back."""
    # A null gpu block is how --caps says "no backend"; there is no separate
    # "available" key, and asking for one would skip this test on every machine.
    gpu = _caps(runner_bin)["gpu"]
    if gpu is None:
        pytest.skip("no GPU present to check the claim against")
    model = _make(tmp_path / "claim.gguf", shared_kv=3, ple=16)
    err = _gpu_run(runner_bin, model, tokens=1).stderr
    ran_on_gpu = b"CUDA backend" in err or b"Metal" in err
    if gpu["eseries"]:
        assert ran_on_gpu, (
            "caps claims E-series support but the model fell back:\n"
            + err.decode(errors="replace"))
    else:
        assert not ran_on_gpu, (
            "caps denies E-series support but the model ran on the device:\n"
            + err.decode(errors="replace"))


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
    # A silent mid-generation fallback also produces stdout that matches CPU
    # (the fallback IS the CPU oracle), so a stdout-only comparison cannot
    # tell "the GPU path ran and agreed" from "the GPU path never really ran
    # this split and CPU quietly did all the work". A missing per-layer
    # embedding tensor binding shipped exactly that way once: every partial
    # split fell back and every one of these parametrizations still passed.
    err = gpu.stderr.decode(errors="replace")
    assert "falling back to CPU" not in err, err
    assert "not resident on the device" not in err, err
    assert gpu.stdout == cpu.stdout


# --- per-layer FFN widths (the E2B export shape) -----------------------------
#
# gemma-4 E2B publishes REAL per-layer FFN width variation (6144/12288) as an
# ARRAY-typed feed_forward_length. A loader that only reads the scalar form
# silently sees 0 and refuses the whole model, so every E2B conversion (QAT or
# not) was REFUSED — cert-matrix roster item 7. The fixture alternates two
# widths on the same 6-layer E-series skeleton.

E2B_WIDTHS = "96,96,192,96,96,192"


def _make_varff(path):
    subprocess.run(
        [sys.executable, ROOT / "scripts/make-test-model.py",
         "--eseries", "3,16", "--ffn-widths", E2B_WIDTHS, str(path)],
        check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    return path


def test_array_ffn_widths_load_and_decode(runner_bin, tmp_path):
    model = _make_varff(tmp_path / "varff.gguf")
    proc = _run(runner_bin, model)
    err = proc.stderr.decode(errors="replace")
    assert proc.returncode == 0, err
    assert "missing model hyperparameters" not in err
    assert b"tok/s" in proc.stdout + proc.stderr


def test_array_ffn_widths_greedy_is_deterministic(runner_bin, tmp_path):
    model = _make_varff(tmp_path / "det-varff.gguf")
    outs = {_run(runner_bin, model, "--temp", "0", tokens=8).stdout
            for _ in range(3)}
    assert len(outs) == 1


def test_array_ffn_widths_on_gpu_are_identical_or_refused_loudly(runner_bin, tmp_path):
    """Per-layer FFN widths landed on Metal (0.1.11); CUDA still has no device
    path for them. Either is fine — computing with one global width is not.

    So: if a backend engaged, its output must match the CPU byte for byte; if
    it declined, it must say so rather than fall back silently.
    """
    model = _make_varff(tmp_path / "gpu-varff.gguf")
    cpu = _run(runner_bin, model, "--gpu", "off", tokens=8)
    gpu = _run(runner_bin, model, "--gpu", "auto", tokens=8,
               env={"RUNNER_METAL_MM": "0"})
    err = gpu.stderr.decode(errors="replace")
    engaged = "Metal backend" in err or "CUDA backend" in err
    if engaged:
        assert gpu.stdout == cpu.stdout, err
    else:
        assert "using CPU" in err or "no device path" in err, err
