"""Mamba-2 hybrid admission and the granitehybrid decode step (tracer 2).

`granitehybrid` (Granite-4 h-series) now RUNS: the runner interleaves GQA
attention with a Mamba-2 selective-SSD recurrence, applies the granite muP
scaling and treats the attention layers as NoPE, gated token-identically
against llama.cpp b10353 on the real granite-4.0-h-small (a scripted check;
this file is the unit level). `nemotron_h_moe` (Nemotron-3.5 Lightning) shares
the tensor set but adds a grouped scan this tracer has not certified, so it is
still RECOGNIZED and refused with a specific "forward not yet implemented"
message — never the generic unknown-architecture refusal, which would read as a
typo. Either way, a missing required SSM tensor must FAIL CLOSED naming it (the
hostile-GGUF discipline). No download: the fixtures are generated and mirror a
real granite-4.0-h / Nemotron GGUF (sparse-MoE, NoPE, inner == 2*embd).
"""
import os
import pathlib
import re
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


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    d = tmp_path_factory.mktemp("hybrid")
    out = {}
    for arch in ("granitehybrid", "nemotron_h_moe"):
        pfx = d / arch
        subprocess.run(
            [sys.executable, ROOT / "scripts/make-test-hybrid.py", str(pfx),
             "--arch", arch], check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
        out[arch] = (pfx.with_suffix(".gguf"),
                     d / (arch + ".missing-ssm_d.gguf"))
    # granite-4.0-h-micro shape: expert_count 0, dense gated MLP on every layer
    pfx = d / "granitehybrid-dense"
    subprocess.run(
        [sys.executable, ROOT / "scripts/make-test-hybrid.py", str(pfx),
         "--dense"], check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    out["granitehybrid-dense"] = (pfx.with_suffix(".gguf"), None)
    return out


def _run(runner_bin, model, n="1"):
    return subprocess.run(
        [runner_bin, "-m", str(model), "-p", "hi", "-n", n, "--temp", "0",
         "--gpu", "off"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)


# A prompt long enough that the byte-level fixture tokenizer yields > 100 tokens,
# so XR_SSM_CHUNK=2 splits the Mamba-2 prefill into dozens of chunks and crosses
# the inter-chunk state boundary many times.
CHUNK_PROMPT = ("hello world this is a somewhat longer prompt used to exercise "
                "the chunked mamba-2 prefill scan over several tokens")


def _run_p(runner_bin, model, prompt, n, env=None):
    e = {**os.environ, **(env or {})}
    return subprocess.run(
        [runner_bin, "-m", str(model), "-p", prompt, "-n", str(n), "--temp", "0",
         "--gpu", "off"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120,
        env=e)


def _act_trace(stderr: bytes) -> bytes:
    """The RUNNER_DEBUG_ACT per-layer activation dump for the prefill pass -- a
    numeric fingerprint (per-layer sum to 6 dp + boundary values) far more
    sensitive than the greedy argmax, which on a tiny random fixture is too
    robust to reveal sub-argmax prefill drift (measured: a dropped inter-chunk
    state carry perturbs the trace but not the 16 greedy tokens)."""
    return b"\n".join(l for l in stderr.splitlines() if l.startswith(b"ACT "))


def _assert_chunked_equals_serial(runner_bin, model):
    """Tracer 3 gate: the chunked-scan prefill must be BIT-IDENTICAL to the serial
    per-token sweep -- chunking changes speed, not output. XR_SSM_SERIAL forces
    the reference path; the default and small chunk sizes take the chunked path
    (tiny sizes cross many inter-chunk state boundaries). Both the prefill
    activation trace AND the greedy token stream must match, at every chunk size.
    A regression in the chunked scan, the conv batching, or the inter-chunk
    state/ring carry moves the activation trace here."""
    dbg = {"RUNNER_DEBUG_ACT": "1"}
    ref = _run_p(runner_bin, model, CHUNK_PROMPT, 16, {"XR_SSM_SERIAL": "1", **dbg})
    assert ref.returncode == 0, ref.stderr.decode(errors="replace")
    ref_act = _act_trace(ref.stderr)
    assert ref_act, "no ACT trace captured; RUNNER_DEBUG_ACT not honored"
    m = re.search(rb"prompt: (\d+) tok", ref.stderr)
    assert m and int(m.group(1)) > 8, \
        "prompt must tokenize to many tokens so XR_SSM_CHUNK=2/3 spans chunks"
    for extra in ({}, {"XR_SSM_CHUNK": "2"}, {"XR_SSM_CHUNK": "3"},
                  {"XR_SSM_CHUNK": "128"}):
        got = _run_p(runner_bin, model, CHUNK_PROMPT, 16, {**dbg, **extra})
        assert got.returncode == 0, got.stderr.decode(errors="replace")
        label = extra or "default-chunk"
        assert _act_trace(got.stderr) == ref_act, (
            f"chunked prefill (env={label}) diverged from the serial sweep in the "
            f"activation trace")
        assert got.stdout == ref.stdout, (
            f"chunked prefill (env={label}) diverged from the serial sweep in the "
            f"token stream: {got.stdout[:200]!r} != {ref.stdout[:200]!r}")


def test_granitehybrid_loads_and_decodes(runner_bin, models):
    """The forward is implemented: granitehybrid must LOAD (not refuse) and
    decode. A build that regressed the loader back to a refusal, or that could
    not run the Mamba-2 step at all, fails here."""
    good, _ = models["granitehybrid"]
    proc = _run(runner_bin, good, n="8")
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    err = proc.stderr.decode(errors="replace")
    # the load announces the hybrid layout, not a refusal
    assert "granitehybrid" in err
    assert "forward not yet implemented" not in err


def test_granitehybrid_dense_loads_and_decodes(runner_bin, models):
    """The DENSE h-variant (granite-4.0-h-micro: expert_count 0, gated MLP on
    every layer). The loader used to bind ffn_gate/ffn_up only inside the
    attention branch, so a dense model's recurrent layers decoded with NULL
    FFN weights — a segfault on the first Mamba-2 layer of the real h-micro."""
    good, _ = models["granitehybrid-dense"]
    proc = _run(runner_bin, good, n="8")
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    err = proc.stderr.decode(errors="replace")
    assert "granitehybrid" in err
    assert "forward not yet implemented" not in err


def test_granitehybrid_dense_decode_is_deterministic(runner_bin, models):
    good, _ = models["granitehybrid-dense"]
    a = _run(runner_bin, good, n="8")
    b = _run(runner_bin, good, n="8")
    assert a.returncode == 0 and b.returncode == 0
    assert a.stdout == b.stdout


def test_granitehybrid_decode_is_deterministic(runner_bin, models):
    """Greedy decode must be reproducible run to run (the property the real
    token-identity gate builds on)."""
    good, _ = models["granitehybrid"]
    a = _run(runner_bin, good, n="8")
    b = _run(runner_bin, good, n="8")
    assert a.returncode == 0 and b.returncode == 0
    assert a.stdout == b.stdout


def test_nemotron_h_moe_loads_and_decodes(runner_bin, models):
    """The forward is implemented: nemotron_h_moe (Nemotron-3.5 Lightning — a
    gate-less squared-ReLU MoE hybrid with an always-on shared expert) must LOAD
    (not refuse) and decode. A regression back to the old refusal fails here."""
    good, _ = models["nemotron_h_moe"]
    proc = _run(runner_bin, good, n="8")
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    err = proc.stderr.decode(errors="replace")
    assert "nemotron_h_moe" in err
    assert "forward not yet implemented" not in err


def test_nemotron_h_moe_decode_is_deterministic(runner_bin, models):
    good, _ = models["nemotron_h_moe"]
    a = _run(runner_bin, good, n="8")
    b = _run(runner_bin, good, n="8")
    assert a.returncode == 0 and b.returncode == 0
    assert a.stdout == b.stdout


@pytest.mark.parametrize("arch", ["granitehybrid", "nemotron_h_moe"])
def test_missing_ssm_tensor_names_the_tensor(runner_bin, models, arch):
    _, broken = models[arch]
    proc = _run(runner_bin, broken)
    assert proc.returncode != 0, "a missing required SSM tensor must fail closed"
    err = proc.stderr.decode(errors="replace")
    assert "ssm_d" in err, "the failure must name the absent tensor"
    assert arch in err, "still recognized as the hybrid, just malformed"


# (nemotron_h_moe is now an admitted, implemented arch — the old
# "RUNNER_ALLOW_UNKNOWN_ARCH must not smuggle it into llama math" test retired
# with the refusal it guarded; the loads-and-decodes test above covers it.)


# --------------------------------------------------------------------------
# nemotron_h (Nemotron-Nano-9B-v2 family) — NON-MoE Mamba-2 / attention / MLP
# hybrid, graduated from refuse to load+decode (SSM tracer 3). Its grouped scan
# (ssm.group_count=2 in the fixture, 8 in the real model) is the crux Lightning
# was refused on; certified token-identically at the noise floor vs llama.cpp
# b10353 on the real Nemotron-Nano-9B-v2 Q8_0 (a scripted check; this is the
# unit level). Distinct fixture generator (non-MoE, three-way layer typing,
# gate-less squared-ReLU MLP, group_count>1, inner != 2*embd).
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def nemo_model(tmp_path_factory):
    d = tmp_path_factory.mktemp("nemotron_h")
    pfx = d / "nemotron_h"
    subprocess.run(
        [sys.executable, ROOT / "scripts/make-test-nemotron.py", str(pfx)],
        check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    return (pfx.with_suffix(".gguf"), d / "nemotron_h.missing-ssm_d.gguf")


def test_nemotron_h_loads_and_decodes(runner_bin, nemo_model):
    """nemotron_h (NON-MoE) RUNS: it must LOAD (not refuse) and decode the
    grouped Mamba-2 scan. A regression to a refusal, or an inability to run the
    n_group>1 step / the three-way (SSM|attn|MLP) block typing, fails here."""
    good, _ = nemo_model
    proc = _run(runner_bin, good, n="8")
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    err = proc.stderr.decode(errors="replace")
    assert "nemotron_h" in err
    assert "grouped scan" in err
    assert "forward not yet implemented" not in err
    assert "refusing to run it through llama-style math" not in err


def test_nemotron_h_decode_is_deterministic(runner_bin, nemo_model):
    """Greedy decode must be reproducible run to run (the property the real
    token-identity gate builds on)."""
    good, _ = nemo_model
    a = _run(runner_bin, good, n="8")
    b = _run(runner_bin, good, n="8")
    assert a.returncode == 0 and b.returncode == 0
    assert a.stdout == b.stdout


def test_nemotron_h_missing_ssm_tensor_fails_closed(runner_bin, nemo_model):
    """The hostile-GGUF discipline holds for the graduated arch too: a missing
    required SSM tensor must FAIL CLOSED naming it, not decode past the map."""
    _, broken = nemo_model
    proc = _run(runner_bin, broken)
    assert proc.returncode != 0, "a missing required SSM tensor must fail closed"
    err = proc.stderr.decode(errors="replace")
    assert "ssm_d" in err, "the failure must name the absent tensor"
    assert "nemotron_h" in err


# --------------------------------------------------------------------------
# Tracer 3 -- Mamba-2 chunked-scan prefill. The chunked path (conv batched, scan
# tiled into chunks and run in parallel across heads, SSD state + conv ring
# carried across chunk boundaries) is bit-identical to the serial per-token
# sweep by construction; these pin that so a chunking regression cannot ship as
# a silent speed-for-correctness trade. Covers n_group=1 (granite) and n_group>1
# (nemotron) grouped scans, and tiny chunk sizes that cross many boundaries.
# --------------------------------------------------------------------------
def test_granitehybrid_chunked_prefill_matches_serial(runner_bin, models):
    good, _ = models["granitehybrid"]
    _assert_chunked_equals_serial(runner_bin, good)


def test_nemotron_h_chunked_prefill_matches_serial(runner_bin, nemo_model):
    good, _ = nemo_model
    _assert_chunked_equals_serial(runner_bin, good)
