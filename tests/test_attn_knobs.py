"""Llama-4 attention knobs: NoPE and the position-dependent attention temperature.

Both are invisible to a text comparison on a small fixture — every variant here
generates byte-identical output — so this reads the activation trace instead.
`RUNNER_DEBUG_ACT=1` dumps per-tensor statistics for one forward pass, and the
post-rope Q row is exactly where skipping the rotation shows up.

The fixtures come from `make-test-model.py --attn-knobs STEP,TEMPSCALE`:

    k_off    no knobs (control)
    k_nope   NoPE on every layer          (step 1)
    k_half   NoPE on every second layer   (step 2)
    k_temp   NoPE on every layer, plus the temperature at scale 0.1
"""
import os
import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "test-attn"


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    return exe


def _q_rows(runner_bin, name):
    """The post-rope Q statistics line for each layer of one forward pass."""
    model = FIXTURES / f"{name}.gguf"
    if not model.exists():
        pytest.skip(f"fixture {model} not generated (see the test target)")
    proc = subprocess.run(
        [runner_bin, "-m", str(model), "-p", "hello world", "-n", "1",
         "--temp", "0", "--gpu", "off"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120,
        env={**os.environ, "RUNNER_DEBUG_ACT": "1"})
    rows = re.findall(r"^ACT L(\d+)\s+q-post-rope\s+(.*)$",
                      proc.stderr.decode(errors="replace"), re.M)
    assert rows, "no q-post-rope rows in the activation trace"
    return {int(layer): stats for layer, stats in rows}


def test_nope_changes_q(runner_bin):
    """Skipping the rotation is not cosmetic — Q must actually differ."""
    off, nope = _q_rows(runner_bin, "k_off"), _q_rows(runner_bin, "k_nope")
    assert off.keys() == nope.keys()
    assert all(off[l] != nope[l] for l in off), \
        "NoPE produced identical Q on every layer — the knob is not applied"


def test_layer_step_is_honoured(runner_bin):
    """step=2 must rope the layers step=1 skips, so the two cannot agree."""
    nope, half = _q_rows(runner_bin, "k_nope"), _q_rows(runner_bin, "k_half")
    off = _q_rows(runner_bin, "k_off")
    # layer 0: (0+1) % 2 != 0, so k_half ropes it and k_nope does not
    assert half[0] == off[0], "k_half should rope layer 0"
    assert half[0] != nope[0], "k_half and k_nope must differ on layer 0"
    # Only layer 0 isolates the step. Layer 1 skips rope under both fixtures,
    # but its Q is computed from layer 0's output — which differs, because
    # k_half roped layer 0 and k_nope did not. So "both skip rope on layer 1"
    # is true and still cannot be asserted as equal Q; checking it here would
    # be asserting that a residual stream forgets its first layer.


def test_temperature_is_a_noop_below_the_floor(runner_bin):
    """This one looks like a broken knob and is not.

    The scale is log(floor((pos + offset) / floor_scale) + 1) * scale + 1, and
    with llama-4's floor_scale of 8192 that is exactly 1.0 for every position
    below 8191. A short prompt must therefore be unaffected. Asserting it stops
    someone "fixing" the knob into applying at low positions, which would not
    match the reference.
    """
    assert _q_rows(runner_bin, "k_temp") == _q_rows(runner_bin, "k_nope")


def test_live_floor_temperature_actually_scales_q(runner_bin):
    """The knob's arithmetic, finally checked against something.

    Every other test here can pass with the temperature never applied: at
    llama-4's floor_scale of 8192 it is exactly 1.0 for every position a test
    prompt reaches, which is why the assertion above is a *no-op* check. The
    k_temp_live fixture sets floor_scale to 4 so the temperature is live at low
    positions, and then NoPE-plus-temperature must differ from NoPE alone.
    Without this, a backend could implement the temperature as `ts = 1.0` and
    every gate in this file would still be green.
    """
    live, nope = _q_rows(runner_bin, "k_temp_live"), _q_rows(runner_bin, "k_nope")
    assert live.keys() == nope.keys()
    assert all(live[l] != nope[l] for l in live), \
        "the live-floor temperature left Q unchanged — the knob is not applied"


def _gen(runner_bin, name, gpu):
    model = FIXTURES / f"{name}.gguf"
    if not model.exists():
        pytest.skip(f"fixture {model} not generated (see the test target)")
    proc = subprocess.run(
        [runner_bin, "-m", str(model), "-p", "hello world", "-n", "16",
         "--temp", "0", "--gpu", gpu],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180)
    return proc.stdout, proc.stderr.decode(errors="replace")


@pytest.fixture(scope="module")
def has_metal(runner_bin):
    """Whether this MACHINE has a Metal backend, independent of any model."""
    proc = subprocess.run([runner_bin, "--caps"], cwd=ROOT,
                          stdout=subprocess.PIPE, timeout=120)
    import json
    caps = json.loads(proc.stdout.decode())
    return ((caps.get("gpu") or {}).get("backend")) == "metal"


@pytest.mark.parametrize("name", ["k_nope", "k_half", "k_temp_live"])
def test_metal_matches_cpu_on_attn_knobs(runner_bin, has_metal, name):
    """CPU/GPU identity on the attention knobs, and proof the GPU ran.

    The engagement assertion is the whole point. Both knobs used to be a loud
    CPU fallback on Metal, and a fallback makes this compare the CPU against
    itself and pass for the wrong reason — the same trap `test-metal-kquant`
    guards against.
    """
    if not has_metal:
        pytest.skip("no Metal backend on this machine")
    cpu_out, _ = _gen(runner_bin, name, "off")
    gpu_out, gpu_err = _gen(runner_bin, name, "auto")
    # Ask the machine, not this run, whether Metal exists. Reading engagement
    # off THIS run's stderr would turn a refusal into a skip, which is exactly
    # the failure being guarded against.
    assert "Metal backend" in gpu_err, \
        f"{name} did not engage Metal: {gpu_err.strip().splitlines()[-3:]}"
    assert cpu_out == gpu_out, f"{name}: Metal output differs from CPU"
