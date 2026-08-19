"""End-to-end for the certified-envelope load gate (slice 3).

A model with an OUTSIDE-envelope sidecar for THIS runtime is refused at load;
--force-uncertified overrides it; certified/experimental sidecars always load.
The sidecar's (version, backend) must match what the runner resolves, so both
are read back from `--version` / `--caps` rather than assumed — the gate is
exact-match and platform-dependent (metal on Apple, cuda/cpu elsewhere).
"""
import json
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNNER = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")


def _runtime_version():
    out = subprocess.run([RUNNER, "--version"], cwd=ROOT, check=True,
                         stdout=subprocess.PIPE, text=True).stdout.strip()
    return out  # "runner X.Y.Z-..."


def _backend():
    caps = json.loads(subprocess.run([RUNNER, "--caps"], cwd=ROOT, check=True,
                                     stdout=subprocess.PIPE, text=True).stdout)
    return (caps.get("gpu") or {}).get("backend", "cpu")


def _manifest(verdict, checks=None):
    m = {
        "schema_version": "xyntetik.runner.envelope.v1",
        "runtime": {"version": _runtime_version(),
                    "kernel_set": {"backend": _backend()}},
        "verdict": verdict,
    }
    if checks is not None:
        m["quality"] = {"checks": checks}
    return json.dumps(m)


@pytest.fixture
def model(tmp_path):
    p = tmp_path / "gate.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", str(p)],
                   check=True, cwd=ROOT)
    return p


def _run(model, *extra):
    return subprocess.run(
        [RUNNER, "-m", str(model), "-p", "hi", "-n", "1", "--gpu", "off",
         "-c", "32", *extra],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)


def test_outside_envelope_refuses_load(model):
    (model.parent / (model.name + ".envelope.json")).write_text(
        _manifest("outside-envelope", {"cpu_gpu_identity": "pass",
                                       "ram_fits": "fail"}))
    proc = _run(model)
    assert proc.returncode != 0, "an outside-envelope verdict must refuse the load"
    err = proc.stderr.decode(errors="replace")
    assert "refusing" in err and "--force-uncertified" in err, err
    assert "ram_fits" in err, "the refusal must name the measured reason: " + err


def test_force_uncertified_overrides(model):
    (model.parent / (model.name + ".envelope.json")).write_text(
        _manifest("outside-envelope", {"ram_fits": "fail"}))
    proc = _run(model, "--force-uncertified")
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    assert "WARNING" in proc.stderr.decode(errors="replace")


def test_certified_loads_with_banner(model):
    (model.parent / (model.name + ".envelope.json")).write_text(
        _manifest("certified"))
    proc = _run(model)
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    assert "certified" in proc.stderr.decode(errors="replace")


def test_no_manifest_is_silent(model):
    proc = _run(model)
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    assert b"envelope:" not in proc.stderr
