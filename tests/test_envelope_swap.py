"""Measured-envelope enforcement on the swap path (gate slice 3b).

The CLI refuses an outside-envelope model by EXITING. A server must not: a swap
set holds several models, and one being outside its envelope is a per-request
refusal (HTTP 409), not a reason to kill a process that is happily serving the
others. --force-uncertified at startup suppresses it, matching the CLI flag.

The sidecar's (version, backend) is read back from --version / --caps so the
exact-match tuple holds on any platform (metal on Apple, cuda/cpu elsewhere).
"""
import contextlib
import json
import pathlib
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNNER = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")


@pytest.fixture(scope="module")
def runner_bin():
    if not RUNNER.exists():
        pytest.skip("runner binary not built")
    return RUNNER


def _free_port():
    with contextlib.closing(socket.socket()) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _runtime_version():
    # BARE --caps version (what the certifier writes + envelope.c matches), not
    # the "runner X" form --version prints.
    caps = json.loads(subprocess.run([RUNNER, "--caps"], cwd=ROOT, check=True,
                                     stdout=subprocess.PIPE, text=True).stdout)
    return caps["version"]


def _backend():
    caps = json.loads(subprocess.run([RUNNER, "--caps"], cwd=ROOT, check=True,
                                     stdout=subprocess.PIPE, text=True).stdout)
    return (caps.get("gpu") or {}).get("backend", "cpu")


def _write_outside_manifest(model_path):
    manifest = {
        "schema_version": "xyntetik.runner.envelope.v1",
        "runtime": {"version": _runtime_version(),
                    "kernel_set": {"backend": _backend()}},
        "verdict": "outside-envelope",
        "quality": {"checks": {"load": "pass", "ram_fits": "fail"}},
    }
    (model_path.parent / (model_path.name + ".envelope.json")).write_text(
        json.dumps(manifest))


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    d = tmp_path_factory.mktemp("swap")
    good, bad = d / "good.gguf", d / "bad.gguf"
    for m in (good, bad):
        subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", str(m)],
                       check=True, cwd=ROOT)
    _write_outside_manifest(bad)   # only `bad` is outside its envelope
    return good, bad


@contextlib.contextmanager
def _serve(runner_bin, spec, *extra):
    port = _free_port()
    proc = subprocess.Popen(
        [str(runner_bin), "-m", spec, "--serve", "--port", str(port),
         "-c", "512", "--gpu", "off", "--no-tray", *extra],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        base = f"http://127.0.0.1:{port}"
        deadline = time.time() + 60
        while time.time() < deadline:
            if proc.poll() is not None:
                raise AssertionError("server exited during startup: "
                                     + proc.stderr.read().decode(errors="replace"))
            try:
                with urllib.request.urlopen(base + "/health", timeout=1):
                    break
            except (urllib.error.URLError, OSError):
                time.sleep(0.1)
        else:
            raise AssertionError("server never answered /health")
        yield base
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=15)


def _chat(base, model_id):
    """Returns the HTTP status; raises nothing for 4xx/5xx (returns the code)."""
    payload = json.dumps({
        "model": model_id,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1, "temperature": 0,
    }).encode()
    req = urllib.request.Request(base + "/v1/chat/completions", data=payload,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status
    except urllib.error.HTTPError as e:
        return e.code


def test_swap_refuses_outside_envelope_but_keeps_serving(runner_bin, models):
    good, bad = models
    with _serve(runner_bin, f"good={good},bad={bad}") as base:
        # The outside-envelope model is refused with 409 — a policy refusal,
        # distinct from a broken model (5xx) or an unknown one (404).
        assert _chat(base, "bad") == 409
        # ...and the server is still up, serving the model that is in-envelope.
        assert _chat(base, "good") == 200
        # A second attempt at the refused model is still refused (not cached bad).
        assert _chat(base, "bad") == 409


def test_force_uncertified_lets_the_swap_load_proceed(runner_bin, models):
    good, bad = models
    with _serve(runner_bin, f"good={good},bad={bad}", "--force-uncertified") as base:
        assert _chat(base, "bad") == 200
