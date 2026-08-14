"""`--chat-template` must reach the server, or be refused.

The flag was parsed into `tmpl_arg` and applied only inside the interactive
chat block, which `--serve` returns before ever reaching. So a documented
public flag was accepted on the command line and silently discarded on the
path most callers take: forcing a template under `--serve` produced a
byte-identical answer and an identical prompt_tokens count to auto-detection.

Two behaviours are pinned here:

  * Serving ONE model, the override reaches the slot. Checked by forcing two
    different templates and requiring the rendered prompt to differ -- which
    is what a template IS. Deliberately not written as "forced X equals the
    auto-detected X", because that passes just as well when the flag does
    nothing at all, which is the bug.
  * Serving a SWAP SET (-m "a=x,b=y") -- TWO OR MORE models -- the flag is
    refused. It names one template; the set holds several models, each with
    its own detection. A single global override would be right for at most one
    of them and would silently mis-render the rest, so it is rejected rather
    than applied or ignored. `-m name=path` on its own is a one-entry registry
    that only pins the /v1/models id, and it still takes the override.

Plus: an unknown template name is an error rather than a quiet fall back to
auto-detection, which is the same silent discard in a smaller place.
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


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    return exe


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    m = tmp_path_factory.mktemp("m") / "test.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", str(m)],
                   check=True, cwd=ROOT)
    return m


def _free_port():
    with contextlib.closing(socket.socket()) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextlib.contextmanager
def _serve(runner_bin, model_spec, *extra):
    port = _free_port()
    proc = subprocess.Popen(
        # --no-tray is load-bearing, not tidiness: the tray deliberately
        # OUTLIVES the session that raised it (README: "LEFT RUNNING
        # afterwards"), so without it every server this fixture starts
        # leaks a process that holds runner.exe open and the next build
        # fails to link on Windows with "Permission denied".
        [str(runner_bin), "-m", str(model_spec), "--serve", "--port", str(port),
         "-c", "512", "--gpu", "off", "--no-tray", *extra],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        base = f"http://127.0.0.1:{port}"
        deadline = time.time() + 60
        while time.time() < deadline:
            if proc.poll() is not None:
                raise AssertionError(
                    "server exited during startup: "
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


def _prompt_tokens(base, model_id=None):
    body = {
        "messages": [{"role": "system", "content": "You answer briefly."},
                     {"role": "user", "content": "How is the weather today?"}],
        "max_tokens": 1,
        "temperature": 0,
    }
    if model_id:
        body["model"] = model_id
    payload = json.dumps(body).encode()
    req = urllib.request.Request(base + "/v1/chat/completions", data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = json.load(resp)
    return body["usage"]["prompt_tokens"]


def test_forced_template_reaches_the_served_slot(runner_bin, model):
    with _serve(runner_bin, model, "--chat-template", "llama2") as base:
        llama2 = _prompt_tokens(base)
    with _serve(runner_bin, model, "--chat-template", "gemma") as base:
        gemma = _prompt_tokens(base)
    assert llama2 != gemma, (
        f"--chat-template did not reach the server: llama2 and gemma both "
        f"rendered {llama2} prompt tokens")


def test_one_entry_registry_still_takes_the_override(runner_bin, model):
    """`-m name=path` pins the /v1/models id. It is ONE model, so the flag
    applies -- and it reaches the slot through the lazy registry load, which
    is the same code path a reload after /unload or a --ttl expiry takes."""
    with _serve(runner_bin, f"solo={model}", "--chat-template", "llama2") as base:
        llama2 = _prompt_tokens(base, "solo")
    with _serve(runner_bin, f"solo={model}", "--chat-template", "gemma") as base:
        gemma = _prompt_tokens(base, "solo")
    assert llama2 != gemma, (
        f"--chat-template did not reach a one-entry registry: llama2 and "
        f"gemma both rendered {llama2} prompt tokens")


def _run(runner_bin, *args):
    # --no-tray for the same reason _serve passes it: the swap-set case below
    # invokes --serve, and the tray is raised BEFORE the refusal is printed,
    # then deliberately outlives the session. A one-shot -p never raises one,
    # but passing it unconditionally keeps that from being a trap later.
    return subprocess.run([str(runner_bin), "--gpu", "off", "--no-tray", *args],
                          cwd=ROOT,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          timeout=60)


def test_chat_template_is_refused_for_a_swap_set(runner_bin, model):
    proc = _run(runner_bin, "-m", f"a={model},b={model}", "--serve",
                "--port", str(_free_port()), "--chat-template", "llama2")
    assert proc.returncode != 0
    err = proc.stderr.decode(errors="replace")
    assert "--chat-template" in err
    # the message has to say WHY, not just no: the caller's next move is to
    # serve one model per instance
    assert "swap" in err


def test_unknown_chat_template_is_an_error(runner_bin, model):
    proc = _run(runner_bin, "-m", str(model), "-p", "hi", "-n", "1",
                "--chat-template", "no-such-template")
    assert proc.returncode != 0
    err = proc.stderr.decode(errors="replace")
    assert "no-such-template" in err
