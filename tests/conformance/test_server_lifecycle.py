"""Process-level lifecycle contracts for the serving runtime."""

import os
import signal
import subprocess
import time
import json
import urllib.request

import pytest

from harness import RunnerServer, find_runner, free_port


@pytest.mark.skipif(os.name == "nt", reason="POSIX signal contract")
@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM])
def test_signal_performs_clean_shutdown(sig):
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    model = os.environ.get("RUNNER_TEST_MODEL", os.path.join(root, "test.gguf"))
    srv = RunnerServer(find_runner(root), model, ctx=1024, parallel=2,
                       extra_args=["--gpu", "off"])
    srv.start()
    try:
        srv.proc.send_signal(sig)
        assert srv.proc.wait(timeout=10) == 0
    finally:
        srv.stop()


@pytest.mark.skipif(os.name == "nt", reason="POSIX signal contract")
def test_signal_during_startup_aborts_startup():
    """A SIGTERM that lands while models are still loading must abort startup.

    The one wrong outcome is a survivor: a signal swallowed mid-load used to
    leave the half-started server running forever, because the stop flag was
    only checked after accept() failed. The delays sweep the load window; a
    signal landing before the handlers install kills by default disposition,
    which is an equally acceptable exit."""
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    model = os.environ.get("RUNNER_TEST_MODEL", os.path.join(root, "test.gguf"))
    exe = find_runner(root)
    for delay in [0.0003, 0.0005, 0.0008, 0.001, 0.002, 0.005] * 2:
        proc = subprocess.Popen(
            [exe, "-m", model, "--serve", "--port", str(free_port()),
             "--parallel", "2", "-c", "1024", "--gpu", "off"],
            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        try:
            time.sleep(delay)
            proc.send_signal(signal.SIGTERM)
            try:
                rc = proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pytest.fail("server survived a SIGTERM sent %.1fms after spawn"
                            % (delay * 1000))
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
        assert rc in (0, -signal.SIGTERM)


@pytest.mark.skipif(os.name == "nt", reason="POSIX signal contract")
def test_second_signal_exits_immediately():
    """A second SIGINT during the shutdown drain must exit now, with 130.

    The drain waits for in-flight work, and a stalled client pins it for the
    full request-read budget; the operator's second Ctrl-C used to find the
    listener already closed, do nothing, and leave only SIGKILL. The stalled
    connection below holds a slot in its read loop to keep the drain open
    while the second signal lands."""
    import socket
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    model = os.environ.get("RUNNER_TEST_MODEL", os.path.join(root, "test.gguf"))
    srv = RunnerServer(find_runner(root), model, ctx=1024, parallel=2,
                       extra_args=["--gpu", "off"])
    srv.start()
    stalled = None
    try:
        stalled = socket.create_connection(("127.0.0.1", srv.port), timeout=15)
        stalled.sendall(b"POST /v1/chat/completions HTTP/1.1\r\n")  # never finishes
        time.sleep(0.5)                    # let a slot enter its read loop
        srv.proc.send_signal(signal.SIGINT)
        time.sleep(0.3)
        assert srv.proc.poll() is None, "drain should still be pinned by the stalled client"
        srv.proc.send_signal(signal.SIGINT)
        assert srv.proc.wait(timeout=3) == 130
    finally:
        if stalled is not None:
            stalled.close()
        srv.stop()


def _chat(base_url, **kw):
    body = json.dumps({"messages": [{"role": "user", "content": "hi"}],
                       "max_tokens": 8, **kw}).encode()
    req = urllib.request.Request(base_url + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def test_draft_model_survives_unload():
    """A --draft configured at startup must come back with the next reload.

    /unload frees the draft with the target; the lazy reload used to re-attach
    only a draft that still existed, so one /unload disabled speculative
    decoding for the life of the server."""
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    model = os.environ.get("RUNNER_TEST_MODEL", os.path.join(root, "test.gguf"))
    with RunnerServer(find_runner(root), model, ctx=1024, parallel=1,
                      extra_args=["--gpu", "off", "--draft", model]) as srv:
        assert _chat(srv.base_url)["runner_telemetry"]["speculative"] is True
        with urllib.request.urlopen(srv.base_url + "/unload", timeout=5) as r:
            assert json.load(r) == {"status": "ok"}
        assert _chat(srv.base_url)["runner_telemetry"]["speculative"] is True


def test_unload_clears_reported_model_context():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    model = os.environ.get("RUNNER_TEST_MODEL", os.path.join(root, "test.gguf"))
    with RunnerServer(find_runner(root), model, ctx=1024, parallel=1,
                      extra_args=["--gpu", "off"]) as srv:
        with urllib.request.urlopen(srv.base_url + "/unload", timeout=5) as r:
            assert json.load(r) == {"status": "ok"}
        with urllib.request.urlopen(srv.base_url + "/v1/capabilities", timeout=5) as r:
            caps = json.load(r)
        assert caps["resident"] is None
        assert caps["context"] == 0
