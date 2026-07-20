"""Process-level lifecycle contracts for the serving runtime."""

import os
import signal
import json
import urllib.request

import pytest

from harness import RunnerServer, find_runner


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
