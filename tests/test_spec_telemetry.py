"""Speculative telemetry reports execution, not a configured draft."""

import json
import pathlib
import sys
import urllib.request


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "conformance"))

from harness import RunnerServer, find_runner  # noqa: E402


def _post_file(server, name):
    payload = (ROOT / "tests" / "fixtures" / name).read_bytes()
    request = urllib.request.Request(
        server.base_url + "/v1/completions", data=payload,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def test_logprobs_reports_speculation_as_not_engaged():
    """Logprob capture selects the solo walk even when --draft is loaded."""
    model = ROOT / "test.gguf"
    with RunnerServer(
        find_runner(ROOT), model, ctx=256, parallel=1,
        extra_args=["--gpu", "off", "--draft", str(model)],
    ) as server:
        plain = _post_file(server, "completion_spec.json")
        assert plain["runner_telemetry"]["speculative"] is True

        logged = _post_file(server, "completion_spec_logprobs.json")
        assert logged["choices"][0]["logprobs"]["token_logprobs"]
        assert logged["runner_telemetry"]["speculative"] is False
