"""Regression gate for Runner's truncation-recovery property.

The product headline is "tool calls survive the token limit": a schema-
constrained tool call stays a parseable ``tool_calls`` entry even when the
generation budget is exhausted mid-object, with ``finish_reason: "length"`` and
no framing leaking into ``content``. This test locks that behaviour so a runner
regression that loses force-close, drops the call, mislabels finish_reason, or
leaks framing turns the gate red.

Two layers:
  * ``check_runner_property`` is exercised directly against synthetic records —
    a passing ladder passes, and each way the property can break is shown to be
    caught (the checker is not vacuous);
  * an end-to-end spawn drives the real ladder against the CPU fixture. The
    property is an engine guarantee, not model quality, so the tiny fixture
    exhibits it and the gate needs no GPU or competitor.
"""

import importlib.util
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "truncation_benchmark", ROOT / "scripts" / "truncation-benchmark.py")
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def _good_records():
    """A ladder that satisfies the property: every truncated rung is a
    parseable, length-finished call with no leak; the control rung completes."""
    records = []
    for rung in MOD.LADDER:
        records.append({
            "max_tokens": rung,
            "finish_reason": "tool_calls" if rung == MOD.CONTROL_RUNG else "length",
            "tool_calls_present": True,
            "tool_call_count": 1,
            "tool_name": "get_weather",
            "arguments_raw": '{"city":"Paris","units":"fahrenheit"}',
            "arguments_parseable": True,
            "arguments": {"city": "Paris", "units": "fahrenheit"},
            "content": "",
            "content_leak": False,
        })
    return records


def test_property_holds_on_a_clean_ladder():
    assert MOD.check_runner_property(_good_records()) == []


def test_ladder_shape_is_the_pinned_contract():
    # the ladder and control rung are load-bearing data, not incidental
    assert MOD.LADDER == (1, 2, 3, 5, 8, 16, 64)
    assert MOD.CONTROL_RUNG == 64
    assert MOD.CONTROL_RUNG not in MOD.TRUNCATED_RUNGS
    assert set(MOD.TRUNCATED_RUNGS) == {1, 2, 3, 5, 8, 16}


def test_unparseable_arguments_are_caught():
    recs = _good_records()
    recs[0]["arguments_parseable"] = False
    recs[0]["arguments_raw"] = '{"city":"Par'
    violations = MOD.check_runner_property(recs)
    assert any("do not parse" in v for v in violations), violations


def test_missing_tool_call_is_caught():
    # vLLM's observed truncation failure: length finish, empty tool_calls list
    recs = _good_records()
    recs[1]["tool_calls_present"] = False
    recs[1]["tool_call_count"] = 0
    violations = MOD.check_runner_property(recs)
    assert any("no tool_calls" in v for v in violations), violations


def test_content_leak_is_caught():
    recs = _good_records()
    recs[2]["content"] = '<tool_call>{"name": "get_wea'
    recs[2]["content_leak"] = True
    violations = MOD.check_runner_property(recs)
    assert any("leaked into content" in v for v in violations), violations


def test_wrong_finish_reason_on_truncated_rung_is_caught():
    recs = _good_records()
    recs[0]["finish_reason"] = "stop"
    violations = MOD.check_runner_property(recs)
    assert any("'length'" in v for v in violations), violations


def test_control_rung_must_complete():
    # if even the 64-token control truncates, the ladder proves nothing about
    # truncation recovery — it is just misconfigured
    recs = _good_records()
    recs[-1]["finish_reason"] = "length"
    violations = MOD.check_runner_property(recs)
    assert any("control" in v for v in violations), violations


def test_request_is_identical_across_engines_bar_the_model_field():
    a = MOD.request_for(8)
    b = MOD.request_for(8, model="granite")
    assert a["tool_choice"] == "required"
    assert a["temperature"] == 0
    assert a["max_tokens"] == 8
    assert a["tools"] == [MOD.TOOL]
    assert "model" not in a
    assert b["model"] == "granite"
    # everything else identical, so both engines answer the same question
    assert {k: v for k, v in b.items() if k != "model"} == a


def test_endpoint_parsing_rejects_off_box_hosts():
    assert MOD.parse_endpoint("127.0.0.1:8000") == 8000
    assert MOD.parse_endpoint("http://localhost:8000/") == 8000
    for bad in ("127.0.0.1", "10.0.0.5:8000", "example.com:8000"):
        with pytest.raises(ValueError):
            MOD.parse_endpoint(bad)


def _find_runner_and_model():
    import sys
    sys.path.insert(0, str(ROOT / "tests" / "conformance"))
    from harness import find_runner  # noqa: E402
    exe = os.environ.get("RUNNER_EXE") or str(ROOT / "runner")
    model = ROOT / "test.gguf"
    if not (Path(exe).exists() and model.is_file()):
        pytest.skip("runner binary or test.gguf not built; run `make test`")
    return exe, model


def test_end_to_end_runner_survives_the_ladder():
    exe, model = _find_runner_and_model()
    import sys
    sys.path.insert(0, str(ROOT / "tests" / "conformance"))
    from harness import Client, RunnerServer  # noqa: E402

    with RunnerServer(exe, str(model), ctx=1024, parallel=1,
                      extra_args=["--gpu", "off"]) as srv:
        client = Client(srv, MOD._Sink())
        records = MOD.run_ladder(client)
    violations = MOD.check_runner_property(records)
    assert violations == [], violations
    # spot-check the observables the headline names, on a real response
    by_rung = {r["max_tokens"]: r for r in records}
    assert by_rung[1]["finish_reason"] == "length"
    assert by_rung[1]["tool_calls_present"] and by_rung[1]["arguments_parseable"]
    assert by_rung[64]["finish_reason"] == "tool_calls"
