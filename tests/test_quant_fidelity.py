"""scripts/quant-fidelity.py: unit tests for the quant-vs-tool-call-fidelity
harness (suite plan P2). No model or runner binary is needed here — every
server call is mocked, either via the harness's own Response/Stream classes
(constructed from hand-built HTTP/SSE bytes, exactly like
tests/test_agent_torture.py does) or via a fake client. The real end-to-end
path (spawn Runner, load a GGUF, hit it over the wire) is the separate smoke
run against models/granite-4.1-8b-Q4_0.gguf, not exercised by pytest.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "conformance"))

SPEC = importlib.util.spec_from_file_location(
    "quant_fidelity", ROOT / "scripts" / "quant-fidelity.py")
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)

from harness import Response, Stream  # noqa: E402


# --------------------------------------------------------------- fixtures
def _http_response(body_obj, status=200):
    body = json.dumps(body_obj).encode()
    headers = {"content-type": "application/json",
              "content-length": str(len(body))}
    return Response("test", status, headers, body, 1.0)


def _tool_call_response(name, arguments_obj, status=200):
    return _http_response({
        "choices": [{"message": {"role": "assistant", "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": name, "arguments": json.dumps(arguments_obj)}}
        ]}, "finish_reason": "tool_calls"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }, status)


def _final_response(document, status=200):
    return _http_response({
        "choices": [{"message": {"role": "assistant", "content": json.dumps(document)},
                    "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }, status)


def _sse_stream(chunks, finish_reason="stop"):
    body = b""
    for chunk in chunks:
        body += b"data: " + json.dumps(chunk).encode() + b"\n\n"
    body += b"data: [DONE]\n\n"
    headers = {"content-type": "text/event-stream"}
    return Stream("test", 200, headers, body, 1.0, 0.5)


def _tool_stream(name, arguments_obj):
    return _sse_stream([
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"name": name,
                                      "arguments": json.dumps(arguments_obj)}}]},
                     "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
    ])


def _prose_stream(text):
    return _sse_stream([
        {"choices": [{"delta": {"content": text}, "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
    ])


NESTED_REQUEST = {
    "tools": [{"function": {"name": "dispatch_job", "parameters": {
        "type": "object", "properties": {"mode": {"type": "string"}},
        "required": ["mode"], "additionalProperties": False}}}],
    "tool_choice": {"type": "function", "function": {"name": "dispatch_job"}},
}
FINAL_REQUEST = {
    "response_format": {"type": "json_schema", "json_schema": {"schema": {
        "type": "object", "properties": {"summary": {"type": "string"}},
        "required": ["summary"], "additionalProperties": False}}},
}
WEATHER_REQUEST = {
    "tools": [{"function": {"name": "lookup_weather", "parameters": {
        "type": "object", "properties": {"city": {"type": "string"}},
        "required": ["city"], "additionalProperties": False}}}],
    "tool_choice": {"type": "function", "function": {"name": "lookup_weather"}},
}


# ------------------------------------------------------------------ judging
def test_judge_tool_response_valid_call_is_executed_and_schema_valid():
    result = MOD.judge_tool_response(
        NESTED_REQUEST, _tool_call_response("dispatch_job", {"mode": "fast"}))
    assert result["executed"] is True
    assert result["schema_valid"] is True
    assert result["tool_name"] == "dispatch_job"
    assert result["arguments"] == {"mode": "fast"}
    assert result["error"] is None


def test_judge_tool_response_schema_violation_is_executed_but_invalid():
    # missing the required "mode" property
    result = MOD.judge_tool_response(
        NESTED_REQUEST, _tool_call_response("dispatch_job", {}))
    assert result["executed"] is True
    assert result["schema_valid"] is False
    assert result["error"]["category"] == "schema"


def test_judge_tool_response_malformed_json_arguments_is_executed_but_invalid():
    body = json.dumps({"choices": [{"message": {"tool_calls": [
        {"function": {"name": "dispatch_job", "arguments": "{not json"}}]},
        "finish_reason": "tool_calls"}]}).encode()
    resp = Response("test", 200, {"content-type": "application/json",
                                  "content-length": str(len(body))}, body, 1.0)
    result = MOD.judge_tool_response(NESTED_REQUEST, resp)
    assert result["executed"] is True
    assert result["schema_valid"] is False
    assert result["error"]["category"] == "schema"


def test_judge_tool_response_wrong_status_is_executed_but_invalid():
    result = MOD.judge_tool_response(
        NESTED_REQUEST, _http_response({"error": {"message": "nope"}}, status=500))
    assert result["executed"] is True
    assert result["schema_valid"] is False
    assert result["error"]["category"] == "protocol"


def test_judge_tool_response_preserves_tool_name_and_arguments_despite_schema_violation():
    # a schema violation must not erase which tool was picked or what it
    # was called with — tool selection and argument agreement are scored
    # independently of schema conformance, on purpose (P2's whole point is
    # telling syntax decay apart from semantic decay)
    result = MOD.judge_tool_response(NESTED_REQUEST,
                                     _tool_call_response("dispatch_job", {}))
    assert result["schema_valid"] is False
    assert result["tool_name"] == "dispatch_job"
    assert result["arguments"] == {}


def test_judge_tool_response_flags_a_tool_not_offered_as_a_protocol_error():
    # regression: the schema lookup must key off the tool the MODEL named,
    # never the tool_choice-forced name — otherwise a hallucinated call to
    # an unoffered tool silently validates against the wrong schema and
    # comes back schema_valid=True
    result = MOD.judge_tool_response(
        NESTED_REQUEST, _tool_call_response("made_up_tool", {"mode": "fast"}))
    assert result["executed"] is True
    assert result["tool_name"] == "made_up_tool"
    assert result["schema_valid"] is False
    assert result["error"]["category"] == "protocol"
    assert "not offered" in result["error"]["message"]


def test_judge_tool_response_schema_valid_for_a_correctly_offered_but_unwanted_tool():
    # the flip side: choosing the WRONG tool among several that were all
    # legitimately offered is a tool-selection miss, not a schema failure —
    # its own arguments still validate against its own schema
    two_tools_request = {
        "tools": [WEATHER_REQUEST["tools"][0],
                 {"function": {"name": "sum_values", "parameters": {
                     "type": "object", "properties": {"values": {"type": "array"}},
                     "required": ["values"], "additionalProperties": False}}}],
        "tool_choice": {"type": "function", "function": {"name": "lookup_weather"}},
    }
    result = MOD.judge_tool_response(
        two_tools_request, _tool_call_response("sum_values", {"values": [1, 2]}))
    assert result["tool_name"] == "sum_values"
    assert result["schema_valid"] is True  # valid against ITS OWN schema
    assert result["arguments"] == {"values": [1, 2]}


def test_judge_tool_stream_flags_a_tool_not_offered_as_a_protocol_error():
    result = MOD.judge_tool_stream(
        WEATHER_REQUEST, _tool_stream("made_up_tool", {"city": "Oslo"}))
    assert result["tool_name"] == "made_up_tool"
    assert result["schema_valid"] is False
    assert result["error"]["category"] == "protocol"


def test_judge_final_preserves_content_despite_schema_violation():
    result = MOD.judge_final(FINAL_REQUEST, _final_response({}))
    assert result["schema_valid"] is False
    assert result["content"] == {}  # not wiped by the schema failure


def test_judge_final_valid_and_invalid_document():
    ok = MOD.judge_final(FINAL_REQUEST, _final_response({"summary": "s"}))
    assert ok["schema_valid"] is True
    assert ok["content"] == {"summary": "s"}

    bad = MOD.judge_final(FINAL_REQUEST, _final_response({}))
    assert bad["executed"] is True
    assert bad["schema_valid"] is False
    assert bad["error"]["category"] == "schema"


def test_judge_final_rejects_a_tool_call_riding_the_final_turn():
    resp = _tool_call_response("dispatch_job", {"mode": "fast"})
    result = MOD.judge_final(FINAL_REQUEST, resp)
    assert result["executed"] is True
    assert result["schema_valid"] is False
    assert result["error"]["category"] == "protocol"


def test_judge_tool_stream_valid_call():
    result = MOD.judge_tool_stream(
        WEATHER_REQUEST, _tool_stream("lookup_weather", {"city": "Oslo"}))
    assert result["executed"] is True
    assert result["schema_valid"] is True
    assert result["tool_name"] == "lookup_weather"
    assert result["arguments"] == {"city": "Oslo"}


def test_judge_tool_stream_missing_done_is_invalid():
    headers = {"content-type": "text/event-stream"}
    raw = b'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}\n\n'
    stream = Stream("test", 200, headers, raw, 1.0, 0.5)
    result = MOD.judge_tool_stream(WEATHER_REQUEST, stream)
    assert result["executed"] is True
    assert result["schema_valid"] is False


def test_judge_prose_has_no_schema_contract_but_still_executes():
    result = MOD.judge_prose({}, _prose_stream("hello there"))
    assert result["executed"] is True
    assert result["schema_valid"] is None  # not applicable, not failed
    assert result["content"] == "hello there"


def test_judge_prose_transport_style_failure_still_marked_executed_with_error():
    headers = {"content-type": "text/event-stream"}
    raw = b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'  # no [DONE]
    stream = Stream("test", 200, headers, raw, 1.0, 0.5)
    result = MOD.judge_prose({}, stream)
    assert result["executed"] is True
    assert result["schema_valid"] is None
    assert result["error"]["category"] == "protocol"


# --------------------------------------------------------------- run_case
class _FakeClient:
    def __init__(self, chat_result=None, stream_result=None, raises=None):
        self.chat_result = chat_result
        self.stream_result = stream_result
        self.raises = raises

    def chat(self, payload, name=None):
        if self.raises:
            raise self.raises
        return self.chat_result

    def chat_stream(self, payload, name=None):
        if self.raises:
            raise self.raises
        return self.stream_result


def test_run_case_transport_failure_is_recorded_not_executed_never_dropped():
    case = {"id": "c1", "category": "nested_arguments", "request": NESTED_REQUEST}
    client = _FakeClient(raises=MOD.TransportError("connection refused"))
    result = MOD.run_case(client, case)
    assert result["id"] == "c1"
    assert result["executed"] is False
    assert "connection refused" in result["not_executed_reason"]
    assert "[transport]" in result["not_executed_reason"]


def test_run_case_success_delegates_to_the_right_judge():
    case = {"id": "c1", "category": "nested_arguments", "request": NESTED_REQUEST}
    client = _FakeClient(chat_result=_tool_call_response("dispatch_job", {"mode": "safe"}))
    result = MOD.run_case(client, case)
    assert result["executed"] is True
    assert result["tool_name"] == "dispatch_job"
    assert result["id"] == "c1"
    assert result["category"] == "nested_arguments"


def test_run_case_routes_stream_categories_through_chat_stream():
    case = {"id": "c1", "category": "stream_normalization", "request": {}}
    client = _FakeClient(stream_result=_prose_stream("ok"))
    result = MOD.run_case(client, case)
    assert result["executed"] is True
    assert result["content"] == "ok"


def test_run_matrix_covers_every_case_id_even_on_failure():
    cases = [
        {"id": "a", "category": "nested_arguments", "request": NESTED_REQUEST},
        {"id": "b", "category": "structured_final", "request": FINAL_REQUEST},
    ]
    client = _FakeClient(raises=MOD.TransportError("boom"))
    by_id = MOD.run_matrix(client, cases)
    assert set(by_id) == {"a", "b"}
    assert all(v["executed"] is False for v in by_id.values())


# --------------------------------------------------------- reference compare
def test_compare_case_agrees_when_tool_and_arguments_match():
    ref = MOD.judge_tool_response(NESTED_REQUEST,
                                  _tool_call_response("dispatch_job", {"mode": "fast"}))
    ref.update(id="c1", category="nested_arguments")
    var = dict(ref)
    out = MOD.compare_case(ref, var)
    assert out["comparable"] is True
    assert out["tool_selection_correct"] is True
    assert out["argument_exact_match"] is True
    assert out["issue_reason"] is None


def test_compare_case_flags_wrong_tool_and_wrong_arguments_separately():
    ref = MOD.judge_tool_response(NESTED_REQUEST,
                                  _tool_call_response("dispatch_job", {"mode": "fast"}))
    ref.update(id="c1", category="nested_arguments")
    wrong_args = MOD.judge_tool_response(NESTED_REQUEST,
                                         _tool_call_response("dispatch_job", {"mode": "safe"}))
    wrong_args.update(id="c1", category="nested_arguments")
    out = MOD.compare_case(ref, wrong_args)
    assert out["tool_selection_correct"] is True
    assert out["argument_exact_match"] is False

    wrong_tool = MOD.judge_tool_response(NESTED_REQUEST,
                                         _tool_call_response("other_tool", {"mode": "fast"}))
    wrong_tool.update(id="c1", category="nested_arguments")
    out2 = MOD.compare_case(ref, wrong_tool)
    assert out2["tool_selection_correct"] is False
    assert out2["argument_exact_match"] is False  # correctness gates argument agreement


def test_compare_case_not_comparable_when_reference_case_failed():
    ref = {"id": "c1", "category": "nested_arguments", "executed": False,
          "not_executed_reason": "[transport] timeout"}
    var = MOD.judge_tool_response(NESTED_REQUEST,
                                  _tool_call_response("dispatch_job", {"mode": "fast"}))
    var.update(id="c1", category="nested_arguments")
    out = MOD.compare_case(ref, var)
    assert out["executed"] is True
    assert out["comparable"] is False
    assert "reference case did not execute" in out["issue_reason"]
    # the variant's OWN schema validity is still an absolute measure, kept
    # even though no reference-relative comparison is possible
    assert out["schema_valid"] is True


def test_compare_case_records_variant_not_executed_with_a_reason():
    ref = MOD.judge_tool_response(NESTED_REQUEST,
                                  _tool_call_response("dispatch_job", {"mode": "fast"}))
    ref.update(id="c1", category="nested_arguments")
    var = {"id": "c1", "category": "nested_arguments", "executed": False,
          "not_executed_reason": "[transport] runner died"}
    out = MOD.compare_case(ref, var)
    assert out["executed"] is False
    assert out["issue_reason"] == "[transport] runner died"
    assert out["schema_valid"] is None


def test_compare_matrix_covers_every_declared_case_id():
    ref_by_id = {"a": {"id": "a", "category": "structured_final", "executed": False,
                       "not_executed_reason": "x"}}
    var_by_id = {}  # "a" missing entirely on the variant side
    out = MOD.compare_matrix(ref_by_id, var_by_id, ["a", "b"])
    ids = [c["id"] for c in out]
    assert ids == [None, None]  # var_judged is None for both -> id comes from var
    # both must be present as issues, never silently dropped
    assert all(c["executed"] is False for c in out)


# ---------------------------------------------------------------- aggregate
def test_aggregate_tool_fidelity_counts_and_rates():
    comparisons = [
        {"schema_valid": True, "tool_selection_correct": True, "argument_exact_match": True},
        {"schema_valid": False, "tool_selection_correct": True, "argument_exact_match": False},
        {"schema_valid": None, "tool_selection_correct": None, "argument_exact_match": None},
    ]
    agg = MOD.aggregate_tool_fidelity(comparisons)
    assert agg["requests"] == 3
    assert agg["schema_conformance"] == {"count": 1, "scored": 2, "rate": 0.5}
    assert agg["tool_selection"] == {"count": 2, "scored": 2, "rate": 1.0}
    assert agg["argument_agreement"] == {"count": 1, "scored": 2, "rate": 0.5}


def test_aggregate_tool_fidelity_handles_nothing_scored():
    agg = MOD.aggregate_tool_fidelity([{"schema_valid": None, "tool_selection_correct": None,
                                        "argument_exact_match": None}])
    assert agg["schema_conformance"] == {"count": 0, "scored": 0, "rate": None}


def test_not_executed_entries_never_silently_dropped():
    comparisons = [
        {"id": "a", "category": "x", "issue_reason": None},
        {"id": "b", "category": "y", "issue_reason": "[transport] timeout"},
        {"id": "c", "category": "z", "issue_reason": "reference case did not execute; no basis for comparison"},
    ]
    entries = MOD.not_executed_entries(comparisons)
    assert entries == [
        {"id": "b", "category": "y", "reason": "[transport] timeout"},
        {"id": "c", "category": "z",
        "reason": "reference case did not execute; no basis for comparison"},
    ]


# ------------------------------------------------------------- zero point
def _perfect_gold_comparisons():
    return [{"id": "a", "category": "nested_arguments", "executed": True,
            "comparable": True, "tool_selection_correct": True,
            "argument_exact_match": True, "schema_valid": True,
            "issue_reason": None}]


PERFECT_KLD = {"positions_scored": 10, "mean_kld": 0.0,
              "top1_agreement_pct": 100.0, "mean_top8_overlap": 1.0}


def test_zero_point_passes_on_perfect_agreement_and_zero_kld():
    verdict = MOD.zero_point_verdict(_perfect_gold_comparisons(), [], [], PERFECT_KLD)
    assert verdict["passed"] is True
    assert verdict["reasons"] == []
    assert verdict["mismatches"] == []


def test_zero_point_refuses_on_a_single_case_mismatch():
    comparisons = _perfect_gold_comparisons()
    comparisons[0]["argument_exact_match"] = False
    verdict = MOD.zero_point_verdict(comparisons, [], [], PERFECT_KLD)
    assert verdict["passed"] is False
    assert verdict["mismatches"] == [{"id": "a", "category": "nested_arguments"}]
    assert any("disagreed" in r for r in verdict["reasons"])


def test_zero_point_refuses_on_nonzero_kld():
    kld = dict(PERFECT_KLD, mean_kld=0.05)
    verdict = MOD.zero_point_verdict(_perfect_gold_comparisons(), [], [], kld)
    assert verdict["passed"] is False
    assert any("mean_kld" in r for r in verdict["reasons"])


def test_zero_point_refuses_on_imperfect_top1_agreement():
    kld = dict(PERFECT_KLD, top1_agreement_pct=99.0)
    verdict = MOD.zero_point_verdict(_perfect_gold_comparisons(), [], [], kld)
    assert verdict["passed"] is False
    assert any("top1_agreement_pct" in r for r in verdict["reasons"])


def test_zero_point_refuses_on_a_kld_position_that_failed_either_run():
    verdict = MOD.zero_point_verdict(
        _perfect_gold_comparisons(), [{"index": 3, "reason": "timeout"}], [], PERFECT_KLD)
    assert verdict["passed"] is False
    assert any("KLD position" in r for r in verdict["reasons"])


def test_zero_point_refuses_on_an_unexecuted_or_uncomparable_case():
    comparisons = _perfect_gold_comparisons()
    comparisons[0]["executed"] = False
    verdict = MOD.zero_point_verdict(comparisons, [], [], PERFECT_KLD)
    assert verdict["passed"] is False


# ----------------------------------------------------------------- report
def _measured_row(label, is_reference, schema_rate=1.0, tool_rate=1.0,
                  arg_rate=1.0, mean_kld=0.0, not_executed=0):
    return {
        "label": label, "path": f"/models/{label}.gguf", "is_reference": is_reference,
        "status": "measured",
        "tool_fidelity": {
            "requests": 4,
            "schema_conformance": {"count": 4, "scored": 4, "rate": schema_rate},
            "tool_selection": {"count": 4, "scored": 4, "rate": tool_rate},
            "argument_agreement": {"count": 4, "scored": 4, "rate": arg_rate},
        },
        "tool_fidelity_not_executed": [{"id": f"x{i}", "category": "c", "reason": "r"}
                                       for i in range(not_executed)],
        "kld_summary": {"positions_scored": 10, "mean_kld": mean_kld,
                        "top1_agreement_pct": 100.0, "mean_top8_overlap": 1.0},
        "kld_not_executed": [],
        "reused_zero_point_run": is_reference, "elapsed_ms": 123.4,
    }


def test_build_report_is_complete_when_zero_point_passes_and_every_variant_is_clean():
    zp = {"passed": True, "reasons": [], "mismatches": [], "kld_summary": PERFECT_KLD}
    rows = [_measured_row("q4_0", True)]
    report = MOD.build_report({"label": "q4_0", "path": "p"}, zp, rows, {"cases": 4})
    assert report["schema_version"] == "xyntetik.quant-fidelity.v1"
    assert report["complete"] is True
    assert report["zero_point"]["passed"] is True


def test_build_report_is_incomplete_if_zero_point_failed():
    zp = {"passed": False, "reasons": ["disagreement"], "mismatches": [], "kld_summary": PERFECT_KLD}
    report = MOD.build_report({"label": "q4_0", "path": "p"}, zp, [], {"cases": 4})
    assert report["complete"] is False


def test_build_report_is_incomplete_if_any_variant_has_not_executed_cases():
    zp = {"passed": True, "reasons": [], "mismatches": [], "kld_summary": PERFECT_KLD}
    rows = [_measured_row("q4_0", True, not_executed=1)]
    report = MOD.build_report({"label": "q4_0", "path": "p"}, zp, rows, {"cases": 4})
    assert report["complete"] is False


def test_build_report_is_incomplete_if_a_variant_did_not_execute_at_all():
    zp = {"passed": True, "reasons": [], "mismatches": [], "kld_summary": PERFECT_KLD}
    rows = [{"label": "q2_k", "path": "missing.gguf", "is_reference": False,
            "status": "not_executed", "status_reason": "model file not found",
            "tool_fidelity": None, "tool_fidelity_not_executed": [],
            "kld_summary": None, "kld_not_executed": [],
            "reused_zero_point_run": False, "elapsed_ms": None}]
    report = MOD.build_report({"label": "q4_0", "path": "p"}, zp, rows, {"cases": 4})
    assert report["complete"] is False


# ---------------------------------------------------------------- markdown
def test_render_markdown_reports_zero_point_and_variant_rows():
    zp = {"passed": True, "reasons": [], "mismatches": [], "kld_summary": PERFECT_KLD}
    rows = [_measured_row("q4_0", True), _measured_row("q2_k", False, schema_rate=0.75)]
    report = MOD.build_report({"label": "q4_0", "path": "p"}, zp, rows, {"cases": 4})
    md = MOD.render_markdown(report)
    assert "PASSED" in md
    assert "q4_0" in md and "q2_k" in md
    assert "75.0%" in md


def test_render_markdown_surfaces_a_not_executed_variant_never_hides_it():
    zp = {"passed": True, "reasons": [], "mismatches": [], "kld_summary": PERFECT_KLD}
    rows = [{"label": "q2_k", "path": "missing.gguf", "is_reference": False,
            "status": "not_executed", "status_reason": "model file not found: missing.gguf",
            "tool_fidelity": None, "tool_fidelity_not_executed": [],
            "kld_summary": None, "kld_not_executed": [],
            "reused_zero_point_run": False, "elapsed_ms": None}]
    report = MOD.build_report({"label": "q4_0", "path": "p"}, zp, rows, {"cases": 4})
    md = MOD.render_markdown(report)
    assert "q2_k" in md
    assert "not_executed" in md


def test_render_markdown_surfaces_zero_point_failure_reason():
    zp = {"passed": False, "reasons": ["mean_kld=0.3 exceeds epsilon=1e-06"],
         "mismatches": [], "kld_summary": PERFECT_KLD}
    report = MOD.build_report({"label": "q4_0", "path": "p"}, zp, [], {"cases": 4})
    md = MOD.render_markdown(report)
    assert "FAILED" in md
    assert "exceeds epsilon" in md


# --------------------------------------------------------------- idle wait
def test_other_runner_pids_parses_and_excludes_self():
    assert MOD.other_runner_pids("123\n456\n") == [123, 456]
    assert MOD.other_runner_pids("123\n456\n", exclude_pid=123) == [456]
    assert MOD.other_runner_pids("") == []


def test_wait_for_idle_runner_returns_immediately_when_clear():
    calls = []

    class _Proc:
        returncode = 1
        stdout = ""
        stderr = ""

    def fake_run(*a, **k):
        calls.append(a)
        return _Proc()

    def fake_sleep(_):
        raise AssertionError("should not sleep when already idle")

    MOD.wait_for_idle_runner(run=fake_run, sleep=fake_sleep)
    assert len(calls) == 1


def test_wait_for_idle_runner_polls_until_the_other_process_exits():
    responses = [
        type("P", (), {"returncode": 0, "stdout": "999\n", "stderr": ""})(),
        type("P", (), {"returncode": 1, "stdout": "", "stderr": ""})(),
    ]
    sleeps = []

    def fake_run(*a, **k):
        return responses.pop(0)

    MOD.wait_for_idle_runner(run=fake_run, sleep=sleeps.append, poll_seconds=15)
    assert sleeps == [15]


# -------------------------------------------------------------------- CLI
def test_parse_labeled_path_accepts_label_equals_path():
    parsed = MOD._parse_labeled_path("q4_0=models/m-Q4_0.gguf")
    assert parsed == {"label": "q4_0", "path": Path("models/m-Q4_0.gguf")}


def test_parse_labeled_path_rejects_a_bare_path():
    with pytest.raises(argparse.ArgumentTypeError):
        MOD._parse_labeled_path("models/m-Q4_0.gguf")


# --------------------------------------------------------------------- KLD
def test_compare_distributions_reuses_kld_raw_and_is_zero_for_identical_dists():
    dist = {"tok_a": -0.1, "tok_b": -2.0}
    positions = [{"index": 0, "dist": dist}]
    summary = MOD.compare_distributions(positions, positions)
    assert summary["positions_scored"] == 1
    assert summary["mean_kld"] == 0.0
    assert summary["top1_agreement_pct"] == 100.0
    assert summary["mean_top8_overlap"] == 1.0


def test_compare_distributions_is_nonzero_for_disagreeing_dists():
    ref = [{"index": 0, "dist": {"tok_a": -0.1, "tok_b": -5.0}}]
    var = [{"index": 0, "dist": {"tok_a": -5.0, "tok_b": -0.1}}]
    summary = MOD.compare_distributions(ref, var)
    assert summary["mean_kld"] > 0
    assert summary["top1_agreement_pct"] == 0.0


def test_collect_distributions_records_position_failures_without_dropping_them(monkeypatch):
    calls = {"n": 0}

    def fake_query(endpoint, model_name, prompt, top_n=20):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated timeout")
        return {"tok": -0.1}

    monkeypatch.setattr(MOD.KLD_RAW, "query", fake_query)
    positions, failures = MOD.collect_distributions(
        "http://127.0.0.1:1", "m", ["a", "b", "c", "d"], max_positions=3, stride=1)
    assert len(positions) == 3  # kept retrying past the one failure
    assert failures == [{"index": 1, "reason": "simulated timeout"}]
