import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "agent_torture", ROOT / "scripts" / "agent-torture.py")
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def test_default_matrix_is_repeatable_and_balanced():
    """105, not 100: seven families divide evenly into it and the published bar
    is a >=100-request matrix. v1's 100/5 results stay valid on their own terms
    but are not case-for-case comparable with v2 — hence the SCHEMA_VERSION
    bump, which is what tells a reader which matrix a result file describes."""
    first = MOD.build_cases()
    second = MOD.build_cases()
    assert first == second
    assert len(first) == 105
    assert len({case["id"] for case in first}) == 105
    assert {case["category"] for case in first} == {
        "nested_arguments", "tool_selection", "forced_truncation",
        "stream_normalization", "large_enum_selection",
        "reasoning_then_tool", "structured_final",
    }
    assert all(sum(c["category"] == category for c in first) == 15
               for category in {c["category"] for c in first})


def test_reasoning_then_tool_replays_prose_before_demanding_a_call():
    """The family exists to catch content bleeding into a call turn, so the
    request must actually carry an earlier assistant turn — a version that
    forgot it would still pass a 'one tool call' check while testing nothing."""
    cases = [c for c in MOD.build_cases() if c["category"] == "reasoning_then_tool"]
    assert cases
    messages = cases[0]["request"]["messages"]
    roles = [m["role"] for m in messages]
    assert roles == ["user", "assistant", "user"], roles
    assert len(messages[1]["content"]) > 40, "the replayed reasoning is trivial"
    assert cases[0]["request"]["tool_choice"]["function"]["name"] == "record_conclusion"


def test_structured_final_constrains_a_final_answer_not_a_call():
    """The other new family goes through response_format, which reaches the
    sampler by a different path than tools do."""
    cases = [c for c in MOD.build_cases() if c["category"] == "structured_final"]
    assert cases
    request = cases[0]["request"]
    assert "tools" not in request, "a structured final must not offer tools"
    schema = request["response_format"]["json_schema"]["schema"]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["owner"]["required"] == ["team", "oncall"]
    # a valid document passes and a near-miss fails, so the verifier is real
    MOD.validate_against_schema(
        {"summary": "s", "severity": "high",
         "owner": {"team": "core", "oncall": True}}, MOD.FINAL_SCHEMA)
    try:
        MOD.validate_against_schema(
            {"summary": "s", "severity": "critical",
             "owner": {"team": "core", "oncall": True}}, MOD.FINAL_SCHEMA)
    except Exception:
        pass
    else:
        raise AssertionError("severity outside the enum was accepted")


def test_large_enum_case_constrains_to_an_exact_taxonomy_member():
    cases = [c for c in MOD.build_cases(100)
             if c["category"] == "large_enum_selection"]
    assert cases, "the large-enum family must be present in the matrix"
    case = cases[0]
    tool = case["request"]["tools"][0]["function"]
    assert tool["name"] == "classify_ticket"
    enum = tool["parameters"]["properties"]["label"]["enum"]
    assert len(enum) >= 50 and len(set(enum)) == len(enum)
    # a valid member passes, a plausible near-miss fails — the exact behaviour
    # the schema-constrained decoder must enforce on a small model
    MOD.validate_against_schema({"label": enum[0]}, MOD.CLASSIFY_SCHEMA)
    try:
        MOD.validate_against_schema({"label": "billing_issue"}, MOD.CLASSIFY_SCHEMA)
        assert False, "a non-member label must be rejected"
    except Exception:
        pass


def test_report_schema_and_totals(tmp_path):
    cases = MOD.build_cases(4)
    results = [
        MOD.result_for(cases[0], "passed", 1.25),
        MOD.result_for(cases[1], "failed", 2.5,
                       failure={"category": "schema", "message": "bad"}),
        MOD.result_for(cases[2], "passed", 3.0),
        MOD.result_for(cases[3], "failed", 4.0,
                       failure={"category": "protocol", "message": "bad"}),
    ]
    report = MOD.make_report(results, "runner", "runner test", "fixture.gguf",
                             123, 456)
    path = tmp_path / "report.json"
    MOD.write_json(path, report)
    decoded = json.loads(path.read_text())

    assert decoded["schema_version"] == "gridcore.agent-torture.v2"
    assert decoded["runtime"] == {"name": "runner", "version": "runner test"}
    assert decoded["configuration"]["model"] == "fixture.gguf"
    assert decoded["totals"] == {
        "requests": 4, "passed": 2, "failed": 2,
        "failures_by_category": {"protocol": 1, "schema": 1},
    }
    assert decoded["metrics"]["valid_structured_tasks_per_second"] == 16.26
    assert decoded["resources"]["peak_rss_kb"] == 456
    assert [c["id"] for c in decoded["cases"]] == [c["id"] for c in cases]


def test_report_is_labeled_with_the_runtime_under_test():
    # the whole point of cross-runtime: a report names which runtime produced
    # it, so llama.cpp / ollama / vllm results compare directly
    report = MOD.make_report([], "llama.cpp", "b3200", "qwen2.5-7b", 1000, None)
    assert report["runtime"] == {"name": "llama.cpp", "version": "b3200"}
    assert report["resources"]["peak_rss_kb"] is None  # foreign process


def test_endpoint_parsing_accepts_host_port_and_urls_rejects_remote():
    assert MOD._parse_endpoint("127.0.0.1:8080") == 8080
    assert MOD._parse_endpoint("http://localhost:11434/") == 11434
    assert MOD._parse_endpoint("localhost:8000") == 8000
    for bad in ("127.0.0.1", "10.0.0.5:8080", "example.com:8080"):
        try:
            MOD._parse_endpoint(bad)
            assert False, f"{bad} should be rejected"
        except ValueError:
            pass


def test_remote_target_exposes_the_client_contract():
    # the harness Client drives a target through .port / assert_alive /
    # sample_rss — a RemoteTarget provides exactly that, so the same matrix
    # runs against any local OpenAI-compatible server
    target = MOD.RemoteTarget(65535)
    assert target.port == 65535
    assert target.peak_rss_kb is None
    assert target.sample_rss() is None


def test_stream_normalization_is_independent_of_tcp_chunks():
    raw = (b'data: {"choices":[{"delta":{"content":"a"}}]}\n\n'
           b'data: {"choices":[{"delta":{"content":"b"},'
           b'"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n')
    reference = MOD.normalize_sse(raw, [raw])
    assert reference["text"] == "ab"
    assert reference["finish_reason"] == "stop"
    assert reference["saw_done"] is True
    for point in range(len(raw) + 1):
        assert MOD.normalize_sse(raw, [raw[:point], raw[point:]]) == reference
