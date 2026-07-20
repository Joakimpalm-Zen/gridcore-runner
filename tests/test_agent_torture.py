import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "agent_torture", ROOT / "scripts" / "agent-torture.py")
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def test_default_matrix_is_repeatable_and_covers_100_cases():
    first = MOD.build_cases(100)
    second = MOD.build_cases(100)
    assert first == second
    assert len(first) == 100
    assert len({case["id"] for case in first}) == 100
    assert {case["category"] for case in first} == {
        "nested_arguments", "tool_selection", "forced_truncation",
        "stream_normalization",
    }
    assert all(sum(c["category"] == category for c in first) == 25
               for category in {c["category"] for c in first})


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
    report = MOD.make_report(results, "runner test", "fixture.gguf", 123, 456)
    path = tmp_path / "report.json"
    MOD.write_json(path, report)
    decoded = json.loads(path.read_text())

    assert decoded["schema_version"] == "gridcore.agent-torture.v1"
    assert decoded["runtime"] == {"name": "runner", "version": "runner test"}
    assert decoded["configuration"]["model"] == "fixture.gguf"
    assert decoded["totals"] == {
        "requests": 4, "passed": 2, "failed": 2,
        "failures_by_category": {"protocol": 1, "schema": 1},
    }
    assert decoded["metrics"]["valid_structured_tasks_per_second"] == 16.26
    assert decoded["resources"]["peak_rss_kb"] == 456
    assert [c["id"] for c in decoded["cases"]] == [c["id"] for c in cases]


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
