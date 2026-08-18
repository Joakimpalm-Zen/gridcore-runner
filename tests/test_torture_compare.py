import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "torture_compare", ROOT / "scripts" / "torture-compare.py")
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def _report(name, version, cases, model="m", elapsed=1000.0, tps=1.0):
    return {"runtime": {"name": name, "version": version},
            "configuration": {"model": model},
            "totals": {"requests": len(cases),
                       "passed": sum(c["status"] == "passed" for c in cases)},
            "metrics": {"elapsed_ms": elapsed,
                        "valid_structured_tasks_per_second": tps},
            "cases": cases}


def _cases(**by_cat):
    out = []
    for cat, (passed, total) in by_cat.items():
        for i in range(total):
            out.append({"category": cat,
                        "status": "passed" if i < passed else "failed"})
    return out


def test_category_counts_and_ordering(tmp_path):
    strong = tmp_path / "a.json"
    weak = tmp_path / "b.json"
    strong.write_text(json.dumps(_report(
        "runner", "runner 0.1", _cases(nested_arguments=(3, 3),
                                       forced_truncation=(3, 3)))))
    weak.write_text(json.dumps(_report(
        "llama.cpp", "b10076", _cases(nested_arguments=(0, 3),
                                      forced_truncation=(0, 3)))))
    reports = MOD.load([str(weak), str(strong)])  # deliberately weak-first
    rows = MOD.build_rows(reports)
    # the stronger runtime is ordered first regardless of input order
    assert rows[0]["label"] == "runner 0.1"       # version-contains-name dedup
    assert rows[1]["label"] == "llama.cpp b10076"


def test_render_table_has_totals_and_no_doubled_name(tmp_path):
    p = tmp_path / "r.json"
    p.write_text(json.dumps(_report(
        "runner", "runner 0.1.1-alpha", _cases(tool_selection=(2, 3)))))
    text = MOD.render(MOD.load([str(p)]), markdown=True)
    assert "| runner 0.1.1-alpha |" in text     # not "runner runner ..."
    assert "valid structured calls" in text
    assert "2/3" in text


def test_ordering_is_by_pass_rate_not_by_absolute_pass_count(tmp_path):
    """build_rows' contract is "best total pass rate first", and the ordering
    it produces is the published LEADERBOARD.md. Sorting on the raw `passed`
    count ranks a runtime that failed three quarters of a larger matrix above
    one that passed all of a smaller one, so any comparison run with unequal
    --cases came out backwards."""
    sloppy = tmp_path / "sloppy.json"
    good = tmp_path / "good.json"
    sloppy.write_text(json.dumps(_report(
        "llama.cpp", "b10076", _cases(nested_arguments=(50, 200)))))
    good.write_text(json.dumps(_report(
        "runner", "runner 0.1", _cases(nested_arguments=(40, 40)))))

    rows = MOD.build_rows(MOD.load([str(sloppy), str(good)]))

    assert [r["label"] for r in rows] == ["runner 0.1", "llama.cpp b10076"]


def test_an_equal_rate_prefers_the_larger_matrix(tmp_path):
    small = tmp_path / "small.json"
    large = tmp_path / "large.json"
    small.write_text(json.dumps(_report(
        "llama.cpp", "b10076", _cases(nested_arguments=(4, 4)))))
    large.write_text(json.dumps(_report(
        "runner", "runner 0.1", _cases(nested_arguments=(40, 40)))))

    rows = MOD.build_rows(MOD.load([str(small), str(large)]))

    assert rows[0]["label"] == "runner 0.1"


def test_reports_on_different_models_are_refused_rather_than_merged(tmp_path):
    """render() takes the model name off whichever report came first on argv
    and prints it as THE model of the table. A leaderboard is a
    same-model-same-hardware comparison; one built across models is a table of
    incomparable rows under a name that is wrong for all but one of them."""
    import pytest

    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(json.dumps(_report("runner", "runner 0.1",
                                    _cases(tool_selection=(3, 3)),
                                    model="qwen2.5-7b.gguf")))
    b.write_text(json.dumps(_report("ollama", "0.32.1",
                                    _cases(tool_selection=(3, 3)),
                                    model="gptoss-20b.gguf")))

    with pytest.raises(SystemExit) as caught:
        MOD.main([str(a), str(b)])

    assert caught.value.code == 2


def test_reports_on_the_same_model_still_render(tmp_path, capsys):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    for path, name in ((a, "runner"), (b, "ollama")):
        path.write_text(json.dumps(_report(name, "1.0",
                                           _cases(tool_selection=(3, 3)),
                                           model="qwen2.5-7b.gguf")))

    assert MOD.main([str(a), str(b), "--md"]) == 0
    assert "qwen2.5-7b.gguf" in capsys.readouterr().out
