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
