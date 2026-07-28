import json
import pathlib
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_llamacpp_comparison_fixture_report(tmp_path):
    proc = subprocess.run(
        [
            sys.executable,
            ROOT / "scripts/compare_llamacpp.py",
            "--fixture",
            "--out-dir",
            tmp_path,
            "--prompt",
            "fixture prompt",
            "--ctx",
            "128",
            "--tokens",
            "4",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    summary = json.loads(proc.stdout)
    assert summary["status"] == "fixture"

    report = json.loads((tmp_path / "comparison.json").read_text())
    assert report["schema_version"] == "gridcore.runner.llamacpp-comparison.v1"
    assert report["status"] == "fixture"
    assert report["real_results"] == "pending"
    assert report["settings"]["prompt"] == "fixture prompt"

    md = (tmp_path / "comparison.md").read_text()
    assert "Runner vs llama.cpp comparison" in md
    assert "Real Qwen3/MoE GPU results are pending" in md
