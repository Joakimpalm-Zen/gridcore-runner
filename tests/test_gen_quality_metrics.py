#!/usr/bin/env python3
"""Unit tests for gen-quality-metrics.py"""

import json
import sys
import tempfile
from pathlib import Path
import importlib.util

script_path = Path(__file__).resolve().parent.parent / "scripts" / "gen-quality-metrics.py"
spec = importlib.util.spec_from_file_location("gen_quality_metrics", script_path)
gen_quality_metrics = importlib.util.module_from_spec(spec)
sys.modules["gen_quality_metrics"] = gen_quality_metrics
spec.loader.exec_module(gen_quality_metrics)

from gen_quality_metrics import (
    analyze_generation, max_ngram_repetition, blank_line_fraction,
    metrics_aggregate, extract_text_from_torture
)


def test_clean_text():
    """Test a clean, non-repetitive text."""
    text = "The quick brown fox jumps over the lazy dog."
    analysis = analyze_generation(text)
    assert analysis["length"] > 0
    assert analysis["max_ngram_repetition"] == 1  # no repeated n-grams
    assert analysis["blank_line_fraction"] == 0  # no blank lines
    assert analysis["word_count"] == 9


def test_looping_text():
    """Test text with significant n-gram repetition."""
    text = "hello world hello world hello world"
    analysis = analyze_generation(text)
    assert analysis["max_ngram_repetition"] >= 3  # 3+ repeats of "hello world"
    # No blank lines in this text
    assert analysis["blank_line_fraction"] == 0


def test_blank_collapse_text():
    """Test text with many blank lines."""
    text = "line one\n\n\n\nline two\n\n\n\n"
    analysis = analyze_generation(text)
    assert analysis["blank_line_fraction"] > 0.5  # more than half are blank


def test_empty_text():
    """Test edge case: empty text."""
    analysis = analyze_generation("")
    assert analysis["length"] == 0
    assert analysis["max_ngram_repetition"] == 0
    # Empty string has one empty line, so blank_line_fraction is 1.0
    assert analysis["blank_line_fraction"] == 1.0


def test_single_word_repetition():
    """Test single-word repetition (1-gram)."""
    text = "test test test test test"
    analysis = analyze_generation(text)
    assert analysis["max_ngram_repetition"] >= 5


def test_aggregate_metrics():
    """Test aggregate computation."""
    analyses = [
        {"length": 10, "word_count": 2, "max_ngram_repetition": 1, "blank_line_fraction": 0},
        {"length": 20, "word_count": 4, "max_ngram_repetition": 2, "blank_line_fraction": 0.1},
    ]
    agg = metrics_aggregate(analyses, rep_threshold=3, blank_threshold=0.2)
    assert agg["count"] == 2
    assert agg["mean_length"] == 15
    assert agg["mean_word_count"] == 3
    assert agg["collapse_count"] == 0  # none exceed thresholds


def test_aggregate_collapse_flag():
    """Test collapse flag in aggregate."""
    analyses = [
        {"length": 50, "word_count": 10, "max_ngram_repetition": 5,
         "blank_line_fraction": 0.1},  # exceeds rep threshold
        {"length": 50, "word_count": 10, "max_ngram_repetition": 2,
         "blank_line_fraction": 0.3},  # exceeds blank threshold
        {"length": 50, "word_count": 10, "max_ngram_repetition": 2,
         "blank_line_fraction": 0.1},  # OK
    ]
    agg = metrics_aggregate(analyses, rep_threshold=3, blank_threshold=0.2)
    assert agg["collapse_count"] == 2
    assert agg["collapse_rate"] == 2 / 3


def test_torture_extraction():
    """Test extraction from torture raw.jsonl format."""
    # Minimal torture record
    torture_json = {
        "id": "test-000",
        "response": {
            "encoding": "base64",
            "body": json.dumps({
                "choices": [{
                    "message": {
                        "content": "test output"
                    }
                }]
            }).encode().hex()  # This will fail, use correct base64
        }
    }
    # Correct version: properly base64-encode the response body
    import base64
    response_body = json.dumps({
        "choices": [{
            "message": {
                "content": "extracted text"
            }
        }]
    })
    torture_json["response"]["body"] = base64.b64encode(
        response_body.encode()
    ).decode()
    line = json.dumps(torture_json)
    text = extract_text_from_torture(line)
    assert text == "extracted text"


def test_script_with_files():
    """Integration test: run the script with files."""
    import subprocess

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create a plain text file
        text_file = tmpdir / "test.txt"
        text_file.write_text("hello world hello world hello world\nline 2\n\n\n")

        # Run the script
        result = subprocess.run(
            [sys.executable, "scripts/gen-quality-metrics.py",
             str(text_file), "--output", str(tmpdir / "out.json")],
            cwd=Path(__file__).parent.parent,
            capture_output=True, text=True
        )

        assert result.returncode == 0, result.stderr

        output_file = tmpdir / "out.json"
        assert output_file.exists(), result.stderr
        output = json.loads(output_file.read_text())
        assert "per_sample" in output
        assert "aggregate" in output
        assert len(output["per_sample"]) == 1
        assert output["aggregate"]["count"] == 1


def main():
    """Run all tests."""
    tests = [
        ("clean_text", test_clean_text),
        ("looping_text", test_looping_text),
        ("blank_collapse_text", test_blank_collapse_text),
        ("empty_text", test_empty_text),
        ("single_word_repetition", test_single_word_repetition),
        ("aggregate_metrics", test_aggregate_metrics),
        ("aggregate_collapse_flag", test_aggregate_collapse_flag),
        ("torture_extraction", test_torture_extraction),
        ("script_with_files", test_script_with_files),
    ]

    failed = []
    for name, test in tests:
        try:
            test()
            print(f"ok: {name}")
        except AssertionError as e:
            print(f"FAIL: {name}: {e}", file=sys.stderr)
            failed.append(name)
        except Exception as e:
            print(f"ERROR: {name}: {e}", file=sys.stderr)
            failed.append(name)

    if failed:
        print(f"\n{len(failed)} test(s) failed", file=sys.stderr)
        return 1
    print(f"\nall {len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
