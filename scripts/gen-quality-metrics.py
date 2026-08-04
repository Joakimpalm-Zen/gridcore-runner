#!/usr/bin/env python3
"""Generation quality metrics for the certification gate.

Reads either a torture raw.jsonl or plain text file of model outputs.
Reports per-sample and aggregate: max n-gram repetition run (n=1..4),
blank/whitespace fraction, output-length distribution, and a collapse flag
(repetition/blank above threshold).

Pure Python, stdlib only, JSON output.
"""

import argparse
import base64
import collections
import json
import sys
from pathlib import Path
from typing import Any


def extract_text_from_torture(line: str) -> str | None:
    """Extract the decoded text from a torture raw.jsonl line."""
    try:
        obj = json.loads(line)
        response = obj.get("response", {})
        body_b64 = response.get("body", "")
        if response.get("encoding") == "base64":
            decoded = base64.b64decode(body_b64).decode("utf-8")
            resp_obj = json.loads(decoded)
            choices = resp_obj.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                return message.get("content", "")
        elif "normalized" in obj:
            normalized = obj.get("normalized", {})
            return normalized.get("text", "")
    except Exception:
        pass
    return None


def max_ngram_repetition(text: str) -> int:
    """Compute the longest run of repeated n-grams for n in 1..4."""
    if not text:
        return 0

    max_run = 0
    words = text.split()

    for n in range(1, 5):
        if len(words) < n:
            continue
        i = 0
        while i < len(words) - n:
            ngram = tuple(words[i : i + n])
            run = 1
            j = i + n
            while j <= len(words) - n:
                if tuple(words[j : j + n]) == ngram:
                    run += 1
                    j += n
                else:
                    break
            max_run = max(max_run, run)
            i += 1

    return max_run


def blank_line_fraction(text: str) -> float:
    """Compute the fraction of blank/whitespace-only lines."""
    lines = text.split("\n")
    if not lines:
        return 0.0
    blank_count = sum(1 for line in lines if not line.strip())
    return blank_count / len(lines)


def analyze_generation(text: str) -> dict[str, Any]:
    """Analyze a single generation."""
    return {
        "length": len(text),
        "word_count": len(text.split()),
        "max_ngram_repetition": max_ngram_repetition(text),
        "blank_line_fraction": blank_line_fraction(text),
    }


def metrics_aggregate(analyses: list[dict[str, Any]],
                      rep_threshold: int = 3,
                      blank_threshold: float = 0.2) -> dict[str, Any]:
    """Compute aggregate metrics and collapse flags."""
    if not analyses:
        return {
            "count": 0,
            "mean_length": 0,
            "mean_word_count": 0,
            "mean_max_ngram_repetition": 0,
            "mean_blank_line_fraction": 0,
            "collapse_count": 0,
            "collapse_rate": 0,
        }

    lengths = [a["length"] for a in analyses]
    word_counts = [a["word_count"] for a in analyses]
    reps = [a["max_ngram_repetition"] for a in analyses]
    blanks = [a["blank_line_fraction"] for a in analyses]

    collapse_count = sum(
        1 for a in analyses
        if a["max_ngram_repetition"] > rep_threshold
        or a["blank_line_fraction"] > blank_threshold
    )

    return {
        "count": len(analyses),
        "mean_length": sum(lengths) / len(lengths),
        "mean_word_count": sum(word_counts) / len(word_counts),
        "mean_max_ngram_repetition": sum(reps) / len(reps),
        "mean_blank_line_fraction": sum(blanks) / len(blanks),
        "collapse_count": collapse_count,
        "collapse_rate": collapse_count / len(analyses),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="raw.jsonl torture file or plain text file")
    parser.add_argument("--format", choices=["torture", "text", "auto"],
                        default="auto",
                        help="Input format (auto-detect by default)")
    parser.add_argument("--rep-threshold", type=int, default=3,
                        help="N-gram repetition run threshold for collapse flag")
    parser.add_argument("--blank-threshold", type=float, default=0.2,
                        help="Blank line fraction threshold for collapse flag")
    parser.add_argument("--output", help="Output JSON file (default: stdout)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"error: {args.input} not found", file=sys.stderr)
        return 1

    fmt = args.format
    if fmt == "auto":
        fmt = "torture" if input_path.suffix == ".jsonl" else "text"

    analyses = []

    if fmt == "torture":
        with open(input_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                text = extract_text_from_torture(line)
                if text:
                    analysis = analyze_generation(text)
                    analyses.append(analysis)
    else:
        with open(input_path) as f:
            text = f.read()
        analysis = analyze_generation(text)
        analyses.append(analysis)

    results = {
        "per_sample": analyses,
        "aggregate": metrics_aggregate(analyses, args.rep_threshold,
                                       args.blank_threshold),
    }

    output = json.dumps(results, indent=2)
    if args.output:
        Path(args.output).write_text(output)
        print(f"ok: wrote {len(analyses)} samples to {args.output}")
    else:
        print(output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
