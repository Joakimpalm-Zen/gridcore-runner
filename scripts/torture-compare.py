#!/usr/bin/env python3
"""Summarize agent-torture reports into a leaderboard.

Reads N `report.json` files (from `agent-torture.py`) and prints one
comparison table — per-category pass rate, total valid calls, valid
structured tasks/second, and elapsed — so publishing or reviewing a
cross-runtime run is one command, not manual diffing.

  torture-compare.py results/*/report.json
  torture-compare.py --md results/2026-07-21-*/*/report.json > LEADERBOARD.md

Columns are ordered by total pass rate, best first. The report's own
`runtime.name` + `runtime.version` label each column, so which runtime
produced which result is never ambiguous.
"""
import argparse
import json
from pathlib import Path

CATEGORIES = ("nested_arguments", "tool_selection", "forced_truncation",
              "stream_normalization")
CATEGORY_LABEL = {
    "nested_arguments": "nested tool arguments",
    "tool_selection": "tool selection",
    "forced_truncation": "truncation mid-call",
    "stream_normalization": "stream normalization",
}


def load(paths):
    reports = []
    for p in paths:
        data = json.loads(Path(p).read_text())
        rt = data.get("runtime", {})
        name = rt.get("name", "?")
        version = rt.get("version", "")
        if version and version not in ("unknown",):
            # some runtimes print their name inside --version ("runner 0.1.1")
            label = version if version.lower().startswith(name.lower()) \
                else f"{name} {version}"
        else:
            label = name
        reports.append({"label": label, "data": data, "path": str(p)})
    return reports


def category_counts(data):
    """Return {category: (passed, total)} from a report's cases."""
    out = {}
    for case in data.get("cases", []):
        c = case.get("category")
        passed, total = out.get(c, (0, 0))
        out[c] = (passed + (case.get("status") == "passed"), total + 1)
    return out


def build_rows(reports):
    """Rows of the table, best total pass rate first."""
    def rate(r):
        t = r["data"].get("totals", {})
        return (t.get("passed", 0), -t.get("requests", 0))
    return sorted(reports, key=rate, reverse=True)


def render(reports, markdown):
    rows = build_rows(reports)
    labels = [r["label"] for r in rows]
    counts = [category_counts(r["data"]) for r in rows]

    def cell(counts_i, cat):
        p, t = counts_i.get(cat, (0, 0))
        return f"{p}/{t}" if t else "-"

    header = ["category", *labels]
    lines = []
    if markdown:
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")
    else:
        width = max(len("stream normalization"), *(len(x) for x in labels))
        lines.append("  ".join(h.ljust(width) for h in header))

    def row(label, cells):
        if markdown:
            return "| " + " | ".join([label, *cells]) + " |"
        width = max(len("stream normalization"), *(len(x) for x in labels))
        return "  ".join(x.ljust(width) for x in (label, *cells))

    for cat in CATEGORIES:
        lines.append(row(CATEGORY_LABEL[cat], [cell(c, cat) for c in counts]))

    # totals + throughput
    def total_cell(r):
        t = r["data"].get("totals", {})
        return f"{t.get('passed', 0)}/{t.get('requests', 0)}"
    lines.append(row("valid structured calls" if markdown else "TOTAL",
                     [total_cell(r) for r in rows]))

    def tps_cell(r):
        m = r["data"].get("metrics", {})
        v = m.get("valid_structured_tasks_per_second")
        return f"{v:g}" if isinstance(v, (int, float)) else "-"
    lines.append(row("valid tasks/s", [tps_cell(r) for r in rows]))

    def elapsed_cell(r):
        m = r["data"].get("metrics", {})
        ms = m.get("elapsed_ms")
        return f"{ms / 1000:.1f}s" if isinstance(ms, (int, float)) else "-"
    lines.append(row("elapsed", [elapsed_cell(r) for r in rows]))

    model = reports[0]["data"].get("configuration", {}).get("model", "?")
    out = []
    if markdown:
        out.append(f"# Agent torture leaderboard — {model}\n")
    else:
        out.append(f"agent torture — model: {model}")
    out.extend(lines)
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reports", nargs="+", help="report.json files to compare")
    ap.add_argument("--md", action="store_true", help="emit a Markdown table")
    args = ap.parse_args(argv)
    reports = load(args.reports)
    if not reports:
        ap.error("no reports")
    print(render(reports, args.md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
