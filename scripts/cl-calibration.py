#!/usr/bin/env python3
"""Calibration report for choice_logprobs decision records (JC-R1).

The runner's `choice_logprobs` response field reports, at every constrained
decision point, a posterior over the legal alternatives. This script answers
the question that decides whether that posterior is USABLE as a confidence
signal: when the model says 0.8, is it right 80% of the time?

Input: JSONL, one labeled decision per line:

    {"alternatives": [{"id": 314, "prob": 0.61, ...}, ...],
     "correct_id": 314,
     "coverage": 0.97}            # optional passthroughs are ignored

`alternatives` is a decision record's array exactly as the server emitted it
(descending prob, renormalized over the legal probed set); `correct_id` is
the ground-truth token id for that decision. A record whose correct id is
not among the stored alternatives counts as a wrong prediction at the
model's stated confidence (that is the honest reading: the model put the
truth outside its top-8 legal set).

Output: top-1 accuracy, Brier score, ECE (10 equal-width bins), and a
reliability table. Exit code 1 if ECE exceeds --max-ece (default: off), so
the script can gate.

No third-party imports — usable anywhere the runner builds.
"""

import argparse
import json
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("jsonl", help="labeled decisions, one JSON object per line"
                                  " ('-' for stdin)")
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--max-ece", type=float, default=None,
                    help="exit 1 if ECE exceeds this")
    args = ap.parse_args()

    fh = sys.stdin if args.jsonl == "-" else open(args.jsonl)
    decisions = []   # (confidence, predicted_correct, brier_term)
    skipped = 0
    for ln, line in enumerate(fh, 1):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            alts = rec["alternatives"]
            correct = rec["correct_id"]
        except (ValueError, KeyError) as e:
            print(f"line {ln}: skipped ({e})", file=sys.stderr)
            skipped += 1
            continue
        if not alts:
            skipped += 1
            continue
        conf = alts[0]["prob"]
        hit = alts[0]["id"] == correct
        # Brier over the full stored distribution: sum (p_i - y_i)^2, with
        # any truth outside the stored alternatives contributing y=1, p=0.
        brier = sum((a["prob"] - (1.0 if a["id"] == correct else 0.0)) ** 2
                    for a in alts)
        if not any(a["id"] == correct for a in alts):
            brier += 1.0
        decisions.append((conf, hit, brier))
    if fh is not sys.stdin:
        fh.close()

    n = len(decisions)
    if n == 0:
        print("no usable decisions", file=sys.stderr)
        return 2

    acc = sum(h for _, h, _ in decisions) / n
    brier = sum(b for _, _, b in decisions) / n

    bins = [[0, 0, 0.0] for _ in range(args.bins)]   # count, hits, conf mass
    for conf, hit, _ in decisions:
        b = min(int(conf * args.bins), args.bins - 1)
        bins[b][0] += 1
        bins[b][1] += hit
        bins[b][2] += conf

    ece = sum(c * abs(h / c - m / c) for c, h, m in bins if c) / n

    print(f"decisions {n}  skipped {skipped}")
    print(f"top-1 accuracy {acc:.4f}   Brier {brier:.4f}   "
          f"ECE({args.bins} bins) {ece:.4f}")
    print(f"{'bin':>11}  {'n':>6}  {'confidence':>10}  {'accuracy':>8}  gap")
    for i, (c, h, m) in enumerate(bins):
        lo, hi = i / args.bins, (i + 1) / args.bins
        if not c:
            print(f"{lo:.2f}-{hi:.2f}  {0:>6}  {'-':>10}  {'-':>8}  -")
            continue
        print(f"{lo:.2f}-{hi:.2f}  {c:>6}  {m / c:>10.4f}  {h / c:>8.4f}  "
              f"{h / c - m / c:+.4f}")

    if args.max_ece is not None and ece > args.max_ece:
        print(f"FAIL: ECE {ece:.4f} > {args.max_ece}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
