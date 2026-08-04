#!/usr/bin/env python3
"""Generate a --prune-experts plan (and a hot/cold precision plan) from
RUNNER_MOE_TRACE JSONL files (src/model.c's moe_trace_emit, {"pos","layer",
"experts","gates","norms"} — "norms" is optional, only present since the
activation-norm trace addition).

Saliency per (layer, expert), summed over every selection in the trace(s):
  mass        sum of gate weights ("gates") — pure popularity/confidence,
              needs no "norms" field. This is the ranking trips 1-3's own
              "standing committee" analysis already used informally.
  gate*norm   sum of gate weight * activation-output L2 norm ("norms") —
              REAP-style saliency: how much each pick actually MOVED the
              residual stream, not just how often/confidently it was
              picked. Needs a trace collected after the norms addition.

Two plan modes (exactly one required):
  --keep-n N     every MoE layer keeps its OWN top-N experts by saliency —
                 GRAPE's uniform-per-layer pruning depth (same COUNT
                 everywhere, different specific ids per layer).
  --coverage T   every MoE layer keeps the smallest top-k covering
                 fraction T of ITS OWN total saliency mass — genuinely
                 non-uniform per-layer allocation: a layer whose mass
                 concentrates in one expert needs fewer kept to hit the
                 same threshold than one with a flat distribution.

Either mode writes a --prune-experts-ready JSON to --out and, doing the
coverage arithmetic either way, prints per-layer and overall realized
coverage to stderr so a --keep-n run can be sanity-checked against a
target coverage number without switching modes.

--precision-out additionally writes a hot/cold split (globally, pooled
across every layer) for a later selective-precision quantize pass: the
top --hot-fraction of (layer, expert) pairs by saliency are "hot" (stay at
higher precision), the rest "cold" (candidates for a lower-bit tier).
"""
import argparse
import json
import sys
from collections import defaultdict


def load_saliency(paths, use_norms):
    """-> {layer: {expert: score}}"""
    mass = defaultdict(lambda: defaultdict(float))
    for path in paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                layer = rec["layer"]
                experts = rec["experts"]
                gates = rec["gates"]
                norms = rec.get("norms") if use_norms else None
                for i, e in enumerate(experts):
                    w = gates[i]
                    if use_norms:
                        if not norms or i >= len(norms) or norms[i] is None:
                            continue
                        w = w * norms[i]
                    mass[layer][e] += w
    return mass


def _ranked(scores):
    # sorted() is stable: equal-score experts keep the order they were
    # first seen in the trace, so a synthetic test's tie-breaking is
    # reproducible rather than dict-hash-order-dependent.
    return sorted(scores.items(), key=lambda kv: -kv[1])


def plan_keep_n(mass, n):
    plan, coverage = {}, {}
    for layer, scores in mass.items():
        ranked = _ranked(scores)
        total = sum(v for _, v in ranked)
        kept = ranked[:n]
        plan[layer] = sorted(e for e, _ in kept)
        coverage[layer] = (sum(v for _, v in kept) / total) if total else None
    return plan, coverage


def plan_coverage(mass, threshold):
    plan, coverage = {}, {}
    for layer, scores in mass.items():
        ranked = _ranked(scores)
        total = sum(v for _, v in ranked)
        target = threshold * total
        acc, kept = 0.0, []
        for e, v in ranked:
            if acc >= target and kept:
                break
            kept.append(e)
            acc += v
        plan[layer] = sorted(kept)
        coverage[layer] = (acc / total) if total else None
    return plan, coverage


def hot_cold(mass, hot_fraction):
    pairs = []
    for layer, scores in mass.items():
        for e, v in scores.items():
            pairs.append((v, layer, e))
    pairs.sort(key=lambda x: -x[0])
    n_hot = int(round(len(pairs) * hot_fraction))
    hot = [[layer, e] for _, layer, e in pairs[:n_hot]]
    cold = [[layer, e] for _, layer, e in pairs[n_hot:]]
    return {"hot": hot, "cold": cold}


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces", nargs="+", help="RUNNER_MOE_TRACE JSONL file(s)")
    ap.add_argument("--keep-n", type=int, help="uniform per-layer expert count to keep")
    ap.add_argument("--coverage", type=float, help="per-layer saliency-mass coverage target (0-1)")
    ap.add_argument("--use-norms", action="store_true",
                     help="weight saliency by gate*norm instead of gate mass alone")
    ap.add_argument("--out", required=True, help="--prune-experts JSON output path")
    ap.add_argument("--precision-out", help="optional hot/cold precision-plan JSON output path")
    ap.add_argument("--hot-fraction", type=float, default=0.5,
                     help="fraction of (layer,expert) pairs, by saliency, kept 'hot' (default 0.5)")
    args = ap.parse_args(argv)

    if (args.keep_n is None) == (args.coverage is None):
        ap.error("exactly one of --keep-n or --coverage is required")

    mass = load_saliency(args.traces, args.use_norms)
    if not mass:
        ap.error(f"no MoE routing records found in {args.traces}")

    if args.keep_n is not None:
        plan, coverage = plan_keep_n(mass, args.keep_n)
    else:
        plan, coverage = plan_coverage(mass, args.coverage)

    out_plan = {f"layer_{l}": plan[l] for l in sorted(plan)}
    with open(args.out, "w") as f:
        json.dump(out_plan, f, indent=2)

    print(f"wrote {args.out}: {len(out_plan)} layer(s), "
          f"saliency={'gate*norm' if args.use_norms else 'gate mass'}", file=sys.stderr)
    for l in sorted(coverage):
        cov = coverage[l]
        cov_s = f"{cov:.4f}" if cov is not None else "n/a"
        print(f"  layer_{l}: keep {len(plan[l])} of {len(mass[l])}, coverage={cov_s}",
              file=sys.stderr)
    covered = [c for c in coverage.values() if c is not None]
    if covered:
        print(f"overall mean coverage: {sum(covered) / len(covered):.4f}", file=sys.stderr)

    if args.precision_out:
        hc = hot_cold(mass, args.hot_fraction)
        with open(args.precision_out, "w") as f:
            json.dump(hc, f, indent=2)
        print(f"wrote {args.precision_out}: {len(hc['hot'])} hot, {len(hc['cold'])} cold",
              file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
