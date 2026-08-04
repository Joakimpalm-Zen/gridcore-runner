#!/usr/bin/env python3
"""JC-R2 Phase 0: classify grammar-draft rejection causes from a
RUNNER_GRAMMAR_TRACE JSONL file.

Classes (per diverged round):
  tail-straddle   actual token's bytes extend the expected token's bytes
                  BEYOND the end of the pinned run (merge into free text)
  coarse-merge    actual extends expected but stays within the pinned run
                  (model merges more of the pin than raw encode did)
  fine-split      expected extends actual (model segments finer than the
                  raw encode's greedy merge)
  seam            divergence at draft index 0 with no prefix relation
                  (context-boundary effect: raw encode of the pin alone
                  starts differently than the model continues)
  other           no prefix relation, not at index 0

Also reports: overall acceptance, fraction of drafts accepted before first
divergence, span-length stats, and what acceptance WOULD be with tail
withholding (drop the final draft token of every round).
"""
import json
import sys
from collections import Counter


def main(path):
    rounds = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            rounds.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # truncated tail line

    n = len(rounds)
    drafted = sum(len(r["draft"]) for r in rounds)
    accepted = sum(r["accepted"] for r in rounds)
    diverged = [r for r in rounds if "div" in r]
    full = n - len(diverged)

    cls = Counter()
    tail_pos_hist = Counter()  # divergence at last draft index?
    for r in diverged:
        pin = bytes.fromhex(r["pin"])
        exp = bytes.fromhex(r["exp_b"])
        act = bytes.fromhex(r["act_b"])
        off = r.get("off", 0)
        last = r["div"] == len(r["draft"]) - 1
        tail_pos_hist[last] += 1
        if act.startswith(exp) and len(act) > len(exp):
            # actual merges further than expected: past the pin end?
            if off + len(act) > len(pin):
                cls["tail-straddle"] += 1
            else:
                cls["coarse-merge"] += 1
        elif exp.startswith(act) and len(act) < len(exp):
            cls["fine-split"] += 1
        elif r["div"] == 0:
            cls["seam"] += 1
        else:
            cls["other"] += 1

    print(f"rounds={n} drafted={drafted} accepted={accepted} "
          f"({100*accepted/drafted:.1f}%)" if drafted else "no drafts")
    print(f"fully-accepted rounds={full}/{n}")
    print("\nrejection classes (of diverged rounds):")
    for k, v in cls.most_common():
        print(f"  {k:14s} {v:5d}  ({100*v/len(diverged):.1f}%)")
    print(f"\ndivergence at final draft index: {tail_pos_hist[True]}"
          f"/{len(diverged)} ({100*tail_pos_hist[True]/len(diverged):.1f}%)"
          if diverged else "")

    # counterfactual: tail withholding = drop the last draft token of every
    # round; a divergence AT the last index becomes a non-event (that token
    # was never drafted), divergences before it are unchanged.
    cf_drafted = sum(max(0, len(r["draft"]) - 1) for r in rounds)
    cf_accepted = 0
    for r in rounds:
        nd = len(r["draft"])
        if "div" not in r:
            cf_accepted += max(0, nd - 1)
        elif r["div"] == nd - 1:
            cf_accepted += r["accepted"]      # accepted everything kept
        else:
            cf_accepted += r["accepted"]      # unchanged
    if cf_drafted:
        print(f"\ncounterfactual tail-withholding: {cf_accepted}/{cf_drafted}"
              f" = {100*cf_accepted/cf_drafted:.1f}% acceptance"
              f" (vs {100*accepted/drafted:.1f}%)")

    # span length distribution (drafted tokens per round)
    lens = Counter(len(r["draft"]) for r in rounds)
    print("\nspan length (drafted tokens/round): "
          + ", ".join(f"{k}:{v}" for k, v in sorted(lens.items())))


if __name__ == "__main__":
    main(sys.argv[1])
