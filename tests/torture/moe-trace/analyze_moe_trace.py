#!/usr/bin/env python3
"""Analyze RUNNER_MOE_TRACE=path JSONL output for expert-routing locality.

Each input line is {"pos":N,"layer":L,"experts":[8 ids],"gates":[8 weights]}
written by src/model.c's moe_trace_emit for every routed token. A single
trace file can hold several independent generations back to back (agent-
torture runs 35 cases through one server process); a per-layer position
reset (next pos <= previous pos for that layer) is treated as the start of
a new segment so sliding-window stats never blend two unrelated documents.

Usage: analyze_moe_trace.py workload_name=path.jsonl [workload_name=path ...]
"""
import sys
import json
from collections import defaultdict

WINDOWS = (64, 256, 1024)
CACHES = (8, 16, 24, 32)
CHECKPOINTS = (32, 128, 512, 2048)


def load_segments(path):
    """-> {layer: [segment, ...]} where segment is a list of (pos, experts, gates)
    ordered by pos, one entry per token that reached this layer."""
    per_layer = defaultdict(list)
    last_pos = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            L = rec["layer"]
            pos = rec["pos"]
            if L not in last_pos or pos <= last_pos[L]:
                per_layer[L].append([])  # new segment
            last_pos[L] = pos
            per_layer[L][-1].append((pos, rec["experts"], rec["gates"]))
    return per_layer


def standing_committee(per_layer):
    """Per layer: smallest expert set covering 50%/70% of total gate mass,
    over every token in every segment."""
    out = {}
    for L, segments in per_layer.items():
        mass = defaultdict(float)
        total = 0.0
        for seg in segments:
            for _, experts, gates in seg:
                for e, g in zip(experts, gates):
                    mass[e] += g
                    total += g
        ranked = sorted(mass.values(), reverse=True)
        out[L] = {}
        for frac in (0.5, 0.7):
            target = frac * total
            acc, n = 0.0, 0
            for v in ranked:
                if acc >= target:
                    break
                acc += v
                n += 1
            out[L][frac] = n
    return out


def cache_hit_curves(per_layer):
    """Per layer, per window size, per cache size: oracle and LRU hit rate.
    Oracle: non-overlapping windows of W tokens; cache = the W window's C
    most-frequently-selected experts (by selection count, not gate mass);
    hit rate = fraction of individual expert *slots* (8 per token) present
    in that set. LRU: a real LRU cache of capacity C, RESET at the start of
    each window (cold start), replayed slot-by-slot in selection order
    within the token; hit rate over the same slot population. Resetting per
    window answers "how fast does locality show up at this granularity",
    not "what would one warm long-running cache achieve" — see README.
    """
    out = {}
    for L, segments in per_layer.items():
        out[L] = {}
        for W in WINDOWS:
            out[L][W] = {}
            for C in CACHES:
                oracle_hits = oracle_total = 0
                lru_hits = lru_total = 0
                for seg in segments:
                    for i in range(0, len(seg), W):
                        window = seg[i:i + W]
                        if not window:
                            continue
                        freq = defaultdict(int)
                        for _, experts, _ in window:
                            for e in experts:
                                freq[e] += 1
                        top = set(sorted(freq, key=lambda e: -freq[e])[:C])
                        lru = []  # most-recent-last
                        for _, experts, _ in window:
                            for e in experts:
                                oracle_total += 1
                                if e in top:
                                    oracle_hits += 1
                                lru_total += 1
                                if e in lru:
                                    lru_hits += 1
                                    lru.remove(e)
                                elif len(lru) >= C:
                                    lru.pop(0)
                                lru.append(e)
                out[L][W][C] = {
                    "oracle_hit_rate": oracle_hits / oracle_total if oracle_total else None,
                    "lru_hit_rate": lru_hits / lru_total if lru_total else None,
                    "n_slots": oracle_total,
                }
    return out


def gate_reuse_overlap(per_layer):
    """Offline approximation (no hidden-state capture): for each token
    present at both layer L and layer L+1, the top-8/top-8 set overlap
    between its expert picks at the two layers, averaged over all tokens.
    This is P(expert overlap) by expert-SET intersection, not the true
    gate-reuse prefetch accuracy P(set@L+1 | set@L) a router probe over
    L's hidden state would give — see README for why.
    """
    layers = sorted(per_layer)
    out = {}
    for L in layers:
        if L + 1 not in per_layer:
            continue
        segs_l = per_layer[L]
        segs_l1 = {i: seg for i, seg in enumerate(per_layer[L + 1])}
        # segments are produced in the same order for every layer within one
        # forward call, so segment index i lines up across adjacent layers.
        total_overlap = 0
        total_tokens = 0
        for i, seg in enumerate(segs_l):
            other = segs_l1.get(i)
            if other is None:
                continue
            by_pos = {pos: set(experts) for pos, experts, _ in other}
            for pos, experts, _ in seg:
                if pos not in by_pos:
                    continue
                total_overlap += len(set(experts) & by_pos[pos])
                total_tokens += 1
        if total_tokens:
            out[f"{L}->{L+1}"] = {
                "mean_overlap_of_8": total_overlap / total_tokens,
                "n_tokens": total_tokens,
            }
    return out


def working_set_growth(per_layer):
    """Per layer: distinct experts touched after the first 32/128/512/2048
    tokens of each segment (segments shorter than a checkpoint are skipped
    for that checkpoint and the gap is reported, not silently dropped)."""
    out = {}
    for L, segments in per_layer.items():
        out[L] = {}
        for cp in CHECKPOINTS:
            seen_counts = []
            skipped = 0
            for seg in segments:
                if len(seg) < cp:
                    skipped += 1
                    continue
                distinct = set()
                for _, experts, _ in seg[:cp]:
                    distinct.update(experts)
                seen_counts.append(len(distinct))
            out[L][cp] = {
                "mean_distinct": sum(seen_counts) / len(seen_counts) if seen_counts else None,
                "n_segments": len(seen_counts),
                "segments_too_short": skipped,
            }
    return out


def summarize(path):
    per_layer = load_segments(path)
    n_segments = sum(len(v) for v in per_layer.values()) // max(len(per_layer), 1)
    return {
        "n_layers": len(per_layer),
        "approx_n_segments": n_segments,
        "standing_committee": standing_committee(per_layer),
        "cache_hit_curves": cache_hit_curves(per_layer),
        "gate_reuse_overlap": gate_reuse_overlap(per_layer),
        "working_set_growth": working_set_growth(per_layer),
    }


def main(argv):
    if not argv:
        print(__doc__)
        return 1
    results = {}
    for arg in argv:
        name, _, path = arg.partition("=")
        if not path:
            name, path = path or name, name
        print(f"analyzing {name} <- {path}", file=sys.stderr)
        results[name] = summarize(path)
    json.dump(results, sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
