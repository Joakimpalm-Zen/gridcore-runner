#!/usr/bin/env python3
"""Cross-engine KLD/top-1/top-8 comparison over RAW (non-chat) completions.

kld-compare.py compares next-token distributions via /v1/chat/completions,
which is correct for two RUNNER instances (same chat-template renderer on
both sides) but unsound across engines for architectures with a structured
chat format: gpt-oss's Harmony template is rendered differently by the
runner (plain conversational continuation) and by llama.cpp (forces
`<|channel|>` markup via response schema), so a chat-endpoint comparison
finds the two engines' TEMPLATES disagree, not their WEIGHTS -- measured
directly during the cert-matrix session (0% top-1, KLD 31.2 via chat,
collapsing to near-perfect agreement via raw completions on the identical
question). This script does the same word-by-word teacher-forcing KLD
protocol as kld-compare.py, but against /v1/completions (no template
applied on either side), and normalizes the two engines' differing
logprobs response schemas (runner: parallel tokens/token_logprobs/
top_logprobs arrays; llama.cpp: content list of {token, logprob,
top_logprobs} dicts).

Same approximation caveat as kld-compare.py: KLD is restricted to the union
of both sides' top-N logprobs and renormalized over just that union.

Usage:
  kld-compare-raw.py --model-a A.gguf --runner ./runner \\
      --endpoint-b http://127.0.0.1:PORT --model-name-b NAME \\
      --corpus FILE.txt --max-positions 400 --out result.json
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request


def start_server(runner, model, port):
    proc = subprocess.Popen(
        [runner, "-m", model, "--serve", "--port", str(port), "--gpu", "off"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(400):
        if proc.poll() is not None:
            raise RuntimeError(f"runner exited early loading {model} (code {proc.returncode})")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=1)
            return proc
        except (urllib.error.URLError, ConnectionError):
            time.sleep(0.5)
    proc.kill()
    raise RuntimeError(f"server for {model} on port {port} did not come up")


def query(endpoint, model_name, prompt, top_n=20):
    payload = {"model": model_name, "prompt": prompt, "max_tokens": 1,
              "temperature": 0, "logprobs": top_n}
    req = urllib.request.Request(
        f"{endpoint}/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read())
    lp = body["choices"][0]["logprobs"]
    if "content" in lp:  # llama.cpp OpenAI-style schema
        step = lp["content"][0]
        top = {e["token"]: e["logprob"] for e in step["top_logprobs"]}
        top[step["token"]] = step["logprob"]
    else:  # runner's schema: parallel arrays, top_logprobs is a dict already
        top = dict(lp["top_logprobs"][0])
        top[lp["tokens"][0]] = lp["token_logprobs"][0]
    return top


# ------------------------------------------------- quality bar v2: near-ties
#
# Plain top-1 agreement counts every argmax flip as a disagreement, including
# flips between two tokens the REFERENCE itself could barely separate.
# Measured 2026-08-13 on this box, that is exactly what the middle of the
# quality ladder fails on: granite-4.1-3b Q6_K scored mean KLD 0.0155 (three
# times inside the 0.05 bound) while missing 97% top-1 by four points, and
# Phi-4-mini Q8_0 scored KLD 0.0082 with 94.75% top-1. Distributions that
# close are not damaged; their argmaxes are coin-flipping.
#
# BAND DERIVATION, against the precedent this project already uses.
# test_tc_tol forgives a top-1 flip whose top-two gap is within TIE_FRAC =
# 0.02 of the logit range. Logprobs are shift-invariant softmax outputs, so a
# top-two LOGPROB gap is numerically the same quantity as a top-two LOGIT gap.
# The mean logit ranges tc-tol measures on real models here are ~25.5
# (Llama-3.2-3B) to ~53.4 (Phi-4-mini), so its band is 0.51 to 1.07 nats. We
# take 0.5 — the conservative end of that translation, forgiving strictly less
# than tc-tol would on every model measured. In probability terms the runner-up
# must hold at least exp(-0.5) = 61% of the leader's mass.
#
# The margin is read from the REFERENCE side, never the variant's. A variant
# that is unsure where the reference was certain has lost something real; a
# variant that picks differently where the reference had no opinion has not.
DEFAULT_TIE_BAND = 0.5


def top_two_margin(dist):
    """Reference-side gap between the best and second-best logprob.

    None when the distribution has fewer than two usable entries: there is no
    gap to measure, and inventing one would forgive a flip on no evidence.
    """
    if len(dist) < 2:
        return None
    ordered = sorted(dist.values(), reverse=True)
    return ordered[0] - ordered[1]


def score_pair(dist_a, dist_b, tie_band=DEFAULT_TIE_BAND):
    """(kld, plain_agree, margin_qualified_agree, top8_overlap).

    dist_a is the variant under test, dist_b the reference. KLD direction and
    the plain-agreement and overlap definitions are unchanged from v1, so every
    published number keeps reproducing.
    """
    union = set(dist_a) | set(dist_b)
    floor_a = min(dist_a.values()) - 1.0 if dist_a else -20.0
    floor_b = min(dist_b.values()) - 1.0 if dist_b else -20.0
    pa = {t: math.exp(dist_a.get(t, floor_a)) for t in union}
    pb = {t: math.exp(dist_b.get(t, floor_b)) for t in union}
    za, zb = sum(pa.values()), sum(pb.values())
    kld = 0.0
    for t in union:
        p, q = pa[t] / za, pb[t] / zb
        if p > 0:
            kld += p * math.log(p / q)
    top1_a = max(dist_a, key=dist_a.get)
    top1_b = max(dist_b, key=dist_b.get)
    agree = top1_a == top1_b
    margin = top_two_margin(dist_b)
    marg_agree = agree or (margin is not None and margin <= tie_band)
    top8_a = set(sorted(dist_a, key=lambda t: -dist_a[t])[:8])
    top8_b = set(sorted(dist_b, key=lambda t: -dist_b[t])[:8])
    denom = max(1, min(8, len(top8_a), len(top8_b)))
    overlap8 = len(top8_a & top8_b) / denom
    return kld, agree, marg_agree, overlap8


def kld_and_agreement(dist_a, dist_b):
    """v1 shape, kept so any caller predating the margin column still works."""
    kld, agree, _marg, overlap8 = score_pair(dist_a, dist_b)
    return kld, agree, overlap8


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-a")
    ap.add_argument("--model-b")
    ap.add_argument("--endpoint-a")
    ap.add_argument("--endpoint-b")
    ap.add_argument("--model-name-a")
    ap.add_argument("--model-name-b")
    ap.add_argument("--runner", default="./runner")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--max-positions", type=int, default=64)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--port-a", type=int, default=58611)
    ap.add_argument("--port-b", type=int, default=58612)
    ap.add_argument("--tie-band", type=float, default=DEFAULT_TIE_BAND,
                    help="near-tie band in nats for margin-qualified top-1 "
                         f"(default {DEFAULT_TIE_BAND}; see the derivation "
                         "against test_tc_tol's 0.02-of-range in this script)")
    ap.add_argument("--out")
    args = ap.parse_args(argv)

    procs = []
    try:
        if args.endpoint_a:
            ep_a = args.endpoint_a
            if not args.model_name_a:
                ap.error("--model-name-a required with --endpoint-a")
        else:
            if not args.model_a:
                ap.error("need --model-a or --endpoint-a")
            if not args.model_name_a:
                args.model_name_a = os.path.basename(args.model_a)
            print(f"starting server A: {args.model_a}", file=sys.stderr)
            procs.append(start_server(args.runner, args.model_a, args.port_a))
            ep_a = f"http://127.0.0.1:{args.port_a}"
        if args.endpoint_b:
            ep_b = args.endpoint_b
            if not args.model_name_b:
                ap.error("--model-name-b required with --endpoint-b")
        else:
            if not args.model_b:
                ap.error("need --model-b or --endpoint-b")
            if not args.model_name_b:
                args.model_name_b = os.path.basename(args.model_b)
            print(f"starting server B: {args.model_b}", file=sys.stderr)
            procs.append(start_server(args.runner, args.model_b, args.port_b))
            ep_b = f"http://127.0.0.1:{args.port_b}"

        words = open(args.corpus, encoding="utf-8").read().split()
        n_scored = 0
        n_failed = 0
        klds, top1s, overlaps, margs, positions = [], [], [], [], []
        prefix = ""
        for i, w in enumerate(words):
            prefix = (prefix + " " + w).strip() if prefix else w
            if i % args.stride != 0:
                continue
            if n_scored >= args.max_positions:
                break
            try:
                da = query(ep_a, args.model_name_a, prefix)
                db = query(ep_b, args.model_name_b, prefix)
            except Exception as e:
                n_failed += 1
                print(f"position {i} failed: {e}", file=sys.stderr)
                continue
            kld, agree, marg, overlap8 = score_pair(da, db, args.tie_band)
            klds.append(kld); top1s.append(agree); overlaps.append(overlap8)
            margs.append(marg)
            # Per-position records so a bar change can be re-scored from the
            # evidence instead of re-running every model. v1 stored summaries
            # only, which is why the 2026-08-14 both-ways re-report had to
            # re-run nine rows from scratch.
            positions.append({"i": i, "kld": kld, "agree": agree,
                              "margin_agree": marg,
                              "ref_margin": top_two_margin(db),
                              "top8": overlap8})
            n_scored += 1
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill()

    result = {
        # v2 (2026-08-14) ADDS the margin-qualified column and per-position
        # records. Every v1 field keeps its name and its meaning, so published
        # v1 numbers stay comparable and reproducible.
        "schema_version": "xyntetik.runner.kld-raw.v2",
# ADOPTED 2026-08-14 (owner decision): margin-qualified top-1 is the
# publication criterion; plain top-1 is reported beside it forever.
        "tie_band_nats": args.tie_band,
        "positions_scored": n_scored,
        "positions_failed": n_failed,
        "mean_kld": sum(klds) / len(klds) if klds else None,
        "top1_agreement_pct": 100.0 * sum(top1s) / len(top1s) if top1s else None,
        "top1_margin_qualified_pct":
            100.0 * sum(margs) / len(margs) if margs else None,
        "mean_top8_overlap": sum(overlaps) / len(overlaps) if overlaps else None,
        "positions": positions,
    }
    print(json.dumps(result, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
