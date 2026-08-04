#!/usr/bin/env python3
"""Compare two runner-served models' next-token distributions over a pinned
corpus, via the chat-completions logprobs surface (top_logprobs, up to 20 —
runner has no raw full-vocab logit dump). KLD here is therefore an
APPROXIMATION: restricted to the union of both models' top-20 token sets at
each position, each side's partial mass renormalized over just that union.
This still gives an exact 0 for two identical distributions (the
self-comparison gate) and a meaningful, monotonic-with-real-divergence
signal for genuinely different models — it is NOT the exact full-vocab KLD
`llama-perplexity --kl-divergence` computes from raw logits (use that for
gemma's own requant fidelity, per the trip-4 goal's Task 5b), but it is
directly comparable at the same top_logprobs depth for any two
runner-served models, which the raw-logits tool isn't (it needs both models
loaded in the SAME llama.cpp process against the same corpus binary).

Teacher forcing without a tokenize endpoint: the corpus is walked WORD BY
WORD (whitespace-delimited), and at each word boundary both models predict
ONE next token (max_tokens=1, temp=0, logprobs on, top_logprobs=20) given
the IDENTICAL text prefix so far. This samples fewer positions than true
per-token teacher forcing (a multi-token word is only scored at its
boundary) but needs no tokenizer access and is exactly reproducible as long
as both models share a tokenizer — true for any two quants/prunes of the
same base model, which is what this tool is for.

Usage:
  kld-compare.py --model-a A.gguf --model-b B.gguf --corpus FILE.txt
  kld-compare.py --endpoint-a http://127.0.0.1:PORT --model-name-a NAME \
      --endpoint-b http://127.0.0.1:PORT2 --model-name-b NAME2 --corpus FILE.txt
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


def query(endpoint, model_name, prompt):
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1,
        "temperature": 0,
        "logprobs": True,
        "top_logprobs": 20,
    }
    req = urllib.request.Request(
        f"{endpoint}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read())
    lp = body["choices"][0]["logprobs"]["content"][0]
    top = {e["token"]: e["logprob"] for e in lp["top_logprobs"]}
    top[lp["token"]] = lp["logprob"]  # the sampled token is always in-distribution
    return top


def kld_and_agreement(dist_a, dist_b):
    """-> (approx_kld, top1_match: bool, top8_overlap: 0..1)"""
    union = set(dist_a) | set(dist_b)
    # A token seen in one side's top-20 but absent from the other's gets a
    # floor one nat below that side's smallest observed logprob in this
    # union — a conservative, divergence-SUPPRESSING stand-in (it assumes
    # "at least this unlikely," never invents a larger gap than observed)
    # for mass this approximation can't see the exact size of.
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
    top8_a = set(sorted(dist_a, key=lambda t: -dist_a[t])[:8])
    top8_b = set(sorted(dist_b, key=lambda t: -dist_b[t])[:8])
    denom = max(1, min(8, len(top8_a), len(top8_b)))
    overlap8 = len(top8_a & top8_b) / denom
    return kld, top1_a == top1_b, overlap8


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-a")
    ap.add_argument("--model-b")
    ap.add_argument("--endpoint-a")
    ap.add_argument("--endpoint-b")
    ap.add_argument("--model-name-a",
                     help="the `model` field to send — must match the server's "
                          "/v1/models id exactly; defaults to --model-a's basename "
                          "when spawning that server, otherwise required")
    ap.add_argument("--model-name-b", help="see --model-name-a")
    ap.add_argument("--runner", default="./runner")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--max-positions", type=int, default=64)
    ap.add_argument("--stride", type=int, default=1, help="compare every Nth word boundary")
    ap.add_argument("--port-a", type=int, default=58601)
    ap.add_argument("--port-b", type=int, default=58602)
    ap.add_argument("--out", help="optional path to also write the JSON result")
    args = ap.parse_args(argv)

    procs = []
    try:
        if args.endpoint_a:
            ep_a = args.endpoint_a
            if not args.model_name_a:
                ap.error("--model-name-a is required with --endpoint-a "
                         "(must match that server's /v1/models id)")
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
                ap.error("--model-name-b is required with --endpoint-b "
                         "(must match that server's /v1/models id)")
        else:
            if not args.model_b:
                ap.error("need --model-b or --endpoint-b")
            if not args.model_name_b:
                args.model_name_b = os.path.basename(args.model_b)
            print(f"starting server B: {args.model_b}", file=sys.stderr)
            procs.append(start_server(args.runner, args.model_b, args.port_b))
            ep_b = f"http://127.0.0.1:{args.port_b}"

        words = open(args.corpus).read().split()
        if not words:
            ap.error(f"{args.corpus} is empty")

        klds, top1_hits, overlaps = [], [], []
        n_scored, n_failed = 0, 0
        for k in range(1, len(words), args.stride):
            if n_scored >= args.max_positions:
                break
            prefix = " ".join(words[:k])
            try:
                da = query(ep_a, args.model_name_a, prefix)
                db = query(ep_b, args.model_name_b, prefix)
            except Exception as e:
                print(f"warning: position {k} failed: {e}", file=sys.stderr)
                n_failed += 1
                continue
            kld, hit, overlap = kld_and_agreement(da, db)
            klds.append(kld)
            top1_hits.append(hit)
            overlaps.append(overlap)
            n_scored += 1

        if not klds:
            print("error: no positions scored", file=sys.stderr)
            return 1

        result = {
            "positions_scored": n_scored,
            "positions_failed": n_failed,
            "mean_kld": sum(klds) / len(klds),
            "top1_agreement_pct": 100.0 * sum(top1_hits) / len(top1_hits),
            "mean_top8_overlap": sum(overlaps) / len(overlaps),
        }
        text = json.dumps(result, indent=2)
        print(text)
        if args.out:
            with open(args.out, "w") as f:
                f.write(text + "\n")
        return 0
    finally:
        for p in procs:
            p.kill()
            p.wait()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
