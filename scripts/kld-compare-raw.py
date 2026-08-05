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


def kld_and_agreement(dist_a, dist_b):
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
    ap.add_argument("--model-name-a")
    ap.add_argument("--model-name-b")
    ap.add_argument("--runner", default="./runner")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--max-positions", type=int, default=64)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--port-a", type=int, default=58611)
    ap.add_argument("--port-b", type=int, default=58612)
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
        klds, top1s, overlaps = [], [], []
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
            kld, agree, overlap8 = kld_and_agreement(da, db)
            klds.append(kld); top1s.append(agree); overlaps.append(overlap8)
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
        "positions_scored": n_scored,
        "positions_failed": n_failed,
        "mean_kld": sum(klds) / len(klds) if klds else None,
        "top1_agreement_pct": 100.0 * sum(top1s) / len(top1s) if top1s else None,
        "mean_top8_overlap": sum(overlaps) / len(overlaps) if overlaps else None,
    }
    print(json.dumps(result, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
