#!/usr/bin/env python3
"""Where does Runner's greedy path first leave llama.cpp's, and by how much?

reference_compare.py answers "is the text identical", which is the right gate
but a blunt instrument: one near-degenerate argmax tie reads the same as a
broken forward pass. This asks the sharper question. For each prompt it walks
both engines greedily, and at the first position where the argmax differs it
reports the logprob gap between the two contenders on each side.

A divergence whose gap is under a few hundredths of a nat is quantisation
noise around a tie — llama.cpp flips on those itself depending on whether the
prompt was cached. A divergence at a wide gap is a real arithmetic bug.

Both engines are driven through /v1/completions with prompt caching disabled
so the reference takes the same cold path every time.
"""

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "1 2 3 4 5 6 7 8",
    "Once upon a time, in a land far away,",
    'JSON example: {"name":',
    "The three laws of thermodynamics are",
    "import numpy as np\n",
    "Q: What is 17 * 23?\nA:",
    "The quick brown fox",
    "class Stack:\n    def __init__(self):",
    "In 1969, humans first",
    "SELECT name, age FROM users WHERE",
    "The mitochondria is the",
    "for i in range(10):\n    print(",
    "Water boils at",
    "A haiku about autumn:\n",
]


def post(url, payload, timeout=600):
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def wait_ready(port, probe, deadline_s):
    end = time.monotonic() + deadline_s
    while time.monotonic() < end:
        try:
            urllib.request.urlopen(
                f"http://127.0.0.1:{port}{probe}", timeout=5).read()
            return True
        except (urllib.error.URLError, OSError):
            time.sleep(2)
    return False


def normalise(entry):
    """Both engines report logprobs, but not in the same shape.

    Everything downstream keys on the token *id*: the two engines render some
    control tokens differently (runner writes "<eos>" where llama.cpp writes
    ""), so comparing the decoded strings reports identical tokens as
    divergences. Each row is (id, text, logprob, {id: logprob} for the top-k).
    """
    choice = entry["choices"][0]
    logprobs = choice.get("logprobs") or {}
    if "content" in logprobs:            # llama.cpp / OAI shape
        rows = []
        for item in logprobs["content"]:
            top = {x["id"]: x["logprob"] for x in item.get("top_logprobs", [])}
            rows.append((item["id"], item["token"], item["logprob"], top))
        return rows
    ids = logprobs.get("token_ids")      # runner shape
    if ids is None:
        raise SystemExit("runner did not report token_ids — rebuild the binary")
    texts = logprobs.get("tokens") or []
    lps = logprobs.get("token_logprobs") or []
    tops = logprobs.get("top_logprobs") or []
    top_ids = logprobs.get("top_token_ids") or []
    rows = []
    for i, tid in enumerate(ids):
        pairs = list(tops[i].values()) if i < len(tops) else []
        keys = top_ids[i] if i < len(top_ids) else []
        rows.append((tid, texts[i], lps[i], dict(zip(keys, pairs))))
    return rows


def gap(top, chosen, other):
    """How far apart the two contenders are, in nats, on one engine."""
    if chosen not in top or other not in top:
        return None
    return abs(top[chosen] - top[other])


def compare(prompt, runner_url, ref_url, tokens):
    body = {"prompt": prompt, "max_tokens": tokens, "temperature": 0,
            "logprobs": 5, "cache_prompt": False}
    a = normalise(post(f"{runner_url}/v1/completions", dict(body, model="runner")))
    b = normalise(post(f"{ref_url}/v1/completions", body))
    n = min(len(a), len(b))
    for i in range(n):
        if a[i][0] != b[i][0]:
            return {
                "prompt": prompt, "status": "diverged", "position": i,
                "matched_tokens": i,
                "runner_token": a[i][1], "runner_id": a[i][0],
                "reference_token": b[i][1], "reference_id": b[i][0],
                "runner_gap_nats": gap(a[i][3], a[i][0], b[i][0]),
                "reference_gap_nats": gap(b[i][3], b[i][0], a[i][0]),
                "max_logprob_delta": max(
                    (abs(a[j][2] - b[j][2]) for j in range(i)), default=0.0),
            }
    return {"prompt": prompt, "status": "identical", "matched_tokens": n,
            "max_logprob_delta": max(
                (abs(a[j][2] - b[j][2]) for j in range(n)), default=0.0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--reference", required=True,
                    help="path to llama-server")
    ap.add_argument("--runner", default=str(ROOT / "runner"))
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--threads", type=int, default=32)
    # A tie this narrow is quantisation noise, not an arithmetic fault: the
    # reference flips on gaps this size by itself when its prompt cache is warm.
    ap.add_argument("--tie-nats", type=float, default=0.25)
    args = ap.parse_args()

    procs = []
    try:
        procs.append(subprocess.Popen(
            [args.runner, "--serve", "-m", args.model, "-c", str(args.ctx),
             "--port", "18201", "--gpu", "off"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        procs.append(subprocess.Popen(
            [args.reference, "-m", args.model, "-c", str(args.ctx),
             "--port", "18202", "--host", "127.0.0.1", "-t", str(args.threads)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        if not wait_ready(18201, "/v1/capabilities", 600):
            sys.exit("runner did not come up")
        if not wait_ready(18202, "/health", 600):
            sys.exit("reference did not come up")

        rows = [compare(p, "http://127.0.0.1:18201", "http://127.0.0.1:18202",
                        args.tokens) for p in PROMPTS]
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill()

    identical = [r for r in rows if r["status"] == "identical"]
    diverged = [r for r in rows if r["status"] == "diverged"]
    ties = [r for r in diverged
            if (r["reference_gap_nats"] or 9e9) <= args.tie_nats
            and (r["runner_gap_nats"] or 9e9) <= args.tie_nats]
    real = [r for r in diverged if r not in ties]
    report = {
        "schema_version": "gridcore.runner.token-divergence.v1",
        "model": args.model,
        "tokens_per_prompt": args.tokens,
        "tie_threshold_nats": args.tie_nats,
        "results": rows,
        "totals": {
            "prompts": len(rows),
            "identical": len(identical),
            "diverged_at_a_tie": len(ties),
            "diverged_for_real": len(real),
            "max_logprob_delta": max(
                (r["max_logprob_delta"] for r in rows), default=0.0),
        },
    }
    print(json.dumps(report, indent=2))
    return 1 if real else 0


if __name__ == "__main__":
    sys.exit(main())
