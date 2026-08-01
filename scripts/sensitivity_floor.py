#!/usr/bin/env python3
"""How much does THIS model amplify a small numeric change?

A logprob gap against another engine means nothing on its own. Some models are
numerically placid and some are chaotic, and a chaotic one will disagree with
any second implementation — including a second run of the same implementation.
Measure the floor before calling a gap a bug.

Two modes, both comparing an engine against ITSELF on the same weights:

  --mode runner     runner with an f16 KV cache vs runner with a q8_0 KV cache.
                    A bounded arithmetic perturbation, entirely inside one
                    build.

  --mode reference  llama.cpp serving a cold-prefilled prompt vs the same
                    prompt from a warm prefix cache. No quantisation change at
                    all — just a different order of floating-point work, which
                    is the smallest honest perturbation available.

Read the output next to scripts/token_divergence.py run on the same model with
the same --tokens. If a model disagrees with itself on N prompts, no
cross-engine comparison can be expected to do better than N.

Worked example (16 prompts, 16 tokens, b10076):

    model                        reference-vs-self   runner-vs-llama.cpp
    Qwen2.5-7B Q4_K_M                    16/16              12/16
    gemma-4-E4B Q4_K_M                   12/16              11/16
    gemma-4-26B-A4B Q4_0                 12/16               7/16

E4B sits at llama.cpp's own instability floor. The 26B MoE does not — but its
runner-vs-runner mean logprob movement is ~18x E4B's, which is a larger factor
than the cross-engine gap it shows, so the amplification accounts for it.
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


def post(url, payload, timeout=900):
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def wait_ready(port, probe, needle, deadline_s):
    end = time.monotonic() + deadline_s
    while time.monotonic() < end:
        try:
            body = urllib.request.urlopen(
                f"http://127.0.0.1:{port}{probe}", timeout=5).read()
            if needle is None or needle in body:
                return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(2)
    return False


def rows(entry):
    """(id, logprob) per generated token, from either engine's shape."""
    logprobs = entry["choices"][0]["logprobs"]
    if "content" in logprobs:                       # llama.cpp / OAI
        return [(c["id"], c["logprob"]) for c in logprobs["content"]]
    ids = logprobs.get("token_ids")
    if ids is None:
        raise SystemExit("runner did not report token_ids — rebuild the binary")
    return list(zip(ids, logprobs["token_logprobs"]))


def summarise(model, mode, pairs, tokens):
    identical, first_div, deltas = 0, [], []
    for a, b in pairs:
        n = min(len(a), len(b))
        div = next((i for i in range(n) if a[i][0] != b[i][0]), None)
        first_div.append(n if div is None else div)
        if div is None:
            identical += 1
        deltas.extend(abs(a[i][1] - b[i][1]) for i in range(first_div[-1]))
    return {
        "schema_version": "gridcore.runner.sensitivity-floor.v1",
        "model": model,
        "mode": mode,
        "tokens_per_prompt": tokens,
        "prompts": len(pairs),
        "identical": identical,
        "mean_tokens_before_divergence": round(
            sum(first_div) / len(first_div), 2),
        "max_logprob_delta": round(max(deltas), 4) if deltas else 0.0,
        "mean_logprob_delta": round(sum(deltas) / len(deltas), 5) if deltas else 0.0,
    }


def run_runner(args):
    procs, pairs = [], []
    try:
        for port, kv in ((18301, "f16"), (18302, "q8")):
            procs.append(subprocess.Popen(
                [args.runner, "--serve", "-m", args.model, "-c", str(args.ctx),
                 "--port", str(port), "--gpu", "off", "--kv", kv],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        for port in (18301, 18302):
            if not wait_ready(port, "/v1/capabilities", None, 900):
                sys.exit(f"runner on {port} did not come up")
        for prompt in PROMPTS:
            body = {"prompt": prompt, "max_tokens": args.tokens, "temperature": 0,
                    "logprobs": 3, "cache_prompt": False, "model": "runner"}
            pairs.append((
                rows(post("http://127.0.0.1:18301/v1/completions", body)),
                rows(post("http://127.0.0.1:18302/v1/completions", body))))
    finally:
        stop(procs)
    return pairs


def run_reference(args):
    if not args.reference:
        sys.exit("--mode reference needs --reference <path to llama-server>")
    procs, pairs = [], []
    try:
        procs.append(subprocess.Popen(
            [args.reference, "-m", args.model, "-c", str(args.ctx),
             "--port", "18303", "--host", "127.0.0.1", "-t", str(args.threads)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        if not wait_ready(18303, "/health", b"ok", 900):
            sys.exit("reference did not come up")
        url = "http://127.0.0.1:18303/v1/completions"
        for prompt in PROMPTS:
            base = {"prompt": prompt, "max_tokens": args.tokens,
                    "temperature": 0, "logprobs": 2}
            cold = rows(post(url, dict(base, cache_prompt=False)))
            post(url, dict(base, cache_prompt=True))   # warm this exact prefix
            warm = rows(post(url, dict(base, cache_prompt=True)))
            pairs.append((cold, warm))
    finally:
        stop(procs)
    return pairs


def stop(procs):
    for p in procs:
        p.terminate()
    for p in procs:
        try:
            p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            p.kill()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=("runner", "reference"), default="runner")
    ap.add_argument("--reference", help="path to llama-server (--mode reference)")
    ap.add_argument("--runner", default=str(ROOT / "runner"))
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--threads", type=int, default=32)
    args = ap.parse_args()

    pairs = run_runner(args) if args.mode == "runner" else run_reference(args)
    label = ("runner f16-KV vs runner q8-KV" if args.mode == "runner"
             else "llama.cpp cold-cache vs warm-cache")
    print(json.dumps(summarise(args.model, label, pairs, args.tokens), indent=2))


if __name__ == "__main__":
    main()
