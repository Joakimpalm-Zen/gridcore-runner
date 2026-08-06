#!/usr/bin/env python3
"""Six-prompt greedy-identity gate: runner vs a pinned llama.cpp reference.

Same protocol as the afmoe cert session (docs/afmoe-cert-goal-2026-08-05.md
gate 3): four short domains at 64 tokens, two of them repeated at 256.
Reference queried via /v1/completions with an explicit pure-greedy sampler
(repeat_penalty 1.0, top_k 0, top_p 1.0, min_p 0.0) and cache_prompt:false
(llama-server's prompt cache makes the reference itself n_predict-dependent
otherwise -- found the hard way during the afmoe session). Runner queried
via its CLI; stdout echoes the prompt, so the completion is stdout[len(prompt):].
Trailing whitespace trimmed on both sides before comparing.

Usage: cert-greedy-identity.py --model M.gguf --runner ./runner --endpoint http://127.0.0.1:PORT --out result.json
"""
import argparse
import hashlib
import json
import subprocess
import sys
import urllib.request

PROMPTS = [
    ("a", "The capital of France is", 64),
    ("b", "Write a python function that reverses a linked list.", 64),
    ("c", "In 1969, exactly 1234567 people watched as Apollo 11 landed. Summarize this event.", 64),
    ("d", "Beskriv kort hur en kvicksilvertermometer fungerar.", 64),
    ("b-long", "Write a python function that reverses a linked list.", 256),
    ("c-long", "In 1969, exactly 1234567 people watched as Apollo 11 landed. Summarize this event.", 256),
]


def ref_completion(endpoint, prompt, n):
    body = json.dumps({
        "prompt": prompt, "temperature": 0, "n_predict": n,
        "repeat_penalty": 1.0, "top_k": 0, "top_p": 1.0, "min_p": 0.0, "seed": 0,
        "cache_prompt": False,
    }).encode()
    req = urllib.request.Request(endpoint + "/completion", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.loads(r.read())
    return d["content"]


def runner_completion(runner, model, prompt, n):
    p = subprocess.run(
        [runner, "-m", model, "-p", prompt, "-n", str(n), "--temp", "0", "--gpu", "off"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=3600)
    out = p.stdout.decode("utf-8", "replace")
    if not out.startswith(prompt):
        raise SystemExit(f"runner stdout did not echo the prompt: {out[:120]!r}")
    return out[len(prompt):]


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return len(a) if len(a) != len(b) else -1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--runner", default="./runner")
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--out")
    args = ap.parse_args()

    results = []
    for tag, prompt, n in PROMPTS:
        sys.stderr.write(f"[{tag}] n={n} ...\n"); sys.stderr.flush()
        ref = ref_completion(args.endpoint, prompt, n)
        run = runner_completion(args.runner, args.model, prompt, n)
        ref_t, run_t = ref.rstrip(), run.rstrip()
        pos = first_diff(ref_t, run_t)
        ok = pos == -1
        results.append({
            "tag": tag, "n": n, "identical": ok, "first_divergent_byte": pos,
            "ref_sha256": hashlib.sha256(ref_t.encode()).hexdigest(),
            "runner_sha256": hashlib.sha256(run_t.encode()).hexdigest(),
            "ref_tail": ref_t[max(0, pos - 10):pos + 50] if pos >= 0 else "",
            "runner_tail": run_t[max(0, pos - 10):pos + 50] if pos >= 0 else "",
        })
        sys.stderr.write(f"[{tag}] identical={ok} first_diff_byte={pos}\n"); sys.stderr.flush()

    n_pass = sum(1 for r in results if r["identical"])
    summary = {"n_pass": n_pass, "n_total": len(results), "results": results}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
