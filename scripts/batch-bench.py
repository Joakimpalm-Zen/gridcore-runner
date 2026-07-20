#!/usr/bin/env python3
"""Continuous-batching gate: real concurrent HTTP requests against a real server.

Answers the two Phase 6 exit questions that only an end-to-end run can answer:

  1. Does concurrent throughput materially exceed independent-slot execution?
     Measured against a baseline binary serving the same model with the same
     --parallel, which is exactly "independent slots" — not against a synthetic
     harness and not against single-stream, so the comparison isolates
     batching rather than parallelism.
  2. Is a batched request's output identical to a solo one's? Every concurrent
     response is compared byte-for-byte with the same prompt run alone.

Single-request latency is reported alongside, because the failure mode of a
batching scheduler is paying for throughput with it.

    python scripts/batch-bench.py --candidate ./runner --baseline ./runner-old \
        --model models/Qwen3-4B-Q4_K_M.gguf --concurrency 4
"""

import argparse
import json
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

PROMPTS = [
    "Write a haiku about the sea.",
    "List three prime numbers.",
    "Explain gravity in one sentence.",
    "Name a color and a fruit.",
    "What is the capital of Japan?",
    "Count from one to five.",
    "Describe rain briefly.",
    "Say hello politely.",
]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class Server:
    def __init__(self, exe, model, parallel, ctx, extra):
        self.port = free_port()
        self.args = [str(Path(exe).resolve()), "-m", str(Path(model).resolve()),
                     "--serve", "--port", str(self.port),
                     "--parallel", str(parallel), "-c", str(ctx)] + list(extra)
        self.log = None

    def __enter__(self):
        self.logf = open(self.log, "w") if self.log else subprocess.DEVNULL
        self.p = subprocess.Popen(self.args, stdout=subprocess.DEVNULL,
                                  stderr=self.logf, text=True)
        deadline = time.time() + 300
        while time.time() < deadline:
            if self.p.poll() is not None:
                raise RuntimeError("server exited during startup")
            try:
                urllib.request.urlopen(
                    f"http://127.0.0.1:{self.port}/health", timeout=1).read()
                return self
            except Exception:
                time.sleep(0.25)
        raise RuntimeError("server did not become healthy")

    def __exit__(self, *exc):
        self.p.terminate()
        try:
            self.p.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.p.kill()

    def chat(self, prompt, max_tokens):
        body = json.dumps({
            "model": "m",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            # each request must stand alone: a shared prefix cache would make
            # the second run of a prompt faster for reasons unrelated to batching
            "cache_prompt": False,
        }).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/v1/chat/completions", data=body,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=900) as r:
            d = json.loads(r.read())
        return (d["choices"][0]["message"].get("content") or "",
                int(d.get("usage", {}).get("completion_tokens", 0)))


def run_concurrent(srv, prompts, max_tokens):
    out = [None] * len(prompts)
    errs = []

    def work(i):
        try:
            out[i] = srv.chat(prompts[i], max_tokens)
        except Exception as e:  # noqa: BLE001
            errs.append(f"{prompts[i]!r}: {e}")

    ths = [threading.Thread(target=work, args=(i,)) for i in range(len(prompts))]
    t0 = time.time()
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    dt = time.time() - t0
    if errs:
        raise RuntimeError("; ".join(errs))
    return out, dt


def measure(srv, prompts, max_tokens, label):
    out, dt = run_concurrent(srv, prompts, max_tokens)
    toks = sum(n for _, n in out)
    print(f"  {label}: {toks} tok in {dt:.2f}s = {toks / dt:.1f} tok/s "
          f"({len(prompts)} concurrent)")
    return [t for t, _ in out], toks / dt


def solo_latency(srv, prompt, max_tokens, reps):
    best = None
    for _ in range(reps):
        t0 = time.time()
        _, n = srv.chat(prompt, max_tokens)
        dt = time.time() - t0
        r = n / dt
        best = r if best is None else max(best, r)
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--baseline", help="binary without batching, for the ratio")
    ap.add_argument("--model", required=True)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--server-args", default="", help="extra flags for both servers")
    ap.add_argument("--min-speedup", type=float, default=1.5)
    ap.add_argument("--max-latency-regression", type=float, default=0.05)
    args = ap.parse_args()

    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(args.concurrency)]
    extra = args.server_args.split()
    fail = 0

    print(f"candidate: {args.candidate}  ({args.concurrency} slots)")
    with Server(args.candidate, args.model, args.concurrency, args.ctx,
                extra) as srv:
        cand_solo = solo_latency(srv, prompts[0], args.max_tokens, 2)
        print(f"  single-request: {cand_solo:.1f} tok/s")
        cand_texts, cand_tps = measure(srv, prompts, args.max_tokens, "concurrent")

        # identity: the same prompt, alone, on the same server
        print("  identity (concurrent vs solo on the same server):")
        for i, p in enumerate(dict.fromkeys(prompts)):
            solo_text, _ = srv.chat(p, args.max_tokens)
            batched = cand_texts[prompts.index(p)]
            if solo_text == batched:
                print(f"    ok   {p!r}")
            else:
                fail += 1
                print(f"    FAIL {p!r}\n      solo:    {solo_text!r}\n"
                      f"      batched: {batched!r}")

    if args.baseline:
        print(f"baseline: {args.baseline}  ({args.concurrency} independent slots)")
        with Server(args.baseline, args.model, args.concurrency, args.ctx,
                    extra) as srv:
            base_solo = solo_latency(srv, prompts[0], args.max_tokens, 2)
            print(f"  single-request: {base_solo:.1f} tok/s")
            base_texts, base_tps = measure(srv, prompts, args.max_tokens,
                                           "concurrent")

        speedup = cand_tps / base_tps
        lat = cand_solo / base_solo
        print(f"\nthroughput  {base_tps:.1f} -> {cand_tps:.1f} tok/s  = {speedup:.2f}x")
        print(f"single-req  {base_solo:.1f} -> {cand_solo:.1f} tok/s  = {lat:.2f}x")

        # the batched answer must equal the unbatched server's answer too
        for i, (a, b) in enumerate(zip(base_texts, cand_texts)):
            if a != b:
                fail += 1
                print(f"FAIL output differs from baseline for {prompts[i]!r}\n"
                      f"  baseline: {a!r}\n  candidate: {b!r}")
        if speedup < args.min_speedup:
            fail += 1
            print(f"FAIL throughput {speedup:.2f}x below --min-speedup "
                  f"{args.min_speedup}")
        if lat < 1.0 - args.max_latency_regression:
            fail += 1
            print(f"FAIL single-request latency regressed {(1 - lat) * 100:.1f}%")

    print("FAIL" if fail else "PASS")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
