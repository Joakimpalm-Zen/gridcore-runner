#!/usr/bin/env python3
"""CPU vs CUDA byte-identity gate for one model, over several prompts.

The compatibility program's `cpu_cuda` check: the same greedy generation must
come out byte-for-byte identical whether the forward pass ran on the host or
on the device.

Each backend is loaded ONCE and driven over HTTP, rather than re-launching the
CLI per prompt. That is not a style preference: on a box whose RAM is smaller
than the model, alternating two processes over the same mmap evicts the page
cache on every switch, so a ten-launch run re-reads tens of gigabytes and
measures the disk instead of the engine.

MoE byte identity is defined over the EAGER routing path
(`RUNNER_MOE_EAGER=1`), because the fused path's device `expf` differs from the
host libm by 1-2 ulp by construction — so this harness pins it, exactly as the
other certification harnesses do. `RUNNER_CUDA_TC=0` pins the scalar GEMM for
the same reason (the tensor-core default is tolerance-gated, not identical).

Usage: cpu_cuda_check.py MODEL [--tokens N] [--runner PATH] [--fused]
Exit status is 0 only if every prompt matched.
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "1 2 3 4 5 6 7 8",
    "Once upon a time, in a land far away,",
    "Q: What is 17 * 23?\nA:",
]


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def post(url, body, timeout):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def wait_ready(base, process, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise SystemExit(f"server exited during startup (rc={process.returncode})")
        try:
            urllib.request.urlopen(base + "/health", timeout=2).read()
            return
        except Exception:
            time.sleep(0.5)
    raise SystemExit("server startup timed out")


def generate_all(runner, model, gpu, tokens, ctx, env, extra, timeout, log_path):
    """Load the model once on `gpu`, return the greedy text for every prompt."""
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    cmd = [str(runner), "-m", str(model), "--serve", "--port", str(port),
           "-c", str(ctx), "--gpu", gpu, *extra]
    log = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    try:
        wait_ready(base, proc, timeout)
        out = []
        for prompt in PROMPTS:
            body = {"prompt": prompt, "max_tokens": tokens,
                    "temperature": 0, "top_p": 1, "stream": False}
            resp = post(base + "/v1/completions", body, timeout)
            out.append(resp["choices"][0]["text"])
            print(f"  [{gpu}] {prompt!r} -> {out[-1][:48]!r}", flush=True)
        return out
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        log.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--runner", default=str(ROOT / "runner"))
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--fused", action="store_true",
                    help="do NOT pin the eager router (measures the fused "
                         "default, whose contract is weaker than identity)")
    ap.add_argument("--extra-arg", action="append", default=[],
                    help="extra flag passed to BOTH runs (e.g. --cpu-moe)")
    ap.add_argument("--out")
    args = ap.parse_args()

    env = dict(os.environ)
    env["RUNNER_CUDA_TC"] = "0"      # identity evidence is over the scalar path
    if not args.fused:
        env["RUNNER_MOE_EAGER"] = "1"

    logs = Path(args.out).parent if args.out else Path(".")
    print("loading CPU backend...", flush=True)
    cpu = generate_all(args.runner, args.model, "off", args.tokens, args.ctx,
                       env, args.extra_arg, args.timeout, logs / "cpu_cuda-cpu.log")
    print("loading CUDA backend...", flush=True)
    gpu = generate_all(args.runner, args.model, "auto", args.tokens, args.ctx,
                       env, args.extra_arg, args.timeout, logs / "cpu_cuda-gpu.log")

    rows, ok = [], True
    for prompt, c, g in zip(PROMPTS, cpu, gpu):
        same = c == g
        ok = ok and same
        rows.append({"prompt": prompt, "identical": same,
                     "cpu": c, "gpu": None if same else g})
        print(f"{'ok  ' if same else 'FAIL'}: {prompt!r}", flush=True)
        if not same:
            print(f"  cpu: {c!r}\n  gpu: {g!r}", flush=True)

    report = {"model": str(args.model), "tokens": args.tokens,
              "context": args.ctx,
              "routing": "fused" if args.fused else "eager",
              "extra_args": args.extra_arg,
              "identical": sum(r["identical"] for r in rows),
              "total": len(rows), "rows": rows}
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1))
    print(f"\ncpu_cuda: {report['identical']}/{report['total']} identical "
          f"({report['routing']} routing)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
