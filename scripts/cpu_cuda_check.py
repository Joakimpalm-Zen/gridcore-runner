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
import re
import subprocess
import sys
import time
import urllib.request
import pathlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# The five short prompts are the historical set (5 to 15 tokens). The four
# long ones were added 2026-08-15 for a specific reason: the TC prefill GEMM
# published uninitialised shared memory for token columns 16 and above from
# 2026-08-08 to 2026-08-13, and this gate ran in every certification through
# that window without catching it, because its longest prompt was FIFTEEN
# tokens. One more token and the bug would have been caught on day one.
#
# A CPU/CUDA identity gate whose prompts all fit in a single sub-tile silently
# exempts every batched prefill path from the contract it claims to enforce.
# These span roughly 24, 40, 64 and 96 tokens so the gate crosses the 16-column
# boundary, the 32-column boundary and a full 64-column tile, and so a future
# tile widening cannot hide behind short prompts either.
PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "1 2 3 4 5 6 7 8",
    "Once upon a time, in a land far away,",
    "Q: What is 17 * 23?\nA:",
    # ~24 tokens: crosses the first tile boundary
    "The lighthouse keeper recorded the weather every morning: wind "
    "direction, cloud cover, and the hour the fog lifted.",
    # ~40 tokens: crosses 32
    "In 1929 the observatory published a revised catalogue listing 4218 "
    "objects, of which roughly one in nine turned out on later inspection "
    "to be a duplicate entry under a second designation. The correction",
    # ~64 tokens: a full tile
    "def parse_header(buf, size):\n"
    "    if size < 8:\n"
    "        raise ValueError('short header')\n"
    "    magic, version = struct.unpack('<II', buf[:8])\n"
    "    if magic != GGUF_MAGIC:\n"
    "        raise ValueError('not a gguf file')\n"
    "    return magic, version\n\n"
    "# The loader above rejects a truncated header before unpacking it, which",
    # ~96 tokens: past one tile, into a second
    "The city of Lisbon sits on seven hills above the Tagus estuary, and its "
    "oldest quarter survived the 1755 earthquake largely intact because the "
    "bedrock there is firmer than the reclaimed ground downriver. Rebuilding "
    "the lower town took decades, and the grid of streets laid out afterwards "
    "was among the first in Europe designed with seismic loads in mind, which "
    "is why the district still reads as unusually regular to a visitor who",
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


def read_split(log_path):
    """The offload split the CUDA run actually achieved, from its own log.

    "5/5 identical" means something different at 24/24 than at 13/24: the
    second only certifies the layers that were on the device. Recording the
    split removes the need to remember which box produced a report.
    """
    try:
        text = pathlib.Path(log_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    hit = re.search(r"gpu-split: .*?G=(\d+)/(\d+).*?full=(\d)", text)
    if not hit:
        return None
    on, total, full = hit.group(1), hit.group(2), hit.group(3)
    return {"layers_on_gpu": int(on), "layers_total": int(total),
            "output_weight_on_gpu": full == "1",
            "whole_graph": int(on) == int(total) and full == "1"}


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
    # 128, not 16: the documented `cpu_cuda` contract has always said 128, and
    # the tool quietly defaulting to 16 meant "certified" asserted less than the
    # matrix claimed. CPU/GPU divergence that only opens up after a few dozen
    # tokens — KV-cache paths, accumulated rounding, sliding-window edges — is
    # precisely what this gate exists to catch, and a 16-token run cannot see it.
    # Owner decision 2026-08-08: raise the tool to the contract and re-certify,
    # publishing whatever un-certifies, rather than lower the contract to the tool.
    ap.add_argument("--tokens", type=int, default=128)
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
    gpu_log = logs / "cpu_cuda-gpu.log"
    print("loading CPU backend...", flush=True)
    cpu = generate_all(args.runner, args.model, "off", args.tokens, args.ctx,
                       env, args.extra_arg, args.timeout, logs / "cpu_cuda-cpu.log")
    print("loading CUDA backend...", flush=True)
    gpu = generate_all(args.runner, args.model, "auto", args.tokens, args.ctx,
                       env, args.extra_arg, args.timeout, gpu_log)

    rows, ok = [], True
    for prompt, c, g in zip(PROMPTS, cpu, gpu):
        same = c == g
        ok = ok and same
        rows.append({"prompt": prompt, "identical": same,
                     "cpu": c, "gpu": None if same else g})
        print(f"{'ok  ' if same else 'FAIL'}: {prompt!r}", flush=True)
        if not same:
            print(f"  cpu: {c!r}\n  gpu: {g!r}", flush=True)

    split = read_split(gpu_log)
    report = {"model": str(args.model), "tokens": args.tokens,
              "context": args.ctx,
              "routing": "fused" if args.fused else "eager",
              "extra_args": args.extra_arg,
              "gpu_split": split,
              "identical": sum(r["identical"] for r in rows),
              "total": len(rows), "rows": rows}
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1))
    print(f"\ncpu_cuda: {report['identical']}/{report['total']} identical "
          f"({report['routing']} routing)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
