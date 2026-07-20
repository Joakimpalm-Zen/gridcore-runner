#!/usr/bin/env python3
"""Kernel throughput gate: prefill + decode tok/s for one binary + model.

Runs a deterministic long-prompt greedy completion and reports the numbers
the CUDA throughput project must move. JSON on stdout so agents and CI can
diff runs mechanically.

    python scripts/kernel-bench.py --runner runner.exe \
        --model C:/ProjectGrid/models/Qwen_Qwen3-4B-Q8_0.gguf
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

STATS_RE = re.compile(
    r"prompt:\s*(\d+)\s*tok,\s*([\d.]+)\s*tok/s\s*\|\s*gen:\s*(\d+)\s*tok,\s*([\d.]+)\s*tok/s")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", default=str(Path(__file__).resolve().parents[1] / "runner.exe"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-words", type=int, default=400,
                        help="itemNNNN words tokenize ~5:1 -> ~2k tokens")
    parser.add_argument("--gen", type=int, default=64)
    parser.add_argument("--ctx", type=int, default=4096)
    parser.add_argument("--batch", type=int, default=0, help="runner -b override")
    parser.add_argument("--kv", choices=("f16", "q8"),
                        help="KV cache storage to benchmark (default: runner's)")
    args = parser.parse_args()

    prompt = " ".join(f"item{i:04d}" for i in range(args.prompt_words))
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False,
                                     encoding="utf-8") as handle:
        handle.write(prompt)
        prompt_file = handle.name
    command = [str(Path(args.runner).resolve()), "-m", args.model, "-f", prompt_file,
               "-n", str(args.gen), "-c", str(args.ctx), "--temp", "0"]
    if args.batch:
        command += ["-b", str(args.batch)]
    if args.kv:
        command += ["--kv", args.kv]
    started = time.time()
    proc = subprocess.run(command, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=1800)
    wall = time.time() - started
    Path(prompt_file).unlink(missing_ok=True)
    match = None
    for line in (proc.stderr or "").splitlines():
        found = STATS_RE.search(line)
        if found:
            match = found
    if match is None:
        print(json.dumps({"error": "no stats line",
                          "stderr_tail": (proc.stderr or "")[-400:]}))
        return 1
    result = {
        "model": Path(args.model).name,
        "runner": str(Path(args.runner)),
        "kv": args.kv or "default",
        "prompt_tokens": int(match.group(1)),
        "prefill_tok_s": float(match.group(2)),
        "gen_tokens": int(match.group(3)),
        "decode_tok_s": float(match.group(4)),
        "wall_s": round(wall, 1),
    }
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
