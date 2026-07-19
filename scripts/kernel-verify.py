#!/usr/bin/env python3
"""Kernel correctness gate: two binaries must produce token-identical greedy
output on the same model + prompts. Run it between every kernel change and
the last known-good binary — a faster wrong kernel is a regression.

    python scripts/kernel-verify.py --baseline runner-baseline.exe \
        --candidate runner.exe --model test-q8.gguf
Exit 0 = identical on every prompt; 1 = divergence (diff shown).
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "1 2 3 4 5 6 7 8",
    "Once upon a time, in a land far away,",
    "JSON example: {\"name\":",
]


def generate(binary: str, model: str, prompt: str, n: int, ctx: int) -> str:
    proc = subprocess.run(
        [str(Path(binary).resolve()), "-m", model, "-p", prompt,
         "-n", str(n), "-c", str(ctx), "--temp", "0"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=900)
    if proc.returncode != 0:
        raise RuntimeError(f"{binary} failed: {(proc.stderr or '')[-300:]}")
    return proc.stdout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--gen", type=int, default=48)
    parser.add_argument("--ctx", type=int, default=2048)
    args = parser.parse_args()

    failures = 0
    for prompt in PROMPTS:
        base = generate(args.baseline, args.model, prompt, args.gen, args.ctx)
        cand = generate(args.candidate, args.model, prompt, args.gen, args.ctx)
        if base == cand:
            print(f"ok   {prompt[:40]!r}")
        else:
            failures += 1
            print(f"DIFF {prompt[:40]!r}")
            print(f"  baseline : {base[:160]!r}")
            print(f"  candidate: {cand[:160]!r}")
    model_name = Path(args.model).name
    if failures:
        print(f"FAIL: {failures}/{len(PROMPTS)} prompts diverge on {model_name}")
        return 1
    print(f"PASS: token-identical on {len(PROMPTS)} prompts ({model_name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
