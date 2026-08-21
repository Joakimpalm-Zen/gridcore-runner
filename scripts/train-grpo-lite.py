#!/usr/bin/env python3
"""GRPO-lite: reinforcement fine-tuning with zero train/infer mismatch (D6).

The loop: sample K completions per prompt FROM THE RUNNER ITSELF, score each
with a task reward, turn group-relative advantages into weighted training
examples, and take one weighted supervised pass (`--train` on a .jsonl).
REINFORCE at example granularity — deliberately the simplest correct member
of the GRPO family, because the demonstration is the MECHANISM: the policy
that samples is the policy that trains, same binary, same kernels, same
bits. Sampling is seeded per request and training is deterministic, so an
entire run replays exactly from its config.

Built-in task `tool-call`: the model must answer with ONLY a JSON object
{"tool": "<expected>", "args": {"pattern": "..."}}. Reward 1.0 for a parse
with the right tool and a non-empty pattern, 0.3 for valid JSON with the
wrong shape, 0 otherwise.

    train-grpo-lite.py --runner ./runner --model M.gguf \
        --rounds 5 --k 8 --lr 5e-5 --out-dir /tmp/grpo
"""
import argparse
import json
import os
import pathlib
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request

PROMPTS = [
    ("search_files", "find all Python files in the src directory"),
    ("search_files", "locate every log file under /var/log"),
    ("read_file", "open the README file and show it"),
    ("read_file", "display the contents of config.yaml"),
]


def instruction(tool, request):
    return ("You are a tool-calling assistant. Reply with ONLY a JSON "
            "object of the form {\"tool\": \"%s\", \"args\": {\"pattern\": "
            "\"<pattern>\"}} and nothing else.\nRequest: %s\nJSON:" %
            (tool, request))


def reward(tool, text):
    text = text.strip()
    start = text.find("{")
    if start < 0:
        return 0.0
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                text = text[start:i + 1]
                break
    try:
        obj = json.loads(text)
    except Exception:
        return 0.0
    if not isinstance(obj, dict):
        return 0.0
    if obj.get("tool") != tool:
        return 0.3
    args = obj.get("args")
    if not isinstance(args, dict) or not str(args.get("pattern", "")).strip():
        return 0.3
    return 1.0


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def sample(port, prompt, seed, max_tokens, temperature):
    body = json.dumps({"prompt": prompt, "max_tokens": max_tokens,
                       "temperature": temperature, "seed": seed}).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:%d/v1/completions" % port, data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["choices"][0]["text"]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runner", default="./runner")
    ap.add_argument("--model", required=True)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out-dir", default="grpo-out")
    args = ap.parse_args()

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    adapter = None
    tflag = ["-t", str(args.threads)] if args.threads > 0 else []
    history = []

    for rnd in range(args.rounds + 1):   # round 0 = measure-only baseline
        # ---- serve the CURRENT policy and sample K per prompt
        port = free_port()
        cmd = [args.runner, "-m", args.model, "--serve", "--port", str(port),
               "--no-tray", "--gpu", "off", *tflag]
        if adapter:
            cmd += ["--lora", str(adapter)]
        srv = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
        for _ in range(300):
            try:
                urllib.request.urlopen(
                    "http://127.0.0.1:%d/health" % port, timeout=1)
                break
            except Exception:
                if srv.poll() is not None:
                    sys.exit("server died at round %d" % rnd)
                time.sleep(1)
        examples, rewards = [], []
        for pi, (tool, request) in enumerate(PROMPTS):
            prompt = instruction(tool, request)
            group = []
            for ki in range(args.k):
                seed = 100000 + rnd * 1000 + pi * 100 + ki
                text = sample(port, prompt, seed, args.max_tokens,
                              args.temperature)
                group.append((text, reward(tool, text)))
            mean_r = sum(r for _, r in group) / len(group)
            var = sum((r - mean_r) ** 2 for _, r in group) / len(group)
            std = var ** 0.5
            rewards += [r for _, r in group]
            for text, r in group:
                # GRPO-standard advantage: group-normalized and clipped.
                # The raw (r - mean) form was measured to COLLAPSE the
                # policy on a high-baseline task (0.91 -> 0.09 by round 5):
                # with mean ~0.9 the rare failures carry -0.9 while
                # successes carry +0.1, and the asymmetric negative mass
                # unlearns the shared structure. Normalizing by the group
                # std and clipping to +-1 restores the symmetric update.
                a = (r - mean_r) / (std + 1e-4)
                a = max(-1.0, min(1.0, a))
                if a != 0.0:
                    examples.append({"prompt": prompt, "completion": text,
                                     "weight": round(a, 6)})
        srv.send_signal(signal.SIGTERM)
        srv.wait(timeout=30)
        rate = sum(1 for r in rewards if r >= 1.0) / len(rewards)
        mean = sum(rewards) / len(rewards)
        history.append({"round": rnd, "valid_rate": rate,
                        "mean_reward": round(mean, 4),
                        "trainable": len(examples)})
        print(json.dumps(history[-1]), flush=True)
        if rnd == args.rounds:
            break
        if not examples:
            print(json.dumps({"round": rnd,
                              "note": "no advantage signal, skipping step"}),
                  flush=True)
            continue
        # ---- one weighted pass over this round's examples
        data = out / ("round%d.jsonl" % rnd)
        data.write_text("\n".join(json.dumps(e) for e in examples) + "\n")
        next_adapter = out / ("adapter-r%d.gguf" % (rnd + 1))
        cmd = [args.runner, "-m", args.model, "--train", str(data),
               "--train-steps", str(len(examples)), "--lr", str(args.lr),
               "--train-out", str(next_adapter), *tflag]
        if adapter:
            cmd += ["--lora", str(adapter)]
        else:
            cmd += ["--lora-rank", str(args.rank)]
        p = subprocess.run(cmd, stdout=subprocess.DEVNULL,
                           stderr=subprocess.PIPE, timeout=7200)
        if p.returncode != 0:
            sys.exit("train step failed:\n" +
                     p.stderr.decode(errors="replace")[-1500:])
        adapter = next_adapter

    print(json.dumps({"summary": history}), flush=True)
    if adapter:
        shutil.copy(adapter, out / "adapter-final.gguf")


if __name__ == "__main__":
    main()
