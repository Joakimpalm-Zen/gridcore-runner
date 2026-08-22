#!/usr/bin/env python3
"""Greedy tool-calling eval over an eval.jsonl (see make-tooluse-data.py).

Serves the model (optionally with an adapter), sends every eval prompt at
temperature 0, and scores four nested rates: JSON parses / right tool /
args schema-shaped / exact match with the gold call. Prints one JSON line.

Usage: eval-tooluse.py --runner ./runner --model M.gguf \
           [--lora A.gguf [--lora-scale S]] --eval eval.jsonl [--threads N]
"""
import argparse
import json
import signal
import socket
import subprocess
import time
import urllib.request

SCHEMAS = {
    "search_files": {"pattern", "path"},
    "read_file": {"path"},
    "write_file": {"path", "content"},
    "list_dir": {"path"},
    "none": set(),
}


def extract_json(text):
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except Exception:
                    return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runner", default="./runner")
    ap.add_argument("--model", required=True)
    ap.add_argument("--lora")
    ap.add_argument("--lora-scale", type=float, default=0.0)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=64)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.eval) if l.strip()]
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    cmd = [args.runner, "-m", args.model, "--serve", "--port", str(port),
           "--no-tray", "--gpu", "off"]
    if args.threads > 0:
        cmd += ["-t", str(args.threads)]
    if args.lora:
        cmd += ["--lora", args.lora]
        if args.lora_scale > 0:
            cmd += ["--lora-scale", str(args.lora_scale)]
    srv = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
    for _ in range(300):
        try:
            urllib.request.urlopen("http://127.0.0.1:%d/health" % port,
                                   timeout=1)
            break
        except Exception:
            time.sleep(1)
    n = len(rows)
    parses = tool_ok = schema_ok = exact = 0
    for row in rows:
        body = json.dumps({"prompt": row["prompt"],
                           "max_tokens": args.max_tokens,
                           "temperature": 0}).encode()
        req = urllib.request.Request(
            "http://127.0.0.1:%d/v1/completions" % port, data=body,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            text = json.loads(r.read())["choices"][0]["text"]
        obj = extract_json(text)
        gold = row["gold"]
        if obj is None or not isinstance(obj, dict):
            continue
        parses += 1
        if obj.get("tool") != gold["tool"]:
            continue
        tool_ok += 1
        a = obj.get("args")
        want = SCHEMAS.get(gold["tool"], set())
        if not isinstance(a, dict) or set(a.keys()) != want or \
                any(not str(v).strip() for v in a.values()):
            if gold["tool"] != "none" or a not in ({}, None):
                continue
        schema_ok += 1
        if obj == gold:
            exact += 1
    srv.send_signal(signal.SIGTERM)
    srv.wait(timeout=30)
    print(json.dumps({
        "model": args.model.split("/")[-1], "adapter": bool(args.lora),
        "n": n, "json_parses": round(parses / n, 4),
        "right_tool": round(tool_ok / n, 4),
        "schema_args": round(schema_ok / n, 4),
        "exact_call": round(exact / n, 4)}))


if __name__ == "__main__":
    main()
