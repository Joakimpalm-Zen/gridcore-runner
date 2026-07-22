#!/usr/bin/env python3
"""Compare greedy raw completions through Runner and llama.cpp's shared API.

Using `/v1/completions` removes CLI banners, ANSI output, prompt echo and chat
template differences. Exact generated UTF-8 text is compared at temperature 0.
"""

import argparse
import json
from pathlib import Path
import socket
import subprocess
import time
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "1 2 3 4 5 6 7 8",
    "Once upon a time, in a land far away,",
    'JSON example: {"name":',
]


def completion_text(response):
    try:
        text = response["choices"][0]["text"]
    except (KeyError, IndexError, TypeError):
        raise ValueError("response has no completion text") from None
    if not isinstance(text, str):
        raise ValueError("completion text is not a string")
    return text


def command_version(command):
    process = subprocess.run([str(command), "--version"], text=True,
                             capture_output=True, timeout=15)
    lines = (process.stdout + process.stderr).strip().splitlines()
    return lines[0] if lines else None


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def request_json(url, body=None, timeout=10):
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(url, data=data,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def wait_ready(base, process, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("server exited during startup")
        try:
            request_json(base + "/health", timeout=1)
            return
        except (OSError, urllib.error.HTTPError, json.JSONDecodeError):
            time.sleep(.1)
    raise RuntimeError("server startup timed out")


def serve(command, log_path, startup_timeout):
    log = log_path.open("w")
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
    try:
        port = int(command[command.index("--port") + 1])
        base = f"http://127.0.0.1:{port}"
        wait_ready(base, process, startup_timeout)
        return process, log, base
    except Exception:
        process.terminate()
        process.wait(timeout=10)
        log.close()
        raise


def stop(process, log):
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    log.close()


def collect(command, log_path, prompts, tokens, startup_timeout, request_timeout):
    process, log, base = serve(command, log_path, startup_timeout)
    try:
        outputs = []
        for prompt in prompts:
            response = request_json(base + "/v1/completions", {
                "model": "compat", "prompt": prompt, "max_tokens": tokens,
                "temperature": 0, "stream": False,
            }, timeout=request_timeout)
            outputs.append(completion_text(response))
        return outputs
    finally:
        stop(process, log)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=Path, default=ROOT / "runner")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--ctx", type=int, default=2048)
    parser.add_argument("--startup-timeout", type=int, default=300)
    parser.add_argument("--request-timeout", type=int, default=300)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    port = free_port()
    work = args.out.parent if args.out else ROOT / "tests/compatibility/out"
    work.mkdir(parents=True, exist_ok=True)
    runner_command = [str(args.runner.resolve()), "-m", str(args.model.resolve()),
                      "--serve", "--port", str(port), "-c", str(args.ctx),
                      "--gpu", "off"]
    runner_outputs = collect(runner_command, work / "reference-runner.log",
                             PROMPTS, args.tokens, args.startup_timeout,
                             args.request_timeout)

    port = free_port()
    reference_command = [str(args.reference.resolve()), "-m", str(args.model.resolve()),
                         "--host", "127.0.0.1", "--port", str(port),
                         "-c", str(args.ctx), "-ngl", "0"]
    reference_outputs = collect(reference_command, work / "reference-llama.log",
                                PROMPTS, args.tokens, args.startup_timeout,
                                args.request_timeout)

    comparisons = []
    for prompt, runner_text, reference_text in zip(
            PROMPTS, runner_outputs, reference_outputs):
        comparisons.append({"prompt": prompt, "runner": runner_text,
                            "reference": reference_text,
                            "status": "pass" if runner_text == reference_text else "fail"})
    report = {
        "schema_version": "gridcore.runner.greedy-reference.v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runner": command_version(args.runner),
        "reference": command_version(args.reference),
        "model": str(args.model), "tokens": args.tokens,
        "comparison": "exact generated UTF-8 text from /v1/completions at temperature 0",
        "comparisons": comparisons,
        "totals": {"passed": sum(x["status"] == "pass" for x in comparisons),
                   "failed": sum(x["status"] == "fail" for x in comparisons)},
    }
    rendered = json.dumps(report, indent=2) + "\n"
    if args.out:
        args.out.write_text(rendered)
    print(rendered, end="")
    return 1 if report["totals"]["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
