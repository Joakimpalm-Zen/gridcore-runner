#!/usr/bin/env python3
"""Exercise real consumer libraries against one live Runner process."""

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request


ROOT = Path(__file__).resolve().parents[1]


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_ready(base, proc):
    for _ in range(100):
        if proc.poll() is not None:
            raise RuntimeError("runner exited before becoming ready")
        try:
            with urllib.request.urlopen(base + "/health", timeout=.2) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(.05)
    raise RuntimeError("runner did not become ready")


def check(name, fn, results):
    started = time.monotonic()
    try:
        fn()
        status, error = "pass", None
    except Exception as exc:
        status, error = "fail", f"{type(exc).__name__}: {exc}"
    results.append({"id": name, "status": status,
                    "elapsed_ms": round((time.monotonic() - started) * 1000, 1),
                    "error": error})


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=Path, default=ROOT / "runner")
    parser.add_argument("--model", type=Path, default=ROOT / "test.gguf")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "tests/compatibility/out/consumers.json")
    parser.add_argument("--node", default="node")
    args = parser.parse_args(argv)

    import anthropic
    import litellm
    import openai
    from langchain_openai import ChatOpenAI

    port = free_port()
    base = f"http://127.0.0.1:{port}"
    log = args.out.with_suffix(".runner.log")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as output:
        proc = subprocess.Popen(
            [str(args.runner.resolve()), "-m", str(args.model.resolve()),
             "--serve", "--port", str(port), "--parallel", "2", "--gpu", "off"],
            stdout=output, stderr=subprocess.STDOUT)
        try:
            wait_ready(base, proc)
            results = []
            py_openai = openai.OpenAI(base_url=base + "/v1", api_key="compat-test")

            check("openai-python-chat", lambda: py_openai.chat.completions.create(
                model="runner", messages=[{"role": "user", "content": "Say OK"}],
                max_tokens=4, temperature=0), results)
            check("openai-python-stream", lambda: list(
                py_openai.chat.completions.create(
                    model="runner", messages=[{"role": "user", "content": "Say OK"}],
                    max_tokens=4, temperature=0, stream=True)), results)
            check("openai-python-responses", lambda: py_openai.responses.create(
                model="runner", input="Say OK", max_output_tokens=4), results)

            py_anthropic = anthropic.Anthropic(base_url=base, api_key="compat-test")
            check("anthropic-python", lambda: py_anthropic.messages.create(
                model="runner", max_tokens=4,
                messages=[{"role": "user", "content": "Say OK"}]), results)

            check("litellm", lambda: litellm.completion(
                model="openai/runner", api_base=base + "/v1",
                api_key="compat-test", messages=[{"role": "user", "content": "Say OK"}],
                max_tokens=4, temperature=0), results)

            chain = ChatOpenAI(model="runner", base_url=base + "/v1",
                               api_key="compat-test", max_tokens=4, temperature=0)
            check("langchain-openai", lambda: chain.invoke("Say OK"), results)

            node_script = ROOT / "tests/compatibility/node/smoke.mjs"
            def node_check():
                subprocess.run([args.node, str(node_script), base], check=True,
                               text=True, capture_output=True, timeout=60)
            check("openai-and-anthropic-node", node_check, results)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

    packages = {}
    for package in ("openai", "anthropic", "litellm", "langchain-openai"):
        packages[package] = version(package)
    npm = subprocess.run(["npm", "list", "--json", "--depth=0"],
                         cwd=ROOT / "tests/compatibility/node", text=True,
                         capture_output=True)
    try:
        node_packages = {k: v.get("version") for k, v in
                         json.loads(npm.stdout).get("dependencies", {}).items()}
    except json.JSONDecodeError:
        node_packages = {}
    report = {
        "schema_version": "gridcore.runner.consumer-compat-report.v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runner": subprocess.run([str(args.runner), "--version"], text=True,
                                 capture_output=True).stdout.strip(),
        "python": sys.version.split()[0], "packages": packages,
        "node_packages": node_packages, "results": results,
        "totals": {"passed": sum(x["status"] == "pass" for x in results),
                   "failed": sum(x["status"] == "fail" for x in results)},
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 1 if report["totals"]["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
