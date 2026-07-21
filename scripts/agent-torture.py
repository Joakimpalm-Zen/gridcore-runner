#!/usr/bin/env python3
"""The public agent torture suite driver.

Executes one repeatable, adversarial request matrix and preserves enough
evidence to audit every verdict. It runs against Runner (spawned locally) OR
any OpenAI-compatible endpoint — llama.cpp's server, Ollama, vLLM — so the
same matrix produces comparable, runtime-labeled reports on identical
hardware. The verdicts (valid tool call, correct tool selection, schema
conformance, transport-invariant streaming) are the same across runtimes;
only the target changes.

  # Runner (default): spawn it and run the matrix on the CPU
  agent-torture.py --model test.gguf

  # Any OpenAI-compatible server already listening locally:
  agent-torture.py --endpoint 127.0.0.1:8080 --runtime llama.cpp \\
      --runtime-version b3200 --model qwen2.5-7b
  agent-torture.py --endpoint 127.0.0.1:11434 --runtime ollama --model qwen2.5
  agent-torture.py --endpoint 127.0.0.1:8000  --runtime vllm --model ...
"""

import argparse
import base64
from collections import Counter
import json
from pathlib import Path
import socket
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "conformance"))

from harness import (Client, ProtocolError, RunnerServer,  # noqa: E402
                     categorize, decode_events, find_runner, parse_stream,
                     rss_kind, validate_against_schema)

SCHEMA_VERSION = "gridcore.agent-torture.v1"


class RemoteTarget:
    """A stand-in the harness Client can drive like a spawned RunnerServer, but
    pointing at an already-running local OpenAI-compatible server (Runner,
    llama.cpp, Ollama, vLLM). The Client only needs ``port`` + ``assert_alive``
    + ``sample_rss`` from its target, so this is all it takes to run the whole
    matrix against any of them. Localhost only: the runtimes under comparison
    run on the same box (identical-hardware is the point); a remote host wants
    an SSH tunnel to a local port."""

    def __init__(self, port: int):
        self.port = int(port)
        self.peak_rss_kb = None   # cannot read a foreign process's RSS

    def assert_alive(self):
        with socket.create_connection(("127.0.0.1", self.port), timeout=2):
            pass

    def sample_rss(self):
        return None

    def __enter__(self):
        self.assert_alive()
        return self

    def __exit__(self, *exc):
        return False


def _parse_endpoint(value: str) -> int:
    """Accept host:port or a full URL; return the local port. Non-loopback
    hosts are rejected — tunnel them to a local port first."""
    text = value.strip()
    for prefix in ("http://", "https://"):
        if text.startswith(prefix):
            text = text[len(prefix):]
    text = text.rstrip("/")
    host, _, port = text.rpartition(":")
    host = host or "127.0.0.1"
    if host not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError(f"endpoint must be loopback (got {host!r}); "
                         "tunnel a remote runtime to a local port")
    if not port.isdigit():
        raise ValueError(f"endpoint needs a port (got {value!r})")
    return int(port)

NESTED_SCHEMA = {
    "type": "object",
    "properties": {
        "job": {
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": ["fast", "safe"]},
                "targets": {
                    "type": "array", "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "retries": {"type": "integer"},
                        },
                        "required": ["path", "retries"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["mode", "targets"],
            "additionalProperties": False,
        },
        "notify": {"type": "boolean"},
    },
    "required": ["job", "notify"],
    "additionalProperties": False,
}


def _tool(name, schema, description="deterministic torture-suite tool"):
    return {"type": "function", "function": {
        "name": name, "description": description, "parameters": schema}}


NESTED_TOOL = _tool("dispatch_job", NESTED_SCHEMA)
SELECT_TOOLS = [
    _tool("lookup_weather", {"type": "object", "properties": {
        "city": {"type": "string"}}, "required": ["city"]}),
    _tool("sum_values", {"type": "object", "properties": {
        "values": {"type": "array", "minItems": 1,
                   "items": {"type": "integer"}}}, "required": ["values"]}),
]


def build_cases(count=100):
    """Return the stable public request matrix (round-robin by category)."""
    if count < 1:
        raise ValueError("count must be positive")
    cases = []
    categories = ("nested_arguments", "tool_selection", "forced_truncation",
                  "stream_normalization")
    ordinals = Counter()
    for index in range(count):
        category = categories[index % len(categories)]
        ordinal = ordinals[category]
        ordinals[category] += 1
        base = {
            "messages": [{"role": "user", "content":
                          f"torture case {index:03d}; follow the selected contract"}],
            "temperature": 0,
        }
        if category == "nested_arguments":
            payload = dict(base, max_tokens=96, tools=[NESTED_TOOL],
                           tool_choice={"type": "function", "function": {
                               "name": "dispatch_job"}})
        elif category == "tool_selection":
            wanted = SELECT_TOOLS[ordinal % len(SELECT_TOOLS)]["function"]["name"]
            payload = dict(base, max_tokens=64, tools=SELECT_TOOLS,
                           tool_choice={"type": "function", "function": {
                               "name": wanted}})
        elif category == "forced_truncation":
            payload = dict(base, max_tokens=(1, 2, 3, 5, 8)[ordinal % 5],
                           tools=[NESTED_TOOL], tool_choice="required")
        else:
            payload = dict(base, max_tokens=4 + ordinal % 5, stream=True)
        cases.append({"id": f"runner-{index:03d}-{category}",
                      "ordinal": index, "category": category,
                      "request": payload})
    return cases


def normalize_sse(raw, chunks=None):
    """Normalize a chat SSE stream using the conformance parser."""
    events = parse_stream(raw, chunks)
    decoded, saw_done = decode_events(events)
    text = []
    finish = None
    for event in decoded:
        for choice in event.get("choices", []):
            delta = choice.get("delta") or {}
            text.append(delta.get("content") or choice.get("text") or "")
            finish = choice.get("finish_reason") or finish
    return {"events": decoded, "saw_done": saw_done,
            "text": "".join(text), "finish_reason": finish}


def _only_tool(response):
    calls = (response.choice.get("message") or {}).get("tool_calls")
    if not isinstance(calls, list) or len(calls) != 1:
        raise ProtocolError("expected exactly one tool call", got=calls)
    function = calls[0].get("function") or {}
    arguments = function.get("arguments")
    if not isinstance(arguments, str):
        raise ProtocolError("tool arguments are not a JSON string")
    try:
        parsed = json.loads(arguments)
    except ValueError as exc:
        raise ProtocolError("tool arguments are invalid JSON",
                            arguments=arguments[:200]) from exc
    return function.get("name"), parsed


def _verify_buffered(case, response):
    response.expect_status(200)
    name, arguments = _only_tool(response)
    if case["category"] in ("nested_arguments", "forced_truncation"):
        if name != "dispatch_job":
            raise ProtocolError("wrong tool selected", expected="dispatch_job",
                                got=name)
        validate_against_schema(arguments, NESTED_SCHEMA)
    else:
        wanted = case["request"]["tool_choice"]["function"]["name"]
        if name != wanted:
            raise ProtocolError("wrong tool selected", expected=wanted, got=name)
        schema = next(t["function"]["parameters"] for t in SELECT_TOOLS
                      if t["function"]["name"] == wanted)
        validate_against_schema(arguments, schema)


def _verify_stream(stream):
    stream.expect_sse()
    reference = normalize_sse(stream.raw, [stream.raw])
    # Three deterministic transport segmentations catch normalization errors
    # without turning each request into an O(n) boundary benchmark.
    points = (0, len(stream.raw) // 2, len(stream.raw))
    for point in points:
        if normalize_sse(stream.raw, [stream.raw[:point], stream.raw[point:]]) != reference:
            raise ProtocolError("SSE normalization depends on transport chunks",
                                split_at=point)
    if not reference["saw_done"]:
        raise ProtocolError("SSE stream omitted [DONE]")
    if reference["finish_reason"] not in ("stop", "length"):
        raise ProtocolError("SSE stream has no usable finish reason",
                            got=reference["finish_reason"])
    return reference


class _MetricsSink:
    def __init__(self):
        self.requests = []

    def record_request(self, record):
        self.requests.append(record)


def result_for(case, status, latency_ms, failure=None):
    result = {"id": case["id"], "ordinal": case["ordinal"],
              "category": case["category"], "status": status,
              "latency_ms": latency_ms}
    if failure:
        result["failure"] = failure
    return result


def make_report(results, runtime_name, version, model, elapsed_ms, peak_kb):
    failures = Counter(r["failure"]["category"] for r in results
                       if r["status"] == "failed")
    passed = sum(r["status"] == "passed" for r in results)
    seconds = max(elapsed_ms / 1000, 1e-9)
    return {
        "schema_version": SCHEMA_VERSION,
        "runtime": {"name": runtime_name, "version": version},
        "configuration": {"model": model, "temperature": 0},
        "totals": {"requests": len(results), "passed": passed,
                   "failed": len(results) - passed,
                   "failures_by_category": dict(sorted(failures.items()))},
        "metrics": {"elapsed_ms": elapsed_ms,
                    "valid_structured_tasks_per_second": round(passed / seconds, 3)},
        "resources": {"peak_rss_kb": peak_kb, "peak_rss_kind": rss_kind()},
        "cases": results,
    }


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _version(exe):
    proc = subprocess.run([str(exe), "--version"], text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          check=False)
    return proc.stdout.strip() or f"exit {proc.returncode}"


def run(target, runtime_name, version, model_label, out, count,
        request_model=None):
    """Run the matrix against ``target`` (a spawned RunnerServer or a
    RemoteTarget) and write report.json + raw.jsonl. Runtime-labeled so
    reports from different runtimes compare directly. ``request_model`` sets
    the OpenAI ``model`` field on every request — Runner and llama.cpp serve
    one model and ignore it, but Ollama routes on it, so it is required there.
    It is recorded verbatim in raw.jsonl, so the request stays reproducible."""
    out.mkdir(parents=True, exist_ok=True)
    sink = _MetricsSink()
    cases = build_cases(count)
    if request_model:
        for case in cases:
            case["request"] = dict(case["request"], model=request_model)
    results = []
    artifacts = []
    started = time.monotonic()
    with target as server:
        client = Client(server, sink)
        for case in cases:
            t0 = time.monotonic()
            artifact = {"schema_version": SCHEMA_VERSION, "id": case["id"],
                        "category": case["category"], "request": case["request"]}
            failure = None
            try:
                if case["category"] == "stream_normalization":
                    stream = client.chat_stream(case["request"], name=case["id"])
                    artifact["response"] = {
                        "encoding": "base64", "media_type": "text/event-stream",
                        "body": base64.b64encode(stream.raw).decode("ascii")}
                    artifact["normalized"] = _verify_stream(stream)
                else:
                    response = client.chat(case["request"], name=case["id"])
                    artifact["response"] = {
                        "status": response.status, "headers": response.headers,
                        "encoding": "base64", "media_type":
                        response.headers.get("content-type", ""),
                        "body": base64.b64encode(response.body).decode("ascii")}
                    _verify_buffered(case, response)
            except Exception as exc:  # verdicts belong in the report, not traceback-only
                failure = {"category": categorize(exc), "message": str(exc)}
                artifact["failure"] = failure
            latency = round((time.monotonic() - t0) * 1000, 2)
            results.append(result_for(case, "failed" if failure else "passed",
                                      latency, failure))
            artifacts.append(artifact)
    elapsed = round((time.monotonic() - started) * 1000, 2)
    report = make_report(results, runtime_name, version, model_label, elapsed,
                         getattr(target, "peak_rss_kb", None))
    write_json(out / "report.json", report)
    with (out / "raw.jsonl").open("w") as raw:
        for artifact in artifacts:
            raw.write(json.dumps(artifact, sort_keys=True) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--endpoint",
                        help="target an already-running OpenAI-compatible "
                             "server (host:port or URL) instead of spawning "
                             "Runner; use with --runtime")
    parser.add_argument("--runtime",
                        help="runtime label for the report (e.g. llama.cpp, "
                             "ollama, vllm). Defaults to 'runner'.")
    parser.add_argument("--runtime-version", default="unknown",
                        help="version string for the report when --endpoint "
                             "is used")
    parser.add_argument("--runner", type=Path)
    parser.add_argument("--model-name", dest="model_name",
                        help="OpenAI `model` field to set on every request "
                             "(required for Ollama, which routes on it; Runner "
                             "and llama.cpp ignore it)")
    parser.add_argument("--model", type=Path, default=ROOT / "test.gguf")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "tests" / "torture" / "out")
    parser.add_argument("--cases", type=int, default=100)
    args = parser.parse_args(argv)

    if args.endpoint:
        try:
            port = _parse_endpoint(args.endpoint)
        except ValueError as exc:
            parser.error(str(exc))
        target = RemoteTarget(port)
        runtime_name = args.runtime or "openai-compatible"
        version = args.runtime_version
        model_label = str(args.model)
    else:
        exe = args.runner or Path(find_runner(str(ROOT)))
        if not args.model.is_file():
            parser.error(f"model not found: {args.model}")
        target = RunnerServer(str(exe), str(args.model), ctx=1024, parallel=2,
                              extra_args=["--gpu", "off"],
                              log_path=str(args.out / "runner.log"))
        args.out.mkdir(parents=True, exist_ok=True)
        runtime_name = args.runtime or "runner"
        version = _version(exe)
        model_label = str(args.model)

    report = run(target, runtime_name, version, model_label, args.out,
                 args.cases, request_model=args.model_name)
    print(f"report: {args.out / 'report.json'}")
    print(f"raw: {args.out / 'raw.jsonl'}")
    print(f"runtime={report['runtime']['name']} "
          f"requests={report['totals']['requests']} "
          f"passed={report['totals']['passed']} failed={report['totals']['failed']}")
    return 1 if report["totals"]["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
