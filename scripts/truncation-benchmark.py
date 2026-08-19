#!/usr/bin/env python3
"""Truncation-recovery benchmark: does a tool call survive the token limit?

Runner's headline property is that a tool call constrained by a JSON-schema
grammar stays a *parseable tool call* even when the generation budget runs out
mid-object. The closer force-closes the constrained JSON, so the client always
sees a well-formed ``tool_calls`` entry with ``finish_reason: "length"`` rather
than a half-written string or protocol framing leaking into ``content``.

This probe pins that property to an executable, someone-else-runnable artifact.
It drives a fixed tool schema and prompt across a token ladder
(1, 2, 3, 5, 8, 16, 64) and records, per rung: the client-visible
``tool_calls`` (present? parseable JSON?), the ``finish_reason``, and any
``content`` leakage.

It runs against Runner (spawned locally, CPU is enough — the property is an
engine guarantee, not a model-quality one, so the tiny CI fixture exhibits it)
OR against any already-running OpenAI-compatible server via ``--endpoint`` so a
competitor column (vLLM, llama.cpp, ...) is measured on the same box with the
same schema, prompt, and budgets.

  # Runner (default): spawn it on the CPU and assert the property (the gate)
  truncation-benchmark.py --model test.gguf --assert

  # Refresh a competitor column against a server already listening locally:
  truncation-benchmark.py --endpoint 127.0.0.1:8000 --runtime vllm \\
      --runtime-version 0.27.1 --model-name granite \\
      --out tests/torture/truncation/<date>-<model>/vllm

The token ladder, tool schema, prompt, and generation settings live here as
data so the two engines are asked exactly the same question. Deterministic,
temperature 0.
"""

import argparse
import base64
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "conformance"))

from harness import Client, RunnerServer, find_runner  # noqa: E402

SCHEMA_VERSION = "xyntetik.truncation-benchmark.v1"

# The token ladder. Below 64 the budget is too small to finish the object, so a
# constrained call must be force-closed to stay parseable; 64 is the control
# rung that completes normally and proves the small rungs measure truncation,
# not a misconfiguration that would fail at every budget.
LADDER = (1, 2, 3, 5, 8, 16, 64)
CONTROL_RUNG = 64
TRUNCATED_RUNGS = tuple(r for r in LADDER if r != CONTROL_RUNG)

# One fixed tool and one fixed prompt, held constant across both engines. The
# schema is a two-required-field object so the grammar has real structure to
# close (an opening brace and one unfinished string are enough to make a naive
# emitter produce unparseable output at rung 1).
TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "units": {"type": "string",
                          "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city", "units"],
            "additionalProperties": False,
        },
    },
}
PROMPT = "What is the weather in Paris? Use fahrenheit."


def request_for(rung, model=None):
    """The exact request sent at one ladder rung. tool_choice 'required' forces
    a call at every budget, so the only variable is whether it survives."""
    req = {
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 0,
        "max_tokens": rung,
        "tools": [TOOL],
        "tool_choice": "required",
    }
    if model is not None:
        req["model"] = model
    return req


def _first_tool_call(message):
    calls = message.get("tool_calls")
    if not isinstance(calls, list) or not calls:
        return None, calls
    return calls[0], calls


def observe(response, rung):
    """Reduce one HTTP response to the client-visible truncation observables."""
    record = {"max_tokens": rung, "http_status": response.status}
    try:
        body = response.json
    except Exception as exc:  # non-JSON / malformed body is itself a datum
        record["error"] = f"response not JSON: {exc}"
        return record
    choices = body.get("choices") or [{}]
    choice = choices[0] if choices else {}
    message = choice.get("message") or {}
    record["finish_reason"] = choice.get("finish_reason")

    call, calls = _first_tool_call(message)
    record["tool_calls_present"] = bool(calls)
    record["tool_call_count"] = len(calls) if isinstance(calls, list) else 0
    record["tool_name"] = None
    record["arguments_raw"] = None
    record["arguments_parseable"] = False
    record["arguments"] = None
    if call:
        function = call.get("function") or {}
        record["tool_name"] = function.get("name")
        args = function.get("arguments")
        record["arguments_raw"] = args
        if isinstance(args, str):
            try:
                record["arguments"] = json.loads(args)
                record["arguments_parseable"] = True
            except ValueError:
                record["arguments_parseable"] = False

    content = message.get("content")
    record["content"] = content
    # Leakage = protocol framing / partial text where a tool-call turn should
    # carry no assistant prose. Empty string and null are both "no leak".
    record["content_leak"] = bool(isinstance(content, str) and content.strip())
    return record


def run_ladder(client, model=None, name="trunc"):
    """Drive the whole ladder against a Client and return one record per rung,
    in ladder order."""
    records = []
    for rung in LADDER:
        response = client.chat(request_for(rung, model), name=f"{name}-{rung}")
        rec = observe(response, rung)
        rec["raw_response_b64"] = base64.b64encode(response.body).decode("ascii")
        records.append(rec)
    return records


def check_runner_property(records):
    """Return a list of human-readable violations of the runner guarantee.

    The guarantee, rung by rung:
      * every truncated rung (1,2,3,5,8,16): a tool call is present, its
        arguments parse as JSON, finish_reason is 'length', and no framing
        leaks into content;
      * the control rung (64): a tool call is present and parses, and it
        completed (finish_reason 'tool_calls'), not truncated.

    An empty list means the property holds. This is what the CI gate asserts,
    so a runner regression that loses force-close (unparseable JSON), drops the
    call, mislabels finish_reason, or leaks framing turns the gate red.
    """
    by_rung = {r["max_tokens"]: r for r in records}
    violations = []
    for rung in LADDER:
        rec = by_rung.get(rung)
        if rec is None:
            violations.append(f"rung {rung}: missing from results")
            continue
        if not rec.get("tool_calls_present"):
            violations.append(
                f"rung {rung}: no tool_calls (finish_reason="
                f"{rec.get('finish_reason')!r}, content={rec.get('content')!r})")
            continue
        if not rec.get("arguments_parseable"):
            violations.append(
                f"rung {rung}: tool-call arguments do not parse as JSON "
                f"({rec.get('arguments_raw')!r})")
        if rec.get("content_leak"):
            violations.append(
                f"rung {rung}: framing leaked into content "
                f"({rec.get('content')!r})")
        if rung == CONTROL_RUNG:
            if rec.get("finish_reason") != "tool_calls":
                violations.append(
                    f"rung {rung} (control): expected finish_reason "
                    f"'tool_calls', got {rec.get('finish_reason')!r} — the "
                    "control rung must complete, else the ladder is "
                    "misconfigured, not truncated")
        else:
            if rec.get("finish_reason") != "length":
                violations.append(
                    f"rung {rung}: expected finish_reason 'length', got "
                    f"{rec.get('finish_reason')!r}")
    return violations


# --------------------------------------------------------------------- targets
class RemoteTarget:
    """A stand-in the harness Client can drive, pointing at an already-running
    local OpenAI-compatible server. The Client needs only port + assert_alive +
    sample_rss. Loopback only: the runtimes under comparison run on the same
    box (identical hardware is the point); tunnel a remote host to a local
    port first."""

    def __init__(self, port):
        self.port = int(port)
        self.peak_rss_kb = None

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


def parse_endpoint(value):
    """host:port or a full URL -> local port. Non-loopback hosts are rejected
    so a published result never names a machine or reaches off-box."""
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


class _Sink:
    def record_request(self, record):
        pass


def _version(exe):
    proc = subprocess.run([str(exe), "--version"], text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          check=False)
    return proc.stdout.strip() or f"exit {proc.returncode}"


def make_report(records, runtime_name, version, model, elapsed_ms):
    violations = check_runner_property(records) if runtime_name == "runner" else None
    report = {
        "schema_version": SCHEMA_VERSION,
        "runtime": {"name": runtime_name, "version": version},
        "configuration": {
            "model": model, "temperature": 0,
            "tool_choice": "required", "ladder": list(LADDER),
            "control_rung": CONTROL_RUNG,
            "prompt": PROMPT, "tool": TOOL,
        },
        "metrics": {"elapsed_ms": elapsed_ms},
        "rungs": records,
    }
    if violations is not None:
        report["runner_property"] = {
            "holds": not violations, "violations": violations}
    return report


def write_report(out, report):
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")


def print_table(report):
    print(f"runtime={report['runtime']['name']} "
          f"version={report['runtime']['version']} "
          f"model={report['configuration']['model']}")
    header = f"{'rung':>4}  {'finish_reason':>13}  {'tool_call':>9}  " \
             f"{'parseable':>9}  {'content_leak':>12}"
    print(header)
    for rec in report["rungs"]:
        print(f"{rec['max_tokens']:>4}  {str(rec.get('finish_reason')):>13}  "
              f"{('yes' if rec.get('tool_calls_present') else 'NO'):>9}  "
              f"{str(rec.get('arguments_parseable')):>9}  "
              f"{str(rec.get('content_leak')):>12}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--endpoint",
                        help="target an already-running OpenAI-compatible "
                             "server (host:port or URL) instead of spawning "
                             "Runner; use with --runtime and --model-name")
    parser.add_argument("--runtime",
                        help="runtime label for the report (e.g. vllm, "
                             "llama.cpp). Defaults to 'runner'.")
    parser.add_argument("--runtime-version", default="unknown",
                        help="version string for the report when --endpoint "
                             "is used")
    parser.add_argument("--runner", type=Path,
                        help="path to the runner binary (default: auto-detect)")
    parser.add_argument("--model", type=Path, default=ROOT / "test.gguf",
                        help="GGUF model to serve when spawning Runner")
    parser.add_argument("--model-name",
                        help="OpenAI `model` field to set on every request "
                             "(what the endpoint serves; required by vLLM's "
                             "--served-model-name)")
    parser.add_argument("--out", type=Path,
                        help="write report.json here (default: no file, print "
                             "only)")
    parser.add_argument("--assert", dest="do_assert", action="store_true",
                        help="assert Runner's truncation-recovery property and "
                             "exit non-zero on any violation (the CI gate)")
    args = parser.parse_args(argv)

    started = time.monotonic()
    if args.endpoint:
        if args.do_assert:
            parser.error("--assert enforces Runner's guarantee; it is not "
                         "meaningful against a competitor --endpoint")
        try:
            port = parse_endpoint(args.endpoint)
        except ValueError as exc:
            parser.error(str(exc))
        target = RemoteTarget(port)
        runtime_name = args.runtime or "openai-compatible"
        version = args.runtime_version
        model_label = args.model_name or "unknown"
        request_model = args.model_name
    else:
        exe = args.runner or Path(find_runner(str(ROOT)))
        if not args.model.is_file():
            parser.error(f"model not found: {args.model}")
        target = RunnerServer(str(exe), str(args.model), ctx=1024, parallel=1,
                              extra_args=["--gpu", "off"])
        runtime_name = args.runtime or "runner"
        version = _version(exe)
        model_label = str(args.model)
        request_model = args.model_name

    with target as server:
        client = Client(server, _Sink())
        records = run_ladder(client, request_model)
    elapsed = round((time.monotonic() - started) * 1000, 2)

    report = make_report(records, runtime_name, version, model_label, elapsed)
    print_table(report)
    if args.out:
        write_report(args.out, report)
        print(f"report: {args.out / 'report.json'}")

    violations = check_runner_property(records)
    if args.do_assert:
        if violations:
            print("\nTRUNCATION-RECOVERY GATE FAILED:", file=sys.stderr)
            for v in violations:
                print(f"  - {v}", file=sys.stderr)
            return 1
        print("\ntruncation-recovery gate: PASS "
              f"({len(TRUNCATED_RUNGS)} truncated rungs closed to parseable "
              "tool calls, control rung completed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
