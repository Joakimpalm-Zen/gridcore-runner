#!/usr/bin/env python3
"""Reproducible Runner-vs-llama.cpp comparison harness.

The real mode expects an already-built llama.cpp server binary and an existing
GGUF. It does not download models, build dependencies, or fill in missing
numbers. Fixture mode is only for CI/testing the report writer; it marks real
results as pending.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT = "The capital of France is"


def run(cmd, timeout=20):
    try:
        return subprocess.run(cmd, text=True, capture_output=True,
                              timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        return exc


def first_line_version(exe):
    proc = run([str(Path(exe).resolve()), "--version"], timeout=15)
    if isinstance(proc, Exception):
        return None
    text = (proc.stdout + proc.stderr).strip().splitlines()
    return text[0] if text else None


def git_head(path):
    cur = Path(path).resolve()
    candidates = [cur if cur.is_dir() else cur.parent, *cur.parents]
    for p in candidates:
        proc = run(["git", "-C", str(p), "rev-parse", "HEAD"], timeout=5)
        if not isinstance(proc, Exception) and proc.returncode == 0:
            return proc.stdout.strip()
    return None


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def request_json(url, body=None, timeout=60):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def wait_ready(base, process, timeout):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited during startup: {process.args}")
        try:
            request_json(base + "/health", timeout=1)
            return
        except Exception as exc:
            last = exc
            time.sleep(0.1)
    raise RuntimeError(f"server startup timed out: {last}")


def nvidia_snapshot():
    proc = run([
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.used,memory.free,memory.total",
        "--format=csv,noheader,nounits",
    ], timeout=5)
    if isinstance(proc, Exception) or proc.returncode != 0:
        return None
    rows = []
    def memory_mib(value):
        try:
            return int(value)
        except ValueError:
            return None

    for line in proc.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 5:
            rows.append({
                "name": parts[0],
                "driver_version": parts[1],
                "memory_used_mib": memory_mib(parts[2]),
                "memory_free_mib": memory_mib(parts[3]),
                "memory_total_mib": memory_mib(parts[4]),
            })
    return rows or None


def hardware_info():
    info = {
        "system": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "nvidia_smi": nvidia_snapshot(),
    }
    if sys.platform == "darwin":
        proc = run(["sysctl", "-n", "machdep.cpu.brand_string"], timeout=5)
        if not isinstance(proc, Exception) and proc.returncode == 0:
            info["cpu_brand"] = proc.stdout.strip()
    return info


def vram_delta(before, after):
    """Return model-load VRAM deltas without pretending they are peak use."""
    if not before or not after or len(before) != len(after):
        return None
    rows = []
    for index, (old, new) in enumerate(zip(before, after)):
        if old.get("memory_used_mib") is None or new.get("memory_used_mib") is None:
            return None
        rows.append({
            "device": index,
            "name": new.get("name") or old.get("name"),
            "used_delta_mib": new["memory_used_mib"] - old["memory_used_mib"],
        })
    return rows


def serve(command, log_path, startup_timeout):
    log = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                               text=True)
    try:
        port = int(command[command.index("--port") + 1])
        base = f"http://127.0.0.1:{port}"
        wait_ready(base, process, startup_timeout)
        return process, log, base
    except Exception:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
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


def completion_text(response):
    try:
        text = response["choices"][0]["text"]
    except (KeyError, IndexError, TypeError):
        return None
    return text if isinstance(text, str) else None


def timing_fields(response):
    out = {
        "prompt_tok_s": None,
        "decode_tok_s": None,
        "prompt_ms": None,
        "decode_ms": None,
        "generated_tokens": None,
    }
    usage = response.get("usage") or {}
    out["generated_tokens"] = usage.get("completion_tokens")
    timings = response.get("timings") or {}
    if timings:
        out["prompt_tok_s"] = timings.get("prompt_per_second")
        out["decode_tok_s"] = timings.get("predicted_per_second")
        out["prompt_ms"] = timings.get("prompt_ms")
        out["decode_ms"] = timings.get("predicted_ms")
        out["generated_tokens"] = timings.get("predicted_n", out["generated_tokens"])
    telemetry = response.get("runner_telemetry") or {}
    if telemetry:
        out["decode_tok_s"] = telemetry.get("generation_tok_s", out["decode_tok_s"])
    return out


def completion_request(prompt, tokens, stream):
    return {
        "prompt": prompt,
        "max_tokens": tokens,
        "temperature": 0,
        "top_p": 1,
        "stream": stream,
    }


def stream_ttft(base, prompt, tokens, request_timeout):
    body = completion_request(prompt, tokens, True)
    req = urllib.request.Request(base + "/v1/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    first = None
    text = []
    with urllib.request.urlopen(req, timeout=request_timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            evt = json.loads(data)
            piece = completion_text(evt) or ""
            if piece and first is None:
                first = time.perf_counter() - start
            text.append(piece)
    return {"time_to_first_token_s": first, "stream_text": "".join(text)}


def top_logprobs(base, prompt, tokens, request_timeout):
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": min(tokens, 8),
        "temperature": 0,
        "logprobs": True,
        "top_logprobs": 5,
    }
    try:
        response = request_json(base + "/v1/chat/completions", body,
                                timeout=request_timeout)
        content = response["choices"][0].get("logprobs", {}).get("content")
        if isinstance(content, list) and content:
            return {"status": "captured", "request": body,
                    "positions": content}
        return {"status": "unsupported", "request": body,
                "reason": "response had no top logprobs"}
    except Exception as exc:
        return {"status": "unsupported", "request": body,
                "reason": str(exc)}


def legacy_completion_logprobs(response):
    try:
        raw = response["choices"][0]["logprobs"]
        if isinstance(raw.get("content"), list):
            return {
                "status": "captured",
                "positions": [{
                    "token": row["token"],
                    "logprob": row["logprob"],
                    "top_logprobs": [{
                        "token": alt["token"], "logprob": alt["logprob"],
                    } for alt in row.get("top_logprobs", [])],
                } for row in raw["content"]],
            }
        tokens = raw["tokens"]
        chosen = raw["token_logprobs"]
        alternatives = raw["top_logprobs"]
        if not (len(tokens) == len(chosen) == len(alternatives)):
            raise ValueError("legacy logprob arrays have different lengths")
        positions = []
        for token, logprob, top in zip(tokens, chosen, alternatives):
            positions.append({
                "token": token,
                "logprob": logprob,
                "top_logprobs": [
                    {"token": alt, "logprob": lp}
                    for alt, lp in top.items()
                ],
            })
        return {"status": "captured", "positions": positions}
    except (KeyError, TypeError, ValueError) as exc:
        return {"status": "unsupported", "reason": str(exc)}


def first_token_divergence(runner, llama):
    if runner.get("status", "captured") != "captured" or \
       llama.get("status", "captured") != "captured":
        return {"status": "pending", "reason": "raw token logprobs unavailable"}
    a = runner.get("positions", [])
    b = llama.get("positions", [])
    for i, (ar, br) in enumerate(zip(a, b)):
        if ar.get("token") != br.get("token"):
            return {"status": "diverged", "position": i, "shared_tokens": i,
                    "runner": ar, "llamacpp": br}
    if len(a) == len(b):
        return {"status": "identical", "shared_tokens": len(a)}
    i = min(len(a), len(b))
    return {"status": "length_mismatch", "position": i, "shared_tokens": i,
            "runner": a[i] if i < len(a) else None,
            "llamacpp": b[i] if i < len(b) else None}


def runner_bench_json(runner, model, prompt, ctx, tokens, runner_gpu):
    cmd = [str(runner), "-m", str(model), "-p", prompt, "-c", str(ctx),
           "-n", str(tokens), "--temp", "0", "--bench-json", "--gpu", runner_gpu]
    proc = run(cmd, timeout=600)
    if isinstance(proc, Exception):
        return {"command": cmd, "error": str(proc)}
    try:
        parsed = json.loads(proc.stdout)
    except json.JSONDecodeError:
        parsed = None
    return {"command": cmd, "returncode": proc.returncode, "stdout": proc.stdout,
            "stderr": proc.stderr, "parsed": parsed}


def measure_runtime(label, command, log_path, prompt, tokens,
                    startup_timeout, request_timeout):
    before = nvidia_snapshot()
    process, log, base = serve(command, log_path, startup_timeout)
    try:
        after_start = nvidia_snapshot()
        body = completion_request(prompt, tokens, False)
        body["logprobs"] = 20
        t0 = time.perf_counter()
        response = request_json(base + "/v1/completions", body,
                                timeout=request_timeout)
        wall = time.perf_counter() - t0
        stream = stream_ttft(base, prompt, tokens, request_timeout)
        logprobs = top_logprobs(base, prompt, tokens, request_timeout)
        metrics = timing_fields(response)
        metrics["request_wall_s"] = wall
        metrics["time_to_first_token_s"] = stream["time_to_first_token_s"]
        return {
            "label": label,
            "command": command,
            "response": response,
            "generated_text": completion_text(response),
            "stream_text": stream["stream_text"],
            "metrics": metrics,
            "top_logprobs": logprobs,
            "raw_completion_logprobs": legacy_completion_logprobs(response),
            "nvidia_smi_before": before,
            "nvidia_smi_after_start": after_start,
            "vram_load_delta_mib": vram_delta(before, after_start),
            "log": str(log_path),
        }
    finally:
        stop(process, log)


def compare_top_logprobs(a, b):
    if a.get("status") != "captured" or b.get("status") != "captured":
        return {
            "status": "pending",
            "reason": "top-logprob capture unsupported by at least one endpoint",
        }
    apos = a["positions"]
    bpos = b["positions"]
    n = min(len(apos), len(bpos))
    rows = []
    max_delta = None
    for i in range(n):
        ar = apos[i]
        br = bpos[i]
        amap = {item.get("token"): item.get("logprob")
                for item in ar.get("top_logprobs") or []
                if item.get("token") is not None
                and isinstance(item.get("logprob"), (int, float))}
        bmap = {item.get("token"): item.get("logprob")
                for item in br.get("top_logprobs") or []
                if item.get("token") is not None
                and isinstance(item.get("logprob"), (int, float))}
        common = []
        for token in amap.keys() & bmap.keys():
            delta = round(abs(amap[token] - bmap[token]), 12)
            max_delta = delta if max_delta is None else max(max_delta, delta)
            common.append({
                "token": token,
                "runner_logprob": amap[token],
                "llamacpp_logprob": bmap[token],
                "abs_delta": delta,
            })
        common.sort(key=lambda item: (item["abs_delta"], item["token"]))
        chosen_delta = None
        if (ar.get("token") == br.get("token")
                and isinstance(ar.get("logprob"), (int, float))
                and isinstance(br.get("logprob"), (int, float))):
            chosen_delta = round(abs(ar["logprob"] - br["logprob"]), 12)
        rows.append({
            "position": i,
            "runner_token": ar.get("token"),
            "llamacpp_token": br.get("token"),
            "chosen_logprob_delta": chosen_delta,
            "common_top_logprobs": common,
            "runner_top_logprobs": ar.get("top_logprobs"),
            "llamacpp_top_logprobs": br.get("top_logprobs"),
        })
    return {"status": "captured", "positions_compared": n,
            "max_abs_common_logprob_delta": max_delta, "positions": rows}


def correctness_gate(divergence, logprobs, min_shared_tokens, max_logprob_delta):
    """Judge cross-engine equivalence without requiring brittle text identity."""
    shared = divergence.get("shared_tokens")
    delta = logprobs.get("max_abs_common_logprob_delta")
    reasons = []
    if not isinstance(shared, int) or shared < min_shared_tokens:
        reasons.append(f"shared prefix {shared} is below {min_shared_tokens}")
    if not isinstance(delta, (int, float)) or delta > max_logprob_delta:
        reasons.append(f"common-token logprob delta {delta} exceeds {max_logprob_delta}")
    return {
        "status": "pass" if not reasons else "fail",
        "minimum_shared_tokens": min_shared_tokens,
        "maximum_common_logprob_delta": max_logprob_delta,
        "observed_shared_tokens": shared,
        "observed_max_common_logprob_delta": delta,
        "reasons": reasons,
    }


def quote_cmd(cmd):
    return " ".join(shlex.quote(str(x)) for x in cmd)


def render_markdown(report):
    settings = report["settings"]
    runner = report["runner"]
    llama = report["llamacpp"]
    top = report["top_logprob_comparison"]
    lines = [
        "# Runner vs llama.cpp comparison",
        "",
        "## Provenance",
        "",
        f"- Schema: `{report['schema_version']}`",
        f"- Generated UTC: `{report['generated_utc']}`",
        f"- Status: `{report['status']}`",
        f"- Model path: `{report['model'].get('path')}`",
        f"- Model SHA256: `{report['model'].get('sha256')}`",
        f"- Model bytes: `{report['model'].get('bytes')}`",
        f"- Runner: `{runner.get('version')}`",
        f"- Runner commit: `{runner.get('commit')}`",
        f"- llama.cpp: `{llama.get('version')}`",
        f"- llama.cpp commit: `{llama.get('commit')}`",
        "",
        "## Settings",
        "",
        f"- Context: `{settings['context']}`",
        f"- Maximum generated tokens: `{settings['tokens']}`",
        f"- Quantization: `{settings.get('quantization')}`",
        f"- Temperature: `{settings.get('temperature')}`",
        f"- Top-p: `{settings.get('top_p')}`",
        f"- Sampling: `{settings.get('sampling')}`",
        f"- Prompt: `{settings['prompt']}`",
        "",
        "The TTFT request is a separate warmed streaming request after model load. "
        "The auxiliary top-k comparison sends the same chat payload to both "
        "runtimes; it is distinct from the raw-completion throughput request.",
        "",
        "## Hardware and driver",
        "",
        "```json",
        json.dumps(report.get("hardware"), indent=2, sort_keys=True),
        "```",
        "",
        "## Commands",
        "",
        f"Runner: `{quote_cmd(runner['command'])}`",
        "",
        f"llama.cpp: `{quote_cmd(llama['command'])}`",
        "",
        "## Results",
        "",
        "| Runtime | Prompt tok/s | Decode tok/s | TTFT s | Generated tokens | Wall s |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key in ("runner", "llamacpp"):
        m = report[key].get("metrics") or {}
        lines.append("| {name} | {prompt} | {decode} | {ttft} | {tokens} | {wall} |".format(
            name=key,
            prompt=m.get("prompt_tok_s"),
            decode=m.get("decode_tok_s"),
            ttft=m.get("time_to_first_token_s"),
            tokens=m.get("generated_tokens"),
            wall=m.get("request_wall_s"),
        ))
    lines += [
        "",
        "## VRAM",
        "",
        "Load deltas are `nvidia-smi` used-memory changes from immediately "
        "before process start to server readiness; they are not peak VRAM.",
        "",
        "```json",
        json.dumps({
            "runner_load_delta_mib": runner.get("vram_load_delta_mib"),
            "llamacpp_load_delta_mib": llama.get("vram_load_delta_mib"),
            "runner_after_start": runner.get("nvidia_smi_after_start"),
            "llamacpp_after_start": llama.get("nvidia_smi_after_start"),
        }, indent=2, sort_keys=True),
        "```",
        "",
        "## Correctness comparison",
        "",
        f"Text comparison: `{report['text_comparison']['status']}`",
        "",
        f"Top-logprob comparison: `{top['status']}`",
        f"Maximum absolute common-token logprob delta: "
        f"`{top.get('max_abs_common_logprob_delta')}`",
        f"Correctness gate: `{report.get('correctness_gate', {}).get('status', 'pending')}`",
        "",
        "## Generated output",
        "",
        "Runner:",
        "",
        "```text",
        runner.get("generated_text") or "",
        "```",
        "",
        "llama.cpp:",
        "",
        "```text",
        llama.get("generated_text") or "",
        "```",
        "",
        "## Raw artifacts",
        "",
        "The complete buffered responses, benchmark JSON, top-k values, exact "
        "requests, and VRAM snapshots are in `comparison.json`. Server output "
        "is in `runner.log` and `llamacpp.log` for real runs.",
        "",
        "Real Qwen3/MoE GPU results are pending unless this report status is `complete`.",
    ]
    return "\n".join(lines) + "\n"


def fixture_report(args):
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    command = ["fixture-only"]
    report = {
        "schema_version": "gridcore.runner.llamacpp-comparison.v1",
        "generated_utc": now,
        "status": "fixture",
        "real_results": "pending",
        "model": {"path": None, "sha256": None, "bytes": None},
        "settings": {
            "prompt": args.prompt,
            "context": args.ctx,
            "tokens": args.tokens,
            "temperature": 0,
            "top_p": 1,
            "sampling": "greedy",
            "quantization": args.quantization,
        },
        "hardware": {"fixture": True},
        "runner": {"version": "fixture", "commit": None, "command": command,
                   "metrics": {}, "generated_text": "fixture"},
        "llamacpp": {"version": "fixture", "commit": None, "command": command,
                     "metrics": {}, "generated_text": "fixture"},
        "text_comparison": {"status": "fixture"},
        "top_logprob_comparison": {"status": "pending",
                                    "reason": "fixture mode does not query logits"},
        "correctness_gate": {"status": "pending"},
    }
    return report


def runtime_commands(args, runner_port, llama_port):
    runner_cmd = [str(args.runner.resolve()), "-m", str(args.model.resolve()),
                  "--serve", "--port", str(runner_port), "-c", str(args.ctx),
                  "--gpu", args.runner_gpu, "-n", str(args.tokens)]
    llama_cmd = [str(args.llamacpp.resolve()), "-m", str(args.model.resolve()),
                 "--host", "127.0.0.1", "--port", str(llama_port),
                 "-c", str(args.ctx), "-ngl", str(args.llamacpp_gpu_layers)]
    llama_cmd.extend(args.llamacpp_arg or [])
    return runner_cmd, llama_cmd


def real_report(args):
    if not args.model:
        raise SystemExit("--model is required outside --fixture")
    if not args.llamacpp:
        raise SystemExit("--llamacpp is required outside --fixture")
    if not args.model.exists():
        raise SystemExit(f"model not found: {args.model}")
    if not args.runner.exists():
        raise SystemExit(f"runner not found: {args.runner}")
    if not args.llamacpp.exists():
        raise SystemExit(f"llama.cpp server not found: {args.llamacpp}")

    out = args.out_dir
    runner_port = free_port()
    llama_port = free_port()
    runner_cmd, llama_cmd = runtime_commands(args, runner_port, llama_port)

    runner = measure_runtime("runner", runner_cmd, out / "runner.log",
                             args.prompt, args.tokens, args.startup_timeout,
                             args.request_timeout)
    llama = measure_runtime("llama.cpp", llama_cmd, out / "llamacpp.log",
                            args.prompt, args.tokens, args.startup_timeout,
                            args.request_timeout)

    bench = runner_bench_json(args.runner.resolve(), args.model.resolve(),
                              args.prompt, args.ctx, args.tokens,
                              args.runner_gpu)
    parsed = bench.get("parsed") or {}
    if parsed:
        runner["metrics"]["prompt_tok_s"] = parsed.get("prompt_tok_s")
        runner["metrics"]["decode_tok_s"] = parsed.get("gen_tok_s")
        runner["metrics"]["generated_tokens"] = parsed.get("generated_tokens")
        runner["metrics"]["context"] = parsed.get("context")

    text_status = "pass" if runner["generated_text"] == llama["generated_text"] else "fail"
    divergence = first_token_divergence(runner["raw_completion_logprobs"],
                                        llama["raw_completion_logprobs"])
    # Once greedy text diverges the engines condition on different histories,
    # so later logits are not comparable. Judge only the shared prefix.
    shared = divergence.get("shared_tokens", 0)
    runner_shared = {**runner["raw_completion_logprobs"],
                     "positions": runner["raw_completion_logprobs"].get("positions", [])[:shared]}
    llama_shared = {**llama["raw_completion_logprobs"],
                    "positions": llama["raw_completion_logprobs"].get("positions", [])[:shared]}
    raw_comparison = compare_top_logprobs(runner_shared, llama_shared)
    gate = correctness_gate(divergence, raw_comparison,
                            args.min_shared_tokens, args.max_logprob_delta)
    return {
        "schema_version": "gridcore.runner.llamacpp-comparison.v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "complete",
        "real_results": "captured",
        "model": {
            "path": str(args.model),
            "sha256": sha256_file(args.model),
            "bytes": args.model.stat().st_size,
        },
        "settings": {
            "prompt": args.prompt,
            "context": args.ctx,
            "tokens": args.tokens,
            "temperature": 0,
            "top_p": 1,
            "sampling": "greedy",
            "quantization": args.quantization,
        },
        "hardware": hardware_info(),
        "runner": {
            "version": first_line_version(args.runner),
            "commit": git_head(args.runner),
            "command": runner_cmd,
            "bench_json": bench,
            **runner,
        },
        "llamacpp": {
            "version": first_line_version(args.llamacpp),
            "commit": args.llamacpp_commit or git_head(args.llamacpp),
            "command": llama_cmd,
            **llama,
        },
        "text_comparison": {
            "status": text_status,
            "runner": runner["generated_text"],
            "llamacpp": llama["generated_text"],
            "first_token_divergence": divergence,
        },
        "top_logprob_comparison": compare_top_logprobs(
            runner["top_logprobs"], llama["top_logprobs"]),
        "raw_logprob_comparison": raw_comparison,
        "correctness_gate": gate,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=Path, default=ROOT / "runner")
    parser.add_argument("--llamacpp", type=Path,
                        help="path to llama.cpp llama-server")
    parser.add_argument("--llamacpp-commit")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--ctx", type=int, default=2048)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--runner-gpu", default="auto", choices=["auto", "off"])
    parser.add_argument("--llamacpp-gpu-layers", type=int, default=-1)
    parser.add_argument("--llamacpp-arg", action="append")
    parser.add_argument("--quantization")
    parser.add_argument("--min-shared-tokens", type=int, default=32)
    parser.add_argument("--max-logprob-delta", type=float, default=2.0)
    parser.add_argument("--startup-timeout", type=int, default=300)
    parser.add_argument("--request-timeout", type=int, default=300)
    parser.add_argument("--out-dir", type=Path,
                        default=ROOT / "tests/compatibility/out/llamacpp-comparison")
    parser.add_argument("--fixture", action="store_true",
                        help="write a schema-valid pending report without running binaries")
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = fixture_report(args) if args.fixture else real_report(args)
    json_path = args.out_dir / "comparison.json"
    md_path = args.out_dir / "comparison.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(md_path),
                      "status": report["status"]}))
    return 1 if report.get("correctness_gate", {}).get("status") == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
