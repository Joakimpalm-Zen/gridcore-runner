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


def relativize_path(path: Path | str | None) -> str | None:
    """Convert absolute path to repo-relative or basename for reproducibility.

    Returns repo-relative path if within ROOT, otherwise basename only.
    """
    if path is None:
        return None
    p = Path(path).resolve()
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        # Path is outside ROOT, use basename only
        return p.name


def relativize_command(cmd: list[str]) -> list[str]:
    """Relativize all paths in a command line."""
    result = []
    for arg in cmd:
        try:
            p = Path(arg).resolve()
            if p.exists() and p.is_file():
                result.append(relativize_path(p))
            else:
                result.append(arg)
        except (ValueError, OSError):
            result.append(arg)
    return result


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
    # Certification pins the scalar GEMM path (TC default-on for promoted
    # combos is tolerance-gated, not byte-identical). Inert for llama.cpp.
    env = dict(os.environ, RUNNER_CUDA_TC="0", RUNNER_MOE_EAGER="1")
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                               text=True, env=env)
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


def completion_request(prompt, tokens, stream, model=None):
    body = {
        "prompt": prompt,
        "max_tokens": tokens,
        "temperature": 0,
        "top_p": 1,
        "stream": stream,
    }
    # Runner and llama.cpp serve one model and ignore the field; Ollama and
    # LM Studio route on it, so an extra endpoint must name what it loaded.
    if model:
        body["model"] = model
    return body


def timing_request(prompt, tokens, model=None):
    """The body used for the throughput measurement.

    `cache_prompt: false` is the whole point: Runner and llama.cpp both keep a
    prefix cache, so a prompt served twice is not prefilled twice, and a
    measurement taken after any earlier identical request reports the cache
    rather than the engine. With it left on, this harness measured "prefill" at
    16k-18k tok/s for both engines — a cache hit wearing a throughput number.
    Engines that do not know the field ignore it, so those are additionally
    measured as the FIRST request against a freshly loaded model.
    """
    body = completion_request(prompt, tokens, True, model)
    body["cache_prompt"] = False
    return body


def stream_ttft(base, prompt, tokens, request_timeout, model=None):
    body = timing_request(prompt, tokens, model)
    req = urllib.request.Request(base + "/v1/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    first = None
    text = []
    chunks = 0
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
            if piece:
                chunks += 1
            text.append(piece)
    return {"time_to_first_token_s": first, "stream_text": "".join(text),
            "stream_total_s": time.perf_counter() - start,
            "stream_chunks": chunks}


def derived_metrics(stream, prompt_tokens):
    """Throughput measured the SAME way for every runtime, from the wire.

    Engines disagree about what they self-report — llama.cpp and Runner return
    a `timings` block, Ollama's OpenAI layer returns none at all — so a table
    built from each engine's own accounting is not one measurement repeated,
    it is several different measurements printed in a column. These fields are
    derived from the streaming response alone (arrival time of the first token
    and of the last), so every row means the same thing.

    Prefill is charged the whole time-to-first-token, which slightly
    understates it: TTFT also contains request handling and sampling for one
    token. It is the same overhead for every engine measured here.
    """
    ttft = stream.get("time_to_first_token_s")
    total = stream.get("stream_total_s")
    n = stream.get("stream_chunks") or 0
    out = {"derived_decode_tok_s": None, "derived_prefill_tok_s": None,
           "derived_streamed_tokens": n}
    if ttft and total and n > 1 and total > ttft:
        out["derived_decode_tok_s"] = (n - 1) / (total - ttft)
    if ttft and prompt_tokens:
        out["derived_prefill_tok_s"] = prompt_tokens / ttft
    return out


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
        return {"command": relativize_command(cmd), "error": str(proc)}
    try:
        parsed = json.loads(proc.stdout)
    except json.JSONDecodeError:
        parsed = None
    return {"command": relativize_command(cmd), "returncode": proc.returncode, "stdout": proc.stdout,
            "stderr": proc.stderr, "parsed": parsed}


def measure_runtime(label, command, log_path, prompt, tokens,
                    startup_timeout, request_timeout):
    before = nvidia_snapshot()
    process, log, base = serve(command, log_path, startup_timeout)
    try:
        after_start = nvidia_snapshot()
        # Timing request FIRST, while nothing is cached. The buffered
        # correctness request runs after it and may be served warm — it is
        # compared for text and logprobs, which caching does not change.
        stream = stream_ttft(base, prompt, tokens, request_timeout)
        body = completion_request(prompt, tokens, False)
        body["logprobs"] = 20
        t0 = time.perf_counter()
        response = request_json(base + "/v1/completions", body,
                                timeout=request_timeout)
        wall = time.perf_counter() - t0
        logprobs = top_logprobs(base, prompt, tokens, request_timeout)
        metrics = timing_fields(response)
        metrics["request_wall_s"] = wall
        metrics["time_to_first_token_s"] = stream["time_to_first_token_s"]
        metrics["prompt_tokens"] = (response.get("usage") or {}).get("prompt_tokens")
        metrics.update(derived_metrics(stream, metrics["prompt_tokens"]))
        return {
            "label": label,
            "command": relativize_command(command),
            "response": response,
            "generated_text": completion_text(response),
            "stream_text": stream["stream_text"],
            "metrics": metrics,
            "top_logprobs": logprobs,
            "raw_completion_logprobs": legacy_completion_logprobs(response),
            "nvidia_smi_before": before,
            "nvidia_smi_after_start": after_start,
            "vram_load_delta_mib": vram_delta(before, after_start),
            "log": relativize_path(log_path),
        }
    finally:
        stop(process, log)


def measure_endpoint(label, base, model_name, prompt, tokens, request_timeout):
    """Throughput for an OpenAI-compatible server this script did not start.

    Ollama and LM Studio are daemons that already hold the model, so there is
    no process to spawn and no load-time VRAM delta to attribute — the single
    snapshot here is labelled `during`, not a delta, rather than reporting a
    number that would read like one.

    Deliberately speed-only. The correctness gate in this report is defined
    against the pinned llama.cpp b10076 reference, and these runtimes do not
    expose completion logprobs comparably (Ollama's OpenAI layer does not
    expose them at all), so asking them for a correctness verdict would
    produce an unfalsifiable one.
    """
    # Timing request first, for the reason in timing_request().
    stream = stream_ttft(base, prompt, tokens, request_timeout, model_name)
    body = completion_request(prompt, tokens, False, model_name)
    t0 = time.perf_counter()
    response = request_json(base + "/v1/completions", body,
                            timeout=request_timeout)
    wall = time.perf_counter() - t0
    metrics = timing_fields(response)
    metrics["request_wall_s"] = wall
    metrics["time_to_first_token_s"] = stream["time_to_first_token_s"]
    metrics["prompt_tokens"] = (response.get("usage") or {}).get("prompt_tokens")
    metrics.update(derived_metrics(stream, metrics["prompt_tokens"]))
    return {
        "label": label,
        "base_url": base,
        "request_model": model_name,
        "spawned_by_harness": False,
        "measurement": "throughput_only",
        "response": response,
        "generated_text": completion_text(response),
        "stream_text": stream["stream_text"],
        "metrics": metrics,
        "nvidia_smi_during": nvidia_snapshot(),
    }


def parse_endpoint(spec):
    """`label=base_url=model_name` -> the three parts.

    Split from the left with maxsplit=2 so a model name may contain `=`;
    labels and URLs in practice do not.
    """
    parts = spec.split("=", 2)
    if len(parts) != 3 or not all(p.strip() for p in parts):
        raise SystemExit(
            f"--endpoint expects label=base_url=model_name, got: {spec!r}")
    label, base, model_name = (p.strip() for p in parts)
    base = base.rstrip("/")
    # Requests are sent to base + "/v1/completions", so the base is the server
    # root. Ollama and LM Studio document their OpenAI surface as ".../v1",
    # and pasting that would silently produce /v1/v1/completions and a 404
    # that reads like the engine being down. Accept either spelling.
    if base.endswith("/v1"):
        base = base[: -len("/v1")]
    return label, base, model_name


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
    skipped_unrenderable = 0
    for i in range(n):
        ar = apos[i]
        br = bpos[i]
        # Keyed by the RENDERED string, because llama.cpp's completions
        # endpoint exposes no token ids to key on. The empty string is
        # therefore excluded: it is not a token identity but whatever each
        # engine's renderer emits for a token with no printable form, and the
        # two engines disagree about which token that is (runner renders the
        # stop token as `<eos>` where llama.cpp renders `""`). Comparing them
        # measures the renderers, not the arithmetic — on Ministral-8B it
        # manufactured a 4.59-nat "divergence" while every real token agreed
        # to 0.0013. Excluded rather than tolerated, and counted so the
        # exclusion is visible instead of silent.
        amap = {item.get("token"): item.get("logprob")
                for item in ar.get("top_logprobs") or []
                if item.get("token")
                and isinstance(item.get("logprob"), (int, float))}
        bmap = {item.get("token"): item.get("logprob")
                for item in br.get("top_logprobs") or []
                if item.get("token")
                and isinstance(item.get("logprob"), (int, float))}
        skipped_unrenderable += sum(
            1 for side in (ar, br)
            for item in side.get("top_logprobs") or []
            if item.get("token") == "")
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
            "max_abs_common_logprob_delta": max_delta,
            "unrenderable_entries_excluded": skipped_unrenderable,
            "positions": rows}


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


def fmt(value, places=2):
    return "" if value is None else f"{value:.{places}f}"


def render_endpoints_markdown(report):
    settings = report["settings"]
    lines = [
        "# Third-party runtime throughput",
        "",
        "Measured with `compare_llamacpp.py --endpoints-only`: each daemon is "
        "measured while it is the ONLY engine holding the GPU, because VRAM is "
        "exclusive and a co-resident measurement would report a spilled engine's "
        "host speed as if it were its device speed. No correctness gate is "
        "emitted here — that gate is defined against the pinned llama.cpp "
        "reference build, which this mode does not run.",
        "",
        f"- Generated UTC: `{report['generated_utc']}`",
        f"- Model path: `{report['model'].get('path')}`",
        f"- Model SHA256: `{report['model'].get('sha256')}`",
        f"- Context: `{settings['context']}`, max tokens: "
        f"`{settings['tokens']}`, sampling: `{settings['sampling']}`",
        f"- Prompt: `{settings['prompt']}`",
        "",
        "Derived columns are measured identically for every engine from the "
        "streaming response (see `derived_metrics`); self-reported columns are "
        "whatever the engine's own `timings` block claims, and are blank for "
        "engines that report none.",
        "",
        "| Runtime | Model id | Derived prefill tok/s | Derived decode tok/s | "
        "Self-reported prompt tok/s | Self-reported decode tok/s | TTFT s | Tokens | Wall s |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report.get("extra_runtimes") or []:
        if row.get("error"):
            lines.append(f"| {row['label']} | {row['request_model']} | "
                         f"unreachable: {row['error']} | | | | | | |")
            continue
        m = row.get("metrics") or {}
        lines.append(
            "| {n} | {mid} | {dp} | {dd} | {p} | {d} | {t} | {g} | {w} |".format(
                n=row["label"], mid=row["request_model"],
                dp=fmt(m.get("derived_prefill_tok_s")),
                dd=fmt(m.get("derived_decode_tok_s")),
                p=fmt(m.get("prompt_tok_s")), d=fmt(m.get("decode_tok_s")),
                t=fmt(m.get("time_to_first_token_s"), 4),
                g=m.get("generated_tokens"), w=fmt(m.get("request_wall_s"))))
    lines += ["", "## Hardware and driver", "", "```json",
              json.dumps(report.get("hardware"), indent=2, sort_keys=True),
              "```", ""]
    return "\n".join(lines) + "\n"


def render_markdown(report):
    if report.get("status") == "endpoints_only":
        return render_endpoints_markdown(report)
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
        "Derived columns are measured identically for every engine from the "
        "streaming response (see `derived_metrics`); self-reported columns come "
        "from each engine's own `timings` block, which not every engine emits "
        "and which engines do not all define the same way.",
        "",
        "| Runtime | Derived prefill tok/s | Derived decode tok/s | "
        "Self-reported prompt tok/s | Self-reported decode tok/s | TTFT s | Tokens | Wall s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    rows = [(key, report[key].get("metrics") or {}) for key in ("runner", "llamacpp")]
    for row in report.get("extra_runtimes") or []:
        if row.get("error"):
            rows.append((f"{row['label']} (unreachable: {row['error']})", {}))
        else:
            rows.append((row["label"], row.get("metrics") or {}))
    for name, m in rows:
        lines.append("| {n} | {dp} | {dd} | {p} | {d} | {t} | {g} | {w} |".format(
            n=name,
            dp=fmt(m.get("derived_prefill_tok_s")),
            dd=fmt(m.get("derived_decode_tok_s")),
            p=fmt(m.get("prompt_tok_s")), d=fmt(m.get("decode_tok_s")),
            t=fmt(m.get("time_to_first_token_s"), 4),
            g=m.get("generated_tokens"), w=fmt(m.get("request_wall_s")),
        ))
    if report.get("extra_runtimes"):
        lines += [
            "",
            "Extra runtimes are measured for throughput only, over the same "
            "prompt, token budget and greedy settings. They are daemons this "
            "harness did not start, so their model load is not timed here and "
            "the correctness gate below does not include them — that gate is "
            "defined against the pinned llama.cpp reference build.",
        ]
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
        "schema_version": "xyntetik.runner.llamacpp-comparison.v1",
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
    runner_cmd.extend(args.runner_arg or [])
    llama_cmd = [str(args.llamacpp.resolve()), "-m", str(args.model.resolve()),
                 "--host", "127.0.0.1", "--port", str(llama_port),
                 "-c", str(args.ctx), "-ngl", str(args.llamacpp_gpu_layers)]
    llama_cmd.extend(args.llamacpp_arg or [])
    return runner_cmd, llama_cmd


def endpoints_only_report(args):
    """Measure ONLY already-running daemons, spawning nothing.

    Exists because VRAM is exclusive: three engines each holding a 5.4 GB
    model do not fit on an 8 GB card, so a run that measured them together
    would report whichever ones got spilled to the host as if that were their
    speed. Each engine is therefore measured with only itself resident, and
    the rows are assembled afterwards. No correctness gate is emitted here —
    that gate is defined against the pinned llama.cpp reference, which this
    mode does not run.
    """
    if not args.endpoint:
        raise SystemExit("--endpoints-only needs at least one --endpoint")
    extra = []
    for spec in args.endpoint:
        label, base, model_name = parse_endpoint(spec)
        try:
            extra.append(measure_endpoint(label, base, model_name, args.prompt,
                                          args.tokens, args.request_timeout))
        except Exception as exc:
            extra.append({"label": label, "base_url": base,
                          "request_model": model_name,
                          "spawned_by_harness": False,
                          "measurement": "throughput_only",
                          "error": f"{type(exc).__name__}: {exc}"})
    return {
        "schema_version": "xyntetik.runner.llamacpp-comparison.v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "endpoints_only",
        "real_results": "captured",
        "model": {"path": relativize_path(args.model) if args.model else None,
                  "sha256": sha256_file(args.model) if args.model else None,
                  "bytes": args.model.stat().st_size if args.model else None},
        "settings": {
            "prompt": args.prompt, "context": args.ctx, "tokens": args.tokens,
            "temperature": 0, "top_p": 1, "sampling": "greedy",
            "quantization": args.quantization,
        },
        "hardware": hardware_info(),
        "extra_runtimes": extra,
    }


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

    extra = []
    for spec in (args.endpoint or []):
        label, base, model_name = parse_endpoint(spec)
        try:
            extra.append(measure_endpoint(label, base, model_name, args.prompt,
                                          args.tokens, args.request_timeout))
        except Exception as exc:
            # A third-party daemon being down must not silently vanish from
            # the report and must not fake a number for it either.
            extra.append({"label": label, "base_url": base,
                          "request_model": model_name,
                          "spawned_by_harness": False,
                          "measurement": "throughput_only",
                          "error": f"{type(exc).__name__}: {exc}"})

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
        "schema_version": "xyntetik.runner.llamacpp-comparison.v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "complete",
        "real_results": "captured",
        "model": {
            "path": relativize_path(args.model),
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
            "command": relativize_command(runner_cmd),
            "bench_json": bench,
            **runner,
        },
        "llamacpp": {
            "version": first_line_version(args.llamacpp),
            "commit": args.llamacpp_commit or git_head(args.llamacpp),
            "command": relativize_command(llama_cmd),
            **llama,
        },
        "extra_runtimes": extra,
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
    parser.add_argument("--runner-arg", action="append",
                        help="extra flag for the Runner server, the twin of "
                             "--llamacpp-arg. Needed for a fair MoE row: recent "
                             "llama.cpp auto-overrides expert tensors to the "
                             "host, so a Runner run without --cpu-moe is "
                             "answering a different question")
    parser.add_argument("--endpoint", action="append", metavar="LABEL=URL=MODEL",
                        help="also measure an already-running OpenAI-compatible "
                             "server, e.g. "
                             "ollama=http://127.0.0.1:11434/v1=gpt-oss:20b . "
                             "Throughput only; the correctness gate stays "
                             "defined against the pinned llama.cpp build.")
    parser.add_argument("--quantization")
    parser.add_argument("--min-shared-tokens", type=int, default=32)
    parser.add_argument("--max-logprob-delta", type=float, default=2.0)
    parser.add_argument("--startup-timeout", type=int, default=300)
    parser.add_argument("--request-timeout", type=int, default=300)
    parser.add_argument("--out-dir", type=Path,
                        default=ROOT / "tests/compatibility/out/llamacpp-comparison")
    parser.add_argument("--fixture", action="store_true",
                        help="write a schema-valid pending report without running binaries")
    parser.add_argument("--endpoints-only", action="store_true",
                        help="measure only --endpoint daemons and spawn nothing "
                             "(VRAM is exclusive; engines must be measured one "
                             "at a time on a single-GPU box)")
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.fixture:
        report = fixture_report(args)
    elif args.endpoints_only:
        report = endpoints_only_report(args)
    else:
        report = real_report(args)
    json_path = args.out_dir / "comparison.json"
    md_path = args.out_dir / "comparison.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"json": relativize_path(json_path), "markdown": relativize_path(md_path),
                      "status": report["status"]}))
    return 1 if report.get("correctness_gate", {}).get("status") == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
