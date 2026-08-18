#!/usr/bin/env python3
"""Run and record the pinned real-model compatibility matrix.

The manifest is committed, but model files are not.  A run never silently
substitutes a similarly named GGUF: its SHA-256 must match before execution.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "tests" / "compatibility" / "models.json"
TOKENIZER_SCRIPT = ROOT / "scripts" / "difftok.py"
GREEDY_SCRIPT = ROOT / "scripts" / "reference_compare.py"


# v2 (2026-08-15) adds per-model `check_params` so the classes that were
# declared-but-unrunnable (cpu_cuda, chat, tool) carry an executable definition.
# v1 manifests still load: they simply declare no parameters, and any
# parameterised class they list lands as not_executed with a specific reason.
SCHEMA_VERSIONS = ("xyntetik.runner.model-compat.v1",
                   "xyntetik.runner.model-compat.v2")



def _kill_group(pid, grace=2.0):
    """Kill one process GROUP, servers included."""
    if os.name == "nt":
        return                       # no process groups; Popen.kill is all there is
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(os.getpgid(pid), sig)
        except OSError:
            return
        time.sleep(grace)
        try:
            os.killpg(os.getpgid(pid), 0)
        except OSError:
            return


def run_group(cmd, timeout, grace=2.0, **kwargs):
    """subprocess.run, but a timeout kills the child's whole process group.

    subprocess.run's own timeout kills the DIRECT child only. The check
    scripts below spawn `runner --serve` grandchildren, and start_new_session
    puts the child in its own group, so those grandchildren outlive the kill:
    the 2026-08-15 ledger run left a 32B and a 30B server resident after two
    cpu_cuda timeouts, and a ledger that leaks a server per timeout eventually
    eats the box it is measuring.

    The fix written then read the group leader off `exc.process`.
    TimeoutExpired carries cmd/timeout/output/stderr and no process at all, so
    that lookup was always None and the kill never ran. Owning the Popen is
    what makes the pid available.
    """
    if kwargs.pop("capture_output", False):
        kwargs.setdefault("stdout", subprocess.PIPE)
        kwargs.setdefault("stderr", subprocess.PIPE)
    stdin_text = kwargs.pop("stdin_text", None)
    if stdin_text is not None:
        kwargs["stdin"] = subprocess.PIPE
    with subprocess.Popen(cmd, start_new_session=True, **kwargs) as proc:
        try:
            out, err = proc.communicate(stdin_text, timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_group(proc.pid, grace)
            proc.kill()
            out, err = proc.communicate()
            raise subprocess.TimeoutExpired(cmd, timeout, output=out,
                                            stderr=err) from None
        return subprocess.CompletedProcess(cmd, proc.returncode, out, err)


# ---------------------------------------------- parameterised check runners
#
# Each returns the same shape as the older runners: status plus enough detail to
# argue with. A missing parameter block is not_executed with a named reason, so
# a manifest that declares a contract without defining it is visible rather than
# quietly green.

def run_cpu_cuda(runner, model, params, timeout):
    """CPU vs CUDA byte identity under the pins the manifest names."""
    script = ROOT / "scripts" / "cpu_cuda_check.py"
    if not script.is_file():
        return {"status": "not_executed", "reason": "cpu_cuda_script_not_found"}
    env = dict(os.environ)
    env.update({k: str(v) for k, v in (params.get("pins") or {}).items()})
    cmd = [sys.executable, str(script), str(model),
           # Forwarded, never left to the sibling's default: cpu_cuda_check.py
           # falls back to the repo root's ./runner, so a ledger run against a
           # candidate binary would certify whichever build happens to be
           # sitting there instead.
           "--runner", str(runner),
           # 128, not 64: the decided certification length (owner 2026-08-08)
           # and cpu_cuda_check.py's own default. A row that declares nothing
           # must not silently certify at half the contract.
           "--tokens", str(params.get("tokens", 128))]
    started = time.time()
    try:
        proc = run_group(cmd, timeout, capture_output=True, text=True,
                         env=env, cwd=str(ROOT))
    except subprocess.TimeoutExpired:
        return {"status": "not_executed", "reason": "cpu_cuda_timeout"}
    out = (proc.stdout or "") + (proc.stderr or "")
    # Capture WHICH prompts diverged, not just how many. The 2026-08-15 run
    # recorded only a 400-character tail, which happened to hold the passing
    # prompts, and three failures could not be attributed without re-running
    # them by hand. The failing prompt is the whole diagnostic value here.
    score = re.search(r"(\d+)/(\d+) identical", out)
    failed_prompts = re.findall(r"^FAIL: (.+)$", out, re.M)
    return {"status": "pass" if proc.returncode == 0 else "fail",
            "returncode": proc.returncode,
            "elapsed_ms": round((time.time() - started) * 1000, 2),
            "pins": params.get("pins") or {},
            "min_prompt_tokens": params.get("min_prompt_tokens"),
            "identical": score.group(0) if score else None,
            "failed_prompts": [p[:160] for p in failed_prompts],
            "output_tail": out[-1200:]}


def run_chat(runner, model, params, timeout):
    """One-shot chat smoke: the model must answer, stop, and not leak markup."""
    prompt = params.get("prompt", "What is 2+2?")
    cmd = [str(runner), "-m", str(model), "-i", "--temp", "0",
           "--gpu", params.get("gpu", "auto"), "-n",
           str(params.get("max_tokens", 200)), "--wait-for-vram", "300"]
    started = time.time()
    try:
        proc = run_group(cmd, timeout, stdin_text=f"{prompt}\n/exit\n",
                         capture_output=True, text=True, cwd=str(ROOT))
    except subprocess.TimeoutExpired:
        return {"status": "not_executed", "reason": "chat_timeout"}
    out = proc.stdout or ""
    reasons = []
    if params.get("expect_nonempty_content", True) and not out.strip():
        reasons.append("empty_output")
    for bad in params.get("forbid_substrings") or []:
        if bad in out:
            reasons.append(f"leaked:{bad}")
    for want in params.get("expect_substrings") or []:
        if want not in out:
            reasons.append(f"missing:{want}")
    # a run that hit the token ceiling did not stop on its own
    if params.get("expect_stop", True) and f"[{params.get('max_tokens', 200)} tok" in (proc.stderr or ""):
        reasons.append("hit_token_ceiling")
    return {"status": "pass" if not reasons and proc.returncode == 0 else "fail",
            "returncode": proc.returncode,
            "elapsed_ms": round((time.time() - started) * 1000, 2),
            "prompt": prompt, "failure_reasons": reasons,
            "output_tail": out[-300:]}


def run_tool(runner, model, params, timeout):
    """Tool/structured-output fidelity via the scenario matrix the manifest names."""
    matrix = params.get("scenario_matrix", "agent-torture")
    script = ROOT / "scripts" / ("agent-torture.py" if matrix == "agent-torture" else "")
    if not script or not script.is_file():
        return {"status": "not_executed",
                "reason": f"tool_matrix_not_available:{matrix}"}
    # --runner forwarded for the same reason as run_cpu_cuda: agent-torture.py
    # otherwise resolves one with find_runner() out of the repo root.
    cmd = [sys.executable, str(script), "--runner", str(runner),
           "--model", str(model)]
    if params.get("cases"):
        cmd += ["--cases", str(params["cases"])]
    started = time.time()
    try:
        proc = run_group(cmd, timeout, capture_output=True, text=True,
                         cwd=str(ROOT))
    except subprocess.TimeoutExpired:
        return {"status": "not_executed", "reason": "tool_timeout"}
    return {"status": "pass" if proc.returncode == 0 else "fail",
            "returncode": proc.returncode,
            "elapsed_ms": round((time.time() - started) * 1000, 2),
            "scenario_matrix": matrix,
            "output_tail": ((proc.stdout or "") + (proc.stderr or ""))[-400:]}


def load_manifest(path):
    data = json.loads(Path(path).read_text())
    if data.get("schema_version") not in SCHEMA_VERSIONS:
        raise ValueError("unsupported model compatibility manifest")
    seen = set()
    for model in data.get("models", []):
        if model["id"] in seen:
            raise ValueError("duplicate model id: " + model["id"])
        seen.add(model["id"])
        digest = model.get("sha256", "")
        if len(digest) != 64:
            raise ValueError("invalid sha256 for " + model["id"])
        int(digest, 16)
    return data


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command_version(command):
    try:
        proc = subprocess.run([str(command), "--version"], text=True,
                              capture_output=True, timeout=15)
        return (proc.stdout + proc.stderr).strip().splitlines()[0]
    except (OSError, subprocess.TimeoutExpired, IndexError):
        return None


def run_load(runner, model, gpu, timeout):
    started = time.monotonic()
    proc = subprocess.run(
        [str(runner), "-m", str(model), "-p", "Compatibility probe:",
         "-n", "1", "--temp", "0", "--gpu", gpu] +
        (["--wait-for-vram", "300"] if gpu == "auto" else []),
        text=True, capture_output=True, timeout=timeout,
        # certification evidence is scalar-path (TC promotion is
        # tolerance-gated separately; see docs/compatibility-program.md)
        env=dict(os.environ, RUNNER_CUDA_TC="0", RUNNER_MOE_EAGER="1"))
    return {
        "status": "pass" if proc.returncode == 0 else "fail",
        "returncode": proc.returncode,
        "elapsed_ms": round((time.monotonic() - started) * 1000, 1),
        "stderr_tail": proc.stderr[-1000:],
    }


def resolve_model(entry, models_root):
    declared = Path(entry["file"])
    if declared.is_absolute() and declared.is_file():
        return declared.resolve(), None
    local = ROOT / declared
    if local.is_file():
        return local.resolve(), None
    matches = sorted(models_root.rglob(declared.name)) if models_root.is_dir() else []
    matches = [path.resolve() for path in matches if path.is_file()]
    if len(matches) == 1:
        return matches[0], None
    if not matches:
        return None, "model_file_not_found"
    return None, "model_basename_ambiguous"


def run_command(command, timeout):
    started = time.monotonic()
    try:
        proc = subprocess.run(command, text=True, capture_output=True,
                              timeout=timeout)
    except OSError as exc:
        return {
            "status": "fail", "reason": "process_start_failed",
            "elapsed_ms": round((time.monotonic() - started) * 1000, 1),
            "error": str(exc),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "fail", "reason": "timeout",
            "elapsed_ms": round((time.monotonic() - started) * 1000, 1),
            "stdout_tail": (exc.stdout or "")[-2000:],
            "stderr_tail": (exc.stderr or "")[-2000:],
        }
    return {
        "status": "pass" if proc.returncode == 0 else "fail",
        "returncode": proc.returncode,
        "elapsed_ms": round((time.monotonic() - started) * 1000, 1),
        "stdout_tail": proc.stdout[-2000:], "stderr_tail": proc.stderr[-2000:],
    }


def run_tokenizer(model, reference, corpus, timeout, reference_ids=None):
    command = [sys.executable, str(TOKENIZER_SCRIPT),
               "--gguf", str(model)]
    if reference_ids is not None:
        command += ["--ref-ids", str(reference_ids)]
    else:
        command += ["--ref", reference]
    command += ["--corpus", str(corpus), "--expect", "0"]
    result = run_command(command, timeout)
    if result["status"] == "fail" and "strings differ" not in result.get("stdout_tail", ""):
        detail = (result.get("stdout_tail", "") + result.get("stderr_tail", "")).lower()
        if any(marker in detail for marker in (
                "access to model", "restricted", "entry not found",
                "tokenizer.json", "repository not found")):
            result["status"] = "not_executed"
            result["reason"] = "tokenizer_reference_unavailable"
    return result


def run_greedy(runner, reference, model, timeout):
    return run_command([
        sys.executable, str(GREEDY_SCRIPT), "--runner", str(runner),
        "--reference", str(reference), "--model", str(model),
        "--tokens", "8", "--port-base", "9300",
        "--startup-timeout", str(timeout), "--request-timeout", str(timeout),
    ], timeout * 2 + 30)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--runner", type=Path, default=ROOT / "runner")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--models-root", type=Path, default=ROOT / "models")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--gpu", choices=("auto", "off"), default="auto")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--verify-files", action="store_true")
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--execute-checks", action="store_true",
                        help="run supported declared checks, recording skips with reasons")
    # RNR-007: a report may only be called complete if every declared check
    # actually ran and passed. --require-complete makes an incomplete report a
    # gate failure; without it the report is still honest (unrun checks are
    # recorded as not_executed) but the exit code reflects only what ran.
    parser.add_argument("--require-complete", action="store_true")
    # Evidence is append-only: refuse to clobber an existing report file.
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing --out report (default: refuse)")
    args = parser.parse_args(argv)

    if args.out and args.out.exists() and not args.force:
        parser.error(
            f"refusing to overwrite existing evidence {args.out} "
            "(reports are append-only; write to a new name or pass --force)")

    manifest = load_manifest(args.manifest)
    selected = [m for m in manifest["models"]
                if not args.model or m["id"] in args.model]
    unknown = set(args.model) - {m["id"] for m in selected}
    if unknown:
        parser.error("unknown model(s): " + ", ".join(sorted(unknown)))

    report = {
        "schema_version": "xyntetik.runner.model-compat-report.v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": {"os": platform.system(), "machine": platform.machine()},
        "runner": {"path": str(args.runner),
                   "version": command_version(args.runner)},
        "reference": {"path": str(args.reference) if args.reference else None,
                      "version": command_version(args.reference) if args.reference else None},
        "models": [],
    }
    failed = False
    for entry in selected:
        path, path_reason = resolve_model(entry, args.models_root)
        item = {"id": entry["id"], "architecture": entry["architecture"],
                "file": entry["file"], "expected_sha256": entry["sha256"],
                "checks": {}}
        if path is None:
            item["file_status"] = "missing"
            item["file_reason"] = path_reason
            failed = True
        elif args.verify_files or args.load or args.execute_checks:
            item["resolved_file"] = str(path)
            actual = sha256(path)
            item["actual_sha256"] = actual
            item["file_status"] = "pass" if actual == entry["sha256"] else "fail"
            failed |= actual != entry["sha256"]
            if (args.load or args.execute_checks) and actual == entry["sha256"] and \
                    "load" in entry.get("checks", []):
                item["checks"]["load"] = run_load(
                    args.runner.resolve(), path.resolve(), args.gpu, args.timeout)
                failed |= item["checks"]["load"]["status"] != "pass"
            if args.execute_checks and actual == entry["sha256"]:
                declared = entry.get("checks", [])
                if "tokenizer" in declared:
                    tok_ref = entry.get("tokenizer_reference")
                    tok_ids = entry.get("tokenizer_reference_ids")
                    if tok_ids:
                        tok_ids = Path(tok_ids)
                        if not tok_ids.is_absolute():
                            tok_ids = ROOT / tok_ids
                    corpus = ROOT / "tests/fixtures/tokenizer-corpus.txt"
                    if not tok_ref and not tok_ids:
                        item["checks"]["tokenizer"] = {
                            "status": "not_executed", "reason": "tokenizer_reference_not_declared"}
                    elif tok_ids and not tok_ids.is_file():
                        item["checks"]["tokenizer"] = {
                            "status": "not_executed",
                            "reason": "tokenizer_reference_capture_not_found"}
                    elif not corpus.is_file():
                        item["checks"]["tokenizer"] = {
                            "status": "not_executed", "reason": "tokenizer_corpus_not_found"}
                    else:
                        item["checks"]["tokenizer"] = run_tokenizer(
                            path, tok_ref, corpus, args.timeout,
                            reference_ids=tok_ids)
                        failed |= item["checks"]["tokenizer"]["status"] != "pass"
                for cls, fn in (("cpu_cuda", run_cpu_cuda),
                                ("chat", run_chat),
                                ("tool", run_tool)):
                    if cls not in declared:
                        continue
                    cp = (entry.get("check_params") or {}).get(cls)
                    if not cp:
                        item["checks"][cls] = {
                            "status": "not_executed",
                            "reason": f"{cls}_params_not_declared"}
                        continue
                    item["checks"][cls] = fn(args.runner.resolve(), path.resolve(),
                                             cp, args.timeout)
                    failed |= item["checks"][cls]["status"] == "fail"
                if "greedy_reference" in declared:
                    if not args.reference:
                        item["checks"]["greedy_reference"] = {
                            "status": "not_executed", "reason": "reference_binary_not_provided"}
                    elif not args.reference.is_file():
                        item["checks"]["greedy_reference"] = {
                            "status": "not_executed", "reason": "reference_binary_not_found"}
                    else:
                        item["checks"]["greedy_reference"] = run_greedy(
                            args.runner.resolve(), args.reference.resolve(), path,
                            args.timeout)
                        failed |= item["checks"]["greedy_reference"]["status"] != "pass"
        else:
            item["file_status"] = "not_run"

        # Never let a declared check be silently absent from the report — an
        # omitted check reads as "fine". Every check the manifest declares but
        # this run did not execute is recorded explicitly as not_executed, and
        # the model is "complete" only when all of them ran and passed.
        declared = entry.get("checks", [])
        for name in declared:
            reason = (path_reason if path is None else
                      "sha256_mismatch" if item.get("file_status") == "fail" else
                      "check_class_not_executable" if args.execute_checks else
                      "check_not_requested")
            item["checks"].setdefault(
                name, {"status": "not_executed", "reason": reason})
        item["complete"] = bool(declared) and all(
            item["checks"][name].get("status") == "pass" for name in declared)
        report["models"].append(item)

    report["complete"] = bool(report["models"]) and all(
        m.get("complete") for m in report["models"])
    report["summary"] = {
        "models": len(report["models"]),
        "files": {},
        "checks": {},
    }
    for item in report["models"]:
        status = item["file_status"]
        report["summary"]["files"][status] = \
            report["summary"]["files"].get(status, 0) + 1
        for name, check in item["checks"].items():
            statuses = report["summary"]["checks"].setdefault(name, {})
            status = check["status"]
            statuses[status] = statuses.get(status, 0) + 1

    rendered = json.dumps(report, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered)
    print(rendered, end="")
    # Executed-check failures always fail the gate; incompleteness fails it only
    # when the caller demands a complete report.
    return 1 if failed or (args.require_complete and not report["complete"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
