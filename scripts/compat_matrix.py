#!/usr/bin/env python3
"""Run and record the pinned real-model compatibility matrix.

The manifest is committed, but model files are not.  A run never silently
substitutes a similarly named GGUF: its SHA-256 must match before execution.
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "tests" / "compatibility" / "models.json"


def load_manifest(path):
    data = json.loads(Path(path).read_text())
    if data.get("schema_version") != "gridcore.runner.model-compat.v1":
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
         "-n", "1", "--temp", "0", "--gpu", gpu],
        text=True, capture_output=True, timeout=timeout)
    return {
        "status": "pass" if proc.returncode == 0 else "fail",
        "returncode": proc.returncode,
        "elapsed_ms": round((time.monotonic() - started) * 1000, 1),
        "stderr_tail": proc.stderr[-1000:],
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--runner", type=Path, default=ROOT / "runner")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--gpu", choices=("auto", "off"), default="auto")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--verify-files", action="store_true")
    parser.add_argument("--load", action="store_true")
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
        "schema_version": "gridcore.runner.model-compat-report.v1",
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
        path = ROOT / entry["file"]
        item = {"id": entry["id"], "architecture": entry["architecture"],
                "file": entry["file"], "expected_sha256": entry["sha256"],
                "checks": {}}
        if not path.is_file():
            item["file_status"] = "missing"
            failed = True
        elif args.verify_files or args.load:
            actual = sha256(path)
            item["actual_sha256"] = actual
            item["file_status"] = "pass" if actual == entry["sha256"] else "fail"
            failed |= actual != entry["sha256"]
            if args.load and actual == entry["sha256"]:
                item["checks"]["load"] = run_load(
                    args.runner.resolve(), path.resolve(), args.gpu, args.timeout)
                failed |= item["checks"]["load"]["status"] != "pass"
        else:
            item["file_status"] = "not_run"

        # Never let a declared check be silently absent from the report — an
        # omitted check reads as "fine". Every check the manifest declares but
        # this run did not execute is recorded explicitly as not_executed, and
        # the model is "complete" only when all of them ran and passed.
        declared = entry.get("checks", [])
        for name in declared:
            item["checks"].setdefault(name, {"status": "not_executed"})
        item["complete"] = bool(declared) and all(
            item["checks"][name].get("status") == "pass" for name in declared)
        report["models"].append(item)

    report["complete"] = bool(report["models"]) and all(
        m.get("complete") for m in report["models"])

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
