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
    args = parser.parse_args(argv)

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
        report["models"].append(item)

    rendered = json.dumps(report, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered)
    print(rendered, end="")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
