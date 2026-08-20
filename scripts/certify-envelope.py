#!/usr/bin/env python3
"""Assemble a measured-envelope manifest (`<model>.envelope.json`) for an artifact.

Slice 1 of the certified-envelope gate (design:
xyntetik-suite/docs/plans/envelope-manifest-design.md). This does NOT run gates
or measure anything — it is an INDEX OVER EVIDENCE that already exists: it reads
`runner --caps` for the runtime identity, an optional compat report for the
per-model quality/load evidence, and the artifact's own sha, and emits one JSON
sidecar per the schema. Every field is traceable to the run that produced it,
and absent evidence is recorded as null rather than invented (H8/H10: a dated
measurement, never a standing property).

    scripts/certify-envelope.py --model models/X.gguf --runner ./runner \
        --compat-report docs/compat-reports/0.1.19-alpha-...json \
        --reference-sha <upstream-sha> --gpu off --kv f16 --ctx 4096 \
        --out models/X.envelope.json

Verdict is `certified` when the model's gate evidence passes, `outside-envelope`
when a check recorded a measured refusal/failure, and `experimental` when there
is no gate evidence for it. A `--reference-sha` is REQUIRED for a `certified`
verdict — a quality number relative to no named reference is not a measurement,
so its absence fails the artifact closed to `experimental` rather than blessing
it. Resolution is exact-match-only by design; there is no closest-class fallback.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def runner_caps(runner):
    proc = subprocess.run([str(runner), "--caps"], text=True,
                          capture_output=True, timeout=30)
    if proc.returncode != 0:
        raise SystemExit(f"runner --caps failed: {proc.stderr.strip()}")
    return json.loads(proc.stdout)


def kernel_set_identity(caps):
    """The envelope key for the runtime's compute kernels. On Metal the shader
    source sha is exact and automatic; the CPU kernel set is the build's ISA
    tier, which --caps does not hash, so it is recorded coarsely and flagged."""
    gpu = caps.get("gpu") or {}
    backend = gpu.get("backend", "cpu")
    ident = {"backend": backend, "os": caps.get("os"), "arch": caps.get("arch")}
    if gpu.get("shader_source_sha256"):
        ident["shader_source_sha256"] = gpu["shader_source_sha256"]
    else:
        # CPU-only build: the ISA tier decides the kernels; --caps has no hash
        # for it, so the class carries the arch and this is marked incomplete.
        ident["cpu_kernel_hash"] = None
    return ident


def find_model_evidence(report, sha, model_path):
    """Return the compat-report entry for this artifact, matched by sha then
    by filename basename, or None."""
    if not report:
        return None
    for m in report.get("models", []):
        if m.get("sha256") == sha:
            return m
    base = Path(model_path).name
    for m in report.get("models", []):
        if Path(m.get("file", "")).name == base or \
                Path(m.get("resolved_file", "")).name == base:
            return m
    return None


def summarize_checks(entry):
    """Reduce a compat-report model entry to (gate, checks, detail) — gate is
    'pass' only if every executed check passed and at least one ran.

    `pass_margin_qualified` (owner-ratified 2026-08-20) is a PASSING status: a
    MoE cpu_cuda run whose only divergences are routing near-ties inside bar-v2's
    tie band (dense models never reach it). It counts as pass for the gate, and
    its exception detail is surfaced in `detail` so the certificate REPORTS it
    rather than presenting a clean 9/9 it did not measure."""
    if not entry:
        return None, {}, {}
    PASSING = ("pass", "pass_margin_qualified")
    checks = {}
    detail = {}
    ran = passed = 0
    for name, res in (entry.get("checks") or {}).items():
        status = res.get("status") if isinstance(res, dict) else res
        checks[name] = status
        if isinstance(res, dict) and res.get("cpu_cuda_identity"):
            detail["cpu_cuda_identity"] = res["cpu_cuda_identity"]
        if status in PASSING or status == "fail":
            ran += 1
            passed += status in PASSING
    if ran == 0:
        return "not_executed", checks, detail
    return ("pass" if passed == ran else "fail"), checks, detail


def build_manifest(args):
    caps = runner_caps(args.runner)
    sha = sha256_file(args.model)
    report = json.loads(Path(args.compat_report).read_text()) \
        if args.compat_report else None
    entry = find_model_evidence(report, sha, args.model)
    gate, checks, detail = summarize_checks(entry)

    # Verdict resolution — sharpened three states (owner 2026-08-20):
    #   certified       — all cert gates pass + a reference sha.
    #   outside-envelope — a measured reason says this config SHOULD NOT RUN
    #                      (the model won't LOAD, or a check is an explicit refusal
    #                      like wont_fit/load_fatal). The runner REFUSES it.
    #   experimental    — anything else short of certified: the model LOADS and
    #                      runs but a stricter gate fails (cpu_cuda identity, chat,
    #                      ...), or there's no reference sha / no gate evidence. It
    #                      is USABLE, so it loads with a banner — never refused.
    # Reserving `outside-envelope` for won't-run keeps a serviceable model (a MoE
    # that fails byte-identity but passes fidelity) out of the refusal path.
    checks_raw = (entry or {}).get("checks") or {}
    def _stat(v):
        return v.get("status") if isinstance(v, dict) else v
    refused = (_stat(checks_raw.get("load")) == "fail"
               or any(_stat(v) in ("wont_fit", "load_fatal") for v in checks_raw.values())
               or any(isinstance(v, dict) and v.get("refusal") for v in checks_raw.values()))
    if gate == "pass" and args.reference_sha:
        verdict = "certified"
    elif refused:
        verdict = "outside-envelope"
    else:
        verdict = "experimental"

    return {
        "schema_version": "xyntetik.runner.envelope.v1",
        "artifact": {
            "sha256": sha,
            "path": Path(args.model).name,
            "architecture": (entry or {}).get("architecture"),
            "derivation": args.derivation,
            "reference": args.reference_sha,
        },
        "runtime": {
            "version": caps.get("version"),
            "kernel_set": kernel_set_identity(caps),
        },
        "hardware": {
            "class": f"{caps.get('arch')}/{(caps.get('gpu') or {}).get('backend','cpu')}"
                     f"/{round((caps.get('ram_bytes') or 0) / 1e9)}G",
            "cpu_cores": caps.get("cpu_cores"),
        },
        "config": {
            "gpu": args.gpu, "threads": args.threads,
            "kv": args.kv, "ctx_max": args.ctx,
        },
        "quality": {
            "gate": gate,
            "checks": checks,
            # Surface a margin-qualified cpu_cuda exception so `certified` never
            # presents a clean identity it did not measure. Absent for a strict
            # 9/9 pass or a non-MoE model.
            **({"cpu_cuda_identity": detail["cpu_cuda_identity"]}
               if detail.get("cpu_cuda_identity") else {}),
            "reference_report": Path(args.compat_report).name
            if args.compat_report else None,
        },
        "verdict": verdict,
        "measured": {
            "date": args.date,
            "source_report_schema": (report or {}).get("schema_version"),
        },
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--runner", default="./runner")
    ap.add_argument("--compat-report")
    ap.add_argument("--reference-sha",
                    help="upstream/parent artifact sha the quality was measured "
                         "against; required for a 'certified' verdict")
    ap.add_argument("--derivation", help="human derivation chain, e.g. "
                    "'keep-30 prune of <sha>, plan <sha>'")
    ap.add_argument("--gpu", default="off")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--kv", default="f16")
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--date", default="", help="measurement date (YYYY-MM-DD); "
                    "pass it in, this script does not read the clock")
    ap.add_argument("--out")
    args = ap.parse_args(argv)

    if not Path(args.model).exists():
        sys.exit(f"model not found: {args.model}")
    manifest = build_manifest(args)
    text = json.dumps(manifest, indent=2) + "\n"
    if args.out:
        Path(args.out).write_text(text)
        print(f"wrote {args.out}  (verdict: {manifest['verdict']})")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
