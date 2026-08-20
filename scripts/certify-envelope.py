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

The optional TOOL-CALLING axis (`--truncation-report`, `--quant-fidelity-report`,
and `runner --tool-info`) is REPORTED-ONLY: it records a per-artifact tool-calling
guarantee (truncation-recovery, schema-shape-held, agent-torture, native protocol)
beside the fidelity verdict but never changes it. The differentiator as measured
data — each field null when its evidence is absent, never invented.
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


# --- tool-calling axis (reported-only; never changes the fidelity verdict) ----
# Each sub-block indexes evidence that already exists; absent evidence is null,
# never invented. The rollup gate reflects only the three MEASURABLE guarantees
# (truncation, schema-shape, agent-torture) — the native protocol is reported for
# information (a generic-envelope model is a valid, guaranteed path, not a fail).

def _load_json(path):
    return json.loads(Path(path).read_text()) if path else None


def tool_info(runner, model):
    """Per-model native tool protocol from `runner --tool-info` (template.c is
    the single source of truth). None if the runtime lacks the flag or the call
    fails — never guessed from the architecture."""
    try:
        proc = subprocess.run([str(runner), "--tool-info", "-m", str(model)],
                              text=True, capture_output=True, timeout=60)
        if proc.returncode != 0 or not proc.stdout.strip():
            return None
        info = json.loads(proc.stdout.strip().splitlines()[-1])
    except (OSError, subprocess.TimeoutExpired, ValueError, IndexError):
        return None
    fam = info.get("tool_family")
    if not fam:
        return None
    return {"tool_family": fam, "native": bool(info.get("native_tool_protocol"))}


def truncation_axis(report):
    """Truncation-recovery: an ENGINE property (measured on a proxy model), so it
    is labelled `scope: engine` — presenting it as artifact-measured would
    overclaim (H8/H10)."""
    if not report:
        return None
    rungs = report.get("rungs") or []
    control = max((r.get("max_tokens", 0) for r in rungs), default=0)
    truncated = [r for r in rungs if r.get("max_tokens", 0) < control]
    passed = sum(1 for r in truncated
                 if r.get("tool_calls_present") and r.get("arguments_parseable"))
    rt = report.get("runtime") or {}
    cfg = report.get("configuration") or {}
    return {
        "holds": bool((report.get("runner_property") or {}).get("holds")),
        "rungs_passed": passed,
        "rungs_total": len(truncated),
        "source_schema": report.get("schema_version"),
        "measured": {"scope": "engine",
                     "runtime_version": rt.get("version") or rt.get("runner"),
                     "model": cfg.get("model") or rt.get("model")},
    }


def schema_shape_axis(report, quant):
    """Schema shape + tool selection held down to `quant`. NOTE recorded
    explicitly: shape and selection hold at low quants; argument VALUES do not
    (they need Q6_K+), so the certificate never implies full tool calling at
    Q4_0."""
    if not report:
        return None
    row = next((v for v in (report.get("variants") or [])
                if v.get("quant") == quant or v.get("label") == quant), None)
    if not row:
        return None
    tf = row.get("tool_fidelity") or {}

    def _rate(k):
        return (tf.get(k) or {}).get("rate")
    return {
        "held_to_quant": quant,
        "schema_conformance_rate": _rate("schema_conformance"),
        "tool_selection_rate": _rate("tool_selection"),
        "note": "schema shape and tool selection hold at this quant; "
                "argument VALUES need Q6_K+",
        "source_schema": report.get("schema_version"),
        "measured": {"model": report.get("model") or row.get("label")},
    }


def agent_torture_axis(entry, report_name):
    """Per-artifact tool conformance, from the compat report's `checks.tool`
    (sha-keyed evidence the certifier already reads). Accepts the old plain-string
    status and the newer {status, totals} dict."""
    if not entry:
        return None
    tool = (entry.get("checks") or {}).get("tool")
    if tool is None:
        return None
    if isinstance(tool, dict):
        status, totals = tool.get("status"), (tool.get("totals") or {})
    else:
        status, totals = tool, {}
    if status not in ("pass", "fail"):
        return None
    return {
        "gate": status,
        "requests": totals.get("requests"),
        "passed": totals.get("passed"),
        "failed": totals.get("failed"),
        "source_schema": "xyntetik.agent-torture.v4",
        "measured": {"reference_report": report_name},
    }


def build_tool_calling(trunc, shape, torture, native):
    """Assemble the reported-only axis. gate is `pass` only when every PRESENT
    guarantee block passes AND all three are present; `partial` when guarantee
    evidence is missing; `fail` when a present guarantee block fails. The native
    protocol is reported but does not gate. None when there is no evidence at all
    (the manifest simply omits the axis)."""
    blocks = {"truncation_recovery": trunc, "schema_shape": shape,
              "agent_torture": torture, "native_tool_protocol": native}
    if not any(blocks.values()):
        return None
    GATE = ("truncation_recovery", "schema_shape", "agent_torture")

    def _ok(k, v):
        if k == "truncation_recovery":
            return bool(v.get("holds")) and v.get("rungs_passed") == v.get("rungs_total")
        if k == "schema_shape":
            return v.get("schema_conformance_rate") == 1.0 and \
                v.get("tool_selection_rate") == 1.0
        if k == "agent_torture":
            return v.get("gate") == "pass"
        return False
    present = {k: blocks[k] for k in GATE if blocks.get(k)}
    if any(not _ok(k, v) for k, v in present.items()):
        gate = "fail"
    elif len(present) == len(GATE):
        gate = "pass"
    else:
        gate = "partial"
    return {**blocks, "gate": gate}


def build_manifest(args):
    caps = runner_caps(args.runner)
    sha = sha256_file(args.model)
    report = json.loads(Path(args.compat_report).read_text()) \
        if args.compat_report else None
    entry = find_model_evidence(report, sha, args.model)
    gate, checks, detail = summarize_checks(entry)

    # Reported-only tool-calling axis, indexed from evidence that already exists.
    tool_calling = build_tool_calling(
        truncation_axis(_load_json(args.truncation_report)),
        schema_shape_axis(_load_json(args.quant_fidelity_report),
                          args.quant_fidelity_quant),
        agent_torture_axis(
            entry, Path(args.compat_report).name if args.compat_report else None),
        tool_info(args.runner, args.model))

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
        **({"tool_calling": tool_calling} if tool_calling else {}),
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
    ap.add_argument("--truncation-report",
                    help="xyntetik.truncation-benchmark.v1 report.json — the "
                    "ENGINE truncation-recovery evidence (labelled scope:engine, "
                    "measured on a proxy model)")
    ap.add_argument("--quant-fidelity-report",
                    help="xyntetik.quant-fidelity.v1 report.json for the "
                    "schema-shape-held claim")
    ap.add_argument("--quant-fidelity-quant", default="Q4_0",
                    help="which quant row of the fidelity report the schema-shape "
                    "claim reads (default Q4_0)")
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
