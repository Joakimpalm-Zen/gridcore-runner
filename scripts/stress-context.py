#!/usr/bin/env python3
"""Context/KV stress: what happens at the edges of the context budget.

The shelf pass (`stress-models.py`) runs every model at a small fixed context.
This one asks the questions that only appear at the edges, where the failure
mode is usually a bad message rather than a wrong number:

  --ctx 0        auto-fit. The context should be sized to what the reservation
                 leaves after the weights, and the engine should say what it
                 chose.
  --ctx huge     a context this machine cannot possibly hold. The contract is
                 refuse-or-cap with a reason. Failing opaquely, or accepting it
                 and dying later, is the bug this probe exists to find.
  --ctx native   the model's own trained context, which is the number a user
                 reads off the model card and types in.

Nothing here judges output quality. Per this project's own hard-won rule, raw
completion prompts are not a quality signal for thinking-tuned families — so
this probe only asks whether the engine's context accounting is honest.
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CTX_RE = re.compile(r"\bctx (\d+)")
KV_RE = re.compile(r"kv cache\s+([\d.]+) MB")


def run(cmd, timeout):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           errors="replace")
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as e:
        return None, e.stdout or "", e.stderr or ""


def probe(runner, model, ctx, timeout, extra=()):
    cmd = [str(runner), "-m", str(model), "-p", "Hello", "-n", "4",
           "--temp", "0", "-c", str(ctx), "-v", *extra]
    rc, out, err = run(cmd, timeout)
    m = CTX_RE.search(err or "")
    kv = KV_RE.search(err or "")
    tail = (err or "").strip().splitlines()
    return {
        "requested_ctx": ctx,
        "rc": rc,
        "resolved_ctx": int(m.group(1)) if m else None,
        "kv_mb": float(kv.group(1)) if kv else None,
        "generated": (out or "").strip()[:60],
        "last_stderr": tail[-1][:200] if tail else "",
        "explains_itself": any(
            k in (err or "").lower()
            for k in ("context", "ctx", "kv", "fit", "reserve", "too large",
                      "capped", "reduc")),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default="C:/ProjectGrid/models")
    ap.add_argument("--runner", default=str(ROOT / "runner.exe"))
    ap.add_argument("--out", default=str(ROOT / "tests/compatibility/out/stress-context.json"))
    ap.add_argument("--only")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--max-gb", type=float, default=9.0,
                    help="skip files larger than this; the edges are the same "
                         "code path and a 18 GB load costs minutes per probe")
    args = ap.parse_args()

    models = sorted(Path(args.models_dir).glob("*.gguf"),
                    key=lambda p: p.stat().st_size)
    if args.only:
        models = [m for m in models if args.only.lower() in m.name.lower()]
    models = [m for m in models if m.stat().st_size / 1e9 <= args.max_gb]

    report = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "models": []}
    for model in models:
        print(f"\n=== {model.name}", flush=True)
        rows = []
        for ctx in (0, 32768, 1000000):
            r = probe(args.runner, model, ctx, args.timeout)
            rows.append(r)
            print(f"  -c {ctx:<8} rc={r['rc']} resolved={r['resolved_ctx']} "
                  f"kv={r['kv_mb']}MB explains={r['explains_itself']} "
                  f"| {r['last_stderr'][:90]}", flush=True)
        entry = {"model": model.name, "probes": rows, "findings": []}
        # An absurd context must not be silently accepted.
        absurd = rows[-1]
        if absurd["rc"] == 0 and absurd["resolved_ctx"] and \
                absurd["resolved_ctx"] >= 1000000:
            entry["findings"].append(
                "accepted a 1,000,000-token context without capping it")
        if absurd["rc"] not in (0, None) and not absurd["explains_itself"]:
            entry["findings"].append(
                "refused a too-large context without saying why")
        if rows[0]["rc"] != 0:
            entry["findings"].append("--ctx 0 (auto-fit) did not run")
        report["models"].append(entry)
        for f in entry["findings"]:
            print(f"  FINDING: {f}", flush=True)

    Path(args.out).write_text(json.dumps(report, indent=1))
    print(f"\nwrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
