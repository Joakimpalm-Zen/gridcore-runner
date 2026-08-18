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


def notes_from(stderr):
    """Every line the engine wrote to explain itself, in order.

    `last_stderr` is the throughput line, so a probe that kept only that could
    not see the one message that answers its own question — the note naming
    what a large -c cost in offloaded layers.
    """
    return [ln.strip() for ln in (stderr or "").splitlines()
            if ln.strip().startswith(("note:", "warning:", "error:"))]


def refused_architecture(rows):
    """True when the model never reached context handling at all.

    An unsupported architecture is refused by name before any context is
    sized, so scoring that refusal against context keywords says nothing about
    context accounting. It is a classification, not a finding.
    """
    return bool(rows) and all(
        "unsupported architecture" in (r.get("last_stderr") or "").lower()
        for r in rows)


def findings_for(rows):
    """Context-accounting findings, or none if context was never reached."""
    if refused_architecture(rows):
        return []
    out = []
    absurd = rows[-1]
    # An absurd context must not be silently accepted.
    if absurd["rc"] == 0 and absurd["resolved_ctx"] and \
            absurd["resolved_ctx"] >= 1000000:
        out.append("accepted a 1,000,000-token context without capping it")
    if absurd["rc"] not in (0, None) and not absurd["explains_itself"]:
        out.append("refused a too-large context without saying why")
    if rows[0]["rc"] != 0:
        out.append("--ctx 0 (auto-fit) did not run")
    return out


def probe(runner, model, ctx, timeout, extra=()):
    cmd = [str(runner), "-m", str(model), "-p", "Hello", "-n", "4",
           "--temp", "0", "-c", str(ctx), "-v", *extra]
    rc, out, err = run(cmd, timeout)
    m = CTX_RE.search(err or "")
    kv = KV_RE.search(err or "")
    tail = (err or "").strip().splitlines()
    notes = notes_from(err)
    # Scored over the engine's OWN explanation lines, never the whole stream.
    # `-v` is always passed above, and every successful load prints "context N
    # (train M)", "kv cache N MB" and a banner ending "| ctx N |" before
    # anything downstream can go wrong -- so a raw-stderr match made
    # explains_itself true for every model that reached the engine at all, and
    # the probe's headline finding ("refused without saying why") could not
    # fire on any post-load failure however opaque.
    explained = any(k in " ".join(notes).lower()
                    for k in ("context", "ctx", "kv", "fit", "reserve",
                              "too large", "capped", "reduc"))
    return {
        "requested_ctx": ctx,
        "rc": rc,
        "resolved_ctx": int(m.group(1)) if m else None,
        "kv_mb": float(kv.group(1)) if kv else None,
        "generated": (out or "").strip()[:60],
        "last_stderr": tail[-1][:200] if tail else "",
        "notes": notes,
        "explains_itself": explained,
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
        entry = {"model": model.name, "probes": rows,
                 "refused_architecture": refused_architecture(rows),
                 "findings": findings_for(rows)}
        report["models"].append(entry)
        if entry["refused_architecture"]:
            print("  refused at the architecture gate — context never reached",
                  flush=True)
        for f in entry["findings"]:
            print(f"  FINDING: {f}", flush=True)

    Path(args.out).write_text(json.dumps(report, indent=1))
    print(f"\nwrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
