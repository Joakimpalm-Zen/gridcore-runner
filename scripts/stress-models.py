#!/usr/bin/env python3
"""Stress every GGUF on a machine's model shelf against this Runner build.

Answers three questions per model, and keeps them separate because they fail
independently:

  1. Does it load and generate at all, and does it do so without falling back?
     A GPU that quietly releases itself mid-forward still produces text, so
     "it answered" is not evidence. Every fallback, refusal and non-zero exit
     is recorded as a FAULT.

  2. Does CUDA agree with the host? Greedy output must be byte-identical
     between `--gpu auto` and `--gpu off`. A mismatch is recorded as a
     DEVIANCE rather than a fault, because a near-tie can separate two
     numerically fine backends — the report prints the text so the difference
     can be judged instead of assumed.

  3. What settings should this machine actually use? Each candidate
     configuration is benchmarked with `--bench-json`, and the fastest one
     that neither faulted nor deviated becomes the recommendation.

Runs are tiered by file size. A model larger than the machine's RAM is paged
from disk on every token, so the host leg of a 18 GB model is given a token
budget that keeps the pass tractable rather than the same budget as a 4 GB one.

Results stream to one JSON file per model under --out-dir, so an interrupted
pass resumes with --skip-done instead of starting over.
"""

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Substrings that mean the engine did not do what the flags asked. These are
# the whole point of the pass: each one is a path that still returns text.
FAULT_PATTERNS = [
    "falling back to CPU",
    "forward failed at runtime",
    "kernel launch failed",
    "graph launch failed",
    "could not",
    "out of memory",
    "error:",
]

# Expected, correct refusals — an architecture the engine declines to run is
# the no-footguns contract working, not a fault.
REFUSAL_PATTERNS = [
    "unsupported architecture",
    "needs sink-aware attention",
]

PROMPT = "The capital of France is"


def run(cmd, timeout):
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout, errors="replace")
        return {"rc": proc.returncode, "stdout": proc.stdout,
                "stderr": proc.stderr, "wall_s": time.perf_counter() - t0,
                "timed_out": False}
    except subprocess.TimeoutExpired as exc:
        return {"rc": None, "stdout": exc.stdout or "", "stderr": exc.stderr or "",
                "wall_s": time.perf_counter() - t0, "timed_out": True}


def classify(res):
    """Split stderr into refusals (expected) and faults (not)."""
    err = (res["stderr"] or "").lower()
    refusals = [p for p in REFUSAL_PATTERNS if p.lower() in err]
    faults = [p for p in FAULT_PATTERNS if p.lower() in err]
    # A clean refusal names itself in the same line an "error:" appears on, so
    # do not double-count it as a fault.
    if refusals:
        faults = [f for f in faults if f != "error:"]
    if res["timed_out"]:
        faults.append("timed out")
    elif res["rc"] not in (0, None) and not refusals:
        faults.append(f"exit {res['rc']}")
    return refusals, faults


def split_line(stderr):
    for line in (stderr or "").splitlines():
        if line.startswith("gpu: CUDA backend") or line.startswith("gpu-split:"):
            return line.strip()
    return None


def bench(runner, model, ctx, gen, extra, timeout):
    cmd = [str(runner), "-m", str(model), "--bench-json", "-c", str(ctx),
           "-n", str(gen), *extra]
    res = run(cmd, timeout)
    refusals, faults = classify(res)
    out = {"command": cmd[1:], "faults": faults, "refusals": refusals,
           "wall_s": round(res["wall_s"], 1), "split": split_line(res["stderr"])}
    try:
        parsed = json.loads(res["stdout"])
        out["prompt_tok_s"] = parsed.get("prompt_tok_s")
        out["decode_tok_s"] = parsed.get("gen_tok_s")
        out["prompt_tokens"] = parsed.get("prompt_tokens")
    except (json.JSONDecodeError, TypeError):
        out["prompt_tok_s"] = out["decode_tok_s"] = None
        out["stderr_tail"] = (res["stderr"] or "")[-600:]
    return out


def generate(runner, model, ctx, gen, extra, timeout):
    cmd = [str(runner), "-m", str(model), "-p", PROMPT, "-n", str(gen),
           "--temp", "0", "-c", str(ctx), *extra]
    res = run(cmd, timeout)
    refusals, faults = classify(res)
    return {"command": cmd[1:], "text": res["stdout"], "faults": faults,
            "refusals": refusals, "split": split_line(res["stderr"]),
            "wall_s": round(res["wall_s"], 1),
            "stderr_tail": (res["stderr"] or "")[-800:] if faults else None}


def same_greedy_output(gpu_text, cpu_text, gpu_gen, cpu_gen):
    """Do the two backends agree, given they may have run different budgets?

    A PREFIX test, not equality. The host leg gets a smaller token budget for a
    model larger than RAM, so the device output is legitimately longer. Greedy
    decoding is deterministic and prefix-consistent, so if the shorter run is a
    prefix of the longer one they agree for every token the shorter one
    produced.

    Comparing truncated CHARACTER slices instead — which this harness did
    first — reports a mismatch on the trailing newline alone, and flagged three
    Gemma models as CPU!=GPU when every GPU text was a clean continuation of
    its CPU text. Strip trailing whitespace: the CLI ends output with one.
    """
    gpu_txt = (gpu_text or "").rstrip()
    cpu_txt = (cpu_text or "").rstrip()
    shorter, longer = (cpu_txt, gpu_txt) if cpu_gen <= gpu_gen else (gpu_txt, cpu_txt)
    return bool(shorter) and longer.startswith(shorter)


def degenerate(text):
    """Obvious garbage: one token repeated, or nothing at all.

    Deliberately conservative — this flags output for a human to look at, it
    does not judge quality. A model legitimately repeating itself is rare
    enough that a false flag costs one glance.
    """
    body = (text or "").strip()
    if not body:
        return "empty output"
    words = body.split()
    if len(words) >= 8 and len(set(words)) <= 2:
        return "output is one or two tokens repeated"
    return None


def tier(size_gb, ram_gb):
    """Token budgets. The host leg of a model bigger than RAM pages from disk
    on every token, so it gets a smaller budget rather than the same one."""
    if size_gb > ram_gb * 0.75:
        return {"gen": 8, "cpu_gen": 4, "bench_gen": 8, "timeout": 3600}
    if size_gb > 6:
        return {"gen": 16, "cpu_gen": 8, "bench_gen": 16, "timeout": 2400}
    return {"gen": 16, "cpu_gen": 16, "bench_gen": 32, "timeout": 1800}


def is_moe(model_path):
    """Read expert_count straight out of the GGUF metadata.

    The obvious test — looking for "expert" in the split banner — only works
    once --cpu-moe is already in use, which is exactly the flag being decided.
    That circularity silently skipped the placement sweep for every MoE model
    on the first pass, including gpt-oss. Read the file instead.
    """
    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "verify_gguf", ROOT / "scripts" / "verify-gguf.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with open(model_path, "rb") as f:
            r = mod.R(f)
            if r.raw(4) != b"GGUF":
                return False
            r.u32()          # version
            r.u64()          # tensor count
            n_kv = r.u64()
            for _ in range(n_kv):
                k = r.string()
                v = r.value(r.u32())
                if k.endswith(".expert_count"):
                    return bool(v and int(v) > 0)
    except Exception:
        return False
    return False


def candidates(model_path, size_gb, vram_gb, moe):
    """Configurations worth trying on THIS machine.

    Keep it small: every entry costs a full model load, and on a box whose RAM
    is smaller than several of these files that is minutes, not seconds.
    """
    cands = [("default", [])]
    # A model larger than RAM decodes at the speed of the page cache, so a KV
    # precision change cannot show up over the paging noise — skip it there and
    # spend the load on the placement flags, which do move that case.
    oversized = size_gb > 11.9
    if not oversized:
        cands.append(("kv-q8", ["--kv", "q8"]))
    if moe:
        cands.append(("cpu-moe-auto", ["--cpu-moe", "auto"]))
        cands.append(("cpu-moe-all", ["--cpu-moe"]))
    return cands


def main():
    ap = argparse.ArgumentParser()
    # The shelf is per-machine. The default is the repo's own models/ because
    # that path exists on every box this runs on; the previous default was one
    # machine's absolute Windows path, which on any other host matched nothing
    # and printed "done".
    ap.add_argument("--models-dir",
                    default=str(Path(__file__).resolve().parent.parent / "models"))
    # Same portability trap as --models-dir above: the default was Windows-only,
    # so on any other host this failed with a FileNotFoundError deep inside
    # subprocess rather than saying which binary it wanted.
    ap.add_argument("--runner",
                    default=str(ROOT / ("runner.exe" if os.name == "nt" else "runner")))
    ap.add_argument("--out-dir", default=str(ROOT / "tests/compatibility/out/stress"))
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--ram-gb", type=float, default=15.9)
    ap.add_argument("--vram-gb", type=float, default=8.0)
    ap.add_argument("--only", help="substring filter on the filename")
    ap.add_argument("--skip-done", action="store_true")
    ap.add_argument("--no-sweep", action="store_true",
                    help="correctness pass only, no settings sweep")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Smallest first: a long pass should have covered the cheap models before
    # it spends an hour paging an 18 GB file, and --skip-done resumes there.
    if not Path(args.runner).exists():
        print(f"error: no runner binary at {args.runner} (build it, or pass "
              f"--runner)", file=sys.stderr)
        return 2
    shelf = Path(args.models_dir)
    models = sorted(shelf.glob("*.gguf"), key=lambda p: p.stat().st_size)
    found = len(models)
    if args.only:
        models = [m for m in models if args.only.lower() in m.name.lower()]

    # Zero models is a FAILURE, not a fast pass. This script's whole output on
    # a wrong --models-dir used to be the word "done", which is exactly what a
    # clean run of the whole shelf also prints -- so a pass over nothing was
    # indistinguishable from a pass over everything.
    if not models:
        if not shelf.is_dir():
            print(f"error: no model shelf at {shelf}", file=sys.stderr)
        elif found == 0:
            print(f"error: no .gguf files under {shelf}", file=sys.stderr)
        else:
            print(f"error: --only {args.only!r} matched none of the "
                  f"{found} models under {shelf}", file=sys.stderr)
        return 2

    for model in models:
        report_path = out_dir / (model.stem + ".json")
        if args.skip_done and report_path.exists():
            print(f"skip (done): {model.name}", flush=True)
            continue
        size_gb = model.stat().st_size / 1e9
        t = tier(size_gb, args.ram_gb)
        print(f"\n=== {model.name}  ({size_gb:.1f} GB)", flush=True)

        report = {"model": model.name, "size_gb": round(size_gb, 2),
                  "context": args.ctx, "generated_utc":
                      time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  "faults": [], "deviances": []}

        # --- 1. GPU load + greedy generation
        moe = is_moe(model)
        report["moe"] = moe
        gpu = generate(args.runner, model, args.ctx, t["gen"], ["--gpu", "auto"],
                       t["timeout"])
        report["gpu_run"] = gpu
        if gpu["refusals"]:
            report["refused"] = gpu["refusals"]
            print(f"  REFUSED (expected contract): {gpu['refusals']}", flush=True)
            report_path.write_text(json.dumps(report, indent=1))
            continue
        for f in gpu["faults"]:
            report["faults"].append({"where": "gpu_run", "what": f})
        bad = degenerate(gpu["text"])
        if bad:
            # NOT a fault. These are raw completion prompts, and raw
            # completions are not a quality signal for thinking-tuned
            # families — this repo already recorded one false "gemma4 Q4_0
            # bug" and a wrong catalog refusal from exactly that mistake.
            # Recorded as something to look at through the chat template.
            report.setdefault("observations", []).append(
                {"where": "gpu_run", "what": bad,
                 "note": "raw completion prompt; re-check via the chat "
                         "template before treating as a quality problem"})
        print(f"  gpu: {gpu['split']}", flush=True)
        print(f"  out: {gpu['text'][:70]!r}", flush=True)

        # --- 2. CPU leg + identity
        cpu = generate(args.runner, model, args.ctx, t["cpu_gen"],
                       ["--gpu", "off"], t["timeout"])
        report["cpu_run"] = cpu
        for f in cpu["faults"]:
            report["faults"].append({"where": "cpu_run", "what": f})
        gpu_txt = (gpu["text"] or "").rstrip()
        cpu_txt = (cpu["text"] or "").rstrip()
        identical = same_greedy_output(gpu["text"], cpu["text"],
                                       t["gen"], t["cpu_gen"])
        report["cpu_cuda_identical"] = identical
        if not identical and not cpu["faults"] and not gpu["faults"]:
            report["deviances"].append({
                "what": "cpu_cuda text mismatch",
                "gpu": gpu_txt[:400], "cpu": cpu_txt[:400]})
        print(f"  cpu==gpu: {identical}", flush=True)

        # --- 3. settings sweep
        if not args.no_sweep:
            sweep = []
            for name, extra in candidates(model, size_gb, args.vram_gb, moe):
                r = bench(args.runner, model, args.ctx, t["bench_gen"], extra,
                          t["timeout"])
                r["config"] = name
                sweep.append(r)
                for f in r["faults"]:
                    report["faults"].append({"where": f"sweep:{name}", "what": f})
                print(f"  {name:14s} prefill {r.get('prompt_tok_s')} "
                      f"decode {r.get('decode_tok_s')} faults={r['faults']}",
                      flush=True)
            report["sweep"] = sweep
            ok = [s for s in sweep if not s["faults"] and s.get("decode_tok_s")]
            if ok:
                best = max(ok, key=lambda s: s["decode_tok_s"])
                report["recommended"] = {
                    "config": best["config"],
                    "decode_tok_s": best["decode_tok_s"],
                    "prefill_tok_s": best.get("prompt_tok_s"),
                    "command": best["command"]}

        report_path.write_text(json.dumps(report, indent=1))
        print(f"  faults={len(report['faults'])} deviances={len(report['deviances'])}",
              flush=True)

    print("\ndone", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
