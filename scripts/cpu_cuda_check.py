#!/usr/bin/env python3
"""CPU vs CUDA byte-identity gate for one model, over several prompts.

The compatibility program's `cpu_cuda` check: the same greedy generation must
come out byte-for-byte identical whether the forward pass ran on the host or
on the device.

Each backend is loaded ONCE and driven over HTTP, rather than re-launching the
CLI per prompt. That is not a style preference: on a box whose RAM is smaller
than the model, alternating two processes over the same mmap evicts the page
cache on every switch, so a ten-launch run re-reads tens of gigabytes and
measures the disk instead of the engine.

MoE byte identity is defined over the EAGER routing path
(`RUNNER_MOE_EAGER=1`), because the fused path's device `expf` differs from the
host libm by 1-2 ulp by construction — so this harness pins it, exactly as the
other certification harnesses do. `RUNNER_CUDA_TC=0` pins the scalar GEMM for
the same reason (the tensor-core default is tolerance-gated, not identical).

Usage: cpu_cuda_check.py MODEL [--tokens N] [--runner PATH] [--fused]
Exit status is 0 only if every prompt matched.
"""
import argparse
import json
import os
import socket
import re
import subprocess
import sys
import time
import urllib.request
import pathlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# The five short prompts are the historical set (5 to 15 tokens). The four
# long ones were added 2026-08-15 for a specific reason: the TC prefill GEMM
# published uninitialised shared memory for token columns 16 and above from
# 2026-08-08 to 2026-08-13, and this gate ran in every certification through
# that window without catching it, because its longest prompt was FIFTEEN
# tokens. One more token and the bug would have been caught on day one.
#
# A CPU/CUDA identity gate whose prompts all fit in a single sub-tile silently
# exempts every batched prefill path from the contract it claims to enforce.
# These span roughly 24, 40, 64 and 96 tokens so the gate crosses the 16-column
# boundary, the 32-column boundary and a full 64-column tile, and so a future
# tile widening cannot hide behind short prompts either.
PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "1 2 3 4 5 6 7 8",
    "Once upon a time, in a land far away,",
    "Q: What is 17 * 23?\nA:",
    # ~24 tokens: crosses the first tile boundary
    "The lighthouse keeper recorded the weather every morning: wind "
    "direction, cloud cover, and the hour the fog lifted.",
    # ~40 tokens: crosses 32
    "In 1929 the observatory published a revised catalogue listing 4218 "
    "objects, of which roughly one in nine turned out on later inspection "
    "to be a duplicate entry under a second designation. The correction",
    # ~64 tokens: a full tile
    "def parse_header(buf, size):\n"
    "    if size < 8:\n"
    "        raise ValueError('short header')\n"
    "    magic, version = struct.unpack('<II', buf[:8])\n"
    "    if magic != GGUF_MAGIC:\n"
    "        raise ValueError('not a gguf file')\n"
    "    return magic, version\n\n"
    "# The loader above rejects a truncated header before unpacking it, which",
    # ~96 tokens: past one tile, into a second
    "The city of Lisbon sits on seven hills above the Tagus estuary, and its "
    "oldest quarter survived the 1755 earthquake largely intact because the "
    "bedrock there is firmer than the reclaimed ground downriver. Rebuilding "
    "the lower town took decades, and the grid of streets laid out afterwards "
    "was among the first in Europe designed with seismic loads in mind, which "
    "is why the district still reads as unusually regular to a visitor who",
]


# ---- MoE margin-qualified tolerance (owner-ratified 2026-08-20) ------------
#
# Dense models keep strict byte-identity. But top-k MoE routing has a hard
# ceiling: a genuine near-tie between two expert gates flips under CPU-vs-CUDA
# reduction-order FP rounding (warp-tree vs SIMD-lane), and that flip cascades
# into the token stream. Forcing the CUDA reduction to mimic host libm summation
# order purely to pass a byte-equality gate tests floating-point EXECUTION ORDER,
# not correctness. So a divergence is TOLERATED, and only then, when ALL hold:
#   (1) near-tie: the first divergent token is a coin-flip the model itself could
#       barely separate — the pick_margin (reference=CPU) to the token the OTHER
#       side chose is within the band. Reuses bar-v2's EXACT definition
#       (scripts/kld-compare-raw.py): band 0.5 nats, measured to the picked token.
#   (2) fidelity: the model's independent bar-v2 fidelity gate passes (checked by
#       certify-envelope.py, not here).
#   (3) no systematic bias: the tolerated flips are a few INDEPENDENT near-ties,
#       not a pattern — capped at NEAR_TIE_CAP of the probes; more than that reads
#       as a real skew, not routing chaos, and FAILS.
#   (4) exactness elsewhere: any divergence whose first flip is NOT a near-tie is
#       a hard FAIL — a real, non-coin-flip difference.
# The result is REPORTED, never hidden: `pass` (all exact) / `pass_margin_qualified`
# (some near-ties tolerated) / `fail`.
DEFAULT_TIE_BAND = 0.5   # nats; mirror of scripts/kld-compare-raw.py
NEAR_TIE_CAP = 0.34      # tolerate near-ties in at most ~1/3 of probes (cond. 3)
TOP_N = 20               # request the model's top-N logprobs per token


def is_moe(model_path):
    """expert_count straight from the GGUF header (the same read as
    scripts/stress-models.py, via verify-gguf.py's reader).

    The margin-qualified tolerance exists for MoE ROUTING near-ties only — a
    dense model has no router whose reduction order could flip a coin, so a
    dense divergence is a real defect even when the logprob gap happens to sit
    inside the band, and it must stay under strict byte-identity. Fail-closed:
    an unreadable header classifies as dense, which only makes the gate
    stricter, never looser."""
    try:
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


def dist_at(lp, i):
    """The token->logprob distribution at position i of a runner logprobs block
    (parallel tokens / token_logprobs / top_logprobs arrays)."""
    d = dict(lp["top_logprobs"][i])
    d[lp["tokens"][i]] = lp["token_logprobs"][i]
    return d


def pick_margin(dist_ref, token):
    """Reference-side gap between its own best logprob and `token`'s (bar-v2).
    None when the reference never reported `token` (unmeasured != tied)."""
    if token not in dist_ref:
        return None
    return max(dist_ref.values()) - dist_ref[token]


def classify_divergence(cpu, gpu, band=DEFAULT_TIE_BAND):
    """Compare two runner logprobs blocks token-by-token. Returns
    (kind, detail): kind in {'exact','near_tie','real'}. A near-tie is the first
    divergent token being a coin-flip the CPU distribution rates within `band`
    of the GPU's pick (and vice-versa, so neither side is confidently right)."""
    ct, gt = cpu["tokens"], gpu["tokens"]
    n = min(len(ct), len(gt))
    i = 0
    while i < n and ct[i] == gt[i]:
        i += 1
    if i == n and len(ct) == len(gt):
        return "exact", None
    if i >= n:  # one ran shorter (EOS) but agreed up to there — a real length split
        return "real", {"pos": i, "reason": "length"}
    cd, gd = dist_at(cpu["logprobs"], i), dist_at(gpu["logprobs"], i)
    m_cpu = pick_margin(cd, gt[i])   # how far CPU rates GPU's pick below its own best
    m_gpu = pick_margin(gd, ct[i])   # and symmetrically
    detail = {"pos": i, "cpu_token": ct[i], "gpu_token": gt[i],
              "cpu_margin_to_gpu_pick": m_cpu, "gpu_margin_to_cpu_pick": m_gpu}
    # near-tie only if BOTH sides could barely separate the two competitors —
    # a divergence where one side is confident (or the other's pick is off its
    # top-N list, margin None) is a real difference.
    if m_cpu is not None and m_gpu is not None and m_cpu <= band and m_gpu <= band:
        return "near_tie", detail
    return "real", detail


def run_result(near_tie, real, total):
    """Fold per-prompt kinds into the run verdict. Any real (non-near-tie)
    divergence fails (cond. 4); near-ties across more than NEAR_TIE_CAP of the
    probes fail as a systematic skew (cond. 3); a handful of near-ties is a
    margin-qualified pass; all-exact is a strict pass."""
    if real > 0 or near_tie > max(1, int(NEAR_TIE_CAP * total)):
        return "fail"
    return "pass" if near_tie == 0 else "pass_margin_qualified"


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def post(url, body, timeout):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def wait_ready(base, process, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise SystemExit(f"server exited during startup (rc={process.returncode})")
        try:
            urllib.request.urlopen(base + "/health", timeout=2).read()
            return
        except Exception:
            time.sleep(0.5)
    raise SystemExit("server startup timed out")


def read_split(log_path):
    """The offload split the CUDA run actually achieved, from its own log.

    "5/5 identical" means something different at 24/24 than at 13/24: the
    second only certifies the layers that were on the device. Recording the
    split removes the need to remember which box produced a report.
    """
    try:
        text = pathlib.Path(log_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    hit = re.search(r"gpu-split: .*?G=(\d+)/(\d+).*?full=(\d)", text)
    if not hit:
        return None
    on, total, full = hit.group(1), hit.group(2), hit.group(3)
    return {"layers_on_gpu": int(on), "layers_total": int(total),
            "output_weight_on_gpu": full == "1",
            "whole_graph": int(on) == int(total) and full == "1"}


def generate_all(runner, model, gpu, tokens, ctx, env, extra, timeout, log_path):
    """Load the model once on `gpu`, return the greedy text for every prompt."""
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    cmd = [str(runner), "-m", str(model), "--serve", "--port", str(port),
           "-c", str(ctx), "--gpu", gpu, *extra]
    log = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    try:
        wait_ready(base, proc, timeout)
        out = []
        for prompt in PROMPTS:
            body = {"prompt": prompt, "max_tokens": tokens,
                    "temperature": 0, "top_p": 1, "stream": False,
                    "logprobs": TOP_N}
            resp = post(base + "/v1/completions", body, timeout)
            ch = resp["choices"][0]
            lp = ch.get("logprobs") or {}
            out.append({"text": ch["text"], "tokens": lp.get("tokens", []),
                        "logprobs": lp})
            print(f"  [{gpu}] {prompt!r} -> {out[-1]['text'][:48]!r}", flush=True)
        return out
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        log.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    # 128, not 16: the documented `cpu_cuda` contract has always said 128, and
    # the tool quietly defaulting to 16 meant "certified" asserted less than the
    # matrix claimed. CPU/GPU divergence that only opens up after a few dozen
    # tokens — KV-cache paths, accumulated rounding, sliding-window edges — is
    # precisely what this gate exists to catch, and a 16-token run cannot see it.
    # Owner decision 2026-08-08: raise the tool to the contract and re-certify,
    # publishing whatever un-certifies, rather than lower the contract to the tool.
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--runner", default=str(ROOT / "runner"))
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--fused", action="store_true",
                    help="do NOT pin the eager router (measures the fused "
                         "default, whose contract is weaker than identity)")
    ap.add_argument("--extra-arg", action="append", default=[],
                    help="extra flag passed to BOTH runs (e.g. --cpu-moe)")
    ap.add_argument("--out")
    args = ap.parse_args()

    env = dict(os.environ)
    env["RUNNER_CUDA_TC"] = "0"      # identity evidence is over the scalar path
    if not args.fused:
        env["RUNNER_MOE_EAGER"] = "1"

    logs = Path(args.out).parent if args.out else Path(".")
    gpu_log = logs / "cpu_cuda-gpu.log"
    print("loading CPU backend...", flush=True)
    cpu = generate_all(args.runner, args.model, "off", args.tokens, args.ctx,
                       env, args.extra_arg, args.timeout, logs / "cpu_cuda-cpu.log")
    print("loading CUDA backend...", flush=True)
    gpu = generate_all(args.runner, args.model, "auto", args.tokens, args.ctx,
                       env, args.extra_arg, args.timeout, gpu_log)

    # Dense models never get the near-tie tolerance (see is_moe): the
    # written policy always said so, but until 2026-08-20 nothing ENFORCED it —
    # classify_divergence itself is architecture-blind, so the downgrade to
    # "real" happens here, where the model file is in hand.
    moe = is_moe(args.model)
    rows = []
    exact = near_tie = real = 0
    for prompt, c, g in zip(PROMPTS, cpu, gpu):
        kind, detail = classify_divergence(c, g)
        if kind == "near_tie" and not moe:
            kind = "real"
            detail["dense_strict"] = True
            print("dense model: in-band flip is NOT a routing near-tie — "
                  "strict identity applies", flush=True)
        exact += kind == "exact"
        near_tie += kind == "near_tie"
        real += kind == "real"
        rows.append({"prompt": prompt, "kind": kind, "detail": detail,
                     "cpu": c["text"], "gpu": None if kind == "exact" else g["text"]})
        print({"exact": "ok  ", "near_tie": "TIE ",
               "real": "FAIL"}[kind] + f": {prompt!r}", flush=True)
        if kind == "near_tie":
            print(f"  near-tie flip @tok {detail['pos']}: cpu {detail['cpu_token']!r} "
                  f"vs gpu {detail['gpu_token']!r} (margins "
                  f"{detail['cpu_margin_to_gpu_pick']:.3f}/"
                  f"{detail['gpu_margin_to_cpu_pick']:.3f} nats ≤ "
                  f"{DEFAULT_TIE_BAND})", flush=True)
        elif kind == "real":
            print(f"  cpu: {c['text']!r}\n  gpu: {g['text']!r}", flush=True)

    total = len(rows)
    result = run_result(near_tie, real, total)
    over_cap = result == "fail" and real == 0   # failed purely on the near-tie cap

    split = read_split(gpu_log)
    report = {"model": str(args.model), "tokens": args.tokens,
              "context": args.ctx,
              "routing": "fused" if args.fused else "eager",
              "moe": moe,
              "extra_args": args.extra_arg,
              "gpu_split": split,
              "cpu_cuda_identity": {"result": result, "exact": f"{exact}/{total}",
                                    "near_tie_tolerated": near_tie,
                                    "tie_band_nats": DEFAULT_TIE_BAND,
                                    "over_cap": over_cap},
              # legacy shape kept so older readers still parse:
              "identical": exact, "total": total, "rows": rows}
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1))
    print(f"\ncpu_cuda: {result} — {exact}/{total} exact, {near_tie} near-tie "
          f"tolerated ({report['routing']} routing)")
    return 0 if result in ("pass", "pass_margin_qualified") else 1


if __name__ == "__main__":
    sys.exit(main())
