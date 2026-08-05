#!/usr/bin/env python3
"""Gate 3: greedy identity, runner vs llama.cpp b10280.

Reference side: llama-server /completion, temperature 0, explicit pure-greedy
sampler params (repeat_penalty 1.0, top_k 0, top_p 1.0, min_p 0.0) so neither
side has a truncation/penalty advantage.
Runner side: ./runner -p <prompt> -n N --temp 0 --gpu off; stdout echoes the
prompt verbatim, so the completion is stdout[len(prompt):].

Trailing whitespace is trimmed on both sides before comparison (the doc's
stated acceptable exception), and that is reported per case.
"""
import json
import hashlib
import pathlib
import subprocess
import sys
import urllib.request

MODEL = "/home/lab/workspace/models/trinity-nano/Trinity-Nano-Preview-Q8_0.gguf"
RUNNER = "./runner"
SERVER = "http://127.0.0.1:8188"
OUT = pathlib.Path("/tmp/afmoe-evidence/gate3")
OUT.mkdir(parents=True, exist_ok=True)

PROMPTS = [
    ("a", "The capital of France is", 64),
    ("b", "Write a python function that reverses a linked list.", 64),
    ("c", "In 1969, exactly 1234567 people watched as Apollo 11 landed. Summarize this event.", 64),
    ("d", "Beskriv kort hur en kvicksilvertermometer fungerar.", 64),
    ("b-long", "Write a python function that reverses a linked list.", 256),
    ("c-long", "In 1969, exactly 1234567 people watched as Apollo 11 landed. Summarize this event.", 256),
]


def ref_completion(prompt, n):
    body = json.dumps({
        "prompt": prompt, "temperature": 0, "n_predict": n,
        "repeat_penalty": 1.0, "top_k": 0, "top_p": 1.0, "min_p": 0.0, "seed": 0,
        # llama-server's slot prompt cache makes the REFERENCE itself
        # n_predict-dependent (verified: prompt (c) at n=64 was not a prefix of
        # the same prompt at n=256 until this was disabled), which would make
        # the comparison unfair to the runner. The runner was self-consistent
        # throughout. Disabled so each reference completion is computed cold.
        "cache_prompt": False,
    }).encode()
    req = urllib.request.Request(SERVER + "/completion", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.loads(r.read())
    return d["content"], d


def runner_completion(prompt, n):
    p = subprocess.run(
        [RUNNER, "-m", MODEL, "-p", prompt, "-n", str(n), "--temp", "0", "--gpu", "off"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=3600)
    out = p.stdout.decode("utf-8", "replace")
    if not out.startswith(prompt):
        raise SystemExit(f"runner stdout did not echo the prompt: {out[:120]!r}")
    return out[len(prompt):], p.stderr.decode("utf-8", "replace")


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return len(a) if len(a) != len(b) else -1


results = []
for tag, prompt, n in PROMPTS:
    sys.stderr.write(f"[{tag}] n={n} ...\n")
    sys.stderr.flush()
    ref, refmeta = ref_completion(prompt, n)
    run, runerr = runner_completion(prompt, n)
    ref_t, run_t = ref.rstrip(), run.rstrip()
    trimmed = (ref != ref_t) or (run != run_t)
    pos = first_diff(ref_t, run_t)
    ok = pos == -1
    (OUT / f"{tag}.ref.txt").write_text(ref)
    (OUT / f"{tag}.runner.txt").write_text(run)
    (OUT / f"{tag}.runner.stderr.txt").write_text(runerr)
    results.append({
        "tag": tag, "prompt": prompt, "n": n, "identical": ok,
        "first_divergent_byte": pos,
        "trailing_ws_trimmed": trimmed,
        "ref_sha256": hashlib.sha256(ref_t.encode()).hexdigest(),
        "runner_sha256": hashlib.sha256(run_t.encode()).hexdigest(),
        "ref_prefix": ref_t[:pos] if pos > 0 else "",
        "ref_tail_at_diff": ref_t[pos:pos + 60] if pos >= 0 else "",
        "runner_tail_at_diff": run_t[pos:pos + 60] if pos >= 0 else "",
        "ref_tokens_evaluated": refmeta.get("tokens_evaluated"),
        "ref_tokens_predicted": refmeta.get("tokens_predicted"),
    })
    sys.stderr.write(f"[{tag}] identical={ok} first_diff_byte={pos}\n")
    sys.stderr.flush()

(OUT / "summary.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
n_pass = sum(1 for r in results if r["identical"])
print(f"GATE3: {n_pass}/{len(results)} byte-identical")
for r in results:
    print(f"  {r['tag']:8s} n={r['n']:3d} identical={r['identical']} "
          f"first_diff_byte={r['first_divergent_byte']}")
