# Handover — runner performance/hardening session

**Status: no open handover. Everything the 2026-07-20 session flagged has
shipped.** This file is now kept only for the durable operating notes at the
bottom (the verification gates and the Windows/Makefile quirks), which are
still true and save real time. The roadmap lives in
`gridcore-main/docs/plans/suite-wide-plan.md`; open cross-repo findings are in the same file.

## What that session flagged, and where it landed

- **Crash-on-OOM in `src/json.c` (three unchecked `malloc`/`realloc` sites,
  reachable from untrusted HTTP input).** FIXED in `bdf2c90` — the parser now
  survives allocation failure on every path (checked allocs, temp-pointer
  realloc idiom, failure returns `NULL` which every caller already handled).
- **Wider allocation-failure survey (~70 sites beyond `json.c`).** DONE
  (`1d4d7e0` / `07b1807` / `d8d80fd`): the request/model/schema paths return an
  error instead of dereferencing NULL under memory pressure.
- **Fuzz suite for the failure paths.** SHIPPED — malformed bodies, oversized
  parameters, and stalled clients run under ASan/UBSan in CI on every push.

The `session/2026-07-20-runner-perf` branch was only a marker (no unique
commits) and is gone. Line numbers from the old version of this file are all
stale — do not trust them; read the current `src/`.

---

## Verification gates you should use (they are why the kernel work was safe)

- `python scripts/kernel-verify.py --baseline <known-good.exe> --candidate <new.exe> --model <gguf>`
  → demands **token-identical** greedy output on 5 prompts. A faster or
  tidier binary that changes tokens is a **regression**, not a win. Copy
  the current `runner.exe` aside before building a candidate.
- `python scripts/kernel-bench.py --runner <exe> --model <gguf>`
  → prefill/decode tok/s as JSON.
- `RUNNER_CUDA_PROFILE=1` → per-phase GPU timings (matvec/attention/norms/
  memcpy/launch), off by default.
- `RUNNER_DEBUG_ACT=1` → per-layer activation stats incl. inf/nan counts.
  This is what proved a suspected gemma4 numerical bug did **not** exist.
- `RUNNER_FORCE_REQUANT=1`, `RUNNER_REQUANT_ONLY=<substr>` → let the
  quantizer do same-size conversions / only matching tensors, for
  bisecting a model group by group.

Reference numbers on an RTX 3070 (2k-token prompt), so you can spot a
regression: **Qwen3-4B-Q8 114 prefill / 59 decode**, **Qwen2.5-7B-Q4_K
112 / 36**, **Llama-3.1-8B-Q5_K_M 98 / 31**.

## Build / test quirks

- `scripts/kernel-bench.py` uses a synthetic `item0000 item0001 ...`
  prompt; some instruct models emit EOS immediately on it and the script
  then reports decode as 0 tok/s (seen with Ministral-8B). Both binaries
  do it equally so A/B comparisons stay valid, but it reads like a
  regression. Give it a more natural prompt.
- `make test` skips the Python client tests unless `PYTHON` points at an
  interpreter with pytest (msys2's python lacks it):
  `make test PYTHON=/c/Users/zen/AppData/Local/Programs/Python/Python312/python.exe`.
  The skip is deliberate and non-fatal.
- Windows quirk already worked around in the Makefile: a `python3`
  **app-execution-alias stub** exists on PATH but fails when run, so the
  interpreter is probed by *running* it, not by `command -v`. Keep that.
- `docs/specs/2026-07-11-cuda-backend-design.md` is marked **historical**:
  it lists partial/layer-split offload as a non-goal (it shipped) and
  describes a kernel design that has been replaced. README is
  authoritative where they disagree.

## Operating the assistant models (cost me real time — save yourself)

- Run **one `codex` at a time**. A second concurrent instance (even
  read-only) dies with `0xC0000142`.
- Always redirect stdin for background runs: `codex exec ... < /dev/null`.
  Without it, `codex exec` blocks waiting on stdin — this idled for 32
  minutes with 0% CPU before I noticed.
- Never let a subagent wait on a Monitor event or a background batch to
  wake it; it will simply stop. Give it foreground commands.
- Treat any agent's "verified / all green" as a **claim**. One reported
  `make smoke` passing when it failed here; another's confident root-cause
  hypothesis was refuted in three minutes by reading the code. Re-run the
  gates yourself.
