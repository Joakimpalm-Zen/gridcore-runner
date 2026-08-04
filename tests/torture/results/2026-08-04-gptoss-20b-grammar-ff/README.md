# gpt-oss-20b — grammar fast-forward A/B (2026-08-04)

`agent-torture.py --cases 35`, both arms CPU (`--gpu off`), same model, same
cases; ON arm adds `RUNNER_GRAMMAR_FF=1`. Runner `8a7804a` (includes
"withhold the pin-final draft token", a concurrent same-day fix to the exact
mechanism this measures — see "Caught mid-session" below) + the
`RUNNER_MOE_PROBE` instrumentation from this trip (off in these runs).

| | OFF (baseline) | ON (`RUNNER_GRAMMAR_FF=1`) |
|---|---|---|
| elapsed_ms | 175,147.40 | 183,463.20 |
| passed/failed | 35/0 | 35/0 (same cases) |

Grammar acceptance summed over the ON arm's `spec: ... grammar A/D` lines:
**168/277 = 60.65%**. Net wall-clock: **4.75% slower** with the flag on.
Still a net loss despite acceptance now well inside the range smaller dense
models showed a win — both in-scope MoE data points (this and the 26B, see
`../2026-08-04-gemma4-26b-grammar-ff/`) show grammar fast-forward losing
wall-clock even at healthy acceptance, which is consistent with the
speculative-verify-amortizes-less-on-MoE hypothesis: each verify position
still pays the full per-position routing + expert-gather cost regardless of
how many draft tokens that round validates.

**Identity: 0 mismatches across all 35 cases** — `content`/`tool_calls`
compared after base64-decoding `response.body`, with `stream_normalization`
cases reconstructed from their SSE chunks (delta/tool-call concatenation)
rather than raw-byte-diffed.

## Caught mid-session

The first pass through this measurement (runner `408e26c`) got 40.0%
acceptance (168/420) and -8.5% wall-clock. Before committing, `git pull
--ff-only` picked up `8a7804a` — a same-day, same-machine-family commit
that changes grammar-FF's draft-token selection (withholds the token that
straddles the grammar pin's tail, based on a phase-0 trace showing 100% of
rejections were exactly that straddle). That's the mechanism this A/B
measures, so the first-pass numbers would have shipped a stale finding.
Rebuilt, re-ran both arms against the merged tree, got the numbers above
instead — same 168 accepted, far fewer bad drafts attempted (420 → 277),
so acceptance rose sharply and the wall-clock deficit narrowed (-8.5% →
-4.75%) without disappearing. The already-committed 26B result in
`../2026-08-04-gemma4-26b-grammar-ff/` was measured at `021e5bc`, predates
this fix the same way, and was **not** re-run this trip (out of scope here;
flagging it as now-stale rather than silently re-measuring someone else's
delivered result). **Update (trip 3):** it was re-measured — see that
directory's "Gen 2": 46.8%→75.87% acceptance, -3.6%→-1.43% wall-clock. Same
shape as this file's update, smaller absolute narrowing in percentage-point
terms but the model still nets a loss at high acceptance either way.

Machine: 128 logical cores, 268.7 GB RAM (CPU run). Toolchain: conda
`ccbuild` env, `CC=x86_64-conda-linux-gnu-gcc`, per the precedent in
`../2026-08-03-smollm2-1.7b-v2/README.md`.

Model: `models/gpt-oss-20b-MXFP4.gguf` (12.1 GB), native MXFP4 quant,
`general.name = gpt-oss-20b` with no ambiguous-provenance flags (unlike the
26B's two local gemma conversions).

Directories `gptoss-off/`, `gptoss-on/` hold each arm's `report.json`,
`raw.jsonl`, and `runner.log`; `*.stdout.log` are the driver stdouts.
