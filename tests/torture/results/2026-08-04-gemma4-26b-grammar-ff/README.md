# gemma-4-26B-A4B — grammar fast-forward A/B (2026-08-04)

`agent-torture.py --cases 35`, both arms CPU (`--gpu off`), same model, same
cases; ON arm adds `RUNNER_GRAMMAR_FF=1`. Runner `021e5bc` + the
`RUNNER_MOE_TRACE` instrumentation later committed as `b0c7108` (off in
these runs).

| | OFF (baseline) | ON (`RUNNER_GRAMMAR_FF=1`) |
|---|---|---|
| elapsed_ms | 294,168.85 | 304,689.19 |
| passed/failed | 31/4 | 31/4 (same 4 case ids) |
| median gen tok/s | 13.61 | 12.06 |

Grammar acceptance summed over the ON arm's `spec: ... grammar A/D` lines:
**217/464 = 46.8%**. Net wall-clock: **3.6% slower** with the flag on —
acceptance is inside the range where smaller dense models showed a win, so
the deficit is not acceptance alone. Plausible mechanism, reported as
observed: each speculative-verify forward on this 128-expert MoE pays the
full per-position routing + expert-gather cost, so verify amortizes less
than on a dense model.

**Identity: 0 mismatches across all 35 cases** — `content`/`tool_calls`
compared after base64-decoding `response.body`, with `stream_normalization`
cases reconstructed from their SSE chunks (delta/tool-call concatenation)
rather than raw-byte-diffed, which would have flagged `created`-timestamp
and chunk-boundary noise.

Machine: 128 logical cores, 268.7 GB RAM (CPU run; the box's Blackwell MIG
slice was unused — the driver spawns `--gpu off`). Toolchain: conda
`ccbuild` env, `CC=x86_64-conda-linux-gnu-gcc`, per the precedent in
`../2026-08-03-smollm2-1.7b-v2/README.md`.

Model provenance caveat: `models/gemma-4-26B-A4B-it-Q4_0.gguf` (14.6 GB)
traces to `google/gemma-4-26B-A4B-it` in metadata but carries no
official-QAT provenance markers — these numbers characterize *a* Q4_0
quantization of the model, not verified-identical-to-release behavior.

Directories `26b-off/`, `26b-on/` hold each arm's `report.json`,
`raw.jsonl`, and `runner.log`; `*.stdout.log` are the driver stdouts.

**Update (trip 2, same day):** `8a7804a` changed grammar-FF's draft-token
selection after this measurement was taken (`021e5bc`) — see
`../2026-08-04-gptoss-20b-grammar-ff/README.md`'s "Caught mid-session" for
what changed. That fix raised gpt-oss's acceptance from 40.0% to 60.65% and
narrowed its wall-clock deficit from -8.5% to -4.75%; this file's 46.8%/
-3.6% numbers likely moved too but were not re-measured this trip — treat
them as pre-fix, not current-main.
