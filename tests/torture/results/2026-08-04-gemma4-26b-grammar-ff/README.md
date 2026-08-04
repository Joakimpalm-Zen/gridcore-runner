# gemma-4-26B-A4B — grammar fast-forward A/B (2026-08-04)

`agent-torture.py --cases 35`, both arms CPU (`--gpu off`), same model, same
cases; ON arm adds `RUNNER_GRAMMAR_FF=1`. Two generations of this
measurement exist, straddling `8a7804a` ("withhold the pin-final draft
token" — the tail-straddle fix). **Gen 2 (below) is current; Gen 1 is kept
for the record, not as the live number.**

## Gen 2 — current (trip 3, runner `63fc829`)

| | OFF (baseline) | ON (`RUNNER_GRAMMAR_FF=1`) |
|---|---|---|
| elapsed_ms | 296,967.80 | 301,207.86 |
| passed/failed | 31/4 | 31/4 (same 4 case ids as Gen 1) |

Grammar acceptance: **217/286 = 75.87%** (same 217 tokens accepted as Gen
1, far fewer bad drafts attempted: 464 → 286). Net wall-clock: **1.43%
slower** — narrowed sharply from Gen 1's 3.6%, but still net negative even
at 75.9% acceptance, which is comfortably inside the range that nets a
*win* on dense models per the E4B/gpt-oss data points. This is the cleanest
single measurement of the MoE-verify-overhead hypothesis's size yet: even
with the tail-straddle rejections almost entirely removed, the remaining
per-position routing + expert-gather cost on this 128-expert MoE is still
enough to erase grammar-FF's gain, just by a much smaller margin than
before (-3.6% → -1.43%).

**Identity: 0 mismatches across all 35 cases**, same SSE-reconstruction
method as Gen 1.

Directories `26b-off-v2/`, `26b-on-v2/` hold Gen 2's `report.json`,
`raw.jsonl`, `runner.log`; `*-v2.stdout.log` are the driver stdouts.

## Gen 1 — pre-fix (trip 1, runner `021e5bc`), kept for the record

| | OFF (baseline) | ON (`RUNNER_GRAMMAR_FF=1`) |
|---|---|---|
| elapsed_ms | 294,168.85 | 304,689.19 |
| passed/failed | 31/4 | 31/4 (same 4 case ids) |
| median gen tok/s | 13.61 | 12.06 |

Grammar acceptance: **217/464 = 46.8%**. Net wall-clock: **3.6% slower**.
Identity: 0 mismatches. Predates `8a7804a` — this is the number trip 2
flagged as stale; Gen 2 above supersedes it.

Directories `26b-off/`, `26b-on/` hold Gen 1's `report.json`, `raw.jsonl`,
`runner.log`; `26b-off.stdout.log`/`26b-on.stdout.log` are the driver
stdouts.

## Common to both generations

Machine: 128 logical cores, 268.7 GB RAM (CPU run; the box's Blackwell MIG
slice was unused — the driver spawns `--gpu off`). Toolchain: conda
`ccbuild` env, `CC=x86_64-conda-linux-gnu-gcc`, per the precedent in
`../2026-08-03-smollm2-1.7b-v2/README.md`.

Model provenance caveat: `models/gemma-4-26B-A4B-it-Q4_0.gguf` (14.6 GB)
traces to `google/gemma-4-26B-A4B-it` in metadata but carries no
official-QAT provenance markers — these numbers characterize *a* Q4_0
quantization of the model, not verified-identical-to-release behavior.
