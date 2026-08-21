# Qwen3.5-4B precision-vs-fidelity — BF16 vs upstream Q4_K_M (2026-08-21)

The first bar-v2 row for the `qwen35` architecture. Parent
`unsloth/Qwen3.5-4B-GGUF` BF16 (8.4 GB) vs the same repo's Q4_K_M (2.7 GB),
400 teacher-forced positions over `tests/fixtures/mixed-corpus.txt`,
`scripts/kld-compare-raw.py` (the bar's canonical raw-completion protocol,
tie band 0.5 nats), CPU serve, Blackwell host, runner `465a238`.

| metric | value | bar | verdict |
|---|---|---|---|
| margin-qualified top-1 | **97.5 %** | ≥ 97 % | pass |
| plain top-1 | 87.0 % | (reported) | — |
| mean KLD | **0.0709** | ≤ 0.05 | **FAIL** |
| mean top-8 overlap | 0.893 | (reported) | — |

**Verdict: the upstream Q4_K_M FAILS the adopted bar on KLD.** The flips are
near-ties (margin-qualified clears comfortably), but the distribution moves
too much — the same signature as every other small-model 4-bit failure this
program has measured, and consistent with the standing observation that
4-bit-class quantization becomes viable at roughly the 8B scale, not at 4B.
At 4B, Q8_0 remains the recommended tier.

Context number: the chat-templated variant (`kld-compare.py`, same positions)
reads mean KLD 0.0085 — an order of magnitude lower. The chat template's
forced prefix tokens dilute the statistic; the bar is defined on the RAW
protocol precisely because of this, and the two numbers are committed side by
side as a live example of why protocol choice is part of the measurement.

Evidence: `qwen35-kld-evidence-2026-08-21.json` (raw, per-position),
`qwen35-kld-chat-2026-08-21.json` (chat-path summary, context only).

Related same-day coverage: the qwen35 tensor-core gate row landed green
(0/64 flips, mean|dlogit| 2e-5 of range — recorded at the promotion table in
`src/cuda.c`; arch stays opt-in pending multi-checkpoint coverage).
