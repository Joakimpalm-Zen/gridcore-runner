# afmoe greedy-identity failure: root cause found — 2026-08-05 evening

Follow-up to `docs/afmoe-cert-report-2026-08-05.md` and
`docs/afmoe-sensitivity-floor-2026-08-05.md`. Method: layer-by-layer
activation differential on the M1 (same Q8_0 file as the cert run,
sha `5fcc2428…`), runner `RUNNER_DEBUG_ACT` vs llama.cpp b10280
`llama-eval-callback`, prompt "The three laws of thermodynamics are",
last-token values, CPU both sides. The `' Zer'`/`' The'` flip reproduces
on ARM exactly as on x86 — the divergence is ISA-independent.

## The mechanism (measured, layer 0)

| step | agreement runner vs llama.cpp |
|---|---|
| tokenization | identical ids (6 tokens) |
| scaled embeddings (muP) | identical to 4 decimals |
| attention input norm | identical to 4 decimals |
| V projection | ~4e-4 (quantization noise) |
| post-gate attention output | ~2e-4 — QK-norm, rope, softmax, output gate all agree |
| **attn_output matvec (wo)** | **~1e-3 absolute = ~10% RELATIVE (branch output is tiny, ~0.01)** |
| **after post-attention sandwich norm** | **~0.01–0.05 absolute — the relative error, re-scaled to full branch magnitude** |

The engines compute quantized matvecs by different arithmetic — the runner
dequantizes weights and does f32 FMA dots; llama.cpp quantizes activations
(q8_K) and does integer dots. On every certified arch to date this
difference stays ~1e-3 *absolute* in the residual and is invisible.
**afmoe re-normalizes every branch before adding it (`x + norm(f(norm x))`),
which converts the engines' small relative disagreement into full
normalized scale — injected at every one of 56 layers.** Layer trajectory:
a steady 3–40% relative delta on sampled dims from L0 through L45,
fluctuating, never healing, never exploding. Both engines agree the
residual stream itself nearly cancels in early layers (a property of the
trained muP weights, present identically on both sides).

This explains every observation in the floor study: flat-distribution
drift, deltas up to 1.87 nats co-existing with identical tokens,
cross-engine gap exceeding the runner's own KV-precision floor, and
ISA independence.

## Verdict

The afmoe implementation is arithmetically faithful — the divergence is
an architecture-amplified consequence of a documented, by-design
inter-engine difference, not an implementation defect. Token identity
against llama.cpp is structurally unachievable for this architecture
until the engines share dot arithmetic (the VNNI integer-dots roadmap
item is the retirement path for this caveat — a testable prediction).

## Recommended certification wording (owner decision)

`afmoe` row: CPU only; tokenizer differential 0/721; layer-0 attention
path verified against llama.cpp to ~2e-4; greedy token identity NOT
claimed — afmoe's per-branch re-normalization amplifies inter-engine
quantized-matvec arithmetic differences (measured mechanism:
`docs/afmoe-divergence-triage-2026-08-05.md`); cross-engine agreement
instead evidenced by the floor study (5/6 domains byte-identical at 64
tokens; divergences confined to sub-0.41-nat near-ties).
