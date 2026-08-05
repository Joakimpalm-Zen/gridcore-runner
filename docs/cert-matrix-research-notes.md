# Cert-matrix research notes — size-reduction proposals assessed, note-only

Owner-supplied research proposals (2026-08-05, ecosystem/LLM-sourced),
assessed against this project's measured results. None of these gets
Blackwell time in this session; they are on record so the survey is
complete and the arithmetic isn't re-derived every time one resurfaces.
Context for all byte math: keep-30's measured delta (2 of 32 experts per
layer = 0.6 GB of 11.5 GB) puts gpt-oss-20b expert tensors at ~85–90% of
file bytes; attention/linear ~1 GB; embeddings ~5–8%.

## 1. Per-layer-embedding (PLE) codebook vectorization — note-only

Vector-quantize embedding tables against a shared codebook (~7× claimed
on embedding tensors). Genuinely unexplored here: our codebook kill
experiment (cb8: top-1 22.3%, KLD 1.384) was on MXFP4-QAT **expert FFNs**,
whose grid-lock does not transfer to embeddings. But the lever targets
the wrong tensor class for our targets: 7× on gpt-oss-20b's ~5–8%
embedding share saves ~0.6 GB — keep-30-sized, at unknown quality risk.
Relevant niche: embedding-heavy small models (Gemma E-series per-layer
embeddings — the arch where "half the footprint is embeddings" is
actually true). Revisit only if an E-series-class model on a <8 GB device
becomes a target; gate would be the standard 97%/0.05 bar.

## 2. "TurboQuant" structural attention/linear reduction — note-only, unverified

Claimed 28–42% structural reduction of MHA/linear tensors. The
description circulating does not match the TurboQuant literature we know
(online vector quantization, KV-adjacent), and no quality evaluation
accompanies the claim — it has the texture of LLM-generated plausibility.
Arithmetic: 40% off gpt-oss-20b's ~1 GB attention/linear share saves
~0.4 GB. For dense Gemma (12B/31B) a real cut of that size would matter,
IF it survives the gate — which is exactly what the claim never shows.
Falsify-first if ever pursued; no engine work on hearsay.

## 3. Granular expert paging (O_DIRECT + sector-aligned experts) — folded into the M1 kill experiment

The one proposal with a live question in it. Our expert-cache rejection
measured expert-granular streaming at 0.05 tok/s raw mmap → 0.65 sealed
with 3× residency; the physics ceiling for perfect streaming is
NVMe_bandwidth / miss_bytes_per_token (order 1 GB of active-expert bytes
per token uncached on gpt-oss-20b → ~2–3 tok/s absolute best on this
disk class, before cache hits raise it). The measured 0.65 sits well
under that ceiling, so alignment + async direct reads have real headroom
to chase — but the ceiling itself is below the ≥5 tok/s usable bar until
combined with load amortization. Disposition: NOT a standalone
experiment; it is one arm of the already-designed speculative-decode
amortization kill experiment on the M1 (K draft tokens per expert load
multiplies the ceiling; aligned direct I/O then determines how close you
get). macOS note: no O_DIRECT — F_NOCACHE + aligned pread is the
equivalent. Run only if Nano-9B-class pruned/healed models fail their
gates — a resident model beats a streamed one at any I/O cleverness.

## 4. Paged KV cache with disk swap — non-problem for our targets

Measured KV at our envelopes is small (GQA models at -c 8192, q8: low
hundreds of MB — gpt-oss-20b ~0.2 GB); the runner already sizes KV by
`-c`, supports `--kv q8`, and auto-fits under reservations. Token-level
KV paging to disk during decode adds a per-token disk dependency to save
memory we are not short of. Dead as proposed; the real lever it gestures
at (context caps under a budget) already exists.

## 5. Layer-fused whole-model streaming — dead by arithmetic

Streaming every layer per token means reading the whole file per token:
11.5 GB / ~3 GB/s NVMe ≈ 4 s/token ≈ 0.25 tok/s as the CEILING, with
perfect pipelining, forever. The proposal reduces RAM by moving the model
size into the per-token bandwidth term — the one term that cannot be
optimized past the disk. This is the expert-cache rejection generalized
to 100% of the weights. No.

## 6. Decoupled lookahead router (prefetch experts 2–3 tokens ahead) — the wall is prediction, and we measured its best case

Routing at layer L depends on the token's hidden state entering layer L —
knowing the *next* token's experts requires running the next token, and
approximate predictors trade exactly the accuracy the 97% gate protects.
The degenerate-but-optimal form of "keep likely experts ready" is an LRU
over routing history — which the expert-cache tier HAD, measured against
real routing traces (the same standing-committee analysis that produced
keep-30), and its hit economics are what 0.65 tok/s already includes.
The unpredictable tail experts are the misses by definition. Overlap of
I/O with compute is likewise already in the rejected tier. Nothing new
survives here that item 3's experiment would not subsume.
