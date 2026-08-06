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

---

# Addendum (same day): items 3/6 re-measured from archived routing traces — the amortization math clears the bar for gemma-26B

The owner supplied a follow-up proposal set (async zero-syscall I/O,
token-tree speculative expert prefetch, lock-free ring buffers) aimed at
the streaming gap, focused on **gemma-4-26B-A4B**. Instead of assessing
it rhetorically, we computed the answers from the trip2/r0 archived
routing traces (real gpt-oss-20b and gemma-26B routing decisions, plus
gemma lookahead probes). Three measured results:

**1. Lookahead expert prediction is confirmed dead** (the wall item 6
predicted): gemma-26B lookahead-1 probes, predicted-vs-actual top-8:
**60.0% hit rate** (chat, 127k probes), **53.5%** (agent-torture, 400k
probes). Any prefetch scheme relying on predicting the *next* token's
experts stalls on ~40–47% of its reads. Token-tree prefetch "fixes" this
by fetching the union of several branches — i.e. spending multiples of
the scarcest resource (bandwidth) to hedge a coin flip. Dead as proposed.

**2. But batch-verify amortization needs NO prediction — and its measured
ceiling clears the usable bar.** In speculative decoding the K draft
tokens are *known* during verification; each layer computes routing for
all K positions, then reads the union of needed experts once. Union
sizes from real traces (per layer, K consecutive tokens):

gemma-4-26B-A4B QAT Q4_0 (30L, 128E top-8, ~3.2 MB/expert, experts
~12.3 GB of 14.4 GB) — ceiling at 3 GB/s NVMe, zero expert cache:

| K | mean union (of 128) | GB/token | ceiling tok/s |
|---|---|---|---|
| 1 | 8.00 | 0.77 | 3.9 |
| 2 | ~13 | 0.62 | ~4.8 |
| 4 | ~20 | 0.49 | ~6.1 |
| 8 | ~31 | 0.37 | ~8.1 |
| 16 | ~44 | 0.26 | ~11.5 |

(chat/doc/agent-torture traces agree within ~10%.)

gpt-oss-20b MXFP4 (24L, 32E top-4, ~12.5 MB/expert): K=1 → 1.20 GB/token
→ 2.5 tok/s; K=8 → 0.50 → ~6.0; K=16 → ~9.4. Worse than gemma at every K
(fewer, fatter experts share less).

**3. The target machine's real bandwidth:** measured 2.35 GB/s
sequential on the 8 GB M1's SSD (3 GB cold-ish dd). Scaling: gemma K=1
→ ~3.0 tok/s, K=4 → ~4.8, K=8 → ~6.4 — **the K≥4 amortized ceiling
crosses the ≥5 tok/s usable bar on the actual hardware**, before any
expert-cache hits (standing-committee residency only improves it; gemma's
always-on dense branch + attention ≈ 2 GB must be resident anyway and
fits the envelope).

**What this changes:** the expert-cache rejection measured 1-token-per-
load streaming (0.65 tok/s sealed, gpt-oss). Batch verification changes
the loads-per-token arithmetic by 2–3×, and gemma-26B has an official
MTP drafter (roster item 17 tests its losslessness + speedup on this
very box). The pieces now compose into a falsifiable target:

> **gemma-4-26B-A4B QAT (25B-class MoE) at ≥5 tok/s sustained on an
> 8 GB M1**, via MTP draft + batch-verify union reads + aligned
> F_NOCACHE streaming + hot-expert residency.

Honest unknowns that decide it (in kill order): (a) MTP draft acceptance
rate — effective K is acceptance-scaled, and K=2-3 effective is the bar's
edge; item 17's measurement answers this; (b) achievable fraction of
sequential bandwidth with ~3 MB scattered aligned reads (measure with a
50-line standalone I/O probe before any engine work); (c) compute/I-O
overlap efficiency on 4 P-cores. The async/ring-buffer engineering from
the proposal is real but subordinate: it matters only if (a) and (b)
survive. STOP rule unchanged: no engine work until all three numbers are
in and the composed ceiling still clears 5 tok/s.

## Addendum 2: Google's MoE batch-1 MTP warning, answered with residency math

Google's Gemma-4 MTP guidance (surfaced via owner's web sweep, buried in
SEO noise) warns the 26B MoE can see ZERO MTP speedup at batch size 1:
verifying K drafts loads the expert UNION, offsetting draft gains when
the amortizable dense fraction is small and effective K (acceptance-
scaled) is low. Mechanism accepted — at effective K=2–3 the raw union
amortization is only ~1.3–1.5× on expert bytes.

But that analysis is for RAM-resident models with no expert cache. Our
regime streams misses past a resident standing committee, and the two
levers COMPOSE (measured, doc trace, misses = union minus top-C-frequent
residents per layer, M1 2.35 GB/s):

| resident set | cache | eff. K=2 | K=3 | K=4 | K=8 |
|---|---|---|---|---|---|
| top-16/layer | 1.5 GB | 6.9 | 7.5 | 7.9 | 9.5 tok/s ceiling |
| top-32/layer | 3.1 GB | 11.3 | 12.1 | 12.7 | 14.7 tok/s ceiling |

Single-token committee hit rates: 50.6% (top-16), 70.3% (top-32).
Budget check on 8 GB: dense/attention ~2 GB + committee 1.5 GB + KV/
scratch ≈ 4 GB — fits with OS headroom; the 3.1 GB cache is the stretch
config. **Even at effective K=2 — Google's pessimistic regime — the
composed ceiling is ~7 tok/s, comfortably above the 5 tok/s bar.** The
kill-order unknowns collapse to: (b) scattered-read efficiency (~7–12
concurrent 3.2 MB aligned reads per layer — the standalone I/O probe
answers this) and (c) compute/IO overlap; acceptance (a) merely moves us
along a curve that clears the bar everywhere. Source-hygiene note: the
same web sweep attributes "TurboQuant" to an SEO cluster shilling a
patched llama.cpp fork — consistent with note 2's assessment.

## Addendum 3: unknown (b) RESOLVED — scattered reads run at 90% of sequential on the M1

Standalone probe (no engine code): K concurrent page-aligned F_NOCACHE
preads of 3.25 MiB at random offsets in a real 11.5 GB GGUF, 30
layer-serialized rounds per pass (the barrier the real engine cannot
avoid), 3 reps. Measured on the 8 GB M1:

| pattern | sustained | note |
|---|---|---|
| K=1 serial (QD1) | 2.07–2.11 GB/s | the SSD saturates at queue depth 1 |
| K=7 concurrent | 2.13–2.17 GB/s | top-16-cache miss traffic, eff K=2 |
| K=12 concurrent | 2.10 GB/s | eff K=4 miss traffic |
| K=21 concurrent | 2.09–2.11 GB/s | no-cache K=4 union |

**Scattered ~3 MB aligned reads cost nothing on this hardware: 90% of
the 2.35 GB/s sequential figure, at ANY concurrency, layer barriers
included.** Gemini's proposed probe was directionally right but had a
10x read-size error (32 MB vs 3.2 MB), strided-not-random offsets (SSD
prefetch flatters strides), a broken percentage printf, and no layer
serialization; this probe fixes all four.

I/O-only token rates from the measured runs (pass time / effective K):
eff K=2 with top-16 cache: 329 ms / 2 = **~6.1 tok/s**; eff K=4:
585 ms / 4 = **~6.8 tok/s**; no-cache K=4: ~3.9. The cached
configurations clear the 5 tok/s bar ON MEASURED I/O, not estimates.

Kill-order status: (a) MTP acceptance — pending, item 17; (b) scattered
bandwidth — **RESOLVED, no penalty**; (c) compute/IO overlap — now the
live risk, and honestly the bigger one: gemma-26B is ~4B-active, and the
Blackwell CPU row for it was 6.6 tok/s on 128 threads. Four M1 P-cores
computing K verify positions must land at or above ~5 tok/s themselves
for the overlapped pipeline to hold the bar. Next measurement (still no
engine work): a resident-compute proxy — batch-K CPU forward throughput
of a 4B-active-class model on the M1 (E4B or a truncated-layer 26B
slice), overlapped-vs-serial with the probe's I/O pattern.

## Local M1 finding (owner's machine, same day): E-series QAT GGUFs refuse to load

Attempting the (c) compute-proxy measurement surfaced a cert-grade
result: BOTH independent conversions of Google's official E4B QAT export
fail identically in the runner —
`unsloth/gemma-4-E4B-it-qat-GGUF/gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf` and
`lmstudio-community/gemma-4-E4B-it-QAT-GGUF/gemma-4-E4B-it-QAT-Q4_0.gguf`
both die at load with `error: missing tensor blk.24.attn_k.weight`,
while the certified non-QAT conversion (bartowski Q4_K_M) loads fine.
The QAT export evidently encodes the E-series shared-KV layer pattern
differently (omitting K/V weights on shared layers, or declaring the
share map differently) than the conversions the loader was validated
against. **Expect roster item 7 (E2B QAT) to REFUSE the same way — that
is a correct result; record the metadata dump.** Loader-side support is
owner-machine work, out of scope for the Blackwell session per the STOP
rules.
