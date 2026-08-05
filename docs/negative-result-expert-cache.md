# Negative result: an expert-residency cache tier for MoE models larger than RAM

*2026-08-05. Status: investigated, measured, rejected. Not in the product.*

## What was tried

For sparse MoE models whose weights exceed system RAM (e.g. gpt-oss-20b,
12.1 GB, on an 8 GB machine), we prototyped an explicit expert-residency
cache: a bounded slot pool that pins routed-expert weights in memory and
reads past the OS page cache, instead of relying on `mmap` and letting the
kernel thrash. Routing locality is real and reproducible — the pool held
85.6–86.8 % of routed-expert touches across macOS/ARM, Windows/AVX2 and
Linux/x86 with byte-identical outputs to the uncached path — so the
mechanism works exactly as designed. The question was whether it makes any
configuration *usable*.

## What was measured

Decode throughput, gpt-oss-20b class, keep-30 pruned artifact (11.5 GB),
same prompt and token budget everywhere. RAM ceilings on the Linux box were
enforced with a frozen, integrity-checked memory ballast; arms with any
ballast loss were invalidated and excluded.

| regime | naive mmap | cache tier (best) | verdict |
|---|---|---|---|
| model fits in RAM | 13.34 tok/s | 8.35 tok/s | tier costs 37 % |
| 1.4× over RAM (8 GB Apple M1) | 0.78 | 0.39 | tier costs 50 % |
| ~2× over RAM (capped Linux) | 0.39 | 0.56 | +44 %, both unusable |
| ~3× over RAM (capped, swap sealed) | 0.05 | 0.65 | +13×, both unusable |
| 8× over RAM (120b on 8 GB) | 0.10 | 0.08–0.11 | neutral, unusable |

## Why it is rejected

There is no measured regime in which the tier converts an unusable
configuration into a usable one. Where the model fits (or nearly fits),
the OS page cache already wins and the tier only adds overhead. Where the
model is far larger than RAM, the tier reduces I/O exactly as designed —
page faults halve, reads drop 3–9× — and decode is *still* an order of
magnitude below usable, because the residual misses dominate. A 13×
improvement on 0.05 tok/s is not a feature; it is a mitigation for a
configuration that should not be run at all.

An earlier measurement in this program showed the tier winning 2.1× on the
8 GB machine. That result did not survive: it predated proper SIMD kernels
for the model's MXFP4 expert format, and the missing kernels masked the
comparison (both arms were compute-starved). With kernels in place the
ordering inverted. We publish this so nobody — including us — re-derives
the idea from the earlier number.

## What replaces it

The runner will not pretend such configurations are supported. Model-
much-larger-than-RAM setups produce a clear warning today and are the
first candidates for refusal under the certified-operating-envelope work:
a configuration that runs, generates fluent text, and delivers 0.4 tok/s
with pathological I/O is *outside envelope*, and the runtime should say
so rather than let it masquerade as a working install.

The measurement methodology (SIMD-unmasked kernels first, cold-cache
discipline, ballast integrity checks, byte-identity verification of every
artifact) is the durable output of this investigation.
