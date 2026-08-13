# Negative result: Metal prefill GEMM occupancy and tile shape

*2026-08-15. Apple M1, 8 GB, 8 GPU cores. Swept, measured, nothing adopted —
the shipped tile shape is already at a local optimum.*

## What prompted it

`docs/metal-dispatch-census-2026-08-13.md` ended by measuring prefill's
per-forward floor at 18.0 ms for 145 MB of resident weights — **8 GB/s against
roughly 68 GB/s of roofline** — and concluding that the floor is latency and
occupancy inside the tiled GEMM rather than memory traffic. This is the sweep
that conclusion asked for.

## The instrument that made it interpretable

Threadgroup memory is the occupancy currency on Apple GPUs: a core has 32 KB,
so a kernel asking 14 KB gets two resident threadgroups and one asking 8 KB
gets four. None of that is visible from throughput. `RUNNER_METAL_STATS` now
prints each pipeline's `staticThreadgroupMemoryLength`:

```
metal-pipeline k_mm_q8_0        tgmem=14336 B  max_threads=1024
metal-pipeline k_attn           tgmem= 1024 B  max_threads=1024
```

That one line is the only part of this work that was kept. Every conclusion
below depends on it, and the sweep would have been guesswork without it.

## 1. It is not bandwidth, and it is not dequantization

Same 135M geometry, three weight formats, prefill GPU time at n=4:

| model | size | GPU time | effective |
|---|---:|---:|---:|
| SmolLM2-135M Q8_0 | 138 MB | 18.36 ms | 13 GB/s |
| s135 f16 | 258 MB | 18.84 ms | 25 GB/s |
| s135 bf16 | 258 MB | 19.89 ms | 25 GB/s |

**The f16 model is 1.87× larger and takes the same time.** Time does not track
bytes, so the floor is not bandwidth. And f16 needs no dequantization at all,
so it is not dequantization either. What is constant across the three is the
tile geometry and the dispatch count — which is what pointed the sweep at
occupancy.

## 2. More occupancy does not help (the aliasing experiment)

`tg_c` (the result tile) is only written after the k-loop's trailing barrier,
past which `tg_w`/`tg_x` (the staging buffers) are dead. Overlapping them drops
the kernel from 14336 B to 8192 B — measured, confirmed by the probe — which
takes residency from **two** threadgroups per core to **four**.

| | fixed a | slope b |
|---|---:|---:|
| 14336 B (2 TG/core) | 18.15 ms | 0.145 ms/token |
| 8192 B (4 TG/core) | 18.23 ms | 0.144 ms/token |

Neutral. Doubling occupancy changes nothing, so two resident threadgroups
already saturate this kernel. **Not adopted**: it buys no time and it
introduces a lifetime invariant (a future edit that writes `tg_c` before the
last barrier corrupts the staging buffers silently).

It also introduced a real bug worth recording, because the gate caught it
rather than a review: sizing the shared pool to the result tile alone
(`MM_TN * MM_TM` floats) is correct at 64×32, where the result needs 8192 B
against 6144 B of staging — and **wrong at 64×16**, where staging needs 5120 B
against a 4096 B pool. Narrow tiles overran `tg_x`. `test-metal-kquant`
reported it as a wrong answer immediately. The first narrow-tile timings in
this session came from that out-of-bounds kernel and were discarded.

## 3. The tile sweep, against the metric that actually matters

Two metrics disagree, and choosing the wrong one inverts the conclusion.
Prefill in production runs at the default batch of 64; the n=4..16 regime only
occurs for very short prompts, which are cheap anyway.

Small-batch GPU time (SmolLM2, `MM_TM=64`, `MM_TK=32`), all parity-clean:

| MM_TN | tgmem | n=4 | n=16 |
|---:|---:|---:|---:|
| 32 | 8192 B | 19.33 ms | 21.72 ms |
| 16 | 5120 B | 13.86 ms (−28 %) | 16.70 ms (−23 %) |
| 8 | 4608 B | 11.67 ms (−40 %) | 15.51 ms (−29 %) |

Production prefill throughput, same builds:

| MM_TN | e2b prompt tok/s |
|---:|---:|
| 32 | **82.85** |
| 16 | 73.10 (−12 %) |
| 8 | 57.55 (−31 %) |

A narrow column tile wins big on tiny batches and loses badly where prefill
actually lives, because the weight tile is re-read once per column tile: at
n=64, `MM_TN=8` reads the weights eight times instead of twice.

## 4. The one candidate that looked better, and was not

`MM_TM=64, MM_TN=64` read 84.64 tok/s against 83.53 in single measurements —
a plausible +1.3 %. Round-robin, five interleaved iterations plus a discarded
warmup:

| | e2b prompt tok/s (median) |
|---|---:|
| 64×32 (shipped) | **77.6** |
| 64×64 | 70.8 (**−8.8 %**) |

The single-shot reading was noise of the opposite sign. The mechanism is
consistent with everything above: 64×64 needs 24576 B of threadgroup memory,
which is **one** resident threadgroup per core. Going 2 → 4 buys nothing
(§2); going 2 → 1 costs 8.8 %. Two is both necessary and sufficient.

## 5. MM_TK is not sweepable as the code stands

`MM_TK=64` measured 86.69 tok/s and **failed parity**, and the reason is
structural rather than a tuning limit: the weight-staging loop is hardcoded to
32 values per row (`int r = p * 32 + (tid >> 2), sub = (tid & 3) * 8` — 4
threads × 8 values). At `MM_TK=64` it fills half of each row and leaves the
rest stale, so the kernel is fast because it does less work. The number is
discarded.

Making `MM_TK` a real knob is **larger than "rewrite that loop", and the
evidence for doing it is absent.** Scoped properly on 2026-08-16:

The staging loop's index arithmetic generalises easily — stride linearly over
`MM_TM * MM_TK` values at 8 per thread, `r = idx / MM_TK`, `sub = idx % MM_TK`,
which reproduces the current mapping exactly at 64×32. That part is contained.

What is not contained is every `DEQ_CHUNK`. Each computes its block from `k0`
alone and then offsets by `sub` **assuming `sub < 32`**:

```c
// k_mm_q8_0
device const uchar *blk = wb + a.w_off + ((ulong)row * nb + k0 / 32) * 34;
device const char  *q   = (device const char *)(blk + 2) + sub;
```

At `MM_TK=64`, `sub` reaches 56 and `q[j]` walks off the end of a 34-byte block
into the next block's scale. Every one of the ~12 `k_mm_*` kernels needs its
block selection moved from `k0` to `k0 + sub`, with per-type granularity (q4_K
indexes 256-element superblocks and a sub-block scale, mxfp4 and the iq types
each differ again).

And the reason to do it is missing. The `MM_TK=64` figure that motivated this —
86.69 tok/s against 83.53 — came from exactly the bug above: the loop filled
half of each row, so the kernel was fast because it read less. It failed parity.
There is no measurement suggesting a larger k-step helps. The theoretical
argument is fewer barriers (18 k-iterations instead of 36 at `n_in=576`), but
this same document shows the kernel is not barrier- or occupancy-bound —
doubling resident threadgroups was neutral.

So: a dozen per-type numerical edits, each able to produce a silent wrong
answer, to test a hypothesis with no supporting evidence and one refuted
argument. **Skipped deliberately**, and the shape is written down so a future
session can take it if a reason appears.

## Conclusion

The shipped `64 × 32 × 32` is at a local optimum on this hardware, and the
occupancy hypothesis that motivated the sweep is **refuted**: the kernel is
saturated at two resident threadgroups per core, so there is no occupancy to
recover. Combined with §1 — not bandwidth, not dequantization — what remains is
per-dispatch latency and the dependent simdgroup chain inside each threadgroup,
neither of which tile shape addresses.

Kept from this work: the `tgmem` diagnostic. Everything else measured neutral or
worse and is not in the product.

The untried levers, in the order the evidence favours them: a general `MM_TK`
staging loop (§5); a batch-dependent tile choice, which would need two compiled
variants and is only worth it if short-prompt prefill ever becomes a workload
that matters (§3); and splitting K across threadgroups, which adds a reduction
and is a different kernel rather than a tuning change.
