# What the long-context decode loss actually is: the KV read, at 1.5 GB/s

*2026-08-15. Apple M1, 8 GB, 8 GPU cores, e2b-q40 (gemma-4 E2B, 35 layers).
Diagnosis. No kernel changed.*

`docs/metal-long-context-decode-2026-08-14.md` closed the "flash-shaped
rewrite" item as prior-art and left one question open, deliberately unanswered
rather than guessed: decode falls **52 %** between a 694- and an 8,110-token
span, attention parallelism is not the cause, and the traffic estimate that
looked like the right order omitted the sliding-window correction. This
measures it.

## The instrument

The dispatch census now also accumulates the KV bytes each attention dispatch
will read, split by whether the layer slides, for **decode only** (`n == 1`).
Decode-only because at `n > 1` each column attends over its own growing range
and a single `pos` does not describe the read — counting prefill there would
produce an underestimate that looks authoritative.

```
metal-kv decode-cumulative: global 465203200 B over 28 layer-dispatches
                          | sliding 58720256 B over 112 | sliding share 11.2%
```

The layer split falls straight out and confirms the architecture: **7 global
layers, 28 sliding** — 35 total.

## The measurement

Per decode token, e2b-q40, four spans:

| span | KV/token | global | sliding | measured ms/token |
|---:|---:|---:|---:|---:|
| 694 | 24.67 MB | 9.99 | 14.68 | 64.27 |
| 2,326 | 48.06 MB | 33.38 | 14.68 | 79.30 |
| 4,641 | 81.25 MB | 66.57 | 14.68 | 101.94 |
| 8,110 | 130.98 MB | 116.30 | 14.68 | 134.23 |

**The sliding layers are exactly constant at 14.68 MB**, which is the
window doing its job, and the whole growth is the 7 global layers. At the
longest span the sliding 80 % of the model contributes 11 % of the KV traffic.

## The fit, and the answer

Regressing measured decode time on measured KV bytes:

```
ms/token = 47.91 + 0.6600 x KV_MB
```

| span | predicted | measured |
|---:|---:|---:|
| 694 | 64.19 | 64.27 |
| 2,326 | 79.64 | 79.30 |
| 4,641 | 101.54 | 101.94 |
| 8,110 | 134.37 | 134.23 |

Every point inside 0.5 %. Consecutive-pair increments agree independently:
1.56, 1.47 and 1.54 GB/s.

**So the 52 % is entirely the KV read, and the KV read runs at 1.52 GB/s** —
about 2 % of this machine's ~68 GB/s roofline. The span-independent 47.91 ms
covers the weights and everything else; against e2b's ~1.3 GB of per-token
weight traffic that is roughly 27 GB/s. **The KV read is ~18x less efficient
per byte than the weight read on the same hardware in the same forward.**

That is the answer the previous document declined to guess at. It is not
attention parallelism, and it is not the volume of KV bytes being unreasonable
— it is the rate at which those bytes are read.

## Why, from the code

In `k_attn`, each thread owns whole rows:

```c
for (int t = t0 + tid; t <= pos; t += tpg)
    ah[t] = kv_dot(kc + base + (ulong)t * row_b, qh, hd, a.q8) * a.scale;
```

Adjacent lanes are `row_b` apart — for e2b, `kv_dim` 512 at f16 is **1024
bytes**. A 32-lane simdgroup issuing one load therefore touches 32 distinct
cache lines and uses a fraction of each. Compare the weight matvec, where lane
`i` takes block `i` and 32 lanes cover a contiguous ~576-byte run: that route
measures ~27 GB/s here, and the difference in access pattern is the difference
in rate.

**Stated as the leading hypothesis, not a result**: the fix implied is to make
the threads cooperate on one row at a time — parallelise across `head_dim`
within a row and reduce, then advance rows — so a simdgroup's lanes read
contiguous bytes. That inverts the current parallelisation and is a real kernel
change, not a tuning knob. It has not been built or measured, and the 1.5 GB/s
figure is evidence that *something* about the access is wrong, not proof that
this specific rewrite fixes it.

## What this retires

- "~10 % beyond traffic remains unexplained" (the framing this item carried for
  weeks): there is no unexplained residue. A two-parameter fit on measured KV
  bytes accounts for every point within 0.5 %.
- The upper-bound traffic estimate in the 2026-08-14 document (~568 MB per
  token at the longest span) was **4.3x too high** precisely because it omitted
  the sliding-window cap. The measured figure is 131 MB.
- Any remaining suspicion of attention *parallelism*: the chunked path already
  dispatches 256 threadgroups, and this shows the bottleneck is per-byte read
  efficiency, which more threadgroups do not change.


---

## Follow-up, 2026-08-16: the cooperative read was built, and coalescing is
## only a small part of it

`k_attn_coop` / `k_attn_chunk_coop` (behind `RUNNER_METAL_ATTN_COOP=1`) do
exactly what this document proposed: one simdgroup owns a KV row, its 32 lanes
split `head_dim`, and the per-row dot ends in a `simd_sum`.

A correction to the proposal above, found by the gate: patching `k_attn` alone
does nothing, because **at decode the CHUNKED path is taken whenever the span
is worth splitting**. `k_attn` only ever runs on short spans; the kernel that
matters at long context is `k_attn_chunk`. The tolerance gate reported "never
dispatched" before any timing was taken from it.

Measured, round-robin, one env-switched binary:

| span | coop off | coop on | delta |
|---:|---:|---:|---:|
| 2,326 | 12.26 | 12.63 | +3.0 % |
| 4,641 | 9.68 | 10.10 | +4.3 % |
| 8,110 | 7.24 | 7.47 | +3.2 % |

Real and consistent — no overlap between arms — but **the prediction was that
the slope would fall, and it barely does**: 0.681 → 0.660 ms/MB, i.e. the KV
read goes 1.47 → 1.52 GB/s. Some of the +3–4 % is a lower intercept, not a
lower slope.

So K-side coalescing is **not** the dominant cause of the 18× gap. The
hypothesis in the section above is thereby narrowed rather than confirmed.

### What the next candidate is, with arithmetic

The scores scratch. `ah` is `att_all + h * n_ctx` — **device** memory, one
float per position — and the kernel makes four passes over it per head per
layer per token: write the scores, read for the max, read-modify-write for
`exp`/sum, and read again in the V accumulation.

At an 8,110-token span with 8 heads over 35 layers that is
`8110 × 4 B × 8 × 4 passes × 35 ≈ 36 MB` of scratch traffic per decode token,
against 131 MB of KV. Roughly **28 % of the KV volume, in a buffer that never
needed to leave the chip** — a chunk's scores are `chunk` floats (≤ 256 at the
measured settings) and would fit threadgroup memory, which `k_attn_chunk`
already allocates 1024 B of.

That is arithmetic, not a measurement, and it is offered as the next thing to
test rather than the next thing to build.

## The scores buffer (`ah`), measured 2026-08-19 — smaller than the estimate

The open follow-up was whether the per-head attention scores buffer `ah`,
which the decode kernel round-trips through device memory, was worth moving to
threadgroup memory. The plan carried the arithmetic estimate "~36 MB per decode
token at an 8k span, 28 % of the KV volume." The dispatch census now counts the
`ah` round-trips directly (`RUNNER_METAL_STATS`, decode-only, ~4 passes per head
over the attended span), so it is a measurement rather than arithmetic.

Measured on e2b-q40 (gemma-4 E2B, 35 layers, 7 global / 28 sliding), 4,002-token
context, `--gpu auto`:

```
metal-kv     decode-cumulative: global 57387008 B | sliding 14680064 B  (72 MB/token)
metal-scores decode-cumulative: 5421696 B (ah round-trips) = 7.5% of KV read
```

**7.5 %, not 28 %.** The ratio is span-independent (both scale linearly with the
attended span), so it holds at 8k too; the earlier 28 % overcounted. The KV read
is the established decode bottleneck (1.5 GB/s, this doc); at 7.5 % of that
traffic, staging `ah` in threadgroup memory could return at most a few percent
of decode — and only if the kernel change were free, which it is not (a chunk's
scores fit threadgroup memory but a long span does not, so it needs a device
fallback above the threadgroup capacity).

**Verdict: deprioritized, not built.** The measurement shrinks the lever below
the threshold that would justify the kernel restructure ahead of the KV-read
work itself. Recorded as measured so it is not re-estimated. The census counter
stays (instrumentation only; Metal decode parity byte-identical).
