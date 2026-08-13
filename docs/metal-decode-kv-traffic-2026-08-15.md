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
