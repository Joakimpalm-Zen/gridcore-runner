# Metal dispatch census: where the batch dimension actually collapses

*2026-08-13. Apple M1, 8 GB, 8 GPU cores. Diagnosis, not a fix.*

Two questions had been open since the +23.6 % prefill GEMM work moved the
numbers without answering them:

1. where does prefill's batch dimension collapse — the standing suspicion was
   "matvec fallback is still per-token dispatches inside the command buffer";
2. what does decode execute at `n == 1` that the profiler cannot account for.

Both are now answered with numbers, and the first answer is **not** the
suspected one.

## The instrument

`RUNNER_METAL_STATS=1` now prints a per-kind dispatch census beside the
existing command-buffer line:

```
metal-census n=64 total=15947 | mm=275 mv=1 mvf=0 rmsnorm=176 qknorm=3200 \
  headnorm=960 rope=50 store=15 attn=35 attn_chunk=0 elem=11235 moe=0
```

Counters are cumulative over the process, so a single forward is read as a
difference — or, as below, from the first forward of a fresh run. This exists
because throughput cannot distinguish a kernel that takes the batch in `grid.y`
from one encoded `n` times: both cost the same to write and look identical from
outside.

## 1. Where prefill's batch dimension collapses

Same 69-token prompt, e2b-q40 (gemma-4 E2B, 35 layers), first prefill forward
at four batch sizes:

| n | total | mm | mv | rmsnorm | rope | store | attn | qknorm | headnorm | elem |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 2507 | 275 | 1 | 176 | 50 | 15 | 35 | 400 | 120 | 1435 |
| 16 | 4427 | 275 | 1 | 176 | 50 | 15 | 35 | 800 | 240 | 2835 |
| 32 | 8267 | 275 | 1 | 176 | 50 | 15 | 35 | 1600 | 480 | 5635 |
| 64 | 15947 | 275 | 1 | 176 | 50 | 15 | 35 | 3200 | 960 | 11235 |

The split is exact, with no fitting required:

- **constant in n** (properly batched): `mm` 275, `mv` 1, `rmsnorm` 176,
  `rope` 50, `store` 15, `attn` 35 — 552 dispatches, plus 35 of `elem`.
- **linear in n** (collapsed): `qknorm` = 50n, `headnorm` = 15n,
  `elem` = 175n — **240 dispatches per token**.

So `total = 240n + 587`. At n = 64 that is **15,360 of 15,947 dispatches
(96.3 %) encoded one token at a time.**

### The suspect is refuted

`mv` is **1**, and constant in n. The matvec fallback is not per-token: since
the `n_col`/`col_tile` work, `enc_mv_n` takes the whole batch in one dispatch
with `grid.y` tiling whether or not a tiled-GEMM kernel exists for the type.
Only one weight in this model lacks an `mm` kernel, and it costs exactly one
dispatch per forward, not n.

### What is actually collapsed

Three explicit `for (int b = 0; b < n; b++)` loops in
`gpu_forward_native_batch`, all of them offsetting with `foff(b * stride)`:

- `elem` — 175n, **73 % of all per-token dispatches**: residual adds,
  activation multiplies (`silu`/`gelu`/`sigmoid` × gate) and the gemma-class
  post-norm scales, in both the attention and FFN halves of every layer.
- `qknorm` — 50n: Q-norm on all 35 layers plus K-norm on the 15 KV-owning ones.
- `headnorm` — 15n: the V head-norm on the KV-owning layers.

Rope, KV store and attention are **already** batched — each derives its
column's position from `pos + col` and takes n in `grid.y`. Only the
element-wise and per-head-norm ops never received that treatment.

## 2. What decode executes at n == 1

Steady-state decode, warm, twelve consecutive tokens (e2b-q40, short context):

```
encode=0.45ms  submit+wait=45.5ms  gpu=44.9ms  total=46.0ms   (medians)
```

**GPU execution accounts for 98.7 % of the wall time.** Encode is 1.0 %; the
residue between `submit+wait` and `gpu` is 0.8 ms, about 1.8 %.

That retires the hypothesis written into `metal.m`'s own timing comment — "the
two candidate explanations, encode overhead at ~420 dispatches per token versus
the kernels themselves, are indistinguishable from throughput alone." They are
now distinguished. It is the kernels. There is no unaccounted decode work: the
profiler's missing time is GPU-side kernel execution, and
`docs/negative-result-metal-multirow-matvec.md` establishes what bounds *that* —
memory traffic, not instruction issue.

## 3. Does the dispatch count cost anything?

Encode time says no, directly: 3.09 ms of CPU encoding against 451 ms of GPU
execution at n = 64 (0.7 %). The 15,360 per-token dispatches are nearly free to
*write*. The question is what they cost to *run*.

Fitting `t(n) = a + b·n` to warm GPU times:

| model | resident? | fixed a | slope b |
|---|---|---:|---:|
| e2b-q40, 2.6 GB | no — 1.1 GB free, pages | 180 ms | 4.24 ms/token |
| SmolLM2-135M-Q8_0, 145 MB | yes | 17.2 ms | 0.33 ms/token |

**This slope does not attribute cleanly, and it would be wrong to claim it
does.** It contains two things at once: the batched GEMM genuinely doing n
times more arithmetic, and the 240 per-token dispatches. Nothing measured here
separates them.

What can be said is a bound. On the resident model, *if* the whole slope were
per-token dispatch overhead, it would be 0.33 ms / 240 = **1.4 µs per
dispatch** — squarely the right order for Apple Silicon dispatch overhead,
which means the per-token dispatches cannot be dismissed as noise and cannot be
convicted either. The e2b figures are additionally paging-confounded (2.6 GB of
weights against 1.1 GB of free RAM; the fixed 180 ms works out to 14.4 GB/s,
well under the M1's ~68) and should not be read as bandwidth numbers.

## The experiment that would settle it

Batch **one** of the three collapsed kinds and re-measure the slope. `qknorm`
is the contained choice — one kernel, one encode function, one call site — and
is 50n of the 240n, i.e. 21 % of per-token dispatches. If the slope falls by
roughly a fifth of its dispatch-attributable part, the attribution is proved
and batching `elem` (73 %) becomes a quantified win rather than a plausible
one. If the slope does not move, the per-token dispatches are free and the
whole line of attack is retired for the cost of one kernel.

That experiment is the natural next step and is deliberately **not** bundled
into this diagnosis.

## The experiment, run

`qknorm` and `headnorm` are now batched in `grid.y`. Each `(head, column)` pair
was always an independent reduction, so this is **bit-identical**: only the
encoding changes, and the CPU/GPU parity gates still pass byte for byte across
six models including the E-series path that exercises both kernels.

Census confirms the intended effect exactly — `qknorm` 50n → 50, `headnorm`
15n → 15, so `total` falls from `240n + 587` to `175n + 587`. At batch 64 that
is **15,947 → 11,852 dispatches, −25.7 %**.

Round-robin A/B, two binaries built from the same tree, five interleaved
iterations plus a discarded warmup, e2b-q40 (the model that *has* QK-norm):

| | before | after | delta |
|---|---:|---:|---:|
| prefill tok/s | 73.74 | 75.29 | **+2.10 %** |
| decode tok/s | 15.09 | 15.18 | +0.58 % |

Every "after" reading (75.16–75.76) sits above every "before" reading
(73.33–74.14), with no overlap across the five pairs — this is signal, not
spread. Decode is unchanged as expected: at `n == 1` batching is a no-op.

**So the attribution is settled: per-token dispatches do cost real time.**
Removing 27 % of them (65n of 240n) bought +2.10 % of prefill.

Extrapolating linearly in dispatch count — an assumption, stated as one —
batching `elem` (175n, the remaining 73 %) is worth on the order of **+5–6 %**
more prefill. That is now a quantified target rather than a plausible one, and
it is the single largest remaining prefill lever on this backend.

### Two methodology notes that cost time here

- **SmolLM2-135M has no QK-norm at all** (`qknorm=0, headnorm=0`). The first
  attempt at this measurement used it and read a meaningless ±9 % of noise on
  a model where the patch is a no-op. Check the census says your change is
  even *reachable* on the model you are about to measure.
- **`git stash` + `make` does not rebuild.** Stash restores mtimes that make
  considers current, so "before" and "after" came out byte-identical
  (`shasum` caught it — same hash twice). This is the rebuild trap AGENTS.md
  documents, in a new disguise: `touch` the sources after any checkout or
  stash that is meant to change the binary, and verify the two binaries
  actually differ before trusting a single number from them.

## Standing conclusions

- The prefill batch dimension collapses in `elem`, `qknorm` and `headnorm` —
  96.3 % of dispatches at batch 64 — and **not** in the matvec fallback, which
  is one batched dispatch.
- Decode at `n == 1` is 98.7 % GPU execution. Encode overhead is retired as an
  explanation for anything.
- Dispatch *encoding* is free (0.7 % of prefill). Dispatch *execution* is
  **not**: batching `qknorm`/`headnorm` removed 27 % of per-token dispatches
  and bought +2.10 % prefill, bit-identically. Batching `elem` (the remaining
  73 %) is the largest prefill lever left on this backend, estimated at
  +5–6 %.
