# Negative result: multi-row-per-simdgroup Metal decode matvec

*2026-08-12. Status: implemented, measured, rejected. Not in the product.*

## What prompted it

An external evaluation measured Runner's Metal decode at 36 % of the
memory-bandwidth roofline on an M5 Max — 79.6 tok/s on a 4.65 B Q4 model
against llama.cpp's 182.7, roofline ~224 — and named the matvec shape as the
cause. `src/kernels.metal` gives one output row to one simdgroup and walks it
with per-byte scalar loads, so each lane has a single outstanding fetch and
nothing to overlap it with. llama.cpp's Metal matvec gives each simdgroup four
rows.

## The constraint that shapes the whole problem

The scalar/decode route is the project's CPU↔GPU byte-identity contract. Each
output row accumulates over `for (i = tiisg; i < n; i += 32)` into one scalar
and finishes with one `simd_sum`. That pins the lane→block mapping: **which**
lane owns **which** block determines the partial sums that enter the reduction
tree, so it cannot be changed without changing the output bytes.

That rules out most of what llama.cpp does to reach 82 %:

- its `sumy` factorisation, which turns `Σ (q−8)·y` into `d·Σ q·y − 8d·Σ y`,
- giving each thread half a block so consecutive lanes read adjacent halves,
- float4 accumulators with a horizontal sum at the end.

All three reassociate the sum. Only two levers survive the contract:

1. **wider loads** — read the same bytes a lane already owns as `uchar4` /
   `packed_float4` instead of one at a time,
2. **multi-row per simdgroup** — give one simdgroup MV_NR rows, so a lane
   carries MV_NR independent fetch chains and reads the activation block once
   for all of them.

Both were implemented, byte-identity-preserving, and measured.

## What was measured

Apple M1, 8 GB, 8 GPU cores, shared with other agents. `--bench-json -c 1024`,
gen_tok_s. Arms **round-robin inside each iteration**, six iterations: a
blocked A/B put the two arms in different minutes of a shared box's load curve
and read +14 % on one pass and 0 % on the next, with a 132 tok/s outlier inside
a 105–108 tok/s block. Round-robin makes both arms sample the same drift.

Model: `e2b-q40.gguf`, 2.6 GB of q4_0. It is the only local model that is
bandwidth-bound — ~40 GB/s of the M1's ~68. `SmolLM2-135M-Q8_0` is not: 145 MB
at 130 tok/s is 19 GB/s, so it is dispatch-bound, and all variants landed
within 2 % of each other there. It cannot answer this question.

| arm | rows/simdgroup | gen tok/s (median of 6) | vs base |
|---|---:|---:|---:|
| shipped kernels | 1 | 15.41 | — |
| wider loads only | 1 | 15.47 | +0.4 % |
| wider loads, 64-wide threadgroups | 1 | 15.44 | +0.2 % |
| wider loads + multi-row | 2 | 15.42 | 0.0 % |
| wider loads + multi-row | 4 | 14.76 | **−4.2 %** |
| wider loads + multi-row | 8 | 50.97* | **−53 %** |
| wider loads + multi-row | 16 | 43.86* | **−59 %** |

\* MV_NR 8 and 16 were measured on SmolLM2-Q8_0 (109.7 tok/s base) — the
collapse there is so large it needed no bandwidth-bound confirmation.

## Why multi-row is rejected

Under the identity contract, simdgroups × rows-per-simdgroup is fixed at
n_out. Multi-row therefore does not *add* parallelism, it **trades** it: four
rows per simdgroup means a quarter of the simdgroups. On silicon with 8 GPU
cores the old shape was already keeping the machine fed, so the trade is a
straight loss of resident simdgroups for latency hiding that was not the
binding constraint. At MV_NR 8 and beyond the per-thread accumulator and row-
pointer arrays stop fitting in registers, spill to scratch, and decode halves.

Wider loads survive: neutral, not a regression, and a strict reduction in
issued memory instructions (48 → 12 per q4_0 block, 64 → 16 per q8_0 block,
128 → 32 per q4_K quarter-superblock). That part is in the product.

## What this does not settle

The M5 Max measurement is not reproduced or refuted here — that silicon has
several times the bandwidth and core count, and the balance between resident
simdgroups and per-lane fetch depth is exactly what differs. What the M1 does
establish is the shape of the ceiling: **on this route the only reachable
levers are load width and row assignment, and row assignment is
zero-sum.** Closing a 36 %-of-roofline gap needs the reassociating
transformations, which means a second kernel promoted by teacher-forced
tolerance the way the tiled prefill GEMM already is — not a change to the
byte-identical path.

An earlier note in `k_mv_q4_K` recorded that "hand-vectorising this loop is
SLOWER (6.58 → 6.43 tok/s)". This work explains it: that attempt moved the sum
into float4 accumulators and paid for four horizontal reductions. Widening only
the loads, with every `+=` keeping its original expression, does not.
