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

The committed code (wider loads, one row per simdgroup) was re-measured on its
own, five interleaved rounds, against the branch point:

| model | before | after | delta |
|---|---:|---:|---:|
| e2b-q40, bandwidth-bound | 15.11 | 15.14 | +0.2 % |
| SmolLM2-135M-Q8_0, dispatch-bound | 106.68 | 107.13 | +0.4 % |

Both are inside the run-to-run spread. The honest summary is **no measurable
change on this machine**, in either direction.

### The box these numbers come from

An 8 GB M1 shared with other agents. Two methodology notes that cost real time
and are worth inheriting:

- **Interleave the arms.** A blocked A/B (all of A, then all of B) put the two
  arms in different minutes of the load curve and read +14 % on one pass and
  0 % on the next, with a 132 tok/s outlier inside a 105–108 block. Every
  number above is round-robin.
- **Discard the first run per (build, model).** The first touch of a
  multi-gigabyte mapped file measures page-cache misses: the opening A/B
  recorded 2.30 and 0.68 tok/s against a settled 15–16.

granite-4.1-8b-Q4_0 cannot be measured here at all, and says so: whole-model
Metal is refused because 5.1 GB of weights do not fit beside 2.6–3.8 GB of free
RAM, and `--gpu auto` silently falls back to CPU. Its byte-identity check uses
`--gpu-layers 8` to force the split the refusal message points at.

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

## Follow-up, 2026-08-13: the reassociating kernel was built, and is also neutral

The section above ends by saying the roofline gap needs the transformations
identity forbids, on a second kernel promoted by teacher-forced tolerance. That
kernel now exists — `k_mvf_q4_0` / `k_mvf_q8_0`, selected at `n_col == 1`
behind `RUNNER_METAL_MV=1`, gated by `tests/test_mv_tol.c`. It ships **off**.

It takes both levers identity forbids: float4 accumulation instead of a scalar
`+=` chain, and the q4_0 zero-point factored out of the inner loop
(`Σ(q−8)·y·d` → `d·Σq·y − 8d·Σy`), which deletes a per-element integer
subtract. Together those cut issued ALU instructions per consumed weight byte
roughly fourfold.

The hypothesis was that at 36 % of roofline the route is **issue**-bound rather
than traffic-bound — weight bytes are irreducible, every byte is read exactly
once either way, so instructions per byte is the only thing left to cut. Same
machine, same round-robin protocol, same first-run discard; both arms are one
binary switched by env, so no build difference confounds it.

| model | bound by | scalar | fast | delta |
|---|---|---:|---:|---:|
| e2b-q40, 2.6 GB q4_0 | bandwidth (~40 of ~68 GB/s) | 15.54 | 15.51 | **−0.16 %** |
| SmolLM2-135M-Q8_0, 145 MB | dispatch (~19 GB/s) | 111.87 | 111.86 | **−0.01 %** |

Both are inside the run-to-run spread, at both ends of the bound spectrum. **The
hypothesis is refuted on this silicon.** Cutting ALU issue fourfold changes
nothing, which is what a genuinely memory-bound kernel looks like: the M1's
decode matvec is waiting on bytes, not on instructions, and no arithmetic
rearrangement will move it.

The tolerance side passed cleanly, on both formats:

| model | fast matvecs | mean\|dlogit\| | as fraction of range | top-1 flips |
|---|---:|---:|---:|---:|
| e2b-q40 | 5,798 | 0.000767 | 0.00002 | **0/64** |
| SmolLM2-135M-Q8_0 | 13,295 | 0.000664 | 0.00002 | **0/64** |

against a 0.005 deviation limit and a zero-flip promotion bar. The determinism
control — the reference re-run — read exactly 0.000000000 and 0/64 both times,
which also pins something no other gate states: Metal decode is run-to-run
reproducible, so a difference this harness reports is the kernel and not the
GPU.

So the artifact is correct, promotable on the numbers, and **not promoted**,
because it buys nothing here and costs the byte-identity contract. That is the
same disposition as the fused CPU int8 dot (`RUNNER_CPU_I8`), and for a
stronger reason: that one at least failed on flips, this one passes and simply
does not pay.

### What this DOES settle, and what it does not

Settled on M1: the decode matvec has now been attacked from three independent
directions — load width, row assignment, and arithmetic reassociation — and all
three measured neutral or worse. The remaining levers are not in the kernel's
arithmetic at all. They are **traffic** (fewer bytes moved: smaller weights, or
a KV/activation layout that reads less) and **dispatch** (fewer, larger command
encodings). That is a genuinely useful narrowing: it retires a whole family of
proposals.

Not settled: the M5 Max 36 %-of-roofline measurement that started this. That
silicon has several times the bandwidth and core count, so the balance between
resident simdgroups and per-lane fetch depth — and therefore whether it is
issue-bound where the M1 is bandwidth-bound — is exactly what differs. The
difference now is that the experiment no longer needs writing: the kernel, the
env switch and the gate are all in the tree, so answering it on M5-class
hardware is `RUNNER_METAL_MV=1`, one bench and one `./test-mv-tol`.
