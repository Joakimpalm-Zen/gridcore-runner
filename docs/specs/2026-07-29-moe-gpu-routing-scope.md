# MoE GPU decode/prefill: device routing + expert batching — scope & design

Status: **designed, not started** (2026-07-29, Blackwell box). Implementation
belongs on the CUDA 13.3 / RTX 3070 machine — every lever below is
`kernels.cu` work requiring PTX header regeneration (`make ptx`), and the
3070 is also the small-GPU deployment the work primarily serves.

## Why (owner direction)

Sparse MoE is the small-GPU strategy: a 30B-A3B carries 30B-class quality
while streaming ~3.3B active params per token, and `--cpu-moe` already puts
that on an 8 GB card. The published benchmark (docs/benchmarks.md) shows the
gap: dense decode is 73–79% of llama.cpp; **qwen3moe decode is 48%, gemma-moe
21%; MoE prefill is 0.6–3.3%**. Closing the MoE rows toward the dense band is
the highest-value performance work left on the table.

## Measured baseline (v0.1.4-alpha `956af0b`, Blackwell MIG 1g.24gb, 512/128 greedy, median of 3)

| model | quant | decode: runner / llama.cpp | prefill: runner / llama.cpp |
|---|---|---|---|
| Qwen3-30B-A3B | Q4_K_M | 72.5 / 151.7 (**48%**) | 106.4 / 3233.5 (**3.3%**) |
| gemma-4-26B-A4B | Q4_0 | 23.7 / 114.2 (**21%**) | 22.7 / 3694.2 (**0.6%**) |

Reference: dense Q4_K decode 73–79% on the same box. TC on qwen3-30b prefill
measured **+6% only** — the expert FFN dominates and runs non-TC
single-column, so TC is gated on expert batching (P2 below), not the fix
itself.

## Where the time goes (read from `gpu_moe_ffn` / `gpu_gemma_moe_ffn`, 2026-07-29)

Per token, per MoE layer, the eager path does:

1. router `enc_mv` → `cuStreamSynchronize` → `cuMemcpyDtoH` of `n_expert`
   logits → **host** softmax + top-k + renormalize;
2. per selected expert (top-8 on qwen3moe): gate `enc_mv`, up `enc_mv`,
   `enc_actmul`, `enc_scale`, down `enc_mv`, `enc_add` — each a
   single-column GEMV over `n_ff_exp` (768 on qwen3-30b), far too narrow to
   feed the GPU.

Qwen3-30B (48 MoE layers): ~48 sync+DtoH round-trips and ~1300 launches per
decoded token, and the host round-trip forces `graph_bad` — no CUDA graph.
gemma-4 adds its dense shared branch per token on top. Prefill runs the same
per-token loop; the CPU path's group-tokens-by-expert trick (`cabdad1`,
~5.6×) was never ported to CUDA.

## P1 — device-side routing + fused indirect expert GEMV (decode lever)

**Routing kernel.** One small kernel per MoE layer computes softmax → top-k →
renormalized weights entirely on device, writing `sel[k]`/`selw[k]` to device
memory. **Bit-identity requirement:** the CPU host routing is the certified
reference; to keep the byte-identical CPU==GPU gate, implement the routing
serially (one thread per token) mirroring the host arithmetic exactly — same
max-scan order, same `expf` per element, same summation order, same
tie-to-lowest-index rule. At `n_expert ≤ 256` a serial thread is
microseconds; do not "optimize" this kernel with reductions that reorder
floats.

**Indirect expert GEMV.** One launch (or two: gate+up fused, then down)
computes all top-k experts' matvecs, reading expert indices from `sel[]` on
device — block y-dim = expert slot, weight base computed in-kernel from the
expert id (the fused-3D expert stride is already computed host-side today in
`moe_expert_weight`; move the same arithmetic into the kernel args). Reuse
the 7ef0209 GEMV patterns (aligned 8-byte quant loads, `float4` activation
loads). Accumulate `selw`-scaled down-projections with the existing add
pattern.

**CUDA graph re-enable.** With no host-dependent branching left, drop the
`is_moe → graph_bad` forcing for this path and let decode capture. Launch
count per token falls from ~1300 to ~150-ish; the ~48 syncs go to zero.

**Expected effect:** decode moves toward the dense band (70%+ of llama.cpp)
— same bandwidth wall, the overhead is what's being removed. Publish the
measured number, not this estimate.

## P2 — expert-grouped prefill GEMM (prefill lever, prerequisite for TC)

After routing a prefill tile (device kernel from P1; one DtoH of the tile's
routing is acceptable here — prefill is not graph-captured), build per-expert
token lists and run **one batched GEMM per active expert** over its token
set, using the existing `k_gemm` family. This is the CUDA port of the CPU
path's `cabdad1` grouping. Only after this lands does TC matter for MoE:
the expert GEMMs become `k_gemm_q4_K_tc`-eligible, and qwen3moe already
**passed** its TC tolerance gate (0.216% of range, one near-tie in 64) — the
promotion decision should be revisited with fresh gate rows once grouped
prefill exists. gemma-4's dual-branch dense shared FFN should batch across
tokens in the same pass.

## P3 — Q8_0 / Q4_0 TC twins (separate, already tracked)

gemma-4-26B is Q4_0 and has no TC kernel at all. The Q8_0 twin is a tracked
remainder from the TC promotion; add Q4_0 with the same MMQ-style structure
and gate each (type, arch) row through `make test-tc-tol` before any
promotion.

## Interplay and non-goals

- **`--cpu-moe` is untouched** by P1/P2 (host-resident experts keep the host
  FFN); its own lever is the tracked VNNI CPU-dots item. The small-VRAM
  benchmark scenario (`--cpu-moe` vs llama.cpp `--n-cpu-moe`) is a
  measurement task, independent of this spec.
- No new router topologies here — sigmoid/bias/group-limited routing is the
  separate "generalized MoE router" plan item. This spec only accelerates
  the existing certified top-k path.
- Keep `compute_75` portability: no arch-specific intrinsics in the new
  kernels; PTX stays single-target.

## Gates (all existing, all must hold)

1. `make test` green incl. `test-moe` (7 tests — includes the
   GPU-actually-executed fallback guard) — on the 3070 the drift check also
   passes since that box owns the 13.3 header.
2. **Byte-identical CPU==GPU greedy** on Qwen3-30B-A3B and gemma-4-26B-A4B
   (the routing bit-identity requirement above exists for exactly this).
3. `greedy_reference` unchanged vs pinned b10076 on both MoE models.
4. `bench.sh` md5s unchanged.
5. Per-phase perf numbers recorded on BOTH boxes (3070 full-card and
   Blackwell MIG) before updating docs/benchmarks.md — the published page
   only changes with re-measured rows.
6. Any TC involvement goes through `make test-tc-tol` per (type, arch);
   promotion remains an owner decision.

## Baseline repro

```sh
# runner side (this spec's baseline used the default path)
./runner -m Qwen3-30B-A3B-Q4_K_M.gguf -f prompt-512tok.txt -n 128 \
         --temp 0 -s 1 --ignore-eos --gpu-layers 99
# llama.cpp side
llama-bench -m Qwen3-30B-A3B-Q4_K_M.gguf -p 512 -n 128 -ngl 99 -r 2
# profile where the time goes
RUNNER_CUDA_PROFILE=1 ./runner ... -n 8
```
