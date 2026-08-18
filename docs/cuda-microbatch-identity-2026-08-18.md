# The CUDA decode microbatch is not bit-identical on quantized models

Found 2026-08-18 during the CUDA backend review. **Not fixed** — the fix is in
`kernels.cu`, and the only CUDA box available cannot regenerate the committed
PTX (see "Why this is deferred"). This note is so the next attempt starts from
the diagnosis rather than the symptom.

## The claim that is false

`gpu.h`, on `gpu_batch_decode`:

> a batched step must produce, for each sequence, the bits a lone step would
> have produced. Nothing here reduces across sequences, and the matvecs pick
> the multi-column twin of whatever kernel the batch-1 path would have used, so
> that holds by construction rather than by luck.

`kernels.cu` repeats it: *"Identity is not an accident here; it is the
selection rule. tests/test_batch.c holds it down."*

It does not hold. `./test-batch <model> 4` on the Blackwell box, at
`8111c36`:

| model | quant | result |
|---|---|---|
| `test.gguf` (the fixture `make test` uses) | F32 | **ok** |
| SmolLM2-135M-Instruct | Q8_0 | FAIL, first difference at step 0 |
| Qwen3-4B | Q4_K_M | FAIL, first difference at step 0 |
| Qwen3-0.6B / Qwen3-1.7B | Q4_0 | FAIL (fixed separately — see below) |

Sampled tokens matched the sequential reference in every run above; the
divergence is at the last mantissa bits. It is a broken invariant, not visibly
broken output — but the invariant is what lets a sequence leave a microbatch
and continue solo, so it is worth having.

## Why `make test` is green

`make test` runs `./$(TEST_BATCH)` with no argument, i.e. on `test.gguf`, which
is F32. F32 is the one case where the twin relationship still holds: batch-1
takes `k_mv_f32` (no `f_gemv` for F32) and the microbatch takes `k_mv_f32_b`,
and those two really do accumulate in the same order. The gate is therefore
vacuous exactly where it is run, and sound nowhere it is not.

## Two independent causes

**1. The width-classed twins are twins of a kernel that no longer exists.**
`k_gemvb_q8_0_x4/_x8` and friends were written in `d0439ea` (2026-07-20)
against the then-current decode GEMVs: lane `l` owns element `l` of one block,
blocks walked sequentially. `7ef0209` (2026-07-28, the decode-bandwidth pass)
rewrote `k_gemv_q8_0`/`_q4_K`/`_q5_K`/`_q6_K` to their v2 forms — each with its
own wider load shape (Q8_0: four blocks in flight, lane `l` taking a
four-element slice of block `l>>3` via `float4`; Q4_K/Q5_K: lane `l` taking
elements `[l*8, l*8+8)` with the per-group affine factored out as
`dg·Σ(q·x) − mmg·Σx`; Q6_K: two blocks per trip with independent
accumulators) — and left the `GEMVB_*` macros at v1. A different
lane→element mapping is a different `warp_sum` partition, hence different
rounding, and the factored affine is a different expression again. The twins
were correct for eight days.

**2. `k_mv_*` and `k_mv_*_b` were never twins for quantized types.**
`k_mv_q8_0` accumulates `d * (Σ q[j]·x[j])` per block; `k_mv_q8_0_b` accumulates
`Σ (d·q[j])·x[j]`. Same values, different association. The same split exists in
every quantized `_b` pair. Confirmed by forcing both paths onto the `k_mv`
family: `test-batch` still fails.

Note that `k_mv_*_b` is also the **prefill** tile kernel, and prefill is
certified as it stands. Changing its reduction to match `k_mv_*` would move
prefill numerics for every model without a `k_gemm_*`, so cause 2 is not a
free edit — the microbatch should stop reusing it, rather than the kernel
changing under prefill.

## What was fixed instead

Q4_0 had no width-classed twin at all (it grew `k_gemv_q4_0` on 2026-08-13 and
never got `k_gemvb_q4_0_x4/_x8`), so `enc_mv_batch` fell through to `f_mvb` —
which has no shared-memory activation staging. That cost both halves at once:
`Qwen3-0.6B-q4_0` at N=4 ran **0.11x of sequential decode**, nine times slower,
*and* failed identity. `8111c36` makes `batch_eligible` refuse a model whose
weights have a decode GEMV but no width-classed twin, so those sequences decode
one at a time (1.09x, and identity restored trivially). The types that do have
twins still batch and are unaffected: Q8_0 measured 1.84x and Q4_K 1.57x at
N=4, both still not bit-identical. Q5_K and Q6_K were not measured.

## Isolation: it is the matvec, and only the matvec

Three builds, `./test-batch <model> 4`, Blackwell MIG 1g.24gb, rebuild confirmed
between each:

| `enc_mv_batch` does | Q8_0 SmolLM2-135M | Q4_K Qwen3-4B | identity |
|---|---:|---:|:--|
| shipping: `f_gemvb` width-classed twins | 1.83x | 1.56x | **FAIL** |
| forced to `f_mvb` | 0.29x | 0.05x | **FAIL** |
| the EXACT batch-1 kernel, once per column | 1.62x | 1.35x | **ok** |

The third row is the proof that nothing else diverges: rope, KV store,
flash-decoding attention and the merge are shared or per-column already, and
with the matvec made per-column the batched logits are bit-identical to
sequential on both models. It is also a **host-only** fix — no kernel change,
no PTX — because launching `f_gemv`/`f_mv` with `batch = 1` and the column's own
x/y offsets is literally the solo forward's launch.

Q4_0 measured on the same build (`Qwen3-0.6B-q4_0`, with the
`8111c36` refusal disabled): **1.48x, identity ok** — better than the 1.09x
that refusing gives it.

So there are two ways to make the tree honest, both measured:

- **Per-column (host-only, landable anywhere).** Exact identity everywhere.
  Q4_0 +48% against refusing; Q8_0 1.83x -> 1.62x and Q4_K 1.56x -> 1.35x, i.e.
  11-13% of the batching win spent on the contract. Leaves `k_gemvb_*` unused
  by the decode path.
- **Fix the twins (needs CUDA 13.3).** Exact identity at no throughput cost,
  and `k_gemvb_*` keeps its job. This is the better end state.

Neither was landed. The first is a throughput trade-off that is a product
decision rather than a bug fix, and the second is blocked below. What WAS
landed is the case where no trade-off exists: Q4_0, which was slower than not
batching at all.

## Not the cause: KV copy-back

`RUNNER_CUDA_PROFILE=1` on Qwen3-4B-Q4_K_M, 128 decode tokens, full offload:
matvec is 71.1% of GPU-busy time, logits-mv 8.4%, norms 9.7%, attention 4.9%.
Host-side `kv_copyback` is 67.8 ms of a 1364.9 ms wall — about 5%, despite
being `2 x n_layer` separate small DtoH copies per token. Batching those copies
is not worth pursuing; the decode path is matvec-bound, as expected.

## Why this is deferred

`src/kernels_ptx.h` is generated by **CUDA 13.3** (`CL-38244171`). The
measurement box has 13.0 in the `ccbuild` conda env, 13.0 in `cudatk`, and
`/usr/local/cuda-13.2`; there is no 13.3 anywhere on it. Regenerating the header
here would replace every kernel's PTX with a different compiler's output — a
whole-file diff that moves register allocation across kernels nobody touched and
invalidates the certs that were taken against the committed header. So no
`kernels.cu` change can land from this box, and `make ptx` additionally needs a
plain `gcc` on PATH (the conda toolchain only provides
`x86_64-conda-linux-gnu-gcc`; a symlink is enough).

## The fix, when a 13.3 box is available

1. Re-derive the four `GEMVB_*` macros from the current v2 `k_gemv_*` bodies —
   same lane→element mapping, same per-lane accumulation, x read from shared
   memory instead of global. Reading x from smem changes no value; that was the
   original design and it is still the right one.
2. Add `k_gemvb_q4_0_x4/_x8` from `k_gemv_q4_0`, and drop the Q4_0 refusal that
   `8111c36` added.
3. Make `enc_mv_batch` refuse rather than substitute `f_mvb` for any remaining
   type, so cause 2 cannot come back silently through the fallback.
4. Give `make test` a non-vacuous batch gate: a small **quantized dense**
   fixture, so `./test-batch` exercises a `f_gemv`/`f_gemvb` pair rather than
   the F32 path. Without this the gate goes back to proving nothing.

Step 4 is the one that matters most and is worth doing even if the rest waits:
without it the next regression here is equally invisible. It cannot land on its
own today, because on the current kernels it would be a red gate.

If a 13.3 box is not coming, take the per-column option above instead — it is
correct today, it costs 11-13% of the batching win on Q8_0/Q4_K, and it gains
48% on Q4_0. The one thing not worth doing is leaving the contract as written
while the code does something else.
