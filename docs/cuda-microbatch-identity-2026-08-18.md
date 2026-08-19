# The CUDA decode microbatch is not bit-identical on quantized models

Found 2026-08-18 during the CUDA backend review. **FIXED 2026-08-19 — see
"Resolved 2026-08-19" at the end.** Everything between here and that section is
the original diagnosis, kept as written: it was right about the mechanism, and
the two things it got wrong ("no 13.3 box exists", "Q5_K and Q6_K unmeasured")
are worth being able to see. The deferral reason in "Why this is deferred" is
the part that turned out to be false.

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

---

## Resolved 2026-08-19

The kernel fix was taken, not the host-only per-column option. All four steps
landed: `739329b` (kernels + regenerated PTX), `2ffdfb9` (cuda.c), `b943ccc`
(the quantized batch fixture). The diagnosis above stands as written; this
section records what it cost and what it turned out to have missed.

### The 13.3 box was the Windows one

`nvcc` release 13.3, V13.3.73, Build ID **CL-38244171** — the build stamped in
the committed header — was installed on the Windows box all along. `where cl`
comes back empty there because MSVC is not on PATH until
`VC\Auxiliary\Build\vcvars64.bat` runs; that, and nothing else, is why the
first attempt concluded no 13.3 box existed.

Regenerating from the UNMODIFIED `kernels.cu` on that box did NOT reproduce the
committed header byte for byte, and the difference is accounted for rather than
waved away. Three functions of ~100 differ (`k_mv_q3_K_b`, `k_store_kv`,
`k_store_kv_seq`), and across the WHOLE FILE the per-function multiset of
`.f32`/`.f16`/`.f64` opcodes is identical: the only opcodes that differ anywhere
are 64-bit integer address arithmetic (`mul.lo.s64` vs `mul.wide.s32`, a
hoisted `neg.s64`, a sunk `cvt.s64.s32`) — strength-reduction and scheduling
choices that cannot change a computed value. Two consecutive runs on that box
are byte-identical to each other, so this is a host instruction-selection
difference (MSVC vs whatever host produced the committed header), not
run-to-run nondeterminism.

The consequence is a methodological one worth keeping: **attribute a source
change by diffing the fixed regen against a BASELINE REGEN FROM THE SAME BOX**,
never against the committed header. Done that way, the only kernels whose
instruction stream changed are the eight rewritten twins and the two new ones;
every other kernel is identical after normalizing virtual-register and
basic-block numbering.

### Q5_K and Q6_K were failing too

The table above left them unmeasured. They were not fine — `./test-batch
<model> 4` at `3ae5d2b`, Blackwell MIG 1g.24gb:

| model | quant | before | after |
|---|---|---:|:--|
| SmolLM2-135M-Instruct | Q8_0 | FAIL, 1.83x | **ok, 1.89x** |
| Qwen3-4B | Q4_K_M | FAIL, 1.57x | **ok, 1.53x** |
| Qwen3-8B | Q5_K_M | FAIL, 1.51x | **ok, 1.69x** |
| Qwen3-8B | Q6_K | FAIL, 2.04x | **ok, 1.97x** |
| Qwen3-0.6B | Q4_0 | ok, 1.07x (refused) | **ok, 1.62x** |
| Qwen3-1.7B | Q4_0 | ok (refused) | **ok, 1.47x** |
| test.gguf | F32 | ok | ok |
| test-q8.gguf | Q8_0 | **FAIL** (new gate) | **ok** |

Identity therefore cost nothing measurable: two families gained throughput, two
lost 2-4%, against the 11-13% the host-only per-column option was measured to
cost. Q4_0 gained 48% over refusing, and beat the 1.48x per-column figure too.

### A third cause the note did not name

`batch_mv_twin_ok` returned `true` for **any** type without a decode GEMV, on
the comment "batch-1 uses f_mv; f_mvb is its twin". That is only true for the
MV_FMA family, where the per-element weight is the loaded element itself: F32
and F16. It is false for every quantized `_b` kernel, for exactly the reason
cause 2 gives. So a dense Q4_1/Q5_0/Q5_1/Q3_K/IQ4_NL/IQ4_XS/MXFP4 model was
being admitted to a microbatch that was not bit-identical, by the very check
that exists to keep it out. Those now decode sequentially.

The claim under that was measured rather than assumed: a probe build with
`f_gemvb` forced to 0, so a Q8_0 microbatch takes `f_mvb` — the substitution
being removed — exits 1 on SmolLM2-135M-Instruct-Q8_0 with the first bitwise
difference at step 0, at 0.29x, reproducing this note's isolation table on the
current binary. F16's twinhood is argued from the source, not measured:
`--quantize` emits only q8_0/q4_0/f16, its f16 pass leaves a BF16 model BF16,
and no F16 GGUF was available on the box.

### What holds it down now

`make test` runs `./test-batch` twice — on test.gguf (F32) and on test-q8.gguf,
a Q8_0 fixture from `scripts/make-test-model.py --quant q8_0`. The second run
was verified to FAIL on the pre-fix kernels (exit 1 at N=2/4/8, first bitwise
difference at step 0) and pass on the fixed ones, so it is a gate and not
decoration. Without it the next regression here would be equally invisible.
