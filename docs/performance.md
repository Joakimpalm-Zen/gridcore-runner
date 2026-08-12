# Performance: closing the CPU/GPU gap

Where Runner stood against llama.cpp/Ollama, what was fixed, and the levers that
remain. All numbers below are Llama-3.2-3B-Instruct-Q4_K_M, 128-token greedy
decode (includes model load), on a Ryzen Threadripper 9980X (Zen 5, 64c/128t,
full AVX-512 + VNNI + BF16) with a Blackwell GPU (MIG 1g.24gb slice).

## Fixed 2026-07-22 — the CPU default was leaving ~6x on the table

Two default-configuration bugs, not algorithmic ones, made the CPU path far
slower than it should be. Both are fixed in `40bf1b9`.

### 1. The SIMD build was silently disabled (the big one)

`quants.c` gates its AVX2/FMA/F16C dot kernels behind
`#if defined(__AVX2__)`. The Makefile intends `-march=native` to define that,
but used `CFLAGS ?=` — a conditional assignment that a conda/distro toolchain
exporting `CFLAGS=-march=nocona -O2` **skips entirely**. The result: `__AVX2__`
undefined, every SIMD kernel `#if`-compiled out, and a **scalar binary shipped
on AVX-512 hardware** (`objdump`: zero `ymm`/`zmm` instructions).

Fixed with a plain `CFLAGS +=` (not `override`): it appends to an *environment*
CFLAGS so `-march=native -O3` win back the codegen, but is ignored for a
*command-line* CFLAGS so the release build's portable `-march=x86-64-v3` pin
survives. After the fix: `ymm` 0 → 2083, `zmm` 0 → 883, and the output is
**token-identical** to the scalar build (the kernels preserve accumulation
order), so it passes the verification gate cleanly.

### 2. The thread default was `min(8, cpus)`

8 threads on a 64-core box. Raised to a physical-core proxy `min(nc/2, 64)`
(SMT siblings add nothing to a compute-bound decode — measured plateau at
physical cores). Per-row partitioning makes it deterministic, so token-identical
across thread counts. Scope a shared box with `--reserve-cpu` or pin with `-t`.

### Measured effect

| build / config | time | vs old |
|---|---:|---:|
| old: scalar, default 8 threads | 35.05s | 1.0x |
| SIMD only, `-t 64` (vs scalar `-t 64` 9.97s) | 5.78s | 1.7x |
| threads only, scalar `-t 64` (vs 32.2s at `-t 8`) | 9.95s | 3.2x |
| **new: SIMD + physical-core default** | **5.75s** | **6.1x** |

The 100.5s SmolLM2 torture run would now be ~16s. GPU decode is **2.23s**
(2.6x the fixed CPU), and CPU/GPU top-1 tokens match 0/64.

## Measured and rejected — CUDA virtual-arch bump

The embedded PTX is built from `compute_75` and targets `sm_75`; the documented
minimum for CUDA offload is **NVIDIA Turing / compute capability 7.5 or newer**.
The measurement GPU is Blackwell (`sm_120`). Regenerating the PTX at
`compute_120` and re-benchmarking gave **2.12s vs 2.23s — within noise.** The
driver JITs `compute_75` PTX to Blackwell SASS at load either way, and Runner's
hand-written matvec kernels use no features (tensor cores, async copy) that a
newer *virtual* arch would unlock. Bumping would only cost portability
(`compute_75` JITs to supported GPUs at Turing or newer; older NVIDIA GPUs fall
back to CPU). **Kept at `compute_75`.**

> **Update (later 2026-07-22):** the tensor-core lever was then built and
> measured — and **lost at the runner's batch width**. A correct WMMA Q4_K GEMM
> (token-identical, opt-in `RUNNER_CUDA_TC`) is ~7× *slower* than the scalar
> kernel at N=8; TC needs N≥16 plus a batch-widening rewrite first, for a
> prefill-only ~3× ceiling. Full analysis:
> `docs/specs/2026-07-22-tensor-core-gemm-scope.md`. Lever 2 below is therefore
> a measured go/no-go, not a live next step.
>
> **Update (2026-07-29): the go was taken and the lever won.** MVB widened to
> 16, the kernel was rebuilt MMQ-style (Phase 2 of the spec), the tolerance
> gate was built (`make test-tc-tol`), and TC is now the **default** prefill
> path for the gated dense (Q4_K, arch) combos — measured +47–77% prefill on
> the Blackwell MIG with decode unchanged and 0/64 teacher-forced top-1 flips
> on every promoted row. The spec carries the gate table and the promotion
> record; `RUNNER_CUDA_TC=0` pins the scalar path.

## 2026-08-13 — lever 1 built and measured; the wall was somewhere else

Lever 1 below (the fused int8 dot) was built, gated and measured on the
Threadripper 9980X. Two results, and the second is the one that moved tok/s.

### The fused int8 dot: 2.4x on the kernel, ~6% end to end, NOT promoted

`vec_dot_i8` keeps the whole dot in int8 with an int32 accumulator — the
activation row is quantized once per matvec into 32-element q8 blocks carrying
their quant sum, and `_mm512_dpbusd_epi32` (AVX-512 VNNI; AVX2 `maddubs`
fallback, both gated by `test-quants-simd`) replaces the convert-to-f32-and-FMA
chain for Q4_K, Q4_0 and Q8_0. In isolation it does what the lever promised:

| format | scalar | fused int8 | speedup |
|---|---:|---:|---:|
| Q4_K | 6.9 GB/s | 17.1 GB/s | **2.48x** |
| Q4_0 | 15.0 GB/s | 24.6 GB/s | 1.64x |
| Q8_0 | 36.5 GB/s | 47.3 GB/s | 1.30x |

*(single core, 3072x3072 weight tile, `scratchpad/mvbench.c`)*

End to end it bought **~6%**, because the dot was only ~10% of decode wall
time. And it does not clear its promotion bar: activation quantization flips
near-tie tokens.

| model | teacher-forced top-1 flips | mean\|dlogit\| / range | decode |
|---|---:|---:|---:|
| Llama-3.2-3B Q4_K_M | **2/64 — fail** | 0.00057 (limit 0.005) | +6% |
| granite-4.1-8b Q4_0 | **2/64 — fail** | 0.00100 | +6% |
| SmolLM2-135M Q8_0 | 0/64 — pass | 0.00389 | -4% |

Both failing rows flip only near-ties (worst margin ~0.001 of the logit range)
and both sit far inside the deviation bound — but the bar for a tolerance-gated
fast route on this engine is 0/64, the same bar TC prefill had to clear, and it
is not widened to fit a lever. **No combo promoted.** The route ships behind
`RUNNER_CPU_I8=1`, `./test-i8-tol MODEL.gguf` is the gate, and the scalar route
is unchanged and byte-identical to the pre-branch binary at every thread count.

### The actual wall: the thread pool, at 65-138 us per matvec

Decode issues ~200 `tpool_run` calls per token. The condvar-only pool cost two
kernel round trips per worker per run:

| threads | before | after | per-token handoff (200 runs) |
|---|---:|---:|---|
| 8 | 14.0 us | 2.6 us | 2.8 ms → 0.5 ms |
| 16 | 23.1 us | 2.7 us | 4.6 ms → 0.5 ms |
| 32 | 65.4 us | 4.5 us | **13.1 ms → 0.9 ms** |
| 64 | 137.9 us | 7.1 us | **27.6 ms → 1.4 ms** |

At 32 threads that was 38% of the token, and at 64 threads 59% — which is why
runner decode got *slower* above 32 threads while llama.cpp kept scaling.
Workers now spin on the generation counter for a bounded window (`~50 us`,
`RUNNER_TPOOL_SPIN`, `0` restores the old pool) before parking, and the
publisher skips the broadcast entirely while the pool is hot. `tp_slice` is
untouched: this changes when threads wake, never which rows they compute, so
**output is byte-identical** — verified against the pre-branch binary on three
models at three thread counts.

### Where that leaves CPU decode vs llama.cpp

Best-of-thread-count, `--bench-json` pp512/tg128 vs `llama-bench` b10353 built
from the same source tree on this box, all under the benchmark lock:

| model | decode: before | +pool | +pool+int8 | llama.cpp |
|---|---:|---:|---:|---:|
| Llama-3.2-3B Q4_K_M | 27.0 (50%) | 35.7 (**66%**) | 41.2 (76%) | 54.5 |
| granite-4.1-8b Q4_0 | 15.0 (60%) | 19.7 (**78%**) | 20.4 (81%) | 25.3 |
| SmolLM2-135M Q8_0 | 134.5 (25%) | 179.7 (**34%**) | 180.4 (34%) | 528.3 |

| model | prefill: before | +pool | llama.cpp |
|---|---:|---:|---:|
| Llama-3.2-3B Q4_K_M | 113.8 (8.7%) | 153.9 (11.8%) | 1306.8 |
| granite-4.1-8b Q4_0 | 50.7 (7.9%) | 61.2 (9.6%) | 638.0 |
| SmolLM2-135M Q8_0 | 578.0 (6.5%) | 765.0 (8.7%) | 8825.2 |

Dense decode on the **default** path moved from 50-60% of llama.cpp to 66-78%,
with every output byte unchanged. The remaining decode gap is real work, not
mystery: measured aggregate DRAM read bandwidth on this box saturates at
~135 GB/s, llama.cpp's 54.5 tok/s on the 3B is ~109 GB/s of that, and runner at
35.7 is ~71 GB/s — so roughly half the remaining gap is still per-core dot
throughput (which is what the int8 route addresses, if a route that holds 0/64
can be found) and the rest is the non-matvec serial work between barriers.
CPU prefill remains the larger, untouched gap: it still dequantizes each weight
row to f32 and never uses the int8 path.

## 2026-08-13 — CUDA prefill: the depth lever found a correctness bug first

The plan for this pass was "widen the batch, deepen the MMQ tile, promote more
types". The measurements redirected all three.

### Widening the batch: already done, and measured as a no-op

`MVB` and `TC_N` are both 64 already. pp512 on Llama-3.2-3B rises steeply with
`-b` (119 → 229 → 428 → 748 tok/s at 8/16/32/64) and then stops dead: 744.4,
749.4, 749.9 at `-b` 64, 128, 256. That curve is not headroom, it is the fixed
64-column tile filling up. Going wider requires deepening `TC_N`, which needs
the shared-memory budget reworked (`sh_c` alone is 16 KB at TC_N=64 and the
three arrays already total 48 KB) and risks the accumulator spill this codebase
already documents for the widened Q3_K tile. **Not attempted; scoped, not
half-landed.**

### What the tile lever found instead: 48 of 64 columns were garbage

`TC_GEMM_32B`, the shared TC GEMM for the 32-byte-block quants (Q8_0, Q4_0),
was left at the original `TC_N=16` shape when TC_N was widened 16 -> 64 in
`6cf8c70` (**2026-08-08** — not 2026-07-29 as first recorded here; that is the
Q4_K promotion date, and the correction matters because it sets the exposure
window). Its q4_K and q6_K twins were updated; this one was not. It staged 16
activation columns, accumulated one 16-wide fragment, stored one 16x16 tile,
and the epilogue then published `sh_c` columns 16..batch-1, which nothing had
written, as logits. The dispatcher hands it 64-column tiles.

Q8_0 is promoted by default, so this was the **default CUDA prefill path**:
greedy output matched the scalar path at `-b 16` and diverged at `-b 32` and
`-b 64`, and the runner's own default is `-b 64`. Why the gate missed it — and the first answer here was wrong.
"Measured before the widening" does not hold: Q8_0 was promoted 2026-08-09, one
day *after* it. Re-run on 2026-08-13 against a rebuilt pre-fix binary, the
broken kernel **passes** the teacher-forced gate on phi3 and gemma4 q4_0
("BIT-IDENTICAL over 448 and 820 dispatches") while the same binary diverges in
free-running greedy at `-b 64`. The gate ran at `n_ctx = n_tok + 8`; at a
production context the block inherits zeroed shared memory and the corruption
surfaces — the new arm reports "first divergence at token 0, tc 0", the argmax
of an all-zero logit vector. `test_tc_tol` now carries a free-running arm at
ctx 4096 that fails against that kernel and passes against the fixed one.

Correctness is not free — the broken kernel was fast because it computed a
quarter of the work:

| model | before (wrong) | after (correct) | scalar |
|---|---:|---:|---:|
| SmolLM2-135M Q8_0 | 5206 | 4001 | 2214 |
| Phi-4-mini Q8_0 | 742 | 489 | 165 |

### Promotions the fix unlocked

With the macro correct, Q4_0 gates clean on six models across four archs, and
granite — which had no admitted combo at all — gates clean on every promoted
type. All rows 0/64 flips, 3e-5 to 8e-5 of logit range, decode unchanged.

| model | prefill before | after | |
|---|---:|---:|---:|
| granite-4.1-8b Q4_0 | 8.0 | 230.6 | **28.8x** |
| granite-4.1-3b q4_0 | 23.3 | 521.7 | 22.4x |
| Phi-4-mini q4_0 | 21.2 | 446.5 | 21.1x |
| Qwen3-1.7B q4_0 | 57.8 | 976.0 | 16.9x |
| granite-3.3-8b Q4_K_M | 108.3 | 326.6 | 3.0x |

granite-4.1-8b Q4_0 is the row worth staring at: 8.0 tok/s of prefill on a
**fully GPU-offloaded** model, slower than the same model on the CPU, purely
because neither its quant type nor its architecture was admitted to the TC
path. That is the shape of the published "Q4_0 prefill 0.6% of llama.cpp" row.

## 2026-08-13 — CUDA decode: the same story, one layer down

Phase 3 of the plan was "squeeze decode from 73-79% toward 90% with
multi-row-per-warp and vectorized loads". Measuring first killed the premise
and found a bigger prize. Implied weight bandwidth during decode (file size x
tok/s) on the same MIG slice:

| model | decode | implied |
|---|---:|---:|
| Llama-3.2-3B Q4_K_M | 129.1 | 260.8 GB/s |
| granite-3.3-8b Q4_K_M | 61.1 | 301.8 GB/s |
| Phi-4-mini Q8_0 | 80.4 | 328.2 GB/s |
| Qwen3-4B Q4_K_M | 99.9 | 249.4 GB/s |
| **granite-4.1-8b Q4_0** | **11.9** | **60.0 GB/s** |

Everything with a coalesced GEMV sits in a 250-330 GB/s band — that is the
slice's wall, and no amount of multi-row-per-warp moves a kernel already
standing on it. **Phase 3 as written had no headroom.** The outlier was not a
tuning gap: Q4_K, Q5_K, Q6_K and Q8_0 each have a lane-per-element decode GEMV
and **Q4_0 never got one**, so it fell through to `k_mv_q4_0`, where a single
lane walks an entire 32-element block through a serial 16-iteration scalar loop.

`k_gemv_q4_0` (2026-08-13) mirrors `k_gemv_q8_0`: four blocks in flight across
the warp, eight lanes each, one 2-aligned `ushort` weight load and two aligned
`float2` activation loads per lane.

| model | before | after | |
|---|---:|---:|---:|
| granite-4.1-8b Q4_0 | 11.9 | 64.3 (325 GB/s) | **5.4x** |
| Phi-4-mini q4_0 | 24.5 | 125.8 (273 GB/s) | 5.1x |
| Qwen3-1.7B q4_0 | 50.8 | 224.1 (258 GB/s) | 4.4x |

All three land in the healthy band. Identity is empirical, as it is for the
other GEMVs: kernel-verify token-identical on 5 prompts x 3 models,
`cpu_cuda_check` 5/5 CPU-vs-GPU on two of them, and granite-4.1-8b
byte-identical over a 256-token greedy generation against the pre-branch binary.

Note what this row and the Q4_0 prefill row have in common: `granite-4.1-8b
Q4_0` was, before today, **slower on a fully offloaded GPU than on the CPU**, in
both phases, for the same reason — nobody had written or admitted its kernels.
The published CUDA table's weakest rows are worth re-reading as coverage gaps
before they are read as kernel-quality gaps.

### The llama.cpp CUDA denominator — found 2026-08-13b

The blocker was a PARTIAL conda package. `conda create -n cudatk -c nvidia
cuda-toolkit=13.0` (the full metapackage, not the bare `cuda-nvcc` tried first)
carries both missing pieces — `fatbinary_section.h` and `libcublas.so` — and
builds llama.cpp b10353 with `-DGGML_CUDA=ON` without further argument. The
same-slice table is in [benchmarks.md](benchmarks.md); the short version is
that the llama.cpp side reproduces the 2026-07-29 published numbers almost
exactly (8440.6 vs 8373.6 prefill, 169.0 vs 169.0 decode on Llama-3.2-3B), so
the old denominators were sound and the movement in the ratios is runner's:
decode 73-79% -> **77-87%**, prefill 4.3-5.6% -> **6.1-9.8%**.

## The levers that remain (bigger, and deliberately not rushed)

Both are architectural changes with real correctness/token-identity risk. They
are the honest next steps, scoped here rather than half-landed.

1. **CPU: fused quantized dot products — BUILT 2026-08-13, not promoted.** The
   premise recorded here was half wrong and the measurement says so. Decode
   never dequantized to a scratch row: `vec_dot` already reads the weights in
   their on-disk quantized form, so the memory traffic this lever was supposed
   to cut was never being spent. What it did cost was the convert-to-f32 chain,
   and removing it with VNNI is worth 2.4-2.5x **on the kernel** — but only
   ~6% end to end, because the dot is ~10% of decode wall time. See the
   2026-08-13 section above for the gate table and why nothing was promoted.
   **PREFILL is where the original premise still holds:** the batched path
   (`mv_rows`, `n_batch > 1`) does dequantize each weight row to an f32 buffer,
   and it does not use the int8 kernels at all. That is the open remainder.

2. **GPU: tensor-core matmul — LANDED (2026-07-28/29), kept here for the
   history.** Phase 1 (WMMA `k_gemm_q4_K_tc`) was correct but ~7× slower than
   scalar at the then-8-token batch; the prerequisite — widening the batch
   tile to 16 — was met on 2026-07-28, and the MMQ-style rewrite was promoted
   to the default prefill path on 2026-07-29 (the update block above). See the
   TC scope spec for the full evidence chain. Only lever 1 remains open.

Widening the *existing* f32 dot to `__m512` was considered and de-prioritized:
the decode matvec is largely memory-bandwidth bound (dequantized weights), so
doubling FMA lane width buys little without also cutting the memory traffic —
which is exactly what lever 1 does. Do lever 1, not a wider f32 dot.
