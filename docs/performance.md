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
