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

The embedded PTX targets `compute_75` (Turing). The GPU is Blackwell
(`sm_120`). Regenerating the PTX at `compute_120` and re-benchmarking gave
**2.12s vs 2.23s — within noise.** The driver JITs `compute_75` PTX to Blackwell
SASS at load either way, and Runner's hand-written matvec kernels use no
features (tensor cores, async copy) that a newer *virtual* arch would unlock.
Bumping would only cost portability (`compute_75` JITs to any ≥Turing GPU).
**Kept at `compute_75`.**

> **Update (later 2026-07-22):** the tensor-core lever was then built and
> measured — and **lost at the runner's batch width**. A correct WMMA Q4_K GEMM
> (token-identical, opt-in `RUNNER_CUDA_TC`) is ~7× *slower* than the scalar
> kernel at N=8; TC needs N≥16 plus a batch-widening rewrite first, for a
> prefill-only ~3× ceiling. Full analysis:
> `docs/specs/2026-07-22-tensor-core-gemm-scope.md`. Lever 2 below is therefore
> a measured go/no-go, not a live next step.

## The levers that remain (bigger, and deliberately not rushed)

Both are architectural changes with real correctness/token-identity risk. They
are the honest next steps, scoped here rather than half-landed.

1. **CPU: fused quantized dot products (the real llama.cpp gap).** Runner
   dequantizes each Q4_K weight row to an f32 buffer and then does an f32 FMA
   dot. llama.cpp keeps weights quantized and dots int8·int8 directly, reading
   ~8x less memory per weight — decisive when the matvec is memory-bandwidth
   bound. Fusing dequant into the dot (operate on Q4 blocks in-register, VNNI
   `_mm512_dpbusd_epi32` for the int8 MAC) attacks that traffic directly and
   would use the AVX-512 + VNNI silicon this box has and the current kernels
   ignore. Large rewrite of the matvec path; must stay token-identical.

2. **GPU: tensor-core matmul — measured, currently a no-go at N=8.** Phase 1
   (WMMA `k_gemm_q4_K_tc`) was correct but ~7× slower than scalar at the
   runner's 8-token batch; the prerequisite is widening the batch tile to ≥16,
   with a prefill-only ~3× ceiling. See the TC scope spec for the evidence and
   the go/no-go framing.

Widening the *existing* f32 dot to `__m512` was considered and de-prioritized:
the decode matvec is largely memory-bandwidth bound (dequantized weights), so
doubling FMA lane width buys little without also cutting the memory traffic —
which is exactly what lever 1 does. Do lever 1, not a wider f32 dot.
