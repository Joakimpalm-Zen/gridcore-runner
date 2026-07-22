# GPU optimization: tensor-core GEMM — scope & profiling

Goal: close the GPU speed gap with llama.cpp/vLLM by using Blackwell's tensor
cores. This doc records the profiling that says *where* to spend the effort, and
scopes the work into phases with the correctness/portability constraints.

Profiled on the dev box: Ryzen Threadripper 9980X + NVIDIA RTX PRO 6000 Blackwell
(MIG 1g.24gb slice), Llama-3.2-3B-Instruct-Q4_K_M, via the built-in
`RUNNER_CUDA_PROFILE=1` (and `RUNNER_CUDA_GRAPH_OFF=1` to expose the decode
phases the CUDA graph otherwise hides). No `nsys`/`ncu` on the box; the built-in
CUDA-event profiler is the instrument.

## Where GPU time actually goes

Per-phase GPU time (CUDA events, authoritative):

| phase | decode (batch 1) | prefill (batch ~8, 541-tok prompt) |
|---|---:|---:|
| **matvec (weight matmuls)** | **80.2%** | **84.1%** |
| logits-mv (final vocab proj) | 7.0% | 0.0% |
| attention | 3.8% | 12.6% |
| norms (rms+qk) | 4.7% | 1.7% |
| elementwise | 3.0% | 1.1% |
| rope | 1.4% | 0.5% |

**Weight matmuls are 80–92% of all GPU time.** Everything else combined is under
15%. Attention is the only other phase that grows (quadratic in sequence length —
12.6% at 541 tokens, worth a second look only for very long contexts). Norms,
rope, elementwise are noise. **matvec is the target; nothing else moves the needle.**

## The decisive distinction: bandwidth-bound vs compute-bound

The matvec kernels (`k_mv_*` decode, `k_gemm_*`/`k_mv_*_b` prefill) are **scalar-FMA**
— they dequantize Q4_K inline and accumulate into `float` registers (confirmed in
`k_gemm_q4_K`, `kernels.cu:530`). No tensor cores anywhere (`grep` for `wmma`/`mma`
= 0). Whether tensor cores help depends entirely on the regime:

- **Single-stream decode (batch 1) is MEMORY-BANDWIDTH bound.** It reads all ~1.8 GB
  of Q4 weights from VRAM per token; measured 9.06 ms/token ≈ **~200 GB/s**, at the
  MIG 1g slice's bandwidth ceiling. Tensor cores do not read memory faster — they
  will **not** speed up batch-1 decode. The only levers there are less weight
  traffic (already Q4) or a bigger MIG slice / full GPU.

- **Batched matmul (prefill, concurrent slots, speculative drafts) is COMPUTE
  bound on the scalar-FMA kernels.** At batch ~8, prefill matvec is 3.25 ms/token
  while the bandwidth floor (weights read once per 8-token tile) is ~1.1 ms/token —
  so the kernel runs **~3× above the memory floor**, spending that gap in scalar
  FMA. That gap is exactly what tensor-core MMA collapses.

**Conclusion: tensor cores are a ~3× lever on the compute-bound batched matmul
(prefill + batched/concurrent decode), and roughly a no-op on single-stream
decode.** Set expectations accordingly — this is a serving-throughput and
prompt-latency win, not a batch-1 tok/s win. (The runner is fleet/serving-oriented,
so batched decode is a first-class path, and long agent prompts make prefill matter.)

## Target and plan

Replace the scalar-FMA accumulation in the batched matmul kernels with tensor-core
MMA over dequantized fp16/bf16 (or int8) tiles — the fused "dequant Q4 → TC-GEMM"
pattern vLLM/llama.cpp use.

**Phase 0 — this doc.** Profiling + scope. Done.

**Phase 1 — one kernel, portable, measured.** A tensor-core `k_gemm_q4_K` using the
**WMMA** API (works on sm_70+, i.e. every GPU ≥ Turing including Blackvell via the
existing `compute_75` PTX — no portability change yet). Dequantize a Q4_K weight
tile + the x tile into fp16 in shared memory, `wmma::mma_sync` into fp32
accumulators. Bench vs the scalar kernel on prefill; target approaching the ~1.1
ms/token floor. If WMMA on `compute_75` PTX JITs to Blackwell tensor cores and
wins, ship it behind the existing kernel-selection with a tolerance gate.

**Phase 2 — Blackwell-native, multi-arch PTX.** Add a `compute_120` (WGMMA / 5th-gen
tensor core, fp8) kernel variant, embed BOTH PTX blobs, and select at runtime by
detected SM — preserving the `compute_75` baseline for older nodes (the "one binary,
any node" promise). This is where the arch string finally matters (Phase-0 measured
a bare `compute_75→120` PTX bump as within noise precisely because the *kernels* used
no TC features; WGMMA is the feature that makes a newer arch pay).

**Phase 3 — coverage + integration.** Extend TC-GEMM to the other common quant types
(q8_0, q6_K, q5_K) and wire it into the batched-decode / speculative path so
concurrent serving gets the compute win, not just prefill.

## Correctness & portability constraints (non-negotiable)

- **Not bit-identical.** TC accumulation (fp16/bf16 inputs, fp32 or fp16 accumulate)
  will not match the scalar-fp32 kernels bit-for-bit. The token-identity gate must be
  a **tolerance gate**, modeled on the existing q8-KV tolerance test
  (`tests/…test_kv_tol`): teacher-force N positions, require top-1 token match and a
  bounded logit deviation. A TC path that flips greedy tokens is rejected.
- **Portability.** WMMA (Phase 1) keeps `compute_75`. WGMMA (Phase 2) requires
  multi-arch PTX + runtime dispatch; never bump the single baseline to a
  Blackwell-only arch (it would fail to JIT on Ampere/Ada/Turing nodes).
- **MIG reality.** Bench numbers here are a 1/7 slice; absolute tok/s will differ on
  the full GPU, but the *ratios* (matvec share, compute-vs-bandwidth) hold.
- **Fallback stays.** The scalar kernels remain the default/fallback; TC is a
  measured, gated opt-in per (quant type, arch).

## First measurable step

Implement Phase 1 (`k_gemm_q4_K` WMMA variant) + a TC tolerance gate, and bench
prefill matvec ms/token against the scalar 3.25 ms/token → target ~1.1–1.5.
