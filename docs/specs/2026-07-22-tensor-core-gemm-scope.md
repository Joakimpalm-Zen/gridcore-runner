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

**Phase 1 — one kernel, portable, measured. DONE 2026-07-22 — decisive NEGATIVE
result that re-scopes the rest.** A tensor-core `k_gemm_q4_K_tc` (WMMA m32n8k16,
fp16 inputs, fp32 accumulate; `kernels.cu`, opt-in via `RUNNER_CUDA_TC`, off by
default) was implemented, verified **token-identical** to the scalar kernel on
four prompts (the dequant is factored to be bit-exact up to fp16 rounding), and
benchmarked against it on the profiling workload.

**It is ~7× SLOWER: prefill matvec 3.25 → 22.66 ms/token.** The mechanism is
fundamental, not a tuning miss:

- **The runner's batch width is N = MVB = 8.** WMMA's fp16 tiles want N ≥ 16;
  m32n8k16 half-utilizes the tensor core at N=8.
- **At N=8 the FMA is not the bottleneck** — each dequantized weight feeds only 8
  MACs. The scalar kernel dequantizes into registers and FMAs immediately.
- **Portable WMMA forces a shared-memory round-trip:** fragments can only be
  filled via `load_matrix_sync` from memory, so every dequantized fp16 weight
  must be written to shared and read back. Trading a cheap register FMA for
  TC + a shared round-trip, at a batch too narrow to amortize it, is a net loss.

**Conclusion: tensor cores are the wrong lever at N ≤ 8, which is every batched
path the runner has today.** The batched GEMM at N=8 is dequant/bandwidth-bound,
not FMA-bound; TC only accelerates the FMA. The prerequisite for *any* TC win is
widening the batch tile to N ≥ 16 — a large, invasive change (MVB is baked into
every `_b` kernel and every activation buffer), and even fully realized the
ceiling is **prefill-only ~3×**, because single-stream decode stays
memory-bandwidth bound and TC-immune on this MIG slice.

**Phase 2 (re-scoped) — batch-tile widening is the real prerequisite, not WGMMA.**
Before any WGMMA/arch work, MVB must widen to ≥16 and the scalar batched kernels
must stay correct/fast at that width. Only *then* does a TC kernel have enough N
to amortize the shared round-trip. This is where the effort and risk actually
are; WGMMA (below) is downstream of it. Given the prefill-only ~3× ceiling vs the
size of the change, this needs a deliberate go/no-go — the CPU fix already landed
6.1× across the board at a fraction of the risk.

**Phase 3 — Blackwell-native WGMMA + coverage, only if Phase 2 pays.** A
`compute_120` WGMMA variant (5th-gen tensor core, fp8; register-operand `mma` that
avoids the shared round-trip), multi-arch PTX + runtime SM dispatch (preserving the
`compute_75` baseline), extended to q8_0/q6_K/q5_K and the concurrent-decode path.
Only worth building once a widened batch has shown TC beats scalar.

### Honest recommendation

The profiling + Phase 1 empirics say the GPU is already near-optimal for
single-stream inference on this MIG slice (decode is at the bandwidth ceiling), and
the tensor-core lever pays only on prefill/concurrent serving *after* a large
batch-widening rewrite, for a ~3× prefill-only ceiling. That is a real option for a
serving-throughput push, but it is a multi-day, higher-risk change with a bounded
payoff — a deliberate decision, not an automatic next step. The correct kernel and
this measurement are committed so the decision is made on evidence.

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

## Reproducing the Phase 1 result

`make ptx && make` (with nvcc), then compare on a long prompt:

    RUNNER_CUDA_PROFILE=1              ./runner -m <q4_K.gguf> -p "<long>" -n 1 --gpu auto   # scalar
    RUNNER_CUDA_TC=1 RUNNER_CUDA_PROFILE=1 ./runner -m <q4_K.gguf> -p "<long>" -n 1 --gpu auto   # TC

Correctness: `RUNNER_CUDA_TC=1` produces token-identical greedy output to the
scalar path. Speed: the TC `matvec` line is ~7× the scalar one. The kernel is
kept, off by default, as the correct foundation for the widened-batch work Phase 2
would need — not as a shipped win.
