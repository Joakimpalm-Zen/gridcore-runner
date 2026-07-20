# CUDA backend for runner — design

> **Status: historical (superseded in part).** This records the design as of
> 2026-07-11 and is kept for the reasoning behind it. Two things have since
> changed and the README is authoritative where they disagree:
>
> - **Partial/layer-split offload shipped.** Listed below as a non-goal for
>   this milestone, it was implemented afterwards: `gpu_forward_batch` runs
>   layers `[0, gpu_layers)` on the device and hands the boundary activation
>   back to the CPU loop, with `--reserve`/`--reserve-vram` budgeting the
>   split. Models larger than VRAM now run hybrid rather than falling back to
>   CPU wholesale.
> - **Kernel design moved on.** The per-token/8-token-tile matvec shape
>   described here was replaced on 2026-07-19/20 by coalesced GEMV kernels for
>   decode and shared-memory tiled GEMM kernels for prefill (Q8_0, Q4_K, Q5_K,
>   Q6_K), plus flash-decoding attention. The measured effect and the
>   diagnosis that motivated it are in the commit history; the launch-overhead
>   reasoning in this document was measured at 0.2-1.4% and is not the
>   bottleneck it was assumed to be.

Owner directive: whole-model GPU offload first; long-term goal is full use of
gpu/cpu/ram/vram. This spec covers the first milestone: a CUDA port of the Metal
backend behind the existing 4-function GPU interface, no interface changes.

## Goals / non-goals

- **Goal:** models whose weights + KV + activations fit free VRAM run the full
  per-token forward pass on the GPU (RTX 3070-class → 7B/8B quantized models).
- **Goal:** zero new build/runtime dependencies. The shipped binary stays a
  single static exe; a machine without an NVIDIA driver just gets CPU.
- **Non-goal (this milestone):** partial/layer-split offload for models larger
  than VRAM. That is the planned follow-up and will extend the interface.

## Approach

`src/cuda.c` implements `gpu_available/gpu_init/gpu_forward/gpu_free` using the
**CUDA driver API loaded dynamically** (`nvcuda.dll` on Windows, `libcuda.so.1`
on Linux) — no CUDA toolkit needed at build or run time. Kernels are written in
`src/kernels.cu` (1:1 port of `kernels.metal`), compiled **at development time**
to PTX (`nvcc -ptx -arch=compute_75`) and committed as the generated header
`src/kernels_ptx.h` (mirrors the `embed-metal.py` convention). The driver JIT
compiles PTX for whatever GPU is present (sm_75+).

Key differences from Metal, forced by discrete-GPU memory:

- **Weights:** copied to VRAM once at `gpu_init` (Metal wraps the mmap
  zero-copy; PCIe GPUs cannot). Fit check against `cuMemGetInfo` free VRAM with
  a 256 MB headroom; failure → return false → CPU path, message to stderr.
- **KV cache coherence:** host `m->kcache/vcache` stays authoritative for CPU
  writes (batched prompt processing). The backend tracks a device watermark;
  `gpu_forward(pos)` uploads host rows `[wm..pos)` for all layers first, and
  after the forward copies the GPU-written row at `pos` back to host
  (~4 KB/token). Either side can therefore run any token; interleaving is safe.
- **Logits:** copied device→host each token (n_vocab × 4 B ≈ 0.6 MB, ≪1 ms).

CUDA driver API note: allocation/copy/context entry points must be resolved by
their `_v2` names (`cuMemAlloc_v2`, `cuMemcpyHtoD_v2`, …); the unsuffixed
symbols are the legacy 32-bit-era ABI.

## Kernel port map (kernels.metal → kernels.cu)

| Metal | CUDA |
|---|---|
| threadgroup / threadgroup_barrier | `__shared__` / `__syncthreads()` |
| simdgroup (32 lanes), `simd_sum` | warp, `__shfl_down_sync` reduction |
| `k_mv_*`: 128 threads = 4 simdgroups, 1 row/simdgroup | identical: 4 warps/block, row = `blockIdx.x*4 + warp` |
| `half` | `__half` (`cuda_fp16.h`, device-only) |
| dispatchThreads (grid-exact) | 1D/2D grid with bounds guard (unchanged guards) |

Same kernel set: rmsnorm, qknorm, rope, store_kv, attn, silu_mul, add, and
mv_{f32,f16,q8_0,q4_0,q4_1,q5_0,q5_1,q4_K,q5_K,q6_K}. Bit layouts are already
mirrored from quants.c and stay byte-identical.

## Build integration

Makefile: Windows and Linux compile `src/cuda.c` (+ committed `kernels_ptx.h`)
instead of `src/gpu_none.c`; macOS keeps Metal. `make ptx` regenerates the PTX
header (requires nvcc, dev machines only). CI needs no CUDA: `cuda.c` is plain
C; on GPU-less runners the dynamic load fails and `gpu_available` returns false,
exactly like `gpu_none.c`.

## Verification

1. Synthetic model (`make-test-model.py`): CPU vs GPU logits/output at temp 0.
2. Qwen2.5-1.5B / Qwen3-4B / Qwen2.5-7B / Llama-3.1-8B-Q5: identical greedy
   output CPU vs GPU on a fixed prompt; tok/s recorded both ways.
3. `--gpu off` still forces CPU; over-VRAM model (14B) falls back to CPU with a
   clear stderr message.
4. runner CI green on all three platforms; gridcore-interpreter tracer bullet +
   `--caps` reports the GPU.
