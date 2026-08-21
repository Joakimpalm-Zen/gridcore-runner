# SSM hybrid cross-ISA replay + the Clang FP-regime re-measure — 2026-08-21

Two measurements triggered by the `-Werror=unknown-attributes` finding
(commit `2727b00`): Clang silently ignored `__attribute__((optimize
("no-fast-math")))`, so the Mamba-2 "precise" core had been compiling under
`-ffast-math` on macOS. These runs quantify what that regime change actually
did, and add the first cross-ISA replay evidence for the Mamba-2 family.

Model: **granite-4.0-h-micro Q4_K_M** (dense granitehybrid, 36 recurrent +
4 attention layers), upstream `ibm-granite/granite-4.0-h-micro-GGUF`, sha256
`bcc78b9b25450101d1ad90d4b9a264e1bac892f534dfb76066f4eec792fdf023`, byte-copied
to both hosts. Protocol: CPU serve, 3 prompts x 200 greedy tokens, chosen +
top-5 logprobs at the server's %.6f serialization (the D0 probe protocol).
Runner `d21564a` both sides. (Running this model at all required the dense
granitehybrid loader fix in `d21564a` — the recurrent-layer MLP was only
bound for MoE h-models before it.)

## 1. Pre-fix vs post-fix on the SAME machine (M1, clang/NEON)

The isolated effect of restoring precise FP semantics to the Mamba-2 core
(pre-fix binary = `0c82c3c` + the loader fix only):

| prompt | token divergence | mean \|dlp\| | max \|dlp\| |
|---|---|---|---|
| "Once upon a time…" | none (200/200) | 0.000029 | 0.000334 |
| "The capital of France is" | none | 0.000000 | 0.000000 |
| "def fibonacci(n):" | none | 0.000002 | 0.000030 |

The regime change is real but small: no token flips in 600 positions, logit
movement two orders below the cross-ISA noise ceiling (0.032). Tiny
granitehybrid/nemotron_h fixtures show the same shape (token-identical,
max |dlp| 1e-6). No token-level result measured on macOS before the fix is
invalidated; the certified SSM greedy-reference numbers were measured on a
GCC host, where the attribute always worked.

## 2. Cross-ISA replay, post-fix: M1 (clang, NEON) vs x86 (gcc, AVX2)

| prompt | token divergence | mean \|dlp\| | max \|dlp\| |
|---|---|---|---|
| "Once upon a time…" | none (200/200) | 0.000039 | 0.000500 |
| "The capital of France is" | none | 0.000004 | 0.000008 |
| "def fibonacci(n):" | none | 0.000008 | 0.000215 |

First cross-ISA measurement of the Mamba-2 path: 600/600 greedy tokens
identical across compiler, ISA and SIMD width, per-run logprob means inside
the band measured for transformer architectures (per-run mean 0.000046–
0.000222 on the 2026-08-20 matrix — the hybrid runs sit at or below its
floor). The recurrent scan does not add cross-ISA divergence beyond what
dense transformers already show.

Raw probes: `probe-m1-neon-clang.json`, `probe-x86-avx2-gcc.json`.
