# Mamba-2 hybrids on CUDA: two silent-fallback bugs, and the identity cert — 2026-08-21

An external review flagged that the decode microbatch (`fwd_batch`) has no
recurrent path and that nothing in `batch_eligible` declines hybrid models.
Chasing that finding empirically surfaced something larger: **the Mamba-2
CUDA path (`k_mamba2_*`) had never actually run.** Two validation bugs in the
shared-weights build pushed every hybrid model to the CPU, silently:

1. The recurrent weight-table check demanded `ssm_norm` — a field only the
   qwen35 branch fills. The Mamba-2 branch fills `ssm_gnorm`/`ssm_D`/
   `ssm_conv_b` instead, so every granitehybrid and nemotron_h layer failed
   the check.
2. The check demanded `ffn_norm` on every layer. nemotron_h SSM and attention
   blocks carry no FFN at all (three-way block design), so after fix 1 every
   nemotron_h model still failed.

Both fallbacks printed nothing. Measured before the fix on an RTX PRO 6000
Blackwell: granite-4.0-h-micro Q4_K_M decoded at 11.9 tok/s, Nemotron-Nano-9B
Q4_K_M at 1.8 tok/s — CPU speeds, behind a "gpu-split ... full=1" banner.
After: h-micro 114.5 tok/s; Nano's CUDA backend initializes (6.5 GB in VRAM).

The original microbatch finding was real but latent — unreachable only
*because* hybrids never had a GPU backend. With the fallback fixed it became
one refactor from live, so `batch_eligible` now declines the recurrent
families (and NoPE-stepped models) explicitly, and `make test` gates it with
a rope-enabled dense hybrid fixture at width 2.

## Identity cert (the gate for enabling kernels that had never executed)

Same protocol as the cross-ISA report: 3 prompts x 200 greedy tokens, chosen
+ top-5 logprobs at %.6f, same GGUF, `--gpu off` vs `--gpu auto`, same box.

| model | tokens | mean \|dlp\| (per run) | max \|dlp\| |
|---|---|---|---|
| granite-4.0-h-micro Q4_K_M (granitehybrid, dense) | 600/600 identical | 0.000003–0.000024 | 0.000422 |
| Nemotron-Nano-9B-v2 Q4_K_M (nemotron_h) | 600/600 identical | 0.000004–0.000023 | 0.000572 |

Token-identical across the full matrix; logprob deltas sit in the same
envelope as the cross-ISA CPU runs. NOT bit-exact — the device scan
reassociates its reductions, as every CUDA path here does; the claim is
token-identity with the measured logprob envelope, per the cpu_cuda
convention. MoE granitehybrid (h-small) remains CPU-only by the router/
shared-expert guards, which print their reason.

Raw probes: `p_hyb_{cpu,gpu}.json`, `p_nano_{cpu,gpu}.json`. Runner at the
commit introducing this report; box gate (`make test`, CUDA build) green.
