# Negative result: Runner-built selective quants of Muse Glimmer 30B

*2026-08-20. Status: six candidates built, gated and rejected. Not published.
The finding is a Runner gap, not a Muse one.*

## What was tried

Six `--type-plan` candidates for `muse-glimmer` (dense, 52 layers, `d_model`
6656, `d_ff` 19968, vocab 202048), aimed at the 14–16 GB band where Meta ships
a 16.76 GB Q4_K_M — the artifact this project would have to beat on
quality-per-byte or on size at equal quality.

Unlike Nemotron, the geometry cooperates: every quantizable Muse row width
divides by 256 (6656 = 256·26, 19968 = 256·78, 4096, 256), so Q3_K is
genuinely available here.

## The parent

`--type-plan` candidates carry a precision claim, so they need a real
high-precision parent, and no BF16/F16 Muse GGUF is published. The 59.5 GB
safetensors (`meta-models/Muse-Glimmer-30B`, revision `a4e59da`) were
downloaded and converted text-only with llama.cpp `521a64cd`
(`convert_hf_to_gguf.py --outtype bf16`), giving a 731-tensor,
55,725,514,176-byte BF16 GGUF, sha256 `d22e290d…`. Starting from Meta's
4-bit file instead would have been quant→quant and could not carry the claim.

## What was measured

`scripts/kld-compare-raw.py`, 400 teacher-forced positions, 0 failed, zero
point exact (0.000 / 100.00% / 100.00%), every arm against that BF16 parent,
same box and same runner build.

| arm | bytes | vs Meta | plain top-1 | margin-qual. | mean KLD |
|---|---:|---:|---:|---:|---:|
| **Meta `KQuant-17GB-Q4_K_M`** | 16,756,683,904 | — | 90.25% | **99.25%** | **0.030** |
| FFN Q3_K, everything else Q8_0 | 16,492,547,520 | ‑1.6% | 80.25% | 92.00% | 0.123 |
| + embed/output Q4_0 | 15,147,716,032 | ‑9.6% | 79.25% | 92.00% | 0.125 |
| Q3_K except `attn_k/v` + `output` | 13,801,180,608 | ‑17.6% | 77.00% | 89.75% | 0.170 |
| gate/up Q3_K, `ffn_down` Q4_0, embed Q4_0 | 16,065,605,056 | ‑4.1% | 83.00% | 94.50% | 0.090 |
| no Q3_K at all: Q4_0 but `attn_v` Q8_0 | 15,730,568,640 | ‑6.1% | 89.50% | **97.75%** | 0.056 |

Bar: margin-qualified top-1 ≥ 97% **and** mean KLD ≤ 0.05.

## Why they are rejected

**Meta's own 4-bit k-quant passes the bar** (99.25% / 0.030) against the BF16
it came from. That sets the standard the stage had to beat, and nothing did:
every Q3_K-bearing plan is three to six times its divergence, and the one plan
that gets close is a plain Q4_0.

Three sub-results are worth keeping:

1. **Q3_K is the damage.** Spending Q8_0 on every attention tensor and both
   embeddings around it still measures 0.123 — four times Meta's number at
   1.6% less size.
2. **The embeddings are nearly free.** Dropping `token_embd` and `output` from
   Q8_0 to Q4_0 saved 1.35 GB (‑8.2%) and moved KLD 0.123 → 0.125. On a 202k
   vocabulary that is the cheapest 1.3 GB on the table.
3. **`ffn_down` is the sensitive FFN tensor** — as Meta's own file implies by
   protecting it with Q6_K on 26 of 52 layers. Moving only `ffn_down` off Q3_K
   to Q4_0 improved 0.123 → 0.090 for +0.9 GB, with `ffn_gate`/`ffn_up` left
   at Q3_K.

## The actual finding: `--quantize` cannot write k-quants

The closest candidate removes Q3_K entirely and lands at **97.75%
margin-qualified — over the bar — with 0.056 mean KLD, under it**, at 6.1%
less size than Meta's file. Closing that last ~11% of divergence requires
precision spent where Runner cannot spend it: `--type-plan` writes only
`q8_0`, `q4_0`, `q3_k` and `f16`, and every remaining route costs more bytes
than the comparator has (`ffn_down` at Q8_0 alone adds 3.45 GB, landing at
19.2 GB against Meta's 16.76 GB).

Runner **reads** Q4_K/Q5_K/Q6_K already. Until it can **write** them, a
Runner-built selective quant cannot compete with a well-made k-quant at the
4-bit size point on a dense model. That is the follow-up this stage produces,
and it is engine work rather than artifact work.

## What was not measured

Downstream task benchmarks; native atem tool fidelity on the candidates (the
tool-fidelity edition selects among survivors, and there are none); Metal or
CUDA paths (all arms CPU); any context length beyond the corpus. The vision
encoder was never in scope — `mmproj` is a separate upstream file and stays out.
