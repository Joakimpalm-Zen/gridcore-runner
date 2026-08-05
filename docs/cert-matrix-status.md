# Cert-matrix status — GPT-OSS x Gemma 4 derivative ecosystems

Live status table, one row per roster item. Updated after EVERY verdict.
Full evidence in `docs/cert-matrix-2026-08-05.md`. Environment: runner
`0.1.8-alpha`, branch `cert-matrix`, llama.cpp reference `b10280 (61881b1f7)`
unless a row notes a split reference.

Verdicts: CERTIFIED / CERTIFIED-WITH-CAVEAT / FAILED / REFUSED / NOT FOUND / SKIPPED.

## Tier 1 — canonical + QAT

| # | artifact | resolved repo/file | verdict | note |
|---|---|---|---|---|
| 1 | ggml-org gpt-oss-20b MXFP4 | `ggml-org/gpt-oss-20b-GGUF/gpt-oss-20b-MXFP4.gguf` (sha `27cd6c43...`) | **FAILED** | tokenizer 222/721 diverge, cpu_cuda not byte-identical on this GPU, chat smoke runs away non-coherent; KLD (raw-completions, chat endpoint unusable for gpt-oss cross-engine) 83% top1/0.128 KLD misses 97%/0.05 bar but matches an already-diagnosed MXFP4 vec_dot_type gap |
| 2 | Bartowski gpt-oss-20b Q6_K_L | `bartowski/openai_gpt-oss-20b-GGUF/openai_gpt-oss-20b-Q6_K_L.gguf` (sha `e729b05f...`) | **FAILED** | mixed-tensor trap confirmed: experts stay MXFP4_MOE (72/72 unchanged), only 24 non-expert tensors move Q8_0->Q6_K; tokenizer/cpu_cuda/chat-smoke fail identically to item 1 (same root causes); KLD 84%/0.116, consistent with item 1 |
| 3 | Unsloth gpt-oss-20b Q4_K_M-class | pending | pending | |
| 4 | Gemma-4-26B-A4B-it QAT Q4_0 | pending | pending | |
| 5 | Bartowski gemma-4-26B-A4B-it Q4_K_M | pending | pending | |
| 6 | gemma-4-12B-it QAT Q4_0 | pending | pending | |
| 7 | gemma-4-E2B-it QAT Q4_0 | pending | pending | |
| 8 | gemma-4-31B-it QAT Q4_0 | pending | pending | |

## Tier 2 — mutations

| # | artifact | resolved repo/file | verdict | note |
|---|---|---|---|---|
| 9 | GPT-OSS Nano 9B (~12 experts) | pending | pending | |
| 10 | GPT-OSS 120B REAP 58B | pending | pending | |
| 11 | gpt-oss-safeguard-20b | pending | pending | |
| 12 | GPT-OSS 20B coder fine-tune | pending | pending | |
| 13 | abliterated GPT-OSS 20B MXFP4 | pending | pending | |
| 14 | Gemma 4 12B Coder fine-tune | pending | pending | |
| 15 | HauhauCS gemma-4-26B-A4B QAT uncensored | pending | pending | |
| 16 | BrainStorm GPT-OSS 36B | pending | pending | |

## Tier 3 — speculative decoding (MTP)

| # | artifact | resolved repo/file | verdict | note |
|---|---|---|---|---|
| 17 | Gemma 4 26B-A4B/12B + MTP drafter | pending | pending | |

## Tier 4 — big iron

| # | artifact | resolved repo/file | verdict | note |
|---|---|---|---|---|
| 18 | gpt-oss-120b MXFP4 | pending | pending | |
| 19 | gpt-oss-safeguard-120b | pending | pending | |
| 20 | 220A20B expanded-expert FrankenMoE | pending | pending | |

## Note-only (no download)

| format | why out of scope |
|---|---|
| NVFP4 | pending |
| FP8 | pending |
| BnB | pending |
| MLX | pending |

## 16 GB envelope sweep

Table filled in after Tiers 1-2. See `docs/cert-matrix-2026-08-05.md` for the
prune-lever numbers and the ballast-capped live verification.

| candidate | fits? | lever | top-1 vs parent | mean KLD | tok/s (cap) | verdict |
|---|---|---|---|---|---|---|
| pending | | | | | | |
