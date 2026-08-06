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
| 3 | Unsloth gpt-oss-20b Q4_K_M-class | `unsloth/gpt-oss-20b-GGUF/gpt-oss-20b-Q4_K_M.gguf` (sha `c2753664...`) | **FAILED** | mixed-tensor trap confirmed again (experts untouched MXFP4); tokenizer/cpu_cuda fail same as items 1/2; chat smoke fails DIFFERENTLY — stops cleanly (Unsloth patched the chat template) but still leaks analysis-channel meta-commentary instead of a direct answer; KLD 78%/0.137 |
| 4 | Gemma-4-26B-A4B-it QAT Q4_0 | `google/gemma-4-26B-A4B-it-qat-q4_0-gguf/gemma-4-26B_q4_0-it.gguf` (sha `3eca3b8f...`) | **CERTIFIED-WITH-CAVEAT** | tokenizer 0/721 clean, cpu_cuda byte-identical, chat smoke clean "4"/stop — all PASS; only KLD misses 97%/0.05 (80.5%/0.126), matching an already-documented numerically-chaotic floor for this exact file (top-8-of-128 routing ties) |
| 5 | Bartowski gemma-4-26B-A4B-it Q4_K_M | `bartowski/google_gemma-4-26B-A4B-it-GGUF/google_gemma-4-26B-A4B-it-Q4_K_M.gguf` (sha `a07f7222...`) | **CERTIFIED-WITH-CAVEAT** | experts genuinely requantized here (non-uniform per-layer Q8_0/Q5_0/Q4_K), unlike gpt-oss rows; tokenizer/cpu_cuda/chat all PASS; **headline: QAT (item 4) vs this PTQ build disagree with EACH OTHER 62.5% of the time (head-to-head KLD 1.95), more than either disagrees with llama.cpp — QAT (80.5% top1) is measurably closer to llama.cpp than PTQ (65.5% top1)** |
| 6 | gemma-4-12B-it QAT Q4_0 | `google/gemma-4-12B-it-qat-q4_0-gguf/gemma-4-12b-it-qat-q4_0.gguf` (sha `93567e57...`) | **CERTIFIED-WITH-CAVEAT** | dense (not MoE) — greedy identity 4/6 exact vs llama.cpp, all 4 short prompts pass; the 2 long-run misses both degenerate into a repeat loop on both engines (small-model artifact, not content disagreement); tokenizer 0/721, cpu_cuda byte-identical, chat clean |
| 7 | gemma-4-E2B-it QAT Q4_0 | `google/gemma-4-E2B-it-qat-q4_0-gguf/gemma-4-E2B_q4_0-it.gguf` (sha `fa401b55...`) | **REFUSED** | clean error "missing model hyperparameters for arch 'gemma4'" — root cause identified (read-only): file publishes real per-layer FFN width variation (6144/12288) via a GGUF ARRAY-typed `feed_forward_length` KV; `gguf_get_u32` has no array branch and silently defaults to 0, tripping the hyperparameter check. Tokenizer independently clean (0/721). **Scope corrected 2026-08-06 (M1):** this is NOT QAT-specific — the non-QAT `E2B-it-Q4_K_M` conversion refuses with the identical error, because per-layer FFN width variation is a property of the E2B *architecture*, not of any one export. E4B is unaffected only because its FFN width is uniform (scalar 10240), which is why the E4B conversions load. Every E2B GGUF is therefore blocked until the loader grows an array branch |
| 8 | gemma-4-31B-it QAT Q4_0 | `google/gemma-4-31B-it-qat-q4_0-gguf/gemma-4-31B_q4_0-it.gguf` (sha `179cfb99...`) | **CERTIFIED-WITH-CAVEAT** | best identity result of session: 5/6 exact greedy vs llama.cpp incl. BOTH 256-tok long runs; tokenizer/cpu_cuda/chat all PASS; real caveat is perf — GPU is 8x SLOWER than CPU on this box (17.7GB weights nearly fill the 24GB MIG slice, VRAM pressure, not a correctness issue) |

## Tier 2 — mutations

| # | artifact | resolved repo/file | verdict | note |
|---|---|---|---|---|
| 9 | GPT-OSS Nano 9B (~12 experts) | `squ11z1/gpt-oss-nano/gpt-oss-9b-q4_k_m.gguf` (sha `794da0a9...`) | **FAILED** | admission PASS on 12-expert (not 32) file — validates runner's per-layer expert-count handling on a real 3rd-party prune; fully requantized off MXFP4 (Q8_0/Q5_0 mixed); cpu_cuda PASSES (no MXFP4 left to disagree about); KLD 88%/0.040 (best KLD of any gpt-oss row, clears the 0.05 bar, misses top-1); chat smoke still fails (same Harmony issue) |
| 10 | GPT-OSS 120B REAP 58B | `12bitmisfit/OpenAI_GPT-OSS-120B_Pruned_REAP_58B-GGUF` Q5_0, 9 shards (39GB) | **REFUSED** | runner has NO support for GGUF split/multi-part files at all — reads only shard 1's tensor table, reports later-layer tensors "missing" rather than discovering sibling shards; real capability gap (multi-part is the standard distribution format past ~30-40GB), not specific to REAP |
| 11 | gpt-oss-safeguard-20b | `unsloth/gpt-oss-safeguard-20b-GGUF/gpt-oss-safeguard-20b-Q4_K_M.gguf` (sha `7c70a6d0...`) | **FAILED** | mixed-tensor trap + tokenizer/cpu_cuda/chat all fail same as base model; Harmony-tag-leak check specifically PASSES (no raw channel tokens in content) but general coherence still fails; KLD 73.75%/0.136 |
| 12 | GPT-OSS 20B coder fine-tune | `DavidAU/Openai_gpt-oss-20b-CODER-NEO-CODE-DI-MATRIX-GGUF/OpenAI-20B-NEO-CODE-DIMAT-MXFP4_MOE2.gguf` (sha `09888bd7...`) | **FAILED** | mixed-tensor claim confirmed accurate; tokenizer/cpu_cuda fail same as family; KLD 84%/0.129; **plain chat fails (runaway) but tool-calling is clean and correct** (valid function name+args, finish_reason=tool_calls) — a genuinely isolated positive result |
| 13 | abliterated GPT-OSS 20B MXFP4 | `noctrex/Huihui-gpt-oss-20b-abliterated-v2-MXFP4_MOE-GGUF` (sha `9d060cce...`) | **FAILED** | tokenizer/KLD match family pattern (82.75%/0.114); cpu_cuda is PROMPT-DEPENDENT (1 of 2 tested prompts byte-identical, confirming near-tie sensitivity); chat smoke hallucinates a FOREIGN `[/INST]` template marker (not gpt-oss's own Harmony) — a distinct, more specific failure than the rest of the family |
| 14 | Gemma 4 12B Coder fine-tune | `yuxinlu1/gemma-4-12B-coder-fable5-composer2.5-v1-GGUF/gemma4-coding-Q4_K_M.gguf` (sha `1fe90b72...`) | **FAILED** | tokenizer 0/721 clean, cpu_cuda byte-identical; greedy identity only 1/6 (regression from base's 4/6) — both engines emit garbled `<\|channel\|>`-style tokens even on RAW completions, looks like an imperfect merge, not a runner bug; chat smoke PASSES cleanly ("2+2 = 4") — the instability doesn't surface under real chat usage |
| 15 | HauhauCS gemma-4-26B-A4B QAT uncensored | `HauhauCS/Gemma4-26B-A4B-QAT-Uncensored-HauhauCS-Balanced-MTP/...-Q4_K_M.gguf` (sha `3c131334...`) | **CERTIFIED-WITH-CAVEAT** | tokenizer 0/721, cpu_cuda byte-identical, chat clean "4"/stop; KLD 77.25%/0.155 (documented routing chaos); **"QAT" in the name is misleading — experts are mixed Q8_0/Q4_K/Q5_0 non-uniform PTQ (matches item 5's Bartowski signature), NOT item 4's uniform QAT Q4_0** |
| 16 | BrainStorm GPT-OSS 36B | `DavidAU/OpenAi-GPT-oss-36B-BrainStorm20x-uncensored-gguf/OpenAI-36B-Brains20x-Uncensored-IQ4_NL.gguf` (sha `cc08c58b...`) | **FAILED** | goal doc's "if one exists" confirmed real: genuine 43-layer expansion (base 24L), expert count unchanged 32/top4; layer duplication forced experts off native MXFP4 into mixed IQ4_NL/Q5_1 — only gpt-oss row this session where experts were unavoidably touched; chat smoke fails same runaway pattern as family; tokenizer 222/721 matches family; KLD 87.5%/0.077, second-best of the gpt-oss family (after item 9's 88%/0.040) |

## Tier 3 — speculative decoding (MTP)

| # | artifact | resolved repo/file | verdict | note |
|---|---|---|---|---|
| 17 | Gemma 4 26B-A4B/12B + MTP drafter | main = item 15 (sha `3c131334...`); drafter `HauhauCS/Gemma4-26B-A4B-QAT-Uncensored-HauhauCS-Balanced-MTP/mtp-gemma-4-26B-A4B-it.gguf` (sha `62bd3af7...`) | **REFUSED** | drafter arch `gemma4-assistant` (real 4-layer NextN/MTP drafter, confirmed via a distinct "invalid NextN/MTP layer count" error even under the experimental force-load flag) unsupported by runner's `--draft` path; safe fallback confirmed — 6/6 byte-identical greedy output WITH vs WITHOUT `--draft`, no speedup/slowdown outside noise |

## Tier 4 — big iron

| # | artifact | resolved repo/file | verdict | note |
|---|---|---|---|---|
| 18 | gpt-oss-120b MXFP4 | `ggml-org/gpt-oss-120b-GGUF/gpt-oss-120b-MXFP4.gguf` (sha `582bd40f...`) | **FAILED** | canonical scale-up (36L/128E/top4, uniform MXFP4, same QAT signature as 20b family); disk-exception download (~59GB) worked cleanly, runner auto-fit 13/36 layers to GPU; cpu_cuda and chat smoke fail same as family; tokenizer 222/721 identical to family; **KLD 64.5%/0.254 is the WORST of the entire gpt-oss family — 128 experts vs 20b's 32 compounds the already-diagnosed near-tie routing sensitivity** |
| 19 | gpt-oss-safeguard-120b | `lmstudio-community/gpt-oss-safeguard-120b-GGUF` MXFP4, shard 1/2 only (sha `c53a801f...`); shard 2 deliberately not downloaded | **REFUSED** | every uploader's conversion is a 2-part split; identical failure to item 10 ("missing tensor blk.22...") confirms the runner's split-GGUF gap is universal, not checkpoint-specific; same 36L/128E/top4 uniform-MXFP4 shape as item 18, confirming a genuine same-arch safety fine-tune |
| 20 | 220A20B expanded-expert FrankenMoE | `LLMWildling/gpt-oss-220a20b` (safetensors only, no GGUF) | **NOT FOUND** | no GGUF conversion exists anywhere on HF for this checkpoint or any sibling in the same author's prolific gpt-oss/gemma4 expert-expansion catalog (checked bartowski/unsloth/mradermacher/ggml-org and the author's full repo list) — author publishes safetensors/NVFP4 only |

## Note-only (no download)

| format | why out of scope |
|---|---|
| NVFP4 | NVIDIA block-scaled 4-bit float for Blackwell tensor cores (TensorRT-LLM/vLLM); no GGUF tensor type, no CPU reference kernel, no conversion path |
| FP8 | 8-bit float (E4M3/E5M2), native H100/Blackwell serving format; lives in safetensors/HF checkpoints, not representable as a GGUF tensor type |
| BnB | bitsandbytes NF4/INT8 is a PyTorch runtime quantization applied on load via `transformers`, not a file format — nothing to point a GGUF runtime at |
| MLX | Apple's own array framework/format for `mlx-lm` on Apple Silicon; entirely separate runtime and encoding from GGUF/llama.cpp, no shared schema to test admission against |

## 16 GB envelope sweep

**Recommendation: no candidate beats keep-30** — fastest under a real 16 GiB
ballast-capped test (13.2-13.3 tok/s) and the only pruning ratio (2 of 32
experts) that clears the 97%/0.05 quality bar. Full numbers, ballast
methodology, and the QAT-vs-PTQ pruning-tolerance experiment in
`docs/cert-matrix-2026-08-05.md`.

| candidate | fits? | lever | top-1 vs parent | mean KLD | tok/s cold/warm (16GiB cap) | verdict |
|---|---|---|---|---|---|---|
| gpt-oss-20b keep-30 (current holder) | fits (11.5GB) | 32→30 experts, native MXFP4 | ≥97% (prior session) | ≤0.05 (prior session) | **13.32 / 13.18** | holds |
| Gemma 4 26B-A4B QAT Q4_0 (item 4), unpruned | fits with levers (`--kv q8`, 14.41GB) | none needed to fit | n/a (not pruned) | n/a | 7.08 / 7.32 | fits, usable, doesn't beat keep-30 |
| Gemma 4 26B-A4B QAT Q4_0, keep-96/64/48 (further pruned) | n/a (didn't need pruning) | 128→96/64/48 experts | 67.75% / 50.25% / 45.25% | 0.377 / 0.719 / 0.896 | not live-tested (failed KLD gate) | **FAIL all 3 points** — refutes "QAT tolerates pruning better" |
| GPT-OSS Nano 9B (item 9), unpruned | fits trivially (6.36GB) | none | n/a (not pruned) | n/a | 12.23 / 12.11 | fits, usable, doesn't beat keep-30 |
| GPT-OSS Nano 9B, keep-10/8/6 (further pruned) | n/a (didn't need pruning) | 12→10/8/6 experts | 79.5% / 72.0% / 59.25% | 0.099 / 0.180 / 0.344 | not live-tested (failed KLD gate) | **FAIL all 3 points** — already-pruned base has no headroom left |
| GPT-OSS 120B REAP 58B (item 10) | does not fit | none rescues it (39GB min quant, 2.4x budget) | n/a | n/a | not tested | **does not fit**, arithmetic only |
