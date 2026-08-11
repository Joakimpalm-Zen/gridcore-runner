# muse-glimmer (Meta Muse Glimmer 30B) certification — 2026-08-11

## Verdict: CERTIFIED-WITH-CAVEAT (official kquant, text path)

Load, tokenizer (0/721), cpu_cuda (6/6 byte-identical), fixture-scale
CPU↔Metal identity, and chat all PASS. The caveat is greedy identity vs
llama.cpp b10353: **3/6 byte-identical**, and the sensitivity-floor
protocol (the afmoe-session lesson) shows the gap sits at the model's own
numerical floor rather than above it:

- llama.cpp vs ITSELF (cold vs warm cache): 15/16 identical — the
  reference alone cannot do better than 1-in-16 disagreement.
- runner vs itself (f16 → q8 KV): 12/16 identical, mean Δlogprob 0.0029.
- runner vs llama.cpp (16 prompts, 64 tok, tie bar 0.25 nats): 11/16
  identical, 4 divergences at near-ties, **1 for real** — max Δlogprob
  0.178 (afmoe FAILED this same reading at 1.867 nats with 3/6 real).

Cross-engine disagreement ≈ the runner's own KV-perturbation floor
(11/16 vs 12/16), so the 3/6 headline is numerical tie-breaking on a
4-bit quant, not a wrong implementation — the same class as the certified
gemma-4-12B row (4/6 with degenerate long runs).

Environment: runner `2ce691f` (the arch-port commit), llama.cpp reference
**b10353** (the build that introduced muse-glimmer support, merged
2026-08-10 — pinned as this arch's reference revision; the rest of the
matrix stays on its own pins), Blackwell box (RTX PRO 6000 MIG 1g.24gb,
conda ccbuild toolchain). Reference built CPU-only, static
(`-DBUILD_SHARED_LIBS=OFF` — the shared build silently resolved the conda
env's older libllama via RPATH and refused the arch; a stale-reference
failure mode worth remembering).

Artifact under test (arch certification): the OFFICIAL meta-models GGUF
`muse-glimmer-30B-kquant-17gb.gguf`, 16.8 GB 4-bit k-quant, sha256
`7e9b74b7c8875e9e265695df9613bf6290f2392e479ce740495a129019c488d8`.
Text path only: the separate vision encoder (`mmproj-kquant.gguf`) and the
atem tool-call syntax are not implemented and not claimed.

## Gates

| gate | result |
|---|---|
| fixture suite (M1 + CI-shape) | **PASS** — load/determinism/gate-participation/pattern-array tests green; CPU↔Metal byte-identical on all three muse fixtures (default, gate-flat, all-swa) on an 8 GB M1 |
| tokenizer differential | **PASS — 0/721** vs the HF reference tokenizer (`meta-models/Muse-Glimmer-30B` tokenizer.json, `tokenizers` 0.22.2). The new `llama4` → o200k split rules are exact on the full corpus |
| cpu_cuda | **PASS — 6/6 byte-identical** on the full cert prompt set (a/b/c/d at 64 tokens, b-long/c-long at 256) at full offload (52/52 layers, 16.8 GB in VRAM). Notably prompts a, d and b-long are internally CPU==CUDA identical while diverging from llama.cpp — engine-internal determinism holds everywhere |
| greedy identity vs llama.cpp b10353 | **3/6 byte-identical** (b, c, and the 256-token c-long). The three divergences (a @byte 114, d @byte 222, b-long @byte 323) all split DEEP into an identical prefix — the near-tie signature, not early structural divergence; both tails stay fluent and on-topic (b-long: `reverse_linked_list` vs `reverseList` naming). Read against the sensitivity floor below |
| chat smoke | **PASS** — real-weight `/v1/chat/completions` "What is 2+2? Answer with just the number." → `content: "4"`, reasoning cleanly separated into `reasoning_content` (the ` to=self` turn), `finish_reason: stop` via the new `<|eot|>` stop probe. TMPL_MUSE renders the model's own template shape |

## Footprint (route A) — 16 GB-Mac quant certification

KLD protocol: `kld-compare-raw.py`, raw completions, word-boundary teacher
forcing over `tests/fixtures/mixed-corpus.txt`, top-20 union approximation,
all arms CPU-served by the same runner build. Harness validated first:
official-vs-official self-comparison over 100 positions scored **exactly
mean KLD 0.0 / top-1 100% / top-8 overlap 1.0**.

| candidate | size | result |
|---|---|---|
| bartowski IQ3_XXS | 11.5 GB | **REFUSED — engine gap, not a quality verdict.** `error: tensor token_embd.weight has unsupported type 21` (IQ3_S; the file mixes IQ3_XXS/IQS3_S tensors). The runner's quant roster has no IQ1–IQ3 family. The plan's preferred 11.5–12.3 GB tier is unreachable until IQ3 dequant support exists |
| bartowski IQ3_XS | 12.3 GB | **REFUSED** — same IQ3 gap |
| bartowski Q3_K_S | 12.8 GB | measured below |
| bartowski Q2_K | 11.0 GB | measured below |

## DFlash drafter

The published 1.63 GB `dflash-kquant.gguf` is its own architecture
(`general.architecture: dflash`, 5 blocks over the shared 202k vocab) — a
DFlash block drafter, not a small muse-glimmer. It does NOT ride the plain
`--draft` path: the CPU verify path refuses it at admission with the loud
`unsupported architecture 'dflash'` error (fail-closed, same shape as the
Tier-3 `gemma4-assistant` REFUSED precedent), and a fully-GPU-offloaded
target ignores `--draft` by design. Consuming it would be a new drafter
arch, tracked as an open item, not smuggled in here.

## Notes

- CPU decode 5.5 tok/s (96 threads), CUDA full-offload decode 19.0 tok/s on
  the MIG slice; prompt 13–15 tok/s at this trivial prompt length.
- The model thinks by default (reasoning strength high in the reference
  system turn); `enable_thinking:false` pins the generation prompt to
  `<|start|>assistant to=user<|message|>`, which is the format's own way to
  suppress the reasoning turn.
