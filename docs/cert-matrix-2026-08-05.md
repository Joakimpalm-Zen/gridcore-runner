# Cert-matrix detailed report — GPT-OSS x Gemma 4 derivative ecosystems

Per-artifact gate evidence. Status table (quick view): `docs/cert-matrix-status.md`.

## Environment

| item | value |
|---|---|
| repo | fresh clone, branch `cert-matrix`, into `~/workspace/Gridcore/cert-matrix/` |
| runner version | `runner 0.1.8-alpha` |
| build | `make runner CC=x86_64-conda-linux-gnu-gcc -j`, conda env `ccbuild` |
| llama.cpp reference | `b10280 (61881b1f7)`, prebuilt, `~/workspace/Gridcore/lcpp-bin/llama-b10280/` (unless a section notes a split reference) |
| box | 128 cores, big NVIDIA GPU (MIG slice), CPU gates use `--gpu off` explicitly |

## Gate battery (recap, see goal doc for full text)

1. Identity (sha256 + metadata dump)
2. Admission (load or clean refusal)
3. Tokenizer differential vs HF reference
4. Reference gate (KLD for gpt-oss/gemma4-moe; token identity for gemma4 dense/E-series)
5. cpu_cuda byte-identity
6. Chat smoke (`--serve`, OpenAI completion)
7. Perf row (`--bench-json`, GPU and CPU)

---

<!-- Per-artifact sections appended below, one per roster item, in order. -->

## 1. ggml-org gpt-oss-20b MXFP4 (re-baseline)

**Verdict: FAILED** — gate 3 (tokenizer, 222/721 diverge), gate 5 (cpu_cuda, not byte-identical on this box's GPU), and gate 6 (chat smoke, runaway non-coherent completion) all fail. Gates 1/2/7 pass; gate 4 (KLD) misses the session's stated bar but matches an already-documented characteristic rather than revealing something new (see below) — not counted as a fresh failure, but not a pass either. Three of seven gates failing outright is not a "certified with a footnote" result.

**Resolved:** `ggml-org/gpt-oss-20b-GGUF/gpt-oss-20b-MXFP4.gguf`. Already present on the box at `gridcore-runner/models/gpt-oss-20b-MXFP4.gguf`; sha256 verified to match the HF LFS blob's sha256 exactly (`27cd6c432c7672cb812a92f611cf3ba7bbc35928262bb1e1253ff4ee6ae35901`, 12,109,566,624 bytes) before symlinking in, rather than re-downloading.

### Gate 1 — Identity

**Manifest:** `GPTOSS / 24L / 32E / top4 / MXFP4_MOE / Harmony`

```
architecture: gpt-oss   block_count: 24   expert_count: 32   expert_used_count: 4
embedding_length: 2880  feed_forward_length: 2880  context_length: 131072 (train)
tensor histogram (459 tensors): Q8_0 x98, F32 x289, MXFP4 x72
expert-tensor types (192 tensors, all 24 layers uniform): F32 x120 (biases/router), MXFP4 x72 (gate/up/down)
output.weight: Q8_0   output_norm.weight: F32   token_embd.weight: Q8_0
```

Confirms the file's expert FFNs are genuinely MXFP4_MOE (this is the reference every later gpt-oss row diffs against for gate 1's "record what each file ACTUALLY is").

### Gate 2 — Admission: PASS

Loads cleanly, both `--gpu off` and `--gpu auto` (full 24/24 layer offload on this box's RTX PRO 6000 Blackwell MIG 1g.24gb slice, 12.1 GB weights in VRAM).

### Gate 3 — Tokenizer: **222/721 diverge (30.8%)** vs `openai/gpt-oss-20b`

Not clean. Characterized, not rationalized — two dominant patterns account for the bulk of the divergent strings:
- **CRLF-as-one-token**: the reference tokenizer merges `\r\n` into a single token (id 370); the runner emits `\r` and `\n` as two separate tokens. Reproduces on every `\r\n`-containing test string (`"\r\nhello"`, `"\r\n1"`, `"\r\n!"`, `"\r\nthe"`, …).
- **Cross-hyphen BPE merges**: the reference merges e.g. `"pre-tokenizer"` as `['pre', '-token', 'izer']` (a single `-token` token, id 73397); the runner keeps `-` and the following word fragment separate (`['pre', '-', 'token', 'izer']`). Same pattern for `"byte-level"` -> ref merges `-level` into one token, runner does not.

Full log: `docs/cert-matrix-evidence/t1.1-difftok.log`. This is a real merge-table/BPE-engine gap, not a fine-tune-changed-vocab case (this is the canonical file) — recorded as measured.

### Gate 4 — Reference (KLD, gpt-oss protocol)

**The literal `/v1/chat/completions` protocol the goal doc specifies is unsound for cross-engine gpt-oss comparison** and was not used for the final number — found and documented before being trusted, same discipline as the afmoe cert session:

- Via `scripts/kld-compare.py` (chat-completions endpoint) at 400 positions: `top1_agreement_pct: 0.0`, `mean_kld: 31.17`. This number is **invalid**, not a quality finding: llama.cpp's Harmony chat-template rendering forces `<|channel|>` structured markup as the first emitted token (`logprob 0.0`, i.e. grammar-forced), while the runner's chat-template rendering for the same message starts with plain text (`" [/"`). The two engines are scoring completely different literal token sequences, not disagreeing about the same one. Evidence: `docs/cert-matrix-evidence/t1.1-kld-chatendpoint-INVALID.json`, plus the two raw chat-completion responses saved during triage.
- Wrote `scripts/kld-compare-raw.py` (new, committed) — same word-by-word teacher-forcing KLD protocol, but against `/v1/completions` (no chat template on either side), with response-schema normalization (runner's parallel-array logprobs vs llama.cpp's OpenAI-style `content` list). Verified sound via self-comparison first: `mean_kld 0.0, top1 100%, top8_overlap 1.0` (40 positions).
- **Real cross-engine result, 400 positions:** `mean_kld: 0.1276, top1_agreement_pct: 83.0, mean_top8_overlap: 0.898`. Evidence: `docs/cert-matrix-evidence/t1.1-kld-raw.json`.

**This misses the stated bar (top-1 >= 97%, KLD <= 0.05), but it is not a new finding** — it is a precise, 400-position confirmation of an already-diagnosed, already-documented gap: `tests/compatibility/out/divergence-study-gpt-oss-2026-08-01.json` measured this exact runner-vs-llama.cpp pair at 16 prompts x 16 tokens and found llama.cpp perfectly self-consistent (16/16) while cross-engine agreement was only 4/16, diagnosing the systematic (not chaotic) cause: `ggml` gives MXFP4 a `vec_dot_type` of `Q8_0` (llama.cpp quantizes activations to int8 before every expert dot product), while the runner dots against full fp32 activations — a different computation by construction, documented in CHANGELOG.md. The README's current claim ("agreement with llama.cpp is at this model's own sensitivity floor rather than token identity, so `greedy_reference` is not claimed") is consistent with this measurement in spirit, but the specific numeric bar this session's goal doc sets (97%/0.05) is **not met and was never claimed to be** — the goal doc's phrasing ("agreement... sits at the model's own sensitivity floor; do NOT gate on token identity... Bar: top-1 >= 97%") appears to assume the floor is tighter than it measurably is for this architecture. Recording the miss rather than reinterpreting the bar.

### Gate 5 — cpu_cuda: **FAIL**

`--gpu off` vs `--gpu auto` (full 24/24 offload) are **not byte-identical**, at both 32 and 64 tokens, on two different prompts, contradicting the current README claim ("CUDA GPU/CPU byte-identical") for this architecture:

- Both sides individually self-consistent across repeated runs (ruled out nondeterminism before concluding divergence).
- "The capital of France is", 64 tokens: first divergent byte at offset 43 (`"...plants.  \n**Answer**: Photosynthesis..."` CPU vs `"...plants.\n\nPhotosynthesis..."` GPU).
- "The capital of France is" / "Explain how photosynthesis works in plants." both diverge at 32 tokens too — the token count `docs/bench-2026-08-01-3070.md` recorded as "5/5 byte-identical" for this model.

**Likely explanation, not confirmed:** that prior byte-identity measurement was on an RTX 3070; this box's GPU is an RTX PRO 6000 Blackwell **MIG 1g.24gb slice** — a different compute capability and very likely a different reduction/tensor-core code path for the MXFP4 kernels. This reads as GPU-architecture-dependent numerical behavior, not a universal regression, but it was not re-verified on a second GPU this session (none available) — recorded as measured on this box, full evidence in `docs/cert-matrix-evidence/t1.1-cpu-64tok.txt` / `t1.1-gpu-64tok.txt`.

### Gate 6 — Chat smoke: **FAIL**

`--serve`, `"What is 2+2? Answer briefly."`, temperature 0. Reproducible (identical output across two independent requests, capped and uncapped):

```
" Assistant: 4. \nWe have a conversation. The user says: \"I want to create a new user in the database...
```

Answers correctly ("4") in the first four words, then the model runs away into unrelated, repetitive fabricated content (a fake user-database schema, looping "The user has a name... phone number... email..." dozens of times) until hitting `max_tokens` (`finish_reason: "length"` even at 1024 tokens — never stops cleanly). No literal `<|channel|>`/`<|message|>` Harmony tokens leak into the `content` field, so the specific tag-leak check the goal doc names is technically clean, but the response is **not coherent**, which is the gate's actual PASS bar. The leading `" Assistant: 4."` prefix (echoing a role label into content) is itself a mild template-rendering artifact, consistent with the Harmony-rendering divergence found in gate 4 — the runner's chat-template output for gpt-oss does not look like a clean single-turn assistant message. Full response: `docs/cert-matrix-evidence/t1.1-chat-smoke.json`.

### Gate 7 — Perf row

```
CPU: {"gpu_layers":0,"layers":24,"prompt_tokens":512,"generated_tokens":256,"prompt_tok_s":62.842,"gen_tok_s":12.329,"prompt_s":8.147,"gen_s":20.765}
GPU: {"gpu_layers":24,"layers":24,"prompt_tokens":512,"generated_tokens":256,"prompt_tok_s":37.158,"gen_tok_s":31.904,"prompt_s":13.779,"gen_s":8.024}
```

(GPU prompt throughput lower than CPU here is plausibly the MIG slice's shared/contended compute — not investigated further, out of scope for a perf-row record.)

### Summary

Admission and the file-identity manifest are clean. Three of the remaining four gates surface real, evidence-backed gaps: tokenizer (30.8% divergent), cpu_cuda (not byte-identical on this box's GPU), and chat smoke (runaway, non-coherent completion). The KLD reference gate misses the session's stated numeric bar but matches an already-documented, already-diagnosed characteristic rather than revealing something new. None of this was fixed (STOP rule) — every gate result here is the deliverable.

## 2. Bartowski gpt-oss-20b Q6_K_L

**Verdict: FAILED** — same three gates as item 1 fail (tokenizer, cpu_cuda, chat smoke), for the same underlying reasons. Included in full regardless, per "record with the same care as passes" — a repeat failure with identical root cause is still evidence, and gate 1 is the interesting one here.

**Resolved:** `bartowski/openai_gpt-oss-20b-GGUF/openai_gpt-oss-20b-Q6_K_L.gguf`. Downloaded (curl over HTTP/2 reset mid-transfer once — retried with `--http1.1 --retry 5`, succeeded). sha256 `e729b05fa245760f29e230c71aa7a8afa3065838dd95dea169d50788babb10c5` matches the HF LFS blob exactly, 12,040,998,976 bytes. Deleted after this verdict per disk discipline.

### Gate 1 — Identity: **the mixed-tensor trap, confirmed as the goal doc predicted**

**Manifest:** `GPTOSS / 24L / 32E / top4 / MXFP4_MOE / Harmony` — architecturally identical to item 1.

```
tensor histogram (459 tensors): Q8_0 x74, F32 x289, Q6_K x24, MXFP4 x72
expert-tensor types (192 tensors, all 24 layers uniform): F32 x120, MXFP4 x72   <- UNCHANGED from item 1
non-expert-tensor types: Q8_0 x74, F32 x169, Q6_K x24                          <- 24 tensors moved Q8_0 -> Q6_K vs item 1
```

Confirmed exactly as the goal doc named this gate: **"Q6_K_L" in the filename refers only to 24 non-expert tensors** (Bartowski's embed/output-layer precision knob); **all 72 expert FFN tensors remain byte-for-byte the same MXFP4_MOE type** as the canonical ggml-org file. File size is actually marginally *smaller* than item 1 (12.04 GB vs 12.11 GB) despite the "Q6_K" name suggesting more precision than Q8_0 — it doesn't; Q6_K is lower-bit than Q8_0, so this is a small size/quality trade on a handful of non-expert tensors only. This is exactly the "record what each file ACTUALLY is, not what its name says" case gate 1 exists for.

### Gate 2 — Admission: PASS

### Gate 3 — Tokenizer: **222/721 diverge (30.8%)**, identical count to item 1

Same divergent-string set (CRLF-as-one-token, cross-hyphen merges) — confirms this is a runner-side tokenizer characteristic of the gpt-oss vocab/BPE table, not something introduced by Bartowski's conversion. `docs/cert-matrix-evidence/t1.2-difftok.log`.

### Gate 4 — Reference (KLD, raw-completions protocol): consistent with item 1

400 positions: `mean_kld: 0.1155, top1_agreement_pct: 84.0, mean_top8_overlap: 0.9003`. Within ~1 point of item 1's 83.0%/0.1276 — as expected, since the expert tensors (where the diagnosed MXFP4 `vec_dot_type` mismatch lives) are byte-identical between the two files; the 24 requantized non-expert tensors move the number only slightly. Misses the session's 97%/0.05 bar for the same already-diagnosed reason as item 1. Evidence: `docs/cert-matrix-evidence/t1.2-kld-raw.json`.

### Gate 5 — cpu_cuda: **FAIL**, same pattern as item 1

`--gpu off` vs `--gpu auto` diverge at 64 tokens on this box's Blackwell MIG slice (first divergent byte 33: CPU continues `"...Paris."\n\nSure! Here's a simple example..."`, GPU continues `"...Paris."\n    # Test with a non-existent page..."`). Consistent with item 1's finding, not re-litigated as a new mystery.

### Gate 6 — Chat smoke: **FAIL**, same pattern as item 1

`"What is 2+2? Answer briefly."` -> `" Assistant: 4. \nWe need to produce a response that is a single word, no punctuation..."` — answers correctly then continues into unrelated meta-commentary about the response format rather than stopping. Same Harmony-rendering / stop-handling characteristic as item 1. `docs/cert-matrix-evidence/t1.2-chat-smoke.json`.

### Gate 7 — Perf row

```
CPU: {"gpu_layers":0,"layers":24,"prompt_tok_s":62.078,"gen_tok_s":12.147,"prompt_s":8.248,"gen_s":21.076}
GPU: {"gpu_layers":24,"layers":24,"prompt_tok_s":37.342,"gen_tok_s":31.568,"prompt_s":13.711,"gen_s":8.110}
```

Within noise of item 1's numbers, as expected for a file differing in only 24 non-expert tensors.

### Summary

The interesting result here is gate 1, exactly as the goal doc anticipated: Bartowski's "Q6_K_L" name is honest about *quantizing something*, but the experts — the overwhelming majority of the file's parameters and the reason MXFP4 exists at all — are untouched. Gates 3/5/6 reproduce item 1's failures with the same root causes (not re-diagnosed from scratch), which is itself informative: these are properties of the *architecture implementation on this runner build*, not properties of any one file.

## 3. Unsloth gpt-oss-20b Q4_K_M

**Verdict: FAILED** — same three gates fail as items 1/2, but gate 6 (chat smoke) fails *differently* here: it terminates cleanly instead of running away, which is a real, distinct partial improvement worth recording precisely rather than lumping in as "same failure."

**Resolved:** `unsloth/gpt-oss-20b-GGUF/gpt-oss-20b-Q4_K_M.gguf`. Downloaded, sha256 `c27536640e410032865dc68781d80a08b98f8db5e93575919af8ccc0568aeb4f` verified against the HF LFS blob, 11,624,759,488 bytes. Deleted after this verdict.

### Gate 1 — Identity

**Manifest:** `GPTOSS / 24L / 32E / top4 / MXFP4_MOE / Harmony`

```
tensor histogram (459 tensors): Q8_0 x13, F32 x289, Q5_0 x61, Q4_K x24, MXFP4 x72
expert-tensor types (192 tensors): F32 x120, MXFP4 x72   <- experts UNCHANGED, same as items 1/2
non-expert-tensor types: Q8_0 x13, F32 x169, Q5_0 x61, Q4_K x24
token_embd.weight: Q5_0 (items 1/2 had this at Q8_0)
```

Third confirmation of the mixed-tensor pattern: "Q4_K_M" names a granular per-tensor-role scheme (Unsloth's own "Dynamic" quant strategy — a genuine 3-way split across Q8_0/Q5_0/Q4_K for non-expert tensors, not a uniform Q4_K_M), while the 72 expert tensors are, again, untouched MXFP4. Notably the GGUF's `tokenizer.chat_template` KV begins with a literal comment: `"{# Chat template fixes by Unsloth #}"` — Unsloth is aware of and has patched gpt-oss's chat template, which matters for gate 6 below.

### Gate 2 — Admission: PASS

### Gate 3 — Tokenizer: **222/721 diverge (30.8%)**, identical to items 1/2

Confirms (third file, third source) this is a runner-side characteristic of the gpt-oss vocab, not introduced by any specific conversion. `docs/cert-matrix-evidence/t1.3-difftok.log`.

### Gate 4 — Reference (KLD, raw-completions protocol)

400 positions: `mean_kld: 0.1374, top1_agreement_pct: 78.25, mean_top8_overlap: 0.874`. Slightly lower top-1 than items 1/2 (83-84%), consistent with this file's more aggressive non-expert quantization (Q4_K/Q5_0 vs mostly-Q8_0) compounding the already-diagnosed MXFP4 activation-quantization gap. Still the same underlying cause, not a new one. Evidence: `docs/cert-matrix-evidence/t1.3-kld-raw.json`.

### Gate 5 — cpu_cuda: **FAIL**, same pattern as items 1/2

Diverges at 64 tokens, same GPU-architecture-dependent characteristic.

### Gate 6 — Chat smoke: **FAIL, but a genuinely different failure mode**

`"What is 2+2? Answer briefly."`:

```
'"\n\nWe need to produce a short answer: 4. But we must follow the style guidelines: no mention of policies, no mention of being an AI, no mention of the policy. Just answer. So answer: 4.'
```

`finish_reason: "stop"` — **this is the first gpt-oss file this session where the completion terminates on its own** rather than running to `max_tokens`/`"length"` (items 1 and 2 both ran away to 1024 tokens). Unsloth's chat-template fix (see gate 1) plausibly explains the difference: stopping is a real, measurable improvement. But the content is still not a coherent, direct answer — the visible text is entirely reasoning/meta-commentary about how to format a response ("we must follow the style guidelines... no mention of policies... Just answer.") with the actual answer buried at the very end, and it opens with a stray, dangling `"` character. Still fails the gate's "coherent answer" bar, but the specific way it fails is materially different from items 1/2 and worth an engineer's attention as a smaller, more tractable gap. `docs/cert-matrix-evidence/t1.3-chat-smoke.json`.

### Gate 7 — Perf row

```
CPU: {"gpu_layers":0,"layers":24,"prompt_tok_s":60.462,"gen_tok_s":12.346,"prompt_s":8.468,"gen_s":20.735}
GPU: {"gpu_layers":24,"layers":24,"prompt_tok_s":33.061,"gen_tok_s":28.644,"prompt_s":15.487,"gen_s":8.937}
```

### Summary

Three-for-three on the mixed-tensor-trap gate 1 confirmation across independent conversion sources (ggml-org, Bartowski, Unsloth) — the finding generalizes: nobody who converts gpt-oss touches the expert tensors, because MXFP4 is gpt-oss's native/required format and there is no reason to requantize it. Gate 6 is the one genuinely new data point this file adds: Unsloth's documented chat-template patch changes *how* the failure manifests (clean stop vs runaway) without fixing the underlying issue (analysis-channel content leaking into the visible answer) — useful signal for whoever eventually debugs the runner's Harmony handling.

## 4. Google gemma-4-26B-A4B-it QAT Q4_0

**Verdict: CERTIFIED-WITH-CAVEAT** — six of seven gates pass cleanly; the one miss (gate 4, KLD) reproduces an already-documented "numerically chaotic" characteristic for this exact file (not a fresh finding). This is the strongest result of the session so far, and the goal doc's own framing ("QAT + MoE; the single most important new artifact in the list") is borne out.

**Resolved:** `google/gemma-4-26B-A4B-it-qat-q4_0-gguf/gemma-4-26B_q4_0-it.gguf`. Already present on the box (`gridcore-runner/models/gemma-4-26B_q4_0-it.gguf`); sha256 `3eca3b8f6d7baf218a7dd6bba5fb59a56ee25fe2d567b6f5f589b4f697eca51d` verified against the HF LFS blob before symlinking in — no download needed. 14,439,363,584 bytes.

### Gate 1 — Identity

**Manifest:** `GEMMA4-MOE / 30L / 128E / top8 / Q4_0-QAT / gemma-canonical-chat`

```
architecture: gemma4   block_count: 30   expert_count: 128   expert_used_count: 8
embedding_length: 2816  feed_forward_length: 2112  expert_feed_forward_length: 704
context_length: 262144
tensor histogram (658 tensors): F32 x392, Q6_K x1 (token_embd), Q4_0 x265
expert-tensor types (150 tensors, fused gate_up_exps + down_exps x30 layers = 60 quantized + 90 F32 aux, uniform): F32 x90, Q4_0 x60
```

Genuinely QAT: the experts are quantization-*aware*-trained Q4_0, not a post-hoc requant of a bf16 checkpoint — a materially different provenance from every gpt-oss file above, where MXFP4 is native but nothing else in the file is QAT'd.

### Gate 2 — Admission: PASS

### Gate 3 — Tokenizer: **0/721 diverge** vs `google/gemma-4-26B-A4B-it` — clean

### Gate 4 — Reference (KLD, gemma4-moe protocol): misses the numeric bar, but matches a documented floor

400 positions (raw-completions protocol, same tool built for the gpt-oss rows): `mean_kld: 0.1255, top1_agreement_pct: 80.5, mean_top8_overlap: 0.841`. Misses 97%/0.05.

**This is not a new finding.** `tests/compatibility/out/divergence-study-gemma4-moe-2026-08-01.json` already measured **this exact file** (sha `3eca3b8f`, listed by name) at 16 prompts x 16 tokens: `self_runner_identical: 3/8 (24 tok)` (i.e. the runner disagrees with *itself* under a KV-precision perturbation more than it disagrees with llama.cpp: `cross_engine_identical: 5/16`). Root cause already diagnosed there: discrete top-8-of-128 expert routing over Q4_0 weights is unstable at ties — layer 2's 6th/7th-ranked experts sat 0.0002 apart, so ordinary rounding flips which expert fires and rewrites an eighth of the FFN output. My 400-position measurement is a more statistically robust confirmation of the same documented chaos, at the same order of magnitude (roughly 1-in-5 disagreement either way). README's existing framing — "too numerically chaotic to gate on token identity" — holds. Evidence: `docs/cert-matrix-evidence/t1.4-kld-raw.json`.

### Gate 5 — cpu_cuda: **PASS** — byte-identical

`--gpu off` vs `--gpu auto` (full 30/30 offload, 14.4 GB in VRAM on this box's Blackwell MIG slice) produce **byte-identical** 64-token output. Notably different from every gpt-oss row above (all FAIL this gate on this box) — gemma4's dense+routed dual-branch MoE math evidently reproduces across CPU/CUDA on this specific hardware even though gpt-oss's MXFP4 kernels do not.

### Gate 6 — Chat smoke: **PASS**

`"What is 2+2? Answer briefly."` -> `"4"`, `finish_reason: "stop"`. Clean, direct, correctly terminated — no analysis-channel leakage, no runaway. `docs/cert-matrix-evidence/t1.4-chat-smoke.json`.

### Gate 7 — Perf row

```
CPU: {"gpu_layers":0,"layers":30,"prompt_tok_s":8.886,"gen_tok_s":6.638,"prompt_s":57.616,"gen_s":38.566}
GPU: {"gpu_layers":30,"layers":30,"prompt_tok_s":23.788,"gen_tok_s":24.127,"prompt_s":21.524,"gen_s":10.610}
```

### Summary

Every gate that measures *implementation correctness* (admission, tokenizer, cpu_cuda, chat) passes cleanly. The one gate that misses its numeric bar (KLD) misses it for a reason already on record for this precise file, not a new defect. This is the cleanest result of the session and squarely supports treating this artifact as the lead QAT+MoE reference the goal doc expected it to be.

## 5. Bartowski gemma-4-26B-A4B-it Q4_K_M — the QAT-vs-PTQ pair

**Verdict: CERTIFIED-WITH-CAVEAT** — same shape of result as item 4 (admission/tokenizer/cpu_cuda/chat all pass, KLD misses the numeric bar for reasons already understood), but this row's real value is the **head-to-head against item 4's QAT build**, which the roster called out as "the whole QAT-vs-PTQ story in one row." It delivers a striking number.

**Resolved:** `bartowski/google_gemma-4-26B-A4B-it-GGUF/google_gemma-4-26B-A4B-it-Q4_K_M.gguf`. Downloaded (17,035,039,872 bytes; one HTTP/2 hiccup mid-transfer, `--http1.1 --retry 5` completed it). sha256 `a07f72221e8e3f77455ab0d7f7652d01a9f63c262b954aa6932a53275a0e895a` verified. Deleted after this verdict; item 4's file was briefly re-symlinked (it costs nothing — it lives on the box already, outside this session's download budget) specifically to run the head-to-head comparison below, then removed again.

### Gate 1 — Identity: **the experts ARE requantized here — the opposite of every gpt-oss row**

**Manifest:** `GEMMA4-MOE / 30L / 128E / top8 / MIXED-PTQ(Q8_0/Q5_0/Q4_K, non-uniform per-layer) / gemma-canonical-chat`

```
tensor histogram (658 tensors): F32 x392, Q6_K x15, Q8_0 x83, Q5_K x30, Q4_K x106, Q5_0 x32
expert-tensor types (150 tensors): F32 x90, Q8_0 x14, Q4_K x30, Q5_0 x16   <- MIXED, unlike item 4's uniform Q4_0
non-uniform expert dtype ACROSS LAYERS: confirmed (some layers' experts sit at Q8_0, others Q5_0, others Q4_K)
```

This is the mirror image of the gpt-oss mixed-tensor-trap finding: where every gpt-oss conversion leaves MXFP4 experts untouched (because MXFP4 is gpt-oss's native/required format), Bartowski's standard PTQ pipeline for gemma-4 **does** requantize the experts, and does so **non-uniformly per layer** — almost certainly imatrix-guided, giving more bits to layers whose activations are more sensitive. Worth recording precisely: "Q4_K_M" here is not one quant type applied once, it is a per-tensor-role, per-layer decision.

### Gate 2 — Admission: PASS

### Gate 3 — Tokenizer: **0/721 diverge** — clean, same as item 4

### Gate 4 — Reference: two numbers, and the interesting one is the comparison between them

**Standard protocol (vs llama.cpp, 400 positions):** `mean_kld: 1.0055, top1_agreement_pct: 65.5, mean_top8_overlap: 0.660`. Misses the bar, and misses it by *more* than item 4 did (item 4: 80.5%/0.126). `docs/cert-matrix-evidence/t1.5-kld-raw.json`.

**Head-to-head, this file vs item 4's QAT build (both runner-served, 400 positions, `kld-compare-raw.py --model-a --model-b` — no third engine involved):**

```json
{"positions_scored": 400, "mean_kld": 1.9503, "top1_agreement_pct": 37.5, "mean_top8_overlap": 0.351}
```

**The two "same" 26B-A4B models disagree with EACH OTHER (37.5% top-1) even more than either disagrees with llama.cpp.** This is not a contradiction, and not a harness bug (the tool's self-comparison zero point was independently re-verified during the T1.1 investigation earlier this session, `mean_kld 0.0 / top1 100%`): it is the direct, expected consequence of the already-documented finding for this architecture — top-8-of-128 expert routing is decided by ties as fine as 0.0002 in weight, so ANY precision difference between two builds can flip which experts fire and rewrite up to an eighth of the FFN output. QAT-vs-PTQ is a *much* larger precision difference than the KV-cache-only perturbation the sensitivity-floor study used (which alone produced 11/16 runner-self-disagreements) or than switching inference engines on the *same* weights, so a bigger disagreement here than either of those is the predicted result, not an anomaly.

**The comparative signal worth keeping**: item 4 (QAT) agrees with llama.cpp on 80.5% of positions; item 5 (PTQ) agrees on only 65.5%. Whatever the QAT training does, it makes this model's output *more* consistent with an independent reference implementation than post-hoc quantization of the same architecture does — a real, measurable point in QAT's favor for a model this routing-sensitive, precisely the experiment the roster asked this pairing to run.

### Gate 5 — cpu_cuda: **PASS** — byte-identical (64 tokens; generation ended at 48 tokens via natural EOS on this prompt)

### Gate 6 — Chat smoke: **PASS**

`"What is 2+2? Answer briefly."` -> `"4"`, `finish_reason: "stop"`. Same clean behavior as item 4.

### Gate 7 — Perf row (re-measured on a quiet box after the head-to-head comparison's two concurrent servers exited)

```
CPU: {"gpu_layers":0,"layers":30,"prompt_tok_s":8.567,"gen_tok_s":6.369,"prompt_s":59.764,"gen_s":40.192}
GPU: {"gpu_layers":30,"layers":30,"prompt_tok_s":83.562,"gen_tok_s":47.944,"prompt_s":6.127,"gen_s":5.340}
```

Notably faster on GPU than item 4's QAT build (47.9 vs 24.1 gen tok/s) — plausibly Q4_K's dequant kernels are better optimized on this hardware than Q4_0's, though this was not investigated further (out of scope for a perf-row record).

### Summary

Gates 1-3, 5-6 pass; gate 4 misses the numeric bar for a documented, architecture-level reason. The distinguishing result of this row is the QAT-vs-PTQ head-to-head the roster specifically asked for: two nominally-equivalent 26B-A4B builds disagree with each other more than either disagrees with an independent engine, and the QAT build is measurably closer to that independent engine than the PTQ build is. That is a genuine, actionable data point about which quantization strategy to prefer for this architecture — delivered exactly as the roster's own framing predicted it would be.

## 6. Google gemma-4-12B-it QAT Q4_0

**Verdict: CERTIFIED-WITH-CAVEAT** — the first dense-gemma4 row this session, and the greedy-identity gate (not KLD — this architecture is not MoE) passes on all four short-domain prompts; the only misses are two 256-token runs that both degenerate into repetitive loops, a known small-model failure mode rather than an architecture defect.

**Resolved:** `google/gemma-4-12B-it-qat-q4_0-gguf/gemma-4-12b-it-qat-q4_0.gguf`. Downloaded, sha256 `93567e57a8fe10b23569b9d9ec38cd005deedf71e29477c421a4b83f418a538b` verified, 6,975,879,296 bytes. Deleted after this verdict.

### Gate 1 — Identity

**Manifest:** `GEMMA4-DENSE / 48L / Q4_0-QAT / gemma-canonical-chat`

```
architecture: gemma4   block_count: 48   embedding_length: 3840   feed_forward_length: 15360
(no expert_count key — genuinely dense, 0 expert tensors)
tensor histogram (667 tensors): F32 x338, Q6_K x1 (token_embd), Q4_0 x328
```

### Gate 2 — Admission: PASS

### Gate 3 — Tokenizer: **0/721 diverge** vs `google/gemma-4-12B-it` — clean

### Gate 4 — Reference: **greedy token identity, 4/6 exact** (dense arch, not KLD)

Six-prompt protocol (the afmoe cert session's gate 3 shape: four 64-token short-domain prompts + two of them repeated at 256), reference queried with explicit pure-greedy params and `cache_prompt:false`:

| case | n | identical |
|---|---:|---|
| a: "The capital of France is" | 64 | **yes** |
| b: linked-list reversal | 64 | **yes** |
| c: Apollo 11 summary | 64 | **yes** |
| d: Swedish thermometer | 64 | **yes** |
| b-long | 256 | no (diverges byte 191) |
| c-long | 256 | no (diverges byte 231) |

**4/6 exact — every short prompt passes.** Both misses are 256-token runs where *both engines* degenerate into a repetitive loop (`"1.\n1.\n1.\n..."` and `"111111...111"` respectively) — a 12B QAT model with no repetition penalty at greedy temp=0 is a textbook case for this, and the divergence point in each case is exactly where the loop's period desynchronizes between the two engines' rounding, not a coherent-text disagreement. Confirmed the runner's own output is deterministic/stable across repeated runs at 256 tokens (ruling out flakiness) before recording this. Evidence: `docs/cert-matrix-evidence/t1.6-greedy-identity.json`.

### Gate 5 — cpu_cuda: **PASS** — byte-identical (64 tokens, full 48/48 offload)

### Gate 6 — Chat smoke: **PASS**

`"What is 2+2? Answer briefly."` -> `"4"`, `finish_reason: "stop"`.

### Gate 7 — Perf row

```
CPU: {"gpu_layers":0,"layers":48,"prompt_tok_s":35.868,"gen_tok_s":8.766,"prompt_s":14.274,"gen_s":29.203}
GPU: {"gpu_layers":48,"layers":48,"prompt_tok_s":5.540,"gen_tok_s":8.330,"prompt_s":92.421,"gen_s":30.731}
```

GPU prompt throughput here is oddly *lower* than CPU's (5.5 vs 35.9 tok/s) — recorded as measured, not investigated further (out of scope for a perf-row record; possibly this box's MIG slice under different load at measurement time, or a placement/scheduling quirk specific to this model's shape).

### Summary

This is the strongest identity result of the session: a real architecture (dense gemma4, QAT) with 4/6 exact greedy matches against an independent engine, clean tokenizer, and byte-identical CPU/CUDA. The two long-run misses are the kind of divergence the goal doc's own "known acceptable exception" language anticipates in spirit (two engines' rounding disagreeing inside a degenerate repeat loop is a different animal from disagreeing about real content) — recorded honestly as misses rather than waived, but clearly distinguished from a real behavioral gap.

## 7. Google gemma-4-E2B-it QAT Q4_0

**Verdict: REFUSED** — clean, specific, correctly-triggered refusal. Per the goal doc, this is a complete result, not a setback.

**Resolved:** `google/gemma-4-E2B-it-qat-q4_0-gguf/gemma-4-E2B_q4_0-it.gguf`. Downloaded, sha256 `fa401b55b07ee70a54c6dae3903c783a6e65064312529ea57175cb5f8dec6634` verified, 3,349,516,256 bytes. Deleted after this verdict.

### Gate 1 — Identity

```
architecture: gemma4   block_count: 35   embedding_length: 1536   context_length: 131072
gemma4.feed_forward_length: [6144 x15, 12288 x20]   <- an ARRAY of 35 per-layer values, not a scalar
tensor histogram (541 tensors): F32 x263, F16 x1, Q6_K x2, Q4_0 x275
(no expert_count key — dense, E-series per-layer-embedding architecture)
```

The interesting fact this gate surfaces: **this file's FFN width genuinely varies per layer** (15 layers at 6144, 20 at 12288) and Google's own conversion publishes that as a GGUF array-typed KV rather than the scalar `feed_forward_length` every other file in this session has. That is not a malformed file — it is an accurate encoding of a real heterogeneous architecture — but it is a format the runner's loader does not read.

### Gate 2 — Admission: **REFUSED**

```
gemma4: E-series (per-layer embeddings + shared KV) — verified against llama.cpp at the Q4_K noise floor rather than token-identically
error: missing model hyperparameters for arch 'gemma4'
```

The runner correctly *recognizes* this as a gemma4 E-series file (the diagnostic line fires first) but then refuses to load it. Root cause identified by direct inspection (read-only — no code changed, per the STOP rule): `src/model.c:1346-1348` requires `n_ff > 0` after populating hyperparameters from `gguf_get_u32(g, "gemma4.feed_forward_length", ...)`; `src/gguf.c:280-299`'s `gguf_get_u32` has no branch for `GGUF_T_ARR` (array) KVs — every branch tests unsigned/signed/float scalar storage — so it silently returns the caller's default (0) when the key holds an array, which then trips the hyperparameter-completeness check and refuses the load. **This is exactly the "clean refusal with a correct reason" the goal doc asks for**: the runner does not silently misread the array as garbage or crash; it notices the resulting hyperparameter is missing and stops. The specific gap is that this loader path was written assuming `feed_forward_length` is always a scalar, which held for every other file in this session (including the 26B-A4B/12B/31B gemma4 QAT releases, all uniform per-layer) but not for this real, heterogeneous E2B release.

Tokenizer vocab was still checked independently (it does not require the hyperparameter path): **0/721 diverge** vs `google/gemma-4-E2B-it` — clean. `docs/cert-matrix-evidence/t1.7-difftok.log`.

Gates 3 (partially, see above) through 7 do not apply — a refused load ends the battery.

### Summary

A specific, well-characterized REFUSED verdict: the file is a legitimate, differently-shaped release (real per-layer FFN width variation, not a corrupt or unusual quant), and the runner's gemma4 loader has a real, narrow gap — no array-typed-KV handling for `feed_forward_length` — that a future engine session could close in one place (`gguf_get_u32`'s array branch, or a per-layer-array reader alongside the existing per-layer-override fields like `l_head_kv`/`l_head_dim` the struct already has for heterogeneous archs). No code was touched to test or work around this, per the STOP rule.

## 8. Google gemma-4-31B-it QAT Q4_0 (flagship dense)

**Verdict: CERTIFIED-WITH-CAVEAT** — the best identity result of the session: **5/6 exact greedy matches**, including both 256-token long runs this time (contrast item 6's dense 12B, which degenerated at 256 tokens on both engines). Only VRAM pressure on this box's 24 GB MIG slice — not correctness — is the real caveat here.

**Resolved:** `google/gemma-4-31B-it-qat-q4_0-gguf/gemma-4-31B_q4_0-it.gguf`. Downloaded, sha256 `179cfb99212709597eae5929112cfca677e1bbf566178b479ae1da0c4772874b` verified, 17,651,001,568 bytes. Deleted after this verdict.

### Gate 1 — Identity

**Manifest:** `GEMMA4-DENSE / 60L / Q4_0-QAT / gemma-canonical-chat`

```
architecture: gemma4   block_count: 60   embedding_length: 5376   feed_forward_length: 21504 (scalar — unlike item 7)
tensor histogram (833 tensors): F32 x422, Q6_K x1, Q4_0 x410
```

### Gate 2 — Admission: PASS

### Gate 3 — Tokenizer: **0/721 diverge** vs `google/gemma-4-31B-it` — clean

### Gate 4 — Reference: **5/6 exact greedy identity**

| case | n | identical |
|---|---:|---|
| a: "The capital of France is" | 64 | **yes** |
| b: linked-list reversal | 64 | **yes** |
| c: Apollo 11 summary | 64 | **yes** |
| d: Swedish thermometer | 64 | no (diverges byte 68) |
| b-long | 256 | **yes** |
| c-long | 256 | **yes** |

The one miss (prompt d, Swedish) is not degenerate text on either side — both continuations are fluent, on-topic Swedish about thermal expansion, just phrased differently from that point on (`"...kallas **termisk expansi[on]**"` vs `"...**termisk expansion**, vilket innebär..."`), consistent with a genuine near-tie argmax disagreement rather than an architecture bug. Confirmed reproducible/stable on the runner side across repeated runs before recording. Both 256-token runs — the length class that broke item 6's 12B model into a repeat loop — are exact here, for whatever it is worth about scale/stability at this size. Evidence: `docs/cert-matrix-evidence/t1.8-greedy-identity.json`.

### Gate 5 — cpu_cuda: **PASS** — byte-identical (64 tokens, full 60/60 offload)

### Gate 6 — Chat smoke: **PASS** — `"4"`, `finish_reason: "stop"`

### Gate 7 — Perf row: **the real caveat is here, not correctness**

```
CPU: {"gpu_layers":0,"layers":60,"prompt_tok_s":15.819,"gen_tok_s":4.829,"prompt_s":32.365,"gen_s":53.016}
GPU: {"gpu_layers":60,"layers":60,"prompt_tok_s":2.096,"gen_tok_s":3.250,"prompt_s":244.227,"gen_s":78.778}
```

**GPU is slower than CPU on every axis** — prompt throughput drops nearly 8x (15.8 -> 2.1 tok/s). This model's 17.7 GB of weights leaves only ~7.6 GB of this box's 24 GB MIG slice for KV cache and scratch; the load log showed `kv 3.69 GB` against `3.75 GB free` after init — the device is nearly full. This reads as VRAM-pressure-induced slowdown (some combination of reduced batching/parallelism headroom or spillover) specific to this box's small MIG slice, not a correctness defect — cpu_cuda byte-identity above confirms the GPU path computes the *same* answer, just much more slowly under this constraint. Recorded as measured; not investigated further (out of scope for a perf-row record, and doing so would risk turning into engine debugging under the STOP rule).

### Summary

The identity gates are unambiguously the best of the session (5/6, both long runs exact) for what is also the largest dense model tested. The perf row is the interesting anti-correlated result: bigger and more correct here does not mean faster on this specific, VRAM-constrained GPU slice — a genuinely useful data point for anyone sizing which of these artifacts to run on a small GPU versus CPU.

## 9. GPT-OSS Nano 9B (squ11z1, community ~12-expert prune)

**Verdict: FAILED** — the "expert roster differs from family" admission test the roster wanted: admission itself is a clean, unqualified PASS (the runner's per-layer expert-count handling works correctly on a real third-party pruned file, not just ones pruned by this project's own `--prune-experts`), but the same tokenizer/chat-smoke issues from every gpt-oss row persist, and gate 4 only half-clears its bar.

**Resolved:** `squ11z1/gpt-oss-nano/gpt-oss-9b-q4_k_m.gguf` (base model gpt-oss-20b, per the repo's own `base_model` tag). Downloaded, sha256 `794da0a902b161bc1ba0eb4a7f0e4e5ef804f5ee93ec7f01ba8927b755a278fa` verified, 6,825,064,608 bytes. Deleted after this verdict.

### Gate 1 — Identity: **confirmed 12-expert prune, and a full requant — not a metadata trick**

**Manifest:** `GPTOSS / 24L / 12E / top4 / MIXED-REQUANT(Q8_0/Q5_0, non-uniform per-layer) / Harmony`

```
gpt-oss.expert_count: 12   (was 32 in every Tier-1 gpt-oss file)   gpt-oss.expert_used_count: 4 (unchanged)
tensor histogram (459 tensors -- SAME count as the 32-expert files): Q8_0 x25, F32 x289, Q5_0 x121, Q4_K x24
expert-tensor types (192 tensors): F32 x120, Q8_0 x12, Q5_0 x60   <- NOT MXFP4 at all; fully requantized
non-uniform expert dtype across layers: confirmed
```

This is the mirror image of items 1-3's finding: `--prune-experts`-style pruning shrinks each expert tensor's 3rd dimension from 32 to 12 (same 459-tensor file structure, smaller `ne[2]`) *and* the whole file was requantized off MXFP4 onto a mixed Q8_0/Q5_0 scheme in the process — a real dequant-prune-requant pipeline, not a lightweight metadata edit.

### Gate 2 — Admission: **PASS** — the interesting result this row exists to produce

Loads cleanly on both CPU and CUDA with no complaint about the non-standard expert count. This is a genuine, valuable cross-check: the runner's per-layer `n_expert` handling (read from each layer's own router tensor, independent of a model-wide constant — the same mechanism this project's own `--prune-experts` pruning depends on) works correctly on a file pruned by an entirely different, third-party pipeline.

### Gate 3 — Tokenizer: **222/721 diverge (30.8%)** — identical count to every other gpt-oss row

Confirms (again) this is a runner-side vocab/BPE characteristic independent of expert count or quantization.

### Gate 4 — Reference (KLD): **mixed — KLD bar clears, top-1 bar does not**

400 positions: `mean_kld: 0.0402, top1_agreement_pct: 88.0, mean_top8_overlap: 0.901`.

**This is the best KLD result of any gpt-oss file this session** (previous best: item 1's 0.1276), and mean KLD **clears** the session's own 0.05 bar — only top-1 (88.0% vs. 97%) misses. Plausible reading, not confirmed: a 12-expert pool has fewer near-tie routing decisions than a 32-expert pool, so the already-diagnosed MXFP4-vs-fp32 activation-quantization mismatch (see item 1) has fewer opportunities to flip a selection — consistent with, not contradicting, that root cause. Evidence: `docs/cert-matrix-evidence/t2.9-kld-raw.json`.

### Gate 5 — cpu_cuda: **PASS** — byte-identical, unlike every native-MXFP4 gpt-oss file

`--gpu off` vs `--gpu auto` (full 24/24 offload) match exactly at 64 tokens. Every Tier-1 gpt-oss row (native MXFP4 experts) failed this gate on this box; this fully-requantized-to-Q8_0/Q5_0 file passes it. Consistent with the item 1 diagnosis being specifically about the MXFP4 kernel path — a file with no MXFP4 tensors left has nothing to disagree about there.

### Gate 6 — Chat smoke: **FAIL** — same shape as items 1/2

```
'"\n\nWe need to parse the instruction: "What is 2+2? Answer briefly." ... So the answer: 2+2 is 4. The user wants a brief answer. So we should answer: '
```

`finish_reason: "length"` — runs to the token cap without emitting a clean final answer, though the analysis text shows it "knows" the answer internally ("2+2 is 4"). Same Harmony-rendering characteristic as the base model's own conversions, inherited through the prune. `docs/cert-matrix-evidence/t2.9-chat-smoke.json`.

### Gate 7 — Perf row

```
CPU: {"gpu_layers":0,"layers":24,"prompt_tok_s":68.784,"gen_tok_s":9.941,"prompt_s":7.444,"gen_s":25.752}
GPU: {"gpu_layers":24,"layers":24,"prompt_tok_s":35.883,"gen_tok_s":31.762,"prompt_s":14.269,"gen_s":8.060}
```

### Summary

The admission test the roster designed this row to run — does the runner's per-layer expert-count machinery handle a real third-party expert prune, not just this project's own — is an unqualified yes. That is genuinely valuable evidence for the pruning infrastructure generally. The overall verdict is still FAILED because gate 6 fails and gate 4 only half-clears, both consistent with characteristics already seen across the gpt-oss family rather than new problems specific to pruning.
