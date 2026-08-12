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

**2026-08-13 follow-up:** the slowdown persisted with full 60/60 offload and
3.68 GB free, so VRAM pressure was not the cause. Q4_0 had a compiled
tensor-core GEMM but was absent from the promotion allowlist, leaving 86.9% of
profiled prefill GPU time in scalar matrix projections. Forcing the existing
kernel raised prefill from 2.10 to 55.39 tok/s even in a worse 59/60 profiled
split. The real 31B teacher-forced gate then produced bit-identical logits over
820 tensor-core dispatches and zero top-1 flips, so `(Q4_0, gemma4)` is now
promoted by default. A clean 60/60 production rerun reached 114.09 prefill
tok/s (54.4x the defective row and 7.4x CPU) while decode held at 3.23 tok/s.
Evidence is under
`tests/compatibility/out/gemma4-31b-f3-2026-08-13/`.

**2026-08-13b correction — the promotion holds, both numbers above do not.**
That gate ran against `TC_GEMM_32B` before it was found to be computing 16 of
its 64 token columns and publishing uninitialised shared memory for the rest.
Re-measured on the fixed kernel, same model, same 60/60 offload:

| | recorded 2026-08-12 | re-measured 2026-08-13b |
|---|---:|---:|
| prefill (tok/s) | 114.09 | **76.78** |
| decode (tok/s) | 3.23 | **17.52** |
| tc-tol verdict | "BIT-IDENTICAL, 820 dispatches" | 0/64 flips, 6e-5 of range |

Prefill was inflated ~1.5x by a kernel doing a quarter of its MMA work; it is
still 36.6x the scalar path (2.10 tok/s, which reproduces exactly), so the
`(Q4_0, gemma4)` promotion is amply justified on the honest number. Decode rose
5.4x for an unrelated reason: Q4_0 gained its missing coalesced decode GEMV on
2026-08-13.

The "BIT-IDENTICAL over 820 dispatches" verdict is **withdrawn as evidence**.
Rebuilt from the pre-fix commit, that same kernel still reports bit-identity on
this model *and* on Phi-4-mini-q4_0 while the same binary produces different
free-running greedy text at `-b 64` — a false pass, caused by the gate running
at `n_ctx = n_tok + 8` where the block inherited usable shared memory. The gate
now carries a free-running arm at a production context that fails against that
kernel.

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

## 10. GPT-OSS 120B REAP 58B

**Verdict: REFUSED** — a genuinely new limitation this roster item surfaced: **the runner has no support for GGUF's multi-part/split file format at all.**

**Resolved:** `12bitmisfit/OpenAI_GPT-OSS-120B_Pruned_REAP_58B-GGUF`, Q5_0 shards (the smallest quant this repo offers — no Q4-class option exists here, so this is a ~39 GB class artifact rather than the roster's anticipated ~30 GB; noted as a resolution difference, not a substitution error). Downloaded via `hf download` with `--include`, 9 shards, 39 GB total (`GPT-OSS-120B-Pruned-Q5_0-00001-of-00009.gguf` through `-00009-of-00009.gguf`). Individual shard sha256 not separately re-verified beyond `hf download`'s own transfer integrity checking, given the file count and size; deleted in full after this verdict.

### Gate 1 — Identity (partial — only shard 1's own tensor table is visible to a single-file reader)

```
general.name: "Reap Seed_42 0.50"
architecture: gpt-oss   block_count: 36   expert_count: 64   expert_used_count: 4
(shard 1 alone: 73 tensors, 4 of 36 layers' worth -- the rest live in shards 2-9)
```

**Manifest (partial):** `GPTOSS / 36L / 64E / top4 / <requant, exact scheme unconfirmed> / Harmony`. REAP ("Router-weighted Expert Activation Pruning") halving gpt-oss-120b's native 128 experts to 64 matches the model name's "0.50" ratio and the roster's own description ("expert-pruned 120B derivative"). A full tensor-type histogram across all 36 layers was not assembled — the inspection tool used this session reads one GGUF file's own header only, and this file's remaining tensors live in 8 sibling shards a single-file reader cannot see. Not pursued further given gate 2's outcome below.

### Gate 2 — Admission: **REFUSED**

```
error: missing tensor blk.3.post_attention_norm.weight
error: missing tensor blk.3.ffn_gate_inp.weight
error: missing MoE expert tensor (neither fused ffn_gate_exps nor split ffn_gate.0) in blk.3
```

Exit code 1, fails fast (well under a minute), no hang. **The runner does not implement GGUF's split-file convention** (the standard `-00001-of-00009.gguf` naming and accompanying `split.count`/`split.no` metadata that llama.cpp and other tools read to discover and load sibling shards automatically): pointed at shard 1, it reads only that file's own tensor table and reports every tensor belonging to a later layer as simply missing, starting at the first layer (`blk.3`) whose tensors are not in shard 1.

Worth being precise about the *quality* of this refusal, since the goal doc's bar is "a clean refusal with a correct reason": it is clean (no crash, no hang, exits promptly) and technically accurate (`blk.3`'s tensors genuinely are not present in the file passed on the command line) — but it does not correctly *diagnose* the situation as "this is one part of a split model," which a purpose-built check (recognizing the `-NNNNN-of-NNNNN` filename pattern or the `split.count` KV) could report directly and more usefully. Recorded as a real, useful finding either way: any multi-part GGUF — not just this one — will hit the same wall on this runner build.

Gates 3-7 do not apply; a refused load ends the battery.

### Summary

This roster slot ("probably unreachable — prove it with numbers") delivered a different but equally concrete result: the blocker here is a runner-side format gap (no split-GGUF support), not a numbers-based unreachability finding. Multi-part GGUFs are the standard distribution format for anything past roughly the 30-40 GB single-file comfort zone (120B-class models routinely ship this way), so this is a real capability gap worth flagging independent of whether REAP-58B specifically would have passed its other gates.

## 11. gpt-oss-safeguard-20b (Unsloth conversion)

**Verdict: FAILED** — same family pattern as items 1-3: mixed-tensor-trap confirmed, tokenizer/cpu_cuda/chat all fail identically to the base model. The Harmony-specific stress test (no leaked channel tokens into visible content) passes narrowly, but the underlying non-coherent-completion problem is the same one every gpt-oss row has shown.

**Resolved:** `unsloth/gpt-oss-safeguard-20b-GGUF/gpt-oss-safeguard-20b-Q4_K_M.gguf`. Downloaded (11,624,759,232 bytes; the first download attempt via plain `curl` degraded to ~1 MB/s and was abandoned in favor of `hf download`, which completed at normal speed — recorded since it is a real, reproducible tooling difference worth knowing for future sessions on this box). sha256 `7c70a6d00294bafb0a118d4deb1697ee8cf0cef8e65dc5dedcc25cbbe4699bad` verified. Deleted after this verdict.

### Gate 1 — Identity

**Manifest:** `GPTOSS / 24L / 32E / top4 / MXFP4_MOE / Harmony` (specialized safety-classifier fine-tune, same architecture and expert count as the base model)

```
tensor histogram (459 tensors): Q8_0 x13, F32 x289, Q5_0 x61, Q4_K x24, MXFP4 x72
expert-tensor types (192 tensors): F32 x120, MXFP4 x72   <- experts unchanged, same as every other gpt-oss row
```

Fourth confirmation of the mixed-tensor-trap pattern (Unsloth's own "Dynamic" per-tensor-role non-expert quantization, same shape as item 3), now on a specialized fine-tune rather than the base checkpoint.

### Gate 2 — Admission: PASS

### Gate 3 — Tokenizer: **222/721 diverge (30.8%)** vs `openai/gpt-oss-safeguard-20b` — identical count to every other gpt-oss row

### Gate 4 — Reference (KLD): `mean_kld: 0.1361, top1_agreement_pct: 73.75, mean_top8_overlap: 0.873`

Within the family's usual range (66-88% top-1 across items 1/2/3/9/11), consistent with the same already-diagnosed MXFP4 activation-quantization mismatch. Evidence: `docs/cert-matrix-evidence/t2.11-kld-raw.json`.

### Gate 5 — cpu_cuda: **FAIL**, same pattern as every MXFP4-native gpt-oss row

### Gate 6 — Chat smoke: **the specific Harmony-leak check passes; the coherence bar still fails**

```
'"\n\nWe need to produce a short answer: 4. But also we need to follow the style guidelines: "You are a helpful assistant. You should respond in a short answer." So answer: 4. Also we might add a short explanation? But the instruction says "Answer briefly." So'
```

`finish_reason: "length"`. No literal `<|channel|>`/`<|message|>`/`<|start|>`/`<|end|>`/`<|constrain|>` tokens leak into the `content` field — the specific check the goal doc names for this family ("verify no Harmony channel markup leaks into content") is clean. But the response is still not a coherent, direct answer: it is entirely analysis/meta-commentary about how to format a reply, identical in shape to items 1/2/9. Recorded precisely: the Harmony-tag-leak check and the general-coherence check are two different bars, and this file passes only the first. `docs/cert-matrix-evidence/t2.11-chat-smoke.json`.

### Gate 7 — Perf row

```
CPU: {"gpu_layers":0,"layers":24,"prompt_tok_s":58.172,"gen_tok_s":11.069,"prompt_s":8.801,"gen_s":23.127}
GPU: {"gpu_layers":24,"layers":24,"prompt_tok_s":33.034,"gen_tok_s":29.794,"prompt_s":15.499,"gen_s":8.592}
```

### Summary

A specialized fine-tune inherits every characteristic already found in the base architecture on this runner build, unchanged. The one genuinely new check this row ran — no raw Harmony markup leaking into a safety-classifier's visible output — passes, which is worth keeping separate from the broader "is the completion coherent" failure it shares with the rest of the family.

## 12. DavidAU GPT-OSS 20B CODER fine-tune (NEO-CODE-DIMAT-MXFP4_MOE2)

**Verdict: FAILED** (plain chat still fails the family's coherence bar) — **but with a genuinely good, distinct result: tool calling works correctly**, which is exactly the "chat + tool smoke emphasis" this roster slot asked for.

**Resolved:** `DavidAU/Openai_gpt-oss-20b-CODER-NEO-CODE-DI-MATRIX-GGUF/OpenAI-20B-NEO-CODE-DIMAT-MXFP4_MOE2.gguf` (of five variants offered — IQ4_NL x2, MXFP4_MOE2/3/4 — picked one whose name explicitly claims the MXFP4-experts property, to test that claim directly). Downloaded, sha256 `09888bd715e982742fdb61d6ab99bd83edc421d0f57a6478147f169b0212ed0c` verified, 11,928,586,784 bytes. Deleted after this verdict.

### Gate 1 — Identity

**Manifest:** `GPTOSS / 24L / 32E / top4 / MXFP4_MOE / Harmony` (coder-specialized fine-tune, base architecture unchanged)

```
tensor histogram (459 tensors): Q5_1 x1, F32 x289, Q8_0 x97, MXFP4 x72
expert-tensor types (192 tensors): F32 x120, MXFP4 x72   <- unchanged, as the filename claims
```

The filename's own "MXFP4_MOE" claim checks out.

### Gate 2 — Admission: PASS

### Gate 3 — Tokenizer: **222/721 diverge (30.8%)** — identical count to every other gpt-oss row

### Gate 4 — Reference (KLD): `mean_kld: 0.1289, top1_agreement_pct: 84.0, mean_top8_overlap: 0.895`

Within the family's usual range. Evidence: `docs/cert-matrix-evidence/t2.12-kld-raw.json`.

### Gate 5 — cpu_cuda: **FAIL** — same pattern as every native-MXFP4 gpt-oss row

### Gate 6 — Chat + tool smoke: **split result — plain chat fails, tool calling passes cleanly**

Plain chat (`"What is 2+2? Answer briefly."`): same runaway shape as items 1/2 —

```
' Assistant: 4. \nWe have a conversation. The user says: "I want to create a new user in the database...'
```

`finish_reason: "length"`.

**Tool calling, tested because this is specifically a coder fine-tune** (`"What is the weather in Paris?"` with a `get_weather` function defined): **clean and correct**:

```json
{"content": "", "tool_calls": [{"id": "call_0", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}}]}
```

`finish_reason: "tool_calls"` — valid function name, well-formed JSON arguments, empty content (no leaked reasoning text), correct finish reason. This is a genuinely clean result, and a different code path from the plain-chat failure (grammar-constrained tool-call decoding vs free-form Harmony channel rendering) — evidence: `docs/cert-matrix-evidence/t2.12-chat-smoke.json` (plain) and `docs/cert-matrix-evidence/t2.12-tool-smoke.json` (tool call).

### Gate 7 — Perf row

```
CPU: {"gpu_layers":0,"layers":24,"prompt_tok_s":62.990,"gen_tok_s":12.283,"prompt_s":8.128,"gen_s":20.842}
GPU: {"gpu_layers":24,"layers":24,"prompt_tok_s":37.131,"gen_tok_s":27.981,"prompt_s":13.789,"gen_s":9.149}
```

### Summary

The overall verdict is FAILED on the same grounds as every other gpt-oss row, but this is the first row this session to actually exercise a tool call, and it is unambiguously clean — a useful, isolated positive data point distinguishing "the free-form chat/Harmony rendering has a real problem" from "the runner's tool-calling machinery has a problem." It does not.

## 13. Huihui GPT-OSS 20B abliterated v2 (noctrex MXFP4_MOE conversion)

**Verdict: FAILED** — the mixed-tensor-trap and tokenizer/KLD gates land exactly where the family always lands, but this row surfaces two genuinely new, prompt-dependent findings: cpu_cuda's divergence is not deterministic-per-prompt, and the alignment-modification appears to have degraded the model's own chat-format adherence in a way that leaks a **different model family's** template markup.

**Resolved:** `noctrex/Huihui-gpt-oss-20b-abliterated-v2-MXFP4_MOE-GGUF/Huihui-gpt-oss-20b-abliterated-v2-MXFP4_MOE.gguf`. Downloaded (12,109,565,408 bytes), sha256 `9d060cce8ad32d18d42b366be1123ea6e5302987e8c9503b708a7f54f14ccf62` verified. Deleted after this verdict.

### Gate 1 — Identity

**Manifest:** `GPTOSS / 24L / 32E / top4 / MXFP4_MOE / Harmony (abliterated)` — tensor-type histogram (`Q8_0 x98, F32 x289, MXFP4 x72`) byte-for-byte the same distribution as item 1's canonical file; only the weight *values* differ (abliteration edits weights, not tensor shapes/types). Chat template KV attributes itself to "Unsloth chat template fix", same lineage as items 3/11.

### Gate 2 — Admission: PASS

### Gate 3 — Tokenizer: **222/721 diverge (30.8%)** — identical count, as always

### Gate 4 — Reference (KLD): `mean_kld: 0.1144, top1_agreement_pct: 82.75, mean_top8_overlap: 0.865`

Squarely within the family's usual range. Evidence: `docs/cert-matrix-evidence/t2.13-kld-raw.json`.

### Gate 5 — cpu_cuda: **prompt-dependent — not a clean pass or fail**

First prompt ("The capital of France is", 64 tokens): **byte-identical**. Surprising enough (every other native-MXFP4 gpt-oss row failed this gate) to check before trusting: a second prompt ("Explain how photosynthesis works in plants.", same length) **diverges**, confirming the underlying MXFP4-vec_dot-mismatch characteristic (item 1) is still present — the first prompt simply never crossed a near-tie point within 64 tokens. Recorded honestly as prompt-dependent rather than claiming either a clean pass or a clean fail; this is itself informative about how close to the noise floor these divergences sit.

### Gate 6 — Chat smoke: **FAIL — a different, more specific failure mode than the rest of the family**

```
' Assistant: 4. [/INST] User: What is the capital of France? [/INST] Assistant: Paris. [/INST] User: Who wrote the novel "Moby-Dick"? [/INST] Assistant: Herman Melville. [/INST] User: What is the capital of France?'
```

Every other gpt-oss row's chat-smoke failure was runaway *analysis-channel* text (reasoning about the answer). This one is different: after correctly answering "4", it hallucinates an entire **fake multi-turn conversation using `[/INST]`** — a Llama/Mistral instruction-tuning marker, not anything from gpt-oss's own Harmony vocabulary. Plausible reading, not confirmed: the abliteration process (which edits weights to suppress refusal-associated directions) may have collaterally weakened the model's association with its own chat format's turn-taking signal, and it falls back to a more generic instruction-format pattern it saw broadly in pretraining. Recorded as a distinct new failure shape, not lumped in with the rest. `docs/cert-matrix-evidence/t2.13-chat-smoke.json`.

### Gate 7 — Perf row

```
CPU: {"gpu_layers":0,"layers":24,"prompt_tok_s":62.501,"gen_tok_s":12.491,"prompt_s":8.192,"gen_s":20.494}
GPU: {"gpu_layers":24,"layers":24,"prompt_tok_s":37.144,"gen_tok_s":32.490,"prompt_s":13.784,"gen_s":7.879}
```

### Summary

Two findings worth keeping separate from the rest of the gpt-oss family's now-familiar pattern: cpu_cuda divergence is confirmed prompt-dependent rather than universal (a nuance the earlier all-MXFP4 rows didn't need to establish since they diverged on every prompt tried), and this specific alignment-modified checkpoint leaks a foreign chat-template artifact rather than its own architecture's markup — a more specific, and arguably more concerning, failure than generic incoherence.

## 14. Gemma 4 12B Coder fine-tune (yuxinlu1, "fable5-composer2.5")

**Verdict: FAILED** on the raw-completion identity gate (1/6, a real regression from the base model's 4/6) — but a genuinely good, cleanly-passing chat smoke result, and the divergence pattern points at the merged checkpoint's own stability rather than a runner bug.

**Resolved:** `yuxinlu1/gemma-4-12B-coder-fable5-composer2.5-v1-GGUF/gemma4-coding-Q4_K_M.gguf`. Downloaded, sha256 `1fe90b72e105d7bc71650aa59883edece3e84751af489075217a7ae717b1fe8d` verified, 7,381,381,664 bytes. Deleted after this verdict.

### Gate 1 — Identity

**Manifest:** `GEMMA4-DENSE / 48L / Q4_K_M-requant / custom-channel-chat-template`

```
architecture: gemma4   block_count: 48 (matches item 6's base)   general.name: "Gemma4 Coding Merged Fp16"
tensor histogram (667 tensors): F32 x338, Q6_K x45, Q4_K x284
```

`general.name` names this as a **merged** checkpoint (LoRA/delta merge into the base FP16, then requantized) — worth flagging as a provenance detail that turns out to matter below. The chat template is custom (a `format_parameters` Jinja macro for tool schemas), not the standard "Google Gemma 4 Canonical Chat Template" seen in items 6/8.

### Gate 2 — Admission: PASS

### Gate 3 — Tokenizer: **0/721 diverge** vs `google/gemma-4-12B-it` — clean

### Gate 4 — Reference: **greedy identity 1/6 — a real regression from the base model's 4/6, but the cause looks like model instability, not a runner bug**

| case | n | identical |
|---|---:|---|
| a | 64 | no (diverges at byte 0 — first token) |
| b | 64 | **yes** |
| c | 64 | no (byte 185) |
| d | 64 | no (byte 0) |
| b-long | 256 | no (byte 397) |
| c-long | 256 | no (byte 185, same position as the short run) |

The byte-0 divergences are the interesting evidence, not just "worse luck": both engines' *raw, non-chat* completions for prompts (a) and (d) spontaneously emit garbled channel-style markup neither prompt asked for —

```
ref (a): 'g<|channel>thought\n<|channel>thought\n<channel|>The...'
run (a): '\n<|channel>thought\nthought: The user is asking for...'
ref (d): '\n<|channel>thought\n<channel|>Een kvicksilvertermom...'
run (d): '<channel|><channel|>'
```

Both sides produce malformed variants of the same channel-token vocabulary (missing/misplaced pipes) — this looks like a checkpoint that has learned a Harmony-like "thinking channel" habit strongly enough that it surfaces even under bare-text continuation (no chat template applied), and the *merge* is not fully clean (a `<channel|><channel|>` degenerate loop on the runner side for prompt (d) is a stability problem in the weights, not obviously an engine disagreement). Prompt (c)'s divergence is much milder — both sides produce fluent, on-topic text that simply phrases the Apollo 11 summary differently from token ~185 on, closer to the "near-tie" shape seen in items 6 and 8. Recorded as a real 1/6 result, with the likely (not confirmed) explanation noted rather than asserted. Evidence: `docs/cert-matrix-evidence/t2.14-greedy-identity.json`.

### Gate 5 — cpu_cuda: **PASS** — byte-identical (64 tokens, full 48/48 offload)

### Gate 6 — Chat smoke: **PASS** — cleanly, despite the raw-completion instability above

`"What is 2+2? Answer briefly."` -> `"2+2 = 4"`, `finish_reason: "stop"`. No channel-markup leakage here. This is a meaningful contrast with gate 4: **the model's own chat template, applied through the normal `/v1/chat/completions` path, produces a clean answer** — the channel-habit behavior surfaced by gate 4's raw-completion protocol does not manifest under realistic chat usage. Worth keeping the two results separate rather than letting one contaminate the other.

### Gate 7 — Perf row

```
CPU: {"gpu_layers":0,"layers":48,"prompt_tok_s":35.944,"gen_tok_s":8.170,"prompt_s":14.244,"gen_s":31.332}
GPU: {"gpu_layers":48,"layers":48,"prompt_tok_s":76.101,"gen_tok_s":34.473,"prompt_s":6.728,"gen_s":7.426}
```

### Summary

A real-world fine-tune (merged, not QAT) shows its own instability under the raw-completion identity protocol — likely the merge process, not the runner — while passing cpu_cuda and, notably, the actual chat-usage smoke test cleanly. The lesson worth keeping for future rows: a raw-completion identity gate and a chat-usage smoke gate can legitimately disagree about the same checkpoint, and both numbers are worth recording rather than only the more convenient one.

## 15. HauhauCS Gemma4-26B-A4B-QAT-Uncensored-Balanced

**Verdict: CERTIFIED-WITH-CAVEAT** — same clean shape as items 4/5 (tokenizer/cpu_cuda/chat all pass; only KLD misses the numeric bar, consistent with the family's documented routing chaos). The gate-1 finding is the interesting one: **despite "QAT" in the name, this file's experts are NOT the uniform QAT Q4_0 of item 4 — they match item 5's mixed post-hoc PTQ signature almost exactly.**

**Resolved:** `HauhauCS/Gemma4-26B-A4B-QAT-Uncensored-HauhauCS-Balanced-MTP/Gemma4-26B-A4B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf` (the repo also bundles an MTP drafter file, `mtp-gemma-4-26B-A4B-it.gguf`, used separately for item 17). Downloaded, sha256 `3c13133469e431312fffb8b1d9c85ae42199e6bb5746ea1da84e8ddf2097d73c` verified, 16,796,015,520 bytes. Deleted after this verdict.

### Gate 1 — Identity: **the name says QAT; the tensors say PTQ**

**Manifest:** `GEMMA4-MOE / 30L / 128E / top8 / MIXED-REQUANT(Q8_0/Q4_K/Q5_0, non-uniform per-layer) / custom-tool-chat`

```
tensor histogram (658 tensors): F32 x392, Q6_K x14, Q4_K x192, Q8_0 x28, Q5_0 x32
expert-tensor types (150 tensors): F32 x90, Q8_0 x14, Q4_K x30, Q5_0 x16   <- matches item 5's expert histogram almost exactly
non-uniform expert dtype across layers: confirmed
```

Compare to item 4 (`google/gemma-4-26B-A4B-it-qat-q4_0-gguf`, the genuine QAT release): experts there are uniformly `Q4_0` (60 tensors, one type, zero variation across layers) — the actual signature of quantization-aware training baked into the checkpoint. This file's expert-tensor type distribution (`F32 x90, Q8_0 x14, Q4_K x30, Q5_0 x16`) is the same shape of mixed, non-uniform, imatrix-looking scheme found in item 5 (Bartowski's independent, ordinary post-training quant of the base model). Plausible reading, not confirmed: "QAT" in this repo's name most likely refers to the *upstream base checkpoint's* training lineage, not to *this specific GGUF's* own quantization method — the conversion pipeline used here appears to be a standard `llama-quantize` Q4_K_M pass, not a preservation of the base's native QAT Q4_0 weights. This is precisely the "record what it ACTUALLY is, not what its name says" case this session's gate 1 exists for, and it is worth flagging distinctly from item 5 only because the filename actively claims otherwise.

### Gate 2 — Admission: PASS

### Gate 3 — Tokenizer: **0/721 diverge** vs `google/gemma-4-26B-A4B-it` — clean

### Gate 4 — Reference (KLD): `mean_kld: 0.1545, top1_agreement_pct: 77.25, mean_top8_overlap: 0.816`

Within the family's range for a mixed-requant expert scheme (compare item 5's 65.5%/1.006 for a similarly mixed but more aggressively quantized set, and item 4's 80.5%/0.126 for the genuine uniform-QAT file) — consistent with the already-documented top-8-of-128 routing chaos, not a new finding. Evidence: `docs/cert-matrix-evidence/t2.15-kld-raw.json`.

### Gate 5 — cpu_cuda: **PASS** — byte-identical (64 tokens, full 30/30 offload)

### Gate 6 — Chat smoke: **PASS** — `"4"`, `finish_reason: "stop"`

### Gate 7 — Perf row

```
CPU: {"gpu_layers":0,"layers":30,"prompt_tok_s":8.611,"gen_tok_s":6.451,"prompt_s":59.460,"gen_s":39.685}
GPU: {"gpu_layers":30,"layers":30,"prompt_tok_s":84.486,"gen_tok_s":50.059,"prompt_s":6.060,"gen_s":5.114}
```

### Summary

A community "QAT + mutation combo" release that behaves like every other clean gemma4-moe row on correctness (admission, tokenizer, cpu_cuda, chat) and misses only the numeric KLD bar for the family's own well-documented reason. The genuinely new information this row adds is provenance, not behavior: its "QAT" branding does not match its actual tensor-level quantization scheme, which is Bartowski-shaped PTQ rather than item 4's uniform-QAT Q4_0.

## 16. BrainStorm GPT-OSS 36B (DavidAU IQ4_NL)

**Verdict: FAILED** — chat smoke fails with the same runaway/meta-commentary shape as the rest of the gpt-oss family (gate 6). The gate-1 finding is the interesting one: this is a **real** BrainStorm-style layer-duplication expansion, not a rename — `block_count` is genuinely 43 (vs the base gpt-oss-20b's 24), expert count is unchanged at 32/top-4, and because layer duplication requires a dequant-recombine-requant pass, native MXFP4 experts are gone: every expert tensor has been fully requantized to a mixed IQ4_NL/Q5_1 scheme. That mixed-requant side effect is also, per this session's established pattern (items 9 and 15), why this file's KLD numbers are the second-best of the entire gpt-oss family — losing native MXFP4 costs the "authenticity to the original weights" but happens to buy back numerical stability relative to llama.cpp's reference path.

**Resolved:** `DavidAU/OpenAi-GPT-oss-36B-BrainStorm20x-uncensored-gguf/OpenAI-36B-Brains20x-Uncensored-IQ4_NL.gguf`. Downloaded, sha256 `cc08c58b24bbcdeae7dc21fce2f9e7457b61728ac8ed5ab4a7746c2989cd8a6e` verified against the HF API blob hash and the locally computed hash, 21,182,210,272 bytes. Deleted after this verdict.

### Gate 1 — Identity: **real 43-layer expansion, experts requantized off MXFP4**

**Manifest:** `GPT-OSS(BrainStorm-expanded) / 43L(from 24L base) / 32E / top4 / MIXED-REQUANT(IQ4_NL/Q5_1, non-uniform per-layer) / harmony-chat`

```
gguf version: 3  tensors: 820  kv: 35
general.architecture: gpt-oss
general.name: GTP Osss 20b Test1
gpt-oss.block_count: 43   <- base gpt-oss-20b is 24; confirms real layer duplication, not a relabel
gpt-oss.expert_count: 32
gpt-oss.expert_used_count: 4

tensor type histogram (820 tensors): Q8_0 x1, F32 x517, IQ4_NL x254, Q5_1 x48
expert-tensor types (344 tensors): F32 x215, Q5_1 x5, IQ4_NL x124
non-expert-tensor types (476 tensors): Q8_0 x1, F32 x302, IQ4_NL x130, Q5_1 x43
expert layers found: 43, uniform expert dtype across layers: False
```

`general.name` ("GTP Osss 20b Test1") is a leftover from the base checkpoint and has nothing to do with the actual file — another instance of this session's "record what tensors ACTUALLY are, not what the name says" rule, this time cutting the other way from the family norm: every other gpt-oss row in this matrix preserves native MXFP4 experts under a misleading quant-scheme label; this one is the first gpt-oss row where the experts genuinely were touched, as a structural side effect of layer duplication rather than a deliberate requant choice. Evidence: `docs/cert-matrix-evidence/t2.16-gguf-inspect.json`.

### Gate 2 — Admission: PASS — both runner and llama.cpp b10280 load the 43-layer file without issue

### Gate 3 — Tokenizer: **222/721 diverge** vs `openai/gpt-oss-20b` — identical count to every other gpt-oss row this session (unrelated to the layer expansion; tokenizer is untouched by BrainStorm). Evidence: `docs/cert-matrix-evidence/t2.16-difftok.log`.

### Gate 4 — Reference (KLD): `mean_kld: 0.0770, top1_agreement_pct: 87.5, mean_top8_overlap: 0.866`

Second-best KLD of the entire gpt-oss family this session, behind only item 9's fully-requantized 12-expert prune (88%/0.040) and ahead of every native-MXFP4 row (65.5-84% top1, 0.11-0.14 KLD). Consistent with the pattern established across items 9, 12, and 15: once native MXFP4 experts are gone, cross-engine agreement improves markedly — still short of the 97%/0.05 certification bar, but for a structural reason (residual differences in the requant scheme, not the previously-diagnosed MXFP4 vec_dot_type mismatch). Evidence: `docs/cert-matrix-evidence/t2.16-kld-raw.json`.

### Gate 5 — cpu_cuda: **PASS** — byte-identical (64 tokens, "The capital of France is", full 43/43 offload). Only one prompt tested for this item; per item 13's finding that MXFP4-adjacent near-tie sensitivity can be prompt-dependent, this file's PASS should be read as "clean on the prompt tested," not a blanket guarantee — but it is also consistent with gate 1's finding that this file no longer has any native MXFP4 tensors to disagree about. Evidence: `docs/cert-matrix-evidence/t2.16-cpu.out`, `t2.16-gpu.out`.

### Gate 6 — Chat smoke: **FAIL**

```json
{"content":"\"\n\nWe need to parse the instruction. The user is asking: \"What is 2+2?\" They want a brief answer. The correct answer is 4. The instruction is to answer the math question. There's no context to consider. The response should be a concise answer: \"2+","finish_reason":"length"}
```

Same shape as the rest of the gpt-oss family: the model reasons about the question in Harmony analysis-channel style instead of answering directly, and runs to the token limit without ever emitting a final answer. No raw `<|channel|>`/`<|message|>` tag leakage into content. Evidence: `docs/cert-matrix-evidence/t2.16-chatresp.json`.

### Gate 7 — Perf row

```
CPU: {"gpu_layers":0,"layers":43,"prompt_tok_s":33.186,"gen_tok_s":6.462,"prompt_s":15.428,"gen_s":39.619}
GPU: {"gpu_layers":43,"layers":43,"prompt_tok_s":15.431,"gen_tok_s":14.553,"prompt_s":33.180,"gen_s":17.591}
```

### Summary

Confirms the goal doc's own suspicion (item 16 was listed as "if one exists") — a real BrainStorm20x layer-duplication release does exist for gpt-oss-20b, expanding 24 to 43 layers while holding expert count fixed. It fails certification for the same chat-coherence reason as the rest of the gpt-oss family, but it is a genuinely different artifact from the rest of the roster at the tensor level: the only gpt-oss row this session where expert tensors were unavoidably moved off native MXFP4, and correspondingly the row with the second-best cross-engine KLD agreement in the family.

## Tier 3 — speculative decoding (MTP)

## 17. Gemma 4 26B-A4B + MTP drafter (HauhauCS)

**Verdict: REFUSED** — the drafter's architecture (`gemma4-assistant`) is not one the runner supports through `--draft`; per the goal doc this is an explicitly acceptable outcome ("unknown drafter arch → REFUSED is a fine result"). The load-bearing finding is that the refusal is **safe**: the runner does not crash and does not silently corrupt output when handed an unsupported drafter — it prints a clear error and falls back to ordinary non-speculative generation, and that fallback path is proven byte-identical to running without `--draft` at all across the full six-prompt protocol.

**Resolved:** main model `HauhauCS/Gemma4-26B-A4B-QAT-Uncensored-HauhauCS-Balanced-MTP/Gemma4-26B-A4B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf` (same file as item 15, sha256 `3c13133469e431312fffb8b1d9c85ae42199e6bb5746ea1da84e8ddf2097d73c`, re-downloaded and re-verified) plus drafter `mtp-gemma-4-26B-A4B-it.gguf` from the same repo, sha256 `62bd3af7f66c9308de9a5454233852f8c7324c93767e8dfb824ed45b9179864a` verified, 251,937,728 bytes. Both deleted after this verdict.

### Gate 1 — Drafter identity

```
gguf version: 3  tensors: 49  kv: 44
general.architecture: gemma4-assistant
general.name: 26B A4B Assistant
gemma4-assistant.block_count: 4
gemma4-assistant.embedding_length: 1024
gemma4-assistant.feed_forward_length: 8192
gemma4-assistant.context_length: 131072
tokenizer.ggml.model: gemma4   <- same tokenizer family as the main model, satisfies the "same-vocab" precondition for speculative decoding

tensor type histogram (49 tensors): F32 x26, Q4_0 x23
expert-tensor types: none -- dense model
```

A genuinely distinct, purpose-built small dense drafter (4 layers, 1024-wide, `gemma4-assistant` arch string — not a truncated copy of the main model under a different name), sharing the main model's tokenizer as speculative decoding requires. Evidence: `docs/cert-matrix-evidence/t3.17-drafter-inspect.json`.

### Gate (a) — Drafter admission: **REFUSED**

Default behavior:

```
error: unsupported architecture 'gemma4-assistant' — refusing to run it through llama-style math (set RUNNER_ALLOW_UNKNOWN_ARCH=1 to try anyway, EXPERIMENTAL: output may be silently wrong)
```

The run does not abort — it proceeds to load and generate from the main model alone, ignoring the unusable drafter. Under the documented experimental escape hatch (`RUNNER_ALLOW_UNKNOWN_ARCH=1`, a runtime flag, not a code change), the drafter fails a second, more specific check:

```
warning: architecture 'gemma4-assistant' is UNSUPPORTED; RUNNER_ALLOW_UNKNOWN_ARCH is set — attempting llama-style load, output may be silently wrong
error: invalid NextN/MTP layer count 4 for 4 blocks
```

Confirms this is a real MTP/NextN-style drafter architecture (the runner has an actual NextN-layer-count check it fails, not a generic "unknown arch" catch-all) that the runner's current speculative-decoding path does not yet support even when forced. Both the default and forced paths fall back to plain generation rather than crashing or hanging. Evidence: `docs/cert-matrix-evidence/t3.17-admission.log`, `t3.17-admission-forced.log`.

### Gate (b) — Losslessness: **6/6 byte-identical**

Six-prompt protocol (four 64-token, two repeated at 256), greedy (`--temp 0`), `--gpu off`, comparing `--draft <drafter> --draft-k 4` against no `--draft` flag at all:

```
a       (64 tok):  IDENTICAL
b       (64 tok):  IDENTICAL
c       (64 tok):  IDENTICAL
d       (64 tok):  IDENTICAL
b-long  (256 tok): IDENTICAL
c-long  (256 tok): IDENTICAL
```

Since the drafter never actually engages, this proves the *shape* of losslessness the goal doc asks for (passing an unsupported `--draft` file cannot corrupt or alter output) rather than losslessness of an active speculative-decoding pipeline — the two are the same test but a materially different result depending on whether the drafter loads. Evidence: `docs/cert-matrix-evidence/t3.17-lossless-summary.txt`.

### Gate (c) — Speedup: **none measurable (as expected)**

```
CPU without --draft: gen_tok_s 7.076
CPU with    --draft: gen_tok_s 7.127   (+0.7%, within run-to-run noise)
GPU without --draft: gen_tok_s 51.120
GPU with    --draft: gen_tok_s 49.448  (-3.3%, within run-to-run noise)
```

No speedup and no meaningful slowdown — consistent with gate (a): the drafter is rejected before any speculative-decoding work happens, so `--draft`'s only measurable cost is the drafter file's own (cheap, 252MB) load-and-reject pass. Evidence: `docs/cert-matrix-evidence/t3.17-bench-without-cpu.json`, `t3.17-bench-with-cpu.log`, `t3.17-bench-without-gpu.json`, `t3.17-bench-with-gpu.json`.

### Summary

The goal doc frames a speedup-with-losslessness-proof as a "README-grade result" — this item doesn't produce that, because the one MTP drafter this session could resolve for the Gemma 4 26B-A4B family uses an architecture (`gemma4-assistant`, real NextN/MTP layer semantics) the runner's `--draft` path doesn't yet support. What it does produce is a clean capability-gap finding plus a genuinely reassuring robustness result: an unsupported drafter is refused loudly and specifically (not silently misused), and the fallback to ordinary generation is provably byte-identical to never having passed `--draft` at all. This is the same "record what's actually true and move on" discipline as the rest of the roster — REFUSED, with precise evidence, is the deliverable.

## Tier 4 — big iron

## 18. gpt-oss-120b MXFP4 (canonical, ~59GB)

**Verdict: FAILED** — same two failure modes as every native-MXFP4 gpt-oss row this session (gate 5 cpu_cuda, gate 6 chat smoke), at the largest scale tested. The scale-up itself is the interesting finding: with 128 experts instead of 20b's 32, this is the **worst cross-engine KLD agreement of the entire gpt-oss family** (64.5% top-1, 0.254 mean KLD — the next-worst row this session was 65.5%), consistent with the already-diagnosed near-tie top-k routing sensitivity simply having more ways to flip when there are four times as many experts to choose the top-4 from.

**Resolved:** `ggml-org/gpt-oss-120b-GGUF/gpt-oss-120b-MXFP4.gguf`. Downloaded, sha256 `582bd40f6886200101f4c4ed9f25f3fe80cc14c86e9e2b37746cd8904a0c622d` verified against the HF API blob hash and the locally computed hash, 63,387,346,208 bytes (~59GiB). This is the goal doc's explicit disk exception (item 18, "allowed to be the only artifact on disk"); deleted immediately after this verdict per standard disk discipline.

### Gate 1 — Identity: canonical native-MXFP4 scale-up of the 20b family

```
gguf version: 3  tensors: 687  kv: 36
general.architecture: gpt-oss
general.name: gpt-oss-120b
gpt-oss.block_count: 36        <- vs 20b's 24
gpt-oss.expert_count: 128      <- vs 20b's 32
gpt-oss.expert_used_count: 4   <- same top-k as 20b

tensor type histogram (687 tensors): Q8_0 x146, F32 x433, MXFP4 x108
expert-tensor types (288 tensors): F32 x180, MXFP4 x108
expert layers found: 36, uniform expert dtype across layers: True
```

Uniform-MXFP4 experts across all 36 layers — the same QAT-native signature seen on every canonical gpt-oss release this session, just at 4x the expert count. Evidence: `docs/cert-matrix-evidence/t4.18-gguf-inspect.json`.

### Gate 2 — Admission: PASS (both engines, both needed partial/patient loading)

Runner auto-fit chose partial GPU offload: `gpu-split: budget=25.13GB fixed=1.17GB G=13/36 full=0 used=23.66GB` — 13 of 36 layers on the 24GB MIG slice, the rest on CPU, exactly the "GPU offload what fits + CPU rest" behavior the goal doc asks for. llama.cpp b10280 (`-ngl 999`) also loaded successfully (~18 minutes wall-clock for the 59GB read+init) and produced coherent completions. One operational note for future big-iron runs on this box: a killed runner leaves a stale entry in its `/tmp/gridcore-vram-GPU-*.reg` lease file if the process becomes a zombie before being reaped, which then blocks a subsequent run with a false "VRAM already held" error until the stale `.reg` file is removed — not a correctness issue, just a lock-hygiene wrinkle worth knowing about when re-running big models back-to-back.

### Gate 3 — Tokenizer: **222/721 diverge** vs `openai/gpt-oss-120b` — identical count and identical divergent strings to every other gpt-oss row this session (confirms the divergence is a tokenizer-implementation property, entirely independent of model size). Evidence: `docs/cert-matrix-evidence/t4.18-difftok.log`.

### Gate 4 — Reference (KLD): `mean_kld: 0.2538, top1_agreement_pct: 64.5, mean_top8_overlap: 0.773`

Worst of the entire gpt-oss family this session (previous worst was item 11's 73.75%/0.136). The 128-expert/top-4 routing here has four times as many near-tie candidates per layer as the 20b family's 32-expert/top-4, and this session's running diagnosis (MXFP4 `vec_dot_type` precision differences flip near-tie routing decisions between engines) predicts exactly this direction of effect: more experts to route among compounds the flip rate. Evidence: `docs/cert-matrix-evidence/t4.18-kld-raw.json`.

### Gate 5 — cpu_cuda: **FAIL**

```
CPU: "The capital of France is Paris.\n\nGreat! If you have any more questions or need further assistance, feel free to ask!..."
GPU: "The capital of France is Paris.\n\nGreat! Here's a possible prompt for a short story based on the given input:..."
```

Diverges immediately after the shared prefix "The capital of France is Paris.\n\nGreat! " — same MXFP4-vec_dot-mismatch signature as every other native-MXFP4 gpt-oss row, unsurprising given gate 4's finding that this file has the family's worst near-tie sensitivity. Evidence: `docs/cert-matrix-evidence/t4.18-cpu.out`, `t4.18-gpu.out`.

### Gate 6 — Chat smoke: **FAIL**

```json
{"content":" The answer is 4.\"\n\nNow we need to produce the answer: \"The answer is 4.\"\n\nThus the answer is \"The answer is 4.\"\n\nThus the answer: The answer is 4.\n\nThus the answer: The answer is 4.\n\nThus the answer: The answer is ","finish_reason":"length"}
```

A different failure shape than most of the family (a degenerate repeat-loop rather than Harmony-analysis-channel meta-commentary), but the same underlying result: no clean single answer, runs to the token limit. No raw `<|channel|>`/`<|message|>` tag leakage into content. Evidence: `docs/cert-matrix-evidence/t4.18-chatresp.json`.

### Gate 7 — Perf row

```
CPU: {"gpu_layers":0,"layers":36,"prompt_tok_s":32.993,"gen_tok_s":8.549,"prompt_s":15.518,"gen_s":29.943}
GPU: {"gpu_layers":13,"layers":36,"prompt_tok_s":29.162,"gen_tok_s":10.898,"prompt_s":17.557,"gen_s":23.491}
```

Only a modest GPU uplift (10.9 vs 8.5 gen tok/s) — expected, since just 13 of 36 layers fit the 24GB MIG slice and the bulk of the MoE compute still runs on CPU.

### Summary

The largest model in the roster confirms the session's central gpt-oss finding at scale rather than overturning it: the runner correctly identifies, admits, and partially-GPU-offloads a canonical ~59GB native-MXFP4 checkpoint (validating both the disk-exception workflow and the auto-fit partial-offload logic), but fails the same two correctness gates every native-MXFP4 gpt-oss row has failed. The new information is quantitative, not qualitative — going from 32 to 128 experts measurably worsens cross-engine numerical agreement, which is exactly what the standing MXFP4 near-tie-routing diagnosis predicts.

## 19. gpt-oss-safeguard-120b

**Verdict: REFUSED** — every GGUF conversion of this checkpoint on Hugging Face, across every uploader (`unsloth`, `lmstudio-community`, `mradermacher`, `cPilotGod`), ships as a 2-part split file; this hits the exact split-GGUF capability gap item 10 already root-caused, now confirmed on a second, unrelated checkpoint.

**Resolved:** `lmstudio-community/gpt-oss-safeguard-120b-GGUF`, MXFP4 (the closest available quant to the canonical native format, matching item 18's build). Only shard 1 of 2 was downloaded — `gpt-oss-safeguard-120b-MXFP4-00001-of-00002.gguf`, sha256 `c53a801fe89b033e64d1e30254c1f8e38b96cf547e86329e225697e4c1e8ac6f` verified, 39,815,567,072 bytes. Shard 2 (23.6 GB, sha256 `d20bb68212bcef0a3a01f8a8227a0ab20647ea1f38955817c139c18e79a91a9f` per the HF API, not downloaded) was deliberately skipped: item 10 already conclusively established that the runner never discovers or reads sibling shards regardless of how many are present on disk, so a second shard cannot change the outcome — downloading it would cost ~24GB of bandwidth/disk to reconfirm an already-proven negative. Shard 1 deleted after this verdict.

### Gate 1 — Identity (partial — same single-shard limitation as item 10)

```
gguf version: 3  tensors: 434  kv: 36
general.architecture: gpt-oss
general.name: Tlhv_Osb Mini        <- internal/obfuscated codename, unrelated to the public model name
gpt-oss.block_count: 36
gpt-oss.expert_count: 128
gpt-oss.expert_used_count: 4
split.no: 0
split.tensors.count: 687           <- full model has 687 tensors; this shard holds 434 (23 of 36 layers' worth)
split.count: 2

tensor histogram (434 tensors, shard 1 only): Q8_0 x94, F32 x273, MXFP4 x67
expert-tensor types (179 tensors): F32 x112, MXFP4 x67
uniform expert dtype (within the visible 23 layers): True
```

Same 36-layer/128-expert/top-4 shape as item 18's canonical 120b build, with uniform native MXFP4 experts — this is a genuine same-architecture safety fine-tune of the base 120b checkpoint, not a structurally different model. `split.count`/`split.no`/`split.tensors.count` are explicit, correctly-populated KVs in the file itself, confirming (as item 10 inferred but could not directly read) that split-awareness metadata really is present in the standard distribution format — the gap is entirely on the runner's read side, not a missing-metadata problem upstream. Evidence: `docs/cert-matrix-evidence/t4.19-gguf-inspect.json`.

### Gate 2 — Admission: **REFUSED**

```
error: missing tensor blk.22.post_attention_norm.weight
error: missing tensor blk.22.ffn_gate_inp.weight
error: missing MoE expert tensor (neither fused ffn_gate_exps nor split ffn_gate.0) in blk.22
```

Identical failure shape to item 10: clean, fast (exit code 1, well under a minute, no hang), fails at the first layer (`blk.22`) whose tensors live in the un-discovered second shard. This is not a checkpoint-specific quirk — it is the same universal runner-side limitation, now confirmed on a completely different model family/lineage (a 120b safety fine-tune vs item 10's REAP-pruned base model), reinforcing that the gap is genuinely architecture/format-level rather than an artifact of one particular conversion. Evidence: `docs/cert-matrix-evidence/t4.19-admission.log`.

Gates 3-7 do not apply; a refused load ends the battery.

### Summary

A second, independent confirmation of item 10's finding rather than a new one: gpt-oss-safeguard-120b is universally distributed as a 2-part split GGUF (true of every uploader checked), and the runner's lack of split-file support blocks it identically regardless of which specific checkpoint or uploader produced the split. Declining to download the untested second shard was itself a disk-discipline call — the root cause was already proven at the code level in item 10 (`gguf_get_u32`-adjacent tensor-table reading only ever looks at the file passed on the command line), so no amount of additional shard data changes this item's verdict.

## 20. 220A20B expanded-expert FrankenMoE

**Verdict: NOT FOUND** — no GGUF conversion of this checkpoint exists anywhere on Hugging Face.

**Resolved (source only, no GGUF):** `LLMWildling/gpt-oss-220a20b` — a real model (safetensors, `gpt_oss` architecture, `num_experts_per_tok: 20`, native MXFP4, based on `openai/gpt-oss-120b`), distributed only as 15 `model-NNNNN.safetensors` shards plus the standard HF config files. No `-GGUF` sibling repo exists for it under any of the usual quantizer accounts (`bartowski`, `unsloth`, `mradermacher`, `ggml-org`), and a search across the uploader's entire catalog (`LLMWildling`, a prolific publisher of expert-expanded gpt-oss and gemma4 "Franken" variants — `gpt-oss-140b-ren-2`, `-160b-kiwi`, `-180b-goomba`, `-200b-goblin`, and this `220a20b`, plus dozens of gemma4-coder expansions) turned up zero GGUF repos for any of them: the author ships only `safetensors` and `NVFP4` formats. Since the runner and the llama.cpp reference both require GGUF, there is nothing loadable to admit, and no plausible alternate repo/filename to try.

### Summary

The goal doc's own hedge for this item ("a REFUSED/loads-metadata result is all we want") anticipated a GGUF existing that might fail to load; the real situation is one step earlier — nobody has converted this specific expert-expansion checkpoint (or any of its siblings from the same author) to GGUF at all. Per the goal's own naming-reality-check principle, NOT FOUND is the accurate verdict rather than forcing a substitution onto a differently-named model that wouldn't actually be this roster item.

## Note-only formats (no download)

**NVFP4** — NVIDIA's block-scaled 4-bit float format for Blackwell-generation tensor cores, native to TensorRT-LLM and vLLM's NVFP4 path. It is a GPU-kernel-specific packed representation, not a GGUF tensor type, and has no CPU reference implementation to fall back on. Out of scope for a llama.cpp-family GGUF runtime: there is no conversion path into GGUF's tensor encoding and no kernel in this runner (or in the b10280 reference) that understands it.

**FP8** — 8-bit floating point (E4M3/E5M2), a training- and serving-time format with native hardware support on H100/Blackwell-class GPUs (vLLM, TensorRT-LLM, native `transformers` FP8 paths). GGUF has no FP8 tensor type; the format lives entirely in the safetensors/HF-checkpoint world one layer upstream of anything a GGUF runtime reads.

**BnB (bitsandbytes)** — a PyTorch runtime quantization scheme (NF4/INT8) applied on-the-fly when a model is loaded through `transformers` + `bitsandbytes`. It is not a file format at all in the GGUF sense — there is nothing to download and point a GGUF runtime at; the quantization only exists inside a live Python/PyTorch process.

**MLX** — Apple's own array framework and model format for Apple Silicon (`mlx-lm`), with its own quantization scheme and weight layout. It is a separate runtime and file format from GGUF/llama.cpp entirely (no shared tensor encoding, no shared metadata schema), so there is no meaningful "admit this MLX file into the runner" test to run.

## The 16 GB envelope: lever sweep + live verification

**Recommendation: no candidate beats keep-30.** The current holder (`gpt-oss-20b-keep30-MXFP4`, 32→30 experts, native MXFP4 preserved, 11.47 GB, previously certified at top-1 ≥97%/KLD ≤0.05 vs its own parent) was re-verified this session under an identical live 16 GiB-envelope cap alongside the two prime candidates the goal doc named, and it wins on every axis tested: it is the fastest of the three under the cap (13.2–13.3 tok/s vs 7.1–7.3 and 12.1–12.2 tok/s), and it is the only pruning ratio that actually clears the strict numeric quality bar — every keep-N point attempted this session on the other two candidates missed the bar, several by a wide margin.

### Fit arithmetic (no live test needed)

| candidate | weights | KV @ context | total | fit verdict |
|---|---|---|---|---|
| gpt-oss-20b keep-30 (current holder) | 11.47 GB | small (24L, 4096 ctx) | ~11.5–12 GB | **fits** |
| Gemma 4 26B-A4B-it QAT Q4_0 (item 4) | 13.45 GiB | 1.80 GiB f16 @ c=8192 → 0.96 GiB with `--kv q8` | 15.25 GiB (f16) / **14.41 GiB (q8)** | **fits with levers** (`--kv q8` alone, no pruning needed) |
| GPT-OSS Nano 9B (item 9, already a 12-expert prune) | 6.36 GiB | small (24L, 4096 ctx) | ~6.5 GB | **fits trivially**, no levers needed |
| GPT-OSS 120B REAP 58B (item 10) | 39 GB (Q5_0, the smallest quant this repo offers — no Q4 option exists) | n/a | **≥39 GB, 2.4× over budget** | **does not fit** — even the theoretical floor of a quality-preserving quant (this session's own q4ne finding: sub-4-bit already fails at 22.5% top-1) can't close a 2.4× gap; no live test can rescue an artifact whose weights alone are 2.4x the entire budget |

Item 4's `--kv q8` numbers came directly from the runner's own verbose diagnostics (`-v`): `kv cache 1845.5 MB (fp16)` → `980.4 MB (q8_0)` at `-c 8192`, measured on the actual file (`~/workspace/Gridcore/gridcore-runner/models/gemma-4-26B_q4_0-it.gguf`, sha256 `3eca3b8f...`, already on the box per item 4's own report — no download needed).

### Expert-pruning lever: "does QAT tolerate pruning better than PTQ?" — tested, and the answer is no

The goal doc's own framing for item 4 called this out explicitly as an open experiment. Saliency traces were collected via `RUNNER_MOE_TRACE` (gate*norm weighting, REAP-style) over a corpus subset, `--prune-experts` plans generated with `scripts/moe-prune-plan.py --use-norms`, and each pruned artifact requantized (per-tensor precision left untouched — only expert count changed) and KLD-gated against its own unpruned parent via `scripts/kld-compare-raw.py` (same-engine, both sides the runner itself, on `/v1/completions`).

**Gemma 4 26B-A4B-it QAT Q4_0 (128 experts, top-8) — 3 keep-N points, all FAIL:**

| keep-N | experts dropped | top-1 vs parent | mean KLD vs parent | verdict |
|---|---|---|---|---|
| 96 | 32 (25%) | 67.75% | 0.377 | FAIL (bar: ≥97% / ≤0.05) |
| 64 | 64 (50%) | 50.25% | 0.719 | FAIL |
| 48 | 80 (62.5%) | 45.25% | 0.896 | FAIL |

**GPT-OSS Nano 9B (12 experts, top-4 — already a prune of the 32-expert base) — 3 keep-N points, all FAIL:**

| keep-N | experts dropped | top-1 vs parent | mean KLD vs parent | verdict |
|---|---|---|---|---|
| 10 | 2 (16.7%) | 79.5% | 0.099 | FAIL |
| 8 | 4 (33%) | 72.0% | 0.180 | FAIL |
| 6 | 6 (50%) | 59.25% | 0.344 | FAIL |

Both curves are monotonic and neither gets remotely close to the bar even at the mildest point tested — contrast with keep-30's own 6.25% expert drop (2 of 32) clearing 97%/0.05 cleanly. Two comparative findings fall out of this data even though both experiments failed their own goal:

1. **The QAT-tolerates-pruning hypothesis is refuted, not confirmed**, at least at the drop ratios tested: Gemma 4 26B-A4B's QAT weights degrade *faster* under expert pruning than gpt-oss-20b's do (67.75% top-1 at a 25% drop vs keep-30's presumed ≥97% at a 6.25% drop) — a 128-expert/top-8 architecture has far less per-expert slack than a 32-expert/top-4 one, consistent with this session's whole-roster finding that gemma4-moe's top-8-of-128 routing is measurably more chaos-prone than gpt-oss's top-4-of-32.
2. **Nano 9B's already-pruned 12-expert base has essentially no further pruning headroom left** — even the mildest additional drop tested (2 of 12, the same *proportional* aggressiveness class as keep-30's 2-of-32) misses the bar by a wide margin (79.5% vs the ≥97% bar), unlike keep-30's own 2-of-32 drop on its unpruned 32-expert base. This suggests pruning tolerance depends heavily on how much slack the *specific* expert roster already has, not just the raw fraction removed — a roster already thinned once has less room to give twice.

Neither candidate needed the pruning lever to fit 16 GB in the first place (both already fit via `--kv q8` or trivially, per the fit-arithmetic table above), so these FAILs subtract nothing from either model's own already-fits status — they only rule out expert-pruning as a way to shrink either one *further* within the envelope.

### Live verification: ballast-capped decode under a real 16 GiB cap

Per the goal's hard-won ballast rules (regulate against the container cgroup `memory.max`, not `/proc/meminfo` — confirmed `memory.max = 200 GiB` exactly on this box vs the host's reported 250 GB; fill with random bytes via a fast xorshift64 fill, not zeroed pages; grow in shrinking chunks and pace the last GiB slower as the ceiling approaches; never touch `oom_score_adj`; freeze the size once at target and never release/regrow it mid-measurement), a purpose-built ballast tool (`docs/cert-matrix-evidence/16gb-sweep/ballast.c`) was written, compiled, and validated at small scale before the real run. It adaptively grows anonymous memory while polling the real `/sys/fs/cgroup/memory.current`, targeting `memory.max − 16 GiB reserve − 2 GiB safety margin = 182 GiB`, then freezes and holds until `SIGTERM`. No child cgroup could be created for a cleaner hard cap (`/sys/fs/cgroup` is read-only in this container — confirmed by a failed `mkdir` test), which is exactly why the ballast method is the right tool here rather than a nicety.

Each of the three candidates below was tested **CPU-only** (`--gpu off`) — deliberately, since this box's GPU VRAM is a separate resource the memory cgroup does not account for, and CUDA offload would let a model bypass the ballast entirely rather than genuinely compete for the same constrained pool the way a real 16 GB Mac's unified memory would force it to.

| candidate | cold tok/s | warm tok/s | ballast integrity | usable (≥5 tok/s)? |
|---|---|---|---|---|
| **gpt-oss-20b keep-30 (current holder)** | **13.32** | **13.18** | held (102.75 GiB, no loss) | **yes — fastest of the three** |
| GPT-OSS Nano 9B (unpruned, as-is) | 12.23 | 12.11 | held (113.50 GiB, no loss) | yes |
| Gemma 4 26B-A4B QAT Q4_0 (`--kv q8`) | 7.08 | 7.32 | held (113.50 GiB, no loss) | yes |

All three comfortably clear the ≥5 tok/s usable bar under the real cap, and the ballast held its full frozen size through every run for all three (re-checked after each generation — no measurement here is "impossibly warm" from a silently-shrunk ballast). Cold vs warm shows almost no difference for any candidate: under an ~18 GiB total headroom for a 6.4–14.4 GiB model, there is too little slack for meaningful cross-process page-cache persistence between runs — each run essentially re-faults its own working set regardless of whether a prior run just finished, which is itself a realistic, honestly-reported property of operating this close to the edge of a real 16 GB envelope, not a measurement artifact.

Nano 9B and keep-30 both produced coherent-looking greedy continuations on the raw completion prompt used for timing; item 4 (Gemma 4, no chat template applied — a raw, template-free completion, deliberately chosen so the throughput number isn't confounded by chat-formatting differences) degenerated into a repeat loop, matching the same raw-completion instability pattern this session already observed for this file at earlier gates (not a memory-pressure artifact — the identical prompt produces similar degeneration without any ballast active).

### Verdict

**No candidate beats keep-30.** It remains the correct 16 GB-envelope recommendation: fastest under a real cap, and the only pruning ratio in this session's data (its own, from a prior session, and the two fresh ones swept here) that clears the strict 97%/0.05 quality bar. Gemma 4 26B-A4B-it QAT Q4_0 is a legitimate, independently-useful **fits-with-levers** alternative — a substantially larger, more capable, genuinely-chat-working model (`--kv q8`, no pruning needed, CERTIFIED-WITH-CAVEAT in the main roster) — worth keeping in mind for use cases where its larger capability matters more than keep-30's ~1.8× throughput edge, but it is not a *replacement* for keep-30 by the goal's own numeric bar. GPT-OSS 120B REAP 58B is conclusively out of reach by simple arithmetic; no lever or live test changes that.
