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
