# Changelog

All notable changes to gridcore-runner. This project is in **alpha**; the HTTP
protocol and CLI may still change between alpha releases.

## Unreleased

- **Shared always-on expert (Qwen2-MoE / DeepSeek) is supported.** It was
  refused at load. A dense FFN runs over the same normed input the router saw
  and is summed into the routed output; Qwen2-MoE additionally scales it by
  sigmoid of a scalar router (llama.cpp writes that sigmoid as `silu(x)/x`),
  DeepSeek has no router tensor and adds it unscaled. Both shapes are handled,
  and the width falls back to the routed expert width exactly as the reference
  does. The branch is added at the call sites rather than inside `moe_ffn`, so
  the routed path is untouched — Qwen3-Coder-30B, gpt-oss and gemma-4-26B-A4B
  are byte-identical across the change.

  **CUDA refuses**: the routed kernels write the FFN output and nothing adds a
  second dense branch, so a shared-expert model runs on CPU with a message
  saying why.

  Gated against the dense oracle with the routed experts zeroed, so the output
  can only match if the shared branch ran exactly once — ignoring it collapses
  the FFN to zero, adding it twice doubles it. The gated variant's router
  weight is zero, making the gate `sigmoid(0) = 0.5`, with the shared FFN
  doubled to compensate: it only comes out right if the gate is really applied.
  The previous binary refuses both fixtures outright, which is the negative
  control.

- **Llama-4 attention knobs: NoPE and the position-dependent attention
  temperature.** Every `no_rope_layer_step`-th layer skips rope entirely, and
  on *those same layers* — it is the else-branch of the rope test in
  llama.cpp's llama4 graph, not a separate pass — Q is scaled by
  `log(floor((pos + offset) / floor_scale) + 1) * scale + 1`. Both default off,
  and Qwen2.5-7B, gemma-4-E4B and gpt-oss are byte-identical across the change.
  CPU and CUDA both implement it, so the two backends cannot diverge; the
  multi-sequence batch path declines a NoPE model instead, because it keeps
  positions on the device and would have to scale with the wrong factor — the
  caller then decodes sequentially, the retreat it already takes for a bad
  index.

  Gated by `tests/test_attn_knobs.py` reading the activation trace, because
  greedy text sees none of this: all four fixtures generate byte-identical
  output while their layer-0 Q rows plainly differ. Two of the three checks
  fail on a knob-blind binary, so the gate can fail.

  One check exists to stop a "fix": the temperature is **exactly 1.0 below
  position 8191**, since `floor(pos / 8192)` is 0 there and `log(1) = 0`. That
  reads as a dead knob and matches the reference.

- **`top_k` is served by selection instead of by sorting the vocabulary, and
  that is most models.** The sampler's head fast path could only satisfy top-k
  by widening until the head held `k` entries, and its loosening schedule
  multiplies a *negative* log-threshold by 4 — one step goes from `p_max/1024`
  to `p_max/e^27`, admitting most of the vocabulary and overflowing the
  4096-entry head cap. Instrumented on gemma-4-E4B: the first head carries 99%
  of the mass in 7 entries, fails `m >= 64`, and the next step overflows. It
  could not be fixed by relaxing the criterion either, because `pick_scaled`
  renormalises over exactly the `k` it is handed, so serving 7 where 64 were
  asked changes every probability.

  Now a quickselect partition takes the true top-k in O(n) and only those `k`
  are sorted. This is **exact, not approximate**: `cand_cmp` is a total order
  (logit descending, then id ascending, and ids are unique), so the k-largest
  set is unique and sorting it reproduces the first `k` of a full sort element
  for element. Verified bit-identical against the previous sampler across
  seeds on two real models and twelve seeds on the small fixture.

  Decode throughput, same seed, 96 tokens:

  | model | backend | before | after |
  |---|---|---|---|
  | gemma-4-E4B-it Q4_K_M | CUDA | 26.1 tok/s | **59.0** |
  | gemma-4-E4B-it Q4_K_M | CPU | 9.6 | **12.5** |
  | Qwen2.5-7B Q4_K_M | CUDA | 54.1 | **72.6** |
  | Qwen2.5-7B Q4_K_M | CPU | 15.2 | **16.4** |

  The win scales with vocabulary size and decode speed, because sampling is a
  fixed per-token cost: 2.3x on a 262k-vocabulary model that decodes fast, 8%
  on a slow CPU decode where the model dominates.

- **`--cpu-moe auto` crashed mid-forward on every sparse-MoE model, and the
  first diagnosis was wrong.** It was not a VRAM over-commit. `full` (is
  `output.weight` uploaded) needs the output tensor to fit the budget; `partial`
  (does the device compute logits) is the layer count alone. The auto fit places
  attention for every layer, then spends what survives on expert banks — so the
  last of the budget could go to a bank, dropping `output.weight` while `G` still
  equalled `n_layer`. The forward then asked the device for logits from a tensor
  that was never uploaded. `--cpu-moe 20` "worked" only because fewer banks left
  room by accident, and the plain split cannot reach the state at all. The fix
  reserves what a full split still owes before placing banks, plus an invariant
  guard that makes the split honestly partial when the reserve cannot be met.
  Not only a crash fix: gemma-4-26B-A4B decode goes 4.74 → **10.89 tok/s**.

- **Two UTF-8 defects in server output.** Ill-formed sequences are now rejected
  rather than only ill-shaped JSON, and multi-byte characters whose bytes span
  two tokens are held back while streaming instead of being emitted as
  replacement characters.

- **The unfiltered sampling path is served from a head instead of a full sort.**
  `top_p = 1.0` was the slowest path in the sampler, and two shipped presets use
  it.

- **`compare_llamacpp.py` was reporting prefill from a warm prefix cache.** The
  timing request now runs first with `cache_prompt: false`, and the figures
  agree with each engine's self-reported timings — which is the check that the
  method is sound. Top-logprob entries keyed by rendered string also excluded
  (and counted) the empty string, which is not a token identity: on Ministral-8B
  that manufactured a 4.59-nat divergence on a run whose greedy text was
  byte-identical. MoE rows are now like-for-like via `--runner-arg`.

- **gpt-oss and gemma-4-E4B certified at whole-graph offload.** The kernel box
  could only reach a 13/24 split for a 12.1 GB model on an 8 GB card and said so
  rather than implying whole-graph identity; re-run on the 25 GB slice, both are
  5/5 byte-identical with every layer and `output.weight` on device (24/24 and
  42/42). `cpu_cuda_check.py` now records the split it achieved, so a partial
  report can no longer read as a whole-graph one.

- **`--reserve`, `--reserve-vram`, `--reserve-ram` and `--reserve-cpu` are in
  `--help`.** They were implemented and documented in the README but absent from
  the binary's own option list.

- **CHANGELOG.md restored.** All 737 lines were deleted as collateral in
  `82e799b`, a commit about `--cpu-moe`; six commits carried it empty and
  `make release-check` was red for the whole stretch. Recovered from `45572e2`
  and the entries above reconstructed from the intervening commit messages.


- **gpt-oss runs on CUDA.** The load-time GPU refusal is gone; everything it
  guarded landed together, generated on the CUDA 13.3 box that owns
  `kernels_ptx.h`. *MXFP4 matvec kernels* (`k_mv_mxfp4`, `k_mv_mxfp4_b`,
  `k_moe_mv_mxfp4`): 17-byte block, E8M0 power-of-two scale decoded with
  `ldexpf` exactly as the CPU's `dq_mxfp4`, nibbles through the same signed
  codebook. *Sink-aware attention softmax*: the per-head sink logit joins the
  max scan and the denominator with no value row — transcribed from
  `softmax_sink()` — in `k_attn` and in `k_attn_merge`'s global reduction
  only, so the flash-decoding split partials (`k_attn_dec`,
  `k_attn_dec_seq`) needed no change and every other arch's arithmetic is
  untouched. *`ACT_SWIGLU_OAI` on device* (dense actmul + both MoE actmul
  kernels, one shared `swiglu_oai()` mirroring the CPU's clamp and early-zero
  guard). *Router + per-expert biases through both CUDA MoE paths*: the
  router bias rides the matvec tail (bitwise what the CPU's add-after
  computes); gate/up biases land before the activation; the down bias lands
  before the routing weight scales it — on the eager path by deferring the
  weight fold until after the down matvec, on the fused path as
  `selw*db` inside `k_moe_sum`. The kernel dispatch tables also grew past
  `T_MXFP4 = 39` (they were sized 32 and would have indexed out of bounds),
  and the grouped prefill declares itself ineligible for biased experts
  rather than dropping them. `make test` green; Qwen3-8B and the MoE
  tolerance fixture verified unchanged (CPU==GPU byte identity holds).

- **`parallel_tool_calls: true` is supported on buffered requests.** The
  envelope becomes a bounded `{"calls": [ ... ]}` array over the *same*
  discriminated union, so a direct answer is simply a one-element array
  holding the final branch — the model chooses how many entries to emit, not
  which document shape to use. Capped at 8 entries by construction, because an
  unbounded array under a token budget is a truncation waiting to happen and
  every legal document must stay completable by `sval_close`. Calls map back
  with distinct ascending ids through one shared per-entry mapper, so the
  single-call and multi-call paths cannot drift in how they render a call.
  **Streaming still refuses it**, with its own reason: the demultiplexer
  tracks one call per turn, and silently downgrading to a single call would
  leave the caller expecting calls it never gets. Where the strict envelope
  does not apply at all — no tools declared, or the ornith template's native
  protocol — the flag stays tolerated exactly as before, since ordinary
  OpenAI-shaped traffic sends it alongside requests that will never call
  anything. Gates: multi-call mapping driven directly in tests/test_tools.c
  (ids, separators, the mixed and wrong-shape documents) rather than through
  a sampled model, plus the rewritten conformance contract.

- **Gemma-4 E-series (E2B/E4B) runs, on CPU.** Both missing halves landed.
  *Per-layer embeddings*: a second embedding table gives each token one
  `n_embd_per_layer` slice per layer; those slices are mixed once per batch
  with a projection of the input embedding, and each layer then gates its
  post-FFN residual through them and adds the result back before the layer
  output scale. *Shared-KV layers*: every layer at or past
  `n_layer - shared_kv_layers` computes no K/V at all and attends over the
  cache of the last KV-owning layer **of its own sliding/full type**
  — `kv_from_start - 2` sliding, `kv_from_start - 1` full. Those layers
  reserve no cache rows (E4B's allocation drops by 18/42) and the prefix
  cache skips them so a snapshot cannot save the same rows twice. Their
  `attn_k`/`attn_v` tensors exist in the file and are deliberately never read.
  A mismatched-geometry alias is refused at load rather than reinterpreted.

  **The GPU is refused** for these models, with its own message: the device
  graph has no per-layer-embedding stage and its KV allocator sizes one
  independent region per layer, so aliasing would silently attend over zeros.

  Verified against llama.cpp b10076 on `gemma-4-E4B-it-Q4_K_M`. Greedy
  agreement is at the **quantisation noise floor, not token identity**, and
  the control run is what makes that claim meaningful: over 16 prompts × 32
  tokens the E-series scores 8/16 identical with a 0.29-nat worst-case
  logprob delta, while **Qwen2.5-7B — a long-verified dense architecture on
  the same harness — scores 6/16 with 0.24 nats.** The E-series profile is
  indistinguishable from a model already known correct; both engines flip on
  sub-0.3-nat argmax ties, and llama.cpp flips on them by itself depending on
  whether its prompt cache was warm. New `scripts/token_divergence.py` is the
  tool that measures this: it walks both engines greedily and reports the
  first differing position with the logprob gap between the two contenders on
  each side, which distinguishes a coin-flip tie from an arithmetic fault —
  something `reference_compare.py`'s exact-text gate cannot do.

- **gpt-oss: sliding-window layers were roped in the wrong regime.** Runner
  treats SWA layers as a separate rope world — right for gemma, whose locals
  rope at base 10k with no scaling while its globals run 1M + YaRN — so it
  built the local frequency table from the raw base *before* the YaRN scaling
  block and forced the YaRN magnitude factor to 1.0 there. llama.cpp's
  openai-moe graph passes the same `freq_base`, `freq_scale`, `ext_factor` and
  `attn_factor` to **every** layer and varies only the KV window. With a
  sliding-window pattern of period 2, half of gpt-oss's 24 layers were wrong.

  The base for those layers had already been fixed when the CPU path landed,
  which made the regime look handled; the frequency scaling and the magnitude
  factor live in two other places and were missed. Output stayed fluent
  throughout — the failure mode is a systematic bias, not garbage.

  Layer 0's relative divergence from the reference drops **7.31% → 0.70%**.
  Over 16 prompts × 16 greedy tokens, runner vs llama.cpp goes from 4/16 to
  **9/16** identical and the worst-case logprob delta from **0.589 to 0.151** —
  now *below* runner's own 0.272 sensitivity to a KV-precision change, i.e. at
  or under the model's own floor.

  `model_rope_mscale()` is now the single definition of the per-layer YaRN
  magnitude factor, shared by the CPU and both CUDA rope sites, because a
  disagreement between them is invisible in output that still reads fluently.
  The new `swa_rope_global` flag is set only for gpt-oss; gemma-4-E4B and
  gemma-4-26B-A4B are byte-identical across the change.

- **The gemma-4 E-series runs on CUDA.** The refusal is lifted; both mechanisms
  have a device path and it is byte-identical to the host. Per-layer
  embeddings: the pre-pass stays on the host (it reads a bf16 projection and a
  q5_K table that have no device kernels) and ships its result once per
  forward — 43 KB for a single E4B token — after which each layer gates,
  multiplies by its slice, projects, norms and adds entirely on device.
  Shared-KV layers project Q and skip K/V, attending over the owning layer's
  rows through the same aliased offsets the host path uses, so they cost no
  device cache either.

  **No new kernels**, which matters because the committed PTX is generated by a
  different machine's nvcc: the stage composes `k_mv_f32`, the existing
  `k_gelu_mul` (already exactly `gelu(a)*b`), `k_rmsnorm` and `k_add`.
  `src/kernels_ptx.h` is untouched.

  Gate is CPU/GPU identity, which for this family is exact even though
  cross-engine identity is not. Byte-identical on gemma-4-E4B-it-Q4_K_M over
  four prompts at 32 tokens and one at 128, at full 42/42 offload; and on the
  generated fixture at every partial-offload split, including the ones that put
  a shared-KV layer on a different device from the layer whose rows it reads.
  Throughput on the Blackwell MIG slice, 134-token prompt / 64 decode:
  prompt 76 → 301 tok/s, decode 13.7 → 63.7 tok/s.

  The layer-weight accounting now includes the per-layer embedding matrices —
  they are f32 in the published GGUFs and add ~5 MB per layer, so omitting them
  would under-budget the offload and overcommit VRAM.

- **The gemma-4 MoE identity claim is withdrawn, and replaced with a
  measurement.** Re-running the gate on the *certified* artifact
  (`ggml-org/gemma-4-26B-A4B-it-GGUF`, sha `d208665a…`, now matching the pin in
  `tests/compatibility/models.json`) gives 4/5 on `reference_compare.py` at
  b10076, not the token identity the README claimed. No archived artifact ever
  backed that claim, and the dense gemma-4 claim in `model.c` cites b9964 — a
  different revision.

  The claim is withdrawn because it is **unachievable for this model, by any
  engine pair**, not because runner regressed. New
  `scripts/sensitivity_floor.py` measures what a model does to a small numeric
  change, and on the certified 26B runner disagrees with **itself** — same
  build, same weights, only the KV cache precision changed — on 11 of 16
  prompts, against the 9 it disagrees with llama.cpp on. A perturbation
  strictly inside one binary moves the output further than switching engines
  does, which leaves no room to attribute the gap to a fault. Direct checks
  agree: layer 0's `attn_norm` matches the reference exactly and the
  pre-softmax router logits match to ~0.1–0.5%. The amplifier is discrete
  top-8-of-128 routing over Q4_0 weights — at layer 2 the 6th and 7th selected
  experts sat 0.0002 apart in weight, so a rounding difference flips an expert
  and rewrites an eighth of the FFN output.

  For contrast, on the same harness gemma-4-E4B is perfectly self-consistent
  under that perturbation (16/16) and matches llama.cpp on 11/16 — essentially
  llama.cpp's own cold-vs-warm-cache floor of 12/16. Evidence:
  `tests/compatibility/out/divergence-study-gemma4-moe-2026-08-01.json`.

- **`RUNNER_DEBUG_ACT` traces are now diffable against llama.cpp.** Each line
  carries the sum plus the leading and trailing three values in
  `llama-eval-callback`'s layout, and gemma-4 MoE layers additionally dump the
  pre-softmax router logits and the selected expert ids with their weights.
  That is what localises a divergence to a layer: comparing aggregate stats
  against another engine's per-row values cannot.

- **Completion logprobs now carry token ids** (`token_ids` and
  `top_token_ids`, alongside the existing decoded strings). Two distinct ids
  can decode to the same text, and control tokens render differently across
  runtimes — runner writes `<eos>` where llama.cpp writes `""` — so a
  cross-engine comparison keyed on the rendered string reports identical
  tokens as divergences. It did, until this. The two duplicated emitters
  behind the streaming and buffered paths were also collapsed onto one.

- **Gemma-4 E-series: array-form sliding-window patterns are read correctly,
  and the refusal now names what is missing.**
  `attention.sliding_window_pattern` is published two ways — dense gemma3/4
  give an integer period, the E-series gives a per-layer BOOLEAN ARRAY. Read
  as a u32 an array key silently yields the default, mis-marking every layer,
  so both forms are now handled with the array winning when present (dense
  models are unaffected: verified byte-identical CPU-vs-GPU output on
  gemma-4-26B before and after). The E-series load refusal changed from
  naming the family to naming the two missing mechanisms — per-layer
  embeddings and shared-KV layers — because those are what a reader needs.
  (Both halves have since been implemented — see the entry above.)

- **Measured: partial expert offload helps prefill and hurts decode, and
  plain layer offload beats both.** Qwen3-Coder-30B on the Blackwell MIG with
  the budget capped to a 12 GB card (`--reserve-vram 48`, `-c 4096`,
  512-token prompt / 64-token decode, median of 3):

  | config | experts on GPU | prompt tok/s | decode tok/s |
  |---|---|---|---|
  | `--cpu-moe` (all host) | 0/48 | 15.5 | 5.35 |
  | `--cpu-moe auto` | 29/48 | **28.5** (+84%) | 4.32 (**-19%**) |
  | `--gpu-layers 24` | — | **43.9** (+184%) | **8.27** (+55%) |

  A control at the full 25 GB budget shows the new binding path is free:
  `--cpu-moe 0` (every expert on the GPU through bindings) measures
  194.1 / 101.5 against the ordinary full-offload upload's 194.0 / 100.5. So
  `auto`'s decode cost is inherent to *mixed* placement — the interleaved
  per-layer host bounces — not to the implementation. Two consequences worth
  stating plainly: the first outside install's conclusion that layer offload
  beat `--cpu-moe` is confirmed and much larger than its own numbers showed;
  and `--reserve-vram` **without an explicit `-c`** grows the KV cache to fill
  the reservation, which starves expert placement (auto placed 0/48 banks
  until the context was pinned). No defaults changed — the advisor's
  moe-hybrid preference is an owner call, now with numbers under it.

- **gpt-oss (OpenAI MoE) runs on the CPU path.** `gpt-oss` joins the
  architecture allowlist with the four pieces it actually needs, each
  transcribed from llama.cpp rather than inferred: per-head **attention
  sinks** (a learned logit joining only the softmax max and denominator, with
  no value row — `softmax_sink()`), the clamped alpha-sigmoid
  **`ACT_SWIGLU_OAI`** activation (alpha 1.702, limit 7; plain SwiGLU here is
  silently-wrong output), the **router bias plus per-expert gate/up/down
  biases** in both CPU MoE paths, and `post_attention_norm` as the FFN input
  norm (the qwen35 shape). Its sliding-window layout needed no new logic —
  the existing `((i + 1) % period)` form with period 2 is identical to
  llama.cpp's `set_swa_pattern(2)` — but its SWA layers inherit the GLOBAL
  rope base (150k), where the runner's generic default would have used 10k.
  A vendor sampling preset is included (temperature 1.0 / top_p 1.0, no
  repetition penalty, per the model card).
  **The GPU refuses this architecture at load**, with a stated reason: a
  sink-aware attention softmax and MXFP4 kernels do not exist on that
  backend, and running there would silently drop the sinks.
  **Not certified, deliberately.** Greedy agreement with pinned llama.cpp
  b10076 over the five standard prompts is **4/5 exact at 8 tokens** and 1/5
  at 32 (evidence: `tests/compatibility/out/reference-gpt-oss-2026-07-31-*`).
  The residual is diagnosed, not mysterious: ggml gives MXFP4 a
  `vec_dot_type` of `Q8_0`, i.e. llama.cpp quantizes the *activation* vector
  to int8 before each expert dot, while the runner dots against full fp32
  activations. The two are different computations by construction, so the
  paths drift apart at near-ties — every divergence is mid-sentence with both
  continuations plausible, after 10–88 characters of exact agreement. For
  scale, the same 8-token method scored the *certified* archs at llama3 3/5,
  qwen3 4/5 and gemma4 0/5 on 2026-07-30. The README certified table is
  untouched: the full certification method includes CPU-vs-GPU identity,
  which an arch the GPU refuses cannot have.

- **Fused-vs-eager MoE routing tolerance gate (`make test`).** Certification
  defines MoE byte identity over the eager host-routing path
  (`RUNNER_MOE_EAGER=1`); the shipping fused default's weaker contract —
  selection identical, routing weights within ~2 ulp — had only ever been spot
  checked by hand at the first routing. `tests/test_moe_tol.c` now gates it in
  the shape of test-kv-tol/test-tc-tol: teacher-forced top-1 agreement with the
  near-tie escape (a decisive-margin flip means selection diverged, not just its
  weights) plus a mean|dlogit| bound. New `gpu_moe_eager_force()` test hook lets
  one process run both paths. The gate needs a fixture whose router is not zero —
  the dense-oracle MoE fixtures are 0.5/0.5 either way and can only compare a
  path with itself — so `make-test-moe.py` gained `moe4` (real router, four
  distinct experts, top-2), deliberately not dense-equivalent. Measured row on
  that fixture: mean|dlogit| 1.47e-08 = 2.0e-08 of range, 0/32 flips. Real
  quantized-model rows want a free full-offload slice; the gate self-skips
  (never passes) without one.
- **MTP heads are admitted as training-only, for every architecture.** An
  export whose `block_count` includes auxiliary NextN/MTP predictor blocks
  declares them with `<arch>.nextn_predict_layers`; those blocks are now
  excluded from the backbone on any architecture (this generalizes the
  qwen35-only handling — qwen35 exports read the same key and are unchanged),
  so dense decoding is bit-for-bit identical to an export without them. The
  load line and `/v1/capabilities` report `mtp.declared_layers` with
  `consumed: false`, so a controller sees the exclusion instead of inferring
  it from a layer count. A profile whose `required_features` contains `mtp`
  is refused with its own reason — requiring consumption asks for a verifier
  this build does not implement, which is not the same as an unknown feature
  name. Consumption itself remains unbuilt by design (the staged contract:
  admission first, verifier second). Gate: tests/test_mtp_admission.py.
- **`--bench-json` benchmarks a realistic prefill and reports both phases in
  seconds.** The default prompt was one ten-token sentence, so the instrument
  reported healthy numbers on the first outside install while a realistic
  2,100-token prompt took 89 s to reach its first word — prefill, not decode,
  was the wall. The default is now a synthesized ~512-token prompt (clamped to
  the context and to whatever `-n` needs), `-p`/`-f` still override it, and the
  JSON gained `prompt_s` / `gen_s` beside the existing rates so time-to-first-
  token is directly readable. Numbers from earlier `--bench-json` runs are not
  comparable to these, which is the point.
- **The `--cpu-moe` x `--gpu-layers` worst-of-both is no longer silent.**
  Capping the attention split under tensor-role placement moves attention to
  the CPU while freeing almost nothing (the expert banks are what fill VRAM):
  measured 10.3 tok/s against 12.7 all-host and 14.6 layer-split-alone on a
  12 GB card. The pair stays legal — it is meaningful once a partial expert
  count is what the headroom is reserved for — but a run that caps below what
  already fits now says so and points at `--cpu-moe N|auto`. A bare
  `--cpu-moe` that leaves room for expert banks also reports how many
  `--cpu-moe auto` would place, counted by the same greedy rule so the advice
  cannot over-promise.
- **Partial expert offload — `--cpu-moe [N|auto]`.** Expert placement is now
  per-layer instead of all-or-nothing: `auto` fills whatever VRAM the
  attention split leaves with whole expert banks (shallowest first) and hosts
  only the remainder, `N` pins exactly the deepest N expert layers to the
  host, and a bare `--cpu-moe` keeps its original all-on-host meaning. Device
  banks upload as ordinary offset-resolved bindings, so a device-resident and
  a host-resident bank coexist inside one forward. The split line now reports
  `experts N/M layers on GPU, K on host` — the first outside install could not
  see that all-or-nothing was leaving 8.8 GB of a 12 GB card idle, because
  nothing reported it. `--caps` advertises `cpu_moe_partial`. Measured on the
  Blackwell MIG (slice mostly occupied by another process, so only 3 of 48
  banks fit): Qwen3-Coder-30B prompt 15.82 -> 16.96 tok/s, decode 5.49 -> 5.85
  (+7% each for 6% of the banks moved). Gates: dense-oracle identity for all
  three expert layouts x {0, 1, auto} with the silent-fallback guard, strict
  count parsing, and real-model CPU==GPU byte identity on Qwen3-Coder-30B
  under the eager pin.

- **Grammar fast-forward (JC-R2, runner half) — opt-in.** Under an active
  constraint, when the validator pins a unique byte continuation (probed by
  trial on validator copies — the same validator-by-trial design as the
  sampler filter), the pinned run's tokenization is drafted for free and
  verified by the target through the existing speculative walk, so output is
  byte-identical to plain decoding with or without a `--draft` model in the
  loop (`make test` gate: tests/test_grammar_ff.c, identity + engagement +
  exclusions, ASan-clean). `RUNNER_SPEC_STATS=1` now reports grammar
  acceptance per generation (`grammar a/d`) — the constrained-segment
  acceptance measurement the judgment co-processor plan calls for.
  **Default OFF** (`RUNNER_GRAMMAR_FF=1` to enable, value parsed strictly):
  measured on EuroLLM-9B and Mistral-Nemo (CPU, contract schema), today's
  raw-encode drafter is 4-12% slower at 33-38% acceptance — real subword
  vocabs tokenize pinned runs differently than the model samples them. The
  toy byte-level vocab accepts 100%; the structural fix is a
  model-canonical (Syntetik) drafter, which is the remaining JC-R2 half.
- **Speculative verify batch bound fix.** `model_forward_batch_keep` bounded
  its batch by `spec_batch` only; with `-b` smaller than the draft window
  (n_batch < 16) the verify batch overwrote the activation buffers past
  their allocation (heap corruption in the `--draft` path, found by ASan
  under the new grammar-ff test). Both the primitive and the draft-window
  clamps now respect `n_batch`.
- New `tok_encode_raw`: raw-byte encode without BOS/specials/segment
  normalization (SPM's leading-space prefix), so a token list can
  round-trip to exactly the input bytes wherever the vocab allows.
- **`choice_logprobs` — constrained-choice posteriors (JC-R1).** Constrained
  requests (JSON mode / `json_schema` / tool schemas) can set
  `"choice_logprobs": true` to get, per decision point (a step where ≥ 2 of
  the probed top-`M` candidates were grammar-legal; `choice_logprobs_probe`
  8–64, default 32), the legal alternatives with a posterior renormalized
  over the legal probed set, raw full-vocab logprobs, and probed coverage.
  Captured from raw logits before the repeat penalty, payload phase only
  (thinking preludes have no decision points), legality decided by the same
  validator trial the sampler uses. Buffered responses only; rejected with
  spec-decode. New `scripts/cl-calibration.py` turns labeled decision
  records into accuracy/Brier/ECE + a reliability table and can gate via
  `--max-ece`. Conformance: `tests/conformance/test_choice_logprobs.py`.
- Published benchmark MoE rows updated after the device-routing work:
  Qwen3-30B-A3B decode 102.2 tok/s (67% of llama.cpp, was 48% at
  v0.1.4) and prefill 194.0 (6.0%, was 3.3%) on the Blackwell MIG;
  gemma-4-26B 24.7 / 23.6. docs/benchmarks.md and the shareable page
  carry the same rows.
- `runner` now depends on `src/kernels_ptx.h` in the Makefile: a pull
  that changed only the regenerated PTX header rebuilt nothing, and a
  publication run measured yesterday's kernels for half an hour before
  the stale binary was caught. Same class as the certification footguns
  this week — the build must never silently serve old code.

- MoE routing normalizations reverted to per-element division — the
  k_moe_route PTX section is byte-identical to the 4719de6 body again
  (verified by section hash), which is exactly what the Blackwell
  splice-proof restored to 102.9 tok/s decode (the reciprocal-multiply
  mirror's rcp.rn.f32 codegen JITed ~58 µs/launch slower on that MIG,
  ×48 layers = 23% of MoE decode; the mirror's bit-identity purpose was
  already retired by the eager certification pin). Fused-path selw bound
  restated in the compat doc: within ~2 ulp of the host reference (two
  independent 1-ulp sources), observed 1 ulp at the first routing on
  both cert boxes. Gates on the 3070: make test green; eager-pinned
  CPU==GPU identity byte-green on both MoE models; bench md5 unchanged.

- MoE routing exp reverted to fp32 device expf (keeping the
  reciprocal-multiply mirror), now that certification pins the eager
  path (`RUNNER_MOE_EAGER=1` in the harnesses since bf93510): the
  correctly-rounded double-exp's only purpose was bit-matching
  correctly-rounded hosts, a property void on the fast-math cert box.
  NOTE the property downgrade this trades away: the fused default is no
  longer byte-identical to the host routing even on correctly-rounded
  (UCRT-class) hosts — its contract is now the verified weaker class,
  expert selection identical + selw within 1 ulp of the host reference
  (re-verified on the 3070 with RUNNER_DEBUG_MOE after the revert;
  docs/compatibility-program.md updated to match). Gates on the 3070:
  make test green; certified (eager-pinned) CPU==GPU identity green on
  both MoE models over 128 greedy tokens; bench md5 unchanged.

- Fixed the full-offload CPU==GPU byte-identity regression the Blackwell
  box found in the P1 MoE path (near-tie flip at ~token 60 on
  Qwen3-30B). Isolated with a new routing-bits discriminator: expert
  SELECTION was identical, but selw differed in the last mantissa bit
  from two compounding 1-ulp sources in k_moe_route — device expf vs the
  host libm, and exact IEEE division vs the -freciprocal-math
  reciprocal-multiply the -ffast-math host build actually emits. The
  kernel now computes the softmax exponential as
  (float)exp((double)x) — the correctly-rounded float exp, which
  bit-matches a correctly-rounded host expf (verified against UCRT on
  20M sampled inputs; residual double-rounding probability ~2^-28 per
  call) — and mirrors the reciprocal-multiply normalization. Verified on
  the 3070: fused output byte-identical to the eager (v0.1.4-certified)
  path over 128 greedy tokens on BOTH MoE models at ngl 17 and 19, and
  CPU==GPU identical; layer-1 routing bits equal, which transitively
  certifies every P1 kernel in layer 0's pipeline. Caveat for the
  full-offload re-cert: if the Blackwell CPU build auto-vectorizes the
  host softmax through libmvec's ~4-ulp expf, its CPU routing bits are
  unmatchable from device code — the new RUNNER_DEBUG_MOE dump (hex
  sel/selw from both paths) + RUNNER_MOE_EAGER (force the v0.1.4
  host-routing path) discriminate that case in two runs.
- (Q4_0, gemma4-moe) tc-tol failure investigated: the tail-safe K loop
  is NOT the cause — a synthetic n_ff=704 Q4_0 model (the exact
  X.5x128-step class gemma exercises) passes at 0.00004 of range with
  0/64 flips, and a 10x-weight variant scales the deviation smoothly
  (0.00022, still 0 flips), pointing at gemma-4's activation magnitudes
  under fp16 tile staging rather than a kernel defect. The gate is doing
  its job; the row stays unpromoted. If that row should ever pass:
  per-column activation absmax scaling in the TC tile is the identified
  follow-up.

- Tensor-core GEMM twins for Q8_0 and Q4_0 (moe-gpu-routing spec P3),
  same MMQ-style structure as the Q4_K TC kernel — block-cooperative
  fp16 weight-tile dequant with per-element values matching the scalar
  kernels exactly, fp32-accumulated m16n16k16 MMAs — plus a tail-safe K
  loop (gemma-4-MoE's n_ff_exp=704 is not a 128-multiple; elements past
  n_in stage as zeros). Both OPT-IN behind RUNNER_CUDA_TC and the
  per-(type, arch) test-tc-tol gate; tc_promoted() is unchanged, so no
  default path moved. test-tc-tol now recognizes all TC-capable types
  (was: hard-skip without Q4_K). Fresh gate rows measured on the 3070
  (full offload, 64 teacher-forced positions, 0 top-1 flips each):
  (Q8_0, qwen3) 0.00005 of range, (Q8_0, phi3) 0.00002, (Q4_0, qwen3
  requantized) 0.00003 — all far under the 0.005 bound. Promotion
  remains the owner's decision with the Blackwell rows.

- MoE GPU prefill: expert-grouped GEMM (moe-gpu-routing spec P2) — the
  CUDA port of the CPU `cabdad1` grouping. A prefill tile is routed on
  device, its routing read back once (prefill is never graph-captured),
  and each active expert then runs ONCE over all its routed tokens with
  the batched k_gemm/k_mv_b kernels — expert weights stream through the
  SMs once per tile instead of once per (token, slot); the per-token
  sync+DtoH per MoE layer collapses to one per layer per tile.
  Accumulation mirrors the CPU grouped path (ascending expert index,
  routing weight at the scatter). gemma-4's dense shared branch now also
  batches across the tile instead of looping per token.
  Follow-up measured on the 3070: the fixed-width GEMM kernels compute
  all 16 tile columns whatever the batch, and a 16-token tile routes only
  ~1-2 tokens per expert, so naive grouping LOST GPU time (gemma Q4_0
  prefill −19%). Expert matmuls now pick the narrowest kernel that
  covers the token count (batch-1 GEMV at 1, the width-classed f_gemvb
  twins to 8, the full GEMM beyond — TC whenever promoted/forced), and
  grouping engages only for expert types with width-classed kernels
  (Q8_0/Q4_K/Q5_K/Q6_K); gemma-4 Q4_0 keeps the per-token fused prefill
  until a batched Q4_0 kernel exists. 3070 result: gemma regression
  erased (14.8 tok/s prefill, above the fused path), qwen at end-to-end
  parity with GPU-busy still 1.65× the fused path's — the grouped win at
  this tile size has to come from TC on the expert GEMMs; the Blackwell
  box should A/B fused-vs-grouped prefill when re-measuring.
  Certified: CPU==GPU
  greedy byte-identical (short 128-tok and 510-tok-prefill configs on
  Qwen3-30B; short config on gemma-4-26B), pinned-b10076 text unchanged,
  bench md5s unchanged, make test green. Noted: gemma-4-26B long-prefill
  CPU-vs-GPU greedy divergence on a pathological repetitive prompt
  pre-exists in v0.1.4-alpha (P2's GPU output is byte-identical to the
  eager path's there — no regression; tracked in the suite plan).

- MoE GPU decode: device-side routing + fused indirect expert matvecs
  (moe-gpu-routing spec P1). Softmax → top-k → renormalize now runs in a
  serial-per-token device kernel that mirrors the host reference bit for
  bit (same scan/summation order, ties to lowest index), and one indirect
  launch per projection covers all top-k experts, reading `sel[]` on
  device — the per-token `cuStreamSynchronize` + DtoH round-trips are
  gone (~48/token on Qwen3-30B) and launches per decoded token fall
  ~4× on the MoE fixture (52.0 → 1.3 with the graph, see below). With no
  host-dependent branching left, fused-layout MoE no longer forces
  `graph_bad`: full-offload MoE decode is CUDA-graph captured. The
  legacy split-expert layout and expert quant types without an indirect
  kernel (outside F32/F16/Q8_0/Q4_0/Q4_K/Q5_K/Q6_K) keep the eager path
  unchanged. gemma-4's dual-branch routed experts use the same device
  routing (per-expert down scales uploaded per layer). Certified on this
  box (RTX 3070, partial offload): CPU==GPU greedy byte-identical over
  128 tokens on Qwen3-30B-A3B-Q4_K_M and gemma-4-26B-A4B-it-Q4_0;
  Qwen3-30B greedy text unchanged vs the pinned b10076 comparison;
  `make test` + `test-moe` green; bench.sh md5s unchanged.

## v0.1.4-alpha — 2026-07-29

### Headline: tensor-core prefill by default, published benchmarks, and the European roster

The tensor-core prefill GEMM is now the **default** on seven
tolerance-gated dense (Q4_K, arch) combos (+47–77% prefill, decode
unchanged), backed by a new teacher-forced tolerance gate
(`make test-tc-tol`); the decode GEMV bandwidth pass lifts dense decode to
73–79% of llama.cpp on the reference box, and the first head-to-head
benchmark is published (`docs/benchmarks.md`) with the losing rows
included. Six European models join the SHA-pinned compatibility manifest
under the new Europe & US model-scope policy (`docs/model-scope.md`) —
whose evidence runs found and fixed three real defects (a silent MoE
GPU→CPU fallback, a fast-math expf-overflow UB in CPU silu, and two
option footguns) and reported a GGUF conversion bug upstream to
OpenLLM-France. Full detail below.

- Fixed CPU decode corruption on models with extreme FFN gate values
  (found by TildeOpen-30b's certification run): silu computes expf(-g),
  fp32 expf overflows past ~88, and the -ffast-math build treats that
  overflow as UB — the auto-vectorized libmvec expf returned garbage for
  TildeOpen's last-layer gates (|g| up to ~2.7e3), degrading every
  pure-CPU decode step into <unk> emissions while GPU (CUDA expf
  saturates properly) was unaffected. gated_act now short-circuits silu
  to 0 below g = -80, where |silu| < 1.5e-33, so expf never sees an
  overflowing argument. Verified: TildeOpen CPU==GPU byte-identical over
  128 greedy tokens, greedy_reference 4/5 vs pinned b10076 (was 1/5),
  and lucie/eurollm CPU outputs bit-identical pre/post fix (the guard is
  a no-op for in-range models). RUNNER_DEBUG_ACT=N now dumps the N-th
  forward pass (was: first only) — the instrument this debug needed.
- `--gpu-layers 0` now means what it says — no GPU (same as `--gpu off`).
  It used to be the auto-fit sentinel and silently ran FULL GPU, a
  documented footgun that bit its own certification run: the roster's
  first cpu_cuda evidence used it as the CPU side and compared GPU with
  GPU. Re-verified with a true `--gpu off` CPU side: all five EU models
  and Qwen3-30B-A3B are byte-identical CPU vs GPU over 128 greedy tokens.
  Omit the flag for auto-fit.
- TildeOpen-30b added to the compatibility manifest (SHA-pinned): loads
  and generates coherently on GPU (llama arch, 60 layers, full 19.4 GB
  offload) — and its evidence run exposed an OPEN ENGINE DEFECT: the
  pure-CPU path emits <unk> for content tokens from the second generated
  token onward (batched CPU prefill is sane per the activation dump; the
  failure is single-token CPU decode, deterministic, independent of -b/-t).
  TildeOpen is the first model to trip it; its unique geometry — GQA 48:8
  (ratio 6), vocab 131072, n_embd 6144 — is the suspect surface. cpu_cuda
  and greedy_reference are recorded FAILED for TildeOpen until the defect
  is fixed; GPU serving is unaffected.
- Certified the European roster into the compatibility program: EuroLLM-9B,
  Lucie-7B, Mistral-Nemo-12B, Teuken-7B and salamandra-7b are SHA-pinned in
  `tests/compatibility/models.json` with a recorded evidence run — all five
  pass load, cpu_cuda (128-token greedy byte-identical, scalar path), chat
  and tool, plus the 8-token greedy_reference sweep against pinned llama.cpp
  b10076 (salamandra 5/5 exact; the others show the same divergence class as
  the long-certified models). Honest gaps recorded rather than skipped:
  Lucie's tokenizer FAILS the 721-string differential (259 divergences) —
  root-caused to the GGUF, not the engine: the conversion exports Lucie's
  BPE tokenizer as SentencePiece with all 65,024 merge ranks flattened to
  -1000, so the reference tokenization is unreproducible from the file by
  any engine (Runner and llama.cpp b10076 are token-identical on it, and
  the vendor's own official GGUF shares the defect — an upstream
  conversion bug affecting every GGUF consumer of Lucie);
  EuroLLM's reference repo is gated and Teuken's has no tokenizer.json, so
  their tokenizer checks are not_executed; long_context was not run for the
  roster. `reference_compare.py` fixed en route: Runner rejects unknown
  model names now, so the harness asks each server for its served model id.
- Vendor sampling presets for four European families (each cites its
  source): `mistral-nemo` — Mistral's card is explicit that Nemo "requires
  smaller temperatures. We recommend to use a temperature of 0.3", so the
  0.7 `mistral` preset was actively wrong for it; `lucie` (temp 0.6 /
  top_p 0.9, generation_config.json) and `salamandra` (temp 0.6 /
  repetition_penalty 1.2, generation_config.json); `teuken` (temp 0.7 /
  top_p 0.95, model card usage example — the weakest citation grade, and
  marked as such). EuroLLM and TildeOpen publish nothing verifiable and
  deliberately stay on `generic`. Name matching requires BOTH "mistral"
  and "nemo" so NVIDIA's Nemotron cannot land on Mistral's temperature.
- Preset matching now runs over `general.name` PLUS the load path's
  basename (`sampler_ident`): quantizer metadata is unreliable — a real
  community salamandra GGUF ships `general.name` "snapshots" (the
  converter's HF cache directory) — and the filename still carries the
  family. All three resolution sites use the combined identity.
- **Promoted the tensor-core prefill GEMM to the default for gated dense
  (Q4_K, arch) combos** (owner decision on the tolerance-gate numbers):
  `llama`, `phi3`, `gemma4`, `qwen3`, `mistral`, `gemma3`, `smollm` — every
  row measured by `test_tc_tol` on real weights with 0/64 teacher-forced
  top-1 flips and ≤0.012% mean logit deviation. Measured effect on the
  Blackwell MIG: dense Q4_K prefill +47–77% with decode unchanged
  (llama-3.2-3b 263→438 tok/s, qwen3-4b 212→352). `qwen3moe` passed its
  gate too (0.216%, one near-tie) but stays opt-in: MoE routing amplifies
  fp16 noise ~86× over dense, and this promotion covers the dense family.
  Unmeasured archs (`qwen2`, `qwen35`, `stablelm`) remain scalar.
  `RUNNER_CUDA_TC=1` still forces the path on everywhere (how a gate
  candidate is measured), `=0`/`off` forces it off; unset now means "per
  the promotion table" instead of "off". `test_kv_tol` pins the GEMM path
  scalar (`gpu_tc_force(0)`) so its strict f16 CPU==GPU invariant keeps
  measuring the KV cache format, not the GEMM. NOTE for certification:
  on promoted combos, byte-identical CPU==GPU comparisons now compare TC
  against scalar — certify the scalar path with `RUNNER_CUDA_TC=0` or use
  the tolerance form (`test_tc_tol`); free-running greedy output was
  byte-identical TC-vs-scalar on all four models checked (512+128), but
  near-tie flips are possible in principle on other prompts.
- Added the TC tolerance gate (`tests/test_tc_tol.c`, `make test-tc-tol`),
  the promotion instrument the tensor-core plan required: teacher-forced
  logits over 64 positions, gated on top-1 agreement (near-tie escape as in
  the q8-KV gate) and a bounded mean logit deviation (≤0.5% of the mean
  logit range, computed over real logits — suppression sentinels excluded).
  Skips rather than passes when the TC kernel cannot engage (no GPU, no
  Q4_K tensor, or bit-identical logits meaning the kernel never launched).
  First measurements on the Blackwell MIG: llama 0.003%, phi3 0.004%,
  gemma4 0.012% of range with 0/64 flips; qwen3moe 0.216% with 1/64 — a
  near-tie at 0.001 of range. All four pass; the qwen3moe free-running
  divergence is thereby classified as near-tie amplification (the q8-KV
  class), not decisive error. Adds `gpu_tc_force()` so one process can
  compare both paths; `RUNNER_CUDA_TC` env behavior is unchanged.
- Fixed a silent MoE GPU→CPU fallback introduced by the `--cpu-moe` binding
  layer (active even without the flag): `binding_find` bounds-checks
  `t->nbytes` on every dispatch, but the per-expert slice descriptors built
  by `moe_expert_weight` and the gemma-4 fused `gate_up` slice kept the full
  multi-expert tensor size, so expert `e >= 1` failed the check, `enc_mv`
  returned false, and the whole forward silently ran on the CPU while the
  load banner still reported a full GPU split. Clamping the slice `nbytes`
  restores the GPU path. Measured on the Blackwell MIG 1g.24gb:
  Qwen3-30B-A3B-Q4_K_M decode 4.5 → 76.3 tok/s (above the 56.5 pre-regression
  rate — expert matvecs now also use the new decode GEMVs);
  gemma-4-26B-A4B-Q4_0 6.9 → 23.5 tok/s (restored). Greedy output verified
  token-identical between CPU and GPU on both models. Note the CPU/GPU
  identity gates could not catch this class of defect: the fallback *is* the
  CPU path, so outputs matched while decode ran up to 12× slow.
- CUDA decode matvec pass (the suite plan's P1 decode lever): the Q4_K and
  Q5_K decode GEMVs now use aligned 8-byte quant loads, `float4` activation
  loads and a factored per-group affine; Q8_0 covers four blocks per warp
  iteration; Q6_K unrolls two blocks for load-level parallelism. Measured on
  an RTX 3070: Qwen3-8B-Q4_K_M decode 31.7 → 53.3 tok/s (+68%),
  Llama-3.1-8B-Q5_K_M 31.0 → 54.0 tok/s (+74%), Qwen3-4B-Q8_0 58.4 → 60.5
  tok/s. GPU output remains token-identical to the CPU path on all verified
  models.
- Prefill matvec tiles widened from 8 to 16 tokens (MVB 16), halving the
  per-tile weight passes. The Q8_0 prefill GEMM keeps its proven 8-column
  tile and runs wide tiles as two launches. Known cost on the 3070:
  Qwen3-4B-Q8_0 prefill ~-4% (113.5 → 108.7 tok/s) from extra attention-score
  L2 pressure at 16 columns; Q4_K prefill is unchanged on the default path.
- Rebuilt the opt-in tensor-core prefill GEMM (`RUNNER_CUDA_TC=1`) as an
  MMQ-style kernel: the block dequantizes a 64-row × 128-K fp16 weight tile
  to shared memory once — 8-byte quant loads, two threads per row — and four
  warps' m16n16k16 MMAs reuse it against a 16-token fp16 activation tile with
  fp32 register accumulation. The previous per-warp variant measured 6-7×
  slower than the scalar GEMM; this one measures Qwen3-8B-Q4_K_M prefill
  96.4 → 138.2 tok/s (+43%) on the RTX 3070, and its greedy output matched
  the CPU path token-for-token on the verification prompts. It remains
  opt-in behind the tolerance-gate promotion decision.
- Added sparse MoE tensor-role placement with `--cpu-moe`. CUDA retains
  attention/dense tensors and KV while expert FFNs execute from system RAM;
  packed uploads omit the expert bank instead of reserving GGUF-sized holes.
- `--caps` now advertises `tensor_placement.cpu_moe` for schedulers.
- Current Qwen3.5 GGUFs that include declared NextN/MTP blocks in
  `block_count` now load only the autoregressive backbone.
- Added native Qwen3.5/Ornith CUDA execution for recurrent Gated DeltaNet and
  full-attention blocks, including causal convolution/state kernels, gated
  attention, partial offload, pre-forward state snapshots for correct runtime
  CPU fallback, and compatibility with both `ssm_dt` tensor spellings.

## v0.1.3-alpha — 2026-07-24

### Headline: sparse Mixture-of-Experts (MoE) support

The runner now runs real sparse **mixture-of-experts** models — the class the
field is converging on for modest-VRAM hardware — on CPU, fully on the GPU, and
with **partial CPU offload for cards smaller than the model**.

- **Architectures:** Mixtral-style `llama`-with-experts and `qwen3moe`
  (Qwen3-MoE). Both the modern fused 3D expert tensors and the legacy
  split-per-expert layout (older Mixtral GGUFs) are supported by one accessor —
  no forward-path branch on layout.
- **Qwen3-30B-A3B (Q4_K_M, 128 experts, top-8)** loads in **18.6 GB**, fits a
  **24 GB MIG slice on an NVIDIA RTX PRO 6000 Blackwell** with ~6 GB free, and
  decodes at **~55 tok/s on that hardware**. This is not presented as a
  representative result for every 24 GB consumer GPU. Greedy GPU output is
  **token-identical to Runner's CPU path on the same quantized GGUF**.
- **Partial CPU offload (8–16 GB cards):** the runner fits as many leading
  layers on the GPU as the VRAM budget allows and runs the rest on the CPU.
  Every configuration tested is token-identical to Runner's CPU path on the
  same quantized GGUF, or to the full-GPU quantized run where the model fits.
- **Q3_K GPU kernel (new):** Q3_K MoE now runs on the GPU. **Mixtral-8x7B
  Q3_K_M (20.4 GB) is fully GPU-resident on the Blackwell 24 GB MIG slice**,
  with GPU output token-identical to CPU.
- **Prefill throughput:** MoE prefill groups tokens by shared expert (batched
  per-expert matmul instead of one token at a time), ~5.6× the per-token CPU
  prefill rate. Decode is unchanged and bit-identical.
- **MXFP4 read support (gpt-oss format):** the OCP microscaling FP4 quant type
  (E8M0 × E2M1) is read and dequantized; validated against the real
  `gpt-oss-20b-MXFP4.gguf` (all 72 expert tensors read; a real row dequantizes
  to spec). *(gpt-oss as a whole needs architecture support to actually run;
  the MXFP4 tensors read correctly today.)*
- **Runnable == validated:** the loader refuses at load — rather than
  miscompute — shared-expert MoE (Qwen2-MoE / DeepSeek) and non-gemma GELU
  sparse MoE until each is validated on its own. Gemma-4's GELU dual-branch MoE
  is implemented and validated separately.

Correctness is checked with synthetic MoE configurations constructed to equal a
dense FFN (asserted token-identical in CI) and CPU/GPU agreement on real
quantized models. CPU/GPU agreement is an internal consistency check; independent
Runner-vs-llama.cpp comparison is handled by
`scripts/compare_llamacpp.py` when the same GGUF, hardware, and llama.cpp build
are available. See [`docs/moe-support.md`](docs/moe-support.md).

### Reliability & security hardening (July 2026 code review, RNR-###)

The release gate from the July code review is cleared, with the remaining
hardware-only Metal validation documented separately:

- Metal runtime fallback preserves the backend resource owner after
  `gpu_disable()`, so CPU fallback can keep using unified-memory KV buffers and
  `model_free()` never frees `MTLBuffer.contents` (RNR-001).
- Quantizer honors `general.alignment` and writes atomically (RNR-002/015).
- Load/scheduler lifecycle, an OOM tranche, and VRAM rollback; the
  OOM-as-truncated-prompt semantic bug is fixed (RNR-003/005/006/013).
- Architecture admission allowlist; unknown archs are experimental behind
  `RUNNER_ALLOW_UNKNOWN_ARCH=1` (RNR-004).
- GGUF typed getters validate type/sign/range/finiteness (RNR-010); one strict
  numeric parser for CLI + env (RNR-021); bounded CLI file reads (RNR-011).
- `load_cancel` is a C atomic (RNR-008); startup lease compares process
  start-time, not just PID (RNR-017); a drift gate guards the committed
  generated GPU headers (RNR-020).
- Python client: streamed `tool_calls` assembly + preserved `finish_reason`
  (RNR-016).
- `make test-moe` runs the synthetic MoE correctness suite in Linux/macOS CI;
  release packaging now checks tag, binary and Python versions, current release
  docs, changelog, and the generated `BUILD-INFO.txt` tag/commit before creating
  archives.
- CUDA compatibility is documented as NVIDIA Turing / compute capability 7.5 or
  newer, matching the embedded `sm_75` PTX target; older NVIDIA GPUs fall back
  to CPU.
- Windows `make test` now builds the prefix-cache and VRAM-registry tests as
  distinct `.exe` targets. Native file IDs and 100 ns last-write timestamps
  prevent an in-place GGUF edit from reusing a stale prefix within the same
  second; the VRAM and output tests are portable across Windows/POSIX.
- GPU header embedding is explicitly UTF-8/LF, so the generated-header drift
  gate is deterministic across Windows, Linux, and macOS.

### Agent conformance

- New agent-torture family, **schema-constrained selection from a large enum**
  (~50 labels) — the structured-labeling task small models fail by emitting a
  plausible near-miss; schema-constrained decoding forces an exact member.

### Notes

- `--gpu-layers N` forces N leading layers on the GPU; `--reserve-vram PCT`
  caps usage. Runner still binds loopback-only by default.

## v0.1.2-alpha — 2026-07-22

- Compatibility evidence: real OpenAI/Anthropic SDKs, LiteLLM, LangChain, and a
  llama.cpp reference matrix. Earlier phases: strict tool-call schema engine,
  streaming agent events, Responses + Messages APIs, shared weights, continuous
  batching, prefix caching, q8 KV cache.

## v0.1.1-alpha — 2026-07-19
## v0.1.0-alpha — 2026-07-17

- Initial public alpha: dependency-free C inference server for GGUF models
  (CPU/CUDA/Metal), OpenAI-compatible HTTP API, sampler-level JSON-schema
  enforcement.
