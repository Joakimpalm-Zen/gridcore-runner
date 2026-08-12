# Changelog

All notable changes to xyntetik-runner (formerly gridcore-runner). This project
is in **alpha**; the HTTP protocol and CLI may still change between alpha
releases. Entries below the rename keep the names that were true when they
were written.

## v0.1.16-alpha — 2026-08-12

Four days, three overnight measurement campaigns, and one external
evaluation's worth of findings. Grouped by what a user notices first.

### Correctness fixes, some of them shipped defects

- **CUDA tensor-core prefill for Q8_0/Q4_0 computed 16 of its 64 token
  columns** and published uninitialised shared memory as logits for the
  rest, on the DEFAULT path, since the 2026-07-29 batch widening. Found by
  measurement, fixed, and the gate that had passed the broken kernel was
  itself fixed: `test-tc-tol` now carries a free-running arm at a
  production context, verified to fail against the broken kernel. The
  cpu/cuda identity corpus now crosses the 16/32/64-column tile
  boundaries; its old longest prompt was one token short of ever seeing
  the bug.
- **Truncated tool calls reported `finish_reason:"tool_calls"`** on all
  three API dialects, converting the loud failure this engine exists to
  prevent into a silent one. A truncated call now keeps its truncation
  signal on every surface (external evaluation finding A).
- **`--quantize` copied `general.file_type` from the parent**, mislabeling
  mixed-retention outputs. The declared type now derives from the output
  histogram (finding B).
- **An unrecognised chat template silently fell back to llama2 markup.**
  gpt-oss chat was fed `[INST]`/`<<SYS>>` it had never seen and ran away
  past every stop; the fallback now warns loudly — and gpt-oss no longer
  needs it (see Harmony below).
- The gpt-oss tokenizer differential went **222/721 divergent strings to
  0/721** by mapping its `gpt-4o` pre-type onto the o200k splitter.
- A sweep-and-review pass fixed a heap over-read in the atem parser
  (ASan), bound-aware recovery for truncated numeric arguments,
  closest-prefix recovery for truncated enums, thread-safety of quant
  arithmetic under fast-math (now confined to a `-fno-fast-math`
  translation unit), and `system()` calls in the tray (now `posix_spawn`).

### New capabilities

- **Muse native atem tool calling**, constrained by the schema compiler:
  tool definitions rendered in the model's own format, generation
  constrained from the recipient header, truncation-surviving closes,
  multi-call turns, buffered and SSE parsing, and the model's own
  protocol control tokens satisfying the automaton so generation stays
  on-distribution. Certified on the real model, 120/120 torture matrix.
- **TMPL_HARMONY**: gpt-oss chat renders its real format — channel-aware
  turns, developer role, its own stop set — with the analysis channel
  suppressed from content and surfaced as `reasoning_content`,
  `enable_thinking` as the control. Before/after transcripts in
  `docs/gpt-oss-harmony-2026-08-14.md`.
- **Multi-part GGUF loads natively.** Standard numbered split sets map
  through independent mappings; missing or inconsistent sets refuse
  loudly; a real split model reproduces the whole file's output SHA.
- **Streaming parallel tool calls** on the generic JSON envelope path;
  the `parallel_tool_calls`+`stream` refusal is gone.
- **VRAM registry priorities and cooperative yield**: advisory priority
  tags on claims (`--vram-priority`), priority-ordered `--wait-for-vram`
  acquisition, and `--yield-on-request` — a holder can be asked, never
  forced, to release at an idle point.
- **`--type-plan` per-tensor precision in the quantizer**, integrity-gated
  (untouched tensors byte-identical to the source). This is the tool
  behind the selective-precision artifact class: per-tensor-class
  precision is the finest GGUF can express (experts are stacked), and it
  is enough — see Artifacts.
- **Quality bar v2**: gate tooling reports margin-qualified top-1
  (0.5-nat reference-side tie band, derived from the tc-tol precedent)
  beside plain top-1, both always printed. Adopted as the publication
  criterion 2026-08-14.
- **`scripts/quant-fidelity.py`**: the quant-vs-tool-call-fidelity
  harness — per-quant schema conformance, tool selection, argument
  agreement and KLD against a reference variant, self-validating
  zero-point that refuses to measure on any disagreement.
- **Machine-readable compatibility ledger**: `docs/compat-reports/`
  carries dated per-release reports; the manifest schema now declares
  executable contracts for cpu_cuda/chat/tool check classes, and checks
  that did not run say so with reasons.

### Performance, measured against llama.cpp on the same hardware

- **CPU decode +32-34%, byte-identical**: the thread pool spent
  65-138 us per matvec handoff (38-59% of every decode token);
  spin-then-park wakeups fixed it. Dense CPU decode moved from 50-60% to
  66-78% of llama.cpp.
- **CUDA prefill up to 28.8x on previously uncovered combos**: Q4_0
  promoted through the tolerance gate across four architectures, granite
  admitted to the TC path, and Q4_0 gained its missing coalesced decode
  GEMV (4.4-5.4x, token-identical). Honest same-slice table: decode
  77-87%, prefill 6.1-9.8% of llama.cpp (`docs/benchmarks.md`).
- **Metal prefill +23.6% at defaults on M1**: half threadgroup staging,
  64x32 tiles, memory-aware batch default. The matvec multi-row rework
  was measured NEGATIVE under the byte-identity contract and not shipped
  (`docs/negative-result-metal-multirow-matvec.md`); a latent activation
  alignment hazard was fixed on the way.
- A fused int8 CPU dot ships gated OFF (`RUNNER_CPU_I8=1`): 2.4-2.5x in
  kernel isolation, ~6% end to end, and it flips near-tie tokens, so it
  did not meet the 0/64 promotion bar.

### Artifacts and measured claims

- **New flagship: Qwen3-30B-A3B selective precision** (attention Q8_0 /
  experts Q4_0, 17.99 GB) — PASSES the adopted bar where the official
  uniform Q4_K_M fails it, from a byte-verified first-party Q8_0 source.
- **Qwen3-Coder-30B keep-120** passes both the original and the current
  bar — the only published artifact to clear the original bar unaided.
- **gpt-oss-20b keep-30's quality claim is withdrawn**: the published
  97.5% does not reproduce at its own protocol on byte-identical files.
  The file stays published as a measured near-miss; its card leads with
  the current numbers.
- **The 4-bit size threshold**: Q4_K_M passes the bar at 8B and 14B,
  fails at 5B and below, on everything measured. Legacy Q4_0 is 6.7x
  worse than Q4_K_M on the same 14B model. Ship k-quants, never Q4_0.
- **The split story, measured on two families**: constrained decoding
  holds schema conformance and tool selection at 100% down to Q4_0 while
  argument agreement decays to 50%. The shape of a tool call is
  quant-proof; its contents are not (`docs/quant-fidelity.md`).
- Quickstart now recommends granite-4.1-3b Q8_0 (3.6 GB, the smallest
  model passing the fidelity gate against its own BF16).

## v0.1.15-alpha — 2026-08-11

- **New architecture: `muse-glimmer` (Meta Muse Glimmer 30B, text path),
  certified.** Gated attention, per-head QK norms, sandwich norms at a fixed
  1e-8 epsilon, a sliding-window pattern array whose full-attention layers are
  NoPE, and a scaled + softcapped logit head — transcribed from the reference
  implementation and gated on real weights: tokenizer differential 0/721,
  CPU/CUDA 6/6 byte-identical at full offload, greedy reference identity at
  the model's measured sensitivity floor, chat with the reasoning turn split
  into `reasoning_content`. Evidence: `docs/muse-glimmer-cert-2026-08-11.md`.
  The separate vision encoder and the atem tool syntax are not implemented.
- **New tokenizer pre-type: `llama4`** (the o200k family) — cased letter runs
  with attached case-insensitive contractions, three-digit runs, and `/` in
  the punctuation tail. Previously these files silently fell back to GPT-2
  split rules.
- **New chat template: `muse`** (`<|start|>role<|message|>…<|eot|>`), with
  `<|eot|>` added to the stop-token probes and `enable_thinking:false`
  pinning the generation prompt to a direct answer.
- **Metal: dense gated attention and NoPE layers now run on the GPU** (new
  `k_sigmoid_mul` kernel; NoPE layers skip the rope dispatch). CUDA gains the
  same dense attention gate. Gate-plus-MoE models (afmoe) deliberately keep
  their CPU fallback — that combination has not been run against the
  identity gates.
- **Codebook i-quants: IQ1_S/M, IQ2_XXS/XS/S, IQ3_XXS/S now load and run**
  (CPU only; CUDA and Metal refuse them loudly), with NEON and AVX2 dequant
  kernels — measured 2.9x decode / 2.5x prefill on a 30B IQ3_XXS versus the
  scalar path. Differentially gated against the reference engine's own
  quantizations of the test fixture plus independent double-precision
  decoders in `test_quants_simd`. `--caps` advertises the new formats.
- **New tools:** `scripts/gguf-depth-slice.py` (drop whole transformer layers
  from a GGUF with per-layer metadata arrays filtered to the survivors; its
  fixture gate compares surviving tensor bytes) and the `RUNNER_LAYER_SIM`
  diagnostic (per-layer residual input/output cosine, the measurement a
  depth-prune plan needs).

- **The project is renamed: Gridcore → Xyntetik.** The repo is now
  `Joakimpalm-Zen/xyntetik-runner` (GitHub redirects the old URL), the Python
  client distribution is `xyntetik-runner-client` with import package
  `xyntetik_runner`, and the engine binary is unchanged in behaviour.
- **State directory moved with a one-time migration.** `~/.gridcore/runner/`
  (POSIX) and `%APPDATA%\gridcore\runner\` (Windows) become `~/.xyntetik/…` /
  `%APPDATA%\xyntetik\…`. On first start after the upgrade the old tree —
  `config.json`, `managed.log`, the instance registry — is moved to the new
  path if the new one does not exist yet, so a live install keeps its config
  and registry.
- **Tray registrations migrated.** macOS: the LaunchAgent label/file
  `ai.gridcore.runner.tray` is unloaded and re-created as
  `ai.xyntetik.runner.tray`, preserving autostart. Windows: the `HKCU\…\Run`
  value `GridcoreTray` is re-registered as `XyntetikTray` and the old value
  deleted; the window class is now `XyntetikTrayWnd`.
- **BREAKING: environment variables renamed without aliases.**
  `GRIDCORE_TRAY_DUMP` → `XYNTETIK_TRAY_DUMP`, `GRIDCORE_TRAY_ICON_DUMP` →
  `XYNTETIK_TRAY_ICON_DUMP` (and the test-only `GRIDCORE_TEST_HOME` →
  `XYNTETIK_TEST_HOME`). Old names are ignored; update shell profiles.
- **Unchanged, deliberately:** the `gridcore.agent.*` GGUF metadata keys and
  the `gridcore` sampler-preset family (matching `general.name`
  `"gridcore-<size>"` / `"Syntetik…"`) — both are contracts with already
  published model artifacts, and the model project's rename is sequenced
  separately. The VRAM registry filename prefix is now `xyntetik-vram-`; a
  stale `gridcore-vram-*.reg` from an older build is simply ignored.

## v0.1.14-alpha — 2026-08-10

- **Metal stops falling back to the CPU on the formats small Macs actually
  use.** `--caps` advertised IQ4_NL and IQ4_XS as GPU-capable while the Metal
  backend accepted neither, so an Apple Silicon user got a silent full-CPU
  fallback on exactly the quants an 8–16 GB machine reaches for. Five matvec
  kernels are new — q2_K, q3_K, iq4_nl, iq4_xs and bf16 — each byte-identical
  to the CPU path, and `--caps` now reports Metal's real support.
- **Metal prefill went from a regression to a large win.** The new matvec
  kernels sped up decode but left prefill on a fallback that re-reads every
  weight once per token column, which was slower on the GPU than on the CPU.
  Tiled GEMM for the same five types fixes it (prompt tok/s, M1):

  | model | CPU | Metal (matvec) | Metal (tiled) |
  |---|---:|---:|---:|
  | tinyllama-1.1b Q2_K | 36.91 | 28.20 | **192.10** |
  | SmolLM2-360M IQ4_XS | 101.21 | 87.50 | **445.80** |
  | SmolLM2-135M Q2_K | 285.79 | — | **892.83** |
  | SmolLM2-135M bf16 | 296.23 | — | **848.69** |

  0.76x against the CPU becomes 5.20x, and 0.86x becomes 4.40x.
- **f16 and bf16 decode now beat the CPU instead of losing to it.** Both loaded
  one weight per lane step while every quant kernel already loaded wide; four
  per step takes f16 from 0.91x to 1.03x and bf16 from 0.91x to 1.04x on an M1
  (135M model, medians of three interleaved runs), output unchanged.
- **CUDA: Q8_0 runs on tensor cores by default.** Prefill goes 113.72 → 475.71
  tok/s, 4.18x, on Qwen3-4B-Q8_0. The logits are bit-identical across 504
  counted tensor-core dispatches rather than merely within tolerance: int8
  weights convert to fp16 without loss and the accumulation order matches the
  scalar path, so there is no numerical difference for an architecture to
  amplify. The kernel already existed and was simply never dispatched. Measured
  on sm_86.
- **Structured output supports bounded repetition.** Schema patterns accept
  `{n}` and `{n,m}` alongside the two unbounded forms. The ceiling is enforced
  during decoding rather than at the closing quote, which is what the previous
  floor-only validator could not express.
- **Qwen3 can be asked not to reason.** A ChatML variant whose own template
  declares `<think>` carries a tri-state thinking control, because the
  references disagree; detection reads the model's template rather than
  matching names, so it follows the checkpoint.
- **`reasoning_limit` is no longer emitted on the OpenAI surfaces.** It is not
  in the OpenAI `finish_reason` enum; it maps to the standard `length`.
  `finish_detail` is now carried into streamed turns.
- **A thread count above the pool maximum is clamped loudly.** `-t 128`
  behaved exactly like `-t 64` with nothing on stderr; the caller asked for N,
  got 64, and had no way to find out.
- **Two gates were vacuous and are now real.** The VRAM leak gate could not
  detect the bug it exists for: a deliberate 4.1 GB leak, six times, on an
  8.59 GB card produced 0.0 MB of drift and a green result, because releasing
  the CUDA primary context reclaims every allocation inside it. It now pins the
  context, and treats a single catastrophic window as a leak rather than
  requiring both windows to lose memory — the two-window rule inverts when the
  first leak exhausts the card. The tensor-core tolerance gate now tells
  "matched exactly" apart from "never ran", which are bit-identical outcomes
  demanding opposite verdicts.
- **A gemma-4 prompt change was reverted after measuring it.** An empty
  thought-block pre-seed was added to the non-thinking generation prompt from a
  summary of the reference template; the construct does not exist in the
  model's own `tokenizer.chat_template`, and it cost 0.275. The original
  unconditional form was correct.
- **The build fails on discarded recipes.** "overriding commands for target" is
  never benign — it always means a recipe was silently thrown away, and it had
  been hiding a dead test target on every build. Windows warnings are
  actionable and the fast-math sentinel warnings are gone.
- New `scripts/weight-io-bench.py` reports mapped-fault versus explicit-read
  bandwidth for large weight files, with tests and `make test-weight-io-bench`.
- New gates cover the Metal f16 parity path, Metal K-quant index geometry, the
  prefix half-budget truncation clamp, reservation auto-fit arithmetic, and the
  KV-evicts-weights threshold.

Known limitation: `--caps` still describes CPU and GPU quant support with one
hardcoded list, which cannot be correct for both backends — CUDA has IQ4 and no
q2_K, Metal now has q2_K. It currently understates Metal, which is the safe
direction, but it is not accurate.

## v0.1.13-alpha — 2026-08-08

- **CUDA prefill is 82% faster: 199.5 → 363.4 tok/s** (RTX PRO 6000 Blackwell
  MIG slice, Qwen3-4B-Q4_K_M, ~3.1k prompt, three interleaved rounds each,
  spreads under 0.1%). Three independent changes:
  - **Prefill token tile widened 16 → 64 (+35%).** `MVB` was doing two jobs —
    the width every activation buffer is allocated for, *and* the compile-time
    tile baked into the scalar kernels (`float s[MVB]`,
    `__shared__ float xsm[MVB][256]`). At 64 the register array spills and the
    shared tile is 64 KB against a 48 KB cap, which is why this looked blocked.
    `MVT` is now the scalar tile and `MVB` the buffer width; fixed-tile kernels
    run `ceil(batch/tile)` launches, generalising the split `k_gemm_q8_0`
    already used. The epilogue tile aliases the weight tile (both 16 KB,
    disjoint lifetimes) to stay under the cap. The decoupling alone was worth
    +14% before any tile widened, because wider buffers widen the outer tile
    too and norms/rope/attention now run 64 tokens per pass instead of 16.
  - **Tensor-core Q6_K GEMM (+9.8%).** Only q4_K, q8_0 and q4_0 had `_tc`
    kernels, so `attn_v` and `ffn_down` in every `Q4_K_M` file — 40% of all
    GEMM calls — ran the scalar path. The new kernel is 2.35x faster on that
    work; unpacking is lifted verbatim from `k_gemm_q6_K`, including the
    integer `sc * q` before the float multiply, so it matches element for
    element.
  - **Attention V accumulation spread across the block (+22.8%).**
    `for (i2 = tid; i2 < hd/2; i2 += tpg)` used 64 of 128 threads at
    head_dim 128 while each walked the entire context serially. Each thread
    now owns a (lane, position-chunk) pair. Where lanes exceed the block
    (head_dim > 512, which gemma4's `key_length` default reaches) the original
    loop is kept — the chunked form would leave the upper dims unwritten.
  Decode is unchanged at matched context. CPU/GPU byte-identical on Qwen3-4B,
  gemma-4-E4B, SmolLM2-135M, Qwen2.5-7B and Llama-3.1-8B, including the
  sliding-window path.
- **The paging warning is MoE-aware.** It told a 16 GB Mac that every token
  would page from disk while gemma-4-26B-A4B was serving 8+ tok/s there. It
  now estimates the per-token hot set — everything except the *unrouted*
  expert banks — and reports both figures: 2.4 GB touched per token against a
  14.4 GB file. Dense models keep the original wording verbatim, and the
  `--mlock` hint is dropped when the hot set fits but the file does not,
  because pinning a file that does not fit is not the fix for that.
- **`--caps` publishes `gpu.eseries`**, so a placement scheduler can tell a box
  that runs gemma-4 E-series on the device from one that silently falls back.
  Follows `gpu_moe_ok()` exactly: Metal true since 0.1.11, CUDA true, stub
  false.
- **The tray re-reads an externally edited `config.json`.** A hand-edited model
  path or port needed a tray restart; `cfg_load()` now stats the file and
  re-reads on change, with size compared alongside `st_mtime` because
  whole-second timestamps miss an edit landing in the same second as the read.
- **A split (multi-part) GGUF is refused by name.** Handed part 1 of a 3-part
  model, runner bound whatever that part held and printed a wall of
  `missing tensor blk.N...` while `split.count` sat in metadata it had already
  parsed. It now says which part, out of how many, names the expected set and
  points at `llama-gguf-split --merge`. `scripts/gguf-split.py` produces
  fixtures in llama.cpp's layout.
- **`make test` no longer dies on dash.** `test-shader-embed` used a bash
  here-string, so the suite failed at parse time on Debian/Ubuntu — i.e. most
  Linux. CI never caught it because CI does not run `make test`.
- Metal fixture coverage for the geometries and MoE layouts nothing pinned:
  dense gemma3/gemma4 with and without SWA, qwen3 SWA, E-series,
  gemma4-hetero, and identical-or-refused-loudly for the split expert layout
  and the shared always-on expert. `--arch gemma4` fixtures did not load at
  all before this — and because both arms failed identically, a CPU-vs-Metal
  comparison of their empty output reported a match.

## v0.1.12-alpha — 2026-08-07

- **A bare `runner` now starts the tray** (macOS, Windows). Double-clicking
  the binary — or typing `runner` with no arguments in a terminal — launches
  the menu-bar controller instead of printing help text that vanishes with
  the window. The conditions are deliberately narrow, and everything else is
  untouched: literally zero arguments (every scripted invocation passes
  something, so none can be affected), stdin *and* stdout must be a real
  terminal (pipes, CI, and probes still get usage text and a nonzero exit),
  and platforms without a tray backend (Linux) keep the old behavior
  exactly. `--no-tray` is the standing opt-out for wrappers that want
  bare-invocation behavior guaranteed text-only, whatever the conditions
  above may become later. Pinned by `make test-bare-invocation`, which runs
  the piped-stdin case — a GUI event loop there would hang every script that
  probes the binary, including the test itself.

- **MoE expert prefetch: the default is now decided per machine class, and
  Apple Silicon earned ON.** v0.1.11 shipped it off pending a second machine
  agreeing. The second machine agreed the next morning: a 16 GB M2 Pro at
  3.2x oversubscription measured decode 2.68 → **3.37 tok/s (1.26x)** and
  prompt 1.65 → **2.54 (1.54x)**, three interleaved rounds, the arms fully
  disjoint — the worst prefetch-on run beat the best prefetch-off run on
  both metrics. Since the M2 Pro ran at *higher* oversubscription than the
  Linux null (3.2x vs 2.24x), the differentiator is the storage class's
  fault cost, not the oversubscription ratio. So: **auto = on for Apple
  Silicon when weights exceed available RAM** (the two-machine measured
  case), **off everywhere else** (the measured null). The Linux default
  changes when someone measures a win there, not before.

- **`--moe-prefetch on|off|auto` — the opt-in becomes a real flag.** Field
  report from the 16 GB deployment: `RUNNER_MOE_PREFETCH=1` set in a shell
  profile does not survive the tray's login LaunchAgent, which runs with
  launchd's environment — so after a reboot the same server came up ~26%
  slower with nothing in the banner chain to say why. A measured win that
  silently evaporates on relaunch reads as a regression. The flag persists
  through the tray's `last_args` like every other knob; precedence is flag
  over env over the per-class default, pinned by tests including both
  override directions.

## v0.1.11-alpha — 2026-08-06

- **MoE expert prefetch: routed experts are handed to the OS as whole blocks.**
  On a model larger than RAM the engine's cost is not bandwidth and not
  arithmetic — it is the NUMBER of I/O operations. Reaching an expert through
  the mmap costs ~200 synchronous 16 KB faults; measured on gemma-4-26B on an
  8 GB M1 that is **~17,000 faults per token at ~45–60 µs each**, which is the
  entire token budget (4x the threads bought 1.32x, and warming the page cache
  halved the faults without moving throughput at all). The router has just
  named the experts this layer will read, so those byte ranges now go to the
  OS as whole blocks. Measured, CPU, weights ~4x available RAM:

  | model | prefill | decode |
  |---|---|---|
  | gemma-4-26B-A4B QAT Q4_0 | 2.61 → **4.33** tok/s (1.66x) | 1.37 → **1.96** tok/s (1.43x) |
  | gpt-oss-20b keep-30 MXFP4 | 0.42 → **0.80** tok/s (1.90x) | 0.61 → **1.04** tok/s (1.70x) |

  **It cannot change output.** The advice is purely about how many faults a
  read costs, never what it returns — verified byte-identical on every MoE
  fixture with it forced on and off.
  **It is not a cache**, deliberately: no residency set, no eviction, no
  hit-rate policy. A standalone probe replaying real routing traces against the
  real GGUF measured cross-workload committee hit rates at 13.7% (top-8) and
  24.8% (top-16) — caching contributes little and does not generalize, while
  granularity alone captures the win.
  **It is architecture-agnostic.** The only thing it takes from the model is
  the list of expert ids a router just produced. It is fed by *whichever*
  router ran, so it engages on gemma-4's dual-branch MoE and gpt-oss's generic
  path alike — unlike the rejected `--expert-cache` tier, which hooked one
  specific FFN path and therefore never engaged on gemma-4 at all
  (`docs/negative-result-expert-cache.md`).

  **It is OFF by default, and the table above is why it does not get to be
  on.** Every number in it is Apple Silicon, far oversubscribed, on slow
  storage. Re-measured on Linux with XFS at 425 MB/s and 2.24x
  oversubscription, against a RAM cap verified to hold, the feature is
  **neutral to slightly negative** — 0.22 tok/s off versus 0.21 on, with
  marginally *more* major faults enabled. That is coherent (fewer faults to
  save when faults are cheap), but it means the win is a property of one
  machine class rather than of the technique, and one machine class is not
  grounds for a default. `RUNNER_MOE_PREFETCH=1` opts in, `=0` forces off.
  It gets the default back when a second machine class agrees, with the
  number attached.


- **Metal prefill is batched: 3.0x on the prompt path, and the Gemma
  families get it too.** Issue #6 — prefill ran at decode speed because
  the prompt batch encoded per-token chains: every weight matrix was
  re-streamed once per token, and each token cost ~10 tiny dispatches per
  layer. The batch now issues one dispatch per layer for each of rmsnorm,
  the Q/K/V/O and FFN matmuls, rope, KV store and attention. Kernels gained
  an explicit column dimension with strides passed in (the scratch is not
  uniformly strided — `xb` is strided by `xdim = max(q_dim, n_embd)`), plus
  a tunable `col_tile` (`RUNNER_METAL_COL_TILE`, default 8): one threadgroup
  per row across ALL columns maximizes cache reuse but serializes the batch
  and starves the GPU of threadgroups, so columns are tiled to keep
  parallelism while still amortizing each weight fetch. Per-element
  arithmetic and the `simd_sum` order are unchanged, so a batched prefill is
  bit-identical to per-token submits. Logits are no longer computed for
  every prompt position when only the last is read.
  Measured on an M1, SmolLM2-135M Q8_0, 601-token prompt, interleaved A/B
  against a binary built from the previous commit: **prefill 119 -> 358
  tok/s (3.0x)**, and Metal now beats the CPU path (273 tok/s) instead of
  losing to it by 2.3x. Decode is unchanged (~68 tok/s) — n=1 has nothing
  to batch.
- **A Metal shader-compile failure can no longer pass unnoticed.** It was the
  quietest failure in the backend: `gpu_init` printed a line, returned false,
  and the model ran correctly on the CPU — so a hand-run "CPU vs GPU" check
  compared the CPU path with itself and passed. That happened during this
  release's prefill work, and briefly made a fallback look like a speedup.
  Two guards: `make test` now runs a shader gate on macOS
  (`tests/test_metal_shaders.m`) that compiles the embedded library and looks
  up every kernel `mk_pipeline()` asks for — verified to fail on both a broken
  kernel and a renamed one; and `--caps` no longer reports a Metal backend
  when the library will not compile, because `--caps` exists to let a
  scheduler place work before dispatching and a device that cannot run is not
  a backend. A missing *function* was quieter still: `mk_pipeline` returns nil
  and the tiled-GEMM path silently degrades to the matvec path — a large
  performance regression with no wrong answer for a correctness gate to catch.
- **MoE and E-series join the Metal batched path; one encoder now, not two.**
  MoE layers were excluded because `enc_gemma_moe_ffn` read the residual at a
  fixed offset 0; both MoE encoders now take an explicit token offset, so
  attention, the projections and the norms run once for the batch while the
  MoE FFN stays per-token (routing picks different experts per token — there
  is no shared weight tile to amortize).
  **gemma-4 E-series runs on Metal at all for the first time** (a standing
  field report: "E-series gets no Metal path"). `gpu_init` used to refuse it
  outright. Implemented: the per-layer-embedding branch, and per-layer FFN
  widths — which together unblock E2B *and* E4B. Real gemma-4 E2B Q4_K_M is
  byte-identical CPU vs Metal, and prefills at 40.3 tok/s against the CPU
  path's 19.6.
  Building the PLE table needed care: `model_forward_batch` deliberately skips
  its own prepass under full offload because CUDA stages the table on-device,
  so Metal was reading a stale one; it now builds the table from the scaled
  embeddings it just wrote.
- **Fixed: shared-KV layers were silently wrong on Metal.** A gemma-4 E-series
  layer past `kv_from_start` owns no cache rows — it projects Q only and
  attends over the layer `model_kv_byte_off()` aliases to, exactly as the CUDA
  backend does. Metal projected K/V anyway and stored them, overwriting the
  source layer's rows. Real E-series models never reached it because the
  per-layer-embedding refusal turned them away first, so this had no symptom
  until that refusal was lifted — but a shared-KV-only model would have
  produced wrong output with no gate to catch it. Pinned by the new
  `make test-metal-eseries` (PLE only, shared-KV only, and both).
- **The per-token Metal encoder is gone.** Keeping a second implementation of
  the same layer loop cost three defects — features silently missing behind an
  eligibility check, a double-applied logit softcap, and wrong output on real
  gemma-4 E2B weights — each invisible to the gates until something else
  exposed it. One encoder now serves every batch size including decode; a
  scratch-allocation failure falls back to the CPU loudly instead of to a
  second, drift-prone path. `tests/test_eseries.py` goes from 9 passed /
  8 skipped to 17 passed: its GPU tests had been skipping because the backend
  refused the models.
- **Tiled prefill GEMM on Metal, behind the tolerance gate: 768 tok/s.**
  The matvec kernels give one output element per simdgroup — 32 lanes each
  doing one FMA, then a log-depth reduction — so almost none of the work is
  reuse. `k_mm_*` computes an output TILE per threadgroup with Apple's
  simdgroup matrix units (32 rows x 16 columns, k-step 32, weights loaded
  transposed straight out of threadgroup memory). Implemented for F32, F16,
  Q8_0, Q4_0, Q4_K, Q6_K and MXFP4; anything else keeps the matvec path.
  It is deliberately NOT bit-identical and cannot be — the weight is
  dequantized before the multiply, and the sum is reassociated into 8-element
  matrix steps — so, exactly like the CUDA tensor-core prefill, it answers to
  `tests/test_tc_tol.c` rather than to an identity claim it cannot honour.
  `gpu_tc_force()` drives it from the gate; `RUNNER_METAL_MM=0` pins the
  matvec path, and the Metal identity smokes now pin it (as the CUDA
  harnesses pin `RUNNER_CUDA_TC=0`), so each gate tests the path it claims.
  Measured, SmolLM2-135M Q8_0, 601-token prompt on an M1: **prefill 360 ->
  768 tok/s** on top of the batching below — **6.4x over the 119 tok/s this
  release started from**, and 2.8x the CPU path. Gate results on real
  weights: Q8_0 mean|dlogit| 0.00003 of logit range and 0/64 top-1 flips;
  gemma-3-4b Q4_K_M 0.00001 of range and 0/64 flips (limits 0.005 and 5%).
  The gate earned its keep immediately — it caught a genuine Q6_K nibble/byte
  indexing bug (14/64 flips with decisive margins) that greedy output and
  every identity smoke had missed.
- **The two Metal forward paths were merged.** The prompt-batch encoder
  duplicated the layer loop and silently lacked features, which an
  eligibility check papered over — so every Gemma model took per-token
  submits and got none of the above. It now implements embedding scale,
  the weightless V norm and V-less layers, sandwich norms, per-layer output
  scale, GELU and heterogeneous per-layer geometry. Real gemma-3-4b-it
  Q4_K_M: byte-identical CPU vs Metal, prefill 5.3 -> 7.2 tok/s on a
  paging-bound M1. Still on per-token submits, honestly: MoE layers (their
  encoders address scratch at a fixed offset 0) and E-series per-layer
  embeddings (no batched prepass).
- **Fixed: Metal applied logit softcap twice.** `model_forward_batch`
  applies head transforms on the host for both the batched and solo paths —
  by design, so they cannot drift — and the Metal encoder applied them
  again. Softcap is monotonic, so greedy argmax never saw it and no
  identity gate could fail; it distorted logprobs and any temperature/top-p
  sampling on softcapped models (the Gemma families). The backend-side copy
  is removed.

## v0.1.10-alpha — 2026-08-06

The Metal release: the Gemma families stop falling back to CPU on Apple
Silicon, and a latent fast-math defect that would have produced NaN on real
Gemma weights is fixed and pinned before anyone could hit it.

- **gemma-4 runs on Metal.** The heterogeneous-attention refusal is
  retired: the per-token Metal path already carried per-layer
  head_dim/KV-head/rope geometry, V-less layers (V from the raw K
  projection), the weightless V head-norm, sandwich norms, per-layer
  output scale and the logit softcap — it was gated off, never exercised.
  Now: scratch buffers size off the per-layer maxima, the dense-FFN
  activation selects GELU when the arch calls for it (unlocks dense
  gemma3/gemma4 class models past the activation gate), and the native
  prompt-batch fast path gained an explicit eligibility check naming the
  features it implements — models carrying anything beyond the plain
  llama shape (MoE, GELU, embedding scale, V norms, sandwich norms,
  out-scales, softcap/suppress, heterogeneous geometry) take the
  per-token path instead of a silently-wrong batch. Pinned by two new
  heterogeneous fixtures (dense + dual-branch MoE, both with V-less full
  layers) under `make test-metal-gemma4-hetero`, byte-identical CPU vs
  Metal.

- **Metal GELU produced NaN on real Gemma weights — fixed.** Enabling the
  path above exposed it immediately: gemma-3-4b-it emitted nothing but
  token 0, with NaN logits, from layer 0. Metal compiles kernels with fast
  math, where `tanh()` is evaluated through `exp(2a)`; the GELU tanh
  argument grows as x^3, overflows to `inf`, and `inf/inf` is NaN. libm's
  `tanhf` on the CPU oracle saturates instead, and CUDA is unaffected
  (nvcc is not built with `-use_fast_math`), which is why the arch was
  certified on CUDA with the defect latent on Metal. The kernel now clamps
  the tanh argument to a magnitude where the result is already exactly
  ±1.0f in fp32, so the guard cannot change any representable value — the
  same hazard and the same remedy as the `g < -80` early-out in the CPU
  silu path. Pinned by `make test-metal-gelu-overflow` (a fixture whose
  gate weights reach the overflow; verified to fail without the fix), and
  validated on real weights: **gemma-3-4b-it Q4_K_M is byte-identical CPU
  vs Metal over 32 greedy tokens** on an M1.

- `RUNNER_METAL_NAN_TRACE=1`: submit after each Metal layer and report the
  first buffer carrying NaN/Inf. The GPU path had no equivalent of
  `RUNNER_DEBUG_ACT`, so a backend silently producing NaN logits could
  only be bisected by guesswork; this found the defect above in three
  runs. Off by default, one env read per forward.

## v0.1.9-alpha — 2026-08-06

The desktop release, plus a certification survey that sharpened the
recommendations: the tray controller lands on macOS and Windows, gemma-4
E2B loads (the last locally-blocked Gemma-4 variant), a 19-model
derivative-certification campaign is folded into the docs, and the README
now carries **standing model recommendations per machine RAM**, each backed
by a measured run.

- **Certified-models README restructured**: one merged table (EU roster
  folded in), plus the new *Recommended models by machine RAM* section —
  8 GB Apple Silicon: Trinity-Nano (13.25 tok/s, resident); 16 GB:
  keep-30 (13.2–13.3 tok/s capped) or gemma-4-26B QAT `--kv q8`
  (7.1–7.3 tok/s capped, live validation pending); 24 GB GPU:
  Qwen3-30B-A3B (~72 tok/s). Models larger than RAM are recommended in
  no configuration — the streaming/paging regimes were measured and
  rejected (see `docs/negative-result-expert-cache.md`).
- **Cert-matrix campaign docs** (`docs/cert-matrix-status.md`,
  `docs/cert-matrix-2026-08-05.md` + evidence): 19 GPT-OSS × Gemma-4
  derivatives certified/failed/refused against the gates; only the
  Gemma-4 family certifies. Two capability gaps documented with root
  causes: no split/multi-part GGUF support, and the E2B loader gap fixed
  below. The compatibility manifest gained the missing `afmoe` pin and
  the README table the missing `apertus` row.

- **gemma-4 E2B loads: per-layer FFN widths.** E2B publishes real
  per-layer width variation (6144/12288) as an ARRAY-typed
  `feed_forward_length`, which the u32 getter silently read as 0 —
  refusing every E2B conversion, QAT or not (cert-matrix roster item 7).
  A new per-index getter (`gguf_get_u32_idx`) serves scalar and array
  forms, each layer carries its own dense-FFN width through the CPU
  forward path, and `--ffn-widths` on the fixture generator pins it in
  `tests/test_eseries.py`. Heterogeneous widths are CPU-only for now:
  CUDA and Metal refuse loudly and fall back rather than compute with one
  global width (verified byte-identical output either way on the real
  E2B-it Q4_K_M: loads, decodes coherently at 10.7 tok/s on an M1).

- Metal fit ceiling made visible (16 GB-Mac field report): the weight
  buffer allocation failure now prints requested bytes vs the device
  working-set limit ("11.5 GB requested, device working-set limit 5.7 GB —
  model exceeds Metal fit ceiling") instead of a bare "allocation" line,
  and `--caps` publishes the ceiling as `gpu.max_working_set_bytes` so a
  scheduler can apply the placement rule before attempting a load.
  `docs/moe-support.md`'s open item is closed with the measured floor:
  full `gpt-oss-20b-MXFP4` on Metal needs ≥ 24 GB unified memory; 16 GB
  machines get the CPU paging path (2.33 tok/s measured) and should run
  the pruned keep-30 artifact instead.

- **`--tray`: desktop tray / menu-bar controller (macOS + Windows).** A
  code-drawn grid icon lists every live runner instance on the machine
  with its loaded models (swap-mode servers are asked live via
  `GET /v1/models`), stops any of them, and starts one pre-configured
  desktop-managed server; login autostart via LaunchAgent / HKCU Run.
  Backed by a new instance discovery registry: every run-mode process
  writes `<config>/gridcore/runner/instances/<pid>.json` at startup
  (atomic, best-effort, swept by readers when the pid dies), so nothing
  in the serve or inference paths changed. `docs/tray-controller.md`.
- Makefile: recursive sub-make calls quote `$(PYTHON)`, fixing
  `make test PYTHON="py -3"` (the Python launcher is the normal recipe
  on Windows boxes without an MSYS2 python).

## v0.1.8-alpha — 2026-08-05

- **New architecture: `afmoe` (Arcee Trinity family — Trinity-Nano-Preview,
  Trinity-Mini).** Plain-transformer sparse MoE with muP-scaled embeddings,
  Qwen-G1 output-gated attention (a per-element sigmoid gate from a separate
  `attn_gate` projection — shares the qwen35 gate machinery), per-head-dim
  QK norms, a 3-local:1-global sliding-window pattern whose global layers
  are NoPE (the Llama-4 knob at step 4), DeepSeek-style sigmoid routing with
  a selection-only bias plus renormalized weights scaled by
  `expert_weights_scale`, one always-on shared expert, and leading dense
  blocks. New `afmoe` pre-tokenizer (right-aligned digit-triplet splitting,
  CJK/Asian-script isolation, punctuation-letter contractions). CPU only:
  Metal and CUDA refuse loudly and fall back. Certification record in the
  README table. Admission gates: tokenizer differential 0/721; layer-0
  attention path verified vs llama.cpp b10280 to ~2e-4; chat smoke and
  perf rows green on an 8 GB M1 (Q4_K_M resident: 13.25 tok/s decode
  CPU-only; Q8_0 under memory pressure: 8.9). Token identity not claimed,
  with the mechanism measured, not presumed — afmoe's per-branch
  re-normalization amplifies the engines' by-design quantized-matvec
  arithmetic difference (docs/afmoe-divergence-triage-2026-08-05.md).
- The `sampling:` banner at `--temp 0` now says what is actually true —
  greedy argmax, shaping knobs inactive — instead of printing a
  `repeat_penalty` value that does not apply. The penalty was already
  correctly bypassed in greedy mode; the banner misled an external
  reviewer during the afmoe certification run into ruling it out by hand.

## v0.1.7-alpha — 2026-08-05

### We found a 13× optimization and rejected it

We prototyped an expert-residency cache for MoE models larger than RAM. At
the extreme it turned 0.05 tok/s into 0.65 tok/s — a 13× speedup, with the
mechanism working exactly as designed (85%+ hit rates, byte-identical
output, reproduced on three platforms). We rejected it anyway: 0.65 tok/s
is not a configuration worth running, and everywhere the model actually
fits in memory the cache made things *slower*. The full measurements and
reasoning are in `docs/negative-result-expert-cache.md`.

That distinction is becoming a core principle of how gridcore-runner is
built: **an optimization doesn't pass because the benchmark got faster. It
has to preserve the model and produce a configuration worth running.** The
same gates that enforce this killed two other superficially attractive
ideas this cycle (deeper expert pruning, sub-4-bit expert requantization —
the latter produced a model that generated fluent text while agreeing with
the reference on 22% of tokens; nothing at load time would have noticed).

### CPU SIMD kernels, measured and gated

- **ARM NEON kernels** for the quantized dot/dequant path on Apple Silicon
  and other aarch64 — added *only* where they measured faster than the
  compiler's auto-vectorized scalar code, per format: Q6_K 8×, IQ4_NL 5×,
  IQ4_XS 4×, MXFP4 1.4×, Q5_K and Q4_0 modest wins. Formats where the
  auto-vectorizer won (F16, BF16, Q8_0, Q4_K) deliberately keep the scalar
  path, with the measurements noted in the source so nobody "optimizes"
  them back in.
- **MXFP4 gets a dedicated dot kernel on every ISA** (NEON, AVX2, scalar).
  It previously fell through to a generic block-dequant path on *all*
  platforms — including x86. On a Windows/AVX2 machine this took gpt-oss
  class decode from 0.21 to 3.4–4.0 tok/s (~16–20×) where the model ≈ fits
  RAM, and 0.15 → 0.54 tok/s on an 8 GB M1 where it doesn't.
- New gate `tests/test_quants_simd.c` (in `make test`): every quant format
  checked against an independent double-precision reference, and q8 KV-row
  quantization pinned byte-identical to its scalar definition. The gate has
  passed on macOS/NEON, Windows/AVX2, Linux/AVX2 and Linux/x86-64-v3.

### Fixed

- **gemma-4 E-series (E2B/E4B) produced silently wrong output under partial
  CUDA offload — shipped in `v0.1.5-alpha` and `v0.1.6-alpha`.** Full offload
  and CPU-only were both correct; any `--gpu-layers N` splitting the model was
  not, for three independent reasons found while trying to measure grammar
  fast-forward on the real E4B model (that path needs >=1 CPU layer). (1) the
  partial-offload upload's byte prefix omitted the per-layer-embedding
  tensors, so every partial split silently kernel-launch-failed and fell back
  to CPU — correct output, wrong device, the existing test only ever compared
  stdout so this went undetected. (2) with that fixed, the CPU-continued
  tail's per-layer-embedding table went stale/zero, because the prepass that
  fills it only ran when the CPU handled the *whole* forward. (3) even
  isolated from per-layer embeddings entirely, the device KV buffer was
  undersized whenever the split boundary landed on a shared-KV (non-cache-
  owning) layer — the real E4B case (`shared_kv_layers=18` of 42, so any
  split in [24,41] hit it) — because the sizing call redirected through "where
  does this layer's data live" when it needed "how many bytes do the first N
  layers need". Verified byte-identical to `--gpu off` on the real E4B model
  at every practical split point; `tests/test_eseries.py` gained an assertion
  that a partial split's stderr shows no fallback (proven red against the
  unfixed code). Metal never does partial offload (only full or fully off),
  so it was never exposed to any of the three.

## v0.1.6-alpha — 2026-08-04

- **Windows: real checkpoints share weights between `--parallel` slots again —
  the split defect is fixed.** The trigger was Branch A of the 2026-08-04
  investigation: MinGW's `stat()` has a 32-bit `st_size`, so on any file past
  2 GB it fails with `EOVERFLOW` ("value too large"), `model_file_identity()`
  lost the identity on **every real checkpoint**, and each `--serve --parallel`
  slot loaded privately and re-decided its own CPU/GPU split under the previous
  slot's VRAM pressure (measured on Qwen3-4B-Q8: slot A 36/36 layers, slot B
  20/36, B wrong on 149,477 of 151,936 logits at step 0). `model_file_identity()`
  now owns the `GetFileInformationByHandle` path that the prefix-cache key
  already used — 64-bit size, stable file index, 100 ns timestamps — so one
  function decides what a file is for the host registry, the device registry
  and the prefix-cache key alike (`stat()` remains the non-NTFS fallback).
  `tests/test_file_identity.c` pins identity at real-checkpoint size with a
  sparse 5 GB file (red before the fix, green after; skips loudly when the disk
  can't spare it). Verified on real weights: `test-shared-weights` exits 0 on
  Qwen3-4B-Q8 and Phi-4-mini-Q8 with identical splits, and a `--serve
  --parallel 2` server answered the same greedy request byte-identically from
  both slots (previously slot-dependent). macOS/Linux were never affected —
  their `stat()` is 64-bit.

- **A load that re-decides its split without a file identity is now reported
  loudly.** Even with the Windows trigger fixed, a genuinely unidentifiable
  file still loads privately and re-decides its split. The GPU registry now
  keeps no-identity entries visible (flagged, never matched) and
  `split_guard()` reports, as an `error:` on stderr, any same-path same-config
  pair whose splits disagree when either side lacks an identity — that
  disagreement is two slots of one server answering one request differently.
  A warning rather than a refusal, because refusing would fall back to CPU,
  which diverges from the resident GPU instance just as silently.
  `make test-split-guard` gates it (proven falsifiable: the harness goes red
  against a guard-less build).

- **A model that cannot be identified on disk now says so instead of silently
  giving up weight sharing.** Both shared-weight registries — the host parse in
  `model.c` and the device upload in `cuda.c` — key on a `stat()`-derived file
  identity, and both treated a failed `stat()` as "load this one privately".
  That fallback is not free: a privately loaded instance also re-decides its own
  CPU/GPU split against whatever VRAM the earlier instances already took, so two
  slots of one `--parallel` server can end up running different numbers of
  layers on the GPU and answering the same request differently. Nothing else in
  a load touches `stat()` — the Windows mapping path uses `GetFileSizeEx` and
  the POSIX one `fstat` — so the failure had no other symptom. The two
  duplicated helpers are now one `model_file_identity()` that reports the path,
  the `errno`, which registry was lost, and the consequence.

  `RUNNER_TEST_NO_FILE_ID=1` forces the failure, because on a machine whose
  `stat()` works nothing in a load can reach that branch. `make
  test-shared-noid` uses it to assert that `test-shared-weights` goes **red**
  without a file identity: the sharing gate had never been shown capable of
  failing, and at the 370 KB default fixture it could not. Under the hook it now
  fails on all four sharing invariants, which is the same host-side signature
  the 2026-08-04 RTX 3070 shelf pass reported when it found the defect.

  Diagnostic and gate only at the time it landed; the underlying split defect
  (Follow-up 3/3a) is fixed by the Windows file-identity entry above.

- **Apertus now runs on CUDA.** The dense device forward path handles its
  ungated `up -> xIELU -> down` FFN and evaluates the four effective per-layer
  xIELU parameters in a native CUDA kernel. The pinned 8B Q4_K_M checkpoint
  fully offloaded 32/32 layers on an RTX 3070 and matched CPU greedy output on
  all five 128-token `cpu_cuda` prompts.

- `--gpu off` now also keeps a speculative draft model on CPU instead of
  silently auto-offloading the draft and consuming VRAM.

- **One-shot and interactive CLI runs now release their model state.** Running
  the README's `make debug` binary under LeakSanitizer found about 240 KB of
  tokenizer, runtime, engine-history, schema/draft, and prompt allocations left
  for process exit. Normal CLI teardown is now explicit, and Linux CI includes
  a leak-enabled CPU smoke (the CUDA driver itself has process-lifetime noise).

- **The model-shelf stress harness now measures the machine it runs on.** Its
  RAM/VRAM defaults were still hard-coded to the earlier 16 GB / 8 GB test box,
  so tiering and placement recommendations were wrong elsewhere. It now reads
  both budgets from `runner --caps`, reports them, and keeps explicit overrides
  for controlled comparisons. It also no longer calls a CPU-only `--gpu auto`
  run CPU/CUDA identity: CUDA evidence is recorded only when the log proves an
  actual GPU split. Apertus now correctly reports `not_executed`, not a vacuous
  pass from comparing CPU with itself.

- **Speculative draft models now honor `--kv q8`.** Draft KV had been forced to
  f16 without explanation even when q8 was requested. The target verifies every
  proposal, so draft-cache loss can affect acceptance and speed but cannot alter
  the target-defined result. On the real Qwen2.5 7B/0.5B pair, 64-token greedy
  speculative output exactly matched plain decoding under both cache types;
  drafting was non-vacuous (92 proposals, 41 accepted with f16 and 42 with q8),
  while draft GPU allocation fell from 0.98 GB to 0.96 GB.

- **Ambiguous duplicate JSON object keys are rejected.** The parser now hashes
  decoded member names while reading each object, so literal duplicates and
  escape-equivalent spellings such as `model` / `m\u006fdel` fail in expected
  O(n) time. This prevents runner's former first-key interpretation from
  disagreeing with last-key-wins clients or proxies. Unit and live HTTP gates
  cover top-level, nested, escaped, and separate-object cases.

- **DNS rebinding and drive-by browser traffic are rejected at the HTTP
  boundary.** Every route, including accept-loop fast paths such as `/health`
  and `/v1/models`, now requires exactly one loopback `Host`; an `Origin`, when
  present, must be an HTTP(S) loopback origin too. Foreign, missing, duplicated,
  malformed, userinfo-bearing, and path-bearing authorities fail with 403.
  This closes the remaining browser-exposure half after `/unload` became
  POST-only. A table-driven C gate exercises the parser directly and live
  conformance checks both fast-path and slot-handled requests.

- **The real write-side stall is now reproducible and gated locally.**
  `scripts/write-stall.py` uses Qwen2.5-7B at 32k context plus a minimum-length
  structured stream to exceed the actual loopback buffers, and reads the exact
  connection's `/proc/net/tcp` transmit queue before judging either path. With
  a 2,304-byte effective client receive buffer, the production timeout run
  queued **396,365 bytes**, released its socket/slot in **31.062 s**, and served
  the next request; the SIGPIPE run queued **267,300 bytes**, survived the
  installed signal disposition, reset the stalled peer, and served again in
  **1.339 s**.

  The experiment found and fixed two bugs. `--ignore-eos` was silently dropped
  by `--serve` because only the one-shot engine received it; an EOS-only server
  fixture now reaches its exact requested limit. And Linux `SO_SNDTIMEO` alone
  left the measured zero-window slot wedged beyond 50 seconds, so Linux now
  applies `TCP_USER_TIMEOUT` at the same 30-second bound as well. The timeout-
  blind negative build stayed wedged at 570,154 queued bytes; the SIGPIPE-
  default build died with signal 13. RST/FIN attempts produced `ECONNRESET`,
  not a kernel-delivered SIGPIPE, so the deterministic signal gate injects
  SIGPIPE only after real queue pressure is proved and records that fact.

- **Published competitor rows now have a scheduled freshness alarm.** A cheap
  weekly workflow reads runtime name/version metadata from the committed
  agent-torture reports and compares each competitor's newest published row
  with official llama.cpp/Ollama GitHub releases and vLLM's PyPI release. It
  performs no inference in GitHub CI. Stale rows open or update one tracking
  issue and make the job red; unreachable registries are explicit skips, while
  malformed committed reports remain hard errors. The first live metadata
  check found llama.cpp **b10076 → b10241** and Ollama **0.32.1 → 0.32.5**
  stale; vLLM **0.26.0** is current.

- **Speculative decoding is now a first-class agent-torture runtime axis.**
  `scripts/agent-torture.py --draft PATH --draft-k N` runs the unchanged
  provider-neutral matrix once without and once with the draft server flags,
  then compares every pass/fail verdict by case ID. Any change is a failing
  result because speculation is an optimization, not a new capability row.
  The report records total proposed/accepted tokens, acceptance rate, grammar
  counters, and mismatches; the baseline report and raw evidence are retained
  alongside it. A negative-control unit test changes one draft verdict and
  proves the comparator fails instead of accidentally comparing the baseline
  to itself; a zero-proposal run also fails, catching ignored draft flags.
  Measured on Qwen2.5-7B-Instruct Q4_K_M plus the same-tokenizer
  Qwen2.5-0.5B-Instruct Q4_K_M draft: **105/105 matched case for case**, with
  2,325/4,044 proposals accepted (**57.49%**). The CPU-target run took 369.1 s
  versus 337.0 s target-only, so this pair proves correctness and real draft
  activity but is not a speed win on this hardware/configuration.

- **BREAKING: `/unload` is POST-only.** It was a `GET`, which made it reachable
  from any web page the user happened to be visiting —
  `<img src="http://127.0.0.1:PORT/unload">` frees the resident model with no
  preflight, no CORS check and no DNS rebinding needed, because binding to
  loopback does not stop a browser. Verified against the old build: `GET`
  returned 200 and the server logged a real unload. A POST is not a CORS
  *simple* request unless its `Content-Type` says so, so requiring one restores
  the preflight that stands between a drive-by page and a freed model.

  `GET /unload` now answers **405** naming the method, not 404, so an existing
  script says what changed rather than looking like a missing route. A POST
  carrying a body is refused too — `/unload` takes none, and the accept loop is
  the only thread calling `accept()`, so it must not sit draining one.

  This is half the fix. `Host` and `Origin` are still unvalidated, which is what
  DNS rebinding needs, and that is tracked separately.

- **Follow-up to the request-validation hardening: accurate rejection
  messages, and one dead check removed.** The new rules are right and stay;
  what they *said* was not. `max_tokens: 1.5` answered "max_tokens must be a
  number" — 1.5 is a number, so that sends a caller looking for a type bug they
  do not have — and `top_k: 2.5` answered "out of range" when 2.5 is inside
  every bound `top_k` has. They now say "must be a whole number", with a
  separate sentinel so a genuine type error still reads as one.

  The added `seed >= 2^64` guard was **unreachable**: `request_number` already
  caps seed at 18446744073709549568.0, which is 2^64 − 2048, the largest double
  below 2^64 — so the `uint64_t` cast was never at risk and the branch could
  not fire. Verified at the boundary (2^64 − 2048 → 200, 2^64 → 400). Replaced
  with a named `SEED_MAX` and a `_Static_assert` tying the bound to the cast's
  safety, so widening it stops the build instead of quietly reintroducing the
  undefined conversion. Confirmed the assertion fires when the bound is raised.

From a field report on an M2 Pro / 16 GB MacBook Pro driving Continue in
VS Code — the second outside install. The Metal-side findings need a Mac and are
filed; these are the ones that did not.

- **Runner can now tell you its weights are being paged, instead of looking
  healthy while it stalls.** This was the report's worst finding, because every
  signal stayed green through it: on a loaded 16 GB machine an 8B sat at ~0.5 GB
  RSS against a 4.9 GB file and a five-token reply took **53 s at 0.0% CPU**;
  later a 1,200-token prompt returned **nothing in 300 s** while `/v1/models`
  answered instantly and warm-prefix chats came back in 1.5 s. A health check
  and a smoke test both pass while the real workload is dead. Four additions:

  - **`--mlock`** wires the mapping into RAM. Opt-in and fail-soft on purpose —
    locking 5 GB on a 16 GB laptop can cause the pressure it was meant to avoid,
    so a refusal is reported and the load continues.
  - **A load-time warning** when the weights are larger than the RAM actually
    available, because *file size ≤ total RAM* is the test that passes right
    before a machine starts thrashing.
  - **A per-request paging signal.** `runner_telemetry.major_page_faults`, and a
    `[N page-ins — weights not resident]` note on the `[slot]` line when N is
    nonzero. Major faults are the mechanism itself, so a slow request that took
    none was slow for some other reason — better than inferring paging from
    wall-clock.
  - **`ram_available_bytes` in `--caps`**, so a launcher can refuse a model that
    will not stay resident rather than discovering it later.

- **A request that never finishes now leaves a trace.** The `[slot]` line was
  printed on completion only, so the 300-second stall above produced an empty
  log for five minutes. Every request now logs a start line carrying its id,
  prompt length and cache hit, and the id is allocated before any work so both
  lines share one name.

- **`-m name=path` no longer silently costs you `--parallel`.** Naming one model
  to pin the `/v1/models` id — which is what the reporter wanted for a Continue
  config — put the server in swap mode and dropped `--parallel 2` without
  asking. Swap mode really is single-slot, but with one entry there is nothing
  to swap to, so runner now keeps the slots and gives up the registry instead,
  and says which: `/unload` and `--ttl` need more than one model. Two or more
  entries behave exactly as before.

- **`--caps` no longer implies GPU MoE where there is none.** It listed `MXFP4`
  and the `Q*_K` family under `gpu_quants` on every backend, but Metal's
  `gpu_init()` refuses a model with experts *before* it looks at the quant — so
  a Mac user read that as a promise and watched gpt-oss-20b run CPU-only at 0.38
  tok/s. The GPU block now carries `moe` and `kv_q8` booleans, and a
  sparse-MoE model whose experts fell back to the CPU says so in the serve
  banner rather than only in an init line that scrolls past.

- **The JSON-schema test now builds `schema.c` the way the binary does.** It was
  compiled with neither `-O3` nor `-ffast-math`, so `exclusiveMinimum` /
  `exclusiveMaximum` — which use `nextafter(x, ±INFINITY)` — were gated in a
  configuration that does not ship. Measured on gcc the results are identical
  either way, so this is gate integrity rather than a bug fix; but clang warns
  here (`-Wnan-infinity-disabled`) and clang is what a Mac uses, so the test
  should be what finds out.

- **README: verify the byte size of a downloaded model.** `download-model.sh`
  already does. A hand-rolled wrapper did not, and an 8B download truncated at
  290 MB by an HF-CDN reset still reported success, because `curl` exited 56
  inside a compound command that returned 0.

- **The Claude Code end-to-end check is a script now**
  (`scripts/claude-code-e2e.sh`), and re-run against **Claude Code 2.1.220**:
  PASS. It starts a server, writes a fixture holding a sentinel generated for
  that run, points the real CLI at Runner via `ANTHROPIC_BASE_URL`, and
  requires the sentinel back — so the README's compatibility claim rests on
  something repeatable rather than on one manual validation against one
  version. Runner served the loop with 22,942 of 22,943 prompt tokens coming
  from the prefix cache.

  Two things the script had to learn the hard way, both recorded in it:
  `--allowedTools` governs permission, not what is *declared*, so Claude Code
  sends its entire built-in tool set on the first request and does not fit in
  a 16k context; and the model has to be named explicitly or the CLI keeps
  whatever model the developer's own session is configured with and dies before
  making a request.

- **Torture matrix v2: seven families, 105 cases** (`scripts/agent-torture.py`).
  Two new families, both request-level so other runtimes can be asked the same
  questions: `reasoning_then_tool` (an assistant turn of prose already in the
  history, then a forced call — the failure it catches is content bleeding into
  the call turn) and `structured_final` (a schema-constrained *final answer*
  via `response_format`, which reaches the sampler by a different path than
  tools do and had no torture coverage). 105 = 7 x 15; `SCHEMA_VERSION` is
  bumped to v2 because v1's 100/5 results are not case-for-case comparable.

- **A published benchmark result is corrected: vLLM scores 80/105, not 20/100.**
  The `2026-08-02` row started vLLM without `--tool-call-parser`, and vLLM
  *refuses* `tool_choice` outright in that state — all 80 tool cases came back
  `400 ... requires --tool-call-parser to be set` before the model was asked
  anything. The 20 that passed are exactly the 20 cases that send no `tools`.
  Started correctly it scores **80/105** on the harder v2 matrix. The old
  README now carries the correction, and the new result keeps a reproduction of
  the misconfigured run beside the fixed one so the claim is checkable. Runner
  is 105/105 on the same matrix and its column is unaffected.

  A comparison harness that lets the competitor fail at admission and files it
  as a capability difference is measuring its own setup.

- **First cross-runtime resource footprint**
  (`tests/torture/results/2026-08-03-smollm2-1.7b-v2/`), with the differences
  in kind stated rather than averaged away: runner 1.1 GB weights + 0.81 GB KV
  in VRAM and 1.10 GB host RSS, against vLLM's 3.19 GiB bf16 weights, a 9.57
  GiB KV pool **preallocated by policy rather than need**, and 3.36 GB host RSS
  across its two processes — measuring only the process named `vllm` reports
  1.01 GB and understates it by more than half.

- **Write-side coverage: an interrupted client cannot take the server with it**
  (`tests/conformance/test_write_side.py`, 4 tests). A client that RSTs
  mid-stream, one that RSTs before reading a byte, ten in a row, and one that
  stops reading entirely — after each, the process is alive and both slots
  still serve.

  **Two things this deliberately does not claim.** The write *stall* — a
  `send()` that blocks until the 30 s `SO_SNDTIMEO` fires — cannot be produced
  in this harness: the suite model is capped by `n_ctx` at ~68 KB of SSE, and
  on loopback that fits in the socket buffers even with the client's
  `SO_RCVBUF` pinned to 1 KB (measured — the whole response is delivered to a
  client that never calls `recv()` once). For the same reason it does **not**
  gate `signal(SIGPIPE, SIG_IGN)`: a build with the handler at `SIG_DFL` was
  constructed and passes all four, because the write completes before the peer
  resets. Both need a real model and a large context, which is a `scripts/`
  experiment rather than a conformance test, and is filed as such.

- **Anthropic prompt caching and replayed reasoning are now gated** (5 tests in
  `tests/conformance/test_messages.py`). Both behaviours already worked and
  neither had a test, which is the state in which a behaviour quietly stops
  being true.

  `cache_control` is accepted wherever the Anthropic SDK and Claude Code put it
  — on system blocks, on message content, and on a tool — because refusing the
  marker would make runner unusable with those clients rather than merely
  uncached. The matching *decision* is pinned too: runner still does not claim
  `cache_read_input_tokens` / `cache_creation_input_tokens`, since those carry
  Anthropic's semantics (`input_tokens` excludes what they cover) and filling
  them with runner's unrelated prefix-cache figures would misstate a client's
  accounting. That figure stays in `runner_telemetry`. **Owner call if it
  should change** — the test now says so out loud instead of the code saying it
  in a comment.

  Replayed `thinking` / `redacted_thinking` blocks are accepted and dropped.
  The test that matters is not the 200 but the cost: via `count_tokens`, an
  assistant turn carrying a long thinking block counts **identically** to the
  same turn without one, with the same text sent as a `text` block as the
  control — without it the test would pass against a server that ignored
  content blocks entirely.

- **Fixed: `/v1/embeddings` refused every official OpenAI client.** The SDKs
  send `encoding_format: "base64"` by *default* and decode it themselves;
  runner answered `400 encoding_format must be float`, so `client.embeddings
  .create(...)` could not be called at all. base64 is now emitted as
  little-endian float32, spelled out byte by byte so the wire format does not
  depend on host endianness. `float` is unchanged and `dimensions` is still
  refused unless it equals the model's width.

  The conformance suite had a test **asserting the 400** — it pinned the bug in
  place, and every hand-written embeddings test passed because none of them sent
  the field the real client sends. It is replaced by a check that the base64
  payload decodes to the same vector as the float form, which merely accepting
  the field would not pass.

- **A second, independent client too: the Vercel AI SDK**
  (`tests/aisdk/smoke.mjs`, driven by `tests/conformance/test_ai_sdk.py`).
  `ai` + `@ai-sdk/openai` is what most local-model tooling actually uses — Cline,
  Continue, anything on Next.js — and it has its own request builder and stream
  parser, so it is a real second opinion rather than a restatement of the first.
  Eight cases: text, streamed text, streamed usage, a tool call, a two-turn tool
  round trip with the result replayed, `generateObject` typechecked against a
  zod schema, batched embeddings, and a typed error. All pass. Skipped unless
  `tests/aisdk/node_modules` exists, so CI gains no npm step.

- **The official OpenAI SDK now has conformance coverage**
  (`tests/conformance/test_openai_sdk.py`, 10 tests, skipped when the package is
  absent — the same rule `test_messages.py` applies to `anthropic`, so CI gains
  no network-installed dependency). It covers models, buffered and streamed
  chat, `stream_options.include_usage`, tool calls and the tool-result second
  turn, `json_schema` output, legacy completions, embeddings, and typed errors.
  The embeddings defect above was found by its first run.

- **A long prefill no longer blocks the other slots for its full length.** On a
  4-slot GPU server, a short request arriving during a 2,891-token prefill
  waited **26.2 s — 110x its 0.237 s solo time**. Prefill now gives the device
  turn back between chunks:

  | | long request | worst short during it | vs solo |
  |---|---|---|---|
  | before | 26.50 s | 26.17 s | 110x |
  | after | 27.01 s | **5.45 s** | **23x** |

  The remaining 5.4 s is the honest cost of fair interleaving, not a defect: a
  short request decodes 8 tokens and each queues behind one 64-token prefill
  chunk. Prefill itself pays 1.9%. CPU builds are unaffected — the device turn
  only exists where CUDA graph capture does.

  **Yielding the turn was not enough, and shipping it alone would have been a
  no-op.** `dev_mu` was a plain mutex, and a plain mutex is not a hand-off: the
  releasing thread is already on-CPU with its threadpool hot and re-acquires
  before the woken waiter is scheduled. The waiter lost 44 of 45 races and
  still waited 25.3 s. `sched_yield()` before retaking changed nothing. The
  turn is now a FIFO ticket turnstile, so giving it back means giving it up.
  `tests/test_sched_turn.c` guards the ordering by reproducing the barge —
  it fails against a plain-mutex build and passes against the turnstile.

- **The prefix cache stores to the divergence point, not to the end of the
  prompt.** Agent traffic is a system prompt, a tool list and a schema, then a
  request that differs. Publishing each turn whole held the shared block once
  per turn. Measured on Qwen2.5-7B with six sibling prompts over a ~2,300-token
  shared prefix:

  | | entries | bytes | evictions |
  |---|---|---|---|
  | before | 4 | 532.4 MB | 2 |
  | after | **1** | **133.1 MB** | **0** |

  Reuse is unchanged — every sibling reuses 2,314 of 2,321 tokens either way —
  so this is 4x less memory for the same work, and the two evictions it removes
  were the cache throwing away a genuinely shared block in order to store tails
  nobody shares.

  The tail past the divergence point is exactly the part another request has
  been *observed* not to share, and it is also the part a repeat of the same
  prompt recovers for free from its own slot via `engine_rewind`. Truncating is
  free in correctness terms for the same reason the existing half-budget cap
  is: a prefix of a prefix is still a valid prefix.

  The new conformance gate was **vacuous on its first version** and only caught
  it by checking: its five sibling tails were of different lengths, so one was
  a prefix of another and the pre-existing "strictly extends" branch collapsed
  them to one entry for a reason unrelated to this policy — it passed against a
  build with the policy compiled out. With equal-length tails that diverge
  early it now fails without the policy and passes with it.

- **A certified `cpu_cuda` claim does not hold at the documented token count,
  and the tool's default hides it. OWNER call.** Chasing the plan's "Qwen3-4B
  CPU/GPU divergence at token 24" turned up something broader.

  `docs/compatibility-program.md` states the contract as *"Exact 128-token
  identity is required between Runner CPU and GPU."* `scripts/cpu_cuda_check.py`
  defaults to **`--tokens 16`**. On Qwen3-4B-Q4_K_M — which declares `cpu_cuda`
  and whose file matches the manifest's pinned sha256 — the two answers differ:

  | tokens | result |
  |---|---|
  | 16 (the default) | 5/5 identical |
  | 32 | 5/5 identical |
  | 48, 64, 96, 128 | **4/5** |

  **Pre-existing, not from this release's work**: the published `v0.1.5-alpha`
  build, rebuilt from the tag, gives the same 4/5. So the certification passes
  because the tool checks 16 tokens, not because the claim holds at 128.

  What it is *not*, measured: not the GPU split (both runs reach `G=36/36
  full=1`), not `RUNNER_MOE_EAGER=1` (5/5 with it set — and Qwen3-4B is dense
  anyway), not cross-request KV reuse (5/5 with `cache_prompt:false`), and not
  simply arithmetic — the failing prompt is byte-identical CPU vs GPU when run
  alone on a fresh server, and a reproduction matching the harness's request
  body exactly also gives 5/5. It is **intermittent on the GPU side** and so far
  reproduces only inside the tool's own process.

  Two things follow, and the first is not mine to decide: whether to re-certify
  Qwen3-4B, raise the tool's default to the documented 128, or amend the
  documented contract, is a **certification decision — surfaced, not taken**.
  The second is a plain tool bug: the report records `"gpu_split": null` even
  though the line is present in the log it reads, so a report cannot say what
  was actually certified — which is exactly what `read_split`'s own docstring
  says it exists to prevent.

- **A CPU server no longer serializes on a device turn it does not have.** The
  scheduler's `dev_mu` exists for one reason, stated in its own comment: a
  microbatch captures a CUDA graph on its lead sequence's stream, and any other
  launch in that context breaks the capture. So prefill, decode and any solo
  generation take turns. A **CPU** build has no capture and no shared device
  context — every `model_t` owns its activation scratch and its thread pool,
  and the weights are read-only — so the turn buys nothing there and costs a
  lot.

  Measured on Qwen2.5-7B, `--gpu off --parallel 4`, with a grammar-fast-forward
  request holding the turn for its whole generation: a plain request arriving
  during it waited **10.3 s against 4.8 s alone**, i.e. for the entire
  generation. After gating the turn on `m->gpu != NULL`: **4.3 s, 1.0x**. The
  constrained request itself goes 6.8 → 7.7 s, which is the correct trade — it
  is now sharing the box instead of monopolising it.

  Untouched on the GPU path, where the capture hazard is real. And the bigger
  half is still open and now says so in the code: `sched_generate` holds the
  turn for a whole speculative generation rather than per forward, so on a GPU
  one `--draft` slot still stalls every other slot for its full length. Taking
  the turn per forward needs `engine_generate` to call a hook around each one.

- **Not done: the per-forward device turn on GPU.** `sched_generate` holds
  `dev_mu` for a whole speculative generation, so on a GPU one `--draft` slot
  stalls every other slot for its full length. The fix was built — a
  `engine_set_device_turn()` hook the scheduler hands down, wrapping each of
  the five forwards a speculative round issues, deliberately per-call rather
  than per-round because that loop exits through several `goto`s and a missed
  release is a deadlock — and then **reverted, because two measurements
  disagreed**:

  | run | whole-generation hold | per-forward hold |
  |---|---|---|
  | 1 | worst concurrent request 3.0x solo | 2.1x |
  | 2 | 2.3x | 3.0x |

  Contradictory results are not evidence, and the change costs a lock/unlock
  per decoded token on the hot path. What defeated the measurement is that a
  schema-constrained generation completes its object and stops early, so no
  configuration available here produced a speculative generation long enough
  for the hold to dominate — which is exactly the case the theory is about (an
  unbounded hold versus one bounded by a single forward). A conclusive test
  needs either a schema that does not self-terminate or a real draft/target
  pair, neither of which is on this box. Patch and numbers filed.

- **Prefix snapshots survive a restart (`runner.prefix.v1`).** A warm prefix
  cache is worth minutes of prefill and it died with the process.
  `prefix_cache_save()` / `prefix_cache_load()` write it to a file and read it
  back, so a restarted server answers the first agent request at fork speed.

  The trust question is the whole design, and it is answered by refusal rather
  than by adaptation. A snapshot is raw KV bytes: installing one that does not
  belong to this model does not error, it produces fluent wrong output — the
  same hazard `engine_prefix_reuse` was built around, now with a file as the
  surface. So the file carries the engine's `model_key`, which already binds
  the weights, the geometry, the tokenizer, the context length and the KV
  element type, and **every entry whose key does not match is dropped, not
  fitted**. The file is also checked for a magic, a length-consistent body and
  a payload digest before any of it is believed, and a failure discards the
  whole load rather than keeping the part that parsed — a partial load of a
  corrupt file is the worst outcome, because it looks like success. It is
  opt-in with an explicit path and no discovery: a cache directory someone else
  can write is a way to hand this process someone else's KV.

  `tests/test_prefix_persist.c` is fifteen checks and most of them are
  refusals: a foreign model key, a mismatched entry width, a file that is not
  ours, a truncated file, a single flipped byte. The round trip is checked by
  **content** — a fresh engine must fork the reloaded snapshot — because a
  save/load pair that wrote zeros would still keep the entry count.

  Landed only after a detour worth recording. The test segfaulted 8 times out
  of 8 at `-O2` and above while passing at `-O0`, `-O1`, and under ASan at both
  — the signature of corruption ASan cannot see. It was **not the feature**:
  `engine_init` calls `free(e->hist)` on entry (its comment says "e must be
  zeroed"), and the test declared `engine e;` on the stack uninitialized, so
  each init freed a wild pointer. `engine e = {0}` and it is 5/5 clean at full
  optimization. The feature was reverted once on the strength of that crash
  before the cause was known, which was the right call at the time and the
  wrong conclusion.

- **Phase 8: the 8k→32k retrieval gate — a q8 cache does not cost recall.**
  Needle-in-a-haystack on Qwen2.5-7B: a unique fact planted at 10%, 50% and 90%
  depth in a filler context, two codes per depth, scored on exact digits.

  | KV | context | prompt tokens | recalled |
  |---|---|---|---|
  | f16 | 8k | ~5,237 | 6/6 |
  | f16 | 32k | ~20,817 | 6/6 |
  | q8_0 | 8k | ~5,237 | 6/6 |
  | q8_0 | 32k | ~20,817 | **6/6** |

  Together with the throughput row below — q8 costs 0.9% prefill and 0.7%
  decode — the case for a q8 cache is that it halves the KV for no measured
  loss on either axis. Whether it becomes a default is still an owner call, and
  the clu item asking for that decision is unchanged.

  The scorer was wrong first and reported 4/9 for answers that were all
  correct: it demanded the hyphens in `62-05-31` from a prompt that asked for
  "the digits only". Scoring on digits is the fix. A gate that marks correct
  answers wrong is the same class of defect as one that cannot fail.

- **Phase 8: q8 KV attention is essentially free.** Measured on the now
  uncontended MIG slice, Qwen2.5-7B-Instruct-Q4_K_M, 512-token prompt and 256
  generated, full offload (28/28 layers) in every row:

  | KV | context | prefill | decode |
  |---|---|---|---|
  | f16 | 4096 / 16384 / 32768 | 129.2 / 129.3 / 129.3 tok/s | 70.51 / 70.44 / 70.59 tok/s |
  | q8_0 | 4096 / 16384 / 32768 | 128.1 / 128.2 / 128.1 tok/s | 70.06 / 69.73 / 70.12 tok/s |

  **0.9% of prefill and 0.7% of decode**, for half the cache. What this does
  *not* show, and the reason the context column is flat: `--bench-json`
  generates 256 tokens whatever the context *capacity* is, so the cache stays
  nearly empty and this measures the q8 attention kernels, not decode against a
  full 32k window. The long-context half is the separate 8k→32k retrieval gate,
  still open.

- **Phase 7 measurements: the fork bottleneck is not the mutex, and an SWA
  snapshot is ~3x larger than it needs to be.** Neither is a code change here;
  both correct a premise the plan was carrying.

  **The "snapshot/fork mutex as a scaling bottleneck" names the wrong mutex.**
  Measured on Qwen2.5-7B with a 2,310-token shared prefix and `--parallel 4`:
  one fork takes 0.095 s, four concurrent forks take 0.44 s in a clean
  staircase (0.385 / 0.422 / 0.422 / 0.437) — fully serialized. Moving the
  132 MB snapshot copy out from under `PFX.mu`, with a pin/dead refcount so
  eviction cannot free an entry mid-copy, was implemented and measured:
  **0.433 s. No improvement.** So it was reverted rather than shipped as
  unmeasured complexity.

  The real serializer is one line up in `completion.c`: `engine_prefix_reuse`
  runs **inside `sched_prefill_begin()/sched_prefill_end()`**, the device turn.
  The comment there justifies it — "on CUDA it issues a forward" — and that is
  true of exactly one single-token forward needed to break the device KV
  mirror. The other 132 MB is a host memcpy holding a device lock. Confirmed on
  the CPU path too, where there is no device work at all and four forks still
  take 4.2x one. The fix is to take the turn around the sync forward alone,
  which needs `engine_prefix_reuse` split into a lookup half and a copy half;
  filed rather than attempted, because a fork that lands wrong produces a
  plausible wrong answer rather than an error.

  **SWA prefixes, measured separately as the plan asked.** gemma-4-E4B has 42
  layers of which **35 are sliding-window with a 512-token window**, and
  `prefix_cache_entry_bytes` stores the full prefix length for every
  KV-owning layer regardless. For a 2,310-token prefix that is 2,310 rows per
  sliding layer where only 512 can ever be attended to. Storing the window
  instead would take the snapshot from 42x2310 row-equivalents to
  7x2310 + 35x512 — **2.85x smaller**, which is also 2.85x more prefixes inside
  the same 512 MB budget. Filed: the KV layout is absolute-indexed, so this is
  a placement change rather than a smaller `memcpy`.

- **`server_run` could not run twice, and a SIGTERM could fail to wake
  `accept()`.** RNR-019's remaining half was "de-globalise `SV`". Rather than
  start from the shape, the property was written down first: *a server must be
  able to start, serve, stop and start again in one process.* A global
  initialized once and torn down once can hide an asymmetry forever, because
  nothing ever asks the state to come back. `tests/test_server_restart.c` asks,
  twice — and found two defects, neither of which any existing test could see.

  **The state had no lifetime.** `q.shutdown`, `shutdown` and `load_cancel` are
  raised during teardown and never lowered, and `reaper_started` stayed true
  next to a `reaper_th` whose thread had already been joined. A second
  `server_run` therefore listened, accepted connections, and generated nothing:
  every slot worker saw a shut-down queue and exited at once. `server_run` now
  resets the state at entry, as the documented counterpart of the teardown at
  the bottom.

  **A SIGTERM could leave the server parked in `accept()`.** The handler closed
  the listener and the comment said that "wakes accept()". It does not — a
  blocked `accept()` is woken by the signal only in the thread the signal was
  *delivered to*, and a process may deliver SIGTERM to any thread that has it
  unblocked: a slot worker, the decode thread, the TTL reaper. Observed
  directly in `/proc`: the accept thread sat in `inet_csk_accept` long after
  the handler had run and closed the fd. `shutdown(fd, SHUT_RDWR)` now precedes
  the `close()`; both are async-signal-safe. This is a **plausible but
  unproven** explanation for the `test_signal_during_startup` sighting above —
  "server survived a SIGTERM" is exactly the symptom, and its rarity matches
  delivery usually landing on the main thread — but it was never reproduced, so
  the link is offered, not claimed.

  Both fixes are demonstrated load-bearing: without the reset the second cycle
  fails with *"no worker answered a completion"*; without `shutdown()` the
  first cycle hangs in `pthread_join`. The gate had to be strengthened to show
  the first — its initial version asked for `/v1/models`, which
  `accept_fastpath` answers on the accept path with no worker involved, so it
  passed against the very bug it was written for. A gate that cannot fail is
  worse than none.

  What this is **not**: `SV` is still one global, so two servers in one process
  would still share it. What changed is that its lifetime is now explicit and
  its init/teardown symmetry is tested. Threading a per-instance context
  through six translation units is the remaining half, filed rather than done —
  nothing needs two servers in a process today, and the defects above were the
  part that was actually costing something.

- **The `test_signal_during_startup` flake did not reproduce, and the test now
  records enough to chase the next one.** It was seen once on 2026-08-02 at the
  2 ms delay and estimated at "~1 in 20 runs" from that single sighting — which
  is the same mistake the earlier speculation flake was corrected for, since
  one occurrence establishes no rate. Not reproduced in **960 spawns**: 240 on
  an idle box, 240 with 24 spinners loading it, 120 on the pre-Phase-5 binary
  and 120 on the current one, plus 20 consecutive whole-suite runs (307/307
  each), whose 240 in-suite spawns are included. Rule of three puts the 95%
  upper bound on the per-spawn rate at 0.31%, just under the 0.42% the original
  estimate implies.

  A guess that Phase 5's faster loads had shortened the window was **wrong**
  and measured as wrong: startup to `listening` is 2.7 ms before and 2.4 ms
  after, because the test fixture's vocabulary is trivial and the parse saving
  needs a real one.

  Left open rather than closed — absence over 960 spawns is not proof of a
  fix. What changed is that the test no longer discards the survivor's output,
  so a future occurrence says how far startup had got instead of only that it
  happened.

- **A context that fits but evicts weights now says so.** Refusing a context
  that cannot fit is loud and correct — `-c 1000000` names the 131072 MB of KV
  it would need and exits non-zero. A context that merely *costs layers* was
  accepted in silence, and the bill arrived as throughput: `-c 32768` on an 8B
  model takes decode from 66.5 to 8.6 tok/s on an 8 GB card because a 4.3 GB
  cache pushed layers onto the host. The split line already reported the
  placement; nothing connected it to the context that caused it.

  ```
  gpu-split: budget=6.34GB fixed=0.91GB G=26/28 full=0 used=6.28GB
  note: the KV cache for ctx 32768 is 1.74 GB on the device and 2 of 28 layers
        ran out of room because of it — a smaller -c or --kv q8 (about half)
        moves layers back
  ```

  It fires only when the KV is a large share of what is on the device, because
  a partial split for any other reason — a model simply bigger than the card —
  is not a trade the user can take back by lowering `-c`. Verified on three
  cases: the one above fires, a full offload is silent, and Qwen3-Coder-30B at
  18 of 48 layers on a 7.6 GB budget is silent. Acting on it works and the
  note then stops: `--kv q8` on the same run recovers a layer (26/28 → 27/28)
  and goes quiet. Written host-side rather than into the CUDA split banner, so
  it covers Metal too and leaves `cuda.c` untouched.

- **A filed prefix-cache "inefficiency" was a misread telemetry field.** The
  note from the flake work said that after a cancellation plus concurrent
  traffic a sequential request "never forks at all — deterministic, not a
  race", and filed it as a possible caching inefficiency. Measured: sequential
  requests after exactly that sequence **do** fork, 930 and 931 tokens.

  The zeros that prompted the note are on the *concurrent* requests, and they
  do not mean the prefix went unused — those requests report
  `prompt_cached_tokens` 927 with `prompt_eval_tokens` **1**. The slot's own KV
  already held the prefix, so `engine_prefix_reuse` declined to fork: its gate
  is `best > r.keep`, and forking when the snapshot holds no more than the slot
  does would copy identical rows over themselves. `prompt_forked_tokens` is the
  subset that came from the *shared snapshot*, not the amount reused.

  `test_agent_runtime_composition.py` now asserts the property that actually
  matters to a caller — every concurrent request reused the prefix rather than
  re-prefilling it — alongside the existing "at least one forked", and its
  comment no longer states the wrong claim as fact.

- **Apertus generated gibberish, and `0.1.5-alpha` shipped it that way.** The
  architecture landed in `d7eda52` with the honest caveat that it was "not yet
  verified against a real checkpoint — the shape is right, the numbers are
  unconfirmed". They were wrong. The first run against
  `Apertus-8B-Instruct-2509-Q4_K_M` produced
  *"The capital of Switzerland isus ROIgg Sylosl Suombe…"* where llama.cpp on
  the same file produces *"Bern, which is also the country's largest city"*.

  `ggml_xielu` does not pass the file's parameters to `op_xielu`. It transforms
  them when it builds the node — `alpha_p` becomes `softplus(alpha_p)`, and
  `alpha_n` becomes `beta + softplus(alpha_n)` — and only the transformed
  values reach the activation. Runner transcribed `op_xielu`, the leaf
  function, and fed it the raw values. **Read the graph, not the op**, which is
  the second time this exact lesson has been recorded here (the E-series PLE
  injection point was the first).

  It hid well. `softplus` is the identity above ~20 to float precision, and
  most of Apertus-8B's alphas are in the tens or hundreds — layer 0 has
  `alpha_p` 166.0, which needs no correction at all. The middle layers are
  where it bites: layer 15 has `alpha_n` **0.00296** against an effective
  **1.19463**, a factor of 403. Folded at load time, once per layer, so the
  hot path is unchanged.

  Now verified rather than asserted. The residual disagreement with llama.cpp
  is **below the model's own noise floor**: a max log-probability delta of
  0.4148 nats cross-engine against 0.4596 nats for runner-versus-runner under
  a KV precision change, with 7 of 16 divergences landing on a tie. So Apertus
  carries `load` and `chat` but not `greedy_reference`, for the same measured
  reason as gemma-4-26B and gemma-4-E4B. Evidence:
  `tests/compatibility/out/sensitivity-apertus-2026-08-03.json` and
  `divergence-apertus-2026-08-03.json`.

- **Apertus tokenizer: 3 of 721 → 0 of 721.** The three known divergences
  (नमस्ते, हिन्दी, สวัสดี) were combining-mark sequences, and the cause was a
  correct fix applied one regex too widely. `cp_mark` exists because treating
  every non-symbol codepoint above ASCII as a letter glued Indic and Thai vowel
  signs into `\p{L}+` runs — right for `llama-bpe`, `qwen2` and `smollm`, whose
  regexes all spell a plain `\p{L}+`. **`tekken` is the exception**: it carries
  `\p{M}` in both letter classes,

  ```
  [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+
  ```

  so a virama or a Thai vowel sign has to stay *inside* the run. The comments
  on `cp_letter_upperish` / `cp_letter_lowerish` had spelled the class with
  `\p{M}` in it all along; only the code disagreed. Marks now count as letters
  in the tekken split alone. Verified against the HuggingFace references:
  Apertus 0/721, and Mistral-Nemo — the other `tekken` model — still 0/721,
  with Qwen2.5-7B and gemma-4-E4B spot-checked at 0/721 to confirm the other
  families are untouched.

- **The merge loops were quadratic in the length of one segment.** Phase 5
  asked for shared *tokenized* prefixes, on the theory that re-tokenizing the
  same system prompt every request was the cost. Measuring first found
  something worse: tokenization was **O(n²)**, and a cache would have hidden
  it rather than fixed it.

  Both merge loops rescanned every adjacent pair after every merge. That is
  fine when the loop is handed a word at a time, which is what a GPT-2 style
  pre-tokenizer does — but the SentencePiece path never splits at all, and
  gemma-4's BPE path splits only on newline runs, so **one long line is one
  unit**. On 4,000 characters of prose in a single line:

  | tokenizer | before | after |
  |---|---|---|
  | gemma-4 (26B, E4B) | 71.6 ms | 0.87 ms (**82×**) |
  | Lucie-7B | 30.5 ms | 0.44 ms (69×) |
  | Teuken-7B | 30.0 ms | 0.56 ms (54×) |
  | salamandra-7B | 31.0 ms | 1.28 ms (24×) |
  | TildeOpen-30b | 30.8 ms | 1.70 ms (18×) |
  | EuroLLM-9B | 31.6 ms | 2.13 ms (15×) |
  | Qwen2.5 / Qwen3-Coder / gpt-oss / OLMo-2 / granite | 0.31–0.34 ms | 0.27–0.30 ms (1.1×) |

  Nine of the fourteen models on the bench box were affected, including **five
  of the six certified European models**. It is genuinely quadratic — `ms/n²`
  is flat at 4.5 across a 30× range of input — so 16,000 characters on one
  line took **940 ms** on gemma-4, single-threaded, before a token is
  generated, on every request. End to end, a warm one-token request with a
  4,000-character single-line prompt on gemma-4-E4B goes **110 ms → 39 ms**.

  The fix is a priority queue of merge candidates, O(n log n). It must be
  **exact** — ids are load-bearing, and every `greedy_reference` certification
  is a claim about output that shifting one id would invalidate — so ordering
  reproduces the old scan exactly: best key first, leftmost on a tie, because
  the old loops compared with a strict `>` / `<` while walking left to right.
  Verified byte-identical against the previous implementation on all 14 local
  models over 1,965 records × 4 flag combinations = **110,040 comparisons**
  per model, covering prose, source, JSON, CJK, Devanagari, Thai combining
  marks, emoji ZWJ sequences and adversarial no-space runs.

  It did not start out exact, and the harness is why that was caught: an
  absorbed symbol keeps its length and its `next` pointer, so a queued
  candidate naming a symbol that had itself been swallowed as somebody's
  right-hand side still passed the liveness test and merged a symbol no longer
  in the list. On EuroLLM that turned `"  index."` into three ids instead of
  two, in 1,100 of 7,860 dumps. Absorbed symbols now have their length zeroed.
  `tests/test_tokenizer_merge.c` is the permanent gate: `tok_merge_force()`
  runs both paths on one binary and requires identical ids across every
  committed vocabulary fixture, the 721-line corpus and the adversarial
  shapes. Confirmed it fails on the pre-fix build.

  Short segments keep the rescan (`MERGE_QUEUE_MIN`), because the first
  measured version was **20% slower** on Qwen2.5 and gpt-oss while being 82×
  faster on gemma-4 — a queue costs more than re-reading three pairs. With the
  threshold and one allocation instead of four, the short-word models come out
  ~1.1× faster rather than slower.

  Also measured and **declined**: caching compiled schemas. A realistic strict
  tool envelope parses and compiles in 10 µs (3 tools) to 35 µs (12 tools),
  against requests that run for hundreds of milliseconds. A cache would add a
  key, an eviction policy and a lifetime hazard — the `snode` is live for the
  whole generation — to save 0.003% of a request.

- **Phase 5 — a reservation is a budget for the server, not for one slot.**
  `--reserve-vram P` with `-c 0` auto-fits the context to whatever the
  reservation leaves after the weights. Every slot ran that arithmetic alone,
  which billed **the weights N times and the KV cache once** — backwards on
  both halves, and the KV half is the dangerous direction. Measured on
  Qwen2.5-7B, `--reserve-vram 40 --parallel 4 -c 0` on a 25.37 GB device
  (10.15 GB budget): all four slots independently auto-fit to 32768 and each
  allocated its own 1.88 GB cache, for **12.47 GB — 23% over the
  reservation**. The only thing that kept it from being far worse is that the
  train context capped the window; a model with a longer train context would
  have overrun by close to the slot count.

  `model_params` gains `n_seq`, and the auto-fit now divides the KV and the
  activation head by it while still counting one weights copy. Same
  configuration after: context 19139, **9.25 GB, inside the budget**. A
  single slot and the one-shot CLI are unchanged (still 32768).

  Worth recording because it was measured rather than assumed: setting `n_seq`
  only for the slots `server_run` creates made it **worse, not better** —
  14.75 GB. Slot 0 is the model `main.c` preloaded, so it kept sizing itself
  alone at 32768 while slots 1–3 chose 19139, and the CUDA shared-weight
  registry keys on context — so the disagreement forced a *second* 4.7 GB
  upload of the same weights. The flag is set before the first load.

- **Phase 5 — one parse per file, not one per slot.** `model_t` fused the
  weight side and the per-sequence side, and the header had said so for a
  while: *"the struct itself is not yet split into two types… the sharing was
  pushed into the backend first, where the duplication actually cost
  gigabytes."* The host side was never done, and the bill was not the weights
  — those are mmap'd and the page cache already dedupes them — it was the
  **parse**.

  Measured on Qwen2.5-7B-Instruct-Q4_K_M, `--serve -c 512 --gpu off`, touched
  host memory after startup:

  | slots | before | after |
  |---|---|---|
  | 1 | 51.3 MB | 51.2 MB |
  | 2 | 81.0 MB | 51.3 MB |
  | 4 | 140.3 MB | 51.4 MB |

  **29.7 MB per extra slot → 87 KB.** Reading the file's own metadata
  accounts for 15.3 MB of what was being repeated: `tokenizer.ggml.tokens` and
  `tokenizer.ggml.merges` are 303,454 strings, each its own `malloc`, re-made
  for every slot — for a vocabulary the slots **never read**, because they
  share one tokenizer built from slot 0's file. The rest is the f32 conversion
  of every norm and bias, the tensor directory, and the pages each duplicate
  mapping faults in separately.

  `model_load` now splits at the line where it stops reading the file and
  starts sizing buffers: `model_bind_weights` produces the immutable half,
  `model_alloc_runtime` the per-sequence half. The immutable half lives in a
  refcounted record keyed on path plus file identity (size, inode, mtime,
  ctime) — a model rebuilt on disk between two loads must not be served out of
  the previous parse — plus the only two parameters the bind phase reads.
  Every `model_t` sharing a record holds **aliasing pointers**, so field
  access is unchanged and only ownership moved; that is what kept this out of
  `cuda.c` and `metal.m` entirely. It is deliberately the same shape
  `cuda.c` has used for the device upload since the MoE work.

  What is **not** shared, and why: the rope tables. YaRN auto-extension keys
  off the requested context and phi3 picks its LongRoPE factor set the same
  way, so two slots of one file with different `-c` legitimately want
  different tables — they are built after the seam and stay per-instance,
  along with the KV cache, all activation scratch, the thread pool, the VRAM
  lease and the expert placement array.

  `tests/test_shared_weights.c` already checked that two instances agree, stay
  isolated, and free exactly once in any order. All three still pass on a
  build that copies the whole file per instance, so they could not have caught
  a regression here — the test now also asserts the aliasing directly, and
  that a load differing in a weight-side parameter gets its own parse.
  Verified the four new assertions fail on a sharing-blind build.

  Also: `make test-shared-asan` was **red before this change and unrelated to
  it** — 5,280 bytes in 12 allocations, every one below `cuInit` in
  `libcuda.so`, reported identically at the previous commit. A leak gate that
  always fails is a leak gate nobody reads, and this is precisely the change
  it exists to check, so `tests/lsan.supp` suppresses that library (not a
  wildcard: a leak in runner's own frames during a CUDA call is still
  reported). The gate is green and reports the suppression matching exactly
  those 12 allocations.

- **RNR-019 — `server.c` is seven files instead of one.** 4,702 lines
  combining socket portability, HTTP parsing, routing, request validation,
  three protocol translations, SSE streaming, model registry and swap
  lifecycle, thread queues, continuous batching and shutdown. Now:

  | file | lines | owns |
  |---|---|---|
  | `http.c` | 211 | sockets, request parsing, response writing |
  | `registry.c` | 324 | model residency, swap, TTL reaper, admission queue |
  | `scheduler.c` | 343 | the continuous-batching decode thread |
  | `completion.c` | 1,757 | the generation loop and all three wire framings |
  | `api_responses.c` | 414 | Responses request → chat |
  | `api_anthropic.c` | 495 | Anthropic Messages request → chat |
  | `server.c` | 1,135 | routes, HTTP dispatch, capabilities, listener, shutdown |

  Every extraction is a **verbatim text move** — linkage is the only edit —
  each verified by comparing the multiset of non-comment lines before and
  after, and each gated on `make test` plus the 307-case conformance suite
  before the next one started. That gate is the reason this work waited: the
  suite had a ~30% flaky test until 0.1.5 and could not have told a broken
  refactor from a fired flake.

  Two of the seams are genuinely narrow and one is not, which is worth being
  precise about. `scheduler.h` is six functions and `SCH` is fully private —
  `sched_shutdown` was the only outside reference, so it moved too.
  `completion.h` is six declarations, because the routes reached the whole
  generation-and-framing complex through `run_completion` and nothing else.
  `api.h` is three. But `server_int.h` exposes the `SV` global as a declared
  `server_state` type rather than hiding it: that is the minimum needed for
  these to be separate translation units at all, and **de-globalising `SV`
  into a context threaded down from `server_run` is the rest of the finding**,
  a behavioural change rather than a move, deliberately not attempted here.

  Two things found on the way. `test_bind.c` — the source-text gate on the
  loopback-only bind — would have been quietly hollowed out: it asserted
  `typedef SOCKET sock_t;` appears in `server.c`, which is no longer where
  that lives, and a check left pointing at a file the code moved out of passes
  vacuously. Each check now follows the code it guards, and the
  forbidden-resolver scan (`getaddrinfo`, `inet_pton`, `INADDR_ANY`,
  `SO_BINDTODEVICE`…) was widened to the new transport and admission files,
  which are the more natural place to smuggle in an escape hatch. Confirmed
  each still fails by planting the forbidden text.

  And the scheduler is `scheduler.c`, not `sched.c`, because `src/` goes on
  the include path with `-I` for every test target and `-I` directories are
  searched **before** the system ones — so `src/sched.h` silently shadowed the
  standard `<sched.h>`, which `<pthread.h>` includes. Every test that reached
  `pthread.h` got the batching scheduler instead and failed with
  `unknown type 'slot_t'` inside a system header. `scripts/check-generated.py`
  now fails the build if any `src/*.h` collides with a C or POSIX header name;
  the symptom is a wall of type errors in a file nobody edited and the cause
  is invisible from there.

- **RNR-018 — `runner.h` is thirteen module headers instead of one.** The core
  header declared the whole engine: GGUF internals, the tokenizer maps, the
  full mutable `model_t`, the backend contract, the VRAM registry, the sampler,
  the JSON and schema validators, chat templates, the tool-call envelope and
  the generation engine — 1,291 lines visible to every translation unit and
  every test. `model.c` could reach the HTTP-facing tool envelope; `sample.c`
  could reach the GGUF tensor directory. Nothing did, but nothing stopped it.

  Now: `fp16.h`, `quants.h`, `gguf.h`, `tpool.h`, `tokenizer.h`, `model.h`,
  `vramreg.h`, `gpu.h`, `sample.h`, `jsonmode.h`, `schema.h`, `template.h`,
  `engine.h`. `runner.h` remains and includes all thirteen, so every consumer
  that wants the whole engine — the CLI, the server, the GPU backends, the
  tests — keeps one include and sees exactly what it saw before. Twelve
  single-module translation units now include only their own boundary:

  | TU | module headers visible |
  |---|---|
  | `sample.c`, `jsonmode.c`, `vramreg.c` | 1 of 13 |
  | `gguf.c`, `tokenizer.c`, `schema.c` | 2 |
  | `quants.c`, `template.c`, `quantize.c` | 3 |
  | `gpu_none.c` | 6 |
  | `model.c` | 7 |
  | `engine.c` | 12 |

  The split is a **verbatim text move**: every declaration was checked to
  appear exactly once across the thirteen files and to be identical to the
  original, by comparing the multiset of non-comment lines before and after.
  No signature, type or comment was reworded — a move can be audited, a
  rewrite has to be re-reviewed.

  `src/cuda.c` and `src/metal.m` deliberately still include `runner.h`: they
  are owned by the CUDA box, and one of them has CRLF line endings, so
  touching them here would have handed that machine a whitespace conflict for
  no gain.

  Two things this does **not** buy, stated because it would be easy to assume
  otherwise. It does not speed up builds: `make` compiles all sixteen sources
  in a single command with no object files, so header granularity has never
  affected rebuild cost. And it does not split immutable weights from
  per-sequence state inside `model_t`, which is the other half of RNR-018 and
  a real interface change rather than a move. `Makefile` gained
  `HDR = $(wildcard src/*.h)`, replacing 26 hardcoded `src/runner.h`
  prerequisites — without it the split would have quietly stopped a change to
  any of the new headers from triggering a rebuild.

## v0.1.5-alpha — 2026-08-02

- **Generalized MoE router.** The router was hardcoded to softmax + top-k +
  renormalize. It now carries the knobs the Llama-4, DeepSeek-V3 and GroveMoE
  families need, transcribed from llama.cpp's `build_moe_ffn`:
  `expert_gating_func` (softmax | sigmoid | softmax-over-selected-weights |
  sqrt-softplus), `exp_probs_b` (a bias applied to **selection only** — the
  weights still come from the unbiased probabilities, which is the point of
  DeepSeek's aux-loss-free balancing), group-limited top-k
  (`expert_group_count` / `expert_group_used_count`), `expert_weights_scale`
  and `expert_weights_norm`. Also picked up from the reference: the
  renormalization divisor is clamped to the smallest normal fp16, so a
  degenerate all-zero row cannot divide by zero — the old code had no clamp.

  Every default reproduces the previous path bit-for-bit: Qwen3-Coder-30B,
  gpt-oss-20b and gemma-4-26B-A4B are byte-identical across the change.
  **CUDA refuses rather than approximates** — `k_moe_route` is softmax + top-k
  with no bias input, so a model needing any non-default knob falls back to the
  host naming the knob.

  Gated by `tests/test_moe_router.c`, one dense-oracle fixture per knob,
  compared in **logit** space. That distinction is not academic:
  `expert_weights_norm` and all three alternative gating functions pass a
  *text* comparison on a binary that does not implement them at all, because a
  2x change in the FFN contribution often does not move a greedy argmax.

- **The flaky composition test is fixed, and it was not the assertion everyone
  thought.** `test_schema_batch_prefix_cancel_and_speculation_compose` failed
  about three runs in ten under whole-suite load and passed every time in
  isolation. It was read as the speculative-acceptance assertion — both render
  as `assert 0 > 0` — and an earlier fix hardened that one. The failing line
  was `prompt_forked_tokens > 0`, the prefix-cache fork.

  Root cause, measured rather than inferred: **a resident prefix can be forked
  by at most `parallel` requests at once.** With `--parallel 2`, four
  concurrent requests sharing a warm prefix report forks `[66, 67, 0, 0]`. The
  test issues *three* concurrent requests — a cancellation plus both schema
  ones — and required both schema responses to fork, which holds only when the
  cancellation loses the race. A coin flip, and the ~1-in-3 rate follows from
  it directly.

  It now asserts that at least one concurrent request forked: the shared prefix
  survived poisoning, cancellation and concurrency. Twenty whole-suite runs
  with no occurrence of this failure, against a 0.08% chance of that if the
  original rate remained.

  Two things found on the way, both filed rather than folded in: after this
  sequence a *sequential* request never forks at all — deterministic, not a
  race, and a possible caching inefficiency; and `test_signal_during_startup`
  has its own much rarer flake (~1 in 20), previously masked by this one.

- **Socket errors are reported through the right platform channel.** A failed
  bind printed `strerror(errno)`, which on Windows is simply the wrong source —
  Winsock reports through `WSAGetLastError`, so a genuine bind failure printed
  "Success" or a stale unrelated error, which is worse than printing no reason.
  `sock_errstr()` now sits beside the other socket shims: `strerror(errno)` on
  POSIX, `FormatMessage` on Winsock with a numeric fallback. `listen()` reports
  a reason too, where it previously reported none at all.

- **First published vLLM row on the agent-torture matrix.** Same 100-case
  matrix, same model (SmolLM2-1.7B-Instruct), same box: **Runner 100/100,
  vLLM 0.26.0 20/100**, every vLLM failure in the `protocol` category. Recorded
  with its deviations stated rather than buried — it is the full 100-case
  matrix, not the 12-case subset the earlier rows used, so it is not comparable
  to the published `12/12 vs 5/12`; and vLLM ran on CUDA against the fp16
  checkpoint while Runner ran on CPU against the Q4_K_M GGUF.
  Evidence: `tests/torture/results/2026-08-02-smollm2-1.7b-vllm/`.

  Getting vLLM to start took three attempts, and the reasons are themselves the
  comparison: torch/triton JIT-compile at startup and needed a C compiler this
  box does not have; then flashinfer needed full CUDA toolkit headers, also
  absent. That chain — a C compiler, a CUDA toolkit and an ~8 GB Python
  environment before the first token — is what "one file to ship" is measured
  against.

  **LM Studio remains unrun**: it is a GUI desktop application whose CLI
  requires the installed app, with no headless server install.

- **Apertus (`apertus`) is admitted: ungated MLP plus xIELU.** Its FFN has no
  `ffn_gate` — it is up → xIELU → down — so the gate tensor became optional
  for the ungated activation and stays required for every other dense arch.
  xIELU is transcribed from ggml's `op_xielu`: `alpha_p*x² + beta*x` above
  zero, `(expm1(min(x, eps)) - x)*alpha_n + beta*x` at or below. Its four
  parameters are read per layer, and from **un-prefixed** keys — llama.cpp
  spells them `xielu.alpha_n`, not `apertus.xielu.alpha_n` — accepting either
  a scalar shared by all layers or a per-layer array.

  Runner already had Apertus's `tekken` tokenizer and chat template; this is
  the forward pass, so the EU-column Apache-2.0 blocker is now down to
  verifying against a real checkpoint.

  **CUDA refuses**: the dense FFN encoder always issues a gate matvec and
  there is no xIELU kernel, so it runs on CPU with a message saying so.
  Gated by `tests/test_apertus.py`, which compares identity xIELU parameters
  (`alpha_p = alpha_n = 0`, `beta = 1`, making it the identity map) against
  real ones — a build ignoring the parameters produces the same output for
  both. The previous binary refuses the architecture outright.

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
