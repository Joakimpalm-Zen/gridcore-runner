# MoE measurements on gemma-4-26B-A4B — results

Produced 2026-08-04 against the handover `ai-platform-handover-2026-08-04-moe-measurements.md`,
on `gridcore-runner` @ `021e5bc` (this trip's tree started there; instrumentation
described below is layered on top, uncommitted — see "Instrumentation diff").

## Machine

- 128 logical cores, 268.7 GB RAM, NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation
  Edition (MIG 1g.24gb slice, 25.4 GB VRAM). `./runner --caps`.
- Toolchain: this box has no system `cc`; built with
  `CC=x86_64-conda-linux-gnu-gcc CXX=x86_64-conda-linux-gnu-g++` from the
  `~/.conda/envs/ccbuild` env (gcc 15.2.0 / clang 22.1.8 / nvcc 13.0), per the
  precedent in `tests/torture/results/2026-08-03-smollm2-1.7b-v2/README.md`.
- Model: `models/gemma-4-26B-A4B-it-Q4_0.gguf` (14.6 GB). Both local
  `gemma-4-26B-A4B` GGUFs on this box (`...-it-Q4_0.gguf` and
  `gemma-4-26B_q4_0-it.gguf`) trace back to `google/gemma-4-26B-A4B-it` in
  their metadata but **neither carries official-QAT provenance** — no
  `general.quantized_by`/source pointing at
  `google/gemma-4-26B-A4B-it-qat-q4_0-gguf`, and the second file's
  `general.name` is a bare `"Hf"` placeholder, consistent with a local
  conversion rather than a downloaded release. Both pass agent-torture's
  structured-output smoke test (2/2) so neither is obviously broken; picked
  `...-it-Q4_0.gguf` for having complete `general.base_model.*` metadata. This
  means the measurements below characterize *a* Q4_0 quantization of this
  model, not verified-identical-to-the-official-release behavior.
- 128 experts, top-8 routed, 30 layers, 262144 trained context (`gemma4.expert_count`,
  `expert_used_count`, `block_count`, `context_length` from the GGUF header) —
  matches the handover's assumed 8-expert schema exactly.

## Task A — grammar fast-forward A/B (`out/26b-off`, `out/26b-on`)

Both arms: `agent-torture.py --cases 35`, CPU (`--gpu off`, the driver's
default), same model, same 35 cases.

| | OFF (baseline) | ON (`RUNNER_GRAMMAR_FF=1`) |
|---|---|---|
| elapsed_ms | 294,168.85 | 304,689.19 |
| passed/failed | 31/4 | 31/4 (same 4 case ids) |
| median gen tok/s | 13.61 | 12.06 |

**Grammar fast-forward is 3.6% *slower* on this model, not faster.** That's
the opposite of the pattern the handover cites for smaller/other models
(33–58% acceptance paired with a net wall-clock win). It is not for lack of
acceptance — grammar accepted/drafted summed over `out/26b-on/runner.log`'s
`spec: ... grammar A/D` lines is **217/464 = 46.8%**, comfortably inside the
range other models showed. The likely mechanism: this is a 128-expert MoE
verified one token at a time — each speculative-verify forward still pays the
full per-token MoE routing + expert-gather cost regardless of how many draft
tokens it validates in one round, so the per-round overhead the fast-forward
adds isn't being amortized over cheaper per-token compute the way it would be
on a dense model. Reported as found; no code changed on this path (measurement
only, per the handover's scope).

**Identity: 0 mismatches across all 35 cases.** Compared `content`/`tool_calls`
between arms after decoding `response.body` (base64) and, for the
`stream_normalization` category, reconstructing the full message from its SSE
`data:` chunks (delta content concatenation + tool-call argument
concatenation by index) rather than comparing raw bytes — a naive raw-body
diff would have flagged all `stream_normalization` cases as "mismatched"
over `created`-timestamp and chunk-boundary noise that isn't a real
correctness difference.

## Task B — R0 routing-locality instrumentation

### Instrumentation

`RUNNER_MOE_TRACE=path` (env, off by default, one cached `getenv` + cached
`FILE*` when unset — same shape as the existing `RUNNER_DEBUG_ACT`) appends
one JSONL record per routed token per MoE layer:
`{"pos":N,"layer":L,"experts":[8 ids],"gates":[8 weights]}`.

Hooked at all three CPU MoE call sites in `src/model.c`: `moe_ffn_token`
(decode), `moe_ffn_grouped` (prefill), and `gemma_moe_ffn` — the last one is
the actual dispatch for this model (`ly->moe_gemma`), not the generic
`moe_ffn`; missing it would have silently traced nothing for gemma-4.
Token position comes from a new `model_t.fwd_pos` field, set once per
`model_forward_batch` call and read as `fwd_pos + b` for batch element `b`.

**CPU path only.** `src/cuda.c` has its own device kernel (`k_moe_route`) that
the code there says "mirrors moe_route() bit for bit" — routing on GPU never
reaches this call site. All three replay workloads below therefore ran
`--gpu off`; see "Gotchas" for how that was discovered the hard way.

Diff (`src/model.c` + `src/model.h`, 41 lines, uncommitted — see "Handoff"):

```diff
diff --git a/src/model.c b/src/model.c
index 1e388e5..ba28110 100644
--- a/src/model.c
+++ b/src/model.c
@@ -2259,6 +2259,39 @@ static void dbg_stat_f16(const char *tag, int layer, const f16_t *v, size_t n) {
             layer, tag, n, absmx, n_inf, n_nan);
 }
 
+// ------------------------------------------------------ MoE routing trace
+// RUNNER_MOE_TRACE=path appends one JSONL record per routed token per MoE
+// layer: {"pos":N,"layer":L,"experts":[...],"gates":[...]}. Off by default;
+// zero cost when unset (one cached getenv + a cached FILE*, same shape as
+// dbg_act_mode above). CPU path only — the CUDA MoE kernels route on-device
+// and never reach this call site.
+static FILE *moe_trace_file(void) {
+    static FILE *fp = NULL;
+    static int opened = 0;
+    if (!opened) {
+        opened = 1;
+        const char *path = getenv("RUNNER_MOE_TRACE");
+        if (path && *path) {
+            fp = fopen(path, "a");
+            if (!fp)
+                fprintf(stderr, "warning: RUNNER_MOE_TRACE=%s: could not open for append\n", path);
+        }
+    }
+    return fp;
+}
+
+static void moe_trace_emit(int pos, int layer, const int *sel, const float *selw, int used) {
+    FILE *fp = moe_trace_file();
+    if (!fp) return;
+    fprintf(fp, "{\"pos\":%d,\"layer\":%d,\"experts\":[", pos, layer);
+    for (int t = 0; t < used; t++) fprintf(fp, "%s%d", t ? "," : "", sel[t]);
+    fprintf(fp, "],\"gates\":[");
+    for (int t = 0; t < used; t++) fprintf(fp, "%s%.6g", t ? "," : "", selw[t]);
+    fprintf(fp, "]}\n");
+    fflush(fp);   // durable across a killed server: --serve traces run for the
+                   // process's whole lifetime, not to a matching fclose
+}
+
 // ---------------------------------------------------------------- forward
 
 // suppress_tokens checkpoint workaround: a large finite constant instead of
@@ -2610,6 +2643,7 @@ static void moe_ffn_token(model_t *m, const layer_t *ly, float *xin) {
     int   sel[256];
     float selw[256];
     moe_route(m, ly, xin, n_embd, ne, used, sel, selw);
+    moe_trace_emit(m->fwd_pos, (int)(ly - m->layers), sel, selw, used);
     for (int i = 0; i < n_embd; i++) m->moe_out[i] = 0.0f;
     for (int t = 0; t < used; t++) {
         int e = sel[t];
@@ -2654,6 +2688,8 @@ static void moe_ffn_grouped(model_t *m, const layer_t *ly, int n, int xdim) {
         float *xin = m->xb + (size_t)b * xdim;
         moe_route(m, ly, xin, n_embd, ne, used,
                   m->moe_sel + (size_t)b * used, m->moe_selw + (size_t)b * used);
+        moe_trace_emit(m->fwd_pos + b, (int)(ly - m->layers),
+                       m->moe_sel + (size_t)b * used, m->moe_selw + (size_t)b * used, used);
         float *out = m->moe_out_b + (size_t)b * n_embd;
         for (int i = 0; i < n_embd; i++) out[i] = 0.0f;
     }
@@ -2769,6 +2805,7 @@ static void gemma_moe_ffn(model_t *m, const layer_t *ly, int n, int xdim) {
             sel[t] = best; selw[t] = bp; denom += bp; m->moe_logits[best] = -1.0f;
         }
         for (int t = 0; t < used; t++) selw[t] /= denom;
+        moe_trace_emit(m->fwd_pos + b, (int)(ly - m->layers), sel, selw, used);
         if (dbg_act_now() && b == n - 1) {
             fprintf(stderr, "ACT L%-3d %-16s", (int)(ly - m->layers), "moe-experts");
             for (int t = 0; t < used; t++)
@@ -2840,6 +2877,7 @@ bool model_moe_ffn_cpu(model_t *m, int layer, int n) {
 
 float *model_forward_batch(model_t *m, const int32_t *tokens, int n, int pos,
                            bool want_logits) {
+    m->fwd_pos = pos;
     if (m->qwen35 && pos == 0) {
         int convdim = 2 * m->ssm_state * m->ssm_groups + m->ssm_inner;
         int hv = m->ssm_inner / m->ssm_v_heads;
diff --git a/src/model.h b/src/model.h
index 611132d..5400037 100644
--- a/src/model.h
+++ b/src/model.h
@@ -178,6 +178,9 @@ typedef struct {
     float    *moe_selw;    // [n_batch][n_expert_used] per-token renormalized weights
     int      *moe_gidx;    // [n_batch] current expert's token indices
     float    *moe_gw;      // [n_batch] current expert's per-token weights
+    int       fwd_pos;     // starting KV position of the batch this forward call
+                            // is processing; RUNNER_MOE_TRACE reads it to log
+                            // each routed token's absolute sequence position
     bool      rope_neox;     // NeoX-style rotation (qwen2) vs adjacent pairs (llama)
     float     rms_eps, rope_base;
     float     rope_mscale;   // YaRN attention magnitude scale (1.0 = off)
```

### Gotchas hit collecting the traces (fixed before trusting any numbers)

1. **`--serve` defaults to `--gpu auto`.** First chat-workload attempt ran on
   a server started without `--gpu off`; all 30 layers landed on the CUDA
   MIG slice, the CPU trace hook never fired, and the output file was never
   created. Silent — no error, just an absent file. Re-ran with `--gpu off`
   explicit.
2. **Raw completion prompts break down on this `-it` model.** Both the
   "free chat" and "long document continuation" workloads were first tried
   as raw `-p`/`-f` completions (no chat template). Both degenerated into
   token loops (`"English, English, English..."`; later
   `"...dethought-dethought-dethought..."`) after a short coherent run —
   not evidence of a corrupt quantization (the same file produces clean,
   on-topic, multi-paragraph output through `/v1/chat/completions`), just an
   instruction-tuned model going out-of-distribution on a prompt shape it
   wasn't tuned for. Re-ran both through the chat-completions endpoint: the
   chat workload as a direct user turn, the document workload as an
   instruction to continue the pasted document verbatim ("don't summarize,
   just continue writing it").
3. **`kill -9` on the traced `--serve` process loses the last buffered
   line.** stdio is not flushed by `SIGKILL`. Both re-run workloads had
   exactly one truncated trailing JSONL line (confirmed by comparing against
   `agent-torture`'s trace, which exits its server normally via the
   torture script and had zero corrupt lines). Added `fflush(fp)` after
   every trace write in `moe_trace_emit` — cheap (this path only exists when
   tracing is on) and makes the trace durable against a killed server, which
   is the realistic way this instrumentation gets used. Dropped the one
   truncated line from each of the two affected files rather than
   re-running ~10 minutes of generation for one record out of 45k/161k.

### Workloads replayed (`traces/*.jsonl`, `analyze_moe_trace.py`)

| workload | tokens (prompt+gen) | trace records | segments |
|---|---|---|---|
| `agent-torture` (35 cases, schema-constrained) | — | 152,940 | 26 (some cases share KV via prefix cache) |
| `chat` (mixed EN/SV, free-form) | 178+1335=1513 | 45,368 | 1 |
| `doc` (README.md continuation) | 4466+914=5380 | 161,376 | 1 |

All 30 layers are MoE (`gemma4.block_count` == number of `is_moe`/`moe_gemma`
layers here — no dense layers to average out).

### Findings

**Standing committee.** Smallest per-layer expert set covering 50%/70% of
routing mass, averaged over all layers:

| workload | mean @50% | mean @70% | range |
|---|---|---|---|
| agent-torture | 14.9 | 29.3 | 9–22 / 19–40 |
| chat | 16.0 | 30.3 | 10–23 / 21–41 |
| doc | 15.8 | 30.8 | 9–29 / 19–51 |

Out of 128 experts. This is **larger than the 2–5-expert "standing
committee" the handover's literature summary cites for other fine-grained
MoEs** — routing mass here is meaningfully less concentrated. Consistent with
the same summary's caveat that shared-expert architectures (gemma-4-26B-A4B
has a dense shared branch alongside the routed one — see `gemma_moe_ffn`)
measurably lower routing consistency: the shared branch can absorb the
common-case signal, leaving the routed experts a flatter, less
committee-like distribution to work with.

**Segment cache-hit curves** (oracle = per-window top-C by selection
frequency; LRU = a real LRU of capacity C, cold-started at each window —
see the analysis script's docstring for why windowed-reset LRU was chosen
over one warm cache for the whole segment). Headline config, cache=16
(2× the 8 active experts), window=256:

| workload | oracle | LRU |
|---|---|---|
| agent-torture | 0.564 | 0.502 |
| chat | 0.574 | 0.564 |
| doc | 0.555 | 0.473 |

At cache=32 (4× active), window=256:

| workload | oracle | LRU |
|---|---|---|
| agent-torture | 0.771 | 0.752 |
| chat | 0.782 | 0.769 |
| doc | 0.762 | 0.709 |

**Against the handover's own decision line — oracle ≥ ~75% at cache = 2×
active experts ⇒ 8 GB streaming is alive — this model misses it by a wide
margin (~0.56–0.57, not ≥0.75) at 2× active.** It only clears that bar
(barely, 0.76–0.78) at 4× active (cache=32 experts, still just 25% of the
table). LRU tracks oracle within ~0.02–0.09 at every setting tried, so an
actual online cache captures most of the achievable locality — the ceiling
itself is just lower than hoped for a 2×-sized cache. Window size (64 vs 256
vs 1024) barely moves the number at fixed cache size, so locality here isn't
a "look at short bursts" effect — it holds (or doesn't) at the same rate
across the timescales tested.

**Gate-reuse / prefetch accuracy.** Used the offline approximation, not the
true `P(set@L+1 | set@L)` a router probe over layer L's hidden state would
give — no hidden-state capture was added (kept the diff small, per the
handover's own fallback clause). What's reported is the raw expert-set
overlap between adjacent layers for the same token:

| workload | mean overlap /8 |
|---|---|
| agent-torture | 0.55 |
| chat | 0.50 |
| doc | 0.47 |

~6% of a token's 8 experts at layer L happen to also be selected at layer
L+1 — barely above what 8-of-128 random draws would give by chance
(8×8/128 ≈ 0.5), i.e. **this offline proxy finds essentially no
gate-reuse signal.** That is a property of the proxy as much as the model:
raw set-overlap ignores the router's actual next-layer decision function
entirely, so it's a strong underestimate of what a real hidden-state-driven
prefetcher could do. Don't read "0.5/8" as "prefetching doesn't work here" —
read it as "this cheap proxy couldn't find the signal a proper probe might."

**Working-set growth** (distinct experts touched, out of 128):

| workload | @32 tok | @128 tok | @512 tok | @2048 tok |
|---|---|---|---|---|
| agent-torture | 60.7 | 91.9 | — (segments too short) | — |
| chat | 66.7 | 87.9 | 109.8 | — (segment too short) |
| doc | 66.8 | 94.6 | 112.6 | 121.0 |

By 2048 tokens (doc workload, the only segment long enough), **121 of 128
experts have been touched at least once** — near-total coverage — even
though a ~16-expert subset carries half the routing *mass*. Concentration of
mass and a small working set are different properties here: this model has
the former but not much of the latter, which is exactly why the cache-hit
numbers above land in the "moderate, not high" range instead of near-100%.

### Conclusion (per the handover's decision thresholds)

Oracle hit rate at cache = 2× active experts is **~0.56–0.57, below the
~75% bar** for "8 GB expert streaming is alive." By the handover's own
rule this **pivots the expert-residency program toward 16 GB-stability +
pruning for gemma-4-26B-A4B**, not 8 GB streaming — unless a 4×-active
cache size (~32 experts/layer, still just 32 MB–ish of routed weights per
layer at this quant) is an acceptable target, where oracle hit crosses the
line (~0.76–0.78) and LRU is within a few points of it. Consistent across
three quite different workloads (schema-constrained tool calls, free chat,
document continuation) — this reads as an intrinsic property of the model's
routing, not a workload artifact.

## What's here

- `README.md` — this file.
- `analyze_moe_trace.py` — the analysis script (standing committee, cache-hit
  curves, gate-reuse overlap, working-set growth). Usage:
  `python3 analyze_moe_trace.py name=path.jsonl [name=path.jsonl ...]` → JSON
  on stdout.
- `analysis.json` — full per-layer output for all three workloads (the tables
  above are means over layers; per-layer detail, including the 9–51 expert
  range on standing-committee size, is in here).
- `traces/{agent-torture,chat,doc}.jsonl` — the raw traces.
- `chat-prompt.txt`, `doc-prompt.txt` — the two free-form prompts used
  (`doc-prompt.txt` is the first 16 KB of this repo's `README.md`).
- `*-server.log`, `*-gen.log` — raw runner stdout/stderr from each collection
  run, kept for anyone who wants to audit the tok/s or reproduce a run.

## Handoff

The `src/model.c`/`src/model.h` instrumentation diff above is **not
committed** — this box's git state had multiple machines pushing to `main`
today (see the session's other work), so a local commit was left uncommitted
rather than guessed at push rights or race a concurrent push. It's small,
off-by-default, and builds/runs clean: `make test` exits 0 on this tree with
the diff applied (re-run after the diff, not just before it). Apply it
directly from the diff above, or ask and it'll be committed/pushed from here.

Didn't get to: gpt-oss-20b (secondary priority per the handover, "if time and
disk allow" — disk allows, `models/gpt-oss-20b-MXFP4.gguf` is present and
12.1 GB, but this trip's time went to getting Task A's honest number and
chasing the three Task B data-collection bugs above to ground before trusting
any of it).
