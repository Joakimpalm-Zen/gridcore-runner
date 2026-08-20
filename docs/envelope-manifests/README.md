# Measured-envelope manifests

Committed copies of the `<model>.envelope.json` sidecars assembled by
`scripts/certify-envelope.py` (schema `xyntetik.runner.envelope.v1`). Each file
here is byte-identical to the sidecar that lives beside its GGUF on the
measurement box, so the load-time envelope gate (`src/envelope.c`, read as
`<model-path>.envelope.json`) can be reproduced offline. Models are not tracked
in this repo; these manifests are the durable evidence of what was measured.

A manifest is an **index over evidence that already exists**, never a hand-written
claim: `certified` requires a passing compat gate for the artifact's sha *and* a
named reference sha; `outside-envelope` records a measured failure; `experimental`
means no gate evidence exists for the artifact yet.

## The `tool_calling` block (reported-only)

A manifest may carry an optional, additive `tool_calling` block (the schema
version is unchanged — `xyntetik.runner.envelope.v1` — because it only adds
fields). It summarises how a model behaves under tool use. It is **reported-only
by design**: the load-time gate reads *only* the top-level `verdict`, so nothing
in this block ever certifies, refuses, or otherwise changes a load. Runner
surfaces it as one extra banner line at load; `runner --tool-info -m <model>`
reports the native-protocol fields alone, straight from the model, without a
manifest.

Any sub-block may be `null` when its evidence was not gathered; the banner omits
whatever is absent. The fields:

| Field | Meaning | Evidence it indexes |
|---|---|---|
| `truncation_recovery` | Does the engine recover a tool call when the response is cut off mid-emit? `holds`, `rungs_passed`/`rungs_total`. | A `xyntetik.truncation-benchmark.v1` report, passed to the certifier as `--truncation-report`. Scope is the **engine** (`measured.scope:"engine"`), measured on a proxy model — not a per-artifact claim. |
| `schema_shape` | Does the tool-call **shape** still parse and select the right tool at a low quant? `held_to_quant`, `schema_conformance_rate`, `tool_selection_rate`. | A `xyntetik.quant-fidelity.v1` report, passed as `--quant-fidelity-report`; the quant row read defaults to `Q4_0` and is set with `--quant-fidelity-quant`. |
| `agent_torture` | A pass/fail gate over an agent-style request matrix. `gate`, `requests`, `passed`, `failed`. | The certifier's agent-torture measurement (`measured`). |
| `native_tool_protocol` | The model's tool-call family and whether it speaks a native tool protocol. `tool_family`, `native`. | Detected from the model's chat template — the same source as `runner --tool-info`. |
| `gate` | A reported-only rollup (`pass`/`partial`/`fail`) over whatever sub-blocks are present. | Derived from the sub-blocks. **Never** changes the load decision. |

**Honesty note.** `schema_shape` holding at `Q4_0` is a claim about the call
**shape** — that a quantised model still emits a well-formed call and picks the
right tool — **not** about the correctness of the *argument values* it fills in.
Read it as "the envelope of the call survives the quant", nothing more. Likewise
`truncation_recovery` is an engine property measured on a proxy model, so it is
labelled `scope:"engine"` rather than attributed to this specific artifact.

## Manifests

### `Qwen3-30B-A3B-expq4_0-attnq8_0.gguf.envelope.json`

The flagship **selective-precision 30B** (HF `Joakimpalm-Zen`): Qwen3-30B-A3B with
attention + embeddings kept Q8_0 and the expert banks at Q4_0, produced via
`--type-plan` and measured against its Q8_0 source (sha `4ad960d1…`). It passes the
adopted quality bar at 17.99 GB (v2 top-1 99.50 %, mean KLD 0.0345) per the suite's
`exhaust-results-2026-08-15` phase 6.

- artifact sha256 `df02efa8…`, reference (Q8_0 source) sha256 `4ad960d1…`
- runtime tuple: runner 0.1.19-alpha / CUDA on the RTX PRO 6000 Blackwell box
- **verdict: `experimental`, and certification is measured-blocked.** The
  fidelity/adopted-bar PASS is a *different* bar from the compat gate this
  schema's `quality.gate` indexes. The `cpu_cuda` identity gate WAS run against
  this exact sha on 2026-08-20 (runner 0.1.19-alpha, full 48/48 CUDA offload,
  eager routing, TC=0; evidence:
  `docs/compat-reports/cpu-cuda-128/qwen3-30b-a3b-expq4_0-attnq8_0/report.json`):
  **8/9 prompts byte-exact**, and the ninth diverges at generated token 58
  (`" over"` on CPU vs `" cloudy"` on CUDA). That flip was evaluated under the
  MoE margin-qualified routing near-tie tolerance (owner-ratified 2026-08-20)
  and **correctly rejected**: the CPU side rates the CUDA pick 0.707 nats below
  its own best (CUDA side 0.090), outside the 0.5-nat band required on both
  sides — the CPU is confident, so this is a real divergence, not a routing
  coin-flip. The sidecar therefore stays un-certified rather than borrowing a
  tolerance it does not qualify for; recording `outside-envelope` (a measured
  failure) is the owner's call. Verified consumable: the runner loads the model
  (qwen3moe, 48 layers, 18.0 GB in VRAM) and the load-time gate reads this
  sidecar and reports its state.

### `granite-4.1-3b-Q8_0.gguf.envelope.json`

The quickstart model (first-party IBM `granite-4.1-3b`, Q8_0) certified on the
**Metal** backend — the runtime dimension the CUDA flagship above cannot show.
Assembled on Apple Silicon (`arm64/metal`), so the runtime carries a real Metal
shader-source sha rather than a CPU/CUDA kernel-set. It demonstrates the gate and
the reported-only tool-calling axis end to end on a real model:

- artifact sha256 `c31f09b9…`, runtime tuple: runner 0.1.20-alpha / Metal on an M1
- **verdict: `experimental`** — no fidelity compat report was attached in this
  run, so there is no `quality.gate` to certify against (the model does pass the
  project's fidelity bar; that evidence simply is not indexed by this sidecar).
- **`tool_calling` gate `partial`** (reported-only, never affects the load):
  - `truncation_recovery` **holds, 6/6 truncated rungs recover** — `scope:"engine"`,
    from the committed `granite-4.1-3b` truncation benchmark (the property is an
    engine guarantee, independent of this artifact's quant/backend).
  - `native_tool_protocol` `generic` / **non-native** — granite has no native
    tool protocol; it uses the runner's generic constrained JSON envelope, which
    is the honest, accurate report for this family.
  - `schema_shape` and `agent_torture` are `null` — that evidence (a quant-fidelity
    ladder and an agent-torture matrix for this artifact) was not gathered here,
    so the fields are absent rather than invented, and the rollup is `partial`.

Verified end to end: with this sidecar beside the model the load-time gate reports
`envelope: measured for 0.1.20-alpha / metal, verdict experimental (not certified)`
and `envelope: tool-calling gate=partial — truncation 6/6, non-native generic`.
