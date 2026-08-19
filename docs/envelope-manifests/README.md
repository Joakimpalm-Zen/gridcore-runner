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

## Manifests

### `Qwen3-30B-A3B-expq4_0-attnq8_0.gguf.envelope.json`

The flagship **selective-precision 30B** (HF `Joakimpalm-Zen`): Qwen3-30B-A3B with
attention + embeddings kept Q8_0 and the expert banks at Q4_0, produced via
`--type-plan` and measured against its Q8_0 source (sha `4ad960d1…`). It passes the
adopted quality bar at 17.99 GB (v2 top-1 99.50 %, mean KLD 0.0345) per the suite's
`exhaust-results-2026-08-15` phase 6.

- artifact sha256 `df02efa8…`, reference (Q8_0 source) sha256 `4ad960d1…`
- runtime tuple: runner 0.1.19-alpha / CUDA on the RTX PRO 6000 Blackwell box
- **verdict: `experimental`.** The fidelity/adopted-bar PASS is a *different* bar
  from the compat gate this schema's `quality.gate` indexes. No compat-matrix
  report exists for this exact sha, so the manifest fails closed to `experimental`
  rather than inventing a `certified` verdict. Reaching `certified` needs one full
  compat gate run (load / tokenizer / greedy_reference / cpu_cuda / chat / tool)
  against sha `df02efa8…` — a 30B gate run that also needs a committed qwen3moe
  tokenizer reference-ids capture and a matching reference binary, neither of which
  exists yet. Verified consumable: the runner loads the model (qwen3moe, 48 layers,
  18.0 GB in VRAM) and the load-time gate reads this sidecar and reports its state.
