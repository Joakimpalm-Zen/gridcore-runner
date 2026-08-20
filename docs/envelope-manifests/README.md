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
