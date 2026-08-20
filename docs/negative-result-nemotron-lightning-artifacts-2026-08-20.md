# Negative result: three of the four planned Nemotron-3.5-Lightning artifacts

*2026-08-20. Status: inspected, measured, rejected before any of them was built.
Not published.*

The artifact program for `nemotron_h_moe`
(`NVIDIA-Nemotron-3.5-Lightning-30B-A3B`) opened with four candidate
derivatives. Three are dead, and each died to a different, cheap check that is
worth writing down because the arithmetic looks promising right up until the
check runs.

Evidence: `run/w0-inspection.md`, `run/w0/*` in the artifact-pipeline branch.
Runner `fb4309f`, `runner 0.1.20-alpha`, sources pinned at
`ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF`
revision `9d425fe18d84ab04da6aabb757d2e2807083d054`.

## 1. "Runner-NoMTP" — the MTP heads are not in the file

**The idea.** Runner loads and skips MTP/NextN tensors, so a copy with them
removed would be free bytes with a mathematically identical dense decode.

**What killed it.** A ranged read of the first 26 MB of the 63.18 GB BF16 GGUF
(header + tensor directory, no tensor data) and a full read of the local Q4_0's
directory: **401 tensors each, blocks `0..51`, zero matches for `nextn`, `mtp`,
`eh_proj`, `enorm`, `hnorm` or `shared_head`.** ggml-org ships the MTP head as a
self-contained `blk.52` in a *separate* `mtp-*.gguf` (19 tensors, 1.16 GB at
Q4_0), which a Runner user never downloads.

**Cost of the check:** minutes and ~26 MB of transfer. **Cost if skipped:** a
63 GB download, a new tensor-strip tool with its tests, and a byte-identity
proof between a file and a copy of itself.

**Generalisation worth keeping:** before building a strip tool, grep the tensor
directory. "The architecture has feature X" is not the same claim as "this file
carries feature X's tensors."

## 2. "Selective C1 (conservative)" — it is already the published file

**The idea.** Quantize the BF16 parent with Mamba/attention/shared-expert/
embedding at Q8_0 and the routed experts at Q4_0.

**What killed it.** That *is* ggml-org's Q4_0 build. The filename says Q4_0;
the tensor directory says otherwise:

| class | count | upstream `…-Q4_0.gguf` |
|---|---|---|
| `ffn_up_exps` / `ffn_down_exps` | 46 | **Q4_0** |
| `ffn_up_shexp` / `ffn_down_shexp` | 46 | Q8_0 |
| `ssm_in` / `ssm_out` | 46 | Q8_0 |
| `attn_q/k/v/output` | 24 | Q8_0 |
| `token_embd` / `output` | 2 | Q8_0 |
| `ffn_gate_inp` (router) | 23 | F32 |

Confirmed from the other direction: Runner's own C1-equivalent `--type-plan`
(`{"default":"q8_0","rules":[{"match":"_exps.weight","type":"q4_0"}]}`) reports
the output histogram `F32:237 Q4_0:46 Q8_0:118`, which is the upstream file's
histogram exactly.

**Generalisation:** run `scripts/gguf-inspect.py` on the comparator *before*
designing a plan against it. A "uniform quant" in the filename is a guess.

## 3. "Selective C3 (compact, routed experts Q3_K)" — the rows do not divide

**The idea.** Push the routed experts from Q4_0 to Q3_K for roughly a 4 GB
saving on an 18.9 GB file.

**What killed it.** Q3_K's super-block is 256 values wide and
`src/quantize.c:type_fits_row()` requires `ne[0] % 256 == 0`; a row that fails
is **retained at its source type and counted as `declined_width`**, so the run
succeeds and silently produces the un-narrowed file. Every Nemotron tensor that
carries mass has a row width that is an odd multiple of 128:

| tensor | `ne[0]` | ÷ 256 |
|---|---|---|
| `ffn_down_exps` | 1856 | 7.25 |
| `ffn_up_exps` | 2688 | 10.5 |
| `ffn_down_shexp` | 3712 | 14.5 |
| `ffn_up_shexp`, `ssm_in`, `attn_q/k/v`, `token_embd`, `output` | 2688 | 10.5 |
| `attn_output`, `ssm_out` | 4096 | 16 ✓ (0.9 % of weights) |

C3 would have emitted a file byte-identical to C2 and reported success.

**Generalisation:** a K-quant plan needs a geometry check, not just a size
estimate, and the failure mode is a silent no-op rather than an error.

## 4. What survives, and why the flagship changed shape

C2 (shared expert Q8_0 → Q4_0) is real but small: 458.7 M parameters moving
from 1.0625 to 0.5625 bytes each is **‑229 MB on 18.90 GB (‑1.2 %)**, bought by
lowering precision on an expert that is active on *every* token. It is not a
standalone artifact; it is at most a component.

That leaves **expert pruning** as the family's only live size lever, which
inverts the planned order: the flagship is prune-first, not precision-first.

Pruning was then measured, and it produced a second negative result plus one
methodological finding worth more than either:

- **The frontier is keep-126** — two experts of 128 per layer. keep-127
  (100.00% margin-qualified / 0.009 KLD) and keep-126 (99.50% / 0.026) pass the
  bar against the unpruned upstream Q4_0; keep-125 (96.75% / 0.066) is the
  first miss, and it degrades monotonically below. 1.37% of the file, with no
  throughput change — a sparse MoE decodes at the cost of its top-k, not its
  roster — so neither passing rung is worth publishing as weights.
- **The saliency ranking was worth more than three rungs of depth.** The first
  ladder used `--use-norms` (REAP-style gate × activation L2 norm) and measured
  **0.232** KLD at keep-124, from which the conclusion "Lightning tolerates no
  pruning at all" was written. Plain **gate mass** at the same depth measures
  **0.074** — 3.1× better, on a ranking that disagrees about 38 of the 92
  experts it drops. The conclusion was wrong and a second opinion caught it.
  Plausibly the squared-ReLU activation makes output magnitude track loudness
  rather than importance, so the norm term ranks loud experts above load-bearing
  ones; that mechanism is a hypothesis, the 3.1× is measured. **On a new family,
  measure both rankings before trusting either.**
- **A non-uniform (coverage-based) prune writes a file this architecture cannot
  load.** `--coverage 0.999` yields 122–128 experts per layer; the quantizer
  writes it and correctly leaves `expert_count` alone, and the loader then
  refuses with `blk.3 ffn_gate_inp is [2688,127], expected [2688,128]`. The
  general MoE path already reads each layer's own expert count (see
  `check_shape3`'s comment); the `nemotron_h_moe` branch validates against the
  model-level count instead. Fail-closed, so nothing miscomputes — but it makes
  the coverage mode unusable on this family and leaves `--prune-experts`
  offering a mode whose output is unloadable.

Full tables and raw JSON: `run/RESULTS.md`, `run/NEGATIVE-RESULTS.md`.

## 5. A defect this stage exposed

`--prune-experts` could not build *any* `nemotron_h_moe` artifact before this
session: the pruner's expert-tensor list knew `exp_probs_b.weight` but not the
`exp_probs_b.bias` spelling this family ships, so a pruned file kept a
128-entry selection bias beside a 120-expert router and the loader refused it.
Fixed in the same branch with a `.bias`-spelling fixture and a test
(`tests/test_prune_experts.py::test_prune_slices_exp_probs_b_bias_spelling`).
It failed closed rather than mis-routing, which is why it presented as a dead
work item instead of a wrong number.
