# Model scope — geographic focus: Europe & US (standing policy, 2026-07-29)

Runner's model-support effort focuses on **European and US model families**. This is
a prioritization of where new support work goes, decided by the project owner;
it is not a removal of anything already shipped.

## The rules

1. **New architecture and family support targets EU/US-origin open-weight
   models.** A support request or roadmap item for a family outside that focus
   is declined by default, the same way Tier-B architectures already are.
2. **Nothing existing is retired.** Every certified architecture keeps its
   certification, tests, kernels and maintenance — including the Qwen line
   (`qwen2`, `qwen3`, `qwen3moe`, `qwen35`), which remains first-class for the
   models already admitted. Completing in-flight evidence for an existing arch
   (e.g. a missing tolerance-gate row) is maintenance, not new support work.
3. **Successor families outside the focus are not picked up.** New Qwen/GLM/
   Hunyuan/MiniMax/DeepSeek/Kimi generations get no new arch work. (DeepSeek/
   Kimi MLA was already declined as Tier B on engineering grounds; this policy
   is a second, independent reason.)
4. **Protocol compatibility is unaffected** — the OpenAI and Anthropic API
   surfaces are US ecosystems and stay the compatibility targets.
5. **Boundary (owner decision, 2026-07-29): wider Europe.** The European
   side of the focus includes non-EU Europe — the UK (Stability AI /
   `stablelm`) and Switzerland (Apertus) are full members of the focus, not
   just grandfathered existing work.

## Focus families

**US (open weights, runnable here):** Meta Llama · Google Gemma ·
Microsoft Phi · OpenAI gpt-oss · IBM Granite (currently refused at load with
an explicit reason — the natural first US addition) · AI2 OLMo/OLMoE ·
NVIDIA Nemotron/Minitron · xAI Grok open releases (size-impractical on a
24 GB slice; listed for completeness).

**US (API/protocol relevance only):** OpenAI GPT series, Anthropic Claude,
Google Gemini, xAI Grok (hosted) — served *through* Runner-compatible
surfaces, not run by Runner.

**EU:** Mistral AI (Mistral Small/Large open releases, Mixtral, Nemo,
Ministral, Codestral, Devstral, Magistral) · EuroLLM / OpenEuroLLM
(EuroLLM-9B is llama-shaped per its model card — likely loads today; cheapest
EU certification row, verify at admission) · TildeOpen (Latvia; ~30B, fits
the 24 GB slice) · Aleph Alpha Pharia/Luminous (Germany; custom arch, a real
porting effort) · Teuken / OpenGPT-X (Germany) · Salamandra (Spain, BSC) ·
Lucie / OpenLLM-France · Poro/Viking (Silo AI, Finland) · BLOOM (BigScience;
historical — ALiBi attention would be a new knob, low priority).

## What this changes in the current roadmap

- The arch-broadening "per-family remainder" list loses GLM 5.x, Hunyuan and
  MiniMax M2/M3 as support targets. The **generalized MoE router** item
  survives on its own merits: gpt-oss (US) and the Mixtral line (EU) both
  need it.
- **gpt-oss** ("human eye: decide whether in scope") is scope-eligible under
  this policy; the remaining decision is engineering effort, not direction.
- Candidate admission order stays evidence-first: gate/certify with the same
  program regardless of origin — this policy chooses *which* families enter
  the queue, never how strictly they are checked.

> **Update (2026-08-04):** three of the calls above have since resolved the
> way the policy pointed. gpt-oss shipped — CPU (2026-07-31), CUDA
> (2026-08-01), and a Metal smoke — and is a certified architecture in the
> README. The generalized MoE router landed (it now covers the Llama-4 /
> DeepSeek-V3 knobs). TildeOpen-30b moved from candidate to the SHA-pinned
> European roster with cpu_cuda byte-identity. The policy itself is unchanged.
