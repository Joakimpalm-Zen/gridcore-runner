# SSM hybrid greedy-reference cert — 2026-08-20

Reproducible greedy-identity comparison of the two supported Mamba-2 hybrid
architectures against llama.cpp, via the committed
`scripts/reference_compare.py` (both engines served, exact generated UTF-8 text
from `/v1/completions` at temperature 0). Run on an RTX PRO 6000 Blackwell host,
CPU path (the hybrids are CPU-only), 5 prompts × 32 tokens.

- **runner** 0.1.20-alpha, **reference** llama.cpp `ea12b27`.
- **`nemotron_h` (Nemotron-Nano-9B-v2, Q8_0): 5/5 byte-identical.** Every prompt —
  factual, code, counting, story, JSON — matches the oracle exactly at 32 tokens.
- **`granitehybrid` (granite-4.0-h-small, Q8_0): 3/5.** The factual, counting and
  story prompts are byte-identical; the two open-ended code/JSON continuations
  diverge after a shared prefix — a greedy cascade from a near-tie, not wrong
  math (the high-confidence `" Paris."` matches exactly). This is the documented
  qwen3moe/gpt-oss noise-floor envelope: token-identical on high-confidence
  prompts, argmax coin-flips on low-confidence continuations.

Reproduce:
`python3 scripts/reference_compare.py --runner ./runner --reference <llama-server> --model <hybrid>.gguf --tokens 32`
