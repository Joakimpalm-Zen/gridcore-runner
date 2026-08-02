# vLLM vs Runner — SmolLM2-1.7B-Instruct, 2026-08-02

The first published vLLM row. Same 100-case matrix, same model, same box.

| runtime | version | passed | failed | failure category |
|---|---|---|---|---|
| Runner | 0.1.4-alpha | **100/100** | 0 | — |
| vLLM   | 0.26.0      | 20/100 | 80 | protocol |

**Read the deviations before quoting this.**

- **Not the same 12-case subset** as the earlier `2026-07-21` and `2026-07-22`
  results, which ran 12 cases. This is the full 100-case matrix, so the numbers
  are not comparable to the published `12/12 vs 5/12` and `12/12 vs 3/12` rows
  — only to each other.
- **Not the same backend.** Runner ran on CPU and vLLM on CUDA, because vLLM is
  GPU-first. The verdicts are tool-call validity and schema conformance, not
  throughput, so the backend should not change them — but it is a difference
  and it is stated rather than buried.
- vLLM served the fp16 HuggingFace checkpoint; Runner served the Q4_K_M GGUF of
  the same model. Quantisation could in principle move a verdict.

## Getting vLLM to run at all

Worth recording, because it took three attempts and the reasons are the point:

1. `Failed to find C compiler` — torch inductor and triton JIT-compile at
   startup, and this box has no system C compiler. Fixed with `CC` pointed at
   the conda toolchain.
2. `fatal error: curand.h: No such file or directory` — vLLM's flashinfer path
   JIT-compiles CUDA kernels and needs full CUDA toolkit headers, which are not
   installed here. Worked around with `VLLM_USE_FLASHINFER_SAMPLER=0` and
   `VLLM_ATTENTION_BACKEND=TORCH_SDPA`.
3. Then it starts, with `--enforce-eager`.

That dependency chain — a C compiler, a CUDA toolkit, and a ~8 GB Python
environment before the first token — is the deployment difference the README's
"one file to ship" claim is about, observed rather than asserted.

## LM Studio

Still not run. It is a GUI desktop application; its `lms` CLI requires the
installed app, and there is no headless install path for a server. The harness
supports it as a `--runtime` target for anyone running it on a workstation.
