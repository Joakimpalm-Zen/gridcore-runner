# Cert-matrix detailed report — GPT-OSS x Gemma 4 derivative ecosystems

Per-artifact gate evidence. Status table (quick view): `docs/cert-matrix-status.md`.

## Environment

| item | value |
|---|---|
| repo | fresh clone, branch `cert-matrix`, into `~/workspace/Gridcore/cert-matrix/` |
| runner version | `runner 0.1.8-alpha` |
| build | `make runner CC=x86_64-conda-linux-gnu-gcc -j`, conda env `ccbuild` |
| llama.cpp reference | `b10280 (61881b1f7)`, prebuilt, `~/workspace/Gridcore/lcpp-bin/llama-b10280/` (unless a section notes a split reference) |
| box | 128 cores, big NVIDIA GPU (MIG slice), CPU gates use `--gpu off` explicitly |

## Gate battery (recap, see goal doc for full text)

1. Identity (sha256 + metadata dump)
2. Admission (load or clean refusal)
3. Tokenizer differential vs HF reference
4. Reference gate (KLD for gpt-oss/gemma4-moe; token identity for gemma4 dense/E-series)
5. cpu_cuda byte-identity
6. Chat smoke (`--serve`, OpenAI completion)
7. Perf row (`--bench-json`, GPU and CPU)

---

<!-- Per-artifact sections appended below, one per roster item, in order. -->
