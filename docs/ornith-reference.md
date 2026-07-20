# Ornith 1.0 9B reference gate

The compatibility target is the official Q4_K_M GGUF at this immutable
revision:

| Field | Value |
|---|---|
| Repository | `deepreinforce-ai/Ornith-1.0-9B-GGUF` |
| Revision | `3296bc7a404871a72ac3f1903f561459c09b5c17` |
| File | `ornith-1.0-9b-Q4_K_M.gguf` |
| Size | `5,629,108,704` bytes |
| SHA-256 | `5720d1f671b4996481274fffe01868c3c36e87c135cc8538471cc7bd6087b106` |

Download and validate the external artifact:

```sh
mkdir -p models
curl -L --fail --continue-at - \
  -o models/ornith-1.0-9b-Q4_K_M.gguf \
  https://huggingface.co/deepreinforce-ai/Ornith-1.0-9B-GGUF/resolve/3296bc7a404871a72ac3f1903f561459c09b5c17/ornith-1.0-9b-Q4_K_M.gguf

python3 scripts/ornith-reference.py \
  --model models/ornith-1.0-9b-Q4_K_M.gguf \
  --sha256 5720d1f671b4996481274fffe01868c3c36e87c135cc8538471cc7bd6087b106
```

The gate checks the complete GGUF extent, architecture and tokenizer metadata,
all 32 blocks, and the published hybrid pattern: three linear-attention blocks
followed by one full-attention block. The official Q4_K_M contains 427 tensors:
24 linear-attention blocks and 8 full-attention blocks.

## Independent inference smoke

Build a current llama.cpp outside this repository and pass its `llama-cli`
alongside Runner. Both smokes use a short greedy CPU decode and emit a JSON
report suitable for archiving in CI:

```sh
python3 scripts/ornith-reference.py \
  --model models/ornith-1.0-9b-Q4_K_M.gguf \
  --runner ./runner \
  --reference /path/to/llama.cpp/build/bin/llama-cli \
  --report /tmp/ornith-reference.json
```

The script deliberately does not download or build llama.cpp itself. This
keeps Runner's committed test surface dependency-free and makes the independent
reference executable explicit. Model weights and generated reports remain
uncommitted.
