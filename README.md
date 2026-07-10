# runner

A tiny llama.cpp-style local LLM inference engine, written from scratch in plain C
(~4,000 lines, no dependencies beyond libc/pthreads). It loads standard **GGUF**
model files — the same files llama.cpp and Ollama use — and runs them on the CPU,
with a particular focus on squeezing **large contexts out of small models**.

```
./runner -m models/SmolLM2-135M-Instruct-Q8_0.gguf -i          # interactive chat
./runner -m llama3.2:1b -p "Hello"                             # straight from your Ollama store
./runner -m tinyllama.gguf -f big-document.txt -c 8192 -n 200  # 4x the training context
./runner -m model.gguf --serve --parallel 2                    # OpenAI-compatible API server
./runner -m model.gguf -p "..." --json                         # guaranteed-valid JSON output
```

## Build

```
make          # produces ./runner
make debug    # ASan/UBSan build for development
```

Requires only a C11 compiler (tested with Apple clang on arm64 macOS).

## Get a model

Three ways:

1. **Use your Ollama models directly.** Pass an Ollama model name (`-m llama3.2:1b`,
   `-m qwen2.5:0.5b`) and runner resolves it through the local Ollama store
   (`~/.ollama/models`, or `$OLLAMA_MODELS`) to the underlying GGUF blob. No copy,
   no conversion. (Cloud-only tags have no local weights and can't be run.)
2. **`./download-model.sh`** fetches a small test model.
3. **Any GGUF from Hugging Face**, e.g.:
   ```
   curl -L -O "https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q8_0.gguf"
   ```

Note: safetensors checkpoints must be converted to GGUF first (Ollama and
llama.cpp both convert on import) — runner runs the converted GGUF.

## Large contexts on small models

This is runner's specialty. Three pieces work together:

- **Automatic context extension.** Ask for more context than the model was
  trained on (`-c 8192` on a 2k model) and runner applies YaRN rope scaling
  automatically — no flags needed. Verified: TinyLlama (trained at 2,048)
  retrieves a fact from the start of a 4,285-token prompt at 4x extension.
  Models with rope-scaling metadata (linear/YaRN) or llama-3.x frequency
  factors (`rope_freqs.weight`) get their native scaling applied; manual
  control via `--rope-scale` and `--rope-base`.
- **fp16 KV cache** — half the memory per context token, so twice the context
  fits. A 1.1B model at 32k context needs ~740 MB of cache; the verbose flag
  (`-v`) prints the exact number before committing.
- **Batched prompt processing.** Long prompts are evaluated in batches
  (default 64 tokens): each weight row is dequantized once and reused across
  the whole batch, and logits are skipped for all but the last token.
  ~8x faster prompt ingestion than token-at-a-time (TinyLlama Q4_K_M:
  5 → 40 tok/s; Qwen2.5-0.5B: ~97 tok/s), with live progress on stderr.

## HTTP server (OpenAI-compatible)

```
./runner -m model.gguf --serve --port 8080 --parallel 2
```

Endpoints: `POST /v1/chat/completions`, `POST /v1/completions`,
`GET /v1/models`, `GET /health`. Works with any OpenAI client:

```python
import openai
client = openai.OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="none")
r = client.chat.completions.create(
    model="runner",
    messages=[{"role": "user", "content": "Hello!"}],
    response_format={"type": "json_object"},   # optional: forced-valid JSON
    stream=True,                                # optional: SSE streaming
)
```

`--parallel N` creates N independent inference slots (each with its own KV
cache and thread pool, splitting `-t` threads between them); requests are
served concurrently, and model weights are shared between slots through the
mmap page cache, so memory grows only by KV cache per slot. The server binds
to 127.0.0.1 only. Streaming clients that disconnect stop generation
immediately. Multiple *processes* also share weights the same way — running
several `runner` instances against one GGUF costs the file size once.

## Structured output (JSON)

`--json` on the CLI, or `"response_format": {"type": "json_object"}` over the
API, constrains sampling with an incremental JSON validator: every candidate
token is checked against the grammar state and rejected unless the output
remains a valid prefix of a single JSON object. If the token budget runs out
mid-object, runner closes strings and containers minimally so the result
still parses (and reports `finish_reason: "length"` so you know it was cut).
The syntax is guaranteed; key names and semantics are still up to the model,
so keep prompts explicit about the schema you want.

## Usage

```
runner -m model [options]

  -m PATH|NAME   GGUF file, or an Ollama model name (e.g. llama3.2:1b)
  -p TEXT        prompt (one-shot completion; \n etc. are unescaped)
  -f FILE        read prompt from file (appended after -p text)
  -i             interactive chat mode
  --serve        HTTP server mode (OpenAI-compatible API)
  --port N       server port (default 8080)
  --parallel N   parallel inference slots in server mode (default 1)
  --json         constrain output to a single valid JSON object
  -n N           max new tokens (default 256, -1 = until EOS)
  -c N           context length (default: min(model max, 4096));
                 beyond the training context, YaRN is applied automatically
  -b N           prompt batch size (default 64)
  -t N           threads (default: min(8, cpus))
  -s N           RNG seed
  --temp F       temperature (default 0.8, 0 = greedy)
  --top-k N      top-k sampling (default 40)
  --top-p F      nucleus sampling (default 0.95)
  --repeat-penalty F   (default 1.1)
  --rope-scale F force linear rope position scaling
  --rope-base F  override rope frequency base
  --system TEXT  system prompt for chat mode
  --chat-template chatml|llama2|llama3|zephyr|raw   (default: auto-detect)
  --no-bos       don't prepend BOS
  --ignore-eos   keep generating past end-of-text tokens
  -v             print model hyperparameters and memory use
```

Chat mode keeps the KV cache across turns (no re-processing of history) and
auto-detects the chat template (ChatML, Llama-2/3, Zephyr) from the model's
metadata and vocabulary.

## What's implemented

| Area | Support |
|---|---|
| File format | GGUF v2/v3, memory-mapped (weights are never copied); Ollama store resolution |
| Architectures | `llama` (Llama 2/3, Mistral, TinyLlama, SmolLM2, …), `qwen2` (QKV biases) |
| Tensor types | F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL, IQ4_XS — every quant Ollama's library commonly serves |
| Long context | fp16 KV cache, batched prompt eval, YaRN / linear / llama-3 freq-factor rope scaling with auto-extension |
| Tokenizers | SentencePiece (llama) with byte fallback; byte-level BPE (gpt2) with merges, special-token parsing |
| Transformer | RMSNorm, RoPE (adjacent-pair and NeoX), grouped-query attention, SwiGLU, tied embeddings |
| Sampling | temperature, top-k, top-p, repeat penalty, greedy; JSON-constrained decoding |
| Server | OpenAI-compatible HTTP API, SSE streaming, N parallel slots |
| Threading | persistent pthread pool; matmul rows and attention heads run in parallel |

Verified end-to-end with: SmolLM2-135M (Q8_0, Q4_K_M, Q3_K_M/IQ4_NL),
TinyLlama-1.1B (Q4_K_M, Q2_K), Qwen2.5-0.5B-Instruct (Q4_K_M), including a
needle-retrieval test at 2x and 4x training context and a 3,600-token
needle test on Qwen2.5.

Not implemented (by design, to stay small): GPU offload, MoE and hybrid-SSM
architectures (Mamba/Jamba/Qwen3.5-style), IQ2/IQ3 codebook quants, full GBNF
grammar sampling (JSON mode only), TLS/auth on the server (bind it behind a
reverse proxy if you need those).

## How it works

```
src/runner.h     shared types
src/gguf.c       GGUF parser — mmaps the file, reads metadata KVs and tensor table
src/quants.c     dequantization + fused dot-product kernels per quant format
                 (bit-exact with ggml's block layouts), fp16 LUT, threadpool
src/tokenizer.c  SPM (score-based bigram merging) and BPE (rank-based merging,
                 GPT-2 byte↔unicode mapping), hash maps for vocab/merges
src/model.c      weight wiring by tensor name, rope scaling setup, and the
                 batched forward pass; fp16 KV cache
src/sample.c     temperature/top-k/top-p sampling with optional validity
                 constraint
src/jsonmode.c   incremental JSON-prefix validator + auto-close
src/template.c   chat template detection/rendering
src/engine.c     prompt feeding + generation loop shared by CLI and server
src/json.c       minimal JSON parser/escaper for the HTTP API
src/server.c     HTTP server, OpenAI-compatible routes, parallel slots
src/main.c       CLI and Ollama store resolution
```

Weights stay quantized in the mmap'd file; matmuls dequantize on the fly, so
memory use is roughly file size + KV cache + a few MB of activations.
