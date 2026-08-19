# Runner vs llama.cpp vs Ollama vs vLLM — SmolLM2-1.7B-Instruct, v3 matrix, 2026-08-19

A version refresh of the competitor comparison. The published competitor rows
(`2026-08-03-smollm2-1.7b-v2` and earlier) recorded llama.cpp `b10076`, Ollama
`0.32.1`, and vLLM `0.26.0`, all behind upstream, and the weekly
`competitor-freshness` workflow opened an issue. This set re-runs the matrix on
identical hardware with each runtime at its **current upstream release**, so the
freshness ledger reads current again and every number comes from a real run.

- **Box:** the Blackwell box — NVIDIA RTX PRO 6000 Blackwell workstation GPU
  (vLLM served on a 24 GB MIG slice of it). Runner and llama.cpp ran on the
  128-core CPU (see the throughput caveat below).
- **Matrix:** v3, 120 cases, eight families, `temperature 0`, deterministic
  (`scripts/agent-torture.py --cases 120`). Not case-for-case comparable with
  the v2 (105-case) rows — v3 adds the `tool_stream_normalization` family.
- **Weights:** Runner, llama.cpp, and Ollama all ran the **same**
  `SmolLM2-1.7B-Instruct-Q4_K_M.gguf` (identical bytes). Ollama imported that
  exact GGUF via `ollama create -f` (autodetected `chatml` template). vLLM
  cannot load a GGUF here, so it ran the upstream `HuggingFaceTB/SmolLM2-1.7B-Instruct`
  **bf16** HF weights — a precision difference, the same one the 2026-08-03 row
  documented, and the reason the throughput row below is not a fair fight.
- Runtimes ran **one at a time** on a loopback port (they contend for the GPU
  and the port otherwise).

| runtime | version | passed | failed | failure category |
|---|---|---|---|---|
| Runner | runner 0.1.19-alpha (CPU) | **120/120** | 0 | — |
| vLLM | 0.27.1, `--tool-call-parser hermes` (GPU) | 80/120 | 40 | protocol |
| llama.cpp | b10488 (CPU) | 30/120 | 90 | protocol |
| Ollama | 0.32.14 (imported GGUF, `chatml`) | 28/120 | 92 | protocol |

## Per-family breakdown

| family | Runner | vLLM | llama.cpp | Ollama |
|---|---|---|---|---|
| `nested_arguments` | 15/15 | 12/15 | 0/15 | 0/15 |
| `tool_selection` | 15/15 | 15/15 | 0/15 | 0/15 |
| `forced_truncation` | 15/15 | 0/15 | 0/15 | 0/15 |
| `stream_normalization` | 15/15 | 15/15 | 15/15 | 15/15 |
| `tool_stream_normalization` | 15/15 | 0/15 | 0/15 | 0/15 |
| `large_enum_selection` | 15/15 | 8/15 | 0/15 | 0/15 |
| `reasoning_then_tool` | 15/15 | 15/15 | 0/15 | 0/15 |
| `structured_final` | 15/15 | 15/15 | 15/15 | 13/15 |

Every failure above is `protocol` — the turn came back with no parseable tool
call (`expected exactly one tool call — got None`), auditable line-by-line in
each `raw.jsonl`.

## Reading the split

- **llama.cpp and Ollama pass exactly the two families that never send
  `tools`** — `stream_normalization` (plain SSE) and `structured_final`
  (`response_format` guided decoding) — and fail every family that requires a
  tool call. This is the same mechanism finding as the 2026-07-22 SmolLM2 and
  2026-07-21 Llama-3.2-3B rows: on a model this small, the runtime's chat-template
  path (llama.cpp's `--jinja`, Ollama's `chatml`) emits no parseable
  `<tool_call>` at all, so the tool-call families come back empty. It is not the
  model reasoning worse under one runtime — Runner runs the identical GGUF and
  lands all 120 — it is that grammar-constrained sampling is template-independent
  and the template path is not. Ollama's two `structured_final` misses are the
  only tool-independent failures on either runtime.
- **vLLM's buffered tool path matches its 2026-08-03 v2 profile**:
  `forced_truncation` 0/15 (a `max_tokens` of 1–8 does not reach the end of a
  hermes `<tool_call>` wrapper), `large_enum_selection` 8/15, `nested_arguments`
  12/15; `tool_selection`, `reasoning_then_tool`, and `structured_final` all
  pass. Its **new** weak spot on v3 is `tool_stream_normalization` (0/15): the
  streamed hermes tool-call path does not produce a transport-invariant, verdict-
  passing stream here. The 40 = the 25 it lost on v2 (forced_truncation 15 +
  large_enum 7 + nested 3) plus the 15 new streamed-tool cases.

## Throughput (not a fair fight — recorded for completeness)

`valid_structured_tasks_per_second`, as the harness reports it:

| | Runner (CPU) | vLLM (GPU) | llama.cpp (CPU) | Ollama (GPU/CPU) |
|---|---|---|---|---|
| valid structured tasks/s | 1.039 | 1.702 | 0.607 | 0.743 |
| host peak RSS | 1.61 GB | — | — | — |

Read with the caveats the 2026-08-03 row spelled out: `agent-torture.py` forces
`--gpu off` for the Runner it spawns, so Runner is CPU-only here against a GPU
vLLM, on Q4_K_M vs bf16 — this is not a throughput claim. A foreign process's
RSS is not read, so only Runner reports one. llama.cpp used the official
`b10488` `ubuntu-x64` release binary (upstream publishes no Linux CUDA asset for
this tag), so it too ran CPU.

## Reproducing

```sh
# Runner — spawned by the harness (CPU, --gpu off), 0.1.19-alpha built on the box
python3 scripts/agent-torture.py \
  --model models/SmolLM2-1.7B-Instruct-Q4_K_M/SmolLM2-1.7B-Instruct-Q4_K_M.gguf \
  --cases 120 --out OUT/runner

# llama.cpp b10488 — official ubuntu-x64 release binary
llama-server -m SmolLM2-1.7B-Instruct-Q4_K_M.gguf --host 127.0.0.1 --port 8080 --jinja -c 4096
python3 scripts/agent-torture.py --endpoint 127.0.0.1:8080 --runtime llama.cpp \
  --runtime-version b10488 --model-name SmolLM2-1.7B-Instruct-Q4_K_M --cases 120 --out OUT/llamacpp

# Ollama 0.32.14 — same GGUF imported so the weights are identical
printf 'FROM %s\n' SmolLM2-1.7B-Instruct-Q4_K_M.gguf > Modelfile
ollama create smollm2-1.7b -f Modelfile          # OLLAMA_HOST=127.0.0.1:11435
python3 scripts/agent-torture.py --endpoint 127.0.0.1:11435 --runtime ollama \
  --runtime-version 0.32.14 --model-name smollm2-1.7b --cases 120 --out OUT/ollama

# vLLM 0.27.1 — the tool-call parser is mandatory (see the 2026-08-03 correction)
VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=TRITON_ATTN \
vllm serve HuggingFaceTB/SmolLM2-1.7B-Instruct --port 8091 --enforce-eager \
  --max-model-len 4096 --gpu-memory-utilization 0.55 \
  --enable-auto-tool-choice --tool-call-parser hermes
python3 scripts/agent-torture.py --endpoint 127.0.0.1:8091 --runtime vllm \
  --runtime-version "0.27.1 (--tool-call-parser hermes)" \
  --model-name HuggingFaceTB/SmolLM2-1.7B-Instruct --cases 120 --out OUT/vllm
```

After committing this directory, `python3 scripts/competitor-freshness.py`
reports all three competitors current (llama.cpp b10488, Ollama 0.32.14, vLLM
0.27.1), because it keeps the newest published version per runtime.
