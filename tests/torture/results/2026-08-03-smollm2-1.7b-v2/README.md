# vLLM vs Runner — SmolLM2-1.7B-Instruct, v2 matrix, 2026-08-03

Same model and same box as the `2026-08-02` row, on the expanded **v2** matrix
(105 cases, seven families). It carries a correction to that row.

| runtime | version | passed | failed | failure category |
|---|---|---|---|---|
| Runner | 0.1.5-alpha | **105/105** | 0 | — |
| vLLM | 0.26.0, `--tool-call-parser hermes` | 80/105 | 25 | protocol |
| vLLM | 0.26.0, as configured on 2026-08-02 | 30/105 | 75 | protocol |

## Correction to the 2026-08-02 result

**That row measured a misconfigured vLLM, and its 20/100 should not be quoted.**

vLLM refuses `tool_choice` outright unless the server was started with a tool
parser: every tool request came back

```
400 tool_choice=function "dispatch_job" requires --tool-call-parser to be set
```

The published 20/100 is exactly the 20 `stream_normalization` cases of the v1
matrix — the only family that does not send `tools`. Every one of the other 80
failed on that 400 before the model was ever asked to do anything. The third
row above reproduces the same mistake on the v2 matrix (30/105 = the 15 stream
cases plus the 15 `structured_final` cases, which also send no tools), and its
report is kept in `vllm-no-tool-parser/` so the correction is checkable rather
than asserted.

Started correctly — `--enable-auto-tool-choice --tool-call-parser hermes` —
vLLM scores **80/105**. That is the number to compare against.

The lesson is not about vLLM. A comparison harness that lets the competitor
fail at admission and records the result as a capability difference is
measuring its own setup, and the earlier row did exactly that for 80 of its
100 cases.

## Where the remaining 25 fall

| family | vLLM | note |
|---|---|---|
| `forced_truncation` | 0/15 | `max_tokens` of 1–8 with a forced call |
| `large_enum_selection` | 8/15 | the ~50-label taxonomy |
| `nested_arguments` | 12/15 | nested object arguments |
| `tool_selection` | 15/15 | — |
| `reasoning_then_tool` | 15/15 | new in v2 |
| `structured_final` | 15/15 | new in v2 |
| `stream_normalization` | 15/15 | — |

Every one of the 25 is `expected exactly one tool call — got None`: the turn
came back with no call at all, not with a malformed one. `forced_truncation`
losing all 15 is the least surprising — hermes has to emit a `<tool_call>`
wrapper before any content, and 1–8 tokens does not reach the end of a call.
Runner passes that family because its envelope is grammar-constrained from the
first token.

The two families added in v2 both pass on vLLM, and `structured_final` passes
even in the misconfigured run — its `response_format` path does not go through
the tool parser. That is worth stating plainly: **on this matrix vLLM's guided
decoding is not the weak spot; its tool-call path is.**

## Resource footprint

Not a headline number, and the two runtimes are not measured in the same units
by choice — the differences in kind are the point.

| | Runner (GPU) | vLLM (GPU) |
|---|---|---|
| weights in VRAM | 1.1 GB (Q4_K_M) | 3.19 GiB (bf16) |
| KV | 0.81 GB, sized by `-c 4096` | 9.57 GiB, **preallocated by policy** |
| activation / non-torch | 0.01 GB scratch | 0.13 + 0.11 GiB |
| host peak RSS | 1.10 GB | 3.36 GB (2.35 EngineCore + 1.01 APIServer) |

Read with three caveats:

- **The KV figures are not comparable.** Runner sizes KV from `-c`; vLLM
  preallocates a pool from `--gpu-memory-utilization` (0.55 here) regardless of
  need. vLLM's own log offers to fit into 9.43 GiB or expand to 19.84 GiB for
  the same workload. A bigger number here is a policy, not a cost.
- **Different precisions.** Q4_K_M against bf16, which is most of the weight
  difference and the reason the throughput rows below are not a fair fight
  either.
- **vLLM is multi-process.** Measuring only the process named `vllm` gives
  1.01 GB and understates it by more than half; the sum above includes the
  `VLLM::EngineCore` worker that actually holds the model.

Throughput is reported by the harness as valid-structured-tasks/second: Runner
0.595 (CPU), vLLM 1.93 (CUDA). Runner's matrix run is CPU-only because
`agent-torture.py` forces `--gpu off` for the server it spawns, so this row
compares a CPU runtime against a GPU one and is not a throughput claim.

## Reproducing

```sh
# runner
python3 scripts/agent-torture.py --model models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf --out OUT

# vLLM — the flags matter, see the correction above
CC=x86_64-conda-linux-gnu-gcc CPATH=$CONDA/envs/py312hdr/include/python3.12 \
VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=TORCH_SDPA \
vllm serve HuggingFaceTB/SmolLM2-1.7B-Instruct --port 8091 --enforce-eager \
  --max-model-len 4096 --gpu-memory-utilization 0.55 \
  --enable-auto-tool-choice --tool-call-parser hermes

python3 scripts/agent-torture.py --endpoint 127.0.0.1:8091 --runtime vllm \
  --model-name HuggingFaceTB/SmolLM2-1.7B-Instruct --out OUT
```

One more startup obstacle beyond the three the 2026-08-02 README lists: triton
JIT-compiles a `cuda_utils.c` against the venv's Python and needs `Python.h`,
which is not installed for the system 3.12. Worked around with a conda
`python=3.12` environment on `CPATH` purely for its headers.
