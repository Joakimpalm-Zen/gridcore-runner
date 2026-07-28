# Runner vs llama.cpp comparison

## Provenance

- Schema: `gridcore.runner.llamacpp-comparison.v1`
- Generated UTC: `2026-07-28T09:54:30Z`
- Status: `complete`
- Model path: `/home/lab/workspace/.Trash-1000/files/Gridcore/gridcore-runner/models/Qwen3-30B-A3B-Q4_K_M.gguf`
- Model SHA256: `0d003f6662faee786ed5da3e31b29c978de5ae5d275c8794c606a7f3c01aa8f5`
- Model bytes: `18556685824`
- Runner: `runner 0.1.3-alpha`
- Runner commit: `e9593131bf5b96df6abfde3c58d484ed40e6e8d5`
- llama.cpp: `version: 1 (305ba51)`
- llama.cpp commit: `305ba519ab61cdff8044922cba2347826a04453f`

## Settings

- Context: `4096`
- Maximum generated tokens: `128`
- Quantization: `Q4_K_M`
- Temperature: `0`
- Top-p: `1`
- Sampling: `greedy`
- Prompt: `The capital of France is`

The TTFT request is a separate warmed streaming request after model load. The auxiliary top-k comparison sends the same chat payload to both runtimes; it is distinct from the raw-completion throughput request.

## Hardware and driver

```json
{
  "machine": "x86_64",
  "nvidia_smi": [
    {
      "driver_version": "610.43.02",
      "memory_free_mib": null,
      "memory_total_mib": null,
      "memory_used_mib": null,
      "name": "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition"
    }
  ],
  "processor": "x86_64",
  "python": "3.14.6",
  "system": "Linux-6.12.0-211.16.1.el10_2.x86_64-x86_64-with-glibc2.39"
}
```

## Commands

Runner: `/home/lab/workspace/Gridcore/gridcore-runner/runner -m /home/lab/workspace/.Trash-1000/files/Gridcore/gridcore-runner/models/Qwen3-30B-A3B-Q4_K_M.gguf --serve --port 37357 -c 4096 --gpu off -n 128`

llama.cpp: `/home/lab/workspace/Gridcore/llama.cpp-b10076/build-cpu/bin/llama-server -m /home/lab/workspace/.Trash-1000/files/Gridcore/gridcore-runner/models/Qwen3-30B-A3B-Q4_K_M.gguf --host 127.0.0.1 --port 45771 -c 4096 -ngl 0`

## Results

| Runtime | Prompt tok/s | Decode tok/s | TTFT s | Generated tokens | Wall s |
| --- | ---: | ---: | ---: | ---: | ---: |
| runner | 5.992 | 4.053 | 0.23925498401513323 | 128 | 32.4121112349967 |
| llamacpp | 90.53706587476913 | 42.11800939758085 | 0.026342616009060293 | 128 | 3.1094883629994 |

## VRAM

Load deltas are `nvidia-smi` used-memory changes from immediately before process start to server readiness; they are not peak VRAM.

```json
{
  "llamacpp_after_start": [
    {
      "driver_version": "610.43.02",
      "memory_free_mib": null,
      "memory_total_mib": null,
      "memory_used_mib": null,
      "name": "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition"
    }
  ],
  "llamacpp_load_delta_mib": null,
  "runner_after_start": [
    {
      "driver_version": "610.43.02",
      "memory_free_mib": null,
      "memory_total_mib": null,
      "memory_used_mib": null,
      "name": "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition"
    }
  ],
  "runner_load_delta_mib": null
}
```

## Correctness comparison

Text comparison: `fail`

Top-logprob comparison: `captured`
Maximum absolute common-token logprob delta: `4.522365358643`
Correctness gate: `pass`

## Generated output

Runner:

```text
 Paris. The capital of Italy is Rome. The capital of Spain is Madrid. The capital of Germany is Berlin. The capital of the United Kingdom is London. The capital of the United States is Washington, D.C. The capital of Brazil is Brasília. The capital of Canada is Ottawa. The capital of Australia is Canberra. The capital of Japan is Tokyo. The capital of China is Beijing. The capital of India is New Delhi. The capital of Russia is Moscow. The capital of Mexico is Mexico City. The capital of Argentina is Buenos Aires. The capital of Egypt is Cairo. The capital of South Africa is Pretoria. The
```

llama.cpp:

```text
 Paris. The capital of Italy is Rome. The capital of Spain is Madrid. The capital of Germany is Berlin. The capital of the United Kingdom is London. The capital of the United States is Washington, D.C. The capital of Brazil is Brasília. The capital of Japan is Tokyo. The capital of India is New Delhi. The capital of China is Beijing. The capital of Russia is Moscow. The capital of Egypt is Cairo. The capital of South Africa is Pretoria. The capital of Australia is Canberra. The capital of Canada is Ottawa. The capital of Mexico is Mexico City. The capital of Argentina is Buenos Aires. The
```

## Raw artifacts

The complete buffered responses, benchmark JSON, top-k values, exact requests, and VRAM snapshots are in `comparison.json`. Server output is in `runner.log` and `llamacpp.log` for real runs.

Real Qwen3/MoE GPU results are pending unless this report status is `complete`.
