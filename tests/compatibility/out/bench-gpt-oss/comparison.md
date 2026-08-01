# Runner vs llama.cpp comparison

## Provenance

- Schema: `gridcore.runner.llamacpp-comparison.v1`
- Generated UTC: `2026-08-01T20:50:06Z`
- Status: `complete`
- Model path: `C:\ProjectGrid\models\gpt-oss-20b-MXFP4.gguf`
- Model SHA256: `27cd6c432c7672cb812a92f611cf3ba7bbc35928262bb1e1253ff4ee6ae35901`
- Model bytes: `12109566624`
- Runner: `runner 0.1.4-alpha`
- Runner commit: `e1a96765dc07631db3c12422494bc4ca180f1a74`
- llama.cpp: `version: 10076 (305ba519a)`
- llama.cpp commit: `305ba519a`

## Settings

- Context: `2048`
- Maximum generated tokens: `32`
- Quantization: `MXFP4`
- Temperature: `0`
- Top-p: `1`
- Sampling: `greedy`
- Prompt: `The history of computing hardware spans several distinct eras. Early mechanical calculators gave way to electromechanical relays, then to vacuum tubes, then to discrete transistors, and finally to integrated circuits. Each transition reduced size and power while increasing reliability and speed. The stored-program architecture, in which instructions and data share one memory, remains the organizing idea behind nearly every general-purpose machine built since. Memory hierarchies developed because fast storage is expensive and dense storage is slow, so designers interpose caches between processors and main memory. The history of computing hardware spans several distinct eras. Early mechanical calculators gave way to electromechanical relays, then to vacuum tubes, then to discrete transistors, and finally to integrated circuits. Each transition reduced size and power while increasing reliability and speed. The stored-program architecture, in which instructions and data share one memory, remains the organizing idea behind nearly every general-purpose machine built since. Memory hierarchies developed because fast storage is expensive and dense storage is slow, so designers interpose caches between processors and main memory. The history of computing hardware spans several distinct eras. Early mechanical calculators gave way to electromechanical relays, then to vacuum tubes, then to discrete transistors, and finally to integrated circuits. Each transition reduced size and power while increasing reliability and speed. The stored-program architecture, in which instructions and data share one memory, remains the organizing idea behind nearly every general-purpose machine built since. Memory hierarchies developed because fast storage is expensive and dense storage is slow, so designers interpose caches between processors and main memory. The history of computing hardware spans several distinct eras. Early mechanical calculators gave way to electromechanical relays, then to vacuum tubes, then to discrete transistors, and finally to integrated circuits. Each transition reduced size and power while increasing reliability and speed. The stored-program architecture, in which instructions and data share one memory, remains the organizing idea behind nearly every general-purpose machine built since. Memory hierarchies developed because fast storage is expensive and dense storage is slow, so designers interpose caches between processors and main memory. The history of computing hardware spans several distinct eras. Early mechanical calculators gave way to electromechanical relays, then to vacuum tubes, then to discrete transistors, and finally to integrated circuits. Each transition reduced size and power while increasing reliability and speed. The stored-program architecture, in which instructions and data share one memory, remains the organizing idea behind nearly every general-purpose machine built since. Memory hierarchies developed because fast storage is expensive and dense storage is slow, so designers interpose caches between processors and main memory. The history of computing hardware spans several distinct eras. Early mechanical calculators gave way to electromechanical relays, then to vacuum tubes, then to discrete transistors, and finally to integrated circuits. Each transition reduced size and power while increasing reliability and speed. The stored-program architecture, in which instructions and data share one memory, remains the organizing idea behind nearly every general-purpose machine built since. Memory hierarchies developed because fast storage is expensive and dense storage is slow, so designers interpose caches between processors and main memory.`

The TTFT request is a separate warmed streaming request after model load. The auxiliary top-k comparison sends the same chat payload to both runtimes; it is distinct from the raw-completion throughput request.

## Hardware and driver

```json
{
  "machine": "AMD64",
  "nvidia_smi": [
    {
      "driver_version": "596.36",
      "memory_free_mib": 7588,
      "memory_total_mib": 8192,
      "memory_used_mib": 431,
      "name": "NVIDIA GeForce RTX 3070"
    }
  ],
  "processor": "Intel64 Family 6 Model 158 Stepping 9, GenuineIntel",
  "python": "3.12.10",
  "system": "Windows-11-10.0.26100-SP0"
}
```

## Commands

Runner: `'C:\ProjectGrid\gridcore-runner\runner.exe' -m 'C:\ProjectGrid\models\gpt-oss-20b-MXFP4.gguf' --serve --port 55488 -c 2048 --gpu auto -n 32`

llama.cpp: `'C:\ProjectGrid\tools\llama-b10076\llama-server.exe' -m 'C:\ProjectGrid\models\gpt-oss-20b-MXFP4.gguf' --host 127.0.0.1 --port 55489 -c 2048 -ngl -1`

## Results

Derived columns are measured identically for every engine from the streaming response (see `derived_metrics`); self-reported columns come from each engine's own `timings` block, which not every engine emits and which engines do not all define the same way.

| Runtime | Derived prefill tok/s | Derived decode tok/s | Self-reported prompt tok/s | Self-reported decode tok/s | TTFT s | Tokens | Wall s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| runner | 4.18 | 0.37 | 4.37 | 0.38 | 153.5850 | 32 | 87.22 |
| llamacpp | 354.35 | 26.48 | 31.80 | 31.05 | 1.7779 | 32 | 1.09 |

## VRAM

Load deltas are `nvidia-smi` used-memory changes from immediately before process start to server readiness; they are not peak VRAM.

```json
{
  "llamacpp_after_start": [
    {
      "driver_version": "596.36",
      "memory_free_mib": 1459,
      "memory_total_mib": 8192,
      "memory_used_mib": 6560,
      "name": "NVIDIA GeForce RTX 3070"
    }
  ],
  "llamacpp_load_delta_mib": [
    {
      "device": 0,
      "name": "NVIDIA GeForce RTX 3070",
      "used_delta_mib": 6129
    }
  ],
  "runner_after_start": [
    {
      "driver_version": "596.36",
      "memory_free_mib": 553,
      "memory_total_mib": 8192,
      "memory_used_mib": 7466,
      "name": "NVIDIA GeForce RTX 3070"
    }
  ],
  "runner_load_delta_mib": [
    {
      "device": 0,
      "name": "NVIDIA GeForce RTX 3070",
      "used_delta_mib": 7035
    }
  ]
}
```

## Correctness comparison

Text comparison: `pass`

Top-logprob comparison: `captured`
Maximum absolute common-token logprob delta: `1.42356465863`
Correctness gate: `pass`

## Generated output

Runner:

```text
 ..."

The user says: "I want you to act as a professional copywriter. I want you to write a short paragraph about the history of computing hardware
```

llama.cpp:

```text
 ..."

The user says: "I want you to act as a professional copywriter. I want you to write a short paragraph about the history of computing hardware
```

## Raw artifacts

The complete buffered responses, benchmark JSON, top-k values, exact requests, and VRAM snapshots are in `comparison.json`. Server output is in `runner.log` and `llamacpp.log` for real runs.

Real Qwen3/MoE GPU results are pending unless this report status is `complete`.
