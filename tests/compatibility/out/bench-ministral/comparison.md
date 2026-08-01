# Runner vs llama.cpp comparison

## Provenance

- Schema: `gridcore.runner.llamacpp-comparison.v1`
- Generated UTC: `2026-08-01T20:35:56Z`
- Status: `complete`
- Model path: `C:\ProjectGrid\models\Ministral-8B-Instruct-2410-Q5_K_M.gguf`
- Model SHA256: `190766d270e0e13a8ea2d1ff5bc2faae8cf5897736881bd1fd1698651cc82c8b`
- Model bytes: `5724146496`
- Runner: `runner 0.1.4-alpha`
- Runner commit: `e1a96765dc07631db3c12422494bc4ca180f1a74`
- llama.cpp: `version: 10076 (305ba519a)`
- llama.cpp commit: `305ba519a`

## Settings

- Context: `2048`
- Maximum generated tokens: `64`
- Quantization: `Q5_K_M`
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

Runner: `'C:\ProjectGrid\gridcore-runner\runner.exe' -m 'C:\ProjectGrid\models\Ministral-8B-Instruct-2410-Q5_K_M.gguf' --serve --port 55455 -c 2048 --gpu auto -n 64`

llama.cpp: `'C:\ProjectGrid\tools\llama-b10076\llama-server.exe' -m 'C:\ProjectGrid\models\Ministral-8B-Instruct-2410-Q5_K_M.gguf' --host 127.0.0.1 --port 55456 -c 2048 -ngl -1`

## Results

Derived columns are measured identically for every engine from the streaming response (see `derived_metrics`); self-reported columns come from each engine's own `timings` block, which not every engine emits and which engines do not all define the same way.

| Runtime | Derived prefill tok/s | Derived decode tok/s | Self-reported prompt tok/s | Self-reported decode tok/s | TTFT s | Tokens | Wall s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| runner | 105.39 | 55.59 | 106.12 | 56.32 | 5.9301 | 64 | 1.23 |
| llamacpp | 2668.68 | 66.21 | 65.44 | 63.62 | 0.2342 | 64 | 1.05 |

## VRAM

Load deltas are `nvidia-smi` used-memory changes from immediately before process start to server readiness; they are not peak VRAM.

```json
{
  "llamacpp_after_start": [
    {
      "driver_version": "596.36",
      "memory_free_mib": 1937,
      "memory_total_mib": 8192,
      "memory_used_mib": 6082,
      "name": "NVIDIA GeForce RTX 3070"
    }
  ],
  "llamacpp_load_delta_mib": [
    {
      "device": 0,
      "name": "NVIDIA GeForce RTX 3070",
      "used_delta_mib": 5651
    }
  ],
  "runner_after_start": [
    {
      "driver_version": "596.36",
      "memory_free_mib": 1687,
      "memory_total_mib": 8192,
      "memory_used_mib": 6332,
      "name": "NVIDIA GeForce RTX 3070"
    }
  ],
  "runner_load_delta_mib": [
    {
      "device": 0,
      "name": "NVIDIA GeForce RTX 3070",
      "used_delta_mib": 5901
    }
  ]
}
```

## Correctness comparison

Text comparison: `pass`

Top-logprob comparison: `captured`
Maximum absolute common-token logprob delta: `0.569922317215`
Correctness gate: `pass`

## Generated output

Runner:

```text
 The history of computing hardware spans several distinct eras. Early mechanical calculators gave way to electromechanical relays, then to vacuum tubes, then to discrete transistors, and finally to integrated circuits. Each transition reduced size and power while increasing reliability and speed. The stored-program architecture, in which instructions and data share one memory
```

llama.cpp:

```text
 The history of computing hardware spans several distinct eras. Early mechanical calculators gave way to electromechanical relays, then to vacuum tubes, then to discrete transistors, and finally to integrated circuits. Each transition reduced size and power while increasing reliability and speed. The stored-program architecture, in which instructions and data share one memory
```

## Raw artifacts

The complete buffered responses, benchmark JSON, top-k values, exact requests, and VRAM snapshots are in `comparison.json`. Server output is in `runner.log` and `llamacpp.log` for real runs.

Real Qwen3/MoE GPU results are pending unless this report status is `complete`.
