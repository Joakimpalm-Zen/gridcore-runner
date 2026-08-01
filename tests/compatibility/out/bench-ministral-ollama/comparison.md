# Third-party runtime throughput

Measured with `compare_llamacpp.py --endpoints-only`: each daemon is measured while it is the ONLY engine holding the GPU, because VRAM is exclusive and a co-resident measurement would report a spilled engine's host speed as if it were its device speed. No correctness gate is emitted here — that gate is defined against the pinned llama.cpp reference build, which this mode does not run.

- Generated UTC: `2026-08-01T20:32:59Z`
- Model path: `C:\ProjectGrid\models\Ministral-8B-Instruct-2410-Q5_K_M.gguf`
- Model SHA256: `190766d270e0e13a8ea2d1ff5bc2faae8cf5897736881bd1fd1698651cc82c8b`
- Context: `2048`, max tokens: `64`, sampling: `greedy`
- Prompt: `The history of computing hardware spans several distinct eras. Early mechanical calculators gave way to electromechanical relays, then to vacuum tubes, then to discrete transistors, and finally to integrated circuits. Each transition reduced size and power while increasing reliability and speed. The stored-program architecture, in which instructions and data share one memory, remains the organizing idea behind nearly every general-purpose machine built since. Memory hierarchies developed because fast storage is expensive and dense storage is slow, so designers interpose caches between processors and main memory. The history of computing hardware spans several distinct eras. Early mechanical calculators gave way to electromechanical relays, then to vacuum tubes, then to discrete transistors, and finally to integrated circuits. Each transition reduced size and power while increasing reliability and speed. The stored-program architecture, in which instructions and data share one memory, remains the organizing idea behind nearly every general-purpose machine built since. Memory hierarchies developed because fast storage is expensive and dense storage is slow, so designers interpose caches between processors and main memory. The history of computing hardware spans several distinct eras. Early mechanical calculators gave way to electromechanical relays, then to vacuum tubes, then to discrete transistors, and finally to integrated circuits. Each transition reduced size and power while increasing reliability and speed. The stored-program architecture, in which instructions and data share one memory, remains the organizing idea behind nearly every general-purpose machine built since. Memory hierarchies developed because fast storage is expensive and dense storage is slow, so designers interpose caches between processors and main memory. The history of computing hardware spans several distinct eras. Early mechanical calculators gave way to electromechanical relays, then to vacuum tubes, then to discrete transistors, and finally to integrated circuits. Each transition reduced size and power while increasing reliability and speed. The stored-program architecture, in which instructions and data share one memory, remains the organizing idea behind nearly every general-purpose machine built since. Memory hierarchies developed because fast storage is expensive and dense storage is slow, so designers interpose caches between processors and main memory. The history of computing hardware spans several distinct eras. Early mechanical calculators gave way to electromechanical relays, then to vacuum tubes, then to discrete transistors, and finally to integrated circuits. Each transition reduced size and power while increasing reliability and speed. The stored-program architecture, in which instructions and data share one memory, remains the organizing idea behind nearly every general-purpose machine built since. Memory hierarchies developed because fast storage is expensive and dense storage is slow, so designers interpose caches between processors and main memory. The history of computing hardware spans several distinct eras. Early mechanical calculators gave way to electromechanical relays, then to vacuum tubes, then to discrete transistors, and finally to integrated circuits. Each transition reduced size and power while increasing reliability and speed. The stored-program architecture, in which instructions and data share one memory, remains the organizing idea behind nearly every general-purpose machine built since. Memory hierarchies developed because fast storage is expensive and dense storage is slow, so designers interpose caches between processors and main memory.`

Derived columns are measured identically for every engine from the streaming response (see `derived_metrics`); self-reported columns are whatever the engine's own `timings` block claims, and are blank for engines that report none.

| Runtime | Model id | Derived prefill tok/s | Derived decode tok/s | Self-reported prompt tok/s | Self-reported decode tok/s | TTFT s | Tokens | Wall s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ollama | ministral-8b-q5km | 2507.68 | 63.99 |  |  | 0.2500 | 64 | 1.56 |

## Hardware and driver

```json
{
  "machine": "AMD64",
  "nvidia_smi": [
    {
      "driver_version": "596.36",
      "memory_free_mib": 1931,
      "memory_total_mib": 8192,
      "memory_used_mib": 6088,
      "name": "NVIDIA GeForce RTX 3070"
    }
  ],
  "processor": "Intel64 Family 6 Model 158 Stepping 9, GenuineIntel",
  "python": "3.12.10",
  "system": "Windows-11-10.0.26100-SP0"
}
```

