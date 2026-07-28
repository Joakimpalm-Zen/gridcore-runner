# Metal runtime fallback ownership

Metal owns the KV cache buffers on Apple Silicon: `model_t.kcache` and
`model_t.vcache` point at `MTLBuffer.contents`, not malloc memory. Runtime
fallback therefore has two separate states:

- `model_t.gpu` is the active backend used for future inference.
- `model_t.gpu_owner` is the backend resource owner that must stay reachable
  until destruction.

On a Metal runtime failure, `gpu_disable()` clears `gpu` and leaves
`gpu_owner`, `kcache` and `vcache` alive so the CPU path can continue over the
same unified-memory KV rows. `model_free()` later calls `gpu_free()`, which
releases every Objective-C object once, detaches the borrowed KV pointers, and
prevents ordinary `free()` from seeing `MTLBuffer.contents`.

CI coverage:

- `make test-metal-fallback` runs on macOS CI.
- `tests/test_metal_ownership.m` exercises repeated disable/free state changes
  without requiring a Metal device.
- When `runner --caps` reports a Metal device, the same target runs a real tiny
  model with `RUNNER_METAL_INJECT_FAILURE=once` plus macOS malloc checking
  (`MallocScribble=1`, `MallocGuardEdges=1`) and compares fallback output to
  `--gpu off`.

Remaining hardware-only validation: induce a real
`MTLCommandBufferStatusError` on Apple Silicon and run the same fallback and
destruction path under AddressSanitizer or Guard Malloc. GitHub-hosted macOS CI
does not provide a reliable way to force that command-buffer failure, so the
checked-in deterministic hook covers the ownership state machine and fallback
path without pretending to be a hardware fault.
