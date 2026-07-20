# Gridcore Runner Python Client

The supported Python endpoint, process-launch, and startup-ownership boundary for Runner consumers.

`StartupLease` atomically arbitrates one parent-owned Runner launch. It only
tracks the owning parent and never inspects or kills an unrelated child process.

`ManagedRunner.start()` owns the child for the whole call. A False return means
nothing is left running: a runner that never answered before the deadline is
terminated rather than handed to a caller that has no reference to it.

`RunnerEndpoint.stream_chat()` treats a malformed `data:` frame as a protocol
error rather than skipping it, so a corrupt stream cannot be certified complete
by a later `finish_reason`; non-data SSE lines (comments, `event:`, `id:`,
`retry:`) are ignored as the spec requires. `stall_seconds` is a watchdog over
the time between stream events, and the raised `RunnerStallError` reports the
measured silence. Both errors carry the text received so far in `.partial`.
