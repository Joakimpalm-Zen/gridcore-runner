# Gridcore Runner Python Client

The supported Python endpoint, process-launch, and startup-ownership boundary for Runner consumers.

`StartupLease` atomically arbitrates one parent-owned Runner launch. It only
tracks the owning parent and never inspects or kills an unrelated child process.
