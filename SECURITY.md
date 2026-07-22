# Security Policy

Runner is in **public alpha** (`0.1.2-alpha`). Only the latest release is
supported; there are no backports.

## Threat model

Runner is a local inference engine for a single trusted machine. Its trust
boundary is the operating-system user account, not the network:

- **The server binds `127.0.0.1`, hardcoded.** There is no flag to bind
  another interface. Nothing outside the machine can reach it directly;
  LAN access in the Gridcore suite goes through the dashboard layer, never
  to runner itself.
- **No authentication, by design.** Any process that can open a loopback
  socket can use the API — the same set of processes that could already
  read your files or burn your GPU. Auth belongs to whatever layer you put
  in front of runner, not inside it.
- **Clients are assumed cooperative but not perfect.** The server defends
  against accidents (stalls, oversized requests, malformed JSON), not
  against a hostile local process — a hostile local process already has
  your account.

If you reverse-proxy or port-forward runner onto a network yourself, you
are outside this model: add authentication and TLS in the proxy, and
understand that runner has never claimed to be safe against hostile
network input.

## The HTTP surface

The built-in server is a deliberate subset of HTTP/1.1, kept in one source
module (`src/server.c`) so its behavior can be audited as a unit. It is not a
general-purpose web server and does not try to be:

| Property | Behavior |
|---|---|
| Body framing | `Content-Length` only — no chunked transfer-encoding |
| Connection lifecycle | `Connection: close` on every response, no keep-alive |
| Request header | capped at 16 KB |
| Request body | capped at 32 MB |
| Request read deadline | header + body must arrive within 10 s, else `408` and the inference slot is released |
| TLS | none — loopback traffic only |

API and schema features outside the documented subset fail closed (`400`).
The current HTTP framing limitations are documented below; the loopback-only
listener is a required part of the threat model, not a substitute for a hardened
internet-facing parser.

## Untrusted input

The one input runner routinely takes from the internet is the **model
file**. GGUF metadata and tensor layouts are bounds-checked before use
(malformed dimension and offset metadata are rejected — this is
CI-tested), but a model file is still a large binary you chose to
download: prefer sources you trust, as you would with any executable.

Request bodies are parsed by runner's own strict JSON parser (no
dependencies, rejects malformed input outright), and JSON Schemas are
validated at compile time before they ever drive sampling.

## Hardening in place

- Zero third-party dependencies (libc/pthreads only) — no transitive CVE
  surface, and the release binary is fully static.
- All request reads are bounded in size and time; buffers are fixed-size
  and NUL-terminated before parsing.
- CI runs the engine under AddressSanitizer/UBSan on every push, plus
  smoke tests for the failure paths above (stalled clients, malformed
  models, oversized parameters).

## Reporting a vulnerability

Found something? Please use
[GitHub private vulnerability reporting](../../security/advisories/new)
for anything sensitive, or an [ordinary issue](../../issues) for
hardening suggestions that don't need coordinated disclosure. Alpha means
reports get read fast — include `runner --version` output and, if you
can, a reproducer.
