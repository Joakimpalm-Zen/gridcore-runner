# Coding-agent compatibility evidence

The original broad sweep below was validated 2026-07-21 with Runner
0.1.1-alpha, Qwen3-4B Q4_K_M,
`--parallel 1`, a 16,384-token context unless noted, and an isolated fixture
whose sentinel was `ORANGE-7319`. Test credentials were the literal `none`.
The versions and outcomes below are observations from installed clients, not
API-shape guesses.

A targeted 0.1.2-alpha rerun on 2026-07-22 covered the consumers still
installed in the test environment: Clu passed 36 runner-client/gateway/CLI
integration tests; Claude Code 2.1.217 and OpenCode 1.18.3 completed real Read
loops; Codex CLI 0.144.6 completed its Responses/`exec_command` loop with a
non-fatal model-metadata warning. Exact results, packaging warnings and
unavailable clients are recorded in
`tests/compatibility/out/2026-07-22-v0.1.2-alpha.json`. The reproducible SDK,
gateway and framework gates are described in `docs/compatibility-program.md`.

| Client | Version | Surface | Result |
|---|---:|---|---|
| OpenCode | 1.18.4 | AI-SDK Chat Completions | Complete Plan/Read/tool-result loop |
| pi coding agent | 0.81.1 | Chat Completions | Complete Read loop |
| pi coding agent | 0.81.1 | Responses | Complete Read loop |
| pi coding agent | 0.81.1 | Anthropic Messages | Complete Read loop |
| Cline CLI | 3.0.46 | AI-SDK Chat Completions | Complete 24-tool Plan/read_files/tool-result loop |
| Continue CLI | 1.5.47 | Chat Completions | Complete ten-tool read/tool-result loop |
| Claude Code | 2.1.217 | Anthropic Messages | Complete two-turn Read loop with `--tools Read`; full captured built-in schemas compile separately |
| Aider | 0.86.2 | Chat Completions | Inference/result PASS under `--dry-run`; edit format is model-profile dependent |
| Codex CLI | 0.144.6 | Responses | Lean tool set completes `exec_command`; feature-rich namespace expansion can exceed 59 tools |

## Client settings that matter

- OpenCode uses `@ai-sdk/openai-compatible`, base URL
  `http://127.0.0.1:8080/v1`, and explicit local limits matching Runner.
- Cline uses provider `openai-compatible`, model `runner`, base URL ending in
  `/v1`, and `thinking none` for a non-reasoning local profile.
- pi's Chat Completions and Responses providers use the `/v1` base. Its
  Anthropic Messages provider uses the server root without `/v1`.
- Continue declares `tool_use` capability and uses its ordinary OpenAI
  provider against the `/v1` base.
- Claude Code uses `ANTHROPIC_BASE_URL=http://127.0.0.1:8080`. For a bounded
  model-quality check, `--tools Read` avoids asking a 4B model to select among
  every harness tool. The full normal
  declaration was validated independently at a 32,768-token context.
- Aider should be compatibility-tested with `--dry-run`. An unknown local
  model falls back to `whole` edit format and may interpret a read task as an
  edit even when the HTTP exchange is correct.
- Codex custom providers use `wire_api = "responses"`. Current feature-rich
  installations may need `apps`, `multi_agent`, and web search disabled to
  remain under Runner's 59-tool constrained-union limit. Hosted web search is
  intentionally rejected rather than advertised as a local function.

## Compatibility changes proven by the clients

OpenCode and Cline exposed integer `minimum`, `maximum`,
`exclusiveMinimum`, and `exclusiveMaximum` in ordinary read/bash schemas.
Runner now enforces those bounds during streaming sampling and truncation
completion; their complete real client loops pass.

Claude Code exposed successive independent requirements:

1. `/v1/messages?beta=true` must route by URL path rather than comparing the
   raw request target.
2. `thinking.type: "adaptive"` is a hint and must not require a resident
   reasoning channel.
3. The harness inserts a system-role turn inside `messages[]`.
4. Metadata uses the tautological
   `propertyNames: {"type":"string"}` with an open object schema.
5. Tool schemas use bounded `number`, an enum-plus-const `anyOf`, JSON Schema
   2020-12 `format` annotations, and anchored repeated ASCII-class patterns.

Each item has a focused regression test. Unsupported constraints continue to
fail at request time rather than being silently weakened.

## Scope and exclusions

Roo Code was considered but not marked verified because its maintained
surface is an editor extension and this test environment did not provide a
representative editor host. Cursor and Windsurf likewise do not provide the
same isolated, provider-configurable headless surface used here. The sweep
therefore concentrates on terminal clients whose real requests can be
captured and whose tool loops can be deterministically replayed.

Large raw HTTP captures were used only as diagnostic evidence. They contained
client system prompts and were deliberately not committed. Every constraint
that changed Runner is represented by a focused test, while compatibility is
claimed only where the installed client itself completed the behavior listed
in the table.
