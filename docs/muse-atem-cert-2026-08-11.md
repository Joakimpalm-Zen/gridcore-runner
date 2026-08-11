# Muse Glimmer native atem certification — 2026-08-11

Environment: `ccbuild`, NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation
Edition MIG 1g.24gb, all 52 Muse Glimmer layers on CUDA. Model:
`models/muse-glimmer-30B-kquant-17gb/muse-glimmer-30B-kquant-17gb.gguf`.

The server was built from `atem-tool-calling` and exercised through
`POST /v1/chat/completions` at temperature 0. Evidence retained from the runs:

- Required `weather.get` rendered native definitions, constrained from the
  recipient header, generated atem, and mapped to
  `weather.get({"city":"Oslo"})` with `finish_reason: "tool_calls"`.
- The equivalent SSE request reconstructed the same name and arguments and
  ended with `finish_reason: "tool_calls"`.
- `enable_thinking:true` emitted a self-addressed reasoning turn that computed
  `17 * 23 = 391`, followed by the constrained call
  `record_conclusion({"result":"391"})` in the same response.
- `parallel_tool_calls:true` generated two native calls separated by Muse's
  turn boundary and returned one response containing `call_0` for Oslo and
  `call_1` for Bergen.
- A one-token forced truncation closed the nested atem call. Raw boolean
  recovery used its declared type (`false`), leaving schema-valid executable
  OpenAI arguments.
- The complete 105-request real-model torture matrix passed 105/105. It
  repeatedly covered nested arguments, rotating named selection, 1/2/3/5/8
  token truncation, large string enums, prior-reasoning history, structured
  finals, ordinary streaming, and tool-bearing streaming normalization. Its
  transport tests reparsed identical SSE bytes at deterministic split points
  and obtained identical calls.
- Explicit reasoning was also verified over SSE: reasoning deltas contained no
  recipient/atem residue, the call reconstructed as
  `record_conclusion({"result":"391"})`, and the terminal reason was
  `tool_calls`.
- The documented `atem_tool_calling:false` override produced the same
  `ping({})` call through the generic JSON envelope in both buffered and SSE
  requests. A non-boolean override was rejected with HTTP 400.
- Muse JSON-schema output passed buffered and SSE without leaking its
  `to=user` header. Native `tools` plus `tool_choice:"auto"` and a final
  response schema selected the user branch and returned a schema-valid JSON
  object.

Targeted C gates (`test-template`, `test-tools`) and the mandatory temperature-0
real-model JSON-schema regression passed before the full repository gate.
