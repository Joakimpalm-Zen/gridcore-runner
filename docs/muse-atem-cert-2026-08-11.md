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

## Addendum — review fixes, 2026-08-12

A branch review before merge found two defects the matrix above never
exercised; both were confirmed live at temperature 0 on the same box and
fixed on this branch.

**Plain thinking chat leaked recipient residue (regression).** Narrowing
`think_close` to ` to=` made the unconstrained splitter close early on
`assistant to=user`: content came back `user391`, reasoning trailed
`assistant` (no tools, `enable_thinking:true` or default; THINK_OFF was
unaffected). The close is restored to the full `assistant to=user`;
constrained runs never depended on those bytes — `constraint_finish_think`
feeds the close on the `<|eom|>` control. Re-measured after the fix:
content `391`, reasoning ends at the model's own last sentence, in
think-on, think-default and think-off modes.

**The `to=user` free-text answer could not terminate.** Its automaton ends
at the SPELLED `<|eot|>` sentinel, but the model's real end-of-turn is the
`<|eot|>` token, which decodes to no bytes and was masked until the
automaton completed — so every native plain answer burned the full
`max_tokens` and finished `length`. `sval_at_raw_tail` now reports the
trailing-raw state and the engine accepts a stop token there as the
answer's natural end. Gated after the fix: tool call via required and
named choice, reasoning-then-call, 3-token truncation recovery, and
streaming required-call all unchanged and green.

**Known issue, pre-existing (control-verified on the unfixed binary):**
with `tool_choice:"auto"`, a model that declines the tool must spell the
` to=user<|message|>` header as literal text; on some prompts greedy
decoding then loops emitting `<|message|>` to the token limit ("Just say
hello" reproduces it; a substantive question does not). The loop is
identical with and without these fixes. The natural-stop fix bounds the
damage only when the model produces a real answer; the header-forcing
design itself is the open item.

## Addendum 2 — the header-forcing loop, fixed 2026-08-12

The known issue above is closed. The engine now accepts a decoded-empty
protocol token (Muse's `<|message|>`, `<|start|>`) in constrained output
whenever feeding its raw vocab spelling would advance the automaton
(`constraint_spelling_ok`), and feeds that spelling through the machine and
the visible stream so downstream parsing is unchanged. The model emits its
natural trained header instead of being forced to type the markers out as
text. Guardrails: stop tokens are excluded (the raw-tail rule owns them),
and the acceptance is refused while the machine is consuming free content
(a raw scalar value, a JSON string), where a protocol marker is corruption.

Measured on the real model at temperature 0 after the fix:

- The loop reproducer ("Just say hello, do not call any tool",
  `tool_choice:"auto"`): content `Hello!`, `finish_reason:"stop"`,
  **5 completion tokens** — previously 200 tokens of `<|message|>` and
  `"length"`. Identical over SSE.
- A substantive auto-mode text answer now ends at the model's own stop
  (74 tokens, `"stop"`), where pre-fix even clean answers burned the full
  budget.
- Full regression battery unchanged: plain thinking modes clean, required
  and named calls, reasoning-then-call, streaming both ways.
- Three-token forced truncation now reports `"length"` with the recovered
  call attached, which matches the documented truncation contract (the
  earlier `"tool_calls"` at 3 tokens predated the natural header path).
- **The complete v3 torture matrix passed 120/120** against this model on
  the same box (schema `xyntetik.agent-torture.v3`, 18.6 minutes).
