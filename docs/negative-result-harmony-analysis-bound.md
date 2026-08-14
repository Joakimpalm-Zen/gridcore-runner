# Negative result: lifting the Harmony analysis bound on the auto branch

*2026-08-14. Status: implemented, measured, reverted. Not in the product.*

> ## CORRECTION 2026-08-14 (later the same day): the motivating symptom is gone
>
> **This document's premise no longer holds.** It was written because a JSON
> tool result made gpt-oss-20b repeat a call it had just been answered, while
> a prose result did not. That asymmetry was real when measured — and it was
> caused by something this document does not mention.
>
> Runner was replaying tool results as
> `<|start|>functions.NAME<|channel|>commentary<|message|>`, which the
> openai-harmony reference parser **rejects outright** (`Unknown role:
> functions.NAME`): a namespaced author is only recognised as the tool role
> via the ` to=assistant` branch. Every multi-turn tool conversation was
> replaying history in a shape gpt-oss never saw in training. Fixed in
> `0ab3238`.
>
> Re-measured on the fixed tree, same box, same model, temperature 0, each
> arm run twice:
>
> | tool result | `enable_thinking` | finish_reason | repeat call |
> |---|---|---|---|
> | JSON  | on  | `stop` | no |
> | prose | on  | `stop` | no |
> | JSON  | off | `tool_calls` | **yes** |
> | prose | off | `tool_calls` | **yes** |
>
> **The JSON-versus-prose axis collapsed completely.** Payload shape no longer
> makes any difference. What remains is a different symptom with a different
> trigger: with thinking disabled the model repeats the call regardless of how
> the result is formatted — and with thinking disabled there is no analysis
> channel, so the 192-byte bound this document is about is not even in play.
>
> **What still stands.** The experiment itself is unaffected: lifting the bound
> on the auto branch really did produce an unbounded loop with corrupted
> arguments, and that measurement is reproducible. Do not re-attempt it on the
> strength of this correction.
>
> **What does not.** The reasoning that connected the bound to the observed
> repeat call. That chain ran: JSON result → analysis cut mid-quotation → hand
> off to the call branch. Prose results now behave identically to JSON, so
> mid-quotation truncation cannot be what distinguishes them, because nothing
> distinguishes them any more.
>
> **What this cost, recorded deliberately.** A later investigation ranked the
> malformed history *below* payload shape as an explanation, reasoning that the
> header was identical in both arms so it could not explain an asymmetry
> between them. That reasoning was sound and the conclusion was still wrong:
> the header was identical, but it was identically *broken*, and repairing it
> removed the asymmetry rather than either arm. A shared defect can be the
> cause of a difference it does not itself vary with.
>
> The open question is now "why does `enable_thinking:false` cause a repeat
> call", which is not this document's subject. See the follow-up plan item.


## What was tried

A strict Harmony tool turn bounds the model's pre-call `analysis` (or visible
`commentary`) message to 192 UTF-8 bytes; at that boundary the trained
assistant handoff becomes the only legal continuation. The bound was written
for the `tool_choice: "required"` case, where masking the final-answer branch
otherwise lets gpt-oss narrate its intended call forever and spend the whole
request budget before generating any arguments.

Reading that rationale, the bound looked wrong on the *auto* branch: there a
final answer is legal, so the model can end its own plan by answering, and
cutting it mid-thought can only strand it. A live symptom agreed. On a
tool-result continuation turn with tools still declared and a **JSON** tool
result, gpt-oss-20b truncated at exactly 192 bytes mid-quotation —

```
The function returned {"temp":
```

— and then re-issued the identical `get_weather({"city":"Oslo","units":"celsius"})`
call it had just been given the answer to, instead of replying. Turning
thinking off on the same history answered immediately, which made the
truncated analysis look causal.

So the bound was made conditional: applied when `!allow_final`, lifted when a
final answer is admitted (`src/schema.c`, `harmony_after_reasoning`).

## What was measured

Blackwell RTX PRO 6000 MIG 1g.24gb, `gpt-oss-20b-MXFP4.gguf`, 24/24 layers on
GPU, temperature 0, `max_tokens` 400. Identical history in every row: one
`get_weather` call, then the tool result `{"temperature": 4, "units": "celsius"}`,
tools still declared, `tool_choice` auto.

| analysis bound | reasoning emitted | finish_reason | call arguments | agent loop |
|---|---|---|---|---|
| 192 bytes (shipped) | 192 B | `tool_calls` | `{"city":"Oslo","units":"celsius"}` | one redundant call, **answers at round 3** |
| unbounded (the change) | 1617–1863 B | `length` | `{"city":"","units":"celsius"}` | **did not terminate in 6 rounds** |

Unbounded, the model does not finish its plan and answer. It keeps
deliberating about the call it has already made, runs out of token budget
mid-argument, and the closer emits a legal-but-empty `city`. Every subsequent
round repeats it. The change is strictly worse than the behavior it was meant
to fix: one wasted round-trip became an unbounded loop that also corrupts the
arguments.

For contrast, on the same history the bound is not what stops a clean answer —
removing the *call branch* is:

| continuation variant | result |
|---|---|
| plain `4 °C` result, tools declared | `stop`, correct final answer |
| JSON result, `tool_choice: "none"` | `stop`, 405 B reasoning, correct answer |
| JSON result, tools omitted | `stop`, 405 B reasoning, correct answer |
| JSON result, `enable_thinking: false` | `stop`, correct answer |

## Why it was rejected

The bound is load-bearing on both branches, not only the masked one. What
drives the runaway is an admitted call branch, not a masked final branch: while
calling remains legal the model keeps re-deliberating, and the bound is what
forces the handoff. Its original comment named only half of its own
justification.

The residual cost is real but bounded and self-correcting: a JSON tool result
costs one redundant call before the model answers. A prose tool result costs
nothing. That is the better of the two measured behaviors, so the shipped
bound stands.

## What would actually address it

Not a different constant — 1024 was not measured, and tuning a magic number
against one model and one prompt is not evidence. The candidates worth testing
are structural: suppressing the call branch once a matching tool result is
already in history (the `tool_choice: "none"` row above answers cleanly, and a
result for the pending call is a strong signal the turn should conclude), or
bounding the analysis by *token* budget share rather than a byte constant.
Both change turn construction rather than the constant, and both need their
own evidence run.
