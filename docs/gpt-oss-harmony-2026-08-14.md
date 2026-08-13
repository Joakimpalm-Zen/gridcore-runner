# gpt-oss Harmony chat — before and after, 2026-08-14

gpt-oss had no chat template. `template_detect()` fell through its terminal
fallback to `TMPL_LLAMA2` and fed the model `[INST]` / `<<SYS>>` markup that
appears **nowhere** in its GGUF — which carries `<|start|>`, `<|channel|>`,
`<|message|>`, `<|end|>` and `<|return|>`. This is the evidence note for the
fix. Model: `gpt-oss-20b-MXFP4.gguf`, temp 0, full GPU offload (24/24) on the
Blackwell MIG slice.

## Before — the llama2 fallback

```
$ ./runner -m gpt-oss-20b-MXFP4.gguf -i --temp 0
chat mode (template: llama2) — Ctrl-D or /exit to quit

>  4

We need to produce a response that is a single line, no markdown, no
explanation. The user asks: "What is 2+2?" The answer is 4. So we output "4".
The instruction says: "Respond with a single line. Do not use any markdown
formatting. Do[64 tok, 2.4 tok/s]
```

Three symptoms, one cause. The answer `4` arrives before anything else because
the model is completing invented markup; the chain-of-thought is served as
ordinary content because nothing declared the channel structure it was trained
on; and generation only stops because it hit the 64-token ceiling — the stops
being watched were llama2's, not `<|return|>`.

## After — TMPL_HARMONY

```
$ ./runner -m gpt-oss-20b-MXFP4.gguf -i --temp 0
chat mode (template: harmony) — Ctrl-D or /exit to quit

> [thinking]
We have a conversation. The user asks: "What is 2+2?" The assistant should
answer. There's no conflict with policy. It's a simple math question. The
answer is 4. There's no policy violation. So we can answer.
[/thinking]
2 + 2 equals **4**.[66 tok, 30.6 tok/s]
```

Reasoning is separated, the answer is the answer, and it **stops on its own**
at 66 tokens rather than running to the ceiling.

## Server gates — `/v1/chat/completions`, temp 0

| request | `content` | `reasoning_content` | finish |
|---|---|---|---|
| default | `4` | the analysis (159 chars) | `stop` |
| `enable_thinking: false` | `The sum of 2 and 2 is 4.` | absent | `stop` |
| `enable_thinking: true` | `4` | the analysis | `stop` |

Three-turn conversation, system message `"Answer with a single number and
nothing else."`, asking 2+2 → ×5 → −3:

```
content           : '17'
reasoning_content : 'We have a conversation. The user asks: "Subtract 3." The
                     context: earlier we had 2+2=4, multiplied by 5 gives 20,
                     subtract 3 gives 17. So answer: 17. ...'
finish_reason     : stop
```

Correct arithmetic across the history, so the turn framing and the developer
turn both survive multi-turn rendering.

## What the implementation is

**Detection.** `<|channel|>` *and* `<|return|>` — the pair no other family
carries. Checked before muse, which shares the `<|start|>role<|message|>`
framing but additionally requires `<|eot|>` that Harmony lacks, so the two
branches cannot claim each other. Also detected from vocabulary when a model
ships no template string.

**Rendering.** A format-owned system turn (identity, `Reasoning: medium`, the
channel declaration), then the caller's system message as a **developer** turn
— the one structural surprise when porting from ChatML. History assistant
turns render as `<|channel|>final`; a past turn's analysis is not replayed,
matching the reference template.

**Stops.** `<|return|>` and `<|call|>` are added; `<|end|>` is *removed* for
this arch. `<|end|>` is in the global stop list for phi3, whose assistant
turns really do end there — but in Harmony it terminates the analysis message,
so leaving it in stops generation on the reasoning and never reaches the
answer.

**Analysis channel.** It rides the existing thinking splitter — no third
mechanism. The generation prompt primes `<|channel|>analysis<|message|>` for
every mode except `enable_thinking: false` (which primes `final`), so the
stream starts inside reasoning and the splitter is initialised with
`think_init_reasoning`, exactly as ornith always does and muse does when
forced.

## Two traps worth recording

**The splitter markers must be the DECODED strings.** Harmony's control tokens
detokenize to nothing, so the bytes that actually arrive are the bare channel
words:

```
raw stream: analysisWe have a conversation...assistantfinal2 + 2 equals **4**.
```

`HARMONY_THINK_OPEN` is therefore `"analysis"` and `HARMONY_THINK_CLOSE` is
`"assistantfinal"`. Writing the control-token spellings compiles, renders, and
silently never matches — the first live run served the whole analysis as
content for exactly that reason. muse recorded the same trap.

**Priming must be matched on both surfaces.** Because the prompt primes the
channel, any surface that initialises the splitter content-first sees no open
marker and prints the analysis as the answer. That had to be fixed in two
places — `completion.c` (`harmony_primed_think`) and `main.c`'s interactive
loop. A single-surface fix looks correct through the API and wrong at the CLI.

## Native tool calling — completed 2026-08-13

Runner now renders the official Harmony `# Tools` TypeScript namespace and
constrains the actual channel-first generation form:

```
<|channel|>commentary to=functions.get_weather<|constrain|>json
<|message|>{"city":"Oslo","units":"celsius"}<|call|>
```

History is intentionally different: the reference renderer canonicalizes a
past call with the recipient before the channel. The native parser accepts
both orders. Runner therefore constrains the model's trained channel-first
form during generation and replays the reference recipient-first form in
history; treating those as one layout trapped the model in commentary and was
caught by the first live run.

The parameter document is compiled through the same strict JSON-schema engine
as other tool paths. `auto`, `required`, named, and `none` choices; structured
final answers; visible commentary; analysis; tool results; multiple prior
calls; and buffered/SSE demultiplexing share that compiler and mapper. A
pre-call analysis/commentary message is bounded to 192 UTF-8 bytes so a
required choice cannot spend the entire request repeating that it intends to
call; the handoff is then the only legal continuation and the model still
generates the arguments.

Blackwell evidence used the canonical `gpt-oss-20b-MXFP4.gguf`, full CUDA
offload (24/24 layers) on the 24 GB MIG slice, temperature 0:

| surface / turn | result |
|---|---|
| Chat, required + reasoning | `finish_reason:"tool_calls"`; 192-byte reasoning; `get_weather({"city":"Oslo","units":"celsius"})` |
| Chat SSE, required | identical call reconstructed from deltas; no protocol text leaked |
| Chat tool-result continuation, prose result | final content `4 °C`, `finish_reason:"stop"`, no second call |
| Chat tool-result continuation, JSON result | one redundant repeat call, then the final answer on the next round — see below |
| Responses, required | completed `function_call` with the same name and arguments |
| Anthropic Messages, `tool_choice:any` | `stop_reason:"tool_use"`; object input with the same city and units |

Every row above was re-measured on 2026-08-14 against the tree that carries
this change.

The continuation rows differ by the *shape* of the tool result, and the
difference is a known cost rather than an open question. A prose result
(`4 °C`) fits inside the 192-byte analysis bound and the model answers. A JSON
result (`{"temperature": 4, "units": "celsius"}`) does not: the analysis is cut
mid-quotation, the forced handoff lands on the call branch, and the model
repeats the call it was just answered before replying on the following round.
Lifting the bound on this branch was implemented and measured — it produces an
unbounded loop with corrupted arguments, strictly worse — and reverted:
[docs/negative-result-harmony-analysis-bound.md](negative-result-harmony-analysis-bound.md).

Focused native grammar, official-render golden, buffered mapper, and every
stream chunk-boundary test pass. The full repository gate is green on all three
supported platforms as of 2026-08-14: macOS/ARM (Apple M1, Metal), Linux/x86_64
(Blackwell MIG slice, CUDA) and Windows x86_64 (MinGW-w64 UCRT64).

## Remaining out of scope

- **Re-certifying gpt-oss's greedy / cpu_cuda rows.** Chat rendering does not
  touch them; they stand as recorded.
