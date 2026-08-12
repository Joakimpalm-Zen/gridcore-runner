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

## Out of scope, not attempted

- **Harmony tool calling.** Its `commentary` channel and `<|call|>` recipient
  syntax are their own project. Not rendered rather than approximated — the
  granite-tool-calling precedent. `<|call|>` is in the stop set so an
  unrendered attempt stops instead of running away.
- **Re-certifying gpt-oss's greedy / cpu_cuda rows.** Chat rendering does not
  touch them; they stand as recorded.
