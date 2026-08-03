# Runner — Qwen2.5-7B-Instruct Q4_K_M, v2 matrix, 2026-08-03

Runner alone, on a bigger model than the cross-runtime rows use. No competitor
column: this is the v2 matrix's first run on a 7B, kept because the two new
families (`reasoning_then_tool`, `structured_final`) had only been exercised
against SmolLM2 and the built-in test model.

| runtime | version | passed | failed |
|---|---|---|---|
| Runner | 0.1.5-alpha | **105/105** | 0 |

CPU — `agent-torture.py` forces `--gpu off` for the server it spawns. Peak host
RSS 4.45 GB; 0.317 valid structured tasks/second, which is a CPU number for a
7B and not a throughput claim.

## Why not Qwen3-4B

Tried first, and it scores **15/105** — every tool and structured family fails
with `finish_reason: "reasoning_limit"` and no call emitted. That is not a
runner defect and not a Qwen3 defect. Qwen3-4B is a thinking model, runner caps
a constrained thinking prelude at half of `max_tokens` (`2866c89`), and the
matrix's budgets are 32–96 tokens: the model is still reasoning when the budget
runs out. **The matrix as written measures non-thinking models.** Running it
against a thinking model measures the budget, not the runtime, and a thinking
row would need its own token budgets to mean anything.

Worth recording separately: `reasoning_limit` is runner's own `finish_reason`,
not an OpenAI one. The Responses surface translates it to
`max_output_tokens` and Anthropic to `max_tokens`, but chat completions emits
it raw — deliberately, per `2866c89`. The official OpenAI SDK deserialises it
without complaint, but a client whose `finish_reason` is a closed union will
have no case for it.
