# Build plan — atem constrained tool calling

**Goal.** Give Muse Glimmer (and future Shade profiles) native, *constrained*
tool calling in its own atem format, reusing the existing schema compiler
rather than adding a parallel path. Three capabilities: render tool
definitions into the system turn, constrain generation to schema-valid atem,
and parse the guaranteed-valid block back into OpenAI `tool_calls`.

**Why this fits (read before coding).** atem is a JSON object in different
delimiters — a literal skeleton, ordered named fields, typed values — so it
maps onto machinery that already exists in `src/schema.c` and `src/template.c`.
Do NOT build a second constraint engine. The atem shape:

```
<atem:function_calls>
<atem:invoke name="TOOL_NAME">
<atem:parameter name="KEY">VALUE</atem:parameter>
...
</atem:invoke>
</atem:function_calls>
```

Key insight that makes this tractable: `<atem:invoke name="TOOL">` puts the
tool discriminator BEFORE any parameter, so a union over N tools diverges on
the tool name. This natively sidesteps the runner's known `SN_UNION`
first-byte-commit limitation (`compile_discriminated_action` in schema.c) — it
is exactly the "const discriminator first, then args" shape the README already
documents as the supported union form. Do NOT attempt the deferred
multi-alternative-tracking redesign; the format removes the need for it.

## Repo rules (AGENTS.md — non-negotiable)

TDD, one failing test/smoke per behavior first. Deep modules: atem work lives
behind `schema.c`/`template.c` public interfaces, no new parallel forward path.
schema.c's failure mode is *silently mis-constraining sampling* — no green
build proves correctness there, so every schema change lands with a fixture
test AND a temp-0 real-model run. Update README in the same commit as any
public-behavior change. Commit to the branch and push after each section.
Build/test on the Blackwell conda toolchain (`source
/opt/conda/etc/profile.d/conda.sh && conda activate ccbuild`, then
`CC=x86_64-conda-linux-gnu-gcc`); the real model is
`~/workspace/Gridcore/muse-glimmer/muse-glimmer-30B-kquant-17gb.gguf`, and a
b10353 reference build is at
`~/workspace/Gridcore/llama.cpp-b10353/build-static/bin`.

## Reference material

The full atem chat template is in the model's GGUF
(`tokenizer.chat_template`) and reproduced in `docs/muse-glimmer-cert-2026-08-11.md`
context — read the `render_atem`, `render_tool_defs`, and tool-result macros
there. The template is the spec. Two facts from it that drive the design:
list/object parameter values are emitted as `| tojson` (real JSON), and scalar
values are emitted RAW (unquoted, unescaped), terminating at
`</atem:parameter>`.

## Sections (land in order; each is a commit)

### S1 — Render tool definitions into the system turn
- `TMPL_MUSE` in `template.c` currently renders no tool defs (the cert doc
  notes it: "Tool declarations are not rendered"). Add the `render_tool_defs`
  path from the template: the fixed preamble, the `// Tool metadata` namespace
  list, then one `{"name":..., "description":..., "parameters":...}` line per
  tool, appended to the system message before `<|eot|>`. Also emit the
  `# Valid recipients:` line to include the tool namespaces.
- Render tool RESULTS: a message with role `tool` becomes
  `<|start|>tool NAME<|message|><tool_output name="NAME">\n...\n</tool_output><|eot|>`.
- Gate: `tests/test_template.c` (or the muse template test) — golden-string
  render of a 2-tool request + a tool-result turn, byte-compared to the
  template's output for the same input.

### S2 — Compile a tool set to a schema-constrained atem automaton
- New entry in `schema.c` that takes the request's `tools` array and builds an
  `SN_UNION` whose branches are one atem `<atem:invoke>` block per tool, each
  branch a literal-chain skeleton with the tool name baked in as the
  discriminator. Reuse `sn_new`, the literal-emit machinery, and the existing
  per-key value sub-automata.
- Parameter values by JSON-schema type: object/array parameters recurse into
  the EXISTING JSON value compiler (they are `| tojson` in the template);
  scalar parameters use a NEW node kind (see S3).
- Keys emit in the tool's `parameters` declared order (schema.c already
  guarantees declared-key order — preserve it).
- Gate: a fixture test that compiles a small tool set and asserts the automaton
  accepts a hand-written valid atem block and rejects malformed ones
  (wrong tag order, missing close). Reuse the `RUNNER_SCHEMA_TRACE=1` harness.

### S3 — New value node: raw-until-sentinel (`SN_RAW` or similar)
- One new `sn_kind` for scalar atem values: free bytes until the literal
  `</atem:parameter>`, no JSON quoting/escaping. Simpler than SN_STR (no escape
  state machine). Wire it into the accept/advance switch in schema.c
  (`SN_UNION` sites at ~1219 and ~1586 show the pattern) and the frame stack.
- Honest limitation to document, not fix: a scalar value containing the literal
  `</atem:parameter>` cannot be expressed — this is atem's own boundary (the
  template says output is regex-parsed, not XML). State it in the README.
- Gate: fixture round-trip — constrain a scalar param, feed bytes incl. one
  that would need escaping in JSON, confirm it passes raw and terminates.

### S4 — Truncation recovery for atem
- Extend the smallest-legal-ending logic: when the budget expires mid-block,
  close any open JSON value first, then emit the deterministic stack of closing
  tags (`</atem:parameter>` if in a value, `</atem:invoke>`,
  `</atem:function_calls>`). This is STRICTLY SIMPLER than JSON truncation
  recovery (a fixed close-tag stack, no structural ambiguity).
- Gate: fixture — cap the token budget mid-parameter, assert the emitted tail
  parses as a complete atem block with the started call intact. This is the
  headline capability; do not skip it.

### S5 — Parse atem output back into `tool_calls`
- Buffered + streaming, in `template.c`. The `tool_stream` demultiplexer
  (TS_TOOL/TS_ARGS states, ~line 876) and the ornith envelope parser
  (`render_atem`/gemma4 `<|tool_call>` parser at ~line 480) are the direct
  precedents — do NOT write a third parser from scratch; extend the envelope
  abstraction. Map `<atem:invoke name="X">` → tool name, each
  `<atem:parameter name="K">V</atem:parameter>` → an arguments key, assembling
  the OpenAI `{"id","type":"function","function":{"name","arguments"}}` shape.
  Because generation was constrained (S2-S4), the block is guaranteed
  well-formed — the parser does not need to handle arbitrary garbage, only the
  format.
- Gate: extend `scripts/agent-torture.py`'s tool matrix to run against the muse
  model with atem calling on; assert valid `tool_calls`, correct tool
  selection, and transport-invariant streaming (same verdicts the JSON path
  passes today).

### S6 — Wire-up, opt-in, README, real-model cert
- Selection: when the loaded arch is muse-glimmer AND the request carries
  `tools`, use the atem path; keep JSON-schema tool calling available. Decide
  and document whether atem is automatic for muse or opt-in via a request flag
  — automatic matches the model's training, so prefer automatic with a
  documented override.
- README: new subsection under tool calling — atem is constrained (not just
  parsed), truncation-surviving, and the scalar-value boundary above. Update
  the muse-glimmer support row to drop "atem tool syntax not implemented".
- Final cert: temp-0 real-model run on the Blackwell — a real tool request end
  to end (render → constrained generate → parse), plus the torture tool matrix
  green. Record a short evidence note beside the muse cert doc.

## Explicit non-goals
- No vision encoder / mmproj (out of scope, by owner decision).
- No DFlash drafter (separate, unscoped item).
- No change to the JSON-schema tool path's behavior for other archs.
- Do NOT implement the general `oneOf` multi-alternative redesign — atem's
  discriminator-first shape makes it unnecessary here.

## Definition of done
S1-S6 landed on this branch, each with its gate green; `make test` green on
the Blackwell; the real muse model completes a constrained atem tool call at
temp 0 and the atem tail still parses when the token budget is cut mid-call;
README and the muse support row updated; evidence note committed. Then stop and
report — do not merge to main or tag.
