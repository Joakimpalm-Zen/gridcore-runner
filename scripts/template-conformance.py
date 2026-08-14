#!/usr/bin/env python3
"""Byte-exact conformance gate: runner's native C chat renderer vs each
family's UPSTREAM jinja chat template.

Runner renders chat templates natively in C instead of executing the model's
jinja template. That is deliberate (no jinja interpreter in the inference
path, and constrained decoding needs the framing as structure rather than as
text), but it means every family can silently DRIFT from its reference and
nothing notices -- goldens written from runner's own output only prove
self-consistency. tests/test_template.c is exactly that kind of golden. This
script supplies the missing oracle.

For each family it:
  1. loads the upstream `chat_template` (HuggingFace tokenizer_config.json /
     chat_template.jinja, or tokenizer.chat_template out of a local GGUF),
  2. renders a fixed conversation matrix through it with transformers-
     compatible jinja semantics (trim_blocks + lstrip_blocks, raise_exception,
     strftime_now, bos_token/eos_token, add_generation_prompt),
  3. renders the SAME matrix through runner's C renderer
     (render_messages_with_tools, via scripts/template-conformance-render.c),
  4. diffs byte-for-byte, and subtracts the deviations runner takes ON PURPOSE
     -- scripts/template-conformance-allowlist.json, every entry carrying a
     written reason and a source citation the harness re-verifies,
  5. subtracts the differences already KNOWN to be bugs --
     scripts/template-conformance-baseline.json, a backlog with per-entry
     status, not permission.

    make template-conformance            run the gate
    make template-conformance-refresh    re-fetch the upstream oracles first
    make template-conformance-baseline   re-record the known-difference backlog

Exit status:
    0   every selected family checked; no new drift; allowlist healthy
    1   NEW drift, drift that CHANGED SHAPE, a fixed-but-still-listed backlog
        entry, an upstream oracle that moved, or a rotted allowlist entry
    2   NOT CHECKED -- jinja2 missing, no compiler, or an oracle that could
        not be obtained. This is never reported as conformance. A gate that
        quietly checks nothing is worse than one that fails.
"""

import argparse
import datetime
import hashlib
import json
import os
import re
import struct
import subprocess
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, "scripts")
ALLOWLIST = os.path.join(SCRIPTS, "template-conformance-allowlist.json")
BASELINE = os.path.join(SCRIPTS, "template-conformance-baseline.json")
HARMONY_SCRIPT = os.path.join(SCRIPTS, "template-conformance-harmony.py")
HARMONY_FIXTURE = os.path.join(SCRIPTS,
                               "template-conformance-harmony-oracle.json")

EXIT_OK, EXIT_DRIFT, EXIT_NOT_CHECKED = 0, 1, 2


# Conversations a family's own reference template REFUSES to render, so there
# is no oracle to compare against and the case asserts nothing. Declared, not
# discovered: every entry is re-verified on each run (see Family.cannot), so
# one that stops being true is reported as rot rather than quietly skipped.
#
# All five reasons are the template's own raise_exception() text, copied from
# the run that first hit them.
ALTERNATE = ("reference",
             "the template raises: 'Conversation roles must alternate "
             "user/assistant/user/assistant/...'. The family has no rendering "
             "for this conversation at all, so there is nothing to conform to")
ALTERNATE_AFTER_SYS = (
    "reference",
    "the template raises: 'After the optional system message, conversation "
    "roles must alternate user/assistant/user/assistant/...'")
NO_MID_SYSTEM_ROLE = (
    "reference",
    "the template raises 'Invalid message role': a system turn is only a role "
    "this template knows in first position")
NO_MID_SYSTEM = (
    "reference",
    "the template raises: 'System message must be at the beginning.'")


# --------------------------------------------------------------- families
#
# Where each family's reference template comes from. `hf` reads the model
# repo's tokenizer_config.json (falling back to chat_template.jinja); `gguf`
# reads tokenizer.chat_template straight out of a local GGUF, which is what
# src/template.c cites for the families whose HF repo is gated.

class Family:
    def __init__(self, runner, source, note="", tool_family=False,
                 thinking_var=None, skip=None, tokenizer=(), cannot=None,
                 oracle_tool_shape=None):
        self.runner = runner            # --chat-template name
        self.source = source            # ("hf", repo) | ("gguf", path)
        self.note = note
        self.tool_family = tool_family  # matrix includes the tool rows
        self.thinking_var = thinking_var
        self.skip = skip                # reason this family has no oracle
        # GGUF basenames that carry THIS family's tokenizer, best first. The
        # token-level check needs a tokenizer, and only a tokenizer from the
        # same family answers the question: borrowing another family's would
        # manufacture agreement (or disagreement) out of an unrelated vocab.
        # Several names because a shelf may hold any quant of the checkpoint.
        self.tokenizer = tokenizer
        # How THIS reference wants tools handed to it. Runner is always fed the
        # OpenAI wire form, because that is what a client sends; only the
        # ORACLE side is reshaped, and only where the template cannot read the
        # wire form at all. None is the common case (an OpenAI-wrapped
        # tools[] and a MAPPING for tool_call.arguments -- see the call site).
        # "flat-string" is apertus: `render_tools` reads tool.name /
        # tool.description / tool.parameters straight off the item, so an
        # OpenAI-wrapped declaration raises UndefinedError; and its call block
        # is `'{"' + function.name + '": ' + function.arguments + '}'`, a
        # string concatenation that raises TypeError on a mapping. It is the
        # one family that wants the arguments STRING the wire actually carries.
        self.oracle_tool_shape = oracle_tool_shape
        # Cases this family genuinely CANNOT express, with the side that
        # refuses ("reference" = the upstream template raises, "runner" = the
        # server refuses the request) and why. Every entry is re-verified on
        # every run: a declared skip that stops refusing is reported as rot,
        # so this cannot quietly become a place to hide a case.
        self.cannot = cannot or {}


FAMILIES = {
    "chatml": Family(
        "chatml", ("hf", "Qwen/Qwen2.5-7B-Instruct"),
        note="generic ChatML; Qwen2.5 is the concrete checkpoint the repo "
             "tests against (models/Qwen2.5-7B-Instruct-Q4_K_M.gguf)",
        tool_family=True,
        tokenizer=("Qwen2.5-7B-Instruct-Q4_K_M.gguf",
                   "Qwen2.5-7B-Instruct-Q8_0.gguf")),
    "chatml-think": Family(
        "chatml-think", ("hf", "Qwen/Qwen3-4B"),
        note="src/template.h cites 'Qwen/Qwen3-* tokenizer_config.json'",
        tool_family=True, thinking_var="enable_thinking",
        tokenizer=("Qwen3-4B-Q4_K_M.gguf", "Qwen3-4B-Q8_0.gguf")),
    "llama2": Family(
        "llama2", ("hf", "unsloth/llama-2-7b-chat"),
        note="meta-llama/Llama-2-7b-chat-hf is gated (HTTP 401); this mirror "
             "carries the identical Llama-2 template",
        # TinyLlama ships Llama-2's tokenizer verbatim (same 32000-entry
        # SPM vocab), so it answers the token question for this family when
        # the 7B chat GGUF is not on the shelf. The report names the file
        # that was actually used, so the substitution is never silent.
        tokenizer=("llama-2-7b-chat-Q4_K_M.gguf", "tinyllama-q2k.gguf"),
        cannot={"consecutive-user": ALTERNATE,
                "consecutive-assistant": ALTERNATE,
                "system-mid-history": ALTERNATE}),
    "llama3": Family(
        "llama3", ("hf", "unsloth/Llama-3.2-3B-Instruct"),
        note="meta-llama/Llama-3.2-3B-Instruct is gated (HTTP 401); this "
             "mirror carries the identical Llama-3.2 template",
        tokenizer=("Llama-3.2-3B-Instruct-Q4_K_M.gguf",
                   "Llama-3.2-3B-Instruct-Q8_0.gguf")),
    "zephyr": Family("zephyr", ("hf", "HuggingFaceH4/zephyr-7b-beta"),
                     tokenizer=("zephyr-7b-beta-Q4_K_M.gguf",)),
    "gemma": Family(
        "gemma", ("gguf", "models/google_gemma-3-4b-it-Q4_K_M.gguf"),
        note="google/gemma-3-4b-it is gated on HF (401); the GGUF the repo "
             "already tests against carries the same tokenizer.chat_template",
        tokenizer=("google_gemma-3-4b-it-Q4_K_M.gguf",),
        cannot={"consecutive-user": ALTERNATE,
                "consecutive-assistant": ALTERNATE,
                "system-mid-history": ALTERNATE}),
    "gemma4": Family(
        "gemma4", ("gguf", "models/e2b-q40.gguf"),
        note="src/template.h cites 'gemma-4 tokenizer.chat_template (read "
             "from the GGUF)'; byte-identical to google/gemma-4-E2B-it's "
             "chat_template.jinja on HF",
        # The reference DOES define a tool protocol, so the tool rows are
        # generated and compared. Read out of the template itself, not
        # inferred from runner: a declaration is `<|tool>` + the
        # format_function_declaration macro + `<tool|>` inside the leading
        # `<|turn>system` block; a call is
        # `<|tool_call>call:NAME{k:v,...}<tool_call|>`; a result is
        # `<|tool_response>response:NAME{k:v}<tool_response|>` emitted from
        # INSIDE the assistant turn that made the call, by the template's
        # forward-scan over the consecutive `role: tool` messages. Until
        # 2026-08-14 this family was declared without tool_family, so the
        # matrix generated no tool rows at all and the entire tool path was
        # UNMEASURED while the family reported CLEAN.
        tool_family=True,
        thinking_var="enable_thinking",
        tokenizer=("e2b-q40.gguf", "e4b-q4km.gguf")),
    "mistral": Family("mistral", ("hf", "mistralai/Mistral-7B-Instruct-v0.3"),
                      tokenizer=("Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
                                 "Mistral-7B-Instruct-v0.3-Q8_0.gguf"),
                      cannot={"consecutive-user": ALTERNATE_AFTER_SYS,
                              "consecutive-assistant": ALTERNATE_AFTER_SYS,
                              "system-mid-history": ALTERNATE_AFTER_SYS}),
    "phi3": Family("phi3", ("hf", "microsoft/Phi-3.5-mini-instruct"),
                   tokenizer=("Phi-3.5-mini-instruct-Q4_K_M.gguf",)),
    "apertus": Family(
        "apertus", ("hf", "swiss-ai/Apertus-8B-Instruct-2509"),
        note="tests/test_template.c cites this repo's chat_template.jinja",
        # The reference DOES define a tool protocol, read out of the template
        # itself: a declaration is the `render_tools` TypeScript block inside
        # the developer turn (`'Tool Capabilities:\n' + render_tools(tools)`,
        # where no tools gives the literal `Tool Capabilities: disabled`); a
        # call is `<|tools_prefix|>[{"NAME": ARGS}]<|tools_suffix|>`; a result
        # has NO token of its own -- it is the raw output text inside a `[`,`]`
        # list appended to the still-open assistant turn. Declared without
        # tool_family until 2026-08-14, so none of it had been compared.
        tool_family=True, oracle_tool_shape="flat-string",
        tokenizer=("Apertus-8B-Instruct-2509-Q4_K_M.gguf",),
        cannot={"system-mid-history": NO_MID_SYSTEM_ROLE}),
    "ornith": Family(
        "ornith", ("hf", "ornith-ai/Ornith-1.0-9B"),
        note="docs/ornith-reference.md names the model; the -GGUF repo is the "
             "same checkpoint", tool_family=True,
        thinking_var="enable_thinking",
        tokenizer=("Ornith-1.0-9B-Q4_K_M.gguf",),
        cannot={"system-mid-history": NO_MID_SYSTEM}),
    "muse": Family(
        "muse", ("hf", "meta-models/Muse-Glimmer-30B"),
        note="src/template.c cites 'the model's OWN tokenizer.chat_template, "
             "read from the official meta-models GGUF'",
        tool_family=True,
        tokenizer=("Muse-Glimmer-30B-Q4_K_M.gguf",)),
    "granite": Family(
        "granite", ("gguf", "models/granite-4.1-8b-Q4_0.gguf"),
        note="src/template.c cites 'the model's OWN tokenizer.chat_template'; "
             "the ibm-granite HF repo is gated (401)",
        tokenizer=("granite-4.1-8b-Q4_0.gguf", "granite-4.1-3b-Q8_0.gguf")),
    # The one family whose oracle is NOT its jinja. gpt-oss's wire format is
    # defined by the openai-harmony library and its docs/format.md; the jinja
    # in the GGUF is a reimplementation of that spec with gaps it cannot
    # close (it has no content_type field, so it can never emit
    # <|constrain|>). Comparing runner against the reimplementation reported
    # runner as drifted in three places where runner matches the spec.
    # scripts/template-conformance-harmony.py holds the mapping and the
    # evidence.
    "harmony": Family(
        "harmony", ("harmony", "openai-harmony"),
        note="the openai-harmony library (docs/format.md + encoding.rs) is "
             "the protocol's reference implementation; the GGUF's jinja is a "
             "lossy reimplementation of it",
        tool_family=True,
        tokenizer=("gpt-oss-20b-MXFP4.gguf",)),
    "raw": Family("raw", None,
                  skip="TMPL_RAW is runner's own no-framing mode: it "
                       "concatenates message contents and has no upstream "
                       "template to conform to."),
}

# The GGUF-sourced families need a multi-GB local model to FETCH their oracle
# (after that the cached .jinja is enough). Anywhere without models/ -- CI,
# a fresh clone -- only this subset is checkable, and asking for the rest
# would make the gate red for a reason that has nothing to do with drift.
NETWORK_ONLY = tuple(n for n, f in FAMILIES.items()
                     if f.source and f.source[0] == "hf")



# ------------------------------------------------------------- the matrix

SYS = "You are a helpful assistant."
U1 = "What color is the sky?"
A1 = "Blue, because of Rayleigh scattering."
U2 = "And at sunset?"

# Non-ASCII that exercises more than "the encoder passes bytes through": a
# combining mark, an astral-plane emoji, a ZWJ sequence that must not be split,
# RTL text, CJK, and the two characters a careless JSON or C-string writer
# breaks on.
UNI = ('Færre "smutthull" \\ 100 % — kombinert: é, '
       'emoji: \U0001f300 \U0001f469‍\U0001f4bb, '
       'RTL: שלום, CJK: 漢字、'
       'أهلاً')

# Long enough to force the prompt buffer past its opening guess several times
# over. src/api.h records that a short buffer does not fail the request, it
# silently truncates the TAIL -- so a renderer edit that changes the size
# arithmetic has to be caught by a case big enough to reach it.
LONG = ("Consider the following at length. " * 700).strip()

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string",
                                    "description": "City name."}},
            "required": ["city"],
        },
    },
}]
CALL = {"id": "call_1", "type": "function",
        "function": {"name": "get_weather",
                     "arguments": '{"city": "Oslo"}'}}
CALL2 = {"id": "call_2", "type": "function",
         "function": {"name": "get_weather",
                      "arguments": '{"city": "Bergen"}'}}


def matrix(family):
    """The same conversations for every family, so results are comparable.

    Each case is (id, messages, add_generation_prompt, thinking,
    oracle_messages). `oracle_messages` is None except where the two sides
    must legitimately be fed different INPUT to be asked the same QUESTION --
    see the content-array case below, which is the only one.

    The universal block runs for every family. The thinking and tool blocks
    run only where the family has the concept at all; anything a specific
    family cannot express is declared in Family.cannot with a reason, and
    re-verified rather than assumed.
    """
    cases = [
        # ---- shape of the conversation
        ("user-only+gen", [{"role": "user", "content": U1}], True, "default"),
        ("user-only-nogen", [{"role": "user", "content": U1}], False,
         "default"),
        ("system+user+gen", [{"role": "system", "content": SYS},
                             {"role": "user", "content": U1}], True,
         "default"),
        ("system+multiturn+gen", [{"role": "system", "content": SYS},
                                  {"role": "user", "content": U1},
                                  {"role": "assistant", "content": A1},
                                  {"role": "user", "content": U2}], True,
         "default"),
        ("multiturn+gen", [{"role": "user", "content": U1},
                           {"role": "assistant", "content": A1},
                           {"role": "user", "content": U2}], True, "default"),
        ("multiturn-nogen", [{"role": "user", "content": U1},
                             {"role": "assistant", "content": A1},
                             {"role": "user", "content": U2}], False,
         "default"),
        # A trailing assistant turn is what a prefill/continuation request
        # looks like. With the generation prompt it must NOT open a second
        # assistant turn; without it the render has to stop mid-turn.
        ("trailing-assistant+gen", [{"role": "user", "content": U1},
                                    {"role": "assistant", "content": A1}],
         True, "default"),
        ("trailing-assistant-nogen", [{"role": "user", "content": U1},
                                      {"role": "assistant", "content": A1}],
         False, "default"),
        # ---- turn ORDER the renderers assume without saying so
        ("consecutive-user", [{"role": "user", "content": U1},
                              {"role": "user", "content": U2}], True,
         "default"),
        ("consecutive-assistant", [{"role": "user", "content": U1},
                                   {"role": "assistant", "content": A1},
                                   {"role": "assistant", "content": "Also, "
                                    "the sky is not actually blue at night."},
                                   {"role": "user", "content": U2}], True,
         "default"),
        # A system message that is not message[0]. Several renderers special-
        # case "the first turn is the system turn" and quietly mis-handle a
        # later one; mistral folds system text into a user turn and llama2
        # wraps it in <<SYS>>, both of which are position-sensitive.
        ("system-mid-history", [{"role": "user", "content": U1},
                                {"role": "assistant", "content": A1},
                                {"role": "system", "content":
                                 "Answer in one sentence from now on."},
                                {"role": "user", "content": U2}], True,
         "default"),
        # ---- shape of the CONTENT
        #
        # The AI-SDK part-array form (Cline, the Vercel AI SDK, and every
        # client that sends images alongside text). src/server.c's
        # message_text() concatenates the text parts with a newline BEFORE
        # anything is rendered, so the reference is asked the equivalent
        # question -- the same conversation with that concatenation already
        # done. What this compares is therefore whether runner's part
        # handling reaches the same prompt as the plain string would: a
        # dropped part, a lost separator or a mangled order all show up.
        ("content-array",
         [{"role": "system", "content": SYS},
          {"role": "user", "content": [
              {"type": "text", "text": "What color is the sky?"},
              {"type": "image_url",
               "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}},
              {"type": "text", "text": "Answer briefly."}]}],
         True, "default",
         [{"role": "system", "content": SYS},
          {"role": "user", "content":
           "What color is the sky?\nAnswer briefly."}]),
        ("empty-assistant-content", [{"role": "user", "content": U1},
                                     {"role": "assistant", "content": ""},
                                     {"role": "user", "content": U2}], True,
         "default"),
        ("empty-user-content", [{"role": "system", "content": SYS},
                                {"role": "user", "content": ""}], True,
         "default"),
        ("unicode-emoji", [{"role": "system", "content": SYS},
                           {"role": "user", "content": UNI},
                           {"role": "assistant", "content": "Ja. " + UNI},
                           {"role": "user", "content": U2}], True, "default"),
        ("long-content", [{"role": "system", "content": SYS},
                          {"role": "user", "content": LONG}], True,
         "default"),
    ]
    if family.thinking_var:
        for mode in ("on", "off"):
            cases.append((f"think-{mode}",
                          [{"role": "system", "content": SYS},
                           {"role": "user", "content": U1}], True, mode))
    if family.tool_family:
        cases.append(("tool-call+result", [
            {"role": "system", "content": SYS},
            {"role": "user", "content": "Weather in Oslo?"},
            {"role": "assistant", "content": "", "tool_calls": [CALL]},
            {"role": "tool", "name": "get_weather", "tool_call_id": "call_1",
             "content": '{"temp_c": -3}'},
        ], True, "default"))
        # Two calls in ONE assistant turn. Muse chains them with an <|eom|>
        # recipient header and Harmony emits a separate turn per call, so this
        # is the case where a renderer edit that assumes "one call per turn"
        # stops being invisible.
        cases.append(("multi-tool-call", [
            {"role": "system", "content": SYS},
            {"role": "user", "content": "Weather in Oslo and Bergen?"},
            {"role": "assistant", "content": "", "tool_calls": [CALL, CALL2]},
            {"role": "tool", "name": "get_weather", "tool_call_id": "call_1",
             "content": '{"temp_c": -3}'},
            {"role": "tool", "name": "get_weather", "tool_call_id": "call_2",
             "content": '{"temp_c": 1}'},
        ], True, "default"))
        # A tool result is not the end of the conversation. Everything after
        # it has to come back to ordinary turns -- the case that catches a
        # renderer leaving a tool block open, or replaying the answer that
        # followed a result in the tool channel.
        cases.append(("tool-then-conversation", [
            {"role": "system", "content": SYS},
            {"role": "user", "content": "Weather in Oslo?"},
            {"role": "assistant", "content": "", "tool_calls": [CALL]},
            {"role": "tool", "name": "get_weather", "tool_call_id": "call_1",
             "content": '{"temp_c": -3}'},
            {"role": "assistant", "content": "It is -3 C in Oslo."},
            {"role": "user", "content": "Should I bring a coat?"},
        ], True, "default"))
    return [c if len(c) == 5 else (c + (None,)) for c in cases]


# ------------------------------------------------- runner-side flattening
#
# There is none, and that is the point.
#
# render_messages_with_tools takes flat {role, content, name, channel} turns,
# not OpenAI messages, and src/server.c:handle_chat() does that flattening --
# content part-arrays, tool_calls, harmony's turn explosion, ornith's tool ->
# user rewrite, the tool-declaration system turn. This harness USED to
# re-implement that in python, which made the tool rows a comparison of jinja
# against a COPY of runner rather than against runner, and which could not
# express a part-array at all.
#
# It now hands scripts/template-conformance-render.c the request body
# verbatim, and that driver #includes src/server.c and runs the real
# handle_chat. Both sides of the diff are fed the same conversation, and the
# only thing between the request and the renderer is runner's own code. See
# the driver's header comment for the single documented substitution
# (add_generation_prompt, which handle_chat hard-codes to true).


def request_body(family, msgs, thinking, tools):
    """The request an OpenAI client would send for this case."""
    req = {"model": "template-conformance", "messages": msgs}
    if tools:
        req["tools"] = tools
    if family.thinking_var and thinking != "default":
        # runner reads this off the request (template.c:req_thinking_mode);
        # the reference takes it as a jinja variable named per family.
        req["enable_thinking"] = (thinking == "on")
    return req


# --------------------------------------------------------- oracle loading

def hf_raw(repo, path, tries=3):
    url = "https://huggingface.co/%s/raw/main/%s" % (repo, path)
    req = urllib.request.Request(url, headers={"User-Agent": "runner-conf"})
    last = None
    for _ in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read().decode("utf-8"), None
        except Exception as e:                                  # noqa: BLE001
            last = e
            # 401/404 are answers, not flakes: a gated or renamed repo will
            # not become reachable on a retry.
            if any(c in str(e) for c in ("401", "403", "404")):
                break
    return None, "%s: %s" % (url, last)


def gguf_kv(path):
    def u32(f):
        return struct.unpack("<I", f.read(4))[0]

    def u64(f):
        return struct.unpack("<Q", f.read(8))[0]

    def rstr(f):
        return f.read(u64(f)).decode("utf-8", "replace")

    def val(f, t):
        if t == 8:
            return rstr(f)
        if t == 9:
            at, n = u32(f), u64(f)
            return [val(f, at) for _ in range(n)]
        fmt = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f",
               7: "<?", 10: "<Q", 11: "<q", 12: "<d"}[t]
        return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]

    with open(path, "rb") as f:
        magic, _ver, _nt, n_kv = struct.unpack("<IIQQ", f.read(24))
        if magic != 0x46554747:
            raise SystemExit("%s is not a GGUF file" % path)
        kv = {}
        for _ in range(n_kv):
            k = rstr(f)
            kv[k] = val(f, u32(f))
    return kv


def fetch_oracle(name, fam, oracle_dir):
    """Write <oracle_dir>/<name>.jinja and <name>.json. Returns (ok, err)."""
    kind, ref = fam.source
    tmpl, meta = None, {}
    if kind == "hf":
        raw, err = hf_raw(ref, "tokenizer_config.json")
        if raw:
            cfg = json.loads(raw)
            t = cfg.get("chat_template")
            if isinstance(t, list):
                t = ([x for x in t if x.get("name") == "default"]
                     or t)[0]["template"]
            tmpl = t
            for k in ("bos_token", "eos_token"):
                v = cfg.get(k)
                meta[k] = v.get("content") if isinstance(v, dict) else v
        if tmpl is None:
            tmpl, err2 = hf_raw(ref, "chat_template.jinja")
            err = err or err2
        if tmpl is None:
            return False, "cannot fetch %s (%s)" % (ref, err)
    else:
        # models/ is gitignored, so the GGUF may live elsewhere:
        # RUNNER_MODELS_DIR overrides, otherwise ROOT/models.
        base = os.environ.get("RUNNER_MODELS_DIR")
        path = (os.path.join(base, os.path.basename(ref)) if base
                else os.path.join(ROOT, ref))
        if not os.path.exists(path):
            return False, ("%s not present. Point RUNNER_MODELS_DIR at the "
                           "model shelf, or run --network-only to check just "
                           "the families whose oracle is fetched over HTTP."
                           % path)
        kv = gguf_kv(path)
        tmpl = kv.get("tokenizer.chat_template")
        toks = kv.get("tokenizer.ggml.tokens") or []
        for k, idk in (("bos_token", "tokenizer.ggml.bos_token_id"),
                       ("eos_token", "tokenizer.ggml.eos_token_id")):
            i = kv.get(idk)
            if isinstance(i, int) and 0 <= i < len(toks):
                meta[k] = toks[i]
    if not tmpl:
        return False, "no chat_template at %s:%s" % (kind, ref)
    os.makedirs(oracle_dir, exist_ok=True)
    with open(os.path.join(oracle_dir, name + ".jinja"), "w") as f:
        f.write(tmpl)
    meta["source"] = "%s:%s" % (kind, ref)
    meta["sha256"] = sha(tmpl)
    meta["fetched_at"] = datetime.datetime.now(
        datetime.timezone.utc).replace(microsecond=0).isoformat()
    with open(os.path.join(oracle_dir, name + ".json"), "w") as f:
        json.dump(meta, f, indent=1)
    return True, None


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ------------------------------------------------------ the harmony oracle
#
# Harmony's reference is a LIBRARY, not a template, so it cannot be cached as
# a .jinja next to the others. Two ways in, in this order:
#
#   1. render live, through any interpreter that has openai_harmony (this
#      one, or RUNNER_HARMONY_PYTHON). Always preferred: a live library
#      cannot go stale.
#   2. the committed fixture scripts/template-conformance-harmony-oracle.json,
#      so a machine without the library still checks harmony instead of
#      skipping it. Regenerate with `make template-conformance-harmony-oracle`.
#
# With neither, harmony is NOT CHECKED -- exit 2, never a pass. And when BOTH
# are present the fixture is verified against the live render, so a fixture
# recorded from an older library version is reported rather than trusted.

# Every subprocess this harness runs is captured as BYTES and decoded here,
# never through `text=True`. Two independent reasons, both found on Windows
# 2026-08-14 while making the gate run there at all:
#
#   * `text=True` returned `stdout is None` -- with `stderr` correctly `''`
#     and returncode 0 -- for a ~31KB `input=` payload on CPython 3.12.10,
#     while the identical call in bytes mode returned all 30604 bytes. Small
#     payloads were fine, which is exactly the kind of size-dependent fault
#     that would have read as "Windows is flaky" forever.
#   * `text=True` decodes with the LOCALE encoding. The box is Swedish, so
#     that is cp1252, and this gate compares prompts BYTE-EXACTLY -- the
#     unicode-emoji and typographic-quote cases would have come back
#     mojibake and diffed against a reference that was never wrong. A
#     byte-exact comparison has no business going through a locale codec.
def _run_captured(argv, payload=None):
    """(returncode, stdout_text, stderr_text). UTF-8 in, UTF-8 out."""
    p = subprocess.run(argv,
                       input=payload.encode("utf-8") if payload is not None
                             else None,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return (p.returncode,
            (p.stdout or b"").decode("utf-8", "replace"),
            (p.stderr or b"").decode("utf-8", "replace"))


def harmony_render_live(cases):
    """(doc, err). doc is {"library_version":..., "renders": {...}}."""
    payload = json.dumps({"cases": cases})
    tried = []
    for py in (os.environ.get("RUNNER_HARMONY_PYTHON"), sys.executable):
        if not py:
            continue
        rc, out, errtext = _run_captured([py, HARMONY_SCRIPT], payload)
        if rc == 0:
            return json.loads(out), None
        tried.append("%s: %s" % (py, errtext.strip().splitlines()[-1]
                                 if errtext.strip() else "exit %d" % rc))
    return None, "; ".join(tried)


def harmony_oracle(cases):
    """(renders, meta, rot, err) for the harmony family."""
    live, lerr = harmony_render_live(cases)
    fixture = None
    if os.path.exists(HARMONY_FIXTURE):
        with open(HARMONY_FIXTURE) as f:
            fixture = json.load(f)

    def meta_for(doc, how):
        return {"source": "%s:openai-harmony %s"
                          % (how, doc.get("library_version", "?")),
                "sha256": sha(json.dumps(doc["renders"], sort_keys=True)),
                "fetched_at": doc.get("generated_at", "live")}

    if live:
        rot = []
        if fixture:
            stale = [c for c, t in live["renders"].items()
                     if fixture.get("renders", {}).get(c) != t]
            if fixture.get("library_version") != live["library_version"]:
                rot.append(
                    "STALE HARMONY ORACLE: the committed fixture was recorded "
                    "from openai-harmony %s, the installed library is %s. "
                    "Regenerate: make template-conformance-harmony-oracle"
                    % (fixture.get("library_version"),
                       live["library_version"]))
            elif stale:
                rot.append(
                    "STALE HARMONY ORACLE: %d case(s) render differently from "
                    "the committed fixture (%s). The fixture is what a machine "
                    "without the library compares against, so it must not "
                    "drift. Regenerate: make template-conformance-harmony-"
                    "oracle" % (len(stale), ", ".join(sorted(stale)[:4])))
        return live["renders"], meta_for(live, "lib"), rot, None
    if fixture:
        return fixture["renders"], meta_for(fixture, "fixture"), [], None
    return None, None, [], (
        "the harmony oracle is the openai-harmony library, and neither the "
        "library nor the committed fixture %s is available (%s). Install it "
        "(pip install openai-harmony), point RUNNER_HARMONY_PYTHON at an "
        "interpreter that has it, or restore the fixture."
        % (os.path.relpath(HARMONY_FIXTURE, ROOT), lerr))


def harmony_cases(fam):
    """The matrix as the harmony oracle script wants it."""
    out = []
    for cid, msgs, add_gen, _thinking, omsgs in matrix(fam):
        out.append({"id": cid,
                    "messages": omsgs if omsgs is not None else msgs,
                    "add_generation_prompt": add_gen,
                    "tools": TOOLS if "tool" in cid else None})
    return out


def write_harmony_fixture():
    """Re-record the committed harmony oracle from the installed library."""
    fam = FAMILIES["harmony"]
    doc, err = harmony_render_live(harmony_cases(fam))
    if not doc:
        print("REFUSING to write the harmony oracle fixture: the "
              "openai-harmony library is not importable (%s)." % err)
        print("The fixture is a RECORDING of the library. Writing one without "
              "it would invent the oracle it is supposed to preserve.")
        return EXIT_NOT_CHECKED
    doc = {
        "_README": [
            "The HARMONY oracle, rendered from the openai-harmony library.",
            "",
            "gpt-oss's wire format is defined by that library and its",
            "docs/format.md, not by the jinja chat_template in the GGUF --",
            "the jinja is a reimplementation with gaps it cannot close (no",
            "content_type field, so it can never emit <|constrain|>).",
            "",
            "This file exists so a machine WITHOUT the library still checks",
            "harmony rather than skipping it. When the library IS installed",
            "the gate renders live and verifies this file still matches; a",
            "mismatch is reported, not accepted.",
            "",
            "Regenerate (needs the library):",
            "    make template-conformance-harmony-oracle",
            "Adding a case to the matrix requires regenerating this file, or",
            "that case reports NOT CHECKED on a machine without the library.",
        ],
        "library_version": doc["library_version"],
        "generated_by": "scripts/template-conformance.py "
                        "--write-harmony-oracle",
        "generated_at": datetime.datetime.now(
            datetime.timezone.utc).replace(microsecond=0).isoformat(),
        "renders": doc["renders"],
    }
    with open(HARMONY_FIXTURE, "w") as f:
        json.dump(doc, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print("wrote %s: %d cases from openai-harmony %s"
          % (os.path.relpath(HARMONY_FIXTURE, ROOT), len(doc["renders"]),
             doc["library_version"]))
    return EXIT_OK


# ------------------------------------------------------- jinja (reference)

def jinja_env():
    import jinja2.ext
    from jinja2.exceptions import TemplateError
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    # transformers' _compile_jinja_template, reproduced: trim_blocks and
    # lstrip_blocks are ON there, and getting them wrong manufactures fake
    # whitespace drift in every template that indents its {% %} tags.
    def raise_exception(message):
        raise TemplateError(message)

    def tojson(x, ensure_ascii=False, indent=None, separators=None,
               sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent,
                          separators=separators, sort_keys=sort_keys)

    def strftime_now(fmt):
        return datetime.datetime.now().strftime(fmt)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                        extensions=[jinja2.ext.loopcontrols])
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = strftime_now
    return env


def render_reference(env, tmpl_src, meta, msgs, add_gen, thinking,
                     thinking_var, tools):
    kwargs = dict(messages=msgs, add_generation_prompt=add_gen,
                  bos_token=meta.get("bos_token") or "",
                  eos_token=meta.get("eos_token") or "")
    if tools:
        kwargs["tools"] = tools
    if thinking_var and thinking != "default":
        kwargs[thinking_var] = (thinking == "on")
    return env.from_string(tmpl_src).render(**kwargs)


# ------------------------------------------------------------ runner side

def build_renderer():
    """Build the runner-side driver. Returns (path, err).

    Through `make`, not a hand-rolled cc line: the driver #includes
    src/server.c and therefore needs the same object set, CFLAGS and platform
    LDFLAGS the server itself is built with (Metal on macOS, winsock on
    Windows). Reproducing that here is exactly the kind of second copy of
    runner's build that this harness just finished deleting from its render
    path.
    """
    exe = "template-conformance-render"
    if sys.platform.startswith("win"):
        exe += ".exe"
    out = os.path.join(ROOT, exe)
    rc, mkout, mkerr = _run_captured(["make", "-C", ROOT, exe])
    if rc or not os.path.exists(out):
        return None, ("cannot build the runner-side renderer (make %s):\n%s%s"
                      % (exe, mkout[-2000:], mkerr[-2000:]))
    return out, None


def run_renderer(binary, payload):
    rc, out, err = _run_captured([binary], json.dumps(payload))
    if rc:
        raise SystemExit("runner renderer failed: %s" % err)
    if not out:
        # Distinguish "the driver printed nothing" from "the JSON was bad",
        # because those have completely different causes and the bare
        # json.loads TypeError named neither.
        raise SystemExit("runner renderer produced no output (exit 0). "
                         "stderr:\n%s" % (err[-2000:] or "(empty)"))
    return json.loads(out)


def render_runner(binary, cases):
    """id -> (prompt, error). error is None on a successful render."""
    doc = run_renderer(binary, {"cases": cases})
    return {r["id"]: (r["prompt"], r.get("error")) for r in doc["results"]}


# --------------------------------------------------------- the token check
#
# The gate compares TEXT, and text is not what the model reads. Whitespace
# that looks cosmetic in a diff is not cosmetic to a tokenizer: `[INST] X
# [/INST]` and `[INST] X[/INST]` are different SentencePiece sequences, and
# that difference -- one extra token per occurrence on
# Mistral-7B-Instruct-v0.3, measured on hardware -- is exactly the defect the
# first family in the fix campaign has. A text-only gate cannot tell whether a
# renderer fix changed what the model actually sees, or only how the diff
# reads.
#
# So the check is ALWAYS ON wherever it can run, rather than a mode or a
# per-family opt-in:
#
#   * a separate mode would not be running during the campaign it exists to
#     protect, and
#   * a per-family opt-in makes "nobody enabled it here" indistinguishable
#     from "it passed", which is the failure this whole script is against.
#
# It runs for any family with a tokenizer on the shelf, needs no flag, and
# reports THREE states per case exactly as the text check does: agreed,
# differed, or NOT CHECKED because no tokenizer for that family was found.
# NOT CHECKED is never folded into the pass -- it is counted, printed, and
# named in the final line. `--require-tokens` promotes it to exit 2, which is
# what the per-family fix runs should use.

def tokenizer_path(fam):
    """Absolute path to a GGUF carrying this family's tokenizer, or None."""
    base = os.environ.get("RUNNER_MODELS_DIR") or os.path.join(ROOT, "models")
    for name in fam.tokenizer:
        p = os.path.join(base, name)
        if os.path.exists(p):
            return p
    return None


def token_check(binary, jobs):
    """jobs: [(id, gguf, want_text, mine_text)] -> id -> (verdict, detail)."""
    if not jobs:
        return {}
    payload = []
    for cid, gguf, want, mine in jobs:
        payload.append({"id": cid + " #ref", "gguf": gguf, "text": want})
        payload.append({"id": cid + " #run", "gguf": gguf, "text": mine})
    doc = run_renderer(binary, {"tokenize": payload})
    got = {t["id"]: t for t in doc["tokens"]}
    out = {}
    for cid, _g, _w, _m in jobs:
        ref, run = got.get(cid + " #ref"), got.get(cid + " #run")
        if not ref or not run or ref["ids"] is None or run["ids"] is None:
            why = ((ref or {}).get("error") or (run or {}).get("error")
                   or "the driver returned no ids")
            out[cid] = ("NOT-CHECKED", why)
            continue
        a, b = ref["ids"], run["ids"]
        if a == b:
            out[cid] = ("same", {"n": len(a), "delta": 0})
        else:
            out[cid] = ("differ", {"n": len(a), "delta": len(b) - len(a),
                                   "at": first_diff_at(a, b)})
    return out


# ----------------------------------------------------------- the allowlist
#
# Deviations runner takes ON PURPOSE. Each `patch` rewrites the ORACLE text
# into what runner would legitimately produce, so what survives into the diff
# is unexplained. Loaded from JSON so the policy is one reviewable file rather
# than lambdas buried in a script, and so a citation can be re-verified.

class Deviation:
    def __init__(self, raw):
        self.id = raw["id"]
        self.family = raw["family"]
        self.why = raw.get("why", "")
        self.cites = [c for c in (raw.get("cite"),
                                  raw.get("corroborating_cite")) if c]
        self.patch_spec = raw["patch"]
        self.used = 0

    def key(self):
        return self.id.split("/", 1)[-1]


def compile_patch(dev, bos):
    """Turn a declarative patch spec into oracle_text -> oracle_text."""
    spec = dev.patch_spec
    op = spec.get("op")
    if op == "strip-bos":
        b = bos or "\0"
        return lambda s: s[len(b):] if s.startswith(b) else s
    if op == "drop-first-match":
        rx = re.compile(spec["pattern"],
                        re.S if spec.get("dotall") else 0)
        return lambda s: rx.sub("", s, count=1)
    if op == "builtin":
        fn = BUILTIN_PATCHES.get(spec["name"])
        if not fn:
            raise SystemExit("allowlist %s: unknown builtin %r"
                             % (dev.id, spec["name"]))
        return fn
    raise SystemExit("allowlist %s: unknown patch op %r" % (dev.id, op))


# harmony's generation prompt is the one deviation that is conditional on the
# rendered text rather than a fixed rewrite: runner primes the analysis
# channel only when there are NO tools (with tools it leaves the header bare
# so the constrained recipient stays reachable). "namespace functions" is how
# the harmony reference spells a tool declaration block.
def _harmony_prime_analysis(s):
    if s.endswith("<|start|>assistant") and "namespace functions" not in s:
        return s + "<|channel|>analysis<|message|>"
    return s


BUILTIN_PATCHES = {"harmony-prime-analysis": _harmony_prime_analysis}


def load_allowlist(path):
    with open(path) as f:
        doc = json.load(f)
    devs = {}
    seen = set()
    for raw in doc["deviations"]:
        d = Deviation(raw)
        if d.id in seen:
            raise SystemExit("allowlist: duplicate id %s" % d.id)
        seen.add(d.id)
        devs.setdefault(d.family, []).append(d)
    return devs


def check_citations(devs, families):
    """Re-verify every citation. Returns a list of rot messages."""
    rot = []
    cache = {}
    for fam in families:
        for d in devs.get(fam, []):
            if not d.why.strip():
                rot.append("%s: no `why` -- an unexplained deviation is a bug,"
                           " not an allowlist entry" % d.id)
            if not d.cites:
                rot.append("%s: no `cite` -- an entry that cannot cite a "
                           "reason is an unfixed bug, move it to the baseline"
                           % d.id)
            for c in d.cites:
                rot += check_one_citation(d.id, c, cache)
    return rot


def check_one_citation(did, c, cache):
    path = os.path.join(ROOT, c["file"])
    if path not in cache:
        try:
            with open(path, encoding="utf-8") as f:
                cache[path] = f.read().splitlines()
        except OSError:
            cache[path] = None
    lines = cache[path]
    if lines is None:
        return ["%s: cited file %s does not exist" % (did, c["file"])]
    m = re.match(r"^(\d+)(?:-(\d+))?$", str(c.get("lines", "")))
    if not m:
        return ["%s: cite.lines %r is not N or N-M"
                % (did, c.get("lines"))]
    lo = int(m.group(1))
    hi = int(m.group(2) or lo)
    anchor = c.get("anchor")
    if not anchor:
        return ["%s: cite has no `anchor`; a bare line number cannot be "
                "re-verified and rots on the next edit" % did]
    if lo < 1 or hi > len(lines):
        return ["%s: %s:%s is past the end of the file (%d lines)"
                % (did, c["file"], c["lines"], len(lines))]
    if anchor in "\n".join(lines[lo - 1:hi]):
        return []
    # STALE vs DEAD: say which, and where it went.
    hits = [i + 1 for i, ln in enumerate(lines) if anchor in ln]
    if hits:
        return ["STALE CITATION %s: anchor %r is no longer in %s:%s -- it is "
                "now at line %s. Update cite.lines."
                % (did, anchor, c["file"], c["lines"],
                   ",".join(map(str, hits)))]
    return ["DEAD CITATION %s: anchor %r is gone from %s entirely. The reason "
            "this deviation was allowed no longer exists in the source; "
            "re-justify it or treat the difference as a bug."
            % (did, anchor, c["file"])]


# ------------------------------------------------------------- the baseline
#
# Differences that are KNOWN BUGS. Not permission -- a backlog. Each entry
# carries a status and an excerpt so the file reads as a bug list, and a
# digest so a difference that CHANGES SHAPE is not mistaken for the one that
# was signed off.

def diff_digest(want, mine):
    return sha(_denow(want) + "\x00" + _denow(mine))


def _denow(s):
    """Neutralise today's date so a backlog entry does not expire at midnight.

    Only TODAY's date in the formats the reference templates actually use is
    substituted -- a general \\d{4}-\\d{2}-\\d{2} scrub would also hide a
    hard-coded date, which is exactly the kind of drift this gate is for.
    """
    now = datetime.datetime.now()
    for fmt in ("%Y-%m-%d", "%d %b %Y", "%d %B %Y", "%b %d, %Y", "%B %Y"):
        s = s.replace(now.strftime(fmt), "<TODAY>")
    return s


def first_diff_at(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def excerpt(s, at, width=90):
    lo = max(0, at - 20)
    return repr(s[lo:lo + width])


def load_baseline(path):
    if not os.path.exists(path):
        return {"version": 1, "oracles": {}, "known_differences": []}
    with open(path) as f:
        return json.load(f)


# ------------------------------------------------------------------ main

def run(args):
    names = args.family or list(FAMILIES)
    for n in names:
        if n not in FAMILIES:
            raise SystemExit("unknown family %r (have: %s)"
                             % (n, ", ".join(FAMILIES)))

    devs = load_allowlist(args.allowlist)
    base = load_baseline(args.baseline)
    known = {e["id"]: e for e in base.get("known_differences", [])}
    base_oracles = base.get("oracles", {})

    # ---- oracle side must actually be available, or nothing is "checked"
    not_checked = []
    try:
        env = jinja_env()
    except ImportError:
        print("NOT CHECKED: jinja2 is not installed, so no reference render "
              "could be produced.")
        print("             install it:  %s -m pip install jinja2"
              % os.path.basename(sys.executable))
        print("             (this is NOT a pass: nothing was compared)")
        return EXIT_NOT_CHECKED

    binary, err = build_renderer()
    if err:
        print("NOT CHECKED: %s" % err)
        return EXIT_NOT_CHECKED

    results = []        # (family, case, verdict, detail)
    oracle_meta = {}
    structural = []     # families with no upstream template BY DESIGN
    stale = []          # --refresh failed but a cached oracle was usable
    rot_skips = []      # Family.cannot entries that stopped being true
    rot_extra = []      # oracle-specific rot (the harmony fixture)
    rebased = []        # caches invalidated because the family's source moved
    tokens = {}         # cid_full -> (verdict, detail)
    tok_sources = {}    # family -> the GGUF its tokenizer came from
    tok_missing = {}    # family -> why the token check could not run
    new_drift = changed = fixed = 0
    checked = 0

    for n in names:
        fam = FAMILIES[n]
        if fam.skip:
            # Not an availability problem: there is nothing upstream to
            # conform to. Reported, but it does not make the run incomplete.
            structural.append((n, fam.skip))
            continue
        cases = matrix(fam)

        # ---- harmony: the oracle is a LIBRARY, not a cached template
        harmony_renders = None
        tmpl_src, meta = None, {}
        if fam.source[0] == "harmony":
            harmony_renders, hmeta, hrot, herr = harmony_oracle(
                harmony_cases(fam))
            if herr:
                not_checked.append((n, herr))
                continue
            oracle_meta[n] = hmeta
            rot_extra += hrot
        else:
            jpath = os.path.join(args.oracle_dir, n + ".jinja")
            mpath = os.path.join(args.oracle_dir, n + ".json")
            # The cache is keyed on the SOURCE, not just the family name.
            #
            # Without this, changing where a family's oracle comes from --
            # a different HF repo, a different GGUF, or (as happened here)
            # switching harmony from its jinja to the openai-harmony
            # library -- silently reuses the old basis, and the run goes
            # green against an oracle the configuration no longer names.
            # A cached artifact that outlives the reason it was cached is
            # the same class of bug as every other one this gate exists to
            # catch, so the mismatch invalidates rather than warns.
            want_source = "%s:%s" % fam.source
            cached = os.path.exists(jpath) and os.path.exists(mpath)
            if cached:
                try:
                    with open(mpath) as f:
                        cached_source = json.load(f).get("source")
                except (OSError, ValueError):
                    cached_source = None
                if cached_source != want_source:
                    rebased.append(
                        "%s: cached oracle was fetched from %s, the family now "
                        "names %s. Re-fetching; the cached copy cannot answer "
                        "for a source it did not come from."
                        % (n, cached_source or "an unreadable meta file",
                           want_source))
                    cached = False
            if args.refresh or not cached:
                ok, ferr = fetch_oracle(n, fam, args.oracle_dir)
                if not ok and not cached:
                    not_checked.append((n, ferr))
                    continue
                if not ok:
                    # A refresh that failed with a usable cache still checks
                    # something; say loudly that the oracle may be stale
                    # rather than throwing away a comparison we can make.
                    stale.append((n, ferr))
            tmpl_src = open(jpath).read()
            with open(mpath) as f:
                meta = json.load(f)
            oracle_meta[n] = {"source": meta.get("source", "?"),
                              "sha256": meta.get("sha256") or sha(tmpl_src),
                              "fetched_at": meta.get("fetched_at", "?")}

        fam_devs = devs.get(n, [])
        patches = [(d, compile_patch(d, meta.get("bos_token")))
                   for d in fam_devs]

        jobs = [{"id": cid, "template": fam.runner,
                 "add_generation_prompt": add_gen,
                 "request": request_body(fam, msgs, thinking,
                                         TOOLS if "tool" in cid else None)}
                for cid, msgs, add_gen, thinking, _om in cases]
        got = render_runner(binary, jobs)

        tok_gguf = tokenizer_path(fam)
        if tok_gguf:
            tok_sources[n] = tok_gguf
        else:
            tok_missing[n] = (
                "no GGUF carrying the %s tokenizer on the model shelf "
                "(looked for %s in %s). The text comparison ran; the TOKEN "
                "comparison did not." % (
                    n, ", ".join(fam.tokenizer) or "nothing -- no tokenizer "
                    "is declared for this family",
                    os.environ.get("RUNNER_MODELS_DIR")
                    or os.path.join(ROOT, "models")))
        tok_jobs = []

        for cid, msgs, add_gen, thinking, omsgs_override in cases:
            cid_full = "%s/%s" % (n, cid)
            tools = TOOLS if "tool" in cid else None
            declined = fam.cannot.get(cid)
            omsgs = omsgs_override if omsgs_override is not None else msgs
            if tools and fam.oracle_tool_shape == "flat-string":
                # apertus: declarations flattened out of the OpenAI wrapper,
                # arguments left as the wire's JSON string. Both are what its
                # template reads; feeding it the common shape raises rather
                # than rendering, which is not a conformance answer.
                tools = [t.get("function", t) for t in tools]
            elif tools:
                # Reference templates assume tool_call.arguments is a MAPPING:
                # muse raises outright on a string, harmony/qwen/granite run
                # it through |tojson and would double-encode one. OpenAI wire
                # format is a JSON string, which is what runner is handed.
                # Parsing it for the oracle side compares the two renderers on
                # the same call rather than on an input the template rejects.
                omsgs = json.loads(json.dumps(omsgs))
                for m in omsgs:
                    for c in m.get("tool_calls") or []:
                        c["function"]["arguments"] = json.loads(
                            c["function"]["arguments"])
            mine, mine_err = got[cid]

            refused = None
            want = None
            if harmony_renders is not None:
                want = harmony_renders.get(cid)
                if want is None:
                    # A case the committed fixture predates. Never silently
                    # skipped: with no library to render it live, this case
                    # was not compared, and that is exit 2.
                    refused = ("reference",
                               "no harmony oracle for this case. The library "
                               "is not installed and the committed fixture "
                               "predates the case. Regenerate: make "
                               "template-conformance-harmony-oracle")
            else:
                try:
                    want = render_reference(env, tmpl_src, meta, omsgs,
                                            add_gen, thinking,
                                            fam.thinking_var, tools)
                except Exception as e:                      # noqa: BLE001
                    refused = ("reference", "%s: %s" % (type(e).__name__, e))
            if refused is None and mine_err:
                refused = ("runner", mine_err)

            # ---- a case this family genuinely cannot express
            #
            # A declared skip is re-verified, never assumed: if the side that
            # was supposed to refuse now accepts the case, the declaration is
            # stale and the gate says so rather than silently dropping a case
            # that has become checkable.
            if declined:
                want_side, why = declined
                if refused is None:
                    rot_skips.append(
                        "STALE SKIP %s: declared as unexpressible (%s refuses: "
                        "%s) but BOTH sides rendered it. The case is now "
                        "checkable -- delete the Family.cannot entry."
                        % (cid_full, want_side, why))
                elif refused[0] != want_side:
                    rot_skips.append(
                        "STALE SKIP %s: declared as refused by the %s side, "
                        "but it was the %s side that refused (%s). Re-check "
                        "the reason."
                        % (cid_full, want_side, refused[0], refused[1]))
                results.append((n, cid, "cannot-express",
                                "%s refuses: %s" % (want_side, why)))
                continue

            if refused is not None:
                # Not a pass and not a difference: this case was never
                # compared. If it is genuinely inexpressible for this family,
                # declare it in Family.cannot with the reason; until then it
                # keeps the run incomplete, which is exit 2.
                not_checked.append((cid_full, "%s refused: %s"
                                    % (refused[0], refused[1])))
                results.append((n, cid, "NOT-CHECKED",
                                "%s refused" % refused[0]))
                continue
            checked += 1

            applied = []
            if not args.raw:
                for d, patch in patches:
                    patched = patch(want)
                    if patched != want:
                        applied.append(d.key())
                        d.used += 1
                        want = patched
            entry = known.get(cid_full)
            if tok_gguf:
                tok_jobs.append((cid_full, tok_gguf, want, mine))

            if mine == want:
                if entry and not args.write_baseline:
                    fixed += 1
                    results.append((n, cid, "FIXED",
                                    "was %s in the baseline; the backlog "
                                    "shrank -- re-record it" % entry.get(
                                        "status", "open")))
                else:
                    tag = ("ok" + (" (-%s)" % ",".join(applied)
                                   if applied else ""))
                    results.append((n, cid, "ok", tag))
                continue

            at = first_diff_at(want, mine)
            dig = diff_digest(want, mine)
            detail = {"applied": applied, "at": at, "digest": dig,
                      "reference_excerpt": excerpt(want, at),
                      "runner_excerpt": excerpt(mine, at)}
            if entry is None:
                new_drift += 1
                results.append((n, cid, "NEW-DRIFT", detail))
            elif entry.get("digest") != dig:
                changed += 1
                results.append((n, cid, "CHANGED", detail))
            else:
                detail["status"] = entry.get("status", "open")
                results.append((n, cid, "known", detail))
            if args.show:
                detail["show"] = (want, mine)

        # ---- token-level, for the cases that produced two renders
        tokens.update(token_check(binary, tok_jobs))

    # ---- allowlist rot
    #
    # Citations are checked for every SELECTED family, oracle or not: they are
    # a static property of the tree, so CI catches a citation that rotted
    # under a family it cannot fetch. UNUSED needs an actual run, so it is
    # only asserted for families that produced results.
    rot = check_citations(devs, names)
    for n in names:
        for d in devs.get(n, []):
            if n in oracle_meta and not d.used and not args.raw:
                rot.append("UNUSED %s: the patch never fired. Runner no "
                           "longer takes this deviation (or the reference "
                           "changed) -- delete the entry." % d.id)

    for n, ferr in stale:
        rot.append("STALE ORACLE %s: --refresh could not reach the upstream "
                   "template, so the comparison used the cached copy "
                   "(fetched %s). %s"
                   % (n, oracle_meta.get(n, {}).get("fetched_at", "?"), ferr))
    # A declared per-family skip that stopped being true is the same class of
    # problem as a rotted allowlist citation: the gate's own premise moved.
    # So is a committed harmony fixture that no longer matches the library.
    rot += rot_skips + rot_extra

    # ---- upstream oracle movement
    moved = []
    for n, om in oracle_meta.items():
        was = base_oracles.get(n)
        if was and was.get("sha256") != om["sha256"]:
            moved.append("%s: upstream template changed (%s -> %s, %s). The "
                         "comparison basis moved; re-review, then re-record."
                         % (n, was.get("sha256"), om["sha256"], om["source"]))

    if args.write_baseline:
        if not_checked:
            print("REFUSING to re-record the baseline: %d families/cases were "
                  "not compared." % len(not_checked))
            for what, why in not_checked:
                print("  %-16s %s" % (what, why))
            print("A baseline written from a partial run would silently drop "
                  "the backlog of everything it could not check.")
            return EXIT_NOT_CHECKED
        return write_baseline(args, results, oracle_meta, base, known,
                              set(oracle_meta), tokens)

    return report(args, results, not_checked, structural, rot, moved,
                  oracle_meta, checked, new_drift, changed, fixed,
                  tokens, tok_sources, tok_missing, known, rebased)


def token_line(tokens, cid_full):
    v = tokens.get(cid_full)
    if not v:
        return ""
    verdict, d = v
    if verdict == "same":
        return "  tokens=%d same" % d["n"]
    if verdict == "differ":
        return "  tokens %+d (ref %d, first differs at %d)" % (
            d["delta"], d["n"], d["at"])
    return "  tokens NOT-CHECKED"


def report(args, results, not_checked, structural, rot, moved, oracle_meta,
           checked, new_drift, changed, fixed, tokens, tok_sources,
           tok_missing, known, rebased):
    for r in rebased:
        print("ORACLE CACHE INVALIDATED  %s" % r)
    per_family = {}
    tok_drift = []
    for n, cid, verdict, detail in results:
        cid_full = "%s/%s" % (n, cid)
        per_family.setdefault(n, []).append((cid, verdict, detail))
        if verdict == "ok":
            line = detail
        elif verdict in ("NEW-DRIFT", "CHANGED", "known"):
            line = "%s%s at byte %d%s" % (
                verdict,
                " [%s]" % detail["status"] if verdict == "known" else "",
                detail["at"],
                " (-%s applied)" % ",".join(detail["applied"])
                if detail["applied"] else "")
        else:
            line = "%s  %s" % (verdict, detail)
        # Identical text whose TOKENS differ would mean the tokenizer is not a
        # function of the text -- report it rather than trust the byte compare.
        if verdict == "ok" and tokens.get(cid_full, ("", ))[0] == "differ":
            tok_drift.append("%s: the two renders are byte-identical but "
                             "tokenize differently. That should be "
                             "impossible; the tokenizer or this harness is "
                             "wrong." % cid_full)
        # A token delta on a KNOWN difference is the number the fix campaign
        # is actually chasing, so it rides on the case line.
        line = "%s%s" % (line, token_line(tokens, cid_full)
                         if verdict != "ok" else "")
        print("  %-13s %-26s %s" % (n, cid, line))
        if isinstance(detail, dict) and detail.get("show"):
            print("      reference: %s" % repr(detail["show"][0]))
            print("      runner   : %s" % repr(detail["show"][1]))
        elif isinstance(detail, dict) and verdict in ("NEW-DRIFT", "CHANGED"):
            # a `known` entry's excerpt is already in the baseline file; only
            # the differences that need a decision get printed here.
            print("      reference: %s" % detail["reference_excerpt"])
            print("      runner   : %s" % detail["runner_excerpt"])

    print()
    for n in per_family:
        bad = [v for _c, v, _d in per_family[n]
               if v in ("NEW-DRIFT", "CHANGED", "FIXED")]
        kn = [v for _c, v, _d in per_family[n] if v == "known"]
        cant = [v for _c, v, _d in per_family[n] if v == "cannot-express"]
        state = ("CLEAN" if not bad and not kn else
                 "%d known" % len(kn) if not bad else
                 "%d NEEDS ATTENTION, %d known" % (len(bad), len(kn)))
        if cant:
            state += " (%d n/a)" % len(cant)
        tk = ("tok:%s" % os.path.basename(tok_sources[n])
              if n in tok_sources else "tok:NOT-CHECKED")
        print("%-15s %-30s [%s %s] %s"
              % (n, state, oracle_meta[n]["source"], oracle_meta[n]["sha256"],
                 tk))
    for n, why in structural:
        print("%-15s %-30s %s" % (n, "no upstream oracle", why))

    known_total = sum(1 for _n, _c, v, _d in results if v == "known")
    cannot_total = sum(1 for _n, _c, v, _d in results if v == "cannot-express")
    print("\n%d cases compared, %d known-bug differences (backlog), "
          "%d new, %d changed, %d fixed-but-still-listed, "
          "%d not expressible by the family"
          % (checked, known_total, new_drift, changed, fixed, cannot_total))

    tok_same = sum(1 for v in tokens.values() if v[0] == "same")
    tok_diff = sum(1 for v in tokens.values() if v[0] == "differ")
    tok_none = checked - tok_same - tok_diff
    print("token-level: %d cases checked (%d identical, %d differing), "
          "%d NOT CHECKED for want of a tokenizer"
          % (tok_same + tok_diff, tok_same, tok_diff, tok_none))
    # The number the fix campaign is chasing: how much of the backlog is
    # costing real tokens rather than only reading badly in a diff.
    costly = [(cid, d["delta"]) for cid, (v, d) in sorted(tokens.items())
              if v == "differ" and d["delta"]]
    if costly:
        print("             %d differing case(s) change the PROMPT LENGTH: %s"
              % (len(costly), ", ".join("%s %+d" % c for c in costly[:6])
                 + (" ..." if len(costly) > 6 else "")))

    bad = False
    if not_checked:
        bad = True
        print("\nNOT CHECKED (%d) -- these produced no comparison at all:"
              % len(not_checked))
        for what, why in not_checked:
            print("  %-24s %s" % (what, why))
    if tok_missing:
        print("\nTOKENS NOT CHECKED (%d families) -- the text was compared, "
              "the token sequence was NOT:" % len(tok_missing))
        for n, why in sorted(tok_missing.items()):
            print("  %-13s %s" % (n, why))
        print("  A whitespace-only text difference can still change the token "
              "sequence the model reads, so this is a real gap, not a "
              "formality. --require-tokens turns it into exit 2.")
    if tok_drift:
        bad = True
        print("\nTOKENISER DISAGREES WITH THE BYTES (%d):" % len(tok_drift))
        for t in tok_drift:
            print("  " + t)
    if rot:
        bad = True
        print("\nALLOWLIST / SKIP ROT (%d) -- scripts/template-conformance-"
              "allowlist.json and Family.cannot:" % len(rot))
        for r in rot:
            print("  " + r)
    if moved:
        bad = True
        print("\nORACLE MOVED (%d):" % len(moved))
        for m in moved:
            print("  " + m)

    # A recorded token delta that MOVED is drift the byte digest cannot see on
    # its own only when the text digest also moved -- but a delta that appears
    # or vanishes under an unchanged digest would mean the tokenizer changed
    # under us, so it is reported either way.
    tok_moved = []
    for cid_full, entry in sorted(known.items()):
        v = tokens.get(cid_full)
        if not v or v[0] != "differ" or "token_delta" not in entry:
            continue
        if entry["token_delta"] != v[1]["delta"]:
            tok_moved.append(
                "%s: recorded token delta %+d, measured %+d. The renderer's "
                "TOKEN cost changed even though the byte difference did not."
                % (cid_full, entry["token_delta"], v[1]["delta"]))
    if tok_moved:
        bad = True
        print("\nTOKEN COST MOVED (%d):" % len(tok_moved))
        for t in tok_moved:
            print("  " + t)

    if new_drift or changed or fixed:
        print("\nFAIL: %d new, %d changed, %d fixed-but-still-listed."
              % (new_drift, changed, fixed))
        print("      New/changed differences are drift: fix the renderer, or "
              "-- if the deviation is deliberate and citable -- add it to "
              "the allowlist.")
        print("      Once you have decided, re-record the backlog:")
        print("        make template-conformance-baseline")
        return EXIT_DRIFT
    if bad and (rot or moved or tok_drift or tok_moved):
        print("\nFAIL: the allowlist, the skips, the oracle pins or the token "
              "costs no longer match the tree. Nothing drifted, but the "
              "gate's own premises did.")
        return EXIT_DRIFT
    if not_checked:
        print("\nNOT CHECKED: %d families/cases were not compared. This is "
              "not a pass." % len(not_checked))
        return EXIT_NOT_CHECKED
    if tok_missing and args.require_tokens:
        print("\nNOT CHECKED: --require-tokens, and %d families had no "
              "tokenizer. This is not a pass." % len(tok_missing))
        return EXIT_NOT_CHECKED
    print("\nOK: every compared case is conformant or an accounted-for "
          "deviation; %d known bugs remain in the backlog." % known_total)
    if tok_missing:
        # Named, not counted. A reader must not be able to infer token
        # conformance from a green run that never tokenized -- and mistral,
        # whose defect IS tokenization, is usually on this list.
        print("    TEXT ONLY for %d of %d families -- no tokenizer was "
              "available, so the token sequence the model reads was NOT "
              "compared for:" % (len(tok_missing), len(per_family)))
        print("      %s" % ", ".join(sorted(tok_missing)))
        print("    Those families are conformant in BYTES. Whether they are "
              "conformant in TOKENS is unmeasured here.")
    return EXIT_OK


def write_baseline(args, results, oracle_meta, old, known, ran, tokens):
    # Entries for families this run did not cover are carried over verbatim.
    # Regenerating from `--family harmony` must not delete everybody else's
    # backlog.
    entries = [e for e in old.get("known_differences", [])
               if e["id"].split("/", 1)[0] not in ran]
    for n, cid, verdict, detail in results:
        if verdict not in ("NEW-DRIFT", "CHANGED", "known"):
            continue
        cid_full = "%s/%s" % (n, cid)
        prev = known.get(cid_full, {})
        e = {
            "id": cid_full,
            "status": prev.get("status", "open"),
            "note": prev.get("note", ""),
            "digest": detail["digest"],
            "allowlist_applied": detail["applied"],
            "first_diff_at": detail["at"],
            "reference_excerpt": detail["reference_excerpt"],
            "runner_excerpt": detail["runner_excerpt"],
        }
        # What this difference COSTS the model, when a tokenizer was
        # available: the delta in prompt tokens between the reference render
        # and runner's. 0 means the difference is genuinely cosmetic; anything
        # else means the model is reading a different sequence. A recorded
        # delta that later moves is reported as TOKEN COST MOVED, so a
        # renderer edit cannot change the cost while leaving the bytes alone.
        tv = tokens.get(cid_full)
        if tv and tv[0] == "differ":
            e["token_delta"] = tv[1]["delta"]
        elif tv and tv[0] == "same":
            e["token_delta"] = 0
        elif prev.get("token_delta") is not None:
            # carried over rather than dropped: re-recording on a machine
            # without the model must not erase a measurement made on one
            # that had it.
            e["token_delta"] = prev["token_delta"]
            e["token_delta_stale"] = True
        entries.append(e)
    entries.sort(key=lambda e: e["id"])
    oracles = dict(old.get("oracles", {}))
    # fetched_at is per-run noise; the sha is the pin worth committing.
    oracles.update({n: {"source": m["source"], "sha256": m["sha256"]}
                    for n, m in oracle_meta.items()})
    doc = {
        "version": 1,
        "_README": [
            "KNOWN DIFFERENCES between runner's native C renderer and each",
            "family's REFERENCE -- its upstream jinja template, except for",
            "harmony, whose reference is the openai-harmony library (the",
            "GGUF's jinja is a lossy reimplementation of that spec; see",
            "scripts/template-conformance-harmony.py). These are BUGS",
            "AWAITING FIXES, not intentional deviations -- intentional",
            "deviations live in scripts/template-conformance-allowlist.json",
            "with a citation, and never appear here.",
            "",
            "`token_delta` is what the difference COSTS the model: the change",
            "in prompt token count, measured with the family's own tokenizer.",
            "0 means genuinely cosmetic. Absent means no tokenizer for that",
            "family was on the shelf, so the cost was NOT measured -- which is",
            "not the same as zero.",
            "",
            "This file exists so the gate is useful TODAY: it fails on NEW",
            "drift while the pre-existing backlog stays visible and counted.",
            "Every entry is a bug ticket. The file should only ever shrink.",
            "",
            "Regenerate after fixing (or after deliberately changing) a case:",
            "    make template-conformance-baseline",
            "`status` and `note` are preserved across regeneration; the",
            "digest and excerpts are recomputed.",
            "",
            "`digest` fingerprints the (reference, runner) pair, with today's",
            "date neutralised. If it moves, the difference changed SHAPE and",
            "the gate says CHANGED rather than quietly accepting a second bug",
            "hiding behind the first.",
            "",
            "`oracles` pins the sha256 of each upstream template this backlog",
            "was recorded against. If upstream moves, the gate says so.",
        ],
        "status_values": {
            "open": "a bug nobody is working on right now",
            "in-progress": "someone is actively fixing this; see `note`",
        },
        "generated_by": "scripts/template-conformance.py --write-baseline",
        "generated_at": datetime.datetime.now(
            datetime.timezone.utc).replace(microsecond=0).isoformat(),
        "oracles": dict(sorted(oracles.items())),
        "known_differences": entries,
    }
    with open(args.baseline, "w") as f:
        json.dump(doc, f, indent=2)
        f.write("\n")
    print("wrote %s: %d known differences across %d families"
          % (os.path.relpath(args.baseline, ROOT), len(entries),
             len({e["id"].split("/")[0] for e in entries})))
    return EXIT_OK


def main():
    ap = argparse.ArgumentParser(
        description="byte-exact conformance of runner's C chat renderer "
                    "against each family's upstream jinja template")
    ap.add_argument("--refresh", action="store_true",
                    help="re-fetch the upstream oracles before comparing "
                         "(needs network, and models/ for the GGUF-sourced "
                         "families)")
    ap.add_argument("--oracle-dir",
                    default=os.path.join(ROOT, ".build", "template-oracles"),
                    help="oracle cache (gitignored; `make clean` wipes it)")
    ap.add_argument("--allowlist", default=ALLOWLIST)
    ap.add_argument("--baseline", default=BASELINE)
    ap.add_argument("--family", action="append",
                    help="limit to these families (repeatable)")
    ap.add_argument("--network-only", action="store_true",
                    help="select only the families whose oracle is fetched "
                         "over the network, i.e. the ones checkable without "
                         "a local models/ tree")
    ap.add_argument("--show", action="store_true",
                    help="print both renders in full for differing cases")
    ap.add_argument("--raw", action="store_true",
                    help="do not subtract the allowlisted deviations")
    ap.add_argument("--write-baseline", action="store_true",
                    help="re-record the known-difference backlog")
    ap.add_argument("--write-harmony-oracle", action="store_true",
                    help="re-record scripts/template-conformance-harmony-"
                         "oracle.json from the installed openai-harmony "
                         "library (the fixture a machine without the library "
                         "compares harmony against)")
    ap.add_argument("--require-tokens", action="store_true",
                    help="treat a family with no local tokenizer as NOT "
                         "CHECKED (exit 2) instead of reporting the gap. Use "
                         "this on the family you are actively fixing: a "
                         "whitespace change that looks cosmetic in the text "
                         "diff can still change the token sequence.")
    args = ap.parse_args()
    if args.network_only:
        args.family = (args.family or []) + list(NETWORK_ONLY)
    if args.write_baseline and args.raw:
        raise SystemExit("--write-baseline with --raw would record the "
                         "intentional deviations as bugs")
    if args.write_harmony_oracle:
        return write_harmony_fixture()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
