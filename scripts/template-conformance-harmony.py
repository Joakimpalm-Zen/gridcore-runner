#!/usr/bin/env python3
"""The HARMONY oracle: openai-harmony, not the GGUF's jinja chat_template.

Every other family in scripts/template-conformance.py is compared against its
upstream jinja, because for those families the jinja IS the reference -- it is
the artifact the model was trained with and the thing every other runtime
executes. Harmony is the exception.

gpt-oss's wire format is defined by the `openai-harmony` library and its
docs/format.md, and the jinja embedded in the GGUF is a REIMPLEMENTATION of
that spec with acknowledged gaps. Three of them, all verified against
openai-harmony 0.0.8 and all cases where runner matches the library and the
jinja does not:

  developer turn, no tools   library: '# Instructions\\n\\nYou are ...<|end|>'
                             jinja:   an extra trailing '\\n\\n'
  replayed tool call         library: '<|channel|>commentary <|constrain|>json'
                             jinja:   a bare 'json', no <|constrain|>
  tool result                library: the result text RAW
                             jinja:   the result run through |tojson

The <|constrain|> gap is structural, not an oversight: the jinja has no
content_type field to carry it, so it could not emit the marker even if it
wanted to. A downstream template with gaps it cannot close does not outrank
the spec it is reimplementing, so for this family the oracle is the library.

    stdin : {"cases":[{"id":"...", "messages":[<OpenAI messages>],
                       "add_generation_prompt":true, "tools":[...]|null}]}
    stdout: {"library_version":"0.0.8", "renders":{"<id>":"<text>", ...}}

Run under any interpreter that has openai_harmony installed:

    python3 scripts/template-conformance-harmony.py < cases.json

scripts/template-conformance.py calls this for itself. It prefers a live
render, and falls back to the committed fixture
scripts/template-conformance-harmony-oracle.json when the library is not
installed anywhere it can find -- see that file's header for how to
regenerate, and `make template-conformance-harmony-oracle`.
"""

import json
import sys

TOOL_NS = "functions"


def library():
    try:
        import openai_harmony as oh
    except ImportError as e:                                    # noqa: BLE001
        raise SystemExit(
            "openai_harmony is not importable by %s (%s).\n"
            "The harmony oracle is the library, not the GGUF jinja. Install "
            "it (pip install openai-harmony) or point RUNNER_HARMONY_PYTHON "
            "at an interpreter that has it." % (sys.executable, e))
    return oh


def version(oh):
    try:
        from importlib.metadata import version as v
        return v("openai-harmony")
    except Exception:                                           # noqa: BLE001
        return getattr(oh, "__version__", "unknown")


def developer(oh, system_text, tools):
    """The developer turn: instructions and the function namespace.

    An OpenAI `system` message is developer INSTRUCTIONS in harmony terms --
    the `system` turn is the model's own identity and channel configuration,
    which the caller does not author. That is also what runner does
    (src/template.c renders a system role into a developer turn under
    '# Instructions').
    """
    if not system_text and not tools:
        return None
    d = oh.DeveloperContent.new()
    if system_text:
        d = d.with_instructions(system_text)
    if tools:
        descs = []
        for t in tools:
            fn = t.get("function", t)
            descs.append(oh.ToolDescription.new(
                fn["name"], fn.get("description", ""),
                parameters=fn.get("parameters")))
        d = d.with_function_tools(descs)
    return d


def convert(oh, messages, tools):
    """OpenAI messages -> a harmony Conversation.

    Mechanical, and deliberately narrow: it maps only the shapes the
    conformance matrix produces. Anything it does not recognise raises, so an
    unmapped shape is a loud failure rather than a silently wrong oracle.
    """
    out = [oh.Message.from_role_and_content(oh.Role.SYSTEM,
                                            oh.SystemContent.new())]
    # Harmony carries every caller instruction in ONE developer turn, so a
    # system message anywhere in the history contributes to it. The matrix's
    # system-mid-history case is exactly that: the position that several jinja
    # templates refuse, and that harmony expresses by construction.
    system_text = "\n\n".join(
        flatten(m.get("content")) for m in messages
        if m.get("role") == "system" and flatten(m.get("content")))
    dev = developer(oh, system_text, tools)
    if dev is not None:
        out.append(oh.Message.from_role_and_content(oh.Role.DEVELOPER, dev))

    for m in messages:
        role = m.get("role")
        if role == "system":
            continue                       # already folded into `developer`
        text = flatten(m.get("content"))
        if role == "user":
            out.append(oh.Message.from_role_and_content(oh.Role.USER, text))
        elif role == "assistant":
            reasoning = m.get("reasoning_content")
            if reasoning:
                out.append(oh.Message.from_role_and_content(
                    oh.Role.ASSISTANT, reasoning).with_channel("analysis"))
            calls = m.get("tool_calls") or []
            if text:
                out.append(oh.Message.from_role_and_content(
                    oh.Role.ASSISTANT, text).with_channel(
                        "commentary" if calls else "final"))
            for c in calls:
                fn = c["function"]
                # The replayed call carries the recipient, the commentary
                # channel AND the <|constrain|> content type. The GGUF jinja
                # cannot express the last of those -- it has no content_type
                # field -- which is the structural gap that makes the library
                # the oracle for this family.
                #
                # These three lines are the library's OWN canonical
                # construction, copied from the test that asserts
                # test-data/test_does_not_drop_if_ongoing_analysis.txt
                # (src/tests.rs:453-459 of openai-harmony 0.0.8). Note
                # "<|constrain|>json" with no space after the marker:
                # encoding.rs renders the leading space itself
                # (render_header_into, the ConstrainedFormat branch), so the
                # spaced spelling used in docs/format.md's python snippet
                # produces a DOUBLE space and does not match the library's own
                # fixture.
                out.append(
                    oh.Message.from_role_and_content(
                        oh.Role.ASSISTANT, fn["arguments"])
                    .with_channel("commentary")
                    .with_recipient("%s.%s" % (TOOL_NS, fn["name"]))
                    .with_content_type("<|constrain|>json"))
        elif role == "tool":
            name = m.get("name") or tool_name_for(messages, m)
            # The result is authored BY the function and addressed back to the
            # assistant on the commentary channel, and its text goes in RAW --
            # not re-encoded as a JSON string, which is the jinja's third
            # deviation.
            #
            # docs/format.md "Handling tool calls" states the shape outright:
            #   <|start|>{toolname} to=assistant<|channel|>commentary
            #   <|message|>{output}<|end|>
            # and two of the library's own fixtures carry it verbatim
            # (test-data/test_tool_response_parsing.txt,
            # test-data/test_streamable_parser.txt). The
            # test_does_not_drop_if_ongoing_analysis fixture omits both fields,
            # but that test constructs a deliberately minimal message to
            # exercise analysis-dropping; it is not a statement about the tool
            # turn's shape. Checked because getting this wrong would
            # manufacture a false backlog entry on every harmony tool case.
            out.append(
                oh.Message.from_author_and_content(
                    oh.Author.new(oh.Role.TOOL, "%s.%s" % (TOOL_NS, name)),
                    text)
                .with_channel("commentary")
                .with_recipient("assistant"))
        else:
            raise SystemExit("harmony oracle: unmapped role %r" % role)
    return oh.Conversation.from_messages(out)


def flatten(content):
    """Content part-arrays, joined the way src/server.c joins them."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = [p.get("text") for p in content
             if p.get("type") == "text" and p.get("text") is not None]
    return "\n".join(parts)


def tool_name_for(messages, msg):
    tcid = msg.get("tool_call_id")
    for m in messages:
        for c in m.get("tool_calls") or []:
            if c.get("id") == tcid:
                return c["function"]["name"]
    raise SystemExit("harmony oracle: tool result %r names no function"
                     % tcid)


def render(oh, enc, case):
    convo = convert(oh, case["messages"], case.get("tools"))
    if case.get("add_generation_prompt", True):
        toks = enc.render_conversation_for_completion(convo, oh.Role.ASSISTANT)
    else:
        toks = enc.render_conversation(convo)
    return enc.decode_utf8(toks)


def main():
    oh = library()
    enc = oh.load_harmony_encoding(oh.HarmonyEncodingName.HARMONY_GPT_OSS)
    req = json.load(sys.stdin)
    renders = {}
    for case in req["cases"]:
        renders[case["id"]] = render(oh, enc, case)
    json.dump({"library_version": version(oh), "renders": renders},
              sys.stdout, indent=1, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
