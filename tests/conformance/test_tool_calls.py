"""Strict tool calls: the schema engine guarantees the call, not a parser.

Phase 1. ``tools[].function.parameters`` is compiled into a discriminated
union with one branch per tool plus a ``final`` branch, and sampling is
constrained to it. So these tests can assert things a post-hoc parser could
never promise against a random-weight test model: the tool name is always one
that was declared, the arguments always parse, and they always conform to the
declared parameter schema — even when ``max_tokens`` cuts generation short.

Buffered requests only. Streaming tool calls are Phase 2 and are pinned by
test_known_gap_no_tool_calls_in_stream in test_streaming.py.
"""

import json

import pytest

from harness import ProtocolError, SchemaError, validate_against_schema

BASE = {"messages": [{"role": "user", "content": "what is the weather?"}],
        "temperature": 0}

WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "look up the weather in a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"},
                           "units": {"enum": ["celsius", "fahrenheit"]}},
            "required": ["city", "units"],
        },
    },
}
ADD = {
    "type": "function",
    "function": {
        "name": "add",
        "parameters": {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
    },
}
TOOLS = [WEATHER, ADD]
BY_NAME = {t["function"]["name"]: t["function"]["parameters"] for t in TOOLS}


def _only_call(r):
    """The single tool_calls entry, checked for OpenAI shape."""
    msg = r.choice.get("message") or {}
    calls = msg.get("tool_calls")
    if not isinstance(calls, list) or len(calls) != 1:
        raise ProtocolError("expected exactly one tool_calls entry",
                            request=r.name, got=calls, body=r.text[:300])
    call = calls[0]
    if call.get("type") != "function":
        raise ProtocolError("tool_calls[].type must be \"function\"",
                            request=r.name, got=call.get("type"))
    if not isinstance(call.get("id"), str) or not call["id"]:
        raise ProtocolError("tool_calls[].id must be a non-empty string",
                            request=r.name, got=call.get("id"))
    fn = call.get("function")
    if not isinstance(fn, dict):
        raise ProtocolError("tool_calls[].function missing", request=r.name)
    if not isinstance(fn.get("arguments"), str):
        # OpenAI carries arguments as a JSON *string*; SDKs call json.loads on
        # it unconditionally and crash on an object
        raise ProtocolError("tool_calls[].function.arguments must be a string",
                            request=r.name, got=type(fn.get("arguments")).__name__)
    return call["id"], fn["name"], fn["arguments"]


def _assert_conforms(r, name, arguments):
    if name not in BY_NAME:
        raise ProtocolError("model named a tool that was never declared",
                            request=r.name, got=name, declared=sorted(BY_NAME))
    try:
        parsed = json.loads(arguments)
    except ValueError as exc:
        raise ProtocolError("tool arguments are not valid JSON",
                            request=r.name, arguments=arguments[:200],
                            error=str(exc)) from exc
    validate_against_schema(parsed, BY_NAME[name], path=f"$.{name}")
    return parsed


def test_required_always_produces_a_conforming_call(client):
    """tool_choice "required" removes the no-call branch, so the union the
    sampler is held to contains nothing but tool calls."""
    r = client.chat(dict(BASE, max_tokens=64, tools=TOOLS,
                         tool_choice="required"),
                    name="tools-required")
    r.expect_status(200)
    if r.finish_reason != "tool_calls":
        raise ProtocolError("a guaranteed tool call must report "
                            "finish_reason \"tool_calls\"",
                            request=r.name, got=r.finish_reason)
    if r.content:
        raise ProtocolError("content must be empty alongside tool_calls",
                            request=r.name, got=r.content[:200])
    _, name, arguments = _only_call(r)
    _assert_conforms(r, name, arguments)


def test_named_tool_choice_selects_exactly_that_tool(client):
    for want in ("get_weather", "add"):
        r = client.chat(
            dict(BASE, max_tokens=64, tools=TOOLS,
                 tool_choice={"type": "function", "function": {"name": want}}),
            name=f"tools-named-{want}")
        r.expect_status(200)
        _, name, arguments = _only_call(r)
        if name != want:
            raise ProtocolError("tool_choice named one tool and another was "
                                "called", request=r.name, want=want, got=name)
        _assert_conforms(r, name, arguments)


@pytest.mark.parametrize("max_tokens", [1, 2, 3, 5, 8, 13, 21])
def test_truncated_call_is_still_valid_and_executable(client, max_tokens):
    """The guarantee that a post-hoc parser cannot make. At tiny max_tokens
    the envelope is cut mid-token; sval_close completes it to the schema's
    minimum, so the caller still receives arguments it can execute rather
    than a fragment it must discard."""
    r = client.chat(dict(BASE, max_tokens=max_tokens, tools=TOOLS,
                         tool_choice="required"),
                    name=f"tools-truncated-{max_tokens}")
    r.expect_status(200)
    _, name, arguments = _only_call(r)
    _assert_conforms(r, name, arguments)


def test_auto_allows_a_plain_answer(client):
    """"auto" adds the final branch back, so a normal reply is legal again.
    Whichever branch the model lands on must be reported coherently: a call
    with finish_reason "tool_calls", or content with no tool_calls at all."""
    r = client.chat(dict(BASE, max_tokens=64, tools=TOOLS, tool_choice="auto"),
                    name="tools-auto")
    r.expect_status(200)
    msg = r.choice.get("message") or {}
    if msg.get("tool_calls"):
        if r.finish_reason != "tool_calls":
            raise ProtocolError("tool_calls present but finish_reason is not "
                                "\"tool_calls\"", request=r.name,
                                got=r.finish_reason)
        _, name, arguments = _only_call(r)
        _assert_conforms(r, name, arguments)
    else:
        if r.finish_reason == "tool_calls":
            raise ProtocolError("finish_reason \"tool_calls\" with no calls",
                                request=r.name, body=r.text[:300])
        # the final branch unwraps to plain text, never to the raw envelope
        if r.content.lstrip().startswith('{"tool"'):
            raise ProtocolError("the internal envelope leaked into content",
                                request=r.name, got=r.content[:200])


def test_tool_choice_none_suppresses_calls(client):
    r = client.chat(dict(BASE, max_tokens=32, tools=TOOLS, tool_choice="none"),
                    name="tools-none")
    r.expect_status(200)
    if (r.choice.get("message") or {}).get("tool_calls"):
        raise ProtocolError("tool_choice \"none\" still produced a call",
                            request=r.name, body=r.text[:300])


def test_tools_and_response_format_in_the_same_request(client):
    """Exit criterion: both work together. The response_format schema becomes
    the shape of the ``final`` branch, so the answer is either a conforming
    tool call or a conforming JSON document — never an unconstrained one."""
    answer_schema = {"type": "object",
                     "properties": {"answer": {"type": "string"}},
                     "required": ["answer"]}
    r = client.chat(dict(BASE, max_tokens=64, tools=TOOLS, tool_choice="auto",
                         response_format={"type": "json_schema",
                                          "json_schema": {"name": "a",
                                                          "schema": answer_schema}}),
                    name="tools-with-response-format")
    r.expect_status(200)
    msg = r.choice.get("message") or {}
    if msg.get("tool_calls"):
        _, name, arguments = _only_call(r)
        _assert_conforms(r, name, arguments)
    else:
        try:
            parsed = json.loads(r.content)
        except ValueError as exc:
            raise ProtocolError("final branch did not honour response_format",
                                request=r.name, got=r.content[:200]) from exc
        validate_against_schema(parsed, answer_schema)


def test_a_parameterless_tool_is_callable(client):
    r = client.chat(dict(BASE, max_tokens=32, tool_choice="required",
                         tools=[{"type": "function",
                                 "function": {"name": "ping"}}]),
                    name="tools-parameterless")
    r.expect_status(200)
    _, name, arguments = _only_call(r)
    if name != "ping":
        raise ProtocolError("wrong tool called", request=r.name, got=name)
    json.loads(arguments)


# ------------------------------------------------------------ rejections
# Same invariant as everywhere else: a tools payload the engine cannot
# compile must 400 rather than fall back to unconstrained generation, which
# would answer 200 while guaranteeing nothing.

@pytest.mark.parametrize("tools,label,contains", [
    ({"a": 1}, "not-an-array", "tools"),
    ([{"type": "function"}], "no-function", "function"),
    ([{"type": "function", "function": {}}], "no-name", "name"),
    ([{"type": "function", "function": {"name": ""}}], "empty-name", "name"),
    ([{"type": "retrieval", "function": {"name": "a"}}], "wrong-type", "type"),
    ([{"type": "function", "function": {"name": "a", "parameters": 7}}],
     "parameters-not-object", "parameters"),
    ([{"type": "function", "function": {"name": "a"}},
      {"type": "function", "function": {"name": "a"}}], "duplicate", "duplicate"),
    ([{"type": "function", "function": {"name": "final"}}], "reserved", "reserved"),
    # a parameter schema the compiler cannot enforce: silently approximating
    # it would mean guaranteeing a constraint that is not there
    ([{"type": "function",
       "function": {"name": "a",
                    "parameters": {"type": "object",
                                   "properties": {"p": {"type": "string",
                                                        "pattern": "^x$"}}}}}],
     "unenforceable-parameters", "keyword"),
])
def test_malformed_tools_are_rejected(client, tools, label, contains):
    client.expect_400(dict(BASE, max_tokens=8, tools=tools,
                           tool_choice="required"),
                      name=f"bad-tools-{label}", contains=contains)


@pytest.mark.parametrize("choice,label", [
    ("maybe", "unknown-string"),
    (7, "number"),
    ([], "array"),
    ({"type": "retrieval", "function": {"name": "add"}}, "wrong-type"),
    ({"type": "function"}, "no-function"),
    ({"type": "function", "function": {"name": "not_declared"}}, "undeclared"),
])
def test_malformed_tool_choice_is_rejected(client, choice, label):
    client.expect_400(dict(BASE, max_tokens=8, tools=TOOLS, tool_choice=choice),
                      name=f"bad-tool-choice-{label}", contains="tool")


def test_tool_choice_without_tools_is_rejected(client):
    """"required" with nothing to call is a contradiction, not a request to
    answer normally."""
    for choice in ("required",
                   {"type": "function", "function": {"name": "add"}}):
        client.expect_400(dict(BASE, max_tokens=8, tool_choice=choice),
                          name=f"tool-choice-no-tools-{choice}",
                          contains="tools")


def test_parallel_tool_calls_true_is_rejected(client):
    """Phase 1 guarantees one call per turn. Accepting the flag and returning
    a single call anyway would leave the caller waiting on results for calls
    that were never made."""
    client.expect_400(dict(BASE, max_tokens=8, tools=TOOLS,
                           tool_choice="required", parallel_tool_calls=True),
                      name="parallel-tool-calls",
                      contains="parallel_tool_calls")


def test_tools_are_still_advisory_when_absent(client):
    """The flags stay tolerated on a request with no tools — rejecting them
    there would break ordinary OpenAI-shaped traffic."""
    r = client.chat(dict(BASE, max_tokens=8, tool_choice="auto",
                         parallel_tool_calls=True),
                    name="tool-flags-without-tools")
    r.expect_status(200)
