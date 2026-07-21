"""The Anthropic Messages surface: POST /v1/messages and /v1/messages/count_tokens.

Phase 4. Like /v1/responses before it, this endpoint is a *translation layer*,
not a second engine: an Anthropic request is reshaped into the same internal
prompt and the same strict tool envelope /v1/chat/completions builds, and the
generated result is rendered in the Anthropic vocabulary on the way out.

So the tests here assert three things, and deliberately not a fourth:

  - the wire shape, and for streams the *order and names* of the six event
    types, because that ordering is exactly what the Anthropic SDKs validate;
  - that unsupported features are refused rather than silently ignored; and
  - that this surface reaches the same engine as the OpenAI ones, by checking
    the two surfaces agree on the same request.

They do not re-assert the generation guarantees themselves (that a tool call
names a declared tool, that its arguments conform) — those belong to the engine
and are covered by test_tool_calls.py.
"""

import json

import pytest

from harness import ProtocolError

BASE = {"messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 8, "temperature": 0}

WEATHER = {
    "name": "get_weather",
    "description": "Look up the weather for a city.",
    "input_schema": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
        "additionalProperties": False,
    },
}


# ------------------------------------------------------------------- buffered
def test_messages_is_advertised_in_capabilities(client):
    """/v1/capabilities is the feature-discovery endpoint. A surface that
    exists but is not discoverable there is one a client has to guess at."""
    caps = client.get("/v1/capabilities").expect_status(200).json
    if not caps.get("features", {}).get("messages_api"):
        raise ProtocolError("/v1/messages is not advertised in capabilities",
                            features=caps.get("features"))


def test_messages_buffered_shape(client, report):
    """The Message object an Anthropic SDK deserialises."""
    r = client.messages(dict(BASE), name="messages-buffered").expect_status(200)
    d = r.json
    if d.get("type") != "message" or d.get("role") != "assistant":
        raise ProtocolError("not an assistant message object", body=d)
    for field in ("id", "model", "content", "stop_reason", "stop_sequence",
                  "usage"):
        if field not in d:
            raise ProtocolError("message missing required field", field=field,
                                keys=sorted(d))
    if not isinstance(d["id"], str) or not d["id"].startswith("msg_"):
        raise ProtocolError("message id is not an msg_ id", got=d.get("id"))
    if not isinstance(d["content"], list) or not d["content"]:
        raise ProtocolError("content is not a non-empty list",
                            got=d.get("content"))
    block = d["content"][0]
    if block.get("type") != "text" or not isinstance(block.get("text"), str):
        raise ProtocolError("first content block is not a text block",
                            block=block)
    # the stub model rarely stops on its own inside max_tokens, so both are
    # legitimate; what is asserted is that the value is one Anthropic defines
    if d["stop_reason"] not in ("end_turn", "max_tokens", "stop_sequence",
                                "tool_use"):
        raise ProtocolError("unknown stop_reason", got=d["stop_reason"])
    u = d["usage"]
    if not isinstance(u.get("input_tokens"), int) or \
            not isinstance(u.get("output_tokens"), int):
        raise ProtocolError("usage is not Anthropic-shaped", usage=u)
    report.check_fixture("messages_buffered", d)


def test_max_tokens_is_required(client):
    """Unlike the OpenAI surfaces, Anthropic makes max_tokens mandatory. A
    caller that forgot it wants a cap of its own, not the server's."""
    client.expect_400({"messages": [{"role": "user", "content": "hi"}]},
                      "messages-no-max-tokens", contains="max_tokens",
                      path="/v1/messages")


# ------------------------------------------------------------------ streaming
def test_messages_stream_event_order(client, report):
    """The ordered event contract, which is exactly what a typed SDK validates.

    Asserted as a grammar rather than a fixed list: the number of deltas
    depends on what the model emitted, but the nesting never does."""
    st = client.messages_stream(dict(BASE), name="messages-stream").expect_sse()
    names = st.names
    if not names:
        raise ProtocolError("stream carried no events", raw=st.raw[:300])
    if names[0] != "message_start":
        raise ProtocolError("stream does not open with message_start",
                            got=names[0], names=names)
    if names[-1] != "message_stop":
        raise ProtocolError("stream does not end with message_stop",
                            got=names[-1], names=names)
    if names[-2] != "message_delta":
        raise ProtocolError("message_stop is not preceded by message_delta",
                            got=names[-2], names=names)
    # an Anthropic stream has no [DONE] sentinel; message_stop is the terminator
    if b"[DONE]" in st.raw:
        raise ProtocolError("an Anthropic stream must not carry data: [DONE]",
                            names=names)

    # every block opened is closed, and its deltas fall strictly inside it
    depth, seen, expect_index = 0, 0, 0
    for name, d in st.typed:
        if name == "content_block_start":
            if depth:
                raise ProtocolError("content block opened inside another",
                                    names=names)
            depth, seen = 1, seen + 1
            if d.get("index") != expect_index:
                raise ProtocolError("content block index is not sequential",
                                    got=d.get("index"), expected=expect_index,
                                    names=names)
        elif name == "content_block_delta":
            if not depth:
                raise ProtocolError("delta emitted outside a content block",
                                    names=names)
            if d.get("index") != expect_index:
                raise ProtocolError("delta index does not match its block",
                                    got=d.get("index"), expected=expect_index)
        elif name == "content_block_stop":
            if depth != 1:
                raise ProtocolError("content block closed while none was open",
                                    names=names)
            depth, expect_index = 0, expect_index + 1
        elif name in ("message_start", "message_delta", "message_stop"):
            if depth:
                raise ProtocolError("message-level event inside an open block",
                                    event=name, names=names)
    if depth:
        raise ProtocolError("stream ended with a content block still open",
                            names=names)
    if not seen:
        raise ProtocolError("stream produced no content blocks", names=names)
    report.check_fixture("messages_stream_events", [d for _, d in st.typed])


def test_message_start_carries_an_empty_shell(client):
    """message_start must describe the turn's identity before any token
    exists, so a client can render it immediately — and must not claim a
    stop_reason it cannot yet know."""
    st = client.messages_stream(dict(BASE), name="messages-start").expect_sse()
    starts = st.payloads("message_start")
    if len(starts) != 1:
        raise ProtocolError("expected exactly one message_start",
                            got=len(starts))
    m = starts[0].get("message")
    if not isinstance(m, dict):
        raise ProtocolError("message_start carries no message object",
                            payload=starts[0])
    if m.get("type") != "message" or m.get("role") != "assistant":
        raise ProtocolError("message_start shell is not an assistant message",
                            message=m)
    if m.get("content") != []:
        raise ProtocolError("message_start must open with empty content",
                            content=m.get("content"))
    if m.get("stop_reason") is not None:
        raise ProtocolError("message_start claims a stop_reason already",
                            stop_reason=m.get("stop_reason"))
    if not isinstance(m.get("usage", {}).get("input_tokens"), int):
        raise ProtocolError("message_start does not report input_tokens",
                            usage=m.get("usage"))


def test_message_delta_reports_the_terminal_reason_and_usage(client):
    st = client.messages_stream(dict(BASE), name="messages-delta").expect_sse()
    deltas = st.payloads("message_delta")
    if len(deltas) != 1:
        raise ProtocolError("expected exactly one message_delta",
                            got=len(deltas))
    d = deltas[0]
    if d.get("delta", {}).get("stop_reason") not in (
            "end_turn", "max_tokens", "stop_sequence", "tool_use"):
        raise ProtocolError("message_delta carries no valid stop_reason",
                            payload=d)
    if not isinstance(d.get("usage", {}).get("output_tokens"), int):
        raise ProtocolError("message_delta does not report output_tokens",
                            payload=d)


def test_streamed_and_buffered_turns_agree(client):
    """The streamed events and the buffered body must describe one turn, not
    two that drifted: same block kinds, same stop_reason."""
    payload = dict(BASE, messages=[{"role": "user", "content": "count to three"}],
                   max_tokens=24)
    st = client.messages_stream(dict(payload), name="messages-eq-stream").expect_sse()
    buf = client.messages(dict(payload), name="messages-eq-buffered") \
                .expect_status(200).json
    s_kinds = [b["type"] for b in st.blocks]
    b_kinds = [b["type"] for b in buf["content"]]
    if s_kinds != b_kinds:
        raise ProtocolError("streamed and buffered block kinds differ",
                            streamed=s_kinds, buffered=b_kinds)
    s_stop = st.payloads("message_delta")[0]["delta"]["stop_reason"]
    if s_stop != buf["stop_reason"]:
        raise ProtocolError("streamed and buffered stop_reason differ",
                            streamed=s_stop, buffered=buf["stop_reason"])
    # temperature 0 and an identical prompt: the text itself must match too
    s_text = "".join(b.get("text", "") for b in st.blocks)
    b_text = "".join(b.get("text", "") for b in buf["content"])
    if s_text != b_text:
        raise ProtocolError("streamed text does not match the buffered message",
                            streamed=s_text, buffered=b_text)


def test_empty_turn_still_carries_a_content_block(client):
    """A turn that generates nothing must still describe itself the way a
    buffered one does — content[] is never empty on either path."""
    payload = dict(BASE, max_tokens=1)
    st = client.messages_stream(dict(payload), name="messages-empty-stream")
    st.expect_sse()
    if not st.blocks:
        raise ProtocolError("a stream produced no content block at all",
                            names=st.names)
    buf = client.messages(dict(payload), name="messages-empty-buffered") \
                .expect_status(200).json
    if not buf["content"]:
        raise ProtocolError("a buffered turn produced empty content",
                            body=buf)


# ---------------------------------------------------------------- tool calling
def _forced_call(client, name, stream=False, **extra):
    """tool_choice {"type":"any"} forces the envelope's call branch, so this
    does not depend on the stub model choosing to call anything."""
    payload = {"messages": [{"role": "user", "content": "what is the weather in Oslo?"}],
               "max_tokens": 48, "temperature": 0, "tools": [WEATHER],
               "tool_choice": {"type": "any"}}
    payload.update(extra)
    if stream:
        st = client.messages_stream(payload, name=name).expect_sse()
        return st, None
    return None, client.messages(payload, name=name).expect_status(200).json


def test_tool_use_block_is_produced(client, report):
    _, d = _forced_call(client, "messages-tool-buffered")
    uses = [b for b in d["content"] if b["type"] == "tool_use"]
    if not uses:
        raise ProtocolError("tool_choice any produced no tool_use block",
                            content=d["content"])
    use = uses[0]
    for field in ("id", "name", "input"):
        if field not in use:
            raise ProtocolError("tool_use block missing field", field=field,
                                block=use)
    if use["name"] != "get_weather":
        raise ProtocolError("model named an undeclared tool", got=use["name"])
    # Anthropic carries the arguments as an object, not the JSON *string*
    # OpenAI uses. A client reads use["input"]["city"] directly.
    if not isinstance(use["input"], dict):
        raise ProtocolError("tool_use.input is not a JSON object",
                            got=type(use["input"]).__name__, block=use)
    if d["stop_reason"] != "tool_use":
        raise ProtocolError("a tool call did not report stop_reason tool_use",
                            got=d["stop_reason"])
    if any(b["type"] == "text" for b in d["content"]):
        raise ProtocolError("a tool call was accompanied by a text block",
                            content=d["content"])
    report.check_fixture("messages_tool_use", d)


def test_streamed_tool_use_rebuilds_the_buffered_block(client):
    st, _ = _forced_call(client, "messages-tool-stream", stream=True)
    _, buf = _forced_call(client, "messages-tool-buffered-eq")
    s_use = [b for b in st.blocks if b["type"] == "tool_use"]
    b_use = [b for b in buf["content"] if b["type"] == "tool_use"]
    if not s_use or not b_use:
        raise ProtocolError("a forced tool call was missing on one path",
                            streamed=st.blocks, buffered=buf["content"])
    if s_use[0]["name"] != b_use[0]["name"]:
        raise ProtocolError("streamed and buffered called different tools",
                            streamed=s_use[0]["name"], buffered=b_use[0]["name"])
    # the input_json_delta run must rebuild exactly the buffered input object
    if s_use[0]["input"] != b_use[0]["input"]:
        raise ProtocolError("input_json_delta run does not rebuild the input",
                            streamed=s_use[0]["input"], buffered=b_use[0]["input"])
    if st.payloads("message_delta")[0]["delta"]["stop_reason"] != "tool_use":
        raise ProtocolError("streamed tool call did not report tool_use",
                            events=st.names)
    # no envelope syntax may leak into a text block
    if st.text:
        raise ProtocolError("a tool call leaked text into content_block_delta",
                            text=st.text)


def test_tool_use_input_is_always_a_json_object(client):
    """Anthropic inlines the arguments as an object where OpenAI carries the
    same document as an escaped string. That difference is load-bearing: a
    string can hold anything, an inlined object cannot. Free-text call syntax
    is only brace-matched, not parsed, so it can be balanced yet invalid —
    inlining that verbatim would emit a response body no client can parse.
    Both the strict and the unconstrained path must therefore be safe."""
    for name, payload in [
        ("strict", {"messages": [{"role": "user", "content": "weather in Oslo?"}],
                    "max_tokens": 48, "temperature": 0, "tools": [WEATHER],
                    "tool_choice": {"type": "any"}}),
        # tool_choice none builds no envelope, so anything that looks like a
        # call in the output goes through the free-text parser instead
        ("unconstrained", {"messages": [{"role": "user", "content": "weather?"}],
                           "max_tokens": 64, "temperature": 0,
                           "tools": [WEATHER],
                           "tool_choice": {"type": "none"}}),
    ]:
        d = client.messages(payload, name=f"messages-input-{name}") \
                  .expect_status(200).json          # .json fails on a bad body
        for block in d["content"]:
            if block["type"] == "tool_use" and not isinstance(block["input"], dict):
                raise ProtocolError("tool_use.input is not a JSON object",
                                    path=name, block=block)


def test_tool_choice_none_answers_directly(client):
    d = client.messages({"messages": [{"role": "user", "content": "hello"}],
                         "max_tokens": 16, "temperature": 0, "tools": [WEATHER],
                         "tool_choice": {"type": "none"}},
                        name="messages-tool-none").expect_status(200).json
    if any(b["type"] == "tool_use" for b in d["content"]):
        raise ProtocolError("tool_choice none still produced a call",
                            content=d["content"])


def test_named_tool_choice_selects_that_tool(client):
    d = client.messages({"messages": [{"role": "user", "content": "weather in Oslo?"}],
                         "max_tokens": 48, "temperature": 0,
                         "tools": [WEATHER, {"name": "get_time",
                                             "input_schema": {"type": "object"}}],
                         "tool_choice": {"type": "tool", "name": "get_time"}},
                        name="messages-tool-named").expect_status(200).json
    uses = [b for b in d["content"] if b["type"] == "tool_use"]
    if not uses or uses[0]["name"] != "get_time":
        raise ProtocolError("named tool_choice did not select that tool",
                            content=d["content"])


def test_tool_result_closes_the_loop(client):
    """The second half of an agent turn: the assistant's tool_use is replayed
    and its tool_result fed back. Asserted by input-token growth, so the test
    proves the result actually reached the prompt rather than merely being
    accepted with a 200."""
    first = {"role": "user", "content": "what is the weather in Oslo?"}
    call = {"role": "assistant",
            "content": [{"type": "tool_use", "id": "toolu_0",
                         "name": "get_weather", "input": {"city": "Oslo"}}]}
    short = {"role": "user",
             "content": [{"type": "tool_result", "tool_use_id": "toolu_0",
                          "content": "ok"}]}
    long = {"role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "toolu_0",
                         "content": "it is " + "extremely " * 12 + "cold"}]}
    base = {"max_tokens": 8, "temperature": 0, "tools": [WEATHER]}
    a = client.messages(dict(base, messages=[first, call, short]),
                        name="messages-tool-result-short") \
              .expect_status(200).json
    b = client.messages(dict(base, messages=[first, call, long]),
                        name="messages-tool-result-long") \
              .expect_status(200).json
    if b["usage"]["input_tokens"] <= a["usage"]["input_tokens"]:
        raise ProtocolError("tool_result content did not reach the prompt",
                            short=a["usage"], long=b["usage"])


# ------------------------------------------------------------ system + blocks
def test_system_reaches_the_model(client):
    """A system string and a system block list say the same thing, and both
    must actually be prepended rather than dropped."""
    base = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4,
            "temperature": 0}
    plain = client.count_tokens(dict(base), name="count-no-system") \
                  .expect_status(200).json["input_tokens"]
    text = "You are a laconic assistant that answers in one word."
    as_str = client.count_tokens(dict(base, system=text),
                                 name="count-system-str") \
                   .expect_status(200).json["input_tokens"]
    as_blocks = client.count_tokens(
        dict(base, system=[{"type": "text", "text": text}]),
        name="count-system-blocks").expect_status(200).json["input_tokens"]
    if as_str <= plain:
        raise ProtocolError("system content did not reach the prompt",
                            without=plain, with_system=as_str)
    if as_str != as_blocks:
        raise ProtocolError("system string and system blocks disagree",
                            as_string=as_str, as_blocks=as_blocks)


def test_text_blocks_and_plain_string_are_equivalent(client):
    """content: "hi" and content: [{"type":"text","text":"hi"}] are the same
    request and must build the same prompt."""
    base = {"max_tokens": 4, "temperature": 0}
    a = client.count_tokens(
        dict(base, messages=[{"role": "user", "content": "hello there"}]),
        name="count-str-content").expect_status(200).json["input_tokens"]
    b = client.count_tokens(
        dict(base, messages=[{"role": "user",
                              "content": [{"type": "text", "text": "hello there"}]}]),
        name="count-block-content").expect_status(200).json["input_tokens"]
    if a != b:
        raise ProtocolError("string and block content built different prompts",
                            as_string=a, as_blocks=b)


# ------------------------------------------------------------- stop sequences
def test_stop_sequence_is_reported_by_name(client):
    """Anthropic reports *which* sequence stopped the turn, which the OpenAI
    surfaces do not. A client uses it to decide what to do next."""
    d = client.messages({"messages": [{"role": "user", "content": "say a b c"}],
                         "max_tokens": 64, "temperature": 0,
                         "stop_sequences": ["a", "b", "c", "e"]},
                        name="messages-stop-seq").expect_status(200).json
    if d["stop_reason"] == "stop_sequence":
        if d.get("stop_sequence") not in ("a", "b", "c", "e"):
            raise ProtocolError("stop_sequence is not one of those requested",
                                got=d.get("stop_sequence"))
        for block in d["content"]:
            if block["type"] == "text" and d["stop_sequence"] in block["text"]:
                raise ProtocolError("the stop sequence was not excluded",
                                    text=block["text"],
                                    stop=d["stop_sequence"])
    elif d.get("stop_sequence") is not None:
        raise ProtocolError("stop_sequence set without stop_reason "
                            "stop_sequence", body=d)


def test_stop_sequences_are_bounded(client):
    client.expect_400({"messages": [{"role": "user", "content": "hi"}],
                       "max_tokens": 4,
                       "stop_sequences": ["a", "b", "c", "d", "e"]},
                      "messages-too-many-stops", contains="stop",
                      path="/v1/messages")


# ---------------------------------------------------------------- count_tokens
def test_count_tokens_matches_a_real_request(client):
    """The count must be the count of the prompt the request would really
    run, not an estimate built by a second code path."""
    payload = {"messages": [{"role": "user", "content": "what is the weather in Oslo?"}],
               "system": "Be brief.", "tools": [WEATHER],
               "temperature": 0}
    counted = client.count_tokens(dict(payload), name="count-tokens") \
                    .expect_status(200).json
    if not isinstance(counted.get("input_tokens"), int) or \
            counted["input_tokens"] <= 0:
        raise ProtocolError("count_tokens did not return a positive count",
                            body=counted)
    real = client.messages(dict(payload, max_tokens=4),
                           name="count-tokens-real").expect_status(200).json
    if counted["input_tokens"] != real["usage"]["input_tokens"]:
        raise ProtocolError("count_tokens disagrees with the request it counts",
                            counted=counted["input_tokens"],
                            actual=real["usage"]["input_tokens"])


def test_count_tokens_does_not_require_max_tokens(client):
    """count_tokens describes an input; there is no output to cap."""
    r = client.count_tokens({"messages": [{"role": "user", "content": "hi"}]},
                            name="count-tokens-no-max")
    r.expect_status(200)


def test_count_tokens_rejects_what_messages_rejects(client):
    """The two endpoints must agree about what a valid request is, or a client
    that pre-flights with count_tokens learns nothing."""
    bad = {"messages": [{"role": "user",
                         "content": [{"type": "image", "source": {}}]}]}
    client.expect_400(dict(bad), "count-tokens-image", contains="image",
                      path="/v1/messages/count_tokens")
    client.expect_400(dict(bad, max_tokens=4), "messages-image",
                      contains="image", path="/v1/messages")


# ------------------------------------------------------------------- refusals
@pytest.mark.parametrize("field,value,explains", [
    ("mcp_servers", [{"type": "url", "url": "https://x", "name": "x"}], "mcp"),
    ("container", "cnt_123", "container"),
    ("thinking", {"type": "enabled", "budget_tokens": 1024}, "thinking"),
])
def test_unsupported_features_are_refused(client, field, value, explains):
    """Silently ignoring a feature the caller asked for is the failure mode
    this project refuses: the caller gets a 200 and believes it happened."""
    payload = {"messages": [{"role": "user", "content": "hi"}],
               "max_tokens": 8, field: value}
    client.expect_400(payload, f"messages-{field}", contains=explains,
                      path="/v1/messages")


@pytest.mark.parametrize("payload,contains", [
    ({"max_tokens": 8}, "messages"),
    ({"messages": [], "max_tokens": 8}, "messages"),
    ({"messages": [{"role": "user", "content": 7}], "max_tokens": 8},
     "content"),
    ({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8,
      "tools": [{"type": "web_search_20250305", "name": "web_search"}]},
     "not supported"),
    ({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8,
      "tools": [WEATHER], "tool_choice": {"type": "sometimes"}}, "tool_choice"),
    ({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8,
      "tools": [WEATHER], "tool_choice": {"type": "tool"}}, "name"),
    ({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8,
      "metadata": "nope"}, "metadata"),
    ({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8,
      "stream": "yes"}, "stream"),
])
def test_malformed_requests_are_rejected(client, payload, contains):
    client.expect_400(payload, "messages-malformed", contains=contains,
                      path="/v1/messages")


def test_parallel_tool_use_is_refused_not_quietly_downgraded(client):
    """The envelope is one call per turn on every surface. A client that asked
    for several and got one could not tell that apart from a considered
    choice, so it is refused exactly as parallel_tool_calls:true is."""
    client.expect_400({"messages": [{"role": "user", "content": "hi"}],
                       "max_tokens": 8, "tools": [WEATHER],
                       "tool_choice": {"type": "auto",
                                       "disable_parallel_tool_use": False}},
                      "messages-parallel", contains="one tool call per turn",
                      path="/v1/messages")


def test_unknown_advisory_fields_are_tolerated(client):
    """The refusal rule is about fields that change the *result*. Advisory
    metadata that cannot is accepted, or ordinary SDK traffic would 400."""
    r = client.messages({"messages": [{"role": "user", "content": "hi"}],
                         "max_tokens": 8, "temperature": 0,
                         "metadata": {"user_id": "u1"},
                         "thinking": {"type": "disabled"},
                         "top_k": 40, "top_p": 0.9},
                        name="messages-advisory")
    r.expect_status(200)


# --------------------------------------------------------------- real SDK
# The wire shape above is asserted by hand, which is necessary but not
# sufficient: Phase 3 found three bugs that only a real client exposed. These
# run the official Anthropic SDK against the same server when it is installed,
# and skip when it is not, so CI does not gain a network-installed dependency
# but a developer who has one gets the stronger check.
try:
    import anthropic as _anthropic
except ImportError:  # pragma: no cover - depends on the environment
    _anthropic = None

needs_sdk = pytest.mark.skipif(_anthropic is None,
                               reason="the anthropic SDK is not installed")


@pytest.fixture(scope="module")
def sdk(server):
    if _anthropic is None:
        pytest.skip("the anthropic SDK is not installed")
    return _anthropic.Anthropic(base_url=f"http://127.0.0.1:{server.port}",
                                api_key="not-used", max_retries=0)


@needs_sdk
def test_real_sdk_parses_a_buffered_turn(sdk):
    m = sdk.messages.create(model="local", max_tokens=8, temperature=0,
                            messages=[{"role": "user", "content": "hello"}])
    if m.type != "message" or m.role != "assistant":
        raise ProtocolError("SDK did not deserialise a Message", got=repr(m))
    if not m.content or m.content[0].type != "text":
        raise ProtocolError("SDK did not deserialise a TextBlock",
                            content=repr(m.content))
    if not isinstance(m.usage.input_tokens, int):
        raise ProtocolError("SDK did not deserialise usage", usage=repr(m.usage))


@needs_sdk
def test_real_sdk_accumulates_a_stream(sdk):
    """The SDK's own event accumulator is far stricter than a shape check: it
    rejects an out-of-order or misnamed event outright."""
    with sdk.messages.stream(model="local", max_tokens=16, temperature=0,
                             messages=[{"role": "user", "content": "hello"}]) as s:
        kinds = [ev.type for ev in s]
        final = s.get_final_message()
    for required in ("message_start", "content_block_start",
                     "content_block_stop", "message_delta", "message_stop"):
        if required not in kinds:
            raise ProtocolError("SDK never saw a required event",
                                missing=required, got=kinds)
    if final.stop_reason not in ("end_turn", "max_tokens", "stop_sequence",
                                 "tool_use"):
        raise ProtocolError("SDK accumulated an invalid stop_reason",
                            got=final.stop_reason)


@needs_sdk
def test_real_sdk_reads_a_tool_use_block(sdk):
    m = sdk.messages.create(model="local", max_tokens=48, temperature=0,
                            tools=[WEATHER], tool_choice={"type": "any"},
                            messages=[{"role": "user", "content": "weather in Oslo?"}])
    uses = [b for b in m.content if b.type == "tool_use"]
    if not uses:
        raise ProtocolError("SDK found no ToolUseBlock", content=repr(m.content))
    if uses[0].name != "get_weather" or not isinstance(uses[0].input, dict):
        raise ProtocolError("SDK deserialised a malformed tool_use",
                            block=repr(uses[0]))
    if m.stop_reason != "tool_use":
        raise ProtocolError("SDK saw the wrong stop_reason for a call",
                            got=m.stop_reason)


@needs_sdk
def test_real_sdk_count_tokens(sdk):
    ct = sdk.messages.count_tokens(
        model="local", messages=[{"role": "user", "content": "hello there"}])
    if not isinstance(ct.input_tokens, int) or ct.input_tokens <= 0:
        raise ProtocolError("SDK did not deserialise MessageTokensCount",
                            got=repr(ct))


@needs_sdk
def test_real_sdk_surfaces_a_refusal_as_an_api_error(sdk):
    """An unsupported feature must reach the client as a typed 400, not as a
    parse failure or a silent success."""
    with pytest.raises(_anthropic.APIStatusError) as e:
        sdk.messages.create(model="local", max_tokens=8,
                            messages=[{"role": "user", "content": "hi"}],
                            extra_body={"container": "cnt_1"})
    if e.value.status_code != 400:
        raise ProtocolError("refusal did not arrive as a 400",
                            got=e.value.status_code)
    if "container" not in str(e.value):
        raise ProtocolError("the SDK error does not name the refused field",
                            got=str(e.value)[:200])


# ------------------------------------------------------- surface equivalence
def test_messages_and_chat_agree_on_the_same_turn(client):
    """Phase 4's exit criterion: the OpenAI and Anthropic surfaces must
    produce equivalent internal agent actions, not merely both return 200."""
    prompt = "what is the weather in Oslo?"
    anth = client.messages(
        {"messages": [{"role": "user", "content": prompt}], "max_tokens": 48,
         "temperature": 0, "tools": [WEATHER],
         "tool_choice": {"type": "any"}},
        name="messages-equiv").expect_status(200).json
    chat = client.chat(
        {"messages": [{"role": "user", "content": prompt}], "max_tokens": 48,
         "temperature": 0,
         "tools": [{"type": "function",
                    "function": {"name": WEATHER["name"],
                                 "description": WEATHER["description"],
                                 "parameters": WEATHER["input_schema"]}}],
         "tool_choice": "required"},
        name="messages-equiv-chat").expect_status(200).json
    a_use = [b for b in anth["content"] if b["type"] == "tool_use"]
    c_calls = chat["choices"][0]["message"].get("tool_calls") or []
    if not a_use or not c_calls:
        raise ProtocolError("the two surfaces disagreed about calling a tool",
                            anthropic=anth["content"], chat=chat["choices"][0])
    if a_use[0]["name"] != c_calls[0]["function"]["name"]:
        raise ProtocolError("the two surfaces called different tools",
                            anthropic=a_use[0]["name"],
                            chat=c_calls[0]["function"]["name"])
    if a_use[0]["input"] != json.loads(c_calls[0]["function"]["arguments"]):
        raise ProtocolError("the two surfaces built different arguments",
                            anthropic=a_use[0]["input"],
                            chat=c_calls[0]["function"]["arguments"])
    # the terminal reasons are the same fact in two vocabularies
    if (anth["stop_reason"], chat["choices"][0]["finish_reason"]) != \
            ("tool_use", "tool_calls"):
        raise ProtocolError("the two surfaces reported different terminations",
                            anthropic=anth["stop_reason"],
                            chat=chat["choices"][0]["finish_reason"])


def test_messages_accepts_claude_code_beta_query(client):
    """Claude Code appends `?beta=true` to the Messages request target."""
    payload = {"model": "runner", "max_tokens": 8,
               "messages": [{"role": "user", "content": "say hi"}]}
    r = client.raw("messages-beta-query", "POST", "/v1/messages?beta=true", payload)
    r.expect_status(200)
    assert r.json["type"] == "message"


def test_messages_accepts_claude_code_adaptive_thinking(client):
    """Adaptive thinking is a hint, not a promise that every model thinks."""
    payload = {"model": "runner", "max_tokens": 8,
               "thinking": {"type": "adaptive", "display": "omitted"},
               "messages": [{"role": "user", "content": "say hi"}]}
    r = client.messages(payload, name="messages-adaptive-thinking")
    r.expect_status(200)
    assert r.json["type"] == "message"


def test_messages_accepts_claude_code_system_turn(client):
    payload = {"model": "runner", "max_tokens": 8,
               "system": "provider system",
               "messages": [
                   {"role": "user", "content": "say hi"},
                   {"role": "system", "content": [
                       {"type": "text", "text": "client harness context"}]},
               ]}
    r = client.messages(payload, name="messages-system-turn")
    r.expect_status(200)
    assert r.json["type"] == "message"
