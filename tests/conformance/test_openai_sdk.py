"""The official OpenAI SDK against runner's OpenAI-compatible surface.

The rest of the suite talks to runner with `urllib` and asserts on the JSON it
gets back. That proves the bytes are what we intended; it does not prove a real
client can use them. A hand-written check accepts a field the SDK's model would
reject, tolerates an event order the SDK's stream accumulator would not, and
never exercises the SDK's own error typing at all. Phase 3 found three bugs on
the Anthropic side that only the real SDK exposed — this is the same check for
the OpenAI side, which until now had none.

Skipped when `openai` is not installed, deliberately: the suite must not gain a
network-installed dependency in CI (the same decision `test_messages.py`
records for `anthropic`). A developer who has one gets the stronger check.

Every test here goes through an SDK method rather than a raw request, because
the point is the SDK's deserialisation and not ours.
"""

import pytest

from _errors import ProtocolError

try:
    import openai as _openai
except ImportError:  # pragma: no cover - depends on the environment
    _openai = None

needs_sdk = pytest.mark.skipif(_openai is None,
                               reason="the openai SDK is not installed")

WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Look up the weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        },
    },
}

PERSON = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
    "required": ["name", "age"],
    "additionalProperties": False,
}


@pytest.fixture(scope="module")
def sdk(server):
    if _openai is None:
        pytest.skip("the openai SDK is not installed")
    return _openai.OpenAI(base_url=f"http://127.0.0.1:{server.port}/v1",
                          api_key="not-used", max_retries=0)


@pytest.fixture(scope="module")
def model(sdk):
    """The served id, read from the server rather than assumed.

    A hardcoded name would make every test below pass or fail for a reason that
    has nothing to do with the SDK.
    """
    ids = [m.id for m in sdk.models.list().data]
    if not ids:
        raise ProtocolError("the SDK deserialised an empty model list")
    return ids[0]


@needs_sdk
def test_sdk_lists_models(sdk):
    page = sdk.models.list()
    if not page.data:
        raise ProtocolError("SDK found no models", got=repr(page))
    first = page.data[0]
    # object/id/owned_by are required by the SDK's Model shape; a missing one
    # deserialises to something unusable rather than raising, so check it.
    if first.object != "model" or not first.id:
        raise ProtocolError("SDK deserialised a malformed Model", got=repr(first))


@needs_sdk
def test_sdk_parses_a_buffered_chat_turn(sdk, model):
    c = sdk.chat.completions.create(
        model=model, max_tokens=8, temperature=0,
        messages=[{"role": "user", "content": "hello"}])
    if c.object != "chat.completion" or not c.choices:
        raise ProtocolError("SDK did not deserialise a ChatCompletion", got=repr(c))
    choice = c.choices[0]
    if choice.message.role != "assistant":
        raise ProtocolError("SDK deserialised the wrong role", got=choice.message.role)
    # The OpenAI enum, and nothing else. "reasoning_limit" used to be accepted
    # here: it is runner's own value (`2866c89`), distinguishing a budget
    # exhausted while a thinking model was still in its prelude from ordinary
    # truncation. This test tolerated it so it would not fail for anyone
    # pointing RUNNER_TEST_MODEL at a thinking model — which meant the one test
    # positioned to catch the divergence was the one waived for it.
    #
    # Owner decision 2026-08-08: the chat surface now emits the standard
    # "length" and carries the distinction as runner_telemetry.finish_detail
    # (see openai_finish() in src/completion.c), matching what the Responses
    # and Anthropic surfaces already did. So the tolerance is removed and this
    # assertion is now strict — a thinking model must not be able to push a
    # non-OpenAI value onto an OpenAI-shaped wire.
    if choice.finish_reason not in ("stop", "length", "tool_calls",
                                    "content_filter"):
        raise ProtocolError("SDK deserialised an invalid finish_reason",
                            got=choice.finish_reason)
    if not isinstance(c.usage.prompt_tokens, int) or c.usage.prompt_tokens <= 0:
        raise ProtocolError("SDK did not deserialise usage", usage=repr(c.usage))


@needs_sdk
def test_sdk_accumulates_a_chat_stream(sdk, model):
    """Chunk shape under the SDK's own parser, including the usage chunk.

    `stream_options={"include_usage": True}` is the case most likely to be
    wrong: it adds a final chunk whose `choices` is empty, and a server that
    omits it or fills it in leaves the SDK with either no usage or a phantom
    choice.
    """
    stream = sdk.chat.completions.create(
        model=model, max_tokens=16, temperature=0, stream=True,
        stream_options={"include_usage": True},
        messages=[{"role": "user", "content": "hello"}])
    text, usage, roles = "", None, []
    for chunk in stream:
        if chunk.object != "chat.completion.chunk":
            raise ProtocolError("SDK deserialised a non-chunk in the stream",
                                got=chunk.object)
        if chunk.usage is not None:
            usage = chunk.usage
        for ch in chunk.choices:
            if ch.delta.role:
                roles.append(ch.delta.role)
            if ch.delta.content:
                text += ch.delta.content
    if not text:
        raise ProtocolError("the SDK accumulated no content from the stream")
    if roles and roles[0] != "assistant":
        raise ProtocolError("the first delta named the wrong role", got=roles)
    if usage is None or not isinstance(usage.total_tokens, int):
        raise ProtocolError("include_usage produced no usable usage chunk",
                            got=repr(usage))


@needs_sdk
def test_sdk_reads_a_tool_call(sdk, model):
    c = sdk.chat.completions.create(
        model=model, max_tokens=48, temperature=0,
        tools=[WEATHER], tool_choice="required",
        messages=[{"role": "user", "content": "weather in Oslo?"}])
    calls = c.choices[0].message.tool_calls
    if not calls:
        raise ProtocolError("SDK found no tool_calls",
                            message=repr(c.choices[0].message))
    call = calls[0]
    if call.type != "function" or call.function.name != "get_weather":
        raise ProtocolError("SDK deserialised a malformed tool call", got=repr(call))
    # arguments is a JSON *string* in this API, not an object. An SDK that got
    # an object here would have raised; this asserts we did not send one.
    if not isinstance(call.function.arguments, str):
        raise ProtocolError("tool call arguments must be a JSON string",
                            got=type(call.function.arguments).__name__)
    if c.choices[0].finish_reason != "tool_calls":
        raise ProtocolError("a tool call ended with the wrong finish_reason",
                            got=c.choices[0].finish_reason)


@needs_sdk
def test_sdk_round_trips_a_tool_result(sdk, model):
    """The second turn is where an OpenAI-shaped history is easiest to reject:
    it carries an assistant message with `tool_calls` and no content, then a
    `role: tool` message keyed by id."""
    first = sdk.chat.completions.create(
        model=model, max_tokens=48, temperature=0,
        tools=[WEATHER], tool_choice="required",
        messages=[{"role": "user", "content": "weather in Oslo?"}])
    call = first.choices[0].message.tool_calls[0]
    second = sdk.chat.completions.create(
        model=model, max_tokens=32, temperature=0, tools=[WEATHER],
        messages=[
            {"role": "user", "content": "weather in Oslo?"},
            {"role": "assistant", "tool_calls": [
                {"id": call.id, "type": "function",
                 "function": {"name": call.function.name,
                              "arguments": call.function.arguments}}]},
            {"role": "tool", "tool_call_id": call.id, "content": "12C, raining"},
        ])
    if not second.choices:
        raise ProtocolError("the tool-result turn produced no choice")


@needs_sdk
def test_sdk_structured_output(sdk, model):
    import json

    c = sdk.chat.completions.create(
        model=model, max_tokens=64, temperature=0,
        response_format={"type": "json_schema",
                         "json_schema": {"name": "person", "schema": PERSON}},
        messages=[{"role": "user", "content": "invent a person"}])
    content = c.choices[0].message.content or ""
    try:
        doc = json.loads(content)
    except ValueError as exc:
        raise ProtocolError("json_schema output is not JSON",
                            got=content[:200], error=str(exc)) from exc
    missing = [k for k in ("name", "age") if k not in doc]
    if missing:
        raise ProtocolError("json_schema output is missing required keys",
                            missing=missing, got=content[:200])


@needs_sdk
def test_sdk_legacy_completions(sdk, model):
    c = sdk.completions.create(model=model, prompt="hello", max_tokens=8,
                               temperature=0)
    if c.object != "text_completion" or not c.choices:
        raise ProtocolError("SDK did not deserialise a Completion", got=repr(c))
    if not isinstance(c.choices[0].text, str):
        raise ProtocolError("SDK deserialised no text", got=repr(c.choices[0]))


@needs_sdk
def test_sdk_embeddings(sdk, model):
    e = sdk.embeddings.create(model=model, input=["a car", "a banana"])
    if len(e.data) != 2:
        raise ProtocolError("SDK deserialised the wrong number of embeddings",
                            got=len(e.data))
    widths = {len(row.embedding) for row in e.data}
    if len(widths) != 1:
        raise ProtocolError("embeddings came back at differing widths", got=widths)
    if [row.index for row in e.data] != [0, 1]:
        raise ProtocolError("embedding rows are not indexed in order",
                            got=[row.index for row in e.data])


@needs_sdk
def test_sdk_surfaces_a_refusal_as_a_typed_error(sdk, model):
    """A rejected request must reach the client as a typed status error, not as
    a deserialisation failure or a silent success."""
    with pytest.raises(_openai.APIStatusError) as e:
        sdk.chat.completions.create(
            model=model, max_tokens=8,
            messages=[{"role": "user", "content": "hi"}],
            extra_body={"n": 4})
    if e.value.status_code != 400:
        raise ProtocolError("refusal did not arrive as a 400",
                            got=e.value.status_code)


@needs_sdk
def test_sdk_reports_an_unknown_model_as_404(sdk):
    with pytest.raises(_openai.APIStatusError) as e:
        sdk.chat.completions.create(
            model="definitely-not-served", max_tokens=8,
            messages=[{"role": "user", "content": "hi"}])
    if e.value.status_code not in (400, 404):
        raise ProtocolError("an unknown model was not refused",
                            got=e.value.status_code)
