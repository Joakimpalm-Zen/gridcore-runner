"""Streaming chat completions: SSE framing, the token-boundary matrix,
client disconnect, and the chunk-shape gaps Phase 2 is expected to close."""

import json

import pytest

from harness import ProtocolError, SSEParser, decode_events, parse_stream, split_points

BASE = {"messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 8, "temperature": 0}


def test_stream_framing(client, report):
    st = client.chat_stream(dict(BASE), name="stream-basic").expect_sse()
    for c in st.chunks:
        if c.get("object") != "chat.completion.chunk":
            raise ProtocolError("streamed chunk has wrong object type",
                                got=c.get("object"), chunk=c)
        if not c.get("id"):
            raise ProtocolError("streamed chunk has no id", chunk=c)
        if not isinstance(c.get("choices"), list):
            raise ProtocolError("streamed chunk has no choices array", chunk=c)
    if st.finish_reason not in ("stop", "length"):
        raise ProtocolError("stream ended with no usable finish_reason",
                            got=st.finish_reason)
    # exactly one chunk carries a finish_reason, and it is the last one
    fins = [i for i, c in enumerate(st.chunks)
            if any(ch.get("finish_reason") for ch in c.get("choices", []))]
    if fins != [len(st.chunks) - 1]:
        raise ProtocolError("finish_reason must appear once, on the final chunk",
                            at=fins, total=len(st.chunks))
    report.check_fixture("chat_stream_chunk", st.chunks[0])
    report.check_fixture("chat_stream_final", st.chunks[-1])


def test_stream_matches_buffered(client):
    """The same greedy request must produce the same text either way. A
    streaming path that drops or duplicates a token is invisible otherwise."""
    buffered = client.chat(dict(BASE, max_tokens=16), name="stream-vs-buffered-b")
    st = client.chat_stream(dict(BASE, max_tokens=16), name="stream-vs-buffered-s")
    st.expect_sse()
    if st.text != buffered.content:
        raise ProtocolError("streamed text differs from buffered text",
                            buffered=buffered.content, streamed=st.text)
    if st.finish_reason != buffered.finish_reason:
        raise ProtocolError("streamed finish_reason differs from buffered",
                            buffered=buffered.finish_reason,
                            streamed=st.finish_reason)


def test_stream_options_include_usage(client):
    st = client.chat_stream(dict(BASE, stream_options={"include_usage": True}),
                            name="stream-usage").expect_sse()
    usage = [c["usage"] for c in st.chunks if "usage" in c]
    if len(usage) != 1:
        raise ProtocolError("include_usage must add exactly one usage chunk",
                            got=len(usage))
    u = usage[0]
    if u["prompt_tokens"] <= 0 or u["total_tokens"] != u["prompt_tokens"] + u["completion_tokens"]:
        raise ProtocolError("streamed usage is not self-consistent", usage=u)
    if st.chunks[-1].get("usage") is not usage[0] or st.chunks[-1].get("choices") != []:
        raise ProtocolError("usage chunk must be last and carry empty choices",
                            last=st.chunks[-1])


# ------------------------------------------------------- boundary matrix
@pytest.fixture(scope="module")
def recorded_stream(client):
    """One real SSE byte stream, captured once and replayed by the matrix."""
    st = client.chat_stream(dict(BASE, max_tokens=12,
                                 stream_options={"include_usage": True}),
                            name="stream-for-boundary-matrix")
    st.expect_sse()
    return st.raw


def test_every_split_point_parses_identically(recorded_stream):
    """Phase 0: split the stream at EVERY possible byte boundary.

    A client that reassembles differently depending on how TCP happened to
    segment the response is broken in a way that only shows up under load, so
    every split of the byte string must yield the identical event list."""
    raw = recorded_stream
    reference = parse_stream(raw)
    if not reference:
        raise ProtocolError("recorded stream parsed to nothing", bytes=len(raw))
    for chunks in split_points(raw):
        got = parse_stream(raw, chunks)
        if got != reference:
            at = len(chunks[0]) if len(chunks) == 2 else "byte-at-a-time"
            raise ProtocolError("parse depends on chunk boundaries",
                                split_at=at, expected=len(reference),
                                got=len(got),
                                first_difference=_first_diff(reference, got))


def test_every_split_point_decodes_identically(recorded_stream):
    """Same matrix, one level up: the decoded chunk objects and the [DONE]
    terminator must also be boundary-independent."""
    raw = recorded_stream
    ref_chunks, ref_done = decode_events(parse_stream(raw))
    for chunks in split_points(raw):
        got_chunks, got_done = decode_events(parse_stream(raw, chunks))
        if got_chunks != ref_chunks or got_done != ref_done:
            at = len(chunks[0]) if len(chunks) == 2 else "byte-at-a-time"
            raise ProtocolError("decoded stream depends on chunk boundaries",
                                split_at=at)


def test_partial_event_is_never_dispatched(recorded_stream):
    """Every strict prefix of the stream must parse to a prefix of the events.

    This is the property a real client depends on: bytes received so far may
    only ever yield events that are genuinely complete. Half of an event must
    never surface as a whole one."""
    raw = recorded_stream
    reference = parse_stream(raw)
    for i in range(len(raw) + 1):
        p = SSEParser()
        p.feed(raw[:i])
        got = p.close()
        if got != reference[:len(got)]:
            raise ProtocolError("prefix parse is not a prefix of the full parse",
                                prefix_bytes=i, got=len(got))


def test_malformed_sse_is_an_error_not_a_skip(recorded_stream):
    """FUTURE.md, "Python client defects": swallowing a bad ``data:`` line
    turns protocol corruption into output that looks complete. The harness
    must raise instead."""
    corrupted = recorded_stream.replace(b"data: {", b"data: {,", 1)
    with pytest.raises(ProtocolError):
        decode_events(parse_stream(corrupted))


def _first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return {"index": i, "expected": x[:120], "got": y[:120]}
    return {"index": min(len(a), len(b)), "expected": None, "got": None}


# ------------------------------------------------------------ disconnect
def test_client_disconnect_mid_stream_releases_the_slot(client, server):
    """Hang up partway through an SSE response.

    The server must notice the dead socket, stop generating, release the slot
    and stay healthy. A leaked slot here is invisible until the next request
    blocks forever, which is why the assertion is "the next requests still
    work", not "no error was logged"."""
    partial, _, _ = client.stream_raw(
        "stream-disconnect", "/v1/chat/completions",
        dict(BASE, max_tokens=256, stream=True), close_after_bytes=1)
    if not partial:
        raise ProtocolError("no bytes arrived before disconnecting")
    if b"200" not in partial.split(b"\r\n", 1)[0]:
        raise ProtocolError("stream did not start OK before disconnect",
                            head=partial[:80].decode("utf-8", "replace"))
    server.assert_alive()
    # every slot must still be usable: run more requests than there are slots
    for i in range(server.parallel + 1):
        client.get("/health").expect_status(200)
        client.chat(dict(BASE, max_tokens=4),
                    name=f"post-disconnect-{i}").expect_status(200)


# ------------------------------------------------------- streamed tool calls
TOOL = {"type": "function",
        "function": {"name": "f", "description": "d",
                     "parameters": {"type": "object",
                                    "properties": {"x": {"type": "string"}},
                                    "required": ["x"]}}}


def collect_tool_calls(st):
    """Reassemble streamed tool_calls the way an SDK does: an opening event
    per index carrying the identity, then argument text keyed by that index."""
    opened, args = {}, {}
    for d in st.deltas():
        for call in d.get("tool_calls") or []:
            i = call.get("index")
            if not isinstance(i, int):
                raise ProtocolError("tool_calls delta has no integer index",
                                    got=call)
            fn = call.get("function") or {}
            if i not in opened:
                if not call.get("id") or call.get("type") != "function" \
                        or not fn.get("name"):
                    raise ProtocolError(
                        "the first delta for a tool call must carry id, type "
                        "and function.name", index=i, got=call)
                opened[i] = {"id": call["id"], "name": fn["name"]}
                args[i] = ""
            elif call.get("id") or call.get("type"):
                raise ProtocolError("identity fields repeated on a later "
                                    "tool_calls delta", index=i, got=call)
            args[i] += fn.get("arguments") or ""
    return [dict(opened[i], arguments=args[i]) for i in sorted(opened)]


def test_stream_emits_tool_calls_with_incremental_arguments(client):
    """Phase 2 (was a known gap).

    A streamed tool call must arrive as tool_calls deltas — identity once,
    then argument text — and end with finish_reason "tool_calls". Streaming
    the internal envelope as ordinary content, which is what runner used to
    do, silently hands the caller a string instead of a call."""
    st = client.chat_stream(
        dict(BASE, max_tokens=32, tools=[TOOL], tool_choice="required"),
        name="stream-tools").expect_sse()

    if st.finish_reason != "tool_calls":
        raise ProtocolError("a guaranteed call must end the stream with "
                            "finish_reason \"tool_calls\"",
                            got=st.finish_reason)
    calls = collect_tool_calls(st)
    if len(calls) != 1:
        raise ProtocolError("expected exactly one streamed tool call",
                            got=calls)
    if calls[0]["name"] != "f":
        raise ProtocolError("streamed call names an undeclared tool",
                            got=calls[0]["name"])
    try:
        json.loads(calls[0]["arguments"])
    except ValueError as exc:
        raise ProtocolError("streamed arguments do not parse as JSON",
                            arguments=calls[0]["arguments"][:200],
                            error=str(exc)) from exc
    # the envelope is internal: not one byte of it may reach `content`
    if st.text:
        raise ProtocolError("content was streamed alongside a tool call",
                            got=st.text[:200])


def test_stream_never_leaks_the_envelope_into_content(client):
    """"auto" may land on either branch. Whichever it picks, the client sees
    a normal answer or a normal call — never the discriminated union runner
    uses internally to guarantee them."""
    st = client.chat_stream(
        dict(BASE, max_tokens=32, tools=[TOOL], tool_choice="auto"),
        name="stream-tools-auto").expect_sse()
    for marker in ('{"tool"', '"args"', '<|tool_call>'):
        if marker in st.text:
            raise ProtocolError("internal envelope syntax leaked into content",
                                marker=marker, got=st.text[:200])
    calls = collect_tool_calls(st)
    if calls and st.finish_reason != "tool_calls":
        raise ProtocolError("streamed tool_calls but finish_reason is not "
                            "\"tool_calls\"", got=st.finish_reason)
    if not calls and st.finish_reason == "tool_calls":
        raise ProtocolError("finish_reason \"tool_calls\" with no streamed call")


def test_stream_tool_call_survives_every_split_point(client):
    """The boundary matrix, applied to a stream that carries a tool call:
    argument deltas are the part most likely to be reassembled differently
    depending on where the bytes were cut."""
    st = client.chat_stream(
        dict(BASE, max_tokens=32, tools=[TOOL], tool_choice="required"),
        name="stream-tools-matrix").expect_sse()
    reference = parse_stream(st.raw)
    for chunks in split_points(st.raw):
        if parse_stream(st.raw, chunks) != reference:
            at = len(chunks[0]) if len(chunks) == 2 else "byte-at-a-time"
            raise ProtocolError("a tool-call stream parses differently "
                                "depending on chunk boundaries", split_at=at)


def test_stream_chunk_carries_created_model_and_role(client):
    """Phase 2 (was a known gap).

    OpenAI's chat.completion.chunk carries ``created`` and ``model`` on every
    chunk and ``role: "assistant"`` on the first delta. SDKs that key off the
    first delta's role (or that echo the model back) reject a stream without
    them, so all three are required here rather than merely tolerated."""
    buffered = client.chat(dict(BASE), name="stream-chunk-fields-b")
    st = client.chat_stream(dict(BASE), name="stream-chunk-fields").expect_sse()
    for c in st.chunks:
        if not isinstance(c.get("created"), int) or c["created"] <= 0:
            raise ProtocolError("streamed chunk has no usable created",
                                got=c.get("created"), chunk=c)
        if not isinstance(c.get("model"), str) or not c["model"]:
            raise ProtocolError("streamed chunk has no model", chunk=c)
        if c["model"] != buffered.json["model"]:
            raise ProtocolError("streamed model differs from the buffered one",
                                buffered=buffered.json["model"],
                                streamed=c["model"])
    role = st.chunks[0]["choices"][0].get("delta", {}).get("role")
    if role != "assistant":
        raise ProtocolError("first delta must carry role \"assistant\"",
                            got=role, chunk=st.chunks[0])
    # ...and only the first: a repeated role confuses accumulating clients
    later = [i for i, d in enumerate(st.deltas()[1:], 1) if "role" in d]
    if later:
        raise ProtocolError("role repeated after the first delta", at=later)


def test_stream_flag_must_be_boolean(client):
    """KNOWN CURRENT BEHAVIOUR (and a deliberate one): "stream":"true" used to
    read as false and answer with a buffered body, leaving an SSE client
    waiting forever. It must 400."""
    for bad in ("true", 1, 0, [], {}):
        client.expect_400(dict(BASE, stream=bad), name=f"stream-nonbool-{bad!r}",
                          contains="stream")
    # ...but a real boolean false is fine and means "buffered"
    r = client.chat(dict(BASE, stream=False), name="stream-false")
    r.expect_status(200)
    if r.json.get("object") != "chat.completion":
        raise ProtocolError("stream:false did not produce a buffered response",
                            got=r.json.get("object"))


def test_stream_error_is_reported_before_the_stream_starts(client):
    """A request rejected at validation time must fail as a normal HTTP error,
    not as a 200 SSE stream carrying an error event — clients branch on the
    status code long before they parse events."""
    raw, _, _ = client.stream_raw(
        "stream-rejected", "/v1/chat/completions",
        dict(BASE, stream=True, response_format={"type": "nonsense"}))
    head = raw.split(b"\r\n", 1)[0]
    if b"400" not in head:
        raise ProtocolError("invalid streaming request did not fail with 400",
                            head=head.decode("utf-8", "replace"))
    if b"text/event-stream" in raw:
        raise ProtocolError("rejected request still opened an SSE stream")
    body = raw.split(b"\r\n\r\n", 1)[1]
    if "error" not in json.loads(body):
        raise ProtocolError("rejection body is not an error envelope",
                            body=body[:200].decode("utf-8", "replace"))
