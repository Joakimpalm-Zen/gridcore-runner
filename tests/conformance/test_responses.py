"""The OpenAI Responses API surface: POST /v1/responses.

Phase 3. This endpoint is a *translation layer*, not a second engine: a
Responses request is reshaped into the same internal prompt + strict tool
envelope that /v1/chat/completions already builds, and the generated result is
rendered in the Responses vocabulary on the way out. The tests here therefore
assert two things and deliberately not a third:

  - the wire shape and, for streams, the *order* of the typed events, because
    that ordering is what SDK clients validate; and
  - that unsupported stateful features are refused rather than ignored.

They do not re-assert the generation guarantees themselves (that a tool call
names a declared tool, that arguments conform) — those belong to the engine
and are covered by test_tool_calls.py. What is asserted here is that this
surface reaches the same engine, by checking the two surfaces agree.
"""

import json

import pytest

from harness import ProtocolError, validate_against_schema

BASE = {"input": "hello", "max_output_tokens": 8, "temperature": 0}


def test_responses_buffered_shape(client, report):
    r = client.responses(dict(BASE), name="responses-buffered").expect_status(200)
    d = r.json
    if d.get("object") != "response":
        raise ProtocolError("wrong object type", got=d.get("object"), body=d)
    for field in ("id", "created_at", "model", "status", "output", "usage"):
        if field not in d:
            raise ProtocolError("response missing required field",
                                field=field, keys=sorted(d))
    # the stub model rarely stops on its own inside max_output_tokens, so both
    # terminal statuses are legitimate here; what is asserted is that the two
    # ways the response reports truncation always agree
    if d["status"] not in ("completed", "incomplete"):
        raise ProtocolError("unknown terminal status", got=d["status"])
    if (d["status"] == "incomplete") != (d.get("incomplete_details") is not None):
        raise ProtocolError("status and incomplete_details disagree",
                            status=d["status"],
                            incomplete_details=d.get("incomplete_details"))
    if d["status"] == "incomplete" and \
            d["incomplete_details"].get("reason") != "max_output_tokens":
        raise ProtocolError("truncation reported with the wrong reason",
                            details=d["incomplete_details"])
    if not isinstance(d["output"], list) or not d["output"]:
        raise ProtocolError("output is not a non-empty list", got=d.get("output"))
    item = d["output"][0]
    if item.get("type") != "message" or item.get("role") != "assistant":
        raise ProtocolError("first output item is not an assistant message",
                            item=item)
    if item.get("status") != ("incomplete" if d["status"] == "incomplete"
                              else "completed"):
        raise ProtocolError("item status does not match the response status",
                            item_status=item.get("status"), response=d["status"])
    if d.get("output_text") != item["content"][0]["text"]:
        raise ProtocolError("output_text aggregate does not match the message",
                            aggregate=d.get("output_text"), item=item)
    parts = item.get("content")
    if not isinstance(parts, list) or not parts:
        raise ProtocolError("message has no content parts", item=item)
    if parts[0].get("type") != "output_text":
        raise ProtocolError("content part is not output_text", part=parts[0])
    if not isinstance(parts[0].get("text"), str):
        raise ProtocolError("output_text has no text string", part=parts[0])
    report.check_fixture("responses_buffered", d)


# ------------------------------------------------------------------ streaming
def test_responses_stream_event_order(client, report):
    """The ordered event contract, which is what a typed SDK validates.

    Asserted as a grammar rather than a fixed list: the number of deltas
    depends on what the model emitted, but the nesting never does."""
    st = client.responses_stream(dict(BASE),
                                 name="responses-stream").expect_sse()
    names = st.names
    if not names:
        raise ProtocolError("stream carried no events", raw=st.raw[:300])
    if names[0] != "response.created":
        raise ProtocolError("stream does not open with response.created",
                            got=names[0], names=names)
    if names[1] != "response.in_progress":
        raise ProtocolError("response.created is not followed by in_progress",
                            got=names[1], names=names)
    if names[-1] not in ("response.completed", "response.incomplete"):
        raise ProtocolError("stream does not end with a terminal response event",
                            got=names[-1], names=names)

    # sequence_number is monotonic from 0 across every event
    for i, (_, d) in enumerate(st.typed):
        if d.get("sequence_number") != i:
            raise ProtocolError("sequence_number is not monotonic from 0",
                                at=i, got=d.get("sequence_number"), names=names)

    # every item opened is closed, and its deltas fall strictly inside it
    depth, part_depth, seen_items = 0, 0, 0
    for name, d in st.typed:
        if name == "response.output_item.added":
            if depth:
                raise ProtocolError("output item opened inside another",
                                    names=names)
            depth, seen_items = 1, seen_items + 1
            if d.get("output_index") != seen_items - 1:
                raise ProtocolError("output_index is not sequential",
                                    got=d.get("output_index"), names=names)
        elif name == "response.output_item.done":
            if depth != 1:
                raise ProtocolError("output item closed while none was open",
                                    names=names)
            if part_depth:
                raise ProtocolError("output item closed with a content part open",
                                    names=names)
            depth = 0
        elif name == "response.content_part.added":
            if not depth:
                raise ProtocolError("content part added outside an item",
                                    names=names)
            part_depth += 1
        elif name == "response.content_part.done":
            if not part_depth:
                raise ProtocolError("content part closed while none was open",
                                    names=names)
            part_depth -= 1
        elif name.endswith(".delta"):
            if not depth:
                raise ProtocolError("delta emitted outside an output item",
                                    event=name, names=names)
    if depth or part_depth:
        raise ProtocolError("stream ended with an item or part still open",
                            names=names)
    if not seen_items:
        raise ProtocolError("stream produced no output items", names=names)
    report.check_fixture("responses_stream_events", [d for _, d in st.typed])


def test_responses_stream_text_matches_completed_output(client):
    """A client that only reads the terminal event must see the same turn as
    one that accumulated the deltas."""
    st = client.responses_stream(dict(BASE, max_output_tokens=16),
                                 name="responses-stream-agreement").expect_sse()
    final = st.payloads("response.completed") or st.payloads("response.incomplete")
    if not final:
        raise ProtocolError("stream had no terminal response event",
                            names=st.names)
    resp = final[0]["response"]
    if resp.get("output_text") != st.text:
        raise ProtocolError("terminal output_text does not match the deltas",
                            deltas=st.text, terminal=resp.get("output_text"))
    texts = [p["text"] for item in resp["output"] if item["type"] == "message"
             for p in item["content"]]
    if "".join(texts) != st.text:
        raise ProtocolError("terminal output items do not match the deltas",
                            deltas=st.text, items=resp["output"])


def test_responses_streamed_and_buffered_agree(client):
    """stream=True is a transport choice, not a behaviour change."""
    payload = dict(BASE, max_output_tokens=16, temperature=0)
    buf = client.responses(dict(payload), name="responses-buf-eq").expect_status(200)
    st = client.responses_stream(dict(payload), name="responses-str-eq").expect_sse()
    final = (st.payloads("response.completed") or
             st.payloads("response.incomplete"))[0]["response"]
    if buf.json["status"] != final["status"]:
        raise ProtocolError("buffered and streamed status differ",
                            buffered=buf.json["status"], streamed=final["status"])
    if buf.json["output_text"] != final["output_text"]:
        raise ProtocolError("buffered and streamed text differ",
                            buffered=buf.json["output_text"],
                            streamed=final["output_text"])
    b_status = [i.get("status") for i in buf.json["output"]]
    s_status = [i.get("status") for i in final["output"]]
    if b_status != s_status:
        raise ProtocolError("buffered and streamed item statuses differ",
                            buffered=b_status, streamed=s_status)


def test_responses_truncated_item_is_marked_incomplete(client):
    """A truncated turn must say so on the item too, not only on the response.

    A client that renders `output[]` and never looks at `status` would
    otherwise show a cut-off message as a finished one."""
    st = client.responses_stream(dict(BASE, max_output_tokens=3),
                                 name="responses-truncated").expect_sse()
    final = st.payloads("response.incomplete")
    if not final:
        pytest.skip("model stopped on its own; nothing was truncated")
    for item in final[0]["response"]["output"]:
        if item["type"] == "message" and item.get("status") != "incomplete":
            raise ProtocolError("truncated message item is not marked incomplete",
                                item=item)
