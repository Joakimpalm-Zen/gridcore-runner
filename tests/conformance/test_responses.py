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
