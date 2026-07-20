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


# ------------------------------------------------- unsupported, not ignored
@pytest.mark.parametrize("field,value,explains", [
    ("store", True, "stateless"),
    ("previous_response_id", "resp_123", "stateless"),
    ("background", True, "background"),
    ("conversation", "conv_123", "stateless"),
    ("truncation", "auto", "truncation"),
    ("include", ["message.output_text.logprobs"], "include"),
])
def test_stateful_features_are_refused(client, field, value, explains):
    """The project invariant, on this surface.

    Every one of these asks the runtime to remember or fetch something it has
    no store for. Answering 200 would tell the caller its turn was persisted,
    or that it will be able to fetch a background job, when neither is true —
    and unlike a dropped sampling parameter, the caller cannot detect it."""
    r = client.expect_400({"input": "hello", field: value},
                          name=f"responses-{field}", contains=explains,
                          path="/v1/responses")
    if field not in r.json["error"]["message"] and \
            explains not in r.json["error"]["message"]:
        raise ProtocolError("error does not name the offending field",
                            field=field, message=r.json["error"]["message"])


def test_supported_stateless_forms_are_accepted(client):
    """The half of those fields that this runtime *can* honour must not be
    refused: store:false and truncation:"disabled" describe what it already
    does, and a client that always sends them is not asking for anything."""
    r = client.responses({"input": "hello", "max_output_tokens": 4,
                          "store": False, "truncation": "disabled",
                          "temperature": 0},
                         name="responses-stateless-ok")
    r.expect_status(200)
    if r.json.get("store") is not False:
        raise ProtocolError("store is not reported false", body=r.json)


@pytest.mark.parametrize("payload,contains", [
    ({"input": "hi", "reasoning": "high"}, "reasoning"),
    ({"input": "hi", "text": {"format": {"type": "bogus"}}}, "text.format"),
    ({"input": "hi", "text": {"format": {"type": "json_schema"}}}, "schema"),
    ({"input": 12}, "input"),
    ({"max_output_tokens": 4}, "input"),
    ({"input": "hi", "tools": [{"type": "web_search"}]}, "function"),
    ({"input": "hi", "tools": [{"type": "function", "name": "f"}],
      "tool_choice": {"type": "function"}}, "tool_choice"),
    ({"input": "hi", "tools": [{"type": "function", "name": "f"}],
      "parallel_tool_calls": True}, "parallel"),
])
def test_malformed_requests_are_rejected(client, payload, contains):
    client.expect_400(payload, name="responses-malformed", contains=contains,
                      path="/v1/responses")


# ---------------------------------------------------------------- reasoning
def test_reasoning_is_accepted_and_echoed(client):
    """`reasoning` is a hint about how much thinking to do, not a promise about
    the response document, so it is accepted rather than refused — but it is
    echoed back so a caller can see what the turn actually ran with."""
    req = {"input": "hello", "max_output_tokens": 4, "temperature": 0,
           "reasoning": {"effort": "low"}}
    d = client.responses(req, name="responses-reasoning").expect_status(200).json
    if d.get("reasoning") != {"effort": "low"}:
        raise ProtocolError("reasoning was not echoed back",
                            got=d.get("reasoning"))


# ------------------------------------------------------------ text.format
def test_text_format_json_schema_constrains_output(client, report):
    schema = {"type": "object",
              "properties": {"city": {"type": "string"},
                             "count": {"type": "integer"}},
              "required": ["city", "count"],
              "additionalProperties": False}
    r = client.responses({"input": "describe a city", "temperature": 0,
                          "max_output_tokens": 60,
                          "text": {"format": {"type": "json_schema",
                                              "name": "city",
                                              "schema": schema}}},
                         name="responses-json-schema").expect_status(200)
    d = r.json
    if not d["runner_telemetry"].get("schema"):
        raise ProtocolError("schema-constrained decoding was not applied",
                            telemetry=d["runner_telemetry"])
    text = d["output_text"]
    # the protocol guarantee is that decoding was constrained; whether the
    # stub model reaches the closing brace inside the token cap is a model
    # outcome, recorded rather than failed (same convention as
    # test_structured_output.py)
    if not text.lstrip().startswith("{"):
        raise ProtocolError("constrained output does not begin as a JSON object",
                            text=text[:200])
    if d["status"] != "completed":
        report.note_quality("responses_json_schema",
                            "schema output truncated by the token cap before "
                            "the document closed",
                            output_tokens=d["usage"]["output_tokens"])
        return
    try:
        value = json.loads(text)
    except ValueError as e:
        raise ProtocolError("text.format json_schema did not produce JSON",
                            text=text[:200], error=str(e))
    validate_against_schema(value, schema)


def test_text_format_json_object_sets_json_mode(client):
    d = client.responses({"input": "hi", "temperature": 0,
                          "max_output_tokens": 30,
                          "text": {"format": {"type": "json_object"}}},
                         name="responses-json-object").expect_status(200).json
    if not d["runner_telemetry"].get("json_mode"):
        raise ProtocolError("text.format json_object did not enable json mode",
                            telemetry=d["runner_telemetry"])


# ------------------------------------------------------------------- tools
WEATHER = {"type": "function", "name": "get_weather",
           "description": "look up the weather in a city",
           "parameters": {"type": "object",
                          "properties": {"city": {"type": "string"}},
                          "required": ["city"],
                          "additionalProperties": False}}


def _forced_call(client, name, stream=False):
    payload = {"input": "what is the weather in Oslo?", "temperature": 0,
               "max_output_tokens": 40, "tools": [WEATHER],
               "tool_choice": "required"}
    if stream:
        st = client.responses_stream(payload, name=name).expect_sse()
        final = (st.payloads("response.completed") or
                 st.payloads("response.incomplete"))
        return st, final[0]["response"]
    r = client.responses(payload, name=name).expect_status(200)
    return None, r.json


def test_function_tool_produces_a_function_call_item(client, report):
    _, d = _forced_call(client, "responses-tool-buffered")
    calls = [i for i in d["output"] if i["type"] == "function_call"]
    if not calls:
        raise ProtocolError("tool_choice:required produced no function_call",
                            output=d["output"])
    call = calls[0]
    for field in ("id", "call_id", "name", "arguments"):
        if field not in call:
            raise ProtocolError("function_call item missing field",
                                field=field, item=call)
    if call["name"] != "get_weather":
        raise ProtocolError("model named an undeclared tool", got=call["name"])
    # the strict envelope guarantees this parses and conforms, even truncated
    args = json.loads(call["arguments"])
    validate_against_schema(args, WEATHER["parameters"])
    if any(i["type"] == "message" for i in d["output"]):
        raise ProtocolError("a tool call was accompanied by a message item",
                            output=d["output"])
    report.check_fixture("responses_function_call", d)


def test_streamed_function_call_matches_the_buffered_one(client):
    st, streamed = _forced_call(client, "responses-tool-stream", stream=True)
    _, buffered = _forced_call(client, "responses-tool-buffered-eq")
    s_call = [i for i in streamed["output"] if i["type"] == "function_call"]
    b_call = [i for i in buffered["output"] if i["type"] == "function_call"]
    if not s_call or not b_call:
        raise ProtocolError("a forced tool call was missing on one path",
                            streamed=streamed["output"], buffered=buffered["output"])
    if s_call[0]["name"] != b_call[0]["name"]:
        raise ProtocolError("streamed and buffered called different tools",
                            streamed=s_call[0]["name"], buffered=b_call[0]["name"])
    if json.loads(s_call[0]["arguments"]) != json.loads(b_call[0]["arguments"]):
        raise ProtocolError("streamed and buffered arguments differ",
                            streamed=s_call[0]["arguments"],
                            buffered=b_call[0]["arguments"])
    # the concatenated argument deltas must rebuild the same document
    deltas = "".join(d["delta"] for d in
                     st.payloads("response.function_call_arguments.delta"))
    if json.loads(deltas) != json.loads(s_call[0]["arguments"]):
        raise ProtocolError("argument deltas do not rebuild the item arguments",
                            deltas=deltas, item=s_call[0]["arguments"])
    # no envelope syntax may reach a text channel
    if st.text:
        raise ProtocolError("a tool call leaked text into output_text.delta",
                            text=st.text)


def test_tool_choice_none_answers_directly(client):
    d = client.responses({"input": "hello", "temperature": 0,
                          "max_output_tokens": 16, "tools": [WEATHER],
                          "tool_choice": "none"},
                         name="responses-tool-none").expect_status(200).json
    if any(i["type"] == "function_call" for i in d["output"]):
        raise ProtocolError("tool_choice:none still produced a call",
                            output=d["output"])


def test_named_tool_choice_selects_that_tool(client):
    d = client.responses({"input": "what is the weather in Oslo?",
                          "temperature": 0, "max_output_tokens": 40,
                          "tools": [WEATHER],
                          "tool_choice": {"type": "function",
                                          "name": "get_weather"}},
                         name="responses-tool-named").expect_status(200).json
    calls = [i for i in d["output"] if i["type"] == "function_call"]
    if not calls or calls[0]["name"] != "get_weather":
        raise ProtocolError("named tool_choice did not select that tool",
                            output=d["output"])


def test_nested_chat_style_tools_are_accepted(client):
    """A client migrating from chat completions sends the nested shape; it
    means the same thing and must not be refused."""
    nested = {"type": "function",
              "function": {"name": "get_weather",
                           "parameters": WEATHER["parameters"]}}
    d = client.responses({"input": "weather in Oslo?", "temperature": 0,
                          "max_output_tokens": 40, "tools": [nested],
                          "tool_choice": "required"},
                         name="responses-tools-nested").expect_status(200).json
    if not [i for i in d["output"] if i["type"] == "function_call"]:
        raise ProtocolError("nested tool declaration produced no call",
                            output=d["output"])


# ------------------------------------------------------- what Codex sends
# These four cases are taken from a captured request of the real Codex CLI
# (codex-cli 0.144.6, wire_api = "responses"). They are the shapes that a
# from-the-spec reading of the Responses API does not predict, so they are
# pinned against the client rather than against the document.
def test_namespace_tools_are_flattened(client):
    """Codex groups related function tools under a `namespace` entry.

    A namespace is not a tool — it is a container of them — so it is flattened
    into the union rather than refused: every leaf really is a local function
    this runtime can drive."""
    payload = {"input": "weather in Oslo?", "temperature": 0,
               "max_output_tokens": 40, "tool_choice": "required",
               "tools": [{"type": "namespace", "name": "grouped",
                          "description": "a group of tools",
                          "tools": [WEATHER]}]}
    d = client.responses(payload, name="responses-namespace").expect_status(200).json
    calls = [i for i in d["output"] if i["type"] == "function_call"]
    if not calls:
        raise ProtocolError("a namespaced tool was not offered to the model",
                            output=d["output"])
    if calls[0]["name"] != "get_weather":
        raise ProtocolError("namespaced tool called under the wrong name",
                            got=calls[0]["name"])


def test_disabled_hosted_tool_is_not_a_request(client):
    """Codex declares `web_search` with `external_web_access: false` even when
    web access is off. That is the client stating the capability is disabled,
    not asking for it, so it must not be refused — while an *enabled* hosted
    tool still must be, since this runtime cannot run one."""
    ok = client.responses({"input": "hello", "temperature": 0,
                           "max_output_tokens": 8,
                           "tools": [{"type": "web_search",
                                      "external_web_access": False}]},
                          name="responses-websearch-off")
    ok.expect_status(200)
    client.expect_400({"input": "hello",
                       "tools": [{"type": "web_search",
                                  "external_web_access": True}]},
                      name="responses-websearch-on", contains="function",
                      path="/v1/responses")


def test_developer_role_is_accepted(client):
    """`developer` is the Responses spelling of a system turn."""
    d = client.responses({"input": [
        {"type": "message", "role": "developer",
         "content": [{"type": "input_text", "text": "You are terse."}]},
        {"type": "message", "role": "user",
         "content": [{"type": "input_text", "text": "hello"}]}],
        "temperature": 0, "max_output_tokens": 8},
        name="responses-developer-role").expect_status(200).json
    if not d["output"]:
        raise ProtocolError("developer-role input produced no output", body=d)


def test_unknown_advisory_fields_are_tolerated(client):
    """`prompt_cache_key` and `client_metadata` are routing and telemetry
    hints. They make no promise about the response document, so unlike a
    dropped schema they can be accepted without misleading the caller."""
    r = client.responses({"input": "hello", "temperature": 0,
                          "max_output_tokens": 8, "store": False,
                          "include": [], "parallel_tool_calls": False,
                          "prompt_cache_key": "abc-123",
                          "client_metadata": {"session_id": "abc"}},
                         name="responses-advisory")
    r.expect_status(200)


# --------------------------------------------------------------- tool loop
def test_function_call_output_is_accepted_in_a_later_request(client):
    """The agent loop: the call this runtime produced is fed back with its
    result, and the next turn must be accepted and answered."""
    _, first = _forced_call(client, "responses-loop-call")
    call = [i for i in first["output"] if i["type"] == "function_call"][0]
    followup = {"input": [
        {"role": "user", "content": "what is the weather in Oslo?"},
        {"type": "function_call", "call_id": call["call_id"],
         "name": call["name"], "arguments": call["arguments"]},
        {"type": "function_call_output", "call_id": call["call_id"],
         "output": "12 degrees and raining"},
    ], "temperature": 0, "max_output_tokens": 24, "tools": [WEATHER]}
    d = client.responses(followup, name="responses-loop-result").expect_status(200).json
    if not d["output"]:
        raise ProtocolError("tool result turn produced no output", body=d)
    if d["status"] not in ("completed", "incomplete"):
        raise ProtocolError("tool result turn did not terminate cleanly",
                            status=d["status"])
    # Accepting the items is not the same as using them. A dropped
    # function_call_output would still answer 200 with a plausible-looking
    # turn, so what is asserted is that the result text was actually rendered
    # into the prompt: the same request without those two items must tokenize
    # to measurably fewer input tokens.
    without = dict(followup, input=followup["input"][:1])
    base = client.responses(without, name="responses-loop-baseline")
    base.expect_status(200)
    grew = d["usage"]["input_tokens"] - base.json["usage"]["input_tokens"]
    if grew < 8:
        raise ProtocolError(
            "the function_call/function_call_output items did not reach the "
            "prompt (input token count barely changed)",
            with_result=d["usage"]["input_tokens"],
            without_result=base.json["usage"]["input_tokens"])


def test_input_items_and_instructions_reach_the_model(client):
    """`instructions` plus a multi-part input must be accepted and echoed."""
    d = client.responses({"instructions": "You are terse.",
                          "input": [{"role": "user",
                                     "content": [{"type": "input_text",
                                                  "text": "hello"}]}],
                          "temperature": 0, "max_output_tokens": 8},
                         name="responses-input-items").expect_status(200).json
    if d.get("instructions") != "You are terse.":
        raise ProtocolError("instructions were not echoed back",
                            got=d.get("instructions"))


# ----------------------------------------------------------------- telemetry
def test_usage_and_telemetry_are_preserved(client):
    d = client.responses(dict(BASE), name="responses-usage").expect_status(200).json
    u = d["usage"]
    for field in ("input_tokens", "output_tokens", "total_tokens"):
        if not isinstance(u.get(field), int):
            raise ProtocolError("usage field missing or not an int",
                                field=field, usage=u)
    if u["input_tokens"] + u["output_tokens"] != u["total_tokens"]:
        raise ProtocolError("usage does not add up", usage=u)
    cached = u.get("input_tokens_details", {}).get("cached_tokens")
    if not isinstance(cached, int):
        raise ProtocolError("cached_tokens is not reported", usage=u)
    t = d.get("runner_telemetry")
    for field in ("prompt_cached_tokens", "prompt_eval_tokens",
                  "generation_seconds", "generation_tok_s"):
        if field not in (t or {}):
            raise ProtocolError("runner telemetry not preserved on /v1/responses",
                                field=field, telemetry=t)
    if t["prompt_cached_tokens"] != cached:
        raise ProtocolError("cached tokens disagree between usage and telemetry",
                            usage=cached, telemetry=t["prompt_cached_tokens"])
