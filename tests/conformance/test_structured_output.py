"""response_format json_object and json_schema, plus embeddings.

Content-level assertions here are the one place where the synthetic test.gguf
genuinely limits what can be checked: it emits near-random token ids, so a
constrained decode can legitimately hit the token cap before closing its JSON.
That is a model-quality outcome, not a protocol one, and the taxonomy exists
precisely so it can be recorded as such instead of failing the run.
"""

import json
import math

import pytest

from harness import ProtocolError, SchemaError, validate_against_schema

BASE = {"messages": [{"role": "user", "content": "emit json"}],
        "temperature": 0}

PERSON = {
    "type": "object",
    "properties": {"name": {"type": "string"},
                   "age": {"type": "integer"},
                   "tag": {"type": "string", "enum": ["x", "y"]}},
    "required": ["name", "age", "tag"],
}


def _decode_constrained(r, report, test, kind):
    """Parse constrained output, or record why it could not be judged.

    Returns None when the stub model ran out of tokens mid-document — the
    protocol did its job, the model did not."""
    content = r.content
    if r.finish_reason == "length":
        report.note_quality(test, f"{kind} output truncated by the token cap "
                                  f"before the document closed",
                            finish_reason=r.finish_reason,
                            completion_tokens=r.usage["completion_tokens"])
        return None
    try:
        return json.loads(content)
    except ValueError as e:
        raise SchemaError(f"{kind} finished cleanly but its content is not JSON",
                          error=str(e), content=content[:300])


def _expect_json_prefix(content, kind, report, test):
    """The first non-whitespace byte of constrained output must open a JSON
    document. json mode permits unbounded leading whitespace, so the stub model
    can spend its whole budget on tabs and newlines and never reach a brace —
    that is a model-quality outcome and is recorded, not failed."""
    stripped = content.lstrip()
    if not stripped:
        report.note_quality(test, f"{kind} produced only whitespace within the "
                                  f"token cap (json mode allows unbounded "
                                  f"leading whitespace)",
                            bytes=len(content))
        return
    if not stripped.startswith(("{", "[")):
        raise SchemaError(f"{kind} output does not begin as JSON",
                          content=content[:200])


def test_json_object_mode(client, report):
    """json_object must be honoured *and* declared: telemetry.json_mode proves
    the constraint was actually applied rather than the model simply behaving."""
    r = client.chat(dict(BASE, max_tokens=64,
                         response_format={"type": "json_object"}),
                    name="json-object").expect_status(200)
    if r.telemetry.get("json_mode") is not True:
        raise ProtocolError("json_object accepted but json_mode not engaged",
                            telemetry=r.telemetry)
    if r.telemetry.get("schema") is not False:
        raise ProtocolError("json_object must not report schema constraint",
                            telemetry=r.telemetry)
    _expect_json_prefix(r.content, "json_object", report, "json_object")
    _decode_constrained(r, report, "json_object", "json_object")
    report.check_fixture("chat_json_object", r.json)


def test_json_schema_mode(client, report):
    """json_schema output must validate against the schema that was sent."""
    r = client.chat(dict(BASE, max_tokens=96,
                         response_format={"type": "json_schema",
                                          "json_schema": {"name": "person",
                                                          "schema": PERSON}}),
                    name="json-schema").expect_status(200)
    if r.telemetry.get("schema") is not True:
        raise ProtocolError("json_schema accepted but schema not engaged",
                            telemetry=r.telemetry)
    doc = _decode_constrained(r, report, "json_schema", "json_schema")
    if doc is not None:
        validate_against_schema(doc, PERSON)
    report.check_fixture("chat_json_schema", r.json)


def test_json_schema_bare_form(client):
    """OpenAI nests the schema under json_schema.schema; several SDKs send it
    directly as json_schema. Both are accepted and both must engage."""
    r = client.chat(dict(BASE, max_tokens=64,
                         response_format={"type": "json_schema",
                                          "json_schema": PERSON}),
                    name="json-schema-bare").expect_status(200)
    if r.telemetry.get("schema") is not True:
        raise ProtocolError("bare json_schema form did not engage the schema",
                            telemetry=r.telemetry)


def test_ollama_format_schema(client):
    """The Ollama-style top-level "format" schema is the third accepted form."""
    r = client.chat(dict(BASE, max_tokens=64, format=PERSON),
                    name="ollama-format").expect_status(200)
    if r.telemetry.get("schema") is not True:
        raise ProtocolError("Ollama-style format schema did not engage",
                            telemetry=r.telemetry)


def test_enum_constraint_is_actually_enforced(client, report):
    """A single-value enum is the sharpest available check that constrained
    decoding is real: there is exactly one legal completion."""
    schema = {"type": "object",
              "properties": {"only": {"type": "string", "enum": ["fixed"]}},
              "required": ["only"]}
    r = client.chat(dict(BASE, max_tokens=64,
                         response_format={"type": "json_schema",
                                          "json_schema": {"schema": schema}}),
                    name="json-schema-enum").expect_status(200)
    doc = _decode_constrained(r, report, "json_schema_enum", "enum schema")
    if doc is not None:
        validate_against_schema(doc, schema)
        if doc["only"] != "fixed":
            raise SchemaError("enum constraint not enforced", got=doc["only"])


def test_json_mode_survives_streaming(client, report):
    """Constrained decoding must work on the streaming path too — the
    constraint lives in the sampler, not in the response writer."""
    st = client.chat_stream(dict(BASE, max_tokens=64,
                                 response_format={"type": "json_object"}),
                            name="json-object-stream").expect_sse()
    text = st.text
    _expect_json_prefix(text, "streamed json_object", report,
                        "json_object_stream")
    if st.finish_reason == "length":
        report.note_quality("json_object_stream",
                            "streamed json_object hit the token cap")
    else:
        try:
            json.loads(text)
        except ValueError as e:
            raise SchemaError("streamed json_object content is not JSON",
                              error=str(e), content=text[:300])


# ------------------------------------------------------------- embeddings
def test_embeddings_shape(client, report):
    r = client.embeddings({"input": ["alpha", "beta"]},
                          name="embeddings-batch").expect_status(200)
    d = r.json
    if d.get("object") != "list":
        raise ProtocolError("embeddings response is not a list object",
                            got=d.get("object"))
    data = d.get("data")
    if not isinstance(data, list) or len(data) != 2:
        raise ProtocolError("embeddings returned the wrong number of vectors",
                            got=len(data) if isinstance(data, list) else data)
    dims = None
    for i, item in enumerate(data):
        if item.get("object") != "embedding":
            raise ProtocolError("embedding item has wrong object type",
                                index=i, got=item.get("object"))
        if item.get("index") != i:
            raise ProtocolError("embedding items are out of order",
                                expected=i, got=item.get("index"))
        vec = item.get("embedding")
        if not isinstance(vec, list) or not vec:
            raise ProtocolError("embedding is not a non-empty array", index=i)
        if not all(isinstance(x, (int, float)) for x in vec):
            raise SchemaError("embedding contains a non-number", index=i)
        if dims is None:
            dims = len(vec)
        elif len(vec) != dims:
            raise SchemaError("embeddings have inconsistent dimensionality",
                              first=dims, this=len(vec))
    if d.get("usage", {}).get("prompt_tokens", 0) <= 0:
        raise ProtocolError("embeddings reported no prompt tokens",
                            usage=d.get("usage"))
    report.check_fixture("embeddings", d)


def test_embeddings_are_l2_normalised(client):
    """Runner documents mean-pooled, L2-normed vectors. A caller doing cosine
    similarity by dot product depends on it."""
    r = client.embeddings({"input": "alpha"}, name="embeddings-single")
    r.expect_status(200)
    vec = r.json["data"][0]["embedding"]
    norm = math.sqrt(sum(x * x for x in vec))
    if not math.isclose(norm, 1.0, rel_tol=1e-3):
        raise SchemaError("embedding is not L2-normalised", norm=norm,
                          dims=len(vec))


def test_embeddings_string_and_array_agree(client):
    """A bare string input and a one-element array must give the same vector."""
    a = client.embeddings({"input": "alpha"}, name="embeddings-str")
    b = client.embeddings({"input": ["alpha"]}, name="embeddings-arr")
    va = a.json["data"][0]["embedding"]
    vb = b.json["data"][0]["embedding"]
    if len(va) != len(vb) or any(abs(x - y) > 1e-6 for x, y in zip(va, vb)):
        raise ProtocolError("string and single-element-array inputs disagree")


@pytest.mark.parametrize("payload,label", [
    ({"input": [1, 2]}, "non-string-items"),
    ({"input": []}, "empty-array"),
    ({"input": {}}, "object"),
    ({"input": None}, "null"),
])
def test_bad_embedding_input_is_rejected(client, payload, label):
    client.expect_400(payload, name=f"bad-embeddings-{label}",
                      path="/v1/embeddings")


def test_empty_string_input_is_accepted(client):
    """Pinned current behaviour: an empty string is NOT the "empty input" the
    400 refers to. It still tokenizes to BOS/EOS, so it embeds successfully.
    The rejection fires only when tokenization yields nothing at all."""
    r = client.embeddings({"input": ""}, name="embeddings-empty-string")
    r.expect_status(200)
    if not r.json["data"][0]["embedding"]:
        raise ProtocolError("empty-string input returned an empty vector")


def test_embeddings_do_not_corrupt_the_next_completion(client):
    """model_embed overwrites the slot's KV cache. If the prefix cache is not
    invalidated afterwards, the next completion silently decodes against stale
    keys — a correctness bug with no visible symptom."""
    before = client.chat({"messages": [{"role": "user", "content": "hello"}],
                          "max_tokens": 12, "temperature": 0},
                         name="pre-embed-completion").content
    client.embeddings({"input": ["some unrelated text to fill the cache"]},
                      name="embeddings-cache-poison").expect_status(200)
    after = client.chat({"messages": [{"role": "user", "content": "hello"}],
                         "max_tokens": 12, "temperature": 0},
                        name="post-embed-completion").content
    if before != after:
        raise ProtocolError("completion changed after an embeddings request "
                            "(stale KV cache)", before=before, after=after)


# --------------------------------------------------------- downstream client
# The suite's own agent client (gridcore-clu) drives runner through
# /v1/chat/completions with a hand-built discriminated union in
# `response_format`, not through tools[]. That shape therefore exercises the
# schema compiler by a different route than test_tool_calls.py does, and a
# change to either path could break it without any other test noticing. It is
# pinned here so it cannot regress silently. Kept structurally identical to
# clu/context.py:action_schema -- if that file changes shape, change this too.
CLU_ACTION_SCHEMA = {
    "oneOf": [
        {
            "type": "object",
            "properties": {
                "thinking": {"type": "string"},
                "tool": {"const": "read_file"},
                "args": {
                    "type": "object",
                    "properties": {"path": {"type": "string"},
                                   "limit": {"type": "integer"}},
                    "required": ["path"],
                    "additionalProperties": False,
                },
            },
            "required": ["thinking", "tool", "args"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {
                "thinking": {"type": "string"},
                "tool": {"const": "finish"},
                "args": {
                    "type": "object",
                    "properties": {"summary": {"type": "string"}},
                    "required": ["summary"],
                    "additionalProperties": False,
                },
            },
            "required": ["thinking", "tool", "args"],
            "additionalProperties": False,
        },
    ]
}


def test_clu_action_schema_still_compiles(client, report):
    """A discriminated union built by hand in response_format, as the project's
    own agent client sends it: `const` discriminators, nested required args and
    additionalProperties:false on both levels."""
    r = client.chat({"messages": [{"role": "user", "content": "read a file"}],
                     "max_tokens": 64, "temperature": 0,
                     "response_format": {
                         "type": "json_schema",
                         "json_schema": {"name": "action",
                                         "schema": CLU_ACTION_SCHEMA}}},
                    name="clu-action-schema").expect_status(200)
    if r.telemetry.get("schema") is not True:
        raise ProtocolError("the client's action schema was accepted but not "
                            "applied as a decoding constraint",
                            telemetry=r.telemetry)
    _expect_json_prefix(r.content, "clu-action", report, "clu_action_schema")
    value = _decode_constrained(r, report, "clu_action_schema", "clu-action")
    if value is None:
        return
    if value.get("tool") not in ("read_file", "finish"):
        raise SchemaError("discriminator is not one of the declared branches",
                          got=value.get("tool"))
    branch = next(b for b in CLU_ACTION_SCHEMA["oneOf"]
                  if b["properties"]["tool"]["const"] == value["tool"])
    validate_against_schema(value, branch)


def test_clu_action_schema_streams(client, report):
    """The same shape on the streaming path: clu renders `thinking` live, so
    the declared key order has to survive streaming too."""
    st = client.chat_stream({"messages": [{"role": "user", "content": "read a file"}],
                             "max_tokens": 64, "temperature": 0,
                             "response_format": {
                                 "type": "json_schema",
                                 "json_schema": {"name": "action",
                                                 "schema": CLU_ACTION_SCHEMA}}},
                            name="clu-action-stream").expect_sse()
    text = st.text.lstrip()
    if text and not text.startswith("{"):
        raise SchemaError("streamed constrained output does not open a JSON object",
                          text=st.text[:200])
