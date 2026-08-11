"""Thane compatibility: the schema shapes Xyntetik Thane actually sends.

Thane (xyntetik-thane) is Runner's primary in-suite consumer and it pins
``xyntetik-runner-client @ git+...xyntetik-runner.git@main``, so a Runner
change lands in Thane the moment it merges. Every request shape asserted here
is a verbatim structural copy of what ``thane/context.py`` builds today:

* ``action_schema()`` — a ``oneOf`` union of one branch per tool, each branch
  discriminated by ``{"tool": {"const": <name>}}``, with ``required`` and
  ``additionalProperties: false`` at both the branch and the ``args`` level.
* ``_SUMMARY_SCHEMA`` — the compaction summary object, which carries
  ``required`` but deliberately no ``additionalProperties``.

The schemas are reproduced here rather than imported: this suite must not
depend on xyntetik-thane being installed, and pinning the shape locally is the
point — if Thane's real schema drifts from this copy, that is a Thane change that
should be made deliberately, and if Runner stops accepting this copy, Thane
breaks in the field.

The keyword gate added in "schema: reject unenforceable JSON Schema keywords"
makes these acceptance tests load-bearing: several constructs Thane relies on
sit next to constructs that are now rejected, and two of them (an empty
``required`` list, and a *property named* ``pattern``) are close enough to
rejected forms that a stricter gate could plausibly catch them by accident.
"""

import json

from harness import ProtocolError

# Thane's real tool surface (xyntetik_loadout.ARG_KEYS) as of this writing.
# A representative subset: one no-arg tool, one single-arg, one multi-arg
# with optionals, one whose argument is *named* like a schema keyword.
THANE_TOOLS = {
    "read_file": (("path",), ("path",)),
    "write_file": (("path", "content", "append"), ("path", "content")),
    "edit_lines": (("path", "first_line", "last_line", "replace"),
                   ("path", "first_line", "last_line", "replace")),
    "list_dir": (("path",), ()),                  # required == []
    "search": (("path", "pattern"), ("pattern",)),  # property named "pattern"
    "done": (("summary",), ("summary",)),
}

_ARG_TYPES = {"first_line": "integer", "last_line": "integer",
              "append": "boolean"}


def _thane_action_schema():
    """Structural copy of thane.context.action_schema()."""
    alts = []
    for tool, (keys, required) in THANE_TOOLS.items():
        alts.append({
            "type": "object",
            "properties": {
                "thinking": {"type": "string"},
                "tool": {"const": tool},
                "args": {
                    "type": "object",
                    "properties": {
                        key: {"type": _ARG_TYPES.get(key, "string")}
                        for key in keys
                    },
                    "required": list(required),
                    "additionalProperties": False,
                },
            },
            "required": ["thinking", "tool", "args"],
            "additionalProperties": False,
        })
    return {"oneOf": alts}


# Structural copy of thane.context._SUMMARY_SCHEMA — note: no
# additionalProperties, by design in Thane.
THANE_SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "progress": {"type": "string"},
        "facts": {"type": "array", "items": {"type": "string"}},
        "open_work": {"type": "array", "items": {"type": "string"}},
        "next_step": {"type": "string"},
    },
    "required": ["progress", "facts", "open_work", "next_step"],
}

# Thane sends exactly this envelope from thane/runner_client.py: RunnerLLM.
BASE = {"messages": [{"role": "user", "content": "list the directory"}],
        "temperature": 0.2, "max_tokens": 64}


def _schema_request(schema, **over):
    payload = dict(BASE)
    payload["response_format"] = {"type": "json_schema",
                                  "json_schema": {"schema": schema}}
    payload.update(over)
    return payload


# --------------------------------------------------------------- compilation
def test_thane_action_union_compiles(client):
    """Thane's discriminated oneOf union must be accepted.

    This is the single most load-bearing request Thane makes: every agent turn
    goes through it. A 400 here means Thane cannot take a single step."""
    client.chat(_schema_request(_thane_action_schema()),
                name="thane-action-union").expect_status(200)


def test_thane_summary_schema_compiles(client):
    """The compaction summary schema carries `required` and no
    `additionalProperties`. Thane compacts through this call; a 400 here means
    Thane dies at ~70% context instead of compacting."""
    client.chat(_schema_request(THANE_SUMMARY_SCHEMA),
                name="thane-summary-schema").expect_status(200)


def test_empty_required_list_is_accepted(client):
    """`list_dir` takes only optional args, so Thane emits `"required": []`.

    An empty list is not the same as a missing `properties` map, which the
    keyword gate does reject — this pins the distinction."""
    schema = {"type": "object",
              "properties": {"path": {"type": "string"}},
              "required": [],
              "additionalProperties": False}
    client.chat(_schema_request(schema),
                name="thane-empty-required").expect_status(200)


def test_property_named_like_a_keyword_is_accepted(client):
    """Thane's `search` tool has an argument called `pattern`, and `pattern` is
    a *rejected* schema keyword. A property name must never be read as a
    keyword of its enclosing schema."""
    schema = {"type": "object",
              "properties": {"pattern": {"type": "string"},
                             "required": {"type": "string"},
                             "const": {"type": "string"}},
              "required": ["pattern"],
              "additionalProperties": False}
    client.chat(_schema_request(schema),
                name="thane-keyword-named-property").expect_status(200)


def test_thane_action_union_output_conforms(client, report):
    """Constrained output must be one of the declared branches, with the
    discriminator and its own args — never another branch's arg names.

    The stub model may exhaust the token cap before closing the document;
    that is a model-quality outcome, recorded rather than failed, exactly as
    test_structured_output.py does."""
    r = client.chat(_schema_request(_thane_action_schema(), max_tokens=256),
                    name="thane-action-conforms")
    r.expect_status(200)
    try:
        truncated = r.finish_reason == "length"
        content = r.content
    except ProtocolError as e:
        # The stub model emits near-random token ids, so the token cap can
        # fall in the middle of a multi-byte UTF-8 sequence and the response
        # body stops being decodable. That is worth knowing about (see the
        # note below) but it is not a statement about Thane's schema.
        report.note_quality("thane-action-conforms",
                            "response body was not decodable UTF-8 — the token "
                            "cap can split a multi-byte sequence",
                            error=str(e))
        return
    if truncated:
        report.note_quality("thane-action-conforms",
                            "constrained action truncated by the token cap "
                            "before the document closed",
                            completion_tokens=r.usage["completion_tokens"])
        return
    doc = json.loads(content)
    assert set(doc) == {"thinking", "tool", "args"}, doc
    assert doc["tool"] in THANE_TOOLS, doc["tool"]
    allowed = set(THANE_TOOLS[doc["tool"]][0])
    assert set(doc["args"]) <= allowed, (doc["tool"], doc["args"])


# ------------------------------------------------------- request envelope
def test_thane_scalar_types_are_accepted(client):
    """Runner now 400s wrong-typed scalars. Thane sends temperature as a float
    and max_tokens as an int (thane/config.py validates both), so its envelope
    must pass — this pins that Thane's types are the accepted ones."""
    client.chat(_schema_request(THANE_SUMMARY_SCHEMA,
                                temperature=0.2, max_tokens=32, stream=False),
                name="thane-scalar-types").expect_status(200)


def test_unnamed_model_is_accepted_on_a_single_model_server(client):
    """Thane addresses its own runner by the tag "thane", not by filename. On a
    non-swap server the name must be ignored rather than 404/400, or every
    Thane request against an attached engine fails."""
    client.chat(_schema_request(THANE_SUMMARY_SCHEMA, model="thane"),
                name="thane-model-tag").expect_status(200)


# ------------------------------------------- constrained-decode termination
def test_constrained_output_opens_immediately(client):
    """A schema-constrained document must begin with its opening token.

    This was the whitespace livelock that cost Thane its compaction summaries.
    A model that would rather emit a preamble (prose, or a thinking block)
    found every prose token rejected and whitespace the only legal move, so it
    could spend the entire max_tokens budget without ever reaching the opening
    brace. The document was then force-closed with empty values and returned
    as `{"progress":"","facts":[],...}` with finish_reason=length — schema
    *valid*, information-free, and indistinguishable from a real answer, so
    Thane's "summary failed, use a placeholder" fallback never fired and the
    distilled session state was silently replaced with nothing.

    Leading whitespace is now refused by the validator, so the only legal
    first token is one that opens the document. Interior whitespace is
    untouched — `{ "a" : "b" }` is ordinary model output.
    """
    r = client.chat(_schema_request(THANE_SUMMARY_SCHEMA, max_tokens=128),
                    name="thane-opens-immediately")
    r.expect_status(200)
    try:
        content = r.content
    except ProtocolError:
        # The stub model emits near-random token ids, so the cap can split a
        # multi-byte UTF-8 sequence. Not a statement about this behaviour.
        return
    if not content:
        return          # nothing generated at all — asserted below
    assert content[0] == "{", (
        "a constrained document must open with its first byte, not whitespace",
        repr(content[:32]))


def test_unproductive_constrained_decode_returns_nothing(client):
    """An empty document must never be invented on the caller's behalf.

    If the budget really does run out before the model produces anything,
    Runner returns empty content rather than synthesizing a conforming
    document out of nothing. A caller then gets a parse error it can act on —
    retry, raise, fall back — instead of a well-formed record of nothing.
    Completing a document the model *started* is truncation and still
    happens; inventing one it never started is fabrication.

    Measured on Qwen3-4B, which spends its budget inside a `<think>` block:
    before this, `{"progress":"","next_step":""}` / finish_reason=length;
    now, empty content / finish_reason=length.
    """
    r = client.chat(_schema_request(THANE_SUMMARY_SCHEMA, max_tokens=128),
                    name="thane-no-fabricated-document")
    r.expect_status(200)
    try:
        content = r.content
    except ProtocolError:
        return
    if not content.strip():
        # nothing generated: it must be reported as truncated, not as a
        # finished answer
        assert r.finish_reason == "length", (
            "an empty constrained answer must be reported truncated",
            r.finish_reason)
        return
    # anything actually generated still has to parse and conform
    doc = json.loads(content)
    assert set(doc) <= set(THANE_SUMMARY_SCHEMA["properties"]), doc
