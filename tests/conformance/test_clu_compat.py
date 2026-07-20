"""Clu compatibility: the schema shapes Gridcore Clu actually sends.

Clu (gridcore-clu) is Runner's primary in-suite consumer and it pins
``gridcore-runner-client @ git+...gridcore-runner.git@main``, so a Runner
change lands in Clu the moment it merges. Every request shape asserted here
is a verbatim structural copy of what ``clu/context.py`` builds today:

* ``action_schema()`` — a ``oneOf`` union of one branch per tool, each branch
  discriminated by ``{"tool": {"const": <name>}}``, with ``required`` and
  ``additionalProperties: false`` at both the branch and the ``args`` level.
* ``_SUMMARY_SCHEMA`` — the compaction summary object, which carries
  ``required`` but deliberately no ``additionalProperties``.

The schemas are reproduced here rather than imported: this suite must not
depend on gridcore-clu being installed, and pinning the shape locally is the
point — if Clu's real schema drifts from this copy, that is a Clu change that
should be made deliberately, and if Runner stops accepting this copy, Clu
breaks in the field.

The keyword gate added in "schema: reject unenforceable JSON Schema keywords"
makes these acceptance tests load-bearing: several constructs Clu relies on
sit next to constructs that are now rejected, and two of them (an empty
``required`` list, and a *property named* ``pattern``) are close enough to
rejected forms that a stricter gate could plausibly catch them by accident.
"""

import json

import pytest

from harness import ProtocolError

# Clu's real tool surface (gridcore_toolbox.ARG_KEYS) as of this writing.
# A representative subset: one no-arg tool, one single-arg, one multi-arg
# with optionals, one whose argument is *named* like a schema keyword.
CLU_TOOLS = {
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


def _clu_action_schema():
    """Structural copy of clu.context.action_schema()."""
    alts = []
    for tool, (keys, required) in CLU_TOOLS.items():
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


# Structural copy of clu.context._SUMMARY_SCHEMA — note: no
# additionalProperties, by design in Clu.
CLU_SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "progress": {"type": "string"},
        "facts": {"type": "array", "items": {"type": "string"}},
        "open_work": {"type": "array", "items": {"type": "string"}},
        "next_step": {"type": "string"},
    },
    "required": ["progress", "facts", "open_work", "next_step"],
}

# Clu sends exactly this envelope from clu/runner_client.py: RunnerLLM.
BASE = {"messages": [{"role": "user", "content": "list the directory"}],
        "temperature": 0.2, "max_tokens": 64}


def _schema_request(schema, **over):
    payload = dict(BASE)
    payload["response_format"] = {"type": "json_schema",
                                  "json_schema": {"schema": schema}}
    payload.update(over)
    return payload


# --------------------------------------------------------------- compilation
def test_clu_action_union_compiles(client):
    """Clu's discriminated oneOf union must be accepted.

    This is the single most load-bearing request Clu makes: every agent turn
    goes through it. A 400 here means Clu cannot take a single step."""
    client.chat(_schema_request(_clu_action_schema()),
                name="clu-action-union").expect_status(200)


def test_clu_summary_schema_compiles(client):
    """The compaction summary schema carries `required` and no
    `additionalProperties`. Clu compacts through this call; a 400 here means
    Clu dies at ~70% context instead of compacting."""
    client.chat(_schema_request(CLU_SUMMARY_SCHEMA),
                name="clu-summary-schema").expect_status(200)


def test_empty_required_list_is_accepted(client):
    """`list_dir` takes only optional args, so Clu emits `"required": []`.

    An empty list is not the same as a missing `properties` map, which the
    keyword gate does reject — this pins the distinction."""
    schema = {"type": "object",
              "properties": {"path": {"type": "string"}},
              "required": [],
              "additionalProperties": False}
    client.chat(_schema_request(schema),
                name="clu-empty-required").expect_status(200)


def test_property_named_like_a_keyword_is_accepted(client):
    """Clu's `search` tool has an argument called `pattern`, and `pattern` is
    a *rejected* schema keyword. A property name must never be read as a
    keyword of its enclosing schema."""
    schema = {"type": "object",
              "properties": {"pattern": {"type": "string"},
                             "required": {"type": "string"},
                             "const": {"type": "string"}},
              "required": ["pattern"],
              "additionalProperties": False}
    client.chat(_schema_request(schema),
                name="clu-keyword-named-property").expect_status(200)


def test_clu_action_union_output_conforms(client, report):
    """Constrained output must be one of the declared branches, with the
    discriminator and its own args — never another branch's arg names.

    The stub model may exhaust the token cap before closing the document;
    that is a model-quality outcome, recorded rather than failed, exactly as
    test_structured_output.py does."""
    r = client.chat(_schema_request(_clu_action_schema(), max_tokens=256),
                    name="clu-action-conforms")
    r.expect_status(200)
    try:
        truncated = r.finish_reason == "length"
        content = r.content
    except ProtocolError as e:
        # The stub model emits near-random token ids, so the token cap can
        # fall in the middle of a multi-byte UTF-8 sequence and the response
        # body stops being decodable. That is worth knowing about (see the
        # note below) but it is not a statement about Clu's schema.
        report.note_quality("clu-action-conforms",
                            "response body was not decodable UTF-8 — the token "
                            "cap can split a multi-byte sequence",
                            error=str(e))
        return
    if truncated:
        report.note_quality("clu-action-conforms",
                            "constrained action truncated by the token cap "
                            "before the document closed",
                            completion_tokens=r.usage["completion_tokens"])
        return
    doc = json.loads(content)
    assert set(doc) == {"thinking", "tool", "args"}, doc
    assert doc["tool"] in CLU_TOOLS, doc["tool"]
    allowed = set(CLU_TOOLS[doc["tool"]][0])
    assert set(doc["args"]) <= allowed, (doc["tool"], doc["args"])


# ------------------------------------------------------- request envelope
def test_clu_scalar_types_are_accepted(client):
    """Runner now 400s wrong-typed scalars. Clu sends temperature as a float
    and max_tokens as an int (clu/config.py validates both), so its envelope
    must pass — this pins that Clu's types are the accepted ones."""
    client.chat(_schema_request(CLU_SUMMARY_SCHEMA,
                                temperature=0.2, max_tokens=32, stream=False),
                name="clu-scalar-types").expect_status(200)


def test_unnamed_model_is_accepted_on_a_single_model_server(client):
    """Clu addresses its own runner by the tag "clu", not by filename. On a
    non-swap server the name must be ignored rather than 404/400, or every
    Clu request against an attached engine fails."""
    client.chat(_schema_request(CLU_SUMMARY_SCHEMA, model="clu"),
                name="clu-model-tag").expect_status(200)


# ------------------------------------------------------------- known gaps
@pytest.mark.known_gap(
    phase="constrained-decode termination",
    what="a schema-constrained decode permits unbounded leading whitespace, so "
         "a model that would rather emit a preamble (prose, or a thinking "
         "block) is clamped to whitespace and can spend its entire max_tokens "
         "budget without reaching the opening brace; the document is then "
         "force-closed with empty values and reported finish_reason=length")
def test_leading_whitespace_livelock_is_pinned(client, report):
    """Pin the whitespace livelock that costs Clu its compaction summaries.

    This was measured against real models, not the stub: Llama-3.2-3B-Instruct
    and Qwen3-4B, given Clu's actual compaction prompt, spend the whole budget
    on whitespace and return `{"progress":"","facts":[],...}` — a
    schema-*valid*, information-free document. Clu raises no exception on it,
    so its documented "summary failed, keep going with a placeholder" fallback
    never fires and the distilled state is silently lost.

    The assertion is deliberately weak (the stub model's behaviour is not the
    point); the marker carries the finding. When constrained decode learns to
    stop at document completion, rewrite this test."""
    r = client.chat(_schema_request(CLU_SUMMARY_SCHEMA, max_tokens=128),
                    name="clu-whitespace-livelock")
    r.expect_status(200)
    try:
        content = r.content
    except ProtocolError:
        return  # stub-model byte soup; the marker carries the finding
    if r.finish_reason == "length" and not content.lstrip().startswith("{"):
        report.note_quality("clu-whitespace-livelock",
                            "whole token budget spent before the document "
                            "opened", bytes=len(content))
    if content.strip():
        # whatever came back, a closed document must still parse
        try:
            json.loads(content)
        except ValueError:
            assert r.finish_reason == "length", (
                "an unparseable document must at least be reported truncated")
