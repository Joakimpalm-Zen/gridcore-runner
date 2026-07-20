"""Malformed and unsupported request fields must be rejected, never ignored.

This is a project invariant, not a style preference. A request that names a
field the server does not implement, or names one it does implement with a
value it cannot use, must fail with 400. Answering 200 while quietly dropping
the field is the worst outcome available: the caller asked for guaranteed
structure, or a stop sequence, or SSE framing, and got a response that looks
successful without it.

Every case below is verified current behaviour.
"""

import pytest

from harness import ProtocolError

CHAT = {"messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 4, "temperature": 0}


# ------------------------------------------------------- response_format
@pytest.mark.parametrize("rf,label,contains", [
    ({"type": "json_schema"}, "json_schema-without-member", "json_schema"),
    ({"type": "json_schema", "json_schema": "nope"}, "json_schema-not-object", "json_schema"),
    ({"type": "json_schema", "json_schema": {"schema": []}}, "schema-not-object", "schema"),
    ({"type": "nonsense"}, "unknown-type", "response_format.type"),
    ({"type": ""}, "empty-type", "response_format.type"),
    ({}, "missing-type", "response_format.type"),
    ({"type": 7}, "non-string-type", "response_format.type"),
    ("json_object", "not-an-object", "response_format"),
    ([], "array", "response_format"),
])
def test_bad_response_format_is_rejected(client, rf, label, contains):
    """A response_format the server cannot honour must 400.

    ``{"type":"json_schema"}`` with no ``json_schema`` member is the headline
    case: it reads as "give me schema-constrained output" and there is no
    schema, so unconstrained 200 would be a silent lie."""
    client.expect_400(dict(CHAT, response_format=rf),
                      name=f"bad-response-format-{label}", contains=contains)


@pytest.mark.parametrize("rf", [{"type": "text"}, {"type": "json_object"}])
def test_supported_response_format_is_accepted(client, rf):
    client.chat(dict(CHAT, max_tokens=16, response_format=rf),
                name=f"good-response-format-{rf['type']}").expect_status(200)


def test_unsupported_schema_construct_is_rejected(client):
    """schema_compile rejects what it cannot enforce rather than approximating
    it — an approximated schema is an unenforced schema."""
    for schema, label in (
            ({"type": "array", "items": {"type": "string"},
              "minItems": 2, "maxItems": 1}, "impossible-bounds"),
            ({"type": []}, "type-is-array"),
            ({"type": "object", "properties": {'bad"key': {"type": "string"}}},
             "escaped-property-key"),
    ):
        client.expect_400(
            dict(CHAT, response_format={"type": "json_schema",
                                        "json_schema": {"schema": schema}}),
            name=f"bad-schema-{label}", contains="schema")


# ----------------------------------------------------- scalar parameters
@pytest.mark.parametrize("field,value,contains", [
    ("stream", "true", "stream"),
    ("stream", 1, "stream"),
    ("stream", [], "stream"),
    ("top_k", 1e300, "sampling"),
    ("temperature", 1e300, "sampling"),
    ("top_p", 2.0, "sampling"),
    ("top_p", -1, "sampling"),
    ("min_p", 5, "sampling"),
    ("keep_alive", 1e300, "keep_alive"),
    ("top_logprobs", 999, "top_logprobs"),
    # Wrong TYPE is rejected for the same reason wrong RANGE is: a value the
    # server cannot use must not be replaced by a default the caller never
    # asked for. Several HTTP layers stringify numbers out of form or
    # env-derived config, so this is the common shape of the mistake.
    ("temperature", "hot", "sampling"),
    ("temperature", "0.7", "sampling"),
    ("top_p", "0.9", "sampling"),
    ("top_k", True, "sampling"),
    ("min_p", [], "sampling"),
    ("repeat_penalty", "1.1", "sampling"),
    ("seed", "1234", "sampling"),
    ("max_tokens", "eight", "max_tokens"),
    ("max_completion_tokens", "eight", "max_tokens"),
    ("keep_alive", "300", "keep_alive"),
    ("top_logprobs", "2", "top_logprobs"),
    ("logprobs", "true", "logprobs"),
])
def test_out_of_range_scalar_is_rejected(client, field, value, contains):
    payload = dict(CHAT, **{field: value})
    if field == "top_logprobs":
        payload["logprobs"] = True
    if field == "max_completion_tokens":
        # max_tokens wins when both are present, so the alias is only
        # reachable once the primary name is gone
        payload.pop("max_tokens")
    client.expect_400(payload, name=f"bad-{field}-{value!r}", contains=contains)


@pytest.mark.parametrize("field", ["temperature", "top_p", "min_p", "top_k",
                                   "seed", "repeat_penalty", "max_tokens",
                                   "max_completion_tokens", "keep_alive",
                                   "stop", "logprobs"])
def test_explicit_null_scalar_reads_as_absent(client, field):
    """The boundary of the rule above. Every mainstream OpenAI SDK serialises
    an unset optional field as ``null`` rather than omitting it, so ``null``
    must mean "absent" and take the default. Treating it as a wrong type
    would 400 on ordinary traffic from an unmodified client."""
    r = client.chat(dict(CHAT, **{field: None}), name=f"null-{field}")
    r.expect_status(200)


@pytest.mark.parametrize("stop,label", [
    ([1], "non-string-item"),
    ([""], "empty-item"),
    ("", "empty-string"),
    (["a", "b", "c", "d", "e"], "too-many"),
    ({"a": 1}, "object"),
    (7, "number"),
])
def test_malformed_stop_is_rejected(client, stop, label):
    client.expect_400(dict(CHAT, stop=stop), name=f"bad-stop-{label}",
                      contains="stop")


# ----------------------------------------------------------- body/routing
def test_invalid_json_body_is_rejected(client):
    for body, label in ((b"{", "truncated"), (b"not json", "garbage"),
                        (b"[]", "array-not-object"), (b"", "empty")):
        r = client.post_bytes(f"bad-body-{label}", "/v1/chat/completions", body)
        if r.status == 200:
            raise ProtocolError("invalid JSON body accepted", case=label,
                                body=r.text[:200])
        r.expect_status(400)
        r.expect_error_envelope()


def test_missing_required_fields_are_rejected(client):
    client.expect_400({"max_tokens": 4}, name="chat-no-messages",
                      contains="messages")
    client.expect_400({"messages": []}, name="chat-empty-messages")
    client.expect_400({"messages": [{"role": "user"}]}, name="chat-no-content")
    client.expect_400({"max_tokens": 4}, name="completion-no-prompt",
                      contains="prompt", path="/v1/completions")
    client.expect_400({}, name="embeddings-no-input", contains="input",
                      path="/v1/embeddings")


def _chat_body(extra):
    """A chat body with a raw JSON fragment spliced in, for values json.dumps
    cannot emit (bare Infinity, overflowing exponents)."""
    return (b'{"messages":[{"role":"user","content":"hi"}],' + extra + b"}")


def test_overflowing_exponent_in_a_sampling_param_is_rejected(client):
    """``1e400`` overflows to infinity. It must never become a sampling
    parameter, whichever guard catches it."""
    r = client.post_bytes("nonfinite-temperature", "/v1/chat/completions",
                          _chat_body(b'"temperature":1e400,"max_tokens":4'))
    if r.status == 200:
        raise ProtocolError("infinite temperature accepted", body=r.text[:200])
    r.expect_status(400)
    r.expect_error_envelope()


def test_overflowing_max_tokens_cannot_run_away(client):
    """FINDING (build-flag dependent, deliberately asserted loosely).

    json.c's number parser rejects non-finite values with ``isfinite``, and
    request_max_tokens re-checks — but the shipping CFLAGS include
    ``-ffast-math``, under which the compiler is licensed to fold ``isfinite``
    to true. So on a stock build ``{"max_tokens": 1e400}`` parses, survives
    both guards as +inf, and is clamped by the ``v > INT_MAX`` branch instead.

    Asserting "400" here would pass on a -fno-fast-math build and fail on the
    shipped one. What actually matters — and holds either way — is that an
    infinite cap cannot produce unbounded generation: either it is rejected, or
    it is clamped to the context window.
    """
    r = client.post_bytes("nonfinite-max-tokens", "/v1/chat/completions",
                          _chat_body(b'"temperature":0,"max_tokens":1e400'))
    if r.status == 400:
        r.expect_error_envelope()
        return
    r.expect_status(200)
    if r.usage["completion_tokens"] >= 1024:
        raise ProtocolError("infinite max_tokens produced unbounded generation",
                            generated=r.usage["completion_tokens"])


def test_max_tokens_above_int_max_is_clamped_not_rejected(client):
    """Finite-but-huge is clamped to the context window (agent clients send
    absurd caps routinely); only non-finite is an error. Pinned because the
    boundary between "clamp" and "reject" is easy to move by accident."""
    r = client.chat(dict(CHAT, max_tokens=1e300), name="max-tokens-huge")
    r.expect_status(200)
    if r.usage["completion_tokens"] >= 1024:
        raise ProtocolError("huge max_tokens was not clamped to the context",
                            generated=r.usage["completion_tokens"])


def test_unknown_route_is_404_not_400(client):
    """Routing errors and validation errors must not be confused: a 400 on an
    unknown path would tell a client its request was malformed."""
    r = client.get("/v1/does-not-exist", name="unknown-route")
    r.expect_status(404)
    r.expect_error_envelope()


def test_error_bodies_are_always_json_envelopes(client, report):
    """Every rejection carries {"error":{"message","type"}}. Clients branch on
    this shape; an HTML or bare-text error page breaks them."""
    r = client.chat(dict(CHAT, response_format={"type": "nope"}),
                    name="error-envelope-fixture")
    r.expect_status(400)
    err = r.expect_error_envelope()
    if not err["message"].strip():
        raise ProtocolError("rejection carried an empty message")
    report.check_fixture("error_envelope", r.json)


def test_extra_unknown_top_level_fields_are_tolerated(client):
    """The other half of the invariant: fields OpenAI clients routinely send
    that runner does not act on (frequency_penalty, user, ...) must NOT 400.
    Rejecting those would break Cline/OpenCode-shaped traffic outright.

    The line is: a field whose *semantics runner cannot honour* is rejected; a
    field that is merely advisory is accepted."""
    r = client.chat(dict(CHAT, frequency_penalty=0, presence_penalty=0,
                         user="conformance", n=1, logit_bias={},
                         parallel_tool_calls=False, tool_choice="auto"),
                    name="advisory-fields")
    r.expect_status(200)


# ---------------------------------------------------------- known gaps
# The following is the *opposite* of the "reject, never ignore" invariant
# above. It is pinned as-is because changing it is a src/ decision, and
# because an unpinned silent-ignore is exactly the kind of thing that quietly
# widens.
#
# The wrong-typed-scalar gap that used to live here is CLOSED: those cases now
# assert 400 in test_out_of_range_scalar_is_rejected, with the null-means-
# absent boundary pinned by test_explicit_null_scalar_reads_as_absent.

@pytest.mark.known_gap("unscheduled", "\"model\" is unvalidated unless the server is in registry mode")
def test_known_gap_model_field_unvalidated_on_single_model_server(client):
    """KNOWN GAP — pins today's behaviour, do not read as desired.

    ``model`` is only checked when the server was started with a registry
    (``-m "a=x.gguf,b=y.gguf"``); a single-model server answers any model name
    with the one model it has. A client that thinks it is talking to gpt-4o
    gets test.gguf and a 200.

    The registry path DOES reject correctly and is covered by the existing CI
    model-swap step, so this is specifically the single-model case.

    WHEN THIS IS FIXED: assert 400 with "unknown model" here instead.
    """
    r = client.chat(dict(CHAT, model="definitely-not-a-real-model"),
                    name="unknown-model-single")
    if r.status == 400:
        pytest.fail("unknown model is now rejected on a single-model server — "
                    "the gap is closed, invert this test")
    r.expect_status(200)
    served = r.json.get("model")
    if served == "definitely-not-a-real-model":
        raise ProtocolError("server echoed back a model it is not serving",
                            got=served)
