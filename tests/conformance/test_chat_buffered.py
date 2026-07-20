"""Buffered chat completions: response shape, truncation, stop sequences."""

import pytest

from harness import ProtocolError

BASE = {"messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 8, "temperature": 0}


def test_health_and_models(client, report):
    client.get("/health").expect_status(200)
    models = client.get("/v1/models").expect_status(200).json
    if models.get("object") != "list" or not models.get("data"):
        raise ProtocolError("/v1/models is not an OpenAI model list", body=models)
    if not models["data"][0].get("id"):
        raise ProtocolError("model entry has no id", entry=models["data"][0])
    report.check_fixture("models", models)


def test_capabilities(client, report):
    caps = client.get("/v1/capabilities").expect_status(200).json
    for feature in ("json_schema", "stop_sequences"):
        if not caps.get("features", {}).get(feature):
            raise ProtocolError("capability not advertised", feature=feature,
                                features=caps.get("features"))
    report.check_fixture("capabilities", caps)


def test_buffered_chat_shape(client, report):
    r = client.chat(dict(BASE), name="buffered-chat").expect_status(200)
    d = r.json
    if d.get("object") != "chat.completion":
        raise ProtocolError("wrong object type", got=d.get("object"))
    for field in ("id", "created", "model", "choices", "usage"):
        if field not in d:
            raise ProtocolError("buffered response missing required field",
                                field=field, keys=sorted(d))
    msg = r.choice["message"]
    if msg.get("role") != "assistant":
        raise ProtocolError("buffered message role is not assistant",
                            got=msg.get("role"))
    if r.finish_reason not in ("stop", "length", "tool_calls"):
        raise ProtocolError("unknown finish_reason", got=r.finish_reason)
    u = r.usage
    if u["prompt_tokens"] + u["completion_tokens"] != u["total_tokens"]:
        raise ProtocolError("usage does not add up", usage=u)
    report.check_fixture("chat_buffered", d)


def test_completions_endpoint_shape(client, report):
    r = client.completion({"prompt": "hello", "max_tokens": 8, "temperature": 0},
                          name="buffered-completion").expect_status(200)
    d = r.json
    if d.get("object") != "text_completion":
        raise ProtocolError("wrong object type", got=d.get("object"))
    if not isinstance(r.choice.get("text"), str):
        raise ProtocolError("completion choice has no text string",
                            choice=r.choice)
    report.check_fixture("completion_buffered", d)


def test_max_tokens_truncates(client):
    """Truncation: the cap is honoured exactly and reported as finish_reason
    'length'. A cap that silently over-runs would blow a caller's budget."""
    for cap in (1, 4, 16):
        r = client.chat(dict(BASE, max_tokens=cap), name=f"truncate-{cap}")
        r.expect_status(200)
        got = r.usage["completion_tokens"]
        if got > cap:
            raise ProtocolError("max_tokens exceeded", cap=cap, generated=got)
        if got == cap and r.finish_reason != "length":
            raise ProtocolError("hit the cap but did not report length",
                                cap=cap, finish_reason=r.finish_reason)


def test_max_tokens_clamped_to_context(client):
    """An absurd max_tokens is clamped to the remaining context, not rejected
    and not obeyed: agent clients routinely send a huge cap."""
    r = client.chat(dict(BASE, max_tokens=999999), name="truncate-clamp")
    r.expect_status(200)
    if r.usage["completion_tokens"] >= 1024:
        raise ProtocolError("max_tokens not clamped to context window",
                            generated=r.usage["completion_tokens"])


def test_prefix_cache_is_deterministic(client):
    """Identical greedy requests must produce identical text whether or not the
    KV prefix cache served them. This is the cheapest correctness canary the
    harness has and it is why temp=0 is used throughout."""
    a = client.chat(dict(BASE, max_tokens=12), name="determinism-1")
    b = client.chat(dict(BASE, max_tokens=12), name="determinism-2")
    if a.content != b.content:
        raise ProtocolError("identical greedy requests diverged",
                            first=a.content, second=b.content)
    if b.telemetry.get("prompt_cached_tokens", 0) <= 0:
        raise ProtocolError("second identical request did not hit the prefix cache",
                            telemetry=b.telemetry)


def _stub_text(client):
    """A prefix of the stub model's greedy output that is safe to slice.

    test.gguf emits near-random token ids, so the decoded bytes can contain
    invalid UTF-8; everything after the first replacement char is dropped."""
    text = client.chat(dict(BASE, max_tokens=24), name="stop-baseline").content
    clean = text.split("�")[0]
    if len(clean) < 3:
        pytest.skip(f"stub model produced too little clean text: {text!r}")
    return text, clean


def test_stop_sequence_truncates_before_match(client):
    text, clean = _stub_text(client)
    stop = clean[:3]
    expected = text[:text.find(stop)]
    for payload_stop, label in ((stop, "string"), ([stop], "array")):
        r = client.chat(dict(BASE, max_tokens=24, stop=payload_stop),
                        name=f"stop-{label}").expect_status(200)
        if r.content != expected:
            raise ProtocolError("stop sequence did not truncate before the match",
                                form=label, expected=expected, got=r.content)
        if r.finish_reason != "stop":
            raise ProtocolError("stop match did not report finish_reason stop",
                                got=r.finish_reason)


def test_unmatched_stop_flushes_held_bytes(client):
    """A partial stop match that never completes must be flushed, not eaten."""
    text, clean = _stub_text(client)
    r = client.chat(dict(BASE, max_tokens=24, stop=[clean + "NEVERMATCHES"]),
                    name="stop-unmatched").expect_status(200)
    if r.content != text:
        raise ProtocolError("unmatched stop sequence swallowed output",
                            expected=text, got=r.content)
    if r.finish_reason == "stop":
        raise ProtocolError("reported a stop match that never happened")
