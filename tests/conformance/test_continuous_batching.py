"""Continuous batching: what must stay true when requests share a microbatch.

The scheduler's whole promise is that being batched is invisible from the
outside. These tests attack that promise from the directions where a batching
server usually breaks: a sequence's output depending on who else was running,
one sequence's cancellation damaging its neighbours, and structured output
being left half-written when a request is cut short mid-document.

The suite's server runs --gpu off, so model_batch_decode takes its sequential
fallback here; what is under test is the scheduler — the state machine, the
per-slot engine reuse, the cancellation and deadline paths — not the CUDA
kernels, which tests/test_batch.c and scripts/kernel-verify.py cover.
"""

import json
import threading

import pytest

from _errors import ProtocolError

BASE = {"model": "test", "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0, "cache_prompt": False}

SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "n": {"type": "integer"}},
    "required": ["name", "n"],
    "additionalProperties": False,
}


def _content(resp):
    return resp.json["choices"][0]["message"].get("content") or ""


def _concurrent(fn, n):
    """Run fn(i) on n threads and return results in index order."""
    out, errs = [None] * n, []

    def work(i):
        try:
            out[i] = fn(i)
        except Exception as e:  # noqa: BLE001
            errs.append(f"[{i}] {e!r}")

    ths = [threading.Thread(target=work, args=(i,)) for i in range(n)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    if errs:
        raise ProtocolError("concurrent requests failed", errors="; ".join(errs))
    return out


def test_batched_greedy_output_matches_solo(client, server):
    """The load-independence guarantee, end to end.

    A greedy request must produce the same tokens whether it decoded alone or
    shared a microbatch with others. This is the property that lets schema and
    determinism guarantees survive concurrency, so it is checked on the wire
    rather than trusted from the primitive's bitwise-identity test."""
    payload = dict(BASE, max_tokens=24)
    solo = _content(client.chat(payload, name="batch-solo"))

    n = server.parallel * 2  # more requests than slots: some must queue
    batched = _concurrent(
        lambda i: _content(client.chat(payload, name=f"batch-concurrent-{i}")), n)

    for i, got in enumerate(batched):
        if got != solo:
            raise ProtocolError(
                "a request's output changed because the server was busy",
                index=i, solo=solo, batched=got)


def test_distinct_prompts_do_not_bleed_between_slots(client, server):
    """Sequences in one microbatch sit at different KV positions with different
    histories. A batch that broadcast one sequence's position or state would
    show up as one request answering another's prompt."""
    prompts = ["hello", "the number is", "once upon a", "def f(x):"]
    payloads = [dict(BASE, messages=[{"role": "user", "content": p}],
                     max_tokens=16) for p in prompts]

    solo = [_content(client.chat(p, name=f"bleed-solo-{i}"))
            for i, p in enumerate(payloads)]

    n = len(payloads)
    batched = _concurrent(
        lambda i: _content(client.chat(payloads[i], name=f"bleed-conc-{i}")), n)

    for i, p in enumerate(prompts):
        if batched[i] != solo[i]:
            raise ProtocolError("concurrent answer differs from the solo answer "
                                "for the same prompt", prompt=p,
                                solo=solo[i], batched=batched[i])


def test_structured_output_stays_valid_under_concurrency(client, server):
    """Every sequence keeps its own schema validator. Concurrency must not be
    able to interleave two documents into one."""
    payload = dict(BASE, max_tokens=64,
                   response_format={"type": "json_schema",
                                    "json_schema": {"name": "s", "schema": SCHEMA}})
    n = server.parallel * 2
    docs = _concurrent(
        lambda i: _content(client.chat(payload, name=f"batch-schema-{i}")), n)
    for i, d in enumerate(docs):
        try:
            obj = json.loads(d)
        except ValueError as e:
            raise ProtocolError("concurrent schema request returned invalid JSON",
                                index=i, body=d, error=str(e))
        missing = {"name", "n"} - set(obj)
        if missing:
            raise ProtocolError("concurrent schema request dropped required keys",
                                index=i, missing=sorted(missing), body=d)


def test_truncated_structured_output_is_still_a_valid_document(client, server):
    """A token ceiling that lands mid-document must still close it. Under
    batching this is the path a deadline and a cancellation also take, so it
    is pinned concurrently as well as alone."""
    payload = dict(BASE, max_tokens=6,  # far too few to finish honestly
                   response_format={"type": "json_schema",
                                    "json_schema": {"name": "s", "schema": SCHEMA}})
    n = server.parallel * 2
    docs = _concurrent(
        lambda i: _content(client.chat(payload, name=f"batch-truncated-{i}")), n)
    for i, d in enumerate(docs):
        try:
            json.loads(d)
        except ValueError as e:
            raise ProtocolError("a truncated constrained document was left "
                                "unclosed", index=i, body=d, error=str(e))


def test_cancelling_one_sequence_does_not_disturb_the_others(client, server):
    """A cancelled sequence is simply absent from the next microbatch. Its
    neighbours must not notice — not in their output, and not by being
    dropped along with it."""
    payload = dict(BASE, max_tokens=24)
    solo = _content(client.chat(payload, name="cancel-reference"))

    results = [None]

    def survivor(i):
        results[0] = _content(client.chat(payload, name=f"cancel-survivor-{i}"))
        return results[0]

    def canceller(i):
        # disconnect after the first byte, mid-generation
        partial, _, _ = client.stream_raw(
            f"cancel-abandoned-{i}", "/v1/chat/completions",
            dict(BASE, max_tokens=256, stream=True), close_after_bytes=1)
        if not partial:
            raise ProtocolError("no bytes arrived before disconnecting")
        return ""

    _concurrent(lambda i: (canceller(i) if i == 0 else survivor(i)),
                server.parallel)

    server.assert_alive()
    if results[0] != solo:
        raise ProtocolError("a neighbour's cancellation changed this request's "
                            "output", solo=solo, got=results[0])

    # and every slot is still usable afterwards
    for i in range(server.parallel + 1):
        client.chat(dict(BASE, max_tokens=4),
                    name=f"post-cancel-{i}").expect_status(200)


def test_request_deadline_truncates_rather_than_erroring(client):
    """A per-request deadline is a truncation, not a failure: the client gets
    a well-formed 200 whose finish_reason is "length", and a constrained
    document is still closed to something valid. An expiring request must
    never hand back a broken body."""
    payload = dict(BASE, max_tokens=512, timeout=0.05,
                   response_format={"type": "json_schema",
                                    "json_schema": {"name": "s", "schema": SCHEMA}})
    r = client.chat(payload, name="batch-deadline").expect_status(200)
    doc = r.json
    body = doc["choices"][0]["message"].get("content") or ""
    try:
        json.loads(body)
    except ValueError as e:
        raise ProtocolError("a deadline-expired constrained document was left "
                            "unclosed", body=body, error=str(e))


def test_deadline_out_of_range_is_rejected(client):
    """The deadline is a request parameter like any other, and is validated
    like one rather than silently clamped."""
    client.chat(dict(BASE, max_tokens=4, timeout=-1),
                name="batch-deadline-negative").expect_status(400)
