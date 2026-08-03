"""Dense agent-runtime composition: all serving features in one request flow."""

import json
import os
import re
import threading

from _errors import ProtocolError
from harness import Client, RunnerServer, find_runner, validate_against_schema

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL = os.path.join(ROOT, "test.gguf")
SYSTEM = " ".join(f"stable agent rule {i}" for i in range(24))
OTHER_SYSTEM = " ".join(f"unrelated poison rule {i}" for i in range(24))

SCHEMA_A = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "n": {"type": "integer"}},
    "required": ["name", "n"],
    "additionalProperties": False,
}
SCHEMA_B = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}, "tag": {"type": "string"}},
    "required": ["ok", "tag"],
    "additionalProperties": False,
}


def _payload(user, schema=SCHEMA_A, **extra):
    body = {
        "model": "test",
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ],
        "temperature": 0,
        "max_tokens": 64,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "composition", "schema": schema},
        },
    }
    body.update(extra)
    return body


def _concurrent(functions):
    results = [None] * len(functions)
    errors = []

    def work(index):
        try:
            results[index] = functions[index]()
        except Exception as exc:  # noqa: BLE001
            errors.append(f"[{index}] {exc!r}")

    threads = [threading.Thread(target=work, args=(i,)) for i in range(len(functions))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise ProtocolError("composition requests failed", errors="; ".join(errors))
    return results


def test_schema_batch_prefix_cancel_and_speculation_compose(report, tmp_path):
    """The target-exact output must survive every agent-serving feature at once."""
    exe = find_runner(ROOT)
    plain_log = tmp_path / "plain.log"
    spec_log = tmp_path / "spec.log"

    with RunnerServer(exe, MODEL, ctx=1024, parallel=1,
                      extra_args=["--gpu", "off"], log_path=plain_log) as plain:
        reference = Client(plain, report).chat(
            _payload("reference", cache_prompt=False), name="composition-reference")
        reference.expect_status(200)
        expected = reference.content
        validate_against_schema(json.loads(expected), SCHEMA_A)

    with RunnerServer(
        exe, MODEL, ctx=1024, parallel=2,
        extra_args=["--gpu", "off", "--draft", MODEL, "--draft-k", "4"],
        log_path=spec_log,
    ) as composed:
        client = Client(composed, report)
        client.chat(_payload("warm prefix"), name="composition-prefix-warm").expect_status(200)

        poison = _payload("poison", max_tokens=8)
        poison["messages"][0]["content"] = OTHER_SYSTEM
        _concurrent([
            lambda: client.chat(poison, name="composition-poison-0").expect_status(200),
            lambda: client.chat(poison, name="composition-poison-1").expect_status(200),
        ])

        def cancel():
            partial, _, _ = client.stream_raw(
                "composition-cancel", "/v1/chat/completions",
                _payload("cancel me", stream=True, max_tokens=256), close_after_bytes=1)
            assert partial

        responses = _concurrent([
            cancel,
            lambda: client.chat(_payload("reference"), name="composition-schema-a"),
            lambda: client.chat(_payload("other schema", SCHEMA_B),
                                name="composition-schema-b"),
        ])
        first, second = responses[1], responses[2]
        for response, schema in ((first, SCHEMA_A), (second, SCHEMA_B)):
            response.expect_status(200)
            validate_against_schema(json.loads(response.content), schema)
            telemetry = response.json["runner_telemetry"]
            assert telemetry["schema"] is True
            assert telemetry["speculative"] is True
        assert first.content == expected

        # Prefix forking is asserted here, and NOT per-response above, because
        # per-response is not a property the engine offers.
        #
        # A resident prefix can be forked by at most `parallel` requests at
        # once. Measured directly: with --parallel 2, four concurrent requests
        # sharing a warm prefix report forks [66, 67, 0, 0]. The block above
        # issues THREE concurrent requests — the cancel plus both schema ones —
        # so one of the three necessarily forks nothing, and requiring both
        # schema responses to fork passed only when the cancel happened to lose
        # the race. That is a coin flip, and it is what made this test fail
        # about three runs in ten under whole-suite load while passing every
        # time in isolation.
        #
        # So the assertion is that at least one of them forked: the shared
        # prefix survived the poisoning and the cancellation and was still
        # usable under concurrency. That still fails if forking breaks.
        #
        # `prompt_forked_tokens == 0` does NOT mean the prefix went unused, and
        # reading it that way is what produced an earlier note claiming the
        # shared prefix became permanently unforkable after this sequence.
        # Measured, it does not: a sequential request after a cancellation plus
        # concurrency forks ~930 tokens. What the zeros above mean is that the
        # slot's OWN KV already held the prefix, so engine_prefix_reuse
        # declined to fork — its gate is `best > r.keep`, and forking when the
        # snapshot holds no more than the slot does would copy identical rows
        # over themselves.
        #
        # So both properties are asserted, and the second is the one that
        # actually matters to a caller: at least one request forked (the shared
        # snapshot survived the poisoning and the cancellation), and EVERY
        # request reused the prefix by one path or the other.
        telem = [r.json["runner_telemetry"] for r in (first, second)]
        concurrent_forks = [t["prompt_forked_tokens"] for t in telem]
        assert max(concurrent_forks) > 0, (
            f"no concurrent request forked the shared prefix: {concurrent_forks}")
        for t in telem:
            reused, fed = t["prompt_cached_tokens"], t["prompt_eval_tokens"]
            assert reused > 0 and fed < reused, (
                "a concurrent request re-prefilled a prompt it should have "
                f"reused: cached={reused} evaluated={fed}")

        # Speculative acceptance has to be PROVABLE here, not lucky. Every
        # request above is schema-constrained, and the drafter generates
        # unconstrained — so the grammar refuses most of what it proposes, and
        # on a random-weight fixture whether ANY draft survives is chance. That
        # is why the acceptance assertion below failed intermittently in a full
        # suite run and passed every time in isolation. One unconstrained
        # greedy request fixes the property rather than the threshold: drafter
        # and target are the same model at temperature 0, so every token it
        # drafts must be accepted. Widening the tolerance instead would have
        # kept the assertion green while measuring nothing.
        unconstrained = _payload("draft with nothing to refuse it")
        unconstrained.pop("response_format")
        plain_spec = client.chat(unconstrained, name="composition-unconstrained")
        plain_spec.expect_status(200)
        assert plain_spec.json["runner_telemetry"]["speculative"] is True

        composed.assert_alive()

    stats = spec_log.read_text(encoding="utf-8")
    accepted = [int(value) for value in re.findall(r"drafted, (\d+) accepted", stats)]
    assert accepted and max(accepted) > 0
