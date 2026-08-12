"""Per-process memory and cumulative throughput, seen from the wire.

These are the two runner-side measurements a capacity dashboard cannot get any
other way. A supervisor watching several runners needs to know what each one is
actually costing in RAM, and how much work it has done, and neither is
answerable from outside the process: RSS attributed to a PID is not the same as
the model's file size (weights are shared, mapped pages come and go), and
tokens-per-second computed by timing HTTP requests measures the client's
network as much as the server's decode.

The boundary this respects: the runner MEASURES and reports raw counters, it
does not PRESENT rates. `tokens_generated` and `generate_seconds` are exposed
as monotonic totals so a dashboard can difference them over whatever window it
cares about. A tok/s field would bake in an averaging window the runner has no
business choosing, and would be wrong for every consumer whose window differs.
"""

import json


def health(client):
    r = client.get("/health", name="health")
    assert r.status == 200, r.status
    return json.loads(r.body)


def test_health_reports_process_memory(client):
    """RSS for THIS process, which is the number a supervisor budgets against."""
    h = health(client)
    assert "rss_bytes" in h, h
    assert "peak_rss_bytes" in h, h
    # A loaded server is worth more than a few pages; a wildly large value
    # means the units are wrong (kilobytes reported as bytes, say).
    assert h["rss_bytes"] > (1 << 20), h["rss_bytes"]
    assert h["rss_bytes"] < (1 << 42), h["rss_bytes"]
    assert h["peak_rss_bytes"] >= h["rss_bytes"], h


def test_throughput_counters_are_monotonic_and_start_sane(client):
    h = health(client)
    for key in ("tokens_generated", "tokens_prompt", "generate_seconds"):
        assert key in h, (key, h)
        assert h[key] >= 0, (key, h[key])


def test_generating_advances_the_counters(client):
    """The counters must count the work, not merely exist."""
    before = health(client)
    body = {"model": "test",
            "messages": [{"role": "user", "content": "count to three"}],
            "temperature": 0, "max_tokens": 8}
    r = client.chat(body, name="metrics-chat").expect_status(200)
    used = r.usage
    after = health(client)

    gained = after["tokens_generated"] - before["tokens_generated"]
    assert gained == used["completion_tokens"], (gained, used)
    assert after["tokens_prompt"] - before["tokens_prompt"] == used["prompt_tokens"]
    # Decode took real time, and the counter is cumulative so it only grows.
    assert after["generate_seconds"] > before["generate_seconds"], (
        before["generate_seconds"], after["generate_seconds"])


def test_health_does_not_count_itself(client):
    """Polling /health must not move the work counters.

    A dashboard polls this endpoint on a timer. If the poll registered as
    activity the dashboard would be measuring its own polling, which is the
    same trap `active_requests` already documents for itself.
    """
    a = health(client)
    for _ in range(3):
        health(client)
    b = health(client)
    assert b["tokens_generated"] == a["tokens_generated"]
    assert b["tokens_prompt"] == a["tokens_prompt"]
    assert b["generate_seconds"] == a["generate_seconds"]
