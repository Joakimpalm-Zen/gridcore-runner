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


def test_batch_counters_report_the_work_the_scheduler_did(client):
    """How much batching actually happened, in the same raw-counter form.

    The scheduler counts microbatch steps and the sequences cut into them. That
    pair is the only wire-visible answer to "is continuous batching earning its
    thread on this box": `batch_sequences / batch_steps` is the mean batch size
    over any window a dashboard cares to difference, and a single-slot or
    non-batched server reports a flat zero rather than a misleading one.

    Kept raw for the same reason `generate_seconds` is: the ratio needs a
    window, and the runner has no business choosing one.
    """
    before = health(client)
    for key in ("batch_steps", "batch_sequences"):
        assert key in before, (key, before)
        assert before[key] >= 0, (key, before[key])

    body = {"model": "test",
            "messages": [{"role": "user", "content": "count to three"}],
            "temperature": 0, "max_tokens": 8, "cache_prompt": False}
    client.chat(body, name="batch-metrics-chat").expect_status(200)
    after = health(client)

    # this suite's server runs two slots, so batching is on (the banner says
    # "continuous batching") and a generation must move the step counter
    assert after["batch_steps"] > before["batch_steps"], (before, after)
    # every step batches at least the one sequence that woke it
    assert after["batch_sequences"] >= after["batch_steps"], after
