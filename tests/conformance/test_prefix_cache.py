"""Shared, forkable KV prefixes (Phase 7), seen from the wire.

An agent client sends the same system prompt, tool list and schema on every
request and changes only the last few hundred bytes. The prefix cache turns
that repetition into prefill nobody does twice: a completed prefix is
snapshotted out of one slot's KV cache and forked into whichever slot takes
the next request.

The reason this suite exists is that the failure mode is *silent*. A fork
installed against the wrong tokens does not error — it answers, fluently, from
another request's context. So the load-bearing tests here are not the counters
but the two identity tests: a forked request must return exactly what the same
request returns with every cache turned off.

The server this runs against has two slots (conftest: parallel=2), which is
what makes the shared tier observable at all. A slot that still holds a prompt
in its own KV rewinds to it for free and never consults the shared store, so
every test that wants to see a fork first has to fill both slots with
something else. `_poison` is that step, and it is why the tests below look
more elaborate than "send it twice".
"""

import threading

import pytest

from _errors import ProtocolError

# Long enough to clear the 16-token floor by a wide margin, short enough that
# prompt + reply fit the suite's 1024-token context (the test model's vocabulary
# is tiny, so it tokenizes close to one token per character).
SYSTEM = ("You are a terse test assistant. " +
          " ".join(f"rule {i}: be short." for i in range(6)))

OTHER_SYSTEM = ("You are a different test fixture. " +
                " ".join(f"directive {i}: enumerate." for i in range(6)))


def _payload(system, user, **kw):
    body = {"model": "test",
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": 0, "max_tokens": 12, "logprobs": True}
    body.update(kw)
    return body


def _telemetry(resp):
    resp.expect_status(200)
    t = resp.json.get("runner_telemetry")
    if t is None:
        raise ProtocolError("response carries no runner_telemetry",
                            keys=sorted(resp.json))
    return t


def _content(resp):
    return resp.json["choices"][0]["message"].get("content") or ""


def _fingerprint(resp):
    """Per-token logprobs — the sharp instrument for "same context?".

    Comparing the decoded text is not enough here. The suite's model is a
    tiny generated GGUF whose greedy continuation is dominated by the last
    token, so it answers almost any prompt with the same characters and text
    equality would pass even for a prefix forked from the wrong request. The
    logprob of each emitted token is a function of the whole attended context,
    so it moves as soon as one KV row is wrong.
    """
    resp.expect_status(200)
    lp = resp.json["choices"][0].get("logprobs")
    if not lp or not lp.get("content"):
        raise ProtocolError("response carries no logprobs",
                            choice=sorted(resp.json["choices"][0]))
    return [(c["token"], c["logprob"]) for c in lp["content"]]


def _same_context(got, want, what):
    """Assert two runs decoded against the same context.

    Not bit equality: prefill tiles the prompt into batches, and reusing a
    prefix moves where those tile boundaries fall, so the same arithmetic is
    reassociated and the last mantissa bits move. That is a known and
    pre-existing property of prefix reuse (engine_rewind has always done it),
    and it is a thousand times smaller than the effect under test — a prefix
    forked from the wrong request moves logprobs by tenths of a nat or drops
    tokens entirely. TOL sits in the gap.
    """
    TOL = 1e-3
    assert [t for t, _ in got] == [t for t, _ in want], what
    worst = max((abs(a - b) for (_, a), (_, b) in zip(got, want)), default=0.0)
    assert worst < TOL, f"{what} (worst logprob delta {worst:.6f})"


def _concurrent(fn, n):
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


def _poison(client, n_slots=2):
    """Leave every slot holding a prompt unrelated to SYSTEM.

    Both slots run one unrelated request at the same time, so neither can
    still rewind into SYSTEM out of its own KV. Whatever happens next has to
    come from the shared store or from a fresh prefill, and the telemetry can
    tell those two apart.
    """
    _concurrent(
        lambda i: client.chat(_payload(OTHER_SYSTEM, f"poison {i}"),
                              name=f"prefix-poison-{i}"),
        n_slots)


@pytest.fixture()
def cold(client):
    client.raw("prefix-clear", "POST", "/v1/runner/prefix-cache/clear")
    return client


# ------------------------------------------------------------------ surface
def test_prefix_cache_endpoint_shape(client):
    r = client.get("/v1/runner/prefix-cache", name="prefix-cache-stats")
    r.expect_status(200)
    doc = r.json
    assert doc["object"] == "runner.prefix_cache"
    for key in ("enabled", "entries", "bytes", "budget_bytes", "ttl_seconds",
                "hits", "misses", "stores", "evictions", "tokens_reused",
                "saved_prefill_seconds", "prefill_seconds_per_token"):
        assert key in doc, f"missing {key}"
    assert doc["enabled"] is True
    assert doc["bytes"] <= doc["budget_bytes"]


def test_telemetry_reports_the_fork(client):
    t = _telemetry(client.chat(_payload(SYSTEM, "hello"), name="prefix-tele"))
    for key in ("prompt_cached_tokens", "prompt_forked_tokens",
                "prompt_eval_tokens", "prefix_cache_saved_seconds"):
        assert key in t, f"missing telemetry field {key}"
    assert t["prompt_forked_tokens"] <= t["prompt_cached_tokens"]


def test_capabilities_advertise_forkable_prefixes(client):
    caps = client.get("/v1/capabilities", name="prefix-caps").json
    assert caps["features"]["shared_prefix_cache"] is True
    assert caps["features"]["forkable_prefixes"] is True


# ------------------------------------------------------------------- the win
def test_a_repeated_system_prompt_is_forked_not_prefilled(cold):
    """The exit criterion: repeated agent requests skip the static prefill."""
    first = _telemetry(cold.chat(_payload(SYSTEM, "first question"),
                                 name="prefix-warm"))
    assert first["prompt_forked_tokens"] == 0, "nothing was cached yet"

    _poison(cold)

    second = _telemetry(cold.chat(_payload(SYSTEM, "second question"),
                                  name="prefix-fork"))
    assert second["prompt_forked_tokens"] > 0, (
        "a repeated system prompt was prefilled from scratch")
    # the fork is the shared block, not the whole prompt: the differing user
    # turn still has to be evaluated
    assert second["prompt_eval_tokens"] > 0
    total = (second["prompt_cached_tokens"] + second["prompt_eval_tokens"])
    assert second["prompt_forked_tokens"] < total

    stats = cold.get("/v1/runner/prefix-cache", name="prefix-stats-after").json
    assert stats["hits"] >= 1
    assert stats["tokens_reused"] >= second["prompt_forked_tokens"]
    assert stats["entries"] >= 1


def test_saved_seconds_are_reported_and_accumulate(cold):
    cold.chat(_payload(SYSTEM, "alpha"), name="prefix-saved-warm")
    _poison(cold)
    t = _telemetry(cold.chat(_payload(SYSTEM, "beta"), name="prefix-saved-hit"))
    assert t["prompt_forked_tokens"] > 0
    assert t["prefix_cache_saved_seconds"] > 0, (
        "a fork that reused tokens reported no saved time")
    stats = cold.get("/v1/runner/prefix-cache", name="prefix-saved-stats").json
    assert stats["saved_prefill_seconds"] >= t["prefix_cache_saved_seconds"]
    assert stats["prefill_seconds_per_token"] > 0


# ------------------------------------------------------------ the real gate
def test_a_forked_prefix_answers_exactly_like_a_cold_one(cold):
    """The anti-contamination gate.

    Greedy decoding is a determinism request, so the same prompt must produce
    the same tokens whether its prefix was forked out of the cache or built
    from nothing. If a fork ever installs rows belonging to another request,
    this is the test that sees it — the output changes and stays fluent, which
    is precisely why a counter-based test would not.
    """
    question = "name one colour"
    reference = cold.chat(_payload(SYSTEM, question, cache_prompt=False),
                          name="prefix-ref")
    reference.expect_status(200)

    # seed the cache with a *different* user turn on the same system prompt,
    # so the fork is partial: the shared block hits, the user turn does not
    cold.chat(_payload(SYSTEM, "something entirely unrelated"),
              name="prefix-seed")
    _poison(cold)

    forked = cold.chat(_payload(SYSTEM, question), name="prefix-forked")
    t = _telemetry(forked)
    assert t["prompt_forked_tokens"] > 0, "this test needs an actual fork"
    _same_context(_fingerprint(forked), _fingerprint(reference),
                  "a forked prefix changed the distribution it sampled from")
    assert _content(forked) == _content(reference)


def test_a_neighbours_prefix_is_never_installed(cold):
    """Two systems that share a leading run must not share past it.

    OTHER_SYSTEM and SYSTEM diverge early, but both start with "You are a".
    A cache that matched on a hash bucket, a length, or a prompt-level key
    would happily hand one to the other; a cache that compares token vectors
    forks only the handful of tokens that genuinely match.
    """
    ref_a = cold.chat(_payload(SYSTEM, "question", cache_prompt=False),
                      name="prefix-neighbour-ref-a")
    ref_b = cold.chat(_payload(OTHER_SYSTEM, "question", cache_prompt=False),
                      name="prefix-neighbour-ref-b")

    # fill the cache with both, in both orders, then interleave
    cold.chat(_payload(SYSTEM, "seed a"), name="prefix-neighbour-seed-a")
    cold.chat(_payload(OTHER_SYSTEM, "seed b"), name="prefix-neighbour-seed-b")
    _poison(cold)

    got_a = cold.chat(_payload(SYSTEM, "question"), name="prefix-neighbour-a")
    got_b = cold.chat(_payload(OTHER_SYSTEM, "question"),
                      name="prefix-neighbour-b")
    assert _telemetry(got_a)["prompt_forked_tokens"] > 0, "no fork happened"
    _same_context(_fingerprint(got_a), _fingerprint(ref_a),
                  "system A picked up system B's prefix")
    _same_context(_fingerprint(got_b), _fingerprint(ref_b),
                  "system B picked up system A's prefix")


def test_concurrent_forks_of_one_prefix_do_not_interfere(cold):
    """Multiple users fork the same prefix at the same time.

    Each fork is a copy into a slot's own KV cache, so the sequences must stay
    independent even though they came from one snapshot.
    """
    questions = ["name one colour", "name one animal"]
    refs = [cold.chat(_payload(SYSTEM, q, cache_prompt=False),
                      name=f"prefix-conc-ref-{i}")
            for i, q in enumerate(questions)]

    cold.chat(_payload(SYSTEM, "seed the shared block"), name="prefix-conc-seed")
    _poison(cold)

    got = _concurrent(
        lambda i: cold.chat(_payload(SYSTEM, questions[i]),
                            name=f"prefix-conc-{i}"),
        len(questions))
    for i, (g, r) in enumerate(zip(got, refs)):
        _same_context(_fingerprint(g), _fingerprint(r),
                      f"concurrent fork {i} decoded against another context")


# ----------------------------------------------------------------- controls
def test_prefix_cache_false_opts_out_of_sharing_only(cold):
    cold.chat(_payload(SYSTEM, "seed for opt-out"), name="prefix-optout-seed")
    _poison(cold)
    t = _telemetry(cold.chat(_payload(SYSTEM, "opted out", prefix_cache=False),
                             name="prefix-optout"))
    assert t["prompt_forked_tokens"] == 0
    assert t["prompt_eval_tokens"] > 0


def test_cache_prompt_false_still_disables_everything(cold):
    cold.chat(_payload(SYSTEM, "seed for cache_prompt"), name="prefix-cp-seed")
    _poison(cold)
    t = _telemetry(cold.chat(_payload(SYSTEM, "no cache", cache_prompt=False),
                             name="prefix-cp-off"))
    assert t["prompt_cached_tokens"] == 0
    assert t["prompt_forked_tokens"] == 0


def test_clear_releases_the_snapshots(client):
    client.chat(_payload(SYSTEM, "fill the cache"), name="prefix-clear-fill")
    before = client.get("/v1/runner/prefix-cache", name="prefix-clear-before").json
    assert before["entries"] >= 1 and before["bytes"] > 0

    r = client.raw("prefix-clear-do", "POST", "/v1/runner/prefix-cache/clear")
    r.expect_status(200)
    after = r.json
    assert after["entries"] == 0
    assert after["bytes"] == 0
    assert after["hits"] == 0 and after["stores"] == 0
