"""JC-R1 constrained-choice posteriors (`choice_logprobs`).

At each constrained sampling step the runner probes the top candidates
against the active grammar; steps where >= 2 probed candidates are legal
are decision points, reported with a posterior renormalized over the legal
set. The synthetic byte-level test.gguf makes this deterministic to force:
a top-level enum schema `["x", "y"]` funnels generation to `"` then a
genuine one-byte choice, so at least one decision point must appear.
"""

import math
import string

# 26 distinct first bytes: with the synthetic model's near-random logits, a
# 2-way enum can land both legal bytes outside the probed top-M (no decision
# recorded, legitimately). A wide enum makes >= 2 legal probed candidates a
# deterministic outcome for this test.gguf at temperature 0.
FRUIT = {"type": "string", "enum": [c + "0" for c in string.ascii_lowercase]}

BASE = {"messages": [{"role": "user", "content": "pick one"}],
        "temperature": 0, "max_tokens": 8}


def _schema_req(**extra):
    req = dict(BASE)
    req["response_format"] = {"type": "json_schema",
                              "json_schema": {"name": "pick",
                                              "schema": FRUIT}}
    req.update(extra)
    return req


def test_requires_constrained_decode(client):
    client.expect_400(dict(BASE, choice_logprobs=True), "cl_unconstrained")


def test_buffered_only(client):
    client.expect_400(_schema_req(choice_logprobs=True, stream=True),
                      "cl_stream")


def test_probe_width_validated(client):
    client.expect_400(_schema_req(choice_logprobs=True,
                                  choice_logprobs_probe=4), "cl_probe_lo")
    client.expect_400(_schema_req(choice_logprobs=True,
                                  choice_logprobs_probe=128), "cl_probe_hi")


def test_absent_unless_requested(client):
    r = client.chat(_schema_req(), name="cl_absent")
    r.expect_status(200)
    assert "choice_logprobs" not in r.json["choices"][0]


def test_decision_points_recorded(client, report):
    r = client.chat(_schema_req(choice_logprobs=True,
                                choice_logprobs_probe=64), name="cl_enum")
    r.expect_status(200)
    choice = r.json["choices"][0]
    recs = choice.get("choice_logprobs", [])
    # the enum branch point (26 first-byte candidates) is a real decision
    assert recs, "enum schema produced no decision points"
    n_gen = r.usage["completion_tokens"]
    prev = -1
    for rec in recs:
        assert prev < rec["index"] < n_gen
        prev = rec["index"]
        assert rec["n_legal"] >= 2
        assert 0.0 < rec["coverage"] <= 1.0 + 1e-6
        alts = rec["alternatives"]
        assert 2 <= len(alts) <= 8
        assert len(alts) == min(rec["n_legal"], 8)
        # descending by probability, renormalized over the legal probed set
        probs = [a["prob"] for a in alts]
        assert probs == sorted(probs, reverse=True)
        if rec["n_legal"] == len(alts):
            assert math.isclose(sum(probs), 1.0, abs_tol=1e-4)
        for a in alts:
            assert a["logprob"] <= 0.0
            assert 0.0 < a["prob"] <= 1.0
    report.check_fixture("chat_choice_logprobs_shape",
                         {"n_recs>=1": True,
                          "keys": sorted(recs[0].keys()),
                          "alt_keys": sorted(recs[0]["alternatives"][0].keys())})


def test_json_object_mode_accepted(client):
    r = client.chat(dict(BASE, max_tokens=16,
                         response_format={"type": "json_object"},
                         choice_logprobs=True), name="cl_json_mode")
    r.expect_status(200)
    for rec in r.json["choices"][0].get("choice_logprobs", []):
        assert rec["n_legal"] >= 2
