"""Unit tests for scripts/token_divergence.py.

The script's verdict — tie vs real arithmetic divergence — is the exit code a
sensitivity-floor run is read by, so every scoring rule here is load-bearing.
"""

import importlib.util
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "token_divergence", ROOT / "scripts/token_divergence.py")
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def runner_wire(tokens, token_ids, token_logprobs, top_logprobs_json,
                top_token_ids):
    """One /v1/completions body in the runner's own logprobs shape.

    Written as TEXT, not as a Python dict, because the whole point of these
    tests is what happens on the wire: src/completion.c writes top_logprobs as
    a string-keyed JSON object with no dedup, and a Python dict literal cannot
    express the duplicate keys that produces.
    """
    return MOD.loads(
        '{"choices":[{"text":"x","logprobs":{'
        f'"tokens":{json.dumps(tokens)},'
        f'"token_ids":{json.dumps(token_ids)},'
        f'"token_logprobs":{json.dumps(token_logprobs)},'
        f'"top_logprobs":{top_logprobs_json},'
        f'"top_token_ids":{json.dumps(top_token_ids)}}}}}]}}')


def test_top_logprobs_survive_two_ids_rendering_to_the_same_string():
    """src/completion.c decodes each top-k id to a string and emits it as a KEY,
    without dedup, and puts the ids in a PARALLEL array "in the same order".
    Two byte-fallback ids decode to the same replacement character, so the
    object carries a duplicate key; json.loads keeps only the last value, and
    zipping the shortened value list against the full id array shifts every
    later logprob onto the wrong token.
    """
    entry = runner_wire(
        ["A"], [10], [-0.5],
        '[{"A":-0.5,"\\ufffd":-4.0,"\\ufffd":-6.0}]', [[10, 20, 30]])

    top = MOD.normalise(entry)[0][3]

    assert top == {10: -0.5, 20: -4.0, 30: -6.0}


def test_a_top_k_without_duplicates_is_unchanged():
    entry = runner_wire(["A"], [10], [-0.5],
                        '[{"A":-0.5,"B":-4.0}]', [[10, 20]])

    assert MOD.normalise(entry)[0][3] == {10: -0.5, 20: -4.0}


# ------------------------------------------------------------------- scoring
def row(token_id, logprob=-0.5, top=None):
    return (token_id, f"t{token_id}", logprob, top or {token_id: logprob})


def test_a_shared_prefix_with_one_engine_stopping_early_is_not_identical():
    """`identical` out of 16 is what a sensitivity floor is read from, so an
    engine that stopped early must not be counted as agreeing."""
    result = MOD.score("p", [row(1), row(2), row(3)],
                       [row(1), row(2), row(3), row(4)])

    assert result["status"] == "length_mismatch"
    assert result["runner_tokens"] == 3 and result["reference_tokens"] == 4


def test_equal_length_agreement_is_still_identical():
    result = MOD.score("p", [row(1), row(2)], [row(1), row(2)])

    assert result["status"] == "identical"
    assert result["matched_tokens"] == 2


def test_a_first_position_disagreement_is_still_a_divergence():
    result = MOD.score("p", [row(1)], [row(2)])

    assert result["status"] == "diverged"
    assert result["position"] == 0


# ---------------------------------------------------------------- classifying
def diverged(runner_gap, reference_gap):
    return {"status": "diverged", "runner_gap_nats": runner_gap,
            "reference_gap_nats": reference_gap}


def test_an_exactly_zero_gap_is_the_most_tie_like_divergence_there_is():
    """`gap or 9e9` reads 0.0 as "no measurement" and fails the gate on the
    one divergence it was written to forgive. Logprobs serialize at %.6f, so
    two contenders within 5e-7 parse to the same float and the gap is exactly
    0.0."""
    ties, real = MOD.classify([diverged(0.0, 0.0)], 0.25)

    assert len(ties) == 1 and real == []


def test_an_unmeasurable_gap_is_not_forgiven():
    ties, real = MOD.classify([diverged(None, 0.0)], 0.25)

    assert ties == [] and len(real) == 1


def test_a_wide_gap_is_a_real_divergence():
    ties, real = MOD.classify([diverged(1.5, 1.5)], 0.25)

    assert ties == [] and len(real) == 1


def test_a_length_mismatch_can_never_be_forgiven_as_a_tie():
    ties, real = MOD.classify([{"status": "length_mismatch"}], 0.25)

    assert ties == [] and len(real) == 1
