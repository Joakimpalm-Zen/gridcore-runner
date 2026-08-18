"""scripts/batch-bench.py: the continuous-batching gate.

Its second exit question is "is a batched request's output identical to a solo
one's?", and the module docstring answers it with "every concurrent response is
compared byte-for-byte with the same prompt run alone". That has to be true of
every slot, not every distinct prompt.
"""

import importlib.util
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "batch_bench", ROOT / "scripts/batch-bench.py")
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def test_every_slot_is_compared_even_when_prompts_repeat():
    """PROMPTS has 8 entries and --parallel goes to 16, so --concurrency 12
    reuses the first four. Deduplicating the prompts and taking
    prompts.index(p) never yielded slots 8-11, so a batching bug that only
    shows when a slot is REUSED mid-batch was invisible."""
    prompts = [MOD.PROMPTS[i % len(MOD.PROMPTS)] for i in range(12)]
    batched = [f"answer-{i}" for i in range(12)]

    rows = MOD.identity_rows(prompts, batched, lambda p: "solo")

    assert [slot for slot, _p, _s, _b in rows] == list(range(12))
    assert [b for _slot, _p, _s, b in rows] == batched


def test_a_mismatch_in_a_reused_slot_is_visible():
    prompts = [MOD.PROMPTS[i % len(MOD.PROMPTS)] for i in range(12)]
    batched = ["solo"] * 12
    batched[9] = "drifted"

    rows = MOD.identity_rows(prompts, batched, lambda p: "solo")

    assert [slot for slot, _p, solo, b in rows if solo != b] == [9]


def test_the_solo_request_is_made_once_per_distinct_prompt():
    prompts = [MOD.PROMPTS[i % len(MOD.PROMPTS)] for i in range(12)]
    asked = []

    MOD.identity_rows(prompts, ["x"] * 12, lambda p: asked.append(p) or "x")

    assert len(asked) == len(set(prompts)) == 8
