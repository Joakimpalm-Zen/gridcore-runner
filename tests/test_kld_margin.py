"""Margin-qualified top-1 agreement (quality bar v2).

Plain top-1 counts every argmax flip as a disagreement, including flips
between two tokens the REFERENCE model itself could barely separate. Measured
2026-08-13, that is what the 5-6 bit band of the quality ladder was failing
on: granite-4.1-3b Q6_K scored mean KLD 0.0155 — three times inside the 0.05
bound — while missing the 97% top-1 criterion by four points, and
Phi-4-mini Q8_0 scored KLD 0.0082 with 94.75% top-1. Distributions that close
are not damaged; their argmaxes are coin-flipping at near-ties.

v2 ADDS a column. Plain top-1 is still computed and still reported; these
tests pin the new one's semantics, especially the two that are easy to get
wrong: the band edge, and which side's margin is consulted.
"""
import importlib.util
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "kld_raw", ROOT / "scripts" / "kld-compare-raw.py")
kld_raw = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(kld_raw)

BAND = kld_raw.DEFAULT_TIE_BAND


def dist(**kw):
    return dict(kw)


def test_agreement_is_agreement_under_both():
    # same argmax: both columns count it, margin never consulted
    a = dist(x=-0.1, y=-3.0)
    b = dist(x=-0.2, y=-2.0)
    kld, agree, marg, overlap = kld_raw.score_pair(a, b)
    assert agree is True
    assert marg is True


def test_exact_tie_on_the_reference_is_forgiven():
    # The reference cannot separate x from y at all, so whichever the variant
    # picks is not evidence of damage. NB max() breaks the reference's tie
    # toward its first key, so the variant must prefer the OTHER one for the
    # argmaxes to differ at all — an exact tie where both happen to land on
    # the same token is plain agreement, not a forgiven flip.
    a = dist(x=-0.9, y=-0.5)     # variant picks y
    b = dist(x=-1.0, y=-1.0)     # reference is exactly tied, max() picks x
    _, agree, marg, _ = kld_raw.score_pair(a, b)
    assert agree is False
    assert marg is True


def test_just_inside_the_band_is_forgiven():
    a = dist(x=-0.1, y=-5.0)                 # variant picks x
    b = dist(x=-1.0 - BAND * 0.9, y=-1.0)    # reference picks y, narrowly
    _, agree, marg, _ = kld_raw.score_pair(a, b)
    assert agree is False
    assert marg is True


def test_just_outside_the_band_is_not_forgiven():
    a = dist(x=-0.1, y=-5.0)                 # variant picks x
    b = dist(x=-1.0 - BAND * 1.1, y=-1.0)    # reference prefers y decisively
    _, agree, marg, _ = kld_raw.score_pair(a, b)
    assert agree is False
    assert marg is False


def test_margin_is_read_from_the_reference_not_the_variant():
    # The asymmetry that matters: the VARIANT is unsure (tiny margin) while the
    # REFERENCE is certain. That is a real disagreement — the reference had a
    # clear opinion and the variant lost it — so it must NOT be forgiven.
    a = dist(x=-1.0, y=-1.0 - BAND * 0.1)    # variant barely picks x
    b = dist(x=-9.0, y=-0.001)               # reference is certain it is y
    _, agree, marg, _ = kld_raw.score_pair(a, b)
    assert agree is False
    assert marg is False, "a confident reference must not be forgiven"

    # and the mirror image: reference unsure, variant certain -> forgiven
    a2 = dist(x=-0.001, y=-9.0)
    b2 = dist(x=-1.0 - BAND * 0.1, y=-1.0)
    _, agree2, marg2, _ = kld_raw.score_pair(a2, b2)
    assert agree2 is False
    assert marg2 is True


def test_single_entry_reference_has_no_margin_and_is_not_forgiven():
    # A reference distribution with one usable entry gives no top-two gap to
    # measure. Refusing to forgive is the conservative reading.
    a = dist(x=-0.1, y=-4.0)
    b = dist(y=-0.2)
    _, agree, marg, _ = kld_raw.score_pair(a, b)
    assert agree is False
    assert marg is False


def test_a_pick_outside_the_band_is_not_forgiven_however_close_the_top_two_are():
    """The criterion must consult what the VARIANT picked, not just the gap.

    Until 2026-08-19 this asked only "were the reference's #1 and #2 close?".
    That question does not mention the variant at all, so a near-tie at the top
    of the reference forgave EVERY flip below it -- including a variant
    confidently emitting a token the reference rates at e^-12. The tie band is
    supposed to forgive a coin-flip between two candidates the reference cannot
    separate; it was forgiving anything at all as long as a coin-flip existed
    somewhere in the distribution.
    """
    b = dist(x=-1.0, y=-1.2, z=-12.0)   # #1/#2 within the band, z is nowhere
    assert kld_raw.top_two_margin(b) < BAND    # the old criterion's whole test
    a = dist(x=-9.0, y=-9.0, z=-0.01)   # the variant is certain it is z
    _, agree, marg, _ = kld_raw.score_pair(a, b)
    assert agree is False
    assert marg is False, "a pick the reference rates at e^-12 is not a near-tie"


def test_a_pick_inside_the_band_is_still_forgiven_in_a_wider_distribution():
    # the other half of the case above: same reference, and the variant picks
    # the runner-up the reference genuinely could not separate from its top.
    b = dist(x=-1.0, y=-1.2, z=-12.0)
    a = dist(x=-9.0, y=-0.01, z=-9.0)   # variant picks y, inside the band
    _, agree, marg, _ = kld_raw.score_pair(a, b)
    assert agree is False
    assert marg is True


def test_a_pick_the_reference_never_reported_is_not_forgiven():
    """Absent from the reference's top-N is not evidence of a near-tie.

    Both sides report a truncated top-N, so a variant's argmax can simply be
    missing from the reference's list. There is no gap to measure against it,
    and treating "unmeasured" as "tied" would forgive exactly the divergences
    the criterion exists to catch."""
    b = dist(x=-1.0, y=-1.05)           # reference's own top two are tied
    a = dist(q=-0.01, x=-9.0)           # variant picks a token b never reported
    _, agree, marg, _ = kld_raw.score_pair(a, b)
    assert agree is False
    assert marg is False


def test_band_is_documented_and_conservative():
    # tc-tol forgives a flip inside 0.02 of the logit range; on the models this
    # project measures that is 0.5-1.1 nats (see the derivation in the script).
    # The default must sit at the conservative end of that translation.
    assert 0.4 <= BAND <= 0.6
