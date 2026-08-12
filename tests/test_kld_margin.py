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


def test_band_is_documented_and_conservative():
    # tc-tol forgives a flip inside 0.02 of the logit range; on the models this
    # project measures that is 0.5-1.1 nats (see the derivation in the script).
    # The default must sit at the conservative end of that translation.
    assert 0.4 <= BAND <= 0.6
