"""scripts/tokenizer-corpus.py: the 721-string tokenizer conformance corpus.

The corpus is the input to the tokenizer differential (difftok.py) and its
length is pinned by the committed reference id captures, so a string that
silently fails to reach the file is a boundary case the gate stops testing
without saying so.
"""

import importlib.util
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "tokenizer_corpus", ROOT / "scripts/tokenizer-corpus.py")
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def test_the_corpus_is_exactly_the_pinned_length():
    assert len(MOD.corpus()) == MOD.TARGET


def test_hand_written_blocks_overflowing_the_target_are_refused(monkeypatch):
    """corpus() ends `return out[:TARGET]`, so hand-written blocks that grow
    past TARGET are silently truncated — the newest additions first, since
    they are appended last. main()'s `len(c) != TARGET` check cannot catch
    that: the slice makes it unsatisfiable by construction.

    Shrinking TARGET below the hand-written count reproduces exactly that
    state without having to add 162 strings to the file.
    """
    monkeypatch.setattr(MOD, "TARGET", 8)

    with pytest.raises(SystemExit) as caught:
        MOD.corpus()

    assert "8" in str(caught.value)
