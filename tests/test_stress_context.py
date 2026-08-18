"""Guards for two context-probe bugs that produced false engine findings.

Both were found by the 2026-08-04 shelf pass (recorded in the CHANGELOG).
Neither was an engine defect; the first was published by the probe as two.
"""
import importlib.util
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load():
    spec = importlib.util.spec_from_file_location(
        "stress_context", ROOT / "scripts" / "stress-context.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sc = load()

GRANITE = ("error: unsupported architecture 'granite' — it would load but "
           "produce incorrect output with llama-style attention")


def rows(stderr, rc=1):
    return [{"requested_ctx": c, "rc": rc, "resolved_ctx": None, "kv_mb": None,
             "last_stderr": stderr, "explains_itself": False}
            for c in (0, 32768, 1000000)]


def test_architecture_refusal_is_not_a_context_finding():
    """Granite never reaches context handling, so it cannot fail at it.

    The probe scored the architecture refusal against its context-keyword
    heuristic and emitted two findings — "refused a too-large context without
    saying why" and "--ctx 0 (auto-fit) did not run". The message is correct
    and says exactly why; it just talks about the architecture, because that
    is the gate that stopped it.
    """
    assert sc.refused_architecture(rows(GRANITE))
    assert sc.findings_for(rows(GRANITE)) == []


def test_a_real_context_failure_is_still_reported():
    """The fix must not silence the case the probe exists to catch."""
    bad = rows("error: something went wrong", rc=1)
    assert not sc.refused_architecture(bad)
    assert sc.findings_for(bad), "a genuine unexplained refusal must still fire"


def test_an_uncapped_million_token_context_is_still_reported():
    good = rows("", rc=0)
    for r in good:
        r["explains_itself"] = True
    good[0]["resolved_ctx"] = 4096
    good[-1]["resolved_ctx"] = 1000000
    assert any("without capping" in f for f in sc.findings_for(good))


def test_the_kv_pressure_note_is_captured():
    """The last stderr line is the throughput line, so it hid the answer.

    `2633d0b` added the note that connects a large -c to the layers it costs.
    A probe asking whether the engine explains itself must record it.
    """
    stderr = (
        "gpu-split: budget=7.46GB fixed=1.40GB G=15/48 full=0 used=7.09GB\n"
        "note: the KV cache for ctx 32768 is 3.62 GB on the device and 33 of "
        "48 layers ran out of room because of it — a smaller -c or --kv q8 "
        "(about half) moves layers back\n"
        "warning: weights are 7.4 GB but only 5.0 GB of RAM is available\n"
        "prompt: 2 tok, 0.37 tok/s | gen: 2 tok, 2.03 tok/s\n")
    notes = sc.notes_from(stderr)
    assert any("ran out of room because of it" in n for n in notes)
    assert any(n.startswith("warning:") for n in notes)
    assert not any(n.startswith("prompt:") for n in notes)


# The probe's headline contract is "refuse-or-cap WITH A REASON". Scoring that
# over the whole of stderr cannot answer it: `-v` is always passed, and every
# successful load prints "context N (train M)", "kv cache N MB" and a banner
# ending "| ctx N |" before anything can go wrong afterwards.

REAL_VERBOSE_LOAD = (
    "heads                    9 (3 kv)\n"
    "context                  4096 (train 8192)\n"
    "kv cache                 94.4 MB (fp16)\n"
    "loaded m.gguf | llama | 30 layers | ctx 4096 | 4 threads | 0.07s\n"
)


def _stub_runner(tmp_path, stderr_text, exit_code):
    import os
    import sys as _sys
    script = tmp_path / "stub_runner.py"
    script.write_text(
        "import sys\n"
        f"sys.stderr.write({stderr_text!r})\n"
        f"sys.exit({exit_code})\n")
    if os.name == "nt":
        launcher = tmp_path / "runner.cmd"
        launcher.write_text(f'@"{_sys.executable}" "{script}" %*\r\n')
    else:
        launcher = tmp_path / "runner"
        launcher.write_text(f'#!/bin/sh\nexec "{_sys.executable}" "{script}" "$@"\n')
        launcher.chmod(0o755)
    return launcher


def test_an_opaque_refusal_after_a_healthy_load_is_still_unexplained(tmp_path):
    runner = _stub_runner(tmp_path, REAL_VERBOSE_LOAD + "error: kaboom\n", 1)

    probe = sc.probe(runner, tmp_path / "m.gguf", 1000000, 60)

    assert probe["rc"] == 1
    assert probe["explains_itself"] is False
    assert sc.findings_for([probe, probe, probe]) == [
        "refused a too-large context without saying why",
        "--ctx 0 (auto-fit) did not run",
    ]


def test_a_refusal_that_names_the_context_cost_is_explained(tmp_path):
    runner = _stub_runner(
        tmp_path,
        REAL_VERBOSE_LOAD +
        "error: cannot allocate buffers (ctx 1000000 needs 23040.0 MB KV cache)\n",
        1)

    probe = sc.probe(runner, tmp_path / "m.gguf", 1000000, 60)

    assert probe["explains_itself"] is True
