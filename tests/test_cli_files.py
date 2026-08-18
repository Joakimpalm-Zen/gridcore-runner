"""What the CLI binary does with its inputs and its debug switches.

Started as RNR-011 (a prompt/schema file that cannot be read must fail with a
clear error, never a crash from an unchecked ftell/fread) and now also covers
the interactive-chat input path and the sampled-token debug env var — the
places where main.c's own handling of input is the behaviour under test rather
than the engine's.
"""
import os
import pathlib
import re
import subprocess
import sys
import time

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    return exe


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    m = tmp_path_factory.mktemp("m") / "test.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", str(m)],
                   check=True, cwd=ROOT)
    return m


def _run(runner_bin, model, *args):
    return subprocess.run([runner_bin, "-m", model, "--gpu", "off", *args],
                          cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          timeout=20)


def test_missing_prompt_file_is_a_clean_error(runner_bin, model):
    proc = _run(runner_bin, model, "-f", "/nonexistent/prompt.txt", "-n", "1")
    assert proc.returncode != 0
    assert b"cannot read" in proc.stderr


def test_missing_schema_file_is_a_clean_error(runner_bin, model):
    proc = _run(runner_bin, model, "-p", "hi", "--json-schema",
                "/nonexistent/schema.json", "-n", "1")
    assert proc.returncode != 0
    assert b"cannot read" in proc.stderr


def test_readable_prompt_file_loads(runner_bin, model, tmp_path):
    pf = tmp_path / "prompt.txt"
    pf.write_text("Hello from a file")
    proc = _run(runner_bin, model, "-f", str(pf), "-n", "1", "--temp", "0")
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")


def test_gpu_off_also_applies_to_speculative_draft(runner_bin, model):
    proc = _run(runner_bin, model, "-p", "hi", "--draft", str(model),
                "--draft-k", "2", "-n", "2", "--temp", "0")
    stderr = proc.stderr.decode(errors="replace")
    assert proc.returncode == 0, stderr
    assert "draft:" in stderr
    assert "gpu-split:" not in stderr
    assert "gpu: CUDA backend" not in stderr


def _chat(runner_bin, model, system, n_ctx, stdin=b"hello\n/exit\n"):
    argv = [runner_bin, "-m", model, "--gpu", "off", "-i",
            "-c", str(n_ctx), "-n", "1", "--temp", "0"]
    if system is not None:
        argv += ["--system", system]
    return subprocess.run(argv, input=stdin, cwd=ROOT,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          timeout=300)


def test_interactive_prompt_is_rendered_whole_or_refused(runner_bin, model):
    """A long --system must never be silently shortened (carry-over from the
    server's own render_prompt_alloc rule).

    Interactive chat rendered into a fixed char rendered[16384] and threw away
    the renderer's needed-size return. render_messages truncates the TAIL, so a
    30 KB system prompt did not fail the turn -- it deleted the user's question
    and the generation header and asked the model something else, at a size the
    context check then reported as comfortably fitting.

    The fixture's byte-fallback vocabulary runs ~1.4 tokens per byte, so the
    two cases below are far apart: a 30000-byte system prompt is ~42k tokens
    and cannot fit -c 30000, while the 16 KB truncation is ~23k tokens and fits
    with room to spare. Before the fix the first case ran silently; the whole
    point is that it now says so.
    """
    big = _chat(runner_bin, model, "word " * 6000, 30000)
    assert b"context full" in big.stderr, big.stderr[-400:]

    # and the complement, so the fix cannot be "always refuse": a system
    # prompt that does fit still runs the turn
    ok = _chat(runner_bin, model, "word " * 200, 30000)
    assert ok.returncode == 0, ok.stderr[-400:]
    assert b"context full" not in ok.stderr


def test_a_pasted_line_longer_than_the_read_buffer_is_one_turn(runner_bin, model):
    """Pasting a file into chat mode must ask ONE question.

    The input line was read with fgets into a fixed char line[8192], so a
    longer paste -- an ordinary "here is my file, what is wrong with it" --
    came back as two fgets calls and was answered as two unrelated turns, the
    second one starting mid-token in the middle of the user's text.
    """
    proc = _chat(runner_bin, model, None, 30000,
                 stdin=b"x" * 10000 + b"\n/exit\n")
    assert proc.returncode == 0, proc.stderr[-400:]
    assert proc.stderr.count(b"tok/s]") == 1, proc.stderr[-400:]


def _sampled_ids(runner_bin, model, env, *extra):
    proc = subprocess.run([runner_bin, "-m", model, "--gpu", "off",
                           "-p", "hi", "-n", "6", "--temp", "0", *extra],
                          cwd=ROOT, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, timeout=60)
    assert proc.returncode == 0, proc.stderr[-400:]
    return re.search(rb"\n(?: \d+){6}", proc.stderr) is not None


def test_debug_tokens_env_prints_every_sampled_id(runner_bin, model):
    """RUNNER_DEBUG_TOKENS traces the sampled ids on BOTH decode paths.

    The switch is undocumented and was ungated, which is how its getenv came to
    sit inside the token loop of the plain and speculative decoders alike.
    Pinned here so the read can be cached without the trace quietly dying.
    """
    on = dict(os.environ, RUNNER_DEBUG_TOKENS="1")
    off = {k: v for k, v in os.environ.items() if k != "RUNNER_DEBUG_TOKENS"}

    assert _sampled_ids(runner_bin, model, on)
    assert not _sampled_ids(runner_bin, model, off)
    # the speculative decoder samples in its own loop, with its own read
    assert _sampled_ids(runner_bin, model, on,
                        "--draft", str(model), "--draft-k", "2")
    assert not _sampled_ids(runner_bin, model, off,
                            "--draft", str(model), "--draft-k", "2")


@pytest.mark.parametrize("flag,arg", [
    ("--prune-experts", "keep.json"),
    ("--type-plan", "plan.json"),
    ("--quant", "q8_0"),
])
def test_rewrite_only_flags_require_quantize(runner_bin, model, flag, arg):
    """A flag that only means something inside --quantize is an error without
    it, never a silent no-op.

    --prune-experts already said so; --type-plan and --quant did not. Both were
    parsed, stored, and then read only inside the `if (quant_out)` block that a
    generating run never enters, so `runner -m m.gguf --quant q8_0 -p hi` ran
    at the model's own precision and said nothing -- the same silent discard
    the --chat-template fix was written about, in a place where the user
    believes they asked for a different file on disk.
    """
    proc = _run(runner_bin, model, flag, arg, "-p", "hi", "-n", "1", "--temp", "0")
    assert proc.returncode != 0, proc.stdout[-200:]
    assert b"requires --quantize" in proc.stderr, proc.stderr[-300:]


def test_quantize_still_accepts_all_three(runner_bin, model, tmp_path):
    out = tmp_path / "requantized.gguf"
    proc = _run(runner_bin, model, "--quantize", str(out), "--quant", "q8_0")
    assert proc.returncode == 0, proc.stderr[-400:]
    assert out.exists()


def _sampled(runner_bin, model, seed):
    return subprocess.run([runner_bin, "-m", model, "--gpu", "off",
                           "-p", "hello", "-n", "12", "-s", str(seed),
                           "--temp", "1.0"],
                          cwd=ROOT, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, timeout=60)


def test_seed_zero_is_refused_rather_than_silently_randomised(runner_bin, model):
    """`-s 0` must not run as if it had been honoured.

    The sampler's xorshift64 has a fixed point at state 0 -- it stays 0 and
    every draw comes back 0.0 -- so 0 genuinely cannot be used as a seed. The
    old code said that by replacing it with time(NULL), which turned a request
    for a REPRODUCIBLE run into a random one with nothing on stderr: measured
    on the fixture, three `-s 0` runs a second apart produced three different
    completions while `-s 7` reproduced exactly.

    The sleep is what makes this able to fail. time(NULL) is whole seconds, so
    two runs inside the same second agreed by accident and hid the bug.
    """
    zero = _sampled(runner_bin, model, 0)
    assert zero.returncode != 0
    assert b"not a usable seed" in zero.stderr, zero.stderr[-200:]

    # every other seed is untouched and still reproduces across the same gap
    a = _sampled(runner_bin, model, 7)
    time.sleep(1.2)
    b = _sampled(runner_bin, model, 7)
    assert a.returncode == 0 and a.stdout == b.stdout
    assert a.stdout != _sampled(runner_bin, model, 8).stdout
