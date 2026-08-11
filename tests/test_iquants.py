"""Codebook i-quants (IQ1_S/M, IQ2_XXS/XS/S, IQ3_XXS/S): dequant parity.

The dequant paths are transcriptions of llama.cpp b10353's per-block
arithmetic over verbatim codebook grids, so the gate is differential, not
unit-shaped: quantize ONE tiny f32 model into every claimed i-quant type
with llama.cpp's own quantizer, then require the runner's greedy CPU
output on each file to be byte-identical to llama-cli on the same file.
A transcription slip (wrong grid index math, wrong sign table, wrong
scale packing) cannot survive that comparison, while a unit test against
hand-computed blocks would only re-state the transcription.

Needs a llama.cpp build directory in RUNNER_LLAMA_CPP_BIN (llama-quantize,
llama-imatrix, llama-cli); skipped when absent. IQ2/IQ1 quantization
requires an importance matrix, generated here from the fixture itself.
"""
import os
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
BIN = os.environ.get("RUNNER_LLAMA_CPP_BIN")

IQ_TYPES = ["IQ3_XXS", "IQ3_S", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ1_S", "IQ1_M"]

pytestmark = pytest.mark.skipif(
    not BIN or not (pathlib.Path(BIN) / "llama-quantize").exists(),
    reason="RUNNER_LLAMA_CPP_BIN with llama-quantize/llama-imatrix/llama-cli required")


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    return exe


@pytest.fixture(scope="module")
def iq_files(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("iq")
    base = tmp / "base.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", str(base)],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    corpus = tmp / "corpus.txt"
    corpus.write_text("the quick brown fox jumps over the lazy dog " * 40)
    imatrix = tmp / "imatrix.gguf"
    subprocess.run([pathlib.Path(BIN) / "llama-imatrix", "-m", base,
                    "-f", corpus, "-o", imatrix, "--ctx-size", "128"],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                   timeout=300)
    files = {}
    for t in IQ_TYPES:
        out = tmp / f"m-{t}.gguf"
        subprocess.run([pathlib.Path(BIN) / "llama-quantize", "--imatrix", imatrix,
                        str(base), str(out), t],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=300)
        files[t] = out
    return files


@pytest.mark.parametrize("t", IQ_TYPES)
def test_iquant_matches_llamacpp_greedy(runner_bin, iq_files, t):
    model = iq_files[t]
    ours = subprocess.run(
        [runner_bin, "-m", model, "-p", "hello world", "-n", "16",
         "--temp", "0", "--gpu", "off"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
    assert ours.returncode == 0, ours.stderr.decode(errors="replace")
    theirs = subprocess.run(
        [pathlib.Path(BIN) / "llama-cli", "-m", model, "-p", "hello world",
         "-n", "16", "--temp", "0", "-no-cnv", "--no-warmup", "--seed", "0"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=300)
    assert theirs.returncode == 0
    # llama-cli echoes the prompt inside its own framing; compare the
    # completion tail after the shared prompt text on both sides
    a = ours.stdout.decode(errors="replace")
    b = theirs.stdout.decode(errors="replace")
    ta = a.split("hello world", 1)[1].strip()
    tb = b.split("hello world", 1)[1].strip()
    assert ta and ta == tb, f"{t}: runner={ta!r} llama.cpp={tb!r}"
