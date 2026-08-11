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
llama-imatrix, llama-server); skipped when absent. IQ2/IQ1 quantization
requires an importance matrix, generated here from the fixture itself.
The reference is queried through llama-server's /completion (explicit
pure-greedy sampler, cache_prompt off) — the same protocol as
cert-greedy-identity.py, because llama-cli's interactive TUI cannot be
scripted reliably.
"""
import json
import os
import pathlib
import socket
import subprocess
import sys
import time
import urllib.request

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


def _ref_completion(model, prompt, n):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    proc = subprocess.Popen(
        [pathlib.Path(BIN) / "llama-server", "-m", model,
         "--port", str(port), "--host", "127.0.0.1"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(120):
            if proc.poll() is not None:
                raise RuntimeError(f"llama-server exited {proc.returncode}")
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
                break
            except OSError:
                time.sleep(0.5)
        body = json.dumps({
            "prompt": prompt, "temperature": 0, "n_predict": n,
            "repeat_penalty": 1.0, "top_k": 0, "top_p": 1.0, "min_p": 0.0,
            "seed": 0, "cache_prompt": False,
        }).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/completion", data=body,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.loads(r.read())["content"]
    finally:
        proc.kill()
        proc.wait()


@pytest.mark.parametrize("t", IQ_TYPES)
def test_iquant_matches_llamacpp_greedy(runner_bin, iq_files, t):
    model = iq_files[t]
    prompt = "hello world"
    ours = subprocess.run(
        [runner_bin, "-m", model, "-p", prompt, "-n", "16",
         "--temp", "0", "--gpu", "off"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
    assert ours.returncode == 0, ours.stderr.decode(errors="replace")
    a = ours.stdout.decode(errors="replace")
    assert a.startswith(prompt), a[:80]
    ta = a[len(prompt):].strip()
    tb = _ref_completion(model, prompt, 16).strip()
    assert ta and ta == tb, f"{t}: runner={ta!r} llama.cpp={tb!r}"
