"""Multi-token-prediction heads are admitted as training-only, never consumed.

The staged contract (suite plan, MTP profile item): first admit exports that
declare present MTP tensors as non-required, without changing dense decoding;
reject a profile that *requires* consumption until a verifier exists. Both
halves are gated here, and the identity half is the one that matters — an
export carrying predictor blocks must decode bit-for-bit like one without.
"""
import json
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / "runner"
    if not exe.exists():
        pytest.skip("runner binary not built")
    return exe


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    d = tmp_path_factory.mktemp("mtp")
    def gen(name, *flags):
        out = d / name
        subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py",
                        *flags, str(out)], check=True, cwd=ROOT,
                       stdout=subprocess.DEVNULL)
        return out
    return {
        "plain": gen("plain.gguf"),
        "mtp": gen("mtp.gguf", "--mtp-layers", "1"),
        "mtp2": gen("mtp2.gguf", "--mtp-layers", "2"),
        "requires": gen("req.gguf", "--mtp-layers", "1", "--agent-feature", "mtp"),
    }


def _run(runner_bin, model, n=8):
    return subprocess.run(
        [runner_bin, "-m", str(model), "-p", "hello", "-n", str(n),
         "--temp", "0", "--gpu", "off"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)


def test_declared_mtp_blocks_do_not_change_decoding(runner_bin, models):
    """The whole point of admitting training-only heads: they are inert."""
    plain = _run(runner_bin, models["plain"])
    assert plain.returncode == 0, plain.stderr.decode(errors="replace")
    for key in ("mtp", "mtp2"):
        got = _run(runner_bin, models[key])
        assert got.returncode == 0, got.stderr.decode(errors="replace")
        assert got.stdout == plain.stdout, (
            f"{key} changed dense decoding; MTP blocks must be inert")


def test_backbone_depth_excludes_the_predictor_blocks(runner_bin, models):
    plain = _run(runner_bin, models["plain"]).stderr.decode(errors="replace")
    two = _run(runner_bin, models["mtp2"]).stderr.decode(errors="replace")
    def layers(text):
        line = [l for l in text.splitlines() if l.startswith("loaded ")][0]
        return int(line.split("|")[2].strip().split()[0])
    assert layers(two) == layers(plain), "MTP blocks must not inflate the depth"
    assert "mtp: 2 predictor block(s) declared and excluded" in two
    assert "mtp:" not in plain, "a model without MTP must not mention it"


def test_profile_requiring_consumption_is_refused(runner_bin, models):
    """A profile that requires MTP consumption asks for a verifier this build
    does not have. That must fail with its own reason, not read as a typo in
    an unknown feature name."""
    proc = _run(runner_bin, models["requires"])
    assert proc.returncode != 0
    err = proc.stderr.decode(errors="replace")
    assert "requires MTP consumption" in err
    assert "does not implement" in err
    assert "unsupported required agent feature" not in err


def test_server_reports_declared_but_unconsumed_heads(runner_bin, models):
    """A controller must be able to see that predictor blocks were present and
    excluded, instead of inferring it from a layer count."""
    import socket
    import time
    import urllib.request
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    proc = subprocess.Popen(
        [runner_bin, "-m", str(models["mtp"]), "--serve", "--port", str(port),
         "--gpu", "off"], cwd=ROOT,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(120):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
                break
            except Exception:
                time.sleep(0.5)
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/capabilities",
                                    timeout=10) as r:
            caps = json.load(r)
    finally:
        proc.terminate()
        proc.wait(timeout=30)
    assert caps["mtp"] == {"declared_layers": 1, "consumed": False}
