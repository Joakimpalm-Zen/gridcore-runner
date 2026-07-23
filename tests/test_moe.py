"""Sparse-MoE inference is validated against the trusted dense path.

make-test-moe.py emits a dense model plus two MoE variants (all F32, weights
shared except the FFN) each constructed to be MATHEMATICALLY IDENTICAL to the
dense FFN, so the runner's already-verified dense path is the oracle — no
separate reference engine is needed:

  moe1  expert_count=2, expert_used=1, zero router -> top-1 picks expert 0,
        expert 0 == dense, expert 1 = zeros. Weight 1.0 -> output == dense.
  moe2  expert_count=2, expert_used=2, zero router -> weights [0.5, 0.5], both
        experts == dense. 0.5*y + 0.5*y == y -> output == dense.

Both exercise the full MoE pipeline (metadata parse, router matmul + softmax +
top-k + renormalization, 3D expert slicing at the correct per-expert offsets,
SwiGLU, weighted sum). The FFN is scaled up in the generator so it drives the
logits — a broken MoE produces different tokens (verified during development).
"""
import pathlib
import subprocess

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
    base = tmp_path_factory.mktemp("moe") / "m"
    subprocess.run(["python3", ROOT / "scripts/make-test-moe.py", str(base)],
                   check=True, cwd=ROOT)
    return base


def _generate(runner_bin, model, prompt="hello world", n=12):
    proc = subprocess.run(
        [runner_bin, "-m", str(model), "-p", prompt, "-n", str(n),
         "--gpu", "off", "--temp", "0"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    return proc.stdout


def test_moe_top1_matches_the_dense_oracle(runner_bin, models):
    dense = _generate(runner_bin, f"{models}.dense.gguf")
    moe1 = _generate(runner_bin, f"{models}.moe1.gguf")
    assert moe1 == dense, "expert_used=1 MoE must be token-identical to the dense FFN"


def test_moe_top2_renormalized_matches_the_dense_oracle(runner_bin, models):
    dense = _generate(runner_bin, f"{models}.dense.gguf")
    moe2 = _generate(runner_bin, f"{models}.moe2.gguf")
    assert moe2 == dense, "expert_used=2 (0.5/0.5) MoE must be token-identical to dense"
