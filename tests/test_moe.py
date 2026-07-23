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


def _generate(runner_bin, model, prompt="hello world", n=12, extra=("--gpu", "off")):
    proc = subprocess.run(
        [runner_bin, "-m", str(model), "-p", prompt, "-n", str(n),
         "--temp", "0", *extra],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
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


def test_split_expert_layout_matches_the_dense_oracle(runner_bin, models):
    # legacy split per-expert tensors (older Mixtral GGUFs) must produce the
    # same result as the fused layout — validates the split loader path.
    dense = _generate(runner_bin, f"{models}.dense.gguf")
    moe3 = _generate(runner_bin, f"{models}.moe3.gguf")
    assert moe3 == dense, "split-expert MoE must be token-identical to the dense FFN"


def test_moe_partial_cpu_offload_matches_dense(runner_bin, models):
    # `--gpu-layers 1` runs layer 0 on the GPU and layer 1 on the CPU (partial
    # offload) when a GPU is present; on a GPU-less host it simply falls back to
    # CPU. Either way the output must equal the dense oracle. Guards the MoE
    # VRAM-accounting / partial-upload path (both fused and split layouts).
    dense = _generate(runner_bin, f"{models}.dense.gguf")
    fused = _generate(runner_bin, f"{models}.moe1.gguf", extra=("--gpu-layers", "1"))
    split = _generate(runner_bin, f"{models}.moe3.gguf", extra=("--gpu-layers", "1"))
    assert fused == dense, "fused MoE with partial offload must match dense"
    assert split == dense, "split MoE with partial offload must match dense"
