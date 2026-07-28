"""Sparse-MoE inference is validated against the trusted dense path.

make-test-moe.py emits a dense model plus three MoE variants (all F32, weights
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
import json
import os
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if os.name == "nt" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    return exe


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    base = tmp_path_factory.mktemp("moe") / "m"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-moe.py", str(base)],
                   check=True, cwd=ROOT)
    return base


def _generate(runner_bin, model, prompt="hello world", n=12, extra=("--gpu", "off")):
    proc = _run(runner_bin, model, prompt=prompt, n=n, extra=extra)
    return proc.stdout


def _run(runner_bin, model, prompt="hello world", n=12, extra=("--gpu", "off")):
    proc = subprocess.run(
        [runner_bin, "-m", str(model), "-p", prompt, "-n", str(n),
         "--temp", "0", *extra],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    return proc


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


def test_moe_expert_cpu_placement_matches_dense(runner_bin, models):
    """Attention stays eligible for CUDA while sparse experts remain on CPU."""
    dense = _generate(runner_bin, f"{models}.dense.gguf")
    proc = _run(runner_bin, f"{models}.moe1.gguf",
                extra=("--cpu-moe", "--gpu-layers", "2"))
    assert proc.stdout == dense
    caps = subprocess.run([runner_bin, "--caps"], cwd=ROOT,
                          stdout=subprocess.PIPE, check=True)
    if b'"available":true' in caps.stdout:
        assert b"experts on CPU" in proc.stderr


def test_moe_gpu_forward_does_not_fall_back(runner_bin, models):
    """A GPU MoE run must actually execute on the GPU — assert the runtime-
    fallback warning never fires. The 69b8085 slice-nbytes defect made every
    MoE forward fail its binding bounds check and silently continue on the
    CPU: output stayed token-identical (the fallback IS the CPU oracle), so
    only the fallback warning can catch this class. moe2 routes through BOTH
    experts (top-2), so the e>=1 slice descriptors are exercised."""
    caps = subprocess.run([runner_bin, "--caps"], cwd=ROOT,
                          stdout=subprocess.PIPE, check=True)
    if (b'"backend":"cuda"' not in caps.stdout and
            b'"backend":"metal"' not in caps.stdout):
        pytest.skip("no GPU backend on this host")
    dense = _generate(runner_bin, f"{models}.dense.gguf")
    for variant in ("moe1", "moe2", "moe3"):
        proc = _run(runner_bin, f"{models}.{variant}.gguf", extra=())
        assert proc.stdout == dense, f"{variant} GPU output must match dense"
        assert b"continuing on CPU" not in proc.stderr, (
            f"{variant} silently fell back to the CPU path:\n"
            + proc.stderr.decode(errors="replace"))


def test_caps_advertise_moe_tensor_placement(runner_bin):
    caps = json.loads(subprocess.run(
        [runner_bin, "--caps"], cwd=ROOT, stdout=subprocess.PIPE,
        check=True, text=True).stdout)
    assert caps["tensor_placement"]["cpu_moe"] is True
