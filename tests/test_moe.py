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
        assert b"expert layers on CPU" in proc.stderr


def _split_line(stderr):
    for line in stderr.decode(errors="replace").splitlines():
        if line.startswith("gpu-split: experts "):
            return line
    return ""


@pytest.mark.parametrize("mode", [("--cpu-moe", "0"), ("--cpu-moe", "1"),
                                  ("--cpu-moe", "auto")])
@pytest.mark.parametrize("variant", ["moe1", "moe2", "moe3"])
def test_partial_expert_offload_matches_dense(runner_bin, models, variant, mode):
    """Partial expert offload places only some banks on the device, so the two
    placements must coexist inside one forward: a device-resident bank runs the
    CUDA MoE path against uploaded bindings while a host bank still bounces the
    activation tile. Output must stay identical to the dense oracle for every
    count, in both expert layouts (moe3 is split-per-expert) and at top-2
    (moe2), which is what proves the two paths agree rather than one of them
    quietly serving every layer."""
    dense = _generate(runner_bin, f"{models}.dense.gguf")
    proc = _run(runner_bin, f"{models}.{variant}.gguf", extra=mode)
    assert proc.stdout == dense, f"{variant} {mode} must match the dense oracle"
    # the same silent-fallback guard the full-GPU test applies: a binding miss
    # would degrade to the CPU path and still print identical tokens
    assert b"continuing on CPU" not in proc.stderr, (
        f"{variant} {mode} fell back to the CPU path:\n"
        + proc.stderr.decode(errors="replace"))


def test_partial_expert_counts_are_reported(runner_bin, models):
    """The split line must say where the experts actually went — the 5070
    install could not tell that --cpu-moe was leaving 8.8 GB of its card idle
    because nothing reported the expert placement."""
    caps = subprocess.run([runner_bin, "--caps"], cwd=ROOT,
                          stdout=subprocess.PIPE, check=True)
    if b'"backend":"cuda"' not in caps.stdout:
        pytest.skip("split reporting is CUDA-side")
    all_host = _run(runner_bin, f"{models}.moe1.gguf", extra=("--cpu-moe",))
    assert "experts 0/2 layers on GPU, 2 on host" in _split_line(all_host.stderr)
    none_host = _run(runner_bin, f"{models}.moe1.gguf", extra=("--cpu-moe", "0"))
    assert "experts 2/2 layers on GPU, 0 on host" in _split_line(none_host.stderr)
    one_host = _run(runner_bin, f"{models}.moe1.gguf", extra=("--cpu-moe", "1"))
    assert "experts 1/2 layers on GPU, 1 on host" in _split_line(one_host.stderr)
    auto = _run(runner_bin, f"{models}.moe1.gguf", extra=("--cpu-moe", "auto"))
    assert "(auto fit)" in _split_line(auto.stderr)


def test_composition_and_idle_vram_are_reported(runner_bin, models):
    """Two silent-loss cases the 5070 install hit, now spoken aloud.

    Capping the attention split under tensor-role placement measured 10.3
    tok/s against 12.7 (all-host) and 14.6 (layer split alone) — the worst of
    both, with nothing said. And all-or-nothing placement left 8.8 GB of a
    12 GB card idle because no line reported what the experts were doing."""
    caps = subprocess.run([runner_bin, "--caps"], cwd=ROOT,
                          stdout=subprocess.PIPE, check=True)
    if b'"backend":"cuda"' not in caps.stdout:
        pytest.skip("both warnings are CUDA-side")
    combo = _run(runner_bin, f"{models}.moe1.gguf",
                 extra=("--cpu-moe", "--gpu-layers", "1"))
    assert b"caps the attention split below" in combo.stderr
    assert b"--cpu-moe N (or auto)" in combo.stderr
    idle = _run(runner_bin, f"{models}.moe1.gguf", extra=("--cpu-moe",))
    assert b"of the budget is unused" in idle.stderr
    assert b"expert layers on the GPU" in idle.stderr
    # the advice must never promise more banks than the model has
    line = [l for l in idle.stderr.decode(errors="replace").splitlines()
            if "would keep" in l][0]
    placed, total = line.split("would keep ")[1].split(" expert")[0].split(" of ")
    assert 0 < int(placed) <= int(total) == 2
    # auto already spends the headroom, so it must not also advertise it
    auto = _run(runner_bin, f"{models}.moe1.gguf", extra=("--cpu-moe", "auto"))
    assert b"of the budget is unused" not in auto.stderr


def test_cpu_moe_count_is_parsed_strictly(runner_bin, models):
    """A bare --cpu-moe must keep its original all-on-host meaning even when a
    non-numeric argument follows, and a malformed count must fail loudly rather
    than being silently read as 'all' (the env-parsing footgun class)."""
    bare = _run(runner_bin, f"{models}.moe1.gguf", extra=("--cpu-moe", "--temp", "0"))
    assert bare.returncode == 0
    bad = subprocess.run(
        [runner_bin, "-m", f"{models}.moe1.gguf", "-p", "x", "-n", "1",
         "--cpu-moe", "99999999999999999999"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
    assert bad.returncode != 0
    assert b"expects an integer" in bad.stderr


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
    assert caps["tensor_placement"]["cpu_moe_partial"] is True
