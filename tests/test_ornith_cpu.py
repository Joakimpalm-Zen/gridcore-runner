"""Public CPU tracer for the Ornith-1.0/Qwen3.5 hybrid architecture."""
import math
import json
import pathlib
import subprocess
import sys
import os

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNNER = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")

def test_qwen35_hybrid_loads_and_decodes(tmp_path):
    model = tmp_path / "ornith.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-ornith.py", model],
                   check=True, cwd=ROOT)
    proc = subprocess.run(
        [RUNNER, "-m", model, "-p", "hi", "-n", "2",
         "--temp", "0", "--gpu", "off", "-c", "32"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=20,
    )
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    assert b"hybrid SSM/attention architecture" not in proc.stderr
    # This greedy suffix is the llama.cpp reference result for the exact same
    # generated weights and prompt-token sequence. It protects the recurrent
    # state orientation, Q scaling, group broadcast, convolution ordering and
    # the full-attention post-norm placement as one end-to-end contract.
    assert proc.stdout.splitlines() == [b"hiii"]


def test_qwen35_legacy_ornith_dt_name_loads(tmp_path):
    model = tmp_path / "ornith-legacy.gguf"
    env = {**os.environ, "ORNITH_LEGACY_DT": "1"}
    subprocess.run([sys.executable, ROOT / "scripts/make-test-ornith.py", model],
                   check=True, cwd=ROOT, env=env)
    proc = subprocess.run(
        [RUNNER, "-m", model, "-p", "hi", "-n", "1",
         "--temp", "0", "--gpu", "off", "-c", "32"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20,
    )
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")


def test_qwen35_hybrid_gpu_auto_matches_cpu(tmp_path):
    caps = json.loads(subprocess.run(
        [RUNNER, "--caps"], cwd=ROOT, stdout=subprocess.PIPE,
        check=True, text=True).stdout)
    gpu_caps = caps.get("gpu")
    if not gpu_caps:
        return
    backend = gpu_caps.get("backend")
    model = tmp_path / "ornith-gpu.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-ornith.py", model],
                   check=True, cwd=ROOT)
    base = [RUNNER, "-m", model, "-p", "hi", "-n", "2",
            "--temp", "0", "-c", "32"]
    cpu = subprocess.run([*base, "--gpu", "off"], cwd=ROOT,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         timeout=20)
    gpu = subprocess.run([*base, "--gpu", "auto"], cwd=ROOT,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         timeout=20)
    assert cpu.returncode == 0, cpu.stderr.decode(errors="replace")
    assert gpu.returncode == 0, gpu.stderr.decode(errors="replace")
    assert gpu.stdout == cpu.stdout
    if backend == "cuda":
        assert b"CUDA backend" in gpu.stderr
        assert b"no kernels for the recurrent" not in gpu.stderr
    elif backend == "metal":
        assert b"not on the metal backend" in gpu.stderr
        assert b"Metal backend" not in gpu.stderr

    if backend == "cuda":
        fallback = subprocess.run(
            [*base, "--gpu", "auto"], cwd=ROOT, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, timeout=20,
            env={**os.environ, "RUNNER_CUDA_INJECT_FAILURE": "1"},
        )
        assert fallback.returncode == 0, fallback.stderr.decode(errors="replace")
        assert fallback.stdout == cpu.stdout
        assert b"injected CUDA runtime failure" in fallback.stderr


def test_qwen35_cuda_failure_after_recurrent_state_accumulates(tmp_path):
    """Fail the device *after* it has carried recurrent state, not on tile one.

    An unconditional hook can only ever fire on the first device forward, which
    for any real generation is the prefill tile — before a single SSM state
    update has been carried across a token boundary. That is the cheap half of
    the contract. The half that can actually corrupt output is a failure at
    decode, where `q35_hist`/`q35_state` hold accumulated history and the
    caller has to roll back to the pre-forward snapshot before recomputing on
    the host.

    `RUNNER_CUDA_INJECT_FAILURE=N` fails the Nth device forward, so N=2 lands
    on the first decode step — past prefill, with state live.
    """
    caps = json.loads(subprocess.run(
        [RUNNER, "--caps"], cwd=ROOT, stdout=subprocess.PIPE,
        check=True, text=True).stdout)
    if (caps.get("gpu") or {}).get("backend") != "cuda":
        return

    model = tmp_path / "ornith-late-fail.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-ornith.py", model],
                   check=True, cwd=ROOT)
    base = [RUNNER, "-m", model, "-p", "hi", "-n", "4",
            "--temp", "0", "-c", "32"]
    cpu = subprocess.run([*base, "--gpu", "off"], cwd=ROOT,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         timeout=20)
    late = subprocess.run(
        [*base, "--gpu", "auto"], cwd=ROOT, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, timeout=20,
        env={**os.environ, "RUNNER_CUDA_INJECT_FAILURE": "2"})
    assert cpu.returncode == 0, cpu.stderr.decode(errors="replace")
    assert late.returncode == 0, late.stderr.decode(errors="replace")
    # Naming the ordinal is what proves the failure landed past prefill; a hook
    # that always fired on tile one would report forward 1 here.
    assert b"injected CUDA runtime failure at device forward 2" in late.stderr, \
        late.stderr.decode(errors="replace")
    assert late.stdout == cpu.stdout
