"""Public CPU tracer for the Ornith-1.0/Qwen3.5 hybrid architecture."""
import math
import json
import pathlib
import subprocess
import sys
import os

ROOT = pathlib.Path(__file__).resolve().parents[1]

def test_qwen35_hybrid_loads_and_decodes(tmp_path):
    model = tmp_path / "ornith.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-ornith.py", model],
                   check=True, cwd=ROOT)
    proc = subprocess.run(
        [ROOT / "runner", "-m", model, "-p", "hi", "-n", "2",
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
        [ROOT / "runner", "-m", model, "-p", "hi", "-n", "1",
         "--temp", "0", "--gpu", "off", "-c", "32"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20,
    )
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")


def test_qwen35_hybrid_cuda_matches_cpu(tmp_path):
    caps = json.loads(subprocess.run(
        [ROOT / "runner", "--caps"], cwd=ROOT, stdout=subprocess.PIPE,
        check=True, text=True).stdout)
    if not caps.get("gpu"):
        return
    model = tmp_path / "ornith-gpu.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-ornith.py", model],
                   check=True, cwd=ROOT)
    base = [ROOT / "runner", "-m", model, "-p", "hi", "-n", "2",
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
    assert b"CUDA backend" in gpu.stderr
    assert b"no kernels for the recurrent" not in gpu.stderr

    fallback = subprocess.run(
        [*base, "--gpu", "auto"], cwd=ROOT, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, timeout=20,
        env={**os.environ, "RUNNER_CUDA_INJECT_FAILURE": "1"},
    )
    assert fallback.returncode == 0, fallback.stderr.decode(errors="replace")
    assert fallback.stdout == cpu.stdout
    assert b"injected CUDA runtime failure" in fallback.stderr
