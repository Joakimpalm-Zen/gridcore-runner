"""Public CPU tracer for the Ornith-1.0/Qwen3.5 hybrid architecture."""
import math
import pathlib
import subprocess
import sys

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
    assert proc.stdout == b"hiii\n"
