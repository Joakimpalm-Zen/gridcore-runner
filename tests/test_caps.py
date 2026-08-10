import json
import pathlib
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_gpu_quant_report_matches_metal_backend_support():
    runner = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not runner.exists():
        return

    caps = json.loads(subprocess.run(
        [runner, "--caps"], cwd=ROOT, stdout=subprocess.PIPE, check=True,
    ).stdout)

    quants = set(caps["quants"])
    gpu_quants = set(caps["gpu_quants"])
    assert gpu_quants <= quants

    if (caps.get("gpu") or {}).get("backend") == "metal":
        assert {"F16", "BF16", "Q2_K", "Q3_K", "IQ4_NL", "IQ4_XS"} <= gpu_quants
