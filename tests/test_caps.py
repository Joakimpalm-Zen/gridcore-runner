import json
import pathlib
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]


# What each backend is expected to have kernels for, stated here on purpose.
#
# `--caps` now generates both lists from one table in main.c filtered by the
# backend's own gpu_quant_ok(), so the advertised list and the loader cannot
# drift apart the way they did on Metal. But that also means they would agree
# about being wrong: a kernel added without extending gpu_type_ok, or an entry
# removed from the table, changes both at once and no self-consistency check
# can see it. This restates the fact independently, so a format gained or lost
# has to be admitted in two places by two different authors.
METAL_QUANTS = {
    "F32", "F16", "BF16", "Q8_0", "Q4_0", "Q4_1", "Q5_0", "Q5_1",
    "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "IQ4_NL", "IQ4_XS", "MXFP4",
}
# CUDA has neither Q2_K nor BF16 today; both are CPU-only there.
CUDA_QUANTS = {
    "F32", "F16", "Q8_0", "Q4_0", "Q4_1", "Q5_0", "Q5_1",
    "Q3_K", "Q4_K", "Q5_K", "Q6_K", "IQ4_NL", "IQ4_XS", "MXFP4",
}


def test_gpu_quant_report_matches_backend_support():
    runner = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not runner.exists():
        return

    caps = json.loads(subprocess.run(
        [runner, "--caps"], cwd=ROOT, stdout=subprocess.PIPE, check=True,
    ).stdout)

    quants = set(caps["quants"])
    gpu_quants = set(caps["gpu_quants"])
    assert gpu_quants <= quants, sorted(gpu_quants - quants)

    backend = (caps.get("gpu") or {}).get("backend")
    expected = {"metal": METAL_QUANTS, "cuda": CUDA_QUANTS}.get(backend)
    if expected is not None:
        # equality, not a subset: an over-claim is the failure that shipped
        # once already, and a subset assertion is exactly what missed it
        assert gpu_quants == expected, (
            f"{backend} gpu_quants disagree; symmetric difference: "
            f"{sorted(gpu_quants ^ expected)}"
        )
