"""Metal fixture coverage: the geometries and MoE layouts nothing else pins.

Every case here is one of two shapes:

* **identical** — the model runs on Metal and produces byte-identical output to
  the CPU path.
* **identical or refused loudly** — Metal has no path for this layout, so it
  must say so and fall back, and the CPU answer must still be right. These flip
  to the first shape for free on the day the Metal path lands.

Three assertions guard against the failure this file exists to prevent, which
is a comparison that passes because *both* sides did nothing:

1. the run must exit 0,
2. its output must be non-empty,
3. Metal must actually have engaged (except in the refusal cases, where the
   refusal itself is asserted).

That is not hypothetical. `--arch gemma4` fixtures did not load at all — the
generator omitted `attention.key_length`, the loader's gemma4 default of 512
disagreed with the tensors, and the load failed. Both arms failed the same way,
so a plain `cmp` of their empty output reported a match:

    --arch gemma4   cpu  identical  cpu_bytes=0  rc=1/1

Two more coverage notes, since neither is obvious from the file:

* `moe4` is the only MoE fixture whose experts differ from one another. The
  Metal MoE tests used `moe1`/`moe2`, and in `moe2` *both* experts equal the
  dense FFN — so a Metal router that selected the wrong expert produced the
  right answer anyway. Routing on Metal was effectively untested.
* the split expert layout and the shared always-on expert are both refused by
  Metal today. Untested, that refusal could silently become a wrong answer.
"""
import os
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROMPT = "abcdefghijklmnopqrstuvwxyz0123456789"

# dense geometries: generator flags -> label
DENSE = [
    (["--arch", "gemma3"], "gemma3"),
    (["--arch", "gemma3", "--swa", "8,2"], "gemma3-swa"),
    (["--arch", "gemma4"], "gemma4"),
    (["--arch", "gemma4", "--swa", "8,2"], "gemma4-swa"),
    (["--arch", "qwen3", "--swa", "8,2"], "qwen3-swa"),
    (["--eseries", "3,16"], "eseries"),
    (["--gemma4-hetero"], "gemma4-hetero"),
]

# MoE fixtures that must run ON Metal and agree byte for byte
MOE_ON_METAL = ["moe1", "moe2", "moe4", "gemma4-moe"]

# MoE fixtures Metal has no path for: identical, or refused with this reason
MOE_REFUSED = [
    ("moe3", "split expert layout"),
    ("shexp", "shared-expert MoE"),
    ("shexpg", "shared-expert MoE"),
]


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    return exe


@pytest.fixture(scope="module")
def has_metal(runner_bin):
    import json
    caps = json.loads(subprocess.run([runner_bin, "--caps"], cwd=ROOT,
                                     stdout=subprocess.PIPE, check=True).stdout)
    if (caps.get("gpu") or {}).get("backend") != "metal":
        pytest.skip("no Metal backend")
    return True


@pytest.fixture(scope="module")
def moe_fixtures(tmp_path_factory):
    prefix = tmp_path_factory.mktemp("moecov") / "f"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-moe.py", str(prefix)],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    return prefix


def _run(runner_bin, model, mode):
    # RUNNER_METAL_MM=0 pins the matvec path, which IS byte-identical to the
    # CPU by contract. Every Makefile smoke that does this same CPU-vs-"auto"
    # byte comparison (test-metal-moe, test-metal-swa, test-metal-eseries, ...)
    # already sets this; this file did not, and that gap is exactly what let a
    # 2026-08-12 MM staging change (float -> half in the tiled prefill GEMM,
    # legal under its own tolerance gate, tests/test_tc_tol.c) turn the
    # `eseries` case here red. The tiled GEMM is documented in kernels.metal as
    # deliberately NOT bit-identical to the scalar path -- this file's job is
    # geometry/dispatch coverage (does Metal have a working kernel for this
    # shape), not the MM kernel's numeric behavior, which has its own gate.
    env = dict(os.environ, RUNNER_METAL_MM="0")
    return subprocess.run(
        [runner_bin, "-m", str(model), "-p", PROMPT, "-n", "12", "--temp", "0",
         "--gpu", mode],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300,
        env=env)


def _both(runner_bin, model):
    cpu = _run(runner_bin, model, "off")
    gpu = _run(runner_bin, model, "auto")
    # the anti-vacuity guard: a comparison of two failed runs is not a pass
    assert cpu.returncode == 0, cpu.stderr.decode(errors="replace")
    assert gpu.returncode == 0, gpu.stderr.decode(errors="replace")
    assert cpu.stdout, "CPU produced no output; nothing to compare against"
    return cpu, gpu


@pytest.mark.parametrize("flags,label", DENSE, ids=[d[1] for d in DENSE])
def test_dense_geometry_matches_cpu_on_metal(runner_bin, has_metal, tmp_path,
                                             flags, label):
    model = tmp_path / f"{label}.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py",
                    *flags, str(model)],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    cpu, gpu = _both(runner_bin, model)
    assert b"Metal backend" in gpu.stderr, (
        f"{label} fell back to CPU, so this comparison proves nothing:\n"
        + gpu.stderr.decode(errors="replace"))
    assert cpu.stdout == gpu.stdout, label


@pytest.mark.parametrize("variant", MOE_ON_METAL)
def test_moe_layout_matches_cpu_on_metal(runner_bin, has_metal, moe_fixtures,
                                         variant):
    """moe4 is the one that actually tests routing: four distinct experts,
    top-2. The others are dense-equivalent by construction."""
    cpu, gpu = _both(runner_bin, f"{moe_fixtures}.{variant}.gguf")
    assert b"Metal backend" in gpu.stderr, gpu.stderr.decode(errors="replace")
    assert cpu.stdout == gpu.stdout, variant


@pytest.mark.parametrize("variant,reason", MOE_REFUSED,
                         ids=[v for v, _ in MOE_REFUSED])
def test_unsupported_moe_layout_is_refused_loudly(runner_bin, has_metal,
                                                  moe_fixtures, variant, reason):
    """Metal has no path for these. The requirement is that it says so and
    hands back to the CPU — never that it quietly computes something else."""
    cpu, gpu = _both(runner_bin, f"{moe_fixtures}.{variant}.gguf")
    err = gpu.stderr.decode(errors="replace")
    if b"Metal backend" in gpu.stderr:
        # a Metal path appeared since this was written: then it must be right
        assert cpu.stdout == gpu.stdout, (
            f"{variant} now runs on Metal but disagrees with the CPU")
    else:
        assert reason in err, (
            f"{variant} fell back without naming a reason:\n{err}")
        assert cpu.stdout == gpu.stdout, (
            f"{variant} fell back to CPU but the answers differ")
