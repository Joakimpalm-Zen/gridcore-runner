#!/usr/bin/env python3
"""Drift gate for the committed generated GPU headers (RNR-020).

Normal builds and CI consume the committed byte-embedded headers
(src/kernels_ptx.h, src/kernels_metal.h) rather than regenerating them, so a
source edit to kernels.metal or kernels.ptx could be reviewed and merged while
the shipped binary still embeds the OLD bytes. This regenerates each header from
its committed source into a temporary file and compares when the source artifact
is present — a mismatch means the header is stale and must be re-embedded.

The byte-embedding step is pure Python and reproducible on any host, so this
runs in CI without a CUDA or Metal toolchain. (It does not verify the upstream
kernels.cu -> kernels.ptx compile, which is nvcc/toolchain-dependent; that stays
a development-machine step. When `src/kernels.ptx` is absent, the committed PTX
header's `.target` is still checked against the documented CUDA minimum.)
"""
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (embed script, committed header) pairs
TARGETS = [
    ("embed-metal.py", os.path.join("src", "kernels_metal.h")),
    ("embed-ptx.py", os.path.join("src", "kernels_ptx.h")),
]

SOURCES = {
    "embed-metal.py": os.path.join("src", "kernels.metal"),
    "embed-ptx.py": os.path.join("src", "kernels.ptx"),
}


def check(script: str, header: str) -> bool:
    committed = os.path.join(ROOT, header)
    if not os.path.exists(committed):
        print(f"MISSING: {header} does not exist", file=sys.stderr)
        return False
    src = SOURCES.get(script)
    if src and not os.path.exists(os.path.join(ROOT, src)):
        print(f"skip: {src} is not present; {header} target is checked separately")
        return True
    with tempfile.NamedTemporaryFile(suffix=".h", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        env = dict(os.environ, EMBED_OUT=tmp_path)
        subprocess.run([sys.executable, os.path.join(ROOT, "scripts", script)],
                       check=True, cwd=ROOT, env=env, stdout=subprocess.DEVNULL)
        with open(tmp_path, "rb") as f:
            regenerated = f.read()
        with open(committed, "rb") as f:
            on_disk = f.read()
    finally:
        os.unlink(tmp_path)
    if regenerated != on_disk:
        print(f"DRIFT: {header} is stale — re-run scripts/{script} and commit",
              file=sys.stderr)
        return False
    print(f"ok: {header} matches its source")
    return True


def main() -> int:
    ok = True
    for script, header in TARGETS:
        ok &= check(script, header)
    ok &= check_cuda_target()
    return 0 if ok else 1


def read_rel(path: str) -> str:
    with open(os.path.join(ROOT, path), encoding="utf-8") as f:
        return f.read()


def macro(text: str, name: str) -> str:
    m = re.search(rf'#define\s+{re.escape(name)}\s+"([^"]+)"', text)
    if not m:
        raise KeyError(name)
    return m.group(1)


def check_cuda_target() -> bool:
    try:
        runner_h = read_rel(os.path.join("src", "runner.h"))
        nvcc_arch = macro(runner_h, "RUNNER_CUDA_NVCC_ARCH")
        ptx_target = macro(runner_h, "RUNNER_CUDA_PTX_TARGET")
        min_gpu = macro(runner_h, "RUNNER_CUDA_MIN_GPU")
    except Exception as exc:
        print(f"DRIFT: cannot read CUDA support constants: {exc}", file=sys.stderr)
        return False

    ok = True
    makefile = read_rel("Makefile")
    ptx_h = read_rel(os.path.join("src", "kernels_ptx.h"))
    if f"-arch={nvcc_arch}" not in makefile:
        print(f"DRIFT: Makefile ptx target is not {nvcc_arch}", file=sys.stderr)
        ok = False
    if f'.target {ptx_target}' not in ptx_h:
        print(f"DRIFT: kernels_ptx.h target is not {ptx_target}", file=sys.stderr)
        ok = False
    for doc in ("README.md", os.path.join("docs", "performance.md")):
        text = read_rel(doc)
        if min_gpu not in text:
            print(f"DRIFT: {doc} does not document {min_gpu}", file=sys.stderr)
            ok = False
    if ok:
        print(f"ok: CUDA support docs and PTX target agree on {ptx_target}")
    return ok


if __name__ == "__main__":
    raise SystemExit(main())
