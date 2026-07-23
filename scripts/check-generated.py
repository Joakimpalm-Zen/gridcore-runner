#!/usr/bin/env python3
"""Drift gate for the committed generated GPU headers (RNR-020).

Normal builds and CI consume the committed byte-embedded headers
(src/kernels_ptx.h, src/kernels_metal.h) rather than regenerating them, so a
source edit to kernels.metal or kernels.ptx could be reviewed and merged while
the shipped binary still embeds the OLD bytes. This regenerates each header from
its committed source into a temporary file and compares — a mismatch means the
header is stale and must be re-embedded.

The byte-embedding step is pure Python and reproducible on any host, so this
runs in CI without a CUDA or Metal toolchain. (It does not verify the upstream
kernels.cu -> kernels.ptx compile, which is nvcc/toolchain-dependent; that stays
a development-machine step.)
"""
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (embed script, committed header) pairs
TARGETS = [
    ("embed-metal.py", os.path.join("src", "kernels_metal.h")),
    ("embed-ptx.py", os.path.join("src", "kernels_ptx.h")),
]


def check(script: str, header: str) -> bool:
    committed = os.path.join(ROOT, header)
    if not os.path.exists(committed):
        print(f"MISSING: {header} does not exist", file=sys.stderr)
        return False
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
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
