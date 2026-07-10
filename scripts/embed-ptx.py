#!/usr/bin/env python3
"""Embed src/kernels.ptx into src/kernels_ptx.h as a C string literal.

The PTX is produced by `make ptx` (requires nvcc + MSVC/gcc host compiler,
development machines only); the generated header is committed so normal builds
and CI need no CUDA toolkit.
"""
import os

src = os.path.join(os.path.dirname(__file__), "..", "src", "kernels.ptx")
dst = os.path.join(os.path.dirname(__file__), "..", "src", "kernels_ptx.h")

with open(src) as f:
    lines = f.read().split("\n")

# Pin the PTX ISA version to 7.8 (CUDA 11.8 era): these kernels use nothing
# newer, and drivers reject PTX whose .version is newer than their JIT —
# "compiled with an unsupported toolchain" — even when the code is compatible.
for i, ln in enumerate(lines):
    if ln.startswith(".version"):
        lines[i] = ".version 7.8"
        break

out = ["// Generated from kernels.cu via scripts/embed-ptx.py — do not edit.",
       "static const char *k_ptx_src ="]
for ln in lines:
    esc = ln.replace("\\", "\\\\").replace('"', '\\"')
    out.append(f'    "{esc}\\n"')
out.append(";")

with open(dst, "w") as f:
    f.write("\n".join(out) + "\n")
print(f"wrote {dst}")
