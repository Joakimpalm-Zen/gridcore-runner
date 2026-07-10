#!/usr/bin/env python3
"""Embed src/kernels.metal into src/kernels_metal.h as a C string literal."""
import os

src = os.path.join(os.path.dirname(__file__), "..", "src", "kernels.metal")
dst = os.path.join(os.path.dirname(__file__), "..", "src", "kernels_metal.h")

with open(src) as f:
    lines = f.read().split("\n")

out = ["// Generated from kernels.metal by scripts/embed-metal.py — do not edit.",
       "static const char *k_metal_src ="]
for ln in lines:
    esc = ln.replace("\\", "\\\\").replace('"', '\\"')
    out.append(f'    "{esc}\\n"')
out.append(";")

with open(dst, "w") as f:
    f.write("\n".join(out) + "\n")
print(f"wrote {dst}")
