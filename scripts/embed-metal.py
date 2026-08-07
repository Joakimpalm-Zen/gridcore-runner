#!/usr/bin/env python3
"""Embed src/kernels.metal into src/kernels_metal.h as a C string literal.

Also emits a SHA-256 of the source so a test can prove the BINARY carries the
current kernels. check-generated.py already compares this header to its
source, but nothing compared the compiled binary to the header -- and a
measurement against a stale binary looks exactly like an honest result.
"""
import hashlib
import os

src = os.path.join(os.path.dirname(__file__), "..", "src", "kernels.metal")
# EMBED_OUT redirects the output (used by the drift check to generate to a temp
# without touching the committed header).
dst = os.environ.get("EMBED_OUT") or os.path.join(
    os.path.dirname(__file__), "..", "src", "kernels_metal.h")

with open(src, encoding="utf-8") as f:
    text = f.read()          # text mode: CRLF is normalised, so the hash is
lines = text.split("\n")     # the same on every checkout
sha = hashlib.sha256(text.encode("utf-8")).hexdigest()

out = ["// Generated from kernels.metal by scripts/embed-metal.py — do not edit.",
       "static const char *k_metal_src ="]
for ln in lines:
    esc = ln.replace("\\", "\\\\").replace('"', '\\"')
    out.append(f'    "{esc}\\n"')
out.append(";")
out.append("// SHA-256 of kernels.metal as embedded above.")
out.append(f'static const char *k_metal_sha = "{sha}";')

with open(dst, "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(out) + "\n")
print(f"wrote {dst}")
