#!/usr/bin/env python3
"""Split a GGUF into llama.cpp-compatible shards, for testing multi-part loads.

Layout follows llama.cpp's gguf-split verbatim (checked against
tools/gguf-split/gguf-split.cpp and src/llama.cpp at master, 2026-08-07):

  filenames   <prefix>-%05d-of-%05d.gguf, 1-based
  every shard carries  split.no (u16, 0-based)
                       split.count (u16)
                       split.tensors.count (i32, total across all shards)
  each shard is a complete GGUF: its own header, the full metadata block, its
  own tensor table with offsets relative to its own data section.

The metadata block is copied byte for byte rather than re-serialized — this
only needs to *skip* KV values to find where they end, so it never has to
understand what any of them mean.

    gguf-split.py in.gguf out-prefix [n_shards]
"""
import struct
import sys

T_U8, T_I8, T_U16, T_I16, T_U32, T_I32, T_F32, T_BOOL = range(8)
T_STR, T_ARR, T_U64, T_I64, T_F64 = 8, 9, 10, 11, 12
SCALAR = {T_U8: 1, T_I8: 1, T_U16: 2, T_I16: 2, T_U32: 4, T_I32: 4,
          T_F32: 4, T_BOOL: 1, T_U64: 8, T_I64: 8, T_F64: 8}


class R:
    def __init__(self, b):
        self.b, self.p = b, 0

    def take(self, n):
        v = self.b[self.p:self.p + n]
        if len(v) != n:
            sys.exit("truncated GGUF")
        self.p += n
        return v

    def u32(self):
        return struct.unpack("<I", self.take(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.take(8))[0]

    def st(self):
        return self.take(self.u64())


def skip_value(r, t):
    if t == T_STR:
        r.st()
    elif t == T_ARR:
        at, n = r.u32(), r.u64()
        if at == T_STR:
            for _ in range(n):
                r.st()
        elif at == T_ARR:
            sys.exit("nested arrays are not supported")
        else:
            r.take(SCALAR[at] * n)
    else:
        r.take(SCALAR[t])


def s(x):
    b = x.encode() if isinstance(x, str) else x
    return struct.pack("<Q", len(b)) + b


def main():
    src, prefix = sys.argv[1], sys.argv[2]
    n_shards = int(sys.argv[3]) if len(sys.argv) > 3 else 2

    raw = open(src, "rb").read()
    r = R(raw)
    if r.take(4) != b"GGUF":
        sys.exit("not a GGUF file")
    version = r.u32()
    n_tensors, n_kv = r.u64(), r.u64()

    meta_start = r.p
    align = 32
    for _ in range(n_kv):
        key = r.st().decode()
        t = r.u32()
        vstart = r.p
        skip_value(r, t)
        if key == "general.alignment":
            align = struct.unpack("<I", raw[vstart:vstart + 4])[0]
    meta = raw[meta_start:r.p]

    tensors = []
    for _ in range(n_tensors):
        name = r.st()
        nd = r.u32()
        ne = [r.u64() for _ in range(nd)]
        ttype, off = r.u32(), r.u64()
        tensors.append([name, ne, ttype, off])

    data_start = (r.p + align - 1) & ~(align - 1)
    # each tensor's byte span: up to the next tensor's offset, or EOF
    for i, t in enumerate(tensors):
        end = tensors[i + 1][3] if i + 1 < len(tensors) else len(raw) - data_start
        t.append(raw[data_start + t[3]:data_start + end])

    if n_shards > n_tensors:
        sys.exit(f"cannot make {n_shards} shards from {n_tensors} tensors")
    per = (n_tensors + n_shards - 1) // n_shards
    groups = [tensors[i:i + per] for i in range(0, n_tensors, per)]
    n_shards = len(groups)

    for i, g in enumerate(groups):
        extra = (s("split.no") + struct.pack("<IH", T_U16, i) +
                 s("split.count") + struct.pack("<IH", T_U16, n_shards) +
                 s("split.tensors.count") + struct.pack("<Ii", T_I32, n_tensors))
        head = (b"GGUF" + struct.pack("<IQQ", version, len(g), n_kv + 3) +
                meta + extra)
        info, off = b"", 0
        for name, ne, ttype, _, blob in g:
            info += s(name) + struct.pack("<I", len(ne))
            for d in ne:
                info += struct.pack("<Q", d)
            info += struct.pack("<IQ", ttype, off)
            off = (off + len(blob) + align - 1) & ~(align - 1)
        head += info
        path = f"{prefix}-{i + 1:05d}-of-{n_shards:05d}.gguf"
        with open(path, "wb") as f:
            f.write(head + b"\0" * ((-len(head)) % align))
            for *_, blob in g:
                f.write(blob)
                f.write(b"\0" * ((-len(blob)) % align))
        print(f"{path}: {len(g)} tensors")


main()
