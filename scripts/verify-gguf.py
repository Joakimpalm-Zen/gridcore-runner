#!/usr/bin/env python3
"""Verify a GGUF download is complete, and dump its metadata.

    scripts/verify-gguf.py <model.gguf> [--keys] [--tensors]

A truncated download is the expensive failure mode: the header and the whole
tensor *index* live at the front of the file, so a half-downloaded GGUF opens
fine, reports a plausible architecture and vocabulary, and only fails much
later -- or worse, loads garbage weights. This checks the one thing that
actually proves completeness: the end of the last tensor's data must fall
within the file.

No dependencies; parses the container directly.
"""

import argparse
import struct
import sys

# ggml type -> (block size in elements, bytes per block)
GGML_TYPES = {
    0:  (1, 4),     # F32
    1:  (1, 2),     # F16
    2:  (32, 18),   # Q4_0
    3:  (32, 20),   # Q4_1
    6:  (32, 22),   # Q5_0
    7:  (32, 24),   # Q5_1
    8:  (32, 34),   # Q8_0
    9:  (32, 36),   # Q8_1
    10: (256, 84),  # Q2_K
    11: (256, 110), # Q3_K
    12: (256, 144), # Q4_K
    13: (256, 176), # Q5_K
    14: (256, 210), # Q6_K
    15: (256, 292), # Q8_K
    16: (256, 66),  # IQ2_XXS
    17: (256, 74),  # IQ2_XS
    18: (256, 98),  # IQ3_XXS
    19: (256, 50),  # IQ1_S
    20: (32, 16),   # IQ4_NL
    21: (256, 82),  # IQ3_S
    22: (256, 82),  # IQ2_S
    23: (256, 136), # IQ4_XS
    24: (1, 1),     # I8
    25: (1, 2),     # I16
    26: (1, 4),     # I32
    27: (1, 8),     # I64
    28: (1, 8),     # F64
    29: (256, 56),  # IQ1_M
    30: (1, 2),     # BF16
}

TYPE_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
    14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
    19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS",
    24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M",
    30: "BF16",
}


class R:
    def __init__(self, f):
        self.f = f

    def raw(self, n):
        b = self.f.read(n)
        if len(b) != n:
            raise EOFError("file ended inside the metadata/tensor index -- "
                           "the download is truncated")
        return b

    def u32(self): return struct.unpack("<I", self.raw(4))[0]
    def u64(self): return struct.unpack("<Q", self.raw(8))[0]
    def i32(self): return struct.unpack("<i", self.raw(4))[0]

    def string(self):
        return self.raw(self.u64()).decode("utf-8", "replace")

    def value(self, t):
        if t == 0:  return struct.unpack("<B", self.raw(1))[0]
        if t == 1:  return struct.unpack("<b", self.raw(1))[0]
        if t == 2:  return struct.unpack("<H", self.raw(2))[0]
        if t == 3:  return struct.unpack("<h", self.raw(2))[0]
        if t == 4:  return self.u32()
        if t == 5:  return self.i32()
        if t == 6:  return struct.unpack("<f", self.raw(4))[0]
        if t == 7:  return struct.unpack("<B", self.raw(1))[0] != 0
        if t == 8:  return self.string()
        if t == 9:
            et = self.u32()
            n = self.u64()
            return [self.value(et) for _ in range(n)]
        if t == 10: return self.u64()
        if t == 11: return struct.unpack("<q", self.raw(8))[0]
        if t == 12: return struct.unpack("<d", self.raw(8))[0]
        raise ValueError(f"unknown gguf value type {t}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--keys", action="store_true", help="print metadata keys")
    ap.add_argument("--tensors", action="store_true", help="print tensor list")
    args = ap.parse_args()

    import os
    size = os.path.getsize(args.path)

    with open(args.path, "rb") as f:
        r = R(f)
        magic = r.raw(4)
        if magic != b"GGUF":
            print(f"not a GGUF file (magic {magic!r})", file=sys.stderr)
            return 1
        version = r.u32()
        n_tensors = r.u64()
        n_kv = r.u64()

        kv = {}
        for _ in range(n_kv):
            k = r.string()
            kv[k] = r.value(r.u32())

        tensors = []
        for _ in range(n_tensors):
            name = r.string()
            nd = r.u32()
            dims = [r.u64() for _ in range(nd)]
            ttype = r.u32()
            offset = r.u64()
            tensors.append((name, dims, ttype, offset))

        index_end = f.tell()

    alignment = kv.get("general.alignment", 32)
    data_start = (index_end + alignment - 1) // alignment * alignment

    print(f"file        {args.path}")
    print(f"size        {size:,} bytes")
    print(f"gguf        v{version}, {n_tensors} tensors, {n_kv} kv pairs")
    print(f"arch        {kv.get('general.architecture')}")
    print(f"name        {kv.get('general.name')}")
    print(f"tok model   {kv.get('tokenizer.ggml.model')}")
    print(f"tok pre     {kv.get('tokenizer.ggml.pre')}")
    print(f"data start  {data_start:,}")

    if args.keys:
        print("\n--- metadata")
        for k, v in kv.items():
            s = repr(v)
            if len(s) > 160:
                s = s[:160] + f"... ({len(v)} items)" if isinstance(v, list) else s[:160] + "..."
            print(f"  {k} = {s}")

    # The completeness check: end of the last byte of tensor data.
    end = 0
    unknown = []
    for name, dims, ttype, offset in tensors:
        n = 1
        for d in dims:
            n *= d
        if ttype not in GGML_TYPES:
            unknown.append((name, ttype))
            continue
        blk, bpb = GGML_TYPES[ttype]
        if n % blk:
            print(f"warning: {name} has {n} elements, not a multiple of "
                  f"block size {blk}", file=sys.stderr)
        nbytes = (n + blk - 1) // blk * bpb
        end = max(end, data_start + offset + nbytes)

    if args.tensors:
        print("\n--- tensors")
        for name, dims, ttype, offset in tensors:
            print(f"  {name:<44} {TYPE_NAMES.get(ttype, ttype):<8} {dims}")

    if unknown:
        print(f"\ncannot verify: {len(unknown)} tensors of unknown ggml type "
              f"{sorted({t for _, t in unknown})}", file=sys.stderr)
        return 2

    last = max(tensors, key=lambda t: t[3])
    print(f"\nlast tensor {last[0]} (type {TYPE_NAMES.get(last[2], last[2])}, "
          f"dims {last[1]})")
    print(f"data ends   {end:,}")
    print(f"slack       {size - end:,} bytes")

    if end > size:
        print(f"\nTRUNCATED: needs {end:,} bytes, file has {size:,} "
              f"({end - size:,} missing)", file=sys.stderr)
        return 1
    print("\nCOMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
