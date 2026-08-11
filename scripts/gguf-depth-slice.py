#!/usr/bin/env python3
"""Depth-slice a GGUF: drop whole transformer layers, keep everything else.

    gguf-depth-slice.py in.gguf out.gguf 30,31,32,...

The mergekit-passthrough operation at the GGUF level, for engines and
models where the HF-side toolchain has no architecture support yet:
surviving layers keep their exact on-disk bytes (no requantization),
`blk.N` tensors are renumbered densely in original order, and every
array-typed `<arch>.*` metadata value whose length equals the old
block_count (sliding-window patterns, per-layer widths, per-layer head
counts) is filtered to the survivors so per-layer semantics travel with
their layers. Refuses models that declare NextN/MTP predictor blocks —
their trailing-block numbering would be silently corrupted by renumbering.

Layer choice is the caller's problem: score first (RUNNER_LAYER_SIM=1
prints per-layer residual-stream input/output cosine; block influence is
1 - cos), then cut.
"""
import struct
import sys

T_U8, T_I8, T_U16, T_I16, T_U32, T_I32, T_F32, T_BOOL = range(8)
T_STR, T_ARR, T_U64, T_I64, T_F64 = 8, 9, 10, 11, 12
SCALAR = {T_U8: "<B", T_I8: "<b", T_U16: "<H", T_I16: "<h", T_U32: "<I",
          T_I32: "<i", T_F32: "<f", T_BOOL: "<B", T_U64: "<Q", T_I64: "<q",
          T_F64: "<d"}
SIZES = {T_U8: 1, T_I8: 1, T_U16: 2, T_I16: 2, T_U32: 4, T_I32: 4,
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

    def u32(self): return struct.unpack("<I", self.take(4))[0]
    def u64(self): return struct.unpack("<Q", self.take(8))[0]
    def st(self): return self.take(self.u64())


def s(x):
    b = x if isinstance(x, bytes) else x.encode()
    return struct.pack("<Q", len(b)) + b


def read_value(r, t):
    """Return (raw_bytes, parsed_or_None). Arrays of scalars parse; the rest
    only need their bytes carried through."""
    start = r.p
    if t == T_STR:
        r.st()
        return r.b[start:r.p], None
    if t == T_ARR:
        at, n = r.u32(), r.u64()
        if at == T_STR:
            vals = [r.st() for _ in range(n)]
            return r.b[start:r.p], ("str", vals)
        if at == T_ARR:
            sys.exit("nested arrays are not supported")
        vals = [struct.unpack(SCALAR[at], r.take(SIZES[at]))[0] for _ in range(n)]
        return r.b[start:r.p], (at, vals)
    r.take(SIZES[t])
    return r.b[start:r.p], struct.unpack(SCALAR[t], r.b[start:r.p])[0]


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    src, dst, drop = sys.argv[1], sys.argv[2], sys.argv[3]
    drop = sorted({int(x) for x in drop.split(",")})
    raw = open(src, "rb").read()
    r = R(raw)
    if r.take(4) != b"GGUF":
        sys.exit("not a GGUF file")
    version, n_tensors, n_kv = r.u32(), r.u64(), r.u64()

    kvs = []          # (key:str, type, raw, parsed)
    align = 32
    arch = None
    for _ in range(n_kv):
        key = r.st().decode()
        t = r.u32()
        rawv, parsed = read_value(r, t)
        if key == "general.alignment":
            align = parsed
        if key == "general.architecture":
            arch = parsed if isinstance(parsed, str) else None
            if arch is None:
                # T_STR raw: 8-byte len + bytes
                arch = rawv[8:].decode()
        kvs.append([key, t, rawv, parsed])
    if arch is None:
        sys.exit("no general.architecture")

    old_n = None
    for key, t, rawv, parsed in kvs:
        if key == f"{arch}.block_count":
            old_n = parsed
    if not old_n:
        sys.exit(f"no {arch}.block_count")
    for key, t, rawv, parsed in kvs:
        if key == f"{arch}.nextn_predict_layers" and parsed:
            sys.exit("model declares NextN/MTP predictor blocks — renumbering "
                     "would corrupt them; refusing")
    bad = [d for d in drop if d < 0 or d >= old_n]
    if bad:
        sys.exit(f"drop indices out of range for {old_n} layers: {bad}")
    keep = [i for i in range(old_n) if i not in set(drop)]
    if len(keep) < 2:
        sys.exit("refusing to keep fewer than 2 layers")
    remap = {old: new for new, old in enumerate(keep)}

    # rewrite metadata
    out_kvs = []
    for key, t, rawv, parsed in kvs:
        if key == f"{arch}.block_count":
            out_kvs.append(s(key) + struct.pack("<I", t) +
                           struct.pack(SCALAR[t], len(keep)))
            continue
        if (key.startswith(arch + ".") and t == T_ARR and parsed and
                parsed[0] != "str" and len(parsed[1]) == old_n):
            at, vals = parsed
            vals = [vals[i] for i in keep]
            body = b"".join(struct.pack(SCALAR[at], v) for v in vals)
            out_kvs.append(s(key) + struct.pack("<II", t, at) +
                           struct.pack("<Q", len(vals)) + body)
            print(f"  filtered per-layer array {key}: {old_n} -> {len(vals)}")
            continue
        out_kvs.append(s(key) + struct.pack("<I", t) + rawv)

    # tensor table
    data_start = (r.p + align - 1) & ~(align - 1)
    tensors = []  # (name, ndim_raw, type, abs_off, nbytes)
    table = []
    for _ in range(n_tensors):
        name = r.st().decode()
        nd = r.u32()
        ne = [r.u64() for _ in range(nd)]
        tt = r.u32()
        off = r.u64()
        table.append((name, nd, ne, tt, off))
    # sizes come from the gap to the next offset; compute via sorted offsets
    by_off = sorted(table, key=lambda x: x[4])
    end = len(raw) - data_start
    sizes = {}
    for i, (name, nd, ne, tt, off) in enumerate(by_off):
        nxt = by_off[i + 1][4] if i + 1 < len(by_off) else end
        sizes[name] = nxt - off

    kept = []
    for name, nd, ne, tt, off in table:
        if name.startswith("blk."):
            layer = int(name.split(".")[1])
            if layer in set(drop):
                continue
            new_name = f"blk.{remap[layer]}." + name.split(".", 2)[2]
        else:
            new_name = name
        kept.append((new_name, nd, ne, tt, off, sizes[name]))

    head = struct.pack("<4sIQQ", b"GGUF", version, len(kept), len(out_kvs))
    head += b"".join(out_kvs)
    # new tensor table with recomputed offsets
    off_new = 0
    entries = []
    blobs = []
    for new_name, nd, ne, tt, off, size in kept:
        entries.append(s(new_name) + struct.pack("<I", nd) +
                       b"".join(struct.pack("<Q", d) for d in ne) +
                       struct.pack("<IQ", tt, off_new))
        blobs.append((data_start + off, size))
        off_new = (off_new + size + align - 1) & ~(align - 1)
    head += b"".join(entries)

    with open(dst, "wb") as f:
        f.write(head + b"\0" * ((-len(head)) % align))
        for off, size in blobs:
            f.write(raw[off:off + size])
            f.write(b"\0" * ((-size) % align))
    print(f"wrote {dst}: {len(keep)}/{old_n} layers "
          f"({len(kept)} tensors), dropped {drop}")


if __name__ == "__main__":
    main()
