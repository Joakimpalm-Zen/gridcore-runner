#!/usr/bin/env python3
"""Generate LoRA-adapter fixtures against a make-test-model.py base (F32).

Writes, next to the given output prefix:
  <out>.adapter.gguf      rank-4 adapters on blk.0 attn_q + ffn_down, small
                          deterministic values (llama.cpp adapter naming:
                          blk.N.<proj>.weight.lora_a/_b, adapter.lora.alpha)
  <out>.zero.gguf         the same shapes with lora_b == 0 (a no-op adapter)
  <out>.merged.gguf       the BASE file with W += (alpha/r) * (B @ A) merged
                          into the targeted F32 tensors — the mathematical
                          reference the adapter run must approximate
  <out>.badshape.gguf     lora_a whose in-dim disagrees with the base (must
                          be refused, naming the tensor)
  <out>.halfpair.gguf     lora_a without its lora_b (must be refused)

Usage: make-test-lora.py <base.gguf> <out-prefix>
"""
import struct
import sys

BASE, OUT = sys.argv[1], sys.argv[2]
ALPHA = 8.0
RANK = 4

U32, F32T, STR = 4, 6, 8
T_F32 = 0


def s(x):
    b = x.encode()
    return struct.pack("<Q", len(b)) + b


def ks(k, v): return s(k) + struct.pack("<I", STR) + s(v)
def kf(k, v): return s(k) + struct.pack("<If", F32T, v)


_seed = 0x1234


def rnd():
    global _seed
    _seed = (_seed * 1103515245 + 12345) & 0x7FFFFFFF
    return (_seed / 0x7FFFFFFF - 0.5) * 0.2


def write_gguf(path, kvs, tensors):
    m = b"".join(kvs)
    info, off = b"", 0
    for name, dims, payload in tensors:
        info += s(name) + struct.pack("<I", len(dims))
        info += b"".join(struct.pack("<Q", d) for d in dims)
        info += struct.pack("<IQ", T_F32, off)
        off = (off + len(payload) + 31) & ~31
    head = struct.pack("<IIQQ", 0x46554747, 3, len(tensors), len(kvs)) + m + info
    with open(path, "wb") as f:
        f.write(head + b"\0" * ((-len(head)) % 32))
        for _, _, payload in tensors:
            f.write(payload)
            f.write(b"\0" * ((-len(payload)) % 32))
    print(f"wrote {path}")


# ---- read the base: header walk sufficient for F32 fixture tensors
def read_base(path):
    d = open(path, "rb").read()
    magic, ver, n_t, n_kv = struct.unpack_from("<IIQQ", d, 0)
    assert magic == 0x46554747
    off = 24

    def rstr():
        nonlocal off
        n = struct.unpack_from("<Q", d, off)[0]
        off += 8
        v = d[off:off + n].decode()
        off += n
        return v

    def skip_val(t):
        nonlocal off
        sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        if t == 8:
            rstr()
        elif t == 9:
            et, n = struct.unpack_from("<IQ", d, off)
            off += 12
            for _ in range(n):
                skip_val(et)
        else:
            off += sizes[t]

    for _ in range(n_kv):
        rstr()
        t = struct.unpack_from("<I", d, off)[0]
        off += 4
        skip_val(t)
    tens = []
    for _ in range(n_t):
        name = rstr()
        nd = struct.unpack_from("<I", d, off)[0]
        off += 4
        dims = list(struct.unpack_from("<%dQ" % nd, d, off))
        off += 8 * nd
        ttype, toff = struct.unpack_from("<IQ", d, off)
        off += 12
        tens.append([name, dims, ttype, toff])
    data0 = (off + 31) & ~31
    return d, tens, data0


raw, tens, data0 = read_base(BASE)
targets = {"blk.0.attn_q.weight": None, "blk.0.ffn_down.weight": None}
for name, dims, ttype, toff in tens:
    if name in targets:
        assert ttype == T_F32, f"{name} is not F32 in the base fixture"
        targets[name] = (dims, toff)
for k, v in targets.items():
    assert v, f"base lacks {k}"


def pack_f(vals):
    return struct.pack("<%df" % len(vals), *vals)


def adapter_tensors(zero_b):
    out = []
    for name, (dims, _) in targets.items():
        nin, nout = dims[0], dims[1]
        a = [rnd() for _ in range(RANK * nin)]
        b = [0.0 if zero_b else rnd() for _ in range(nout * RANK)]
        base = name[:-len(".weight")]
        out.append((f"{base}.weight.lora_a", [nin, RANK], pack_f(a)))
        out.append((f"{base}.weight.lora_b", [RANK, nout], pack_f(b)))
    return out


meta = [ks("general.architecture", "llama"), ks("general.type", "adapter"),
        ks("adapter.type", "lora"), kf("adapter.lora.alpha", ALPHA)]

_seed = 0x1234
adap = adapter_tensors(zero_b=False)
write_gguf(OUT + ".adapter.gguf", meta, adap)

_seed = 0x1234
write_gguf(OUT + ".zero.gguf", meta, adapter_tensors(zero_b=True))

# merged reference: base bytes with W += (alpha/r) * B @ A on the targets,
# using the SAME adapter values (same seed replay)
merged = bytearray(raw)
_seed = 0x1234
for name, (dims, toff) in targets.items():
    nin, nout = dims[0], dims[1]
    a = [rnd() for _ in range(RANK * nin)]     # a[k*nin + i]
    b = [rnd() for _ in range(nout * RANK)]    # b[j*RANK + k]
    scale = ALPHA / RANK
    base_off = data0 + toff
    for j in range(nout):
        row = base_off + j * nin * 4
        w = list(struct.unpack_from("<%df" % nin, merged, row))
        for i in range(nin):
            delta = 0.0
            for k in range(RANK):
                delta += b[j * RANK + k] * a[k * nin + i]
            w[i] += scale * delta
        struct.pack_into("<%df" % nin, merged, row, *w)
open(OUT + ".merged.gguf", "wb").write(merged)
print(f"wrote {OUT}.merged.gguf")

# hostile variants
nin, nout = targets["blk.0.attn_q.weight"][0][0], targets["blk.0.attn_q.weight"][0][1]
bad = [("blk.0.attn_q.weight.lora_a", [nin + 2, RANK], pack_f([0.1] * (RANK * (nin + 2)))),
       ("blk.0.attn_q.weight.lora_b", [RANK, nout], pack_f([0.1] * (nout * RANK)))]
write_gguf(OUT + ".badshape.gguf", meta, bad)
half = [("blk.0.attn_q.weight.lora_a", [nin, RANK], pack_f([0.1] * (RANK * nin)))]
write_gguf(OUT + ".halfpair.gguf", meta, half)
