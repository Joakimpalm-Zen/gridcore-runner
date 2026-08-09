# Rewrite every F16 tensor in a GGUF as BF16, in place, to make a BF16 test
# fixture. No BF16 GGUF is published for this model and the runner's own
# quantizer only emits q8_0/q4_0/f16, so the fixture has to be built.
#
# Both types are 2 bytes per element, so the data region keeps its exact layout
# and every tensor offset stays valid -- only the type tag and the payload
# bytes change. Conversion is f16 -> f32 -> bf16 with round-to-nearest-even,
# which is what any honest converter does; the point is a VALID bf16 model, not
# a bit-preserving one (f16 and bf16 cannot represent the same set anyway).
import struct, sys, numpy as np

src, dst = sys.argv[1], sys.argv[2]
buf = bytearray(open(src, 'rb').read())
o = 0
def u32():
    global o; v, = struct.unpack_from('<I', buf, o); o += 4; return v
def u64():
    global o; v, = struct.unpack_from('<Q', buf, o); o += 8; return v
def rd_str():
    global o
    n = u64(); s = bytes(buf[o:o+n]); o += n; return s

assert bytes(buf[0:4]) == b'GGUF'
o = 4
ver = u32(); n_tensors = u64(); n_kv = u64()

def skip_val(t):
    global o
    sz = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
    if t == 8:
        rd_str(); return
    if t == 9:
        et = u32(); n = u64()
        for _ in range(n): skip_val(et)
        return
    o += sz[t]

for _ in range(n_kv):
    rd_str(); t = u32(); skip_val(t)

tensors = []
for _ in range(n_tensors):
    name = rd_str(); nd = u32()
    dims = [u64() for _ in range(nd)]
    ty_off = o; ty = u32(); off = u64()
    n_elem = 1
    for d in dims: n_elem *= d
    tensors.append((name, ty_off, ty, off, n_elem))

# align the data region the way GGUF does (default alignment 32)
ALIGN = 32
data_start = (o + ALIGN - 1) // ALIGN * ALIGN

conv = 0
for name, ty_off, ty, off, n_elem in tensors:
    if ty != 1:          # 1 == F16
        continue
    base = data_start + off
    raw = np.frombuffer(bytes(buf[base:base + n_elem * 2]), dtype=np.float16)
    f32 = raw.astype(np.float32).view(np.uint32)
    # round-to-nearest-even on the discarded low 16 bits
    rounded = ((f32 + 0x7FFF + ((f32 >> 16) & 1)) >> 16).astype(np.uint16)
    buf[base:base + n_elem * 2] = rounded.tobytes()
    struct.pack_into('<I', buf, ty_off, 30)   # 30 == BF16
    conv += 1

open(dst, 'wb').write(bytes(buf))
print(f"converted {conv} F16 tensors to BF16 -> {dst}")
