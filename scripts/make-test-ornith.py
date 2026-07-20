#!/usr/bin/env python3
"""Generate a tiny Qwen3.5/Ornith-shaped GGUF for the CPU tracer test."""
import struct
import sys
import os

OUT = sys.argv[1] if len(sys.argv) > 1 else "test-ornith.gguf"
E, HEADS, KV, FF = 32, 4, 2, 64
LAYERS = int(os.environ.get("ORNITH_TEST_LAYERS", "4"))
STATE, GROUPS, VHEADS, CONV = 8, 2, 4, 4
VOCAB = ["<unk>", "<s>", "</s>"] + [f"<0x{i:02X}>" for i in range(256)]
TTYPE = [2, 3, 3] + [6] * 256
U32, F32, STR, ARR, I32, BOOL = 4, 6, 8, 9, 5, 7

def s(x):
    b = x.encode()
    return struct.pack("<Q", len(b)) + b
def ku(k, v): return s(k) + struct.pack("<II", U32, v)
def kf(k, v): return s(k) + struct.pack("<If", F32, v)
def ks(k, v): return s(k) + struct.pack("<I", STR) + s(v)
def kb(k, v): return s(k) + struct.pack("<IB", BOOL, bool(v))
def kas(k, xs):
    return s(k) + struct.pack("<IIQ", ARR, STR, len(xs)) + b"".join(s(x) for x in xs)
def kaf(k, xs):
    return s(k) + struct.pack("<IIQ", ARR, F32, len(xs)) + struct.pack(f"<{len(xs)}f", *xs)
def kai(k, xs):
    return s(k) + struct.pack("<IIQ", ARR, I32, len(xs)) + struct.pack(f"<{len(xs)}i", *xs)
def kau(k, xs):
    return s(k) + struct.pack("<IIQ", ARR, U32, len(xs)) + struct.pack(f"<{len(xs)}I", *xs)

seed = 0x35
def rnd():
    global seed
    seed = (seed * 1103515245 + 12345) & 0x7fffffff
    return (seed / 0x7fffffff - .5) * .08
def data(n): return struct.pack(f"<{n}f", *(rnd() for _ in range(n)))
def ones(n): return struct.pack(f"<{n}f", *([1.] * n))
def vals(xs): return struct.pack(f"<{len(xs)}f", *xs)
def add(ts, name, dims, payload=None):
    n = 1
    for d in dims: n *= d
    ts.append((name, dims, payload if payload is not None else data(n)))

t = []
add(t, "token_embd.weight", [E, len(VOCAB)])
add(t, "output_norm.weight", [E], ones(E))
for i in range(LAYERS):
    add(t, f"blk.{i}.attn_norm.weight", [E], ones(E))
    add(t, f"blk.{i}.post_attention_norm.weight", [E], ones(E))
    if (i + 1) % 4:
        keydim, valuedim = STATE * GROUPS, STATE * VHEADS
        add(t, f"blk.{i}.attn_qkv.weight", [E, keydim * 2 + valuedim])
        add(t, f"blk.{i}.attn_gate.weight", [E, valuedim])
        add(t, f"blk.{i}.ssm_conv1d.weight", [CONV, keydim * 2 + valuedim])
        add(t, f"blk.{i}.ssm_dt.bias", [VHEADS], vals([0.] * VHEADS))
        add(t, f"blk.{i}.ssm_a", [VHEADS], vals([-1.] * VHEADS))
        add(t, f"blk.{i}.ssm_beta.weight", [E, VHEADS])
        add(t, f"blk.{i}.ssm_alpha.weight", [E, VHEADS])
        add(t, f"blk.{i}.ssm_norm.weight", [STATE], ones(STATE))
        add(t, f"blk.{i}.ssm_out.weight", [valuedim, E])
    else:
        hd, kvdim = E // HEADS, (E // HEADS) * KV
        add(t, f"blk.{i}.attn_q.weight", [E, E * 2])
        add(t, f"blk.{i}.attn_k.weight", [E, kvdim])
        add(t, f"blk.{i}.attn_v.weight", [E, kvdim])
        add(t, f"blk.{i}.attn_output.weight", [E, E])
        add(t, f"blk.{i}.attn_q_norm.weight", [hd], ones(hd))
        add(t, f"blk.{i}.attn_k_norm.weight", [hd], ones(hd))
    add(t, f"blk.{i}.ffn_gate.weight", [E, FF])
    add(t, f"blk.{i}.ffn_up.weight", [E, FF])
    add(t, f"blk.{i}.ffn_down.weight", [FF, E])

kvs = [
    ks("general.architecture", "qwen35"), ku("qwen35.block_count", LAYERS),
    ku("qwen35.context_length", 256), ku("qwen35.embedding_length", E),
    ku("qwen35.feed_forward_length", FF), ku("qwen35.attention.head_count", HEADS),
    ku("qwen35.attention.head_count_kv", KV), ku("qwen35.attention.key_length", E // HEADS),
    ku("qwen35.rope.dimension_count", E // HEADS), kf("qwen35.attention.layer_norm_rms_epsilon", 1e-5),
    kf("qwen35.rope.freq_base", 10000.), ku("qwen35.full_attention_interval", 4),
    kau("qwen35.rope.dimension_sections", [2, 2, 0, 0]),
    ku("qwen35.ssm.conv_kernel", CONV), ku("qwen35.ssm.inner_size", E),
    ku("qwen35.ssm.state_size", STATE), ku("qwen35.ssm.time_step_rank", VHEADS),
    ku("qwen35.ssm.group_count", GROUPS), ks("tokenizer.ggml.model", "llama"),
    kas("tokenizer.ggml.tokens", VOCAB), kaf("tokenizer.ggml.scores", [0.] * len(VOCAB)),
    kai("tokenizer.ggml.token_type", TTYPE), ku("tokenizer.ggml.bos_token_id", 1),
    ku("tokenizer.ggml.eos_token_id", 2), kb("tokenizer.ggml.add_bos_token", True),
]
meta = b"".join(kvs)
info, off = b"", 0
for name, dims, payload in t:
    info += s(name) + struct.pack("<I", len(dims))
    info += b"".join(struct.pack("<Q", d) for d in dims)
    info += struct.pack("<IQ", 0, off)
    off = (off + len(payload) + 31) & ~31
head = struct.pack("<IIQQ", 0x46554747, 3, len(t), len(kvs)) + meta + info
with open(OUT, "wb") as f:
    f.write(head + b"\0" * ((-len(head)) % 32))
    for _, _, payload in t:
        f.write(payload)
        f.write(b"\0" * ((-len(payload)) % 32))
print(f"wrote {OUT}")
