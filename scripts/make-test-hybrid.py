#!/usr/bin/env python3
"""Generate tiny Mamba-2 hybrid GGUF fixtures for the SSM admission gate.

The runner recognizes the Mamba-2 hybrid families (arch `granitehybrid`,
`nemotron_h_moe`) but does not yet implement their forward. Admission must say
so SPECIFICALLY — "recognized, forward not yet implemented" — rather than fall
back to the generic unknown-architecture refusal, and it must FAIL CLOSED with
a named tensor when a required SSM tensor or metadata field is absent (the
hostile-GGUF discipline). These fixtures exercise both branches; no real model
download is needed. The tensor names and `*.ssm_*` keys mirror a real
granite-4.0-h / Nemotron-3.5 GGUF (confirmed off files on the box):

  ssm_in.weight   [E, 2*inner? no: inner + conv_dim + n_heads]  the in-projection
  ssm_conv1d.{weight,bias}  causal depthwise conv over conv_dim = inner+2*g*state
  ssm_a, ssm_d    [1, n_heads]   per-head A (log) and skip D
  ssm_dt.bias     [n_heads]      per-head timestep bias (softplus input)
  ssm_norm.weight [inner]        gated RMSNorm before the out-projection
  ssm_out.weight  [inner, E]     the out-projection

Usage:  make-test-hybrid.py <out-prefix> [--arch granitehybrid]
Writes <out>.gguf (valid) and <out>.missing-ssm_d.gguf (drops one SSM tensor).
"""
import struct
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "test-hybrid"
ARCH = "granitehybrid"
if "--arch" in sys.argv:
    ARCH = sys.argv[sys.argv.index("--arch") + 1]

# tiny geometry (kept structurally faithful to Mamba-2, not to any real size)
E = 32                    # embedding_length
HEADS, KV = 4, 2          # attention head counts (attention layers only)
FF = 64                   # dense FFN width
N_SSM_HEADS = 4           # ssm.time_step_rank
SSM_INNER = 32            # ssm.inner_size (head_dim = INNER/HEADS = 8)
SSM_STATE = 8             # ssm.state_size
SSM_GROUPS = 1            # ssm.group_count
SSM_CONV = 4              # ssm.conv_kernel
CONV_DIM = SSM_INNER + 2 * SSM_GROUPS * SSM_STATE     # x, B, C concatenated
INPROJ = SSM_INNER + CONV_DIM + N_SSM_HEADS           # z, xBC, dt
# layer types: recurrent (SSM) layers carry head_count_kv == 0, attention
# layers a real count. blk.0 recurrent, blk.1 attention.
LAYER_KV = [0, KV]
LAYERS = len(LAYER_KV)

VOCAB = ["<unk>", "<s>", "</s>"] + [f"<0x{i:02X}>" for i in range(256)]
TTYPE = [2, 3, 3] + [6] * 256
U32, F32, STR, ARR, I32, BOOL = 4, 6, 8, 9, 5, 7
T_F32 = 0


def s(x):
    b = x.encode()
    return struct.pack("<Q", len(b)) + b


def ku(k, v): return s(k) + struct.pack("<II", U32, v)
def kf(k, v): return s(k) + struct.pack("<If", F32, v)
def ks(k, v): return s(k) + struct.pack("<I", STR) + s(v)
def kb(k, v): return s(k) + struct.pack("<IB", BOOL, bool(v))
def kas(k, xs): return s(k) + struct.pack("<IIQ", ARR, STR, len(xs)) + b"".join(s(x) for x in xs)
def kaf(k, xs): return s(k) + struct.pack("<IIQ", ARR, F32, len(xs)) + struct.pack(f"<{len(xs)}f", *xs)
def kai(k, xs): return s(k) + struct.pack("<IIQ", ARR, I32, len(xs)) + struct.pack(f"<{len(xs)}i", *xs)
def kau(k, xs): return s(k) + struct.pack("<IIQ", ARR, U32, len(xs)) + struct.pack(f"<{len(xs)}I", *xs)


_seed = 0x5510
def rnd():
    global _seed
    _seed = (_seed * 1103515245 + 12345) & 0x7fffffff
    return (_seed / 0x7fffffff - .5) * .08


def flist(n): return [rnd() for _ in range(n)]
def pack(xs): return struct.pack(f"<{len(xs)}f", *xs)
def ones(n): return pack([1.0] * n)


def meta():
    p = ARCH
    return [
        ks("general.architecture", ARCH),
        ku(f"{p}.block_count", LAYERS), ku(f"{p}.context_length", 256),
        ku(f"{p}.embedding_length", E), ku(f"{p}.feed_forward_length", FF),
        ku(f"{p}.attention.head_count", HEADS),
        kau(f"{p}.attention.head_count_kv", LAYER_KV),
        kf(f"{p}.attention.layer_norm_rms_epsilon", 1e-5),
        kf(f"{p}.rope.freq_base", 10000.0),
        # Mamba-2 SSM block
        ku(f"{p}.ssm.conv_kernel", SSM_CONV),
        ku(f"{p}.ssm.inner_size", SSM_INNER),
        ku(f"{p}.ssm.state_size", SSM_STATE),
        ku(f"{p}.ssm.group_count", SSM_GROUPS),
        ku(f"{p}.ssm.time_step_rank", N_SSM_HEADS),
        # granite scaling multipliers (present so the fixture matches the family)
        kf(f"{p}.embedding_scale", 12.0), kf(f"{p}.logit_scale", 16.0),
        kf(f"{p}.residual_scale", 0.22), kf(f"{p}.attention.scale", 0.0078125),
        ks("tokenizer.ggml.model", "llama"), kas("tokenizer.ggml.tokens", VOCAB),
        kaf("tokenizer.ggml.scores", [0.0] * len(VOCAB)),
        kai("tokenizer.ggml.token_type", TTYPE), ku("tokenizer.ggml.bos_token_id", 1),
        ku("tokenizer.ggml.eos_token_id", 2), kb("tokenizer.ggml.add_bos_token", True),
    ]


def write(path, tensors, kvs):
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


def ssm_layer(i, drop=None):
    kv_dim = (E // HEADS) * KV
    t = [
        (f"blk.{i}.attn_norm.weight", [E], ones(E)),
        (f"blk.{i}.ssm_in.weight", [E, INPROJ], pack(flist(E * INPROJ))),
        (f"blk.{i}.ssm_conv1d.weight", [SSM_CONV, CONV_DIM], pack(flist(SSM_CONV * CONV_DIM))),
        (f"blk.{i}.ssm_conv1d.bias", [CONV_DIM], pack(flist(CONV_DIM))),
        (f"blk.{i}.ssm_a", [1, N_SSM_HEADS], pack(flist(N_SSM_HEADS))),
        (f"blk.{i}.ssm_d", [1, N_SSM_HEADS], pack(flist(N_SSM_HEADS))),
        (f"blk.{i}.ssm_dt.bias", [N_SSM_HEADS], pack(flist(N_SSM_HEADS))),
        (f"blk.{i}.ssm_norm.weight", [SSM_INNER], ones(SSM_INNER)),
        (f"blk.{i}.ssm_out.weight", [SSM_INNER, E], pack(flist(SSM_INNER * E))),
        (f"blk.{i}.ffn_norm.weight", [E], ones(E)),
        (f"blk.{i}.ffn_gate.weight", [E, FF], pack(flist(E * FF))),
        (f"blk.{i}.ffn_up.weight", [E, FF], pack(flist(E * FF))),
        (f"blk.{i}.ffn_down.weight", [FF, E], pack(flist(FF * E))),
    ]
    if drop:
        t = [x for x in t if not x[0].endswith("." + drop)]
    return t


def attn_layer(i):
    kv_dim = (E // HEADS) * KV
    return [
        (f"blk.{i}.attn_norm.weight", [E], ones(E)),
        (f"blk.{i}.attn_q.weight", [E, E], pack(flist(E * E))),
        (f"blk.{i}.attn_k.weight", [E, kv_dim], pack(flist(E * kv_dim))),
        (f"blk.{i}.attn_v.weight", [E, kv_dim], pack(flist(E * kv_dim))),
        (f"blk.{i}.attn_output.weight", [E, E], pack(flist(E * E))),
        (f"blk.{i}.ffn_norm.weight", [E], ones(E)),
        (f"blk.{i}.ffn_gate.weight", [E, FF], pack(flist(E * FF))),
        (f"blk.{i}.ffn_up.weight", [E, FF], pack(flist(E * FF))),
        (f"blk.{i}.ffn_down.weight", [FF, E], pack(flist(FF * E))),
    ]


def build(drop=None):
    ts = [
        ("token_embd.weight", [E, len(VOCAB)], pack(flist(E * len(VOCAB)))),
        ("output_norm.weight", [E], ones(E)),
    ]
    for i, kv in enumerate(LAYER_KV):
        ts += ssm_layer(i, drop) if kv == 0 else attn_layer(i)
    return ts


write(f"{OUT}.gguf", build(), meta())
write(f"{OUT}.missing-ssm_d.gguf", build(drop="ssm_d"), meta())
