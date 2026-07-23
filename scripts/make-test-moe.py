#!/usr/bin/env python3
"""Generate a matched dense + MoE GGUF trio for the MoE equivalence test.

The three files share every weight except the FFN, and each MoE variant is
constructed to be MATHEMATICALLY IDENTICAL to the dense model's FFN, so the
runner's already-trusted dense path is the oracle — no separate reference
engine is needed. All weights are F32, so there is no quantization error and
the greedy output must be byte-identical.

  <out>.dense.gguf   plain llama, dense SwiGLU FFN with down = D
  <out>.moe1.gguf    llama+experts, expert_count=2 expert_used=1, zero router
                     (uniform logits -> top-1 picks expert 0 by index), expert 0
                     FFN == dense, expert 1 = zeros. Weight of the one expert is
                     1.0, so output == dense exactly. Validates: metadata parse,
                     router matmul+softmax+top-1 selection, 3D expert slice at
                     offset 0, SwiGLU, and that expert 1 does not leak in.
  <out>.moe2.gguf    expert_count=2 expert_used=2, zero router -> weights
                     [0.5, 0.5], BOTH experts == dense. 0.5*y + 0.5*y == y
                     exactly, so output == dense. Validates: top-2 selection,
                     renormalization summing to 1, the weighted multi-expert sum,
                     and reading BOTH expert slices (offsets 0 and 1).
"""
import struct
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "test-moe"
E, HEADS, KV, FF, LAYERS = 32, 4, 2, 64, 2
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
def kas(k, xs): return s(k) + struct.pack("<IIQ", ARR, STR, len(xs)) + b"".join(s(x) for x in xs)
def kaf(k, xs): return s(k) + struct.pack("<IIQ", ARR, F32, len(xs)) + struct.pack(f"<{len(xs)}f", *xs)
def kai(k, xs): return s(k) + struct.pack("<IIQ", ARR, I32, len(xs)) + struct.pack(f"<{len(xs)}i", *xs)


_seed = 0x50E
def rnd():
    global _seed
    _seed = (_seed * 1103515245 + 12345) & 0x7fffffff
    return (_seed / 0x7fffffff - .5) * .08


def flist(n): return [rnd() for _ in range(n)]
def pack(xs): return struct.pack(f"<{len(xs)}f", *xs)
def ones(n): return pack([1.0] * n)
def zeros(n): return pack([0.0] * n)


def base_meta(arch, extra):
    kvs = [
        ks("general.architecture", "llama"),
        ku("llama.block_count", LAYERS), ku("llama.context_length", 256),
        ku("llama.embedding_length", E), ku("llama.feed_forward_length", FF),
        ku("llama.attention.head_count", HEADS), ku("llama.attention.head_count_kv", KV),
        kf("llama.attention.layer_norm_rms_epsilon", 1e-5),
        kf("llama.rope.freq_base", 10000.0),
    ] + extra + [
        ks("tokenizer.ggml.model", "llama"), kas("tokenizer.ggml.tokens", VOCAB),
        kaf("tokenizer.ggml.scores", [0.0] * len(VOCAB)),
        kai("tokenizer.ggml.token_type", TTYPE), ku("tokenizer.ggml.bos_token_id", 1),
        ku("tokenizer.ggml.eos_token_id", 2), kb("tokenizer.ggml.add_bos_token", True),
    ]
    return kvs


def write(path, tensors, kvs):
    meta = b"".join(kvs)
    info, off = b"", 0
    for name, dims, payload in tensors:
        info += s(name) + struct.pack("<I", len(dims))
        info += b"".join(struct.pack("<Q", d) for d in dims)
        info += struct.pack("<IQ", 0, off)  # type F32, offset
        off = (off + len(payload) + 31) & ~31
    head = struct.pack("<IIQQ", 0x46554747, 3, len(tensors), len(kvs)) + meta + info
    with open(path, "wb") as f:
        f.write(head + b"\0" * ((-len(head)) % 32))
        for _, _, payload in tensors:
            f.write(payload)
            f.write(b"\0" * ((-len(payload)) % 32))
    print(f"wrote {path}")


# --- shared weights (generated once so all three files agree) -----------------
shared = [
    ("token_embd.weight", [E, len(VOCAB)], pack(flist(E * len(VOCAB)))),
    ("output_norm.weight", [E], ones(E)),
]
kv_dim = (E // HEADS) * KV
for i in range(LAYERS):
    shared += [
        (f"blk.{i}.attn_norm.weight", [E], ones(E)),
        (f"blk.{i}.attn_q.weight", [E, E], pack(flist(E * E))),
        (f"blk.{i}.attn_k.weight", [E, kv_dim], pack(flist(E * kv_dim))),
        (f"blk.{i}.attn_v.weight", [E, kv_dim], pack(flist(E * kv_dim))),
        (f"blk.{i}.attn_output.weight", [E, E], pack(flist(E * E))),
        (f"blk.{i}.ffn_norm.weight", [E], ones(E)),
    ]

# per-layer dense FFN weights (gate/up shared across experts; down is the oracle).
# Scaled up so the FFN materially drives the logits — otherwise the tiny random
# model is degenerate (argmax is FFN-independent) and the test cannot tell a
# correct MoE from a broken one.
ffn = {}
for i in range(LAYERS):
    ffn[i] = dict(gate=[v * 8 for v in flist(E * FF)],
                  up=[v * 8 for v in flist(E * FF)],
                  down=[v * 8 for v in flist(FF * E)])


def dense_ffn(i):
    return [
        (f"blk.{i}.ffn_gate.weight", [E, FF], pack(ffn[i]["gate"])),
        (f"blk.{i}.ffn_up.weight", [E, FF], pack(ffn[i]["up"])),
        (f"blk.{i}.ffn_down.weight", [FF, E], pack(ffn[i]["down"])),
    ]


def moe_ffn(i, n_expert, expert1_down):
    # gate_exps/up_exps: the shared gate/up, repeated per expert (ne {E, FF, n_expert})
    gate_exps = ffn[i]["gate"] * n_expert
    up_exps = ffn[i]["up"] * n_expert
    # down_exps ne {FF, E, n_expert}: expert 0 = dense down, expert 1 = expert1_down
    down_exps = list(ffn[i]["down"]) + list(expert1_down)
    return [
        (f"blk.{i}.ffn_gate_inp.weight", [E, n_expert], zeros(E * n_expert)),  # router
        (f"blk.{i}.ffn_gate_exps.weight", [E, FF, n_expert], pack(gate_exps)),
        (f"blk.{i}.ffn_up_exps.weight", [E, FF, n_expert], pack(up_exps)),
        (f"blk.{i}.ffn_down_exps.weight", [FF, E, n_expert], pack(down_exps)),
    ]


dense = list(shared)
for i in range(LAYERS):
    dense += dense_ffn(i)
write(f"{OUT}.dense.gguf", dense, base_meta("llama", []))

# moe1: expert_used=1, expert 0 = dense, expert 1 = zeros
moe1 = list(shared)
for i in range(LAYERS):
    moe1 += moe_ffn(i, 2, [0.0] * (FF * E))
write(f"{OUT}.moe1.gguf", moe1,
      base_meta("llama", [ku("llama.expert_count", 2), ku("llama.expert_used_count", 1)]))

# moe2: expert_used=2, both experts = dense
moe2 = list(shared)
for i in range(LAYERS):
    moe2 += moe_ffn(i, 2, ffn[i]["down"])
write(f"{OUT}.moe2.gguf", moe2,
      base_meta("llama", [ku("llama.expert_count", 2), ku("llama.expert_used_count", 2)]))
