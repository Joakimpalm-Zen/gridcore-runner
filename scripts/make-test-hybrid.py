#!/usr/bin/env python3
"""Generate tiny Mamba-2 hybrid GGUF fixtures for the SSM path.

The runner now RUNS `granitehybrid` (Granite-4 h-series) — its Mamba-2 decode
step, the granite muP scaling and the NoPE attention are implemented and gated
token-identically against llama.cpp b10353 on the real granite-4.0-h-small
(tracer 2). `nemotron_h_moe` (Nemotron-3.5 Lightning) shares the Mamba-2 tensor
set but adds a grouped scan this tracer has not certified, so it is still
RECOGNIZED and refused with a specific "forward not yet implemented" message.

This fixture is structurally faithful to a real granite-4.0-h GGUF (confirmed
off the file on the box): a sparse-MoE hybrid where a minority of layers are
GQA attention (attention.head_count_kv != 0) and the rest are Mamba-2 recurrent
mixers (head_count_kv == 0), every layer carrying a routed MoE FFN plus an
always-on shared expert. It is small but honors the load-bearing invariants:

  * ssm.inner_size == 2 * embedding_length  (the Mamba-2 expansion factor),
  * ssm_in.weight [E, 2*inner + 2*groups*state + heads]  the zxBCdt projection,
  * ssm_conv1d.{weight,bias} over conv_dim = inner + 2*groups*state,
  * ssm_a, ssm_d [1, heads]; ssm_dt.bias [heads]; ssm_norm.weight [inner];
    ssm_out.weight [inner, E],
  * the four granite scalars (embedding/attention/residual/logit),
  * rope.scaling.finetuned = false, so the attention layers are NoPE.

`granitehybrid` loads and decodes it; `nemotron_h_moe` is refused. The
`.missing-ssm_d` variant drops one SSM tensor so admission must FAIL CLOSED
naming it (the hostile-GGUF discipline).

Usage:  make-test-hybrid.py <out-prefix> [--arch granitehybrid]
Writes <out>.gguf (valid) and <out>.missing-ssm_d.gguf (drops one SSM tensor).
"""
import struct
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "test-hybrid"
ARCH = "granitehybrid"
if "--arch" in sys.argv:
    ARCH = sys.argv[sys.argv.index("--arch") + 1]

# tiny geometry (structurally faithful to Mamba-2, not to any real size)
E = 32                    # embedding_length
HEADS, KV = 4, 2          # attention head counts (attention layers only)
HEAD_DIM = E // HEADS     # 8
N_SSM_HEADS = 4           # ssm.time_step_rank
SSM_INNER = 2 * E         # ssm.inner_size == 2*E (Mamba-2 expansion), = 64
SSM_STATE = 8             # ssm.state_size
SSM_GROUPS = 1            # ssm.group_count
SSM_CONV = 4              # ssm.conv_kernel
SSM_HDIM = SSM_INNER // N_SSM_HEADS                  # 16 (mamba head dim)
CONV_DIM = SSM_INNER + 2 * SSM_GROUPS * SSM_STATE    # x, B, C convolved together
INPROJ = 2 * SSM_INNER + 2 * SSM_GROUPS * SSM_STATE + N_SSM_HEADS  # z, xBC, dt
# MoE
N_EXPERT, N_USED = 4, 2
FF_EXP = 16               # per-expert FFN width
FF_SHEXP = 16            # shared-expert FFN width
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
# ssm_a is stored as the (negative) decay coefficient A, used directly as
# exp(dt*A); make it plausibly negative so the recurrence is a decay.
def negs(n): return pack([-(0.5 + abs(rnd())) for _ in range(n)])


def meta():
    p = ARCH
    return [
        ks("general.architecture", ARCH),
        ku(f"{p}.block_count", LAYERS), ku(f"{p}.context_length", 256),
        ku(f"{p}.embedding_length", E), ku(f"{p}.feed_forward_length", FF_EXP),
        ku(f"{p}.attention.head_count", HEADS),
        kau(f"{p}.attention.head_count_kv", LAYER_KV),
        kf(f"{p}.attention.layer_norm_rms_epsilon", 1e-5),
        ku(f"{p}.rope.dimension_count", HEAD_DIM),
        kf(f"{p}.rope.freq_base", 10000.0),
        # granite-4.0-h uses rope_finetuned as an on/off switch for rope; false
        # means the attention layers are NoPE (position comes from the mixers).
        kb(f"{p}.rope.scaling.finetuned", False),
        # sparse-MoE FFN with an always-on shared expert (granite MoE shared)
        ku(f"{p}.expert_count", N_EXPERT), ku(f"{p}.expert_used_count", N_USED),
        ku(f"{p}.expert_feed_forward_length", FF_EXP),
        ku(f"{p}.expert_shared_feed_forward_length", FF_SHEXP),
        # Mamba-2 SSM block
        ku(f"{p}.ssm.conv_kernel", SSM_CONV),
        ku(f"{p}.ssm.inner_size", SSM_INNER),
        ku(f"{p}.ssm.state_size", SSM_STATE),
        ku(f"{p}.ssm.group_count", SSM_GROUPS),
        ku(f"{p}.ssm.time_step_rank", N_SSM_HEADS),
        # granite scaling multipliers (load-bearing; forward is wrong without them)
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


def moe(i):
    # routed experts (fused 3D: {E, FF_EXP, N_EXPERT} etc.) + shared expert
    return [
        (f"blk.{i}.ffn_norm.weight", [E], ones(E)),
        (f"blk.{i}.ffn_gate_inp.weight", [E, N_EXPERT], pack(flist(E * N_EXPERT))),
        (f"blk.{i}.ffn_gate_exps.weight", [E, FF_EXP, N_EXPERT], pack(flist(E * FF_EXP * N_EXPERT))),
        (f"blk.{i}.ffn_up_exps.weight", [E, FF_EXP, N_EXPERT], pack(flist(E * FF_EXP * N_EXPERT))),
        (f"blk.{i}.ffn_down_exps.weight", [FF_EXP, E, N_EXPERT], pack(flist(FF_EXP * E * N_EXPERT))),
        (f"blk.{i}.ffn_gate_shexp.weight", [E, FF_SHEXP], pack(flist(E * FF_SHEXP))),
        (f"blk.{i}.ffn_up_shexp.weight", [E, FF_SHEXP], pack(flist(E * FF_SHEXP))),
        (f"blk.{i}.ffn_down_shexp.weight", [FF_SHEXP, E], pack(flist(FF_SHEXP * E))),
    ]


def ssm_layer(i, drop=None):
    t = [
        (f"blk.{i}.attn_norm.weight", [E], ones(E)),
        (f"blk.{i}.ssm_in.weight", [E, INPROJ], pack(flist(E * INPROJ))),
        (f"blk.{i}.ssm_conv1d.weight", [SSM_CONV, CONV_DIM], pack(flist(SSM_CONV * CONV_DIM))),
        (f"blk.{i}.ssm_conv1d.bias", [CONV_DIM], pack(flist(CONV_DIM))),
        (f"blk.{i}.ssm_a", [1, N_SSM_HEADS], negs(N_SSM_HEADS)),
        (f"blk.{i}.ssm_d", [1, N_SSM_HEADS], pack(flist(N_SSM_HEADS))),
        (f"blk.{i}.ssm_dt.bias", [N_SSM_HEADS], pack(flist(N_SSM_HEADS))),
        (f"blk.{i}.ssm_norm.weight", [SSM_INNER], ones(SSM_INNER)),
        (f"blk.{i}.ssm_out.weight", [SSM_INNER, E], pack(flist(SSM_INNER * E))),
    ] + moe(i)
    if drop:
        t = [x for x in t if not x[0].endswith("." + drop)]
    return t


def attn_layer(i):
    kv_dim = HEAD_DIM * KV
    return [
        (f"blk.{i}.attn_norm.weight", [E], ones(E)),
        (f"blk.{i}.attn_q.weight", [E, HEAD_DIM * HEADS], pack(flist(E * HEAD_DIM * HEADS))),
        (f"blk.{i}.attn_k.weight", [E, kv_dim], pack(flist(E * kv_dim))),
        (f"blk.{i}.attn_v.weight", [E, kv_dim], pack(flist(E * kv_dim))),
        (f"blk.{i}.attn_output.weight", [HEAD_DIM * HEADS, E], pack(flist(HEAD_DIM * HEADS * E))),
    ] + moe(i)


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
