#!/usr/bin/env python3
"""Generate a tiny `nemotron_h` (Nemotron-Nano-9B-v2 family) GGUF fixture.

The runner RUNS `nemotron_h`: a Mamba-2 / attention / MLP hybrid where each
block is EXACTLY ONE of three kinds (mutually exclusive, one pre-norm + one
residual), typed off two per-layer arrays:

  * recurrent (SSM) : head_count_kv[i] == 0 && feed_forward_length[i] == 0,
  * attention       : head_count_kv[i]  > 0  (feed_forward_length 0), NoPE,
  * MLP             : feed_forward_length[i] > 0, a gate-less squared-ReLU FFN.

Unlike `granitehybrid` this family is NON-MoE and carries NO muP scalars, and
crucially it uses a GROUPED scan (ssm.group_count > 1): the B/C projections are
shared across groups of heads and broadcast (group g covers heads
[g*(H/G), (g+1)*(H/G))). This fixture sets group_count = 2 to exercise that
broadcast (granite's fixture is group_count = 1), plus ssm_inner != 2*embd (as
the real Nano-9B-v2, where inner=10240, embd=4480 — so granite's inner==2*embd
assertion must be ABSENT for this arch). Structurally faithful to the real GGUF
confirmed on the box: ssm_in [E, 2*inner+2*groups*state+heads], ssm_conv1d over
conv_dim = inner+2*groups*state, ssm_a/ssm_d [1,heads], ssm_dt.bias [heads],
ssm_norm.weight [inner/groups, groups], ssm_out [inner, E]; attention blocks
carry only attn_q/k/v/output (no biases, no qk-norm); MLP blocks only
ffn_up/ffn_down (no gate). rope.scaling.finetuned=false ⇒ NoPE attention.

Usage:  make-test-nemotron.py <out-prefix>
Writes <out>.gguf (valid) and <out>.missing-ssm_d.gguf (drops one SSM tensor,
so admission must FAIL CLOSED naming it — the hostile-GGUF discipline).
"""
import struct
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "test-nemotron"
ARCH = "nemotron_h"

# tiny geometry (structurally faithful to Mamba-2, not to any real size)
E = 32                    # embedding_length
HEADS, KV = 4, 2          # attention head counts (attention layers only)
HEAD_DIM = E // HEADS     # 8
N_SSM_HEADS = 4           # ssm.time_step_rank
SSM_INNER = 48            # ssm.inner_size — deliberately NOT 2*E (Nano: 10240 != 2*4480)
SSM_STATE = 8             # ssm.state_size
SSM_GROUPS = 2            # ssm.group_count > 1: exercise the grouped B/C broadcast
SSM_CONV = 4              # ssm.conv_kernel
SSM_HDIM = SSM_INNER // N_SSM_HEADS                  # 12 (mamba head dim)
CONV_DIM = SSM_INNER + 2 * SSM_GROUPS * SSM_STATE    # x, B, C convolved together
INPROJ = 2 * SSM_INNER + 2 * SSM_GROUPS * SSM_STATE + N_SSM_HEADS  # z, xBC, dt
FF = 40                   # MLP feed_forward_length (gate-less)
# layer types (per-layer arrays): SSM, MLP, attention, SSM.
#   head_count_kv : 0 => not attention;  feed_forward_length : >0 => MLP.
LAYER_KV = [0, 0, KV, 0]
LAYER_FF = [0, FF, 0, 0]
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
def negs(n): return pack([-(0.5 + abs(rnd())) for _ in range(n)])


def meta():
    p = ARCH
    return [
        ks("general.architecture", ARCH),
        ku(f"{p}.block_count", LAYERS), ku(f"{p}.context_length", 256),
        ku(f"{p}.embedding_length", E),
        # per-layer FFN width ARRAY: 0 on the SSM/attention blocks, FF on MLP.
        kau(f"{p}.feed_forward_length", LAYER_FF),
        ku(f"{p}.attention.head_count", HEADS),
        kau(f"{p}.attention.head_count_kv", LAYER_KV),
        ku(f"{p}.attention.key_length", HEAD_DIM),
        ku(f"{p}.attention.value_length", HEAD_DIM),
        kf(f"{p}.attention.layer_norm_rms_epsilon", 1e-5),
        ku(f"{p}.rope.dimension_count", HEAD_DIM),
        kf(f"{p}.rope.freq_base", 10000.0),
        # NoPE: the attention layers apply no rope (position comes from the
        # Mamba layers); nemotron marks this with rope.scaling.finetuned=false.
        kb(f"{p}.rope.scaling.finetuned", False),
        # Mamba-2 SSM block — grouped scan (group_count > 1), inner != 2*E.
        ku(f"{p}.ssm.conv_kernel", SSM_CONV),
        ku(f"{p}.ssm.inner_size", SSM_INNER),
        ku(f"{p}.ssm.state_size", SSM_STATE),
        ku(f"{p}.ssm.group_count", SSM_GROUPS),
        ku(f"{p}.ssm.time_step_rank", N_SSM_HEADS),
        # NON-MoE, NO muP scalars: expert_* and embedding/logit/residual/
        # attention scales are all ABSENT, so the runner keeps its off defaults.
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
    # ssm_norm is [inner/groups, groups] (granite's is [inner]); both flatten to
    # `inner` contiguous floats the grouped RMS norm reads.
    t = [
        (f"blk.{i}.attn_norm.weight", [E], ones(E)),
        (f"blk.{i}.ssm_in.weight", [E, INPROJ], pack(flist(E * INPROJ))),
        (f"blk.{i}.ssm_conv1d.weight", [SSM_CONV, CONV_DIM], pack(flist(SSM_CONV * CONV_DIM))),
        (f"blk.{i}.ssm_conv1d.bias", [CONV_DIM], pack(flist(CONV_DIM))),
        (f"blk.{i}.ssm_a", [1, N_SSM_HEADS], negs(N_SSM_HEADS)),
        (f"blk.{i}.ssm_d", [1, N_SSM_HEADS], pack(flist(N_SSM_HEADS))),
        (f"blk.{i}.ssm_dt.bias", [N_SSM_HEADS], pack(flist(N_SSM_HEADS))),
        (f"blk.{i}.ssm_norm.weight", [SSM_INNER // SSM_GROUPS, SSM_GROUPS], ones(SSM_INNER)),
        (f"blk.{i}.ssm_out.weight", [SSM_INNER, E], pack(flist(SSM_INNER * E))),
    ]
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
    ]


def mlp_layer(i):
    # gate-less squared-ReLU MLP: attn_norm doubles as the FFN input norm.
    return [
        (f"blk.{i}.attn_norm.weight", [E], ones(E)),
        (f"blk.{i}.ffn_up.weight", [E, FF], pack(flist(E * FF))),
        (f"blk.{i}.ffn_down.weight", [FF, E], pack(flist(FF * E))),
    ]


def build(drop=None):
    ts = [
        ("token_embd.weight", [E, len(VOCAB)], pack(flist(E * len(VOCAB)))),
        ("output_norm.weight", [E], ones(E)),
        ("output.weight", [E, len(VOCAB)], pack(flist(E * len(VOCAB)))),
    ]
    for i in range(LAYERS):
        if LAYER_KV[i] > 0:
            ts += attn_layer(i)
        elif LAYER_FF[i] > 0:
            ts += mlp_layer(i)
        else:
            ts += ssm_layer(i, drop)
    return ts


write(f"{OUT}.gguf", build(), meta())
write(f"{OUT}.missing-ssm_d.gguf", build(drop="ssm_d"), meta())
