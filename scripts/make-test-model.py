#!/usr/bin/env python3
"""Generate a tiny random llama-architecture GGUF for CI smoke tests.

The model is ~1 MB of random weights with a byte-fallback SPM vocabulary:
output is gibberish, but loading, tokenization, the forward pass, sampling,
and JSON-constrained decoding all execute the same code paths as a real model.
"""
import struct
import sys

OUT = "test.gguf"
SUPPRESS_ALL_BUT_EOS = False
for a in sys.argv[1:]:
    if a == "--suppress-all-but-eos":
        SUPPRESS_ALL_BUT_EOS = True
    else:
        OUT = a

N_EMBD, N_HEAD, N_KV, N_FF, N_LAYER = 64, 4, 2, 128, 2
VOCAB = ["<unk>", "<s>", "</s>"] + [f"<0x{i:02X}>" for i in range(256)]
TTYPE = [2, 3, 3] + [6] * 256  # unknown, control, control, bytes
N_VOCAB = len(VOCAB)

GGUF_U32, GGUF_F32, GGUF_STR, GGUF_ARR, GGUF_I32, GGUF_BOOL = 4, 6, 8, 9, 5, 7


def s(x):
    b = x.encode()
    return struct.pack("<Q", len(b)) + b


def kv_u32(k, v):  return s(k) + struct.pack("<II", GGUF_U32, v)
def kv_f32(k, v):  return s(k) + struct.pack("<If", GGUF_F32, v)
def kv_str(k, v):  return s(k) + struct.pack("<I", GGUF_STR) + s(v)
def kv_bool(k, v): return s(k) + struct.pack("<IB", GGUF_BOOL, 1 if v else 0)


def kv_arr_str(k, items):
    out = s(k) + struct.pack("<IIQ", GGUF_ARR, GGUF_STR, len(items))
    for it in items:
        out += s(it)
    return out


def kv_arr_f32(k, items):
    return (s(k) + struct.pack("<IIQ", GGUF_ARR, GGUF_F32, len(items)) +
            struct.pack(f"<{len(items)}f", *items))


def kv_arr_i32(k, items):
    return (s(k) + struct.pack("<IIQ", GGUF_ARR, GGUF_I32, len(items)) +
            struct.pack(f"<{len(items)}i", *items))


# deterministic pseudo-random floats (no numpy dependency)
_state = 0x12345678


def rnd():
    global _state
    _state = (_state * 1103515245 + 12345) & 0x7FFFFFFF
    return (_state / 0x7FFFFFFF - 0.5) * 0.08


def tensor_data(n):
    return struct.pack(f"<{n}f", *(rnd() for _ in range(n)))


def ones(n):
    return struct.pack(f"<{n}f", *([1.0] * n))


tensors = [("token_embd.weight", [N_EMBD, N_VOCAB], tensor_data(N_EMBD * N_VOCAB)),
           ("output_norm.weight", [N_EMBD], ones(N_EMBD))]
for i in range(N_LAYER):
    kv_dim = N_KV * (N_EMBD // N_HEAD)
    tensors += [
        (f"blk.{i}.attn_norm.weight", [N_EMBD], ones(N_EMBD)),
        (f"blk.{i}.attn_q.weight", [N_EMBD, N_EMBD], tensor_data(N_EMBD * N_EMBD)),
        (f"blk.{i}.attn_k.weight", [N_EMBD, kv_dim], tensor_data(N_EMBD * kv_dim)),
        (f"blk.{i}.attn_v.weight", [N_EMBD, kv_dim], tensor_data(N_EMBD * kv_dim)),
        (f"blk.{i}.attn_output.weight", [N_EMBD, N_EMBD], tensor_data(N_EMBD * N_EMBD)),
        (f"blk.{i}.ffn_norm.weight", [N_EMBD], ones(N_EMBD)),
        (f"blk.{i}.ffn_gate.weight", [N_EMBD, N_FF], tensor_data(N_EMBD * N_FF)),
        (f"blk.{i}.ffn_up.weight", [N_EMBD, N_FF], tensor_data(N_EMBD * N_FF)),
        (f"blk.{i}.ffn_down.weight", [N_FF, N_EMBD], tensor_data(N_FF * N_EMBD)),
    ]

meta_kvs = [
    kv_str("general.architecture", "llama"),
    kv_u32("llama.block_count", N_LAYER),
    kv_u32("llama.context_length", 256),
    kv_u32("llama.embedding_length", N_EMBD),
    kv_u32("llama.feed_forward_length", N_FF),
    kv_u32("llama.attention.head_count", N_HEAD),
    kv_u32("llama.attention.head_count_kv", N_KV),
    kv_f32("llama.attention.layer_norm_rms_epsilon", 1e-5),
    kv_f32("llama.rope.freq_base", 10000.0),
    kv_str("tokenizer.ggml.model", "llama"),
    kv_arr_str("tokenizer.ggml.tokens", VOCAB),
    kv_arr_f32("tokenizer.ggml.scores", [0.0] * N_VOCAB),
    kv_arr_i32("tokenizer.ggml.token_type", TTYPE),
    kv_u32("tokenizer.ggml.bos_token_id", 1),
    kv_u32("tokenizer.ggml.eos_token_id", 2),
    kv_bool("tokenizer.ggml.add_bos_token", True),
]
if SUPPRESS_ALL_BUT_EOS:
    # every token except </s> is suppressed: greedy generation must emit EOS
    # immediately, so a completion prints only the echoed prompt
    meta_kvs.append(kv_arr_i32("tokenizer.ggml.suppress_tokens",
                               [i for i in range(N_VOCAB) if i != 2]))
meta = b"".join(meta_kvs)

header = struct.pack("<IIQQ", 0x46554747, 3, len(tensors), len(meta_kvs))

info = b""
offset = 0
for name, ne, data in tensors:
    info += s(name) + struct.pack("<I", len(ne))
    for d in ne:
        info += struct.pack("<Q", d)
    info += struct.pack("<IQ", 0, offset)  # type F32
    offset += len(data)
    offset = (offset + 31) & ~31

head = header + meta + info
pad = (-len(head)) % 32

with open(OUT, "wb") as f:
    f.write(head + b"\0" * pad)
    for _, _, data in tensors:
        f.write(data)
        f.write(b"\0" * ((-len(data)) % 32))

print(f"wrote {OUT}")
