#!/usr/bin/env python3
"""Dump a GGUF file's real per-tensor-class quant types — not what its
filename claims. Built for the cert-matrix session, whose whole thesis is
that "Q6_K" or "Q4_K_M" in a GPT-OSS/Gemma-4-MoE filename says nothing about
whether the expert FFN tensors are still MXFP4_MOE or got requantized along
with everything else. Pure stdlib, reads only the header + tensor directory
(never the tensor data), so it is fast and safe on files far larger than
memory.

Usage: python3 scripts/gguf-inspect.py MODEL.gguf
"""
import struct
import sys
from collections import Counter, defaultdict

GGML_TYPE_NAME = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
    14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
    19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS",
    24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M",
    30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4",
}
GGUF_KV_TYPE = {0: "u8", 1: "i8", 2: "u16", 3: "i16", 4: "u32", 5: "i32",
                6: "f32", 7: "bool", 8: "str", 9: "arr", 10: "u64",
                11: "i64", 12: "f64"}


def u32(f): return struct.unpack("<I", f.read(4))[0]
def u64(f): return struct.unpack("<Q", f.read(8))[0]


def read_str(f):
    n = u64(f)
    return f.read(n).decode("utf-8", "replace")


def skip_scalar(f, t):
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    if t == 8:
        read_str(f)
    elif t == 9:
        at = u32(f)
        n = u64(f)
        for _ in range(n):
            skip_scalar(f, at)
    else:
        f.read(sizes[t])


def read_value(f, t):
    if t == 8:
        return read_str(f)
    if t == 9:
        at = u32(f)
        n = u64(f)
        return [read_value(f, at) for _ in range(n)]
    fmt = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
          6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}[t]
    return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]


def is_expert_tensor(name):
    # fused stacked layout (gpt-oss / Mixtral / Qwen3-MoE) and gemma-4's
    # fused gate_up_exps -- anything with a per-expert 3rd dimension.
    return any(s in name for s in (
        "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps", "ffn_gate_up_exps",
        "exp_probs_b", "ffn_gate_inp",
    )) or ".ffn_gate." in name and ".weight" in name and any(
        c.isdigit() for c in name.split(".ffn_gate.")[-1].split(".")[0])


def main():
    path = sys.argv[1]
    f = open(path, "rb")
    magic, version, n_tensors, n_kv = struct.unpack("<IIQQ", f.read(24))
    if magic != 0x46554747:
        sys.exit(f"not a GGUF file (magic {magic:#x})")

    kv = {}
    for _ in range(n_kv):
        k = read_str(f)
        t = u32(f)
        kv[k] = read_value(f, t)

    arch = kv.get("general.architecture", "?")
    print(f"file: {path}")
    print(f"gguf version: {version}  tensors: {n_tensors}  kv: {n_kv}")
    print(f"general.architecture: {arch}")
    print(f"general.name: {kv.get('general.name', '?')}")
    for key in (f"{arch}.block_count", f"{arch}.expert_count",
                f"{arch}.expert_used_count", f"{arch}.expert_shared_count",
                f"{arch}.embedding_length", f"{arch}.feed_forward_length",
                f"{arch}.expert_feed_forward_length",
                f"{arch}.context_length", f"{arch}.rope.scaling.type",
                "tokenizer.ggml.model", "tokenizer.chat_template"):
        if key in kv:
            v = kv[key]
            if isinstance(v, str) and len(v) > 80:
                v = v[:80] + f"...({len(v)} chars)"
            print(f"  {key}: {v}")

    tensors = []
    for _ in range(n_tensors):
        name = read_str(f)
        n_dims = u32(f)
        ne = [u64(f) for _ in range(n_dims)]
        ttype = u32(f)
        offset = u64(f)
        tensors.append((name, n_dims, ne, ttype, offset))

    overall = Counter()
    expert = Counter()
    non_expert = Counter()
    expert_layers = defaultdict(set)
    for name, n_dims, ne, ttype, _ in tensors:
        tname = GGML_TYPE_NAME.get(ttype, f"type{ttype}")
        overall[tname] += 1
        if is_expert_tensor(name):
            expert[tname] += 1
            # blk.N.xxx -> layer N
            parts = name.split(".")
            if len(parts) > 1 and parts[0] == "blk" and parts[1].isdigit():
                expert_layers[int(parts[1])].add(tname)
        else:
            non_expert[tname] += 1

    print(f"\ntensor type histogram (all {n_tensors} tensors): {dict(overall)}")
    print(f"expert-tensor types ({sum(expert.values())} tensors): {dict(expert) or '(none -- dense model)'}")
    print(f"non-expert-tensor types ({sum(non_expert.values())} tensors): {dict(non_expert)}")
    if expert_layers:
        all_types = set()
        for s in expert_layers.values():
            all_types |= s
        uniform = all(s == all_types for s in expert_layers.values())
        print(f"expert layers found: {len(expert_layers)}, "
              f"uniform expert dtype across layers: {uniform}")

    # a few named tensors worth calling out explicitly, if present
    for name, n_dims, ne, ttype, _ in tensors:
        if name in ("token_embd.weight", "output.weight", "output_norm.weight"):
            print(f"  {name}: {GGML_TYPE_NAME.get(ttype, ttype)} ne={ne}")


if __name__ == "__main__":
    main()
