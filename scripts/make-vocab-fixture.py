#!/usr/bin/env python3
"""Generate small vocabulary-only GGUFs for the tokenizer tests.

test.gguf carries a byte-fallback-only vocab (<unk>, <s>, </s>, <0x00>..<0xFF>),
which cannot exercise score-based piece merging. These fixtures carry a real
32000-piece SPM vocabulary (TinyLlama-1.1B-Chat, Apache-2.0, itself the Llama-2
sentencepiece vocab) with no tensors, so they stay small enough to commit and
let CI run the tokenizer assertions instead of skipping them.

Two files are written:

  vocab-spm.gguf            correct sentencepiece scores
  vocab-spm-zeroscores.gguf all-zero scores plus a merges list

The second mirrors a real and widely distributed conversion quirk: many GGUFs
(TheBloke's among them) write tokenizer.ggml.scores as all zeros and put the
merge information in tokenizer.ggml.merges instead. With every score equal, a
score-driven merge loop degenerates to left-to-right and mis-tokenizes.

Regenerating needs network access and two dev-only packages:

    pip install sentencepiece huggingface_hub
    python3 scripts/make-vocab-fixture.py

The generated files are committed, so running the tests needs neither.
"""
import json
import os
import struct

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(REPO, "tests", "fixtures")
MODEL_REPO = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

GGUF_U32, GGUF_F32, GGUF_STR, GGUF_ARR, GGUF_I32, GGUF_BOOL = 4, 6, 8, 9, 5, 7


def s(x):
    b = x.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def kv_u32(k, v):  return s(k) + struct.pack("<II", GGUF_U32, v)
def kv_str(k, v):  return s(k) + struct.pack("<I", GGUF_STR) + s(v)
def kv_bool(k, v): return s(k) + struct.pack("<IB", GGUF_BOOL, 1 if v else 0)


def kv_arr_str(k, items):
    out = [s(k) + struct.pack("<IIQ", GGUF_ARR, GGUF_STR, len(items))]
    out.extend(s(it) for it in items)
    return b"".join(out)


def kv_arr_f32(k, items):
    return (s(k) + struct.pack("<IIQ", GGUF_ARR, GGUF_F32, len(items)) +
            struct.pack(f"<{len(items)}f", *items))


def kv_arr_i32(k, items):
    return (s(k) + struct.pack("<IIQ", GGUF_ARR, GGUF_I32, len(items)) +
            struct.pack(f"<{len(items)}i", *items))


def write_gguf(path, kvs):
    meta = b"".join(kvs)
    # no tensors: n_tensors is 0 and the tensor-info block is empty
    head = struct.pack("<IIQQ", 0x46554747, 3, 0, len(kvs)) + meta
    with open(path, "wb") as f:
        f.write(head + b"\0" * ((-len(head)) % 32))
    print(f"wrote {os.path.relpath(path, REPO)} ({os.path.getsize(path) / 1024:.0f} KB)")


def main():
    import sentencepiece as spm
    from huggingface_hub import hf_hub_download

    sp = spm.SentencePieceProcessor(
        model_file=hf_hub_download(MODEL_REPO, "tokenizer.model"))
    n = sp.get_piece_size()

    tokens = [sp.id_to_piece(i) for i in range(n)]
    scores = [sp.get_score(i) for i in range(n)]
    # GGUF token types: 1 normal, 2 unknown, 3 control, 5 unused, 6 byte
    ttype = []
    for i in range(n):
        if sp.is_unknown(i):    ttype.append(2)
        elif sp.is_control(i):  ttype.append(3)
        elif sp.is_byte(i):     ttype.append(6)
        elif sp.is_unused(i):   ttype.append(5)
        else:                   ttype.append(1)

    with open(hf_hub_download(MODEL_REPO, "tokenizer.json"), encoding="utf-8") as f:
        raw_merges = json.load(f)["model"]["merges"]
    # newer tokenizers.json store merges as ["a", "b"] pairs, older as "a b"
    merges = [m if isinstance(m, str) else " ".join(m) for m in raw_merges]

    common = [
        kv_str("general.architecture", "llama"),
        kv_str("tokenizer.ggml.model", "llama"),
        kv_arr_str("tokenizer.ggml.tokens", tokens),
        kv_arr_i32("tokenizer.ggml.token_type", ttype),
        kv_u32("tokenizer.ggml.bos_token_id", 1),
        kv_u32("tokenizer.ggml.eos_token_id", 2),
        kv_u32("tokenizer.ggml.unknown_token_id", 0),
        kv_bool("tokenizer.ggml.add_bos_token", True),
    ]

    os.makedirs(OUTDIR, exist_ok=True)
    write_gguf(os.path.join(OUTDIR, "vocab-spm.gguf"),
               common + [kv_arr_f32("tokenizer.ggml.scores", scores)])
    write_gguf(os.path.join(OUTDIR, "vocab-spm-zeroscores.gguf"),
               common + [kv_arr_f32("tokenizer.ggml.scores", [0.0] * n),
                         kv_arr_str("tokenizer.ggml.merges", merges)])


def bpe_alphabet():
    """GPT-2 byte <-> unicode alphabet: the 256 single-byte vocabulary pieces."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return [chr(c) for _, c in sorted(zip(bs, cs))]


def make_bpe_fixtures():
    """Small synthetic BPE vocabularies for the pre-tokenizer split rules.

    A real llama-bpe vocabulary is ~5 MB of tokens and merges, far too big to
    commit for what is being checked here: where pre-token boundaries fall.
    So the vocab holds the 256 byte pieces plus exactly the pieces the test
    strings should split into, each built by its own left-to-right merges.
    An id per expected pre-token makes a wrong split immediately visible.

    The files differ only in tokenizer.ggml.pre, which decides whether digits
    group in threes (llama-bpe) or stay single (qwen2, smollm), and whether a
    newline run is one pre-token (llama-bpe) or several (smollm's GPT-2 rules).
    """
    tokens = bpe_alphabet()
    merges = []
    # 'G' is U+0120 (byte-mapped space), 'C' is U+010A (byte-mapped newline)
    for piece in ["tokenization", "/end", "123", "456", "789",
                  "ĊĊ", "ĠĠ", "'ll", "Ġgo", "Ġworld"]:
        acc = piece[0]
        for ch in piece[1:]:
            merges.append(f"{acc} {ch}")
            acc += ch
            if acc not in tokens:
                tokens.append(acc)

    common = [
        kv_str("general.architecture", "llama"),
        kv_str("tokenizer.ggml.model", "gpt2"),
        kv_arr_str("tokenizer.ggml.tokens", tokens),
        kv_arr_i32("tokenizer.ggml.token_type", [1] * len(tokens)),
        kv_arr_str("tokenizer.ggml.merges", merges),
        kv_u32("tokenizer.ggml.bos_token_id", 0),
        kv_u32("tokenizer.ggml.eos_token_id", 0),
        kv_bool("tokenizer.ggml.add_bos_token", False),
    ]
    for name, pre in [("vocab-bpe-llama3.gguf", "llama-bpe"),
                      ("vocab-bpe-qwen2.gguf", "qwen2"),
                      ("vocab-bpe-qwen35.gguf", "qwen35"),
                      ("vocab-bpe-smollm.gguf", "smollm")]:
        write_gguf(os.path.join(OUTDIR, name),
                   common + [kv_str("tokenizer.ggml.pre", pre)])


if __name__ == "__main__":
    main()
    make_bpe_fixtures()
