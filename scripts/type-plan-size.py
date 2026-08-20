#!/usr/bin/env python3
"""What will this `--type-plan` actually produce? Answer from the header.

A type plan is written blind. Its rules are substrings matched first-wins, so
`{"match": "output.weight"}` also claims every `attn_output.weight`; and the
quantizer silently DECLINES a rule it cannot honour — a target block width
that does not divide the row (`type_fits_row`, 256 for the K-quants, 32 for
q8_0/q4_0) or a target that would grow the tensor (the never-grow rule). Both
declines produce a successful build and a file that is not what the plan said.
On a 30B parent that is a wasted hour per mistake.

This reads only the GGUF header and tensor directory — never the tensor data —
applies the same rules `src/quantize.c` applies, and prints the resulting byte
size, the per-type histogram the quantizer will print, and every rule it will
decline.

    scripts/type-plan-size.py MODEL.gguf PLAN.json [--json]

Prune plans change the tensor directory's contents and are not modelled here;
this answers the precision question only.
"""
import argparse
import json
import struct
import sys

# ggml type -> (block elements, bytes per block). Only the types this quantizer
# can read as a source or write as a target need to be here.
BLOCK = {
    0: (1, 4, "F32"), 1: (1, 2, "F16"), 2: (32, 18, "Q4_0"), 3: (32, 20, "Q4_1"),
    6: (32, 22, "Q5_0"), 7: (32, 24, "Q5_1"), 8: (32, 34, "Q8_0"),
    10: (256, 84, "Q2_K"), 11: (256, 110, "Q3_K"), 12: (256, 144, "Q4_K"),
    13: (256, 176, "Q5_K"), 14: (256, 210, "Q6_K"), 20: (32, 18, "IQ4_NL"),
    23: (256, 136, "IQ4_XS"), 30: (1, 2, "BF16"), 39: (32, 17, "MXFP4"),
}
BY_NAME = {v[2]: k for k, v in BLOCK.items()}
FILE_TYPE_KEY = "general.file_type"
PLAN_TYPE = {"q8_0": "Q8_0", "q4_0": "Q4_0", "q3_k": "Q3_K", "f16": "F16"}


def _read(f):
    """(tensors, data_start, alignment) from a GGUF header + tensor directory."""
    def u32():
        return struct.unpack("<I", f.read(4))[0]

    def u64():
        return struct.unpack("<Q", f.read(8))[0]

    def s():
        return f.read(u64()).decode("utf-8", "replace")

    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

    def skip(t):
        if t == 9:
            et, n = u32(), u64()
            for _ in range(n):
                skip(et)
        elif t == 8:
            s()
        else:
            f.read(sizes[t])

    if f.read(4) != b"GGUF":
        sys.exit("not a GGUF file")
    u32()
    n_tensors, n_kv = u64(), u64()
    alignment = 32
    have_file_type = False
    for _ in range(n_kv):
        key = s()
        t = u32()
        if key == FILE_TYPE_KEY:
            have_file_type = True
        if key == "general.alignment" and t == 4:
            alignment = u32()
        else:
            skip(t)
    tensors = []
    for _ in range(n_tensors):
        name = s()
        ne = [u64() for _ in range(u32())]
        ttype = u32()
        u64()  # offset
        tensors.append((name, ttype, ne))
    pos = f.tell()
    # The rewrite copies every KV and SETS general.file_type. A source without
    # that key therefore grows by one entry, which can push data_start a whole
    # alignment unit further and make an otherwise exact prediction 32 bytes
    # short (observed on the make-test-moe fixtures, which carry no file_type).
    if not have_file_type:
        pos += 8 + len(FILE_TYPE_KEY) + 4 + 4    # str len + key + kv type + u32
    data_start = (pos + alignment - 1) // alignment * alignment
    return tensors, data_start, alignment


def row_bytes(ttype, n):
    blk, sz, _ = BLOCK[ttype]
    return (n // blk) * sz


def should_quantize(name, ne):
    """src/quantize.c:should_quantize"""
    if len(ne) < 2 or ne[0] % 32 or not name.endswith(".weight"):
        return False
    return "_norm." not in name and "rope_freqs" not in name


def resolve(name, plan):
    for rule in plan.get("rules", []):
        if rule["match"] in name:
            return rule["type"]
    return plan.get("default", "keep")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("plan")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    with open(a.model, "rb") as f:
        tensors, data_start, alignment = _read(f)
    plan = json.loads(open(a.plan).read())

    data, hist, declined, never_grew = 0, {}, [], []
    for name, src, ne in tensors:
        if src not in BLOCK:
            sys.exit(f"unsupported source type {src} on {name}")
        out = src
        want = resolve(name, plan)
        if want != "keep" and should_quantize(name, ne):
            if "ffn_gate_inp" in name:
                pass                                   # keep_source_type
            else:
                w = BY_NAME[PLAN_TYPE[want]]
                if ne[0] % BLOCK[w][0]:
                    declined.append({"tensor": name, "type": PLAN_TYPE[want],
                                     "row": ne[0], "block": BLOCK[w][0]})
                elif row_bytes(src, ne[0]) <= row_bytes(w, ne[0]):
                    never_grew.append({"tensor": name, "type": PLAN_TYPE[want],
                                       "source": BLOCK[src][2]})
                else:
                    out = w
        rows = 1
        for d in ne[1:]:
            rows *= d
        n = row_bytes(out, ne[0]) * rows
        data += (n + alignment - 1) // alignment * alignment
        hist[BLOCK[out][2]] = hist.get(BLOCK[out][2], 0) + 1

    total = data_start + data
    report = {"model": a.model, "plan": a.plan, "predicted_bytes": total,
              "histogram": hist, "declined_row_width": declined,
              "declined_never_grow": never_grew}
    if a.json:
        print(json.dumps(report, indent=1))
        return
    print(f"{a.model}\nplan {a.plan}")
    print(f"  predicted size  {total:,} B  ({total / 1e9:.2f} GB / {total / 2 ** 30:.2f} GiB)")
    print(f"  histogram       {hist}")
    for d in declined:
        print(f"  DECLINED (row width) {d['tensor']}: {d['type']} needs "
              f"ne[0] %% {d['block']} == 0, row is {d['row']}")
    for d in never_grew:
        print(f"  DECLINED (never grow) {d['tensor']}: already {d['source']}, "
              f"{d['type']} would be larger")


if __name__ == "__main__":
    main()
