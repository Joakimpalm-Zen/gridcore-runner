#!/usr/bin/env python3
"""Inspect an Ornith GGUF and optionally run a deterministic reference smoke.

The official model is intentionally not committed.  This script turns that
external artifact into a repeatable gate: first validate its Qwen3.5 hybrid
layout, then prove that Runner and (optionally) llama.cpp can both complete the
same short greedy prompt.
"""

import argparse
import hashlib
import json
import os
import re
import struct
import subprocess
import sys


GGML_TYPES = {
    0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20),
    6: (32, 22), 7: (32, 24), 8: (32, 34), 9: (32, 36),
    10: (256, 84), 11: (256, 110), 12: (256, 144),
    13: (256, 176), 14: (256, 210), 15: (256, 292),
    16: (256, 66), 17: (256, 74), 18: (256, 98),
    19: (256, 50), 20: (32, 16), 21: (256, 82),
    22: (256, 82), 23: (256, 136), 24: (1, 1),
    25: (1, 2), 26: (1, 4), 27: (1, 8), 28: (1, 8),
    29: (256, 56), 30: (1, 2),
}


class Reader:
    def __init__(self, stream):
        self.stream = stream

    def raw(self, size):
        data = self.stream.read(size)
        if len(data) != size:
            raise ValueError("GGUF ended inside its metadata or tensor index")
        return data

    def u32(self):
        return struct.unpack("<I", self.raw(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.raw(8))[0]

    def string(self):
        return self.raw(self.u64()).decode("utf-8", "replace")

    def value(self, value_type):
        formats = {
            0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
            6: "<f", 7: "<B", 10: "<Q", 11: "<q", 12: "<d",
        }
        if value_type in formats:
            size = struct.calcsize(formats[value_type])
            value = struct.unpack(formats[value_type], self.raw(size))[0]
            return bool(value) if value_type == 7 else value
        if value_type == 8:
            return self.string()
        if value_type == 9:
            element_type = self.u32()
            return [self.value(element_type) for _ in range(self.u64())]
        raise ValueError(f"unsupported GGUF metadata type {value_type}")


def read_gguf(path):
    with open(path, "rb") as stream:
        reader = Reader(stream)
        if reader.raw(4) != b"GGUF":
            raise ValueError("not a GGUF file")
        version = reader.u32()
        tensor_count = reader.u64()
        metadata_count = reader.u64()
        metadata = {}
        for _ in range(metadata_count):
            key = reader.string()
            metadata[key] = reader.value(reader.u32())
        tensors = []
        for _ in range(tensor_count):
            name = reader.string()
            dimensions = [reader.u64() for _ in range(reader.u32())]
            tensor_type = reader.u32()
            offset = reader.u64()
            tensors.append((name, dimensions, tensor_type, offset))
        index_end = stream.tell()

    alignment = metadata.get("general.alignment", 32)
    data_start = (index_end + alignment - 1) // alignment * alignment
    required_size = data_start
    for name, dimensions, tensor_type, offset in tensors:
        if tensor_type not in GGML_TYPES:
            raise ValueError(f"{name} uses unsupported GGML type {tensor_type}")
        elements = 1
        for dimension in dimensions:
            elements *= dimension
        block, block_bytes = GGML_TYPES[tensor_type]
        tensor_bytes = (elements + block - 1) // block * block_bytes
        required_size = max(required_size, data_start + offset + tensor_bytes)
    actual_size = os.path.getsize(path)
    if actual_size < required_size:
        raise ValueError(
            f"truncated GGUF: {actual_size} bytes, needs {required_size}"
        )
    return {
        "version": version,
        "metadata": metadata,
        "tensors": tensors,
        "size": actual_size,
        "required_size": required_size,
    }


def validate_layout(metadata, tensor_names):
    errors = []
    block_count = metadata.get("qwen35.block_count")
    interval = metadata.get("qwen35.full_attention_interval")
    if metadata.get("general.architecture") != "qwen35":
        errors.append("general.architecture must be qwen35")
    if metadata.get("tokenizer.ggml.pre") != "qwen35":
        errors.append("tokenizer.ggml.pre must be qwen35")
    if not isinstance(block_count, int) or block_count <= 0:
        errors.append("qwen35.block_count must be a positive integer")
        block_count = 0
    if not isinstance(interval, int) or interval <= 0:
        errors.append("qwen35.full_attention_interval must be positive")
        interval = 1

    full = sorted(
        int(match.group(1))
        for name in tensor_names
        if (match := re.fullmatch(r"blk\.(\d+)\.attn_q\.weight", name))
    )
    linear = sorted(
        int(match.group(1))
        for name in tensor_names
        if (match := re.fullmatch(r"blk\.(\d+)\.attn_qkv\.weight", name))
    )
    expected_full = list(range(interval - 1, block_count, interval))
    expected_linear = [
        block for block in range(block_count) if block not in expected_full
    ]
    if full != expected_full:
        errors.append(
            f"full-attention blocks are {full}, expected {expected_full}"
        )
    if linear != expected_linear:
        errors.append(
            f"linear-attention blocks are {linear}, expected {expected_linear}"
        )
    for required in ("token_embd.weight", "output.weight", "output_norm.weight"):
        if required not in tensor_names:
            errors.append(f"missing required tensor {required}")
    return {
        "ok": not errors,
        "errors": errors,
        "full_attention_blocks": full,
        "linear_attention_blocks": linear,
    }


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def run_smoke(command, timeout):
    completed = subprocess.run(
        command, text=True, capture_output=True, timeout=timeout
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr_tail": completed.stderr[-4000:],
        "ok": completed.returncode == 0 and bool(completed.stdout.strip()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--runner", help="Runner executable to smoke")
    parser.add_argument("--reference", help="llama-cli executable to smoke")
    parser.add_argument("--prompt", default="Reply with exactly: ORNITH_OK")
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--ctx", type=int, default=512)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--sha256", dest="expected_sha256")
    parser.add_argument("--report", help="write the JSON report to this path")
    args = parser.parse_args()

    parsed = read_gguf(args.model)
    metadata = parsed["metadata"]
    layout = validate_layout(metadata, {item[0] for item in parsed["tensors"]})
    report = {
        "schema": "gridcore-runner/ornith-reference/1",
        "model": {
            "path": args.model,
            "size": parsed["size"],
            "required_size": parsed["required_size"],
            "gguf_version": parsed["version"],
            "architecture": metadata.get("general.architecture"),
            "name": metadata.get("general.name"),
            "block_count": metadata.get("qwen35.block_count"),
            "context_length": metadata.get("qwen35.context_length"),
            "tensor_count": len(parsed["tensors"]),
            "layout": layout,
        },
        "smokes": {},
    }
    ok = layout["ok"]
    if args.expected_sha256:
        actual_hash = sha256(args.model)
        report["model"]["sha256"] = actual_hash
        report["model"]["sha256_ok"] = actual_hash == args.expected_sha256
        ok = ok and report["model"]["sha256_ok"]

    if args.runner:
        command = [
            args.runner, "-m", args.model, "-p", args.prompt,
            "-n", str(args.tokens), "--temp", "0", "--ctx", str(args.ctx),
            "--gpu", "off",
        ]
        report["smokes"]["runner"] = run_smoke(command, args.timeout)
        ok = ok and report["smokes"]["runner"]["ok"]
    if args.reference:
        command = [
            args.reference, "-m", args.model, "-p", args.prompt,
            "-n", str(args.tokens), "--temp", "0", "--ctx-size", str(args.ctx),
            "--no-display-prompt", "--no-warmup", "--no-conversation",
            "--simple-io",
        ]
        report["smokes"]["llama_cpp"] = run_smoke(command, args.timeout)
        ok = ok and report["smokes"]["llama_cpp"]["ok"]

    report["ok"] = ok
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as stream:
            stream.write(rendered + "\n")
    print(rendered)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
