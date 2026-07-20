#!/usr/bin/env python3
"""Tokenizer differential test: Runner vs the HuggingFace reference.

    scripts/difftok.py --gguf models/X.gguf --ref <hf-repo-id-or-dir> [--corpus F]

Builds the difftok harness (make difftok), runs it over the corpus, runs
`tokenizers` over the same corpus, and reports how many strings differ.

Both sides encode with no special tokens added: BOS/EOS placement is a chat
template concern, and folding it in here would mask real merge differences
behind a constant offset.

Divergences are printed with the decoded pieces from both sides, because "which
strings" matters far more than "how many" -- Mistral's known 2/721 is only
acceptable because every one of them begins with whitespace.

Exit status is 0 if the divergence count matches --expect (default 0).
"""

import argparse
import json
import os
import subprocess
import sys


def build(root, exe):
    make = os.environ.get("MAKE", "make")
    cc = os.environ.get("CC")
    cmd = [make, "-C", root, exe]
    if cc:
        cmd.append(f"CC={cc}")
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)


def load_corpus(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n").rstrip("\r")
            if not line:
                continue
            out.append(json.loads(line))
    return out


def runner_ids(exe, gguf, corpus_path, n):
    r = subprocess.run([exe, gguf, corpus_path], capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stderr)
        raise SystemExit(f"difftok harness failed ({r.returncode})")
    lines = r.stdout.split("\n")
    while lines and lines[-1] == "":
        lines.pop()
    if len(lines) != n:
        raise SystemExit(f"harness emitted {len(lines)} lines, corpus has {n}")
    return [[int(x) for x in ln.split()] for ln in lines]


def reference_tokenizer(ref):
    from tokenizers import Tokenizer

    if os.path.isdir(ref):
        return Tokenizer.from_file(os.path.join(ref, "tokenizer.json"))
    if os.path.isfile(ref):
        return Tokenizer.from_file(ref)
    return Tokenizer.from_pretrained(ref)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--ref", required=True,
                    help="HF repo id, a directory containing tokenizer.json, "
                         "or a path to tokenizer.json")
    ap.add_argument("--corpus", default="tests/fixtures/tokenizer-corpus.txt")
    ap.add_argument("--expect", type=int, default=0,
                    help="expected number of diverging strings")
    ap.add_argument("--show", type=int, default=20,
                    help="max divergences to print in detail")
    ap.add_argument("--no-build", action="store_true")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    exe = os.path.join(root, "difftok.exe" if os.name == "nt" else "difftok")
    if not args.no_build:
        build(root, os.path.basename(exe))

    corpus = load_corpus(args.corpus)
    mine = runner_ids(exe, args.gguf, args.corpus, len(corpus))

    tok = reference_tokenizer(args.ref)
    theirs = [tok.encode(s, add_special_tokens=False).ids for s in corpus]

    bad = [i for i in range(len(corpus)) if mine[i] != theirs[i]]

    for i in bad[: args.show]:
        s = corpus[i]
        print(f"--- line {i + 1}: {json.dumps(s, ensure_ascii=False)}")
        print(f"    runner: {mine[i]}")
        print(f"            {[tok.decode([t]) if t < tok.get_vocab_size() else '?' for t in mine[i]]}")
        print(f"    ref   : {theirs[i]}")
        print(f"            {[tok.decode([t]) for t in theirs[i]]}")
    if len(bad) > args.show:
        print(f"... and {len(bad) - args.show} more")

    leading_ws = sum(1 for i in bad if corpus[i][:1] in (" ", "\t", "\n", "\r"))
    print(f"\n{len(bad)}/{len(corpus)} strings differ "
          f"({leading_ws} of them begin with whitespace)")

    if len(bad) != args.expect:
        print(f"FAIL: expected {args.expect}", file=sys.stderr)
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
