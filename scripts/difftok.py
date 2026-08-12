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
import hashlib
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


def corpus_digest(corpus):
    """Identity of the exact strings a capture was taken against."""
    h = hashlib.sha256()
    for s in corpus:
        h.update(s.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


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
    ap.add_argument("--ref",
                    help="HF repo id, a directory containing tokenizer.json, "
                         "or a path to tokenizer.json. Not needed with "
                         "--ref-ids.")
    # Offline reference. Loading --ref by repo id needs the network and, for a
    # gated repo, credentials -- which is what keeps this check out of CI and
    # off contributors' machines. The reference tokenization of a fixed corpus
    # is a constant, though, so capture it ONCE from an authenticated run and
    # every later run is offline and reproducible.
    ap.add_argument("--capture", metavar="PATH",
                    help="write the reference ids for this corpus to PATH "
                         "(use during an authenticated run)")
    ap.add_argument("--ref-ids", metavar="PATH",
                    help="read reference ids from a --capture file instead of "
                         "loading a tokenizer; no network, no credentials")
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

    if args.ref_ids:
        with open(args.ref_ids, encoding="utf-8") as f:
            cap = json.load(f)
        # The ids mean nothing without the corpus they were captured against,
        # so refuse a mismatch loudly rather than comparing against the wrong
        # strings and reporting confident nonsense.
        want = corpus_digest(corpus)
        if cap.get("corpus_sha256") != want:
            raise SystemExit(
                "--ref-ids was captured against a different corpus "
                f"({cap.get('corpus_sha256', '?')[:12]} != {want[:12]}); "
                "recapture it")
        theirs = [list(x) for x in cap["ids"]]
        if len(theirs) != len(corpus):
            raise SystemExit("--ref-ids has %d rows, corpus has %d"
                             % (len(theirs), len(corpus)))
        tok = None
    else:
        if not args.ref:
            raise SystemExit("one of --ref or --ref-ids is required")
        tok = reference_tokenizer(args.ref)
        theirs = [tok.encode(s, add_special_tokens=False).ids for s in corpus]
        if args.capture:
            with open(args.capture, "w", encoding="utf-8") as f:
                json.dump({"ref": args.ref,
                           "corpus": os.path.basename(args.corpus),
                           "corpus_sha256": corpus_digest(corpus),
                           "ids": theirs}, f)
            print(f"captured {len(theirs)} reference tokenizations to "
                  f"{args.capture}")

    bad = [i for i in range(len(corpus)) if mine[i] != theirs[i]]

    for i in bad[: args.show]:
        s = corpus[i]
        print(f"--- line {i + 1}: {json.dumps(s, ensure_ascii=False)}")
        print(f"    runner: {mine[i]}")
        if tok is not None:
            print(f"            {[tok.decode([t]) if t < tok.get_vocab_size() else '?' for t in mine[i]]}")
        print(f"    ref   : {theirs[i]}")
        if tok is not None:
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
