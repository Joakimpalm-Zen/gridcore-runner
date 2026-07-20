#!/bin/sh
# Fetch a real SPM (llama) vocabulary for the tokenizer tests.
#
# test.gguf carries a byte-fallback-only vocab, which cannot exercise
# score-based piece merging. TinyLlama-1.1B-Chat uses the Llama-2 SPM vocab
# (32000 pieces with scores), so tests/test_tokenizer.c runs against it.
# Without this file those tests skip rather than fail.
set -e
cd "$(dirname "$0")/.."
mkdir -p models
FILE="models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
# sha256 of the published file — update when bumping the model
SHA256="9fecc3b3cd76bba89d504f29b616eedf7da85b96540e490ca5824d3f7d2776a0"

verify() {
    if command -v sha256sum >/dev/null 2>&1; then
        echo "$SHA256  $FILE" | sha256sum -c - >/dev/null 2>&1
    elif command -v shasum >/dev/null 2>&1; then
        echo "$SHA256  $FILE" | shasum -a 256 -c - >/dev/null 2>&1
    else
        echo "warning: no sha256 tool found — skipping verification" >&2
        return 0
    fi
}

if [ -f "$FILE" ] && verify; then
    echo "$FILE already exists (checksum ok)"
    exit 0
fi
echo "Downloading TinyLlama-1.1B-Chat Q4_K_M (~638 MB)..."
curl -L --progress-bar -C - --retry 3 -o "$FILE" \
  "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
if ! verify; then
    echo "error: checksum mismatch — removing $FILE (rerun to retry)" >&2
    rm -f "$FILE"
    exit 1
fi
echo "Done: $FILE (checksum ok)"
echo "Try:  make test"
