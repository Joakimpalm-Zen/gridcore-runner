#!/bin/sh
# Fetch a small GGUF model for testing runner.
set -e
mkdir -p models
FILE="models/SmolLM2-135M-Instruct-Q8_0.gguf"
# sha256 of the published file — update when bumping the model
SHA256="5a1395716f7913741cc51d98581b9b1228d80987a9f7d3664106742eb06bba83"

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
echo "Downloading SmolLM2-135M-Instruct Q8_0 (~138 MB)..."
# -C -: resume a partial download instead of restarting
curl -L --progress-bar -C - --retry 3 -o "$FILE" \
  "https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q8_0.gguf"
if ! verify; then
    echo "error: checksum mismatch — removing $FILE (rerun to retry)" >&2
    rm -f "$FILE"
    exit 1
fi
echo "Done: $FILE (checksum ok)"
echo "Try:  ./runner -m $FILE -i"
