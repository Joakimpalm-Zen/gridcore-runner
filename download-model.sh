#!/bin/sh
# Fetch a small GGUF model for testing runner.
set -e
mkdir -p models
FILE="models/SmolLM2-135M-Instruct-Q8_0.gguf"
if [ -f "$FILE" ]; then
    echo "$FILE already exists"
    exit 0
fi
echo "Downloading SmolLM2-135M-Instruct Q8_0 (~138 MB)..."
curl -L --progress-bar -o "$FILE" \
  "https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q8_0.gguf"
echo "Done: $FILE"
echo "Try:  ./runner -m $FILE -i"
