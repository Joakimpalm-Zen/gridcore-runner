#!/bin/sh
# temporary: regen PTX with MSVC + nvcc on PATH
export PATH="/c/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64:$PATH"
cd "$(dirname "$0")/.."
"/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/bin/nvcc.exe" \
    -ptx -arch=compute_75 -O3 -o src/kernels.ptx src/kernels.cu || exit 1
python scripts/embed-ptx.py || python3 scripts/embed-ptx.py
