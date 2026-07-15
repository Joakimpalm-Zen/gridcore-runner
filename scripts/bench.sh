#!/bin/sh
# reproducible decode benchmark: 3 greedy runs, per-run gen tok/s + output md5.
# decision metric is the MEDIAN tok/s; the md5s must all match the baseline.
M=${1:?usage: bench.sh model.gguf [n_tokens]}
N=${2:-256}
EXE=./runner
[ -x ./runner.exe ] && EXE=./runner.exe
for i in 1 2 3; do
    "$EXE" -m "$M" -p "Write a short story about a lighthouse keeper." \
        -n "$N" --temp 0 -s 1 > "bench-out.$i.txt" 2> "bench-err.$i.txt"
    grep -o 'gen: [0-9]* tok, [0-9.]* tok/s' "bench-err.$i.txt"
done
md5sum bench-out.1.txt bench-out.2.txt bench-out.3.txt
