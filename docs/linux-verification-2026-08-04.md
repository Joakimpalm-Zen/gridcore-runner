# Linux verification pass — RTX PRO 6000 Blackwell box, 2026-08-04 (`63f077e`)

The Linux-only gates that had accumulated behind Windows-only verification, run
on the borrowed Blackwell machine before it is wiped. Every verdict below was
observed on this box; nothing is carried over from a previous report.

Tree: `origin/main` at `63f077e` ("Record the f27e7bb context/KV edge probe"),
which matched the expected HEAD. The local checkout was 16 commits behind with a
clean working tree and no stashes; `HEAD` was a strict ancestor of `origin/main`,
so it was fast-forwarded (`git merge --ff-only origin/main`) rather than merged
or rebased. No local work existed to preserve.

## Machine

```
$ nvidia-smi
NVIDIA-SMI 610.43.02              KMD Version: 610.43.02     CUDA UMD Version: 13.3
GPU 0: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition   MIG M.: Enabled
MIG dev 0: GI 4, CI 0 — 64MiB / 24192MiB
```

```
$ uname -a
Linux c273b4a8ee04 6.12.0-211.16.1.el10_2.x86_64 #1 SMP PREEMPT_DYNAMIC Mon May 18 10:23:57 EDT 2026 x86_64 x86_64 x86_64 GNU/Linux

$ free -g
               total        used        free      shared  buff/cache   available
Mem:             250          21          78           0         153         228
Swap:             21           0          21
```

**The GPU is a MIG 1g.24gb slice, not the whole card.** Every CUDA result below
is a result on that slice. It is a real CUDA device with a real driver — the
driver-dependent claim in Task 1 is testable here — but a full-card run could
differ on anything sensitive to SM count or total VRAM.

`./runner --caps` (abridged to the fields that matter; run before any test, from
the `runner` binary present in the tree at session start):

```json
{"version":"0.1.5-alpha","os":"linux","arch":"x86_64","cpu_cores":128,
 "ram_bytes":268744687616,"ram_available_bytes":245541113856,
 "gpu":{"backend":"cuda",
        "name":"NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition MIG 1g.24gb",
        "unified_memory":false,"min_compute_capability":"7.5","ptx_target":"sm_75",
        "vram_bytes":25367150592,"vram_free_bytes":25130827776,
        "moe":true,"kv_q8":true},
 "kv_types":["f16","q8"],"kv_type_default":"f16","max_models":16}
```

## Toolchain — one correction to the prescribed invocation

The specified `PATH=/root/.conda/envs/ccbuild/bin` is not usable in this session.
The session runs as `uid=1000(lab)`, `/root` is mode `drwx------ root root`, and
there is no usable sudo (`the "no new privileges" flag is set`). The `ccbuild`
environment itself exists and is complete — it just lives under the invoking
user's home, which is also how `MACHINE-NOTE.md` refers to it (`~/.conda/envs/ccbuild`):

```
$ /home/lab/.conda/envs/ccbuild/bin/x86_64-conda-linux-gnu-gcc --version
x86_64-conda-linux-gnu-gcc (conda-forge gcc 15.2.0-19) 15.2.0
$ /home/lab/.conda/envs/ccbuild/bin/clang --version
clang version 22.1.8 (https://github.com/conda-forge/clangdev-feedstock ...)
$ nvcc --version
Cuda compilation tools, release 13.0, V13.0.88
```

So every command below used

```sh
PATH=/home/lab/.conda/envs/ccbuild/bin:/usr/bin:/bin CC=x86_64-conda-linux-gnu-gcc
```

and is otherwise verbatim as specified. This is a path substitution only; the
same conda toolchain was used. No compilation blocker was hit.

The `micromamba install ... clang` fallback was **not needed and not run**:
clang 22.1.8 is already in the environment. (It could not have been run as
given — `/root/.local/bin/micromamba` is unreadable from this account and no
`micromamba` is on `PATH`.)

There is no system compiler at all outside this environment (`gcc`, `cc` and
`clang` are all absent from `/usr/bin`), so nothing below could have silently
fallen back to a different toolchain.

---

## TASK 1 — sanitizer ownership gate: **PASS**

```sh
PATH=/home/lab/.conda/envs/ccbuild/bin:/usr/bin:/bin CC=x86_64-conda-linux-gnu-gcc \
  make -B test-shared-asan test-shared-weights
./test-shared-weights test.gguf
```

Sanitized half — built `-fsanitize=address,undefined -fno-omit-frame-pointer`,
run by the Makefile as `RUNNER_TEST_GPU_OFF=1 LSAN_OPTIONS=suppressions=tests/lsan.supp`:

```
RUNNER_TEST_GPU_OFF=1 LSAN_OPTIONS=suppressions=tests/lsan.supp \
    ./test-shared-asan-bin
kv: head_dim not a multiple of 32 — keeping f16
shared weight tests ok (test.gguf, cpu only)
```

Clean: no LeakSanitizer report, no UBSan diagnostic, `make` exit 0. The only
compiler output was a pre-existing unused-variable warning in
`src/compat.c:309` (`perf_known`), unrelated to this gate.

**No suppression edits.** `tests/lsan.supp` is byte-identical before and after
(`sha256 1a6667e7953595ce5f48bc00407efad22c50961722ed9e455686f7a3a3f39b67`,
`git status --porcelain tests/lsan.supp` empty). The file still scopes the
suppression to `leak:libcuda.so` and nothing was widened.

Ordinary half — real CUDA backend, load/free cycles:

```
gpu-split: budget=25.13GB fixed=0.54GB G=2/2 full=1 used=0.54GB
gpu: CUDA backend on NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition MIG 1g.24gb (0.0 GB weights in VRAM)
gpu: VRAM 25.13 GB free of 25.37 GB after init (kv 0.00 GB + scratch 0.00 GB this instance)
gpu: reusing resident weights (0.0 GB, now shared by 2 instances)
...
vram stable across load/free cycles (0.0 / 0.0 MB)
vram fully returned after unload (0.0 MB outstanding)
shared weight tests ok (test.gguf, gpu backend)
```

Exit 0. Both halves of the stated PASS criterion are met: sanitized target clean
with no suppression edits, ordinary target on a real CUDA backend with VRAM
stable across cycles and fully returned.

### Two things this run does *not* establish

Recording these because the commit being verified is specifically a claim about
driver behaviour:

1. **It does not re-test the ~648 KB / 39-allocation driver leak.** `8b9282f`
   made the sanitized target CPU-only (`RUNNER_TEST_GPU_OFF=1`), so the ASan
   process never calls `cuInit` and the driver allocations cannot appear
   whatever the driver does. That is the intended design of the change, and the
   gate is green — but "clean under LSan" here means "Runner's own CPU-side
   allocation is clean", not "driver 610.43.02 no longer leaks under ASan". The
   driver-attribution claim in `8b9282f` remains unretested by this gate on any
   platform, by construction.
2. **The VRAM accounting is exercised at near-zero magnitude.** `test.gguf` is a
   370 KB synthetic model, so every figure above reads `0.0 GB` / `0.0 MB`. The
   check that VRAM returns to its starting value is real and it passed, but it
   passed on a workload that puts essentially nothing in VRAM. A leak smaller
   than the reporting resolution would not be visible.

---

## TASK 2 — Linux cross-check of `4ee0fe9`: **PASS**

```sh
PATH=/home/lab/.conda/envs/ccbuild/bin:/usr/bin:/bin CC=x86_64-conda-linux-gnu-gcc \
  make -B test
PATH=/home/lab/.conda/envs/ccbuild/bin:/usr/bin:/bin CC=x86_64-conda-linux-gnu-gcc \
  make test-ornith-cpu test-apertus
```

`make -B test` — full rebuild from source, exit 0. Tail:

```
python/tests/test_client.py ............................       [100%]
28 passed in 0.84s
.............................................................  [100%]
77 passed in 22.59s
...
python3 -m pytest -q tests/test_moe.py
.....................                                          [100%]
21 passed in 15.83s
```

The two targets:

```
python3 -m pytest -q tests/test_ornith_cpu.py
....                                                           [100%]
4 passed in 2.54s
python3 -m pytest -q tests/test_apertus.py
....                                                           [100%]
4 passed in 1.93s
```

**The two new gates ran; they did not self-skip.** This box has a CUDA backend,
so neither guard tripped. Named explicitly:

```
tests/test_ornith_cpu.py::test_qwen35_cuda_failure_after_recurrent_state_accumulates PASSED [ 50%]
tests/test_apertus.py::test_cuda_failure_falls_back_without_corrupting_output PASSED        [100%]
2 passed in 1.87s
```

The ordinal counter — the actual subject of `4ee0fe9` — was also exercised
directly against a fixture model, to confirm the pass is not vacuous:

```
$ RUNNER_CUDA_INJECT_FAILURE=2 ./runner -m ornith.gguf -p hi -n 4 --temp 0 -c 32 --gpu auto
gpu: CUDA backend on NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition MIG 1g.24gb (0.0 GB weights in VRAM)
gpu: injected CUDA runtime failure at device forward 2 — falling back to CPU

$ RUNNER_CUDA_INJECT_FAILURE=1 ...
gpu: injected CUDA runtime failure at device forward 1 — falling back to CPU

$ RUNNER_CUDA_INJECT_FAILURE=5 ...
gpu: injected CUDA runtime failure at device forward 5 — falling back to CPU
```

N=1, 2 and 5 each land on the named forward, so the count is honoured past
prefill on Linux as it was reported on Windows. N=5 reaching the graph decode
path is what the old unconditional `getenv` could not do.

### Defect found while checking for self-skip

`test_qwen35_cuda_failure_after_recurrent_state_accumulates` guards with a bare
`return`, not `pytest.skip()`:

```python
    if (caps.get("gpu") or {}).get("backend") != "cuda":
        return
```

On a machine without CUDA this test reports **PASSED, not SKIPPED** — it is
silently vacuous, which is the same failure mode `4ee0fe9` was written to
eliminate. Its Apertus counterpart does it correctly
(`pytest.skip("CUDA device not available")`). This did not affect the verdict
here, because this box has CUDA and the assertions genuinely executed, but the
gate is not trustworthy on a CPU-only CI runner. Not fixed in this pass —
reported, since changing test behaviour was outside this trip's scope.

---

## TASK 3 — write-side stall: **PASS** (both modes)

Run against `models/Qwen2.5-7B-Instruct-Q4_K_M.gguf` (4.68 GB), which is
resident on this machine. Nothing was downloaded.

```sh
python scripts/write-stall.py --runner ./runner \
  --model models/Qwen2.5-7B-Instruct-Q4_K_M.gguf --mode timeout
```

```json
{
  "client_rcvbuf": 2304,
  "control_response_bytes": 626,
  "mode": "timeout",
  "peak_send_queue_bytes": 396167,
  "timeout_release_ms": 31064.07
}
```

```sh
python scripts/write-stall.py --runner ./runner \
  --model models/Qwen2.5-7B-Instruct-Q4_K_M.gguf --mode sigpipe
```

```json
{
  "client_rcvbuf": 2304,
  "control_response_bytes": 628,
  "mode": "sigpipe",
  "peak_send_queue_bytes": 266708,
  "recovery_ms": 1337.98,
  "sigpipe_injected_under_pressure": true
}
```

Both exit 0. Real pressure was reached in both runs before either gate was
tested — 396 KB and 267 KB queued in `/proc/net/tcp` against a 2,304-byte peer
window, past the script's own 256 KB `queue_under_pressure` threshold.

On the timeout figure: 31,064 ms is a **pass**, not a marginal one. The script
asserts `send_timeout - 5 <= elapsed <= send_timeout + 20`, i.e. 25–50 s for the
default 30 s policy; the overshoot past 30 s is the 1-second polling interval of
the observing loop, not slow release by Runner. The subsequent control request
returned 200 in both modes (626 / 628 bytes), so the single slot was genuinely
released and the server stayed serving. In `sigpipe` mode Runner survived both
the injected SIGPIPE and the stalled-client RST and answered a fresh request
1.34 s later.

---

## TASK 4 — cheap re-confirmations

### `fuzz-run`: **PASS**

```sh
PATH=/home/lab/.conda/envs/ccbuild/bin:/usr/bin:/bin FUZZ_CLANG=clang FUZZ_TIME=30 \
  CC=x86_64-conda-linux-gnu-gcc make -B fuzz-run
```

All five targets (`json_parse schema_compile sval_feed jsonv_feed gguf_open`)
built with clang 22.1.8 and ran 30 s each. Exit 0. Last target's summary:

```
#361804 DONE   cov: 240 ft: 479 corp: 124/158Kb lim: 10506 exec/s: 11671 rss: 120Mb
Done 361804 runs in 31 second(s)
stat::number_of_executed_units: 361804
stat::average_exec_per_sec:     11671
stat::new_units_added:          263
stat::peak_rss_mb:              120
fuzz: all targets clean
```

No crashes, no leaks, no timeouts.

### `scripts/conformance.sh`: **PASS**, with a count that differs from the last-known figure

```sh
PATH=/home/lab/.conda/envs/ccbuild/bin:/usr/bin:/bin CC=x86_64-conda-linux-gnu-gcc \
  scripts/conformance.sh
```

```
338 passed, 2 skipped in 69.02s (0:01:09)
```

Exit 0. Last known was **323 passed, 17 skipped**; this run is **338 passed,
2 skipped**. The total is 340 either way — 15 tests that were skipped when the
earlier figure was recorded ran here and passed. That is consistent with this
being a CUDA-capable Linux box rather than the machine that produced the earlier
number; no test was lost and none regressed.

The 2 remaining skips are the AI-SDK interop cases, which need Node:

```
SKIPPED [1] tests/conformance/test_ai_sdk.py:66: node or tests/aisdk/node_modules is absent (cd tests/aisdk && npm install)
SKIPPED [1] tests/conformance/test_ai_sdk.py:73: node or tests/aisdk/node_modules is absent (cd tests/aisdk && npm install)
```

---

## Summary

| Task | Verdict |
|---|---|
| 1 — sanitizer ownership gate (ASan/LSan + VRAM) | **PASS**, no suppression edits; see the two caveats above |
| 2 — Linux cross-check of `4ee0fe9` | **PASS**; ordinals 1/2/5 confirmed; one test-quality defect reported |
| 3 — write-side stall, `timeout` and `sigpipe` | **PASS** (both) |
| 4 — `fuzz-run` | **PASS** (5 targets, 30 s each, clean) |
| 4 — `conformance.sh` | **PASS**, 338 passed / 2 skipped (was 323/17; 15 fewer skips) |

Nothing was skipped for lack of hardware or lack of a model. The only deviation
from the prescribed commands is the `ccbuild` path, explained above.
