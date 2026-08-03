# Real write-side stall experiment

`tests/conformance/test_write_side.py` cannot fill the loopback socket buffers:
its tiny model emits at most about 68 KB of SSE. `scripts/write-stall.py` is the
local, Linux-only real-model gate for the two paths beyond that fixture.

The experiment starts Qwen2.5-7B with a 32,768-token context, constrains a JSON
string to at least 100,000 bytes, enables streaming logprobs, and connects a
client whose requested receive buffer is 1 KiB (Linux reports 2,304 bytes after
its minimum/doubling rules). The client never reads. `/proc/net/tcp` identifies
the exact connection by both ports and measures the server transmit queue, so
the gate cannot pass merely because generation was slow or the draft request
flags were ignored.

Run the production gates:

```bash
python3 scripts/write-stall.py --model models/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
    --mode timeout
python3 scripts/write-stall.py --model models/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
    --mode sigpipe
```

Measured 2026-08-03 on the Blackwell 1g.24gb box:

| gate | peak queued | result |
|---|---:|---|
| 30 s write timeout | 396,365 B | socket and slot released in 31.062 s; next request 200 |
| ignored SIGPIPE | 267,300 B | signal survived; RST recovery and next request in 1.339 s |

The measurement found two real defects. First, server slot engines silently
dropped the process-level `--ignore-eos` setting, so the initial “large” run
stopped after 44 tokens; the EOS-only integration fixture now proves a serving
slot reaches its requested token limit. Second, `SO_SNDTIMEO` alone did not
release Linux's zero-window connection after more than 50 seconds. Runner now
sets Linux `TCP_USER_TIMEOUT` to the same 30-second policy while retaining
`SO_SNDTIMEO`; the same experiment releases at 31.062 seconds.

## Negative controls

The Makefile exposes local-only knob-blind builds. They reuse the checked-in
PTX header and are not release artifacts:

```bash
make runner-no-write-timeout runner-sigpipe-default
python3 scripts/write-stall.py --runner ./runner-no-write-timeout \
    --model models/Qwen2.5-7B-Instruct-Q4_K_M.gguf --mode timeout \
    --stall-deadline 45                 # must FAIL
python3 scripts/write-stall.py --runner ./runner-sigpipe-default \
    --model models/Qwen2.5-7B-Instruct-Q4_K_M.gguf --mode sigpipe \
    --stall-deadline 60                 # must FAIL with signal 13
```

The timeout-blind build remained wedged with 570,154 bytes queued. The
SIGPIPE-default build died with return code -13. On this kernel, both an RST
and an orderly-close attempt delivered `ECONNRESET` to Runner's first failed
write rather than an `EPIPE`, and Runner correctly made no second write; those
closures therefore cannot deterministically raise SIGPIPE. The gate injects
SIGPIPE into the real pressured process after the queue threshold is proved.
That tests the exact installed disposition without falsely claiming the kernel
delivered a signal it did not.
