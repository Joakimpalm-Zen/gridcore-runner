#!/usr/bin/env python3
"""Reproduce the startup/SIGTERM race that test_signal_during_startup chases.

The bug (reported once, 2026-08-02, never reproduced since): a SIGTERM landing
while models are still loading could leave a half-started server running
forever, because the stop flag was only checked after accept() failed. The
conformance test sweeps a fixed 12-delay ladder once per suite run; a
~0.17%/spawn event needs far more spawns than that, and it needs the race
window held OPEN — the window is between the signal handler installing
(server.c:1199) and the load loop's stop checks, so CPU pressure and many
concurrent spawns widen it. Inspection is exhausted (startup path audited
2026-08-15, no hole found); this is the instrument the plan asks for instead.

    scripts/repro-startup-signal.py --iterations 5000 --concurrency 8 --load

A survivor is a spawn that ignored the SIGTERM and had to be SIGKILLed after
--timeout seconds. Every survivor's captured startup output and the exact
post-spawn delay are printed and saved, so the next occurrence says WHERE
startup was — which the single 2026-08-02 sighting never did. Exit is 1 if any
survivor is seen, 0 otherwise, so this can gate a soak run in CI.
"""
import argparse
import os
import random
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor


def free_port():
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def one_spawn(exe, model, delay, timeout):
    """Spawn a server, SIGTERM it `delay` s later, return a survivor record or None."""
    proc = subprocess.Popen(
        [exe, "-m", model, "--serve", "--port", str(free_port()),
         "--parallel", "2", "-c", "1024", "--gpu", "off"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        time.sleep(delay)
        proc.send_signal(signal.SIGTERM)
        try:
            rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            out = proc.stdout.read().decode("utf-8", "replace")
            return {"delay_ms": delay * 1000,
                    "output": out or "(nothing printed before it hung)"}
        # rc in (0, -SIGTERM) is the intended outcome; any other code is not a
        # survivor but is worth surfacing.
        if rc not in (0, -signal.SIGTERM):
            out = proc.stdout.read().decode("utf-8", "replace")
            return {"delay_ms": delay * 1000, "unexpected_rc": rc,
                    "output": out}
        return None
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        proc.stdout.close()


def burn_cpu(stop):
    """Hold CPU pressure so the load window between handler-install and the
    stop checks stays open longer."""
    x = 0.0
    while not stop.is_set():
        x = (x + 1.234) * 1.0000001
        if x > 1e9:
            x = 0.0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=2000)
    ap.add_argument("--concurrency", type=int, default=4,
                    help="concurrent spawns; >1 is the 'under load' the plan asks for")
    ap.add_argument("--timeout", type=float, default=10.0,
                    help="seconds to wait for a signalled server to exit")
    ap.add_argument("--min-delay-ms", type=float, default=0.1)
    ap.add_argument("--max-delay-ms", type=float, default=5.0)
    ap.add_argument("--load", action="store_true",
                    help="also run CPU-burn threads to widen the race window")
    ap.add_argument("--seed", type=int, default=1,
                    help="delay-sampling seed (fixed so a survivor is reproducible)")
    ap.add_argument("--runner")
    ap.add_argument("--model")
    args = ap.parse_args(argv)

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    exe = args.runner or os.path.join(root, "runner")
    model = args.model or os.environ.get(
        "RUNNER_TEST_MODEL", os.path.join(root, "test.gguf"))
    if not os.path.exists(exe):
        sys.exit(f"runner not found at {exe} (build it, or pass --runner)")
    if not os.path.exists(model):
        sys.exit(f"model not found at {model} (pass --model)")

    rng = random.Random(args.seed)
    delays = [rng.uniform(args.min_delay_ms, args.max_delay_ms) / 1000.0
              for _ in range(args.iterations)]

    burners = []
    stop = threading.Event()
    if args.load:
        for _ in range(max(1, (os.cpu_count() or 2) // 2)):
            t = threading.Thread(target=burn_cpu, args=(stop,), daemon=True)
            t.start()
            burners.append(t)

    survivors = []
    done = 0
    t0 = time.time()
    print(f"spawning {args.iterations} servers, concurrency {args.concurrency}, "
          f"load={'on' if args.load else 'off'}, timeout {args.timeout}s",
          flush=True)
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = [pool.submit(one_spawn, exe, model, d, args.timeout) for d in delays]
        for fut in futs:
            rec = fut.result()
            done += 1
            if rec is not None:
                survivors.append(rec)
                tag = "SURVIVOR" if "unexpected_rc" not in rec else \
                    f"rc={rec['unexpected_rc']}"
                print(f"\n[{tag}] SIGTERM {rec['delay_ms']:.2f}ms after spawn; "
                      f"startup output:\n{rec['output']}\n", flush=True)
            if done % 200 == 0:
                rate = done / max(1e-9, time.time() - t0)
                print(f"  {done}/{args.iterations}  survivors={len(survivors)}  "
                      f"({rate:.0f}/s)", flush=True)
    stop.set()

    n = args.iterations
    print(f"\n{n} spawns, {len(survivors)} survivor(s) "
          f"in {time.time() - t0:.1f}s")
    if survivors:
        # A single survivor establishes the bug is live; a rate needs many more.
        print(f"observed survivor rate: {len(survivors)}/{n} = "
              f"{100.0 * len(survivors) / n:.3f}% (lower bound — one is enough)")
        return 1
    print("no survivors — the stop flag held on every spawn in this run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
