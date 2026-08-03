#!/usr/bin/env python3
"""Reproduce Runner's real write-side socket stall with a large model/context.

Unlike the tiny conformance fixture, this emits enough SSE to fill the actual
loopback TCP buffers while a 1 KiB-window client never reads. Linux /proc TCP
queue measurements prove hundreds of KiB remain queued against that window
before either gate is tested. `timeout` requires the socket and its one slot to
be released in the 30-second policy window;
`sigpipe` resets the stalled connection and requires Runner to survive.

This is intentionally a local, real-model experiment, not a CI test.
"""

import argparse
import json
import os
from pathlib import Path
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def server_tx_queue(tcp_path, server_port, client_port):
    local = f"0100007F:{server_port:04X}"
    remote = f"0100007F:{client_port:04X}"
    for line in Path(tcp_path).read_text().splitlines()[1:]:
        fields = line.split()
        if len(fields) >= 5 and fields[1] == local and fields[2] == remote:
            return int(fields[4].split(":", 1)[0], 16)
    return None


def queue_under_pressure(queued):
    return queued >= 256 * 1024


def request(port, payload, *, receive=True, rcvbuf=None, timeout=60):
    sock = socket.socket()
    if rcvbuf:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, rcvbuf)
    sock.settimeout(timeout)
    sock.connect(("127.0.0.1", port))
    body = json.dumps(payload).encode()
    sock.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\n"
                 b"Content-Type: application/json\r\nConnection: close\r\n"
                 b"Content-Length: %d\r\n\r\n" % len(body) + body)
    if not receive:
        return sock
    chunks = []
    while True:
        chunk = sock.recv(65536)
        if not chunk:
            break
        chunks.append(chunk)
    sock.close()
    raw = b"".join(chunks)
    if not raw.startswith(b"HTTP/1.1 200"):
        raise RuntimeError(f"control request failed: {raw[:160]!r}")
    return raw


def wait_healthy(proc, port, deadline):
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"runner exited during startup ({proc.returncode})")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1) as sock:
                sock.sendall(b"GET /health HTTP/1.1\r\nHost: x\r\n"
                             b"Connection: close\r\n\r\n")
                if b" 200 " in sock.recv(64):
                    return
        except OSError:
            pass
        time.sleep(0.1)
    raise RuntimeError("runner did not become healthy")


def reset(sock):
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER,
                    struct.pack("ii", 1, 0))
    sock.close()


def run_experiment(args):
    port = free_port()
    with tempfile.TemporaryDirectory(prefix="runner-write-stall-") as tmp:
        log_path = Path(tmp) / "runner.log"
        with log_path.open("wb") as log:
            proc = subprocess.Popen([
                str(args.runner), "-m", str(args.model), "--serve",
                "--port", str(port), "--parallel", "1", "-c", str(args.ctx),
                "--ignore-eos", "--gpu", "auto"], stdout=log,
                stderr=subprocess.STDOUT)
            dead = None
            try:
                wait_healthy(proc, port, time.monotonic() + 120)
                payload = {
                    "messages": [{"role": "user",
                                  "content": "Fill the payload string."}],
                    "max_tokens": args.max_tokens, "temperature": 0,
                    "stream": True, "cache_prompt": False,
                    "logprobs": True, "top_logprobs": 20,
                    # Ignoring EOS alone is insufficient: an instruct model
                    # can sample EOS forever, producing tokens but no bytes.
                    # The minimum-length grammar keeps every sampled token in
                    # a nonempty JSON string until well past the TCP buffers.
                    "response_format": {"type": "json_schema",
                        "json_schema": {"name": "large_payload", "schema": {
                            "type": "object", "properties": {
                                "payload": {"type": "string",
                                            "minLength": 100000}},
                            "required": ["payload"],
                            "additionalProperties": False}}},
                }
                dead = request(port, payload, receive=False, rcvbuf=1024,
                               timeout=args.stall_deadline)
                client_port = dead.getsockname()[1]
                peak_queue = 0
                request_at = time.monotonic()
                timed_out_at = None
                deadline = time.monotonic() + args.stall_deadline
                while time.monotonic() < deadline:
                    queued = server_tx_queue("/proc/net/tcp", port, client_port)
                    if queued is not None:
                        peak_queue = max(peak_queue, queued)
                        # For SIGPIPE, reset while a response vastly larger
                        # than the peer's 2.3 KiB window is still in flight.
                        if args.mode == "sigpipe" and queue_under_pressure(queued):
                            break
                    elif args.mode == "timeout" and queue_under_pressure(peak_queue):
                        # TCP_USER_TIMEOUT closes the server half; /proc drops
                        # that exact 4-tuple before the non-reading client has
                        # drained its own receive queue.
                        timed_out_at = time.monotonic()
                        break
                    if proc.poll() is not None:
                        raise RuntimeError("runner died before the socket stalled")
                    time.sleep(1)
                else:
                    raise RuntimeError(
                        f"send queue did not reach/release under pressure; "
                        f"peak={peak_queue} bytes")

                stalled_at = time.monotonic()
                measurement = {"mode": args.mode,
                               "peak_send_queue_bytes": peak_queue,
                               "client_rcvbuf": dead.getsockopt(
                                   socket.SOL_SOCKET, socket.SO_RCVBUF)}
                if args.mode == "sigpipe":
                    # Linux returned ECONNRESET (not EPIPE) for both RST and
                    # orderly-close attempts on this pressured TCP stream, so
                    # neither deterministically delivers SIGPIPE. Inject the
                    # exact signal only after proving real write pressure: the
                    # production disposition must survive, while the SIG_DFL
                    # negative-control build must die here.
                    os.kill(proc.pid, signal.SIGPIPE)
                    time.sleep(0.2)
                    if proc.poll() is not None:
                        raise RuntimeError(
                            f"runner died on SIGPIPE ({proc.returncode})")
                    reset(dead)
                    dead = None
                    time.sleep(1)
                    if proc.poll() is not None:
                        raise RuntimeError(
                            f"runner died after stalled-client RST ({proc.returncode})")
                    raw = request(port, {"messages": [{"role": "user",
                                                        "content": "still alive"}],
                                         "max_tokens": 1, "temperature": 0},
                                  timeout=30)
                    measurement["recovery_ms"] = round(
                        (time.monotonic() - stalled_at) * 1000, 2)
                    measurement["control_response_bytes"] = len(raw)
                    measurement["sigpipe_injected_under_pressure"] = True
                else:
                    elapsed = timed_out_at - request_at
                    if not (args.send_timeout - 5 <= elapsed <=
                            args.send_timeout + 20):
                        raise RuntimeError(
                            f"socket released outside the send-timeout window "
                            f"({elapsed:.2f}s)")
                    raw = request(
                        port, {"messages": [{"role": "user",
                                             "content": "slot released"}],
                               "max_tokens": 1, "temperature": 0}, timeout=20)
                    measurement["timeout_release_ms"] = round(elapsed * 1000, 2)
                    measurement["control_response_bytes"] = len(raw)
                return measurement
            except Exception as exc:
                log.flush()
                tail = log_path.read_text(errors="replace")[-2000:]
                raise RuntimeError(f"{exc}\nrunner log tail:\n{tail}") from exc
            finally:
                if dead is not None:
                    reset(dead)
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, default=ROOT / "runner")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--mode", choices=("timeout", "sigpipe"), required=True)
    parser.add_argument("--ctx", type=int, default=32768)
    parser.add_argument("--max-tokens", type=int, default=16000)
    parser.add_argument("--stall-deadline", type=int, default=360)
    parser.add_argument("--send-timeout", type=int, default=30)
    args = parser.parse_args(argv)
    if sys.platform != "linux":
        parser.error("this /proc TCP experiment is Linux-only")
    for path in (args.runner, args.model):
        if not path.is_file():
            parser.error(f"not found: {path}")
    args.runner = args.runner.resolve()
    args.model = args.model.resolve()
    try:
        measurement = run_experiment(args)
    except RuntimeError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(measurement, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
