"""HTTP-level request cancellation on a clean client close.

The long prompt and batch size one keep prefill live long enough to observe
without a production-only delay hook.  Every model is the generated test.gguf
fixture; no network model or external service is involved.
"""

import json
import os
import socket
import sys
import time


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(HERE, "conformance"))

from _process import RunnerServer, find_runner  # noqa: E402


def _body(prompt, *, max_tokens=1):
    return json.dumps({"prompt": prompt, "max_tokens": max_tokens,
                       "temperature": 0, "cache_prompt": False},
                      separators=(",", ":")).encode()


def _open(port, body):
    s = socket.create_connection(("127.0.0.1", port), timeout=10)
    s.sendall(b"POST /v1/completions HTTP/1.1\r\n"
              b"Host: localhost\r\n"
              b"Content-Type: application/json\r\n" +
              ("Content-Length: %d\r\n\r\n" % len(body)).encode() + body)
    return s


def _read_response(s, timeout=60):
    s.settimeout(timeout)
    data = bytearray()
    while True:
        part = s.recv(65536)
        if not part:
            break
        data.extend(part)
    s.close()
    head, sep, body = bytes(data).partition(b"\r\n\r\n")
    assert sep and b" 200 " in head.split(b"\r\n", 1)[0], bytes(data[:160])
    return json.loads(body)


def _request(port, body, timeout=60):
    return _read_response(_open(port, body), timeout)


def _drain_one_request_connection(s, timeout=60):
    s.settimeout(timeout)
    try:
        while s.recv(65536):
            pass
    except ConnectionResetError:
        pass
    finally:
        s.close()


def _wait_for_starts(path, count, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with open(path, "rb") as f:
                if f.read().count(b": start,") >= count:
                    return
        except FileNotFoundError:
            pass
        time.sleep(0.01)
    raise AssertionError(f"request {count} never entered prefill")


def _wait_for_log_count(path, marker, count, timeout=60):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with open(path, "rb") as f:
            if f.read().count(marker) >= count:
                return
        time.sleep(0.01)
    raise AssertionError(f"server log never recorded {marker!r} {count} times")


def test_disconnect_stops_prefill_without_false_cancellation(tmp_path):
    model = os.path.join(ROOT, "test.gguf")
    assert os.path.exists(model), "build the generated fixture with make test.gguf"
    log = tmp_path / "disconnect-server.log"
    # Byte-fallback tokenization keeps this below 32k context while making a
    # batch-one prefill much longer than one cancellation poll on CI CPUs.
    long_body = _body("x " * 8000)
    short_body = _body("slot reuse", max_tokens=4)

    with RunnerServer(find_runner(ROOT), model, ctx=32768, parallel=1,
                      extra_args=["--gpu", "off", "-b", "1"],
                      log_path=str(log)) as srv:
        reference = _request(srv.port, short_body)

        # Alive and quiet is the common prefill state: delay reads until the
        # server has spent time polling an otherwise empty receive buffer.
        slow = _open(srv.port, long_body)
        _wait_for_starts(log, 2)
        full_t0 = time.monotonic()
        time.sleep(0.1)
        slow_result = _read_response(slow)
        full_s = time.monotonic() - full_t0
        assert slow_result["choices"], "a slow reader was falsely cancelled"
        assert full_s > 0.2, "the long-prefill control was too short to gate cancellation"

        # Bytes becoming readable are allowed to be the next HTTP request.
        # Send them only after prefill starts, so they stay in the kernel receive
        # buffer for the probe rather than being absorbed by initial parsing.
        piped = _open(srv.port, long_body)
        _wait_for_starts(log, 3)
        piped.sendall(b"GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n")
        # Runner serves one request per connection. Closing after the first
        # response with the deliberately unread second request may send RST and
        # discard that response, so the lifecycle log is the honest oracle:
        # two cmpl-2 lines mean start + normal completion. A false cancellation
        # during prefill produces only the start line.
        _drain_one_request_connection(piped)
        _wait_for_log_count(log, b"] cmpl-2:", 2)

        # A clean close must release the only slot within one batch-one chunk.
        abandoned = _open(srv.port, long_body)
        _wait_for_starts(log, 4)
        abandoned.close()
        cancel_t0 = time.monotonic()
        reused = _request(srv.port, short_body)
        cancel_s = time.monotonic() - cancel_t0

        assert reused["choices"][0]["text"] == reference["choices"][0]["text"], \
            "the slot reused after cancellation diverged from its cold output"
        assert cancel_s * 2 < full_s, \
            f"dead prefill was not cut promptly: reuse {cancel_s:.3f}s, full {full_s:.3f}s"
