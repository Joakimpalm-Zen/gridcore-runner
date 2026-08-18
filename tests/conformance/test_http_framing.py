"""Raw-wire request framing contracts for the loopback HTTP server."""

import contextlib
import os
import socket
import time

import pytest

from harness import RunnerServer, find_runner

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def raw_request(server, request):
    with contextlib.closing(socket.create_connection(
            ("127.0.0.1", server.port), timeout=5)) as sock:
        sock.settimeout(5)
        sock.sendall(request)
        response = bytearray()
        while True:
            try:
                part = sock.recv(65536)
            except ConnectionResetError:
                if response:
                    return bytes(response)
                raise
            if not part:
                return bytes(response)
            response += part


def status(response):
    return int(response.split(b" ", 2)[1])


def status_line(response):
    return response.split(b"\r\n", 1)[0]


def chat_request():
    body = b'{"messages":[{"role":"user","content":"hi"}],"max_tokens":1}'
    return (b"POST /v1/chat/completions HTTP/1.1\r\n"
            b"Host: localhost\r\nContent-Type: application/json\r\n" +
            (b"Content-Length: %d\r\n\r\n" % len(body)) + body)


@pytest.mark.parametrize("line", [
    b"BROKEN",
    b"GET /health",
    b"GET /health HTTP/1.1 extra",
    b"GET /health HTTP/1.1 ",
    b"GET\t/health\tHTTP/1.1",
])
def test_malformed_request_line_is_bad_request(server, line):
    response = raw_request(server, line + b"\r\nHost: localhost\r\n\r\n")
    assert status(response) == 400


def framed_request(headers=b""):
    return b"POST /not-found HTTP/1.1\r\nHost: localhost\r\n" + headers + b"\r\n"


def test_content_length_name_is_line_anchored_and_exact(server):
    response = raw_request(server, framed_request(
        b"X-Content-Length: not-a-number\r\n"))
    assert status(response) == 404


def test_content_length_allows_ows_around_decimal(server):
    response = raw_request(server, framed_request(
        b"Content-Length:\t 0 \t\r\n"))
    assert status(response) == 404


def test_content_length_name_is_case_insensitive(server):
    response = raw_request(server, framed_request(b"cOnTeNt-LeNgTh: 0\r\n"))
    assert status(response) == 404


@pytest.mark.parametrize("value", [b"", b"+0", b"-0", b"0x0", b"0 junk",
                                   b"18446744073709551616"])
def test_invalid_content_length_is_bad_request(server, value):
    response = raw_request(server, framed_request(
        b"Content-Length: " + value + b"\r\n"))
    assert status(response) == 400


@pytest.mark.parametrize("headers", [
    b"Content-Length: 0\r\nContent-Length: 0\r\n",
    b"Content-Length: 0\r\nContent-Length: 1\r\n",
])
def test_duplicate_content_length_is_bad_request(server, headers):
    assert status(raw_request(server, framed_request(headers))) == 400


@pytest.mark.parametrize("value", [b"chunked", b"identity", b"gzip"])
def test_any_transfer_encoding_is_bad_request(server, value):
    response = raw_request(server, framed_request(
        b"Transfer-Encoding: " + value + b"\r\n"))
    assert status(response) == 400


def test_fastpath_also_rejects_invalid_framing(server):
    request = (b"GET /health HTTP/1.1\r\nHost: localhost\r\n"
               b"Transfer-Encoding: chunked\r\n\r\n")
    assert status(raw_request(server, request)) == 400


@pytest.mark.parametrize("request_bytes", [
    b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n",
    b"GET /v1/models HTTP/1.1\r\nHost: [::1]\r\n\r\n",
    b"GET /v1/capabilities HTTP/1.1\r\nHost: localhost\r\n\r\n",
])
def test_compact_fastpath_request_closes_cleanly(server, request_bytes):
    """A whole request that fits inside the accept loop's peek must still be
    CONSUMED before the socket is closed.

    The accept fastpath peeks the first bytes to route, then drains the header
    with real reads. A peek leaves the bytes in the receive buffer, so when the
    peek alone already contains the blank line the drain loop never runs and the
    socket is closed with the request still unread — which sends RST instead of
    FIN. The reply is then discarded by the client's stack: the caller sees
    ECONNRESET having read nothing at all. The sleep is what makes it
    deterministic rather than a race the client sometimes wins.
    """
    with contextlib.closing(socket.create_connection(
            ("127.0.0.1", server.port), timeout=5)) as sock:
        sock.settimeout(5)
        sock.sendall(request_bytes)
        time.sleep(0.2)
        response = bytearray()
        while True:
            part = sock.recv(65536)  # a reset propagates and fails the test
            if not part:
                break
            response += part
    assert status(bytes(response)) == 200


def test_503_status_line_uses_service_unavailable_reason():
    model = os.environ.get("RUNNER_TEST_MODEL", os.path.join(ROOT, "test.gguf"))
    old_queue = os.environ.get("RUNNER_MAX_QUEUE")
    os.environ["RUNNER_MAX_QUEUE"] = "1"
    stalled = queued = None
    try:
        with RunnerServer(find_runner(ROOT), model, ctx=1024, parallel=1,
                          extra_args=["--gpu", "off"]) as srv:
            stalled = socket.create_connection(("127.0.0.1", srv.port), timeout=5)
            stalled.sendall(b"POST /v1/chat/completions HTTP/1.1\r\n")

            queued = socket.create_connection(("127.0.0.1", srv.port), timeout=5)
            queued.sendall(chat_request())
            time.sleep(0.2)

            response = raw_request(srv, chat_request())
            assert status(response) == 503
            assert status_line(response) == b"HTTP/1.1 503 Service Unavailable"
    finally:
        if stalled is not None:
            stalled.close()
        if queued is not None:
            queued.close()
        if old_queue is None:
            os.environ.pop("RUNNER_MAX_QUEUE", None)
        else:
            os.environ["RUNNER_MAX_QUEUE"] = old_queue
