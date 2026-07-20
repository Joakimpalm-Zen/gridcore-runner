"""Raw-wire request framing contracts for the loopback HTTP server."""

import contextlib
import socket

import pytest


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
