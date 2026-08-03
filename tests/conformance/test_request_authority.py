"""Browser-origin and DNS-rebinding boundary for the loopback API."""

import socket


def _raw(server, path, headers):
    with socket.create_connection(("127.0.0.1", server.port), timeout=5) as sock:
        request = f"GET {path} HTTP/1.1\r\n{headers}\r\n\r\n".encode()
        sock.sendall(request)
        data = b""
        while True:
            try:
                chunk = sock.recv(4096)
            except ConnectionResetError:
                break
            if not chunk:
                break
            data += chunk
    return data.split(b"\r\n", 1)[0]


def test_foreign_host_and_origin_are_rejected_on_fast_and_slot_paths(server):
    assert _raw(server, "/health", "Host: evil.example") == \
        b"HTTP/1.1 403 Forbidden"
    assert _raw(server, "/not-fast", "Host: evil.example") == \
        b"HTTP/1.1 403 Forbidden"
    assert _raw(server, "/v1/models", "Host: localhost\r\n"
                "Origin: https://evil.example") == b"HTTP/1.1 403 Forbidden"


def test_loopback_host_and_origin_remain_accepted(server):
    assert _raw(server, "/health", "Host: localhost") == b"HTTP/1.1 200 OK"
    assert _raw(server, "/health", f"Host: 127.0.0.1:{server.port}\r\n"
                "Origin: http://localhost:3000") == b"HTTP/1.1 200 OK"
