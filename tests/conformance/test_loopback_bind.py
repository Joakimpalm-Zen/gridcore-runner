"""The loopback-only bind, proved against a running server.

tests/test_bind.c locks the source: one literal INADDR_LOOPBACK, no option and
no environment variable that reaches it. This file is the other half — it asks
the running process the only question that actually matters:

    can anything outside this machine open a socket to it?

The answer must be no on every non-loopback address the host owns, while
127.0.0.1 keeps working. That is what makes it safe for the API to have no
authentication, and what makes the permissive request framing (unanchored
Content-Length, ignored Transfer-Encoding) a latent issue rather than a live
one: neither a missing auth check nor a smuggling primitive is reachable from
off-box.

Context for why this is worth a test rather than a comment: SentinelLabs and
Censys found ~175,000 exposed Ollama instances in January 2026, no auth, no
firewall, nearly half with tool calling enabled. Ollama's default bind is
127.0.0.1, same as runner's and same as llama-server's. The difference is that
setting OLLAMA_HOST=0.0.0.0 is one line, and runner has no such line to write.
"""

import contextlib
import os
import socket
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from harness import find_runner  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

# Options that would, if they existed, choose a bind address.
ADDRESS_OPTIONS = ["--host", "--bind", "--listen", "--address", "--interface", "--ip"]


def local_non_loopback_addresses():
    """Every IPv4 address this host answers on that is not 127.0.0.0/8.

    Two sources, because neither is complete on its own: the hostname's A
    records, and the address the kernel would pick to reach the outside world
    (a connected UDP socket sends nothing, it just does the route lookup).
    """
    found = set()
    with contextlib.suppress(OSError):
        _, _, addrs = socket.gethostbyname_ex(socket.gethostname())
        found.update(addrs)
    with contextlib.suppress(OSError):
        with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_DGRAM)) as s:
            s.connect(("192.0.2.1", 9))  # TEST-NET-1, never routed
            found.add(s.getsockname()[0])
    return sorted(a for a in found if not a.startswith("127.") and a != "0.0.0.0")


def test_loopback_is_reachable(server):
    """The control: the socket really is up on 127.0.0.1, so a refusal on any
    other address below means loopback-only and not merely 'server is down'."""
    with contextlib.closing(socket.create_connection(("127.0.0.1", server.port),
                                                     timeout=5)) as s:
        s.sendall(b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                  b"Connection: close\r\n\r\n")
        assert b" 200 " in s.recv(64)


def test_not_reachable_on_any_non_loopback_address(server):
    """The gate. If this fails, the server is listening off-box: an
    unauthenticated, tool-calling inference API on the network."""
    addrs = local_non_loopback_addresses()
    if not addrs:
        pytest.skip("host has no non-loopback IPv4 address, so the bind cannot "
                    "be probed from one; tests/test_bind.c still locks the source")

    reachable = []
    for addr in addrs:
        try:
            with contextlib.closing(socket.create_connection((addr, server.port),
                                                             timeout=3)):
                reachable.append(addr)
        except OSError:
            pass  # refused, unreachable or timed out — all of them mean "not listening"

    assert not reachable, (
        f"runner accepted a connection on {reachable} (port {server.port}). "
        f"The server must bind 127.0.0.1 only: the API has no authentication and "
        f"the HTTP framing is not hardened against smuggling, and both of those "
        f"are only acceptable because nothing off-box can reach the socket.")


@pytest.mark.parametrize("opt", ADDRESS_OPTIONS)
def test_no_option_selects_a_bind_address(opt):
    """There is no flag to get wrong. Rejected at the CLI, not just undocumented."""
    exe = find_runner(ROOT)
    p = subprocess.run([exe, opt, "0.0.0.0", "--caps"],
                       capture_output=True, text=True, timeout=60)
    assert p.returncode != 0, f"runner accepted {opt}"
    assert "unknown option" in (p.stderr + p.stdout), \
        f"runner did not reject {opt} as an unknown option"


def test_help_advertises_no_bind_address():
    exe = find_runner(ROOT)
    p = subprocess.run([exe, "--help"], capture_output=True, text=True, timeout=60)
    help_text = p.stderr + p.stdout
    assert "--port" in help_text, "sanity: did not capture the usage text"
    for opt in ADDRESS_OPTIONS:
        assert opt not in help_text, f"runner advertises {opt}"
