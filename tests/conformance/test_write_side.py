"""What happens to the server when the client stops holding up its end.

Every other test in this suite is a well-behaved client: it sends a request and
reads the answer. The write side is where a server gets taken down by traffic
that is not malicious, just interrupted — a browser tab closed mid-stream, a
proxy that times out, an agent that cancels. The failure modes are a fatal
SIGPIPE, a slot that is never returned, and a partially-written response that
corrupts the next one on a reused connection.

**The stall proper is not reachable from this harness, and neither is SIGPIPE.**
Both need the server's `send()` to actually block or fail, which needs a
response larger than the socket buffers between the two ends. The suite's model
is capped by `n_ctx` at ~68 KB of SSE, and on loopback that fits even with the
client's `SO_RCVBUF` pinned to 1 KB — measured: the entire response is buffered
and delivered to a client that never calls `recv()` once. A SIG_DFL build was
also constructed and run against these tests; all of them still pass, because
the write has already completed by the time the peer resets. Producing either
case needs a real model and a large context, i.e. a `scripts/`-sized experiment
rather than a conformance test. `scripts/write-stall.py` is that local Linux
gate; this fast portable suite deliberately remains the interrupted-client
half instead of duplicating or weakening it.

What these four do cover is the reachable and still-real half: an interrupted
client does not kill the process, strand its slot, or corrupt what the server
sends next.
"""

import json
import socket
import struct
import time

from _errors import ProtocolError

STREAM = {"model": "test", "messages": [{"role": "user", "content": "hello"}],
          "max_tokens": 400, "temperature": 0, "stream": True,
          "cache_prompt": False}
SHORT = {"model": "test", "messages": [{"role": "user", "content": "hi"}],
         "max_tokens": 8, "temperature": 0, "cache_prompt": False}


def _connect(server, payload, rcvbuf=None):
    s = socket.socket()
    if rcvbuf is not None:
        # Pin the receive window before connect, so it is advertised in the
        # handshake rather than applied after the buffers are already sized.
        s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, rcvbuf)
    s.connect(("127.0.0.1", server.port))
    body = json.dumps(payload).encode()
    s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\n"
              b"Content-Type: application/json\r\n"
              b"Content-Length: %d\r\n\r\n" % len(body) + body)
    return s


def _rst(s):
    """Close with SO_LINGER 0: an RST, not a FIN.

    A FIN is an orderly half-close the server may not even notice until it
    writes. An RST makes the next write fail hard, which is the case that used
    to kill a process on SIGPIPE.
    """
    s.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    s.close()


def _works(client, name):
    r = client.chat(SHORT, name=name)
    r.expect_status(200)
    return r


def test_a_client_that_vanishes_mid_stream_does_not_take_the_server_with_it(
        server, client):
    """The RST arrives while the server is still writing SSE frames.

    **This does not gate `signal(SIGPIPE, SIG_IGN)`**, which is what it looks
    like it should do and what a first version of this docstring claimed. A
    build with the handler left at SIG_DFL was constructed and run against
    these four tests: all four still pass. The reason is the same measurement
    that defeats the stall case — the whole ~68 KB response fits in the socket
    buffers, so the server's `send()` has already returned before the RST
    arrives and there is no broken pipe to signal on. Gating SIGPIPE needs a
    response bigger than the buffers, which this model cannot produce.

    What it does gate is the ordinary shape of the failure: the server is still
    alive afterwards and did not lose the slot.
    """
    s = _connect(server, STREAM)
    s.settimeout(30)
    first = s.recv(256)          # streaming has actually started
    if not first.startswith(b"HTTP/1.1 200"):
        raise ProtocolError("the stream never started", got=first[:120])
    _rst(s)

    # The process must still be there, and still serving. Two requests, because
    # one proves the process is alive and two prove neither slot was lost.
    _works(client, "write-side-after-rst-1")
    _works(client, "write-side-after-rst-2")


def test_a_client_that_vanishes_before_reading_anything(server, client):
    """The same, but the RST lands during the prompt rather than the stream —
    the server has produced nothing yet and must not treat an empty write as a
    success or wedge waiting for a reader."""
    s = _connect(server, STREAM)
    _rst(s)
    _works(client, "write-side-after-early-rst-1")
    _works(client, "write-side-after-early-rst-2")


def test_many_vanishing_clients_do_not_leak_slots(server, client):
    """A slot leaked per abandoned connection would exhaust a 2-slot server in
    two iterations; ten makes the check unambiguous, and the final request
    would hang rather than fail if a slot were still held."""
    for i in range(10):
        s = _connect(server, STREAM)
        try:
            s.settimeout(10)
            s.recv(64)
        except (socket.timeout, OSError):
            pass
        _rst(s)
    _works(client, "write-side-after-ten-rst")


def test_a_reader_that_never_reads_does_not_stall_the_other_slot(server, client):
    """A dead reader holds its own slot; it must not hold anyone else's.

    See the module docstring for why the *stall* itself cannot be produced
    here. What is asserted is the part that is reachable and is also the part
    that matters in practice: whatever the dead reader is doing, the other slot
    keeps serving.
    """
    dead = _connect(server, STREAM, rcvbuf=1024)
    try:
        time.sleep(0.5)
        for i in range(3):
            t0 = time.time()
            _works(client, f"write-side-during-dead-reader-{i}")
            elapsed = time.time() - t0
            # Generous by two orders of magnitude: these return in ~0.02 s.
            # The failure this catches is serialization, not slowness.
            if elapsed > 20.0:
                raise ProtocolError(
                    "a request was stalled behind a client that stopped reading",
                    seconds=round(elapsed, 2), iteration=i)
    finally:
        _rst(dead)
    _works(client, "write-side-after-dead-reader")
