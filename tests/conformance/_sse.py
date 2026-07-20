"""Incremental Server-Sent Events parser.

Exists to be fed at arbitrary byte boundaries: the Phase 0 requirement is that
splitting the stream anywhere produces an identical parse. Keeping the parser
purely incremental (no "read a line from a socket") is what makes that testable
at all, so this is deliberately a byte-buffer state machine and nothing else.

Runner frames every event as ``data: <json>\\n\\n``. This parser is a little
more permissive than that (\\r\\n line ends, comment lines, multi-line data)
because a conformance harness should not encode a stricter reading of the wire
than a real SSE client would apply.
"""

import json

from _errors import ProtocolError

DONE = "[DONE]"


class SSEParser:
    """Feed bytes, get events. Events are the raw ``data:`` payload strings."""

    def __init__(self):
        self._buf = b""
        self._events = []
        self._names = []
        self._closed = False

    def feed(self, chunk):
        if self._closed:
            raise ProtocolError("SSEParser.feed after close")
        self._buf += chunk
        while True:
            cut, sep = self._find_dispatch()
            if cut < 0:
                return
            block = self._buf[:cut]
            self._buf = self._buf[cut + sep:]
            self._dispatch(block)

    def close(self):
        """Signal EOF. A trailing unterminated block is *not* dispatched:
        a half-received event must never look like a complete one."""
        self._closed = True
        self.trailing = self._buf
        return self._events

    @property
    def events(self):
        return list(self._events)

    @property
    def names(self):
        """The ``event:`` field of each dispatched event (None when absent)."""
        return list(self._names)

    # ------------------------------------------------------------------
    def _find_dispatch(self):
        # an event ends at a blank line: \n\n or \r\n\r\n (or the mixed forms)
        best, seplen = -1, 0
        for sep in (b"\r\n\r\n", b"\n\n", b"\r\r"):
            i = self._buf.find(sep)
            if i >= 0 and (best < 0 or i < best):
                best, seplen = i, len(sep)
        return best, seplen

    def _dispatch(self, block):
        data = []
        name = None
        for raw in block.replace(b"\r\n", b"\n").split(b"\n"):
            if not raw or raw.startswith(b":"):
                continue  # blank or comment
            field, _, value = raw.partition(b":")
            if value.startswith(b" "):
                value = value[1:]
            if field == b"data":
                data.append(value)
            elif field == b"event":
                name = value.decode("utf-8", "replace")
        if data:
            self._events.append(b"\n".join(data).decode("utf-8", "replace"))
            self._names.append(name)


def parse_stream(raw, chunks=None):
    """Parse a complete SSE byte string, optionally in explicit chunks."""
    p = SSEParser()
    for c in chunks if chunks is not None else [raw]:
        p.feed(c)
    p.close()
    return p.events


def parse_named_stream(raw, chunks=None):
    """Like ``parse_stream`` but returns ``(event_name, data)`` pairs.

    The Responses API names every event twice — once in the SSE ``event:``
    field and once as ``data.type`` — and typed SDK clients dispatch on the
    former. Keeping both lets a test assert they agree."""
    p = SSEParser()
    for c in chunks if chunks is not None else [raw]:
        p.feed(c)
    p.close()
    return list(zip(p.names, p.events))


def split_points(raw):
    """Every possible single split of ``raw`` into two chunks, plus the two
    degenerate chunkings (whole, and one byte at a time)."""
    for i in range(len(raw) + 1):
        yield [raw[:i], raw[i:]]
    yield [raw[i:i + 1] for i in range(len(raw))]


def decode_events(events):
    """Turn raw data payloads into (json_objects, saw_done).

    Malformed JSON in a ``data:`` line is a protocol error, never something to
    skip — silently swallowing it is exactly the Python-client defect FUTURE.md
    calls out under "Python client defects"."""
    out, saw_done = [], False
    for ev in events:
        if ev == DONE:
            saw_done = True
            continue
        if saw_done:
            raise ProtocolError("event after [DONE]", event=ev)
        try:
            out.append(json.loads(ev))
        except ValueError as e:
            raise ProtocolError("malformed JSON in SSE data field",
                                event=ev[:200], error=str(e))
    return out, saw_done
