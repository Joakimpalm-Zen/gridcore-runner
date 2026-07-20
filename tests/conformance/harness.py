"""Public surface of the agent-protocol conformance harness.

One class does the talking: ``Client``. It speaks HTTP over a raw socket
(runner is one-request-per-connection, Connection: close) so that transport
faults stay distinguishable from protocol faults, and so a test can hang up
mid-stream on purpose.

Everything a test needs is re-exported here; the underscore modules are
implementation detail.

    with RunnerServer(exe, model) as srv:
        c = Client(srv, report)
        r = c.chat({"messages": [...], "max_tokens": 8})   # -> Response
        r.expect_status(200)
        c.expect_400({"messages": [...], "stream": "yes"})
        ev = c.chat_stream({...})                          # -> Stream
"""

import json
import socket
import time

from _errors import (CATEGORIES, ConformanceError, ModelQualityError,
                      ProtocolError, SchemaError, TransportError, categorize)
from _process import RunnerServer, find_runner, free_port, peak_rss_kb, rss_kind
from _report import Report, normalize
from _sse import (DONE, SSEParser, decode_events, parse_named_stream,
                   parse_stream, split_points)

__all__ = [
    "Client", "Response", "Stream", "ResponseStream", "RunnerServer", "Report",
    "ConformanceError", "TransportError", "ProtocolError", "SchemaError",
    "ModelQualityError", "categorize", "CATEGORIES",
    "SSEParser", "parse_stream", "parse_named_stream", "split_points",
    "decode_events", "DONE",
    "normalize", "find_runner", "free_port", "peak_rss_kb", "rss_kind",
    "validate_against_schema",
]

DEFAULT_TIMEOUT = 120.0


# --------------------------------------------------------------------- wire
def _read_all(sock):
    out = bytearray()
    while True:
        try:
            b = sock.recv(65536)
        except socket.timeout as e:
            raise TransportError("read timed out", received=len(out),
                                 error=str(e))
        except OSError as e:
            raise TransportError("socket error while reading", error=str(e))
        if not b:
            return bytes(out)
        out += b


def _split_head(raw):
    i = raw.find(b"\r\n\r\n")
    if i < 0:
        raise TransportError("response has no header terminator",
                             head=raw[:200].decode("utf-8", "replace"))
    head = raw[:i].decode("iso-8859-1")
    body = raw[i + 4:]
    lines = head.split("\r\n")
    parts = lines[0].split(" ", 2)
    if len(parts) < 2 or not parts[0].startswith("HTTP/"):
        raise TransportError("malformed status line", line=lines[0])
    try:
        status = int(parts[1])
    except ValueError:
        raise TransportError("non-numeric status code", line=lines[0])
    headers = {}
    for line in lines[1:]:
        k, _, v = line.partition(":")
        headers[k.strip().lower()] = v.strip()
    return status, headers, body


class Response:
    """A buffered HTTP response with protocol-aware accessors."""

    def __init__(self, name, status, headers, body, latency_ms):
        self.name = name
        self.status = status
        self.headers = headers
        self.body = body
        self.latency_ms = latency_ms
        self._json = None

    # ---- protocol level
    def expect_status(self, code):
        if self.status != code:
            raise ProtocolError(f"expected HTTP {code}", got=self.status,
                                request=self.name,
                                body=self.text[:400])
        return self

    @property
    def text(self):
        return self.body.decode("utf-8", "replace")

    @property
    def json(self):
        if self._json is None:
            ctype = self.headers.get("content-type", "")
            if "json" not in ctype:
                raise ProtocolError("response is not JSON", content_type=ctype,
                                    request=self.name, body=self.text[:200])
            declared = self.headers.get("content-length")
            if declared is not None and int(declared) != len(self.body):
                raise TransportError("body length does not match Content-Length",
                                     declared=declared, actual=len(self.body))
            try:
                self._json = json.loads(self.body)
            except ValueError as e:
                raise ProtocolError("response body is not valid JSON",
                                    request=self.name, error=str(e),
                                    body=self.text[:200])
        return self._json

    def expect_error_envelope(self, contains=None):
        """Runner's documented error shape: {"error":{"message","type"}}."""
        d = self.json
        err = d.get("error")
        if not isinstance(err, dict) or "message" not in err or "type" not in err:
            raise ProtocolError("error response lacks {error:{message,type}}",
                                request=self.name, body=d)
        want = "invalid_request_error" if self.status < 500 else "server_error"
        if err["type"] != want:
            raise ProtocolError("wrong error type for status", status=self.status,
                                expected=want, got=err["type"])
        if contains and contains.lower() not in err["message"].lower():
            raise ProtocolError("error message does not explain the rejection",
                                expected_substring=contains, got=err["message"])
        return err

    # ---- chat convenience
    @property
    def choice(self):
        d = self.json
        if not isinstance(d.get("choices"), list) or not d["choices"]:
            raise ProtocolError("response has no choices", request=self.name,
                                body=d)
        return d["choices"][0]

    @property
    def content(self):
        msg = self.choice.get("message")
        if not isinstance(msg, dict):
            raise ProtocolError("chat choice has no message object",
                                request=self.name, choice=self.choice)
        c = msg.get("content")
        if not isinstance(c, str):
            raise ProtocolError("message.content is not a string",
                                request=self.name, got=type(c).__name__)
        return c

    @property
    def finish_reason(self):
        return self.choice.get("finish_reason")

    @property
    def usage(self):
        u = self.json.get("usage")
        if not isinstance(u, dict):
            raise ProtocolError("response has no usage object", request=self.name)
        return u

    @property
    def telemetry(self):
        return self.json.get("runner_telemetry") or {}


class Stream:
    """A completed SSE response: raw bytes plus the parsed, normalized events."""

    def __init__(self, name, status, headers, raw, latency_ms, first_byte_ms):
        self.name = name
        self.status = status
        self.headers = headers
        self.raw = raw
        self.latency_ms = latency_ms
        self.first_byte_ms = first_byte_ms
        self.events = parse_stream(raw)
        self.chunks, self.saw_done = decode_events(self.events)

    def expect_sse(self):
        if self.status != 200:
            raise ProtocolError("stream did not start with 200",
                                got=self.status, request=self.name,
                                body=self.raw[:300].decode("utf-8", "replace"))
        ctype = self.headers.get("content-type", "")
        if "text/event-stream" not in ctype:
            raise ProtocolError("stream Content-Type is not text/event-stream",
                                got=ctype, request=self.name)
        if not self.saw_done:
            raise ProtocolError("stream did not terminate with data: [DONE]",
                                request=self.name, events=len(self.events))
        if not self.chunks:
            raise ProtocolError("stream carried no chunks before [DONE]",
                                request=self.name)
        return self

    @property
    def text(self):
        out = []
        for c in self.chunks:
            for ch in c.get("choices", []):
                out.append(ch.get("delta", {}).get("content") or ch.get("text") or "")
        return "".join(out)

    @property
    def finish_reason(self):
        fin = None
        for c in self.chunks:
            for ch in c.get("choices", []):
                fin = ch.get("finish_reason") or fin
        return fin

    def deltas(self):
        return [ch.get("delta", {})
                for c in self.chunks for ch in c.get("choices", [])]


class ResponseStream:
    """A completed /v1/responses SSE stream.

    Unlike a chat stream this one is a sequence of *named* typed events with a
    mandatory order, so the parsed form keeps the ``event:`` name alongside the
    payload rather than collapsing to a chunk list."""

    def __init__(self, name, status, headers, raw, latency_ms, first_byte_ms):
        self.name = name
        self.status = status
        self.headers = headers
        self.raw = raw
        self.latency_ms = latency_ms
        self.first_byte_ms = first_byte_ms
        self.pairs = parse_named_stream(raw)
        self.events = [p[1] for p in self.pairs]

    def expect_sse(self):
        if self.status != 200:
            raise ProtocolError("response stream did not start with 200",
                                got=self.status, request=self.name,
                                body=self.raw[:300].decode("utf-8", "replace"))
        ctype = self.headers.get("content-type", "")
        if "text/event-stream" not in ctype:
            raise ProtocolError("stream Content-Type is not text/event-stream",
                                got=ctype, request=self.name)
        return self

    @property
    def typed(self):
        """[(event_name, decoded_payload)], asserting the two names agree."""
        out = []
        for name, data in self.pairs:
            try:
                d = json.loads(data)
            except ValueError as e:
                raise ProtocolError("malformed JSON in Responses SSE data",
                                    request=self.name, event=data[:200],
                                    error=str(e))
            if name is None:
                raise ProtocolError("Responses event has no SSE event: name",
                                    request=self.name, payload=d)
            if d.get("type") != name:
                raise ProtocolError("SSE event name disagrees with data.type",
                                    request=self.name, event_field=name,
                                    data_type=d.get("type"))
            out.append((name, d))
        return out

    @property
    def names(self):
        return [n for n, _ in self.typed]

    def payloads(self, kind):
        return [d for n, d in self.typed if n == kind]

    @property
    def text(self):
        return "".join(d.get("delta", "")
                       for d in self.payloads("response.output_text.delta"))


# ------------------------------------------------------------------- client
class Client:
    """Issues requests against a RunnerServer and records metrics."""

    def __init__(self, server, report, timeout=DEFAULT_TIMEOUT):
        self.server = server
        self.report = report
        self.timeout = timeout

    # ---- low level
    def _connect(self):
        try:
            s = socket.create_connection(("127.0.0.1", self.server.port),
                                         timeout=self.timeout)
        except OSError as e:
            self.server.assert_alive()
            raise TransportError("cannot connect to runner",
                                 port=self.server.port, error=str(e))
        s.settimeout(self.timeout)
        return s

    def _request_bytes(self, method, path, body=None):
        head = (f"{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{self.server.port}\r\n"
                f"Connection: close\r\n")
        if body is None:
            return (head + "\r\n").encode()
        head += ("Content-Type: application/json\r\n"
                 f"Content-Length: {len(body)}\r\n\r\n")
        return head.encode() + body

    def raw(self, name, method, path, payload=None):
        """Buffered request. Returns a Response."""
        body = json.dumps(payload).encode() if payload is not None else None
        t0 = time.monotonic()
        sock = self._connect()
        try:
            sock.sendall(self._request_bytes(method, path, body))
            raw = _read_all(sock)
        finally:
            sock.close()
        latency = (time.monotonic() - t0) * 1000
        if not raw:
            self.server.assert_alive()
            raise TransportError("empty response from runner", request=name)
        status, headers, rbody = _split_head(raw)
        resp = Response(name, status, headers, rbody, round(latency, 2))
        self._record(name, path, False, resp)
        return resp

    def post_bytes(self, name, path, body, content_type="application/json"):
        """POST an arbitrary body — including one that is not valid JSON."""
        t0 = time.monotonic()
        sock = self._connect()
        try:
            head = (f"POST {path} HTTP/1.1\r\nHost: 127.0.0.1:{self.server.port}\r\n"
                    f"Connection: close\r\nContent-Type: {content_type}\r\n"
                    f"Content-Length: {len(body)}\r\n\r\n")
            sock.sendall(head.encode() + body)
            raw = _read_all(sock)
        finally:
            sock.close()
        if not raw:
            self.server.assert_alive()
            raise TransportError("empty response to raw body", request=name)
        status, headers, rbody = _split_head(raw)
        resp = Response(name, status, headers, rbody,
                        round((time.monotonic() - t0) * 1000, 2))
        self._record(name, path, False, resp)
        return resp

    def stream_raw(self, name, path, payload, close_after_bytes=None):
        """Streaming request. If ``close_after_bytes`` is set the socket is
        hung up once that many bytes have arrived, simulating a client that
        disconnects mid-stream; the partial bytes are returned."""
        body = json.dumps(payload).encode()
        t0 = time.monotonic()
        first = None
        sock = self._connect()
        buf = bytearray()
        try:
            sock.sendall(self._request_bytes("POST", path, body))
            while True:
                try:
                    b = sock.recv(65536)
                except socket.timeout as e:
                    raise TransportError("stream stalled", request=name,
                                         received=len(buf), error=str(e))
                except OSError as e:
                    raise TransportError("socket error during stream",
                                         request=name, error=str(e))
                if not b:
                    break
                if first is None:
                    first = (time.monotonic() - t0) * 1000
                buf += b
                if close_after_bytes is not None and len(buf) >= close_after_bytes:
                    break
        finally:
            sock.close()
        latency = (time.monotonic() - t0) * 1000
        return bytes(buf), round(latency, 2), round(first or latency, 2)

    # ---- typed helpers
    def chat(self, payload, name="chat"):
        return self.raw(name, "POST", "/v1/chat/completions", payload)

    def completion(self, payload, name="completion"):
        return self.raw(name, "POST", "/v1/completions", payload)

    def embeddings(self, payload, name="embeddings"):
        return self.raw(name, "POST", "/v1/embeddings", payload)

    def get(self, path, name=None):
        return self.raw(name or path, "GET", path)

    def chat_stream(self, payload, name="chat-stream"):
        payload = dict(payload, stream=True)
        raw, latency, first = self.stream_raw(name, "/v1/chat/completions", payload)
        status, headers, body = _split_head(raw)
        st = Stream(name, status, headers, body, latency, first)
        self._record(name, "/v1/chat/completions", True, st)
        return st

    # ---- Responses API
    def responses(self, payload, name="responses"):
        return self.raw(name, "POST", "/v1/responses", payload)

    def responses_stream(self, payload, name="responses-stream"):
        payload = dict(payload, stream=True)
        raw, latency, first = self.stream_raw(name, "/v1/responses", payload)
        status, headers, body = _split_head(raw)
        st = ResponseStream(name, status, headers, body, latency, first)
        self._record(name, "/v1/responses", True, st)
        return st

    def expect_400(self, payload, name, contains=None, path="/v1/chat/completions"):
        """A malformed or unsupported field must be rejected, never ignored.

        This is a project invariant, not a nicety: a request that silently
        drops a field the caller depended on (a schema, a stop sequence) looks
        like success and is not."""
        r = self.raw(name, "POST", path, payload)
        if r.status == 200:
            raise ProtocolError(
                "request was accepted but should have been rejected with 400 "
                "(silently ignoring an unsupported field is a conformance bug)",
                request=name, payload=payload, body=r.text[:300])
        r.expect_status(400)
        r.expect_error_envelope(contains)
        return r

    # ---- metrics
    def _record(self, name, path, stream, obj):
        rec = {"name": name, "endpoint": path, "stream": stream,
               "status": obj.status, "latency_ms": obj.latency_ms}
        if stream:
            rec["time_to_first_byte_ms"] = obj.first_byte_ms
            rec["events"] = len(obj.events)
        else:
            try:
                if obj.status == 200 and "json" in obj.headers.get("content-type", ""):
                    d = obj.json
                    u = d.get("usage") or {}
                    t = d.get("runner_telemetry") or {}
                    rec["prompt_tokens"] = u.get("prompt_tokens")
                    rec["completion_tokens"] = u.get("completion_tokens")
                    rec["generation_tok_s"] = t.get("generation_tok_s")
                    gen_s = t.get("generation_seconds")
                    evald = t.get("prompt_eval_tokens")
                    if gen_s is not None and evald:
                        prompt_s = max(obj.latency_ms / 1000 - gen_s, 1e-6)
                        rec["prompt_tok_s"] = round(evald / prompt_s, 3)
            except ConformanceError:
                pass
        rec["peak_rss_kb"] = self.server.sample_rss()
        self.report.record_request(rec)


# ------------------------------------------------------- schema validation
def validate_against_schema(value, schema, path="$"):
    """Minimal JSON-Schema check covering the subset runner's schema.c accepts.

    Violations raise SchemaError, so a structurally valid but non-conforming
    response is never confused with a protocol fault."""
    t = schema.get("type")
    if t == "object":
        if not isinstance(value, dict):
            raise SchemaError("expected object", at=path, got=type(value).__name__)
        props = schema.get("properties", {})
        for req in schema.get("required", []):
            if req not in value:
                raise SchemaError("required property missing", at=path, key=req,
                                  got=sorted(value))
        for k, v in value.items():
            if k in props:
                validate_against_schema(v, props[k], f"{path}.{k}")
            elif schema.get("additionalProperties") is False:
                raise SchemaError("unexpected property", at=path, key=k)
    elif t == "array":
        if not isinstance(value, list):
            raise SchemaError("expected array", at=path, got=type(value).__name__)
        if "minItems" in schema and len(value) < schema["minItems"]:
            raise SchemaError("too few items", at=path, got=len(value),
                              minItems=schema["minItems"])
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            raise SchemaError("too many items", at=path, got=len(value),
                              maxItems=schema["maxItems"])
        if "items" in schema:
            for i, v in enumerate(value):
                validate_against_schema(v, schema["items"], f"{path}[{i}]")
    elif t == "string":
        if not isinstance(value, str):
            raise SchemaError("expected string", at=path, got=type(value).__name__)
    elif t == "integer":
        if not isinstance(value, int) or isinstance(value, bool):
            raise SchemaError("expected integer", at=path, got=type(value).__name__)
    elif t == "number":
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise SchemaError("expected number", at=path, got=type(value).__name__)
    elif t == "boolean":
        if not isinstance(value, bool):
            raise SchemaError("expected boolean", at=path, got=type(value).__name__)
    if "enum" in schema and value not in schema["enum"]:
        raise SchemaError("value outside enum", at=path, got=value,
                          enum=schema["enum"])
    return value
