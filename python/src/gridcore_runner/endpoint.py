from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable


class RunnerHttpError(RuntimeError):
    def __init__(self, status: int, detail: str):
        super().__init__(f"runner returned HTTP {status}: {detail}")
        self.status = status
        self.detail = detail


class RunnerProtocolError(RuntimeError):
    def __init__(self, message: str, *, partial: str = ""):
        super().__init__(message)
        self.partial = partial


class RunnerStallError(TimeoutError):
    def __init__(self, message: str, *, partial: str = ""):
        super().__init__(message)
        self.partial = partial


class RunnerCancelledError(RuntimeError):
    def __init__(self, message: str = "runner stream cancelled", *, partial: str = ""):
        super().__init__(message)
        self.partial = partial


@dataclass(frozen=True)
class ToolCall:
    """One assembled streamed tool call. `arguments` is the concatenation of the
    streamed argument fragments — a JSON string the caller parses."""
    index: int
    id: str | None
    name: str | None
    arguments: str


@dataclass(frozen=True)
class StreamResult:
    text: str
    reasoning: str
    estimated_tokens: int
    tool_calls: tuple["ToolCall", ...] = ()
    finish_reason: str | None = None


class RunnerEndpoint:
    """Typed access to Runner's HTTP contract."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 600,
        opener: Callable[..., Any] | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._opener = opener or urllib.request.urlopen

    def capabilities(self, *, timeout: float | None = None) -> dict[str, Any]:
        data = self.get_json("/v1/capabilities", timeout=timeout)
        if data.get("object") != "runner.capabilities":
            raise RunnerProtocolError("endpoint did not return Runner capabilities")
        return data

    def context_size(self, *, timeout: float | None = None) -> int | None:
        value = self.capabilities(timeout=timeout).get("context")
        return value if isinstance(value, int) and value > 0 else None

    def healthy(self, *, timeout: float = 2.0) -> bool:
        try:
            self.capabilities(timeout=timeout)
            return True
        except (OSError, RuntimeError):
            return False

    def get_json(self, path: str, *, timeout: float | None = None) -> dict[str, Any]:
        request = urllib.request.Request(self.base_url + path, method="GET")
        return self._read_json(request, timeout=timeout)

    def post_json(
        self, path: str, payload: dict[str, Any], *, timeout: float | None = None
    ) -> dict[str, Any]:
        request = self._request(path, payload)
        return self._read_json(request, timeout=timeout)

    def stream_chat(
        self,
        payload: dict[str, Any],
        *,
        on_delta: Callable[[str], None] | None = None,
        on_reasoning_delta: Callable[[str], None] | None = None,
        stall_seconds: float | None = None,
        cancel_event: Any | None = None,
    ) -> StreamResult:
        """Consume one SSE chat completion.

        Two guarantees the caller can rely on. A malformed `data:` frame
        raises `RunnerProtocolError` rather than being skipped, because a
        skipped frame produces text that looks whole and is not — and any
        later `finish_reason` would certify the hole as a finished answer.
        And silence is measured: `stall_seconds` is a no-stream-data
        watchdog over the time between events, not merely a socket option,
        so the raised `RunnerStallError` reports how long the stream was
        actually quiet. Both errors carry the text received so far in
        `.partial`, which is often a salvageable answer.

        Genuine non-data SSE lines — comments, `event:`, `id:`, `retry:` —
        are ignored, and count as activity for the watchdog.

        Scope: this parser treats each `data:` line as one complete JSON frame,
        which matches Runner's one-line-per-event wire format. It does not join
        multiple `data:` lines into a single SSE event, so it is Runner-wire-
        specific rather than a general SSE client. Streamed `delta.tool_calls`
        fragments ARE assembled by index into `StreamResult.tool_calls`, and the
        terminal `finish_reason` is preserved on the result.
        """
        body = dict(payload)
        body["stream"] = True
        request = self._request("/v1/chat/completions", body)
        window = stall_seconds if stall_seconds is not None else self.timeout
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        # streamed tool calls arrive as fragments keyed by index: id/name early,
        # arguments in pieces (RNR-016). Accumulate by index, assemble at the end.
        tool_acc: dict[int, dict[str, Any]] = {}
        finish_reason: str | None = None
        complete = False
        last_event = time.monotonic()
        if cancel_event is not None and cancel_event.is_set():
            raise RunnerCancelledError()
        try:
            with self._open(request, window) as response:
                # the watchdog covers the stream, not the connect and prompt
                # processing that precede the first chunk
                last_event = time.monotonic()
                for raw_line in response:
                    if cancel_event is not None and cancel_event.is_set():
                        raise RunnerCancelledError(partial="".join(text_parts))
                    now = time.monotonic()
                    idle, last_event = now - last_event, now
                    if idle > window:
                        # the socket timeout cannot see a gap that ends on its
                        # own; the watchdog can, so a resumed stream that went
                        # quiet past the window still reports as a stall
                        raise RunnerStallError(
                            self._stall_message(idle, window),
                            partial="".join(text_parts),
                        )
                    line = raw_line.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        complete = True
                        break
                    choice = self._stream_choice(data, "".join(text_parts))
                    if choice is None:
                        continue
                    delta = self._stream_delta(choice, "".join(text_parts))
                    piece = delta.get("content") or ""
                    reasoning = delta.get("reasoning_content") or ""
                    if piece:
                        text_parts.append(piece)
                        if on_delta is not None:
                            on_delta(piece)
                    if reasoning:
                        reasoning_parts.append(reasoning)
                        if on_reasoning_delta is not None:
                            on_reasoning_delta(reasoning)
                    for frag in delta.get("tool_calls") or []:
                        idx = frag.get("index", 0)
                        acc = tool_acc.setdefault(
                            idx, {"id": None, "name": None, "arguments": []})
                        if frag.get("id"):
                            acc["id"] = frag["id"]
                        fn = frag.get("function") or {}
                        if fn.get("name"):
                            acc["name"] = fn["name"]
                        if fn.get("arguments"):
                            acc["arguments"].append(fn["arguments"])
                    if cancel_event is not None and cancel_event.is_set():
                        raise RunnerCancelledError(partial="".join(text_parts))
                    if choice.get("finish_reason") is not None:
                        complete = True
                        finish_reason = choice["finish_reason"]
        except (RunnerCancelledError, RunnerStallError, RunnerProtocolError):
            raise
        except (socket.timeout, TimeoutError) as error:
            raise RunnerStallError(
                self._stall_message(time.monotonic() - last_event, window),
                partial="".join(text_parts),
            ) from error
        except urllib.error.HTTPError as error:
            raise self._http_error(error) from error
        except urllib.error.URLError as error:
            if isinstance(getattr(error, "reason", None), (socket.timeout, TimeoutError)):
                raise RunnerStallError(
                    self._stall_message(time.monotonic() - last_event, window),
                    partial="".join(text_parts),
                ) from error
            raise
        text = "".join(text_parts)
        if not complete:
            raise RunnerProtocolError(
                "runner stream ended before a terminal marker", partial=text
            )
        tool_calls = tuple(
            ToolCall(index=idx, id=acc["id"], name=acc["name"],
                     arguments="".join(acc["arguments"]))
            for idx, acc in sorted(tool_acc.items())
        )
        generated_chars = (len(text) + sum(len(item) for item in reasoning_parts)
                           + sum(len(tc.arguments) for tc in tool_calls))
        return StreamResult(
            text=text,
            reasoning="".join(reasoning_parts),
            estimated_tokens=max(0, (generated_chars + 3) // 4),
            tool_calls=tool_calls,
            finish_reason=finish_reason,
        )

    @staticmethod
    def _stall_message(idle: float, window: float) -> str:
        return (
            f"runner sent no stream data for {idle:.1f}s "
            f"(stall window {window:g}s)"
        )

    @staticmethod
    def _stream_choice(data: str, partial: str) -> dict[str, Any] | None:
        """The one choice carried by an SSE `data:` payload, or None for a
        chunk that legitimately has none — the `stream_options.include_usage`
        tail arrives with an empty choices array. Anything else that is not a
        well-formed chunk raises: protocol corruption must not read as the end
        of a healthy stream."""
        try:
            chunk = json.loads(data)
        except ValueError as error:
            raise RunnerProtocolError(
                f"runner sent an unparseable SSE data frame: {data[:120]!r}",
                partial=partial,
            ) from error
        if not isinstance(chunk, dict):
            raise RunnerProtocolError(
                "runner sent an SSE data frame that is not a JSON object",
                partial=partial,
            )
        choices = chunk.get("choices")
        if not isinstance(choices, list):
            raise RunnerProtocolError(
                "runner sent a stream chunk without a choices array",
                partial=partial,
            )
        if not choices:
            return None
        if not isinstance(choices[0], dict):
            raise RunnerProtocolError(
                "runner sent a stream chunk whose choice is not an object",
                partial=partial,
            )
        return choices[0]

    @staticmethod
    def _stream_delta(choice: dict[str, Any], partial: str) -> dict[str, Any]:
        delta = choice.get("delta")
        if delta is None:
            return {}
        if not isinstance(delta, dict):
            raise RunnerProtocolError(
                "runner sent a stream chunk whose delta is not an object",
                partial=partial,
            )
        for field in ("content", "reasoning_content"):
            value = delta.get(field)
            if value is not None and not isinstance(value, str):
                raise RunnerProtocolError(
                    f"runner sent a non-string {field} delta", partial=partial
                )
        return delta

    def _read_json(
        self, request: urllib.request.Request, *, timeout: float | None
    ) -> dict[str, Any]:
        try:
            with self._open(request, timeout or self.timeout) as response:
                body = response.read()
        except urllib.error.HTTPError as error:
            raise self._http_error(error) from error
        try:
            data = json.loads(body.decode("utf-8"))
        except ValueError as error:
            # a non-Runner service squatting on the port answers 200 with
            # HTML; that must read as "not a runner", not crash healthy()
            raise RunnerProtocolError("runner returned a non-JSON response") from error
        if not isinstance(data, dict):
            raise RunnerProtocolError("runner response must be a JSON object")
        return data

    def _open(self, request: urllib.request.Request, timeout: float):
        return self._opener(request, timeout=timeout)

    def _request(self, path: str, payload: dict[str, Any]) -> urllib.request.Request:
        return urllib.request.Request(
            self.base_url + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

    @staticmethod
    def _http_error(error: urllib.error.HTTPError) -> RunnerHttpError:
        try:
            detail = error.read().decode("utf-8", "replace")
        except Exception:
            detail = str(error.reason)
        return RunnerHttpError(error.code, detail or str(error.reason))
