from __future__ import annotations

import json
import socket
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


@dataclass(frozen=True)
class StreamResult:
    text: str
    reasoning: str
    estimated_tokens: int


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
    ) -> StreamResult:
        body = dict(payload)
        body["stream"] = True
        request = self._request("/v1/chat/completions", body)
        timeout = stall_seconds if stall_seconds is not None else self.timeout
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        complete = False
        try:
            with self._open(request, timeout) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        complete = True
                        break
                    try:
                        chunk = json.loads(data)
                        choice = chunk["choices"][0]
                    except (ValueError, KeyError, IndexError, TypeError):
                        continue
                    delta = choice.get("delta") or {}
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
                    if choice.get("finish_reason") is not None:
                        complete = True
        except (socket.timeout, TimeoutError) as error:
            partial = "".join(text_parts)
            raise RunnerStallError(
                f"runner produced no bytes for {timeout:g} seconds", partial=partial
            ) from error
        except urllib.error.HTTPError as error:
            raise self._http_error(error) from error
        except urllib.error.URLError as error:
            if isinstance(getattr(error, "reason", None), (socket.timeout, TimeoutError)):
                raise RunnerStallError(
                    f"runner produced no bytes for {timeout:g} seconds",
                    partial="".join(text_parts),
                ) from error
            raise
        text = "".join(text_parts)
        if not complete:
            raise RunnerProtocolError(
                "runner stream ended before a terminal marker", partial=text
            )
        generated_chars = len(text) + sum(len(item) for item in reasoning_parts)
        return StreamResult(
            text=text,
            reasoning="".join(reasoning_parts),
            estimated_tokens=max(0, (generated_chars + 3) // 4),
        )

    def _read_json(
        self, request: urllib.request.Request, *, timeout: float | None
    ) -> dict[str, Any]:
        try:
            with self._open(request, timeout or self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            raise self._http_error(error) from error
        if not isinstance(data, dict):
            raise RunnerProtocolError("runner response must be a JSON object")
        return data

    def _open(self, request: urllib.request.Request, timeout: float):
        return self._opener(request, timeout)

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
