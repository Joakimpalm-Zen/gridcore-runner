from __future__ import annotations

import io
import json
import os
import socket
import unittest
import urllib.error

from gridcore_runner import (
    ManagedRunner,
    RunnerEndpoint,
    RunnerProtocolError,
    RunnerStallError,
    ServerLaunch,
    build_server_args,
    model_registry_argument,
    query_system_capabilities,
)


class _Response:
    def __init__(self, lines=(), payload=None):
        self._lines = list(lines)
        self._body = json.dumps(payload or {}).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def __iter__(self):
        return iter(self._lines)

    def read(self):
        return self._body


class EndpointTests(unittest.TestCase):
    def test_capabilities_are_runner_identified_and_expose_context(self):
        seen = {}

        def open_request(request, *, timeout):
            seen["timeout"] = timeout
            return _Response(payload={
                "object": "runner.capabilities",
                "context": 6144,
                "features": {"json_schema": True},
            })

        endpoint = RunnerEndpoint(
            "http://127.0.0.1:8080",
            opener=open_request,
        )

        self.assertEqual(endpoint.context_size(), 6144)
        self.assertTrue(endpoint.healthy())
        self.assertEqual(seen["timeout"], 2.0)

    def test_stream_collects_content_and_requires_terminal_marker(self):
        lines = [
            b'data: {"choices":[{"delta":{"content":"hel"}}]}\n',
            b'data: {"choices":[{"delta":{"content":"lo"},"finish_reason":"stop"}]}\n',
        ]
        endpoint = RunnerEndpoint(
            "http://127.0.0.1:8080", opener=lambda request, timeout: _Response(lines)
        )
        seen = []

        result = endpoint.stream_chat({"messages": []}, on_delta=seen.append)

        self.assertEqual(result.text, "hello")
        self.assertEqual(seen, ["hel", "lo"])

    def test_stream_rejects_premature_eof_with_partial_text(self):
        endpoint = RunnerEndpoint(
            "http://127.0.0.1:8080",
            opener=lambda request, timeout: _Response([
                b'data: {"choices":[{"delta":{"content":"partial"}}]}\n'
            ]),
        )

        with self.assertRaises(RunnerProtocolError) as caught:
            endpoint.stream_chat({"messages": []})

        self.assertEqual(caught.exception.partial, "partial")

    def test_socket_timeout_is_a_stall_with_partial_text(self):
        class StallingResponse(_Response):
            def __iter__(self):
                yield b'data: {"choices":[{"delta":{"content":"partial"}}]}\n'
                raise socket.timeout("stalled")

        endpoint = RunnerEndpoint(
            "http://127.0.0.1:8080",
            opener=lambda request, timeout: StallingResponse(),
        )

        with self.assertRaises(RunnerStallError) as caught:
            endpoint.stream_chat({"messages": []}, stall_seconds=3)

        self.assertEqual(caught.exception.partial, "partial")


class LaunchTests(unittest.TestCase):
    def test_system_capabilities_are_parsed_from_runner_binary(self):
        seen = []

        class Completed:
            stdout = json.dumps({
                "os": "windows",
                "arch": "x86_64",
                "cpu_cores": 8,
                "ram_bytes": 16 * 1024**3,
                "gpu": {"backend": "cuda", "name": "RTX", "vram_bytes": 8 * 1024**3},
                "quants": ["Q4_K"],
                "gpu_quants": ["Q4_K"],
            })

        caps = query_system_capabilities(
            "runner.exe", run=lambda args, **kwargs: seen.append((args, kwargs)) or Completed()
        )

        self.assertEqual(caps["gpu"]["backend"], "cuda")
        self.assertEqual(seen[0][0], ["runner.exe", "--caps"])

    def test_registry_argument_is_stable_and_rejects_reserved_names(self):
        self.assertEqual(
            model_registry_argument({"worker": "C:/models/w.gguf", "planner": "C:/models/p.gguf"}),
            "planner=C:/models/p.gguf,worker=C:/models/w.gguf",
        )
        with self.assertRaises(ValueError):
            model_registry_argument({"bad,name": "model.gguf"})

    def test_server_args_use_runner_owned_fit_and_parent_lifetime(self):
        launch = ServerLaunch(
            executable="runner.exe",
            model="model.gguf",
            port=8090,
            reserve_pct=55,
            threads=6,
            parent_pid=1234,
        )

        args = build_server_args(launch)

        self.assertEqual(args[:2], ["runner.exe", "--serve"])
        self.assertEqual(args[args.index("-c") + 1], "0")
        self.assertEqual(args[args.index("--reserve") + 1], "55")
        self.assertEqual(args[args.index("--parent-pid") + 1], "1234")

    def test_managed_runner_waits_for_readiness_and_stops_owned_child(self):
        class Process:
            def __init__(self):
                self.running = True
                self.terminated = False

            def poll(self):
                return None if self.running else 0

            def terminate(self):
                self.terminated = True
                self.running = False

            def wait(self, timeout=None):
                return 0

        class Endpoint:
            def healthy(self, timeout=2):
                return True

        process = Process()
        seen = []
        managed = ManagedRunner(
            ServerLaunch("runner.exe", "model.gguf", 8090),
            spawn=lambda args: seen.append(args) or process,
            endpoint_factory=lambda url: Endpoint(),
        )

        self.assertTrue(managed.start(timeout=0.1, interval=0.01))
        self.assertIn("--parent-pid", seen[0])
        managed.stop()
        self.assertTrue(process.terminated)


if __name__ == "__main__":
    unittest.main()
