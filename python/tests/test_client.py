from __future__ import annotations

import io
import json
import os
import re
import socket
import sys
import tempfile
import time
import threading
import unittest
import urllib.error
from pathlib import Path

from gridcore_runner import (
    ManagedRunner,
    RunnerEndpoint,
    RunnerProtocolError,
    RunnerCancelledError,
    RunnerStallError,
    ServerLaunch,
    StartupLease,
    build_server_args,
    model_registry_argument,
    query_system_capabilities,
    spawn_detached,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER_BIN = REPO_ROOT / ("runner.exe" if os.name == "nt" else "runner")
# smallest first: the integration test only needs a model runner will begin
# loading, and the smallest one starts serving soonest
MODELS = sorted((REPO_ROOT / "models").glob("*.gguf"), key=lambda p: p.stat().st_size)


def free_port() -> int:
    """A port nothing answers on, so health checks fail the way a
    still-loading runner's do."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class StartupLeaseTests(unittest.TestCase):
    def test_live_owner_keeps_a_second_claim_from_overwriting_it(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "runner.lease"
            first = StartupLease(path)
            second = StartupLease(path)

            self.assertTrue(first.acquire())
            original = (path / "owner.json").read_text(encoding="utf-8")
            self.assertFalse(second.acquire())
            second.release()
            self.assertEqual((path / "owner.json").read_text(encoding="utf-8"), original)
            first.release()
            self.assertFalse(path.exists())

    def test_dead_owner_record_is_reclaimed_without_reaping_a_child(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "runner.lease"
            path.write_text(
                json.dumps({"owner_pid": 2147483647, "token": "stale"}),
                encoding="utf-8",
            )

            lease = StartupLease(path)

            self.assertTrue(lease.acquire())
            self.assertNotEqual(json.loads((path / "owner.json").read_text())["token"], "stale")
            lease.release()

    def test_release_is_token_safe(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "runner.lease"
            owner = StartupLease(path)
            non_owner = StartupLease(path)

            self.assertTrue(owner.acquire())
            non_owner.release()

            self.assertTrue(path.exists())


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
    def test_pre_cancelled_stream_never_opens_request(self):
        cancelled = threading.Event()
        cancelled.set()
        opened = []
        endpoint = RunnerEndpoint(
            "http://127.0.0.1:8080",
            opener=lambda *args, **kwargs: opened.append(True))

        with self.assertRaises(RunnerCancelledError):
            endpoint.stream_chat({"messages": []}, cancel_event=cancelled)

        self.assertEqual(opened, [])

    def test_stream_cancellation_preserves_partial_and_closes_response(self):
        cancelled = threading.Event()
        response = _Response([
            b'data: {"choices":[{"delta":{"content":"partial"}}]}\n',
            b'data: {"choices":[{"delta":{"content":"ignored"},"finish_reason":"stop"}]}\n',
        ])
        endpoint = RunnerEndpoint(
            "http://127.0.0.1:8080", opener=lambda *args, **kwargs: response)

        with self.assertRaises(RunnerCancelledError) as caught:
            endpoint.stream_chat(
                {"messages": []}, cancel_event=cancelled,
                on_delta=lambda piece: cancelled.set())

        self.assertEqual(caught.exception.partial, "partial")
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

    def test_stall_message_reports_measured_idle_time_not_the_window(self):
        """The old message asserted "no bytes for N seconds" using the
        configured window, a number nothing had measured. A stall report has
        to state how long the stream was actually silent."""
        class StallingResponse(_Response):
            def __iter__(self):
                yield b'data: {"choices":[{"delta":{"content":"hi"}}]}\n'
                time.sleep(0.2)
                raise socket.timeout("stalled")

        endpoint = RunnerEndpoint(
            "http://127.0.0.1:8080",
            opener=lambda request, timeout: StallingResponse(),
        )

        with self.assertRaises(RunnerStallError) as caught:
            endpoint.stream_chat({"messages": []}, stall_seconds=0.1)

        message = str(caught.exception)
        match = re.search(
            r"no stream data for ([0-9.]+)s \(stall window 0\.1s\)", message
        )
        self.assertIsNotNone(match, message)
        # The measured figure, not an exact one: sleep(0.2) guarantees at
        # least 0.2s of silence, but CI schedulers overshoot it (macOS
        # runners measured 0.3s). What the old bug reported was the 0.1s
        # window itself; any genuinely measured value clears 0.15 and stays
        # far under the seconds a wall-clock-vs-monotonic mixup would show.
        idle = float(match.group(1))
        self.assertGreaterEqual(idle, 0.15)
        self.assertLess(idle, 2.0)

    def test_a_gap_between_events_is_a_stall_even_when_bytes_resume(self):
        """A real inactivity watchdog fires on the gap itself. urllib's
        per-socket-operation timeout does not see one that ends on its own."""
        class SlowResponse(_Response):
            def __iter__(self):
                yield b'data: {"choices":[{"delta":{"content":"before"}}]}\n'
                time.sleep(0.3)
                yield b'data: {"choices":[{"delta":{"content":"after"},"finish_reason":"stop"}]}\n'

        endpoint = RunnerEndpoint(
            "http://127.0.0.1:8080",
            opener=lambda request, timeout: SlowResponse(),
        )

        with self.assertRaises(RunnerStallError) as caught:
            endpoint.stream_chat({"messages": []}, stall_seconds=0.1)

        self.assertEqual(caught.exception.partial, "before")

    def test_malformed_json_frame_raises_with_the_partial_response(self):
        endpoint = RunnerEndpoint(
            "http://127.0.0.1:8080",
            opener=lambda request, timeout: _Response([
                b'data: {"choices":[{"delta":{"content":"good"}}]}\n',
                b'data: {"choices":[{"delta":{"con\n',
                b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n',
            ]),
        )

        with self.assertRaises(RunnerProtocolError) as caught:
            endpoint.stream_chat({"messages": []})

        self.assertEqual(caught.exception.partial, "good")

    def test_chunk_without_choices_raises_instead_of_completing_the_stream(self):
        """The dangerous shape: a corrupt frame skipped, then a later
        finish_reason marks a truncated stream "finished"."""
        endpoint = RunnerEndpoint(
            "http://127.0.0.1:8080",
            opener=lambda request, timeout: _Response([
                b'data: {"choices":[{"delta":{"content":"good"}}]}\n',
                b'data: {"object":"chat.completion.chunk"}\n',
                b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n',
                b'data: [DONE]\n',
            ]),
        )

        with self.assertRaises(RunnerProtocolError) as caught:
            endpoint.stream_chat({"messages": []})

        self.assertEqual(caught.exception.partial, "good")

    def test_usage_only_tail_chunk_is_not_malformed(self):
        """stream_options.include_usage makes runner send one chunk with an
        empty choices array. That is the contract, not corruption."""
        endpoint = RunnerEndpoint(
            "http://127.0.0.1:8080",
            opener=lambda request, timeout: _Response([
                b'data: {"choices":[{"delta":{"content":"hi"},"finish_reason":"stop"}]}\n',
                b'data: {"choices":[],"usage":{"total_tokens":7}}\n',
                b'data: [DONE]\n',
            ]),
        )

        self.assertEqual(endpoint.stream_chat({"messages": []}).text, "hi")

    def test_non_data_sse_lines_are_still_ignored(self):
        endpoint = RunnerEndpoint(
            "http://127.0.0.1:8080",
            opener=lambda request, timeout: _Response([
                b": keep-alive\n",
                b"\n",
                b"event: ping\n",
                b"id: 4\n",
                b"retry: 1000\n",
                b'data: {"choices":[{"delta":{"content":"hi"},"finish_reason":"stop"}]}\n',
                b"data: [DONE]\n",
            ]),
        )

        self.assertEqual(endpoint.stream_chat({"messages": []}).text, "hi")


class ManagedRunnerOwnershipTests(unittest.TestCase):
    def test_failed_start_leaves_no_child_process_behind(self):
        """A False start must not hand the caller an orphan. The child here
        never answers, exactly like a runner still loading weights when the
        health deadline expires."""
        spawned = []

        def spawn(args):
            process = spawn_detached(
                [sys.executable, "-c", "import time; time.sleep(120)"]
            )
            spawned.append(process)
            return process

        managed = ManagedRunner(
            ServerLaunch("runner", "model.gguf", free_port()), spawn=spawn
        )

        self.assertFalse(managed.start(timeout=0.3, interval=0.05))

        self.assertIsNotNone(spawned[0].poll(), "start() orphaned its child")
        self.assertIsNone(managed.process)
        self.assertFalse(managed.alive())

    @unittest.skipUnless(
        RUNNER_BIN.exists() and MODELS, "needs a built runner and a model"
    )
    def test_failed_start_leaves_no_runner_serve_process(self):
        """The reported failure, with the real binary: a runner that cannot
        become answerable in time must not survive holding memory."""
        launch = ServerLaunch(
            executable=RUNNER_BIN,
            model=MODELS[0],
            port=free_port(),
            gpu="off",
        )
        spawned = []

        def spawn(args):
            process = spawn_detached(args, cwd=RUNNER_BIN.parent)
            spawned.append(process)
            return process

        managed = ManagedRunner(launch, spawn=spawn)

        # timeout=0 expires the health deadline deterministically, before this
        # runner can answer — the same path a slow-loading model takes, without
        # racing a load that happens to be fast on this box
        self.assertFalse(managed.start(timeout=0.0, interval=0.1))

        self.assertIsNotNone(
            spawned[0].poll(), "a runner --serve process outlived a failed start()"
        )
        self.assertIsNone(managed.process)
        self.assertFalse(managed.alive())


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
