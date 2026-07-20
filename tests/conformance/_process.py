"""Server lifecycle and peak-RSS sampling.

Runner is started on a free port against a model, waited on until /health
answers, and shut down cleanly (SIGTERM, then SIGKILL). Everything about
"is it up yet" and "did it die" lives here so tests never poll a socket.
"""

import contextlib
import os
import platform
import signal
import socket
import subprocess
import sys
import time

from _errors import TransportError

WINDOWS = os.name == "nt"


def free_port():
    """Bind :0, learn the port, release it. Racy in principle; the server
    binds immediately after and start() fails loudly if it lost the race."""
    with contextlib.closing(socket.socket()) as s:
        s.bind(("127.0.0.1", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def peak_rss_kb(pid):
    """Peak resident set size in KiB, or None where unavailable.

    Linux reports a true high-water mark (VmHWM). macOS has no cheap peak, so
    the current RSS is sampled instead and the report labels it as such."""
    try:
        if platform.system() == "Linux":
            with open(f"/proc/{pid}/status", encoding="ascii") as f:
                for line in f:
                    if line.startswith("VmHWM:"):
                        return int(line.split()[1])
        elif platform.system() == "Darwin":
            out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)],
                                 capture_output=True, text=True, timeout=5)
            if out.stdout.strip():
                return int(out.stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return None


def rss_kind():
    return {"Linux": "peak", "Darwin": "sampled"}.get(platform.system(), "unavailable")


class RunnerServer:
    """A running ``runner --serve``. Use as a context manager."""

    def __init__(self, exe, model, *, extra_args=(), ctx=1024, parallel=1,
                 start_timeout=60.0, log_path=None):
        self.exe = str(exe)
        self.model = str(model)
        self.extra_args = list(extra_args)
        self.ctx = ctx
        self.parallel = parallel
        self.start_timeout = start_timeout
        self.log_path = log_path
        self.port = None
        self.proc = None
        self._log = None
        self.peak_rss_kb = None

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.port}"

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
        return False

    def start(self):
        self.port = free_port()
        argv = [self.exe, "-m", self.model, "--serve",
                "--port", str(self.port), "--parallel", str(self.parallel),
                "-c", str(self.ctx)] + self.extra_args
        self._log = open(self.log_path, "wb") if self.log_path else subprocess.DEVNULL
        self.proc = subprocess.Popen(argv, stdout=self._log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + self.start_timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise TransportError("runner exited during startup",
                                     returncode=self.proc.returncode,
                                     argv=" ".join(argv), log=self._tail())
            if self._health_ok():
                return self
            time.sleep(0.05)
        self.stop()
        raise TransportError("runner did not become healthy in time",
                             seconds=self.start_timeout, argv=" ".join(argv),
                             log=self._tail())

    def sample_rss(self):
        if self.proc and self.proc.poll() is None:
            v = peak_rss_kb(self.proc.pid)
            if v is not None:
                self.peak_rss_kb = max(self.peak_rss_kb or 0, v)
        return self.peak_rss_kb

    def assert_alive(self):
        if self.proc and self.proc.poll() is not None:
            raise TransportError("runner died mid-suite",
                                 returncode=self.proc.returncode, log=self._tail())

    def stop(self):
        if not self.proc:
            return
        self.sample_rss()
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM if not WINDOWS else signal.SIGTERM)
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=10)
        if self._log not in (None, subprocess.DEVNULL):
            self._log.close()
        self._log = None

    # ------------------------------------------------------------------
    def _health_ok(self):
        try:
            with contextlib.closing(socket.create_connection(
                    ("127.0.0.1", self.port), timeout=2)) as s:
                s.sendall(b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                          b"Connection: close\r\n\r\n")
                return b" 200 " in s.recv(64)
        except OSError:
            return False

    def _tail(self, n=2000):
        if not self.log_path or not os.path.exists(self.log_path):
            return "<no log>"
        with open(self.log_path, "rb") as f:
            return f.read()[-n:].decode("utf-8", "replace")


def find_runner(root):
    """Locate the built runner binary, or explain how to build it."""
    env = os.environ.get("RUNNER_EXE")
    candidates = [env] if env else []
    candidates += [os.path.join(root, "runner.exe" if WINDOWS else "runner")]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    sys.exit(f"runner binary not found (looked in {candidates}); "
             f"build it with `make runner` or set RUNNER_EXE")
