from __future__ import annotations

import os
import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from .endpoint import RunnerEndpoint


def spawn_detached(args: list[str], *, cwd: str | Path | None = None):
    """Popen a child isolated from the parent's console signals, so a Ctrl+C
    aimed at the supervisor never tears down the server underneath it."""
    kwargs: dict[str, Any] = {
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if cwd is not None:
        kwargs["cwd"] = str(cwd)
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    return subprocess.Popen(args, **kwargs)


def query_system_capabilities(
    executable: str | Path,
    *,
    run: Callable[..., Any] | None = None,
    timeout: float = 15,
) -> dict[str, Any]:
    """Ask the installed Runner binary what this build can execute."""
    execute = run or subprocess.run
    completed = execute(
        [str(executable), "--caps"],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=True,
    )
    data = json.loads(completed.stdout)
    required = {"os", "arch", "cpu_cores", "ram_bytes", "gpu", "quants", "gpu_quants"}
    if not isinstance(data, dict) or not required.issubset(data):
        raise ValueError("runner --caps returned an incomplete capability report")
    return data


# Must match the native registry's fixed limits (reg_entry in server.c) so the
# builder rejects a spec the server would truncate or drop rather than emitting
# it (RNR-014).
_MAX_MODELS = 16
_MAX_NAME = 63
_MAX_PATH = 1023


def model_registry_argument(models: Mapping[str, str | Path]) -> str:
    if not models:
        raise ValueError("model registry must not be empty")
    if len(models) > _MAX_MODELS:
        raise ValueError(
            f"too many models ({len(models)}); the server accepts at most {_MAX_MODELS}")
    entries: list[str] = []
    for name in sorted(models):
        path = str(models[name])
        if not name or any(char in name for char in ",="):
            raise ValueError(f"invalid model registry name: {name!r}")
        if len(name) > _MAX_NAME:
            raise ValueError(
                f"model registry name too long ({len(name)} > {_MAX_NAME}): {name!r}")
        if not path or "," in path:
            raise ValueError(f"invalid model registry path: {path!r}")
        if len(path) > _MAX_PATH:
            raise ValueError(
                f"model registry path too long ({len(path)} > {_MAX_PATH}) for {name!r}")
        entries.append(f"{name}={path}")
    return ",".join(entries)


@dataclass(frozen=True)
class ServerLaunch:
    executable: str | Path
    model: str | Path | Mapping[str, str | Path]
    port: int
    context_size: int = 4096
    reserve_pct: int = 0
    threads: int | None = None
    gpu: str = "auto"
    parent_pid: int = field(default_factory=os.getpid)
    ttl: int | None = None
    extra_args: tuple[str, ...] = ()


def build_server_args(launch: ServerLaunch) -> list[str]:
    model = (
        model_registry_argument(launch.model)
        if isinstance(launch.model, Mapping)
        else str(launch.model)
    )
    args = [
        str(launch.executable),
        "--serve",
        "--port", str(launch.port),
        "-c", "0" if launch.reserve_pct > 0 else str(launch.context_size),
        "-m", model,
        "--parent-pid", str(launch.parent_pid),
    ]
    if launch.reserve_pct > 0:
        args += ["--reserve", str(launch.reserve_pct)]
    if launch.threads is not None:
        args += ["-t", str(launch.threads)]
    if launch.gpu == "off":
        args += ["--gpu", "off"]
    if launch.ttl is not None:
        args += ["--ttl", str(launch.ttl)]
    return args + list(launch.extra_args)


class ManagedRunner:
    """Owns one Runner child; caller policy decides when it should exist."""

    def __init__(
        self,
        launch: ServerLaunch,
        *,
        spawn: Callable[[list[str]], Any] | None = None,
        endpoint_factory: Callable[[str], RunnerEndpoint] | None = None,
    ):
        self.launch = launch
        self._spawn = spawn or self._default_spawn
        self._endpoint_factory = endpoint_factory or RunnerEndpoint
        self.process: Any = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.launch.port}"

    def start(self, *, timeout: float = 600, interval: float = 0.5) -> bool:
        """Spawn a Runner child and wait until it answers, returning whether it
        did.

        `start()` owns the child for the whole call: False means nothing is
        left running. A runner that never became answerable — still loading
        weights when the deadline expired, wedged, or abandoned by a
        KeyboardInterrupt — is terminated before returning, because the caller
        has no handle to it and would otherwise leak a process holding VRAM
        for as long as the box stays up.
        """
        if self.alive():
            return True
        self.process = self._spawn(build_server_args(self.launch))
        endpoint = self._endpoint_factory(self.base_url)
        deadline = time.monotonic() + timeout
        started = False
        try:
            while time.monotonic() < deadline:
                if not self.alive():
                    break
                if endpoint.healthy(timeout=min(2.0, max(interval, 0.1))):
                    started = True
                    break
                time.sleep(interval)
        finally:
            if not started:
                self.stop()
        return started

    def alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def stop(self, *, timeout: float = 10) -> None:
        process = self.process
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        self.process = None

    @staticmethod
    def _default_spawn(args: list[str]):
        return spawn_detached(args, cwd=Path(args[0]).resolve().parent)
