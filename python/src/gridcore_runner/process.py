from __future__ import annotations

import os
import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from .endpoint import RunnerEndpoint


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


def model_registry_argument(models: Mapping[str, str | Path]) -> str:
    if not models:
        raise ValueError("model registry must not be empty")
    entries: list[str] = []
    for name in sorted(models):
        path = str(models[name])
        if not name or any(char in name for char in ",="):
            raise ValueError(f"invalid model registry name: {name!r}")
        if not path or "," in path:
            raise ValueError(f"invalid model registry path: {path!r}")
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
        if self.alive():
            return True
        self.process = self._spawn(build_server_args(self.launch))
        endpoint = self._endpoint_factory(self.base_url)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self.alive():
                return False
            if endpoint.healthy(timeout=min(2.0, max(interval, 0.1))):
                return True
            time.sleep(interval)
        return False

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
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        return subprocess.Popen(
            args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=str(Path(args[0]).resolve().parent),
            creationflags=flags,
        )
