from __future__ import annotations

import ctypes
import json
import os
import shutil
import time
import uuid
from pathlib import Path

_RECORD = "owner.json"


class StartupLease:
    """Token-owned process lease with atomic claims and stale-owner recovery."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.owner_pid = os.getpid()
        self.token = uuid.uuid4().hex

    def acquire(self) -> bool:
        """Claim the lease, or return False while its recorded owner is alive."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(8):
            claim = self._prepare_claim()
            try:
                try:
                    claim.rename(self.path)
                    return True
                except (FileExistsError, OSError):
                    if not self.path.exists():
                        continue
                    record = self._read_record(self.path)
                    owner_pid = _record_owner_pid(record)
                    if owner_pid is not None and _process_alive(owner_pid):
                        return False
                    stale = self._move_aside(self.path)
                    if stale is not None:
                        _remove_tree(stale)
            finally:
                _remove_tree(claim)
        return False

    def release(self) -> None:
        """Release only when the current record still carries this token."""
        record = self._read_record(self.path)
        if record.get("token") != self.token:
            return
        stale = self._move_aside(self.path)
        if stale is not None:
            _remove_tree(stale)

    def _prepare_claim(self) -> Path:
        claim = self.path.with_name(f".{self.path.name}.{self.token}.claim")
        _remove_tree(claim)
        claim.mkdir()
        (claim / _RECORD).write_text(json.dumps({
            "owner_pid": self.owner_pid,
            "token": self.token,
            "created": time.time(),
        }), encoding="utf-8")
        return claim

    def _read_record(self, path: Path) -> dict:
        try:
            source = path / _RECORD if path.is_dir() else path
            data = json.loads(source.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _move_aside(self, path: Path) -> Path | None:
        stale = path.with_name(f".{path.name}.{uuid.uuid4().hex}.stale")
        try:
            path.rename(stale)
            return stale
        except OSError:
            return None


def _record_owner_pid(record: dict) -> int | None:
    # ``clu_pid`` accepts records written by the former Clu-local lease during
    # migration; Runner owns no other part of that legacy schema.
    value = record.get("owner_pid", record.get("clu_pid"))
    try:
        pid = int(value)
    except (TypeError, ValueError):
        return None
    return pid if pid > 0 else None


def _process_alive(pid: int) -> bool:
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel32.OpenProcess(0x1000, False, pid)
    if handle:
        kernel32.CloseHandle(handle)
        return True
    return ctypes.get_last_error() == 5


def _remove_tree(path: Path) -> None:
    try:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)
    except OSError:
        pass
