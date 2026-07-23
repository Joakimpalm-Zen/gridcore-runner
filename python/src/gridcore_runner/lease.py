from __future__ import annotations

import ctypes
import json
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path

_RECORD = "owner.json"


class StartupLease:
    """Token-owned process lease with atomic claims and stale-owner recovery."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.owner_pid = os.getpid()
        self.owner_start = _process_start_time(self.owner_pid)
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
                    if _still_owned(record):
                        return False
                    stale = self._move_aside(self.path)
                    if stale is not None:
                        # between judging the record stale and moving it, a
                        # rival may have reclaimed and re-claimed: only delete
                        # the exact record we judged; put a fresh one back
                        moved = self._read_record(stale)
                        if moved.get("token") != record.get("token"):
                            try:
                                stale.rename(self.path)
                            except OSError:
                                pass  # a third claim landed; leave theirs be
                            continue
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
            "owner_start": self.owner_start,
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


def _still_owned(record: dict) -> bool:
    """True only if the record's owner process is genuinely still running.

    Comparing the PID alone is not enough: after an unclean exit the OS can
    reuse that PID for an unrelated process, which would keep the lease "live"
    forever and block startup (RNR-017). So when the record carries the owner's
    process start time, it must also match the live process's start time; a
    mismatch means the PID was reused and the lease is stale. Records without a
    start time are honoured PID-only as a bounded migration compatibility.
    """
    owner_pid = _record_owner_pid(record)
    if owner_pid is None or not _process_alive(owner_pid):
        return False
    rec_start = record.get("owner_start")
    if rec_start is None:
        return True  # legacy record: PID-only during migration
    live_start = _process_start_time(owner_pid)
    if live_start is None:
        return True  # cannot read the start time to disprove ownership — stay safe
    return str(live_start) == str(rec_start)  # differ => PID reused => not the owner


def _process_start_time(pid: int) -> str | None:
    """A stable per-process start identity, or None if it cannot be read.

    Combined with the PID this distinguishes an original owner from an unrelated
    process that later inherited the same PID.
    """
    if os.name == "nt":
        try:
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
            kernel32.OpenProcess.restype = ctypes.c_void_p
            handle = kernel32.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFO
            if not handle:
                return None
            try:
                creation = ctypes.c_ulonglong(0)
                exit_t = ctypes.c_ulonglong(0)
                kern_t = ctypes.c_ulonglong(0)
                user_t = ctypes.c_ulonglong(0)
                ok = kernel32.GetProcessTimes(
                    ctypes.c_void_p(handle), ctypes.byref(creation),
                    ctypes.byref(exit_t), ctypes.byref(kern_t), ctypes.byref(user_t))
                return str(creation.value) if ok else None
            finally:
                kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
                kernel32.CloseHandle(ctypes.c_void_p(handle))
        except OSError:
            return None
    # Linux: field 22 of /proc/<pid>/stat is starttime (clock ticks since boot).
    # The comm field (2) may contain spaces and parens, so split after the last
    # ')': the remaining fields start at field 3 (state), making starttime
    # index 19.
    try:
        with open(f"/proc/{pid}/stat", "rb") as f:
            data = f.read()
        rparen = data.rfind(b")")
        if rparen >= 0:
            fields = data[rparen + 2:].split()
            return fields[19].decode()
    except (OSError, IndexError, ValueError):
        pass
    # macOS / BSD: no /proc — ask ps for the start timestamp (second precision,
    # which together with the PID is enough to catch reuse).
    try:
        out = subprocess.run(["ps", "-o", "lstart=", "-p", str(pid)],
                             capture_output=True, text=True, timeout=5)
        stamp = out.stdout.strip()
        return stamp or None
    except (OSError, subprocess.SubprocessError):
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
        except PermissionError:
            return True  # exists, owned by someone else
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
