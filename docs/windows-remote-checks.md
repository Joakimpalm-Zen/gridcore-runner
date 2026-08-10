# Windows remote-check protocol (learned the hard way, 2026-08-10)

`ssh zen@192.168.1.123` reaches ZEN-GAMING and runs cmd.exe, but **stderr from
the remote process is not forwarded**. `gcc --version` (stdout) comes back;
compiler diagnostics (stderr) do not, and neither does `> file 2>&1` reliably
when chained.

Silence therefore means nothing. A file with `return zzz;` produced exactly the
same empty output as a clean compile.

Use the exit code, which is trustworthy:

    gcc ... & if errorlevel 1 (echo FAILED) else (echo CLEAN)

`%errorlevel%` is expanded at parse time in a one-liner and is NOT a substitute.

Two further traps:
  - `;` is not a command separator in cmd; use `&` (or `&&`).
  - Single-file `-fsyntax-only` checks are under-specified for this project:
    the Windows build needs the Makefile's flags and backend defines. Baseline
    `cuda.c` from before any local change also fails a bare check, so a bare
    pass/fail says nothing about a diff. Compare baseline against changed with
    identical flags, or build properly.

`make` is not installed in the MSYS2 environment there (`C:\msys64\usr\bin\make.exe`
absent), so `make OS=Windows_NT` — the check the plan says was used to build the
runner on that box — cannot currently run. Installing it restores real Windows
verification.
