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

The earlier note that `make` was absent is stale. Rechecked 2026-08-13:
`C:\msys64\usr\bin\make.exe` is present (GNU Make 4.4.1), and a fresh detached
Runner checkout completed the native MinGW/UCRT gate at `aa359e2` with
`make OS=Windows_NT -j2 test` (exit 0). The run passed the native, CUDA, and
tray-core gates; Python totals were client 30 passed/1 skipped, main 162/17,
MoE 28/0, and prune 9/0. No validation runner or tray process survived it.

That run found two Windows-only test-harness defects before it went green:
`d302a7a` makes the compatibility test's executable stub native on Windows,
and `aa359e2` stops the MTP admission test from raising a detached tray that
locks `runner.exe` before a later relink. Both were reproduced red on Windows,
then verified by targeted tests and the clean full gate. Do not install or
change the machine based on the old absence claim; verify the current tool
directly before diagnosing the setup.

The headless tray seams are also verified: menu dumping passed, and the idle,
loaded, and running BMPs have distinct hashes and pixel counts. Pixel analysis
confirmed that loaded preserves the idle icon's two opposing sweeps while
filling the core, and running draws the four-segment ring. This proves the GDI
raster states, not visibility or click behavior in the live Windows taskbar.
