# Tray controller v1 — validation report (branch `tray-controller`)

Date: 2026-08-05. Goal: docs/plans/goal-tray-controller-2026-08-05.md in
the suite repo. Branch pushed, **not merged** — owner merges after the
human checklists below.

## What was built

- `src/instances.{h,c}` — discovery registry (gate 1).
- `src/tray.{h,c}` — portable controller core: menu model, config,
  managed-instance spawn/stop, quit semantics, `/v1/models` enrichment,
  single-tray guard, `GRIDCORE_TRAY_DUMP` headless seam.
- `src/tray_macos.m` — NSStatusItem backend, accessory policy, code-drawn
  3×3 template icon with running dot, NSOpenPanel (UTType), LaunchAgent.
- `src/tray_win.c` — Shell_NotifyIcon backend, hidden message window,
  console subsystem kept (FreeConsole in tray mode), GDI icon, HKCU Run.
- `src/tray_stub.c` — honest `--tray` error on Linux.
- `tests/test_instances.c`, `tests/test_tray_core.c` — both in `make test`.

## Gate results

| Gate | Result |
|---|---|
| 1. Registry unit tests, macOS | PASS (6 groups) |
| 1. Registry unit tests, Windows/3070 | PASS |
| 1. Live registry proof (serve appears/vanishes, CLI one-shot, SIGTERM-stale swept) | PASS (M1) |
| 2. Full `make test`, macOS, with registry hooks + tray sources | PASS (RC=0) |
| 2. Full `make test`, Windows/3070 (UCRT64, `PYTHON="py -3"`) | PASS (RC=0) |
| 3. macOS live: tray runs, registers `tray` record, second instance refused | PASS (M1) |
| 3. macOS: live serve row + model submenu correct (headless dump) | PASS |
| 3. macOS: swap-serve models fetched via `/v1/models` (live 2-model server) | PASS |
| 3. macOS: managed spawn/stop/quit semantics | PASS (test_tray_core) |
| 3. macOS: LaunchAgent plist create/read/remove | PASS (harness, fake HOME) |
| 4. Windows: build clean (gcc 16.1.0 UCRT64), tray tests pass | PASS |
| 4. Windows: live tray registers, second instance refused, kill → swept | PASS (via ssh session) |
| 4. Windows: HKCU Run autostart set/get/remove | PASS (harness; key left clean) |

Suite counts on Windows tail: 21 + 9 pytest green in the final sub-makes,
C tests all `ok`. One pre-existing Makefile bug was fixed on the branch:
recursive `$(MAKE) ... PYTHON=$(PYTHON)` was unquoted and exploded with
`PYTHON="py -3"` (the launcher is the only Python on the 3070).

## Fixed during validation

- Windows TEMP paths written raw into test JSON were invalid escapes
  (`\U`, `\g`) — test now writes forward slashes. Production config writes
  were never affected (`cfg_save` escapes via `sb_esc`).
- NSOpenPanel `allowedFileTypes` deprecation → `allowedContentTypes` +
  UniformTypeIdentifiers framework (Darwin-only link flag).
- Owner click-through round 1: no feedback between Start and the record
  appearing (multi-second model load), Start-while-running a silent no-op,
  tray listed itself, running dot always lit → explicit managed lifecycle
  (starting / running / exited+log / stopped), tray filtered from rows
  and dot, managed output captured to `managed.log`.
- Owner click-through round 2: instance submenu (models + Stop) could not
  be opened at all on macOS — AppKit refuses to open a submenu on a
  disabled item; parent rows with submenus are now enabled. Headless
  checks could never have caught this: the menu model was correct, the
  platform rendering swallowed the interaction.

## Human checklist — macOS (owner, at the screen)

Owner click-through 2026-08-05 (two rounds of feedback, both fixed on the
branch — see "Fixed during validation"):

- [x] Grid icon visible in the menu bar; menu opens and matches the dump.
- [x] Start default runner: starting → running lifecycle visible; managed
      ● row appears.
- [x] Instance submenu opens (model list) and **Stop** works.
- [x] **Choose model…** picker (used to select Trinity-Nano from Downloads).
- [ ] **Launch at login** toggle (not yet exercised at the screen; plist
      lifecycle proven by harness).
- [ ] **Quit controller** stops the ● instance only (semantics proven by
      test_tray_core; not yet clicked at the screen).

## Human checklist — Windows (owner, at the 3070's screen)

Build is at `C:\Users\zen\tray-build\runner.exe`. Start with
`runner.exe --tray` from Explorer or a terminal (console detaches):

- [ ] Grid icon appears in the notification area (may hide under the ^
      overflow chevron).
- [ ] Left- or right-click opens the menu; rows match the dump output.
- [ ] Icon is legible on your taskbar theme (v1 draws white-on-black
      regardless of theme — judge whether that is acceptable).
- [ ] Stop on a hand-started `runner.exe --serve` row kills it
      (TerminateProcess — expect immediate, not graceful).
- [ ] **Choose model…** opens the Win32 file dialog, filtered to .gguf.
- [ ] **Launch at login** creates/removes `GridcoreTray` under HKCU Run
      (`reg query HKCU\Software\Microsoft\Windows\CurrentVersion\Run`).
- [ ] **Quit controller** removes the icon and stops only the ● instance.
- [ ] Reboot test (optional): with autostart on, the icon is present
      after sign-in and no console window flashes.

## Deviations from the goal spec (all documented in tray-controller.md)

- Windows stop skips the WM_CLOSE/GenerateConsoleCtrlEvent attempt and
  goes straight to TerminateProcess. Attaching to a foreign console to
  deliver a ctrl event is the classic Win32 tarpit the STOP rules warn
  about; the runner keeps no cross-request state, so a hard stop loses
  nothing. Revisit only if a future runner persists state at shutdown.
- Windows icon does not read `SystemUsesLightTheme` in v1.
- Instance rows read `mode · :port · pid` with models in the submenu
  (spec sketched models inline; submenu keeps long names readable).

## Cleanup state

- 3070: checkout at `C:\Users\zen\tray-build` (kept — the click checklist
  needs it), helper scripts `build*.sh`/`gate4*.sh`/`greplog.sh`/
  `autostart_win.c` in `C:\Users\zen` (delete freely). HKCU Run value:
  removed. Registry dir: empty.
- M1: tray left running for the checklist; `~/.gridcore/runner/` contains
  its record and nothing else.
