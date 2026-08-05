# Tray / menu-bar controller

`runner --tray` puts a small grid icon in the macOS menu bar or the Windows
notification area. Clicking it shows **every runner instance live on the
machine** — however it was started — with its loaded models, and lets you
stop any of them or launch one pre-configured "desktop-managed" server.
macOS and Windows only in v1; on Linux `--tray` prints an honest error.

The CLI is completely unaffected when `--tray` is not passed. The only
thing the tray adds to normal runs is the discovery record described below,
and writing it is best-effort: a failure to write never affects the run.

## How instances are discovered

Every runner process in a *run* mode (one-shot generation, `--serve`,
`--tray` itself) writes one JSON record at startup:

```
~/.gridcore/runner/instances/<pid>.json          (macOS, Linux)
%APPDATA%\gridcore\runner\instances\<pid>.json   (Windows)
```

```json
{"pid": 4711, "started": 1785940000, "mode": "serve", "port": 8080,
 "version": "0.1.8-alpha",
 "models": [{"name": "trinity.gguf", "path": "/abs/path/trinity.gguf"}]}
```

Records are written atomically (tmp + rename) and removed at normal exit.
A crash leaves a stale file; **every reader sweeps** — any record whose pid
is no longer alive is deleted on sight, so the directory is self-healing.
Utility modes (`--quantize`, `--caps`, `--bench-json`, `--version`) do not
register.

A swap-mode server (`--serve` with a `name=path` model list) registers with
an empty models array because its resident set changes at runtime; the tray
asks the instance itself with `GET /v1/models` on loopback (500 ms budget)
each time the menu opens, and shows `(no model resident)` if nothing is
loaded or the call fails.

## The menu

- Header row: `gridcore-runner <version>`.
- One row per live instance: `mode  ·  :port  ·  pid` (`●` marks the
  tray-managed instance). Its submenu lists the model names and a **Stop**
  item.
- **Start default runner** — spawns `runner --serve -m <last_model> --port
  <port> <last_args>` as a detached child. With no model configured the row
  reads `Start… (no model configured)` and opens the native file picker
  (filtered to `*.gguf`); the choice is saved and the server starts.
- **Choose model…** — same picker, any time.
- **Launch at login** — checkbox, see autostart below.
- **Quit controller** — stops the tray **and the instance it manages,
  nothing else**. Instances you started by hand are never touched by quit.

Stop semantics: SIGTERM, 3 s grace, SIGKILL on macOS. On Windows v1 uses
`TerminateProcess` directly — there is no portable graceful signal for a
console process outside your own console group; the registry record is
swept either way, and the server holds no state that outlives the process.

The icon shows a filled center cell while at least one instance is running
and refreshes every 5 s.

Only one tray runs per machine: a second `--tray` exits with an error
naming the live controller's pid.

## Config file

```
~/.gridcore/runner/config.json          (macOS, Linux)
%APPDATA%\gridcore\runner\config.json   (Windows)
```

```json
{"last_model": "/abs/path/model.gguf", "last_args": "-c 4096", "port": 8080}
```

`last_args` is a space-split argument tail appended to the managed server's
command line. This field is the seam the calibration profiles
(Brain/Balanced/Fastest) will write into later; the tray itself only ever
records your file-picker choice in `last_model`.

## Autostart

- macOS: `~/Library/LaunchAgents/ai.gridcore.runner.tray.plist`
  (RunAtLoad, starts only the tray — never a model).
- Windows: `HKCU\Software\Microsoft\Windows\CurrentVersion\Run`, value
  `GridcoreTray`.

The checkbox creates or removes exactly that one artifact; presence of the
artifact is the state.

## Uninstall notes

Remove, if present:

- the autostart artifact above,
- `~/.gridcore/runner/` (POSIX) or `%APPDATA%\gridcore\runner\` (Windows) —
  contains only `config.json` and the self-healing `instances/` directory.

Nothing else is written anywhere.

## v1 simplifications (documented, deliberate)

- Windows stop is `TerminateProcess` (no `WM_CLOSE`/console-ctrl attempt).
- The Windows icon is drawn white-on-black regardless of taskbar theme
  (macOS uses a template image and adapts automatically).
- No load/unload of individual models from the menu — the runner has no
  unload API; you stop the instance instead.
- Registry refresh is poll-on-open plus a 5 s badge timer, not a
  filesystem watch.

## Headless validation seam

`GRIDCORE_TRAY_DUMP=1 runner --tray` prints the exact menu the backend
would render (indentation = submenus, `*` = clickable, `[x]` = checkbox
state) and exits without touching the GUI. CI and remote validation diff
this output; the human checklist only has to confirm the pixels.
