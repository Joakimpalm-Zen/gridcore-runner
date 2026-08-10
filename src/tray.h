// Desktop tray/menu-bar controller (v1).
//
// All logic lives in the portable core (tray.c): reading the discovery
// registry, owning the one desktop-managed instance, config, autostart
// bookkeeping. Platform backends (tray_macos.m, tray_win.c) only render
// the menu the core describes and forward clicks back. The CLI runner is
// untouched unless --tray is passed.
#ifndef RUNNER_TRAY_H
#define RUNNER_TRAY_H

#include <stdbool.h>

// --tray entry point: initializes the core, then runs the platform loop.
// Returns a process exit code. Refuses to run if another tray is alive.
int tray_main(void);

// ------------------------------------------------------------- menu model

enum {
    TRAY_K_LABEL = 0,     // disabled informational row
    TRAY_K_ACTION,        // clickable row
    TRAY_K_CHECK,         // clickable row with a checkmark state
    TRAY_K_SEP,           // separator
    TRAY_K_SUB_BEGIN,     // following items nest under the previous row
    TRAY_K_SUB_END,
};

enum {
    TRAY_ACT_NONE = 0,
    TRAY_ACT_START_MANAGED,   // spawn --serve with the configured model/args
    TRAY_ACT_PICK_MODEL,      // native file picker, saves last_model
    TRAY_ACT_STOP_INSTANCE,   // arg = pid
    TRAY_ACT_TOGGLE_AUTOSTART,
    TRAY_ACT_OPEN_LOG,        // open the managed instance's log file
    TRAY_ACT_QUIT,
};

typedef struct {
    char label[512];
    int  kind;     // TRAY_K_*
    int  action;   // TRAY_ACT_*
    long arg;      // pid for STOP_INSTANCE
    bool checked;  // TRAY_K_CHECK state
} tray_item;

// Build the current menu into items (at most cap). Called by the backend
// every time the menu is about to open. Returns the item count.
int  tray_menu_build(tray_item *items, int cap);

// A menu row was clicked. STOP/START/QUIT/PICK side effects happen here.
// For TRAY_ACT_PICK_MODEL the backend passes the chosen path (or NULL if
// the user cancelled); every other action ignores `text`.
void tray_menu_act(int action, long arg, const char *text);

// True while ≥1 runner instance is live — the backend shows the badge dot.
bool tray_any_running(void);

// Make sure a tray controller exists, without becoming one. Spawns a detached
// `<self> --tray` when no tray record is live, and does nothing when one is.
//
// Spawning rather than hosting is what keeps this cheap: AppKit demands the
// main thread, so a tray inside a serving process would mean moving the server
// off it. A separate process needs none of that, and the tray already lists
// every registered runner, so the serve it was launched beside shows up with no
// extra plumbing.
//
// The child is detached (its own session / process group) on purpose: killing
// the server with Ctrl-C must not take the tray with it, because the whole
// point is that you can load the next model from it afterwards. Racing callers
// are harmless — the child runs the one-tray-per-machine guard itself and the
// loser exits.
void tray_ensure_running(void);

// Which glyph the menu-bar / notification-area icon should draw. Three states,
// each sourced from something real rather than inferred:
//   IDLE    no runner instance registered at all
//   LOADED  a runner is up with a model resident, nothing in flight
//   RUNNING inference is actually in flight (active_requests from /health)
// LOADED is the honest answer whenever the busy question cannot be asked — a
// server that is up but unreachable on loopback is still a loaded model, and
// claiming it is working would be the one reading that is definitely wrong.
typedef enum {
    TRAY_ICON_IDLE = 0,
    TRAY_ICON_LOADED,
    TRAY_ICON_RUNNING,
} tray_icon_state;
tray_icon_state tray_icon(void);

// Design-review / human-click-checklist seam, the icon sibling of
// GRIDCORE_TRAY_DUMP: render all three states to <dir>/tray-{idle,loaded,
// running}.png at `px` square and return true. A menu-bar glyph is 18 px of
// arcs that nobody can review by reading the drawing code, and this is the
// only way to look at it without a display. Backends with no offscreen
// renderer return false.
bool tray_platform_icon_dump(const char *dir, int px);

// True when the platform loop should exit (set by TRAY_ACT_QUIT).
bool tray_should_quit(void);

// ------------------------------------------- provided by platform backends

// Run the icon + menu loop until tray_should_quit(). Returns exit code.
int  tray_platform_run(void);
// Autostart (login item) state, platform-specific storage.
bool tray_platform_autostart_get(void);
bool tray_platform_autostart_set(bool on);

#endif // RUNNER_TRAY_H
