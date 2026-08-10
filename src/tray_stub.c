// Fallback backend for platforms without a tray implementation (Linux in
// v1). Keeps --tray a defined flag everywhere with an honest error.
#include "tray.h"
#include <stdio.h>

int tray_platform_run(void) {
    fprintf(stderr, "error: --tray is not available on this platform "
                    "(macOS and Windows only in v1)\n");
    return 1;
}

bool tray_platform_autostart_get(void) { return false; }

// no offscreen renderer in the stub; the seam reports failure rather than
// writing nothing and claiming success
bool tray_platform_icon_dump(const char *dir, int px) {
    (void)dir; (void)px;
    return false;
}
bool tray_platform_autostart_set(bool on) { (void)on; return false; }
