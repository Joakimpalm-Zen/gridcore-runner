// Tray core gate: menu building against the registry, config-driven managed
// spawn, stop escalation, and quit-stops-managed-only semantics — all
// headless (links the stub backend on every platform). Re-invokes ITSELF as
// the managed child: spawn_managed launches self_exe with --serve args, and
// child mode writes a marker file with its pid+argv then idles until killed.
#include "instances.h"
#include "tray.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#include <direct.h>
#include <process.h>
#define getpid _getpid
#define setenv_compat(k, v) _putenv_s(k, v)
static void msleep(int ms) { Sleep(ms); }
#else
#include <signal.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#define setenv_compat(k, v) setenv(k, v, 1)
static void msleep(int ms) {
    struct timespec ts = { ms / 1000, (ms % 1000) * 1000000L };
    nanosleep(&ts, NULL);
}
#endif

static int fails = 0;
#define CHECK(cond, msg) do { if (!(cond)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, msg); fails++; } \
} while (0)

static char g_home[512];

static void marker_path(char *out, size_t cap) {
    snprintf(out, cap, "%s/spawned.txt", g_home);
}

static bool menu_has(const tray_item *it, int n, const char *needle) {
    for (int i = 0; i < n; i++)
        if (strstr(it[i].label, needle)) return true;
    return false;
}

int main(int argc, char **argv) {
    // ---- child mode: we ARE the "managed runner" the core spawned
    if (argc > 1 && strcmp(argv[1], "--serve") == 0) {
        const char *home = getenv("GRIDCORE_TEST_HOME");
        if (!home) return 2;
        char mp[600];
        snprintf(mp, sizeof mp, "%s/spawned.txt", home);
        FILE *f = fopen(mp, "wb");
        if (!f) return 2;
        fprintf(f, "%ld\n", (long)getpid());
        for (int i = 1; i < argc; i++) fprintf(f, "%s\n", argv[i]);
        fclose(f);
        msleep(30000);  // idle until the controller stops us
        return 0;
    }

    // ---- parent mode: private registry + config under a fake HOME
    snprintf(g_home, sizeof g_home,
#ifdef _WIN32
             "%s\\gridcore-tray-test-%ld", getenv("TEMP"), (long)getpid());
    _mkdir(g_home);
    setenv_compat("APPDATA", g_home);
#else
             "/tmp/gridcore-tray-test-%ld", (long)getpid());
    mkdir(g_home, 0755);
    setenv_compat("HOME", g_home);
#endif
    setenv_compat("GRIDCORE_TEST_HOME", g_home);

    // The core reads config.json once and then owns it in memory (its own
    // saves keep it current), so the file must exist BEFORE the first menu
    // build. instances_dir() is <root>/instances; config.json is its sibling.
    char cfg[700];
    const char *idir = instances_dir();
    CHECK(idir != NULL, "instances dir resolves under fake HOME");
    snprintf(cfg, sizeof cfg, "%s/../config.json", idir);
    // raw backslashes are invalid JSON escapes, so the Windows TEMP path
    // must go into config.json with forward slashes (fopen accepts both)
    char home_json[512];
    snprintf(home_json, sizeof home_json, "%s", g_home);
    for (char *c = home_json; *c; c++) if (*c == '\\') *c = '/';
    FILE *f = fopen(cfg, "wb");
    fprintf(f, "{\"last_model\": \"%s/fake-model.gguf\","
               " \"last_args\": \"-c 512\", \"port\": 8127}\n", home_json);
    fclose(f);
    char mdl[600];
    snprintf(mdl, sizeof mdl, "%s/fake-model.gguf", home_json);
    f = fopen(mdl, "wb"); fputs("x", f); fclose(f);

    // 1. empty registry: setup rows present, no stop rows
    tray_item items[128];
    int n = tray_menu_build(items, 128);
    CHECK(n > 0, "menu builds");
    CHECK(menu_has(items, n, "no runners active"), "empty state row present");
    CHECK(!menu_has(items, n, "Stop"), "no stop rows with empty registry");

    // 2. configured model: start row is offered and START_MANAGED spawns
    // self --serve with the configured argv
    CHECK(menu_has(items, n, "Start default runner"), "configured start row");
    CHECK(menu_has(items, n, "fake-model.gguf"), "start row names the model");

    tray_menu_act(TRAY_ACT_START_MANAGED, 0, NULL);

    char mp[600];
    marker_path(mp, sizeof mp);
    long child = 0;
    char args[512] = "";
    for (int t = 0; t < 60; t++) {  // up to 6 s for the child to come up
        FILE *m = fopen(mp, "rb");
        if (m) {
            char line[512];
            if (fgets(line, sizeof line, m)) child = atol(line);
            while (fgets(line, sizeof line, m)) {
                line[strcspn(line, "\r\n")] = 0;
                strncat(args, line, sizeof args - strlen(args) - 2);
                strncat(args, " ", sizeof args - strlen(args) - 1);
            }
            fclose(m);
            if (child > 0) break;
        }
        msleep(100);
    }
    CHECK(child > 0, "managed child spawned and wrote its marker");
    CHECK(strstr(args, "--serve") != NULL, "child got --serve");
    CHECK(strstr(args, "fake-model.gguf") != NULL, "child got the configured model");
    CHECK(strstr(args, "--port 8127") != NULL, "child got the configured port");
    CHECK(strstr(args, "-c 512") != NULL, "child got last_args tail");
    CHECK(instance_pid_alive(child), "child is alive before quit");

    // 2b. the child never registers (it is not a real runner), so the core
    // must report STARTING: visible row, cancel option, no live Start row
    n = tray_menu_build(items, 128);
    CHECK(menu_has(items, n, "starting fake-model.gguf"), "starting row shown");
    CHECK(menu_has(items, n, "Cancel start"), "cancel available while starting");
    CHECK(menu_has(items, n, "Default runner: starting"), "start row disabled to starting");
    CHECK(!menu_has(items, n, "Start default runner"), "no live start row while starting");

    // 2c. kill the child behind the core's back: menu must flip to the
    // exited warning with a log row and a restart offer
#ifdef _WIN32
    {
        HANDLE h = OpenProcess(PROCESS_TERMINATE, FALSE, (DWORD)child);
        CHECK(h != NULL, "can open child to simulate a crash");
        if (h) { TerminateProcess(h, 1); CloseHandle(h); }
    }
#else
    kill((pid_t)child, SIGKILL);
#endif
    bool exited_row = false;
    for (int t = 0; t < 50 && !exited_row; t++) {
        msleep(100);
        n = tray_menu_build(items, 128);
        exited_row = menu_has(items, n, "exited");
    }
    CHECK(exited_row, "crash surfaces as the exited warning");
    CHECK(menu_has(items, n, "View log"), "log row offered after exit");
    CHECK(menu_has(items, n, "Restart default runner"), "restart offered after exit");

    // 2d. restart spawns a fresh child and clears the warning
    remove(mp);
    tray_menu_act(TRAY_ACT_START_MANAGED, 0, NULL);
    child = 0;
    for (int t = 0; t < 60 && child == 0; t++) {
        FILE *m = fopen(mp, "rb");
        if (m) {
            char line[512];
            if (fgets(line, sizeof line, m)) child = atol(line);
            fclose(m);
        }
        msleep(100);
    }
    CHECK(child > 0, "restart spawned a fresh child");
    n = tray_menu_build(items, 128);
    CHECK(!menu_has(items, n, "exited"), "warning cleared once restarted");

    // 3. a foreign record must survive quit; only the managed child stops
    char foreign[700];
    snprintf(foreign, sizeof foreign, "%s/%ld.json", idir, (long)getpid());
    f = fopen(foreign, "wb");
    fprintf(f, "{\"pid\": %ld, \"started\": 1, \"mode\": \"serve\","
               " \"port\": 9001, \"version\": \"t\", \"models\": []}\n",
            (long)getpid());  // our own pid: always alive, never our child
    fclose(f);

    tray_menu_act(TRAY_ACT_QUIT, 0, NULL);
    CHECK(tray_should_quit(), "quit flag set");

    bool child_dead = false;
    for (int t = 0; t < 50; t++) {
        if (!instance_pid_alive(child)) { child_dead = true; break; }
        msleep(100);
    }
    CHECK(child_dead, "managed child stopped by quit");

    int nl = 0;
    instance_rec *r = instances_list(&nl);
    bool foreign_alive = false;
    for (int i = 0; i < nl; i++)
        if (r[i].port == 9001) foreign_alive = true;
    instances_list_free(r, nl);
    CHECK(foreign_alive, "foreign instance record untouched by quit");

    // 4. after a user-initiated quit the menu is back to a clean Start —
    // no lingering exited warning for an instance the user stopped
    n = tray_menu_build(items, 128);
    CHECK(menu_has(items, n, "Start default runner"), "start row back after quit");
    CHECK(!menu_has(items, n, "exited"), "no exited warning after intentional stop");

    remove(foreign);
    remove(mp);
    remove(cfg);
    remove(mdl);

    if (fails) { fprintf(stderr, "test_tray_core: %d FAILURES\n", fails); return 1; }
    printf("tray core tests ok\n");
    return 0;
}
