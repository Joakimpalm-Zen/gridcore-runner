// Tray controller core — see tray.h. Portable C; no GUI includes here.
#include "tray.h"
#include "instances.h"
#include "json.h"
#include "runner.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <process.h>
#define getpid _getpid
typedef SOCKET sock_t;
#define sock_close closesocket
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
extern char **environ;
typedef int sock_t;
#define INVALID_SOCKET (-1)
#define sock_close close
#endif

// ------------------------------------------------------------------ config

// <config_dir>/config.json — sibling of instances/. Format:
// {"last_model": "...", "last_args": "...", "port": 8080}
// last_args is a space-split argv tail appended to the managed spawn; the
// calibration profiles will write this same field later (the whole seam).
static struct {
    char last_model[1024];
    char last_args[512];
    int  port;
} g_cfg = { "", "", 8080 };

static void cfg_path(char *out, size_t cap) {
    const char *d = instances_dir();  // ensures the tree exists
    if (!d) { out[0] = 0; return; }
    // instances_dir returns <root>/instances; config sits at <root>
    size_t l = strlen(d);
    const char *tail =
#ifdef _WIN32
        "\\instances";
#else
        "/instances";
#endif
    size_t tl = strlen(tail);
    if (l > tl && strcmp(d + l - tl, tail) == 0) l -= tl;
    snprintf(out, cap, "%.*s%cconfig.json", (int)l, d,
#ifdef _WIN32
             '\\'
#else
             '/'
#endif
    );
}

// What the loaded config was read from. The tray used to load once per
// process, so a hand-edited config.json needed a tray restart to take effect
// (workmac issue 5) — and the config is exactly the file a person edits by
// hand, since it is where a model path and port live.
//
// Size is compared alongside the timestamp because st_mtime is whole seconds
// on some filesystems: an edit landing in the same second as our read is
// invisible to the timestamp alone, and a config edit almost always changes
// the length.
static bool   g_cfg_loaded = false;
static time_t g_cfg_mtime = 0;
static long   g_cfg_size = -1;

// True when the file on disk differs from what g_cfg holds. A config that
// cannot be stat'd (deleted, or never written) counts as unchanged: the last
// known-good values are better than blanking the menu.
static bool cfg_is_stale(const char *path) {
    struct stat st;
    if (!path[0] || stat(path, &st) != 0) return false;
    if (g_cfg_loaded && st.st_mtime == g_cfg_mtime &&
        (long)st.st_size == g_cfg_size) return false;
    g_cfg_mtime = st.st_mtime;
    g_cfg_size  = (long)st.st_size;
    return true;
}

static void cfg_load(void) {
    char path[1200];
    cfg_path(path, sizeof path);
    if (g_cfg_loaded && !cfg_is_stale(path)) return;
    g_cfg_loaded = true;
    FILE *f = path[0] ? fopen(path, "rb") : NULL;
    if (!f) return;
    char buf[4096];
    size_t n = fread(buf, 1, sizeof buf - 1, f);
    fclose(f);
    buf[n] = 0;
    jv *v = json_parse(buf, n);
    if (!v) return;
    snprintf(g_cfg.last_model, sizeof g_cfg.last_model, "%s",
             jv_str(jv_get(v, "last_model"), ""));
    snprintf(g_cfg.last_args, sizeof g_cfg.last_args, "%s",
             jv_str(jv_get(v, "last_args"), ""));
    g_cfg.port = (int)jv_num(jv_get(v, "port"), 8080);
    jv_free(v);
}

static void cfg_save(void) {
    char path[1200];
    cfg_path(path, sizeof path);
    if (!path[0]) return;
    FILE *f = fopen(path, "wb");
    if (!f) return;
    sbuf b = {0};
    sb_lit(&b, "{\"last_model\": \"");
    sb_esc(&b, g_cfg.last_model, strlen(g_cfg.last_model));
    sb_lit(&b, "\", \"last_args\": \"");
    sb_esc(&b, g_cfg.last_args, strlen(g_cfg.last_args));
    sb_fmt(&b, "\", \"port\": %d}\n", g_cfg.port);
    if (!b.failed) fwrite(b.s, 1, b.n, f);
    free(b.s);
    fclose(f);
    // Our own write is not an external edit. Re-stamping here keeps the next
    // menu build from re-reading a file whose contents it already holds.
    struct stat st;
    if (stat(path, &st) == 0) {
        g_cfg_mtime = st.st_mtime;
        g_cfg_size  = (long)st.st_size;
        g_cfg_loaded = true;
    }
}

// --------------------------------------------------------- managed process

static long g_managed_pid = 0;
static bool g_managed_spawned = false;  // spawned since the last user stop
static bool g_quit = false;

// Is the configured model still on disk? Publishing a checkpoint and
// reclaiming the local copy is routine, and it silently arms a config that can
// never start again. Checked rather than remembered, so restoring the file
// heals the menu with no bookkeeping.
static bool model_file_present(void) {
    if (!g_cfg.last_model[0]) return false;
    struct stat st;
    return stat(g_cfg.last_model, &st) == 0 && S_ISREG(st.st_mode);
}

// Managed-instance lifecycle, derived — never stored — so the menu can't
// show a state the world has moved past. STARTING = the child process is
// alive but has not yet registered (model still loading); EXITED = it died
// without the user stopping it (start failure or crash), which keeps a
// "view log" row up until the user starts again; MISSING = the configured
// model no longer exists, so no start can succeed and calling it an exit
// would blame the engine for a stale pointer.
enum { MG_OFF, MG_STARTING, MG_RUNNING, MG_EXITED, MG_MISSING };

static int managed_state(const instance_rec *recs, int nrecs) {
#ifndef _WIN32
    // reap a dead child, else the zombie still answers kill(pid, 0)
    if (g_managed_pid > 0) waitpid((pid_t)g_managed_pid, NULL, WNOHANG);
#endif
    if (g_managed_pid <= 0) {
        // Derived exactly like the rest: a missing model outranks "exited",
        // because it explains the exit and survives it.
        if (g_cfg.last_model[0] && !model_file_present()) return MG_MISSING;
        return g_managed_spawned ? MG_EXITED : MG_OFF;
    }
    if (!instance_pid_alive(g_managed_pid))
        return MG_EXITED;
    for (int i = 0; i < nrecs; i++)
        if (recs[i].pid == g_managed_pid) return MG_RUNNING;
    return MG_STARTING;
}

// <root>/managed.log — the spawned server's stdout+stderr, truncated on
// each start, so "it exited" always has a story next to it.
static void log_path(char *out, size_t cap) {
    const char *d = instances_dir();
    if (!d) { out[0] = 0; return; }
    size_t l = strlen(d);
#ifdef _WIN32
    const char *tail = "\\instances";
#else
    const char *tail = "/instances";
#endif
    size_t tl = strlen(tail);
    if (l > tl && strcmp(d + l - tl, tail) == 0) l -= tl;
    snprintf(out, cap, "%.*s%cmanaged.log", (int)l, d,
#ifdef _WIN32
             '\\'
#else
             '/'
#endif
    );
}

static void open_log(void) {
    char lp[1200], cmd[1400];
    log_path(lp, sizeof lp);
    if (!lp[0]) return;
#ifdef _WIN32
    snprintf(cmd, sizeof cmd, "notepad.exe \"%s\"", lp);
    STARTUPINFOA si = { .cb = sizeof si };
    PROCESS_INFORMATION pi;
    if (CreateProcessA(NULL, cmd, NULL, NULL, FALSE, 0, NULL, NULL, &si, &pi)) {
        CloseHandle(pi.hThread);
        CloseHandle(pi.hProcess);
    }
#elif defined(__APPLE__)
    snprintf(cmd, sizeof cmd, "open \"%s\"", lp);
    system(cmd);
#else
    snprintf(cmd, sizeof cmd, "xdg-open \"%s\" >/dev/null 2>&1 &", lp);
    system(cmd);
#endif
}

static void self_exe(char *out, size_t cap);

static void spawn_managed(void) {
    cfg_load();
    if (!g_cfg.last_model[0]) return;
    // Refuse a spawn that cannot succeed. Letting it run would fail at GGUF
    // open and land in MG_EXITED, which is indistinguishable from a crash and
    // sends the user to the log for a cause the menu already knows.
    if (!model_file_present()) return;
    char exe[1200];
    self_exe(exe, sizeof exe);
    char portbuf[16];
    snprintf(portbuf, sizeof portbuf, "%d", g_cfg.port);

    // argv: exe --serve -m model --port N [split last_args...]
    char *argv[40];
    int ac = 0;
    argv[ac++] = exe;
    argv[ac++] = "--serve";
    argv[ac++] = "-m";
    argv[ac++] = g_cfg.last_model;
    argv[ac++] = "--port";
    argv[ac++] = portbuf;
    static char args_copy[512];
    snprintf(args_copy, sizeof args_copy, "%s", g_cfg.last_args);
    for (char *t = strtok(args_copy, " "); t && ac < 38; t = strtok(NULL, " "))
        argv[ac++] = t;
    argv[ac] = NULL;

    char lp[1200];
    log_path(lp, sizeof lp);

#ifdef _WIN32
    // rebuild a command line; CreateProcess wants one string
    char cmd[2600] = "";
    for (int i = 0; argv[i]; i++) {
        strncat(cmd, "\"", sizeof cmd - strlen(cmd) - 1);
        strncat(cmd, argv[i], sizeof cmd - strlen(cmd) - 1);
        strncat(cmd, "\" ", sizeof cmd - strlen(cmd) - 1);
    }
    SECURITY_ATTRIBUTES sa = { sizeof sa, NULL, TRUE };
    HANDLE hlog = lp[0] ? CreateFileA(lp, GENERIC_WRITE, FILE_SHARE_READ, &sa,
                                      CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL)
                        : INVALID_HANDLE_VALUE;
    STARTUPINFOA si = { .cb = sizeof si };
    if (hlog != INVALID_HANDLE_VALUE) {
        si.dwFlags = STARTF_USESTDHANDLES;
        si.hStdOutput = si.hStdError = hlog;
    }
    PROCESS_INFORMATION pi;
    if (CreateProcessA(NULL, cmd, NULL, NULL, hlog != INVALID_HANDLE_VALUE,
                       CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP,
                       NULL, NULL, &si, &pi)) {
        g_managed_pid = (long)pi.dwProcessId;
        g_managed_spawned = true;
        CloseHandle(pi.hThread);
        CloseHandle(pi.hProcess);
    }
    if (hlog != INVALID_HANDLE_VALUE) CloseHandle(hlog);
#else
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    if (lp[0]) {
        posix_spawn_file_actions_addopen(&fa, 1, lp,
                                         O_WRONLY | O_CREAT | O_TRUNC, 0644);
        posix_spawn_file_actions_adddup2(&fa, 1, 2);
    }
    pid_t pid;
    if (posix_spawn(&pid, exe, &fa, NULL, argv, environ) == 0) {
        g_managed_pid = (long)pid;
        g_managed_spawned = true;
    }
    posix_spawn_file_actions_destroy(&fa);
#endif
}

// Graceful-then-firm stop. The 3 s escalation runs inline: stopping is a
// user-initiated action and the runner responds to TERM in well under a
// second; the wait is bounded either way.
static void stop_pid(long pid) {
    if (pid <= 0) return;
#ifdef _WIN32
    HANDLE h = OpenProcess(PROCESS_TERMINATE | SYNCHRONIZE, FALSE, (DWORD)pid);
    if (!h) return;
    // no portable graceful signal for a console process from outside its
    // group: v1 documents TerminateProcess (records are swept either way)
    TerminateProcess(h, 0);
    WaitForSingleObject(h, 3000);
    CloseHandle(h);
#else
    kill((pid_t)pid, SIGTERM);
    bool mine = pid == g_managed_pid, reaped = false;
    for (int i = 0; i < 30; i++) {
        // our own child MUST be reaped here, not just observed dead — a
        // WNOHANG probe that races the kill leaves a zombie nobody ever
        // waits on again (found live: permanent ghost row in the menu)
        if (mine) {
            pid_t w = waitpid((pid_t)pid, NULL, WNOHANG);
            if (w == (pid_t)pid || (w == -1 && errno == ECHILD)) {
                reaped = true;  // dead and reaped (here or by managed_state)
                break;
            }
        }
        if (!mine && !instance_pid_alive(pid)) break;
        struct timespec ts = { 0, 100 * 1000 * 1000 };
        nanosleep(&ts, NULL);
    }
    if (!reaped) {
        if (instance_pid_alive(pid)) kill((pid_t)pid, SIGKILL);
        // blocking wait is safe after SIGKILL: death is guaranteed, and it
        // closes the reap race for good
        if (mine) waitpid((pid_t)pid, NULL, 0);
    }
#endif
    if (pid == g_managed_pid) g_managed_pid = 0;
}

static void self_exe(char *out, size_t cap) {
#ifdef _WIN32
    GetModuleFileNameA(NULL, out, (DWORD)cap);
#elif defined(__APPLE__)
    extern int _NSGetExecutablePath(char *, unsigned int *);
    unsigned int sz = (unsigned int)cap;
    if (_NSGetExecutablePath(out, &sz) != 0) snprintf(out, cap, "runner");
#else
    ssize_t n = readlink("/proc/self/exe", out, cap - 1);
    if (n > 0) out[n] = 0; else snprintf(out, cap, "runner");
#endif
}

// ---------------------------------------------------- swap-serve enrichment

// One loopback GET, best-effort: fills `out` with the raw response (headers
// included) and returns false on any failure. Shared by the model list and the
// activity poll so there is one timeout policy and one socket teardown path.
static bool http_get_loopback(int port, const char *path, char *out, int cap) {
#ifdef _WIN32
    static bool wsa_up = false;
    if (!wsa_up) { WSADATA w; wsa_up = WSAStartup(MAKEWORD(2, 2), &w) == 0; }
    if (!wsa_up) return false;
#endif
    sock_t s = socket(AF_INET, SOCK_STREAM, 0);
    if (s == INVALID_SOCKET) return false;
#ifdef _WIN32
    DWORD tmo = 500;
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, (const char *)&tmo, sizeof tmo);
    setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, (const char *)&tmo, sizeof tmo);
#else
    struct timeval tv = { 0, 500 * 1000 };
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
    setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);
#endif
    struct sockaddr_in a = {0};
    a.sin_family = AF_INET;
    a.sin_port = htons((unsigned short)port);
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (connect(s, (struct sockaddr *)&a, sizeof a) != 0) { sock_close(s); return false; }

    char req[160];
    int rl = snprintf(req, sizeof req,
                      "GET %s HTTP/1.1\r\nHost: 127.0.0.1:%d\r\n"
                      "Connection: close\r\n\r\n", path, port);
    if (send(s, req, rl, 0) != rl) { sock_close(s); return false; }

    int n = 0, r;
    while (n < cap - 1 && (r = (int)recv(s, out + n, cap - 1 - n, 0)) > 0)
        n += r;
    sock_close(s);
    out[n] = 0;
    return n > 0;
}

// A swap-mode server registers with an empty models list (models come and go
// at runtime), so the menu asks the instance itself and parses the ids.
static int fetch_served_models(int port, char names[][128], int cap) {
    char buf[16384];
    if (!http_get_loopback(port, "/v1/models", buf, sizeof buf)) return 0;
    char *body = strstr(buf, "\r\n\r\n");
    if (!body) return 0;
    jv *v = json_parse(body + 4, strlen(body + 4));
    if (!v) return 0;
    int out = 0;
    jv *data = jv_get(v, "data");
    if (data && data->type == J_ARR)
        for (int i = 0; i < data->n && out < cap; i++) {
            const char *id = jv_str(jv_get(data->items[i], "id"), NULL);
            if (id) snprintf(names[out++], 128, "%s", id);
        }
    jv_free(v);
    return out;
}

// ------------------------------------------------------------------- menu

static const char *base_name(const char *p) {
    const char *b = strrchr(p, '/');
#ifdef _WIN32
    const char *b2 = strrchr(p, '\\');
    if (b2 && (!b || b2 > b)) b = b2;
#endif
    return b ? b + 1 : p;
}

static void label_copy(char *out, size_t cap, const char *text) {
    if (!cap) return;
    size_t n = strlen(text);
    if (n >= cap) n = cap - 1;
    memcpy(out, text, n);
    out[n] = 0;
}

int tray_menu_build(tray_item *it, int cap) {
    cfg_load();
    int n = 0;
#define PUT(...) do { if (n < cap) { it[n] = (tray_item){__VA_ARGS__}; n++; } } while (0)

    PUT(.kind = TRAY_K_LABEL, .action = TRAY_ACT_NONE);
    snprintf(it[0].label, sizeof it[0].label, "xyntetik-runner %s", RUNNER_VERSION);
    PUT(.kind = TRAY_K_SEP);

    int ni = 0;
    instance_rec *recs = instances_list(&ni);
    int st = managed_state(recs, ni);

    // the tray's own record is bookkeeping (single-instance guard), not a
    // runner the user can manage — never show it as an instance row
    int shown = 0;
    for (int i = 0; i < ni; i++)
        if (strcmp(recs[i].mode, "tray") != 0) shown++;

    if (shown == 0 && st != MG_STARTING && st != MG_EXITED && st != MG_MISSING) {
        PUT(.kind = TRAY_K_LABEL);
        snprintf(it[n-1].label, sizeof it[n-1].label, "no runners active");
    }
    if (st == MG_MISSING) {
        PUT(.kind = TRAY_K_LABEL);
        snprintf(it[n-1].label, sizeof it[n-1].label,
                 "⚠ model file missing: %.470s", base_name(g_cfg.last_model));
    }
    if (st == MG_STARTING) {
        PUT(.kind = TRAY_K_LABEL);
        snprintf(it[n-1].label, sizeof it[n-1].label,
                 "● starting %.470s… (loading model)", base_name(g_cfg.last_model));
        PUT(.kind = TRAY_K_SUB_BEGIN);
        PUT(.kind = TRAY_K_ACTION, .action = TRAY_ACT_STOP_INSTANCE,
            .arg = g_managed_pid);
        snprintf(it[n-1].label, sizeof it[n-1].label, "Cancel start");
        PUT(.kind = TRAY_K_SUB_END);
    }
    if (st == MG_EXITED) {
        PUT(.kind = TRAY_K_LABEL);
        snprintf(it[n-1].label, sizeof it[n-1].label,
                 "⚠ default runner exited (failed start or crash)");
        PUT(.kind = TRAY_K_ACTION, .action = TRAY_ACT_OPEN_LOG);
        snprintf(it[n-1].label, sizeof it[n-1].label, "View log");
    }
    for (int i = 0; i < ni; i++) {
        if (strcmp(recs[i].mode, "tray") == 0) continue;
        bool managed = recs[i].pid == g_managed_pid;
        PUT(.kind = TRAY_K_LABEL);
        if (recs[i].port > 0)
            snprintf(it[n-1].label, sizeof it[n-1].label, "%s%s  ·  :%d  ·  pid %ld",
                     managed ? "● " : "", recs[i].mode, recs[i].port, recs[i].pid);
        else
            snprintf(it[n-1].label, sizeof it[n-1].label, "%s%s  ·  pid %ld",
                     managed ? "● " : "", recs[i].mode, recs[i].pid);
        PUT(.kind = TRAY_K_SUB_BEGIN);
        for (int m = 0; m < recs[i].n_models; m++) {
            PUT(.kind = TRAY_K_LABEL);
            snprintf(it[n-1].label, sizeof it[n-1].label, "%s", recs[i].model_names[m]);
        }
        if (recs[i].n_models == 0 && recs[i].port > 0) {
            // swap-mode server: registry has no static list, ask the API
            static char served[16][128];
            int ns = fetch_served_models(recs[i].port, served, 16);
            for (int m = 0; m < ns; m++) {
                PUT(.kind = TRAY_K_LABEL);
                label_copy(it[n-1].label, sizeof it[n-1].label, served[m]);
            }
            if (ns == 0) {
                PUT(.kind = TRAY_K_LABEL);
                snprintf(it[n-1].label, sizeof it[n-1].label, "(no model resident)");
            }
        }
        PUT(.kind = TRAY_K_ACTION, .action = TRAY_ACT_STOP_INSTANCE, .arg = recs[i].pid);
        snprintf(it[n-1].label, sizeof it[n-1].label, "Stop");
        PUT(.kind = TRAY_K_SUB_END);
    }
    instances_list_free(recs, ni);

    PUT(.kind = TRAY_K_SEP);
    if (st == MG_RUNNING) {
        // the ● row above carries Stop; a live Start row here would just
        // be a silent no-op, which reads as "nothing happened"
        PUT(.kind = TRAY_K_LABEL);
        snprintf(it[n-1].label, sizeof it[n-1].label,
                 "Default runner: running on :%d", g_cfg.port);
    } else if (st == MG_STARTING) {
        PUT(.kind = TRAY_K_LABEL);
        snprintf(it[n-1].label, sizeof it[n-1].label, "Default runner: starting…");
    } else if (st == MG_MISSING) {
        // No Start row: it could only ever fail. Offer the fix instead.
        PUT(.kind = TRAY_K_ACTION, .action = TRAY_ACT_PICK_MODEL);
        snprintf(it[n-1].label, sizeof it[n-1].label,
                 "Choose another model… (%.470s is gone)",
                 base_name(g_cfg.last_model));
    } else if (g_cfg.last_model[0]) {
        PUT(.kind = TRAY_K_ACTION, .action = TRAY_ACT_START_MANAGED);
        snprintf(it[n-1].label, sizeof it[n-1].label, "%s default runner (%.470s)",
                 st == MG_EXITED ? "Restart" : "Start",
                 base_name(g_cfg.last_model));
    } else {
        PUT(.kind = TRAY_K_ACTION, .action = TRAY_ACT_PICK_MODEL);
        snprintf(it[n-1].label, sizeof it[n-1].label, "Start… (no model configured)");
    }
    PUT(.kind = TRAY_K_ACTION, .action = TRAY_ACT_PICK_MODEL);
    snprintf(it[n-1].label, sizeof it[n-1].label, "Choose model…");
    PUT(.kind = TRAY_K_CHECK, .action = TRAY_ACT_TOGGLE_AUTOSTART,
        .checked = tray_platform_autostart_get());
    snprintf(it[n-1].label, sizeof it[n-1].label, "Launch at login");
    PUT(.kind = TRAY_K_SEP);
    PUT(.kind = TRAY_K_ACTION, .action = TRAY_ACT_QUIT);
    snprintf(it[n-1].label, sizeof it[n-1].label, "Quit controller");
#undef PUT
    return n;
}

void tray_menu_act(int action, long arg, const char *text) {
    switch (action) {
    case TRAY_ACT_START_MANAGED:
#ifndef _WIN32
        if (g_managed_pid > 0) waitpid((pid_t)g_managed_pid, NULL, WNOHANG);
#endif
        if (g_managed_pid && instance_pid_alive(g_managed_pid)) break;
        spawn_managed();
        break;
    case TRAY_ACT_PICK_MODEL:
        if (text && *text) {
            snprintf(g_cfg.last_model, sizeof g_cfg.last_model, "%s", text);
            cfg_save();
            if (!g_managed_pid || !instance_pid_alive(g_managed_pid))
                spawn_managed();
        }
        break;
    case TRAY_ACT_STOP_INSTANCE:
        // a user-initiated stop of the managed instance is not a failure:
        // drop the spawned flag so no "exited" warning lingers
        if (arg == g_managed_pid) g_managed_spawned = false;
        stop_pid(arg);
        break;
    case TRAY_ACT_TOGGLE_AUTOSTART:
        tray_platform_autostart_set(!tray_platform_autostart_get());
        break;
    case TRAY_ACT_OPEN_LOG:
        open_log();
        break;
    case TRAY_ACT_QUIT:
        // desktop-managed semantics: quitting the controller stops the
        // instance it manages — and ONLY that one
        g_managed_spawned = false;
        if (g_managed_pid) stop_pid(g_managed_pid);
        g_quit = true;
        break;
    }
}

void tray_ensure_running(void) {
    int n = 0;
    instance_rec *r = instances_list(&n);
    bool have_tray = false;
    for (int i = 0; i < n; i++)
        if (strcmp(r[i].mode, "tray") == 0) have_tray = true;
    instances_list_free(r, n);
    if (have_tray) return;

    char exe[1200];
    self_exe(exe, sizeof exe);
    char *argv[] = { exe, (char *)"--tray", NULL };

#ifdef _WIN32
    char cmd[1400];
    snprintf(cmd, sizeof cmd, "\"%s\" --tray", exe);
    STARTUPINFOA si = { .cb = sizeof si };
    PROCESS_INFORMATION pi;
    // DETACHED_PROCESS: no console of its own and none inherited, so a tray
    // launched beside a server does not scribble over the server's output
    if (CreateProcessA(NULL, cmd, NULL, NULL, FALSE,
                       DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,
                       NULL, NULL, &si, &pi)) {
        CloseHandle(pi.hThread);
        CloseHandle(pi.hProcess);
    }
#else
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    // stdio to /dev/null: the tray has nothing to say on a terminal it shares
    // with a server, and inheriting the tty would let it interleave output
    posix_spawn_file_actions_addopen(&fa, 0, "/dev/null", O_RDONLY, 0);
    posix_spawn_file_actions_addopen(&fa, 1, "/dev/null", O_WRONLY, 0);
    posix_spawn_file_actions_adddup2(&fa, 1, 2);

    posix_spawnattr_t at;
    posix_spawnattr_init(&at);
#ifdef POSIX_SPAWN_SETSID
    // own session, so SIGINT to the server's foreground process group does
    // not reach the tray — Ctrl-C must stop the server, not the menu bar
    posix_spawnattr_setflags(&at, POSIX_SPAWN_SETSID);
#endif
    pid_t pid;
    posix_spawn(&pid, exe, &fa, &at, argv, environ);
    posix_spawnattr_destroy(&at);
    posix_spawn_file_actions_destroy(&fa);
#endif
}

bool tray_any_running(void) {
    int n = 0, live = 0;
    instance_rec *r = instances_list(&n);
    // the tray's own record must not light the running dot
    for (int i = 0; i < n; i++)
        if (strcmp(r[i].mode, "tray") != 0) live++;
    instances_list_free(r, n);
    return live > 0;
}

// GET /health on loopback and read active_requests. Same best-effort shape as
// fetch_served_models: short timeout, and any failure returns 0 so an
// unreachable server reads as "not working" rather than inventing activity.
static int fetch_active_requests(int port) {
    char buf[2048];
    if (!http_get_loopback(port, "/health", buf, sizeof buf)) return 0;
    char *body = strstr(buf, "\r\n\r\n");
    if (!body) return 0;
    jv *v = json_parse(body + 4, strlen(body + 4));
    if (!v) return 0;
    int active = (int)jv_num(jv_get(v, "active_requests"), 0);
    jv_free(v);
    return active;
}

tray_icon_state tray_icon(void) {
    int n = 0;
    instance_rec *r = instances_list(&n);
    bool loaded = false;
    int  active = 0;
    for (int i = 0; i < n; i++) {
        if (strcmp(r[i].mode, "tray") == 0) continue;  // the tray is not a runner
        loaded = true;
        // Only a serving instance can be asked. A `cli` run holds a model too,
        // and counts as loaded, but has no port to answer on.
        if (r[i].port > 0 && active == 0)
            active = fetch_active_requests(r[i].port);
    }
    instances_list_free(r, n);
    if (!loaded) return TRAY_ICON_IDLE;
    return active > 0 ? TRAY_ICON_RUNNING : TRAY_ICON_LOADED;
}

bool tray_should_quit(void) { return g_quit; }

int tray_main(void) {
    cfg_load();

    // headless validation seam: dump the menu the backend would render and
    // exit. Lets CI and the remote Windows box assert menu content without
    // a display; also what the human-click checklist diffs against. Runs
    // BEFORE the single-instance guard — it is read-only, and the checklist
    // diffs it against a tray that is currently up.
    // icon sibling of the same seam: render the three glyph states to files so
    // a design change can be reviewed, and diffed, without a menu bar.
    const char *icon_dir = getenv("XYNTETIK_TRAY_ICON_DUMP");
    if (icon_dir)
        return tray_platform_icon_dump(icon_dir, 144) ? 0 : 1;

    if (getenv("XYNTETIK_TRAY_DUMP")) {
        tray_item items[128];
        int ni2 = tray_menu_build(items, 128);
        int depth = 0;
        for (int i = 0; i < ni2; i++) {
            if (items[i].kind == TRAY_K_SUB_END) { depth--; continue; }
            printf("%*s", depth * 2, "");
            switch (items[i].kind) {
            case TRAY_K_SEP:       printf("---\n"); break;
            case TRAY_K_SUB_BEGIN: depth++; printf(">\n"); break;
            case TRAY_K_CHECK:
                printf("[%c] %s\n", items[i].checked ? 'x' : ' ', items[i].label);
                break;
            case TRAY_K_ACTION:    printf("* %s\n", items[i].label); break;
            default:               printf("  %s\n", items[i].label); break;
            }
        }
        return 0;
    }

    // one tray per machine: a live record with mode "tray" means another
    // controller owns the icon
    int ng = 0;
    instance_rec *rg = instances_list(&ng);
    for (int i = 0; i < ng; i++)
        if (strcmp(rg[i].mode, "tray") == 0 && rg[i].pid != (long)getpid()) {
            fprintf(stderr, "error: another tray controller is running (pid %ld)\n",
                    rg[i].pid);
            instances_list_free(rg, ng);
            return 1;
        }
    instances_list_free(rg, ng);

    instances_register("tray", 0, NULL, NULL, 0);
    atexit(instances_unregister);
    int rc = tray_platform_run();
    instances_unregister();
    return rc;
}
