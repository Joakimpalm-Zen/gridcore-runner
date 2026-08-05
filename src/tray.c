// Tray controller core — see tray.h. Portable C; no GUI includes here.
#include "tray.h"
#include "instances.h"
#include "json.h"
#include "runner.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <process.h>
#define getpid _getpid
typedef SOCKET sock_t;
#define sock_close closesocket
#else
#include <arpa/inet.h>
#include <netinet/in.h>
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

static bool g_cfg_loaded = false;

static void cfg_load(void) {
    if (g_cfg_loaded) return;
    g_cfg_loaded = true;
    char path[1200];
    cfg_path(path, sizeof path);
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
}

// --------------------------------------------------------- managed process

static long g_managed_pid = 0;
static bool g_quit = false;

static void self_exe(char *out, size_t cap);

static void spawn_managed(void) {
    cfg_load();
    if (!g_cfg.last_model[0]) return;
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

#ifdef _WIN32
    // rebuild a command line; CreateProcess wants one string
    char cmd[2600] = "";
    for (int i = 0; argv[i]; i++) {
        strncat(cmd, "\"", sizeof cmd - strlen(cmd) - 1);
        strncat(cmd, argv[i], sizeof cmd - strlen(cmd) - 1);
        strncat(cmd, "\" ", sizeof cmd - strlen(cmd) - 1);
    }
    STARTUPINFOA si = { .cb = sizeof si };
    PROCESS_INFORMATION pi;
    if (CreateProcessA(NULL, cmd, NULL, NULL, FALSE,
                       CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP,
                       NULL, NULL, &si, &pi)) {
        g_managed_pid = (long)pi.dwProcessId;
        CloseHandle(pi.hThread);
        CloseHandle(pi.hProcess);
    }
#else
    pid_t pid;
    if (posix_spawn(&pid, exe, NULL, NULL, argv, environ) == 0)
        g_managed_pid = (long)pid;
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
    for (int i = 0; i < 30; i++) {
        if (!instance_pid_alive(pid)) break;
        struct timespec ts = { 0, 100 * 1000 * 1000 };
        nanosleep(&ts, NULL);
    }
    if (instance_pid_alive(pid)) kill((pid_t)pid, SIGKILL);
    if (pid == g_managed_pid) waitpid((pid_t)pid, NULL, WNOHANG);
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

// A swap-mode server registers with an empty models list (models come and
// go at runtime), so the menu asks the instance itself: GET /v1/models on
// loopback with a short timeout, parse the ids. Best-effort — on any
// failure the menu just shows the placeholder row.
static int fetch_served_models(int port, char names[][128], int cap) {
#ifdef _WIN32
    static bool wsa_up = false;
    if (!wsa_up) { WSADATA w; wsa_up = WSAStartup(MAKEWORD(2, 2), &w) == 0; }
    if (!wsa_up) return 0;
#endif
    sock_t s = socket(AF_INET, SOCK_STREAM, 0);
    if (s == INVALID_SOCKET) return 0;
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
    if (connect(s, (struct sockaddr *)&a, sizeof a) != 0) { sock_close(s); return 0; }

    char req[128];
    int rl = snprintf(req, sizeof req,
                      "GET /v1/models HTTP/1.1\r\nHost: 127.0.0.1:%d\r\n"
                      "Connection: close\r\n\r\n", port);
    if (send(s, req, rl, 0) != rl) { sock_close(s); return 0; }

    char buf[16384];
    int n = 0, r;
    while (n < (int)sizeof buf - 1 &&
           (r = (int)recv(s, buf + n, sizeof buf - 1 - n, 0)) > 0)
        n += r;
    sock_close(s);
    buf[n] = 0;

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

int tray_menu_build(tray_item *it, int cap) {
    cfg_load();
    int n = 0;
#define PUT(...) do { if (n < cap) { it[n] = (tray_item){__VA_ARGS__}; n++; } } while (0)

    PUT(.kind = TRAY_K_LABEL, .action = TRAY_ACT_NONE);
    snprintf(it[0].label, sizeof it[0].label, "gridcore-runner %s", RUNNER_VERSION);
    PUT(.kind = TRAY_K_SEP);

    int ni = 0;
    instance_rec *recs = instances_list(&ni);
    if (ni == 0) {
        PUT(.kind = TRAY_K_LABEL);
        snprintf(it[n-1].label, sizeof it[n-1].label, "no runners active");
    }
    for (int i = 0; i < ni; i++) {
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
                snprintf(it[n-1].label, sizeof it[n-1].label, "%s", served[m]);
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
    if (g_cfg.last_model[0]) {
        PUT(.kind = TRAY_K_ACTION, .action = TRAY_ACT_START_MANAGED);
        snprintf(it[n-1].label, sizeof it[n-1].label, "Start default runner (%s)",
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
        stop_pid(arg);
        break;
    case TRAY_ACT_TOGGLE_AUTOSTART:
        tray_platform_autostart_set(!tray_platform_autostart_get());
        break;
    case TRAY_ACT_QUIT:
        // desktop-managed semantics: quitting the controller stops the
        // instance it manages — and ONLY that one
        if (g_managed_pid) stop_pid(g_managed_pid);
        g_quit = true;
        break;
    }
}

bool tray_any_running(void) {
    int n = 0;
    instance_rec *r = instances_list(&n);
    instances_list_free(r, n);
    return n > 0;
}

bool tray_should_quit(void) { return g_quit; }

int tray_main(void) {
    // one tray per machine: a live record with mode "tray" means another
    // controller owns the icon
    int n = 0;
    instance_rec *r = instances_list(&n);
    for (int i = 0; i < n; i++)
        if (strcmp(r[i].mode, "tray") == 0 && r[i].pid != (long)getpid()) {
            fprintf(stderr, "error: another tray controller is running (pid %ld)\n",
                    r[i].pid);
            instances_list_free(r, n);
            return 1;
        }
    instances_list_free(r, n);

    cfg_load();

    // headless validation seam: dump the menu the backend would render and
    // exit. Lets CI and the remote Windows box assert menu content without
    // a display; also what the human-click checklist diffs against.
    if (getenv("GRIDCORE_TRAY_DUMP")) {
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

    instances_register("tray", 0, NULL, NULL, 0);
    atexit(instances_unregister);
    int rc = tray_platform_run();
    instances_unregister();
    return rc;
}
