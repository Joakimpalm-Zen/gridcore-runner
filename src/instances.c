// Instance discovery registry — see instances.h for the contract.
#include "instances.h"
#include "json.h"
#include "runner.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <errno.h>
#include <sys/stat.h>

#ifdef _WIN32
#include <windows.h>
#include <direct.h>
#include <io.h>
#define getpid _getpid
#include <process.h>
#else
#include <dirent.h>
#include <limits.h>
#include <signal.h>
#include <unistd.h>
#ifndef PATH_MAX
#define PATH_MAX 4096
#endif
#ifdef __APPLE__
#include <sys/sysctl.h>
#include <sys/proc.h>
#endif
#endif

// ------------------------------------------------------------------ paths

static bool path_child(char *out, size_t cap, const char *base,
                       char separator, const char *name) {
    size_t nb = strlen(base), nn = strlen(name);
    if (nb >= cap || nn >= cap - nb - 1) return false;
    memcpy(out, base, nb);
    out[nb] = separator;
    memcpy(out + nb + 1, name, nn + 1);
    return true;
}

static bool mkdir_one(const char *p) {
#ifdef _WIN32
    return _mkdir(p) == 0 || errno == EEXIST || GetLastError() == ERROR_ALREADY_EXISTS;
#else
    return mkdir(p, 0755) == 0 || errno == EEXIST;
#endif
}

static bool path_exists(const char *p) {
#ifdef _WIN32
    return GetFileAttributesA(p) != INVALID_FILE_ATTRIBUTES;
#else
    struct stat st;
    return stat(p, &st) == 0;
#endif
}

// One-time migration of the pre-rename state dir (Gridcore -> Xyntetik):
// move the whole old tree when the new one does not exist yet, so config.json,
// managed.log and the instance registry survive the rename instead of
// stranding a live install (the tray-reports-missing-model failure of 1aed406).
static void migrate_old_tree(const char *base, char sep, const char *oldname,
                             const char *newbase) {
    char old[1024];
    if (!path_child(old, sizeof old, base, sep, oldname)) return;
    if (!path_exists(newbase) && path_exists(old)) rename(old, newbase);
}

const char *instances_dir(void) {
    static char dir[1024];
    static bool made = false;
    if (made) return dir[0] ? dir : NULL;
    made = true;
#ifdef _WIN32
    const char *base = getenv("APPDATA");
    if (!base || !*base) { dir[0] = 0; return NULL; }
    char a[1024], b[1024], c[1024];
    if (!path_child(a, sizeof a, base, '\\', "xyntetik") ||
        !path_child(b, sizeof b, a, '\\', "runner") ||
        !path_child(c, sizeof c, b, '\\', "instances")) {
        dir[0] = 0;
        return NULL;
    }
    migrate_old_tree(base, '\\', "gridcore", a);
    if (!mkdir_one(a) || !mkdir_one(b) || !mkdir_one(c)) { dir[0] = 0; return NULL; }
    memcpy(dir, c, strlen(c) + 1);
#else
    const char *base = getenv("HOME");
    if (!base || !*base) { dir[0] = 0; return NULL; }
    char a[1024], b[1024], c[1024];
    if (!path_child(a, sizeof a, base, '/', ".xyntetik") ||
        !path_child(b, sizeof b, a, '/', "runner") ||
        !path_child(c, sizeof c, b, '/', "instances")) {
        dir[0] = 0;
        return NULL;
    }
    migrate_old_tree(base, '/', ".gridcore", a);
    if (!mkdir_one(a) || !mkdir_one(b) || !mkdir_one(c)) { dir[0] = 0; return NULL; }
    memcpy(dir, c, strlen(c) + 1);
#endif
    return dir;
}

bool instance_pid_alive(long pid) {
    if (pid <= 0) return false;
#ifdef _WIN32
    HANDLE h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, (DWORD)pid);
    if (!h) return false;
    DWORD code = 0;
    bool alive = GetExitCodeProcess(h, &code) && code == STILL_ACTIVE;
    CloseHandle(h);
    return alive;
#else
    if (kill((pid_t)pid, 0) != 0 && errno != EPERM) return false;
    // a zombie still answers kill(pid, 0) but is dead for every purpose a
    // reader has: it serves nothing and no signal can stop it further.
    // Counting it alive turns an unreaped child into a permanent ghost row
    // that Stop can never clear.
#ifdef __APPLE__
    struct kinfo_proc kp;
    size_t len = sizeof kp;
    int mib[4] = { CTL_KERN, KERN_PROC, KERN_PROC_PID, (int)pid };
    if (sysctl(mib, 4, &kp, &len, NULL, 0) == 0 && len >= sizeof kp &&
        kp.kp_proc.p_stat == SZOMB)
        return false;
#elif defined(__linux__)
    char sp[64], buf[512];
    snprintf(sp, sizeof sp, "/proc/%ld/stat", pid);
    FILE *f = fopen(sp, "rb");
    if (f) {
        size_t rn = fread(buf, 1, sizeof buf - 1, f);
        fclose(f);
        buf[rn] = 0;
        // state is the field after the parenthesized comm
        char *rp = strrchr(buf, ')');
        if (rp && rp[1] == ' ' && (rp[2] == 'Z' || rp[2] == 'X'))
            return false;
    }
#endif
    return true;
#endif
}

// ------------------------------------------------------------- own record

// Remembered so set_port / unregister can rewrite or remove without the
// caller re-passing everything.
static struct {
    bool  registered;
    char  mode[16];
    int   port;
    sbuf  models_json;   // pre-serialized models array content
} g_self;

static void self_path(char *out, size_t cap) {
    const char *d = instances_dir();
    if (!d) { out[0] = 0; return; }
#ifdef _WIN32
    snprintf(out, cap, "%s\\%ld.json", d, (long)getpid());
#else
    snprintf(out, cap, "%s/%ld.json", d, (long)getpid());
#endif
}

static bool write_self(void) {
    char path[1200], tmp[1240];
    self_path(path, sizeof path);
    if (!path[0]) return false;
    snprintf(tmp, sizeof tmp, "%s.tmp", path);
    FILE *f = fopen(tmp, "wb");
    if (!f) return false;
    fprintf(f,
        "{\"pid\": %ld, \"started\": %lld, \"mode\": \"%s\", \"port\": %d,\n"
        " \"version\": \"%s\",\n \"models\": [%s]}\n",
        (long)getpid(), (long long)time(NULL), g_self.mode, g_self.port,
        RUNNER_VERSION, g_self.models_json.s ? g_self.models_json.s : "");
    bool ok = fclose(f) == 0;
#ifdef _WIN32
    // rename() cannot replace an existing file on Windows
    ok = ok && MoveFileExA(tmp, path, MOVEFILE_REPLACE_EXISTING);
#else
    ok = ok && rename(tmp, path) == 0;
#endif
    if (!ok) remove(tmp);
    return ok;
}

bool instances_register(const char *mode, int port,
                        const char *const *model_names,
                        const char *const *model_paths, int n_models) {
    snprintf(g_self.mode, sizeof g_self.mode, "%s", mode ? mode : "cli");
    g_self.port = port;
    free(g_self.models_json.s);
    memset(&g_self.models_json, 0, sizeof g_self.models_json);
    for (int i = 0; i < n_models; i++) {
        if (i) sb_lit(&g_self.models_json, ", ");
        sb_lit(&g_self.models_json, "{\"name\": \"");
        sb_esc(&g_self.models_json, model_names[i], strlen(model_names[i]));
        sb_lit(&g_self.models_json, "\", \"path\": \"");
        // absolute path: a controller in another cwd must be able to
        // restart or inspect the file the instance was launched with.
        // realpath's output buffer MUST be PATH_MAX bytes: glibc's
        // fortified build aborts on a smaller one regardless of how short
        // the resolved path is (macOS PATH_MAX is 1024 so a short buffer
        // passes there — every Linux CI job catches it)
#ifdef _WIN32
        char abs[_MAX_PATH];
        if (!_fullpath(abs, model_paths[i], sizeof abs))
            snprintf(abs, sizeof abs, "%s", model_paths[i]);
#else
        char abs[PATH_MAX];
        if (!realpath(model_paths[i], abs))
            snprintf(abs, sizeof abs, "%s", model_paths[i]);
#endif
        sb_esc(&g_self.models_json, abs, strlen(abs));
        sb_lit(&g_self.models_json, "\"}");
    }
    if (g_self.models_json.failed) return false;
    g_self.registered = write_self();
    return g_self.registered;
}

bool instances_set_port(int port) {
    if (!g_self.registered) return false;
    g_self.port = port;
    return write_self();
}

void instances_unregister(void) {
    if (!g_self.registered) return;
    g_self.registered = false;
    char path[1200];
    self_path(path, sizeof path);
    if (path[0]) remove(path);
}

// ------------------------------------------------------------------- list

static void rec_free_members(instance_rec *r);

// Grow to fit one more record. Returns NULL only when the array could not
// grow, and then WITHOUT disturbing the caller's array: `arr = realloc(arr, n)`
// loses the original pointer, and the old code both leaked it and handed the
// caller a NULL that instances_list still reported a count for.
static instance_rec *push_rec(instance_rec *arr, int *n, int *cap) {
    if (*n == *cap) {
        int want = *cap ? *cap * 2 : 8;
        instance_rec *grown = realloc(arr, sizeof(instance_rec) * (size_t)want);
        if (!grown) return NULL;
        arr = grown;
        *cap = want;
    }
    memset(&arr[*n], 0, sizeof(instance_rec));
    return arr;
}

static bool parse_rec(const char *path, instance_rec *r) {
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    char buf[16384];
    size_t n = fread(buf, 1, sizeof buf - 1, f);
    fclose(f);
    buf[n] = 0;
    jv *v = json_parse(buf, n);
    if (!v) return false;
    r->pid     = (long)jv_num(jv_get(v, "pid"), 0);
    r->started = (long long)jv_num(jv_get(v, "started"), 0);
    r->port    = (int)jv_num(jv_get(v, "port"), 0);
    snprintf(r->mode, sizeof r->mode, "%s", jv_str(jv_get(v, "mode"), "?"));
    snprintf(r->version, sizeof r->version, "%s", jv_str(jv_get(v, "version"), "?"));
    jv *ms = jv_get(v, "models");
    if (ms && ms->type == J_ARR && ms->n > 0) {
        r->model_names = calloc((size_t)ms->n, sizeof(char *));
        r->model_paths = calloc((size_t)ms->n, sizeof(char *));
        if (r->model_names && r->model_paths) {
            for (int i = 0; i < ms->n; i++) {
                r->model_names[i] = strdup(jv_str(jv_get(ms->items[i], "name"), "?"));
                r->model_paths[i] = strdup(jv_str(jv_get(ms->items[i], "path"), ""));
            }
            r->n_models = ms->n;
        }
    }
    jv_free(v);
    return r->pid > 0;
}

instance_rec *instances_list(int *n_out) {
    *n_out = 0;
    const char *d = instances_dir();
    if (!d) return NULL;
    instance_rec *arr = NULL;
    int n = 0, cap = 0;
#ifdef _WIN32
    char pat[1100];
    snprintf(pat, sizeof pat, "%s\\*.json", d);
    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA(pat, &fd);
    if (h == INVALID_HANDLE_VALUE) return NULL;
    do {
        char path[1300];
        snprintf(path, sizeof path, "%s\\%s", d, fd.cFileName);
#else
    DIR *dp = opendir(d);
    if (!dp) return NULL;
    struct dirent *de;
    while ((de = readdir(dp)) != NULL) {
        size_t l = strlen(de->d_name);
        if (l < 6 || strcmp(de->d_name + l - 5, ".json") != 0) continue;
        char path[1300];
        snprintf(path, sizeof path, "%s/%s", d, de->d_name);
#endif
        instance_rec r = {0};
        bool ok = parse_rec(path, &r);
        if (ok && instance_pid_alive(r.pid)) {
            // Out of memory stops the enumeration; it does not discard what has
            // already been read. A short list is a true answer, and the record
            // files are left alone -- this is not a reason to sweep a live
            // process away.
            instance_rec *grown = push_rec(arr, &n, &cap);
            if (!grown) { rec_free_members(&r); break; }
            arr = grown;
            arr[n++] = r;
        } else {
            // stale (crash leftover) or unreadable: sweep it
            rec_free_members(&r);
            remove(path);
        }
#ifdef _WIN32
    } while (FindNextFileA(h, &fd));
    FindClose(h);
#else
    }
    closedir(dp);
#endif
    *n_out = n;
    return arr;
}

static void rec_free_members(instance_rec *r) {
    for (int m = 0; m < r->n_models; m++) {
        free(r->model_names[m]);
        free(r->model_paths[m]);
    }
    free(r->model_names);
    free(r->model_paths);
    r->model_names = r->model_paths = NULL;
    r->n_models = 0;
}

void instances_list_free(instance_rec *recs, int n) {
    for (int i = 0; i < n; i++) rec_free_members(&recs[i]);
    free(recs);
}
