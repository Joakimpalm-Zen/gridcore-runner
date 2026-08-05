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
#include <signal.h>
#include <unistd.h>
#endif

// ------------------------------------------------------------------ paths

static bool mkdir_one(const char *p) {
#ifdef _WIN32
    return _mkdir(p) == 0 || errno == EEXIST || GetLastError() == ERROR_ALREADY_EXISTS;
#else
    return mkdir(p, 0755) == 0 || errno == EEXIST;
#endif
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
    snprintf(a, sizeof a, "%s\\gridcore", base);
    snprintf(b, sizeof b, "%s\\runner", a);
    snprintf(c, sizeof c, "%s\\instances", b);
    if (!mkdir_one(a) || !mkdir_one(b) || !mkdir_one(c)) { dir[0] = 0; return NULL; }
    snprintf(dir, sizeof dir, "%s", c);
#else
    const char *base = getenv("HOME");
    if (!base || !*base) { dir[0] = 0; return NULL; }
    char a[1024], b[1024], c[1024];
    snprintf(a, sizeof a, "%s/.gridcore", base);
    snprintf(b, sizeof b, "%s/runner", a);
    snprintf(c, sizeof c, "%s/instances", b);
    if (!mkdir_one(a) || !mkdir_one(b) || !mkdir_one(c)) { dir[0] = 0; return NULL; }
    snprintf(dir, sizeof dir, "%s", c);
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
    return kill((pid_t)pid, 0) == 0 || errno == EPERM;
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
        // restart or inspect the file the instance was launched with
        char abs[1200];
#ifdef _WIN32
        if (!_fullpath(abs, model_paths[i], sizeof abs))
            snprintf(abs, sizeof abs, "%s", model_paths[i]);
#else
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

static instance_rec *push_rec(instance_rec *arr, int *n, int *cap) {
    if (*n == *cap) {
        *cap = *cap ? *cap * 2 : 8;
        arr = realloc(arr, sizeof(instance_rec) * (size_t)*cap);
    }
    if (arr) memset(&arr[*n], 0, sizeof(instance_rec));
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
            arr = push_rec(arr, &n, &cap);
            if (!arr) { rec_free_members(&r); break; }
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
