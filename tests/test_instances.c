// Discovery-registry gate: register/list roundtrip, port rewrite, stale
// sweep, multi-record listing, unregister. Runs against a private HOME
// (or APPDATA on Windows) so it never touches the user's real registry.
#include "instances.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>

#ifdef _WIN32
#include <process.h>
#include <direct.h>
#define getpid _getpid
#define setenv_compat(k, v) _putenv_s(k, v)
#else
#include <sys/wait.h>
#include <unistd.h>
#define setenv_compat(k, v) setenv(k, v, 1)
#endif

static int fails = 0;
#define CHECK(cond, msg) do { if (!(cond)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, msg); fails++; } \
} while (0)

static void write_fake(const char *dir, long pid, const char *name) {
    char path[1300];
    snprintf(path, sizeof path, "%s/%ld.json", dir, pid);
    FILE *f = fopen(path, "wb");
    fprintf(f, "{\"pid\": %ld, \"started\": %lld, \"mode\": \"serve\","
               " \"port\": 8123, \"version\": \"test\","
               " \"models\": [{\"name\": \"%s\", \"path\": \"/x/%s\"}]}\n",
            pid, (long long)time(NULL), name, name);
    fclose(f);
}

int main(void) {
    char tmpl[512];
#ifdef _WIN32
    snprintf(tmpl, sizeof tmpl, "%s\\gridcore-test-%ld", getenv("TEMP"), (long)getpid());
    _mkdir(tmpl);
    setenv_compat("APPDATA", tmpl);
#else
    snprintf(tmpl, sizeof tmpl, "/tmp/gridcore-test-%ld", (long)getpid());
    mkdir(tmpl, 0755);
    setenv_compat("HOME", tmpl);
#endif

    const char *dir = instances_dir();
    CHECK(dir != NULL, "instances_dir resolves and creates the directory");
    if (!dir) return 1;

    // 1. register + list roundtrip
    const char *const names[] = { "modelA.gguf", "modelB.gguf" };
    const char *const paths[] = { "/m/modelA.gguf", "/m/modelB.gguf" };
    CHECK(instances_register("cli", 0, names, paths, 2), "register succeeds");
    int n = 0;
    instance_rec *r = instances_list(&n);
    CHECK(n == 1, "one live record after register");
    if (n == 1) {
        CHECK(r[0].pid == (long)getpid(), "record carries our pid");
        CHECK(strcmp(r[0].mode, "cli") == 0, "mode roundtrips");
        CHECK(r[0].port == 0, "port roundtrips");
        CHECK(r[0].n_models == 2, "both models roundtrip");
        CHECK(r[0].n_models == 2 && strcmp(r[0].model_names[1], "modelB.gguf") == 0,
              "model name content roundtrips");
        CHECK(r[0].version[0] != 0, "version present");
    }
    instances_list_free(r, n);

    // 2. set_port rewrites in place
    CHECK(instances_set_port(9099), "set_port succeeds");
    r = instances_list(&n);
    CHECK(n == 1 && r[0].port == 9099, "port rewrite visible to readers");
    instances_list_free(r, n);

    // 3. a second record for a live pid is listed alongside ours
    write_fake(dir, (long)getpid() + 2000000, "ghost.gguf"); // dead pid
    char alive_path[1300];
    // reuse OUR pid under a different filename: filename is identity, pid
    // field is what liveness checks — this models a second live process
    snprintf(alive_path, sizeof alive_path, "%s/%ld.json", dir, (long)getpid() + 1);
    {
        FILE *f = fopen(alive_path, "wb");
        fprintf(f, "{\"pid\": %ld, \"started\": 1, \"mode\": \"serve\","
                   " \"port\": 8123, \"version\": \"t\", \"models\": []}\n",
                (long)getpid());
        fclose(f);
    }
    r = instances_list(&n);
    CHECK(n == 2, "live foreign record listed, dead one swept");
    bool found_serve = false;
    for (int i = 0; i < n; i++)
        if (strcmp(r[i].mode, "serve") == 0 && r[i].port == 8123) found_serve = true;
    CHECK(found_serve, "foreign serve record fields intact");
    instances_list_free(r, n);

    // 4. the dead record's file is gone (swept, not just skipped)
    {
        char dead[1300];
        snprintf(dead, sizeof dead, "%s/%ld.json", dir, (long)getpid() + 2000000);
        FILE *f = fopen(dead, "rb");
        CHECK(f == NULL, "stale record file deleted by the sweep");
        if (f) fclose(f);
    }

    // 5. unregister removes ours; double-unregister is safe
    instances_unregister();
    instances_unregister();
    remove(alive_path);
    r = instances_list(&n);
    CHECK(n == 0, "no records after unregister + cleanup");
    instances_list_free(r, n);

    // 6. corrupt record is swept, not fatal
    {
        char junk[1300];
        snprintf(junk, sizeof junk, "%s/999.json", dir);
        FILE *f = fopen(junk, "wb");
        fprintf(f, "not json at all");
        fclose(f);
        r = instances_list(&n);
        CHECK(n == 0, "corrupt record ignored");
        instances_list_free(r, n);
        f = fopen(junk, "rb");
        CHECK(f == NULL, "corrupt record swept");
        if (f) fclose(f);
    }

#ifndef _WIN32
    // 7. a zombie answers kill(pid, 0) but must count as dead: a record
    // pointing at one would otherwise be an unsweepable ghost row that
    // Stop can never clear (found live on the M1, 2026-08-05)
    {
        pid_t z = fork();
        if (z == 0) _exit(0);
        // give it time to die; do NOT reap — the zombie is the fixture
        for (int t = 0; t < 100 && instance_pid_alive((long)z); t++) {
            struct timespec ts = { 0, 10 * 1000 * 1000 };
            nanosleep(&ts, NULL);
        }
        CHECK(!instance_pid_alive((long)z), "zombie counts as dead");
        write_fake(dir, (long)z, "ghost.gguf");
        r = instances_list(&n);
        CHECK(n == 0, "zombie record not listed");
        instances_list_free(r, n);
        char zp[1300];
        snprintf(zp, sizeof zp, "%s/%ld.json", dir, (long)z);
        FILE *f = fopen(zp, "rb");
        CHECK(f == NULL, "zombie record swept");
        if (f) fclose(f);
        waitpid(z, NULL, 0);
    }
#endif

    if (fails) { fprintf(stderr, "test_instances: %d FAILURES\n", fails); return 1; }
    printf("instances registry tests ok\n");
    return 0;
}
