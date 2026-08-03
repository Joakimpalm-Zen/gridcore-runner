// server_run must be able to run, stop, and run again in one process.
//
// This is the property the RNR-019 finding is actually about. The review's
// complaint was not that `SV` is spelled as a global — it was that global
// queue, scheduler and signal state "make partial initialization and teardown
// difficult", and that the scheduler and swap leaks were symptoms. A global
// initialized once and torn down once can hide an asymmetry forever, because
// nothing ever asks it to come back.
//
// So this asks. Two cycles, each one: start the server on its own thread, wait
// until it answers /health, serve a request, request a graceful stop, join. If
// startup and teardown are symmetric both cycles behave identically; if any
// state survives a teardown, the second cycle is where it shows.
//
// It fails before the fix. `SV.shutdown` and `SV.q.shutdown` are set true at
// teardown and never cleared, so on a second call every slot worker sees a
// shut-down queue and exits immediately: the server listens, accepts, and
// answers nothing. The first cycle passes, which is exactly why a
// once-per-process global could carry this indefinitely.
#include "runner.h"
#include "compat.h"
#include "http.h"
#include "server.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *g_model = "test.gguf";
static int g_port = 18099;
static int g_fail = 0;

static void ck(int cond, const char *what) {
    if (!cond) { fprintf(stderr, "FAIL: %s\n", what); g_fail = 1; }
}

typedef struct {
    model_t   m;
    tokenizer tok;
    int       rc;
} run_ctx;

static void *serve_thread(void *arg) {
    run_ctx *c = arg;
    model_params p;
    memset(&p, 0, sizeof(p));
    p.gpu_mode  = GPU_OFF;
    p.n_threads = 1;
    p.n_ctx     = 512;
    p.n_batch   = 8;
    if (!model_load(&c->m, g_model, &p)) { c->rc = -1; return NULL; }
    if (!tokenizer_init(&c->tok, &c->m.gf)) { c->rc = -2; return NULL; }
    sampler smp;
    sampler_reset(&smp);
    sampler_override ov;
    memset(&ov, 0, sizeof(ov));
    sampler_resolve(&smp, c->m.arch, NULL, &ov);
    c->rc = server_run(&c->m, &c->tok, g_model, &p, smp, &ov, g_port,
                       1, 1, -1, NULL, 0, false);
    return NULL;
}

static double mono_s(void) {
    return plat_now();
}

// Connect, send one request, return the response (caller frees) or NULL.
// The budget is WALL CLOCK, not an attempt count: a server that accepts and
// then answers nothing -- which is exactly the failure this test is looking
// for -- costs the socket timeout on every attempt, so counting attempts turns
// a ten-second budget into an hour.
static char *http_req(const char *path, const char *body, double budget_s) {
    double deadline = mono_s() + budget_s;
    while (mono_s() < deadline) {
        sock_t fd = socket(AF_INET, SOCK_STREAM, 0);
        if (fd == SOCK_INVALID) return NULL;
        struct sockaddr_in a;
        memset(&a, 0, sizeof(a));
        a.sin_family = AF_INET;
        a.sin_port   = htons((uint16_t)g_port);
        a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        if (connect(fd, (struct sockaddr *)&a, (int)sizeof(a)) != 0) {
            sock_close(fd);
            plat_sleep_ms(10);
            continue;
        }
        double left = deadline - mono_s();
        if (left < 0.25) left = 0.25;
        sock_recv_timeout(fd, left);
        char req[512];
        int n = body
            ? snprintf(req, sizeof(req),
                       "POST %s HTTP/1.1\r\nHost: localhost\r\n"
                       "Content-Type: application/json\r\n"
                       "Content-Length: %zu\r\n\r\n%s",
                       path, strlen(body), body)
            : snprintf(req, sizeof(req),
                       "GET %s HTTP/1.1\r\nHost: localhost\r\n\r\n", path);
        if (sock_send(fd, req, (size_t)n) != n) { sock_close(fd); return NULL; }
        char *buf = malloc(65536);
        if (!buf) { sock_close(fd); return NULL; }
        size_t len = 0;
        for (;;) {
            int r = sock_recv(fd, buf + len, 65535 - len);
            if (r <= 0) break;
            len += (size_t)r;
            if (len >= 65535) break;
        }
        buf[len] = 0;
        sock_close(fd);
        if (len == 0) { free(buf); return NULL; }
        return buf;
    }
    return NULL;
}

static void step(int n, const char *what) {
    fprintf(stderr, "[restart] cycle %d: %s\n", n, what);
}

static void cycle(int n) {
    step(n, "start");
    run_ctx c;
    memset(&c, 0, sizeof(c));
    pthread_t th;
    if (pthread_create(&th, NULL, serve_thread, &c) != 0) {
        ck(0, "cannot start the server thread");
        return;
    }

    // The server is up when /health answers. A 10 s budget is generous: the
    // fixture loads in milliseconds.
    step(n, "waiting for /health");
    char *health = http_req("/health", NULL, 10.0);
    if (!health) {
        ck(0, "server never answered /health");
    } else {
        char msg[64];
        snprintf(msg, sizeof(msg), "cycle %d: /health returned 200", n);
        ck(strstr(health, "200 OK") != NULL, msg);
        free(health);
    }

    // The real question: does it SERVE, not just listen?
    //
    // It has to be a request a SLOT WORKER answers. /health and /v1/models are
    // both handled on the accept path by accept_fastpath, so a server whose
    // workers all exited on a stale queue-shutdown flag answers those
    // perfectly while generating nothing. The first version of this test used
    // /v1/models and passed against exactly that bug -- a gate that cannot
    // fail is worse than none, so it asks for a generation instead.
    step(n, "requesting a completion (needs a worker)");
    char *gen = http_req("/v1/completions",
                         "{\"prompt\":\"hi\",\"max_tokens\":2,\"temperature\":0}",
                         10.0);
    char msg[96];
    if (!gen) {
        snprintf(msg, sizeof(msg), "cycle %d: no worker answered a completion", n);
        ck(0, msg);
    } else {
        snprintf(msg, sizeof(msg), "cycle %d: completion returned 200", n);
        ck(strstr(gen, "200 OK") != NULL, msg);
        snprintf(msg, sizeof(msg), "cycle %d: completion produced text", n);
        ck(strstr(gen, "text_completion") != NULL, msg);
        free(gen);
    }

    step(n, "requesting graceful stop");
    server_request_stop();
    step(n, "joining");
    pthread_join(th, NULL);
    snprintf(msg, sizeof(msg), "cycle %d: server_run returned 0", n);
    ck(c.rc == 0, msg);
}

int main(int argc, char **argv) {
    if (argc > 1) g_model = argv[1];
    if (argc > 2) g_port = atoi(argv[2]);
    f16_init();
    sock_init();
    cycle(1);
    cycle(2);
    if (g_fail) {
        fprintf(stderr, "server restart: FAILED\n");
        return 1;
    }
    puts("server restart: two start/serve/stop cycles in one process");
    return 0;
}
