// /v1/capabilities is answered on the ACCEPT thread while a slot thread can be
// freeing the very model it reports on.
//
// send_capabilities reads the residency index, then dereferences
// SV.slots[0].m for the agent profile and the MTP declaration. Nothing holds
// SV.swap_mu across those two steps, and swap_to() -- reached by any request
// naming a different model -- calls unload_resident(), which model_free()s and
// frees that container. The window is the handful of instructions between the
// residency read and the dereference, and it is hit within a second of ordinary
// traffic: one client alternating models while another polls capabilities.
//
// This gate only bites when it is built with AddressSanitizer, which is how it
// is wired in the Makefile (`make test-swap-race`). A plain build reads the
// freed bytes and usually gets away with it, which is precisely why the bug
// survived: the failure it produces in the field is a crash under load with no
// pattern to it.
#include "runner.h"
#include "compat.h"
#include "http.h"
#include "server.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *g_model = "test.gguf";
static int  g_port = 18097;
static int  g_fail = 0;
static double g_seconds = 3.0;
static atomic_int g_stop;
static atomic_int g_answered;

static void ck(int cond, const char *what) {
    if (!cond) { fprintf(stderr, "FAIL: %s\n", what); g_fail = 1; }
}

static int g_rc = -1;

static void *serve_thread(void *arg) {
    (void)arg;
    char spec[512];
    snprintf(spec, sizeof spec, "a=%s,b=%s", g_model, g_model);
    model_params p;
    memset(&p, 0, sizeof(p));
    p.gpu_mode  = GPU_OFF;
    p.n_threads = 1;
    p.n_ctx     = 512;
    p.n_batch   = 8;
    sampler smp;
    sampler_reset(&smp);
    sampler_override ov;
    memset(&ov, 0, sizeof(ov));
    sampler_resolve(&smp, NULL, NULL, -1, &ov);
    // ttl 0: the residency reaper never fires, so every unload in this run is
    // one a request asked for. The race is not about the reaper.
    g_rc = server_run(NULL, NULL, spec, &p, smp, &ov, g_port, 1, 1, 0,
                      NULL, 0, false, -1, false);
    return NULL;
}

// One request on its own connection, read to EOF. Returns the response (caller
// frees) or NULL.
static char *http_req(const char *req, size_t n, double budget_s) {
    sock_t fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd == SOCK_INVALID) return NULL;
    struct sockaddr_in a;
    memset(&a, 0, sizeof(a));
    a.sin_family = AF_INET;
    a.sin_port   = htons((uint16_t)g_port);
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (connect(fd, (struct sockaddr *)&a, (int)sizeof(a)) != 0) {
        sock_close(fd);
        return NULL;
    }
    sock_recv_timeout(fd, budget_s);
    sock_send_timeout(fd, budget_s);
    if (sock_send(fd, req, n) != (int)n) { sock_close(fd); return NULL; }
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

// The User-Agent padding is not decoration: a request short enough to fit the
// accept loop's peek takes a different drain path, and this gate is about what
// happens AFTER the routing, not about the framing.
static const char CAPS[] =
    "GET /v1/capabilities HTTP/1.1\r\nHost: 127.0.0.1\r\n"
    "User-Agent: swap-race-probe\r\n\r\n";

static void *poll_thread(void *arg) {
    (void)arg;
    while (!atomic_load(&g_stop)) {
        char *r = http_req(CAPS, sizeof(CAPS) - 1, 5.0);
        if (r) {
            if (strstr(r, "200 OK")) atomic_fetch_add(&g_answered, 1);
            free(r);
        }
    }
    return NULL;
}

// Alternate the named model so every request forces swap_to() to unload the
// resident one and load the other. /v1/embeddings with no `input` is refused
// 400 -- AFTER the swap, which is the only part this gate needs and the
// cheapest way to ask for one.
static void *churn_thread(void *arg) {
    (void)arg;
    for (int i = 0; !atomic_load(&g_stop); i++) {
        char body[64], req[512];
        int bn = snprintf(body, sizeof body, "{\"model\":\"%s\"}",
                          i % 2 ? "a" : "b");
        int n = snprintf(req, sizeof req,
                         "POST /v1/embeddings HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                         "User-Agent: swap-race-churn\r\n"
                         "Content-Length: %d\r\n\r\n%s", bn, body);
        free(http_req(req, (size_t)n, 15.0));
    }
    return NULL;
}

int main(int argc, char **argv) {
    if (argc > 1) g_model = argv[1];
    if (argc > 2) g_port = atoi(argv[2]);
    if (argc > 3) g_seconds = atof(argv[3]);
    f16_init();
    sock_init();

    pthread_t server_th;
    if (pthread_create(&server_th, NULL, serve_thread, NULL) != 0) {
        fprintf(stderr, "FAIL: cannot start the server thread\n");
        return 1;
    }

    // up when /health answers; a swap-mode server has no model resident yet
    char *up = NULL;
    double deadline = plat_now() + 20.0;
    static const char HEALTH[] =
        "GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\n"
        "User-Agent: swap-race-probe\r\n\r\n";
    while (!up && plat_now() < deadline) {
        up = http_req(HEALTH, sizeof(HEALTH) - 1, 2.0);
        if (!up) plat_sleep_ms(20);
    }
    ck(up != NULL, "server answered /health");
    free(up);

    enum { N_POLL = 8 };
    pthread_t poll_th[N_POLL], churn_th;
    for (int i = 0; i < N_POLL; i++)
        pthread_create(&poll_th[i], NULL, poll_thread, NULL);
    pthread_create(&churn_th, NULL, churn_thread, NULL);

    double until = plat_now() + g_seconds;
    while (plat_now() < until) plat_sleep_ms(50);
    atomic_store(&g_stop, 1);
    for (int i = 0; i < N_POLL; i++) pthread_join(poll_th[i], NULL);
    pthread_join(churn_th, NULL);

    // A gate that never reached the code it guards is not a green run.
    int answered = atomic_load(&g_answered);
    ck(answered > 100, "capabilities was polled through the swaps");

    server_request_stop();
    pthread_join(server_th, NULL);
    ck(g_rc == 0, "server_run returned 0");

    if (g_fail) {
        fprintf(stderr, "swap race: FAILED\n");
        return 1;
    }
    printf("swap race: %d capabilities answers across model swaps\n", answered);
    return 0;
}
