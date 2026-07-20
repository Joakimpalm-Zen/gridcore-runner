// The loopback bind is a product decision, and this test is the lock on it.
//
// runner's HTTP server binds INADDR_LOOPBACK and offers no way to change it:
// no --host, no --bind, no environment variable. That is not an oversight to
// be tidied up later. It is the reason two other things in this codebase are
// allowed to be as simple as they are:
//
//   1. There is no authentication on the API. Nothing in the server checks who
//      the caller is, because the kernel already did: only this machine can
//      reach the socket.
//   2. The HTTP request framing is permissive in ways that would be a request
//      smuggling primitive behind a keep-alive proxy — Content-Length is
//      matched anywhere in the header block rather than at a line start, and
//      Transfer-Encoding: chunked is ignored rather than rejected. That is
//      assessed as not exploitable *because* the listener is loopback-only and
//      serves one request per connection. Weaken the bind and that assessment
//      is void.
//
// Since Phases 1-4 the server also does tool calling on three API surfaces,
// which is precisely the capability that turned the ~175,000 exposed Ollama
// hosts found by SentinelLabs and Censys in January 2026 into remote code
// execution. Those operators did not set out to expose anything; they set
// 0.0.0.0 for convenience and did not get to the firewall. runner does not have
// the flag they got wrong.
//
// So this test fails the build if the bind is ever weakened, whether by
// swapping the constant or by adding an option that reaches it. The runtime
// half of the gate — proving the running server is genuinely unreachable on a
// non-loopback address — lives in tests/conformance/test_loopback_bind.py.
//
// If you are here because you legitimately need remote access: the supported
// answer is a reverse proxy, an SSH tunnel or Tailscale, which is where auth
// and TLS belong. If you are here because you intend to add --host anyway,
// read the "Non-Negotiable Invariants" section of FUTURE.md first: the flag
// may not land before the HTTP framing strictness fixes and an auth story.
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// mingw spells the pipe helpers with an underscore
#ifdef _WIN32
#define popen  _popen
#define pclose _pclose
#endif

// Read a whole source file. Tests run from the repo root (same convention as
// the tokenizer tests reading tests/fixtures/), so the paths are relative.
static char *slurp(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "test-bind: cannot open %s (run from the repo root)\n", path);
        exit(1);
    }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    assert(n > 0);
    char *buf = malloc((size_t)n + 1);
    assert(buf);
    size_t got = fread(buf, 1, (size_t)n, f);
    buf[got] = '\0';
    fclose(f);
    return buf;
}

static int count_of(const char *hay, const char *needle) {
    int n = 0;
    for (const char *p = strstr(hay, needle); p; p = strstr(p + 1, needle)) n++;
    return n;
}

static void must_contain(const char *src, const char *path, const char *needle) {
    if (!strstr(src, needle)) {
        fprintf(stderr,
                "test-bind: %s no longer contains `%s`.\n"
                "  The loopback-only bind is a documented invariant, not an\n"
                "  implementation detail. See FUTURE.md, Non-Negotiable Invariants.\n",
                path, needle);
        exit(1);
    }
}

static void must_not_contain(const char *src, const char *path, const char *needle,
                             const char *why) {
    if (strstr(src, needle)) {
        fprintf(stderr,
                "test-bind: %s contains `%s`.\n"
                "  %s\n"
                "  The loopback-only bind is a documented invariant. If you are\n"
                "  adding remote listening on purpose, FUTURE.md's Non-Negotiable\n"
                "  Invariants list the three things that must land with it.\n",
                path, needle, why);
        exit(1);
    }
}

// The socket must be pinned to 127.0.0.1 by a literal constant, so that no
// input — argv, a config file, an environment variable — can move it.
static void test_bind_address_is_a_literal_loopback_constant(void) {
    char *src = slurp("src/server.c");

    must_contain(src, "src/server.c", "addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);");

    // Exactly one listening socket, so exactly one bind address to reason about.
    if (count_of(src, "INADDR_LOOPBACK") != 1) {
        fprintf(stderr, "test-bind: expected exactly one INADDR_LOOPBACK in "
                        "src/server.c, found %d — a second listener is a second "
                        "bind address to audit\n", count_of(src, "INADDR_LOOPBACK"));
        exit(1);
    }

    // Every way there is of naming an address that is not the loopback one.
    static const struct { const char *sym, *why; } forbidden[] = {
        { "INADDR_ANY",    "INADDR_ANY listens on every interface." },
        { "in6addr_any",   "in6addr_any listens on every interface." },
        { "INADDR_BROADCAST", "That is not a bind address for this server." },
        { "inet_addr",     "Parsing an address string means the address came from somewhere." },
        { "inet_pton",     "Parsing an address string means the address came from somewhere." },
        { "inet_aton",     "Parsing an address string means the address came from somewhere." },
        { "getaddrinfo",   "Resolving a host means the host was configurable." },
        { "gethostbyname", "Resolving a host means the host was configurable." },
        { "SO_BINDTODEVICE", "Binding to a named device means an interface was chosen." },
    };
    for (size_t i = 0; i < sizeof forbidden / sizeof *forbidden; i++)
        must_not_contain(src, "src/server.c", forbidden[i].sym, forbidden[i].why);

    free(src);
}

// A flag nobody can pass is the whole point: the defaults of llama-server and
// ollama are loopback too, and it did not save the 175,000 hosts, because both
// keep an override. runner's guarantee is that there is nothing to override.
static void test_no_option_reaches_the_bind_address(void) {
    static const char *files[] = { "src/main.c", "src/server.c" };
    static const char *opts[] = {
        "--host", "--bind", "--listen", "--address", "--addr",
        "--interface", "--ip", "--public", "--expose",
    };
    // Environment variables are the other half of the same escape hatch:
    // OLLAMA_HOST=0.0.0.0 is exactly how the exposed hosts got exposed.
    static const char *envs[] = {
        "RUNNER_HOST", "RUNNER_BIND", "RUNNER_ADDRESS", "RUNNER_LISTEN",
    };

    for (size_t f = 0; f < sizeof files / sizeof *files; f++) {
        char *src = slurp(files[f]);
        for (size_t i = 0; i < sizeof opts / sizeof *opts; i++)
            must_not_contain(src, files[f], opts[i],
                             "That option would let a caller choose the bind address.");
        for (size_t i = 0; i < sizeof envs / sizeof *envs; i++)
            must_not_contain(src, files[f], envs[i],
                             "That variable would let the environment choose the bind address.");
        free(src);
    }
}

// The same check through the shipped binary, so a flag added anywhere at all
// (a new file, a table, a generated parser) still trips the gate. Skipped with
// a notice when the binary is absent — `make test` and both CI jobs build it
// first, so in practice this always runs.
static const char *find_runner(void) {
    static const char *candidates[] = { "./runner", "./runner.exe" };
    for (size_t i = 0; i < sizeof candidates / sizeof *candidates; i++) {
        FILE *f = fopen(candidates[i], "rb");
        if (f) { fclose(f); return candidates[i]; }
    }
    return NULL;
}

static char *run_capture(const char *cmd) {
    FILE *p = popen(cmd, "r");
    if (!p) { fprintf(stderr, "test-bind: popen failed for %s\n", cmd); exit(1); }
    size_t cap = 65536, len = 0;
    char *out = malloc(cap);
    assert(out);
    for (;;) {
        if (cap - len < 4096) { cap *= 2; out = realloc(out, cap); assert(out); }
        size_t n = fread(out + len, 1, cap - len - 1, p);
        if (n == 0) break;
        len += n;
    }
    out[len] = '\0';
    pclose(p);
    return out;
}

static void test_binary_rejects_a_host_option(void) {
    const char *exe = find_runner();
    if (!exe) {
        puts("test-bind: runner binary not built, skipping the CLI surface check");
        return;
    }

    char cmd[512];
    // --help is the advertised surface. It must offer --port and nothing that
    // sounds like an address.
    snprintf(cmd, sizeof cmd, "%s --help 2>&1", exe);
    char *help = run_capture(cmd);
    must_contain(help, "runner --help", "--port");
    static const char *opts[] = { "--host", "--bind", "--listen", "--address",
                                  "--interface", "--ip" };
    for (size_t i = 0; i < sizeof opts / sizeof *opts; i++)
        must_not_contain(help, "runner --help", opts[i],
                         "runner advertises an option that chooses a bind address.");
    free(help);

    // And they must actually be rejected, not merely undocumented.
    for (size_t i = 0; i < sizeof opts / sizeof *opts; i++) {
        snprintf(cmd, sizeof cmd, "%s %s 0.0.0.0 --caps 2>&1", exe, opts[i]);
        char *out = run_capture(cmd);
        if (!strstr(out, "unknown option")) {
            fprintf(stderr,
                    "test-bind: `runner %s 0.0.0.0` was not rejected as an unknown\n"
                    "  option. Something now accepts a bind address.\n", opts[i]);
            exit(1);
        }
        free(out);
    }
}

int main(void) {
    test_bind_address_is_a_literal_loopback_constant();
    test_no_option_reaches_the_bind_address();
    test_binary_rejects_a_host_option();
    puts("loopback bind invariant ok");
    return 0;
}
