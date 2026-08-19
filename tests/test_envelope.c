// envelope_report reads a <model>.envelope.json sidecar and resolves the
// current (version, backend) against it, exact-match, without enforcement.
#include "envelope.h"
#include "runner.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

static const char *MODEL = "/tmp/xyntetik-envelope-test.gguf";

static void write_manifest(const char *body) {
    char path[512];
    snprintf(path, sizeof path, "%s.envelope.json", MODEL);
    FILE *f = fopen(path, "wb");
    assert(f);
    fputs(body, f);
    fclose(f);
}
static void rm_manifest(void) {
    char path[512];
    snprintf(path, sizeof path, "%s.envelope.json", MODEL);
    remove(path);
}

static const char *manifest(const char *version, const char *backend,
                            const char *verdict) {
    static char buf[1024];
    snprintf(buf, sizeof buf,
             "{\"schema_version\":\"xyntetik.runner.envelope.v1\","
             "\"runtime\":{\"version\":\"runner %s\","
             "\"kernel_set\":{\"backend\":\"%s\"}},"
             "\"verdict\":\"%s\"}",
             version, backend, verdict);
    return buf;
}

int main(void) {
    char out[256];

    // No sidecar -> silent, no state.
    rm_manifest();
    assert(envelope_report(MODEL, RUNNER_VERSION, "cpu", out, sizeof out) == ENV_NONE);
    assert(out[0] == 0);

    // Exact match + certified -> certified, and it says so.
    write_manifest(manifest(RUNNER_VERSION, "cpu", "certified"));
    assert(envelope_report(MODEL, RUNNER_VERSION, "cpu", out, sizeof out) == ENV_CERTIFIED);
    assert(strstr(out, "certified") && strstr(out, "measured envelope"));

    // Exact match + outside-envelope -> outside (reported, not enforced).
    write_manifest(manifest(RUNNER_VERSION, "cpu", "outside-envelope"));
    assert(envelope_report(MODEL, RUNNER_VERSION, "cpu", out, sizeof out) == ENV_OUTSIDE);
    assert(strstr(out, "OUTSIDE"));

    // Exact-match only: a manifest measured on a DIFFERENT backend does not
    // describe this configuration -> experimental.
    write_manifest(manifest(RUNNER_VERSION, "cpu", "certified"));
    assert(envelope_report(MODEL, RUNNER_VERSION, "cuda", out, sizeof out) == ENV_EXPERIMENTAL);

    // ... nor a different runtime version.
    write_manifest(manifest("0.0.0-old", "cpu", "certified"));
    assert(envelope_report(MODEL, RUNNER_VERSION, "cpu", out, sizeof out) == ENV_EXPERIMENTAL);

    // Unknown manifest schema is not trusted for this runner.
    write_manifest("{\"schema_version\":\"other.v9\",\"verdict\":\"certified\"}");
    assert(envelope_report(MODEL, RUNNER_VERSION, "cpu", out, sizeof out) == ENV_EXPERIMENTAL);

    // A present-but-malformed sidecar is experimental, never a crash or a load
    // failure.
    write_manifest("{ this is not json");
    assert(envelope_report(MODEL, RUNNER_VERSION, "cpu", out, sizeof out) == ENV_EXPERIMENTAL);

    // ---- slice 3: the enforcing gate ----------------------------------------
    int st = -1;

    // No manifest -> load, silent, state NONE.
    rm_manifest();
    assert(envelope_gate(MODEL, RUNNER_VERSION, "cpu", false, out, sizeof out, &st) == true);
    assert(out[0] == 0 && st == ENV_NONE);

    // Certified -> load, with the informational banner.
    write_manifest(manifest(RUNNER_VERSION, "cpu", "certified"));
    assert(envelope_gate(MODEL, RUNNER_VERSION, "cpu", false, out, sizeof out, &st) == true);
    assert(st == ENV_CERTIFIED && strstr(out, "certified"));

    // Experimental (matches nothing) -> load, banner, never a refusal.
    write_manifest(manifest(RUNNER_VERSION, "cuda", "certified"));
    assert(envelope_gate(MODEL, RUNNER_VERSION, "cpu", false, out, sizeof out, &st) == true);
    assert(st == ENV_EXPERIMENTAL);

    // Outside-envelope -> REFUSE (returns false), and the message tells the
    // user how to override.
    write_manifest(manifest(RUNNER_VERSION, "cpu", "outside-envelope"));
    assert(envelope_gate(MODEL, RUNNER_VERSION, "cpu", false, out, sizeof out, &st) == false);
    assert(st == ENV_OUTSIDE && strstr(out, "refusing") && strstr(out, "--force-uncertified"));

    // ...unless --force-uncertified is set: then it loads with a loud warning.
    assert(envelope_gate(MODEL, RUNNER_VERSION, "cpu", true, out, sizeof out, &st) == true);
    assert(st == ENV_OUTSIDE && strstr(out, "WARNING") && strstr(out, "force-uncertified"));

    // The refusal names the measured reason: the gate check(s) that FAILED.
    write_manifest("{\"schema_version\":\"xyntetik.runner.envelope.v1\","
                   "\"runtime\":{\"version\":\"runner " RUNNER_VERSION "\","
                   "\"kernel_set\":{\"backend\":\"cpu\"}},"
                   "\"verdict\":\"outside-envelope\","
                   "\"quality\":{\"checks\":{\"cpu_gpu_identity\":\"pass\","
                   "\"ram_fits\":\"fail\"}}}");
    assert(envelope_gate(MODEL, RUNNER_VERSION, "cpu", false, out, sizeof out, &st) == false);
    assert(strstr(out, "ram_fits"));

    // Fail-open: a malformed manifest never blocks a load.
    write_manifest("{ this is not json");
    assert(envelope_gate(MODEL, RUNNER_VERSION, "cpu", false, out, sizeof out, &st) == true);
    assert(st == ENV_EXPERIMENTAL);

    rm_manifest();
    printf("test-envelope: OK\n");
    return 0;
}
