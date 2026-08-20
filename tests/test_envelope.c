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
             "\"runtime\":{\"version\":\"%s\","
             "\"kernel_set\":{\"backend\":\"%s\"}},"
             "\"verdict\":\"%s\"}",
             version, backend, verdict);
    return buf;
}

// Same, but the manifest also carries an artifact.sha256 (Unit 7).
static const char *manifest_sha(const char *version, const char *backend,
                                const char *verdict, const char *sha) {
    static char buf[1024];
    snprintf(buf, sizeof buf,
             "{\"schema_version\":\"xyntetik.runner.envelope.v1\","
             "\"artifact\":{\"sha256\":\"%s\"},"
             "\"runtime\":{\"version\":\"%s\","
             "\"kernel_set\":{\"backend\":\"%s\"}},"
             "\"verdict\":\"%s\"}",
             sha, version, backend, verdict);
    return buf;
}

// Write a model file (not just its sidecar) so artifact.sha256 has a real file
// to hash. "abc" is the classic FIPS 180-4 SHA-256 test vector.
static void write_model(const char *bytes, size_t n) {
    FILE *f = fopen(MODEL, "wb");
    assert(f);
    fwrite(bytes, 1, n, f);
    fclose(f);
}
static void rm_model(void) { remove(MODEL); }
static const char *ABC_SHA =
    "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
static const char *ZERO_SHA =
    "0000000000000000000000000000000000000000000000000000000000000000";

int main(void) {
    char out[256];

    // No sidecar -> silent, unclassified (transitional/legacy, not experimental).
    rm_manifest();
    assert(envelope_report(MODEL, RUNNER_VERSION, "cpu", out, sizeof out) == ENV_UNCLASSIFIED);
    assert(out[0] == 0);

    // Exact match + certified -> certified, and it says so.
    write_manifest(manifest(RUNNER_VERSION, "cpu", "certified"));
    assert(envelope_report(MODEL, RUNNER_VERSION, "cpu", out, sizeof out) == ENV_CERTIFIED);
    assert(strstr(out, "certified") && strstr(out, "measured envelope"));

    // Exact match + outside-envelope -> outside (reported, not enforced).
    write_manifest(manifest(RUNNER_VERSION, "cpu", "outside-envelope"));
    assert(envelope_report(MODEL, RUNNER_VERSION, "cpu", out, sizeof out) == ENV_OUTSIDE);
    assert(strstr(out, "OUTSIDE"));

    // Exact match + verdict "experimental" -> EXPERIMENTAL: a real measurement
    // that came back inconclusive (NOT the same as one we could not read).
    write_manifest(manifest(RUNNER_VERSION, "cpu", "experimental"));
    assert(envelope_report(MODEL, RUNNER_VERSION, "cpu", out, sizeof out) == ENV_EXPERIMENTAL);
    assert(strstr(out, "experimental"));

    // Exact-match only: a manifest measured on a DIFFERENT backend does not
    // describe this configuration -> INDETERMINATE (foreign, could not judge).
    write_manifest(manifest(RUNNER_VERSION, "cpu", "certified"));
    assert(envelope_report(MODEL, RUNNER_VERSION, "cuda", out, sizeof out) == ENV_INDETERMINATE);

    // ... nor a different runtime version.
    write_manifest(manifest("0.0.0-old", "cpu", "certified"));
    assert(envelope_report(MODEL, RUNNER_VERSION, "cpu", out, sizeof out) == ENV_INDETERMINATE);

    // Unknown manifest schema is not trusted for this runner -> indeterminate.
    write_manifest("{\"schema_version\":\"other.v9\",\"verdict\":\"certified\"}");
    assert(envelope_report(MODEL, RUNNER_VERSION, "cpu", out, sizeof out) == ENV_INDETERMINATE);

    // A present-but-malformed sidecar is indeterminate, never a crash or a load
    // failure.
    write_manifest("{ this is not json");
    assert(envelope_report(MODEL, RUNNER_VERSION, "cpu", out, sizeof out) == ENV_INDETERMINATE);

    // ---- tool-calling axis (reported-only): summarised in the banner --------
    // A manifest carrying a `tool_calling` block makes envelope_report append a
    // one-line, report-only summary of it. It NEVER changes the resolved state.
    write_manifest("{\"schema_version\":\"xyntetik.runner.envelope.v1\","
                   "\"runtime\":{\"version\":\"" RUNNER_VERSION "\","
                   "\"kernel_set\":{\"backend\":\"cpu\"}},"
                   "\"verdict\":\"certified\","
                   "\"tool_calling\":{"
                   "\"truncation_recovery\":{\"rungs_passed\":6,\"rungs_total\":6},"
                   "\"schema_shape\":{\"held_to_quant\":\"Q4_0\"},"
                   "\"agent_torture\":{\"gate\":\"pass\"},"
                   "\"native_tool_protocol\":{\"tool_family\":\"granite\","
                   "\"native\":true},"
                   "\"gate\":\"pass\"}}");
    assert(envelope_report(MODEL, RUNNER_VERSION, "cpu", out, sizeof out) == ENV_CERTIFIED);
    assert(strstr(out, "tool-calling gate=pass"));
    assert(strstr(out, "truncation 6/6") && strstr(out, "schema-shape@Q4_0"));
    assert(strstr(out, "agent-torture pass") && strstr(out, "native granite"));

    // The same manifest WITHOUT a tool_calling block emits no such line
    // (back-compat: silent for manifests that predate the axis).
    write_manifest(manifest(RUNNER_VERSION, "cpu", "certified"));
    assert(envelope_report(MODEL, RUNNER_VERSION, "cpu", out, sizeof out) == ENV_CERTIFIED);
    assert(!strstr(out, "tool-calling"));

    // ---- slice 3: the enforcing gate ----------------------------------------
    int st = -1;

    // No manifest -> load, silent, unclassified.
    rm_manifest();
    assert(envelope_gate(MODEL, RUNNER_VERSION, "cpu", false, out, sizeof out, &st) == true);
    assert(out[0] == 0 && st == ENV_UNCLASSIFIED);

    // Certified -> load, with the informational banner.
    write_manifest(manifest(RUNNER_VERSION, "cpu", "certified"));
    assert(envelope_gate(MODEL, RUNNER_VERSION, "cpu", false, out, sizeof out, &st) == true);
    assert(st == ENV_CERTIFIED && strstr(out, "certified"));

    // Matching verdict "experimental" -> load, banner, never a refusal.
    write_manifest(manifest(RUNNER_VERSION, "cpu", "experimental"));
    assert(envelope_gate(MODEL, RUNNER_VERSION, "cpu", false, out, sizeof out, &st) == true);
    assert(st == ENV_EXPERIMENTAL && strstr(out, "experimental"));

    // Foreign sidecar (measured on another backend) -> INDETERMINATE, loads
    // with a banner, never a refusal.
    write_manifest(manifest(RUNNER_VERSION, "cuda", "certified"));
    assert(envelope_gate(MODEL, RUNNER_VERSION, "cpu", false, out, sizeof out, &st) == true);
    assert(st == ENV_INDETERMINATE && out[0]);

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
                   "\"runtime\":{\"version\":\"" RUNNER_VERSION "\","
                   "\"kernel_set\":{\"backend\":\"cpu\"}},"
                   "\"verdict\":\"outside-envelope\","
                   "\"quality\":{\"checks\":{\"cpu_gpu_identity\":\"pass\","
                   "\"ram_fits\":\"fail\"}}}");
    assert(envelope_gate(MODEL, RUNNER_VERSION, "cpu", false, out, sizeof out, &st) == false);
    assert(strstr(out, "ram_fits"));

    // Fail-open: a malformed manifest is indeterminate, never blocks a load.
    write_manifest("{ this is not json");
    assert(envelope_gate(MODEL, RUNNER_VERSION, "cpu", false, out, sizeof out, &st) == true);
    assert(st == ENV_INDETERMINATE);

    // ---- Unit 7: artifact.sha256 must match the loaded file -----------------
    // A sidecar carries an artifact.sha256; the manifest resolves on
    // (version, backend) only, so nothing else proves it belongs to THIS file.
    // Verify the hash before applying any verdict.
    write_model("abc", 3);

    // Matching sha + outside-envelope -> the verdict STILL applies (refuse).
    write_manifest(manifest_sha(RUNNER_VERSION, "cpu", "outside-envelope", ABC_SHA));
    assert(envelope_gate(MODEL, RUNNER_VERSION, "cpu", false, out, sizeof out, &st) == false);
    assert(st == ENV_OUTSIDE && strstr(out, "refusing"));
    assert(envelope_report(MODEL, RUNNER_VERSION, "cpu", out, sizeof out) == ENV_OUTSIDE);

    // WRONG sha -> the sidecar names a different artifact: its outside-envelope
    // verdict has NO effect (loads), state INDETERMINATE, and it says why.
    write_manifest(manifest_sha(RUNNER_VERSION, "cpu", "outside-envelope", ZERO_SHA));
    assert(envelope_gate(MODEL, RUNNER_VERSION, "cpu", false, out, sizeof out, &st) == true);
    assert(st == ENV_INDETERMINATE && strstr(out, "does not match"));
    assert(envelope_report(MODEL, RUNNER_VERSION, "cpu", out, sizeof out) == ENV_INDETERMINATE);
    assert(strstr(out, "does not match"));

    // A matching-sha certified sidecar still certifies (the check gates on
    // identity, it does not weaken a legitimate verdict).
    write_manifest(manifest_sha(RUNNER_VERSION, "cpu", "certified", ABC_SHA));
    assert(envelope_gate(MODEL, RUNNER_VERSION, "cpu", false, out, sizeof out, &st) == true);
    assert(st == ENV_CERTIFIED && strstr(out, "certified"));

    // Back-compat: a manifest with NO artifact.sha256 behaves exactly as before
    // even when the model file exists — the outside verdict still enforces.
    write_manifest(manifest(RUNNER_VERSION, "cpu", "outside-envelope"));
    assert(envelope_gate(MODEL, RUNNER_VERSION, "cpu", false, out, sizeof out, &st) == false);
    assert(st == ENV_OUTSIDE);

    rm_model();
    rm_manifest();
    printf("test-envelope: OK\n");
    return 0;
}
