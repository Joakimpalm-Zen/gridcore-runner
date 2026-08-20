#include "envelope.h"
#include "json.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Read the whole sidecar into a heap buffer. Manifests are small (a few KB);
// cap the read so a hostile/huge sidecar cannot exhaust memory.
#define ENV_MAX_BYTES (1 << 20)

static char *read_sidecar(const char *model_path, size_t *n_out) {
    size_t plen = strlen(model_path);
    char *path = malloc(plen + sizeof(".envelope.json"));
    if (!path) return NULL;
    memcpy(path, model_path, plen);
    memcpy(path + plen, ".envelope.json", sizeof(".envelope.json"));
    FILE *f = fopen(path, "rb");
    free(path);
    if (!f) return NULL;
    char *buf = malloc(ENV_MAX_BYTES);
    if (!buf) { fclose(f); return NULL; }
    size_t n = fread(buf, 1, ENV_MAX_BYTES - 1, f);
    fclose(f);
    buf[n] = 0;
    *n_out = n;
    return buf;
}

// Resolve a parsed manifest against this runtime, exact-match. Writes the
// one-line human summary and returns the state. Shared by the report path
// (slice 2) and the enforcing gate (slice 3) so both classify identically.
static int classify(jv *m, const char *runtime_version, const char *backend,
                    char *out, int cap) {
    const char *schema  = jv_str(jv_get(m, "schema_version"), "");
    jv *runtime         = jv_get(m, "runtime");
    const char *m_ver   = jv_str(jv_get(runtime, "version"), "");
    const char *m_back  = jv_str(jv_get(jv_get(runtime, "kernel_set"), "backend"), "");
    const char *verdict = jv_str(jv_get(m, "verdict"), "");

    // The certifier writes runtime.version straight from `--caps`, which is the
    // BARE version ("0.1.19-alpha") — NOT the "runner X" form `--version` prints.
    // An earlier revision here compared against "runner %s", so EVERY real
    // certifier-produced manifest resolved as foreign/indeterminate and never
    // matched its own runtime (the test fixtures used the "runner X" form and
    // masked it). Compare against the bare version, exactly what the manifest and
    // `--caps` carry.
    const char *rv = runtime_version ? runtime_version : "";
    bool ver_match  = m_ver[0]  && !strcmp(m_ver, rv);
    bool back_match = m_back[0] && backend && !strcmp(m_back, backend);

    if (strcmp(schema, "xyntetik.runner.envelope.v1") != 0) {
        // A manifest whose schema we do not understand is not evidence for THIS
        // runner — we cannot judge it, so it is indeterminate (not the same as a
        // measurement that came back inconclusive).
        snprintf(out, (size_t)cap,
                 "envelope: manifest schema %s not recognised (indeterminate)",
                 schema[0] ? schema : "(missing)");
        return ENV_INDETERMINATE;
    }
    if (ver_match && back_match) {
        if (!strcmp(verdict, "certified")) {
            snprintf(out, (size_t)cap,
                     "envelope: matches a measured envelope (certified: %s / %s)",
                     rv, backend);
            return ENV_CERTIFIED;
        }
        if (!strcmp(verdict, "outside-envelope")) {
            snprintf(out, (size_t)cap,
                     "envelope: OUTSIDE the measured envelope for %s / %s "
                     "(measured refusal)", rv, backend);
            return ENV_OUTSIDE;
        }
        if (!strcmp(verdict, "experimental")) {
            snprintf(out, (size_t)cap,
                     "envelope: measured for %s / %s, verdict experimental "
                     "(not certified)", rv, backend);
            return ENV_EXPERIMENTAL;
        }
        // Matching runtime but a verdict we do not recognise: we cannot judge it.
        snprintf(out, (size_t)cap,
                 "envelope: measured for %s / %s but its verdict %s is "
                 "unrecognised (indeterminate)", rv, backend,
                 verdict[0] ? verdict : "(missing)");
        return ENV_INDETERMINATE;
    }
    // Exact-match only: a manifest measured on a different version or backend
    // does not describe this configuration — foreign, so indeterminate here.
    snprintf(out, (size_t)cap,
             "envelope: measured on %s / %s, not this runtime (%s / %s) "
             "— indeterminate here",
             m_ver[0] ? m_ver : "(unknown)", m_back[0] ? m_back : "(unknown)",
             rv, backend ? backend : "(unknown)");
    return ENV_INDETERMINATE;
}

// The measured reason behind an outside-envelope verdict is the set of gate
// checks that FAILED — `quality.checks` is a {name: status} object. Fold the
// failing names into a human phrase; fall back to a generic line if the
// certifier recorded no per-check detail.
static void outside_reason(jv *m, char *out, int cap) {
    jv *checks = jv_get(jv_get(m, "quality"), "checks");
    int written = 0;
    if (checks && checks->type == J_OBJ) {
        for (int i = 0; i < checks->n; i++) {
            const char *status = jv_str(checks->items[i], "");
            if (strcmp(status, "fail") != 0) continue;
            written += snprintf(out + written, (size_t)(cap - written),
                                "%s%s", written ? ", " : "", checks->keys[i]);
            if (written >= cap - 1) break;   // out of room; stop cleanly
        }
    }
    if (written == 0)
        snprintf(out, (size_t)cap, "the configuration is outside what was measured");
}

int envelope_report(const char *model_path, const char *runtime_version,
                    const char *backend, char *out, int cap) {
    if (cap > 0) out[0] = 0;
    if (!model_path) return ENV_UNCLASSIFIED;

    size_t n = 0;
    char *text = read_sidecar(model_path, &n);
    if (!text) return ENV_UNCLASSIFIED;

    jv *m = json_parse(text, n);
    free(text);
    if (!m) {
        snprintf(out, (size_t)cap,
                 "envelope: manifest present but unreadable (indeterminate)");
        return ENV_INDETERMINATE;
    }
    int state = classify(m, runtime_version, backend, out, cap);
    jv_free(m);
    return state;
}

bool envelope_gate(const char *model_path, const char *runtime_version,
                   const char *backend, bool forced,
                   char *msg, int cap, int *out_state) {
    if (cap > 0) msg[0] = 0;
    if (out_state) *out_state = ENV_UNCLASSIFIED;
    if (!model_path) return true;

    size_t n = 0;
    char *text = read_sidecar(model_path, &n);
    if (!text) return true;                 // no manifest: nothing to enforce

    jv *m = json_parse(text, n);
    free(text);
    if (!m) {
        // Fail-open the SAFE way: an unreadable manifest is not evidence of a
        // refusal, so it never blocks a load — it is indeterminate, load on.
        snprintf(msg, (size_t)cap,
                 "envelope: manifest present but unreadable (indeterminate)");
        if (out_state) *out_state = ENV_INDETERMINATE;
        return true;
    }

    char summary[256];
    int state = classify(m, runtime_version, backend, summary, sizeof summary);
    if (out_state) *out_state = state;

    bool allow = true;
    if (state == ENV_OUTSIDE) {
        char reason[192];
        outside_reason(m, reason, sizeof reason);
        if (forced) {
            snprintf(msg, (size_t)cap,
                     "envelope: WARNING --force-uncertified: loading despite an "
                     "OUTSIDE-envelope verdict (%s)", reason);
        } else {
            snprintf(msg, (size_t)cap,
                     "envelope: refusing to load — OUTSIDE the measured envelope "
                     "(%s). Override with --force-uncertified.", reason);
            allow = false;
        }
    } else {
        // certified / experimental / indeterminate: informational banner
        // (never blocks). UNCLASSIFIED already returned above with an empty msg.
        snprintf(msg, (size_t)cap, "%s", summary);
    }

    jv_free(m);
    return allow;
}
