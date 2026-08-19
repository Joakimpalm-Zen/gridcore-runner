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

int envelope_report(const char *model_path, const char *runtime_version,
                    const char *backend, char *out, int cap) {
    if (cap > 0) out[0] = 0;
    if (!model_path) return ENV_NONE;

    size_t n = 0;
    char *text = read_sidecar(model_path, &n);
    if (!text) return ENV_NONE;

    jv *m = json_parse(text, n);
    free(text);
    if (!m) {
        snprintf(out, (size_t)cap,
                 "envelope: manifest present but unreadable (treated as experimental)");
        return ENV_EXPERIMENTAL;
    }

    const char *schema  = jv_str(jv_get(m, "schema_version"), "");
    jv *runtime         = jv_get(m, "runtime");
    const char *m_ver   = jv_str(jv_get(runtime, "version"), "");
    const char *m_back  = jv_str(jv_get(jv_get(runtime, "kernel_set"), "backend"), "");
    const char *verdict = jv_str(jv_get(m, "verdict"), "");

    // `runner --version` prints "runner X"; the manifest records that whole
    // string, so compare against the same shape.
    char rv[64];
    snprintf(rv, sizeof rv, "runner %s", runtime_version ? runtime_version : "");
    bool ver_match  = m_ver[0]  && !strcmp(m_ver, rv);
    bool back_match = m_back[0] && backend && !strcmp(m_back, backend);

    int state;
    if (strcmp(schema, "xyntetik.runner.envelope.v1") != 0) {
        // A manifest whose schema we do not understand is not evidence for THIS
        // runner — report it, do not trust it.
        snprintf(out, (size_t)cap,
                 "envelope: manifest schema %s not recognised (experimental)",
                 schema[0] ? schema : "(missing)");
        state = ENV_EXPERIMENTAL;
    } else if (ver_match && back_match) {
        if (!strcmp(verdict, "certified")) {
            snprintf(out, (size_t)cap,
                     "envelope: matches a measured envelope (certified: %s / %s)",
                     rv, backend);
            state = ENV_CERTIFIED;
        } else if (!strcmp(verdict, "outside-envelope")) {
            snprintf(out, (size_t)cap,
                     "envelope: OUTSIDE the measured envelope for %s / %s "
                     "(measured refusal; not enforced)", rv, backend);
            state = ENV_OUTSIDE;
        } else {
            snprintf(out, (size_t)cap,
                     "envelope: measured for %s / %s but not certified (experimental)",
                     rv, backend);
            state = ENV_EXPERIMENTAL;
        }
    } else {
        // Exact-match only: a manifest measured on a different version or
        // backend does not describe this configuration.
        snprintf(out, (size_t)cap,
                 "envelope: measured on %s / %s, not this runtime (%s / %s) "
                 "— experimental here",
                 m_ver[0] ? m_ver : "(unknown)", m_back[0] ? m_back : "(unknown)",
                 rv, backend ? backend : "(unknown)");
        state = ENV_EXPERIMENTAL;
    }

    jv_free(m);
    return state;
}
