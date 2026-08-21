#include "envelope.h"
#include "json.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ---- SHA-256 (FIPS 180-4) --------------------------------------------------
// Used only to verify a sidecar's artifact.sha256 against the file that was
// actually loaded (Unit 7). Kept local so this stays a leaf translation unit:
// the C test links envelope.c + json.c and nothing else, and no other caller
// needs a hash. It streams the file in a fixed buffer, so a large GGUF costs
// one sequential pass and no extra memory. Runs ONLY when a sidecar carrying a
// sha is present, so the ordinary no-manifest load never touches it.
typedef struct {
    uint32_t h[8];
    uint64_t len;        // total bytes fed
    uint8_t  buf[64];
    size_t   n;          // bytes buffered in `buf`
} sha256_ctx;

static uint32_t sha_rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

static void sha256_init(sha256_ctx *c) {
    static const uint32_t iv[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    memcpy(c->h, iv, sizeof iv);
    c->len = 0;
    c->n = 0;
}

static void sha256_block(sha256_ctx *c, const uint8_t *p) {
    static const uint32_t k[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
        0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
        0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
        0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
        0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
        0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
    uint32_t w[64];
    for (int i = 0; i < 16; i++)
        w[i] = (uint32_t)p[i * 4] << 24 | (uint32_t)p[i * 4 + 1] << 16 |
               (uint32_t)p[i * 4 + 2] << 8 | (uint32_t)p[i * 4 + 3];
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = sha_rotr(w[i - 15], 7) ^ sha_rotr(w[i - 15], 18) ^
                      (w[i - 15] >> 3);
        uint32_t s1 = sha_rotr(w[i - 2], 17) ^ sha_rotr(w[i - 2], 19) ^
                      (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = c->h[0], b = c->h[1], cc = c->h[2], d = c->h[3];
    uint32_t e = c->h[4], f = c->h[5], g = c->h[6], hh = c->h[7];
    for (int i = 0; i < 64; i++) {
        uint32_t S1 = sha_rotr(e, 6) ^ sha_rotr(e, 11) ^ sha_rotr(e, 25);
        uint32_t ch = (e & f) ^ (~e & g);
        uint32_t t1 = hh + S1 + ch + k[i] + w[i];
        uint32_t S0 = sha_rotr(a, 2) ^ sha_rotr(a, 13) ^ sha_rotr(a, 22);
        uint32_t maj = (a & b) ^ (a & cc) ^ (b & cc);
        uint32_t t2 = S0 + maj;
        hh = g; g = f; f = e; e = d + t1; d = cc; cc = b; b = a; a = t1 + t2;
    }
    c->h[0] += a; c->h[1] += b; c->h[2] += cc; c->h[3] += d;
    c->h[4] += e; c->h[5] += f; c->h[6] += g; c->h[7] += hh;
}

static void sha256_update(sha256_ctx *c, const uint8_t *p, size_t n) {
    c->len += n;
    while (n) {
        size_t take = 64 - c->n;
        if (take > n) take = n;
        memcpy(c->buf + c->n, p, take);
        c->n += take;
        p += take;
        n -= take;
        if (c->n == 64) { sha256_block(c, c->buf); c->n = 0; }
    }
}

static void sha256_final(sha256_ctx *c, uint8_t out[32]) {
    uint64_t bits = c->len * 8;   // captured before padding grows c->len
    uint8_t one = 0x80, zero = 0;
    sha256_update(c, &one, 1);
    while (c->n != 56) sha256_update(c, &zero, 1);
    uint8_t lenb[8];
    for (int i = 0; i < 8; i++) lenb[i] = (uint8_t)(bits >> (56 - i * 8));
    sha256_update(c, lenb, 8);
    for (int i = 0; i < 8; i++) {
        out[i * 4]     = (uint8_t)(c->h[i] >> 24);
        out[i * 4 + 1] = (uint8_t)(c->h[i] >> 16);
        out[i * 4 + 2] = (uint8_t)(c->h[i] >> 8);
        out[i * 4 + 3] = (uint8_t)(c->h[i]);
    }
}

// Hash a whole file to lowercase hex. Returns false if it cannot be read.
bool envelope_file_sha256(const char *path, char hex[65]) {
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    sha256_ctx c;
    sha256_init(&c);
    uint8_t buf[1 << 16];
    size_t r;
    while ((r = fread(buf, 1, sizeof buf, f)) > 0) sha256_update(&c, buf, r);
    bool ok = !ferror(f);
    fclose(f);
    if (!ok) return false;
    uint8_t d[32];
    sha256_final(&c, d);
    static const char hexd[] = "0123456789abcdef";
    for (int i = 0; i < 32; i++) {
        hex[i * 2]     = hexd[d[i] >> 4];
        hex[i * 2 + 1] = hexd[d[i] & 15];
    }
    hex[64] = 0;
    return true;
}

// Case-insensitive hex-string equality (a manifest sha may be written in any
// case; hashes from Python's hexdigest are lowercase).
static bool hex_eq_ci(const char *a, const char *b) {
    for (;; a++, b++) {
        char ca = *a, cb = *b;
        if (ca >= 'A' && ca <= 'F') ca += 32;
        if (cb >= 'A' && cb <= 'F') cb += 32;
        if (ca != cb) return false;
        if (!ca) return true;
    }
}

// Trust check tying a sidecar to THE file loaded (Unit 7). A manifest resolves
// on (runtime.version, backend) only; nothing else proves the sidecar belongs
// beside THIS artifact, so a misplaced or stale sidecar for another model would
// otherwise be trusted. When the manifest carries `artifact.sha256`, hash the
// loaded file and require a match before any verdict is applied.
//   returns true   -> no sha in the manifest (back-compat, unchanged behavior),
//                     or it matches: proceed to classify.
//   returns false  -> the manifest names a DIFFERENT artifact (or the file
//                     cannot be hashed to confirm identity): the sidecar is
//                     unusable here and must be treated as INDETERMINATE, never
//                     enforced. `out` carries the reason.
static bool sidecar_matches_artifact(jv *m, const char *model_path,
                                     char *out, int cap) {
    const char *want = jv_str(jv_get(jv_get(m, "artifact"), "sha256"), "");
    if (!want[0]) return true;   // no sha declared: hash nothing, behave as before
    char have[65];
    if (!envelope_file_sha256(model_path, have)) {
        snprintf(out, (size_t)cap,
                 "envelope: could not hash this artifact to verify the sidecar "
                 "— ignored");
        return false;
    }
    if (!hex_eq_ci(have, want)) {
        snprintf(out, (size_t)cap,
                 "envelope: sidecar sha does not match this artifact — ignored");
        return false;
    }
    return true;
}

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

// REPORT-ONLY: append a one-line summary of the manifest's optional
// `tool_calling` block (the reported-only tool-calling axis the certifier
// emits) after whatever classify() wrote into `out`. It is surfaced for the
// operator and NEVER influences the load/gate decision or the resolved
// envelope_state — the load path reads only classify()'s verdict. Emits
// nothing when the manifest carries no `tool_calling` block, or when none of
// its sub-fields are present, so manifests that predate the axis stay silent:
// back-compatible by construction. Only the sub-fields actually present are
// shown; a null sub-block is simply omitted from the line.
static void append_tool_calling(jv *m, char *out, int cap) {
    jv *tc = jv_get(m, "tool_calling");
    if (!tc || tc->type != J_OBJ) return;

    char parts[192];
    int p = 0;
    const char *sep = "";
#define TC_REM (p < (int)sizeof parts ? (int)sizeof parts - p : 0)

    jv *tr = jv_get(tc, "truncation_recovery");
    if (tr && tr->type == J_OBJ) {
        int passed = (int)jv_num(jv_get(tr, "rungs_passed"), 0);
        int total  = (int)jv_num(jv_get(tr, "rungs_total"), 0);
        p += snprintf(parts + p, (size_t)TC_REM, "%struncation %d/%d",
                      sep, passed, total);
        sep = ", ";
    }
    jv *ss = jv_get(tc, "schema_shape");
    if (ss && ss->type == J_OBJ) {
        const char *q = jv_str(jv_get(ss, "held_to_quant"), "");
        if (q[0]) {
            p += snprintf(parts + p, (size_t)TC_REM, "%sschema-shape@%s", sep, q);
            sep = ", ";
        }
    }
    jv *at = jv_get(tc, "agent_torture");
    if (at && at->type == J_OBJ) {
        const char *g = jv_str(jv_get(at, "gate"), "");
        if (g[0]) {
            p += snprintf(parts + p, (size_t)TC_REM, "%sagent-torture %s", sep, g);
            sep = ", ";
        }
    }
    jv *nt = jv_get(tc, "native_tool_protocol");
    if (nt && nt->type == J_OBJ) {
        const char *fam = jv_str(jv_get(nt, "tool_family"), "");
        if (fam[0]) {
            bool native = jv_bool(jv_get(nt, "native"), false);
            p += snprintf(parts + p, (size_t)TC_REM, "%s%s %s",
                          sep, native ? "native" : "non-native", fam);
            sep = ", ";
        }
    }
#undef TC_REM
    if (p == 0) return;   // block present but no usable sub-fields — stay silent

    // Append as a SECOND line so it never disturbs the classify summary the
    // gate decision is based on; the caller prints the whole buffer with one %s.
    int len = (int)strlen(out);
    int rem = cap > len ? cap - len : 0;
    const char *gate = jv_str(jv_get(tc, "gate"), "");
    if (gate[0])
        snprintf(out + len, (size_t)rem,
                 "\nenvelope: tool-calling gate=%s — %s", gate, parts);
    else
        snprintf(out + len, (size_t)rem,
                 "\nenvelope: tool-calling — %s", parts);
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
    // A sidecar that names a different artifact is not evidence for this model.
    if (!sidecar_matches_artifact(m, model_path, out, cap)) {
        jv_free(m);
        return ENV_INDETERMINATE;
    }
    int state = classify(m, runtime_version, backend, out, cap);
    // Reported-only: surface the tool-calling axis when present. Never affects
    // the returned state.
    append_tool_calling(m, out, cap);
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

    // A sidecar whose artifact.sha256 does not match the loaded file is NOT
    // this model's manifest: its verdict must have no effect. Treat it like any
    // other unusable sidecar — indeterminate, fail-open, never a refusal.
    if (!sidecar_matches_artifact(m, model_path, msg, cap)) {
        if (out_state) *out_state = ENV_INDETERMINATE;
        jv_free(m);
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

    // Reported-only: append the tool-calling axis summary to whatever banner
    // was chosen above. This is surfaced for the operator and NEVER changes the
    // load decision already made in `allow` (including the OUTSIDE refusal).
    append_tool_calling(m, msg, cap);

    jv_free(m);
    return allow;
}
