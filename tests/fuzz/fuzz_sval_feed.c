// libFuzzer harness for sval_feed (src/schema.c:827) — the schema-constrained
// streaming validator.
//
// sval_feed needs *two* untrusted things: a compiled schema and a byte stream.
// The input is therefore split with a 2-byte little-endian length prefix:
//
//     [lo][hi][schema JSON, `len` bytes][stream bytes...]
//
// A length prefix survives mutation far better than a textual separator, which
// the mutator deletes almost immediately.
//
// If the schema half does not compile, a fixed fallback schema is used instead
// of returning early. Random bytes rarely compile as a schema, and bailing out
// would mean nearly every execution never reached sval_feed at all — the
// target would look well-fuzzed while being barely touched.
#include <stdint.h>
#include <stddef.h>
#include <string.h>

#include "json.h"
#include "runner.h"

// exercises objects, required props, enums, unions, arrays with bounds and
// nesting — i.e. most of the sframe machinery
static const char FALLBACK[] =
    "{\"type\":\"object\","
    "\"properties\":{"
      "\"a\":{\"type\":\"string\",\"enum\":[\"x\",\"y\"]},"
      "\"b\":{\"type\":\"integer\"},"
      "\"c\":{\"type\":\"array\",\"items\":{\"type\":\"number\"},"
             "\"minItems\":1,\"maxItems\":4},"
      "\"d\":{\"type\":[\"string\",\"null\"]},"
      "\"e\":{\"type\":\"object\",\"properties\":{\"f\":{\"type\":\"boolean\"}}}},"
    "\"required\":[\"a\",\"b\"]}";

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size < 2) return 0;
    size_t len = (size_t)data[0] | ((size_t)data[1] << 8);
    data += 2; size -= 2;
    if (len > size) len = size;

    char err[256];
    jv    *j = json_parse((const char *)data, len);
    snode *s = j ? schema_compile(j, err, (int)sizeof(err)) : NULL;
    jv_free(j);

    jv *fj = NULL;
    if (!s) {
        fj = json_parse(FALLBACK, sizeof(FALLBACK) - 1);
        s  = fj ? schema_compile(fj, err, (int)sizeof(err)) : NULL;
        if (!s) { jv_free(fj); return 0; }   // fallback must compile; nothing to do if not
    }

    const uint8_t *stream = data + len;
    size_t         n      = size - len;

    sval v;
    sval_init(&v, s);
    // ragged chunks: a streaming validator's state must survive a split at any
    // byte, and a single whole-buffer feed would never test that
    for (size_t i = 0; i < n; ) {
        size_t chunk = (size_t)(stream[i] & 0x0f) + 1;
        if (chunk > n - i) chunk = n - i;
        if (!sval_feed(&v, (const char *)stream + i, (int)chunk)) break;
        i += chunk;
    }

    char out[1024];
    int  wrote = sval_close(&v, out, (int)sizeof(out));
    if (wrote < 0 || (size_t)wrote > sizeof(out)) __builtin_trap();

    schema_free(s);
    jv_free(fj);
    return 0;
}
