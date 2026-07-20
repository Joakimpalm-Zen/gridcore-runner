// libFuzzer harness for schema_compile (src/runner.h:359, src/schema.c).
//
// Input is JSON text: it is parsed with json_parse and, if that succeeds,
// handed to schema_compile. Inputs that are not valid JSON are discarded
// early, so the corpus naturally evolves toward JSON-shaped schemas.
//
// schema_compile owns a recursive tree; schema_free must return it all, so a
// leak here is a real defect and not harness noise.
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>

#include "json.h"
#include "runner.h"

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    jv *j = json_parse((const char *)data, size);
    if (!j) return 0;

    char err[256];
    snode *s = schema_compile(j, err, (int)sizeof(err));
    // compile failure must still report a NUL-terminated reason
    schema_free(s);
    jv_free(j);
    return 0;
}
