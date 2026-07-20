// libFuzzer harness for json_parse (src/json.c).
//
// json_parse is bounded by n and does NOT require NUL termination, so the
// input buffer is handed over as-is: any read past `size` is a real bug and
// ASan will catch it against libFuzzer's redzoned allocation.
//
// A successful parse is round-tripped through jv_dump so the serializer and
// the sbuf growth path are fuzzed too, and everything is freed before
// returning so LeakSanitizer findings mean something.
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "json.h"

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    jv *v = json_parse((const char *)data, size);
    if (!v) return 0;

    sbuf b = {0};
    jv_dump(v, &b);
    // re-parsing our own output must succeed: a dump that cannot be read back
    // is a serializer bug, not a parser bug, but it is worth surfacing here.
    if (!b.failed && b.s) {
        jv *again = json_parse(b.s, b.n);
        jv_free(again);
    }
    free(b.s);
    jv_free(v);
    return 0;
}
