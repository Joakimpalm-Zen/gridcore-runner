// libFuzzer harness for jsonv_feed (src/jsonmode.c) — the incremental JSON
// validator behind --json.
//
// jsonv is a fixed-size, memcpy-copyable struct with no heap of its own, so
// the interesting failures are stack-buffer overflows on the nesting stack
// and out-of-range writes from jsonv_close's repair buffer.
//
// The first input byte picks the entry point (jsonv_init vs jsonv_init_any);
// the rest is fed in *ragged chunks* rather than one call, because the whole
// point of a streaming validator is that its state survives a split at an
// arbitrary byte. A single feed of the whole buffer would never exercise
// that.
#include <stdint.h>
#include <stddef.h>
#include <string.h>

#include "runner.h"

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size < 1) return 0;

    jsonv v;
    if (data[0] & 1) jsonv_init_any(&v);
    else             jsonv_init(&v);
    data++; size--;

    // chunk lengths come from the data itself: cheap, deterministic, and it
    // makes the mutator able to steer the split points
    size_t i = 0;
    while (i < size) {
        size_t chunk = (size_t)(data[i] & 0x0f) + 1;
        if (chunk > size - i) chunk = size - i;
        if (!jsonv_feed(&v, (const char *)data + i, (int)chunk)) break;
        (void)jsonv_value_end(&v);
        i += chunk;
    }

    // close must always terminate and must stay inside `out`
    char out[512];
    int n = jsonv_close(&v, out, (int)sizeof(out));
    if (n < 0 || (size_t)n > sizeof(out)) __builtin_trap();
    return 0;
}
