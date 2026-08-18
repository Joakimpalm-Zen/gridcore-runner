// libFuzzer harness for the HTTP request parsers (src/http.c) — the only
// hand-written parser in the tree that eats bytes straight off a socket,
// before anything has authenticated or framed them.
//
// The three functions are driven in the order and with the arguments
// handle_conn uses, because their contracts are joint: parse_request_line
// temporarily NUL-terminates the request line and hands back a pointer to the
// first header, and the other two walk from there to the blank line that
// handle_conn located with strstr. Feeding them independently would test
// three functions that never run that way.
//
// The buffer is NUL-terminated exactly as the read loop leaves it, and it is
// heap-allocated at the input's own size so ASan's redzone sits immediately
// after the last byte: a scan that runs one past the blank line is a report
// here and silent corruption in production.
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "http.h"

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size < 4 || size > 1u << 20) return 0;

    char *buf = malloc(size + 1);
    if (!buf) return 0;
    memcpy(buf, data, size);
    buf[size] = 0;

    // handle_conn only parses once the blank line has arrived; without one it
    // answers 400 and never reaches these functions.
    char *header_end = strstr(buf, "\r\n\r\n");
    if (!header_end) { free(buf); return 0; }

    char method[8] = {0}, path[256] = {0};
    char *first_header = NULL;
    bool line_ok = parse_request_line(buf, method, path, &first_header);

    // first_header is set even when the request line is rejected, and it can
    // point past header_end when the line IS the last thing before the blank
    // line. Both are states handle_conn can be in, so both are fuzzed.
    if (first_header) {
        size_t content_length = 0;
        (void)parse_request_framing(first_header, header_end, &content_length);
        (void)validate_request_authority(first_header, header_end);
    }
    (void)line_ok;
    free(buf);
    return 0;
}
