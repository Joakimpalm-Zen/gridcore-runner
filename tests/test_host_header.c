#include "http.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

static bool valid(const char *text) {
    char buf[512];
    snprintf(buf, sizeof(buf), "%s", text);
    char *end = strstr(buf, "\r\n\r\n");
    assert(end);
    return validate_request_authority(buf, end);
}

int main(void) {
    assert(strcmp(reason_phrase(405), "Method Not Allowed") == 0);
    assert(valid("Host: 127.0.0.1:8080\r\n\r\n"));
    assert(valid("host: localhost\r\n\r\n"));
    assert(valid("Host: [::1]:65535\r\n\r\n"));
    assert(valid("Host: localhost:80\r\nOrigin: http://localhost:3000\r\n\r\n"));
    assert(valid("Host: 127.0.0.1\r\nOrigin: https://127.0.0.1:8443\r\n\r\n"));
    assert(!valid("User-Agent: x\r\n\r\n"));
    assert(!valid("Host: evil.example\r\n\r\n"));
    assert(!valid("Host: localhost.evil.example\r\n\r\n"));
    assert(!valid("Host: localhost@evil.example\r\n\r\n"));
    assert(!valid("Host: localhost:0\r\n\r\n"));
    assert(!valid("Host: localhost:99999\r\n\r\n"));
    assert(!valid("Host: localhost\r\nHost: localhost\r\n\r\n"));
    assert(!valid("Host: localhost\r\nOrigin: null\r\n\r\n"));
    assert(!valid("Host: localhost\r\nOrigin: https://evil.example\r\n\r\n"));
    assert(!valid("Host: localhost\r\nOrigin: http://localhost/path\r\n\r\n"));
    assert(!valid("Host: localhost\r\nOrigin: http://localhost\r\n"
                  "Origin: http://localhost\r\n\r\n"));
    puts("host/origin validation ok");
    return 0;
}
