// HTTP transport and framing. Lifted out of server.c (RNR-019); see http.h.
#include "http.h"
#include "compat.h"

#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
void sock_init(void) {
    WSADATA w;
    WSAStartup(MAKEWORD(2, 2), &w);
}
int  sock_recv(sock_t fd, char *buf, size_t n) { return recv(fd, buf, (int)n, 0); }
int  sock_send(sock_t fd, const char *buf, size_t n) { return send(fd, buf, (int)n, 0); }
int  sock_peek(sock_t fd, char *buf, size_t n) { return recv(fd, buf, (int)n, MSG_PEEK); }
void sock_close(sock_t fd) { closesocket(fd); }
// Winsock reports through WSAGetLastError, NOT errno — strerror(errno) here
// prints "Success" (or a stale unrelated error) for a genuine bind failure,
// which is worse than no reason at all. FormatMessage gives the real text.
const char *sock_errstr(void) {
    static char buf[256];
    DWORD e = (DWORD)WSAGetLastError();
    DWORD n = FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM |
                             FORMAT_MESSAGE_IGNORE_INSERTS, NULL, e,
                             MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                             buf, sizeof(buf) - 1, NULL);
    while (n > 0 && (buf[n - 1] == '\n' || buf[n - 1] == '\r')) buf[--n] = 0;
    if (n == 0) snprintf(buf, sizeof(buf), "winsock error %lu", (unsigned long)e);
    return buf;
}
void sock_recv_timeout(sock_t fd, double s) {
    DWORD ms = (DWORD)(s * 1000.0);
    if (ms == 0) ms = 1; // 0 would mean "block forever" on winsock
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, (const char *)&ms, sizeof(ms));
}
void sock_send_timeout(sock_t fd, double s) {
    DWORD ms = (DWORD)(s * 1000.0);
    if (ms == 0) ms = 1;
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, (const char *)&ms, sizeof(ms));
}
#else
void sock_init(void) { signal(SIGPIPE, SIG_IGN); }
int  sock_recv(sock_t fd, char *buf, size_t n) { return (int)read(fd, buf, n); }
int  sock_send(sock_t fd, const char *buf, size_t n) { return (int)write(fd, buf, n); }
int  sock_peek(sock_t fd, char *buf, size_t n) { return (int)recv(fd, buf, n, MSG_PEEK); }
void sock_close(sock_t fd) { close(fd); }
const char *sock_errstr(void) { return strerror(errno); }
void sock_recv_timeout(sock_t fd, double s) {
    struct timeval tv;
    tv.tv_sec = (time_t)s;
    tv.tv_usec = (suseconds_t)((s - (double)tv.tv_sec) * 1e6);
    if (tv.tv_sec == 0 && tv.tv_usec == 0) tv.tv_usec = 1000;
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
}
void sock_send_timeout(sock_t fd, double s) {
    struct timeval tv;
    tv.tv_sec = (time_t)s;
    tv.tv_usec = (suseconds_t)((s - (double)tv.tv_sec) * 1e6);
    if (tv.tv_sec == 0 && tv.tv_usec == 0) tv.tv_usec = 1000;
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
}
#endif

// case-insensitive prefix compare (strncasecmp is not universal)
static int ci_ncmp(const char *a, const char *b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        int ca = a[i] >= 'A' && a[i] <= 'Z' ? a[i] + 32 : (unsigned char)a[i];
        int cb = b[i] >= 'A' && b[i] <= 'Z' ? b[i] + 32 : (unsigned char)b[i];
        if (ca != cb) return ca - cb;
        if (ca == 0) return 0;
    }
    return 0;
}

// Parse the two HTTP fields that determine request framing. Header names are
// compared as complete, line-anchored fields: text inside another field's
// value must never influence framing. Runner does not implement transfer
// codings, and multiple Content-Length fields are rejected even when equal so
// every accepted request has one unambiguous interpretation.
bool parse_request_framing(char *first_header, char *header_end,
                                  size_t *content_length) {
    bool saw_length = false;
    *content_length = 0;
    for (char *line = first_header; line < header_end;) {
        char *end = strstr(line, "\r\n");
        if (!end || end > header_end) return false;
        char *colon = memchr(line, ':', (size_t)(end - line));
        if (colon) {
            size_t name_n = (size_t)(colon - line);
            bool is_length = name_n == 14 &&
                             ci_ncmp(line, "Content-Length", 14) == 0;
            bool is_transfer = name_n == 17 &&
                               ci_ncmp(line, "Transfer-Encoding", 17) == 0;
            if (is_transfer) return false;
            if (is_length) {
                if (saw_length) return false;
                saw_length = true;
                char *p = colon + 1;
                while (p < end && (*p == ' ' || *p == '\t')) p++;
                char *digits = p;
                size_t value = 0;
                while (p < end && *p >= '0' && *p <= '9') {
                    size_t digit = (size_t)(*p - '0');
                    if (value > ((size_t)-1 - digit) / 10) return false;
                    value = value * 10 + digit;
                    p++;
                }
                if (p == digits) return false;
                while (p < end && (*p == ' ' || *p == '\t')) p++;
                if (p != end) return false;
                *content_length = value;
            }
        }
        line = end + 2;
    }
    return true;
}

bool parse_request_line(char *hdr, char method[8], char path[256],
                               char **first_header) {
    char *end = strstr(hdr, "\r\n");
    if (!end) return false;
    *end = 0;
    char *sp1 = strchr(hdr, ' ');
    char *sp2 = sp1 ? strchr(sp1 + 1, ' ') : NULL;
    bool shape = sp1 && sp1 != hdr && sp2 && sp2 != sp1 + 1 &&
                 sp2[1] != 0 && strchr(sp2 + 1, ' ') == NULL;
    size_t method_n = shape ? (size_t)(sp1 - hdr) : 0;
    size_t path_n = shape ? (size_t)(sp2 - sp1 - 1) : 0;
    bool valid = shape && method_n < 8 && path_n < 256 &&
                 (!strcmp(sp2 + 1, "HTTP/1.1") ||
                  !strcmp(sp2 + 1, "HTTP/1.0"));
    if (valid) {
        memcpy(method, hdr, method_n);
        method[method_n] = 0;
        memcpy(path, sp1 + 1, path_n);
        path[path_n] = 0;
    }
    *end = '\r';
    *first_header = end + 2;
    return valid;
}

// ---------------------------------------------------------------- helpers

bool send_all(sock_t fd, const char *s, size_t n) {
    while (n > 0) {
        int w = sock_send(fd, s, n);
        if (w <= 0) return false;
        s += w;
        n -= (size_t)w;
    }
    return true;
}

const char *reason_phrase(int code) {
    switch (code) {
        case 200: return "OK";
        case 400: return "Bad Request";
        case 404: return "Not Found";
        case 408: return "Request Timeout";
        case 500: return "Internal Server Error";
        case 503: return "Service Unavailable";
        default:  return "Internal Server Error";
    }
}

void send_response(sock_t fd, int code, const char *ctype, const char *body,
                          size_t blen) {
    char hdr[256];
    const char *msg = reason_phrase(code);
    int hn = snprintf(hdr, sizeof(hdr),
                      "HTTP/1.1 %d %s\r\nContent-Type: %s\r\n"
                      "Content-Length: %zu\r\nConnection: close\r\n\r\n",
                      code, msg, ctype, blen);
    if (send_all(fd, hdr, hn)) send_all(fd, body, blen);
}

// Send a built JSON body, or 500 if the builder ran out of memory. A short
// body that still parses as success is worse than an error: the client cannot
// tell anything went wrong.
void send_error(sock_t fd, int code, const char *message);
void send_error_detail(sock_t fd, int code, const char *message,
                              const char *param, const char *error_code);

void send_built(sock_t fd, sbuf *b) {
    if (b->failed) send_error(fd, 500, "out of memory building response");
    else send_response(fd, 200, "application/json", b->s, b->n);
}

void send_error(sock_t fd, int code, const char *message) {
    send_error_detail(fd, code, message, NULL, NULL);
}

void send_error_detail(sock_t fd, int code, const char *message,
                              const char *param, const char *error_code) {
    char body[768], esc[384], ep[128], ec[128];
    json_escape(message, strlen(message), esc, sizeof(esc));
    if (param) json_escape(param, strlen(param), ep, sizeof(ep));
    if (error_code) json_escape(error_code, strlen(error_code), ec, sizeof(ec));
    int n = snprintf(body, sizeof(body),
                     "{\"error\":{\"message\":\"%s\",\"type\":\"%s\","
                     "\"param\":%s%s%s,\"code\":%s%s%s}}",
                     esc, code >= 500 ? "server_error" : "invalid_request_error",
                     param ? "\"" : "", param ? ep : "null", param ? "\"" : "",
                     error_code ? "\"" : "", error_code ? ec : "null",
                     error_code ? "\"" : "");
    send_response(fd, code, "application/json", body, n);
}
