// HTTP transport and framing: sockets, request parsing, response writing.
//
// Everything here was the first 250 lines of server.c. It is the layer that
// knows about file descriptors and CRLF and nothing about inference: what a
// socket is on this platform, how to read one HTTP request off it safely, and
// how to write a response or an OpenAI-shaped error back.
//
// The platform typedefs live in this header rather than in http.c because
// server.c still calls socket/bind/listen/accept/select directly for the
// listener; centralising the winsock-vs-BSD divergence in one place is most of
// what this split is worth.
#ifndef RUNNER_HTTP_H
#define RUNNER_HTTP_H

#include <stddef.h>
#include <stdbool.h>

#include "json.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <ws2tcpip.h>
typedef SOCKET sock_t;
#define SOCK_INVALID INVALID_SOCKET
#else
#include <signal.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/select.h>
#include <netinet/in.h>
#include <sys/socket.h>
typedef int sock_t;
#define SOCK_INVALID (-1)
#endif

void sock_init(void);
int  sock_recv(sock_t fd, char *buf, size_t n);
int  sock_send(sock_t fd, const char *buf, size_t n);
int  sock_peek(sock_t fd, char *buf, size_t n);
void sock_close(sock_t fd);
// The reason a socket call failed, from the platform's own error channel.
const char *sock_errstr(void);
void sock_recv_timeout(sock_t fd, double s);
void sock_send_timeout(sock_t fd, double s);

// Parse the two HTTP fields that decide request framing; false rejects the
// request outright. See http.c for why a duplicate Content-Length is refused.
bool parse_request_framing(char *first_header, char *header_end,
                           size_t *content_length);
// Split "METHOD path HTTP/1.x" and point first_header at the line after it.
bool parse_request_line(char *hdr, char method[8], char path[256],
                        char **first_header);
bool validate_request_authority(char *first_header, char *header_end);

bool send_all(sock_t fd, const char *s, size_t n);
const char *reason_phrase(int code);
void send_response(sock_t fd, int code, const char *ctype, const char *body,
                   size_t blen);
// OpenAI-shaped error bodies. send_built sends a built JSON body, or 500 when
// the builder ran out of memory: a short body that still parses as success is
// worse than an error, because the client cannot tell anything went wrong.
void send_error(sock_t fd, int code, const char *message);
void send_error_detail(sock_t fd, int code, const char *message,
                       const char *param, const char *error_code);
void send_built(sock_t fd, sbuf *b);

#endif // RUNNER_HTTP_H
