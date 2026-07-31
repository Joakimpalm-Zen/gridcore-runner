// The Responses-API framing state machine, tested directly (Phase 3 item
// "C-level Responses state-machine test").
//
// tests/conformance/test_responses.py already validates this surface against
// a live server, but only as far as a real generation happens to exercise it.
// The two properties the framer actually promises are structural, and they are
// cheapest to pin here, with no model and no HTTP:
//
//   1. ORDER. An item is announced (`response.output_item.added`) before any
//      of its deltas and closed (`.done`) after them, with a content part
//      opened and closed inside it. Nothing may appear between a part's
//      `.done` and its item's `.done`, and an item can never close twice.
//   2. NAMING. Every event names itself twice — in the SSE `event:` field and
//      in `data.type` — because typed SDK clients dispatch on the first while
//      validating the second. They are written from one argument precisely so
//      they cannot drift; this test fails if that is ever refactored apart.
//
// Plus the invariant the framer alone can guarantee: `sequence_number` is
// monotonic from 0 with no gaps, because nothing else assigns it.
//
// The state machine is `static` inside server.c, so this file includes that
// translation unit and links the rest of the engine around it. Events are
// captured through a socketpair — the real send path, not a stub, so a
// framing bug that only shows up in the bytes still fails here.
#include "runner.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#include "../src/server.c"

static int g_fail = 0;

static void ck(int cond, const char *what) {
    if (!cond) { fprintf(stderr, "FAIL: %s\n", what); g_fail = 1; }
    else        fprintf(stderr, "ok: %s\n", what);
}

// one captured SSE event: the `event:` name and the `data.type` value
typedef struct { char ev[64]; char type[64]; long seq; } event_t;

enum { MAX_EV = 64 };
static event_t g_ev[MAX_EV];
static int g_n_ev;

// Parse the raw SSE stream the framer wrote into the socket. Deliberately
// literal: if the wire shape changes, this notices.
static void parse_events(const char *buf) {
    g_n_ev = 0;
    for (const char *p = buf; (p = strstr(p, "event: ")) && g_n_ev < MAX_EV; ) {
        p += 7;
        const char *nl = strchr(p, '\n');
        if (!nl) break;
        event_t *e = &g_ev[g_n_ev];
        size_t n = (size_t)(nl - p);
        if (n >= sizeof(e->ev)) n = sizeof(e->ev) - 1;
        memcpy(e->ev, p, n);
        e->ev[n] = 0;

        const char *t = strstr(nl, "\"type\":\"");
        e->type[0] = 0;
        if (t) {
            t += 8;
            const char *q = strchr(t, '"');
            if (q) {
                size_t m = (size_t)(q - t);
                if (m >= sizeof(e->type)) m = sizeof(e->type) - 1;
                memcpy(e->type, t, m);
                e->type[m] = 0;
            }
        }
        const char *s = strstr(nl, "\"sequence_number\":");
        e->seq = s ? strtol(s + 18, NULL, 10) : -1;
        g_n_ev++;
        p = nl;
    }
}

static int index_of(const char *type, int from) {
    for (int i = from; i < g_n_ev; i++)
        if (!strcmp(g_ev[i].type, type)) return i;
    return -1;
}

// Drive the framer over a socketpair and capture what it wrote.
static void run_sequence(const char *kind, bool close_it) {
    int sv[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0) {
        ck(0, "socketpair");
        return;
    }
    gen_ctx g;
    memset(&g, 0, sizeof(g));
    g.fd = sv[0];
    g.close_status = "completed";

    if (resp_open_item(&g, kind)) ck(0, "resp_open_item wrote");
    resp_delta(&g, kind, "he", 2);
    resp_delta(&g, kind, "llo", 3);
    if (close_it) resp_close_item(&g);

    shutdown(sv[0], SHUT_WR);
    static char buf[65536];
    size_t got = 0;
    for (;;) {
        ssize_t r = recv(sv[1], buf + got, sizeof(buf) - 1 - got, 0);
        if (r <= 0) break;
        got += (size_t)r;
        if (got >= sizeof(buf) - 1) break;
    }
    buf[got] = 0;
    close(sv[0]);
    close(sv[1]);
    parse_events(buf);
    free(g.item_text.s);
    free(g.out_items.s);
    free(g.out_text.s);
    free(g.call_name);
}

static void test_message_item_order(void) {
    run_sequence("message", true);
    ck(g_n_ev >= 4, "a message item emits at least added/part/delta/done");

    int added = index_of("response.output_item.added", 0);
    int part  = index_of("response.content_part.added", 0);
    int delta = index_of("response.output_text.delta", 0);
    int tdone = index_of("response.output_text.done", 0);
    int pdone = index_of("response.content_part.done", 0);
    int idone = index_of("response.output_item.done", 0);

    ck(added == 0, "the item is announced first");
    ck(part  > added, "the content part opens inside the item");
    ck(delta > part,  "deltas come after the part opens");
    ck(tdone > delta, "the text is closed after its deltas");
    ck(pdone > tdone, "the part closes after the text");
    ck(idone > pdone, "the item closes last");
    ck(index_of("response.output_item.done", idone + 1) < 0,
       "an item never closes twice");
}

static void test_event_names_agree(void) {
    run_sequence("message", true);
    bool agree = g_n_ev > 0;
    for (int i = 0; i < g_n_ev; i++)
        if (strcmp(g_ev[i].ev, g_ev[i].type) != 0) agree = false;
    ck(agree, "every event's SSE name matches its data.type");
}

static void test_sequence_numbers_are_dense(void) {
    run_sequence("message", true);
    bool dense = g_n_ev > 0;
    for (int i = 0; i < g_n_ev; i++)
        if (g_ev[i].seq != i) dense = false;
    ck(dense, "sequence_number runs 0..n-1 with no gaps or repeats");
}

static void test_function_call_item_has_no_content_part(void) {
    // A function_call item streams argument deltas directly: RESP_CALL
    // declares no part_added/part_done, and emitting them would break typed
    // clients that only expect parts on message items.
    run_sequence("function_call", true);
    ck(index_of("response.output_item.added", 0) == 0,
       "the function_call item is announced first");
    ck(index_of("response.function_call_arguments.delta", 0) > 0,
       "argument deltas are emitted");
    ck(index_of("response.content_part.added", 0) < 0,
       "a function_call item opens no content part");
    ck(index_of("response.output_item.done", 0) > 0,
       "the function_call item is closed");
}

static void test_unclosed_item_emits_no_done(void) {
    // Closing is the caller's decision; the framer must not invent it.
    run_sequence("message", false);
    ck(index_of("response.output_item.added", 0) == 0, "item opened");
    ck(index_of("response.output_item.done", 0) < 0,
       "an item left open emits no .done");
}

int main(void) {
    test_message_item_order();
    test_event_names_agree();
    test_sequence_numbers_are_dense();
    test_function_call_item_has_no_content_part();
    test_unclosed_item_emits_no_done();
    if (!g_fail) fprintf(stderr, "all responses state-machine tests passed\n");
    return g_fail;
}
