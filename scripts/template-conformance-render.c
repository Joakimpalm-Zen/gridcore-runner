// Runner-side renderer for scripts/template-conformance.py.
//
// Reads a JSON job list on stdin and writes each case's NATIVE render to
// stdout, so the python harness can diff runner's C renderer byte-for-byte
// against the model's own jinja chat template.
//
//   stdin : {"cases":[{"id":"...", "template":"chatml",
//                      "add_generation_prompt":true, "thinking":"default",
//                      "messages":[{"role":"user","content":"hi",
//                                   "name":null,"channel":null}],
//                      "tools":[...]}, ...]}
//   stdout: {"results":[{"id":"...","prompt":"..."} ...]}
//
// Nothing here interprets the request the way server.c does (no tool_call
// history expansion, no tool-role rewriting): this calls
// render_messages_with_tools directly, which is exactly the surface the
// conformance question is about. The python side is responsible for handing
// it the same conversation it hands jinja.
#include "template.h"
#include "json.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int think_of(const char *s) {
    if (!s) return THINK_DEFAULT;
    if (!strcmp(s, "on"))  return THINK_ON;
    if (!strcmp(s, "off")) return THINK_OFF;
    return THINK_DEFAULT;
}

static char *read_all(FILE *f, size_t *out_n) {
    size_t cap = 1 << 16, n = 0;
    char *b = malloc(cap);
    if (!b) return NULL;
    for (;;) {
        if (n == cap) {
            char *g = realloc(b, cap * 2);
            if (!g) { free(b); return NULL; }
            b = g; cap *= 2;
        }
        size_t got = fread(b + n, 1, cap - n, f);
        n += got;
        if (got == 0) break;
    }
    b[n < cap ? n : cap - 1] = 0;
    *out_n = n;
    return b;
}

int main(void) {
    size_t n_in = 0;
    char *in = read_all(stdin, &n_in);
    if (!in) { fprintf(stderr, "oom\n"); return 2; }
    jv *req = json_parse(in, n_in);
    if (!req) { fprintf(stderr, "input is not JSON\n"); return 2; }
    jv *cases = jv_get(req, "cases");
    if (!cases || cases->type != J_ARR) {
        fprintf(stderr, "missing cases[]\n");
        return 2;
    }
    sbuf out = {0};
    sb_lit(&out, "{\"results\":[");
    for (int i = 0; i < cases->n; i++) {
        jv *c = cases->items[i];
        const char *id = jv_str(jv_get(c, "id"), "");
        const char *tname = jv_str(jv_get(c, "template"), "raw");
        int tmpl = template_from_name(tname);
        if (tmpl < 0) { fprintf(stderr, "unknown template %s\n", tname); return 2; }
        bool addgen = jv_bool(jv_get(c, "add_generation_prompt"), true);
        int think = think_of(jv_str(jv_get(c, "thinking"), NULL));
        jv *msgs = jv_get(c, "messages");
        int n_msgs = msgs && msgs->type == J_ARR ? msgs->n : 0;
        chat_msg *cm = calloc((size_t)(n_msgs > 0 ? n_msgs : 1), sizeof *cm);
        if (!cm) { fprintf(stderr, "oom\n"); return 2; }
        for (int k = 0; k < n_msgs; k++) {
            jv *m = msgs->items[k];
            cm[k].role    = jv_str(jv_get(m, "role"), "user");
            cm[k].content = jv_str(jv_get(m, "content"), "");
            cm[k].name    = jv_str(jv_get(m, "name"), NULL);
            cm[k].channel = jv_str(jv_get(m, "channel"), NULL);
        }
        jv *tools = jv_get(c, "tools");
        if (tools && tools->type != J_ARR) tools = NULL;
        // "server_tools_turn": mirror src/server.c's non-strict tools path,
        // which is where ORNITH's declarations actually come from --
        // tools_render_for() builds a leading system turn, and an existing
        // system message is folded into it. Calling the real function keeps
        // that text out of this harness (and out of the python side).
        sbuf ts = {0};
        chat_msg *cm2 = NULL;
        if (jv_bool(jv_get(c, "server_tools_turn"), false) && tools) {
            tools_render_for(tmpl, tools, &ts);
            if (ts.n) {
                int drop = (tmpl == TMPL_ORNITH && n_msgs > 0 &&
                            !strcmp(cm[0].role, "system")) ? 1 : 0;
                if (drop) {
                    sb_lit(&ts, "\n\n");
                    sb_put(&ts, cm[0].content, strlen(cm[0].content));
                }
                cm2 = calloc((size_t)(n_msgs - drop + 1), sizeof *cm2);
                if (!cm2) { fprintf(stderr, "oom\n"); return 2; }
                cm2[0] = (chat_msg){ .role = "system", .content = ts.s };
                for (int k = drop; k < n_msgs; k++) cm2[k - drop + 1] = cm[k];
                free(cm);
                cm = cm2;
                n_msgs = n_msgs - drop + 1;
            }
        }
        // grow until the render fits, the same two-signal rule server.c uses
        size_t cap = 4096;
        char *buf = NULL;
        for (;;) {
            free(buf);
            buf = malloc(cap);
            if (!buf) { fprintf(stderr, "oom\n"); return 2; }
            size_t w = render_messages_with_tools(tmpl, cm, n_msgs, addgen,
                                                  think, tools, buf, cap);
            if (w < cap && strlen(buf) + 1 < cap) break;
            cap *= 2;
        }
        if (i) sb_lit(&out, ",");
        sb_lit(&out, "{\"id\":\"");
        sb_esc(&out, id, strlen(id));
        sb_lit(&out, "\",\"prompt\":\"");
        sb_esc(&out, buf, strlen(buf));
        sb_lit(&out, "\"}");
        free(buf);
        free(cm);
        free(ts.s);
    }
    sb_lit(&out, "]}");
    if (out.failed) { fprintf(stderr, "oom building results\n"); return 2; }
    fwrite(out.s, 1, out.n, stdout);
    free(out.s);
    free(in);
    jv_free(req);
    return 0;
}
