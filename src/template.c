// Chat template detection and rendering, plus the model-facing chat
// conventions that ride on top of it: thinking-tag splitting and the
// tool-call syntax (declaration rendering and response parsing).
#include "runner.h"
#include "json.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int template_detect(const char *meta_tmpl, tokenizer *tok) {
    if (meta_tmpl) {
        if (strstr(meta_tmpl, "<|im_start|>"))        return TMPL_CHATML;
        if (strstr(meta_tmpl, "<|start_header_id|>")) return TMPL_LLAMA3;
        if (strstr(meta_tmpl, "<|user|>"))            return TMPL_ZEPHYR;
        if (strstr(meta_tmpl, "<|turn>"))             return TMPL_GEMMA4;
        if (strstr(meta_tmpl, "<start_of_turn>"))     return TMPL_GEMMA;
        if (strstr(meta_tmpl, "[INST]"))              return TMPL_LLAMA2;
    }
    if (tok_find(tok, "<|im_start|>") >= 0)        return TMPL_CHATML;
    if (tok_find(tok, "<|start_header_id|>") >= 0) return TMPL_LLAMA3;
    if (tok_find(tok, "<|user|>") >= 0)            return TMPL_ZEPHYR;
    if (tok_find(tok, "<|turn>") >= 0)             return TMPL_GEMMA4;
    if (tok_find(tok, "<start_of_turn>") >= 0)     return TMPL_GEMMA;
    return TMPL_LLAMA2;
}

int template_from_name(const char *name) {
    if (!strcmp(name, "chatml")) return TMPL_CHATML;
    if (!strcmp(name, "llama2")) return TMPL_LLAMA2;
    if (!strcmp(name, "llama3")) return TMPL_LLAMA3;
    if (!strcmp(name, "zephyr")) return TMPL_ZEPHYR;
    if (!strcmp(name, "gemma"))  return TMPL_GEMMA;
    if (!strcmp(name, "gemma4")) return TMPL_GEMMA4;
    if (!strcmp(name, "raw"))    return TMPL_RAW;
    return -1;
}

const char *template_name(int t) {
    switch (t) {
        case TMPL_CHATML: return "chatml";  case TMPL_LLAMA2: return "llama2";
        case TMPL_LLAMA3: return "llama3";  case TMPL_ZEPHYR: return "zephyr";
        case TMPL_GEMMA:  return "gemma";
        case TMPL_GEMMA4: return "gemma4";
        default: return "raw";
    }
}

static size_t emit(char *out, size_t cap, size_t off, const char *fmt,
                   const char *a, const char *b) {
    if (off >= cap) return off;
    int n = snprintf(out + off, cap - off, fmt, a, b);
    return n > 0 ? off + (size_t)n : off;
}

size_t render_messages(int tmpl, const chat_msg *msgs, int n_msgs,
                       bool add_assistant, char *out, size_t cap) {
    size_t off = 0;
    out[0] = 0;
    switch (tmpl) {
    case TMPL_CHATML:
        for (int i = 0; i < n_msgs; i++)
            off = emit(out, cap, off, "<|im_start|>%s\n%s<|im_end|>\n",
                       msgs[i].role, msgs[i].content);
        if (add_assistant)
            off = emit(out, cap, off, "<|im_start|>assistant\n", NULL, NULL);
        break;
    case TMPL_LLAMA3:
        for (int i = 0; i < n_msgs; i++)
            off = emit(out, cap, off,
                       "<|start_header_id|>%s<|end_header_id|>\n\n%s<|eot_id|>",
                       msgs[i].role, msgs[i].content);
        if (add_assistant)
            off = emit(out, cap, off,
                       "<|start_header_id|>assistant<|end_header_id|>\n\n", NULL, NULL);
        break;
    case TMPL_ZEPHYR:
        for (int i = 0; i < n_msgs; i++)
            off = emit(out, cap, off, "<|%s|>\n%s</s>\n",
                       msgs[i].role, msgs[i].content);
        if (add_assistant)
            off = emit(out, cap, off, "<|assistant|>\n", NULL, NULL);
        break;
    case TMPL_GEMMA4:
        // gemma4 (reference: llama.cpp models/templates/google-gemma-4-31B-it
        // .jinja): <|turn>role\n CONTENT <turn|>\n per turn, a native system
        // role (unlike gemma1-3), assistant role named "model"
        for (int i = 0; i < n_msgs; i++) {
            const char *role = !strcmp(msgs[i].role, "assistant") ? "model"
                                                                  : msgs[i].role;
            off = emit(out, cap, off, "<|turn>%s\n", role, NULL);
            off = emit(out, cap, off, "%s<turn|>\n", msgs[i].content, NULL);
        }
        if (add_assistant)
            off = emit(out, cap, off, "<|turn>model\n", NULL, NULL);
        break;
    case TMPL_GEMMA: {
        // gemma has no system role: fold a system message into the first
        // user turn; the assistant role is named "model"
        const char *gsys = NULL;
        for (int i = 0; i < n_msgs; i++) {
            const chat_msg *m = &msgs[i];
            if (!strcmp(m->role, "system")) { gsys = m->content; continue; }
            const char *role = !strcmp(m->role, "assistant") ? "model" : m->role;
            if (gsys && !strcmp(m->role, "user")) {
                off = emit(out, cap, off, "<start_of_turn>%s\n%s\n\n", role, gsys);
                off = emit(out, cap, off, "%s<end_of_turn>\n", m->content, NULL);
                gsys = NULL;
            } else {
                off = emit(out, cap, off, "<start_of_turn>%s\n", role, NULL);
                off = emit(out, cap, off, "%s<end_of_turn>\n", m->content, NULL);
            }
        }
        if (add_assistant)
            off = emit(out, cap, off, "<start_of_turn>model\n", NULL, NULL);
        break;
    }
    case TMPL_LLAMA2: {
        // fold an initial system message into the first user turn
        const char *sys = NULL;
        for (int i = 0; i < n_msgs; i++) {
            const chat_msg *m = &msgs[i];
            if (!strcmp(m->role, "system")) { sys = m->content; continue; }
            if (!strcmp(m->role, "user")) {
                if (sys) {
                    off = emit(out, cap, off, "[INST] <<SYS>>\n%s\n<</SYS>>\n\n", sys, NULL);
                    off = emit(out, cap, off, "%s [/INST]", m->content, NULL);
                    sys = NULL;
                } else {
                    off = emit(out, cap, off, "[INST] %s [/INST]", m->content, NULL);
                }
            } else { // assistant
                off = emit(out, cap, off, " %s </s>", m->content, NULL);
            }
        }
        break;
    }
    default: // TMPL_RAW: concatenate contents
        for (int i = 0; i < n_msgs; i++)
            off = emit(out, cap, off, "%s", msgs[i].content, NULL);
        break;
    }
    return off;
}

// ------------------------------------------------- thinking-tag splitter

// gemma4 interleaves channel blocks with plain text anywhere in a response
// (thought . answer . thought . ...), so the splitter scans the whole stream:
// bytes between open and close tags are reasoning, the rest is content. The
// last strlen(tag)-1 bytes are held back while scanning so a tag split across
// chunk boundaries still resolves.

enum { TS_CONTENT, TS_WS, TS_THINK };

void think_init(think_split *t, const char *open, const char *close) {
    memset(t, 0, sizeof(*t));
    t->open  = open;
    t->close = close;
}

void think_free(think_split *t) {
    free(t->buf);
    t->buf = NULL;
    t->n = t->cap = 0;
}

static void ts_keep(think_split *t, const char *b, int n) {
    // b always points into t->buf and n <= what feed() already grew cap to,
    // so no realloc may happen here — it would free the memory b points into
    memmove(t->buf, b, n);
    t->n = n;
}

static const char *ts_find(const char *p, int len, const char *tag, int tl) {
    for (int i = 0; i + tl <= len; i++)
        if (memcmp(p + i, tag, tl) == 0) return p + i;
    return NULL;
}

int think_feed(think_split *t, const char *bytes, int n, think_cb cb, void *ud) {
    if (!t->open) return n > 0 ? cb(ud, 0, bytes, n) : 0;

    if (t->n + n > t->cap) {
        t->cap = t->n + n + 64;
        t->buf = realloc(t->buf, t->cap);
    }
    memcpy(t->buf + t->n, bytes, n);
    t->n += n;
    const char *p = t->buf;
    int len = t->n;
    int ol = (int)strlen(t->open), cl = (int)strlen(t->close);

    for (;;) {
        if (t->state == TS_WS) { // drop whitespace after an open tag
            while (len > 0 && (*p == ' ' || *p == '\n' || *p == '\t' || *p == '\r')) {
                p++; len--;
            }
            if (len == 0) { t->n = 0; return 0; }
            t->state = TS_THINK;
        }
        bool     think = t->state == TS_THINK;
        const char *tag = think ? t->close : t->open;
        int          tl = think ? cl : ol;
        const char   *q = ts_find(p, len, tag, tl);
        if (!q) {
            int hold = tl - 1 < len ? tl - 1 : len;   // possible partial tag
            int rc = len - hold > 0 ? cb(ud, think, p, len - hold) : 0;
            ts_keep(t, p + (len - hold), hold);
            return rc;
        }
        int rc = q > p ? cb(ud, think, p, (int)(q - p)) : 0;
        if (rc) { ts_keep(t, q + tl, len - (int)(q - p) - tl); return rc; }
        len -= (int)(q - p) + tl;
        p = q + tl;
        t->state = think ? TS_CONTENT : TS_WS;
    }
}

int think_finish(think_split *t, think_cb cb, void *ud) {
    if (!t->open || t->n == 0) { t->n = 0; return 0; }
    // a held partial tag never completed: it is literal text of whichever
    // section we were in (whitespace after an open tag is dropped)
    int rc = t->state == TS_WS ? 0
           : cb(ud, t->state == TS_THINK, t->buf, t->n);
    t->n = 0;
    return rc;
}

// ------------------------------------------------- tool-call convention

// parse gemma4 tool-call blocks — <|tool_call>call:NAME{json}<tool_call|> —
// out of the content, appending OpenAI tool_calls items to tc. Returns the
// number of calls; content is compacted in place.
// OpenAI "tools" declarations rendered as a system turn, teaching the model
// the call syntax tool_calls_parse reads back. The trained template's native
// declaration macros are more elaborate; this works in practice.
void tools_render(const jv *tools, sbuf *out) {
    if (!tools || tools->type != J_ARR || tools->n == 0) return;
    sb_lit(out, "You have these tools available. To call one, reply with "
                "exactly <|tool_call>call:NAME{json arguments}<tool_call|> "
                "and nothing else. Tools:\n");
    jv_dump(tools, out);
}

int tool_calls_parse(sbuf *content, sbuf *tc) {
    if (!content->s) return 0;
    static const char OPEN[] = "<|tool_call>call:";
    static const char CLOSE[] = "<tool_call|>";
    int n_calls = 0;
    char *w = content->s; // write cursor for the compacted content
    const char *p = content->s, *end = content->s + content->n;
    while (p < end) {
        const char *o = strstr(p, OPEN);
        if (!o) {
            memmove(w, p, end - p);
            w += end - p;
            break;
        }
        memmove(w, p, o - p);
        w += o - p;
        const char *name = o + sizeof(OPEN) - 1;
        const char *brace = name;
        while (brace < end && *brace != '{' && *brace != '<' && brace - name < 128)
            brace++;
        if (brace >= end || *brace != '{') { p = name; continue; } // not a call
        // brace-match the args object (string- and escape-aware)
        const char *q = brace;
        int depth = 0;
        bool in_str = false;
        for (; q < end; q++) {
            if (in_str) {
                if (*q == '\\') q++;
                else if (*q == '"') in_str = false;
            } else if (*q == '"') in_str = true;
            else if (*q == '{') depth++;
            else if (*q == '}' && --depth == 0) { q++; break; }
        }
        if (depth != 0) { p = name; continue; } // truncated: leave as content
        sb_fmt(tc, "%s{\"id\":\"call_%d\",\"type\":\"function\",\"function\":"
                   "{\"name\":\"", n_calls ? "," : "", n_calls);
        sb_esc(tc, name, (int)(brace - name));
        sb_lit(tc, "\",\"arguments\":\"");
        sb_esc(tc, brace, (int)(q - brace));
        sb_lit(tc, "\"}}");
        n_calls++;
        p = q;
        if (end - p >= (int)sizeof(CLOSE) - 1 &&
            memcmp(p, CLOSE, sizeof(CLOSE) - 1) == 0)
            p += sizeof(CLOSE) - 1;
    }
    if (n_calls) content->n = (int)(w - content->s);
    return n_calls;
}

