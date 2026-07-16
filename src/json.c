// Minimal recursive-descent JSON parser, string escaper, string builder,
// and value re-serializer.
#include "json.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>

typedef struct { const char *p, *end; int depth; } jcur;

static void skip_ws(jcur *c) {
    while (c->p < c->end && (*c->p == ' ' || *c->p == '\t' ||
                             *c->p == '\n' || *c->p == '\r')) c->p++;
}

static jv *jv_new(jtype t) {
    jv *v = calloc(1, sizeof(jv));
    v->type = t;
    return v;
}

void jv_free(jv *v) {
    if (!v) return;
    free(v->str);
    for (int i = 0; i < v->n; i++) {
        jv_free(v->items[i]);
        if (v->keys) free(v->keys[i]);
    }
    free(v->items);
    free(v->keys);
    free(v);
}

static int u8_emit(unsigned cp, char *out) {
    if (cp < 0x80) { out[0] = (char)cp; return 1; }
    if (cp < 0x800) {
        out[0] = (char)(0xC0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    }
    if (cp < 0x10000) {
        out[0] = (char)(0xE0 | (cp >> 12));
        out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[2] = (char)(0x80 | (cp & 0x3F));
        return 3;
    }
    out[0] = (char)(0xF0 | (cp >> 18));
    out[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
    out[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
    out[3] = (char)(0x80 | (cp & 0x3F));
    return 4;
}

static int hex4(const char *p) {
    int v = 0;
    for (int i = 0; i < 4; i++) {
        char c = p[i];
        v <<= 4;
        if (c >= '0' && c <= '9') v |= c - '0';
        else if (c >= 'a' && c <= 'f') v |= c - 'a' + 10;
        else if (c >= 'A' && c <= 'F') v |= c - 'A' + 10;
        else return -1;
    }
    return v;
}

// parse a string (cursor at opening quote); returns malloc'd decoded string
static char *parse_string(jcur *c) {
    if (c->p >= c->end || *c->p != '"') return NULL;
    c->p++;
    size_t cap = 32, m = 0;
    char *out = malloc(cap);
    while (c->p < c->end) {
        if (m + 8 > cap) { cap *= 2; out = realloc(out, cap); }
        unsigned char ch = (unsigned char)*c->p;
        if (ch == '"') { c->p++; out[m] = 0; return out; }
        if (ch == '\\') {
            c->p++;
            if (c->p >= c->end) break;
            char e = *c->p++;
            switch (e) {
                case '"': out[m++] = '"'; break;
                case '\\': out[m++] = '\\'; break;
                case '/': out[m++] = '/'; break;
                case 'b': out[m++] = '\b'; break;
                case 'f': out[m++] = '\f'; break;
                case 'n': out[m++] = '\n'; break;
                case 'r': out[m++] = '\r'; break;
                case 't': out[m++] = '\t'; break;
                case 'u': {
                    if (c->p + 4 > c->end) goto fail;
                    int cp = hex4(c->p);
                    if (cp < 0) goto fail;
                    c->p += 4;
                    if (cp >= 0xD800 && cp <= 0xDBFF && c->p + 6 <= c->end &&
                        c->p[0] == '\\' && c->p[1] == 'u') {
                        int lo = hex4(c->p + 2);
                        if (lo >= 0xDC00 && lo <= 0xDFFF) {
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                            c->p += 6;
                        }
                    }
                    m += u8_emit((unsigned)cp, out + m);
                    break;
                }
                default: goto fail;
            }
        } else if (ch < 0x20) {
            goto fail;
        } else {
            out[m++] = (char)ch;
            c->p++;
        }
    }
fail:
    free(out);
    return NULL;
}

static jv *parse_value(jcur *c);

static jv *parse_number(jcur *c) {
    const char *start = c->p;
    const char *p = start;
    if (p < c->end && *p == '-') p++;
    if (p >= c->end) return NULL;
    if (*p == '0') {
        p++;
        if (p < c->end && *p >= '0' && *p <= '9') return NULL;
    } else if (*p >= '1' && *p <= '9') {
        do { p++; } while (p < c->end && *p >= '0' && *p <= '9');
    } else {
        return NULL;
    }
    if (p < c->end && *p == '.') {
        p++;
        if (p >= c->end || *p < '0' || *p > '9') return NULL;
        do { p++; } while (p < c->end && *p >= '0' && *p <= '9');
    }
    if (p < c->end && (*p == 'e' || *p == 'E')) {
        p++;
        if (p < c->end && (*p == '+' || *p == '-')) p++;
        if (p >= c->end || *p < '0' || *p > '9') return NULL;
        do { p++; } while (p < c->end && *p >= '0' && *p <= '9');
    }

    size_t n = (size_t)(p - start);
    char local[128];
    char *tmp = n < sizeof(local) ? local : malloc(n + 1);
    if (!tmp) return NULL;
    memcpy(tmp, start, n);
    tmp[n] = 0;
    char *endp = NULL;
    double d = strtod(tmp, &endp);
    bool ok = endp == tmp + n && isfinite(d);
    if (tmp != local) free(tmp);
    if (!ok) return NULL;

    jv *r = jv_new(J_NUM);
    if (!r) return NULL;
    r->num = d;
    c->p = p;
    return r;
}

static jv *parse_container(jcur *c, char open) {
    char close = open == '{' ? '}' : ']';
    jv *v = jv_new(open == '{' ? J_OBJ : J_ARR);
    c->p++; // consume open
    skip_ws(c);
    if (c->p < c->end && *c->p == close) { c->p++; return v; }
    for (;;) {
        char *key = NULL;
        if (v->type == J_OBJ) {
            skip_ws(c);
            key = parse_string(c);
            if (!key) goto fail;
            skip_ws(c);
            if (c->p >= c->end || *c->p != ':') { free(key); goto fail; }
            c->p++;
        }
        jv *item = parse_value(c);
        if (!item) { free(key); goto fail; }
        v->items = realloc(v->items, sizeof(jv *) * (v->n + 1));
        if (v->type == J_OBJ)
            v->keys = realloc(v->keys, sizeof(char *) * (v->n + 1));
        v->items[v->n] = item;
        if (v->type == J_OBJ) v->keys[v->n] = key;
        v->n++;
        skip_ws(c);
        if (c->p < c->end && *c->p == ',') { c->p++; continue; }
        if (c->p < c->end && *c->p == close) { c->p++; return v; }
        goto fail;
    }
fail:
    jv_free(v);
    return NULL;
}

static jv *parse_value(jcur *c) {
    if (++c->depth > 128) { c->depth--; return NULL; }
    skip_ws(c);
    jv *r = NULL;
    if (c->p < c->end) {
        char ch = *c->p;
        if (ch == '{' || ch == '[') {
            r = parse_container(c, ch);
        } else if (ch == '"') {
            char *s = parse_string(c);
            if (s) { r = jv_new(J_STR); r->str = s; }
        } else if (ch == 't' && c->end - c->p >= 4 && !memcmp(c->p, "true", 4)) {
            r = jv_new(J_BOOL); r->b = true; c->p += 4;
        } else if (ch == 'f' && c->end - c->p >= 5 && !memcmp(c->p, "false", 5)) {
            r = jv_new(J_BOOL); r->b = false; c->p += 5;
        } else if (ch == 'n' && c->end - c->p >= 4 && !memcmp(c->p, "null", 4)) {
            r = jv_new(J_NULL); c->p += 4;
        } else if (ch == '-' || (ch >= '0' && ch <= '9')) {
            r = parse_number(c);
        }
    }
    c->depth--;
    return r;
}

jv *json_parse(const char *s, size_t n) {
    jcur c = { s, s + n, 0 };
    jv *v = parse_value(&c);
    if (!v) return NULL;
    skip_ws(&c);
    if (c.p != c.end) { jv_free(v); return NULL; } // trailing garbage
    return v;
}

jv *jv_get(jv *obj, const char *key) {
    if (!obj || obj->type != J_OBJ) return NULL;
    for (int i = 0; i < obj->n; i++)
        if (strcmp(obj->keys[i], key) == 0) return obj->items[i];
    return NULL;
}

const char *jv_str(jv *v, const char *dflt) {
    return (v && v->type == J_STR) ? v->str : dflt;
}
double jv_num(jv *v, double dflt) {
    return (v && v->type == J_NUM) ? v->num : dflt;
}
bool jv_bool(jv *v, bool dflt) {
    return (v && v->type == J_BOOL) ? v->b : dflt;
}

// length of the valid UTF-8 sequence starting at s[i] (bounded by n), or 0
// when the bytes there are not well-formed UTF-8
static size_t utf8_seq(const char *s, size_t i, size_t n) {
    unsigned char c = (unsigned char)s[i];
    size_t len = c < 0x80 ? 1 : (c & 0xE0) == 0xC0 ? 2 :
                 (c & 0xF0) == 0xE0 ? 3 : (c & 0xF8) == 0xF0 ? 4 : 0;
    if (len == 0 || i + len > n) return 0;
    for (size_t k = 1; k < len; k++)
        if (((unsigned char)s[i + k] & 0xC0) != 0x80) return 0;
    return len;
}

size_t json_escape(const char *s, size_t n, char *out, size_t cap) {
    size_t m = 0;
    for (size_t i = 0; i < n && m + 8 < cap; ) {
        unsigned char c = (unsigned char)s[i];
        switch (c) {
            case '"':  out[m++] = '\\'; out[m++] = '"';  i++; continue;
            case '\\': out[m++] = '\\'; out[m++] = '\\'; i++; continue;
            case '\n': out[m++] = '\\'; out[m++] = 'n';  i++; continue;
            case '\r': out[m++] = '\\'; out[m++] = 'r';  i++; continue;
            case '\t': out[m++] = '\\'; out[m++] = 't';  i++; continue;
            case '\b': out[m++] = '\\'; out[m++] = 'b';  i++; continue;
            case '\f': out[m++] = '\\'; out[m++] = 'f';  i++; continue;
        }
        if (c < 0x20) {
            m += snprintf(out + m, cap - m, "\\u%04x", c);
            i++;
            continue;
        }
        if (c < 0x80) {
            out[m++] = (char)c;
            i++;
            continue;
        }
        // multi-byte: pass through only well-formed UTF-8 — a model's raw
        // byte-fallback tokens can emit stray 0x80..0xFF bytes, and one of
        // those in a response body breaks every strict JSON client
        size_t len = utf8_seq(s, i, n);
        if (len == 0) {
            out[m++] = '\xEF'; out[m++] = '\xBF'; out[m++] = '\xBD'; // U+FFFD
            i++;
            continue;
        }
        for (size_t k = 0; k < len && m < cap - 1; k++) out[m++] = s[i + k];
        i += len;
    }
    out[m] = 0;
    return m;
}

// -------------------------------------------------------- string builder

void sb_put(sbuf *b, const char *s, size_t n) {
    if (b->n + n + 1 > b->cap) {
        b->cap = (b->n + n + 1) * 2 + 256;
        b->s = realloc(b->s, b->cap);
    }
    memcpy(b->s + b->n, s, n);
    b->n += n;
    b->s[b->n] = 0;
}

void sb_fmt(sbuf *b, const char *fmt, ...) {
    char tmp[4096];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(tmp, sizeof(tmp), fmt, ap);
    va_end(ap);
    if (n > 0) sb_put(b, tmp, n < (int)sizeof(tmp) ? (size_t)n : sizeof(tmp) - 1);
}

void sb_esc(sbuf *b, const char *s, size_t n) {
    char *tmp = malloc(n * 6 + 8);
    size_t m = json_escape(s, n, tmp, n * 6 + 8);
    sb_put(b, tmp, m);
    free(tmp);
}

void jv_dump(const jv *v, sbuf *o) {
    if (!v) { sb_lit(o, "null"); return; }
    switch (v->type) {
    case J_NULL: sb_lit(o, "null"); break;
    case J_BOOL: sb_lit(o, v->b ? "true" : "false"); break;
    case J_NUM:
        if (v->num >= (double)LLONG_MIN && v->num <= (double)LLONG_MAX &&
            v->num == (double)(long long)v->num)
            sb_fmt(o, "%lld", (long long)v->num);
        else
            sb_fmt(o, "%.10g", v->num);
        break;
    case J_STR:
        sb_lit(o, "\"");
        sb_esc(o, v->str, strlen(v->str));
        sb_lit(o, "\"");
        break;
    case J_ARR:
        sb_lit(o, "[");
        for (int i = 0; i < v->n; i++) {
            if (i) sb_lit(o, ",");
            jv_dump(v->items[i], o);
        }
        sb_lit(o, "]");
        break;
    case J_OBJ:
        sb_lit(o, "{");
        for (int i = 0; i < v->n; i++) {
            if (i) sb_lit(o, ",");
            sb_lit(o, "\"");
            sb_esc(o, v->keys[i], strlen(v->keys[i]));
            sb_lit(o, "\":");
            jv_dump(v->items[i], o);
        }
        sb_lit(o, "}");
        break;
    }
}
