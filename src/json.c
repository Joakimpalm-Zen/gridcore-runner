// Minimal recursive-descent JSON parser, string escaper, string builder,
// and value re-serializer.
#include "json.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <errno.h>

typedef struct { const char *p, *end; int depth; } jcur;

static size_t utf8_seq(const char *s, size_t i, size_t n);

static void skip_ws(jcur *c) {
    while (c->p < c->end && (*c->p == ' ' || *c->p == '\t' ||
                             *c->p == '\n' || *c->p == '\r')) c->p++;
}

static jv *jv_new(jtype t) {
    jv *v = calloc(1, sizeof(jv));
    if (!v) return NULL;
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

int json_unescape(const char *s, size_t n, char out[4], int *outn) {
    *outn = 0;
    if (n < 1 || s[0] != '\\') return -1;
    if (n < 2) return 0;
    char simple = 0;
    switch (s[1]) {
        case '"':  simple = '"';  break;
        case '\\': simple = '\\'; break;
        case '/':  simple = '/';  break;
        case 'b':  simple = '\b'; break;
        case 'f':  simple = '\f'; break;
        case 'n':  simple = '\n'; break;
        case 'r':  simple = '\r'; break;
        case 't':  simple = '\t'; break;
        case 'u':  break;
        default: return -1;
    }
    if (simple) { out[0] = simple; *outn = 1; return 2; }
    if (n < 6) return 0;
    int cp = hex4(s + 2);
    if (cp < 0) return -1;
    // jv strings are NUL-terminated and intentionally do not carry a byte
    // length. Accepting U+0000 would make every consumer see a silently
    // truncated value/key, so this representation must reject it.
    if (cp == 0) return -1;
    size_t used = 6;
    if (cp >= 0xD800 && cp <= 0xDBFF) {
        // A high surrogate is only half a character: peek for its pair, but
        // only wait for bytes that could still be one. Anything that cannot
        // complete the pair is rejected outright, INCLUDING an escape that is
        // not \u -- emitting the surrogate on its own produces ED A0 80, which
        // is not UTF-8 and breaks every strict client that reads the body.
        if (n < 7) return 0;
        if (s[6] != '\\') return -1;
        if (n < 8) return 0;
        if (s[7] != 'u') return -1;
        if (n < 12) return 0;
        int lo = hex4(s + 8);
        if (lo < 0xDC00 || lo > 0xDFFF) return -1;
        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
        used = 12;
    } else if (cp >= 0xDC00 && cp <= 0xDFFF) return -1;
    *outn = u8_emit((unsigned)cp, out);
    return (int)used;
}

// parse a string (cursor at opening quote); returns malloc'd decoded string
static char *parse_string(jcur *c) {
    if (c->p >= c->end || *c->p != '"') return NULL;
    c->p++;
    size_t cap = 32, m = 0;
    char *out = malloc(cap);
    if (!out) return NULL;
    while (c->p < c->end) {
        if (m + 8 > cap) {
            // via a temporary: `out = realloc(out, cap)` strands the original
            // block on failure and then yields NULL for the next write
            char *tmp = realloc(out, cap * 2);
            if (!tmp) goto fail;
            out = tmp;
            cap *= 2;
        }
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
                    if (cp == 0) goto fail;
                    c->p += 4;
                    if (cp >= 0xD800 && cp <= 0xDBFF) {
                        if (c->p + 6 > c->end || c->p[0] != '\\' ||
                            c->p[1] != 'u') goto fail;
                        int lo = hex4(c->p + 2);
                        if (lo < 0xDC00 || lo > 0xDFFF) goto fail;
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                        c->p += 6;
                    } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
                        goto fail;
                    }
                    m += u8_emit((unsigned)cp, out + m);
                    break;
                }
                default: goto fail;
            }
        } else if (ch < 0x20) {
            goto fail;
        } else if (ch < 0x80) {
            out[m++] = (char)ch;
            c->p++;
        } else {
            size_t len = utf8_seq(c->p, 0, (size_t)(c->end - c->p));
            if (len == 0) goto fail;
            memcpy(out + m, c->p, len);
            m += len;
            c->p += len;
        }
    }
fail:
    free(out);
    return NULL;
}

static jv *parse_value(jcur *c);

// Object member names are already decoded by parse_string(), so hashing those
// names catches both literal duplicates and equivalent escape spellings.  Keep
// this separate from jv's compact public representation: the table exists only
// while an object is being parsed and makes duplicate detection O(n) expected
// time instead of turning large request bodies into an O(n^2) scan.
typedef struct {
    const char **slots;
    size_t cap, used;
} keyset;

static uint64_t key_hash(const char *s) {
    uint64_t h = UINT64_C(1469598103934665603);
    for (; *s; s++) {
        h ^= (unsigned char)*s;
        h *= UINT64_C(1099511628211);
    }
    return h;
}

// Returns 1 when inserted, 0 for a duplicate, and -1 on allocation failure.
static int keyset_insert(keyset *set, const char *key) {
    if (set->used + 1 > set->cap / 2) {
        size_t cap = set->cap ? set->cap * 2 : 16;
        if (cap < set->cap || cap > SIZE_MAX / sizeof(*set->slots)) return -1;
        const char **slots = calloc(cap, sizeof(*slots));
        if (!slots) return -1;
        for (size_t i = 0; i < set->cap; i++) {
            const char *old = set->slots[i];
            if (!old) continue;
            size_t at = (size_t)key_hash(old) & (cap - 1);
            while (slots[at]) at = (at + 1) & (cap - 1);
            slots[at] = old;
        }
        free(set->slots);
        set->slots = slots;
        set->cap = cap;
    }
    size_t at = (size_t)key_hash(key) & (set->cap - 1);
    while (set->slots[at]) {
        if (!strcmp(set->slots[at], key)) return 0;
        at = (at + 1) & (set->cap - 1);
    }
    set->slots[at] = key;
    set->used++;
    return 1;
}

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
    errno = 0;
    double d = strtod(tmp, &endp);
    // isfinite() is folded away by -ffast-math (the release build uses it), so
    // an overflowing literal like 1e400 parsed as +inf and passed every later
    // range check. strtod's ERANGE survives the optimiser.
    bool ok = endp == tmp + n && errno != ERANGE &&
              d > -1.7e308 && d < 1.7e308;
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
    keyset keys = {0};
    if (!v) return NULL;
    c->p++; // consume open
    skip_ws(c);
    if (c->p < c->end && *c->p == close) { c->p++; return v; }
    for (;;) {
        char *key = NULL;
        if (v->type == J_OBJ) {
            skip_ws(c);
            key = parse_string(c);
            if (!key) goto fail;
            if (keyset_insert(&keys, key) != 1) { free(key); goto fail; }
            skip_ws(c);
            if (c->p >= c->end || *c->p != ':') { free(key); goto fail; }
            c->p++;
        }
        jv *item = parse_value(c);
        if (!item) { free(key); goto fail; }
        // item and key are not owned by v until both arrays have grown, so a
        // failure here has to free them itself
        jv **grown = realloc(v->items, sizeof(jv *) * (v->n + 1));
        if (!grown) { free(key); jv_free(item); goto fail; }
        v->items = grown;
        if (v->type == J_OBJ) {
            char **kgrown = realloc(v->keys, sizeof(char *) * (v->n + 1));
            if (!kgrown) { free(key); jv_free(item); goto fail; }
            v->keys = kgrown;
        }
        v->items[v->n] = item;
        if (v->type == J_OBJ) v->keys[v->n] = key;
        v->n++;
        skip_ws(c);
        if (c->p < c->end && *c->p == ',') { c->p++; continue; }
        if (c->p < c->end && *c->p == close) {
            c->p++;
            free(keys.slots);
            return v;
        }
        goto fail;
    }
fail:
    free(keys.slots);
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
            if (s) {
                r = jv_new(J_STR);
                if (r) r->str = s;
                else free(s);
            }
        } else if (ch == 't' && c->end - c->p >= 4 && !memcmp(c->p, "true", 4)) {
            r = jv_new(J_BOOL); if (r) r->b = true; c->p += 4;
        } else if (ch == 'f' && c->end - c->p >= 5 && !memcmp(c->p, "false", 5)) {
            r = jv_new(J_BOOL); if (r) r->b = false; c->p += 5;
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
// Length of the well-formed UTF-8 sequence at s[i], or 0 if there isn't one.
//
// Structure is NOT enough. A byte pattern can have a valid lead byte and valid
// continuation bytes and still not be UTF-8, and a strict decoder — every JSON
// client — rejects it. Checking only the shape let 0xC0 through as a two-byte
// lead and emitted a response body Python could not decode at all
// ("invalid start byte"), which a model reaches whenever a byte-fallback token
// emits a stray byte or `max_tokens` cuts a multi-byte character in half.
//
// The three families a conforming decoder refuses, all rejected here so the
// caller replaces them with U+FFFD:
//   overlong    — a value encoded in more bytes than it needs (0xC0/0xC1 at
//                 two bytes, 0xE0 80..9F, 0xF0 80..8F)
//   surrogates  — U+D800..DFFF (0xED A0..BF) exist only in UTF-16
//   out of range— anything past U+10FFFF (0xF4 90.. and every 0xF5..0xFF lead)
static size_t utf8_seq(const char *s, size_t i, size_t n) {
    unsigned char c = (unsigned char)s[i];
    size_t len = c < 0x80 ? 1 : (c & 0xE0) == 0xC0 ? 2 :
                 (c & 0xF0) == 0xE0 ? 3 : (c & 0xF8) == 0xF0 ? 4 : 0;
    if (len == 0 || i + len > n) return 0;
    for (size_t k = 1; k < len; k++)
        if (((unsigned char)s[i + k] & 0xC0) != 0x80) return 0;
    unsigned char c1 = len > 1 ? (unsigned char)s[i + 1] : 0;
    if (len == 2 && c < 0xC2) return 0;
    if (len == 3 && c == 0xE0 && c1 < 0xA0) return 0;
    if (len == 3 && c == 0xED && c1 >= 0xA0) return 0;
    if (len == 4 && c == 0xF0 && c1 < 0x90) return 0;
    if (len == 4 && (c > 0xF4 || (c == 0xF4 && c1 >= 0x90))) return 0;
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
    if (b->failed) return;
    if (b->n + n + 1 > b->cap) {
        if (n > SIZE_MAX / 2 - b->n - 256) { b->failed = true; return; }
        size_t cap = (b->n + n + 1) * 2 + 256;
        char *grown = realloc(b->s, cap);
        if (!grown) { b->failed = true; return; }
        b->s = grown;
        b->cap = cap;
    }
    memcpy(b->s + b->n, s, n);
    b->n += n;
    b->s[b->n] = 0;
}

void sb_fmt(sbuf *b, const char *fmt, ...) {
    if (b->failed) return;
    // The stack buffer is a fast path, not a limit: a caller formats
    // client-supplied text through here (tool-call arguments, which routinely
    // exceed 4 KB), and silently keeping the first 4095 bytes corrupted the
    // prompt mid-JSON and dropped the closing marker.
    char tmp[4096];
    va_list ap, ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int n = vsnprintf(tmp, sizeof(tmp), fmt, ap);
    va_end(ap);
    if (n < 0) { b->failed = true; va_end(ap2); return; }
    if ((size_t)n < sizeof(tmp)) {
        sb_put(b, tmp, (size_t)n);
    } else {
        char *big = malloc((size_t)n + 1);
        if (!big) { b->failed = true; va_end(ap2); return; }
        vsnprintf(big, (size_t)n + 1, fmt, ap2);
        sb_put(b, big, (size_t)n);
        free(big);
    }
    va_end(ap2);
}

void sb_esc(sbuf *b, const char *s, size_t n) {
    if (b->failed) return;
    if (n > (SIZE_MAX - 8) / 6) { b->failed = true; return; }
    size_t cap = n * 6 + 8;
    char *tmp = malloc(cap);
    if (!tmp) { b->failed = true; return; }
    size_t m = json_escape(s, n, tmp, cap);
    sb_put(b, tmp, m);
    free(tmp);
}

// `comma` and `colon` are the only difference between the two public dumps:
// "," / ":" is the compact wire format, ", " / ": " is what jinja's tojson
// emits. Everything else -- escaping, number formatting, key order -- is
// shared on purpose, so the two can never disagree about anything but spacing.
static void jv_dump_sep(const jv *v, sbuf *o,
                        const char *comma, const char *colon) {
    if (!v) { sb_lit(o, "null"); return; }
    switch (v->type) {
    case J_NULL: sb_lit(o, "null"); break;
    case J_BOOL: sb_lit(o, v->b ? "true" : "false"); break;
    case J_NUM:
        // (double)LLONG_MAX rounds UP to 2^63, which is not representable as
        // long long — the upper bound must exclude it (strict compare)
        if (v->num >= (double)LLONG_MIN && v->num < 9223372036854775808.0 &&
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
            if (i) sb_lit(o, comma);
            jv_dump_sep(v->items[i], o, comma, colon);
        }
        sb_lit(o, "]");
        break;
    case J_OBJ:
        sb_lit(o, "{");
        for (int i = 0; i < v->n; i++) {
            if (i) sb_lit(o, comma);
            sb_lit(o, "\"");
            sb_esc(o, v->keys[i], strlen(v->keys[i]));
            sb_lit(o, "\"");
            sb_lit(o, colon);
            jv_dump_sep(v->items[i], o, comma, colon);
        }
        sb_lit(o, "}");
        break;
    }
}

void jv_dump(const jv *v, sbuf *o) { jv_dump_sep(v, o, ",", ":"); }

void jv_dump_tojson(const jv *v, sbuf *o) { jv_dump_sep(v, o, ", ", ": "); }

static int jv_key_cmp(const char *a, const char *b) {
    for (;; a++, b++) {
        unsigned char x = (unsigned char)*a, y = (unsigned char)*b;
        if (x >= 'A' && x <= 'Z') x = (unsigned char)(x - 'A' + 'a');
        if (y >= 'A' && y <= 'Z') y = (unsigned char)(y - 'A' + 'a');
        if (x != y) return x < y ? -1 : 1;
        if (!x) return 0;
    }
}

// Insertion sort over an index array: an object here holds a handful of
// members, and stability is the point rather than an implementation detail.
void jv_dictsort(const jv *obj, int *order) {
    if (!obj || obj->type != J_OBJ) return;
    for (int i = 0; i < obj->n; i++) {
        int j = i;
        while (j > 0 && jv_key_cmp(obj->keys[order[j - 1]], obj->keys[i]) > 0) {
            order[j] = order[j - 1];
            j--;
        }
        order[j] = i;
    }
}
