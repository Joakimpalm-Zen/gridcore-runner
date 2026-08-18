// Incremental JSON-prefix validator for constrained generation.
// Accepts bytes only while the output remains a valid prefix of exactly one
// top-level JSON object; sets `done` once that object closes.
#include "jsonmode.h"

#include <string.h>

#if defined(__GNUC__) || defined(__clang__)
#define FALLTHROUGH __attribute__((fallthrough))
#else
#define FALLTHROUGH
#endif

enum {
    S_START,        // expecting '{' (whitespace ok)
    S_VALUE,        // expecting any value
    S_KEY_OR_END,   // after '{': '"' or '}'
    S_KEY_EXPECT,   // after ',' in object: '"'
    S_KEY,          // inside key string
    S_COLON,        // after key: ':'
    S_STRING,       // inside value string
    S_AFTER,        // value done inside container: ',' or close
    S_ARR_FIRST,    // after '[': value or ']'
    S_NUM_MINUS, S_NUM_ZERO, S_NUM_INT, S_NUM_FRAC0, S_NUM_FRAC,
    S_NUM_EXP0, S_NUM_EXP1, S_NUM_EXP,
    S_LIT,          // inside true/false/null
    S_DONE,         // top-level object complete
};

static const char *LITS[3] = { "true", "false", "null" };

// See jsonmode.h. The rejections are as early as the digits allow -- a second
// digit that makes DC..DF cannot become anything but an unpaired low
// surrogate -- because a rejection at the FOURTH digit would leave a
// constrained model with all sixteen continuations masked.
bool json_escape_hex(uint8_t *sub, uint16_t *esc, uint8_t c) {
    int d;
    if (c >= '0' && c <= '9') d = c - '0';
    else if (c >= 'a' && c <= 'f') d = c - 'a' + 10;
    else if (c >= 'A' && c <= 'F') d = c - 'A' + 10;
    else return false;
    bool low = *sub >= 8;                     // the low half of a pair
    int nth = (int)(*sub - (low ? 8 : 2));    // 0..3
    uint16_t val = (uint16_t)((nth ? *esc : 0) * 16 + d);
    if (low) {
        if (nth == 0 && d != 0xD) return false;
        if (nth == 1 && (val < 0xDC || val > 0xDF)) return false;
    } else if (nth == 1 && val >= 0xDC && val <= 0xDF) {
        return false;                         // a low surrogate with no high half
    }
    *esc = val;
    if (nth < 3) { (*sub)++; return true; }
    // jv strings are NUL-terminated and carry no length, so U+0000 would make
    // every consumer see a truncated value -- json.c rejects it and so must
    // anything that generates it
    if (!low && val == 0) return false;
    *sub = (!low && val >= 0xD800 && val <= 0xDBFF) ? 6 : 0;
    return true;
}

int json_escape_close(uint8_t *sub, uint16_t *esc, char *out, int cap) {
    int m = 0;
    #define ESC_PUT(ch) do { if (m < cap) out[m++] = (char)(ch); } while (0)
    if (*sub == 1) { ESC_PUT('n'); *sub = 0; return m; }  // dangling backslash
    if (*sub >= 2 && *sub <= 5) {
        int nth = *sub - 2;                     // digits already written
        uint16_t cp = (uint16_t)((nth ? *esc : 0) << (4 * (4 - nth)));
        if (cp == 0) {                          // zeros would spell U+0000
            for (int i = nth; i < 3; i++) ESC_PUT('0');
            ESC_PUT('1');
            cp = 1;
        } else {
            for (int i = nth; i < 4; i++) ESC_PUT('0');
        }
        *sub = (cp >= 0xD800 && cp <= 0xDBFF) ? 6 : 0;
    }
    if (*sub == 6) { ESC_PUT('\\'); *sub = 7; }
    if (*sub == 7) { ESC_PUT('u');  *sub = 8; }
    if (*sub >= 8 && *sub <= 11) {
        // the smallest low surrogate consistent with what is already written;
        // json_escape_hex has pinned digit 1 to D and digit 2 to C..F, so
        // zeros are safe from the third digit on
        static const char LOW[4] = { 'D', 'C', '0', '0' };
        for (int i = *sub - 8; i < 4; i++) ESC_PUT(LOW[i]);
        *sub = 0;
    }
    #undef ESC_PUT
    return m;
}

void jsonv_init(jsonv *v) {
    v->depth = 0;
    v->st = S_START;
    v->sub = 0;
    v->lit = 0;
    v->esc = 0;
    v->done = false;
}

void jsonv_init_any(jsonv *v) {
    jsonv_init(v);
    v->st = S_VALUE;
}

bool jsonv_value_end(const jsonv *v) {
    if (v->done) return true;
    return v->depth == 0 && (v->st == S_NUM_ZERO || v->st == S_NUM_INT ||
                             v->st == S_NUM_FRAC || v->st == S_NUM_EXP);
}

static bool is_ws(uint8_t c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

static bool push(jsonv *v, uint8_t c) {
    if (v->depth >= (int)sizeof(v->stack)) return false;
    v->stack[v->depth++] = c;
    return true;
}

// a value just finished at the current depth
static void value_done(jsonv *v) {
    if (v->depth == 0) {
        v->st = S_DONE;
        v->done = true;
    } else {
        v->st = S_AFTER;
    }
}

// returns false on invalid byte; may set reconsume for number terminators
static bool feed_byte(jsonv *v, uint8_t c, bool *reconsume) {
    *reconsume = false;
    switch (v->st) {
    case S_START:
        // A constrained document must begin with its opening token. Leading
        // whitespace is refused because it is the one position where a model
        // can burn its whole budget without producing any content: with every
        // other byte illegal, spaces and newlines are the only moves left, and
        // the decode livelocks until max_tokens forces a close. Interior
        // whitespace stays legal throughout (see S_VALUE, S_AFTER, and the
        // rest) — that is ordinary pretty-printing, and by then the document
        // has real content in it. See leading_ws_ok() in schema.c.
        if (c == '{') { v->st = S_KEY_OR_END; return push(v, 'O'); }
        return false;

    case S_ARR_FIRST:
        if (c == ']' && !is_ws(c)) { v->depth--; value_done(v); return true; }
        // fall through: anything else must start a value
        FALLTHROUGH;
    case S_VALUE:
        if (is_ws(c)) return true;
        if (c == '{') { v->st = S_KEY_OR_END; return push(v, 'O'); }
        if (c == '[') { v->st = S_ARR_FIRST; return push(v, 'A'); }
        if (c == '"') { v->st = S_STRING; v->sub = 0; return true; }
        if (c == '-') { v->st = S_NUM_MINUS; return true; }
        if (c == '0') { v->st = S_NUM_ZERO; return true; }
        if (c >= '1' && c <= '9') { v->st = S_NUM_INT; return true; }
        if (c == 't') { v->st = S_LIT; v->lit = 0; v->sub = 1; return true; }
        if (c == 'f') { v->st = S_LIT; v->lit = 1; v->sub = 1; return true; }
        if (c == 'n') { v->st = S_LIT; v->lit = 2; v->sub = 1; return true; }
        return false;

    case S_KEY_OR_END:
        if (is_ws(c)) return true;
        if (c == '"') { v->st = S_KEY; v->sub = 0; return true; }
        if (c == '}') { v->depth--; value_done(v); return true; }
        return false;

    case S_KEY_EXPECT:
        if (is_ws(c)) return true;
        if (c == '"') { v->st = S_KEY; v->sub = 0; return true; }
        return false;

    case S_KEY:
    case S_STRING: {
        bool key = v->st == S_KEY;
        if (v->sub == 1) { // after backslash
            if (c == '"' || c == '\\' || c == '/' || c == 'b' || c == 'f' ||
                c == 'n' || c == 'r' || c == 't') { v->sub = 0; return true; }
            if (c == 'u') { v->sub = 2; return true; }
            return false;
        }
        if (v->sub == 6) { if (c != '\\') return false; v->sub = 7; return true; }
        if (v->sub == 7) { if (c != 'u')  return false; v->sub = 8; return true; }
        if (v->sub >= 2) return json_escape_hex(&v->sub, &v->esc, c);
        if (c == '"') {
            if (key) v->st = S_COLON;
            else value_done(v);
            return true;
        }
        if (c == '\\') { v->sub = 1; return true; }
        return c >= 0x20; // control chars forbidden; UTF-8 bytes allowed
    }

    case S_COLON:
        if (is_ws(c)) return true;
        if (c == ':') { v->st = S_VALUE; return true; }
        return false;

    case S_AFTER: {
        if (is_ws(c)) return true;
        uint8_t top = v->depth > 0 ? v->stack[v->depth - 1] : 0;
        if (c == ',' && top == 'O') { v->st = S_KEY_EXPECT; return true; }
        if (c == ',' && top == 'A') { v->st = S_VALUE; return true; }
        if (c == '}' && top == 'O') { v->depth--; value_done(v); return true; }
        if (c == ']' && top == 'A') { v->depth--; value_done(v); return true; }
        return false;
    }

    case S_NUM_MINUS:
        if (c == '0') { v->st = S_NUM_ZERO; return true; }
        if (c >= '1' && c <= '9') { v->st = S_NUM_INT; return true; }
        return false;
    case S_NUM_ZERO:
    case S_NUM_INT:
        if (v->st == S_NUM_INT && c >= '0' && c <= '9') return true;
        if (c == '.') { v->st = S_NUM_FRAC0; return true; }
        if (c == 'e' || c == 'E') { v->st = S_NUM_EXP0; return true; }
        value_done(v); *reconsume = true; return true;
    case S_NUM_FRAC0:
        if (c >= '0' && c <= '9') { v->st = S_NUM_FRAC; return true; }
        return false;
    case S_NUM_FRAC:
        if (c >= '0' && c <= '9') return true;
        if (c == 'e' || c == 'E') { v->st = S_NUM_EXP0; return true; }
        value_done(v); *reconsume = true; return true;
    case S_NUM_EXP0:
        if (c == '+' || c == '-') { v->st = S_NUM_EXP1; return true; }
        if (c >= '0' && c <= '9') { v->st = S_NUM_EXP; return true; }
        return false;
    case S_NUM_EXP1:
        if (c >= '0' && c <= '9') { v->st = S_NUM_EXP; return true; }
        return false;
    case S_NUM_EXP:
        if (c >= '0' && c <= '9') return true;
        value_done(v); *reconsume = true; return true;

    case S_LIT: {
        const char *lit = LITS[v->lit];
        if (c == (uint8_t)lit[v->sub]) {
            v->sub++;
            if (lit[v->sub] == 0) value_done(v);
            return true;
        }
        return false;
    }

    case S_DONE:
        return false; // nothing (not even whitespace) after the object

    default:
        return false;
    }
}

// forcibly complete the JSON from the current state (used when the token
// budget runs out): close strings/escapes, complete literals and numbers
// with minimal filler, then close all open containers. Returns bytes
// written, or 0 if generation never started an object.
int jsonv_close(jsonv *v, char *out, int cap) {
    int m = 0;
    if (v->st == S_START || v->done) return 0;
    #define EMIT(c) do { if (m < cap - 1) out[m++] = (c); } while (0)
    // unfinished string escapes
    if (v->st == S_KEY || v->st == S_STRING) {
        char esc[12];
        int en = json_escape_close(&v->sub, &v->esc, esc, (int)sizeof(esc));
        for (int i = 0; i < en; i++) EMIT(esc[i]);
        EMIT('"');
        if (v->st == S_KEY) v->st = S_COLON;
        else value_done(v);
    }
    if (v->st == S_COLON)      { EMIT(':'); v->st = S_VALUE; }
    if (v->st == S_KEY_EXPECT) { EMIT('"'); EMIT('_'); EMIT('"'); EMIT(':'); v->st = S_VALUE; }
    if (v->st == S_LIT) {
        const char *lit = LITS[v->lit];
        while (lit[v->sub]) EMIT(lit[v->sub++]);
        value_done(v);
    }
    if (v->st == S_NUM_MINUS || v->st == S_NUM_FRAC0 ||
        v->st == S_NUM_EXP0 || v->st == S_NUM_EXP1) {
        EMIT('0');
        value_done(v);
    }
    if (v->st == S_NUM_ZERO || v->st == S_NUM_INT ||
        v->st == S_NUM_FRAC || v->st == S_NUM_EXP) {
        value_done(v); // number already complete as-is
    }
    if (v->st == S_VALUE || v->st == S_ARR_FIRST) {
        EMIT('n'); EMIT('u'); EMIT('l'); EMIT('l');
        value_done(v);
    }
    // close remaining containers
    while (!v->done && v->depth > 0 &&
           (v->st == S_AFTER || v->st == S_KEY_OR_END)) {
        uint8_t top = v->stack[v->depth - 1];
        EMIT(top == 'O' ? '}' : ']');
        v->depth--;
        value_done(v);
    }
    #undef EMIT
    out[m] = 0;
    return m;
}

bool jsonv_feed(jsonv *v, const char *s, int n) {
    for (int i = 0; i < n; i++) {
        bool re;
        if (!feed_byte(v, (uint8_t)s[i], &re)) return false;
        if (re && !feed_byte(v, (uint8_t)s[i], &re)) return false;
    }
    return true;
}

// The machine reads stack[0 .. depth-1] and nothing above it: push() writes
// stack[depth] before depth names it, and every reader indexes depth-1 or
// less. So a snapshot only has to carry that prefix. Anything added here that
// reads a slot above `depth` breaks this — and jsonv_trial's test poisons the
// scratch, so it breaks loudly.
void jsonv_snapshot(jsonv *dst, const jsonv *src) {
    int d = src->depth;
    if (d < 0) d = 0;
    else if (d > (int)sizeof(src->stack)) d = (int)sizeof(src->stack);
    memcpy(dst->stack, src->stack, (size_t)d);
    dst->depth = src->depth;
    dst->st = src->st;
    dst->sub = src->sub;
    dst->lit = src->lit;
    dst->done = src->done;
    dst->esc = src->esc;
}

bool jsonv_trial(const jsonv *v, jsonv *scratch, const char *s, int n) {
    jsonv_snapshot(scratch, v);
    return jsonv_feed(scratch, s, n);
}
