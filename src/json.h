// Minimal JSON tree parser + string escaping (for the HTTP API).
#ifndef RUNNER_JSON_H
#define RUNNER_JSON_H

#include <stddef.h>
#include <stdbool.h>

typedef enum { J_NULL, J_BOOL, J_NUM, J_STR, J_ARR, J_OBJ } jtype;

typedef struct jv jv;
struct jv {
    jtype  type;
    double num;
    bool   b;
    char  *str;      // J_STR (decoded, NUL-terminated)
    jv   **items;    // J_ARR / J_OBJ values
    char **keys;     // J_OBJ keys
    int    n;
};

jv         *json_parse(const char *s, size_t n); // NULL on error
void        jv_free(jv *v);
jv         *jv_get(jv *obj, const char *key);    // NULL if absent / not object
const char *jv_str (jv *v, const char *dflt);
double      jv_num (jv *v, double dflt);
bool        jv_bool(jv *v, bool dflt);

// escape s (n bytes) as JSON string content (no surrounding quotes);
// returns bytes written, always NUL-terminates within cap
size_t json_escape(const char *s, size_t n, char *out, size_t cap);

// Inverse of one escape sequence, for callers decoding a JSON string as it
// arrives rather than from a complete document. `s` points at the backslash
// and `n` is how many bytes are readable from there. Decoded UTF-8 goes to
// `out` (at most 4 bytes) with its length in *outn. Returns the input bytes
// consumed, 0 when the sequence is incomplete and more input would settle it,
// or -1 when it can never be valid.
int json_unescape(const char *s, size_t n, char out[4], int *outn);

// Growable string builder for assembling JSON/HTTP bodies.
//
// `failed` latches on allocation failure and every sb_* call then becomes a
// no-op, so a caller can build a whole body and check once at the end. A
// caller that ignores it would emit a short body that still looks successful,
// which is worse than an error: check it and answer 500. Zero-initialised
// (`sbuf b = {0}`) means healthy.
typedef struct sbuf { char *s; size_t n, cap; bool failed; } sbuf;
void sb_put(sbuf *b, const char *s, size_t n);
#define sb_lit(b, lit) sb_put(b, lit, strlen(lit))
#if defined(__GNUC__) || defined(__clang__)
void sb_fmt(sbuf *b, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
#else
void sb_fmt(sbuf *b, const char *fmt, ...);
#endif
void sb_esc(sbuf *b, const char *s, size_t n); // appends as JSON string content

// re-serialize a parsed value (inverse of json_parse, minus formatting).
// Compact -- no space after ':' or ',' -- and that is the HTTP API's format:
// every response body the server emits comes out of here.
void jv_dump(const jv *v, sbuf *o);

// The same JSON with jinja's `tojson` spacing: ", " between members and ": "
// after a key. NOT a prettier jv_dump -- it exists because several families
// declare their tools through `{{ tool | tojson }}` (chatml.jinja:11,
// chatml-think.jinja:9, ornith.jinja:50, muse.jinja:73) and that block is
// prompt text the model was trained on verbatim, so the separators are part
// of the contract. Use it ONLY when reproducing a reference template that
// pipes the value through `tojson`; the API response path stays on jv_dump,
// where these bytes would be gratuitous drift in every body.
//
// Known residual, NOT fixed here: `jv` stores every number as a double, so
// this cannot reproduce python's int/float distinction (`1.0` comes back as
// `1`) and formats non-integral values with "%.10g" rather than python's
// shortest round-trip repr. Schema numerals in practice (0, 100, 0.01) agree;
// a float needing more than 10 significant digits, a non-integral value
// written as `1.0`, or an integer past 2^63 will not.
void jv_dump_tojson(const jv *v, sbuf *o);

// jinja's `| dictsort` over an object's members: fills `order` (which must
// hold v->n ints) with member indices sorted by key.
//
// It lives here, next to jv, because TWO unrelated modules need the identical
// order and must not each guess it: template.c renders gemma4's declarations
// and call blocks with dictsorted keys, and schema.c compiles the grammar
// that constrains those same blocks. A disagreement between the two would be
// a grammar that rejects the model's own trained output.
//
// jinja's default is case_sensitive=False and python's sort is stable, so the
// order is by LOWERCASED key with ties left in insertion order.
void jv_dictsort(const jv *obj, int *order);

#endif
