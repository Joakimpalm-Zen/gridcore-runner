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

// growable string builder for assembling JSON/HTTP bodies
typedef struct sbuf { char *s; size_t n, cap; } sbuf;
void sb_put(sbuf *b, const char *s, size_t n);
#define sb_lit(b, lit) sb_put(b, lit, strlen(lit))
#if defined(__GNUC__) || defined(__clang__)
void sb_fmt(sbuf *b, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
#else
void sb_fmt(sbuf *b, const char *fmt, ...);
#endif
void sb_esc(sbuf *b, const char *s, size_t n); // appends as JSON string content

// re-serialize a parsed value (inverse of json_parse, minus formatting)
void jv_dump(const jv *v, sbuf *o);

#endif
