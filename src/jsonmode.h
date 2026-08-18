// Incremental JSON validator (json_mode).
#ifndef RUNNER_JSONMODE_H
#define RUNNER_JSONMODE_H

#include <stdint.h>
#include <stdbool.h>

// incremental validator: accepts byte strings only while they remain a valid
// prefix of a single JSON object; small and memcpy-copyable for lookahead
typedef struct {
    uint8_t stack[200];     // container nesting: 'O' object, 'A' array
    int16_t depth;
    uint8_t st, sub, lit;   // micro-state, escape/digit progress, literal id
    bool    done;           // a complete top-level object has been parsed
} jsonv;

void jsonv_init(jsonv *v);      // accept exactly one JSON object
void jsonv_init_any(jsonv *v);  // accept exactly one JSON value of any kind
bool jsonv_feed(jsonv *v, const char *s, int n);
// Copy only the LIVE part of `src` into `dst`: the container stack above
// `depth` is never read before it is written, so copying those bytes is pure
// cost. Only the trial probes below need this; ordinary state keeping should
// just assign the struct.
void jsonv_snapshot(jsonv *dst, const jsonv *src);
// Would `s` keep the validator alive? Answers without touching `v`, running
// the trial in caller-owned `scratch` whose previous contents are irrelevant.
//
// This is the candidate-token oracle: constrained sampling with top_k off
// calls it once per vocabulary entry, per token, so what it copies is a
// decode-speed property. See sval_trial in schema.h.
bool jsonv_trial(const jsonv *v, jsonv *scratch, const char *s, int n);
// true if the machine stopped at a self-terminated value boundary (numbers)
bool jsonv_value_end(const jsonv *v);
// force-complete the object (token budget ran out); returns bytes written
int  jsonv_close(jsonv *v, char *out, int cap);

#endif // RUNNER_JSONMODE_H
