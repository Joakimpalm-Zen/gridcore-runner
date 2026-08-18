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
    uint16_t esc;           // \uXXXX value accumulated so far
} jsonv;

// One hex digit of a `\uXXXX` escape, for both validators.
//
// `sub` is the string micro-state: 2..5 are the four digits of an escape, and
// a HIGH surrogate then requires its low half -- 6 waits for the backslash, 7
// for the `u`, 8..11 for its digits. `esc` accumulates the digits.
//
// The pairing rules are the ones json.c's parser enforces, and they live here
// rather than in each validator because a validator that admitted an escape
// the parser refuses would let a constrained model generate a document this
// program cannot read back. Returns false for a digit that commits the string
// to such an escape; every rejection leaves other digits legal, so a
// constrained model is never left with nothing to sample.
bool json_escape_hex(uint8_t *sub, uint16_t *esc, uint8_t c);
// Finish an escape that generation stopped inside, writing at most 12 bytes
// to `out` and leaving the string state ready for the closing quote. Padding
// the missing digits with zeros is what a caller would do instead, and that
// yields the NUL escape or an unpaired surrogate -- both refused by json.c,
// so the force-closed document would not parse.
int  json_escape_close(uint8_t *sub, uint16_t *esc, char *out, int cap);

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
