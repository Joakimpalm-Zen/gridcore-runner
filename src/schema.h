// JSON-schema compilation and the streaming schema validator.
#ifndef RUNNER_SCHEMA_H
#define RUNNER_SCHEMA_H

#include <stdint.h>
#include <stdbool.h>
#include "jsonmode.h"

enum sn_kind { SN_ANY, SN_NULL, SN_BOOL, SN_NUM, SN_INT, SN_STR, SN_ENUM,
               SN_OBJ, SN_MAP, SN_ARR, SN_UNION, SN_COND, SN_SEQ, SN_RAW };

// One segment of a compiled `pattern`: a literal run, then `min_tail` to
// `max_tail` bytes of `ascii` when `has_class`. max_tail is -1 for the
// unbounded forms (`+`, `{n,}`), which only the final segment may use.
typedef struct {
    char *prefix; int prefix_len;
    bool  has_class;
    int   min_tail, max_tail;
    bool  ascii[128];
} pat_seg;

typedef struct snode snode;
struct snode {
    int     kind;
    char  **lits; int n_lits;                       // enum literals (JSON text)
    char  **keys; int *key_len; snode **props;      // object properties
    bool   *req;  int n_props;                      //   (declared order)
    snode  *items; int min_items, max_items;        // array items / map values
    snode **alts; int n_alts;                       // type unions
    bool    whitespace_significant;                 // protocol literals/branches
    int64_t num_min, num_max;                       // enforced integer interval
    bool    has_num_min, has_num_max;
    double  real_min, real_max;                     // enforced number interval
    bool    has_real_min, has_real_max;
    // A compiled `pattern` is a sequence of segments, each a literal run
    // followed by an optional repeated ASCII class. One segment is the
    // common case (`^wf_[a-z0-9-]{6,}$`); several express shapes like
    // `^[A-Z]{3}[0-9]{4}$`. Every segment but the last is FIXED-length, so
    // the segment holding byte `p` follows from `p` alone and the enforced
    // language is still exactly the declared one -- see compile_pattern.
    pat_seg *pat; int n_pat;
    // SN_RAW's terminator. Its own field: this used to borrow the pattern
    // prefix, which tied two unrelated features to one pointer.
    char   *sentinel; int sentinel_len;
};

struct jv;
snode *schema_compile(struct jv *schema, char *err, int errcap);
// Compile OpenAI tools[] to Muse's native atem call block. Scalar parameter
// values are added in S3; S2 admits structured JSON values.
snode *schema_compile_atem_tools(struct jv *tools, char *err, int errcap);
// Compile the recipient header together with the native call. The duplicated
// invoke name is pinned by the header discriminator; `user` selects raw text
// or final_schema when supplied.
enum atem_turn_start {
    ATEM_TURN_DIRECT,
    ATEM_TURN_AFTER_REASONING,
    ATEM_TURN_EITHER,
};
snode *schema_compile_atem_turn(struct jv *tools, bool allow_user,
                                const char *only_tool,
                                struct jv *final_schema,
                                enum atem_turn_start start,
                                char *err, int errcap);
snode *schema_compile_atem_parallel(struct jv *tools, const char *only_tool,
                                    char *err, int errcap);
// Wrap a JSON/schema payload in Muse's user-recipient header. The second
// branch starts after a reasoning close has already consumed ` to=`.
snode *schema_compile_muse_user_payload(struct jv *schema,
                                        char *err, int errcap);
void   schema_free(snode *n);

// streaming validator state (memcpy-copyable for token lookahead)
typedef struct {
    const snode *node;
    uint8_t  phase, sub;
    int32_t  idx;       // array element / object key counter (int32: max/minItems
                        // are bounded only by INT_MAX, so int16 could wrap and
                        // silently drop maxItems enforcement past 32767)
    int32_t  lit_pos;   // string chars / literal bytes seen (int32: file-sized strings)
    uint16_t disc;      // this object's discriminator choice + 1; 0 = not chosen yet
    uint64_t alive;
    uint64_t num_abs;  // integer magnitude accumulated so far
    char     num_text[96]; // number spelling for bounded-number validation
    uint8_t  num_len;
} sframe;

typedef struct {
    sframe stack[48];
    int    depth;
    bool   done;
    int    last_enum;   // enum literal completed by the most recent child
    jsonv  any;      // generic submachine for open {} schema nodes
} sval;

void sval_init (sval *v, const snode *root);
bool sval_feed (sval *v, const char *s, int n);
// True when whitespace at the current position is string CONTENT, not an
// insignificant separator. Callers suppress separator whitespace to stop a
// constrained model burning its budget on blank runs; see schema.c.
bool sval_ws_is_content(const sval *v);
// true while the validator is inside a trailing raw value whose only
// terminator is the model's own stop token (which decodes to no bytes); the
// engine accepts a stop token there as the answer's natural end
bool sval_at_raw_tail(const sval *v);
int  sval_close(sval *v, char *out, int cap);

#endif // RUNNER_SCHEMA_H
