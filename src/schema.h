// JSON-schema compilation and the streaming schema validator.
#ifndef RUNNER_SCHEMA_H
#define RUNNER_SCHEMA_H

#include <stdint.h>
#include <stdbool.h>
#include "jsonmode.h"

enum sn_kind { SN_ANY, SN_NULL, SN_BOOL, SN_NUM, SN_INT, SN_STR, SN_ENUM,
               SN_OBJ, SN_ARR, SN_UNION, SN_COND };

typedef struct snode snode;
struct snode {
    int     kind;
    char  **lits; int n_lits;                       // enum literals (JSON text)
    char  **keys; int *key_len; snode **props;      // object properties
    bool   *req;  int n_props;                      //   (declared order)
    snode  *items; int min_items, max_items;        // array
    snode **alts; int n_alts;                       // type unions
    int64_t num_min, num_max;                       // enforced integer interval
    bool    has_num_min, has_num_max;
    double  real_min, real_max;                     // enforced number interval
    bool    has_real_min, has_real_max;
    char   *pattern_prefix; int pattern_prefix_len, pattern_min_tail;
    bool    pattern_ascii[128];
};

struct jv;
snode *schema_compile(struct jv *schema, char *err, int errcap);
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
int  sval_close(sval *v, char *out, int cap);

#endif // RUNNER_SCHEMA_H
