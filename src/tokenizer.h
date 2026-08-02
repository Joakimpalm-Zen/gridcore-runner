// SPM / BPE tokenizer built from GGUF vocabulary metadata.
#ifndef RUNNER_TOKENIZER_H
#define RUNNER_TOKENIZER_H

#include "gguf.h"

enum { TOK_SPM, TOK_BPE, TOK_BPE_SPM }; // BPE_SPM: gemma4 (spaces to U+2581, raw-UTF-8 BPE)

// BPE pre-tokenizer split rules, from tokenizer.ggml.pre. GPT2 is the original
// regex (a leading space may join a run); the newer families let any single
// non-letter/non-digit character lead a letter run and cap digit runs, which
// changes where pre-token boundaries fall. Unrecognised values stay on GPT2.
enum { TOK_PRE_GPT2, TOK_PRE_LLAMA3, TOK_PRE_QWEN2, TOK_PRE_SMOLLM,
       TOK_PRE_TEKKEN };

typedef struct { const char *key; uint32_t klen; int32_t val; } hmap_ent;
typedef struct { hmap_ent *e; size_t cap; } hmap;

typedef struct {
    int      model;         // TOK_SPM | TOK_BPE
    int      pre;           // TOK_PRE_* split rules (BPE only)
    int      n_vocab;
    gg_str  *tokens;        // borrowed from gguf kv
    float   *scores;        // SPM (may be NULL)
    int32_t *ttype;         // token type per id (may be NULL)
    int      bos_id, eos_id, unk_id;
    bool     add_bos, add_space_prefix;
    hmap     vocab;         // token string -> id
    hmap     merges;        // "left right" -> rank (BPE)
    char    *merges_buf;    // owned storage for merge keys
    // special (control/user-defined) tokens, sorted by length desc
    int     *special_ids;
    int      n_special;
    int      b2u[256];      // BPE byte -> codepoint
    int      u2b[512];      // BPE codepoint -> byte (-1 = none)
    // Set by an encode helper when it drops a text segment because a temporary
    // allocation failed. tok_encode resets it per call and returns -1 when set,
    // so an OOM is never mistaken for a legitimately shorter prompt.
    bool     encode_oom;
} tokenizer;

bool tokenizer_init(tokenizer *t, gguf_file *g);
void tokenizer_free(tokenizer *t);
// returns number of tokens written to out (capacity cap); add_bos per call
int  tok_encode(tokenizer *t, const char *text, int32_t *out, int cap,
                bool add_bos, bool parse_special);
// raw-byte encode without BOS/specials/segment normalization (see tokenizer.c)
int  tok_encode_raw(tokenizer *t, const char *text, int n,
                    int32_t *out, int cap);
// decode one token into buf (returns bytes written, no NUL); control tokens -> 0
int  tok_decode(tokenizer *t, int id, char *buf, int cap);
// decoded text of token id even if control (for stop-token matching); NULL if oob
const char *tok_raw(tokenizer *t, int id);
bool tok_is_control(tokenizer *t, int id);
int  tok_find(tokenizer *t, const char *s); // exact vocab lookup, -1 if absent

#endif // RUNNER_TOKENIZER_H
