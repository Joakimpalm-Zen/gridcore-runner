// runner — a compact GGUF inference engine (CPU only).
#ifndef RUNNER_H
#define RUNNER_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

// ---------------------------------------------------------------- fp16/bf16

typedef uint16_t f16_t;
extern float g_f16_table[65536];
void f16_init(void);

static inline float f16_to_f32(f16_t h) { return g_f16_table[h]; }

static inline float bf16_to_f32(uint16_t h) {
    union { uint32_t u; float f; } v;
    v.u = (uint32_t)h << 16;
    return v.f;
}

#if defined(__ARM_FP16_FORMAT_IEEE)
static inline f16_t f32_to_f16(float f) {
    __fp16 h = (__fp16)f;
    f16_t r;
    __builtin_memcpy(&r, &h, 2);
    return r;
}
static inline float f16_load(const f16_t *p) { return (float)*(const __fp16 *)p; }
#else
static inline f16_t f32_to_f16(float f) {
    union { float fl; uint32_t u; } v = { .fl = f };
    uint32_t sign = (v.u >> 16) & 0x8000;
    int32_t  exp  = (int32_t)((v.u >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = v.u & 0x7FFFFF;
    if (exp >= 31) return (f16_t)(sign | 0x7C00);           // inf/overflow
    if (exp <= 0) {                                          // subnormal/zero
        if (exp < -10) return (f16_t)sign;
        mant |= 0x800000;
        uint32_t shift = 14 - exp;
        uint32_t h = mant >> shift;
        if ((mant >> (shift - 1)) & 1) h++;                  // round
        return (f16_t)(sign | h);
    }
    uint32_t h = (uint32_t)(exp << 10) | (mant >> 13);
    if ((mant >> 12) & 1) h++;                               // round to nearest
    return (f16_t)(sign | h);
}
static inline float f16_load(const f16_t *p) { return g_f16_table[*p]; }
#endif

// ---------------------------------------------------------------- GGUF file

enum gguf_val_type {
    GGUF_T_U8 = 0, GGUF_T_I8, GGUF_T_U16, GGUF_T_I16, GGUF_T_U32, GGUF_T_I32,
    GGUF_T_F32, GGUF_T_BOOL, GGUF_T_STR, GGUF_T_ARR, GGUF_T_U64, GGUF_T_I64,
    GGUF_T_F64,
};

// tensor data types (subset of ggml)
enum ggml_type {
    T_F32 = 0, T_F16 = 1, T_Q4_0 = 2, T_Q4_1 = 3,
    T_Q5_0 = 6, T_Q5_1 = 7, T_Q8_0 = 8,
    T_Q2_K = 10, T_Q3_K = 11, T_Q4_K = 12, T_Q5_K = 13, T_Q6_K = 14,
    T_IQ4_NL = 20, T_IQ4_XS = 23,
    T_BF16 = 30,
};

typedef struct {
    uint64_t n;
    char    *s;             // NUL-terminated copy
} gg_str;

typedef struct {
    char       *key;
    uint32_t    type;       // gguf_val_type
    // scalar value (widened)
    union { uint64_t u64; int64_t i64; double f64; bool b; } v;
    gg_str      str;        // GGUF_T_STR
    // GGUF_T_ARR
    uint32_t    arr_type;
    uint64_t    arr_n;
    const void *arr_raw;    // scalar arrays: packed little-endian, points into mmap
    gg_str     *arr_str;    // string arrays: parsed copies
} gguf_kv;

typedef struct {
    char     name[128];
    uint32_t type;          // ggml_type
    uint32_t n_dims;
    uint64_t ne[4];         // ne[0] = row length (fastest dim)
    void    *data;
    uint64_t nbytes;
} gguf_tensor;

typedef struct {
    void       *map;
    size_t      map_size;
    uint32_t    version;
    uint64_t    n_tensors, n_kv;
    gguf_kv    *kv;
    gguf_tensor *tensors;
} gguf_file;

bool         gguf_open(gguf_file *g, const char *path);
void         gguf_close(gguf_file *g);
gguf_kv     *gguf_get(gguf_file *g, const char *key);
uint32_t     gguf_get_u32 (gguf_file *g, const char *key, uint32_t dflt);
float        gguf_get_f32 (gguf_file *g, const char *key, float dflt);
bool         gguf_get_bool(gguf_file *g, const char *key, bool dflt);
const char  *gguf_get_str (gguf_file *g, const char *key, const char *dflt);
gguf_tensor *gguf_find_tensor(gguf_file *g, const char *name);

// ---------------------------------------------------------------- quants

int         ggml_block_size(int type);   // elements per block
size_t      ggml_type_size(int type);    // bytes per block
const char *ggml_type_name(int type);
bool        ggml_type_supported(int type);
static inline size_t ggml_row_size(int type, int64_t n) {
    return (size_t)(n / ggml_block_size(type)) * ggml_type_size(type);
}

// dequantize a full row of n elements
void  dequant_row(int type, const void *src, float *dst, int n);
// dot(row, x) over n elements
float vec_dot(int type, const void *row, const float *x, int n);

// ---------------------------------------------------------------- threadpool

typedef void (*tp_fn)(void *ctx, int i0, int i1); // process items [i0, i1)
typedef struct tpool tpool;
tpool *tpool_create(int n_threads);
void   tpool_run(tpool *tp, tp_fn fn, void *ctx, int n_items);

// ---------------------------------------------------------------- tokenizer

enum { TOK_SPM, TOK_BPE };

typedef struct { const char *key; uint32_t klen; int32_t val; } hmap_ent;
typedef struct { hmap_ent *e; size_t cap; } hmap;

typedef struct {
    int      model;         // TOK_SPM | TOK_BPE
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
} tokenizer;

bool tokenizer_init(tokenizer *t, gguf_file *g);
// returns number of tokens written to out (capacity cap); add_bos per call
int  tok_encode(tokenizer *t, const char *text, int32_t *out, int cap,
                bool add_bos, bool parse_special);
// decode one token into buf (returns bytes written, no NUL); control tokens -> 0
int  tok_decode(tokenizer *t, int id, char *buf, int cap);
// decoded text of token id even if control (for stop-token matching); NULL if oob
const char *tok_raw(tokenizer *t, int id);
bool tok_is_control(tokenizer *t, int id);
int  tok_find(tokenizer *t, const char *s); // exact vocab lookup, -1 if absent

// ---------------------------------------------------------------- model

typedef struct {
    gguf_tensor *attn_norm, *wq, *wk, *wv, *wo;
    float       *bq, *bk, *bv, *bo;      // optional biases (f32, converted)
    gguf_tensor *ffn_norm, *w_gate, *w_up, *w_down;
    float       *attn_norm_w, *ffn_norm_w; // norm weights as f32
} layer_t;

typedef struct {
    gguf_file gf;
    char      arch[32];
    int       n_layer, n_embd, n_head, n_head_kv, head_dim, n_ff;
    int       n_vocab, n_ctx_train, rope_dim;
    bool      rope_neox;     // NeoX-style rotation (qwen2) vs adjacent pairs (llama)
    float     rms_eps, rope_base;
    float     rope_mscale;   // YaRN attention magnitude scale (1.0 = off)
    gguf_tensor *tok_embd;
    gguf_tensor *output;     // may equal tok_embd (tied)
    float       *out_norm_w;
    layer_t     *layers;
    float       *rope_inv_freq; // [rope_dim/2], scaling factors folded in
    // runtime state
    tpool *tp;               // worker pool used by this instance
    int    n_ctx, n_batch;
    f16_t *kcache, *vcache;  // [n_layer][n_ctx][kv_dim], fp16 to halve memory
    float *x, *xb, *xb2, *q, *hb, *hb2;   // [n_batch][dim] activations
    float *k_tmp, *v_tmp;                 // [n_batch][kv_dim]
    float *att, *logits;
    void  *gpu;              // GPU backend context (NULL = CPU only)
} model_t;

// ---------------------------------------------------------------- gpu

enum { GPU_AUTO = 0, GPU_OFF = 1 };
bool   gpu_available(char *name, int name_cap);
bool   gpu_init(model_t *m);                     // false = unsupported, use CPU
float *gpu_forward(model_t *m, int token, int pos); // NULL = failed, use CPU

typedef struct {
    int   gpu_mode;    // GPU_AUTO | GPU_OFF
    int   n_threads;   // worker threads for this instance (0 = 1)
    int   n_ctx;       // 0 = default (min(train ctx, 4096))
    int   n_batch;     // prompt batch size, 0 = default 64
    float rope_base;   // >0 overrides model rope theta
    float rope_scale;  // >0 forces linear rope scaling by this factor
    bool  verbose;
} model_params;

bool   model_load(model_t *m, const char *path, const model_params *p);
// process n tokens starting at position pos; returns logits of the last
// token if want_logits, else NULL
float *model_forward_batch(model_t *m, const int32_t *tokens, int n, int pos,
                           bool want_logits);
// single-token convenience wrapper
float *model_forward(model_t *m, int token, int pos);

// ---------------------------------------------------------------- sampler

typedef struct {
    float temp, top_p, repeat_penalty;
    int top_k;
    uint64_t rng;
    int32_t recent[256];
    int n_recent, recent_head;
} sampler;

// validity filter for constrained sampling; return true if token is allowed
typedef bool (*sample_ok_fn)(void *ud, int token);

void sampler_reset(sampler *s);
void sampler_accept(sampler *s, int tok);
// pick next token; ok==NULL means unconstrained; returns -1 if no token allowed
int  sample_pick(sampler *s, float *logits, int n_vocab, sample_ok_fn ok, void *ud);

// ---------------------------------------------------------------- json mode

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
// true if the machine stopped at a self-terminated value boundary (numbers)
bool jsonv_value_end(const jsonv *v);
// force-complete the object (token budget ran out); returns bytes written
int  jsonv_close(jsonv *v, char *out, int cap);

// ------------------------------------------------- schema-constrained output

enum sn_kind { SN_ANY, SN_NULL, SN_BOOL, SN_NUM, SN_INT, SN_STR, SN_ENUM,
               SN_OBJ, SN_ARR, SN_UNION };

typedef struct snode snode;
struct snode {
    int     kind;
    char  **lits; int n_lits;                       // enum literals (JSON text)
    char  **keys; int *key_len; snode **props;      // object properties
    bool   *req;  int n_props;                      //   (declared order)
    snode  *items; int min_items, max_items;        // array
    snode **alts; int n_alts;                       // type unions
};

struct jv;
snode *schema_compile(struct jv *schema, char *err, int errcap);
void   schema_free(snode *n);

// streaming validator state (memcpy-copyable for token lookahead)
typedef struct {
    const snode *node;
    uint8_t  phase, sub;
    int16_t  idx, lit_pos;
    uint64_t alive;
} sframe;

typedef struct {
    sframe stack[48];
    int    depth;
    bool   done;
    jsonv  any;      // generic submachine for open {} schema nodes
} sval;

void sval_init (sval *v, const snode *root);
bool sval_feed (sval *v, const char *s, int n);
int  sval_close(sval *v, char *out, int cap);

// ---------------------------------------------------------------- templates

enum { TMPL_CHATML, TMPL_LLAMA2, TMPL_LLAMA3, TMPL_ZEPHYR, TMPL_RAW };

typedef struct { const char *role, *content; } chat_msg;

int         template_detect(const char *meta_tmpl, tokenizer *tok);
int         template_from_name(const char *name); // -1 if unknown
const char *template_name(int tmpl);
// render messages; add_assistant appends the assistant generation prefix.
// returns bytes written (excl. NUL)
size_t render_messages(int tmpl, const chat_msg *msgs, int n_msgs,
                       bool add_assistant, char *out, size_t cap);

// ---------------------------------------------------------------- engine

// return nonzero to abort generation (e.g. client disconnected)
typedef int (*gen_cb)(void *ud, const char *bytes, int n);

typedef struct {
    model_t   *m;
    tokenizer *tok;
    sampler   *smp;
    int  pos;              // next free KV slot
    int  stop_ids[8];
    int  n_stop;
    bool ignore_eos;
    bool hit_stop;         // last generate ended on a stop token / json done
    bool json_mode;        // constrain output to a single JSON object
    const snode *schema;   // constrain output to a JSON schema (overrides json_mode)
    sval  sv;
    jsonv jv;
    bool progress;         // print prompt progress to stderr
} engine;

void   engine_init(engine *e, model_t *m, tokenizer *tok, sampler *smp);
void   engine_reset(engine *e); // clear KV position + sampler + json state
// feed tokens (batched); returns last-token logits or NULL on ctx overflow
float *engine_feed(engine *e, const int32_t *toks, int n);
// sample until stop/limit, streaming decoded bytes to cb; returns token count
int    engine_generate(engine *e, float *logits, int max_new,
                       gen_cb cb, void *ud, double *gen_time);
double now_s(void);

#endif // RUNNER_H
