// runner — a compact GGUF inference engine (CPU, CUDA, Metal).
#ifndef RUNNER_H
#define RUNNER_H

#define RUNNER_VERSION "0.1.2-alpha"

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <math.h>

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
// Requantize a GGUF at in_path to out_path (written beside + atomically
// renamed). Returns 0 on success, nonzero on failure with the destination
// left untouched. Declared here so tests can drive it without main.c.
int   quantize_gguf(const char *in_path, const char *out_path, int target);
// dot(row, x) over n elements
float vec_dot(int type, const void *row, const float *x, int n);
// out[b] = dot(w, x + b*x_stride) for nb columns sharing one weight row
void  vec_dot_f32_multi(const float *w, const float *x, int x_stride,
                        int nb, int n, float *out);
void  q8_quant_row(const float *x, void *dst, int n); // n % 32 == 0
void  q8_accum_row(const void *src, float a, float *out, int n);

// ---------------------------------------------------------------- threadpool

typedef void (*tp_fn)(void *ctx, int i0, int i1); // process items [i0, i1)
typedef struct tpool tpool;
tpool *tpool_create(int n_threads);
void   tpool_run(tpool *tp, tp_fn fn, void *ctx, int n_items);
void   tpool_destroy(tpool *tp);
int    tpool_size(const tpool *tp);  // workers incl. the calling thread

// ---------------------------------------------------------------- tokenizer

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
    // sparse-MoE FFN (Mixtral / Qwen3-MoE): a router picks expert_used of
    // expert_count experts, each a SwiGLU FFN, weighted-summed. When is_moe is
    // set these replace w_gate/w_up/w_down for this layer.
    bool         is_moe;
    gguf_tensor *ffn_gate_inp;   // router: n_embd -> n_expert logits (F32)
    // modern fused 3D expert tensors ({.., n_expert}); NULL when split
    gguf_tensor *ffn_gate_exps;  // 3D {n_embd, n_ff, n_expert}
    gguf_tensor *ffn_up_exps;    // 3D {n_embd, n_ff, n_expert}
    gguf_tensor *ffn_down_exps;  // 3D {n_ff, n_embd, n_expert}
    // legacy split layout: one 2D tensor per expert (older Mixtral GGUFs)
    bool         moe_split;
    gguf_tensor **moe_g, **moe_u, **moe_d;  // [n_expert] each, when moe_split
    float       *attn_norm_w, *ffn_norm_w; // norm weights as f32
    float       *qnorm_w, *knorm_w;      // per-head Q/K norms (qwen3, gemma3/4)
    float       *post_attn_norm_w, *post_ffn_norm_w; // gemma sandwich norms
    float        out_scale;              // whole-layer output scalar (gemma4; 1.0 = off)
    // Qwen3.5 hybrid layers. Full-attention layers keep using the fields
    // above; recurrent layers use this compact Gated DeltaNet weight set.
    bool         recurrent;
    gguf_tensor *wqkv, *wq_gate, *ssm_conv, *ssm_beta, *ssm_alpha, *ssm_out;
    float       *ssm_dt, *ssm_a, *ssm_norm_w;
} layer_t;

// Return expert `e`'s weight for one MoE FFN projection (which: 0=gate, 1=up,
// 2=down) as a 2D tensor, handling both the fused 3D layout (a slice) and the
// legacy split-per-expert layout. Shared by the CPU and GPU MoE forward paths.
gguf_tensor moe_expert_weight(const layer_t *ly, int which, int e,
                              int n_embd, int n_ff_exp);

// One model_t is one *sequence*: a set of weights plus the mutable state
// needed to decode one stream against them. The fields below are grouped by
// which half of that they belong to, because concurrent serving turns the
// distinction into a memory bill:
//
//   IMMUTABLE (weight side) — derived from the GGUF and never written after
//   model_load returns. Two model_t values loaded from the same file with the
//   same parameters hold bit-identical copies of all of it, so it is safe to
//   share. The CUDA backend already does: gpu_init keys a refcounted device
//   registry on the file identity and geometry below, so `--parallel N`
//   uploads the weights once rather than N times (see src/cuda.c).
//
//   PER-SEQUENCE (state side) — written by every forward pass. Never share:
//   two sequences writing one KV cache is silent cross-contamination.
//
// The struct itself is not yet split into two types. Doing that changes the
// public interface every caller uses (server slots, model swap, speculative
// draft models), so the sharing was pushed into the backend first, where the
// duplication actually cost gigabytes.
typedef struct {
    // ---- immutable: file and geometry ----
    gguf_file gf;
    char     *path;          // owned copy of the load path (shared-weight key)
    char      arch[32];
    int       n_layer, n_embd, n_head, n_head_kv, head_dim, n_ff;
    int       n_vocab, n_ctx_train, rope_dim;
    // sparse-MoE (0 = dense model). n_ff_exp is the per-expert FFN width.
    int       n_expert, n_expert_used, n_ff_exp;
    float    *moe_logits;  // [n_expert] router scratch (forward, single token)
    float    *moe_gate;    // [n_ff_exp]
    float    *moe_up;      // [n_ff_exp]
    float    *moe_dexp;    // [n_embd] one expert's down output
    float    *moe_out;     // [n_embd] weighted expert-sum accumulator
    bool      rope_neox;     // NeoX-style rotation (qwen2) vs adjacent pairs (llama)
    float     rms_eps, rope_base;
    float     rope_mscale;   // YaRN attention magnitude scale (1.0 = off)
    float     embd_scale;    // token embedding multiplier (gemma: sqrt(n_embd))
    int       swa_window;    // sliding-window size for local layers (0 = none)
    // per-layer geometry overrides (NULL = uniform, use the scalars above);
    // heterogeneous archs (gemma4) vary kv heads / head dim / rope dim per layer
    int      *l_head_kv;     // [n_layer] kv heads per layer
    int      *l_head_dim;    // [n_layer] head dim per layer (K == V required)
    int      *l_rope_dim;    // [n_layer] rotated dims per layer
    bool     *l_is_swa;      // [n_layer] sliding-window layer flags
    size_t   *kv_off;        // [n_layer+1] element offsets into kcache/vcache
    float     attn_scale;    // 0 = default 1/sqrt(head_dim(l)), else fixed
    int       ffn_act;       // ACT_SILU (default) or ACT_GELU (gemma)
    bool      v_rmsnorm;     // weightless per-head RMS norm on V (gemma4)
    bool      qwen35;
    int       full_attn_interval;
    int       ssm_conv_kernel, ssm_inner, ssm_state, ssm_v_heads, ssm_groups;
    float     logit_softcap; // final logits = c*tanh(x/c) when > 0
    int32_t  *suppress;      // token ids forced to -inf in the logits
    int       n_suppress;    // (tokenizer.ggml.suppress_tokens)
    const char *think_open, *think_close; // architecture thinking-tag pair, or NULL
    gguf_tensor *tok_embd;
    gguf_tensor *output;     // may equal tok_embd (tied)
    float       *out_norm_w;
    layer_t     *layers;
    // phi3 fuses Q/K/V and the FFN gate/up into single tensors; these are the
    // sliced descriptors the layers point at, owned here so they outlive init
    gguf_tensor *fused_splits;
    float       *rope_inv_freq; // [rope_dim/2], scaling factors folded in
    float       *rope_inv_freq_local; // sliding-window layers, own base, unscaled
    int          rope_dim_local;      // rotated dims on sliding layers
    // ---- per-sequence: mutable state, one set per decoding stream ----
    tpool *tp;               // worker pool used by this instance
    int    n_ctx, n_batch;
    f16_t *kcache, *vcache;  // [n_layer][n_ctx][kv_dim], fp16 (or q8_0 blocks
                             // when kv_q8 — treated as raw bytes then)
    bool   kv_q8;            // KV rows stored as q8_0 blocks (CPU and CUDA)
    float *x, *xb, *xb2, *q, *hb, *hb2;   // [n_batch][dim] activations
    float *k_tmp, *v_tmp;                 // [n_batch][kv_dim]
    float *q_gate;                        // qwen35 full-attention output gate
    float *ssm_qkv, *ssm_z, *ssm_aux;     // qwen35 recurrent scratch
    float *ssm_cw;                         // qwen35 dequantized conv-kernel row
    float *ssm_conv_state, *ssm_state_mem; // qwen35 per-sequence recurrent state
    float *att, *logits;
    float *all_logits;       // lazy [spec_batch][n_vocab] (speculative verify)
    int    spec_batch;       // rows all_logits can hold
    int    reserve_vram_pct; // VRAM cap for the GPU backend (0 = free VRAM)
    int    gpu_layers;       // leading layers run on GPU (n_layer = full,
                             // <n_layer = partial offload, CPU finishes the rest).
                             // Decided by the first instance to upload a given
                             // file and reused by every instance sharing it, so
                             // parallel slots cannot end up on different splits.
    void  *gpu;              // GPU backend context (NULL = CPU only). Per
                             // sequence: it owns this stream's KV and
                             // activations and points at the shared weights.
    struct vram_lease *vram; // this instance's entry in the cross-process VRAM
                             // registry (NULL on CPU-only runs, which are never
                             // accounted and never refused)
} model_t;

// ------------------------------------------------- vram reservation registry
//
// A cross-process ledger of who is holding GPU memory, keyed by GPU device
// identity, living in $XDG_RUNTIME_DIR (else the temp dir). It is NOT a
// single-instance lock: several runners on one GPU are legitimate and the test
// suite depends on it. It exists so that when memory does run out, the refusal
// can name the holders instead of reporting a bare out-of-memory — and so that
// a runner killed with SIGKILL cannot leave a reservation poisoning the box.
//
// Entries owned by dead pids are reaped on every claim. CPU-only runs never
// claim, and so are never refused.
typedef struct vram_lease vram_lease;

// how much the device reports free, right now; the registry re-reads it while
// waiting. Callers on a real GPU pass a gpu_mem_info wrapper.
typedef uint64_t (*vram_free_fn)(void *ud);

// What the ledger looked like at the moment of the decision. `holders` is the
// count of *other* live runners registered on this GPU; a shortfall with zero
// holders is memory held by something outside runner's accounting, which is not
// a reason to refuse — there would be nobody to name and no orphan to blame.
typedef struct {
    int      holders;      // other live runners registered on this GPU
    uint64_t held_bytes;   // what they hold between them
    uint64_t available;    // free VRAM, less claims not yet allocated
} vram_status;

// Claim `need_bytes` on the GPU named by `gpu_id` (a UUID where the backend can
// supply one — a MIG slice must not share a ledger with its parent card).
//
// Returns a lease, or NULL when the request does not fit, in which case `err`
// holds a message naming every live holder: pid, model, bytes, uptime.
// wait_secs > 0 queues and retries until the deadline instead of refusing.
// `cancel` (nullable) aborts the queue wait early when *cancel goes nonzero.
// `st` (nullable) reports the ledger state behind the decision.
vram_lease *vram_claim(const char *gpu_id, const char *model_path,
                       uint64_t need_bytes, vram_free_fn free_fn, void *free_ud,
                       int wait_secs, const _Atomic int *cancel,
                       vram_status *st, char *err, size_t err_cap);

// The allocation happened: `actual_bytes` is now real device memory, visible to
// the driver's free figure, and stops being subtracted from it.
void        vram_commit(vram_lease *l, uint64_t actual_bytes);
// Clean exit. Unclean exits are covered by dead-pid reaping instead.
void        vram_release(vram_lease *l);

// ---------------------------------------------------------------- gpu

enum { GPU_AUTO = 0, GPU_OFF = 1 };
bool   gpu_available(char *name, int name_cap);
// Stable identity of the GPU this backend would use, preferring a UUID: on a
// MIG box the parent card's UUID is the same for every slice, so an identity
// that cannot tell slices apart is worse than useless for VRAM accounting.
// Falls back to a bus id, then to "cuda:0". false when there is no GPU.
bool   gpu_device_id(char *id, int id_cap);
// dedicated GPU memory in bytes; false when there is no discrete GPU
// (Metal's unified memory is governed by the RAM reservation instead)
bool   gpu_mem_info(size_t *free_bytes, size_t *total_bytes);
// does this backend's attention read a q8_0 KV cache? The CPU path always
// can; a backend that cannot forces the cache back to f16 rather than
// handing q8_0 blocks to kernels that would read them as fp16.
bool   gpu_kv_q8_ok(void);
bool   gpu_init(model_t *m);                     // false = unsupported, use CPU
// process n tokens starting at pos (prompt batches); on success returns true
// and sets *logits to the last token's logits when want_logits (else NULL).
// false = failed, use CPU.
bool   gpu_forward_batch(model_t *m, const int32_t *tokens, int n, int pos,
                         bool want_logits, float **logits);
void   gpu_free(model_t *m); // releases GPU buffers; KV pointers become invalid
// Stop using the GPU for this model after a runtime failure, releasing
// whatever the backend can release while leaving the model usable on the CPU.
// Distinct from gpu_free: a backend whose KV cache lives in GPU-owned memory
// (Metal's unified buffers) cannot free it here, because the CPU path is about
// to read those very rows. CUDA can and does — the host KV copy is
// authoritative there — so a slot that falls back mid-run hands its VRAM, and
// its reference to the shared weights, straight back.
void   gpu_disable(model_t *m);

// ------------------------------------------- batched decode (backend half)
//
// One decode step for several independent sequences in a single microbatch.
// Every sequence keeps its own KV cache, its own position and its own logits;
// what is shared is the *work*, because a decode step is weight-bandwidth
// bound and reading the weights once for N tokens costs barely more than
// reading them once for one.
//
// The contract that makes this usable is numerical, not structural: a batched
// step must produce, for each sequence, the bits a lone step would have
// produced. Nothing here reduces across sequences, and the matvecs pick the
// multi-column twin of whatever kernel the batch-1 path would have used, so
// that holds by construction rather than by luck. See gpu_batch_decode.
typedef struct gpu_batch gpu_batch;
// Group n sequences loaded from one file into a reusable microbatch context.
// NULL = this backend/model cannot batch, and the caller decodes one by one.
gpu_batch *gpu_batch_create(model_t **seqs, int n);
void       gpu_batch_free(gpu_batch *b);
// Evaluate one token for each of the n sequences named by idx (indices into
// the create() array). tok[i]/pos[i] are that sequence's token and KV write
// position; out[i] receives its logits, owned by the batch and valid until
// the next call. false = nothing was evaluated, decode these sequences singly.
bool       gpu_batch_decode(gpu_batch *b, const int *idx, const int32_t *tok,
                            const int *pos, int n, float **out);

typedef struct {
    int   gpu_mode;    // GPU_AUTO | GPU_OFF
    int   n_threads;   // worker threads for this instance (0 = 1)
    // When the count above came from the small "GPU does the math" default
    // and the architecture turns out to be CPU-forced (the recurrent qwen3.5
    // path has no GPU kernels), model_load raises the pool to this cap
    // instead. 0 = the user pinned -t or a CPU reservation; never raise.
    int   cpu_fallback_threads;
    int   n_ctx;       // 0 = default (min(train ctx, 4096)), or reservation
                       // auto-fit when a reservation is set
    int   n_batch;     // prompt batch size, 0 = default 64
    float rope_base;   // >0 overrides model rope theta
    float rope_scale;  // >0 forces linear rope scaling by this factor
    bool  verbose;
    // resource reservation: percentage of *total* VRAM / RAM this instance
    // may use (0 = unmanaged). With -c 0, the context window is sized to
    // fill whatever the reservation leaves after the weights — small models
    // grow their context into the reserved room, capped at the train ctx.
    int   reserve_vram_pct;
    int   reserve_ram_pct;
    // --wait-for-vram: seconds to queue behind other registered runners rather
    // than refusing outright. 0 = refuse immediately (the default).
    int   vram_wait_secs;
    // Load cancellation: when non-NULL and it becomes nonzero, a load queued in
    // the --wait-for-vram retry loop gives up promptly and the load fails. A
    // lock-free atomic, read across threads (RNR-008). The serving layer points
    // this at its unload/shutdown flag; standalone loads leave it NULL.
    const _Atomic int *load_cancel;
    // store the KV cache as q8_0 instead of fp16: halves cache bytes, so it
    // roughly doubles the context that fits a given VRAM/RAM reservation.
    // Lossy — output is NOT token-identical to an fp16 cache — so f16 stays
    // the default. Requires every layer's head_dim to be a multiple of 32.
    bool  kv_q8;
} model_params;

bool   model_load(model_t *m, const char *path, const model_params *p);
void   model_free(model_t *m); // frees everything incl. GPU context and mmap
// process n tokens starting at position pos; returns logits of the last
// token if want_logits, else NULL
float *model_forward_batch(model_t *m, const int32_t *tokens, int n, int pos,
                           bool want_logits);
// single-token convenience wrapper
float *model_forward(model_t *m, int token, int pos);
// speculative verify: forward a small batch keeping hidden states, then pull
// each row's logits lazily (false/NULL on full GPU offload or n > spec_batch)
bool   model_forward_batch_keep(model_t *m, const int32_t *tokens, int n, int pos);
float *model_spec_row_logits(model_t *m, int b);
// mean-pooled L2-normalized embedding of toks; clobbers KV slots [0, n)
bool   model_embed(model_t *m, const int32_t *toks, int n, float *out);

// ------------------------------------------------ continuous batching (Phase 6)
//
// The decode primitive continuous batching is built on: advance N independent
// sequences by exactly one token each, in one pass over the weights.
//
// This is deliberately *only* the primitive. Deciding which sequences are
// ready, when to cut a batch, how to admit and evict them, and what to do with
// the logits afterwards is scheduling, and scheduling belongs to the server.
// What this owns is the part the server cannot express: the microbatch itself,
// and the guarantee that being in one changes nothing about the answer.
//
//   model_batch *b = model_batch_create(slots, n_slots);
//   ...
//   model_batch_decode(b, idx, tok, pos, n_ready, logits);   // once per step
//
// Sequences are the same model_t values the server already owns, so they keep
// their own KV cache, position, sampler and schema state, and a sequence can
// leave or rejoin a batch between steps at no cost — it is named per call, not
// enrolled. A sequence that is prefilling, cancelled, or not yet ready simply
// is not listed that step.
typedef struct model_batch model_batch;

// Largest number of sequences one microbatch evaluates at once. Longer lists
// are split into consecutive microbatches, so this is a performance boundary,
// not a limit on the caller.
//
// It is the backends' token-tile width: the activation scratch is already this
// many columns wide and the multi-column matvec kernels unroll over it, so a
// microbatch of this size reuses machinery that exists rather than adding any.
#define MODEL_BATCH_MAX 8
int model_batch_max(void);

// Group sequences loaded from the same file with the same parameters into a
// batch context. Never fails in a way the caller must handle: with no batching
// backend the context still works and decodes sequentially, so the scheduler
// above it needs no backend-specific branch.
model_batch *model_batch_create(model_t **seqs, int n);
void         model_batch_free(model_batch *b);

// Advance the n sequences named by idx (indices into the create() array) by
// one token each. tok[i] is the token to evaluate for that sequence and pos[i]
// the KV position it occupies; out[i] receives that sequence's logits, valid
// until the next call.
//
// Guarantee: out[i] is bit-identical to what model_forward(seq, tok[i], pos[i])
// would have returned. Batching is a throughput decision and never a numerical
// one — see tests/test_batch.c, which is that claim as a test.
//
// Returns false only if a sequence could not be evaluated at all.
bool model_batch_decode(model_batch *b, const int *idx, const int32_t *tok,
                        const int *pos, int n, float **out);

// ---------------------------------------------------------------- sampler

typedef struct {
    float temp, top_p, min_p, repeat_penalty;
    int top_k;
    uint64_t rng;
    int32_t recent[256];
    int n_recent, recent_head;
    // Ids exempt from the repeat penalty. A chat template puts its own turn
    // terminator in the prompt, and the prompt seeds the penalty window, so
    // penalising terminators can stop a model ever ending its turn.
    int32_t no_penalty[12];
    int n_no_penalty;
} sampler;

// validity filter for constrained sampling; return true if token is allowed
typedef bool (*sample_ok_fn)(void *ud, int token);

void sampler_reset(sampler *s);
void sampler_accept(sampler *s, int tok);
// Pick next token; ok==NULL means unconstrained; returns -1 if no token allowed.
//
// Contract: temp <= 0 returns the model's argmax (the first *valid* token in
// argmax order when constrained). The repeat penalty is a diversity knob and
// is not applied there — greedy decoding is a determinism request, and a
// caller who wants the most likely token does not want a different one back.
// Returns the chosen token id, -1 when no candidate is valid (a clean stop),
// or -2 on an allocation failure (the caller must treat this as an error, not
// a stop).
int  sample_pick(sampler *s, float *logits, int n_vocab, sample_ok_fn ok, void *ud);

// -------------------------------------------------- per-family sampling presets
//
// Model families publish recommended sampling settings and they differ a lot
// (Gemma 3 wants temp 1.0, SmolLM2 wants 0.2). One fixed set of defaults is
// wrong for most models, so runner picks a preset from the GGUF's
// `general.architecture` and `general.name`. Presets are visible rather than
// magic: logged at load, listed by --caps, reported by /v1/capabilities.

typedef struct {
    const char *name;    // preset id ("qwen3", "llama3", ..., "generic")
    const char *source;  // provenance of these numbers, printed by --caps
    float temp, top_p, min_p, repeat_penalty;
    int   top_k;
} sampler_preset;

// Values the caller set explicitly on the CLI. An explicit value always beats
// the preset, including an explicit zero.
typedef struct {
    bool  has_temp, has_top_p, has_min_p, has_top_k, has_repeat_penalty;
    float temp, top_p, min_p, repeat_penalty;
    int   top_k;
} sampler_override;

// Preset for a model. Never NULL: an unrecognised model gets "generic".
// Either string may be NULL.
const sampler_preset *sampler_preset_for(const char *arch, const char *name);
// Enumerate the preset table; NULL past the end.
const sampler_preset *sampler_preset_at(int i);
// Apply preset then overrides to s. Touches only the five sampling knobs —
// rng state and the penalty window are left alone — and returns the preset
// used, so callers can report it.
const sampler_preset *sampler_resolve(sampler *s, const char *arch,
                                      const char *name,
                                      const sampler_override *ov);
// One-line human summary, e.g.
// "gemma3 (temp 1.00, top_p 0.95, top_k 64, min_p 0.00, repeat_penalty 1.10)"
void sampler_describe(const sampler *s, const sampler_preset *p,
                      char *buf, size_t cap);

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
    int16_t  idx;
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
int  sval_close(sval *v, char *out, int cap);

// ---------------------------------------------------------------- templates

// MISTRAL is LLAMA2's [INST] framing without the <<SYS>> block: Mistral's own
// template rejects a system role outright, so the text is folded into the
// first user turn rather than wrapped in markers it never saw in training.
enum { TMPL_CHATML, TMPL_LLAMA2, TMPL_LLAMA3, TMPL_ZEPHYR, TMPL_GEMMA,
       TMPL_GEMMA4, TMPL_MISTRAL, TMPL_PHI3, TMPL_APERTUS, TMPL_ORNITH,
       TMPL_RAW };

enum { ACT_SILU = 0, ACT_GELU = 1 };

// per-layer geometry accessors: uniform models keep the scalars, heterogeneous
// archs (gemma4) override per layer
static inline int model_head_dim(const model_t *m, int l) {
    return m->l_head_dim ? m->l_head_dim[l] : m->head_dim;
}
static inline int model_n_head_kv(const model_t *m, int l) {
    return m->l_head_kv ? m->l_head_kv[l] : m->n_head_kv;
}
static inline int model_kv_dim(const model_t *m, int l) {
    return model_n_head_kv(m, l) * model_head_dim(m, l);
}
static inline int model_q_dim(const model_t *m, int l) {
    return m->n_head * model_head_dim(m, l);
}
static inline int model_rope_dim(const model_t *m, int l) {
    return m->l_rope_dim ? m->l_rope_dim[l] : m->rope_dim;
}
static inline bool model_is_swa(const model_t *m, int l) {
    return m->l_is_swa != NULL && m->l_is_swa[l];
}
static inline float model_attn_scale(const model_t *m, int l) {
    return m->attn_scale > 0 ? m->attn_scale
                             : 1.0f / sqrtf((float)model_head_dim(m, l));
}
// bytes per cached KV row / per layer start, honoring the storage format.
// q8_0 packs each 32 values into a 34-byte block; kv_dim is always a
// multiple of 32 when kv_q8 is enabled (checked at load)
static inline size_t model_kv_row_bytes(const model_t *m, int l) {
    int d = model_kv_dim(m, l);
    return m->kv_q8 ? (size_t)(d / 32) * 34 : (size_t)d * sizeof(f16_t);
}
static inline size_t model_kv_byte_off(const model_t *m, int l) {
    return m->kv_q8 ? m->kv_off[l] / 32 * 34 : m->kv_off[l] * sizeof(f16_t);
}

typedef struct { const char *role, *content; } chat_msg;

int         template_detect(const char *meta_tmpl, tokenizer *tok);
int         template_from_name(const char *name); // -1 if unknown
const char *template_name(int tmpl);
// render messages; add_assistant appends the assistant generation prefix.
// returns bytes written (excl. NUL)
size_t render_messages(int tmpl, const chat_msg *msgs, int n_msgs,
                       bool add_assistant, char *out, size_t cap);

// chat tool-call convention (template.c; sbuf/jv live in json.h)
struct sbuf;
struct jv;
// render OpenAI "tools" declarations as a system turn (no-op when absent)
void tools_render(const struct jv *tools, struct sbuf *out);
void tools_render_for(int tmpl, const struct jv *tools, struct sbuf *out);
void tool_history_render_for(int tmpl, const struct jv *calls, struct sbuf *out);
// parse tool-call blocks out of content into OpenAI tool_calls items;
// returns the call count, content is compacted in place
int  tool_calls_parse(struct sbuf *content, struct sbuf *tc);
int  tool_calls_parse_for(int tmpl, struct sbuf *content, struct sbuf *tc);

// ------------------------------------------- strict tool-call envelope
//
// OpenAI tools[] compiled into a discriminated union — one branch per tool
// plus a `final` branch for an ordinary or schema-constrained answer:
//
//   {"tool":"get_weather","args":{"city":"Oslo"}}
//   {"tool":"final","args":{"content":"it is cold"}}
//
// Constraining sampling to that union is what makes the guarantee: the model
// cannot name a tool that was not declared, cannot invent an argument key or
// get its type wrong, and a max_tokens truncation closes to a document that
// is still a legal call. That replaces parsing hopeful output afterward,
// which is what tool_calls_parse above does and remains as the fallback for
// requests that do not opt in.
enum { TCH_AUTO, TCH_REQUIRED, TCH_NONE, TCH_NAMED };

typedef struct {
    int   kind;           // TCH_*
    char *schema_src;     // envelope JSON schema, for schema_compile (owned)
    char *system_turn;    // system message teaching the envelope (owned)
    bool  final_is_text;  // final branch is {"content": "..."} rather than
                          // the caller's own response_format schema
} tool_envelope;

// Build the envelope for one request. `final_schema` is the caller's
// response_format schema, or NULL for a plain text answer. Returns
//    1  strict mode applies; free with tool_envelope_free
//    0  strict mode does not apply (no tools, or tool_choice "none")
//   -1  malformed tools / tool_choice; err holds the client-facing reason
int  tool_envelope_build(struct jv *tools, struct jv *tool_choice,
                         struct jv *final_schema, tool_envelope *out,
                         char *err, int errcap);
void tool_envelope_free(tool_envelope *e);

// Map a generated envelope document back to the OpenAI response shape.
// Returns 1 for a tool call (one tool_calls[] item appended to tc), 0 for
// the final branch (assistant content appended to content), -1 when doc is
// not a well-formed envelope.
int  tool_envelope_map(const tool_envelope *e, const char *doc, size_t n,
                       struct sbuf *content, struct sbuf *tc);

// Streaming counterpart of tool_envelope_map.
//
// tool_envelope_map needs the whole document, which is exactly what a stream
// does not have: the envelope is decided one token at a time, and the client
// must see the decision as it happens rather than after the fact. This is the
// same mapping run incrementally — bytes in, demultiplexed events out — so a
// streamed request reaches the same call as a buffered one.
//
// Nothing is emitted until the branch is known, which is what keeps envelope
// syntax out of the client's `content`: the discriminator is buffered, and by
// the time anything is forwarded it is already known to be either assistant
// text or tool arguments. Argument text is forwarded raw (insignificant
// whitespace removed) so it stays a JSON *string* the caller can execute;
// `final` text is unescaped, matching what the buffered path hands back.
//
// Every callback returns non-zero to stop generation (the client went away);
// that result propagates out of tool_stream_feed unchanged.
typedef struct {
    void *ud;
    int (*content)(void *ud, const char *bytes, int n);
    int (*call_begin)(void *ud, const char *name);
    int (*call_args)(void *ud, const char *bytes, int n);
} tool_stream_sink;

typedef struct {
    const tool_envelope *env;
    tool_stream_sink     sink;
    int   state;
    char  *head;          // undecided prefix, held back from the client
    size_t head_n, head_cap;
    char *name;           // selected branch, once known (owned)
    bool  called;         // a tool branch, rather than `final`, was selected
    int   depth;          // nesting inside the value being forwarded
    bool  started;        // the forwarded value has produced its first byte
    bool  in_str, esc;    // JSON string state within that value
    char  pend[16];       // partial escape sequence awaiting more bytes
    int   n_pend;
} tool_stream;

void tool_stream_init(tool_stream *s, const tool_envelope *e,
                      const tool_stream_sink *sink);
int  tool_stream_feed(tool_stream *s, const char *bytes, int n);
// true once a tool branch was selected, i.e. finish_reason is "tool_calls"
bool tool_stream_called(const tool_stream *s);
void tool_stream_free(tool_stream *s);

// streaming splitter for thinking-tag models: bytes between
// open and close tags, including architectures that interleave them with
// plain text — reach the callback as reasoning (reasoning=1), the rest as
// content (reasoning=0). Partial tags at chunk boundaries are held back until
// they resolve. With open == NULL every byte passes straight through.
typedef int (*think_cb)(void *ud, int reasoning, const char *bytes, int n);
typedef struct {
    const char *open, *close;
    int   state;
    char *buf;
    int   n, cap;
} think_split;
void think_init(think_split *t, const char *open, const char *close);
void think_init_reasoning(think_split *t, const char *open, const char *close);
int  think_feed(think_split *t, const char *bytes, int n, think_cb cb, void *ud);
int  think_finish(think_split *t, think_cb cb, void *ud); // flush held bytes
void think_free(think_split *t);

// ---------------------------------------------------------------- engine

// return nonzero to abort generation (e.g. client disconnected)
typedef int (*gen_cb)(void *ud, const char *bytes, int n);

typedef struct { int32_t id; float lp; } lp_alt; // logprob alternative

typedef struct {
    model_t   *m;
    tokenizer *tok;
    sampler   *smp;
    int  pos;              // next free KV slot
    int  stop_ids[12];
    int  n_stop;
    bool ignore_eos;
    bool hit_stop;         // last generate ended on a stop token / json done
    bool oom;              // generation aborted on an allocation failure — the
                           // finish reason is "error", never a silent "stop"
    bool json_mode;        // constrain output to a single JSON object
    const snode *schema;   // constrain output to a JSON schema (overrides json_mode)
    // Constrained mode hides the thinking prelude from the output callback by
    // default (a raw completion's contract is "constrained payload only").
    // Chat serving sets this so the prelude reaches the callback too, where
    // the server's thinking-tag splitter routes it to the reasoning channel —
    // without it, a thinking model's schema call that exhausts max_tokens
    // mid-think returns empty content AND empty reasoning: undiagnosable.
    bool emit_think_prelude;
    sval  sv;
    jsonv jv;
    // constrained thinking-tag prelude: probe for think_open, sample freely
    // through think_close, then enforce sv/jv and emit only the payload
    uint8_t constraint_phase;
    bool    constraint_tag_possible;
    int     constraint_tag_match, constraint_close_match;
    bool progress;         // print prompt progress to stderr
    int32_t *hist;         // tokens whose KV occupies slots [0, pos)
    // optional per-token logprob capture (server "logprobs"): caller points
    // these at buffers sized [lp_cap] / [lp_cap * lp_n] before generating
    float   *lp_chosen;    // chosen-token logprob per emitted token
    int32_t *lp_ids;       // chosen token id per emitted token
    lp_alt  *lp_top;       // top-N alternatives per token
    int      lp_n, lp_cap, lp_count;
    // speculative decoding: a small draft model proposes draft_k tokens per
    // round, the target verifies them in one batched forward
    model_t *dm;           // draft model (NULL = off)
    int      dpos;         // draft KV position (may trail pos)
    int      draft_k;      // drafts per round
    // in-flight generation budget, owned by engine_gen_begin/step/end
    int      gen_max, gen_count;
    double   gen_t0;
    // identity of everything that decides what this engine's KV bytes mean:
    // the weights, the geometry, the tokenizer and the cache element type.
    // Computed once by engine_init; see engine_prefix_reuse.
    uint64_t model_key;
} engine;

// Returns false if the per-context history buffer could not be allocated;
// the caller must not use the engine in that case.
bool   engine_init(engine *e, model_t *m, tokenizer *tok, sampler *smp);
void   engine_reset(engine *e); // clear KV position + sampler + json state
void   engine_think_started(engine *e); // prompt already contains think_open
// keep the KV for the longest common prefix of hist and toks, reset the rest
// of the engine state; returns how many prompt tokens can be skipped
int    engine_rewind(engine *e, const int32_t *toks, int n);
// feed tokens (batched); returns last-token logits or NULL on ctx overflow
float *engine_feed(engine *e, const int32_t *toks, int n);
// sample until stop/limit, streaming decoded bytes to cb; returns token count
int    engine_generate(engine *e, float *logits, int max_new,
                       gen_cb cb, void *ud, double *gen_time);

// The same generation, one step at a time, for a caller that owns the forward.
//
// Continuous batching needs the forward hoisted out of the loop: one thread
// must issue every model_batch_decode, while the sampler, schema validator,
// stop check, logprob capture and streaming callback stay per-sequence and run
// wherever the caller likes. Splitting engine_generate at that seam — rather
// than writing a second batched generation loop — is what keeps a batched
// request's output identical to a solo one by construction.
//
//   engine_gen_begin(e, max_new);
//   while (engine_gen_step(e, logits, cb, ud, &tok, &pos) == ENGINE_STEP_MORE)
//       logits = <forward tok at pos, batched or not>;
//   n = engine_gen_end(e, cb, ud, &secs);
//
// A caller that abandons the loop early (deadline, cancellation) must still
// call engine_gen_end: that is where a truncated constrained document is
// closed to something valid.
enum { ENGINE_STEP_DONE = 0, ENGINE_STEP_MORE = 1 };
void   engine_gen_begin(engine *e, int max_new);
int    engine_gen_step(engine *e, const float *logits, gen_cb cb, void *ud,
                       int32_t *next_tok, int *next_pos);
int    engine_gen_end(engine *e, gen_cb cb, void *ud, double *gen_time);
// load a draft model for speculative decoding (shared by CLI and server);
// see engine.c for the gates. NULL = speculation could not be enabled.
model_t *spec_draft_load(const char *path, const model_t *target,
                         const model_params *mp);
double now_s(void);

// ------------------------------------------- shared, forkable KV prefixes
//
// engine_rewind reuses the KV a slot happens to still hold from its own last
// request. This is the same idea made durable and shareable: a completed
// prefix is snapshotted out of a slot's KV cache into host memory, and any
// slot serving a compatible model can fork it back in. Agent traffic is what
// pays for it — every request in a session repeats the same system prompt,
// tool list and schema verbatim, and that block is usually most of the prompt.
//
// Two calls, in the order a request meets them:
//
//   prefix_reuse r = engine_prefix_reuse(e, toks, n);   // before prefill
//   logits = engine_feed(e, toks + r.keep, n - r.keep);
//   engine_prefix_publish(e, toks, n, r.keep, prefill_seconds); // after
//
// There is deliberately no "look up" / "install" pair: installing a prefix
// that does not match the prompt is the one failure mode that produces a
// plausible wrong answer instead of an error, so the decision and the install
// are one operation that never sees a token vector it did not itself compare.
typedef struct {
    int    keep;     // prompt tokens whose KV is in place; feed from here
    int    forked;   // of those, tokens installed from a shared snapshot
    double saved_s;  // prefill seconds those forked tokens would have cost,
                     // priced at this process's measured seconds-per-token
} prefix_reuse;
prefix_reuse engine_prefix_reuse(engine *e, const int32_t *toks, int n);
// Offer the KV now occupying [0, n) for future forks. fed is how many of those
// tokens this request actually prefilled and prefill_s how long that took;
// together they price future hits. Cheap and safe to call unconditionally.
void engine_prefix_publish(engine *e, const int32_t *toks, int n,
                           int fed, double prefill_s);

typedef struct {
    uint64_t hits, misses, stores, evictions;
    uint64_t tokens_reused;
    double   saved_prefill_s;   // cumulative, same pricing as prefix_reuse
    double   cost_per_token_s;  // measured prefill cost behind that figure
    size_t   bytes, budget;
    int      entries;
    double   ttl;
} prefix_cache_stats;

// budget 0 disables the shared cache (each slot's own rewind is unaffected);
// ttl_s is how long an unused snapshot survives.
void   prefix_cache_configure(size_t budget_bytes, double ttl_s);
void   prefix_cache_stats_get(prefix_cache_stats *out);
void   prefix_cache_clear(void);
// host bytes one n-token snapshot of this model would occupy
size_t prefix_cache_entry_bytes(const model_t *m, int n);

#endif // RUNNER_H
