// Weights, geometry and per-sequence state: the inference core.
#ifndef RUNNER_MODEL_H
#define RUNNER_MODEL_H

#include <math.h>
#include "fp16.h"
#include "gguf.h"
#include "quants.h"
#include "tpool.h"

typedef enum {
    KV_OWNER_MALLOC = 0,
    KV_OWNER_GPU_BACKEND = 1,
} kv_owner_t;
// ACT_SWIGLU_OAI is gpt-oss's clamped alpha-sigmoid GLU. Verified against
// llama.cpp ggml_compute_forward_swiglu_oai_f32 (alpha 1.702, limit 7):
//   x = min(gate, limit); y = clamp(up, -limit, limit)
//   out = (x / (1 + exp(-alpha*x))) * (y + 1)
// Plain SwiGLU here is silently-wrong output, which is why it is its own op.
enum { ACT_SILU = 0, ACT_GELU = 1, ACT_SWIGLU_OAI = 2, ACT_XIELU = 3 };

// softplus, in ggml's exact form including the linear cutover at 20 — above
// it log1p(exp(x)) is the identity to float precision and expf overflows.
static inline float softplus_f32(float x) {
    return x > 20.0f ? x : logf(1.0f + expf(x));
}

// xIELU (Apertus). Transcribed from ggml's op_xielu:
//   x >  0 : alpha_p * x^2 + beta * x
//   x <= 0 : (expm1(min(x, eps)) - x) * alpha_n + beta * x
// `an` and `ap` are the EFFECTIVE parameters, already through the transform
// ggml_xielu applies when it builds the node (softplus, and beta folded into
// alpha_n); model_load does that once per layer. Passing the file's raw values
// here is the bug this signature exists to prevent -- see model.c.
// Unlike every other activation here it is UNGATED — Apertus has no
// ffn_gate tensor, so the FFN is up -> xielu -> down.
static inline float xielu(float x, float an, float ap, float b, float eps) {
    if (x > 0.0f) return ap * x * x + b * x;
    float mn = x < eps ? x : eps;
    return (expm1f(mn) - x) * an + b * x;
}

// Router gating functions, numbered as llama.cpp's llama_expert_gating_func_type
// so a GGUF's expert_gating_func value maps across without translation.
enum {
    EXPERT_GATE_NONE           = 0,
    EXPERT_GATE_SOFTMAX        = 1,
    EXPERT_GATE_SIGMOID        = 2,
    EXPERT_GATE_SOFTMAX_WEIGHT = 3,  // softmax over the SELECTED weights
    EXPERT_GATE_SQRT_SOFTPLUS  = 4,
};
typedef struct {
    gguf_tensor *attn_norm, *wq, *wk, *wv, *wo;
    float       *bq, *bk, *bv, *bo;      // optional biases (f32, converted)
    gguf_tensor *ffn_norm, *w_gate, *w_up, *w_down;
    int          n_ff;          // THIS layer's dense-FFN width. gemma-4 E2B
                                 // publishes real per-layer variation
                                 // (6144/12288) via an ARRAY-typed
                                 // feed_forward_length; everywhere else this
                                 // equals model_t.n_ff, which holds the MAX
                                 // (scratch buffers size off the max)
    // sparse-MoE FFN (Mixtral / Qwen3-MoE): a router picks expert_used of
    // expert_count experts, each a SwiGLU FFN, weighted-summed. When is_moe is
    // set these replace w_gate/w_up/w_down for this layer.
    bool         is_moe;
    int          n_expert;      // THIS layer's expert count, read from its own
                                 // router tensor's ne[1] at load time — usually
                                 // equals model_t.n_expert, but --prune-experts
                                 // writes fused expert tensors with fewer
                                 // blocks for pruned layers, and every MoE
                                 // forward path routes over n_expert, not the
                                 // model-wide count (model.c's check_shape3
                                 // calls require every one of a layer's expert
                                 // tensors to agree on this exact value)
    gguf_tensor *ffn_gate_inp;   // router: n_embd -> n_expert logits (F32)
    // modern fused 3D expert tensors ({.., n_expert}); NULL when split
    // gpt-oss: per-head learned attention-sink logits [n_head], F32. They
    // join the attention softmax DENOMINATOR only — no value row — so the
    // head's output is scaled down without attending anywhere.
    float       *attn_sinks;
    // Selection-only probability bias (DeepSeek V3 aux-loss-free balancing).
    // Added to the probabilities for TOP-K SELECTION and deliberately not to
    // the weights the selected experts are scaled by.
    float       *exp_probs_b;
    // Shared always-on expert (Qwen2-MoE / DeepSeek): a dense FFN over the
    // same normed input, added to the routed output. Qwen2-MoE additionally
    // gates it by sigmoid of a scalar router; DeepSeek has no gate.
    gguf_tensor *w_gate_shexp, *w_up_shexp, *w_down_shexp;
    gguf_tensor *ffn_gate_inp_shexp;
    // Gemma-4 E-series per-layer embeddings: a gate into the PLE width, an
    // elementwise product with this layer's slice of the per-layer table, a
    // projection back to n_embd and its own RMS norm.
    gguf_tensor *ple_gate;       // [n_embd, n_embd_ple]  (blk.N.inp_gate)
    gguf_tensor *ple_proj;       // [n_embd_ple, n_embd]  (blk.N.proj)
    float       *ple_post_norm;  // [n_embd]              (blk.N.post_norm)
    // gpt-oss: router bias [n_expert] and per-expert FFN biases. The expert
    // biases are added to each expert's own gate/up/down result BEFORE the
    // routing weight multiplies it (llama.cpp build_moe_ffn ordering).
    float       *ffn_gate_inp_b;     // [n_expert]
    float       *ffn_gate_exps_b;    // [n_ff_exp * n_expert]
    float       *ffn_up_exps_b;      // [n_ff_exp * n_expert]
    float       *ffn_down_exps_b;    // [n_embd * n_expert]
    gguf_tensor *ffn_gate_exps;  // 3D {n_embd, n_ff, n_expert}
    gguf_tensor *ffn_up_exps;    // 3D {n_embd, n_ff, n_expert}
    gguf_tensor *ffn_down_exps;  // 3D {n_ff, n_embd, n_expert}
    // legacy split layout: one 2D tensor per expert (older Mixtral GGUFs)
    bool         moe_split;
    gguf_tensor **moe_g, **moe_u, **moe_d;  // [n_expert] each, when moe_split
    // gemma-4 MoE: gate and up fused in one 3D tensor {n_embd, 2*n_ff_exp,
    // n_expert} (first half gate, second up); a per-expert down scale; a router
    // input scale; and — every gemma-4 MoE layer ALSO runs a dense GELU FFN as a
    // shared expert (w_gate/w_up/w_down), summed with the routed experts.
    bool         moe_gemma;              // gemma-4 dual-branch (dense + routed) MoE layer
    gguf_tensor *ffn_gate_up_exps;       // fused 3D {n_embd, 2*n_ff_exp, n_expert}
    float       *down_exps_scale;        // [n_expert] per-expert down-projection scale
    float       *gate_inp_scale;         // [n_embd] router-input scale (gemma-4)
    float       *ffn_pre_norm2_w, *ffn_post_norm1_w, *ffn_post_norm2_w; // gemma-4 MoE branch norms
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
// The struct is still one type, but the OWNERSHIP is split: everything in the
// immutable section belongs to a refcounted `model_weights` record shared by
// every instance loaded from the same file with the same weight-side
// parameters, and the pointers here alias into it. Keeping one struct is what
// let that happen without touching a single field access in the backends.
typedef struct {
    // ---- immutable: file and geometry ----
    gguf_file gf;
    // --mlock succeeded for this mapping, so model_free must unlock it before
    // unmapping. Tracked rather than re-derived: an munlock of a mapping that
    // was never locked is a silent no-op that would hide a failed lock.
    bool      weights_locked;
    char     *path;          // owned copy of the load path (shared-weight key)
    bool      file_id_ok;    // stat-backed identity below is valid
    uint64_t  file_size, file_ino;
    int64_t   file_mtime_ns, file_ctime_ns;
    char      arch[32];
    bool      agent_profile;
    uint32_t  agent_protocol_version, agent_tokenizer_version;
    const char *agent_schema_id, *agent_schema_digest;
    gg_str    *agent_required_features;
    uint64_t  n_agent_required_features;
    int       n_layer, n_embd, n_head, n_head_kv, head_dim, n_ff;
    int       n_vocab, n_ctx_train, rope_dim;
    // sparse-MoE (0 = dense model). n_ff_exp is the per-expert FFN width.
    int       n_expert, n_expert_used, n_ff_exp;
    float    *moe_logits;  // [n_expert] router scratch (forward, single token)
    float    *shexp_in, *shexp_g, *shexp_u, *shexp_o;  // shared-expert scratch
    float    *moe_sel_scores;   // [n_expert] selection scores when biased/grouped
    float    *moe_group_score;  // [n_expert_groups] group-limited top-k scratch
    float    *moe_gate;    // [n_ff_exp]
    float    *moe_up;      // [n_ff_exp]
    float    *moe_dexp;    // [n_embd] one expert's down output
    float    *moe_out;     // [n_embd] weighted expert-sum accumulator
    // Grouped-by-expert prefill scratch (sized for a full prompt batch): route
    // all tokens, then run each expert once over ALL its routed tokens as a
    // batched matmul instead of one-at-a-time. Decode (n==1) never uses these.
    float    *moe_out_b;   // [n_batch][n_embd] per-token output accumulator
    float    *moe_gath;    // [n_batch][n_embd] one expert's gathered inputs
    float    *moe_gate_b;  // [n_batch][n_ff_exp] batched gate
    float    *moe_up_b;    // [n_batch][n_ff_exp] batched up
    float    *moe_dexp_b;  // [n_batch][n_embd] batched down output
    int      *moe_sel;     // [n_batch][n_expert_used] per-token selected experts
    float    *moe_selw;    // [n_batch][n_expert_used] per-token renormalized weights
    float    *moe_trace_norms; // [n_batch][n_expert_used] scratch: L2 norm of
                                // each selected expert's own FFN output this
                                // forward, filled only when RUNNER_MOE_TRACE is
                                // set (moe_ffn_grouped's prefill path needs it
                                // held across its per-expert loop, since trace
                                // emission there happens per-token afterward)
    int      *moe_gidx;    // [n_batch] current expert's token indices
    float    *moe_gw;      // [n_batch] current expert's per-token weights
    int       fwd_pos;     // starting KV position of the batch this forward call
                            // is processing; RUNNER_MOE_TRACE reads it to log
                            // each routed token's absolute sequence position
    float    *moe_probe_hist;  // [3][n_batch][n_embd] ring of the last three
                                // gemma-4 layers' post-attention hidden states,
                                // for RUNNER_MOE_PROBE's lookahead router replay
    int       moe_probe_depth; // how many of the 3 ring slots are valid for
                                // the forward call in progress (reset with
                                // fwd_pos; layer 0 always sees 0)
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
    bool      moe_gemma;     // gemma-4 dual-branch MoE (CPU-only; no GPU dual-branch kernel yet)
    bool      moe_prefetch;  // hand routed experts to the OS as whole blocks
                             // (only pays when weights exceed RAM)
    bool      ffn_var;       // per-layer FFN widths differ (gemma-4 E2B):
                             // CPU sizes per layer; device backends refuse
    int       full_attn_interval;
    int       ssm_conv_kernel, ssm_inner, ssm_state, ssm_v_heads, ssm_groups;
    float     logit_softcap; // final logits = c*tanh(x/c) when > 0
    float     logit_scale;   // final logits *= this BEFORE the softcap
                             // (muse-glimmer <arch>.logit_scale; 1 = off)
    float     post_norm_eps; // eps for the sandwich norms (post_attention_norm
                             // / post_ffw_norm) ONLY: muse-glimmer fixes these
                             // at 1e-8 while pre-norms keep rms_eps. 0 = use
                             // rms_eps, which every other arch does.
    bool      embd_norm;     // weightless RMS norm on the embedding row
                             // (muse-glimmer), applied where embd_scale is
    float     resid_scale;   // granite muP: BOTH branch outputs (attention
                             // and FFN) x this before their residual adds;
                             // 1 = off, which every other arch keeps
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
    kv_owner_t kv_owner;     // malloc on CPU/CUDA; backend-owned on Metal
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
    int    xdim;             // max(n_embd, per-layer q_dim); sizes xb/xb2/q
    int    reserve_vram_pct; // VRAM cap for the GPU backend (0 = free VRAM)
    int    gpu_layers_override; // forced leading GPU layer count (0 = auto)
    int    mtp_layers;       // declared multi-token-prediction blocks excluded
                             // from the backbone (training-only; consuming
                             // them is a separate unimplemented feature)
    // Gemma-4 E-series. n_embd_ple > 0 turns on per-layer embeddings;
    // kv_from_start < n_layer turns on shared KV, where every layer at or
    // past it computes no K/V of its own and reads kv_src[l] instead.
    int    n_embd_ple;
    int    kv_from_start;
    int   *kv_src;               // [n_layer] cache-owning layer for each layer
    gguf_tensor *ple_tok_embd;   // [n_embd_ple * n_layer, n_vocab]
    gguf_tensor *ple_model_proj; // [n_embd, n_embd_ple * n_layer]
    float       *ple_proj_norm;  // [n_embd_ple]
    float       *ple;            // scratch [n_batch][n_layer][n_embd_ple]
    float       *ple_tmp;        // scratch [n_batch][n_embd_ple]
    // Sliding-window layers are usually their own rope regime (gemma locals
    // rope at 10k with no scaling while globals run 1M + YaRN). gpt-oss is
    // not like that: llama.cpp passes the same freq_base, freq_scale,
    // ext_factor and attn_factor for EVERY layer and varies only the KV
    // window, so its sliding layers must rope exactly like its global ones.
    // Generalized MoE router (llama.cpp build_moe_ffn). Defaults reproduce the
    // softmax + top-k + renormalize path every currently-certified MoE uses,
    // so an arch that sets none of these is bit-for-bit unaffected.
    int    expert_gating;        // EXPERT_GATE_* above
    // xIELU parameters, one per layer (Apertus publishes them as arrays).
    float *xielu_an, *xielu_ap, *xielu_b, *xielu_eps;
    int    n_ff_shexp;           // shared-expert FFN width (0 = no shared expert)
    int    n_expert_groups;      // >1 enables group-limited top-k (DeepSeek V3)
    int    n_group_used;         // groups kept when n_expert_groups > 1
    float  expert_w_scale;       // 0 or 1 = no scaling
    bool   expert_norm_w;        // renormalize the selected weights
    // Llama-4 attention knobs. Both default off, so every other arch is
    // untouched. NoPE: every no_rope_layer_step-th layer skips rope entirely.
    // Attention temperature: on those SAME layers (and only those — it is the
    // else-branch of the rope test in llama.cpp's llama4 graph) Q is scaled by
    // a per-token factor that grows with position.
    int    no_rope_layer_step;    // 0 = every layer ropes
    bool   nope_on_full;          // muse-glimmer: rope exactly the sliding
                                  // layers; full-attention layers are NoPE.
                                  // Follows l_is_swa (a bool pattern array in
                                  // the GGUF) rather than a periodic step.
    int    attn_temp_floor_scale;
    float  attn_temp_scale;
    float  attn_temp_offset;
    bool   swa_rope_global;
    bool   gptoss;           // gpt-oss: attention sinks + swiglu_oai + MoE
                             // biases; no GPU kernels for those yet
    bool   attn_out_gate;    // afmoe: per-element sigmoid output gate from a
                             // separate blk.N.attn_gate projection of the
                             // normed block input (Qwen "G1"), applied to the
                             // concatenated heads before attn_output. Reuses
                             // the qwen35 q_gate scratch; CPU only for now.
    int    n_dense_lead;     // afmoe leading_dense_block_count: the first N
                             // layers of an n_expert>0 model are plain dense
                             // FFN (no router). 0 everywhere else.
    bool   cpu_moe;          // keep sparse expert FFNs on the host while CUDA
                             // runs the remaining tensors/layers
    bool  *moe_host;         // [n_layer] per-layer expert placement under
                             // cpu_moe: true = this layer's expert FFN runs on
                             // the host. NULL when cpu_moe is off (no layer
                             // bounces). Decided once per file at upload and
                             // shared by every instance on that upload.
    int    cpu_moe_layers;   // requested host-expert layer count (params copy)
    int    gpu_layers;       // leading layers run on GPU (n_layer = full,
                             // <n_layer = partial offload, CPU finishes the rest).
                             // Decided by the first instance to upload a given
                             // file and reused by every instance sharing it, so
                             // parallel slots cannot end up on different splits.
    void  *gpu;              // active GPU backend context (NULL = CPU path).
                             // Runtime fallback clears this without implying
                             // that backend-owned resources may be released.
    void  *gpu_owner;        // backend resource owner, possibly retained after
                             // m->gpu is detached so CPU fallback can keep
                             // reading backend-owned unified-memory KV.
    struct vram_lease *vram; // this instance's entry in the cross-process VRAM
                             // registry (NULL on CPU-only runs, which are never
                             // accounted and never refused)
    // The immutable half above is not owned by this instance: it belongs to a
    // refcounted record holding the mmap, the parsed metadata, the layer array
    // and every f32 conversion, shared by every model_t loaded from the same
    // file with the same weight-side parameters. Every pointer in the
    // immutable section aliases into it, so field access is unchanged and only
    // ownership moved. NULL only for a load that failed before publishing.
    struct model_weights *W;
} model_t;
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
// The YaRN magnitude factor for this layer. One definition, because the CPU
// and CUDA rope paths both need it and a disagreement between them would be
// invisible in output that still looks fluent.
// Does this model's routing need anything the device k_moe_route cannot do?
// That kernel is hardcoded to softmax + top-k + renormalize and takes no bias,
// so any other gating function, a selection-only bias, group-limited top-k, or
// a weight scale must route on the host. The router bias is exempt: it rides
// the matvec tail and never reaches the kernel.
static inline bool model_moe_router_is_plain(const model_t *m) {
    return m->expert_gating == EXPERT_GATE_SOFTMAX &&
           m->n_expert_groups <= 1 && m->expert_norm_w &&
           (m->expert_w_scale == 0.0f || m->expert_w_scale == 1.0f);
}
// Does layer l apply rope? llama.cpp: n_no_rope_layer_step > 0 &&
// (il + 1) % n_no_rope_layer_step != 0.
// muse-glimmer instead ropes exactly its sliding layers (use_rope =
// is_swa(il)), whose pattern is a per-layer bool array, so nope_on_full
// follows l_is_swa rather than a periodic step.
static inline bool model_layer_ropes(const model_t *m, int l) {
    if (m->nope_on_full) return m->l_is_swa != NULL && m->l_is_swa[l];
    return !(m->no_rope_layer_step > 0 &&
             (l + 1) % m->no_rope_layer_step == 0);
}
// The per-token Q scale a NoPE layer applies, from llama.cpp's
// llm_graph_input_attn_temp::set_input. Returns 1 when the knob is off.
static inline float model_attn_temp(const model_t *m, int pos) {
    if (m->attn_temp_scale == 0.0f || m->attn_temp_floor_scale == 0) return 1.0f;
    return logf(floorf(((float)pos + m->attn_temp_offset) /
                       (float)m->attn_temp_floor_scale) + 1.0f)
           * m->attn_temp_scale + 1.0f;
}
static inline float model_rope_mscale(const model_t *m, int l) {
    bool local = m->l_is_swa != NULL && m->l_is_swa[l];
    return (local && !m->swa_rope_global) ? 1.0f : m->rope_mscale;
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

// Reservation auto-fit (`-c 0` with --reserve-ram/--reserve-vram): how many
// context tokens fit a budget once the weights and every slot's activation head
// are paid for, and the clamp that turns that into a context length. Public so
// the arithmetic can be gated without a machine big enough to reach the regime.
#define MODEL_AUTOFIT_HEAD (256u << 20)   // activations + slack, per slot
long long model_autofit_tokens(uint64_t budget, uint64_t weights,
                               uint64_t head_per_seq, uint64_t kv_per_tok,
                               int n_seq);
int       model_autofit_clamp(long long best, int n_ctx_train);
// Whether to warn that the KV cache, not the model's own size, is what pushed
// layers off the device -- the one case a smaller -c can take back.
bool      model_kv_trade_note(int gpu_layers, int n_layer, uint64_t kv_dev,
                              uint64_t weights);
// Which layer physically owns layer l's KV rows. Identity everywhere except
// gemma4 E-series shared-KV layers, which compute no K/V and read an earlier
// layer's cache. l == n_layer is the total-size sentinel and never remapped.
static inline int model_kv_owner(const model_t *m, int l) {
    return (m->kv_src && l < m->n_layer) ? m->kv_src[l] : l;
}
static inline size_t model_kv_byte_off(const model_t *m, int l) {
    size_t e = m->kv_off[model_kv_owner(m, l)];
    return m->kv_q8 ? e / 32 * 34 : e * sizeof(f16_t);
}
// Bytes covering the first `l` layers' cache rows -- the raw cumulative
// boundary. Unlike model_kv_byte_off(), this does NOT redirect through
// model_kv_owner(): that redirect answers "where does layer l's own data
// live", which is wrong for a boundary/count question. Sizing the whole host
// cache (l = n_layer) or a partial device split's buffer (l = gpu_layers)
// needs this one -- calling model_kv_byte_off(m, gpu_layers) instead
// undersizes the buffer whenever gpu_layers itself lands on a shared-KV
// layer, since it would be redirected back to that layer's (earlier) owner.
static inline size_t model_kv_boundary_bytes(const model_t *m, int l) {
    size_t e = m->kv_off[l];
    return m->kv_q8 ? e / 32 * 34 : e * sizeof(f16_t);
}
void        model_ple_prepass(model_t *m, const int32_t *tokens, int n,
                              const float *x, float *out, float *scratch);
// cpu_moe_layers sentinels; any value >= 0 is a literal host-expert layer count
enum { CPU_MOE_ALL = -1, CPU_MOE_AUTO = -2 };

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
    // force exactly this many leading layers onto the GPU (the rest run on the
    // CPU); 0 = auto-fit to available/reserved VRAM. Like llama.cpp's -ngl.
    int   gpu_layers_override;
    // Keep sparse MoE expert FFNs in system RAM while the CUDA backend runs
    // attention, norms, embeddings/output and any dense layers. This is
    // tensor-role placement, distinct from the leading-layer split above.
    bool  cpu_moe;
    // How many MoE layers keep their experts on the host, when cpu_moe is set.
    // CPU_MOE_ALL reproduces the original all-or-nothing flag; CPU_MOE_AUTO
    // fills the remaining VRAM budget with expert banks and hosts the rest;
    // N >= 0 hosts exactly the deepest N MoE layers (device-resident experts
    // stay a leading run, matching the leading-layer split's direction).
    int   cpu_moe_layers;
    int   reserve_ram_pct;
    // How many sequences will share this reservation. A reservation is a
    // budget for the whole server, and its two halves scale differently: the
    // weights are uploaded once and shared, the KV cache and activation head
    // are paid per slot. 0 or 1 means a lone sequence. Only the -c 0 auto-fit
    // reads it; an explicit -c is the caller's own arithmetic.
    int   n_seq;
    // --wait-for-vram: seconds to queue behind other registered runners rather
    // than refusing outright. 0 = refuse immediately (the default).
    int   vram_wait_secs;
    // --vram-priority / RUNNER_VRAM_PRIORITY: advisory small-integer tag on
    // this claim, default 0. Shown in the refusal listing; among several
    // --wait-for-vram waiters on one GPU, decides who is admitted first once
    // space frees (see vram_claim's header comment for the exact, bounded
    // rule). No effect at all outside a --wait-for-vram queue.
    int   vram_priority;
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
    // --moe-prefetch: hand routed experts to the OS as whole blocks before the
    // FFN reads them. 0 = auto (platform default: on for Apple Silicon when
    // weights exceed available RAM, off elsewhere — the A/B'd split), 1 =
    // force on, -1 = force off. A flag rather than env-only because a GUI
    // relaunch (tray LaunchAgent) does not inherit a shell's environment.
    int   moe_prefetch;
    // --mlock: wire the mmap'd weights into RAM so the OS cannot reclaim them.
    // Opt-in and fail-soft by design: locking 5 GB on a 16 GB laptop can cause
    // the pressure it was meant to prevent, so a refusal is reported and the
    // load continues. Without it, weights are clean file-backed pages and are
    // the first thing a loaded machine evicts.
    bool  mlock;
    // --yield-on-request: opt in to the cooperative VRAM yield primitive.
    // --serve polls vram_yield_requested() for the resident model only while
    // idle between requests (no --ttl/-m swap concept applies otherwise), and
    // on a hit releases it cleanly and logs why. Off by default: nothing
    // outside this process can make it give memory back except this flag.
    bool  yield_on_request;
} model_params;

bool   model_load(model_t *m, const char *path, const model_params *p);

// The identity a shared-weights record is keyed on: which file this actually
// is, beyond the path it was spelled with. Every keyed view of a file — the
// host parse in model.c, the device upload in cuda.c, and the prefix-cache key
// — goes through it, so it lives in one place and reports its own failures.
// `ctime` is optional (nullable). A NULL `registry` suppresses the diagnostic
// for callers whose load path has already reported the loss.
//
// Returns false when the file cannot be identified, and SAYS SO on stderr with
// the errno, naming `registry` and the consequence. That diagnostic is the
// point of the function: an unidentifiable file is loaded privately, which
// silently turns off weight sharing AND lets the instance re-decide its own
// CPU/GPU split. Nothing else on either platform notices — the Windows mmap
// path uses GetFileSizeEx and the POSIX one fstat, so a broken stat() shows up
// nowhere else in a load. A correctness difference hiding inside an
// optimization's fallback is exactly the 2026-08-04 shared-weights split
// defect (see the CHANGELOG entry for the fix).
//
// RUNNER_TEST_NO_FILE_ID forces the failure so the fallback is reachable from a
// test on a machine where stat() works; see `make test-shared-noid`.
bool   model_file_identity(const char *path, const char *registry,
                           uint64_t *size, uint64_t *ino,
                           int64_t *mtime, int64_t *ctime);

// Fraction of this model's mapped weights currently in physical memory, or
// -1.0 when the platform will not say. Cheap enough to call per request.
double model_resident_fraction(const model_t *m);
// The architecture admission allowlist, exposed so `--caps` publishes exactly
// what `model_load` accepts (one source of truth). Wrong-math archs
// (granite/gemma2/gemma) are deliberately absent — they load but miscompute.
const char *const *model_supported_archs(size_t *count);
void   model_free(model_t *m); // frees everything incl. GPU context and mmap
// process n tokens starting at position pos; returns logits of the last
// token if want_logits, else NULL
float *model_forward_batch(model_t *m, const int32_t *tokens, int n, int pos,
                           bool want_logits);
// single-token convenience wrapper
float *model_forward(model_t *m, int token, int pos);
// Internal backend bridge for tensor-role placement: apply one complete MoE
// FFN (including its residual) to m->x on the host. CUDA uses this after its
// attention sublayer and then resumes on-device. False rejects a non-MoE layer.
bool   model_moe_ffn_cpu(model_t *m, int layer, int n);
// byte cost of one layer's expert half (router + experts + gemma shared branch)
uint64_t model_layer_expert_bytes(const layer_t *ly, int n_expert);
// bytes a single token actually touches: the whole file for a dense model
// (returned as 0 — the caller already has the file size), or everything except
// the unrouted expert banks for a sparse MoE
uint64_t model_hot_set_bytes(const model_t *m);
// residency warning text; false when no warning is warranted (see model.c)
bool     model_residency_warning(uint64_t need, uint64_t hot, uint64_t have,
                                 bool locked, char *buf, size_t n);
// Advisory prefetch of the experts a router just selected. Fed by whichever
// router ran, so it is architecture-agnostic; cannot change output.
void     model_moe_prefetch(const model_t *m, const layer_t *ly,
                            const int *sel, int used);
// fill m->moe_host for a requested host-expert layer count (see model.c)
void   model_moe_place_host(model_t *m, int host_layers);
// speculative verify: forward a small batch keeping hidden states, then pull
// each row's logits lazily (false/NULL on full GPU offload or n > spec_batch)
bool   model_forward_batch_keep(model_t *m, const int32_t *tokens, int n, int pos);
// whether the batched verify path above can run at all for this model
bool   model_spec_verify_ok(const model_t *m);
float *model_spec_row_logits(model_t *m, int b);
// mean-pooled L2-normalized embedding of toks; clobbers KV slots [0, n)
bool   model_embed(model_t *m, const int32_t *toks, int n, float *out);
// The per-row embedding transforms every forward path applies right after the
// dequantized table row lands in a host buffer: the gemma-family sqrt(n_embd)
// scale, then muse-glimmer's weightless RMS norm. One function so the CPU,
// CUDA and Metal host-side copies cannot drift (same reasoning as RNC-2).
void   model_embd_transform(const model_t *m, float *row);
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

#endif // RUNNER_MODEL_H
