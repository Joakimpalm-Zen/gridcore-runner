// One generation: prompt, sampling loop, constraints, KV reuse.
#ifndef RUNNER_ENGINE_H
#define RUNNER_ENGINE_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "model.h"
#include "tokenizer.h"
#include "sample.h"
#include "schema.h"
#include "jsonmode.h"

// return nonzero to abort generation (e.g. client disconnected)
typedef int (*gen_cb)(void *ud, const char *bytes, int n);

typedef struct { int32_t id; float lp; } lp_alt; // logprob alternative

// JC-R1 constrained-choice posterior: one DECISION POINT — a constrained
// sampling step where >= 2 of the probed top candidates were legal under
// the active schema/JSON constraint. prob[] is the softmax renormalized
// over the legal probed set (raw logits, temperature-free); coverage is
// the full-softmax mass of everything probed, so a consumer can see how
// much distribution the probe examined. Alternatives beyond CL_MAX_ALT
// count toward n_legal/renormalization but are not stored.
#define CL_MAX_ALT 8
typedef struct {
    int32_t pos;                 // emitted-token index in this generation
    int32_t n_legal;             // legal candidates among the probed set
    float   coverage;            // full-softmax mass of ALL probed candidates
    int32_t ids[CL_MAX_ALT];     // legal alternatives, descending probability
    float   prob[CL_MAX_ALT];    // renormalized over the legal probed set
    float   raw_lp[CL_MAX_ALT];  // full-vocab log-softmax of each
} cl_rec;

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
    int     prelude_max, prelude_count;
    bool    prelude_exhausted;
    bool progress;         // print prompt progress to stderr
    int32_t *hist;         // tokens whose KV occupies slots [0, pos)
    // optional per-token logprob capture (server "logprobs"): caller points
    // these at buffers sized [lp_cap] / [lp_cap * lp_n] before generating
    float   *lp_chosen;    // chosen-token logprob per emitted token
    int32_t *lp_ids;       // chosen token id per emitted token
    lp_alt  *lp_top;       // top-N alternatives per token
    int      lp_n, lp_cap, lp_count;
    // JC-R1 "choice_logprobs": constrained-choice posteriors. When cl_cap>0
    // and a schema/JSON constraint is active, each payload sampling step
    // probes the top cl_probe candidates by raw logit against the validator
    // and records a decision point when >= 2 are legal — the renormalized
    // posterior over the legal set is the calibrated verdict surface the
    // judgment co-processor reads (suite docs/plans/judgment-coprocessor.md).
    cl_rec  *cl_recs;      // caller-owned buffer [cl_cap]
    int      cl_cap, cl_count, cl_probe;
    // speculative decoding: a small draft model proposes draft_k tokens per
    // round, the target verifies them in one batched forward
    model_t *dm;           // draft model (NULL = off)
    int      dpos;         // draft KV position (may trail pos)
    int      draft_k;      // drafts per round
    // JC-R2 grammar fast-forward: when the active constraint pins a unique
    // byte continuation, its tokenization is drafted for free (no draft
    // forwards) and verified by the target exactly like a draft-model
    // proposal, so output stays sampler-identical to plain decoding.
    // Opt-in via RUNNER_GRAMMAR_FF=1 (see engine_init for the measured
    // reason); needs the batched verify path (not full GPU offload).
    bool     gram_ff;
    struct { int rounds, drafted, accepted,      // all drafts (either source)
                 gr_drafted, gr_accepted; }      // grammar-pinned drafts only
             spec_st;                            // reset per generation
    // JC-R2 Phase 0 trace (RUNNER_GRAMMAR_TRACE=path): the current grammar
    // round's pinned bytes + drafted ids, stashed at draft time and emitted
    // as one JSONL record when the verify walk resolves the round. Slot-safe
    // (per-engine, not file-static) because --serve runs parallel slots.
    char     gtr_pin[96];  // pinned bytes (GRAM_FF_BYTES)
    int      gtr_plen;
    int32_t  gtr_d[16];    // drafted ids (SPEC_DRAFT_MAX)
    int      gtr_nd;
    // in-flight generation budget, owned by engine_gen_begin/step/end
    int      gen_max, gen_count;
    double   gen_t0;
    // yielded between prefill chunks (see engine_set_prefill_yield)
    void (*prefill_yield)(void *ud);
    void *prefill_ud;
    // identity of everything that decides what this engine's KV bytes mean:
    // the weights, the geometry, the tokenizer and the cache element type.
    // Computed once by engine_init; see engine_prefix_reuse.
    uint64_t model_key;
} engine;

// Called between prefill chunks so a caller holding the device turn can drop
// it and let someone else in.
//
// Prefill is the longest single thing a slot does, and it used to hold that
// turn for all of it: measured on Qwen2.5-7B, a short request arriving during
// a 2,300-token prefill waited 26.2 s against 0.237 s alone -- 110x, i.e. the
// whole prefill. engine_feed already walks the prompt in n_batch chunks, so
// the fix is to let the caller yield between them; the hold becomes one chunk
// instead of one prompt.
//
// Unset is a no-op: the CLI, and any server without a scheduler running.
void engine_set_prefill_yield(engine *e, void (*yield)(void *ud), void *ud);

// True when this request should take the speculative walk (a draft model
// and/or grammar fast-forward under an active constraint) — shared by
// engine_generate and the server's scheduler dispatch so they cannot drift.
bool engine_wants_spec(const engine *e);

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

// ---- snapshot persistence (runner.prefix.v1) --------------------------
//
// A warm prefix cache is worth minutes of prefill and it dies with the
// process. These write it to a file and read it back, so a restarted server
// answers the first agent request at fork speed instead of prefill speed.
//
// The trust question is the whole design. A snapshot is raw KV bytes: loading
// one that does not belong to this model does not error, it produces a
// confident wrong answer, which is the single failure mode this cache's
// lookup path was built to avoid. So the file carries the engine's
// `model_key` -- which already binds the weights, the geometry, the tokenizer,
// the context length and the KV element type -- and every entry whose key does
// not match the live engine is refused, not adapted. The file is also checked
// for a magic, a version, a length-consistent body and a payload digest before
// any of it is believed; a truncated or edited file loads nothing.
//
// It is opt-in and explicit. There is no default path and no automatic
// discovery: a cache directory that another user can write is a way to hand
// this process someone else's KV, and the way not to have that problem is not
// to go looking for files nobody asked for.
//
// Returns the number of entries written / loaded, or -1 on an I/O or format
// error (with a reason on stderr). Loading is additive and skips anything that
// does not fit the live budget.
int prefix_cache_save(const char *path);
int prefix_cache_load(const char *path, const engine *e);

#endif // RUNNER_ENGINE_H
