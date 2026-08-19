// Generation engine shared by the CLI and the server: prompt feeding,
// stop-token handling, optional JSON-constrained sampling.
#include "engine.h"
#include "compat.h"


#include <limits.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

double now_s(void) { return plat_now(); }

enum { CP_PROBE, CP_THINK, CP_AFTER_THINK, CP_OUTPUT };

// everything that decides what this engine's KV bytes mean; see the shared
// prefix cache below, which is the only thing that consumes it
static uint64_t model_identity(const model_t *m, const tokenizer *tok);
// the constrained document recorded for engine_constraint_truncate; defined
// with the rest of the constraint layer
static void cdoc_release(engine *e);

static void constraint_reset(engine *e) {
    e->constraint_phase = !e->constraint_includes_prelude &&
                          e->m && e->m->think_open && e->m->think_close
                            ? CP_PROBE : CP_OUTPUT;
    e->constraint_tag_possible = e->constraint_phase == CP_PROBE;
    e->constraint_tag_match = 0;
    e->constraint_close_match = 0;
}

// Admit one stop token by its vocabulary spelling. Silent on a token this
// model does not have, on a duplicate, and on a full list — each is a reason
// this spelling contributes nothing, not a reason to fail an engine.
static void stop_add(engine *e, tokenizer *tok, const char *spelling) {
    int id = tok_find(tok, spelling);
    int cap = (int)(sizeof(e->stop_ids) / sizeof(*e->stop_ids));
    if (id < 0 || e->n_stop >= cap) return;
    for (int j = 0; j < e->n_stop; j++) if (e->stop_ids[j] == id) return;
    e->stop_ids[e->n_stop++] = id;
}

bool engine_init(engine *e, model_t *m, tokenizer *tok, sampler *smp) {
    free(e->hist); // slot engines are re-inited on model swap; e must be zeroed
    free(e->cdoc);
    memset(e, 0, sizeof(*e));
    e->cdoc_rec = true;
    e->m = m;
    e->tok = tok;
    e->smp = smp;
    e->think_end_id = -1;
    e->pending_pos = -1;
    e->stop_ids[e->n_stop++] = tok->eos_id;
    static const char *stops[] = { "<|im_end|>", "<|eot_id|>", "<|end_of_text|>",
                                   "<|endoftext|>", "</s>",
                                   // gemma turn terminators (gemma1-3 / gemma4)
                                   "<end_of_turn>", "<turn|>",
                                   // phi3 ends assistant turns with <|end|>,
                                   // which is not its declared eos_token_id
                                   "<|end|>",
                                   // muse-glimmer ends turns with <|eot|>,
                                   // also not its declared eos (<|eom|> is
                                   // NOT a stop: it separates a reasoning
                                   // turn from the answer that follows it)
                                   "<|eot|>" };
    // gpt-oss / Harmony inverts one of these. <|end|> terminates a NON-final
    // Harmony message — the assistant writes its analysis, closes it with
    // <|end|>, then opens the final channel — so treating <|end|> as a stop
    // ends generation on the reasoning and never reaches the answer. It is in
    // the list above for phi3, whose assistant turns really do end there.
    // <|return|> is Harmony's own end-of-generation marker and takes its place;
    // <|call|> joins it so an unrendered tool attempt stops instead of running
    // away (Harmony tool calling is deliberately not rendered — see template.c).
    bool harmony = strcmp(m->arch, "gpt-oss") == 0;
    for (size_t i = 0; i < sizeof(stops) / sizeof(*stops); i++) {
        if (harmony && !strcmp(stops[i], "<|end|>")) continue;
        stop_add(e, tok, stops[i]);
    }
    if (harmony) {
        static const char *hstops[] = { "<|return|>", "<|call|>" };
        for (size_t i = 0; i < sizeof(hstops) / sizeof(*hstops); i++)
            stop_add(e, tok, hstops[i]);
    }
    if (!strcmp(m->arch, "muse-glimmer"))
        e->think_end_id = tok_find(tok, "<|eom|>");
    // A stop token is not repetition: the chat template puts it in the prompt,
    // the prompt seeds the penalty window, and penalising it can leave a model
    // unable to end its turn at all.
    if (smp) {
        smp->n_no_penalty = 0;
        for (int i = 0; i < e->n_stop &&
             smp->n_no_penalty < (int)(sizeof(smp->no_penalty) / sizeof(*smp->no_penalty));
             i++)
            smp->no_penalty[smp->n_no_penalty++] = e->stop_ids[i];
    }
    jsonv_init(&e->jv);
    constraint_reset(e);
    // grammar fast-forward is opt-in (RUNNER_GRAMMAR_FF=1/on) and needs the
    // batched verify path. Measured 2026-07-30 (EuroLLM-9B, Mistral-Nemo,
    // CPU, contract schema): byte-identical output but 4-12% SLOWER at
    // 33-38% acceptance — real subword vocabs tokenize pinned runs
    // differently than the model samples them, and a rejected round's
    // batched forward costs more than the accepted ones save. The value
    // parses strictly (=1/on), never by existence — the RUNNER_CUDA_TC=0
    // bug class. The structural fix is a model-canonical drafter (the
    // Syntetik half of JC-R2).
    const char *ff = getenv("RUNNER_GRAMMAR_FF");
    e->gram_ff = model_spec_verify_ok(m) &&
                 ff && (!strcmp(ff, "1") || !strcmp(ff, "on"));
    e->hist = malloc(sizeof(int32_t) * m->n_ctx);
    if (!e->hist) return false;   // no history buffer: the engine is unusable
    e->model_key = model_identity(m, tok);
    return true;
}

void engine_reset(engine *e) {
    e->pos = 0;
    e->hit_stop = false;
    sampler_reset(e->smp);
    jsonv_init(&e->jv);
    if (e->schema) sval_init(&e->sv, e->schema);
    constraint_reset(e);
    cdoc_release(e);
    e->dpos = 0;
}

void engine_think_started(engine *e) {
    if (!e || !e->m || !e->m->think_close) return;
    e->constraint_phase = CP_THINK;
    e->constraint_tag_possible = false;
    e->constraint_tag_match = 0;
    e->constraint_close_match = 0;
}

int engine_rewind(engine *e, const int32_t *toks, int n) {
    int keep = 0;
    if (e->hist)
        while (keep < e->pos && keep < n - 1 && e->hist[keep] == toks[keep])
            keep++; // n - 1: always feed at least one token to get logits
    // Recurrent (SSM) layers hold a FOLD over [0, e->pos), not a per-position
    // prefix, so keeping KV rows [0, keep) does not by itself put the recurrent
    // state at `keep` — it is still the fold as-of the old e->pos. Attention's
    // Fact 1 does not hold here. Only act on an actual rewind (keep < e->pos):
    //   - an exact snapshot at `keep` (the spec-decode / abandoned-step rollback
    //     boundary) restores the fold in a memcpy — cheap and bit-identical;
    //   - otherwise the fold cannot be sliced, so recompute from 0: keep nothing
    //     and let the re-feed rebuild both the recurrent state (model_forward's
    //     pos==0 reset) and the reused-anyway attention KV. That is qwen35's
    //     pre-seam behavior — never wrong, just not reused. Reusing the KV tail
    //     for keep>0 on the recompute path is tracer 5 (prefix-cache), which
    //     stores the recurrent blob in the snapshot and drives this same
    //     restore. keep == e->pos is a pure extension: the fold is already right.
    if (model_has_recurrent(e->m) && keep < e->pos) {
        if (!model_recurrent_restore(e->m, keep)) {
            model_recurrent_reset(e->m);
            keep = 0;
        }
    }
    e->pos = keep;
    // the draft's KV beyond the kept prefix was computed from the previous
    // request's tokens; the catch-up loop re-feeds hist[dpos..pos)
    if (e->dpos > keep) e->dpos = keep;
    e->hit_stop = false;
    sampler_reset(e->smp);
    // the kept prefix still counts toward the repeat-penalty window
    for (int i = 0; i < keep; i++) sampler_accept(e->smp, toks[i]);
    jsonv_init(&e->jv);
    if (e->schema) sval_init(&e->sv, e->schema);
    constraint_reset(e);
    cdoc_release(e);
    return keep;
}

// ==================================================== shared KV prefix cache
//
// Correctness here rests on two facts and one hazard.
//
//   Fact 1 — attention is causal, so KV row i is a function of tokens [0, i]
//   only. Any prefix of a snapshot is therefore itself a valid snapshot, which
//   is what lets one stored prompt serve every later prompt that shares any
//   leading run of tokens with it, rather than only exact repeats.
//
//   Fact 2 — on every backend the host kcache/vcache is the authoritative
//   copy. CUDA keeps a device mirror and resyncs it from the host whenever a
//   forward's position is not the previous one plus 1 (see kv_upload in
//   cuda.c); Metal's unified buffer *is* the host cache. So a snapshot is a
//   plain memcpy in both directions, with one caveat handled at the install.
//
//   Hazard — a fork that does not match the new prompt's true prefix silently
//   conditions generation on another request's context. It does not fail; it
//   answers, plausibly, from the wrong premises. Nothing below hashes token
//   ids into a bucket and trusts the bucket: candidates keep their whole token
//   vector and it is compared element by element with the prompt, so only the
//   run that demonstrably matches is ever installed. Hashing is used solely
//   for the things that are *not* tokens — the weights, the geometry, the
//   tokenizer, the cache element type — and a hash miss there costs a prefill,
//   never a wrong answer.

static uint64_t h64(uint64_t h, const void *p, size_t n) {
    const uint8_t *b = (const uint8_t *)p;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 0x100000001B3ull; }
    return h;
}
static uint64_t h64s(uint64_t h, const char *s) {
    return s ? h64(h, s, strlen(s) + 1) : h64(h, "\xff", 1);
}
static uint64_t h64i(uint64_t h, long long v) { return h64(h, &v, sizeof v); }
static uint64_t h64f(uint64_t h, double v) { return h64(h, &v, sizeof v); }

// The carried-over cache-key hazard, answered.
//
// A prefix key must cover everything that changes the tokens or the cache
// bytes, not just the prompt text. `tokenizer.ggml.pre` selects between
// distinct BPE split rules and SPM scores may be rebuilt from merge ranks at
// load, so two models with identical system prompts and different `pre` values
// must never share a prefix. Rather than enumerate the metadata that has
// mattered so far, this digests the tokenizer that was actually built —
// every vocabulary string, every score, every token type, the split rule and
// the BOS/space flags — so a tokenizer that behaves differently keys
// differently whatever the metadata said.
//
// Weight bytes are represented by the load-time file identity, not by a full
// digest on every request. That keeps prefix admission O(prompt) while still
// distinguishing a model rebuilt or replaced in place at the same path. The
// small tensor samples below remain a fallback guard for platforms where stat
// cannot report a useful identity.
static uint64_t model_identity(const model_t *m, const tokenizer *tok) {
    uint64_t h = 0xCBF29CE484222325ull;
    h = h64s(h, "runner.prefix.v1");
    h = h64s(h, m->path);
    h = h64s(h, m->arch);
    h = h64i(h, m->file_id_ok);
    h = h64i(h, (long long)m->file_size);
    h = h64i(h, (long long)m->file_ino);
    h = h64i(h, (long long)m->file_mtime_ns);
    h = h64i(h, (long long)m->file_ctime_ns);
    const long long geo[] = {
        m->n_layer, m->n_embd, m->n_head, m->n_head_kv, m->head_dim, m->n_ff,
        m->n_vocab, m->n_ctx_train, m->rope_dim, m->rope_neox, m->swa_window,
        m->ffn_act, m->v_rmsnorm, m->n_suppress,
        // context geometry and KV element type: both change what a row means
        // and where it lives (--kv f16|q8 is a real axis, not a hint)
        m->n_ctx, m->kv_q8,
    };
    h = h64(h, geo, sizeof geo);
    h = h64f(h, m->rms_eps);       h = h64f(h, m->rope_base);
    h = h64f(h, m->rope_mscale);   h = h64f(h, m->embd_scale);
    h = h64f(h, m->attn_scale);    h = h64f(h, m->logit_softcap);
    h = h64f(h, m->logit_scale);   h = h64f(h, m->post_norm_eps);
    h = h64f(h, m->resid_scale);
    h = h64i(h, m->embd_norm);     h = h64i(h, m->nope_on_full);
    for (int l = 0; l < m->n_layer; l++) {
        h = h64i(h, model_kv_dim(m, l));
        h = h64i(h, model_rope_dim(m, l));
        h = h64i(h, model_is_swa(m, l));
        h = h64i(h, (long long)model_kv_row_bytes(m, l));
        const gguf_tensor *sample[2] = { m->layers[l].wq, m->layers[l].w_down };
        for (int k = 0; k < 2; k++) {
            if (!sample[k] || !sample[k]->data) { h = h64s(h, NULL); continue; }
            h = h64i(h, (long long)sample[k]->nbytes);
            h = h64(h, sample[k]->data,
                    sample[k]->nbytes < 64 ? (size_t)sample[k]->nbytes : 64);
        }
    }
    if (m->tok_embd && m->tok_embd->data) {
        h = h64i(h, (long long)m->tok_embd->nbytes);
        h = h64(h, m->tok_embd->data,
                m->tok_embd->nbytes < 256 ? (size_t)m->tok_embd->nbytes : 256);
    }
    if (!tok) return h64s(h, "no-tokenizer");
    h = h64i(h, tok->model); h = h64i(h, tok->pre);
    h = h64i(h, tok->n_vocab);
    h = h64i(h, tok->bos_id); h = h64i(h, tok->eos_id); h = h64i(h, tok->unk_id);
    h = h64i(h, tok->add_bos); h = h64i(h, tok->add_space_prefix);
    h = h64i(h, tok->scores != NULL); h = h64i(h, tok->ttype != NULL);
    for (int i = 0; i < tok->n_vocab; i++) {
        if (tok->tokens && tok->tokens[i].s)
            h = h64(h, tok->tokens[i].s, (size_t)tok->tokens[i].n);
        else
            h = h64s(h, NULL);
        if (tok->scores) h = h64(h, &tok->scores[i], sizeof(float));
        if (tok->ttype)  h = h64(h, &tok->ttype[i], sizeof(int32_t));
    }
    return h;
}

// Snapshots are packed per layer as { K[n][row_bytes] , V[n][row_bytes] }.
// The n-major layout is what makes a partial restore a straight memcpy of the
// first c rows of each block, which is Fact 1 turned into code.
size_t prefix_cache_entry_bytes(const model_t *m, int n) {
    size_t t = 0;
    for (int l = 0; l < m->n_layer; l++) {
        // shared-KV layers alias an earlier layer's rows: snapshotting them
        // would copy the same bytes twice and make save/load disagree
        if (model_kv_owner(m, l) != l) continue;
        t += 2 * (size_t)n * model_kv_row_bytes(m, l);
    }
    return t;
}

static void pfx_save(const model_t *m, uint8_t *dst, int n) {
    size_t off = 0;
    for (int l = 0; l < m->n_layer; l++) {
        if (model_kv_owner(m, l) != l) continue;
        size_t blk = (size_t)n * model_kv_row_bytes(m, l);
        size_t lo  = model_kv_byte_off(m, l);
        memcpy(dst + off,       (const uint8_t *)m->kcache + lo, blk);
        memcpy(dst + off + blk, (const uint8_t *)m->vcache + lo, blk);
        off += 2 * blk;
    }
}

// install the first `n` rows of a snapshot taken over `stride` rows
static void pfx_load(const model_t *m, const uint8_t *src, int stride, int n) {
    size_t off = 0;
    for (int l = 0; l < m->n_layer; l++) {
        if (model_kv_owner(m, l) != l) continue;
        size_t row = model_kv_row_bytes(m, l);
        size_t blk = (size_t)stride * row, take = (size_t)n * row;
        size_t lo  = model_kv_byte_off(m, l);
        memcpy((uint8_t *)m->kcache + lo, src + off, take);
        memcpy((uint8_t *)m->vcache + lo, src + off + blk, take);
        off += 2 * blk;
    }
}

// A prefix shorter than this is not worth a snapshot: the copy costs more than
// the prefill it saves, and it must be at least 2 for the GPU resync argument
// at the install site to hold.
#define PFX_MIN_TOKENS  16
#define PFX_MAX_ENTRIES 64

typedef struct pfx_entry {
    struct pfx_entry *next;
    uint64_t  key;
    int32_t  *toks;
    int       n;
    uint8_t  *kv;
    size_t    bytes;
    double    used;
    uint64_t  hits;
} pfx_entry;

static struct {
    pfx_entry      *head;
    size_t          bytes, budget;
    double          ttl;
    bool            configured;
    uint64_t        hits, misses, stores, evictions, tokens_reused;
    double          saved_s, cost_per_tok;
    pthread_mutex_t mu;
} PFX = { NULL, 0, 0, 600.0, false, 0, 0, 0, 0, 0, 0.0, 0.0,
          PTHREAD_MUTEX_INITIALIZER };

// caller holds mu
static void pfx_defaults(void) {
    if (PFX.configured) return;
    PFX.configured = true;
    PFX.budget = (size_t)env_u64("RUNNER_PREFIX_CACHE_MB", 0, 1u << 20, 512)
                 * 1024 * 1024;
    PFX.ttl    = env_f64("RUNNER_PREFIX_CACHE_TTL", 0.0, 1e9, 600.0);
}

static void pfx_drop(pfx_entry **pp) {
    pfx_entry *e = *pp;
    *pp = e->next;
    PFX.bytes -= e->bytes;
    free(e->toks); free(e->kv); free(e);
}

static void pfx_expire(double now) {
    for (pfx_entry **pp = &PFX.head; *pp; ) {
        if (now - (*pp)->used > PFX.ttl) { PFX.evictions++; pfx_drop(pp); }
        else pp = &(*pp)->next;
    }
}

static int pfx_count(void) {
    int n = 0;
    for (pfx_entry *p = PFX.head; p; p = p->next) n++;
    return n;
}

// Evict until `need` more bytes fit — least-recently-used, but *unproven*
// entries first.
//
// Plain LRU is the wrong policy here and it fails in the exact case this
// cache exists for. Publishing stores the whole prompt, so every one-shot
// request leaves a large entry behind, and a run of unrelated traffic is
// enough to push out the shared system/tool/schema block that a dozen agent
// requests were about to hit — the newest entry is always the one nobody has
// used yet. Measured: a 2900-token shared prefix was evicted by two 800-token
// one-shot prompts between the request that stored it and the request that
// wanted it, on the 512 MB default.
//
// So an entry that has been forked at least once has proved it is shared, and
// is only evicted once nothing unproven is left. Within each class the order
// is still least-recently-used.
static void pfx_trim(size_t need) {
    while (PFX.head && (PFX.bytes + need > PFX.budget ||
                        pfx_count() >= PFX_MAX_ENTRIES)) {
        pfx_entry **victim = NULL;
        for (int proven = 0; proven <= 1 && !victim; proven++)
            for (pfx_entry **pp = &PFX.head; *pp; pp = &(*pp)->next) {
                if (((*pp)->hits > 0) != (proven == 1)) continue;
                if (!victim || (*pp)->used < (*victim)->used) victim = pp;
            }
        if (!victim) break; // every entry belongs to one of the two classes
        PFX.evictions++;
        pfx_drop(victim);
    }
}

void prefix_cache_configure(size_t budget_bytes, double ttl_s) {
    pthread_mutex_lock(&PFX.mu);
    PFX.configured = true;
    PFX.budget = budget_bytes;
    PFX.ttl    = ttl_s;
    pfx_trim(0);
    pthread_mutex_unlock(&PFX.mu);
}

void prefix_cache_clear(void) {
    pthread_mutex_lock(&PFX.mu);
    while (PFX.head) pfx_drop(&PFX.head);
    PFX.bytes = 0;
    PFX.hits = PFX.misses = PFX.stores = PFX.evictions = PFX.tokens_reused = 0;
    PFX.saved_s = 0;
    pthread_mutex_unlock(&PFX.mu);
}

void prefix_cache_stats_get(prefix_cache_stats *out) {
    pthread_mutex_lock(&PFX.mu);
    pfx_defaults();
    out->hits = PFX.hits; out->misses = PFX.misses;
    out->stores = PFX.stores; out->evictions = PFX.evictions;
    out->tokens_reused = PFX.tokens_reused;
    out->saved_prefill_s = PFX.saved_s;
    out->cost_per_token_s = PFX.cost_per_tok;
    out->bytes = PFX.bytes; out->budget = PFX.budget;
    out->entries = pfx_count();
    out->ttl = PFX.ttl;
    pthread_mutex_unlock(&PFX.mu);
}

prefix_reuse engine_prefix_reuse(engine *e, const int32_t *toks, int n) {
    prefix_reuse r = { 0, 0, 0.0 };
    // the slot's own KV first: it is already in place and costs nothing
    r.keep = engine_rewind(e, toks, n);
    if (!e->hist || n < 2) return r;

    pthread_mutex_lock(&PFX.mu);
    pfx_defaults();
    double now = now_s();
    pfx_expire(now);

    // n - 1: at least one prompt token must still be fed, or there are no
    // logits to sample the first output token from
    int best = 0;
    pfx_entry *hit = NULL;
    for (pfx_entry *p = PFX.head; p; p = p->next) {
        if (p->key != e->model_key) continue;
        if (p->bytes != prefix_cache_entry_bytes(e->m, p->n)) continue; // stale layout
        int c = 0;
        while (c < p->n && c < n - 1 && p->toks[c] == toks[c]) c++;
        if (c > best) { best = c; hit = p; }
    }
    // A shared snapshot stores KV only (Fact 1). Forking it for a recurrent
    // model would install attention rows at `best` while leaving the recurrent
    // fold as-of this slot's previous end — silently wrong. Storing the
    // recurrent blob in the snapshot and restoring it on an exact hit is tracer
    // 5; until then, recurrent models decline the shared fork and keep the
    // self-rewind result above (which forced a correct recompute).
    if (!model_has_recurrent(e->m) &&
        hit && best >= PFX_MIN_TOKENS && best > r.keep) {
        // CUDA's device KV is a mirror that only resyncs when a forward's
        // position is not the previous one plus 1. A fork writes host rows
        // behind the device's back, so it has to *create* that discontinuity
        // rather than hope for one: without this, a slot whose previous
        // generation happened to stop at position best-1 would keep decoding
        // against the previous request's device rows. One forward of toks[0]
        // at position 0 recomputes a row the snapshot overwrites with
        // identical bytes a moment later and leaves the mirror at position 0,
        // so the next forward at `best` (>= PFX_MIN_TOKENS) always uploads.
        if (e->m->gpu) model_forward_batch(e->m, toks, 1, 0, false);
        pfx_load(e->m, hit->kv, hit->n, best);
        memcpy(e->hist, toks, sizeof(int32_t) * (size_t)best);
        e->pos = best;
        e->dpos = 0;          // the draft model's KV was not forked
        e->hit_stop = false;
        sampler_reset(e->smp);
        for (int i = 0; i < best; i++) sampler_accept(e->smp, toks[i]);
        jsonv_init(&e->jv);
        if (e->schema) sval_init(&e->sv, e->schema);
        constraint_reset(e);

        hit->used = now;
        hit->hits++;
        r.keep    = best;
        r.forked  = best;
        r.saved_s = PFX.cost_per_tok * best;
        PFX.hits++;
        PFX.tokens_reused += (uint64_t)best;
        PFX.saved_s += r.saved_s;
    } else {
        PFX.misses++;
    }
    pthread_mutex_unlock(&PFX.mu);
    return r;
}

void engine_prefix_publish(engine *e, const int32_t *toks, int n,
                           int fed, double prefill_s) {
    if (!e->hist || n < PFX_MIN_TOKENS || e->pos < n) return;

    pthread_mutex_lock(&PFX.mu);
    pfx_defaults();
    // price future hits off real measured prefill, smoothed: one request's
    // wall clock on a shared box is noise, the running average is not
    if (fed >= PFX_MIN_TOKENS && prefill_s > 0) {
        double c = prefill_s / fed;
        PFX.cost_per_tok = PFX.cost_per_tok > 0
                             ? 0.8 * PFX.cost_per_tok + 0.2 * c : c;
    }
    if (PFX.budget == 0) { pthread_mutex_unlock(&PFX.mu); return; }

    double now = now_s();
    pfx_expire(now);

    // A snapshot may not eat more than half the budget on its own, or one
    // long prompt evicts everything worth keeping. Truncating it is free:
    // a prefix of a prefix is still a valid prefix (Fact 1).
    size_t per_tok = prefix_cache_entry_bytes(e->m, 1);
    int store_n = n;
    if (per_tok > 0 && prefix_cache_entry_bytes(e->m, store_n) > PFX.budget / 2)
        store_n = (int)((PFX.budget / 2) / per_tok);
    if (store_n < PFX_MIN_TOKENS) { pthread_mutex_unlock(&PFX.mu); return; }

    // Already covered? Refresh it. A strict extension of an existing entry
    // replaces it rather than sitting beside it.
    int diverge_at = 0;   // longest prefix shared with an entry we diverge from
    for (pfx_entry **pp = &PFX.head; *pp; ) {
        pfx_entry *p = *pp;
        int lim = p->n < store_n ? p->n : store_n, c = 0;
        if (p->key != e->model_key) { pp = &p->next; continue; }
        while (c < lim && p->toks[c] == toks[c]) c++;
        if (c == lim && p->n >= store_n) {   // nothing new to say
            p->used = now;
            pthread_mutex_unlock(&PFX.mu);
            return;
        }
        if (c == lim) pfx_drop(pp);          // ours strictly extends p
        else {
            if (c > diverge_at) diverge_at = c;
            pp = &p->next;
        }
    }

    // Store to the DIVERGENCE POINT, not to the end of the prompt.
    //
    // Two requests in an agent session share a system prompt, a tool list and
    // a schema, then differ. Storing each one whole means the shared block is
    // held once per request: measured with four sibling prompts over a
    // ~2,300-token shared prefix, the cache held four entries totalling
    // 530 MB of a 512 MB budget for what is one 132 MB block of shared KV, and
    // the fourth store evicted the first.
    //
    // The tail past `diverge_at` is exactly the part another request has
    // already been observed NOT to share, so it is the part least worth a slot
    // in a fixed budget. It is also the part a repeat of this same prompt
    // recovers for free from its own slot's KV via engine_rewind. Truncating
    // is free in correctness terms for the same reason the half-budget cap
    // above is: a prefix of a prefix is still a valid prefix.
    if (diverge_at >= PFX_MIN_TOKENS && diverge_at < store_n) {
        store_n = diverge_at;
        // ...unless that exact prefix is already stored, in which case there
        // is nothing to add and the scan above would have said so.
        for (pfx_entry *p = PFX.head; p; p = p->next) {
            if (p->key != e->model_key || p->n < store_n) continue;
            int c = 0;
            while (c < store_n && p->toks[c] == toks[c]) c++;
            if (c == store_n) { p->used = now; pthread_mutex_unlock(&PFX.mu); return; }
        }
    }

    size_t need = prefix_cache_entry_bytes(e->m, store_n);
    pfx_trim(need);
    if (PFX.bytes + need > PFX.budget) { pthread_mutex_unlock(&PFX.mu); return; }

    pfx_entry *ne = calloc(1, sizeof(*ne));
    int32_t *nt = malloc(sizeof(int32_t) * (size_t)store_n);
    uint8_t *kv = malloc(need);
    if (!ne || !nt || !kv) {
        free(ne); free(nt); free(kv);
        pthread_mutex_unlock(&PFX.mu);
        return;
    }
    memcpy(nt, toks, sizeof(int32_t) * (size_t)store_n);
    pfx_save(e->m, kv, store_n);
    ne->key = e->model_key; ne->toks = nt; ne->n = store_n;
    ne->kv = kv; ne->bytes = need; ne->used = now;
    ne->next = PFX.head; PFX.head = ne;
    PFX.bytes += need;
    PFX.stores++;
    pthread_mutex_unlock(&PFX.mu);
}

// ---------------------------------------------------- snapshot persistence

#define PFX_MAGIC   "runner.prefix.v1"
#define PFX_MAGIC_N 16

// FNV-1a over the body. Not a security primitive -- it is here so a truncated
// or half-written file is refused rather than believed. The refusal that
// matters (wrong model) is the key check, which a digest cannot do.
static uint64_t pfx_digest(const void *p, size_t n) {
    const uint8_t *b = p;
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 0x100000001b3ull; }
    return h;
}

static bool wr(FILE *f, const void *p, size_t n) { return fwrite(p, 1, n, f) == n; }
static bool rd(FILE *f, void *p, size_t n) { return fread(p, 1, n, f) == n; }

int prefix_cache_save(const char *path) {
    if (!path || !*path) return -1;
    // write beside and rename, so a crash mid-write cannot leave a file that
    // passes its own digest over half a cache
    char tmp[1024];
    if (snprintf(tmp, sizeof(tmp), "%s.partial", path) >= (int)sizeof(tmp)) return -1;
    FILE *f = fopen(tmp, "wb");
    if (!f) { fprintf(stderr, "prefix: cannot write %s\n", tmp); return -1; }

    pthread_mutex_lock(&PFX.mu);
    uint32_t count = 0;
    for (pfx_entry *p = PFX.head; p; p = p->next) count++;

    uint64_t digest = 0xcbf29ce484222325ull;
    bool ok = wr(f, PFX_MAGIC, PFX_MAGIC_N) && wr(f, &count, sizeof count);
    for (pfx_entry *p = PFX.head; ok && p; p = p->next) {
        uint64_t key = p->key, bytes = p->bytes;
        int32_t  n   = p->n;
        ok = wr(f, &key, sizeof key) && wr(f, &n, sizeof n) &&
             wr(f, &bytes, sizeof bytes) &&
             wr(f, p->toks, sizeof(int32_t) * (size_t)n) && wr(f, p->kv, p->bytes);
        // the digest covers exactly what a load will read back
        digest ^= pfx_digest(&key, sizeof key);   digest *= 0x100000001b3ull;
        digest ^= pfx_digest(p->toks, sizeof(int32_t) * (size_t)n);
        digest *= 0x100000001b3ull;
        digest ^= pfx_digest(p->kv, p->bytes);    digest *= 0x100000001b3ull;
    }
    pthread_mutex_unlock(&PFX.mu);

    ok = ok && wr(f, &digest, sizeof digest);
    if (fclose(f) != 0) ok = false;
    if (!ok || rename(tmp, path) != 0) {
        remove(tmp);
        fprintf(stderr, "prefix: failed to write %s\n", path);
        return -1;
    }
    return (int)count;
}

int prefix_cache_load(const char *path, const engine *e) {
    if (!path || !*path || !e || !e->m) return -1;
    FILE *f = fopen(path, "rb");
    if (!f) return -1;                 // absent is not an error worth shouting

    char magic[PFX_MAGIC_N];
    uint32_t count = 0;
    if (!rd(f, magic, PFX_MAGIC_N) || memcmp(magic, PFX_MAGIC, PFX_MAGIC_N) != 0 ||
        !rd(f, &count, sizeof count)) {
        fprintf(stderr, "prefix: %s is not a %s file\n", path, PFX_MAGIC);
        fclose(f);
        return -1;
    }

    pthread_mutex_lock(&PFX.mu);
    pfx_defaults();
    double now = now_s();
    int loaded = 0, refused = 0;
    uint64_t digest = 0xcbf29ce484222325ull;
    bool bad = false;
    for (uint32_t i = 0; i < count && !bad; i++) {
        uint64_t key = 0, bytes = 0;
        int32_t  n = 0;
        if (!rd(f, &key, sizeof key) || !rd(f, &n, sizeof n) ||
            !rd(f, &bytes, sizeof bytes) || n < PFX_MIN_TOKENS ||
            bytes == 0 || bytes > (uint64_t)1 << 40) { bad = true; break; }
        int32_t *toks = malloc(sizeof(int32_t) * (size_t)n);
        uint8_t *kv   = malloc((size_t)bytes);
        if (!toks || !kv || !rd(f, toks, sizeof(int32_t) * (size_t)n) ||
            !rd(f, kv, (size_t)bytes)) { free(toks); free(kv); bad = true; break; }
        digest ^= pfx_digest(&key, sizeof key);  digest *= 0x100000001b3ull;
        digest ^= pfx_digest(toks, sizeof(int32_t) * (size_t)n);
        digest *= 0x100000001b3ull;
        digest ^= pfx_digest(kv, (size_t)bytes); digest *= 0x100000001b3ull;

        // The refusal that matters. A snapshot from another model, another
        // context length or another KV dtype is not adapted -- it is dropped.
        // Adapting it would be inventing KV rows, which reads as fluent output
        // and is wrong.
        if (key != e->model_key ||
            bytes != prefix_cache_entry_bytes(e->m, n)) {
            free(toks); free(kv); refused++;
            continue;
        }
        pfx_entry *ne = calloc(1, sizeof(*ne));
        if (!ne) { free(toks); free(kv); bad = true; break; }
        pfx_trim((size_t)bytes);
        if (PFX.bytes + bytes > PFX.budget) {   // does not fit the live budget
            free(toks); free(kv); free(ne);
            continue;
        }
        ne->key = key; ne->toks = toks; ne->n = n;
        ne->kv = kv; ne->bytes = (size_t)bytes; ne->used = now;
        ne->next = PFX.head; PFX.head = ne;
        PFX.bytes += (size_t)bytes;
        loaded++;
    }
    uint64_t want = 0;
    bool have_digest = !bad && rd(f, &want, sizeof want);
    pthread_mutex_unlock(&PFX.mu);
    fclose(f);

    if (bad || !have_digest || want != digest) {
        // Everything read so far is suspect, so none of it is kept. A partial
        // load of a corrupt file is the worst outcome: it looks like success.
        prefix_cache_clear();
        fprintf(stderr, "prefix: %s is truncated or corrupt — nothing loaded\n", path);
        return -1;
    }
    if (refused)
        fprintf(stderr, "prefix: %d snapshot(s) in %s belong to a different "
                "model or context — refused\n", refused, path);
    return loaded;
}

// load a draft model for speculative decoding, with the same gates in CLI
// and server mode: the target must keep a CPU verify path, and the vocabs
// must match modulo family padding. NULL (with a stderr note) = run plain.
model_t *spec_draft_load(const char *path, const model_t *target,
                         const model_params *mp) {
    if (target->gpu && target->gpu_layers >= target->n_layer) {
        fprintf(stderr, "draft: target is fully GPU-offloaded — speculative "
                "decoding needs the CPU verify path, ignoring --draft\n");
        return NULL;
    }
    model_params dmp = *mp;
    dmp.n_ctx = target->n_ctx; // draft must cover the target's positions
    // Draft KV may use the requested q8 storage too. It can change which
    // tokens the draft proposes (and therefore acceptance/throughput), but the
    // target still verifies every proposal and defines the final output.
    // Respect the caller's backend policy. In particular, --gpu off is a
    // process-wide promise; silently auto-offloading the draft consumed VRAM
    // even though the target correctly stayed on CPU.
    model_t *dm = malloc(sizeof(model_t));
    if (!dm) {
        fprintf(stderr, "draft: out of memory — ignoring --draft\n");
        return NULL;   // callers treat NULL as "run plain decoding"
    }
    if (!model_load(dm, path, &dmp)) {
        // model_load memsets the struct on entry and may allocate before failing
        // (late load failures must free partial buffers to avoid leaks)
        model_free(dm);
        free(dm);
        return NULL;
    }
    if (abs(dm->n_vocab - target->n_vocab) > 512) {
        // model families pad the vocab differently per size; small
        // differences are padding ids the draft never emits
        fprintf(stderr, "draft: vocab mismatch (%d vs %d) — ignoring --draft\n",
                dm->n_vocab, target->n_vocab);
        model_free(dm);
        free(dm);
        return NULL;
    }
    return dm;
}

static bool is_stop(engine *e, int id) {
    for (int i = 0; i < e->n_stop; i++) if (e->stop_ids[i] == id) return true;
    return false;
}

// hist, pos and the sampler's penalty window advance together, one batch at a
// time, because everything downstream reads hist as the answer to "which
// tokens produced the KV in [0, pos)?".
//
// Writing hist up front instead used to break that on exactly one path: a run
// too long for the remaining context skipped the write altogether (the guard
// was `pos + n <= n_ctx`) while the loop below still fed and counted every
// batch that did fit. pos then pointed past rows belonging to this request
// over hist entries belonging to the *previous* one, and the next request's
// engine_rewind happily "kept" a prefix that matched nothing in the cache.
void engine_set_prefill_yield(engine *e, void (*yield)(void *), void *ud) {
    e->prefill_yield = yield;
    e->prefill_ud    = ud;
}

float *engine_feed(engine *e, const int32_t *toks, int n) {
    float *logits = NULL;
    model_t *m = e->m;
    for (int i = 0; i < n; ) {
        int chunk = n - i < m->n_batch ? n - i : m->n_batch;
        if (e->pos + chunk > m->n_ctx) return NULL;
        bool last = (i + chunk == n);
        if (e->hist) memcpy(e->hist + e->pos, toks + i, sizeof(int32_t) * chunk);
        logits = model_forward_batch(m, toks + i, chunk, e->pos, last);
        for (int j = 0; j < chunk; j++) sampler_accept(e->smp, toks[i + j]);
        e->pos += chunk;
        i += chunk;
        if (e->progress && n > 512 && (i % 512 < m->n_batch || last))
            fprintf(stderr, "\rprompt: %d/%d tokens%s", i, n, last ? "\n" : "");
        // Between chunks, never inside one: a chunk is one model_forward_batch
        // and splitting that is the backend's business, not this loop's. Not
        // after the last chunk either -- the caller still owns the turn when
        // engine_feed returns, and releasing it here would leave the pairing
        // to be reasoned about at every call site instead of one.
        if (!last && e->prefill_yield) e->prefill_yield(e->prefill_ud);
    }
    return logits;
}

// Advance a tag-prefix match by one byte, retaining the longest suffix that
// can still begin the tag. Tags are tiny, so the direct overlap check keeps
// this state self-contained without another allocation or public parser API.
static int tag_advance(const char *tag, int match, char c) {
    int tl = (int)strlen(tag);
    int max = match + 1 < tl ? match + 1 : tl;
    for (int k = max; k > 0; k--) {
        int start = match + 1 - k;
        bool same = true;
        for (int j = 0; j < k; j++) {
            int at = start + j;
            char got = at < match ? tag[at] : c;
            if (got != tag[j]) { same = false; break; }
        }
        if (same) return k;
    }
    return 0;
}

// A token that is nothing but insignificant whitespace is rejected under a
// schema. It is always legal JSON, which is exactly the problem: it keeps the
// token in the candidate set at every position, and an unsure model emits runs
// of it until -n is exhausted. Whitespace is optional wherever it is legal, so
// refusing it removes no reachable document; inside a string it is content and
// sval_ws_is_content() says so, leaving those tokens admissible.
// Read once per process, not once per sampled token. Both decoders call this
// from inside their token loop, on every slot thread, and getenv walks the
// whole environment block each time. Same cached shape as the other switches
// in this file; NULL vs set is the whole test, so RUNNER_DEBUG_TOKENS=0 still
// enables the trace exactly as it did before.
static bool debug_tokens(void) {
    static int on = -1;
    if (on < 0) on = getenv("RUNNER_DEBUG_TOKENS") ? 1 : 0;
    return on == 1;
}

static bool all_insignificant_ws(const sval *sv, const char *bytes, int n) {
    // RUNNER_SCHEMA_ALLOW_WS=1 restores the old behaviour, so the effect can
    // be A/B'd in one binary rather than across two builds -- the only way
    // this project has found to keep a before/after honest.
    static int allow = -1;
    if (allow < 0) {
        const char *e = getenv("RUNNER_SCHEMA_ALLOW_WS");
        allow = e && *e && strcmp(e, "0") ? 1 : 0;
    }
    if (allow) return false;
    if (n <= 0 || sval_ws_is_content(sv)) return false;
    for (int i = 0; i < n; i++) {
        char c = bytes[i];
        if (c != ' ' && c != '\t' && c != '\n' && c != '\r') return false;
    }
    return true;
}

// Record payload bytes so the validator can be re-seated on a prefix of them.
// A failure to grow simply stops recording: the buffer exists to make a stop
// match honest, and losing it must degrade that one behaviour rather than end
// a generation that is otherwise fine.
static void cdoc_put(engine *e, const char *bytes, int n) {
    if (!e->cdoc_rec || n <= 0) return;
    if (e->cdoc_n > INT_MAX - n) { e->cdoc_rec = false; return; }
    if (e->cdoc_n + n > e->cdoc_cap) {
        int cap = e->cdoc_cap ? e->cdoc_cap : 512;
        while (cap < e->cdoc_n + n) {
            if (cap > (1 << 26)) { e->cdoc_rec = false; return; }
            cap *= 2;
        }
        char *p = realloc(e->cdoc, (size_t)cap);
        if (!p) { e->cdoc_rec = false; return; }
        e->cdoc = p;
        e->cdoc_cap = cap;
    }
    memcpy(e->cdoc + e->cdoc_n, bytes, (size_t)n);
    e->cdoc_n += n;
}

// The recording is per generation, not per engine: it is dead weight between
// requests, and freeing it here is what keeps every teardown path that frees
// only e->hist correct.
static void cdoc_release(engine *e) {
    free(e->cdoc);
    e->cdoc = NULL;
    e->cdoc_n = e->cdoc_cap = 0;
    e->cdoc_rec = true;
}

// The validator is fed through a copy so a token whose LAST byte is illegal
// does not leave it half-consumed. On a scratch engine that protection is
// already paid for by the copy the caller made, and paying it twice is what
// constrained decoding actually costs: sval is 7 KB and the sampler probes
// this once per candidate token -- with top_k off, once per vocabulary entry.
// Measured with `sample` (SmolLM2-135M-Instruct-Q8_0, 49k vocab, an array
// schema, --top-k 0 --gpu off): 1644 of 5985 decode-thread samples were in
// memmove, against 249 in sval_feed -- the copies cost 6.6x the validation
// they were protecting, and two thirds of them were these.
static bool constraint_payload_feed(engine *e, bool schema,
                                    const char *bytes, int n) {
    if (schema) {
        if (all_insignificant_ws(&e->sv, bytes, n)) return false;
        if (e->constraint_scratch) return sval_feed(&e->sv, bytes, n);
        sval tmp = e->sv;
        if (!sval_feed(&tmp, bytes, n)) return false;
        e->sv = tmp;
    } else {
        if (e->constraint_scratch) return jsonv_feed(&e->jv, bytes, n);
        jsonv tmp = e->jv;
        if (!jsonv_feed(&tmp, bytes, n)) return false;
        e->jv = tmp;
    }
    cdoc_put(e, bytes, n);
    return true;
}

// Would these bytes survive the payload validator? Same question
// constraint_payload_feed answers, asked without changing anything.
//
// The engine copy above is what a caller needs to feed a THROWAWAY engine;
// probing a candidate token needs no such thing, only a scratch validator, and
// the trial APIs carry just the live part of one. Measured on
// SmolLM2-135M-Instruct-Q8_0 with --top-k 0 (the whole-vocabulary walk, where
// this runs once per vocabulary entry per token), the engine copy this
// replaces was 2928 bytes against roughly 300 here.
static bool constraint_payload_trial(const engine *e, bool schema,
                                     const char *bytes, int n) {
    if (schema) {
        if (all_insignificant_ws(&e->sv, bytes, n)) return false;
        sval scratch;
        return sval_trial(&e->sv, &scratch, bytes, n);
    }
    jsonv scratch;
    return jsonv_trial(&e->jv, &scratch, bytes, n);
}

static void constraint_payload_reset(engine *e, bool schema) {
    if (schema) sval_init(&e->sv, e->schema);
    else        jsonv_init(&e->jv);
    e->cdoc_n = 0;
}

// Feed one decoded token through the optional thinking prelude and then the
// JSON/schema validator. On success, *visible is the first byte belonging to
// the constrained payload, or -1 when this token is entirely prelude/tag.
// The function mutates only its engine argument, so sample lookahead calls it
// on a shallow engine copy and acceptance calls it on the real engine.
static bool constraint_feed(engine *e, bool schema, const char *bytes, int n,
                            int *visible) {
    *visible = -1;
    if (e->constraint_phase == CP_OUTPUT) {
        if (!constraint_payload_feed(e, schema, bytes, n)) return false;
        *visible = 0;
        return true;
    }

    if (e->constraint_phase == CP_AFTER_THINK) {
        e->constraint_phase = CP_OUTPUT;
        if (!constraint_payload_feed(e, schema, bytes, n)) return false;
        *visible = 0;
        return true;
    }

    const char *open = e->m->think_open;
    const char *close = e->m->think_close;
    if (e->constraint_phase == CP_THINK) {
        int cl = (int)strlen(close);
        for (int i = 0; i < n; i++) {
            e->constraint_close_match = tag_advance(
                close, e->constraint_close_match, bytes[i]);
            if (e->constraint_close_match != cl) continue;
            e->constraint_phase = CP_OUTPUT;
            e->constraint_close_match = 0;
            constraint_payload_reset(e, schema);
            int at = i + 1;
            if (at < n && !constraint_payload_feed(e, schema, bytes + at, n - at))
                return false;
            if (at < n) *visible = at;
            return true;
        }
        return true;
    }

    // CP_PROBE keeps both safe starts alive: a direct constrained payload, or
    // optional leading whitespace followed by the declared opening tag.
    bool payload_ok;
    sval sv = e->sv;
    jsonv jv = e->jv;
    if (schema) payload_ok = sval_feed(&sv, bytes, n);
    else        payload_ok = jsonv_feed(&jv, bytes, n);

    int ol = (int)strlen(open);
    bool tag_ok = e->constraint_tag_possible;
    int match = e->constraint_tag_match;
    for (int i = 0; tag_ok && i < n; i++) {
        char c = bytes[i];
        if (match == 0 && (c == ' ' || c == '\n' || c == '\r' || c == '\t'))
            continue;
        if (c != open[match]) { tag_ok = false; break; }
        match++;
        if (match != ol) continue;

        e->constraint_phase = CP_THINK;
        e->constraint_tag_possible = false;
        e->constraint_tag_match = 0;
        e->constraint_close_match = 0;
        constraint_payload_reset(e, schema);
        int sub = -1;
        if (i + 1 < n &&
            !constraint_feed(e, schema, bytes + i + 1, n - i - 1, &sub))
            return false;
        if (sub >= 0) *visible = i + 1 + sub;
        return true;
    }

    if (!payload_ok && !tag_ok) return false;
    if (payload_ok) {
        if (schema) e->sv = sv;
        else        e->jv = jv;
        // CP_PROBE feeds the payload validator directly rather than through
        // constraint_payload_feed, so the recording has to be made here too --
        // otherwise a document that started without a thinking prelude is
        // untruncatable.
        cdoc_put(e, bytes, n);
    }
    e->constraint_tag_possible = tag_ok;
    e->constraint_tag_match = tag_ok ? match : 0;
    if (!tag_ok) {
        e->constraint_phase = CP_OUTPUT;
        *visible = 0;
    }
    // Ambiguous leading whitespace is deliberately held back. It is optional
    // JSON whitespace if the payload wins, and prelude whitespace otherwise.
    return true;
}

static bool constraint_done(const engine *e, bool schema) {
    return e->constraint_phase == CP_OUTPUT &&
           (schema ? e->sv.done : e->jv.done);
}

// A protocol token the model was trained on (Muse's <|message|>, <|start|>)
// decodes to no bytes, so the byte machine can never observe it — yet the
// atem automata SPELL those markers as literals. Without this, the model is
// forced to type its own protocol out as text, which is off-distribution;
// measured on the real model, greedy decoding then loops "<|message|>" to
// the token limit on some tool_choice:"auto" prompts. Accept a decoded-empty
// token in constrained output when feeding its raw vocab spelling would
// advance the machine, and refuse it while the machine is consuming free
// content (a raw scalar, a JSON string): a protocol marker mid-value is
// corruption, not syntax. Stop tokens are excluded — they end the turn, and
// the raw-tail rule in constraint_token_ok owns them.
static bool constraint_spelling_ok(engine *e, int id, bool schema) {
    if (!schema || e->constraint_phase != CP_OUTPUT) return false;
    if (is_stop(e, id)) return false;
    if (sval_ws_is_content(&e->sv)) return false;
    const char *sp = tok_raw(e->tok, id);
    if (!sp || !*sp) return false;
    // CP_OUTPUT is guaranteed by the first line, so this is exactly what
    // constraint_feed would do with these bytes.
    return constraint_payload_trial(e, true, sp, (int)strlen(sp));
}

// validity filter for constrained mode. Before a declared thinking block
// closes, ordinary tokens are allowed; stop/control tokens remain valid only
// inside that prelude or after the constrained payload is complete.
static bool constraint_token_ok(engine *e, int id, bool schema) {
    // Stop and control tokens are valid only inside a declared thinking
    // prelude or once the constrained payload is complete.
    //
    // Tightening this so a STOP is refused in CP_THINK too was tried on
    // 2026-08-07 and REVERTED: the defect it was aimed at is on gemma4, which
    // declares no think tags at all (only qwen3/qwen3moe/qwen35 do), so the
    // change was a no-op for the case that motivated it and untestable
    // without a think-tag model to hand. It may still be right; it is not
    // shipping unmeasured.
    if (is_stop(e, id) || tok_is_control(e->tok, id))
        return e->constraint_phase == CP_THINK ||
               e->constraint_phase == CP_AFTER_THINK ||
               constraint_done(e, schema) ||
               // A trailing raw value (Muse's to=user free-text answer) can
               // only ever end at the model's own stop token: its byte
               // sentinel is the SPELLED stop marker, which the token
               // decodes to no bytes. A stop there is the answer's natural
               // end; refusing it forced every such answer to burn the whole
               // token budget and finish "length". Controls that are not
               // stops stay refused — a turn header mid-answer is corruption.
               (is_stop(e, id) && schema &&
                e->constraint_phase == CP_OUTPUT &&
                sval_at_raw_tail(&e->sv)) ||
               constraint_spelling_ok(e, id, schema);
    char buf[512];
    int n = tok_decode(e->tok, id, buf, sizeof(buf));
    if (n == 0)
        return e->constraint_phase == CP_THINK ||
               constraint_done(e, schema) ||
               constraint_spelling_ok(e, id, schema);
    // Inside the payload -- every step of an ordinary constrained generation
    // -- constraint_feed is exactly one payload feed, so ask that directly.
    // The engine copy below exists for the prelude phases, which have engine
    // state (tag match, phase) to advance; the payload phase has none.
    if (e->constraint_phase == CP_OUTPUT)
        return constraint_payload_trial(e, schema, buf, n);
    engine tmp = *e;
    tmp.cdoc_rec = false;   // shallow copy: it must not touch e's recording
    tmp.constraint_scratch = true;  // ...and may be fed in place, being one
    int visible;
    return constraint_feed(&tmp, schema, buf, n, &visible);
}

// See engine.h. Replay, not rollback: the validator has no un-feed, but it is
// a deterministic byte machine, so re-seating it on the delivered prefix of a
// document it already accepted reproduces exactly the state the model would
// have been in had it stopped there.
int engine_constraint_truncate(engine *e, int n) {
    if (!e || n <= 0 || (!e->schema && !e->json_mode)) return 0;
    // Nothing was recorded (an allocation failure, or no payload yet), so
    // there is no state to re-seat and the old behaviour stands.
    if (!e->cdoc_rec || e->cdoc_n <= 0) return 0;
    if (n > e->cdoc_n) n = e->cdoc_n;
    int keep = e->cdoc_n - n;
    bool schema = e->schema != NULL;
    constraint_payload_reset(e, schema);   // clears cdoc_n
    if (keep > 0) {
        // Fed directly rather than through constraint_payload_feed: the
        // whitespace suppression there is a rule about which TOKENS a
        // constrained model may sample, and replay must reproduce the bytes
        // the validator actually saw, not re-apply that rule to them.
        bool ok = schema ? sval_feed(&e->sv, e->cdoc, keep)
                         : jsonv_feed(&e->jv, e->cdoc, keep);
        if (ok) e->cdoc_n = keep;
        else constraint_payload_reset(e, schema); // unreachable: a prefix of an
                                                  // accepted document is accepted
    }
    return n;
}

static bool schema_ok(void *ud, int id) {
    return constraint_token_ok(ud, id, true);
}

static bool json_ok(void *ud, int id) {
    return constraint_token_ok(ud, id, false);
}

// RUNNER_SCHEMA_TRACE=1: one line per accepted token showing what the
// constraint layer did with it -- phase, visible offset, validator depth and
// done flag, and the decoded bytes. Added 2026-08-07 after two hypotheses
// about a "generation stops with no payload" defect were both wrong from
// reading the code alone; the first trace run answered it in one line.
static void schema_trace(const engine *e, bool schema, const char *bytes,
                         int n, int visible, bool fed_ok) {
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("RUNNER_SCHEMA_TRACE");
        on = v && *v && strcmp(v, "0") ? 1 : 0;
    }
    if (!on) return;
    char esc[128];
    int k = 0;
    for (int i = 0; i < n && k < (int)sizeof(esc) - 5; i++) {
        unsigned char c = (unsigned char)bytes[i];
        if (c == '\n') { esc[k++] = '\\'; esc[k++] = 'n'; }
        else if (c == '\t') { esc[k++] = '\\'; esc[k++] = 't'; }
        else if (c < 32 || c > 126) k += snprintf(esc + k, sizeof(esc) - k, "\\x%02x", c);
        else esc[k++] = (char)c;
    }
    esc[k] = 0;
    fprintf(stderr, "schema-trace phase=%d n=%d visible=%d fed=%d done=%d "
            "depth=%d bytes=\"%s\"\n",
            e->constraint_phase, n, visible, (int)fed_ok,
            schema ? (int)e->sv.done : (int)e->jv.done,
            schema ? e->sv.depth : 0, esc);
}

static int constraint_accept(engine *e, bool schema, const char *bytes, int n,
                             gen_cb cb, void *ud) {
    int visible;
    bool fed = constraint_feed(e, schema, bytes, n, &visible);
    schema_trace(e, schema, bytes, n, visible, fed);
    if (!fed) return 1;
    // the hidden span is the thinking prelude (tags included); chat serving
    // opts in to receive it so its splitter can surface the reasoning
    int hidden = visible >= 0 ? visible : n;
    if (hidden > 0 && e->emit_think_prelude && cb) {
        int rc = cb(ud, bytes, hidden);
        if (rc) return rc;
    }
    return cb && visible >= 0 && visible < n
             ? cb(ud, bytes + visible, n - visible) : 0;
}

// Muse's <|eom|> is a decoded-empty control token, so the byte matcher cannot
// observe it. It is nevertheless the authoritative boundary between the
// self-addressed reasoning turn and the next assistant recipient. Feed the
// model's logical close marker to the channel splitter, reset the payload,
// then permit decoded-empty start/role controls until the recipient arrives.
static int constraint_finish_think(engine *e, bool schema,
                                   gen_cb cb, void *ud) {
    int rc = 0;
    // Only an OPENED thinking block gets the close-and-swallow treatment.
    // CP_PROBE means the model never opened one — an ambiguous-whitespace
    // prelude, say — and feeding a close tag there injects it into content:
    // measured as `    </think>` (no payload) on the ornith conformance
    // fixture, where the pre-atem behavior was `    {"answer":""}`. From the
    // probe the correct exit is the old silent flip to output.
    if (e->constraint_phase == CP_THINK) {
        if (e->emit_think_prelude && cb && e->m->think_close)
            rc = cb(ud, e->m->think_close, (int)strlen(e->m->think_close));
        e->constraint_phase = CP_AFTER_THINK;
    } else {
        e->constraint_phase = CP_OUTPUT;
    }
    e->constraint_tag_possible = false;
    e->constraint_tag_match = e->constraint_close_match = 0;
    constraint_payload_reset(e, schema);
    return rc;
}

static int constraint_control_accept(engine *e, int tok, bool schema,
                                     gen_cb cb, void *ud) {
    if (e->constraint_phase != CP_THINK || tok != e->think_end_id) return 0;
    return constraint_finish_think(e, schema, cb, ud);
}

// logprob capture: raw-logit log-softmax stats taken BEFORE sample_pick
// mutates the recent tokens' logits with the repeat penalty
typedef struct {
    float lse;                 // log sum exp of the raw logits
    float snap[256];           // raw values of the penalty-window tokens
    int   snap_ids[256], n_snap;
} lp_pre;

static void lp_capture_pre(engine *e, const float *logits, lp_pre *p) {
    int V = e->m->n_vocab;
    float mx = logits[0];
    for (int i = 1; i < V; i++) if (logits[i] > mx) mx = logits[i];
    double sum = 0;
    for (int i = 0; i < V; i++) sum += expf(logits[i] - mx);
    p->lse = mx + logf((float)sum);
    p->n_snap = 0;
    sampler *s = e->smp;
    // n_recent is capped at 256 in sample.c (sampler.recent[256]); the extra
    // bound keeps this copy in range even if that invariant ever changes.
    int snap_cap = (int)(sizeof(p->snap) / sizeof(p->snap[0]));
    for (int i = 0; i < s->n_recent && p->n_snap < snap_cap; i++) {
        p->snap_ids[p->n_snap] = s->recent[i];
        p->snap[p->n_snap++]   = logits[s->recent[i]];
    }
    // top-N alternatives for this position (insertion into a small list)
    if (e->lp_n <= 0) return;
    lp_alt *top = e->lp_top + (size_t)e->lp_count * e->lp_n;
    int filled = 0;
    for (int i = 0; i < V; i++) {
        float lp = logits[i] - p->lse;
        if (filled == e->lp_n && lp <= top[filled - 1].lp) continue;
        int j = filled < e->lp_n ? filled++ : e->lp_n - 1;
        while (j > 0 && top[j - 1].lp < lp) { top[j] = top[j - 1]; j--; }
        top[j].id = i; top[j].lp = lp;
    }
    for (int j = filled; j < e->lp_n; j++) { top[j].id = -1; top[j].lp = 0; }
}

// JC-R1: probe the top candidates against the active constraint and record
// a decision point when the schema leaves a real choice (>= 2 legal). Raw
// logits, before sample_pick's repeat penalty mutates them, so the posterior
// is calibration-meaningful; legality uses the same validator trial the
// sampler itself uses, so "legal" is exactly "what sample_pick could have
// returned". Runs only inside the constrained payload — the thinking
// prelude samples freely and has no decision points.
static void cl_capture(engine *e, const float *logits) {
    if (!e->cl_cap || e->cl_count >= e->cl_cap) return;
    if (!e->schema && !e->json_mode) return;
    if (e->constraint_phase != CP_OUTPUT) return;
    int V = e->m->n_vocab;
    enum { PROBE_CAP = 64 };
    int M = e->cl_probe >= 8 && e->cl_probe <= PROBE_CAP ? e->cl_probe : 32;
    if (M > V) M = V;
    float mx = logits[0];
    for (int i = 1; i < V; i++) if (logits[i] > mx) mx = logits[i];
    double sum = 0;
    for (int i = 0; i < V; i++) sum += expf(logits[i] - mx);
    float lse = mx + logf((float)sum);
    int   ids[PROBE_CAP]; float lps[PROBE_CAP];
    int filled = 0;
    for (int i = 0; i < V; i++) {
        float lp = logits[i] - lse;
        if (filled == M && lp <= lps[filled - 1]) continue;
        int j = filled < M ? filled++ : M - 1;
        while (j > 0 && lps[j - 1] < lp) {
            lps[j] = lps[j - 1]; ids[j] = ids[j - 1]; j--;
        }
        lps[j] = lp; ids[j] = i;
    }
    cl_rec *r = &e->cl_recs[e->cl_count];
    r->pos = e->gen_count;
    r->n_legal = 0;
    r->coverage = 0;
    double legal_mass = 0;
    bool schema = e->schema != NULL;
    for (int c = 0; c < filled; c++) {
        float pm = expf(lps[c]);
        r->coverage += pm;
        if (!constraint_token_ok(e, ids[c], schema)) continue;
        if (r->n_legal < CL_MAX_ALT) {
            r->ids[r->n_legal]    = ids[c];
            r->raw_lp[r->n_legal] = lps[c];
            r->prob[r->n_legal]   = pm;   // renormalized below
        }
        legal_mass += pm;
        r->n_legal++;
    }
    if (r->n_legal < 2 || legal_mass <= 0) return;  // forced: no decision
    int stored = r->n_legal < CL_MAX_ALT ? r->n_legal : CL_MAX_ALT;
    for (int i = 0; i < stored; i++)
        r->prob[i] = (float)(r->prob[i] / legal_mass);
    e->cl_count++;
}

// JC-R2 grammar fast-forward: byte-level forced-continuation discovery by
// trial. In the constrained payload phase, when exactly one next byte keeps
// the validator alive, that byte is pinned by the grammar — no model opinion
// is involved in emitting it. The pinned run's tokenization is proposed as a
// free draft (no draft-model forwards) and verified by the target like any
// other draft, so even a non-canonical tokenization can only cost acceptance,
// never correctness. Probing is by trial on validator copies — the same
// validator-by-trial design the sampler filter uses, no materialized mask.
static int grammar_forced_bytes(const engine *e, bool schema,
                                char *out, int cap, bool *doc_done) {
    *doc_done = false;
    if (e->constraint_phase != CP_OUTPUT) return 0;
    sval  sv = e->sv;
    jsonv jv = e->jv;
    if (schema ? sv.done : jv.done) return 0;
    int n = 0;
    while (n < cap) {
        int legal = -1;
        for (int c = 1; c < 256; c++) {          // NUL is never a legal byte
            char b = (char)c;
            bool ok;
            // 255 trials per pinned byte, so the trial API's short copy is
            // what keeps this affordable: a full sval copy here was 2240
            // bytes against roughly 300.
            if (schema) { sval  t; ok = sval_trial(&sv, &t, &b, 1); }
            else        { jsonv t; ok = jsonv_trial(&jv, &t, &b, 1); }
            if (!ok) continue;
            if (legal >= 0) return n;            // a real choice: stop here
            legal = c;
        }
        if (legal < 0) return n;                 // completion boundary
        char b = (char)legal;
        if (!(schema ? sval_feed(&sv, &b, 1) : jsonv_feed(&jv, &b, 1)))
            return n;
        out[n++] = b;
        if (schema ? sv.done : jv.done) { *doc_done = true; return n; }
    }
    return n;
}

// JC-R2 Phase 0 trace: one JSONL record per grammar round, emitted when the
// verify walk resolves it. Byte fields are hex-encoded so arbitrary pinned
// bytes never fight JSON escaping; the analysis script decodes them. Same
// opt-in shape as RUNNER_MOE_TRACE (cached getenv + FILE*, off by default).
static FILE *gtrace_file(void) {
    static FILE *fp = NULL;
    static int opened = 0;
    if (!opened) {
        opened = 1;
        const char *path = getenv("RUNNER_GRAMMAR_TRACE");
        if (path && *path) {
            fp = fopen(path, "a");
            if (!fp)
                fprintf(stderr, "warning: RUNNER_GRAMMAR_TRACE=%s: "
                                "could not open for append\n", path);
        }
    }
    return fp;
}

static void gtrace_hex(FILE *fp, const char *b, int n) {
    for (int i = 0; i < n; i++) fprintf(fp, "%02x", (unsigned char)b[i]);
}

// div_i < 0: every draft accepted. Otherwise the walk diverged at draft
// index div_i: exp is the draft's token, act the target's actual pick, and
// their decoded bytes are logged so the classifier needs no tokenizer.
static void gtrace_emit(engine *e, int accepted, int div_i, int exp, int act) {
    FILE *fp = gtrace_file();
    if (!fp || e->gtr_nd <= 0) return;
    fprintf(fp, "{\"pos\":%d,\"pin\":\"", e->pos);
    gtrace_hex(fp, e->gtr_pin, e->gtr_plen);
    fprintf(fp, "\",\"draft\":[");
    for (int i = 0; i < e->gtr_nd; i++)
        fprintf(fp, "%s%d", i ? "," : "", e->gtr_d[i]);
    fprintf(fp, "],\"accepted\":%d", accepted);
    if (div_i >= 0) {
        char tb[512];
        int tn, off = 0;
        // byte offset of the diverged draft position inside the pinned run,
        // so the analyzer can separate within-pin from beyond-pin merges
        // without needing the tokenizer
        for (int i = 0; i < div_i; i++) {
            tn = tok_decode(e->tok, e->gtr_d[i], tb, sizeof(tb));
            if (tn > 0) off += tn;
        }
        fprintf(fp, ",\"div\":%d,\"off\":%d,\"exp\":%d,\"act\":%d,\"exp_b\":\"",
                div_i, off, exp, act);
        tn = exp >= 0 ? tok_decode(e->tok, exp, tb, sizeof(tb)) : 0;
        gtrace_hex(fp, tb, tn > 0 ? tn : 0);
        fprintf(fp, "\",\"act_b\":\"");
        tn = act >= 0 ? tok_decode(e->tok, act, tb, sizeof(tb)) : 0;
        gtrace_hex(fp, tb, tn > 0 ? tn : 0);
        fprintf(fp, "\"");
    }
    fprintf(fp, "}\n");
    fflush(fp);   // durable across a killed server, same reasoning as the
                  // MoE routing trace
    e->gtr_nd = 0;
}

// tokenize a pinned byte run into draft proposals, keeping only the token
// prefix that provably round-trips to a prefix of the pinned bytes — any
// tokenizer normalization the raw encode still leaks (or a byte the vocab
// cannot express) just shortens the draft, never corrupts it.
static int grammar_draft(engine *e, int32_t *d, int max_d) {
    enum { GRAM_FF_BYTES = 96 };
    char fb[GRAM_FF_BYTES], tb[512];
    bool doc_done;
    int fn = grammar_forced_bytes(e, e->schema != NULL, fb, GRAM_FF_BYTES,
                                  &doc_done);
    if (fn <= 0) return 0;
    int nd = tok_encode_raw(e->tok, fb, fn, d, max_d);
    if (e->tok->encode_oom) return 0;
    int off = 0, keep = 0;
    for (int i = 0; i < nd; i++) {
        if (d[i] < 0 || d[i] >= e->m->n_vocab || tok_is_control(e->tok, d[i]))
            break;
        int tn = tok_decode(e->tok, d[i], tb, sizeof(tb));
        if (tn <= 0 || off + tn > fn || memcmp(fb + off, tb, tn) != 0) break;
        off += tn;
        keep++;
    }
    // Tail withholding (measured 2026-08-04, gemma-4-E4B over the torture
    // suite: 100% of draft rejections were the model merging the final
    // pinned token with the free bytes that follow the pin). Unless the pin
    // completes the document — nothing follows, so no merge is possible —
    // don't draft the token that ends exactly at the pin boundary; it is
    // the one position the target tokenizes unpredictably. Only when at
    // least one draft survives: a single-token draft keeps the round alive
    // (a rejected slot costs no more than the plain decode it replaces),
    // and zeroing rounds would also blind the engagement gate.
    if (!doc_done && keep > 1 && off == fn) keep--;
    if (keep > 0 && gtrace_file()) {
        memcpy(e->gtr_pin, fb, fn);
        e->gtr_plen = fn;
        memcpy(e->gtr_d, d, sizeof(int32_t) * (size_t)keep);
        e->gtr_nd = keep;
    }
    return keep;
}

// speculative decoding: sampler-equality verification (llama.cpp-style) —
// sample each position from the TARGET's logits with the full sampler chain;
// a draft is accepted when the sampled token equals it, so output follows
// exactly the same distribution as the non-speculative path
// budget expired mid-document: complete it so constrained output stays valid
// JSON (the caller's finish_reason stays "length")
static void constraint_close(engine *e, gen_cb cb, void *ud) {
    e->constraint_closing = true;
    // A hard token ceiling may land inside the reasoning prelude. Preserve the
    // constrained-output contract by synthesizing the same minimal valid tail
    // used for an ordinary mid-document ceiling; reasoning tokens still count
    // toward max_new and no extra model sampling occurs here.
    if ((e->schema || e->json_mode) && e->constraint_phase != CP_OUTPUT) {
        constraint_payload_reset(e, e->schema != NULL);
        e->constraint_phase = CP_OUTPUT;
    }
    if (e->schema && !e->sv.done) {
        char cbuf[4096];
        int cn = sval_close(&e->sv, cbuf, sizeof(cbuf));
        if (cn > 0 && cb) cb(ud, cbuf, cn);
    } else if (!e->schema && e->json_mode && !e->jv.done) {
        char cbuf[600];
        int cn = jsonv_close(&e->jv, cbuf, sizeof(cbuf));
        if (cn > 0 && cb) cb(ud, cbuf, cn);
    }
    // the recorded document has served its only purpose; every generation ends
    // here, so this is where it stops costing memory
    cdoc_release(e);
    e->constraint_closing = false;
}

static int engine_generate_spec(engine *e, float *logits, int max_new,
                                gen_cb cb, void *ud, double *gen_time) {
    char buf[512];
    int n_gen = 0;
    e->hit_stop = false;
    e->oom = false;
    e->lp_count = 0;
    e->prelude_count = 0;
    e->prelude_exhausted = false;
    e->prelude_max = max_new < 0 ? 0 : (max_new > 1 ? max_new / 2 : max_new);
    double t0 = now_s();
    model_t *m = e->m, *dm = e->dm;
    memset(&e->spec_st, 0, sizeof(e->spec_st));
    // Draft window size. Bounded below by 1, above by the model's spec_batch,
    // and — defensively — by the fixed stack buffer d[] so a future spec_batch
    // bump can never overflow it (RNC-4: d[] used to be a bare 16 unlinked to
    // spec_batch).
    enum { SPEC_DRAFT_MAX = 16 };  // capacity of d[]; must be >= any spec_batch
    int K = e->draft_k;
    if (K < 1) K = 1;
    if (K > m->spec_batch - 1) K = m->spec_batch - 1;
    if (K > SPEC_DRAFT_MAX - 1) K = SPEC_DRAFT_MAX - 1;
    if (K > m->n_batch) K = m->n_batch; // activation buffers hold n_batch rows
    int32_t d[SPEC_DRAFT_MAX];
    float *dl = NULL; // draft logits for position dpos
    // Even under JSON/schema constraints, speculation stays target-exact:
    // the draft proposes, but only target-sampled tokens feed the validator.
    sample_ok_fn ok = e->schema ? schema_ok : e->json_mode ? json_ok : NULL;
    bool constrained = e->schema || e->json_mode;
    // grammar drafts fill the whole verify window; they cost no forwards and
    // a pinned token is a near-certain accept, so draft_k does not cap them
    int GK = m->spec_batch - 1;
    if (GK > SPEC_DRAFT_MAX - 1) GK = SPEC_DRAFT_MAX - 1;
    if (GK > m->n_batch) GK = m->n_batch; // same activation-buffer bound
    int st_rounds = 0, st_drafted = 0, st_accepted = 0;
    int st_gr_drafted = 0, st_gr_accepted = 0;
    #define SPEC_STATS() do { \
        e->spec_st.rounds = st_rounds; e->spec_st.drafted = st_drafted; \
        e->spec_st.accepted = st_accepted; \
        e->spec_st.gr_drafted = st_gr_drafted; \
        e->spec_st.gr_accepted = st_gr_accepted; \
        if (dm || getenv("RUNNER_SPEC_STATS")) fprintf(stderr, \
            "spec: %d rounds, %d drafted, %d accepted (%.2f tok/round)" \
            ", grammar %d/%d\n", \
            st_rounds, st_drafted, st_accepted, \
            st_rounds ? (double)n_gen / st_rounds : 0, \
            st_gr_accepted, st_gr_drafted); } while (0)

    while ((max_new < 0 || n_gen < max_new) && e->pos < m->n_ctx) {
        st_rounds++;
        // a grammar-pinned run drafts for free and preempts the draft model
        int nd = 0, gr = 0;
        if (e->gram_ff && constrained) {
            int cap = GK;
            if (max_new >= 0 && cap > max_new - n_gen) cap = max_new - n_gen;
            if (cap > m->n_ctx - e->pos - 1) cap = m->n_ctx - e->pos - 1;
            if (cap > 0) nd = gr = grammar_draft(e, d, cap);
            st_drafted += nd; st_gr_drafted += nd;
        }
        // catch the draft up on tokens accepted since its last position
        while (!nd && dm && e->dpos < e->pos) {
            int chunk = e->pos - e->dpos < dm->n_batch ? e->pos - e->dpos
                                                       : dm->n_batch;
            if (e->dpos + chunk > dm->n_ctx) { dl = NULL; break; }
            dl = model_forward_batch(dm, e->hist + e->dpos, chunk, e->dpos,
                                     e->dpos + chunk == e->pos);
            e->dpos += chunk;
        }
        // draft up to K tokens greedily (nd == 0 degrades to plain decoding)
        while (!gr && dl && nd < K && e->pos + nd + 1 < m->n_ctx &&
               e->dpos + 1 < dm->n_ctx) {
            int best = 0;
            for (int i = 1; i < dm->n_vocab; i++)
                if (dl[i] > dl[best]) best = i;
            if (best >= m->n_vocab) break; // padding id the target can't embed
            d[nd++] = best;
            st_drafted++;
            dl = model_forward(dm, best, e->dpos++);
        }
        // one batched target forward computes every draft position's hidden
        // state; row logits are pulled lazily as the walk reaches them
        if (nd && !model_forward_batch_keep(m, d, nd, e->pos))
            nd = 0; // verify unavailable: plain decoding

        // walk the drafts (i < nd) plus one bonus position (i == nd)
        int i = 0;
        for (; i <= nd; i++) {
            if (max_new >= 0 && n_gen >= max_new) goto rewind;
            float *ti = i == 0 ? logits : model_spec_row_logits(m, i - 1);
            int tok = sample_pick(e->smp, ti, m->n_vocab, ok, e);
            if (tok < 0) {
                if (tok == -2) e->oom = true;  // error, not a clean stop
                e->hit_stop = true;
                e->pos += i; // keep the accepted drafts' KV
                if (e->dpos > e->pos) e->dpos = e->pos;
                goto done;
            }
            sampler_accept(e->smp, tok);
            if (debug_tokens()) fprintf(stderr, " %d", tok);
            if (is_stop(e, tok) && !e->ignore_eos) {
                e->hit_stop = true;
                e->pos += i; // keep the accepted drafts' KV
                if (e->dpos > e->pos) e->dpos = e->pos;
                goto done;
            }
            int n = tok_decode(e->tok, tok, buf, sizeof(buf));
            bool in_prelude = constrained && e->constraint_phase != CP_OUTPUT;
            int rc = e->schema && n > 0
                       ? constraint_accept(e, true, buf, n, cb, ud)
                   : e->json_mode && n > 0
                       ? constraint_accept(e, false, buf, n, cb, ud)
                   : cb && n > 0 ? cb(ud, buf, n) : 0;
            if (!rc && constrained)
                rc = constraint_control_accept(e, tok, e->schema != NULL,
                                               cb, ud);
            if (!rc && e->schema && n == 0 &&
                constraint_spelling_ok(e, tok, true)) {
                const char *sp = tok_raw(e->tok, tok);
                rc = constraint_accept(e, true, sp, (int)strlen(sp), cb, ud);
            }
            n_gen++;
            if (in_prelude && (e->constraint_phase == CP_PROBE ||
                               e->constraint_phase == CP_THINK) &&
                e->prelude_max > 0 && ++e->prelude_count >= e->prelude_max) {
                e->prelude_exhausted = true;
                // mirrors engine_gen_step: close the prelude rather than
                // ending the turn with nothing. `in_prelude` already required
                // a constraint, so there is no other case to handle.
                rc = constraint_finish_think(e, e->schema != NULL, cb, ud);
            }
            bool constrained_done = e->schema
                                      ? constraint_done(e, true)
                                      : e->json_mode && constraint_done(e, false);
            if (i < nd && tok == d[i] && rc == 0) {
                e->hist[e->pos + i] = tok; // accepted: its KV is already right
                st_accepted++;
                if (gr) st_gr_accepted++;
                if (constrained_done) {
                    if (gr) gtrace_emit(e, i + 1, -1, -1, -1);
                    e->hit_stop = true;
                    e->pos += i + 1;
                    if (e->dpos > e->pos) e->dpos = e->pos;
                    if (gen_time) *gen_time = now_s() - t0;
                    SPEC_STATS();
                    return n_gen;
                }
                continue;
            }
            // mismatch, bonus position, or aborted: forward the real token
            if (gr) gtrace_emit(e, i < nd ? i : nd,
                                i < nd ? i : -1,
                                i < nd ? d[i] : -1,
                                i < nd ? tok : -1);
            e->pos += i;
            if (constrained_done) {
                e->hit_stop = true;
                if (e->dpos > e->pos) e->dpos = e->pos;
                if (gen_time) *gen_time = now_s() - t0;
                SPEC_STATS();
                return n_gen;
            }
            if (e->hist && e->pos < m->n_ctx) e->hist[e->pos] = tok;
            logits = model_forward(m, tok, e->pos);
            e->pos++;
            // rewind the draft to just before this token: the next catch-up
            // refeeds it, fixing the draft's KV for the rejected position AND
            // refreshing dl (clamping to pos left dl stale from the abandoned
            // round — acceptance collapsed to ~zero)
            if (e->dpos > e->pos - 1) e->dpos = e->pos - 1;
            if (rc) {
                // An aborted callback is not only a client that left: a
                // matched stop sequence arrives here too, and leaving by this
                // door skipped the close entirely, so the caller was served a
                // constrained document with no legal ending at all. The solo
                // loop always closes (engine_gen_end does), and which loop
                // served the request must not decide what the client gets.
                goto done;
            }
            i = -1; // signal: already advanced
            break;
        }
        if (i >= 0) {
rewind:
            e->pos += i > nd ? nd : i; // budget hit mid-walk: keep accepted
            if (e->dpos > e->pos) e->dpos = e->pos;
            if (max_new >= 0 && n_gen >= max_new) break;
        }
    }
done:
    constraint_close(e, cb, ud);
    if (gen_time) *gen_time = now_s() - t0;
    SPEC_STATS();
    #undef SPEC_STATS
    return n_gen;
}

// ------------------------------------------------- the generation step
//
// engine_generate used to own both halves of a decode step: the per-sequence
// work (sample, constrain, stop-check, stream) and the forward that produces
// the next step's logits. Continuous batching needs to keep the first half and
// take the second away, because one thread has to issue every batched forward
// (see model_batch_decode). So the loop is split at exactly that seam.
//
// engine_generate below is written on the same three calls the scheduler uses,
// which is the point: there is one copy of the sampler/schema/stop/stream
// logic, not a solo one and a batched one that can drift apart.

void engine_gen_begin(engine *e, int max_new) {
    e->hit_stop  = false;
    e->oom       = false;
    e->lp_count  = 0;
    e->gen_max   = max_new;
    e->gen_count = 0;
    e->prelude_count = 0;
    e->prelude_exhausted = false;
    e->prelude_max = max_new < 0 ? 0 : (max_new > 1 ? max_new / 2 : max_new);
    e->pending_pos = -1;
    e->gen_t0    = now_s();
}

int engine_gen_step(engine *e, const float *logits, gen_cb cb, void *ud,
                    int32_t *next_tok, int *next_pos) {
    char buf[512];
    // Arriving here at all means the caller forwarded the row the previous
    // step handed out -- `logits` is that forward's result. Anything still
    // outstanding when engine_gen_end runs was abandoned instead.
    e->pending_pos = -1;
    if (!((e->gen_max < 0 || e->gen_count < e->gen_max) && e->pos < e->m->n_ctx))
        return ENGINE_STEP_DONE;
    lp_pre pre;
    pre.lse = 0; pre.n_snap = 0;
    bool want_lp = e->lp_cap && e->lp_count < e->lp_cap;
    if (want_lp) lp_capture_pre(e, logits, &pre);
    if (e->cl_cap) cl_capture(e, logits);
    int tok = sample_pick(e->smp, (float *)logits, e->m->n_vocab,
                          e->schema ? schema_ok :
                          e->json_mode ? json_ok : NULL, e);
    if (tok < 0) { // -1: no valid continuation (clean stop); -2: allocation error
        if (tok == -2) e->oom = true;
        e->hit_stop = true;
        return ENGINE_STEP_DONE;
    }
    sampler_accept(e->smp, tok);
    if (debug_tokens()) fprintf(stderr, " %d", tok);
    if (is_stop(e, tok) && !e->ignore_eos) {
        e->hit_stop = true;
        return ENGINE_STEP_DONE;
    }
    if (want_lp) {
        float raw = logits[tok]; // unmutated unless in the penalty window
        for (int i = 0; i < pre.n_snap; i++)
            if (pre.snap_ids[i] == tok) { raw = pre.snap[i]; break; }
        e->lp_ids[e->lp_count]    = tok;
        e->lp_chosen[e->lp_count] = raw - pre.lse;
        e->lp_count++;
    }
    int n = tok_decode(e->tok, tok, buf, sizeof(buf));
    bool in_prelude = (e->schema || e->json_mode) &&
                      e->constraint_phase != CP_OUTPUT;
    int rc = e->schema && n > 0
               ? constraint_accept(e, true, buf, n, cb, ud)
           : e->json_mode && n > 0
               ? constraint_accept(e, false, buf, n, cb, ud)
           : cb && n > 0 ? cb(ud, buf, n) : 0;
    if (!rc && (e->schema || e->json_mode))
        rc = constraint_control_accept(e, tok, e->schema != NULL, cb, ud);
    if (!rc && e->schema && n == 0 && constraint_spelling_ok(e, tok, true)) {
        const char *sp = tok_raw(e->tok, tok);
        rc = constraint_accept(e, true, sp, (int)strlen(sp), cb, ud);
    }
    if (rc != 0) { e->gen_count++; return ENGINE_STEP_DONE; } // client gone
    e->gen_count++;
    if (in_prelude && (e->constraint_phase == CP_PROBE ||
                       e->constraint_phase == CP_THINK) && e->prelude_max > 0 &&
        ++e->prelude_count >= e->prelude_max) {
        e->prelude_exhausted = true;
        // A model that opens a thinking block and never closes it used to end
        // the request here, having produced nothing: the caller asked for a
        // schema-constrained payload and got an empty document. Measured on
        // e4b-q4km (gemma4, prelude tags <|channel>thought / <channel|>), two
        // of four tool prompts did exactly this -- prelude_max is max_new/2,
        // so -n 200 burned 100 tokens thinking and returned one newline.
        //
        // Under an active constraint the prelude is not the deliverable, so
        // the budget cap now CLOSES the prelude instead of ending the turn:
        // phase moves to output, the payload validator is reset, and the
        // remaining half of the budget goes on the JSON that was asked for.
        // prelude_exhausted still records why, so the server keeps reporting
        // "reasoning_limit".
        // No `else`. `in_prelude` above already required a constraint, so an
        // unconstrained turn never reaches this bound at ALL -- it runs to
        // max_tokens like any other. There WAS an else here returning
        // ENGINE_STEP_DONE, and it was unreachable; on 2026-08-15 it was read
        // as the unconstrained POLICY and presented to the owner as one, which
        // is the specific harm of dead code that looks like a decision.
        if (constraint_finish_think(e, e->schema != NULL, cb, ud) != 0)
            return ENGINE_STEP_DONE;
    }
    if ((e->schema && constraint_done(e, true)) ||
        (!e->schema && e->json_mode && constraint_done(e, false))) {
        e->hit_stop = true;
        return ENGINE_STEP_DONE;
    }
    if (e->hist && e->pos < e->m->n_ctx) e->hist[e->pos] = tok;
    *next_tok = (int32_t)tok;
    *next_pos = e->pos++;
    e->pending_pos = *next_pos;
    return ENGINE_STEP_MORE;
}

int engine_gen_end(engine *e, gen_cb cb, void *ud, double *gen_time) {
    // A step whose forward never happened (a deadline, a failed microbatch, a
    // shutdown -- see sched_generate) owns no KV row: give it back before
    // anything reads e->pos as the extent of this sequence's cache.
    if (e->pending_pos >= 0) {
        e->pos = e->pending_pos;
        if (e->dpos > e->pos) e->dpos = e->pos;
        e->pending_pos = -1;
    }
    constraint_close(e, cb, ud);
    if (gen_time) *gen_time = now_s() - e->gen_t0;
    return e->gen_count;
}

// The speculative walk owns its own forwards and cannot interleave with
// logprob/choice-logprob capture (both hook the solo step path).
bool engine_wants_spec(const engine *e) {
    if (e->lp_cap || e->cl_cap) return false;
    if (e->dm) return true;
    return e->gram_ff && (e->schema || e->json_mode);
}

int engine_generate(engine *e, float *logits, int max_new,
                    gen_cb cb, void *ud, double *gen_time) {
    if (engine_wants_spec(e)) {
        // the speculative walk has exits that bypass constraint_close (a
        // completed document, a client that left); release here so no path
        // out of a generation keeps the recording alive
        int n = engine_generate_spec(e, logits, max_new, cb, ud, gen_time);
        cdoc_release(e);
        return n;
    }
    engine_gen_begin(e, max_new);
    int32_t tok; int pos;
    while (engine_gen_step(e, logits, cb, ud, &tok, &pos) == ENGINE_STEP_MORE)
        logits = model_forward(e->m, tok, pos);
    return engine_gen_end(e, cb, ud, gen_time);
}
