// Generation engine shared by the CLI and the server: prompt feeding,
// stop-token handling, optional JSON-constrained sampling.
#include "runner.h"

#include "compat.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

double now_s(void) { return plat_now(); }

enum { CP_PROBE, CP_THINK, CP_OUTPUT };

// everything that decides what this engine's KV bytes mean; see the shared
// prefix cache below, which is the only thing that consumes it
static uint64_t model_identity(const model_t *m, const tokenizer *tok);

static void constraint_reset(engine *e) {
    e->constraint_phase = e->m && e->m->think_open && e->m->think_close
                            ? CP_PROBE : CP_OUTPUT;
    e->constraint_tag_possible = e->constraint_phase == CP_PROBE;
    e->constraint_tag_match = 0;
    e->constraint_close_match = 0;
}

bool engine_init(engine *e, model_t *m, tokenizer *tok, sampler *smp) {
    free(e->hist); // slot engines are re-inited on model swap; e must be zeroed
    memset(e, 0, sizeof(*e));
    e->m = m;
    e->tok = tok;
    e->smp = smp;
    e->stop_ids[e->n_stop++] = tok->eos_id;
    static const char *stops[] = { "<|im_end|>", "<|eot_id|>", "<|end_of_text|>",
                                   "<|endoftext|>", "</s>",
                                   // gemma turn terminators (gemma1-3 / gemma4)
                                   "<end_of_turn>", "<turn|>",
                                   // phi3 ends assistant turns with <|end|>,
                                   // which is not its declared eos_token_id
                                   "<|end|>" };
    for (size_t i = 0; i < sizeof(stops) / sizeof(*stops); i++) {
        int id = tok_find(tok, stops[i]);
        if (id < 0 || e->n_stop >= (int)(sizeof(e->stop_ids) / sizeof(*e->stop_ids)))
            continue;
        bool dup = false;
        for (int j = 0; j < e->n_stop; j++) if (e->stop_ids[j] == id) dup = true;
        if (!dup) e->stop_ids[e->n_stop++] = id;
    }
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
// The weights are sampled rather than digested whole: two finetunes of one
// architecture share every tensor name, dtype and shape, so the table alone
// would collide. A slice of each layer's Q and down projections separates them
// at a cost that does not scale with model size.
static uint64_t model_identity(const model_t *m, const tokenizer *tok) {
    uint64_t h = 0xCBF29CE484222325ull;
    h = h64s(h, "runner.prefix.v1");
    h = h64s(h, m->path);
    h = h64s(h, m->arch);
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
    for (int l = 0; l < m->n_layer; l++)
        t += 2 * (size_t)n * model_kv_row_bytes(m, l);
    return t;
}

static void pfx_save(const model_t *m, uint8_t *dst, int n) {
    size_t off = 0;
    for (int l = 0; l < m->n_layer; l++) {
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
    if (hit && best >= PFX_MIN_TOKENS && best > r.keep) {
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
        else pp = &p->next;
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
    dmp.kv_q8 = false;
    dmp.gpu_mode = GPU_AUTO;   // a small draft usually fits VRAM whole
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

static bool constraint_payload_feed(engine *e, bool schema,
                                    const char *bytes, int n) {
    if (schema) {
        sval tmp = e->sv;
        if (!sval_feed(&tmp, bytes, n)) return false;
        e->sv = tmp;
    } else {
        jsonv tmp = e->jv;
        if (!jsonv_feed(&tmp, bytes, n)) return false;
        e->jv = tmp;
    }
    return true;
}

static void constraint_payload_reset(engine *e, bool schema) {
    if (schema) sval_init(&e->sv, e->schema);
    else        jsonv_init(&e->jv);
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

// validity filter for constrained mode. Before a declared thinking block
// closes, ordinary tokens are allowed; stop/control tokens remain valid only
// inside that prelude or after the constrained payload is complete.
static bool constraint_token_ok(engine *e, int id, bool schema) {
    if (is_stop(e, id) || tok_is_control(e->tok, id))
        return e->constraint_phase == CP_THINK || constraint_done(e, schema);
    char buf[512];
    int n = tok_decode(e->tok, id, buf, sizeof(buf));
    if (n == 0)
        return e->constraint_phase == CP_THINK || constraint_done(e, schema);
    engine tmp = *e;
    int visible;
    return constraint_feed(&tmp, schema, buf, n, &visible);
}

static bool schema_ok(void *ud, int id) {
    return constraint_token_ok(ud, id, true);
}

static bool json_ok(void *ud, int id) {
    return constraint_token_ok(ud, id, false);
}

static int constraint_accept(engine *e, bool schema, const char *bytes, int n,
                             gen_cb cb, void *ud) {
    int visible;
    if (!constraint_feed(e, schema, bytes, n, &visible)) return 1;
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
    for (int i = 0; i < s->n_recent; i++) {
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

// speculative decoding: sampler-equality verification (llama.cpp-style) —
// sample each position from the TARGET's logits with the full sampler chain;
// a draft is accepted when the sampled token equals it, so output follows
// exactly the same distribution as the non-speculative path
// budget expired mid-document: complete it so constrained output stays valid
// JSON (the caller's finish_reason stays "length")
static void constraint_close(engine *e, gen_cb cb, void *ud) {
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
}

static int engine_generate_spec(engine *e, float *logits, int max_new,
                                gen_cb cb, void *ud, double *gen_time) {
    char buf[512];
    int n_gen = 0;
    e->hit_stop = false;
    e->oom = false;
    e->lp_count = 0;
    double t0 = now_s();
    model_t *m = e->m, *dm = e->dm;
    int K = e->draft_k;
    if (K < 1) K = 1;
    if (K > m->spec_batch - 1) K = m->spec_batch - 1;
    int32_t d[16];
    float *dl = NULL; // draft logits for position dpos
    // Even under JSON/schema constraints, speculation stays target-exact:
    // the draft proposes, but only target-sampled tokens feed the validator.
    sample_ok_fn ok = e->schema ? schema_ok : e->json_mode ? json_ok : NULL;
    int st_rounds = 0, st_drafted = 0, st_accepted = 0;
    #define SPEC_STATS() fprintf(stderr, \
        "spec: %d rounds, %d drafted, %d accepted (%.2f tok/round)\n", \
        st_rounds, st_drafted, st_accepted, \
        st_rounds ? (double)n_gen / st_rounds : 0)

    while ((max_new < 0 || n_gen < max_new) && e->pos < m->n_ctx) {
        st_rounds++;
        // catch the draft up on tokens accepted since its last position
        while (e->dpos < e->pos) {
            int chunk = e->pos - e->dpos < dm->n_batch ? e->pos - e->dpos
                                                       : dm->n_batch;
            if (e->dpos + chunk > dm->n_ctx) { dl = NULL; break; }
            dl = model_forward_batch(dm, e->hist + e->dpos, chunk, e->dpos,
                                     e->dpos + chunk == e->pos);
            e->dpos += chunk;
        }
        // draft up to K tokens greedily (nd == 0 degrades to plain decoding)
        int nd = 0;
        while (dl && nd < K && e->pos + nd + 1 < m->n_ctx &&
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
            if (getenv("RUNNER_DEBUG_TOKENS")) fprintf(stderr, " %d", tok);
            if (is_stop(e, tok) && !e->ignore_eos) {
                e->hit_stop = true;
                e->pos += i; // keep the accepted drafts' KV
                if (e->dpos > e->pos) e->dpos = e->pos;
                goto done;
            }
            int n = tok_decode(e->tok, tok, buf, sizeof(buf));
            int rc = e->schema && n > 0
                       ? constraint_accept(e, true, buf, n, cb, ud)
                   : e->json_mode && n > 0
                       ? constraint_accept(e, false, buf, n, cb, ud)
                   : cb && n > 0 ? cb(ud, buf, n) : 0;
            n_gen++;
            bool constrained_done = e->schema
                                      ? constraint_done(e, true)
                                      : e->json_mode && constraint_done(e, false);
            if (i < nd && tok == d[i] && rc == 0) {
                e->hist[e->pos + i] = tok; // accepted: its KV is already right
                st_accepted++;
                if (constrained_done) {
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
                if (gen_time) *gen_time = now_s() - t0;
                SPEC_STATS();
                return n_gen;
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
    e->gen_t0    = now_s();
}

int engine_gen_step(engine *e, const float *logits, gen_cb cb, void *ud,
                    int32_t *next_tok, int *next_pos) {
    char buf[512];
    if (!((e->gen_max < 0 || e->gen_count < e->gen_max) && e->pos < e->m->n_ctx))
        return ENGINE_STEP_DONE;
    lp_pre pre;
    pre.lse = 0; pre.n_snap = 0;
    bool want_lp = e->lp_cap && e->lp_count < e->lp_cap;
    if (want_lp) lp_capture_pre(e, logits, &pre);
    int tok = sample_pick(e->smp, (float *)logits, e->m->n_vocab,
                          e->schema ? schema_ok :
                          e->json_mode ? json_ok : NULL, e);
    if (tok < 0) { // -1: no valid continuation (clean stop); -2: allocation error
        if (tok == -2) e->oom = true;
        e->hit_stop = true;
        return ENGINE_STEP_DONE;
    }
    sampler_accept(e->smp, tok);
    if (getenv("RUNNER_DEBUG_TOKENS")) fprintf(stderr, " %d", tok);
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
    int rc = e->schema && n > 0
               ? constraint_accept(e, true, buf, n, cb, ud)
           : e->json_mode && n > 0
               ? constraint_accept(e, false, buf, n, cb, ud)
           : cb && n > 0 ? cb(ud, buf, n) : 0;
    if (rc != 0) { e->gen_count++; return ENGINE_STEP_DONE; } // client gone
    e->gen_count++;
    if ((e->schema && constraint_done(e, true)) ||
        (!e->schema && e->json_mode && constraint_done(e, false))) {
        e->hit_stop = true;
        return ENGINE_STEP_DONE;
    }
    if (e->hist && e->pos < e->m->n_ctx) e->hist[e->pos] = tok;
    *next_tok = (int32_t)tok;
    *next_pos = e->pos++;
    return ENGINE_STEP_MORE;
}

int engine_gen_end(engine *e, gen_cb cb, void *ud, double *gen_time) {
    constraint_close(e, cb, ud);
    if (gen_time) *gen_time = now_s() - e->gen_t0;
    return e->gen_count;
}

int engine_generate(engine *e, float *logits, int max_new,
                    gen_cb cb, void *ud, double *gen_time) {
    if (e->dm && e->lp_cap == 0)
        return engine_generate_spec(e, logits, max_new, cb, ud, gen_time);
    engine_gen_begin(e, max_new);
    int32_t tok; int pos;
    while (engine_gen_step(e, logits, cb, ud, &tok, &pos) == ENGINE_STEP_MORE)
        logits = model_forward(e->m, tok, pos);
    return engine_gen_end(e, cb, ud, gen_time);
}
