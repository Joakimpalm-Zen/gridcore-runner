// Persistent and forkable KV prefixes (Phase 7).
//
// The feature's whole value is skipping prefill; its whole risk is that a
// forked prefix does not actually correspond to the prompt it was installed
// for, in which case generation is silently conditioned on somebody else's
// context and nothing anywhere reports an error. So the gate here is
// *identity*, not speed:
//
//   a sequence that forks a cached prefix must produce, bit for bit, the
//   logits it would have produced by prefilling the whole prompt itself.
//
// Everything else in this file exists to make that claim hard to pass by
// accident: the fork happens on a *different* model instance than the one that
// stored it, one fork is partial (fewer tokens than were stored), and the
// negative cases check that a snapshot from an incompatible context length or
// KV type is refused rather than reinterpreted.
//
// The first test is not about the cache at all. It pins the invariant the
// cache is built on: e->hist always describes the tokens whose KV really
// occupies [0, e->pos). engine_rewind and the prefix cache both trust that,
// and it used to be false on the context-overflow path.
#include "runner.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *g_path = "test.gguf";
static int g_fail = 0;

static void ck(int cond, const char *what) {
    if (!cond) { fprintf(stderr, "FAIL: %s\n", what); g_fail = 1; }
    else        fprintf(stderr, "ok: %s\n", what);
}

enum { CTX = 96, BATCH = 4 };

static model_params base_params(void) {
    model_params p;
    memset(&p, 0, sizeof(p));
    p.gpu_mode  = GPU_AUTO;
    p.n_threads = 1;
    p.n_ctx     = CTX;
    p.n_batch   = BATCH;
    return p;
}

// A loaded model plus the tokenizer/sampler/engine a server slot wraps it in.
typedef struct { model_t m; tokenizer tok; sampler smp; engine e; } slot;

static bool slot_open(slot *s, const model_params *p) {
    memset(s, 0, sizeof(*s));
    if (!model_load(&s->m, g_path, p)) return false;
    if (!tokenizer_init(&s->tok, &s->m.gf)) { model_free(&s->m); return false; }
    s->smp.temp = 0;              // greedy; logits are compared directly anyway
    s->smp.repeat_penalty = 1.0f;
    s->smp.rng = 1;
    engine_init(&s->e, &s->m, &s->tok, &s->smp);
    return true;
}

static void slot_close(slot *s) {
    free(s->e.hist);
    tokenizer_free(&s->tok);
    model_free(&s->m);
}

// deterministic pseudo-token stream inside the model's vocabulary
static void fill_tokens(int32_t *t, int n, int vocab, uint32_t seed) {
    for (int i = 0; i < n; i++) {
        seed = seed * 1664525u + 1013904223u;
        t[i] = (int32_t)(seed % (uint32_t)(vocab > 2 ? vocab - 1 : 1)) + 1;
    }
}

// --------------------------------------------------------------- hist coherence
//
// engine_feed used to write the whole token run into e->hist up front and only
// if the run fit the context, then feed it batch by batch. A run that did NOT
// fit therefore skipped the write entirely while the loop still advanced
// e->pos by every batch that did fit, leaving hist describing the *previous*
// request over KV rows belonging to this one. engine_rewind then handed the
// next request a "matching" prefix that matched nothing, which is exactly the
// silent-conditioning failure the prefix cache must never have.
static void test_hist_tracks_kv_on_overflow(void) {
    slot s;
    model_params p = base_params();
    if (!slot_open(&s, &p)) { ck(0, "load model"); return; }

    int32_t a[CTX], b[CTX + BATCH * 3];
    fill_tokens(a, CTX, s.m.n_vocab, 1);
    fill_tokens(b, CTX + BATCH * 3, s.m.n_vocab, 2);

    engine_reset(&s.e);
    ck(engine_feed(&s.e, a, CTX) != NULL, "a full-context feed succeeds");
    ck(s.e.pos == CTX, "pos is at the context end");

    // now a run that cannot fit: some of it is fed, then the feed fails
    engine_reset(&s.e);
    ck(engine_feed(&s.e, b, CTX + BATCH * 3) == NULL, "an oversized feed fails");
    ck(s.e.pos > 0, "the batches that fitted were still fed");

    bool coherent = true;
    for (int i = 0; i < s.e.pos; i++)
        if (s.e.hist[i] != b[i]) coherent = false;
    ck(coherent, "hist describes the tokens whose KV is really in [0, pos)");

    slot_close(&s);
}

// --------------------------------------------------------- the abandoned step
//
// engine_gen_step hands the caller a token and the KV row its forward will
// occupy, and counts that row as occupied the moment it does. A caller that
// abandons the loop instead of forwarding -- sched_generate does, on a request
// deadline, a failed microbatch, or shutdown -- would otherwise leave e->pos
// covering a row nothing ever wrote, over a hist entry naming the token it
// would have held. That is the same lie the overflow path above used to tell,
// and the next request's rewind believes it: it "keeps" the phantom row and
// decodes against KV nobody computed.
static void test_abandoned_step_drops_its_row(void) {
    model_params p = base_params();
    slot warm, cold;
    if (!slot_open(&warm, &p) || !slot_open(&cold, &p)) {
        ck(0, "load two instances for the abandoned step");
        return;
    }

    // N is a multiple of BATCH so the continuation's last chunk covers the same
    // positions in both instances: this is a KV claim, not a batch-shape one.
    enum { N = 20 };
    int32_t prompt[N + 2];
    fill_tokens(prompt, N, warm.m.n_vocab, 31);

    engine_reset(&warm.e);
    float *lg = engine_feed(&warm.e, prompt, N);
    ck(lg != NULL, "prefill before the abandoned step");

    engine_gen_begin(&warm.e, 8);
    int32_t tok = 0;
    int pos = 0;
    ck(engine_gen_step(&warm.e, lg, NULL, NULL, &tok, &pos) == ENGINE_STEP_MORE,
       "the first step asks for a forward");
    // the caller's deadline expires here; that forward is never issued
    engine_gen_end(&warm.e, NULL, NULL, NULL);
    ck(warm.e.pos == N, "an abandoned step gives its KV row back");

    // the observable consequence: the session's next request replays the token
    // that was delivered, and a rewind that kept the phantom row decodes
    // against KV nobody computed
    prompt[N] = tok;
    fill_tokens(prompt + N + 1, 1, warm.m.n_vocab, 41);
    int keep = engine_rewind(&warm.e, prompt, N + 2);
    float *wl = engine_feed(&warm.e, prompt + keep, N + 2 - keep);
    engine_reset(&cold.e);
    float *cl = engine_feed(&cold.e, prompt, N + 2);
    int diffs = (!wl || !cl) ? -1 : 0;
    if (wl && cl)
        for (int i = 0; i < warm.m.n_vocab; i++) if (wl[i] != cl[i]) diffs++;
    ck(diffs == 0, "a rewind after an abandoned step matches a cold prefill");

    slot_close(&warm);
    slot_close(&cold);
}

// ------------------------------------------------------------------ the gate
//
// Store a prefix from one model instance, fork it into another, and require
// the forked instance's logits to be bit-identical to a cold full prefill.
static void test_fork_is_bit_identical(void) {
    model_params p = base_params();
    slot warm, cold, forked;
    if (!slot_open(&warm, &p) || !slot_open(&cold, &p) || !slot_open(&forked, &p)) {
        ck(0, "load three instances");
        return;
    }
    prefix_cache_clear();
    prefix_cache_configure(64u * 1024 * 1024, 3600);

    enum { SHARED = 48, TAIL = 11 };
    int32_t prompt[SHARED + TAIL];
    fill_tokens(prompt, SHARED + TAIL, warm.m.n_vocab, 7);

    // 1. the warm instance prefills the shared block and publishes it
    engine_reset(&warm.e);
    double t0 = now_s();
    ck(engine_feed(&warm.e, prompt, SHARED) != NULL, "warm prefill");
    engine_prefix_publish(&warm.e, prompt, SHARED, SHARED, now_s() - t0);

    prefix_cache_stats st;
    prefix_cache_stats_get(&st);
    ck(st.entries == 1, "the published prefix is resident");
    ck(st.bytes > 0, "the cache reports its snapshot bytes");

    // 2. the cold instance prefills the whole prompt with no reuse at all
    engine_reset(&cold.e);
    float *cold_logits = engine_feed(&cold.e, prompt, SHARED + TAIL);
    ck(cold_logits != NULL, "cold prefill");
    float *ref = malloc(sizeof(float) * (size_t)cold.m.n_vocab);
    memcpy(ref, cold_logits, sizeof(float) * (size_t)cold.m.n_vocab);

    // 3. a third, never-used instance forks the cached prefix and feeds only
    //    the tail: different model_t, different KV cache, and it never saw a
    //    single one of the shared tokens.
    prefix_reuse r = engine_prefix_reuse(&forked.e, prompt, SHARED + TAIL);
    ck(r.forked == SHARED, "the whole stored prefix is forked into a cold slot");
    ck(r.keep == SHARED, "prefill starts after the forked prefix");
    float *fl = engine_feed(&forked.e, prompt + r.keep, SHARED + TAIL - r.keep);
    ck(fl != NULL, "the forked instance feeds the remaining tail");

    int diffs = 0;
    for (int i = 0; i < forked.m.n_vocab; i++)
        if (ref[i] != fl[i]) diffs++;
    ck(diffs == 0, "forked logits are bit-identical to a cold full prefill");

    // hist must describe the forked rows too, or the next request's rewind is
    // back to trusting a lie
    bool coherent = true;
    for (int i = 0; i < SHARED + TAIL; i++)
        if (forked.e.hist[i] != prompt[i]) coherent = false;
    ck(coherent, "a forked prefix leaves hist describing the forked tokens");

    // 4. a PARTIAL fork: a prompt that diverges inside the stored run must
    //    install only the leading tokens that really match.
    enum { DIVERGE_AT = SHARED - 9 };
    int32_t other[SHARED + TAIL];
    memcpy(other, prompt, sizeof(other));
    other[DIVERGE_AT] = other[DIVERGE_AT] % (warm.m.n_vocab - 2) + 1;
    if (other[DIVERGE_AT] == prompt[DIVERGE_AT]) other[DIVERGE_AT] += 1;

    slot part, cold2;
    if (slot_open(&part, &p) && slot_open(&cold2, &p)) {
        prefix_reuse pr = engine_prefix_reuse(&part.e, other, SHARED + TAIL);
        ck(pr.forked == DIVERGE_AT,
           "a partial match forks exactly the tokens that match");
        float *pl = engine_feed(&part.e, other + pr.keep,
                                SHARED + TAIL - pr.keep);
        float *c2 = engine_feed(&cold2.e, other, SHARED + TAIL);
        int d2 = 0;
        for (int i = 0; i < part.m.n_vocab; i++) if (pl[i] != c2[i]) d2++;
        ck(d2 == 0, "a partially forked prefix still matches a cold prefill");
        slot_close(&cold2);
        slot_close(&part);
    }

    prefix_cache_stats_get(&st);
    ck(st.hits >= 2 && st.tokens_reused >= SHARED,
       "hits and reused tokens are counted");

    free(ref);
    slot_close(&warm); slot_close(&cold); slot_close(&forked);
}

// ---------------------------------------------------------------- the key
//
// The carried-over hazard: a prefix may only be shared between contexts whose
// tokenization AND cache bytes are identical. Text equality is not enough.
static void test_key_rejects_incompatible_snapshots(void) {
    prefix_cache_clear();
    prefix_cache_configure(64u * 1024 * 1024, 3600);

    model_params p = base_params();
    slot a;
    if (!slot_open(&a, &p)) { ck(0, "load keyed instance"); return; }

    enum { N = 40 };
    int32_t toks[N];
    fill_tokens(toks, N, a.m.n_vocab, 11);
    engine_reset(&a.e);
    engine_feed(&a.e, toks, N);
    engine_prefix_publish(&a.e, toks, N, N, 0.01);

    // same file, same text, different context geometry: the KV layout and the
    // reservations behind it both differ
    model_params q = base_params();
    q.n_ctx = CTX * 2;
    slot b;
    if (slot_open(&b, &q)) {
        ck(b.e.model_key != a.e.model_key, "context geometry is part of the key");
        prefix_reuse r = engine_prefix_reuse(&b.e, toks, N);
        ck(r.forked == 0, "a different context length does not fork");
        slot_close(&b);
    }

    // same file, same text, q8 KV: the cache bytes are a different format
    // entirely. test.gguf's head_dim may forbid q8, in which case model_load
    // clears the flag — assert on the flag that actually took effect, not the
    // one that was requested.
    model_params k8 = base_params();
    k8.kv_q8 = true;
    slot c;
    if (slot_open(&c, &k8)) {
        if (c.m.kv_q8) {
            ck(c.e.model_key != a.e.model_key, "KV type is part of the key");
            prefix_reuse r = engine_prefix_reuse(&c.e, toks, N);
            ck(r.forked == 0, "a different KV element type does not fork");
        } else {
            fprintf(stderr, "note: this model cannot store q8 KV; the KV-type "
                            "axis of the key is exercised by --kv q8 models\n");
            ck(c.e.model_key == a.e.model_key,
               "a KV type that did not take effect does not change the key");
        }
        slot_close(&c);
    }

    // an identical second instance DOES share
    slot d;
    if (slot_open(&d, &p)) {
        ck(d.e.model_key == a.e.model_key, "identical instances share a key");
        prefix_reuse r = engine_prefix_reuse(&d.e, toks, N - 1);
        ck(r.forked == N - 2, "an identical instance forks the stored prefix");
        slot_close(&d);
    }
    slot_close(&a);
}

static bool copy_file(const char *from, const char *to) {
    FILE *a = fopen(from, "rb");
    FILE *b = fopen(to, "wb");
    if (!a || !b) {
        if (a) fclose(a);
        if (b) fclose(b);
        return false;
    }
    char buf[8192];
    size_t n;
    bool ok = true;
    while ((n = fread(buf, 1, sizeof(buf), a)) > 0)
        if (fwrite(buf, 1, n, b) != n) { ok = false; break; }
    if (ferror(a)) ok = false;
    fclose(a);
    fclose(b);
    return ok;
}

static bool flip_unsampled_weight_byte(const char *path) {
    gguf_file g;
    if (!gguf_open(&g, path)) return false;
    gguf_tensor *t = gguf_find_tensor(&g, "blk.1.ffn_down.weight");
    if (!t || !t->data || t->nbytes <= 128) {
        gguf_close(&g);
        return false;
    }
    long off = (long)((const uint8_t *)t->data - (const uint8_t *)g.map + 128);
    gguf_close(&g);

    FILE *f = fopen(path, "r+b");
    if (!f) return false;
    if (fseek(f, off, SEEK_SET) != 0) { fclose(f); return false; }
    int c = fgetc(f);
    if (c == EOF) { fclose(f); return false; }
    if (fseek(f, off, SEEK_SET) != 0) { fclose(f); return false; }
    unsigned char b = (unsigned char)c ^ 1u;
    bool ok = fwrite(&b, 1, 1, f) == 1;
    fclose(f);
    return ok;
}

static void test_key_changes_after_in_place_weight_edit(void) {
    const char *orig = g_path;
    const char *tmp = "prefix-key-replaced.gguf";
    if (!copy_file(orig, tmp)) {
        ck(0, "copy model for in-place identity test");
        return;
    }

    g_path = tmp;
    model_params p = base_params();
    slot before;
    if (!slot_open(&before, &p)) {
        ck(0, "load identity fixture before edit");
        g_path = orig;
        remove(tmp);
        return;
    }
    uint64_t old_key = before.e.model_key;
    slot_close(&before);

    ck(flip_unsampled_weight_byte(tmp), "edit an unsampled weight byte in place");

    slot after;
    if (slot_open(&after, &p)) {
        ck(after.e.model_key != old_key,
           "model identity changes after an in-place weight edit");
        slot_close(&after);
    } else {
        ck(0, "load identity fixture after edit");
    }

    g_path = orig;
    remove(tmp);
}

// ------------------------------------------------------------- housekeeping
static void test_eviction(void) {
    model_params p = base_params();
    slot s;
    if (!slot_open(&s, &p)) { ck(0, "load evicting instance"); return; }

    enum { N = 40 };
    int32_t toks[N];
    prefix_cache_stats st;

    // TTL: a prefix older than the TTL is gone, not merely unreported
    prefix_cache_clear();
    prefix_cache_configure(64u * 1024 * 1024, -1.0); // already expired
    fill_tokens(toks, N, s.m.n_vocab, 21);
    engine_reset(&s.e);
    engine_feed(&s.e, toks, N);
    engine_prefix_publish(&s.e, toks, N, N, 0.01);
    slot other;
    if (slot_open(&other, &p)) {
        prefix_reuse r = engine_prefix_reuse(&other.e, toks, N);
        ck(r.forked == 0, "an expired prefix is not forked");
        slot_close(&other);
    }
    prefix_cache_stats_get(&st);
    ck(st.entries == 0 && st.bytes == 0, "an expired prefix is freed");

    // budget: a store that cannot fit evicts, and never exceeds the budget
    prefix_cache_clear();
    size_t one = prefix_cache_entry_bytes(&s.m, N);
    prefix_cache_configure(one * 2 + 1, 3600);
    for (int i = 0; i < 6; i++) {
        fill_tokens(toks, N, s.m.n_vocab, 100u + (uint32_t)i);
        engine_reset(&s.e);
        engine_feed(&s.e, toks, N);
        engine_prefix_publish(&s.e, toks, N, N, 0.01);
        prefix_cache_stats_get(&st);
        ck(st.bytes <= one * 2 + 1, "the cache stays inside its byte budget");
    }
    prefix_cache_stats_get(&st);
    ck(st.entries <= 2, "the budget bounds the entry count");
    ck(st.evictions > 0, "eviction is reported");

    // a budget of zero disables the shared cache without disabling the slot's
    // own rewind
    prefix_cache_clear();
    prefix_cache_configure(0, 3600);
    fill_tokens(toks, N, s.m.n_vocab, 55);
    engine_reset(&s.e);
    engine_feed(&s.e, toks, N);
    engine_prefix_publish(&s.e, toks, N, N, 0.01);
    prefix_cache_stats_get(&st);
    ck(st.entries == 0, "a zero budget stores nothing");
    prefix_reuse r = engine_prefix_reuse(&s.e, toks, N);
    ck(r.keep == N - 1 && r.forked == 0,
       "the slot's own rewind still works with the shared cache off");

    // The half-budget truncation clamp. This path only engages when ONE
    // snapshot rivals the whole budget, which never happens at fixture scale
    // with the 512 MB default -- a 135M model's 40-token entry is a rounding
    // error against it. It is reachable in production (a long prompt on a
    // large model) and was previously uncovered, so drive it directly by
    // shrinking the budget until the fixture is "large" relative to it.
    size_t per_tok = prefix_cache_entry_bytes(&s.m, 1);
    if (per_tok == 0) {
        ck(0, "per-token KV size is measurable");
    } else {
        // budget = 40 tokens' worth => half is 20, so a 40-token store must be
        // truncated to 20 (>= PFX_MIN_TOKENS, so it still stores)
        prefix_cache_clear();
        size_t budget = 40 * per_tok;
        prefix_cache_configure(budget, 3600);
        fill_tokens(toks, N, s.m.n_vocab, 77);
        engine_reset(&s.e);
        engine_feed(&s.e, toks, N);
        engine_prefix_publish(&s.e, toks, N, N, 0.01);
        prefix_cache_stats_get(&st);
        ck(st.entries == 1, "an oversized snapshot is truncated, not dropped");
        ck(st.bytes <= budget / 2,
           "one snapshot never eats more than half the budget");

        // truncation must not poison reuse: a prefix of a prefix is still a
        // valid prefix, so a fresh slot must still fork from the short entry
        slot trunc;
        if (slot_open(&trunc, &p)) {
            prefix_reuse tr = engine_prefix_reuse(&trunc.e, toks, N);
            ck(tr.forked > 0 && tr.forked < N && tr.keep == tr.forked,
               "a truncated snapshot is still a usable prefix");
            slot_close(&trunc);
        }

        // and a budget too small to hold even PFX_MIN_TOKENS stores nothing
        // rather than storing a degenerate zero-length entry
        prefix_cache_clear();
        prefix_cache_configure(20 * per_tok, 3600);
        fill_tokens(toks, N, s.m.n_vocab, 78);
        engine_reset(&s.e);
        engine_feed(&s.e, toks, N);
        engine_prefix_publish(&s.e, toks, N, N, 0.01);
        prefix_cache_stats_get(&st);
        ck(st.entries == 0 && st.bytes == 0,
           "a budget below the minimum prefix stores nothing at all");
    }

    prefix_cache_clear();
    prefix_cache_configure(64u * 1024 * 1024, 600);
    slot_close(&s);
}

int main(int argc, char **argv) {
    if (argc > 1) g_path = argv[1];
    f16_init();
    test_hist_tracks_kv_on_overflow();
    test_abandoned_step_drops_its_row();
    test_fork_is_bit_identical();
    test_key_rejects_incompatible_snapshots();
    test_key_changes_after_in_place_weight_edit();
    test_eviction();
    if (g_fail) { fprintf(stderr, "test-prefix FAILED\n"); return 1; }
    fprintf(stderr, "test-prefix: all checks passed\n");
    return 0;
}
