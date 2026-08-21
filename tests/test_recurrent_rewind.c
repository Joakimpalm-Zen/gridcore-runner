// Recurrent-state cache seam (SSM tracer 4) — the keystone gate.
//
// The KV cache rests on Fact 1: any prefix of a stored sequence is a valid
// snapshot, so keeping rows [0, keep) puts attention at exactly position keep.
// Recurrent (SSM) state has NO such property — it is a fold over the whole
// prefix, so the state after position n cannot reproduce the state after k<n.
// Before this tracer, engine_rewind moved e->pos back but left the recurrent
// fold as-of the OLD end position, so a rewound decode read the wrong state and
// diverged. This file pins the fix with the only bar that matters here:
//
//   a decode that rewinds and replays produces, BIT FOR BIT, the logits a
//   decode that never rewound produces.
//
// Two independent gates, on the qwen35 (Ornith) recurrent fixture:
//
//   1. The snapshot/restore PRIMITIVE, isolated at the model_forward level:
//      snapshot the fold at a boundary, run a divergent detour that mutates it,
//      restore, and require the continuation to match a run that never detoured.
//      This is the spec-decode / abandoned-step rollback boundary and the shape
//      the CUDA q35_*_prev pattern already mirrors one level down.
//
//   2. engine_rewind end to end: a warm slot that rewinds across a divergence
//      (no snapshot at the rewind point, so the fold cannot be sliced and the
//      recurrent layers must recompute from 0) must match a cold slot that fed
//      the diverged prompt fresh.
//
// Proven RED first: with the restore/rewind wiring disabled the fold stays
// stale and both gates fail; the wiring makes them green. The fixture has 3
// recurrent layers + 1 full-attention layer (full_attention_interval 4), so the
// recurrent path is genuinely exercised.
#include "runner.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *g_path = "test-ornith.gguf";
static int g_fail = 0;

static void ck(int cond, const char *what) {
    if (!cond) { fprintf(stderr, "FAIL: %s\n", what); g_fail = 1; }
    else        fprintf(stderr, "ok: %s\n", what);
}

enum { CTX = 64, BATCH = 4 };

static model_params base_params(void) {
    model_params p;
    memset(&p, 0, sizeof(p));
    p.gpu_mode  = GPU_OFF;   // CPU-only: the recurrent path is CPU-forced
    p.n_threads = 1;
    p.n_ctx     = CTX;
    p.n_batch   = BATCH;
    return p;
}

typedef struct { model_t m; tokenizer tok; sampler smp; engine e; } slot;

static bool slot_open(slot *s, const model_params *p) {
    memset(s, 0, sizeof(*s));
    if (!model_load(&s->m, g_path, p)) return false;
    if (!tokenizer_init(&s->tok, &s->m.gf)) { model_free(&s->m); return false; }
    s->smp.temp = 0;
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

// Copy the just-returned last-token logits so a later forward can't clobber them.
static float *snap_logits(const model_t *m, const float *lg) {
    float *out = malloc(sizeof(float) * (size_t)m->n_vocab);
    if (out && lg) memcpy(out, lg, sizeof(float) * (size_t)m->n_vocab);
    return out;
}

static int logits_differ(const model_t *m, const float *a, const float *b) {
    if (!a || !b) return -1;
    int diffs = 0;
    for (int i = 0; i < m->n_vocab; i++) if (a[i] != b[i]) diffs++;
    return diffs;
}

// ---- gate 1: the snapshot/restore primitive, isolated ----------------------
//
// Drive model_forward directly so nothing but the recurrent fold can explain a
// divergence. A reference run folds [s0..s4] straight through. The test run
// folds [s0,s1,s2], snapshots at pos 3, folds a DIFFERENT detour [d0,d1] over
// positions 3-4 (which the SSM state absorbs), restores the snapshot, then folds
// [s3,s4] — and must land on the reference's position-4 logits bit for bit.
static void test_snapshot_restore_roundtrip(void) {
    model_params p = base_params();
    slot ref, tst;
    if (!slot_open(&ref, &p) || !slot_open(&tst, &p)) {
        ck(0, "load two instances for the primitive gate");
        return;
    }
    ck(model_has_recurrent(&ref.m), "the fixture has recurrent (SSM) layers");

    const int32_t seq[5]    = { 10, 20, 30, 40, 50 };
    const int32_t detour[2] = { 111, 222 };

    // reference: fold the whole sequence, capture logits entering nothing beyond
    float *rl = NULL;
    for (int i = 0; i < 5; i++) rl = model_forward(&ref.m, seq[i], i);
    float *ref_l = snap_logits(&ref.m, rl);

    // test: fold a prefix, snapshot, detour, restore, replay the tail
    for (int i = 0; i < 3; i++) model_forward(&tst.m, seq[i], i);
    ck(model_recurrent_snapshot(&tst.m, 3), "snapshot the fold at pos 3");
    model_forward(&tst.m, detour[0], 3);   // detour mutates the fold + KV rows 3,4
    model_forward(&tst.m, detour[1], 4);
    ck(model_recurrent_restore(&tst.m, 3), "restore hits the pos-3 snapshot");
    // a stale/missing snapshot restores nothing: prove the key is honored
    ck(!model_recurrent_restore(&tst.m, 4), "no snapshot at pos 4 declines");
    model_forward(&tst.m, seq[3], 3);       // overwrites KV rows 3,4 with s3,s4
    float *tl = model_forward(&tst.m, seq[4], 4);
    float *tst_l = snap_logits(&tst.m, tl);

    int diffs = logits_differ(&ref.m, ref_l, tst_l);
    ck(diffs == 0, "restore makes the replay bit-identical to no detour");

    free(ref_l); free(tst_l);
    slot_close(&ref); slot_close(&tst);
}

// ---- gate 2: engine_rewind across a divergence, end to end -----------------
//
// A warm slot folds a long prompt, then a second request in the session shares
// only a short leading run and diverges. engine_rewind finds no snapshot at the
// divergence point, so the fold cannot be sliced and the recurrent layers must
// recompute from 0. Its logits must equal a cold slot fed the diverged prompt.
static void test_rewind_divergence_matches_cold(void) {
    model_params p = base_params();
    slot warm, cold;
    if (!slot_open(&warm, &p) || !slot_open(&cold, &p)) {
        ck(0, "load two instances for the rewind gate");
        return;
    }

    const int32_t p_long[5] = { 10, 20, 30, 40, 50 };   // first request
    const int32_t p_new[4]  = { 10, 20, 77, 88 };       // shares [10,20], diverges

    engine_reset(&warm.e);
    ck(engine_feed(&warm.e, p_long, 5) != NULL, "warm folds the first request");
    // second request on the same slot: rewind to the shared prefix, feed the tail
    int keep = engine_rewind(&warm.e, p_new, 4);
    float *wl = engine_feed(&warm.e, p_new + keep, 4 - keep);
    float *warm_l = snap_logits(&warm.m, wl);

    engine_reset(&cold.e);
    float *cl = engine_feed(&cold.e, p_new, 4);
    float *cold_l = snap_logits(&cold.m, cl);

    int diffs = logits_differ(&warm.m, warm_l, cold_l);
    ck(diffs == 0, "a rewound+replayed decode matches a cold one");

    free(warm_l); free(cold_l);
    slot_close(&warm); slot_close(&cold);
}

// ---- gate 3: the SHARED prefix cache restores the fold on an exact hit -------
//
// A warm slot folds SHARED tokens and PUBLISHES them to the cross-request cache
// (KV rows + the appended fold blob). A fresh forked slot reuses the exact same
// prefix: tracer 5 forks it — installing the KV AND restoring the fold — so it
// only feeds the tail, and its logits must equal a cold slot fed the whole
// prompt. The fork must actually happen (keep == SHARED); without the fold blob
// a recurrent model declines the fork (keep == 0), which is the RED state.
static void test_prefix_cache_fold_restore_matches_cold(void) {
    enum { SHARED = 24, TAIL = 3 };   // SHARED >= PFX_MIN_TOKENS (16)
    model_params p = base_params();
    slot warm, cold, forked;
    if (!slot_open(&warm, &p) || !slot_open(&cold, &p) || !slot_open(&forked, &p)) {
        ck(0, "load three instances for the prefix-cache gate");
        return;
    }
    prefix_cache_clear();

    int32_t prompt[SHARED + TAIL];
    for (int i = 0; i < SHARED + TAIL; i++) prompt[i] = 10 + i;

    // warm: fold the shared prefix and publish it (KV + fold blob at pos SHARED)
    engine_reset(&warm.e);
    ck(engine_feed(&warm.e, prompt, SHARED) != NULL, "warm folds the shared prefix");
    engine_prefix_publish(&warm.e, prompt, SHARED, SHARED, 0.0);

    // cold: fold the whole prompt for the reference logits
    engine_reset(&cold.e);
    float *cl = engine_feed(&cold.e, prompt, SHARED + TAIL);
    float *cold_l = snap_logits(&cold.m, cl);

    // forked: reuse the exact prefix (fork restores KV + fold), feed only the tail
    engine_reset(&forked.e);
    prefix_reuse r = engine_prefix_reuse(&forked.e, prompt, SHARED + TAIL);
    ck(r.keep == SHARED, "the recurrent model forks the exact prefix (fold restored)");
    float *fl = engine_feed(&forked.e, prompt + r.keep, SHARED + TAIL - r.keep);
    float *fork_l = snap_logits(&forked.m, fl);

    int diffs = logits_differ(&forked.m, fork_l, cold_l);
    ck(diffs == 0, "a fold-restored fork matches a cold decode bit for bit");

    free(cold_l); free(fork_l);
    slot_close(&warm); slot_close(&cold); slot_close(&forked);
}

// ---- gate 4: speculative decode rolls the fold back on divergence ----------
//
// The batched verify advances the recurrent fold through EVERY drafted token; a
// partially-accepted round must roll it back to the accepted prefix (tracer 6:
// round-start snapshot + re-fold). Two gates, honestly scoped:
//
//   4a (decisive, model-level): the exact rollback recipe spec_fold_sync
//      performs — restore the round-start snapshot, re-fold the accepted
//      prefix with a batched forward, continue — must land on logits BIT-
//      IDENTICAL to a run that never folded the rejected drafts at all.
//      Deterministic; no dependence on organic draft divergence.
//
//   4b (engine-level smoke): the full speculative walk with a draft model
//      stays byte-identical to plain decoding. On these tiny fixtures every
//      random model collapses to echoing its last input token, so drafts are
//      organically all-accepted here (measured: 20/20 across seeds and even
//      across architectures) — this smoke therefore pins the full-accept path,
//      while the REJECTED-round engine path is exercised by the grammar-ff
//      identity gate run against this same fixture (make test runs
//      test-grammar-ff on test-ornith.gguf: its unconstrained draft proposals
//      against the constrained target measured 35/67 accepted).
typedef struct { char buf[2048]; int len; } capture;

static int cap_cb(void *ud, const char *s, int n) {
    capture *c = ud;
    if (c->len + n >= (int)sizeof(c->buf)) return 1;
    memcpy(c->buf + c->len, s, (size_t)n);
    c->len += n;
    return 0;
}

static void test_divergent_round_rollback(void) {
    model_params p = base_params();
    slot ref, tst;
    if (!slot_open(&ref, &p) || !slot_open(&tst, &p)) {
        ck(0, "load two instances for the rollback gate");
        return;
    }
    const int32_t seq[6]  = { 10, 20, 30, 40, 50, 60 }; // prompt + a + b
    const int32_t bad[2]  = { 111, 222 };               // rejected drafts

    // reference: never speculated — fold the prompt, then a (50), then b (60)
    float *rl = NULL;
    for (int i = 0; i < 6; i++) rl = model_forward(&ref.m, seq[i], i);
    float *ref_l = snap_logits(&ref.m, rl);

    // test: fold the prompt, then run one divergent speculative round the way
    // the spec walk + spec_fold_sync do it — snapshot, batch-verify 3 drafts
    // [a, X, Y] (the fold now wrongly holds X and Y), accept only a: restore
    // the snapshot, re-fold the accepted prefix batched, forward the real b.
    for (int i = 0; i < 4; i++) model_forward(&tst.m, seq[i], i);
    ck(model_recurrent_snapshot(&tst.m, 4), "snapshot the round-start fold");
    int32_t drafts[3] = { seq[4], bad[0], bad[1] };
    model_forward_batch(&tst.m, drafts, 3, 4, false);   // folds a, X, Y
    ck(model_recurrent_restore(&tst.m, 4), "restore the round-start fold");
    model_forward_batch(&tst.m, drafts, 1, 4, false);   // re-fold accepted a
    float *tl = model_forward(&tst.m, seq[5], 5);        // the real token b
    float *tst_l = snap_logits(&tst.m, tl);

    int diffs = logits_differ(&ref.m, ref_l, tst_l);
    ck(diffs == 0, "a rolled-back divergent round is bit-identical to no round");

    free(ref_l); free(tst_l);
    slot_close(&ref); slot_close(&tst);
}

static void test_spec_full_accept_identity(void) {
    enum { NP = 4, GEN = 24 };
    model_params p = base_params();
    slot plain, spec;
    if (!slot_open(&plain, &p) || !slot_open(&spec, &p)) {
        ck(0, "load two instances for the speculative gate");
        return;
    }
    model_params dp = base_params();
    model_t *draft = malloc(sizeof(model_t));
    if (!draft || !model_load(draft, "test-ornith-draft.gguf", &dp)) {
        ck(0, "load the draft fixture");
        return;
    }
    spec.e.dm = draft;
    spec.e.draft_k = 4;

    const int32_t prompt[NP] = { 10, 20, 30, 40 };
    capture pc = {{0}, 0}, sc = {{0}, 0};

    engine_reset(&plain.e);
    float *pl = engine_feed(&plain.e, prompt, NP);
    engine_generate(&plain.e, pl, GEN, cap_cb, &pc, NULL);

    engine_reset(&spec.e);
    float *sl = engine_feed(&spec.e, prompt, NP);
    engine_generate(&spec.e, sl, GEN, cap_cb, &sc, NULL);

    ck(spec.e.spec_st.drafted > 0, "the draft actually proposed tokens");
    ck(pc.len > 0, "plain decoding produced output");
    ck(sc.len == pc.len && memcmp(pc.buf, sc.buf, (size_t)pc.len) == 0,
       "speculative output is byte-identical to plain decoding");

    model_free(draft); free(draft);
    spec.e.dm = NULL;
    slot_close(&plain); slot_close(&spec);
}

// ---- gate 5: speculative rollback must own every recurrent fold ------------
//
// Partial CUDA offload used to pass model_spec_verify_ok even when one of the
// offloaded leading layers was recurrent. Its live fold is then device-owned,
// while model_recurrent_snapshot/restore above only checkpoints the host fold.
// A rejected speculative round therefore cannot be rolled back correctly.
// Keep CPU recurrent verification enabled, but decline the first split that
// places this fixture's recurrent layer 0 on the device.
static void test_spec_admission_requires_host_recurrent_fold(void) {
    model_params p = base_params();
    slot s;
    if (!slot_open(&s, &p)) {
        ck(0, "load an instance for the speculative admission gate");
        return;
    }

    ck(s.m.layers[0].recurrent, "the fixture's leading layer is recurrent");
    ck(model_spec_verify_ok(&s.m), "CPU recurrent verification is admitted");

    // No backend call is made: the non-NULL marker models the placement fields
    // that CUDA publishes after upload, and is cleared before model_free.
    s.m.gpu = &s;
    s.m.gpu_layers = 1;
    ck(!model_spec_verify_ok(&s.m),
       "a CUDA-resident recurrent fold declines speculative verification");
    ck(spec_draft_load("test-ornith-draft.gguf", &s.m, &p) == NULL,
       "draft loading declines the unsafe partial CUDA split");
    s.m.gpu = NULL;
    s.m.gpu_layers = 0;

    slot_close(&s);
}

int main(void) {
    test_snapshot_restore_roundtrip();
    test_rewind_divergence_matches_cold();
    test_prefix_cache_fold_restore_matches_cold();
    test_divergent_round_rollback();
    test_spec_full_accept_identity();
    test_spec_admission_requires_host_recurrent_fold();
    return g_fail;
}
