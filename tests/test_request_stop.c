// Request-stop predicate gate.
//
// Cancellation is sampled only at the engine's existing safe boundaries:
// between complete prefill chunks and between a sampled decode token and its
// forward.  This test uses the generated Ornith hybrid fixture so an aborted
// slot has both attention KV and a recurrent fold to clean up before reuse.
// A follow-up on that slot must remain bit-identical to a cold run.
#include "scheduler.c"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

server_state SV;

static int g_fail;

static void ck(bool cond, const char *what) {
    if (!cond) { fprintf(stderr, "FAIL: %s\n", what); g_fail = 1; }
    else        fprintf(stderr, "ok: %s\n", what);
}

enum { NSLOTS = 2, CTX = 64, BATCH = 4 };

typedef struct {
    int polls;
    int fire_after;
} stop_probe;

static bool stop_after(void *ud) {
    stop_probe *p = ud;
    p->polls++;
    return p->polls >= p->fire_after;
}

static bool stop_on_peer_close(void *ud) {
    sock_t *fd = ud;
    return sock_peer_closed(*fd);
}

typedef struct {
    unsigned char bytes[2048];
    int n;
} capture;

static int capture_cb(void *ud, const char *s, int n) {
    capture *c = ud;
    if (n < 0 || c->n + n > (int)sizeof(c->bytes)) return 1;
    memcpy(c->bytes + c->n, s, (size_t)n);
    c->n += n;
    return 0;
}

static bool slots_open(void) {
    model_params p;
    memset(&p, 0, sizeof(p));
    p.gpu_mode = GPU_OFF;
    p.n_threads = 1;
    p.n_ctx = CTX;
    p.n_batch = BATCH;

    SV.n_slots = NSLOTS;
    SV.slots = calloc(NSLOTS, sizeof(*SV.slots));
    if (!SV.slots) return false;
    tokenizer *tok = calloc(1, sizeof(*tok));
    if (!tok) return false;
    for (int i = 0; i < NSLOTS; i++) {
        slot_t *s = &SV.slots[i];
        s->id = i;
        s->m = calloc(1, sizeof(*s->m));
        if (!s->m || !model_load(s->m, "test-ornith.gguf", &p)) return false;
        if (i == 0 && !tokenizer_init(tok, &s->m->gf)) return false;
        s->tok = tok;
        s->smp.temp = 0;
        s->smp.repeat_penalty = 1.0f;
        s->smp.rng = (uint64_t)(i + 1);
        if (!engine_init(&s->e, s->m, tok, &s->smp)) return false;
        s->e.ignore_eos = true;
    }
    return sched_start();
}

static void slots_close(void) {
    sched_shutdown();
    tokenizer *tok = SV.slots ? SV.slots[0].tok : NULL;
    for (int i = 0; i < SV.n_slots; i++) {
        free(SV.slots[i].e.hist);
        if (SV.slots[i].m) {
            model_free(SV.slots[i].m);
            free(SV.slots[i].m);
        }
    }
    if (tok) { tokenizer_free(tok); free(tok); }
    free(SV.slots);
    memset(&SV, 0, sizeof(SV));
}

static float *copy_logits(const model_t *m, const float *logits) {
    float *copy = malloc(sizeof(*copy) * (size_t)m->n_vocab);
    if (copy && logits)
        memcpy(copy, logits, sizeof(*copy) * (size_t)m->n_vocab);
    return copy;
}

static bool same_logits(const model_t *m, const float *a, const float *b) {
    return a && b && memcmp(a, b, sizeof(*a) * (size_t)m->n_vocab) == 0;
}

static void followup_matches_cold(const int32_t *prompt, int n,
                                  const char *logits_what,
                                  const char *output_what) {
    slot_t *warm = &SV.slots[0], *cold = &SV.slots[1];
    int keep = engine_rewind(&warm->e, prompt, n);
    float *wl = engine_feed(&warm->e, prompt + keep, n - keep);
    float *warm_logits = copy_logits(warm->m, wl);

    engine_reset(&cold->e);
    float *cl = engine_feed(&cold->e, prompt, n);
    float *cold_logits = copy_logits(cold->m, cl);
    ck(same_logits(warm->m, warm_logits, cold_logits), logits_what);

    capture wc = {{0}, 0}, cc = {{0}, 0};
    int wn = sched_generate(warm, warm_logits, 8, capture_cb, &wc, NULL, 0);
    int cn = sched_generate(cold, cold_logits, 8, capture_cb, &cc, NULL, 0);
    ck(wn == cn && wc.n == cc.n &&
       memcmp(wc.bytes, cc.bytes, (size_t)wc.n) == 0, output_what);
    free(warm_logits);
    free(cold_logits);
}

static void test_prefill_stop_and_reuse(void) {
    slot_t *s = &SV.slots[0];
    int32_t prompt[20];
    for (int i = 0; i < 20; i++) prompt[i] = 10 + i;
    stop_probe p = {0, 2};

    engine_reset(&s->e);
    engine_set_stop(&s->e, stop_after, &p);
    ck(engine_feed(&s->e, prompt, 20) == NULL,
       "prefill stop returns NULL before the prompt completes");
    ck(p.polls == 2 && s->e.pos == BATCH * 2,
       "prefill stop fires only between complete chunks");
    engine_set_stop(&s->e, NULL, NULL);

    const int32_t next[6] = {70, 71, 72, 73, 74, 75};
    followup_matches_cold(next, 6,
        "a slot reused after stopped prefill has cold-identical hybrid logits",
        "a slot reused after stopped prefill has cold-identical output");
}

static void test_decode_stop_and_reuse(void) {
    slot_t *s = &SV.slots[0];
    const int32_t prompt[6] = {30, 31, 32, 33, 34, 35};
    engine_reset(&s->e);
    float *logits = engine_feed(&s->e, prompt, 6);
    stop_probe p = {0, 3};
    capture out = {{0}, 0};

    engine_set_stop(&s->e, stop_after, &p);
    int n = sched_generate(s, logits, 16, capture_cb, &out, NULL, 0);
    ck(p.polls == 3 && n == 3,
       "decode stop ends generation after the requested poll");
    ck(s->e.pending_pos == -1 && s->e.pos == 6 + 2,
       "decode stop rewinds the sampled token whose forward was abandoned");
    engine_set_stop(&s->e, NULL, NULL);

    const int32_t next[6] = {90, 91, 92, 93, 94, 95};
    followup_matches_cold(next, 6,
        "a slot reused after stopped decode has cold-identical hybrid logits",
        "a slot reused after stopped decode has cold-identical output");
}

static void test_solo_decode_stop_and_reuse(void) {
    slot_t *s = &SV.slots[0];
    const int32_t prompt[6] = {40, 41, 42, 43, 44, 45};
    engine_reset(&s->e);
    float *logits = engine_feed(&s->e, prompt, 6);
    stop_probe p = {0, 3};
    capture out = {{0}, 0};

    engine_set_stop(&s->e, stop_after, &p);
    int n = engine_generate(&s->e, logits, 16, capture_cb, &out, NULL);
    ck(p.polls == 3 && n == 3,
       "solo decode polls the request stop between token forwards");
    ck(s->e.pending_pos == -1 && s->e.pos == 6 + 2,
       "solo decode stop rewinds the token whose forward was abandoned");
    engine_set_stop(&s->e, NULL, NULL);

    const int32_t next[6] = {160, 161, 162, 163, 164, 165};
    followup_matches_cold(next, 6,
        "a slot reused after solo decode stop has cold-identical hybrid logits",
        "a slot reused after solo decode stop has cold-identical output");
}

static void test_spec_decode_stop_and_reuse(void) {
    slot_t *s = &SV.slots[0];
    model_params p;
    memset(&p, 0, sizeof(p));
    p.gpu_mode = GPU_OFF;
    p.n_threads = 1;
    p.n_ctx = CTX;
    p.n_batch = BATCH;
    model_t *draft = calloc(1, sizeof(*draft));
    if (!draft || !model_load(draft, "test-ornith-draft.gguf", &p)) {
        free(draft);
        ck(false, "load the generated draft fixture for request-stop");
        return;
    }
    s->e.dm = draft;
    s->e.draft_k = 4;
    const int32_t prompt[6] = {50, 51, 52, 53, 54, 55};
    engine_reset(&s->e);
    float *logits = engine_feed(&s->e, prompt, 6);
    stop_probe probe = {0, 4};
    capture out = {{0}, 0};

    engine_set_stop(&s->e, stop_after, &probe);
    int n = engine_generate(&s->e, logits, 16, capture_cb, &out, NULL);
    ck(probe.polls == 4 && n > 0 && n < 16,
       "speculative decode polls request stop inside a verify round");
    engine_set_stop(&s->e, NULL, NULL);
    model_free(draft);
    free(draft);
    s->e.dm = NULL;

    const int32_t next[6] = {170, 171, 172, 173, 174, 175};
    followup_matches_cold(next, 6,
        "a slot reused after spec decode stop has cold-identical hybrid logits",
        "a slot reused after spec decode stop has cold-identical output");
}

static void test_socket_stop_and_reuse(void) {
#ifdef _WIN32
    fprintf(stderr, "skip: socketpair request-stop gate is POSIX-only\n");
#else
    int sv[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0) {
        ck(false, "create a local socket pair for the peer-close gate");
        return;
    }
    ck(!sock_peer_closed(sv[0]),
       "an alive but quiet peer is not cancellation");
    ck(send(sv[1], "next", 4, 0) == 4,
       "queue pipelined bytes on the peer");
    ck(!sock_peer_closed(sv[0]),
       "readable pipelined bytes are not cancellation");
    char next[4];
    ck(recv(sv[0], next, sizeof(next), 0) == 4 &&
       memcmp(next, "next", 4) == 0,
       "the peer-close probe does not consume pipelined bytes");

    close(sv[1]);
    slot_t *s = &SV.slots[0];
    int32_t prompt[20];
    for (int i = 0; i < 20; i++) prompt[i] = 110 + i;
    engine_reset(&s->e);
    engine_set_stop(&s->e, stop_on_peer_close, &sv[0]);
    ck(engine_feed(&s->e, prompt, 20) == NULL && s->e.pos == BATCH,
       "a clean peer close stops prefill at the next chunk boundary");
    engine_set_stop(&s->e, NULL, NULL);
    close(sv[0]);

    const int32_t followup[6] = {150, 151, 152, 153, 154, 155};
    followup_matches_cold(followup, 6,
        "a slot reused after peer-close prefill has cold-identical hybrid logits",
        "a slot reused after peer-close prefill has cold-identical output");
#endif
}

int main(void) {
    if (!slots_open()) {
        fprintf(stderr, "FAIL: could not open scheduler slots\n");
        slots_close();
        return 1;
    }
    ck(model_has_recurrent(SV.slots[0].m),
       "request-stop fixture has recurrent layers");
    test_prefill_stop_and_reuse();
    test_decode_stop_and_reuse();
    test_solo_decode_stop_and_reuse();
    test_spec_decode_stop_and_reuse();
    test_socket_stop_and_reuse();
    slots_close();
    return g_fail;
}
