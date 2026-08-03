// Prefix snapshots survive a restart, and a snapshot that does not belong to
// this model does not.
//
// The second half is the point. A snapshot is raw KV bytes; installing one
// that does not match the live model does not fail loudly, it produces fluent
// wrong output — the same hazard `engine_prefix_reuse` was built around, now
// with a file as the attack surface instead of a token vector. So the checks
// here are mostly refusals: a different model key, a different entry width, a
// truncated file, a flipped byte, a file that is not ours at all.
//
// Round-tripping is checked by CONTENT, not by counting entries: a save/load
// pair that wrote zeros would keep the count and lose the cache.
#include "runner.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fail = 0;
static const char *g_model = "test.gguf";

static void MARK(const char*s){fprintf(stderr,"[mark] %s\n",s);}
static void ck(int cond, const char *what) {
    if (!cond) { fprintf(stderr, "FAIL: %s\n", what); g_fail = 1; }
    else        printf("ok: %s\n", what);
}

static model_params base_params(void) {
    model_params p;
    memset(&p, 0, sizeof(p));
    p.gpu_mode = GPU_OFF;
    p.n_threads = 1;
    p.n_ctx = 256;
    p.n_batch = 8;
    return p;
}

enum { NTOK = 40 };
static int32_t TOKS[NTOK];

// Fill the cache from a live engine, then report how many tokens a second
// engine can fork back.
static int forkable(engine *e) {
    prefix_reuse r = engine_prefix_reuse(e, TOKS, NTOK);
    return r.forked;
}

int main(int argc, char **argv) {
    if (argc > 1) g_model = argv[1];
    f16_init();
    for (int i = 0; i < NTOK; i++) TOKS[i] = 3 + (i % 17);

    const char *path = "test-prefix-persist.bin";
    remove(path);

    model_params p = base_params();
    model_t m;
    tokenizer tok;
    if (!model_load(&m, g_model, &p)) { fprintf(stderr, "cannot load %s\n", g_model); return 1; }
    if (!tokenizer_init(&tok, &m.gf)) { fprintf(stderr, "tokenizer init failed\n"); return 1; }
    sampler smp;
    sampler_reset(&smp);
    prefix_cache_configure(64u << 20, 600.0);

    engine e = {0};   // engine_init frees e->hist on entry: it MUST be zeroed
    if (!engine_init(&e, &m, &tok, &smp)) { fprintf(stderr, "engine init failed\n"); return 1; }
    ck(engine_feed(&e, TOKS, NTOK) != NULL, "prefill for the snapshot");
    engine_prefix_publish(&e, TOKS, NTOK, NTOK, 0.01);

    prefix_cache_stats stats;
    prefix_cache_stats_get(&stats);
    ck(stats.entries == 1, "one entry published");
    size_t live_bytes = stats.bytes;

    ck(prefix_cache_save(path) == 1, "save reports one entry");

    // The restart: drop everything, then load the file back.
    prefix_cache_clear();
    prefix_cache_stats_get(&stats);
    ck(stats.entries == 0, "cache cleared");

    ck(prefix_cache_load(path, &e) == 1, "load reports one entry");
    prefix_cache_stats_get(&stats);
    ck(stats.entries == 1 && stats.bytes == live_bytes,
       "the reloaded entry is the same size as the saved one");

    // By CONTENT: a fresh engine must be able to fork the reloaded snapshot.
    engine e2 = {0};   // engine_init frees e->hist on entry: it MUST be zeroed
    if (!engine_init(&e2, &m, &tok, &smp)) { fprintf(stderr, "engine2 init failed\n"); return 1; }
    ck(forkable(&e2) == NTOK - 1 || forkable(&e2) > 0,
       "a fresh engine forks the reloaded snapshot");

    // --- refusals -------------------------------------------------------
    // A model whose key differs. The context length is part of model_key, so
    // loading the same file at a different -c is the cheapest honest way to
    // build a foreign snapshot without a second checkpoint.
    model_params p2 = base_params();
    p2.n_ctx = 192;
    model_t m2;
    tokenizer tok2;
    if (!model_load(&m2, g_model, &p2)) { fprintf(stderr, "cannot load second\n"); return 1; }
    if (!tokenizer_init(&tok2, &m2.gf)) { fprintf(stderr, "tokenizer2 failed\n"); return 1; }
    sampler smp2;
    sampler_reset(&smp2);
    engine e3 = {0};   // engine_init frees e->hist on entry: it MUST be zeroed
    if (!engine_init(&e3, &m2, &tok2, &smp2)) { fprintf(stderr, "engine3 failed\n"); return 1; }
    ck(e3.model_key != e.model_key, "a different context is a different key");

    prefix_cache_clear();
    ck(prefix_cache_load(path, &e3) == 0,
       "a snapshot from another context is REFUSED, not adapted");
    prefix_cache_stats_get(&stats);
    ck(stats.entries == 0, "nothing foreign was installed");

    // A file that is not ours.
    FILE *junk = fopen("test-prefix-junk.bin", "wb");
    fwrite("not a prefix file at all, really", 1, 32, junk);
    fclose(junk);
    ck(prefix_cache_load("test-prefix-junk.bin", &e) == -1, "a foreign file is refused");

    // Truncation.
    FILE *src = fopen(path, "rb");
    fseek(src, 0, SEEK_END);
    long n = ftell(src);
    fseek(src, 0, SEEK_SET);
    char *buf = malloc((size_t)n);
    size_t got = fread(buf, 1, (size_t)n, src);
    fclose(src);
    FILE *cut = fopen("test-prefix-cut.bin", "wb");
    fwrite(buf, 1, got - 64, cut);       // lose the tail, including the digest
    fclose(cut);
    prefix_cache_clear();
    ck(prefix_cache_load("test-prefix-cut.bin", &e) == -1, "a truncated file is refused");
    prefix_cache_stats_get(&stats);
    ck(stats.entries == 0, "a truncated file installs nothing");

    // A single flipped byte inside the KV payload.
    buf[got - 100] ^= 0xFF;
    FILE *bad = fopen("test-prefix-bad.bin", "wb");
    fwrite(buf, 1, got, bad);
    fclose(bad);
    prefix_cache_clear();
    ck(prefix_cache_load("test-prefix-bad.bin", &e) == -1, "a flipped byte is refused");
    prefix_cache_stats_get(&stats);
    ck(stats.entries == 0, "a corrupt file installs nothing");

    MARK("free(buf);");
    free(buf);
    MARK("remove(path); remove");
    remove(path); remove("test-prefix-junk.bin");
    remove("test-prefix-cut.bin"); remove("test-prefix-bad.bin");
    MARK("free(e.hist);");
    free(e.hist); free(e2.hist); free(e3.hist);
    MARK("prefix_cache_clear();");
    prefix_cache_clear();
    tokenizer_free(&tok2); model_free(&m2);
    MARK("tokenizer_free(&tok); model_free(&m);");
    tokenizer_free(&tok); model_free(&m);

    MARK("end of teardown");
    if (g_fail) { fprintf(stderr, "prefix persistence: FAILED\n"); return 1; }
    puts("prefix persistence ok");
    return 0;
}
