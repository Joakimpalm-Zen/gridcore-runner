#include <assert.h>
#include <stdlib.h>
#include <malloc/malloc.h>

#include "../src/metal.m"

int ggml_block_size(int type) {
    (void)type;
    abort();
}

size_t ggml_type_size(int type) {
    (void)type;
    abort();
}

void dequant_row(int type, const void *src, float *dst, int n) {
    (void)type; (void)src; (void)dst; (void)n;
    abort();
}

void model_ple_prepass(model_t *m, const int32_t *tokens, int n,
                       const float *x, float *out, float *scratch) {
    (void)m; (void)tokens; (void)n; (void)x; (void)out; (void)scratch;
    abort();
}

void model_embd_transform(const model_t *m, float *row) {
    (void)m; (void)row;
    abort();
}

// Every per-layer buffer table gpu_init() allocates must come back.
//
// gpu_release_state keeps THREE hand-written rosters of the same set — the
// element-release loop, the array-free list, and gpu_init's calloc block — and
// nothing links them. g->ppn was in two of the three: released element-wise,
// never freed, so every Metal model load leaked its pointer array for the
// process lifetime.
//
// A private malloc zone makes that countable without touching the allocator
// metal.m uses: free() dispatches to the owning zone, so gpu_free() releases
// these through the ordinary path while malloc_zone_statistics() sees only
// this test's blocks. n_layer is 0, so the element loop never dereferences the
// (empty) tables and what is under test is purely the array ownership.
static void check_layer_tables_freed(void) {
    malloc_zone_t *z = malloc_create_zone(0, 0);
    assert(z);
    malloc_statistics_t st;
    malloc_zone_statistics(z, &st);
    assert(st.blocks_in_use == 0);

    model_t lm = {0};
    gpu_t *lg = malloc_zone_calloc(z, 1, sizeof(*lg));
    assert(lg);
    id<MTLBuffer> **tables[] = {
        &lg->attn_norm, &lg->ffn_norm, &lg->bq, &lg->bk, &lg->bv, &lg->bo,
        &lg->qn, &lg->kn, &lg->sinks, &lg->gib, &lg->geb, &lg->ueb, &lg->deb,
        &lg->ppn, &lg->pan, &lg->pfn, &lg->gpn1, &lg->gprn2, &lg->gpn2,
        &lg->ggis, &lg->gdsc,
    };
    const unsigned n_tables = (unsigned)(sizeof(tables) / sizeof(*tables));
    for (unsigned i = 0; i < n_tables; i++)
        *tables[i] = malloc_zone_calloc(z, 1, sizeof(id<MTLBuffer>));

    malloc_zone_statistics(z, &st);
    assert(st.blocks_in_use == n_tables + 1);   // the tables plus gpu_t itself

    lm.gpu = lg;
    lm.gpu_owner = lg;
    gpu_free(&lm);

    malloc_zone_statistics(z, &st);
    assert(st.blocks_in_use == 0);
    malloc_destroy_zone(z);
}

int main(void) {
    check_layer_tables_freed();

    model_t m = {0};
    gpu_t *g = calloc(1, sizeof(*g));
    assert(g);

    unsigned char borrowed_k[16] = {0};
    unsigned char borrowed_v[16] = {0};
    m.gpu = g;
    m.gpu_owner = g;
    m.kv_owner = KV_OWNER_GPU_BACKEND;
    m.kcache = (f16_t *)borrowed_k;
    m.vcache = (f16_t *)borrowed_v;
    m.gpu_layers = 42;

    gpu_disable(&m);
    assert(m.gpu == NULL);
    assert(m.gpu_owner == g);
    assert(m.kv_owner == KV_OWNER_GPU_BACKEND);
    assert(m.kcache == (f16_t *)borrowed_k);
    assert(m.vcache == (f16_t *)borrowed_v);
    assert(m.gpu_layers == 0);

    gpu_disable(&m);
    assert(m.gpu == NULL);
    assert(m.gpu_owner == g);
    assert(m.kv_owner == KV_OWNER_GPU_BACKEND);

    gpu_free(&m);
    assert(m.gpu == NULL);
    assert(m.gpu_owner == NULL);
    assert(m.kv_owner == KV_OWNER_MALLOC);
    assert(m.kcache == NULL);
    assert(m.vcache == NULL);

    gpu_disable(&m);
    gpu_free(&m);
    assert(m.gpu == NULL);
    assert(m.gpu_owner == NULL);

    model_t cpu = {0};
    cpu.kv_owner = KV_OWNER_MALLOC;
    cpu.kcache = malloc(16);
    cpu.vcache = malloc(16);
    assert(cpu.kcache && cpu.vcache);
    gpu_free(&cpu);
    assert(cpu.kcache != NULL);
    assert(cpu.vcache != NULL);
    free(cpu.kcache);
    free(cpu.vcache);

    return 0;
}
