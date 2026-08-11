#include <assert.h>
#include <stdlib.h>

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

int main(void) {
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
