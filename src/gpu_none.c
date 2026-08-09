// GPU backend stub for platforms without one.
// CUDA (cuda.c) and Metal (metal.m) implement this same interface.
#include "gpu.h"

bool gpu_available(char *name, int cap) {
    (void)name; (void)cap;
    return false;
}

bool gpu_device_id(char *id, int cap) {
    (void)id; (void)cap;
    return false;   // no GPU, so nothing to account for and nothing to name
}

bool gpu_mem_info(size_t *free_bytes, size_t *total_bytes) {
    (void)free_bytes; (void)total_bytes;
    return false;
}

const char *gpu_shader_source_sha(void) { return NULL; }   // no shaders here

bool gpu_kv_q8_ok(void) {
    return false;   // no backend here; the CPU path handles q8 on its own
}

// No GPU, so no tensor-core GEMM was ever dispatched. Present so the TC
// tolerance gate links against a CPU-only build.
unsigned long gpu_tc_dispatches(void) { return 0; }

bool gpu_moe_ok(void) {
    return false;   // no backend here at all
}

bool gpu_eseries_ok(void) {
    return false;
}

bool gpu_init(model_t *m) {
    (void)m;
    return false;
}

bool gpu_forward_batch(model_t *m, const int32_t *tokens, int n, int pos,
                       bool want_logits, float **logits) {
    (void)m; (void)tokens; (void)n; (void)pos; (void)want_logits; (void)logits;
    return false;
}

void gpu_free(model_t *m) {
    (void)m;
}

void gpu_disable(model_t *m) {
    m->gpu = NULL;
}

// No backend, so no microbatch: model_batch_decode sees NULL and decodes the
// sequences one at a time, which is what this platform would have done anyway.
gpu_batch *gpu_batch_create(model_t **seqs, int n) {
    (void)seqs; (void)n;
    return NULL;
}

void gpu_batch_free(gpu_batch *b) {
    (void)b;
}

bool gpu_batch_decode(gpu_batch *b, const int *idx, const int32_t *tok,
                      const int *pos, int n, float **out) {
    (void)b; (void)idx; (void)tok; (void)pos; (void)n; (void)out;
    return false;
}
