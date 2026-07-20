// GPU backend stub for platforms without one.
// CUDA (cuda.c) and Metal (metal.m) implement this same interface.
#include "runner.h"

bool gpu_available(char *name, int cap) {
    (void)name; (void)cap;
    return false;
}

bool gpu_mem_info(size_t *free_bytes, size_t *total_bytes) {
    (void)free_bytes; (void)total_bytes;
    return false;
}

bool gpu_kv_q8_ok(void) {
    return false;   // no backend here; the CPU path handles q8 on its own
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
