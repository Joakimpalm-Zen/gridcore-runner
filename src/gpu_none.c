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
