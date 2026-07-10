// GPU backend stub for platforms without one (Linux/Windows for now).
// CUDA and Vulkan backends would implement this same interface.
#include "runner.h"

bool gpu_available(char *name, int cap) {
    (void)name; (void)cap;
    return false;
}

bool gpu_init(model_t *m) {
    (void)m;
    return false;
}

float *gpu_forward(model_t *m, int token, int pos) {
    (void)m; (void)token; (void)pos;
    return 0;
}

void gpu_free(model_t *m) {
    (void)m;
}
