// Pure Metal weight-admission policy. Kept free of Objective-C/Metal types so
// the capacity decision can be tested with device sizes this machine does not
// physically have.
#ifndef RUNNER_METAL_ADMISSION_H
#define RUNNER_METAL_ADMISSION_H

#include <stdint.h>

typedef struct {
    uint64_t working_set;
    uint64_t max_buffer;
} metal_weight_limits;

static inline uint64_t metal_full_weight_budget(metal_weight_limits lim,
                                                uint64_t other_bytes) {
    // max_buffer is deliberately not a full-offload ceiling: metal.m wraps a
    // weight file in as many tensor-boundary MTLBuffers as it needs. Only the
    // aggregate working set competes with KV, scratch, and the safety margin.
    (void)lim.max_buffer;
    return lim.working_set > other_bytes
         ? (uint64_t)((double)(lim.working_set - other_bytes) * 0.94)
         : 0;
}

#endif // RUNNER_METAL_ADMISSION_H
