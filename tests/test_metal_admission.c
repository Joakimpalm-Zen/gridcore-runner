#include "metal_admission.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>

int main(void) {
    const uint64_t GB = 1000000000ull;
    metal_weight_limits reported_16gb_mac = {
        .working_set = 127ull * GB / 10,
        .max_buffer = 95ull * GB / 10,
    };
    uint64_t budget = metal_full_weight_budget(reported_16gb_mac, GB / 2);

    // Several zero-copy buffers can hold this 10.3 GB file. It is below the
    // working-set budget but above one MTLBuffer, so selecting a layer split
    // here would preserve the exact ceiling the multi-buffer path removed.
    uint64_t model = 103ull * GB / 10;
    assert(model > reported_16gb_mac.max_buffer);
    assert(model <= budget);

    // Multi-buffer wrapping changes no residency arithmetic: a file beyond
    // the working-set budget must still take the partial/refusal path.
    assert(12ull * GB > budget);

    puts("metal admission policy ok");
    return 0;
}
