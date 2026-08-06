// The Metal shader library must compile, and every kernel the backend looks
// up must exist in it.
//
// Why this is its own gate: a shader compile error is NOT loud in practice.
// gpu_init prints a line and returns false, the model runs on the CPU, and
// output stays correct — so a "CPU vs GPU" comparison run by hand silently
// compares the CPU path with itself and passes. That happened during the
// batched-prefill work: a kernel with mismatched position-attribute types
// stopped the whole library compiling, every run fell back, and an ad-hoc
// identity check reported success while the measured "speedup" was really
// just the CPU path. The Makefile smokes catch it (they grep stderr for
// "Metal backend"), but only if you run them.
//
// A missing FUNCTION is quieter still: mk_pipeline returns nil and the backend
// keeps going with that one pipeline unset, which for the tiled-GEMM kernels
// means silently falling back to the matvec path — a large performance
// regression with no wrong answer to trip any correctness gate.
//
// This test needs no model and no weights: compile the embedded source, then
// look up every name. Skips (not passes) when the machine has no Metal device.
#import <Metal/Metal.h>
#include <stdio.h>
#include <string.h>

#include "../src/kernels_metal.h"   // k_metal_src

// Every kernel mk_pipeline() asks for in src/metal.m. Keep in step with it:
// scripts/check-generated.py cannot see a name that was renamed in both files.
static const char *KERNELS[] = {
    "k_add", "k_attn", "k_gelu_mul", "k_head_rmsnorm", "k_head_transform",
    "k_mm_f16", "k_mm_f32", "k_mm_mxfp4", "k_mm_q4_0", "k_mm_q4_K",
    "k_mm_q6_K", "k_mm_q8_0",
    "k_moe_actmul", "k_moe_mv_f16", "k_moe_mv_f32", "k_moe_mv_mxfp4",
    "k_moe_mv_q4_0", "k_moe_mv_q4_K", "k_moe_mv_q5_K", "k_moe_mv_q6_K",
    "k_moe_mv_q8_0", "k_moe_route", "k_moe_sum",
    "k_mv_f16", "k_mv_f32", "k_mv_mxfp4", "k_mv_q4_0", "k_mv_q4_1",
    "k_mv_q4_K", "k_mv_q5_0", "k_mv_q5_1", "k_mv_q5_K", "k_mv_q6_K",
    "k_mv_q8_0",
    "k_qknorm", "k_rmsnorm", "k_rope", "k_scale", "k_silu_mul", "k_store_kv",
};

int main(void) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) {
            printf("metal shaders: skipped (no Metal device)\n");
            return 0;
        }

        NSError *err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:
                                  [NSString stringWithUTF8String:k_metal_src]
                                              options:nil
                                                error:&err];
        if (!lib) {
            fprintf(stderr, "FAIL: the embedded Metal library does not compile."
                    "\n  Every run would fall back to the CPU, and any hand-run"
                    "\n  CPU-vs-GPU check would compare the CPU with itself.\n%s\n",
                    err ? err.localizedDescription.UTF8String : "(no diagnostic)");
            return 1;
        }

        int missing = 0;
        for (size_t i = 0; i < sizeof(KERNELS) / sizeof(*KERNELS); i++) {
            id<MTLFunction> fn = [lib newFunctionWithName:
                                      [NSString stringWithUTF8String:KERNELS[i]]];
            if (!fn) {
                fprintf(stderr, "FAIL: kernel %s is missing from the library "
                        "(mk_pipeline would return nil and the backend would "
                        "quietly run without it)\n", KERNELS[i]);
                missing++;
                continue;
            }
            // A function can exist and still fail to specialize (bad buffer
            // index, unsupported type); the backend hits that only at init.
            NSError *perr = nil;
            id<MTLComputePipelineState> p =
                [dev newComputePipelineStateWithFunction:fn error:&perr];
            if (!p) {
                fprintf(stderr, "FAIL: kernel %s compiles but has no pipeline: %s\n",
                        KERNELS[i],
                        perr ? perr.localizedDescription.UTF8String : "(no diagnostic)");
                missing++;
            }
            [p release];
            [fn release];
        }
        [lib release];
        [dev release];
        if (missing) return 1;
        printf("metal shaders: library compiles, %zu kernels present\n",
               sizeof(KERNELS) / sizeof(*KERNELS));
    }
    return 0;
}
