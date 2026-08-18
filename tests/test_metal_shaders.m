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

// The roster is READ OUT OF src/metal.m, not restated here.
//
// It used to be a hand-kept array carrying the comment "keep in step with it",
// and it had silently fallen five kernels behind: k_attn_coop and
// k_attn_chunk_coop (the cooperative decode attention PROMOTED to the default
// route on 2026-08-17), k_sigmoid_mul (muse-glimmer's attention output gate)
// and the two k_mvf_* fast matvecs were all looked up by mk_pipeline() and
// checked by nothing. A missing function is precisely the quiet failure this
// file exists to catch, so its roster cannot be a second copy of the backend's.
//
// The scan is deliberately literal — the exact `mk_pipeline(dev, lib, @"name")`
// spelling metal.m uses — so a lookup written some other way shows up as a
// count that stopped growing rather than as a false pass. MIN_KERNELS below is
// the guard against the scan itself failing open.
enum { MAX_KERNELS = 256, NAME_CAP = 64, MIN_KERNELS = 50 };

static int scan_pipelines(const char *path, char names[][NAME_CAP]) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    char line[1024];
    int n = 0;
    while (fgets(line, sizeof line, f) && n < MAX_KERNELS) {
        const char *p = line;
        while ((p = strstr(p, "mk_pipeline(")) != NULL) {
            const char *q = strstr(p, "@\"");
            const char *end = q ? strchr(q + 2, '"') : NULL;
            if (!q || !end || end - (q + 2) >= NAME_CAP) { p += 12; continue; }
            size_t len = (size_t)(end - (q + 2));
            memcpy(names[n], q + 2, len);
            names[n][len] = 0;
            n++;
            p = end;
        }
    }
    fclose(f);
    return n;
}

int main(int argc, char **argv) {
    const char *backend = argc > 1 ? argv[1] : "src/metal.m";
    static char kernels[MAX_KERNELS][NAME_CAP];
    int n_kernels = scan_pipelines(backend, kernels);
    if (n_kernels < 0) {
        fprintf(stderr, "FAIL: cannot read %s — run this from the repo root, "
                "or pass the backend source as argv[1]\n", backend);
        return 1;
    }
    if (n_kernels < MIN_KERNELS) {
        fprintf(stderr, "FAIL: only %d mk_pipeline() lookups found in %s. The "
                "scan, not the backend, is what broke — a gate that checks "
                "nothing passes.\n", n_kernels, backend);
        return 1;
    }

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
        for (int i = 0; i < n_kernels; i++) {
            id<MTLFunction> fn = [lib newFunctionWithName:
                                      [NSString stringWithUTF8String:kernels[i]]];
            if (!fn) {
                fprintf(stderr, "FAIL: kernel %s is missing from the library "
                        "(mk_pipeline would return nil and the backend would "
                        "quietly run without it)\n", kernels[i]);
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
                        kernels[i],
                        perr ? perr.localizedDescription.UTF8String : "(no diagnostic)");
                missing++;
            }
            [p release];
            [fn release];
        }
        [lib release];
        [dev release];
        if (missing) return 1;
        printf("metal shaders: library compiles, %d kernels present "
               "(roster read from %s)\n", n_kernels, backend);
    }
    return 0;
}
