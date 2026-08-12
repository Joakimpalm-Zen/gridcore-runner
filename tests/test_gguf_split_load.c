#include "gguf.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s WHOLE FIRST-SHARD\n", argv[0]);
        return 2;
    }
    gguf_file whole, split;
    assert(gguf_open(&whole, argv[1]));
    assert(gguf_open(&split, argv[2]));
    assert(whole.n_maps == 0);
    assert(split.n_maps == 3);
    assert(gguf_mapped_size(&whole) == whole.map_size);
    uint64_t mapped = 0;
    for (uint32_t i = 0; i < split.n_maps; i++) mapped += split.map_sizes[i];
    assert(gguf_mapped_size(&split) == mapped);
    assert(split.n_tensors == whole.n_tensors);

    for (uint64_t i = 0; i < whole.n_tensors; i++) {
        gguf_tensor *want = &whole.tensors[i];
        gguf_tensor *got = gguf_find_tensor(&split, want->name);
        assert(got != NULL);
        assert(got->type == want->type);
        assert(got->n_dims == want->n_dims);
        assert(memcmp(got->ne, want->ne, sizeof(want->ne)) == 0);
        assert(got->nbytes == want->nbytes);
        assert(memcmp(got->data, want->data, (size_t)want->nbytes) == 0);
    }

    gguf_close(&split);
    gguf_close(&whole);
    puts("split GGUF tensors are byte-identical to the whole file");
    return 0;
}
