// A failed model_load must not leave partially-owned state behind.
//
// The server swap path handles registered models by attempting a real
// model_load and then reporting a request error if the model is broken. When
// admission fails after gguf_open, the public model API must have already
// unwound the mmap/path allocations; otherwise a long-lived server can leak
// one partial model per bad request.
#include "runner.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    unsigned char *b;
    size_t n, cap;
} buf;

static void bput(buf *w, const void *p, size_t n) {
    if (w->n + n > w->cap) {
        size_t nc = (w->n + n) * 2 + 64;
        unsigned char *nb = realloc(w->b, nc);
        assert(nb);
        w->b = nb;
        w->cap = nc;
    }
    memcpy(w->b + w->n, p, n);
    w->n += n;
}

static void bu32(buf *w, uint32_t v) { bput(w, &v, sizeof(v)); }
static void bu64(buf *w, uint64_t v) { bput(w, &v, sizeof(v)); }
static void bstr(buf *w, const char *s) {
    uint64_t n = strlen(s);
    bu64(w, n);
    bput(w, s, (size_t)n);
}
static void bpad(buf *w, size_t align) {
    while (w->n % align) {
        unsigned char z = 0;
        bput(w, &z, 1);
    }
}

static void write_bad_arch_gguf(const char *path) {
    buf w = {0};
    bu32(&w, 0x46554747u);       // GGUF
    bu32(&w, 3);                 // version
    bu64(&w, 0);                 // n_tensors
    bu64(&w, 1);                 // n_kv
    bstr(&w, "general.architecture");
    bu32(&w, GGUF_T_STR);
    bstr(&w, "granite");         // explicitly refused after gguf_open
    bpad(&w, 32);

    FILE *f = fopen(path, "wb");
    assert(f);
    assert(fwrite(w.b, 1, w.n, f) == w.n);
    fclose(f);
    free(w.b);
}

int main(void) {
    const char *path = "bad-arch-load.gguf";
    write_bad_arch_gguf(path);

    model_params p = {0};
    p.gpu_mode = GPU_OFF;
    p.n_threads = 1;
    p.n_ctx = 64;
    p.n_batch = 4;

    model_t m;
    memset(&m, 0xA5, sizeof(m));
    assert(!model_load(&m, path, &p));

    assert(m.gf.map == NULL);
    assert(m.path == NULL);
    assert(m.layers == NULL);
    assert(m.kcache == NULL);
    assert(m.vcache == NULL);
    assert(m.gpu == NULL);
    assert(m.gpu_owner == NULL);

    // A caller that keeps the old "free after failure" habit must still be safe.
    model_free(&m);
    remove(path);
    printf("model load failure cleanup ok\n");
    return 0;
}
