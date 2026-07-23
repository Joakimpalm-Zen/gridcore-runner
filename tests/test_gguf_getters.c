// RNR-010: GGUF typed getters must validate the source type, sign, range, and
// finiteness instead of reinterpreting the union — and gguf_open must reject
// duplicate metadata keys / tensor names. A crafted or corrupt file must not
// turn into huge or type-confused geometry.
#include "runner.h"

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { unsigned char *b; size_t n, cap; } buf;
static void bput(buf *w, const void *p, size_t n) {
    if (w->n + n > w->cap) { w->cap = (w->n + n) * 2 + 64; w->b = realloc(w->b, w->cap); assert(w->b); }
    memcpy(w->b + w->n, p, n); w->n += n;
}
static void bu32(buf *w, uint32_t v) { bput(w, &v, 4); }
static void bu64(buf *w, uint64_t v) { bput(w, &v, 8); }
static void bi32(buf *w, int32_t v)  { bput(w, &v, 4); }
static void bf32(buf *w, float v)    { bput(w, &v, 4); }
static void bstr(buf *w, const char *s) { uint64_t n = strlen(s); bu64(w, n); bput(w, s, n); }
static void bkey(buf *w, const char *k, uint32_t type) { bstr(w, k); bu32(w, type); }

// assemble a GGUF from a prebuilt KV blob and tensor-info blob, padded to 32
static void write_gguf(const char *path, buf *kv, uint64_t nkv, buf *ti, uint64_t nt) {
    buf w = {0};
    bu32(&w, 0x46554747); bu32(&w, 3); bu64(&w, nt); bu64(&w, nkv);
    if (kv) bput(&w, kv->b, kv->n);
    if (ti) bput(&w, ti->b, ti->n);
    while (w.n % 32) { unsigned char z = 0; bput(&w, &z, 1); }
    FILE *f = fopen(path, "wb"); assert(f);
    assert(fwrite(w.b, 1, w.n, f) == w.n);
    fclose(f); free(w.b);
}

static void test_getter_validation(void) {
    // one file carrying every awkward scalar type
    buf kv = {0};
    uint64_t n = 0;
    bkey(&kv, "good_u32", GGUF_T_U32);   bu32(&kv, 42);                 n++;
    bkey(&kv, "neg_i32",  GGUF_T_I32);   bi32(&kv, -5);                 n++;
    bkey(&kv, "huge_u64", GGUF_T_U64);   bu64(&kv, (1ull << 40));       n++;
    bkey(&kv, "nan_f32",  GGUF_T_F32);   bf32(&kv, NAN);                n++;
    bkey(&kv, "frac_f32", GGUF_T_F32);   bf32(&kv, 3.5f);               n++;
    bkey(&kv, "true_b",   GGUF_T_BOOL);  { unsigned char t = 1; bput(&kv, &t, 1); } n++;
    bkey(&kv, "a_str",    GGUF_T_STR);   bstr(&kv, "hello");            n++;

    const char *path = "getters.gguf";
    write_gguf(path, &kv, n, NULL, 0);
    free(kv.b);

    gguf_file g;
    assert(gguf_open(&g, path));

    // u32: valid unsigned survives; negative / out-of-range / non-integral all
    // fall back to the default rather than becoming garbage geometry
    assert(gguf_get_u32(&g, "good_u32", 99) == 42);
    assert(gguf_get_u32(&g, "neg_i32",  99) == 99);   // -5 must not become 4294967291
    assert(gguf_get_u32(&g, "huge_u64", 99) == 99);   // > UINT32_MAX
    assert(gguf_get_u32(&g, "nan_f32",  99) == 99);
    assert(gguf_get_u32(&g, "frac_f32", 99) == 99);   // 3.5 is not an integer
    assert(gguf_get_u32(&g, "true_b",   99) == 99);   // bool is not a u32
    assert(gguf_get_u32(&g, "a_str",    99) == 99);
    assert(gguf_get_u32(&g, "absent",   99) == 99);

    // f32: floats pass through, ints cast, NaN/non-numeric fall back
    assert(gguf_get_f32(&g, "frac_f32", 0.0f) == 3.5f);
    assert(gguf_get_f32(&g, "neg_i32",  0.0f) == -5.0f);
    assert(gguf_get_f32(&g, "good_u32", 0.0f) == 42.0f);
    assert(gguf_get_f32(&g, "nan_f32",  1.0f) == 1.0f);   // NaN -> default
    assert(gguf_get_f32(&g, "a_str",    1.0f) == 1.0f);

    // bool: only a real bool key answers true
    assert(gguf_get_bool(&g, "true_b",  false) == true);
    assert(gguf_get_bool(&g, "good_u32", false) == false); // u32 is not a bool
    assert(gguf_get_bool(&g, "absent",  true)  == true);

    gguf_close(&g);
    remove(path);
    printf("ok: typed getters validate type/sign/range/finiteness\n");
}

static void write_tensor_info(buf *ti, const char *name) {
    bstr(ti, name);
    bu32(ti, 1);            // n_dims
    bu64(ti, 1);            // ne[0]
    bu32(ti, 0);            // type F32
    bu64(ti, 0);            // offset
}

static void test_duplicate_keys_rejected(void) {
    buf kv = {0};
    bkey(&kv, "dup.key", GGUF_T_U32); bu32(&kv, 1);
    bkey(&kv, "dup.key", GGUF_T_U32); bu32(&kv, 2);
    const char *path = "dupkey.gguf";
    write_gguf(path, &kv, 2, NULL, 0);
    free(kv.b);
    gguf_file g;
    assert(!gguf_open(&g, path) && "duplicate metadata key must be rejected");
    remove(path);
    printf("ok: duplicate metadata keys rejected\n");
}

static void test_duplicate_tensor_names_rejected(void) {
    buf ti = {0};
    write_tensor_info(&ti, "blk.0.weight");
    write_tensor_info(&ti, "blk.0.weight");
    const char *path = "duptensor.gguf";
    write_gguf(path, NULL, 0, &ti, 2);
    free(ti.b);
    gguf_file g;
    assert(!gguf_open(&g, path) && "duplicate tensor name must be rejected");
    remove(path);
    printf("ok: duplicate tensor names rejected\n");
}

int main(void) {
    test_getter_validation();
    test_duplicate_keys_rejected();
    test_duplicate_tensor_names_rejected();
    printf("all gguf getter tests passed\n");
    return 0;
}
