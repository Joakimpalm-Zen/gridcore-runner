// Failure-path + correctness tests for the GGUF requantizer (RNR-002/015).
//
// Two properties the July 2026 review flagged:
//   RNR-002  the writer must honor the file's declared general.alignment, not
//            a hardcoded 32 — otherwise every tensor offset past the first
//            misaligned one is read at the wrong address (silent corruption).
//   RNR-015  a failed/interrupted requant must never destroy the destination,
//            and an in-place requant must not truncate its own input.
//
// The fixtures are built here in-process (a minimal GGUF v3 writer) so the test
// is hermetic and pins the exact alignment that triggers the bug: tensor 0 is a
// 32-byte F32 tensor, so tensor 1's offset is 64 under a correct writer but 32
// under the old hardcoded-32 writer — and 32 is not 64-aligned, which the
// offset assertion below catches directly.
#include "runner.h"

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ALIGN 64

// ---- tiny little-endian GGUF writer ------------------------------------
typedef struct { uint8_t *b; size_t n, cap; } buf;

static void bput(buf *w, const void *p, size_t n) {
    if (w->n + n > w->cap) {
        w->cap = (w->n + n) * 2 + 64;
        w->b = realloc(w->b, w->cap);
        assert(w->b);
    }
    memcpy(w->b + w->n, p, n);
    w->n += n;
}
static void bu32(buf *w, uint32_t v) { bput(w, &v, 4); }
static void bu64(buf *w, uint64_t v) { bput(w, &v, 8); }
static void bstr(buf *w, const char *s) { uint64_t n = strlen(s); bu64(w, n); bput(w, s, n); }
static void bpad(buf *w, size_t align) {
    while (w->n % align) { uint8_t z = 0; bput(w, &z, 1); }
}

typedef struct { const char *name; uint64_t ne[2]; int n_dims; float *data; uint64_t n_elem; } tdesc;

// Write a GGUF v3 with one general.alignment=ALIGN KV and the given F32 tensors,
// laid out at ALIGN. If bad_type is set, tensor 0's stored type is corrupted to
// an unsupported value to exercise the pre-write rejection path.
static void write_gguf(const char *path, tdesc *ts, int nt, int bad_type) {
    // offsets: cumulative F32 sizes, ALIGN-padded
    uint64_t *off = calloc(nt, sizeof(uint64_t));
    uint64_t cur = 0;
    for (int i = 0; i < nt; i++) {
        off[i] = cur;
        cur += ts[i].n_elem * sizeof(float);
        cur = (cur + (ALIGN - 1)) & ~(uint64_t)(ALIGN - 1);
    }
    buf w = {0};
    bu32(&w, 0x46554747); bu32(&w, 3);
    bu64(&w, nt); bu64(&w, 1);            // n_tensors, n_kv
    bstr(&w, "general.alignment"); bu32(&w, GGUF_T_U32); bu32(&w, ALIGN);
    for (int i = 0; i < nt; i++) {
        bstr(&w, ts[i].name);
        bu32(&w, ts[i].n_dims);
        for (int d = 0; d < ts[i].n_dims; d++) bu64(&w, ts[i].ne[d]);
        bu32(&w, (i == 0 && bad_type) ? 999u : (uint32_t)T_F32);
        bu64(&w, off[i]);
    }
    bpad(&w, ALIGN);
    for (int i = 0; i < nt; i++) {
        // pad the data region so each tensor lands exactly at header_pad+off[i]
        // (header is already ALIGN-padded, and off[] is ALIGN-cumulative)
        bput(&w, ts[i].data, ts[i].n_elem * sizeof(float));
        bpad(&w, ALIGN);
    }
    FILE *f = fopen(path, "wb");
    assert(f);
    assert(fwrite(w.b, 1, w.n, f) == w.n);
    fclose(f);
    free(w.b); free(off);
}

static float ramp(uint64_t i) { return ((float)(int)(i % 64) - 32.0f) * 0.01f; }

// build the standard 3-tensor fixture; tensor 0 is 32 bytes so tensor 1's
// offset diverges between a 32- and 64-aligned writer
static void make_fixture(const char *path, int bad_type) {
    static float t0[8], t1[64 * 32], t2[64 * 16];
    for (int i = 0; i < 8; i++) t0[i] = 1.0f;
    for (uint64_t i = 0; i < 64 * 32; i++) t1[i] = ramp(i);
    for (uint64_t i = 0; i < 64 * 16; i++) t2[i] = ramp(i);
    tdesc ts[3] = {
        { "output_norm.weight",  {8, 1},     1, t0, 8 },
        { "blk.0.attn_q.weight", {64, 32},   2, t1, 64 * 32 },
        { "blk.0.ffn_gate.weight", {64, 16}, 2, t2, 64 * 16 },
    };
    write_gguf(path, ts, 3, bad_type);
}

static bool exists(const char *p) { FILE *f = fopen(p, "rb"); if (f) { fclose(f); return true; } return false; }

static void check_valid_q8(const char *path) {
    gguf_file g;
    assert(gguf_open(&g, path));
    assert(g.n_tensors == 3);
    // every tensor's data must sit on the declared alignment
    for (uint64_t i = 0; i < g.n_tensors; i++) {
        size_t rel = (uint8_t *)g.tensors[i].data - (uint8_t *)g.map;
        if (rel % ALIGN != 0) {
            fprintf(stderr, "tensor %s data at offset %zu is not %d-aligned "
                    "(alignment not honored)\n", g.tensors[i].name, rel, ALIGN);
            abort();
        }
    }
    gguf_tensor *q = gguf_find_tensor(&g, "blk.0.attn_q.weight");
    assert(q && q->type == T_Q8_0);
    gguf_tensor *nrm = gguf_find_tensor(&g, "output_norm.weight");
    assert(nrm && nrm->type == T_F32);            // 1-D norm stays F32
    // dequantize the first row and compare to the known ramp
    float row[64];
    dequant_row(q->type, q->data, row, 64);
    for (int c = 0; c < 64; c++) {
        float want = ramp((uint64_t)c);
        if (fabsf(row[c] - want) > 0.02f) {
            fprintf(stderr, "round-trip mismatch at col %d: got %f want %f "
                    "(data read from wrong offset?)\n", c, row[c], want);
            abort();
        }
    }
    gguf_close(&g);
}

static void cp(const char *from, const char *to) {
    FILE *a = fopen(from, "rb"), *b = fopen(to, "wb");
    assert(a && b);
    char buf2[4096]; size_t n;
    while ((n = fread(buf2, 1, sizeof(buf2), a)) > 0) assert(fwrite(buf2, 1, n, b) == n);
    fclose(a); fclose(b);
}

// MXFP4 read support: a hand-built OCP microscaling FP4 block must dequantize
// to (E8M0 block scale) x (E2M1 code). The expected magnitudes come from the
// spec, an independent oracle — not a copy of the runner's internal table.
static void check_mxfp4_dequant(void) {
    const float mag[16] = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6,
                           0, -0.5f, -1, -1.5f, -2, -3, -4, -6};
    unsigned char blk[17];
    blk[0] = 129;  // E8M0 exponent byte -> 2^(129-127) = 4.0
    for (int j = 0; j < 16; j++) {
        int lo = j % 8;         // element j     -> positive magnitudes
        int hi = 8 + (j % 8);   // element j+16  -> negative magnitudes
        blk[1 + j] = (unsigned char)((hi << 4) | lo);
    }
    float out[32];
    dequant_row(T_MXFP4, blk, out, 32);
    for (int j = 0; j < 16; j++) {
        assert(fabsf(out[j]      - mag[j % 8]       * 4.0f) < 1e-6f);
        assert(fabsf(out[j + 16] - mag[8 + (j % 8)] * 4.0f) < 1e-6f);
    }
    assert(ggml_type_size(T_MXFP4) == 17);
    assert(ggml_block_size(T_MXFP4) == 32);
    assert(ggml_type_supported(T_MXFP4));
    printf("ok: MXFP4 block dequantizes to spec (E8M0 scale x E2M1 code)\n");
}

int main(void) {
    f16_init();   // dequant_row's scale lookup table (unused by the quantizer
                  // itself, but the round-trip check below dequantizes)
    const char *in = "q_in.gguf", *out = "q_out.gguf";

    // RNR-002: alignment honored end-to-end + round-trip
    make_fixture(in, 0);
    assert(quantize_gguf(in, out, T_Q8_0) == 0);
    check_valid_q8(out);
    assert(!exists("q_out.gguf.partial"));        // temp cleaned up on success
    printf("ok: alignment honored, round-trips, no leftover .partial\n");

    // RNR-015: in-place requant must not truncate its own input
    const char *inplace = "q_inplace.gguf";
    cp(in, inplace);
    assert(quantize_gguf(inplace, inplace, T_Q8_0) == 0);
    check_valid_q8(inplace);
    printf("ok: in-place requant preserved and converted the model\n");

    // RNR-015: a failing requant must not destroy an existing destination
    const char *dest = "q_keep.gguf";
    const char sentinel[] = "DO NOT DESTROY THIS FILE";
    FILE *df = fopen(dest, "wb"); assert(df);
    assert(fwrite(sentinel, 1, sizeof(sentinel), df) == sizeof(sentinel));
    fclose(df);
    const char *bad = "q_bad.gguf";
    make_fixture(bad, 1);                          // unsupported tensor type
    assert(quantize_gguf(bad, dest, T_Q8_0) != 0); // must reject
    FILE *rf = fopen(dest, "rb"); assert(rf);
    char back[64]; size_t rn = fread(back, 1, sizeof(back), rf); fclose(rf);
    assert(rn == sizeof(sentinel) && memcmp(back, sentinel, rn) == 0);
    printf("ok: failed requant left the destination untouched\n");

    check_mxfp4_dequant();

    remove(in); remove(out); remove(inplace); remove(dest); remove(bad);
    printf("all quantize tests passed\n");
    return 0;
}
