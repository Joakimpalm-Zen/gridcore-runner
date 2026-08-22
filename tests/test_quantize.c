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

// `type` is T_F32 (0) unless set; T_BF16 stores the top 16 bits of each float,
// which is exactly what a bf16 conversion does and is all this writer needs to
// build a fixture with a type that is neither f32 nor f16.
typedef struct { const char *name; uint64_t ne[2]; int n_dims; float *data;
                 uint64_t n_elem; int type; } tdesc;

static size_t tdesc_bytes(const tdesc *t) {
    return (size_t)t->n_elem * (t->type == T_BF16 ? 2 : sizeof(float));
}

// Write a GGUF v3 with one general.alignment=ALIGN KV and the given tensors,
// laid out at ALIGN. If bad_type is set, tensor 0's stored type is corrupted to
// an unsupported value to exercise the pre-write rejection path.
static void write_gguf(const char *path, tdesc *ts, int nt, int bad_type) {
    // offsets: cumulative payload sizes, ALIGN-padded
    uint64_t *off = calloc(nt, sizeof(uint64_t));
    uint64_t cur = 0;
    for (int i = 0; i < nt; i++) {
        off[i] = cur;
        cur += tdesc_bytes(&ts[i]);
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
        bu32(&w, (i == 0 && bad_type) ? 999u : (uint32_t)ts[i].type);
        bu64(&w, off[i]);
    }
    bpad(&w, ALIGN);
    for (int i = 0; i < nt; i++) {
        // pad the data region so each tensor lands exactly at header_pad+off[i]
        // (header is already ALIGN-padded, and off[] is ALIGN-cumulative)
        if (ts[i].type == T_BF16) {
            for (uint64_t j = 0; j < ts[i].n_elem; j++) {
                uint32_t u;
                memcpy(&u, &ts[i].data[j], 4);
                uint16_t h = (uint16_t)(u >> 16);
                bput(&w, &h, 2);
            }
        } else {
            bput(&w, ts[i].data, (size_t)ts[i].n_elem * sizeof(float));
        }
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
        { "output_norm.weight",  {8, 1},     1, t0, 8, T_F32 },
        { "blk.0.attn_q.weight", {64, 32},   2, t1, 64 * 32, T_F32 },
        { "blk.0.ffn_gate.weight", {64, 16}, 2, t2, 64 * 16, T_F32 },
    };
    write_gguf(path, ts, 3, bad_type);
}

static bool exists(const char *p) { FILE *f = fopen(p, "rb"); if (f) { fclose(f); return true; } return false; }

static void env_set(const char *k, const char *v) {
#ifdef _WIN32
    assert(_putenv_s(k, v) == 0);
#else
    assert(setenv(k, v, 1) == 0);
#endif
}

static void env_unset(const char *k) {
#ifdef _WIN32
    assert(_putenv_s(k, "") == 0);
#else
    assert(unsetenv(k) == 0);
#endif
}

static void check_bytes(const char *path, const void *want, size_t n) {
    FILE *rf = fopen(path, "rb");
    assert(rf);
    unsigned char back[128];
    assert(n <= sizeof(back));
    size_t rn = fread(back, 1, sizeof(back), rf);
    fclose(rf);
    assert(rn == n && memcmp(back, want, n) == 0);
}

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

static bool files_equal(const char *pa, const char *pb) {
    FILE *a = fopen(pa, "rb"), *b = fopen(pb, "rb");
    assert(a && b);
    bool eq = true;
    for (;;) {
        char ba[4096], bb[4096];
        size_t na = fread(ba, 1, sizeof(ba), a);
        size_t nb = fread(bb, 1, sizeof(bb), b);
        if (na != nb || memcmp(ba, bb, na) != 0) { eq = false; break; }
        if (na == 0) break;
    }
    fclose(a); fclose(b);
    return eq;
}

// tiny adapter-GGUF writer (the shape model_lora_save emits): v3 header,
// general.type=adapter / adapter.type=lora / adapter.lora.alpha, F32
// lora_a/lora_b tensors. `arch` NULL omits general.architecture, matching
// the base fixtures here (which carry none either — both read back "").
typedef struct { const char *name; uint64_t ne[2]; const float *data; } adesc;
// ttype: T_F32 writes the floats verbatim; T_F16 rounds them to half —
// the format llama.cpp's converter emits, which the merge must accept
static void write_adapter(const char *path, float alpha, const char *arch,
                          const adesc *ts, int nt, int ttype) {
    buf w = {0};
    bu32(&w, 0x46554747); bu32(&w, 3);
    bu64(&w, (uint64_t)nt); bu64(&w, 4 + (arch != NULL));
    bstr(&w, "general.alignment"); bu32(&w, GGUF_T_U32); bu32(&w, ALIGN);
    bstr(&w, "general.type"); bu32(&w, GGUF_T_STR); bstr(&w, "adapter");
    bstr(&w, "adapter.type"); bu32(&w, GGUF_T_STR); bstr(&w, "lora");
    bstr(&w, "adapter.lora.alpha"); bu32(&w, GGUF_T_F32); bput(&w, &alpha, 4);
    if (arch) {
        bstr(&w, "general.architecture");
        bu32(&w, GGUF_T_STR);
        bstr(&w, arch);
    }
    uint64_t off = 0;
    size_t esz = ttype == T_F16 ? 2 : 4;
    for (int i = 0; i < nt; i++) {
        bstr(&w, ts[i].name);
        bu32(&w, 2);
        bu64(&w, ts[i].ne[0]); bu64(&w, ts[i].ne[1]);
        bu32(&w, (uint32_t)ttype);
        bu64(&w, off);
        off += ts[i].ne[0] * ts[i].ne[1] * esz;
        off = (off + (ALIGN - 1)) & ~(uint64_t)(ALIGN - 1);
    }
    bpad(&w, ALIGN);
    for (int i = 0; i < nt; i++) {
        uint64_t n = ts[i].ne[0] * ts[i].ne[1];
        if (ttype == T_F16) {
            for (uint64_t j = 0; j < n; j++) {
                f16_t h = f32_to_f16(ts[i].data[j]);
                bput(&w, &h, 2);
            }
        } else {
            bput(&w, ts[i].data, (size_t)(n * 4));
        }
        bpad(&w, ALIGN);
    }
    FILE *f = fopen(path, "wb");
    assert(f);
    assert(fwrite(w.b, 1, w.n, f) == w.n);
    fclose(f);
    free(w.b);
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

// finding B (2026-08-11 external evaluation): general.file_type must
// describe the OUTPUT, not the parent. The published gemma artifact carried
// its Q4_K_M parent's declaration over a mixed Q4_0/Q4_K result, and
// llama.cpp printed the stale label in its own model line.
static void check_file_type(const char *path, uint32_t want) {
    gguf_file g;
    assert(gguf_open(&g, path));
    uint32_t got = gguf_get_u32(&g, "general.file_type", 9999);
    if (got != want) {
        fprintf(stderr, "file_type: got %u want %u for %s\n", got, want, path);
        assert(0);
    }
    gguf_close(&g);
}


// ---- QAT grid-exact repack -----------------------------------------------
//
// A quantization-aware-trained checkpoint's weights ALREADY sit on the q4_0
// grid: every value is d*(q-8) for one per-block scale d and integer q in
// [0,15]. Requantizing such a block should therefore be LOSSLESS -- the answer
// is already there to be read off.
//
// The stock quantizer does not read it off. It derives d from the value of
// largest magnitude, `d = vmax / -8`, which recovers the true scale only when
// the block actually contains q == 0. A QAT block whose codes never reach 0
// yields a scale that is too small, and the values at the other end of the
// range then saturate at q == 15 and come back WRONG -- a pure repack that
// loses information the source had already committed to.
//
// This fixture is that case, built so the arithmetic is checkable by eye:
// d = 0.5 (exact in fp16), codes 1..15 and no 0, so the extreme values are
// -3.5 and +3.5. `vmax` is whichever of those the scan meets first, giving
// d = 3.5/8 = 0.4375, and +3.5 then needs q = 16 -- clamped to 15, i.e.
// 3.0625. Off by 0.4375 on a repack that should have been exact.
#define QAT_D 0.5f
static void fill_qat_grid(float *dst, int n_blocks) {
    for (int b = 0; b < n_blocks; b++)
        for (int j = 0; j < 32; j++) {
            // codes 1..15, never 0, so `vmax / -8` cannot see the true scale
            int code = 1 + ((b * 32 + j) % 15);
            dst[b * 32 + j] = QAT_D * (float)(code - 8);
        }
}

// Every value must come back bit-for-bit. Not "close": the source was already
// on the grid, so any difference at all is information destroyed by a repack.
static void check_qat_exact(const char *path, const float *want, size_t n) {
    gguf_file g;
    assert(gguf_open(&g, path));
    const gguf_tensor *t = NULL;
    for (uint64_t i = 0; i < g.n_tensors; i++)
        if (!strcmp(g.tensors[i].name, "blk.0.attn_q.weight")) t = &g.tensors[i];
    assert(t && t->type == T_Q4_0);
    float *got = malloc(sizeof(float) * n);
    assert(got);
    dequant_row(t->type, t->data, got, (int)n);
    size_t bad = 0;
    double worst = 0;
    size_t first = 0;
    for (size_t i = 0; i < n; i++) {
        if (got[i] == want[i]) continue;
        if (!bad) first = i;
        bad++;
        double d = fabs((double)got[i] - (double)want[i]);
        if (d > worst) worst = d;
    }
    if (bad) {
        fprintf(stderr, "QAT grid repack lost %zu/%zu values; first at %zu "
                "(want %.6f got %.6f), worst |delta| %.6f\n",
                bad, n, first, (double)want[first], (double)got[first], worst);
        assert(0);
    }
    free(got);
    gguf_close(&g);
}

// Q3_K writer tracer bullet: a hermetic expert tensor already on a simple
// Q3_K grid must survive the writer -> existing reader round trip exactly,
// including zero.  The attention tensor is outside the type-plan rule and
// must retain both its type and its source bytes.
static void check_q3_plan_exact(const char *path, const float *want,
                                const float *kept, size_t n) {
    gguf_file g;
    assert(gguf_open(&g, path));
    gguf_tensor *q = gguf_find_tensor(&g, "blk.0.ffn_down_exps.weight");
    gguf_tensor *k = gguf_find_tensor(&g, "blk.0.attn_q.weight");
    assert(q && q->type == T_Q3_K);
    assert(k && k->type == T_F32);
    assert(memcmp(k->data, kept, n * sizeof(float)) == 0);
    float *got = malloc(n * sizeof(float));
    assert(got);
    dequant_row(q->type, q->data, got, (int)n);
    for (size_t i = 0; i < n; i++) {
        if (got[i] != want[i]) {
            fprintf(stderr, "Q3_K exact round-trip mismatch at %zu: got %.9g want %.9g\n",
                    i, (double)got[i], (double)want[i]);
            assert(0);
        }
        if (want[i] == 0.0f)
            assert(memcmp(&got[i], &want[i], sizeof(float)) == 0);
    }
    free(got);
    gguf_close(&g);
}

int main(void) {
    f16_init();   // dequant_row's scale lookup table (unused by the quantizer
                  // itself, but the round-trip check below dequantizes)
    const char *in = "q_in.gguf", *out = "q_out.gguf";

    // RNR-002: alignment honored end-to-end + round-trip
    make_fixture(in, 0);
    assert(quantize_gguf(in, out, T_Q8_0, NULL) == 0);
    check_valid_q8(out);
    assert(!exists("q_out.gguf.partial"));        // temp cleaned up on success
    printf("ok: alignment honored, round-trips, no leftover .partial\n");

    // finding B: the declared file_type follows the OUTPUT histogram.
    // f32 fixture -> q8_0: MOSTLY_Q8_0 (7). That file -> q4_0: every
    // quantized tensor converts, so MOSTLY_Q4_0 (2) — the evaluator's exact
    // local repro, which used to keep MOSTLY_Q8_0. And q4_0 -> q8_0: the
    // never-grow retention rule keeps every tensor at Q4_0, so the honest
    // declaration STAYS MOSTLY_Q4_0 (2), not the requested target's 7.
    check_file_type(out, 7);
    const char *q40 = "q_ft40.gguf", *q80back = "q_ft80.gguf";
    assert(quantize_gguf(out, q40, T_Q4_0, NULL) == 0);
    check_file_type(q40, 2);
    assert(quantize_gguf(q40, q80back, T_Q8_0, NULL) == 0);
    check_file_type(q80back, 2);
    printf("ok: general.file_type follows the output histogram\n");

    // QAT grid-exact repack: a source already on the q4_0 grid must survive
    // a q4_0 repack bit-for-bit.
    {
        const char *qat = "q_qat_in.gguf", *qat_out = "q_qat_out.gguf";
        enum { QAT_BLOCKS = 64, QAT_N = QAT_BLOCKS * 32 };
        static float qw[QAT_N], qn[8];
        fill_qat_grid(qw, QAT_BLOCKS);
        for (int i = 0; i < 8; i++) qn[i] = 1.0f;
        tdesc qts[2] = {
            { "output_norm.weight",  {8, 1},  1, qn, 8, T_F32 },
            { "blk.0.attn_q.weight", {32, QAT_BLOCKS}, 2, qw, QAT_N, T_F32 },
        };
        write_gguf(qat, qts, 2, 0);
        assert(quantize_gguf(qat, qat_out, T_Q4_0, NULL) == 0);
        check_qat_exact(qat_out, qw, QAT_N);
        remove(qat); remove(qat_out);
        printf("ok: a QAT-grid source repacks to q4_0 bit-exactly\n");
    }


    // Q3_K selective writer: only the expert tensor changes precision.
    {
        const char *q3in = "q_q3_in.gguf", *q3out = "q_q3_out.gguf";
        const char *plan = "q_q3_plan.json";
        enum { Q3_N = 256 };
        static float expert[Q3_N], attention[Q3_N];
        for (int i = 0; i < Q3_N; i++) {
            expert[i] = 0.25f * (float)((i % 8) - 4);
            attention[i] = (float)i * 0.001f + 1.0f;
        }
        tdesc q3ts[2] = {
            { "blk.0.ffn_down_exps.weight", {Q3_N, 1}, 2, expert, Q3_N, T_F32 },
            { "blk.0.attn_q.weight",        {Q3_N, 1}, 2, attention, Q3_N, T_F32 },
        };
        write_gguf(q3in, q3ts, 2, 0);
        FILE *pf = fopen(plan, "wb");
        assert(pf);
        const char plan_json[] =
            "{\"default\":\"keep\",\"rules\":[{\"match\":\"_exps.weight\",\"type\":\"q3_k\"}]}";
        assert(fwrite(plan_json, 1, sizeof(plan_json) - 1, pf) == sizeof(plan_json) - 1);
        fclose(pf);
        assert(quantize_gguf_plan(q3in, q3out, T_KEEP, NULL, plan) == 0);
        check_q3_plan_exact(q3out, expert, attention, Q3_N);
        remove(q3in); remove(q3out); remove(plan);
        printf("ok: Q3_K plan round-trips exactly and preserves unselected bytes\n");
    }

    // A type plan naming a K-quant for a row the K-quant cannot describe.
    // should_quantize only demands ne[0] % 32 == 0, but Q3_K's super-block is
    // 256 wide: a 288-wide row has no whole number of Q3_K blocks, so the
    // writer would emit one block (256 values, 32 dropped) under a header that
    // still says 288 -- and gguf.c rejects ne[0] % block_size != 0 at load, so
    // the requant "succeeds" and produces a file nothing can open. Such a
    // tensor must keep its own type instead.
    {
        const char *win = "q_width_in.gguf", *wout = "q_width_out.gguf";
        const char *plan = "q_width_plan.json";
        enum { W_N = 288 };                 // 9 * 32, not a multiple of 256
        static float wide[W_N], narrow[256];
        for (int i = 0; i < W_N; i++) wide[i] = 0.25f * (float)((i % 8) - 4);
        for (int i = 0; i < 256; i++) narrow[i] = 0.25f * (float)((i % 8) - 4);
        tdesc wts[2] = {
            { "blk.0.ffn_down_exps.weight", {W_N, 1}, 2, wide, W_N, T_F32 },
            { "blk.1.ffn_down_exps.weight", {256, 1}, 2, narrow, 256, T_F32 },
        };
        write_gguf(win, wts, 2, 0);
        FILE *pf = fopen(plan, "wb");
        assert(pf);
        const char plan_json[] =
            "{\"default\":\"keep\",\"rules\":[{\"match\":\"_exps.weight\",\"type\":\"q3_k\"}]}";
        assert(fwrite(plan_json, 1, sizeof(plan_json) - 1, pf) == sizeof(plan_json) - 1);
        fclose(pf);
        assert(quantize_gguf_plan(win, wout, T_KEEP, NULL, plan) == 0);
        gguf_file g;
        assert(gguf_open(&g, wout));        // the whole point: it must load
        gguf_tensor *w288 = gguf_find_tensor(&g, "blk.0.ffn_down_exps.weight");
        gguf_tensor *w256 = gguf_find_tensor(&g, "blk.1.ffn_down_exps.weight");
        assert(w288 && w288->type == T_F32);
        assert(memcmp(w288->data, wide, sizeof(wide)) == 0);
        assert(w256 && w256->type == T_Q3_K);   // the representable row still converts
        gguf_close(&g);
        remove(win); remove(wout); remove(plan);
        printf("ok: a row too narrow for the plan's block keeps its own type\n");
    }

    // The MoE router decides WHICH expert runs. Quantizing it turns a small
    // weight error into a swapped FFN, and it is the one weight matrix whose
    // cost of staying exact is negligible -- which is why every reference MoE
    // GGUF ships ffn_gate_inp.weight at F32 beside 4-bit experts. A whole-file
    // --quant must leave it alone; the experts beside it must still convert.
    {
        const char *rin = "q_router_in.gguf", *rout = "q_router_out.gguf";
        enum { R_EMB = 32, R_EXP = 4, R_FF = 64 };
        static float router[R_EMB * R_EXP], shexp[R_EMB], experts[R_EMB * R_FF * R_EXP];
        for (int i = 0; i < R_EMB * R_EXP; i++) router[i] = ramp((uint64_t)i);
        for (int i = 0; i < R_EMB; i++) shexp[i] = ramp((uint64_t)i + 7);
        for (int i = 0; i < R_EMB * R_FF * R_EXP; i++) experts[i] = ramp((uint64_t)i);
        tdesc rts[3] = {
            { "blk.0.ffn_gate_inp.weight",       {R_EMB, R_EXP}, 2, router,
              R_EMB * R_EXP, T_F32 },
            { "blk.0.ffn_gate_inp_shexp.weight", {R_EMB, 1},     2, shexp, R_EMB, T_F32 },
            { "blk.0.ffn_gate_exps.weight",      {R_EMB, R_FF * R_EXP}, 2, experts,
              R_EMB * R_FF * R_EXP, T_F32 },
        };
        write_gguf(rin, rts, 3, 0);
        assert(quantize_gguf(rin, rout, T_Q4_0, NULL) == 0);
        gguf_file g;
        assert(gguf_open(&g, rout));
        gguf_tensor *r = gguf_find_tensor(&g, "blk.0.ffn_gate_inp.weight");
        gguf_tensor *s = gguf_find_tensor(&g, "blk.0.ffn_gate_inp_shexp.weight");
        gguf_tensor *e = gguf_find_tensor(&g, "blk.0.ffn_gate_exps.weight");
        assert(r && r->type == T_F32 && memcmp(r->data, router, sizeof(router)) == 0);
        assert(s && s->type == T_F32 && memcmp(s->data, shexp, sizeof(shexp)) == 0);
        assert(e && e->type == T_Q4_0);
        gguf_close(&g);
        remove(rin); remove(rout);
        printf("ok: the MoE router survives a q4_0 requant untouched\n");
    }

    // "keep" in a --type-plan is documented (quants.h) as "leave this tensor
    // exactly as it is". It did that only for tensors should_quantize accepts;
    // everything else fell through to the whole-file rule that pushes a
    // non-f16 tensor to F32, so a bf16 norm under {"default":"keep"} came out
    // F32 -- a converted, grown tensor in a plan that converts nothing.
    {
        const char *kin = "q_keeptype_in.gguf", *kout = "q_keeptype_out.gguf";
        const char *plan = "q_keeptype_plan.json";
        static float norm[8], wgt[64 * 4];
        for (int i = 0; i < 8; i++) norm[i] = 0.5f + (float)i;
        for (int i = 0; i < 64 * 4; i++) wgt[i] = ramp((uint64_t)i);
        tdesc kts[2] = {
            { "output_norm.weight",  {8, 1},   1, norm, 8, T_BF16 },
            { "blk.0.attn_q.weight", {64, 4},  2, wgt, 64 * 4, T_BF16 },
        };
        write_gguf(kin, kts, 2, 0);
        FILE *pf = fopen(plan, "wb");
        assert(pf);
        const char plan_json[] = "{\"default\":\"keep\"}";
        assert(fwrite(plan_json, 1, sizeof(plan_json) - 1, pf) == sizeof(plan_json) - 1);
        fclose(pf);
        assert(quantize_gguf_plan(kin, kout, T_KEEP, NULL, plan) == 0);
        gguf_file g;
        assert(gguf_open(&g, kout));
        gguf_tensor *n1 = gguf_find_tensor(&g, "output_norm.weight");
        gguf_tensor *w1 = gguf_find_tensor(&g, "blk.0.attn_q.weight");
        assert(n1 && n1->type == T_BF16 && n1->nbytes == 8 * 2);
        assert(w1 && w1->type == T_BF16);
        gguf_close(&g);
        remove(kin); remove(kout); remove(plan);
        printf("ok: a type plan's \"keep\" keeps a type this tool cannot write\n");
    }

    // Q4_K / Q6_K writers: round-trip through the production dq_ readers
    // within each type's grid resolution, and byte-deterministic across
    // runs. A layout disagreement with quants.c cannot pass the error
    // bound; a nondeterministic search cannot pass the byte compare.
    {
        const char *kin = "q_kq_in.gguf", *k1 = "q_kq1.gguf", *k2 = "q_kq2.gguf";
        enum { K_N = 256, K_ROWS = 4 };
        static float kw[K_N * K_ROWS], kn[8];
        for (int i = 0; i < K_N * K_ROWS; i++)
            kw[i] = 0.01f * (float)((i * 7 % 96) - 48) + 0.003f * (float)(i % 5);
        for (int i = 0; i < 8; i++) kn[i] = 1.0f;
        tdesc kts[2] = {
            { "output_norm.weight",  {8, 1},        1, kn, 8, T_F32 },
            { "blk.0.attn_q.weight", {K_N, K_ROWS}, 2, kw, K_N * K_ROWS, T_F32 },
        };
        write_gguf(kin, kts, 2, 0);
        const struct { int t; float bound; } KQ[] = {
            { T_Q4_K, 0.04f }, { T_Q6_K, 0.02f },
        };
        for (size_t k = 0; k < sizeof(KQ) / sizeof(*KQ); k++) {
            assert(quantize_gguf(kin, k1, KQ[k].t, NULL) == 0);
            assert(quantize_gguf(kin, k2, KQ[k].t, NULL) == 0);
            assert(files_equal(k1, k2));
            gguf_file g;
            assert(gguf_open(&g, k1));
            gguf_tensor *q = gguf_find_tensor(&g, "blk.0.attn_q.weight");
            assert(q && (int)q->type == KQ[k].t);
            size_t rs = ggml_row_size(q->type, K_N);
            for (int r = 0; r < K_ROWS; r++) {
                float row[K_N];
                dequant_row(q->type, (const uint8_t *)q->data + (size_t)r * rs,
                            row, K_N);
                for (int c = 0; c < K_N; c++) {
                    float want = kw[r * K_N + c];
                    if (fabsf(row[c] - want) > KQ[k].bound) {
                        fprintf(stderr, "%s round-trip: row %d col %d got %f "
                                "want %f\n", ggml_type_name(KQ[k].t), r, c,
                                row[c], want);
                        abort();
                    }
                }
            }
            gguf_close(&g);
        }
        remove(kin); remove(k1); remove(k2);
        printf("ok: q4_k/q6_k round-trip within grid resolution, "
               "deterministically\n");
    }

    // --merge-lora: fold scale*B·A into the base weights through the writer.
    // On an F32 base the merged floats are asserted BYTE-EXACTLY against the
    // documented fmaf chain; untouched tensors copy verbatim; a zero adapter
    // yields the identical file a keep-type requant yields; hostile adapters
    // (wrong shape, half a pair, unhooked projection, wrong arch) refuse.
    {
        const char *base = "q_mrg_base.gguf", *ad = "q_mrg_ad.gguf";
        const char *out1 = "q_mrg1.gguf", *out2 = "q_mrg2.gguf";
        enum { M_IN = 64, M_OUT = 32, M_R = 2 };
        const float alpha = 4.0f;             // scale = alpha/r = 2.0
        static float mw[M_IN * M_OUT], mg[M_IN * 16], mn[8];
        static float ma[M_R * M_IN], mb[M_OUT * M_R], mz[M_OUT * M_R];
        for (int i = 0; i < M_IN * M_OUT; i++) mw[i] = ramp((uint64_t)i);
        for (int i = 0; i < M_IN * 16; i++) mg[i] = ramp((uint64_t)i + 3);
        for (int i = 0; i < 8; i++) mn[i] = 1.0f;
        for (int k = 0; k < M_R; k++)
            for (int i = 0; i < M_IN; i++)
                ma[k * M_IN + i] = 0.001f * (float)(i + 17 * k);
        for (int j = 0; j < M_OUT; j++)
            for (int k = 0; k < M_R; k++)
                mb[j * M_R + k] = 0.01f * (float)((j % 5) - 2) +
                                  0.005f * (float)k;
        memset(mz, 0, sizeof(mz));
        tdesc bts[3] = {
            { "output_norm.weight",    {8, 1},         1, mn, 8, T_F32 },
            { "blk.0.attn_q.weight",   {M_IN, M_OUT},  2, mw, M_IN * M_OUT, T_F32 },
            { "blk.0.ffn_gate.weight", {M_IN, 16},     2, mg, M_IN * 16, T_F32 },
        };
        write_gguf(base, bts, 3, 0);
        adesc ats[2] = {
            { "blk.0.attn_q.weight.lora_a", {M_IN, M_R},  ma },
            { "blk.0.attn_q.weight.lora_b", {M_R, M_OUT}, mb },
        };
        write_adapter(ad, alpha, NULL, ats, 2, T_F32);
        assert(merge_lora_gguf(base, ad, 1.0f, out1, T_KEEP) == 0);
        assert(merge_lora_gguf(base, ad, 1.0f, out2, T_KEEP) == 0);
        assert(files_equal(out1, out2));      // byte-deterministic
        gguf_file g;
        assert(gguf_open(&g, out1));
        gguf_tensor *q = gguf_find_tensor(&g, "blk.0.attn_q.weight");
        gguf_tensor *fg = gguf_find_tensor(&g, "blk.0.ffn_gate.weight");
        assert(q && q->type == T_F32);
        assert(fg && fg->type == T_F32 && memcmp(fg->data, mg, sizeof(mg)) == 0);
        const float *got = (const float *)q->data;
        for (int j = 0; j < M_OUT; j++)
            for (int i = 0; i < M_IN; i++) {
                float acc = 0.0f;
                for (int k = 0; k < M_R; k++)
                    acc = fmaf(mb[j * M_R + k], ma[k * M_IN + i], acc);
                float want = fmaf(alpha / (float)M_R, acc, mw[j * M_IN + i]);
                assert(memcmp(&got[j * M_IN + i], &want, 4) == 0);
            }
        gguf_close(&g);
        // zero delta: the merge is byte-identical to a keep-type requant —
        // untouched weights are copied, not churned through dequant/requant
        const char *keep = "q_mrg_keep.gguf", *zad = "q_mrg_zad.gguf";
        const char *zout = "q_mrg_zout.gguf";
        adesc zts[2] = {
            { "blk.0.attn_q.weight.lora_a", {M_IN, M_R},  ma },
            { "blk.0.attn_q.weight.lora_b", {M_R, M_OUT}, mz },
        };
        write_adapter(zad, alpha, NULL, zts, 2, T_F32);
        assert(quantize_gguf(base, keep, T_KEEP, NULL) == 0);
        assert(merge_lora_gguf(base, zad, 1.0f, zout, T_KEEP) == 0);
        assert(files_equal(zout, keep));
        // merging into a quantized base: the delta rides the dequant->
        // requant path; the merged row decodes to base row + delta within
        // the type's grid resolution
        const char *q8b = "q_mrg_q8.gguf", *q8m = "q_mrg_q8m.gguf";
        assert(quantize_gguf(base, q8b, T_Q8_0, NULL) == 0);
        assert(merge_lora_gguf(q8b, ad, 1.0f, q8m, T_KEEP) == 0);
        {
            gguf_file gb, gm;
            assert(gguf_open(&gb, q8b) && gguf_open(&gm, q8m));
            gguf_tensor *tb = gguf_find_tensor(&gb, "blk.0.attn_q.weight");
            gguf_tensor *tm = gguf_find_tensor(&gm, "blk.0.attn_q.weight");
            assert(tb && tm && tm->type == T_Q8_0);
            size_t rs = ggml_row_size(T_Q8_0, M_IN);
            for (int j = 0; j < M_OUT; j++) {
                float rb[M_IN], rm[M_IN];
                dequant_row(T_Q8_0, (const uint8_t *)tb->data + (size_t)j * rs,
                            rb, M_IN);
                dequant_row(T_Q8_0, (const uint8_t *)tm->data + (size_t)j * rs,
                            rm, M_IN);
                for (int i = 0; i < M_IN; i++) {
                    float acc = 0.0f;
                    for (int k = 0; k < M_R; k++)
                        acc = fmaf(mb[j * M_R + k], ma[k * M_IN + i], acc);
                    float want = fmaf(alpha / (float)M_R, acc, rb[i]);
                    assert(fabsf(rm[i] - want) < 0.01f);
                }
            }
            gguf_close(&gb); gguf_close(&gm);
        }
        // an F16 adapter (the llama.cpp-converter format) merges too, and
        // lands within half-precision rounding of the F32 merge
        {
            const char *fad = "q_mrg_f16ad.gguf", *fout = "q_mrg_f16.gguf";
            write_adapter(fad, alpha, NULL, ats, 2, T_F16);
            assert(merge_lora_gguf(base, fad, 1.0f, fout, T_KEEP) == 0);
            gguf_file gf;
            assert(gguf_open(&gf, fout));
            gguf_tensor *qf = gguf_find_tensor(&gf, "blk.0.attn_q.weight");
            assert(qf && qf->type == T_F32);
            const float *gotf = (const float *)qf->data;
            for (int j = 0; j < M_OUT; j++)
                for (int i = 0; i < M_IN; i++) {
                    float acc = 0.0f;
                    for (int k = 0; k < M_R; k++)
                        acc = fmaf(mb[j * M_R + k], ma[k * M_IN + i], acc);
                    float want = fmaf(alpha / (float)M_R, acc,
                                      mw[j * M_IN + i]);
                    assert(fabsf(gotf[j * M_IN + i] - want) < 1e-3f);
                }
            gguf_close(&gf);
            remove(fad); remove(fout);
        }
        // hostile adapters refuse the whole merge
        const char *hout = "q_mrg_h.gguf";
        adesc half[1] = {
            { "blk.0.attn_q.weight.lora_a", {M_IN, M_R}, ma },
        };
        write_adapter(ad, alpha, NULL, half, 1, T_F32);
        assert(merge_lora_gguf(base, ad, 1.0f, hout, T_KEEP) != 0);
        adesc badshape[2] = {
            { "blk.0.attn_q.weight.lora_a", {32, M_R},  ma },
            { "blk.0.attn_q.weight.lora_b", {M_R, M_OUT}, mb },
        };
        write_adapter(ad, alpha, NULL, badshape, 2, T_F32);
        assert(merge_lora_gguf(base, ad, 1.0f, hout, T_KEEP) != 0);
        adesc unhooked[2] = {
            { "blk.0.attn_qkv.weight.lora_a", {M_IN, M_R},  ma },
            { "blk.0.attn_qkv.weight.lora_b", {M_R, M_OUT}, mb },
        };
        write_adapter(ad, alpha, NULL, unhooked, 2, T_F32);
        assert(merge_lora_gguf(base, ad, 1.0f, hout, T_KEEP) != 0);
        write_adapter(ad, alpha, "llama", ats, 2, T_F32);   // base carries no arch
        assert(merge_lora_gguf(base, ad, 1.0f, hout, T_KEEP) != 0);
        assert(!exists(hout));                       // nothing was installed
        remove(base); remove(ad); remove(out1); remove(out2);
        remove(keep); remove(zad); remove(zout); remove(q8b); remove(q8m);
        printf("ok: --merge-lora folds the exact fmaf chain, copies "
               "untouched tensors verbatim, and refuses hostile adapters\n");
    }

    // RNR-015: in-place requant must not truncate its own input
    const char *inplace = "q_inplace.gguf";
    cp(in, inplace);
    assert(quantize_gguf(inplace, inplace, T_Q8_0, NULL) == 0);
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
    assert(quantize_gguf(bad, dest, T_Q8_0, NULL) != 0); // must reject
    check_bytes(dest, sentinel, sizeof(sentinel));
    printf("ok: failed requant left the destination untouched\n");

    // A type plan is parsed before the input GGUF is opened. If that open
    // fails, the already-owned rules must still be released (ASan/LSan gate).
    const char *missing_plan = "q_missing_plan.json";
    FILE *mpf = fopen(missing_plan, "wb"); assert(mpf);
    const char missing_plan_json[] =
        "{\"default\":\"keep\",\"rules\":[{\"match\":\"weight\",\"type\":\"q8_0\"}]}";
    assert(fwrite(missing_plan_json, 1, sizeof(missing_plan_json) - 1, mpf) ==
           sizeof(missing_plan_json) - 1);
    fclose(mpf);
    assert(quantize_gguf_plan("q_does_not_exist.gguf", dest, T_KEEP, NULL,
                              missing_plan) != 0);
    remove(missing_plan);
    printf("ok: a missing input releases its parsed type plan\n");

    // RNR-015 on the late install path: the temp file is complete, and only
    // replacing the destination fails. This is the Windows destroy-destination
    // window in a platform-independent smoke.
    FILE *late = fopen(dest, "wb"); assert(late);
    assert(fwrite(sentinel, 1, sizeof(sentinel), late) == sizeof(sentinel));
    fclose(late);
    env_set("RUNNER_QUANTIZE_INSTALL_FAIL", "1");
    assert(quantize_gguf(in, dest, T_Q8_0, NULL) != 0);
    env_unset("RUNNER_QUANTIZE_INSTALL_FAIL");
    check_bytes(dest, sentinel, sizeof(sentinel));
    assert(!exists("q_keep.gguf.partial"));
    printf("ok: failed install left the destination untouched\n");

    check_mxfp4_dequant();

    remove(in); remove(out); remove(inplace); remove(dest); remove(bad);
    printf("all quantize tests passed\n");
    return 0;
}
