// libFuzzer harness for gguf_open (src/gguf.c:84).
//
// gguf_open takes a *path*, not a buffer, so the harness materialises each
// input as a file. The file is created once and rewritten per iteration:
// creating and unlinking a fresh temp file per execution would make the
// syscalls, not the parser, the thing being measured.
//
// The contract under test is "reject, never crash". Every accepted file is
// then walked through the public accessors, because a header that parses but
// leaves kv/tensor entries pointing outside the mapping is exactly the bug
// class worth finding here, and it only shows up on access.
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdbool.h>

#include "runner.h"

static char g_path[] = "/tmp/fuzz-gguf-XXXXXX";
static int  g_fd = -1;
static int  g_null = -1, g_stderr = -1;

static void cleanup(void) {
    if (g_fd >= 0) close(g_fd);
    unlink(g_path);
}

// gguf_open reports every rejection to stderr and nearly every generated input
// is a rejection: left alone that is gigabytes of log per run and it, not the
// parser, dominates the runtime.
//
// The muting is scoped to the call rather than done once with freopen at
// startup, because libFuzzer prints its *own* progress and crash reports to
// stderr between calls -- muting it globally silences the fuzzer itself, which
// makes a failing run indistinguishable from a passing one. A sanitizer report
// raised inside the muted window is still captured: the Makefile sets
// ASAN_OPTIONS/UBSAN_OPTIONS log_path for this target.
static void mute_stderr(void)   { if (g_null    >= 0) dup2(g_null, 2); }
static void unmute_stderr(void) { if (g_stderr  >= 0) dup2(g_stderr, 2); }

int LLVMFuzzerInitialize(int *argc, char ***argv) {
    (void)argc; (void)argv;
    g_fd = mkstemp(g_path);
    if (g_fd < 0) { perror("mkstemp"); exit(1); }
    atexit(cleanup);
    g_stderr = dup(2);
    g_null   = open("/dev/null", O_WRONLY);
    if (g_stderr < 0 || g_null < 0) { perror("dup/open"); exit(1); }
    return 0;
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (ftruncate(g_fd, 0) != 0) return 0;
    if (lseek(g_fd, 0, SEEK_SET) != 0) return 0;
    for (size_t off = 0; off < size; ) {
        ssize_t w = write(g_fd, data + off, size - off);
        if (w <= 0) return 0;
        off += (size_t)w;
    }

    gguf_file g;
    mute_stderr();
    bool ok = gguf_open(&g, g_path);
    unmute_stderr();
    if (!ok) return 0;

    // touch everything the loader would touch
    for (uint64_t i = 0; i < g.n_kv; i++) {
        gguf_kv *kv = &g.kv[i];
        if (kv->key) (void)gguf_get(&g, kv->key);
    }
    (void)gguf_get_u32 (&g, "llama.block_count", 0);
    (void)gguf_get_f32 (&g, "llama.attention.layer_norm_rms_epsilon", 0.0f);
    (void)gguf_get_bool(&g, "tokenizer.ggml.add_bos_token", false);
    (void)gguf_get_str (&g, "general.architecture", "");
    for (uint64_t i = 0; i < g.n_tensors; i++) {
        gguf_tensor *t = &g.tensors[i];
        (void)gguf_find_tensor(&g, t->name);
        // a tensor whose bytes escape the mapping is the bug we are hunting:
        // read the first and last byte the loader would be entitled to read
        if (t->data && t->nbytes) {
            volatile uint8_t sink = ((const uint8_t *)t->data)[0];
            sink ^= ((const uint8_t *)t->data)[t->nbytes - 1];
            (void)sink;
        }
    }

    gguf_close(&g);
    return 0;
}
