// Allocation-failure tests for the chat-template renderer.
//
// render_messages_with_tools has no error channel: it returns the number of
// bytes it would have written, and the caller (server.c's render_prompt_alloc)
// reads that as "grow the buffer and try again". So a failed allocation inside
// it cannot be reported — it can only be survived. What the renderer must
// never do is turn one into a DIFFERENT PROMPT that looks fine: the family
// preambles are built in an sbuf and then formatted into the output with
// "%s", and an sbuf whose realloc failed carries a NULL pointer, which is
// undefined behavior and prints as the four characters "(null)" on the libcs
// where it does not crash. A model then reads "(null)" as its system prompt.
//
// json.c and template.c are compiled straight into this test with their
// allocators macro-substituted, the same technique as test_json_oom.c and
// test_tokenizer_oom.c; tokenizer.c/gguf.c/compat.c link normally, so their
// allocations are not counted.
#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static long alloc_calls;   // allocations attempted since the last reset
static long alloc_live;    // outstanding blocks; must return to 0
static long fail_at = -1;  // index of the allocation to fail, -1 for none

static void *t_malloc(size_t n) {
    if (fail_at >= 0 && alloc_calls++ == fail_at) return NULL;
    if (fail_at < 0) alloc_calls++;
    void *p = malloc(n);
    if (p) alloc_live++;
    return p;
}

static void *t_calloc(size_t a, size_t b) {
    if (fail_at >= 0 && alloc_calls++ == fail_at) return NULL;
    if (fail_at < 0) alloc_calls++;
    void *p = calloc(a, b);
    if (p) alloc_live++;
    return p;
}

static void *t_realloc(void *p, size_t n) {
    if (fail_at >= 0 && alloc_calls++ == fail_at) return NULL;
    if (fail_at < 0) alloc_calls++;
    void *q = realloc(p, n);
    if (q && !p) alloc_live++;   // realloc(NULL, n) behaves as malloc
    return q;
}

static void t_free(void *p) {
    if (p) alloc_live--;
    free(p);
}

#define malloc  t_malloc
#define calloc  t_calloc
#define realloc t_realloc
#define free    t_free
#include "../src/json.c"
#include "../src/template.c"
#undef malloc
#undef calloc
#undef realloc
#undef free

// One declared tool, so the families that render declarations natively (muse's
// atem block, harmony's TypeScript namespace, gemma4's <|tool>) all have
// something to build an sbuf for.
static const char *const TOOLS_SRC =
    "[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\","
    "\"description\":\"Look a city up\",\"parameters\":{\"type\":\"object\","
    "\"properties\":{\"city\":{\"type\":\"string\",\"description\":\"City\"},"
    "\"days\":{\"type\":\"integer\"}},\"required\":[\"city\"]}}}]";

static const int FAMILIES[] = {
    TMPL_HARMONY, TMPL_MUSE, TMPL_GEMMA4, TMPL_CHATML, TMPL_LLAMA2,
    TMPL_MISTRAL, TMPL_APERTUS, TMPL_ORNITH, TMPL_GRANITE, TMPL_LLAMA3,
    TMPL_GEMMA, TMPL_PHI3, TMPL_ZEPHYR,
};

static const chat_msg MSGS[] = {
    { .role = "system",    .content = "You are terse." },
    { .role = "user",      .content = "weather in Oslo?" },
    { .role = "assistant", .content = "checking" },
    { .role = "user",      .content = "and tomorrow?" },
};
#define N_MSGS ((int)(sizeof(MSGS) / sizeof(*MSGS)))

// A prompt built through a failed allocation may be SHORTER than the complete
// one -- there is no way to report the failure, and a missing declaration is
// at least visibly missing. It must never contain the spelling of a NULL
// pointer, which is what a "%s" of a failed sbuf produces.
static void test_render_never_formats_a_failed_buffer(void) {
    char out[8192];
    for (size_t f = 0; f < sizeof(FAMILIES) / sizeof(*FAMILIES); f++) {
        // `tools` is parsed with allocation failure OFF: it stands in for a
        // request the server already accepted, and instrumenting its parse
        // would only re-test json.c.
        fail_at = -1;
        jv *tools = json_parse(TOOLS_SRC, strlen(TOOLS_SRC));
        assert(tools != NULL);

        alloc_calls = 0;
        size_t clean = render_messages_with_tools(FAMILIES[f], MSGS, N_MSGS,
                                                  true, THINK_DEFAULT, tools,
                                                  out, sizeof(out));
        long total = alloc_calls;
        assert(clean > 0 && clean < sizeof(out));
        assert(!strstr(out, "(null)"));

        for (long k = 0; k < total; k++) {
            fail_at = k;
            alloc_calls = 0;
            alloc_live = 0;
            memset(out, 0, sizeof(out));
            render_messages_with_tools(FAMILIES[f], MSGS, N_MSGS, true,
                                       THINK_DEFAULT, tools, out, sizeof(out));
            if (strstr(out, "(null)")) {
                fprintf(stderr, "template %s: failing allocation %ld of %ld "
                        "formatted a NULL buffer:\n%s\n",
                        template_name(FAMILIES[f]), k, total, out);
                abort();
            }
            if (alloc_live != 0) {
                fprintf(stderr, "template %s: failing allocation %ld of %ld: "
                        "%ld block(s) leaked\n",
                        template_name(FAMILIES[f]), k, total, alloc_live);
                abort();
            }
        }
        fail_at = -1;
        jv_free(tools);
    }
}

int main(void) {
    test_render_never_formats_a_failed_buffer();
    fail_at = -1;
    puts("template allocation-failure tests ok");
    return 0;
}
