// Chat template detection and rendering.
//
// Detection reads the GGUF's own chat_template string, so these tests pass the
// marker text directly rather than needing a model per family. A tokenizer is
// still required for the fallback path that looks for special tokens, and any
// fixture vocabulary serves for that.
#include "runner.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

#define FIXTURE "tests/fixtures/vocab-bpe-llama3.gguf"

// Llama-2 and Mistral both frame turns with [INST], and runner used to send
// both down the Llama-2 path. Mistral's own template accepts only user and
// assistant roles and has no <<SYS>> block, so wrapping a system prompt in one
// feeds it markers it never saw in training. <<SYS>> is what separates them.
static void test_detect_llama2_vs_mistral(tokenizer *t) {
    const char *llama2 =
        "{% if messages[0]['role'] == 'system' %}[INST] <<SYS>>\n"
        "{{ messages[0]['content'] }}\n<</SYS>>\n\n{% endif %}";
    const char *mistral =
        "{{ bos_token }}{% for message in messages %}"
        "{{ '[INST] ' + message['content'] + ' [/INST]' }}{% endfor %}";

    assert(template_detect(llama2, t) == TMPL_LLAMA2);
    assert(template_detect(mistral, t) == TMPL_MISTRAL);
}

// phi3 uses zephyr's <|role|> framing but terminates turns with <|end|>
// rather than </s>, so the terminator is what tells them apart.
static void test_detect_zephyr_vs_phi3(tokenizer *t) {
    const char *zephyr = "{{'<|user|>\n' + message['content'] + '</s>\n'}}";
    const char *phi3   = "{{'<|user|>\n' + message['content'] + '<|end|>\n'}}";
    assert(template_detect(zephyr, t) == TMPL_ZEPHYR);
    assert(template_detect(phi3, t) == TMPL_PHI3);

    const chat_msg msgs[] = { { "user", "HI" } };
    char out[512];
    render_messages(TMPL_PHI3, msgs, 1, true, out, sizeof(out));
    assert(strcmp(out, "<|user|>\nHI<|end|>\n<|assistant|>\n") == 0);
}

// Apertus (Swiss AI) frames turns with <|role_start|>...<|role_end|>. The
// prefix "<|user_start|>" contains no substring the other families key on, but
// detection order still matters: the Apertus vocabulary inherits Mistral's
// [/INST] tokens, so anything that reached the [INST] branch or the tok_find
// fallback would come back TMPL_MISTRAL or TMPL_LLAMA2.
//
// Ground truth for the rendering is the model's own chat_template.jinja
// rendered by jinja2 (swiss-ai/Apertus-8B-Instruct-2509).
static void test_detect_and_render_apertus(tokenizer *t) {
    const char *apertus =
        "{%- set user_token = '<|user_start|>' -%}"
        "{%- set end_user_token = '<|user_end|>' -%}"
        "{%- set assistant_token = '<|assistant_start|>' -%}"
        "{%- set end_assistant_token = '<|assistant_end|>' -%}";
    assert(template_detect(apertus, t) == TMPL_APERTUS);

    // The reference template always emits the developer block; with thinking
    // and tools off it is this exact constant.
    const chat_msg msgs[] = {
        { "system", "SYS" }, { "user", "HI" },
        { "assistant", "YO" }, { "user", "BYE" },
    };
    char out[1024];
    render_messages(TMPL_APERTUS, msgs, 4, true, out, sizeof(out));
    assert(strcmp(out,
        "<|system_start|>SYS<|system_end|>"
        "<|developer_start|>Deliberation: disabled\n"
        "Tool Capabilities: disabled<|developer_end|>"
        "<|user_start|>HI<|user_end|>"
        "<|assistant_start|>YO<|assistant_end|>"
        "<|user_start|>BYE<|user_end|>"
        "<|assistant_start|>") == 0);
}

// With no system message the reference template substitutes a default one that
// embeds strftime_now('%Y-%m-%d') -- a live date. Runner omits it, exactly as
// it already omits Llama-3.2's "Cutting Knowledge Date" header, and emits the
// developer block so the turn framing still matches.
static void test_render_apertus_without_system(void) {
    const chat_msg msgs[] = { { "user", "HI" } };
    char out[512];
    render_messages(TMPL_APERTUS, msgs, 1, true, out, sizeof(out));
    assert(strcmp(out,
        "<|developer_start|>Deliberation: disabled\n"
        "Tool Capabilities: disabled<|developer_end|>"
        "<|user_start|>HI<|user_end|>"
        "<|assistant_start|>") == 0);
}

static void test_detect_by_marker(tokenizer *t) {
    assert(template_detect("<|im_start|>system", t) == TMPL_CHATML);
    assert(template_detect("<|start_header_id|>system<|end_header_id|>", t) == TMPL_LLAMA3);
    assert(template_detect("<start_of_turn>user", t) == TMPL_GEMMA);
    assert(template_detect("<|user|>", t) == TMPL_ZEPHYR);
}

static void test_detect_and_render_ornith(tokenizer *t) {
    const char *ornith =
        "{% if tools %}<tools>{% endif %}"
        "<tool_call>\\n<function=example_function_name>"
        "{% if add_generation_prompt %}<think>\\n{% endif %}"
        "<|im_start|>";
    assert(template_detect(ornith, t) == TMPL_ORNITH);

    const chat_msg msgs[] = {
        { "system", "SYS" },
        { "user", "HI" },
        { "assistant", "<think>\nPLAN\n</think>\n\nANSWER" },
    };
    char out[1024];
    render_messages(TMPL_ORNITH, msgs, 3, true, out, sizeof(out));
    assert(strcmp(out,
        "<|im_start|>system\nSYS<|im_end|>\n"
        "<|im_start|>user\nHI<|im_end|>\n"
        "<|im_start|>assistant\n<think>\nPLAN\n</think>\n\nANSWER<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n") == 0);
}

typedef struct { char reason[128], content[128]; int nr, nc; } split_capture;

static int capture_split(void *ud, int reasoning, const char *bytes, int n) {
    split_capture *c = ud;
    char *dst = reasoning ? c->reason : c->content;
    int *len = reasoning ? &c->nr : &c->nc;
    memcpy(dst + *len, bytes, n);
    *len += n;
    dst[*len] = 0;
    return 0;
}

static void test_ornith_split_starts_inside_prompted_think(void) {
    think_split split;
    split_capture got = {0};
    think_init_reasoning(&split, "<think>", "</think>");
    const char *generated = "Thinking Process:</think>answer";
    think_feed(&split, generated, strlen(generated), capture_split, &got);
    think_finish(&split, capture_split, &got);
    assert(!strcmp(got.reason, "Thinking Process:"));
    assert(!strcmp(got.content, "answer"));
    think_free(&split);
}

static void test_ornith_groups_consecutive_tool_responses(void) {
    const chat_msg msgs[] = {
        { "user", "HI" },
        { "assistant", "<think>\nPLAN\n</think>\n\n<tool_call>x</tool_call>" },
        { "user", "<tool_response>\nONE\n</tool_response>" },
        { "user", "<tool_response>\nTWO\n</tool_response>" },
    };
    char out[1024];
    render_messages(TMPL_ORNITH, msgs, 4, true, out, sizeof(out));
    assert(strstr(out,
        "<|im_start|>user\n"
        "<tool_response>\nONE\n</tool_response>\n"
        "<tool_response>\nTWO\n</tool_response><|im_end|>\n"
        "<|im_start|>assistant\n"));
}

// The system prompt is folded into the first user turn either way; only the
// framing differs.
static void test_render_system_prompt(void) {
    const chat_msg msgs[] = { { "system", "SYS" }, { "user", "HI" } };
    char out[1024];

    render_messages(TMPL_MISTRAL, msgs, 2, true, out, sizeof(out));
    assert(strcmp(out, "[INST] SYS\n\nHI [/INST]") == 0);

    render_messages(TMPL_LLAMA2, msgs, 2, true, out, sizeof(out));
    assert(strcmp(out, "[INST] <<SYS>>\nSYS\n<</SYS>>\n\nHI [/INST]") == 0);
}

// Without a system message the two are identical, which is why detection has
// to key on the template text rather than on the rendered output.
static void test_render_without_system(void) {
    const chat_msg msgs[] = { { "user", "HI" } };
    char mistral[512], llama2[512];
    render_messages(TMPL_MISTRAL, msgs, 1, true, mistral, sizeof(mistral));
    render_messages(TMPL_LLAMA2, msgs, 1, true, llama2, sizeof(llama2));
    assert(strcmp(mistral, "[INST] HI [/INST]") == 0);
    assert(strcmp(mistral, llama2) == 0);
}

static void test_name_roundtrip(void) {
    static const char *const names[] = {
        "chatml", "llama2", "llama3", "zephyr", "gemma", "gemma4", "mistral",
        "phi3", "apertus", "ornith", "raw",
    };
    for (size_t i = 0; i < sizeof(names) / sizeof(*names); i++) {
        int id = template_from_name(names[i]);
        assert(id >= 0);
        assert(strcmp(template_name(id), names[i]) == 0);
    }
    assert(template_from_name("nope") == -1);
}

int main(void) {
    gguf_file g;
    if (!gguf_open(&g, FIXTURE)) {
        fprintf(stderr, "cannot open %s (run from the repo root)\n", FIXTURE);
        return 1;
    }
    tokenizer t;
    if (!tokenizer_init(&t, &g)) {
        fprintf(stderr, "tokenizer_init failed\n");
        return 1;
    }

    test_detect_llama2_vs_mistral(&t);
    test_detect_zephyr_vs_phi3(&t);
    test_detect_by_marker(&t);
    test_detect_and_render_ornith(&t);
    test_ornith_groups_consecutive_tool_responses();
    test_ornith_split_starts_inside_prompted_think();
    test_detect_and_render_apertus(&t);
    test_render_apertus_without_system();
    test_render_system_prompt();
    test_render_without_system();
    test_name_roundtrip();

    tokenizer_free(&t);
    gguf_close(&g);
    puts("template tests ok");
    return 0;
}
