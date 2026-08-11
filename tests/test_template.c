// Chat template detection and rendering.
//
// Detection reads the GGUF's own chat_template string, so these tests pass the
// marker text directly rather than needing a model per family. A tokenizer is
// still required for the fallback path that looks for special tokens, and any
// fixture vocabulary serves for that.
#include "runner.h"
#include "json.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
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
    render_messages(TMPL_PHI3, msgs, 1, true, THINK_DEFAULT, out, sizeof(out));
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
    render_messages(TMPL_APERTUS, msgs, 4, true, THINK_DEFAULT, out, sizeof(out));
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
    render_messages(TMPL_APERTUS, msgs, 1, true, THINK_DEFAULT, out, sizeof(out));
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
    render_messages(TMPL_ORNITH, msgs, 3, true, THINK_DEFAULT, out, sizeof(out));
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

// Constrained flow: reasoning bytes arrive freely, the <|eom|> control that
// really ends the turn decodes to no bytes, so engine.c's
// constraint_finish_think feeds the literal close to the splitter itself.
// The splitter only has to flip on that synthetic feed; the payload machine
// owns everything after it.
static void test_muse_split_closes_on_fed_reasoning_boundary(void) {
    think_split split;
    split_capture got = {0};
    think_init_reasoning(&split, " to=self", "assistant to=user");
    const char *reasoning = "compute 17*23";
    for (size_t i = 0; i < strlen(reasoning); i++)
        think_feed(&split, reasoning + i, 1, capture_split, &got);
    const char *fed_close = "assistant to=user"; // constraint_finish_think
    think_feed(&split, fed_close, strlen(fed_close), capture_split, &got);
    const char *payload = "record_conclusion<|message|><atem:invoke>";
    for (size_t i = 0; i < strlen(payload); i++)
        think_feed(&split, payload + i, 1, capture_split, &got);
    think_finish(&split, capture_split, &got);
    assert(!strcmp(got.reason, "compute 17*23"));
    assert(!strcmp(got.content,
                   "record_conclusion<|message|><atem:invoke>"));
    think_free(&split);
}

// Unconstrained plain chat, the shape the real model emits at temp 0:
// ` to=self` THINKING `<|eom|><|start|>` decode to nothing around the
// literal `assistant to=user`, then the answer. The full close string must
// be consumed so no recipient residue reaches content and no `assistant`
// tail sticks to reasoning (both leaked when the close was narrowed to
// ` to=`; measured live 2026-08-11, content came back as "user391").
static void test_muse_plain_thinking_close_leaves_no_recipient_residue(void) {
    think_split split;
    split_capture got = {0};
    think_init(&split, " to=self", "assistant to=user");
    const char *generated = " to=selfSeventeen times 23 is 391."
                            "assistant to=user391";
    for (size_t i = 0; i < strlen(generated); i++)
        think_feed(&split, generated + i, 1, capture_split, &got);
    think_finish(&split, capture_split, &got);
    assert(!strcmp(got.reason, "Seventeen times 23 is 391."));
    assert(!strcmp(got.content, "391"));
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
    render_messages(TMPL_ORNITH, msgs, 4, true, THINK_DEFAULT, out, sizeof(out));
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

    render_messages(TMPL_MISTRAL, msgs, 2, true, THINK_DEFAULT, out, sizeof(out));
    assert(strcmp(out, "[INST] SYS\n\nHI [/INST]") == 0);

    render_messages(TMPL_LLAMA2, msgs, 2, true, THINK_DEFAULT, out, sizeof(out));
    assert(strcmp(out, "[INST] <<SYS>>\nSYS\n<</SYS>>\n\nHI [/INST]") == 0);
}

// Without a system message the two are identical, which is why detection has
// to key on the template text rather than on the rendered output.
static void test_render_without_system(void) {
    const chat_msg msgs[] = { { "user", "HI" } };
    char mistral[512], llama2[512];
    render_messages(TMPL_MISTRAL, msgs, 1, true, THINK_DEFAULT, mistral, sizeof(mistral));
    render_messages(TMPL_LLAMA2, msgs, 1, true, THINK_DEFAULT, llama2, sizeof(llama2));
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

// gemma-4's generation prompt, from the MODEL'S OWN chat template rather than
// a summary of llama.cpp's: it is `<|turn>model\n` and nothing else, in either
// thinking mode. On 2026-08-08 an empty thought-block pre-seed was added here
// on the strength of a web summary; gemma-4-E2B's planning score fell
// 0.575 -> 0.300 and it emitted reasoning prose as its visible answer, because
// that construct only ever appears when re-rendering a prior assistant message
// that contained thinking text.
//
// So the assertion that matters most is the NEGATIVE one: no thought block is
// ever pre-seeded at generation time, whatever the caller asks for.
static void test_gemma4_generation_prompt(void) {
    const chat_msg msgs[] = { { "user", "HI" } };
    char out[512];
    const char *base = "<|turn>user\nHI<turn|>\n<|turn>model\n";

    render_messages(TMPL_GEMMA4, msgs, 1, true, THINK_DEFAULT, out, sizeof(out));
    assert(strcmp(out, base) == 0);
    render_messages(TMPL_GEMMA4, msgs, 1, true, THINK_OFF, out, sizeof(out));
    assert(strcmp(out, base) == 0);
    // THINK_ON does not pre-seed either: the template injects <|think|> into
    // the first SYSTEM turn instead, which this engine does not implement
    render_messages(TMPL_GEMMA4, msgs, 1, true, THINK_ON, out, sizeof(out));
    assert(strcmp(out, base) == 0);
    assert(strstr(out, "channel>thought") == NULL);

    // after a tool response the template emits NO generation prompt at all
    // when thinking is off, and an OPEN thought tag when it is on
    const chat_msg after_tool[] = { { "user", "HI" }, { "tool", "42" } };
    render_messages(TMPL_GEMMA4, after_tool, 2, true, THINK_DEFAULT,
                    out, sizeof(out));
    assert(strstr(out, "<|turn>model") == NULL);
    assert(strstr(out, "channel>thought") == NULL);
    render_messages(TMPL_GEMMA4, after_tool, 2, true, THINK_ON,
                    out, sizeof(out));
    assert(strstr(out, "<|channel>thought\n") != NULL);

    render_messages(TMPL_GEMMA4, msgs, 1, false, THINK_DEFAULT, out, sizeof(out));
    assert(strcmp(out, "<|turn>user\nHI<turn|>\n") == 0);
}

static void test_chatml_think_shape(void) {
    const chat_msg msgs[] = { { "user", "HI" } };
    char out[512];
    const char *base = "<|im_start|>user\nHI<|im_end|>\n<|im_start|>assistant\n";

    // detection comes from the model's own template, not a name list
    assert(template_detect("<|im_start|>system ... <think> ...", NULL)
           == TMPL_CHATML_THINK);
    assert(template_detect("<|im_start|>system", NULL) == TMPL_CHATML);
    assert(template_from_name("chatml-think") == TMPL_CHATML_THINK);
    assert(!strcmp(template_name(TMPL_CHATML_THINK), "chatml-think"));

    // DEFAULT and ON both mean "let the model think" for this family
    render_messages(TMPL_CHATML_THINK, msgs, 1, true, THINK_DEFAULT,
                    out, sizeof(out));
    assert(strcmp(out, base) == 0);
    render_messages(TMPL_CHATML_THINK, msgs, 1, true, THINK_ON,
                    out, sizeof(out));
    assert(strcmp(out, base) == 0);

    // OFF appends the closed block, verbatim from the reference
    render_messages(TMPL_CHATML_THINK, msgs, 1, true, THINK_OFF,
                    out, sizeof(out));
    assert(strcmp(out, "<|im_start|>user\nHI<|im_end|>\n"
                       "<|im_start|>assistant\n<think>\n\n</think>\n\n") == 0);

    // plain ChatML never grows a thought block, whatever is asked of it
    render_messages(TMPL_CHATML, msgs, 1, true, THINK_OFF, out, sizeof(out));
    assert(strcmp(out, base) == 0);
}

// Muse's own template places tool metadata after the reasoning-strength line,
// derives valid recipient namespaces from dotted function names, and renders
// tool results as named tool turns.  Keep the whole turn byte-exact: moving
// any of these fragments changes the prompt the real model sees.
static void test_muse_tools_and_result_golden(void) {
    const char *src =
        "[{\"type\":\"function\",\"function\":{\"name\":\"weather.get\","
        "\"description\":\"Get weather\",\"parameters\":{\"type\":\"object\","
        "\"properties\":{\"city\":{\"type\":\"string\"}}}}},"
        "{\"type\":\"function\",\"function\":{\"name\":\"math.add\","
        "\"description\":\"Add values\",\"parameters\":{\"type\":\"object\","
        "\"properties\":{\"a\":{\"type\":\"integer\"}}}}}]";
    jv *tools = json_parse(src, strlen(src));
    assert(tools != NULL);
    const chat_msg msgs[] = {
        { .role = "user", .content = "Weather?" },
        { .role = "tool", .content = "sunny", .name = "weather.get" },
    };
    char out[8192];
    render_messages_with_tools(TMPL_MUSE, msgs, 2, true, THINK_DEFAULT,
                               tools, out, sizeof(out));
    assert(strcmp(out,
        "<|start|>system<|message|>You are a helpful AI assistant.\n"
        "Knowledge cutoff: 2026-01-04.\n\nReasoning strength: high.\n\n"
        "In this environment you have access to a set of tools you can use to answer the user's question.\n\n"
        "You can invoke a function by writing a \"<atem:function_calls>\" block like the following:\n"
        "<atem:function_calls>\n<atem:invoke name=\"$FUNCTION_NAME\">\n"
        "<atem:parameter name=\"$PARAMETER_NAME\">$PARAMETER_VALUE</atem:parameter>\n"
        "...\n</atem:invoke>\n</atem:function_calls>\n\n"
        "String and scalar parameters should be specified as is, while lists and objects should use JSON format. Note that spaces for string values are not stripped. The output is not expected to be valid XML and is parsed with regular expressions.\n"
        "Here are the functions available in JSONSchema format:\n// Tool metadata\n"
        "{\"name\": \"weather\", \"description\": \"\"}\n"
        "{\"name\": \"math\", \"description\": \"\"}\n"
        "// Function schemas\n"
        "{\"name\": \"weather.get\", \"description\": \"Get weather\", \"parameters\": {\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}}}}\n"
        "{\"name\": \"math.add\", \"description\": \"Add values\", \"parameters\": {\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"integer\"}}}}\n\n"
        "Here's an example of how to call a function in the tool set:\n"
        "(If the tool namespace is not specified, invoke the function directly as `example_function_name` rather than `example_tool_name.example_function_name`)\n\n"
        "to=example_tool_name.example_function_name\n\n"
        "<atem:function_calls>\n<atem:invoke name=\"example_tool_name.example_function_name\">\n"
        "<atem:parameter name=\"example_parameter_1\">value_1</atem:parameter>\n"
        "<atem:parameter name=\"example_parameter_2\">This is the value for the second parameter\nthat can span\n\"multiple\" lines\n</atem:parameter>\n"
        "</atem:invoke>\n</atem:function_calls>\n\n"
        "# Valid recipients: \"self\", \"weather.*\", \"math.*\", \"user\".<|eot|>"
        "<|start|>user<|message|>Weather?<|eot|>"
        "<|start|>tool weather.get<|message|><tool_output name=\"weather.get\">\n"
        "sunny\n</tool_output><|eot|>"
        "<|start|>assistant") == 0);
    jv_free(tools);
}

static void test_muse_tool_result_id_resolves_prior_name(void) {
    const char *src =
        "[{\"role\":\"assistant\",\"tool_calls\":[{\"id\":\"call_7\","
        "\"type\":\"function\",\"function\":{\"name\":\"weather.get\","
        "\"arguments\":\"{}\"}}]},"
        "{\"role\":\"tool\",\"tool_call_id\":\"call_7\",\"content\":\"sunny\"}]";
    jv *messages = json_parse(src, strlen(src));
    assert(messages != NULL);
    assert(!strcmp(tool_result_name(messages, 1), "weather.get"));
    jv_free(messages);
}

static void test_muse_parallel_tool_history_has_native_turn_boundaries(void) {
    const char *src =
        "[{\"type\":\"function\",\"function\":{\"name\":\"weather.get\","
        "\"arguments\":\"{\\\"city\\\":\\\"Oslo\\\"}\"}},"
        "{\"type\":\"function\",\"function\":{\"name\":\"weather.get\","
        "\"arguments\":\"{\\\"city\\\":\\\"Bergen\\\"}\"}}]";
    jv *calls = json_parse(src, strlen(src));
    assert(calls != NULL);
    sbuf out = {0};
    tool_history_render_for(TMPL_MUSE, calls, &out);
    assert(out.s != NULL);
    assert(strstr(out.s,
        "</atem:function_calls><|eom|><|start|>assistant to=weather.get"
        "<|message|><atem:function_calls>") != NULL);
    free(out.s);
    jv_free(calls);
}

static void test_muse_tool_history_skips_bad_calls_without_leading_boundary(void) {
    const char *src =
        "[{\"type\":\"function\",\"function\":{\"arguments\":\"{}\"}},"
        "{\"type\":\"function\",\"function\":{\"name\":\"ping\","
        "\"arguments\":\"{}\"}}]";
    jv *calls = json_parse(src, strlen(src));
    assert(calls != NULL);
    sbuf out = {0};
    tool_history_render_for(TMPL_MUSE, calls, &out);
    assert(out.s != NULL);
    assert(!strncmp(out.s, "<atem:function_calls>", 21));
    assert(strstr(out.s, "<|eom|><|start|>") == NULL);
    free(out.s);
    jv_free(calls);
}

static void test_muse_user_payload_strip_removes_only_recipient_header(void) {
    sbuf payload = {0};
    sb_lit(&payload, " to=user<|message|>{\"summary\":\"ok\"}");
    assert(muse_user_payload_strip(&payload));
    assert(payload.s && !strcmp(payload.s, "{\"summary\":\"ok\"}"));
    free(payload.s);
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
    test_muse_split_closes_on_fed_reasoning_boundary();
    test_muse_plain_thinking_close_leaves_no_recipient_residue();
    test_detect_and_render_apertus(&t);
    test_render_apertus_without_system();
    test_render_system_prompt();
    test_render_without_system();
    test_name_roundtrip();
    test_gemma4_generation_prompt();
    test_chatml_think_shape();
    test_muse_tools_and_result_golden();
    test_muse_tool_result_id_resolves_prior_name();
    test_muse_parallel_tool_history_has_native_turn_boundaries();
    test_muse_tool_history_skips_bad_calls_without_leading_boundary();
    test_muse_user_payload_strip_removes_only_recipient_header();

    tokenizer_free(&t);
    gguf_close(&g);
    puts("template tests ok");
    return 0;
}
