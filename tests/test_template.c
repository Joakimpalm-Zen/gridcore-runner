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
        "{{ '[INST] ' + message['content'] + '[/INST]' }}{% endfor %}";

    assert(template_detect(llama2, t) == TMPL_LLAMA2);
    assert(template_detect(mistral, t) == TMPL_MISTRAL);
}

// "Mistral" is not one framing. Three are in circulation, and the difference
// between them is a SPACE next to each instruction marker -- which is not
// cosmetic: on Mistral-7B-Instruct-v0.3, measured on hardware, `[INST] What is
// 2+2? [/INST]` is 11 SentencePiece tokens and `[INST] What is 2+2?[/INST]` is
// 10. Each divergent space costs a token the model was not trained to see
// there.
//
// The three, with the fragments below copied from the checkpoints' own
// tokenizer_config.json chat_template (fetched 2026-08-14):
//
//   v0.1 / v0.2   ' [INST] ' ... ' [/INST]'   byte-identical templates
//                 (sha256[:16] 796853172235ab3a for both), system text folded
//                 into the FIRST user turn
//   v0.3 / Small  "[INST] " ... "[/INST]"     byte-identical templates
//   -2409         (sha256[:16] e16746b40344d6c5 for both), system text folded
//                 into the LAST user turn
//   Nemo-2407     "[INST]" ... "[/INST]"      (sha256[:16] e4676cb56dffea77),
//                 no space anywhere, system text into the LAST user turn
//
// Detection reads the TEMPLATE TEXT, one line from the <<SYS>> split that
// already separates llama-2 from Mistral, because the template IS the
// artifact: a vocabulary size or a model name is a description of the
// checkpoint and can drift from it independently, while the template cannot
// drift from itself. The predicate is a rendered fact, not a spelling:
//
//   a space immediately before [/INST]  -> the v0.1/v0.2 form
//   otherwise, no space after [INST]    -> Nemo
//   otherwise                           -> the v0.3 form
//
// v0.3 is the default because its form is shared by Mistral-Small-2409 and the
// wider derivative population; Nemo is the outlier, not the rule.
static void test_detect_mistral_variants(tokenizer *t) {
    // mistralai/Mistral-7B-Instruct-v0.1 and -v0.2 (identical templates)
    const char *v1 =
        "{%- if message['role'] == 'user' %}\n"
        "    {%- if loop.first and system_message is defined %}\n"
        "        {{- ' [INST] ' + system_message + '\\n\\n'"
        " + message['content'] + ' [/INST]' }}\n"
        "    {%- else %}\n"
        "        {{- ' [INST] ' + message['content'] + ' [/INST]' }}\n"
        "    {%- endif %}\n"
        "{%- endif %}";
    // mistralai/Mistral-7B-Instruct-v0.3 and Mistral-Small-Instruct-2409
    const char *v3 =
        "{%- if loop.last and system_message is defined %}\n"
        "    {{- \"[INST] \" + system_message + \"\\n\\n\""
        " + message[\"content\"] + \"[/INST]\" }}\n"
        "{%- else %}\n"
        "    {{- \"[INST] \" + message[\"content\"] + \"[/INST]\" }}\n"
        "{%- endif %}";
    // mistralai/Mistral-Nemo-Instruct-2407
    const char *nemo =
        "{%- if loop.last and system_message is defined %}\n"
        "    {{- \"[INST]\" + system_message + \"\\n\\n\""
        " + message[\"content\"] + \"[/INST]\" }}\n"
        "{%- else %}\n"
        "    {{- \"[INST]\" + message[\"content\"] + \"[/INST]\" }}\n"
        "{%- endif %}";

    assert(template_detect(v1, t) == TMPL_MISTRAL_V1);
    assert(template_detect(v3, t) == TMPL_MISTRAL);
    assert(template_detect(nemo, t) == TMPL_MISTRAL_NEMO);

    // A Mistral-shaped template this build has never seen renders the v0.3
    // form rather than guessing: that is the population, and it is also what
    // `--chat-template mistral` selects.
    const char *unknown =
        "{{ bos_token }}{% for m in messages %}"
        "{{ '[INST] ' + m['content'] + '[/INST]' }}{% endfor %}";
    assert(template_detect(unknown, t) == TMPL_MISTRAL);

    // llama-2 keeps its own path: <<SYS>> is still what splits the families,
    // and llama-2's template also writes ' [/INST]' with a leading space, so
    // the variant predicate must never be reached for it.
    const char *llama2 =
        "{% if messages[0]['role'] == 'system' %}[INST] <<SYS>>\n"
        "{{ messages[0]['content'] }}\n<</SYS>>\n\n{% endif %}"
        "{{ content + ' [/INST]' }}";
    assert(template_detect(llama2, t) == TMPL_LLAMA2);
}

// Every expected string below was produced by rendering the checkpoint's OWN
// chat_template through jinja with transformers' semantics (trim_blocks and
// lstrip_blocks on), then removing the leading BOS -- which runner never
// spells into a template because the tokenizer prepends it at encode time.
// They are not derived from runner's own output; that is the mistake this
// family's four-year-old rendering was.
static void test_render_mistral_v1(void) {
    char out[512];
    const chat_msg user[] = { { "user", "HI" } };
    const chat_msg sys_user[] = { { "system", "SYS" }, { "user", "HI" } };
    const chat_msg turns[] = { { "system", "SYS" }, { "user", "U1" },
                               { "assistant", "A1" }, { "user", "U2" } };

    render_messages(TMPL_MISTRAL_V1, user, 1, true, THINK_DEFAULT,
                    out, sizeof(out));
    assert(strcmp(out, " [INST] HI [/INST]") == 0);

    // system text leads the FIRST user turn in this form
    render_messages(TMPL_MISTRAL_V1, sys_user, 2, true, THINK_DEFAULT,
                    out, sizeof(out));
    assert(strcmp(out, " [INST] SYS\n\nHI [/INST]") == 0);

    render_messages(TMPL_MISTRAL_V1, turns, 4, true, THINK_DEFAULT,
                    out, sizeof(out));
    assert(strcmp(out, " [INST] SYS\n\nU1 [/INST] A1</s> [INST] U2 [/INST]")
           == 0);
}

static void test_render_mistral_v3(void) {
    char out[512];
    const chat_msg user[] = { { "user", "HI" } };
    const chat_msg sys_user[] = { { "system", "SYS" }, { "user", "HI" } };
    const chat_msg turns[] = { { "system", "SYS" }, { "user", "U1" },
                               { "assistant", "A1" }, { "user", "U2" } };

    render_messages(TMPL_MISTRAL, user, 1, true, THINK_DEFAULT,
                    out, sizeof(out));
    assert(strcmp(out, "[INST] HI[/INST]") == 0);

    render_messages(TMPL_MISTRAL, sys_user, 2, true, THINK_DEFAULT,
                    out, sizeof(out));
    assert(strcmp(out, "[INST] SYS\n\nHI[/INST]") == 0);

    // and here is the difference that is not whitespace: the system text goes
    // to the LAST user turn, not the first
    render_messages(TMPL_MISTRAL, turns, 4, true, THINK_DEFAULT,
                    out, sizeof(out));
    assert(strcmp(out, "[INST] U1[/INST] A1</s>[INST] SYS\n\nU2[/INST]") == 0);

    // the assistant turn is ' ' + content|trim + eos in this form, so an empty
    // one is a bare ' </s>' and a padded one loses its padding
    const chat_msg empty[] = { { "user", "U1" }, { "assistant", "" },
                               { "user", "U2" } };
    render_messages(TMPL_MISTRAL, empty, 3, true, THINK_DEFAULT,
                    out, sizeof(out));
    assert(strcmp(out, "[INST] U1[/INST] </s>[INST] U2[/INST]") == 0);

    const chat_msg padded[] = { { "user", "U1" }, { "assistant", "  A1  " },
                                { "user", "U2" } };
    render_messages(TMPL_MISTRAL, padded, 3, true, THINK_DEFAULT,
                    out, sizeof(out));
    assert(strcmp(out, "[INST] U1[/INST] A1</s>[INST] U2[/INST]") == 0);
}

static void test_render_mistral_nemo(void) {
    char out[512];
    const chat_msg user[] = { { "user", "HI" } };
    const chat_msg turns[] = { { "system", "SYS" }, { "user", "U1" },
                               { "assistant", "A1" }, { "user", "U2" } };

    render_messages(TMPL_MISTRAL_NEMO, user, 1, true, THINK_DEFAULT,
                    out, sizeof(out));
    assert(strcmp(out, "[INST]HI[/INST]") == 0);

    render_messages(TMPL_MISTRAL_NEMO, turns, 4, true, THINK_DEFAULT,
                    out, sizeof(out));
    assert(strcmp(out, "[INST]U1[/INST]A1</s>[INST]SYS\n\nU2[/INST]") == 0);

    // Nemo does NOT trim the assistant turn -- v0.3 does. Same conversation,
    // two different byte strings, which is the whole reason the variants are
    // separate renderers rather than one with a spacing switch.
    const chat_msg padded[] = { { "user", "U1" }, { "assistant", "  A1  " },
                                { "user", "U2" } };
    render_messages(TMPL_MISTRAL_NEMO, padded, 3, true, THINK_DEFAULT,
                    out, sizeof(out));
    assert(strcmp(out, "[INST]U1[/INST]  A1  </s>[INST]U2[/INST]") == 0);
}

// The one place runner deliberately does NOT follow the v0.3 and Nemo
// references. Their fold is `loop.last`: the system text is attached only when
// the LAST message is a user turn, so a conversation ending in an assistant
// turn -- a prefill or continuation request -- renders with the caller's
// system prompt silently discarded. Verified against both templates on
// 2026-08-14: [system, user, assistant] renders '[INST] U1[/INST] A1</s>',
// with no SYS anywhere.
//
// Runner attaches it to the last USER turn instead. Every conversation that
// ends with a user turn -- which is every generation request -- renders
// byte-identically to the reference; only the prefill shape differs, and it
// differs by keeping a system prompt the caller sent rather than dropping it.
static void test_render_mistral_keeps_system_on_a_prefill(void) {
    const chat_msg msgs[] = { { "system", "SYS" }, { "user", "U1" },
                              { "assistant", "A1" } };
    char out[512];

    render_messages(TMPL_MISTRAL, msgs, 3, false, THINK_DEFAULT,
                    out, sizeof(out));
    assert(strcmp(out, "[INST] SYS\n\nU1[/INST] A1</s>") == 0);
    render_messages(TMPL_MISTRAL_NEMO, msgs, 3, false, THINK_DEFAULT,
                    out, sizeof(out));
    assert(strcmp(out, "[INST]SYS\n\nU1[/INST]A1</s>") == 0);
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

// Phi-3's reference closes an un-prompted render with the eos token:
//
//   {% if add_generation_prompt %}{{ '<|assistant|>\n' }}
//   {% else %}{{ eos_token }}{% endif %}
//
// -- microsoft/Phi-3.5-mini-instruct tokenizer_config.json, whose eos_token is
// '<|endoftext|>'. Runner emitted nothing in the else branch, so a render with
// add_assistant=false stopped one token short of the reference.
//
// Reachability, stated plainly rather than inflated: every HTTP path renders
// with add_assistant=true, so no server request takes this branch today. It is
// reachable through render_messages(), which template.h publishes, and through
// the conformance gate's *-nogen rows. A published renderer that renders the
// wrong bytes is a bug at whatever severity.
//
// Zephyr shares the <|role|> framing but NOT this branch: its reference emits
// the generation prompt inside the loop and nothing at all otherwise
// (HuggingFaceH4/zephyr-7b-beta), so the assertion below pins them apart.
static void test_phi3_without_generation_prompt_ends_with_eos(void) {
    const chat_msg msgs[] = { { "user", "HI" }, { "assistant", "YO" } };
    char out[512];

    render_messages(TMPL_PHI3, msgs, 2, false, THINK_DEFAULT, out, sizeof(out));
    assert(strcmp(out, "<|user|>\nHI<|end|>\n"
                       "<|assistant|>\nYO<|end|>\n"
                       "<|endoftext|>") == 0);

    // with the generation prompt, nothing changes
    render_messages(TMPL_PHI3, msgs, 2, true, THINK_DEFAULT, out, sizeof(out));
    assert(strcmp(out, "<|user|>\nHI<|end|>\n"
                       "<|assistant|>\nYO<|end|>\n"
                       "<|assistant|>\n") == 0);

    render_messages(TMPL_ZEPHYR, msgs, 2, false, THINK_DEFAULT,
                    out, sizeof(out));
    assert(strcmp(out, "<|user|>\nHI</s>\n<|assistant|>\nYO</s>\n") == 0);
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

// enable_thinking=false is a request NOT to reason, and the ornith reference
// answers it with an ALREADY-CLOSED thought block:
//
//   {%- if add_generation_prompt %}
//       {{- '<|im_start|>assistant\n' }}
//       {%- if enable_thinking is defined and enable_thinking is false %}
//           {{- '<think>\n\n</think>\n\n' }}
//       {%- else %}
//           {{- '<think>\n' }}
//       {%- endif %}
//   {%- endif %}
//
// (ornith-ai/Ornith-1.0-9B tokenizer_config.json, rendered through jinja2 with
// transformers' trim_blocks/lstrip_blocks; the same three renders are the
// ornith/think-on and ornith/think-off rows of the conformance gate.)
//
// Runner used to emit the OPEN block in every mode, so a caller asking for no
// reasoning got the exact construct that starts it. Both other modes are
// asserted too: the reference's `else` covers THINK_ON *and* the undefined
// case, so neither may grow a closed block.
static void test_ornith_thinking_off_closes_the_thought_block(void) {
    const chat_msg msgs[] = { { "user", "HI" } };
    char out[512];
    const char *head = "<|im_start|>user\nHI<|im_end|>\n<|im_start|>assistant\n";

    render_messages(TMPL_ORNITH, msgs, 1, true, THINK_OFF, out, sizeof(out));
    assert(strcmp(out, "<|im_start|>user\nHI<|im_end|>\n"
                       "<|im_start|>assistant\n<think>\n\n</think>\n\n") == 0);

    char open_block[512];
    snprintf(open_block, sizeof(open_block), "%s<think>\n", head);
    render_messages(TMPL_ORNITH, msgs, 1, true, THINK_ON, out, sizeof(out));
    assert(strcmp(out, open_block) == 0);
    render_messages(TMPL_ORNITH, msgs, 1, true, THINK_DEFAULT, out, sizeof(out));
    assert(strcmp(out, open_block) == 0);

    // no generation prompt: no thought block in any mode
    render_messages(TMPL_ORNITH, msgs, 1, false, THINK_OFF, out, sizeof(out));
    assert(strcmp(out, "<|im_start|>user\nHI<|im_end|>\n") == 0);
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


// ---------------------------------------------------------------- Harmony
//
// gpt-oss. Before 2026-08-14 this fell through template_detect's terminal
// fallback to llama2 and was fed [INST]/<<SYS>> markup that appears nowhere
// in its own GGUF; chat ran away and hallucinated [/INST]. These pin the
// rendering, the detection, and the analysis-channel split.

static void test_detect_harmony(tokenizer *t) {
    // <|channel|> + <|return|> is the pair no other family has. muse claims
    // <|start|> + <|eot|>, which Harmony lacks, so the branches cannot collide.
    const char *harmony =
        "{%- for m in messages %}<|start|>{{ m.role }}<|message|>{{ m.content }}"
        "<|end|>{%- endfor %}<|start|>assistant<|channel|>analysis<|message|>"
        "...<|return|>";
    assert(template_detect(harmony, t) == TMPL_HARMONY);
    // and the families it must not steal
    const char *muse = "<|start|>assistant to=user<|message|>x<|eot|>";
    assert(template_detect(muse, t) == TMPL_MUSE);
    const char *l3 = "<|start_header_id|>user<|end_header_id|>";
    assert(template_detect(l3, t) == TMPL_LLAMA3);
    const char *granite = "<|start_of_role|>user<|end_of_role|>";
    assert(template_detect(granite, t) == TMPL_GRANITE);
}

// Rendered with openai-harmony 0.0.8 (git abd677f7) through
//
//     Conversation.from_messages([
//         Message.from_role_and_content(Role.SYSTEM, SystemContent.new()),
//         Message.from_role_and_content(Role.DEVELOPER,
//             DeveloperContent.new().with_instructions("Be terse.")),
//         ... user / assistant(final) / user ...])
//
// then two documented Runner deltas applied by hand:
//   - the reference's "Current date: <today>" line is omitted, because it is a
//     live wall-clock value that would evict the prefix cache at midnight (see
//     the comment on the Harmony branch in template.c). The "Knowledge cutoff"
//     line beside it is NOT omitted: it is a frozen literal and costs nothing;
//   - the reference stops at the last turn; Runner appends its own primed
//     generation header, `<|channel|>analysis` for THINK_DEFAULT.
// Everything else below is the reference's bytes verbatim. Note in particular
// that `# Valid channels` lists commentary even with no tools declared: the
// reference's channel list is a constant (chat.rs, SystemContent::default),
// not a function of the tool set.
static void test_harmony_render_golden(void) {
    const chat_msg msgs[] = {
        { .role = "system",    .content = "Be terse." },
        { .role = "user",      .content = "What is 2+2?" },
        { .role = "assistant", .content = "4" },
        { .role = "user",      .content = "And 3+3?" },
    };
    char out[4096];
    render_messages(TMPL_HARMONY, msgs, 4, true, THINK_DEFAULT,
                    out, sizeof(out));
    assert(strcmp(out,
        "<|start|>system<|message|>You are ChatGPT, a large language model "
        "trained by OpenAI.\nKnowledge cutoff: 2024-06\n\nReasoning: medium"
        "\n\n# Valid channels: "
        "analysis, commentary, final. Channel must be included for every "
        "message.<|end|>"
        "<|start|>developer<|message|># Instructions\n\nBe terse.<|end|>"
        "<|start|>user<|message|>What is 2+2?<|end|>"
        "<|start|>assistant<|channel|>final<|message|>4<|end|>"
        "<|start|>user<|message|>And 3+3?<|end|>"
        "<|start|>assistant<|channel|>analysis<|message|>") == 0);
}

// The knowledge cutoff is part of the system preamble, and Runner used to drop
// it along with the current date. Those two lines are not the same kind of
// thing. `Current date:` is strftime_now() — a live wall-clock value that would
// change the prompt prefix at midnight and evict the whole prefix cache, so
// Runner omits it (the same call llama3, apertus and muse make). The cutoff is
// a FROZEN string literal: identical on every render, no interpolation, no
// cache cost. Dropping it cost the model a fact it is trained to have and
// bought nothing.
//
// Both references agree on the bytes. openai-harmony 0.0.8 (git abd677f7)
// renders SystemContent.new() with no conversation_start_date as
//
//     ...trained by OpenAI.\nKnowledge cutoff: 2024-06\n\nReasoning: medium\n\n
//
// (verbatim in its own test-data/test_reasoning_system_message.txt), and the
// `chat_template` in models/gpt-oss-20b-MXFP4.gguf emits the same literal at
// its build_system_message() macro, between the identity line and the
// conditional `Current date:`. Note the separator: identity ends with a SINGLE
// \n now, because the blank line before `Reasoning:` comes from the end of the
// cutoff/date block, not from the identity line.
static void test_harmony_system_turn_carries_the_knowledge_cutoff(void) {
    const chat_msg msgs[] = { { .role = "user", .content = "hi" } };
    char out[2048];
    render_messages(TMPL_HARMONY, msgs, 1, true, THINK_DEFAULT,
                    out, sizeof(out));
    assert(strstr(out,
        "<|start|>system<|message|>You are ChatGPT, a large language model "
        "trained by OpenAI.\nKnowledge cutoff: 2024-06\n\n"
        "Reasoning: medium\n\n") != NULL);
    // and the live date stays out: no `Current date:` line, in any year
    assert(strstr(out, "Current date") == NULL);
}

static void test_harmony_render_without_system(void) {
    const chat_msg msgs[] = { { .role = "user", .content = "hi" } };
    char out[2048];
    render_messages(TMPL_HARMONY, msgs, 1, true, THINK_DEFAULT,
                    out, sizeof(out));
    // no developer turn when the caller supplied no system message
    assert(strstr(out, "<|start|>developer") == NULL);
    assert(strstr(out, "<|start|>user<|message|>hi<|end|>") != NULL);
    assert(strstr(out, "<|start|>assistant") != NULL);
}

// Declaring function tools adds one line to the SYSTEM turn, immediately after
// the channel list. The reference gates it on the conversation containing a
// developer `functions` namespace with at least one tool
// (encoding.rs:175-194, :996-1001), so it is present exactly when Runner has
// tools to render and absent otherwise. Confirmed against openai-harmony
// 0.0.8 (git abd677f7) via DeveloperContent.new().with_function_tools([...]).
static void test_harmony_tools_add_the_commentary_routing_line(void) {
    const char *src =
        "[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\","
        "\"description\":\"Get weather\",\"parameters\":{\"type\":\"object\","
        "\"properties\":{\"city\":{\"type\":\"string\"}},"
        "\"required\":[\"city\"]}}}]";
    jv *tools = json_parse(src, strlen(src));
    assert(tools != NULL);
    const chat_msg msgs[] = { { .role = "user", .content = "hi" } };
    char with[4096], without[4096];
    render_messages_with_tools(TMPL_HARMONY, msgs, 1, true, THINK_DEFAULT,
                               tools, with, sizeof(with));
    render_messages_with_tools(TMPL_HARMONY, msgs, 1, true, THINK_DEFAULT,
                               NULL, without, sizeof(without));
    assert(strstr(with,
        "# Valid channels: analysis, commentary, final. Channel must be "
        "included for every message.\nCalls to these tools must go to the "
        "commentary channel: 'functions'.<|end|>") != NULL);
    assert(strstr(without, "Calls to these tools") == NULL);
    jv_free(tools);
}

// Tool requests leave the assistant header bare so the model can choose
// analysis, commentary, or final.
//
// WHICH ORACLE CALL PRODUCED THIS, and why it matters. openai-harmony has TWO
// slots that render an identical `# Tools` namespace, and picking the wrong one
// changes only the POSITION of the block, never a byte inside it:
//
//   SystemContent.new().with_tools(ns)          -> namespace in the SYSTEM turn
//   DeveloperContent.new().with_function_tools  -> namespace in the DEVELOPER turn
//
// The SYSTEM slot is for the model's BUILT-IN tools (browser, python). OpenAI
// function tools — everything Runner ever renders — belong in the DEVELOPER
// slot. An earlier version of this golden was generated through
// SystemContent.with_tools, so it verified the namespace's CONTENTS while
// blessing the wrong POSITION for two releases. The bytes below came from
//
//     Conversation.from_messages([
//         Message.from_role_and_content(Role.SYSTEM, SystemContent.new()),
//         Message.from_role_and_content(Role.DEVELOPER,
//             DeveloperContent.new().with_instructions("Be terse.")
//                                   .with_function_tools([get_weather])),
//         Message.from_role_and_content(Role.USER, "Weather in Oslo?")])
//
// rendered by openai-harmony 0.0.8 (git abd677f7) and cross-checked against the
// `chat_template` embedded in models/gpt-oss-20b-MXFP4.gguf, which agrees.
// Two documented Runner deltas are applied by hand: the reference's live
// "Current date:" line is omitted (the frozen "Knowledge cutoff" beside it is
// kept), and Runner appends its own generation header (bare
// `<|start|>assistant` when tools are declared).
static void test_harmony_tool_definitions_golden(void) {
    const char *src =
        "[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\","
        "\"description\":\"Get weather\",\"parameters\":{\"type\":\"object\","
        "\"properties\":{\"city\":{\"type\":\"string\"}},"
        "\"required\":[\"city\"]}}}]";
    jv *tools = json_parse(src, strlen(src));
    assert(tools != NULL);
    const chat_msg msgs[] = {
        { .role = "system", .content = "Be terse." },
        { .role = "user", .content = "Weather in Oslo?" },
    };
    char out[4096];
    render_messages_with_tools(TMPL_HARMONY, msgs, 2, true, THINK_DEFAULT,
                               tools, out, sizeof(out));
    assert(strcmp(out,
        "<|start|>system<|message|>You are ChatGPT, a large language model "
        "trained by OpenAI.\nKnowledge cutoff: 2024-06\n\nReasoning: medium"
        "\n\n# Valid channels: "
        "analysis, commentary, final. Channel must be included for every "
        "message.\nCalls to these tools must go to the commentary channel: "
        "'functions'.<|end|>"
        "<|start|>developer<|message|># Instructions\n\nBe terse.\n\n"
        "# Tools\n\n## functions\n\n"
        "namespace functions {\n\n// Get weather\n"
        "type get_weather = (_: {\ncity: string,\n}) => any;\n\n"
        "} // namespace functions<|end|>"
        "<|start|>user<|message|>Weather in Oslo?<|end|>"
        "<|start|>assistant") == 0);
    jv_free(tools);
}

// The developer turn exists for EITHER half of its content. Runner used to emit
// one only for a caller system message, so tools with no system message had
// nowhere left to go once the namespace moved off the system turn. The
// reference emits `# Tools` alone, with no `# Instructions` heading and no
// leading blank line; the GGUF jinja agrees (`{%- if developer_message or
// tools %}`). Bytes from openai-harmony 0.0.8 (git abd677f7),
// DeveloperContent.new().with_function_tools([get_weather]) and no
// .with_instructions(), minus the live current-date line and plus Runner's
// generation header.
static void test_harmony_tools_without_instructions_still_get_a_developer_turn(void) {
    const char *src =
        "[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\","
        "\"description\":\"Get weather\",\"parameters\":{\"type\":\"object\","
        "\"properties\":{\"city\":{\"type\":\"string\"}},"
        "\"required\":[\"city\"]}}}]";
    jv *tools = json_parse(src, strlen(src));
    assert(tools != NULL);
    const chat_msg msgs[] = { { .role = "user", .content = "hi" } };
    char out[4096];
    render_messages_with_tools(TMPL_HARMONY, msgs, 1, true, THINK_DEFAULT,
                               tools, out, sizeof(out));
    assert(strcmp(out,
        "<|start|>system<|message|>You are ChatGPT, a large language model "
        "trained by OpenAI.\nKnowledge cutoff: 2024-06\n\nReasoning: medium"
        "\n\n# Valid channels: "
        "analysis, commentary, final. Channel must be included for every "
        "message.\nCalls to these tools must go to the commentary channel: "
        "'functions'.<|end|>"
        "<|start|>developer<|message|># Tools\n\n## functions\n\n"
        "namespace functions {\n\n// Get weather\n"
        "type get_weather = (_: {\ncity: string,\n}) => any;\n\n"
        "} // namespace functions<|end|>"
        "<|start|>user<|message|>hi<|end|>"
        "<|start|>assistant") == 0);
    jv_free(tools);
}

// Every caller system message lands in ONE developer turn, in order, joined by
// a blank line — and that turn sits directly after the system preamble even
// when a system message arrived late in the list. Both the reference and the
// GGUF jinja put developer content in exactly one turn in exactly that
// position, and the tool namespace has to append to the same turn, so a second
// developer turn opened mid-history would leave the namespace with two possible
// homes. The instruction bytes match openai-harmony 0.0.8 (git abd677f7)
// rendered from DeveloperContent.new().with_instructions("Be terse.\n\nBe
// kind."), which is what folding two messages produces.
static void test_harmony_folds_every_system_message_into_one_developer_turn(void) {
    const chat_msg msgs[] = {
        { .role = "system", .content = "Be terse." },
        { .role = "user",   .content = "hi" },
        { .role = "system", .content = "Be kind." },
    };
    char out[4096];
    render_messages(TMPL_HARMONY, msgs, 3, true, THINK_DEFAULT,
                    out, sizeof(out));
    assert(strstr(out,
        "message.<|end|>"
        "<|start|>developer<|message|># Instructions\n\nBe terse.\n\n"
        "Be kind.<|end|>"
        "<|start|>user<|message|>hi<|end|>") != NULL);
    // exactly one developer turn, not one per system message
    const char *first = strstr(out, "<|start|>developer");
    assert(first != NULL);
    assert(strstr(first + 1, "<|start|>developer") == NULL);
}

// The TypeScript inside `namespace functions` is the tool-calling half of the
// prompt contract, so the schema renderer gets golden coverage per branch.
// Every expected block below was taken from the pinned protocol oracle
// (openai-harmony abd677f7, `Encoding::json_schema_to_typescript`) by
// rendering the same tool through the reference encoder, unless the test says
// otherwise in its own comment.
static void assert_harmony_tool_ts(const char *tools_json, const char *want) {
    jv *tools = json_parse(tools_json, strlen(tools_json));
    assert(tools != NULL);
    const chat_msg msgs[] = { { .role = "user", .content = "hi" } };
    char out[8192];
    render_messages_with_tools(TMPL_HARMONY, msgs, 1, true, THINK_DEFAULT,
                               tools, out, sizeof(out));
    if (!strstr(out, want)) {
        fprintf(stderr, "harmony schema mismatch\n--- want ---\n%s\n"
                        "--- got ---\n%s\n", want, out);
        assert(!"harmony tool namespace mismatch");
    }
    jv_free(tools);
}

// Scalar types collapse to TypeScript's three: integer and number both become
// number, and every type name the reference does not spell — "null" and any
// unknown name alike — becomes any, not a literal null.
static void test_harmony_schema_scalar_branches(void) {
    assert_harmony_tool_ts(
        "[{\"type\":\"function\",\"function\":{\"name\":\"scalars\","
        "\"description\":\"Scalar kinds\",\"parameters\":{\"type\":\"object\","
        "\"properties\":{\"i\":{\"type\":\"integer\"},\"n\":{\"type\":\"number\"},"
        "\"b\":{\"type\":\"boolean\"},\"z\":{\"type\":\"null\"},"
        "\"u\":{\"type\":\"widget\"}},"
        "\"required\":[\"i\",\"n\",\"b\",\"z\",\"u\"]}}}]",
        "namespace functions {\n\n// Scalar kinds\n"
        "type scalars = (_: {\n"
        "i: number,\n"
        "n: number,\n"
        "b: boolean,\n"
        "z: any,\n"
        "u: any,\n"
        "}) => any;\n\n} // namespace functions");
}

// Enums become string-literal unions; an array becomes `T[]` and, with no
// items schema to name, `Array<any>`. Element types nest, so an array of
// objects carries the object's own indentation into the `[]` suffix. Only the
// string alternatives of an enum are offered, and an enum with none left to
// offer degrades to plain string rather than to an empty type.
static void test_harmony_schema_enum_and_array_branches(void) {
    assert_harmony_tool_ts(
        "[{\"type\":\"function\",\"function\":{\"name\":\"arrs\","
        "\"description\":\"Enums and arrays\",\"parameters\":{\"type\":\"object\","
        "\"properties\":{"
        "\"unit\":{\"type\":\"string\",\"enum\":[\"celsius\",\"fahrenheit\"]},"
        "\"mixed\":{\"type\":\"string\",\"enum\":[\"a\",1]},"
        "\"nums\":{\"type\":\"string\",\"enum\":[1,2]},"
        "\"tags\":{\"type\":\"array\",\"items\":{\"type\":\"string\"}},"
        "\"bag\":{\"type\":\"array\"},"
        "\"mat\":{\"type\":\"array\",\"items\":{\"type\":\"array\","
        "\"items\":{\"type\":\"integer\"}}},"
        "\"rows\":{\"type\":\"array\",\"items\":{\"type\":\"object\","
        "\"properties\":{\"id\":{\"type\":\"integer\"}},\"required\":[\"id\"]}}},"
        "\"required\":[\"unit\"]}}}]",
        "namespace functions {\n\n// Enums and arrays\n"
        "type arrs = (_: {\n"
        "unit: \"celsius\" | \"fahrenheit\",\n"
        "mixed?: \"a\",\n"
        "nums?: string,\n"
        "tags?: string[],\n"
        "bag?: Array<any>,\n"
        "mat?: number[][],\n"
        "rows?: {\n"
        "    id: number,\n"
        "    }[],\n"
        "}) => any;\n\n} // namespace functions");
}

// A type given as an array of names is a union of those names. `nullable` adds
// " | null", but only when the type does not already offer null, so "t" below
// stays a two-member union rather than repeating null. Non-string entries in a
// type list are dropped.
static void test_harmony_schema_type_array_and_nullable(void) {
    assert_harmony_tool_ts(
        "[{\"type\":\"function\",\"function\":{\"name\":\"tan\","
        "\"description\":\"Type arrays and nullable\","
        "\"parameters\":{\"type\":\"object\",\"properties\":{"
        "\"x\":{\"type\":[\"number\",\"string\"]},"
        "\"y\":{\"type\":[\"integer\",\"null\"]},"
        "\"s\":{\"type\":\"string\",\"nullable\":true},"
        "\"t\":{\"type\":[\"string\",\"null\"],\"nullable\":true},"
        "\"e\":{\"type\":[\"string\",7]},"
        "\"z\":{\"type\":[]}},"
        "\"required\":[\"x\"]}}}]",
        "namespace functions {\n\n// Type arrays and nullable\n"
        "type tan = (_: {\n"
        "x: number | string,\n"
        "y?: number | null,\n"
        "s?: string | null,\n"
        "t?: string | null,\n"
        "e?: string,\n"
        "z?: any,\n"
        "}) => any;\n\n} // namespace functions");
}

// Titles, descriptions and examples become comment lines above the property;
// a title is followed by a bare `//` separator, and only string examples are
// listed. A default trails the property. The default of an enum property is
// printed bare because the enum alternatives above it are already quoted.
static void test_harmony_schema_comments_and_defaults(void) {
    assert_harmony_tool_ts(
        "[{\"type\":\"function\",\"function\":{\"name\":\"cmt\","
        "\"description\":\"Titles descriptions examples defaults\","
        "\"parameters\":{\"type\":\"object\","
        "\"description\":\"The outer object.\",\"properties\":{"
        "\"a\":{\"type\":\"string\",\"title\":\"The A\",\"description\":\"an a\","
        "\"examples\":[\"one\",2,\"two\"]},"
        "\"b\":{\"type\":\"string\",\"default\":\"hi\"},"
        "\"c\":{\"type\":\"integer\",\"default\":7},"
        "\"d\":{\"type\":\"string\",\"enum\":[\"x\",\"y\"],\"default\":\"x\"},"
        "\"e\":{\"type\":\"boolean\",\"default\":false}},"
        "\"required\":[]}}}]",
        "namespace functions {\n\n// Titles descriptions examples defaults\n"
        "type cmt = (_: // The outer object.\n"
        "{\n"
        "// The A\n"
        "//\n"
        "// an a\n"
        "// Examples:\n"
        "// - \"one\"\n"
        "// - \"two\"\n"
        "a?: string,\n"
        "b?: string, // default: \"hi\"\n"
        "c?: number, // default: 7\n"
        "d?: \"x\" | \"y\", // default: x\n"
        "e?: boolean, // default: false\n"
        "}) => any;\n\n} // namespace functions");
}

// Nested objects indent their members by four spaces per level and close on an
// indented brace. An object's own description is emitted twice — once above
// the property name, once after it — because the reference prints it from both
// the property loop and the object branch.
static void test_harmony_schema_nested_objects(void) {
    assert_harmony_tool_ts(
        "[{\"type\":\"function\",\"function\":{\"name\":\"nest\","
        "\"description\":\"Nested\",\"parameters\":{\"type\":\"object\","
        "\"properties\":{\"outer\":{\"type\":\"object\","
        "\"description\":\"inner obj\",\"properties\":{"
        "\"deep\":{\"type\":\"object\",\"properties\":{"
        "\"k\":{\"type\":\"string\"}},\"required\":[\"k\"]},"
        "\"flat\":{\"type\":\"string\"}},\"required\":[\"deep\"]}},"
        "\"required\":[\"outer\"]}}}]",
        "namespace functions {\n\n// Nested\n"
        "type nest = (_: {\n"
        "// inner obj\n"
        "outer:     // inner obj\n"
        "{\n"
        "    deep: {\n"
        "        k: string,\n"
        "        },\n"
        "    flat?: string,\n"
        "    },\n"
        "}) => any;\n\n} // namespace functions");
}

// DELIBERATELY MIRRORS A KNOWN UPSTREAM QUIRK — do not "fix" this back.
//
// THE RULE: the TOOL-level description splits into one comment per line.
// EVERY other description or title takes a single "// " prefix and is not
// split. openai-harmony abd677f7:
//     tool-level      for line in tool.description.lines()  encoding.rs:775-777
//     object-level    format!("{indent}// {desc_str}\n")    encoding.rs:486-488
//     property title  format!("{indent}// {t}\n{indent}//\n") encoding.rs:508-510
//     property desc   format!("{indent}// {desc_str}\n")    encoding.rs:518
//     oneOf property  format!("{indent}// {desc_str}\n")    encoding.rs:565
// so a multi-line value at any of the four non-tool sites drops its
// continuation into the type body as a bare, uncommented line — and the
// continuation carries no indent, because the newline is inside the formatted
// string rather than between two formatted lines. The official gpt-oss jinja
// chat template does the same, which is what llama.cpp and LM Studio drive the
// model with.
//
// That is invalid TypeScript, and it is nonetheless the contract: the owner's
// call is cross-engine portability, so the same tool schema must produce the
// same prompt bytes in runner as in llama.cpp. Commenting the continuation
// lines here would make runner the odd one out. Every expected block below was
// rendered through openai-harmony abd677f7 itself and copied byte-for-byte.
static void test_harmony_schema_multiline_property_text_mirrors_reference(void) {
    assert_harmony_tool_ts(
        "[{\"type\":\"function\",\"function\":{\"name\":\"md\","
        "\"description\":\"Multi-line description\","
        "\"parameters\":{\"type\":\"object\",\"properties\":{"
        "\"f\":{\"type\":\"string\",\"title\":\"TITLE one\\nTITLE two\","
        "\"description\":\"multi\\nline desc\"}},"
        "\"required\":[\"f\"]}}}]",
        "namespace functions {\n\n// Multi-line description\n"
        "type md = (_: {\n"
        "// TITLE one\n"
        "TITLE two\n"
        "//\n"
        "// multi\n"
        "line desc\n"
        "f: string,\n"
        "}) => any;\n\n} // namespace functions");

    // the description a oneOf property lifts above its own name takes the
    // same single-prefix path in the reference (encoding.rs:565)
    assert_harmony_tool_ts(
        "[{\"type\":\"function\",\"function\":{\"name\":\"oo\","
        "\"description\":\"OneOf multi\","
        "\"parameters\":{\"type\":\"object\",\"properties\":{"
        "\"val\":{\"description\":\"one-of\\nspanning two\",\"oneOf\":["
        "{\"type\":\"string\"},{\"type\":\"integer\"}]}},"
        "\"required\":[\"val\"]}}}]",
        "namespace functions {\n\n// OneOf multi\n"
        "type oo = (_: {\n"
        "// one-of\n"
        "spanning two\n"
        "val:\n"
        " | string\n"
        " | number\n"
        ",\n"
        "}) => any;\n\n} // namespace functions");

    // An object schema's OWN description is the fourth non-tool site
    // (encoding.rs:486-488). At the top level it lands between "(_: " and the
    // opening brace, so the continuation line separates the two.
    assert_harmony_tool_ts(
        "[{\"type\":\"function\",\"function\":{\"name\":\"ob\","
        "\"description\":\"Object level\","
        "\"parameters\":{\"type\":\"object\","
        "\"description\":\"OBJ one\\nOBJ two\",\"properties\":{"
        "\"g\":{\"type\":\"string\"}},\"required\":[\"g\"]}}}]",
        "namespace functions {\n\n// Object level\n"
        "type ob = (_: // OBJ one\n"
        "OBJ two\n"
        "{\n"
        "g: string,\n"
        "}) => any;\n\n} // namespace functions");

    // The same site at a non-empty indent, reached through a nested object.
    // The reference prints the text TWICE — once as the property description
    // and once as the nested object's own — and indents only the line that
    // carries the "// " marker, so the second copy's continuation stays flush
    // left both times.
    assert_harmony_tool_ts(
        "[{\"type\":\"function\",\"function\":{\"name\":\"nb\","
        "\"description\":\"Nested level\","
        "\"parameters\":{\"type\":\"object\",\"properties\":{"
        "\"inner\":{\"type\":\"object\","
        "\"description\":\"INNER one\\nINNER two\",\"properties\":{"
        "\"k\":{\"type\":\"string\"}},\"required\":[\"k\"]}},"
        "\"required\":[\"inner\"]}}}]",
        "namespace functions {\n\n// Nested level\n"
        "type nb = (_: {\n"
        "// INNER one\n"
        "INNER two\n"
        "inner:     // INNER one\n"
        "INNER two\n"
        "{\n"
        "    k: string,\n"
        "    },\n"
        "}) => any;\n\n} // namespace functions");
}

// A top-level oneOf renders as a leading-pipe union. The leading pipe on the
// FIRST alternative is not a bug: the reference writes "\n<indent> | " on
// every iteration including the first, so the union legitimately opens with a
// pipe on its own line. Alternatives render at indent + 3 spaces and carry
// their description and default as one trailing comment.
static void test_harmony_schema_oneof_toplevel(void) {
    assert_harmony_tool_ts(
        "[{\"type\":\"function\",\"function\":{\"name\":\"ot\","
        "\"description\":\"Top-level oneOf\",\"parameters\":{\"oneOf\":["
        "{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"string\"}},"
        "\"required\":[\"a\"]},"
        "{\"type\":\"string\",\"description\":\"as text\"},"
        "{\"type\":\"integer\",\"default\":3},"
        "{\"type\":\"string\",\"enum\":[\"x\",\"y\"],\"default\":\"x\"},"
        "{\"type\":\"boolean\",\"nullable\":true}]}}}]",
        "namespace functions {\n\n// Top-level oneOf\n"
        "type ot = (_: \n"
        " | {\n"
        "   a: string,\n"
        "   }\n"
        " | string // as text\n"
        " | number // default: 3\n"
        " | \"x\" | \"y\" // default: \"x\"\n"
        " | boolean | null) => any;\n\n} // namespace functions");
}

// A property whose schema is a oneOf takes a different shape from a top-level
// oneOf: the property name ends the line with a bare colon, each alternative
// gets its own line, and the trailing comma sits alone on the closing line.
// The property description and default move above the property name, and the
// first alternative's description is suppressed when it duplicates the
// property description ("same" below), otherwise it stays a trailing comment.
static void test_harmony_schema_oneof_property(void) {
    assert_harmony_tool_ts(
        "[{\"type\":\"function\",\"function\":{\"name\":\"op\","
        "\"description\":\"Property oneOf\",\"parameters\":{\"type\":\"object\","
        "\"properties\":{"
        "\"val\":{\"description\":\"the value\",\"default\":\"q\",\"oneOf\":["
        "{\"type\":\"string\",\"description\":\"as text\"},"
        "{\"type\":\"integer\",\"description\":\"as number\",\"default\":4},"
        "{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"string\"}},"
        "\"required\":[\"a\"]}]},"
        "\"same\":{\"description\":\"shared\",\"oneOf\":["
        "{\"type\":\"string\",\"description\":\"shared\"},"
        "{\"type\":\"boolean\",\"description\":\"other\"}]}},"
        "\"required\":[\"val\"]}}}]",
        "namespace functions {\n\n// Property oneOf\n"
        "type op = (_: {\n"
        "// the value\n"
        "// default: \"q\"\n"
        "val:\n"
        " | string\n"
        " | number // as number default: 4\n"
        " | {\n"
        "   a: string,\n"
        "   }\n"
        ",\n"
        "same?:\n"
        " | string\n"
        " | boolean // other\n"
        ",\n"
        "}) => any;\n\n} // namespace functions");
}

// A continued tool conversation must replay the native recipient-bearing
// assistant call and the named tool result. Generic <|tool_call> wrappers or a
// role:"tool" message are both off-protocol for Harmony.
//
// The result turn carries ` to=assistant`. That recipient is not decoration:
// openai-harmony resolves the author token BEFORE consuming the channel, and a
// namespaced author like "functions.get_weather" is no known Role — it is
// recognised as Role::Tool only through the recipient fallback branch
// (encoding.rs:1386-1398). Without it the reference rejects the whole prompt
// with `Unknown role: functions.get_weather`; with it the turn parses to
// author.role=TOOL, name="functions.get_weather", channel="commentary",
// recipient="assistant". The expected bytes below were produced by rendering
// this same exchange through openai-harmony abd677f7's own renderer.
static void test_harmony_tool_history_golden(void) {
    const char *src =
        "[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\","
        "\"description\":\"Get weather\",\"parameters\":{\"type\":\"object\","
        "\"properties\":{\"city\":{\"type\":\"string\"}},"
        "\"required\":[\"city\"]}}}]";
    jv *tools = json_parse(src, strlen(src));
    assert(tools != NULL);
    const chat_msg msgs[] = {
        { .role = "user", .content = "Weather?" },
        { .role = "assistant", .content = "{\"city\":\"Oslo\"}",
          .name = "get_weather" },
        { .role = "tool", .content = "{\"temp\":4}",
          .name = "get_weather" },
    };
    char out[4096];
    render_messages_with_tools(TMPL_HARMONY, msgs, 3, true, THINK_DEFAULT,
                               tools, out, sizeof(out));
    assert(strstr(out,
        "<|start|>user<|message|>Weather?<|end|>"
        "<|start|>assistant to=functions.get_weather<|channel|>commentary "
        "<|constrain|>json<|message|>{\"city\":\"Oslo\"}<|call|>"
        "<|start|>functions.get_weather to=assistant<|channel|>commentary"
        "<|message|>{\"temp\":4}<|end|><|start|>assistant") != NULL);
    jv_free(tools);
}

// The analysis body may legally CONTAIN the handoff text. Inside a free-text
// region the constraint masks the control tokens, so the model types the
// handoff byte-by-byte as ordinary characters — that is the normal path, not an
// exotic one. schema.c bounds the analysis region with the 36-byte
// "<|end|><|start|>assistant<|channel|>", so a bare "<|end|><|start|>assistant"
// NOT followed by "<|channel|>" is accepted as body text. The mapper must use
// the same 36-byte sentinel; with the shorter 25-byte one it stops at the first
// occurrence, the following final/call_prefix compares both miss, and a
// schema-legal document maps to -1.
static void test_harmony_analysis_body_may_contain_the_handoff_text(void) {
    const char *src =
        "[{\"type\":\"function\",\"function\":{\"name\":\"add\","
        "\"description\":\"Add\",\"parameters\":{\"type\":\"object\","
        "\"properties\":{\"a\":{\"type\":\"integer\"},"
        "\"b\":{\"type\":\"integer\"}},\"required\":[\"a\",\"b\"]}}}]";
    jv *tools = json_parse(src, strlen(src));
    assert(tools != NULL);
    tool_envelope e;
    char err[192];
    assert(tool_envelope_build(tools, NULL, NULL, &e, err, sizeof(err)) == 1);
    e.harmony = true;
    e.tools = tools;

    sbuf reason = {0}, content = {0}, calls = {0};
    const char *doc =
        "<|channel|>analysis<|message|>quoting "
        "<|end|><|start|>assistantX"
        "<|end|><|start|>assistant<|channel|>commentary to=functions.add"
        "<|constrain|>json<|message|>{\"a\":1,\"b\":2}";
    assert(tool_envelope_map_channels(&e, doc, strlen(doc), &reason,
                                      &content, &calls) == 1);
    assert(!strcmp(reason.s, "quoting <|end|><|start|>assistantX"));
    assert(content.n == 0);
    assert(strstr(calls.s, "\"name\":\"add\"") != NULL);
    assert(strstr(calls.s, "{\\\"a\\\":1,\\\"b\\\":2}") != NULL);
    free(reason.s); free(content.s); free(calls.s);

    // the ordinary handoff still maps, so the wider sentinel did not simply
    // move the failure to the common case
    reason = (sbuf){0}; content = (sbuf){0}; calls = (sbuf){0};
    doc = "<|channel|>analysis<|message|>Need arithmetic."
          "<|end|><|start|>assistant<|channel|>final<|message|>3<|return|>";
    assert(tool_envelope_map_channels(&e, doc, strlen(doc), &reason,
                                      &content, &calls) == 0);
    assert(!strcmp(reason.s, "Need arithmetic."));
    assert(!strcmp(content.s, "3"));
    free(reason.s); free(content.s); free(calls.s);

    tool_envelope_free(&e);
    jv_free(tools);
}

static void test_harmony_reasoning_preamble_and_parallel_history(void) {
    const chat_msg msgs[] = {
        { .role = "user", .content = "Check both." },
        { .role = "assistant", .content = "Need both records.",
          .channel = "analysis" },
        { .role = "assistant", .content = "I’ll check both sources.",
          .channel = "commentary" },
        { .role = "assistant", .content = "{\"id\":1}", .name = "lookup" },
        { .role = "assistant", .content = "{\"id\":2}", .name = "lookup" },
    };
    char out[4096];
    render_messages(TMPL_HARMONY, msgs, 5, true, THINK_DEFAULT,
                    out, sizeof(out));
    assert(strstr(out,
        "<|start|>assistant<|channel|>analysis<|message|>Need both records."
        "<|end|><|start|>assistant<|channel|>commentary<|message|>"
        "I’ll check both sources.<|end|>"
        "<|start|>assistant to=functions.lookup<|channel|>commentary "
        "<|constrain|>json<|message|>{\"id\":1}<|call|>"
        "<|start|>assistant to=functions.lookup<|channel|>commentary "
        "<|constrain|>json<|message|>{\"id\":2}<|call|>") != NULL);
}

// THINK_ON primes the analysis channel in the prompt, THINK_OFF primes final
// — the same trick muse plays with ` to=self` / ` to=user`.
static void test_harmony_thinking_controls_the_primed_channel(void) {
    const chat_msg msgs[] = { { .role = "user", .content = "hi" } };
    char on[2048], off[2048], def[2048];
    render_messages(TMPL_HARMONY, msgs, 1, true, THINK_ON, on, sizeof(on));
    render_messages(TMPL_HARMONY, msgs, 1, true, THINK_OFF, off, sizeof(off));
    render_messages(TMPL_HARMONY, msgs, 1, true, THINK_DEFAULT, def, sizeof(def));
    assert(strstr(on,  "<|start|>assistant<|channel|>analysis<|message|>"));
    assert(strstr(off, "<|start|>assistant<|channel|>final<|message|>"));
    // DEFAULT primes analysis too: the decoded open marker is the bare word
    // "analysis", too common to hunt for in a free stream, so the splitter
    // starts already inside reasoning instead of searching for it
    assert(strstr(def, "<|start|>assistant<|channel|>analysis<|message|>"));
    assert(strstr(def, "<|channel|>final<|message|>") == NULL);
}

// The analysis channel rides the existing splitter: everything from the
// analysis header to the final header is reasoning, the rest is content.
static void test_harmony_split_hides_analysis_from_content(void) {
    think_split split;
    split_capture got = {0};
    think_init(&split, HARMONY_THINK_OPEN, HARMONY_THINK_CLOSE);
    // the DECODED stream: Harmony's control tokens detokenize to nothing, so
    // only the channel words survive (measured live on gpt-oss-20b)
    const char *generated =
        "analysisUser asks 2+2. Answer 4.assistantfinal4";
    for (size_t i = 0; i < strlen(generated); i++)
        think_feed(&split, generated + i, 1, capture_split, &got);
    think_finish(&split, capture_split, &got);
    assert(!strcmp(got.reason, "User asks 2+2. Answer 4."));
    assert(!strcmp(got.content, "4"));
    think_free(&split);
}

// With the prompt already inside the analysis channel (THINK_ON) the split
// starts in reasoning, exactly as muse's forced-think path does.
static void test_harmony_split_starts_inside_primed_analysis(void) {
    think_split split;
    split_capture got = {0};
    think_init_reasoning(&split, HARMONY_THINK_OPEN, HARMONY_THINK_CLOSE);
    const char *generated = "thinking out loudassistantfinaldone";
    for (size_t i = 0; i < strlen(generated); i++)
        think_feed(&split, generated + i, 1, capture_split, &got);
    think_finish(&split, capture_split, &got);
    assert(!strcmp(got.reason, "thinking out loud"));
    assert(!strcmp(got.content, "done"));
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

// The system prompt is folded into a user turn either way, and the framing
// differs -- but so does the TURN. This assertion used to read
// "[INST] SYS\n\nHI [/INST]" for mistral, which pinned two defects at once:
// the space before [/INST] that no Mistral template writes, and the fold into
// the first user turn, which only the v0.1/v0.2 form does. With one user turn
// first and last are the same turn, so what is left here is the spacing.
static void test_render_system_prompt(void) {
    const chat_msg msgs[] = { { "system", "SYS" }, { "user", "HI" } };
    char out[1024];

    render_messages(TMPL_MISTRAL, msgs, 2, true, THINK_DEFAULT, out, sizeof(out));
    assert(strcmp(out, "[INST] SYS\n\nHI[/INST]") == 0);

    render_messages(TMPL_LLAMA2, msgs, 2, true, THINK_DEFAULT, out, sizeof(out));
    assert(strcmp(out, "[INST] <<SYS>>\nSYS\n<</SYS>>\n\nHI [/INST]") == 0);
}

// This test used to assert `strcmp(mistral, llama2) == 0` under the heading
// "Without a system message the two are identical". That was a claim about the
// REFERENCES, and it was false against all four upstream Mistral templates:
// llama-2 writes ' [/INST]', v0.3 writes '[/INST]', and the two therefore
// cannot agree on any conversation at all. The invariant is now the opposite
// one, and it is the reason the families need separate renderers rather than a
// shared case with a system-block switch.
static void test_render_without_system(void) {
    const chat_msg msgs[] = { { "user", "HI" } };
    char mistral[512], llama2[512];
    render_messages(TMPL_MISTRAL, msgs, 1, true, THINK_DEFAULT, mistral, sizeof(mistral));
    render_messages(TMPL_LLAMA2, msgs, 1, true, THINK_DEFAULT, llama2, sizeof(llama2));
    assert(strcmp(mistral, "[INST] HI[/INST]") == 0);
    assert(strcmp(llama2, "[INST] HI [/INST]") == 0);
    assert(strcmp(mistral, llama2) != 0);
}

static void test_name_roundtrip(void) {
    static const char *const names[] = {
        "chatml", "llama2", "llama3", "zephyr", "gemma", "gemma4", "mistral",
        "mistral-v1", "mistral-nemo",
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
    // THINK_ON injects <|think|> at the top of the first SYSTEM turn, which
    // is where the model's own template puts it (read from the GGUF's
    // tokenizer.chat_template on gemma-4-E2B, 2026-08-12, not from a summary
    // of it). With no system message the template still opens a system turn:
    // its condition is `enable_thinking or tools or messages[0] is system`.
    // There is still no pre-seeded thought block at the generation prompt.
    render_messages(TMPL_GEMMA4, msgs, 1, true, THINK_ON, out, sizeof(out));
    assert(strcmp(out, "<|turn>system\n<|think|>\n<turn|>\n"
                       "<|turn>user\nHI<turn|>\n<|turn>model\n") == 0);
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

    // A system message already opens that turn: the marker goes INSIDE it,
    // above the system text, rather than creating a second system turn.
    const chat_msg with_sys[] = { { "system", "BE BRIEF" }, { "user", "HI" } };
    render_messages(TMPL_GEMMA4, with_sys, 2, true, THINK_ON, out, sizeof(out));
    assert(strcmp(out, "<|turn>system\n<|think|>\nBE BRIEF<turn|>\n"
                       "<|turn>user\nHI<turn|>\n<|turn>model\n") == 0);
    // ...and thinking off renders the same system turn without the marker
    render_messages(TMPL_GEMMA4, with_sys, 2, true, THINK_OFF, out, sizeof(out));
    assert(strcmp(out, "<|turn>system\nBE BRIEF<turn|>\n"
                       "<|turn>user\nHI<turn|>\n<|turn>model\n") == 0);
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

// A tool result is NOT a `tool` turn on ChatML. Every reference in the family
// converts it into a USER turn wrapping the payload in <tool_response>, and
// groups consecutive results under one turn:
//
//   {%- elif message.role == "tool" %}
//       {%- if loop.first or (messages[loop.index0 - 1].role != "tool") %}
//           {{- '<|im_start|>user' }}
//       {%- endif %}
//       {{- '\n<tool_response>\n' }}{{- content }}{{- '\n</tool_response>' }}
//       {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
//           {{- '<|im_end|>\n' }}
//       {%- endif %}
//
// -- Qwen/Qwen3-4B tokenizer_config.json; Qwen/Qwen2.5-7B-Instruct's is the
// same block keyed on `loop.index0 == 0`, and ornith-ai/Ornith-1.0-9B's on
// `loop.previtem`. Three independent references, one shape.
//
// Runner used to emit `<|im_start|>tool\n{...}<|im_end|>` -- a role header no
// ChatML checkpoint is trained on -- while already doing this conversion on
// its own ORNITH path. The goldens are the reference renders of these exact
// conversations, not runner's prior output.
static void test_chatml_tool_result_renders_as_a_user_tool_response(void) {
    char out[1024];

    const chat_msg one[] = {
        { "user", "Weather in Oslo?" },
        { "tool", "{\"temp_c\": -3}" },
    };
    const char *want_one =
        "<|im_start|>user\nWeather in Oslo?<|im_end|>\n"
        "<|im_start|>user\n<tool_response>\n{\"temp_c\": -3}\n"
        "</tool_response><|im_end|>\n"
        "<|im_start|>assistant\n";
    render_messages(TMPL_CHATML, one, 2, true, THINK_DEFAULT, out, sizeof(out));
    assert(strcmp(out, want_one) == 0);
    // the branch is shared: chatml-think must get it too, and its own
    // thinking control must still work on top of the rewritten turn
    render_messages(TMPL_CHATML_THINK, one, 2, true, THINK_DEFAULT,
                    out, sizeof(out));
    assert(strcmp(out, want_one) == 0);
    render_messages(TMPL_CHATML_THINK, one, 2, true, THINK_OFF,
                    out, sizeof(out));
    assert(strcmp(out, "<|im_start|>user\nWeather in Oslo?<|im_end|>\n"
                       "<|im_start|>user\n<tool_response>\n{\"temp_c\": -3}\n"
                       "</tool_response><|im_end|>\n"
                       "<|im_start|>assistant\n<think>\n\n</think>\n\n") == 0);

    // consecutive results share ONE user turn, separated by a bare newline
    const chat_msg two[] = {
        { "user", "Weather in Oslo?" },
        { "tool", "{\"temp_c\": -3}" },
        { "tool", "{\"temp_c\": 1}" },
        { "user", "thanks" },
    };
    render_messages(TMPL_CHATML, two, 4, true, THINK_DEFAULT, out, sizeof(out));
    assert(strcmp(out,
        "<|im_start|>user\nWeather in Oslo?<|im_end|>\n"
        "<|im_start|>user\n<tool_response>\n{\"temp_c\": -3}\n</tool_response>"
        "\n<tool_response>\n{\"temp_c\": 1}\n</tool_response><|im_end|>\n"
        "<|im_start|>user\nthanks<|im_end|>\n"
        "<|im_start|>assistant\n") == 0);

    // and no `tool` header survives anywhere
    assert(strstr(out, "<|im_start|>tool") == NULL);
}

// Muse's own template places tool metadata after the reasoning-strength line,
// derives valid recipient namespaces from dotted function names, and renders
// tool results as named tool turns.  Keep the whole turn byte-exact: moving
// any of these fragments changes the prompt the real model sees.
// The `parameters` object is SPACED (`, ` / `: `), not compact: muse.jinja:73
// renders it through jinja's `tojson`, whose default separators are those.
// This golden previously pinned runner's own compact output -- it was written
// from the implementation rather than the oracle, which is the exact
// circularity that let a family drift. Expected bytes here come from the
// reference render, not from jv_dump.
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
        "{\"name\": \"weather.get\", \"description\": \"Get weather\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}}}}\n"
        "{\"name\": \"math.add\", \"description\": \"Add values\", \"parameters\": {\"type\": \"object\", \"properties\": {\"a\": {\"type\": \"integer\"}}}}\n\n"
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
    test_detect_mistral_variants(&t);
    test_render_mistral_v1();
    test_render_mistral_v3();
    test_render_mistral_nemo();
    test_render_mistral_keeps_system_on_a_prefill();
    test_detect_zephyr_vs_phi3(&t);
    test_phi3_without_generation_prompt_ends_with_eos();
    test_detect_by_marker(&t);
    test_detect_and_render_ornith(&t);
    test_ornith_groups_consecutive_tool_responses();
    test_ornith_thinking_off_closes_the_thought_block();
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
    test_chatml_tool_result_renders_as_a_user_tool_response();
    test_muse_tools_and_result_golden();
    test_muse_tool_result_id_resolves_prior_name();
    test_muse_parallel_tool_history_has_native_turn_boundaries();
    test_muse_tool_history_skips_bad_calls_without_leading_boundary();
    test_muse_user_payload_strip_removes_only_recipient_header();
    test_detect_harmony(&t);
    test_harmony_render_golden();
    test_harmony_system_turn_carries_the_knowledge_cutoff();
    test_harmony_render_without_system();
    test_harmony_tools_add_the_commentary_routing_line();
    test_harmony_tool_definitions_golden();
    test_harmony_tools_without_instructions_still_get_a_developer_turn();
    test_harmony_folds_every_system_message_into_one_developer_turn();
    test_harmony_schema_scalar_branches();
    test_harmony_schema_enum_and_array_branches();
    test_harmony_schema_type_array_and_nullable();
    test_harmony_schema_comments_and_defaults();
    test_harmony_schema_nested_objects();
    test_harmony_schema_multiline_property_text_mirrors_reference();
    test_harmony_schema_oneof_toplevel();
    test_harmony_schema_oneof_property();
    test_harmony_tool_history_golden();
    test_harmony_analysis_body_may_contain_the_handoff_text();
    test_harmony_reasoning_preamble_and_parallel_history();
    test_harmony_thinking_controls_the_primed_channel();
    test_harmony_split_hides_analysis_from_content();
    test_harmony_split_starts_inside_primed_analysis();

    tokenizer_free(&t);
    gguf_close(&g);
    puts("template tests ok");
    return 0;
}
