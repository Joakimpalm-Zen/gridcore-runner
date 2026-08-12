// Chat template detection and rendering, plus the model-facing chat
// conventions that ride on top of it: thinking-tag splitting and the
// tool-call syntax (declaration rendering and response parsing).
#include "template.h"
#include "json.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int template_detect(const char *meta_tmpl, tokenizer *tok) {
    if (meta_tmpl) {
        // apertus first: its vocabulary inherits Mistral's [INST]/[/INST]
        // tokens, so the [INST] branch below would otherwise claim it
        if (strstr(meta_tmpl, "<|assistant_start|>")) return TMPL_APERTUS;
        if (strstr(meta_tmpl, "<function=example_function_name>") &&
            strstr(meta_tmpl, "<think>"))
            return TMPL_ORNITH;
        // muse-glimmer: the atem tool-call syntax and the <|start|>role
        // <|message|> framing appear in no other family's template
        if (strstr(meta_tmpl, "atem:function_calls") ||
            (strstr(meta_tmpl, "<|start|>") && strstr(meta_tmpl, "<|eot|>")))
            return TMPL_MUSE;
        if (strstr(meta_tmpl, "<|start_of_role|>")) return TMPL_GRANITE;
        // Qwen3 and relatives: ChatML whose own template carries a
        // <think> branch. Detected from the model's template rather than
        // a name list, so it follows the checkpoint and not a guess.
        if (strstr(meta_tmpl, "<|im_start|>"))
            return strstr(meta_tmpl, "<think>") ? TMPL_CHATML_THINK
                                                : TMPL_CHATML;
        if (strstr(meta_tmpl, "<|start_header_id|>")) return TMPL_LLAMA3;
        if (strstr(meta_tmpl, "<|user|>"))
            return strstr(meta_tmpl, "<|end|>") ? TMPL_PHI3 : TMPL_ZEPHYR;
        if (strstr(meta_tmpl, "<|turn>"))             return TMPL_GEMMA4;
        if (strstr(meta_tmpl, "<start_of_turn>"))     return TMPL_GEMMA;
        if (strstr(meta_tmpl, "[INST]"))
            return strstr(meta_tmpl, "<<SYS>>") ? TMPL_LLAMA2 : TMPL_MISTRAL;
    }
    if (tok_find(tok, "<|assistant_start|>") >= 0) return TMPL_APERTUS;
    if (tok_find(tok, "<|eot|>") >= 0 && tok_find(tok, "<|message|>") >= 0)
        return TMPL_MUSE;
    if (tok_find(tok, "<|start_of_role|>") >= 0)   return TMPL_GRANITE;
    if (tok_find(tok, "<|im_start|>") >= 0)        return TMPL_CHATML;
    if (tok_find(tok, "<|start_header_id|>") >= 0) return TMPL_LLAMA3;
    if (tok_find(tok, "<|user|>") >= 0)
        return tok_find(tok, "<|end|>") >= 0 ? TMPL_PHI3 : TMPL_ZEPHYR;
    if (tok_find(tok, "<|turn>") >= 0)             return TMPL_GEMMA4;
    if (tok_find(tok, "<start_of_turn>") >= 0)     return TMPL_GEMMA;
    return TMPL_LLAMA2;
}

int template_from_name(const char *name) {
    if (!strcmp(name, "chatml")) return TMPL_CHATML;
    if (!strcmp(name, "chatml-think")) return TMPL_CHATML_THINK;
    if (!strcmp(name, "llama2")) return TMPL_LLAMA2;
    if (!strcmp(name, "llama3")) return TMPL_LLAMA3;
    if (!strcmp(name, "zephyr")) return TMPL_ZEPHYR;
    if (!strcmp(name, "gemma"))  return TMPL_GEMMA;
    if (!strcmp(name, "gemma4")) return TMPL_GEMMA4;
    if (!strcmp(name, "mistral")) return TMPL_MISTRAL;
    if (!strcmp(name, "phi3"))    return TMPL_PHI3;
    if (!strcmp(name, "apertus")) return TMPL_APERTUS;
    if (!strcmp(name, "ornith")) return TMPL_ORNITH;
    if (!strcmp(name, "muse"))   return TMPL_MUSE;
    if (!strcmp(name, "granite")) return TMPL_GRANITE;
    if (!strcmp(name, "raw"))    return TMPL_RAW;
    return -1;
}

const char *template_name(int t) {
    switch (t) {
        case TMPL_CHATML: return "chatml";  case TMPL_LLAMA2: return "llama2";
        case TMPL_CHATML_THINK: return "chatml-think";
        case TMPL_LLAMA3: return "llama3";  case TMPL_ZEPHYR: return "zephyr";
        case TMPL_GEMMA:  return "gemma";
        case TMPL_GEMMA4: return "gemma4";
        case TMPL_MISTRAL: return "mistral";
        case TMPL_PHI3:    return "phi3";
        case TMPL_APERTUS: return "apertus";
        case TMPL_ORNITH: return "ornith";
        case TMPL_MUSE:   return "muse";
        case TMPL_GRANITE: return "granite";
        default: return "raw";
    }
}

static size_t emit(char *out, size_t cap, size_t off, const char *fmt,
                   const char *a, const char *b) {
    if (off >= cap) return off;
    int n = snprintf(out + off, cap - off, fmt, a, b);
    return n > 0 ? off + (size_t)n : off;
}

// Per-request opt-in for the thinking shape. Accepted at the top level and
// inside `chat_template_kwargs`, which is where llama.cpp's server takes
// template variables — a client that already speaks to llama.cpp should not
// have to learn a second spelling to get the same behaviour.
//
// Returns THINK_DEFAULT when the field is absent, so a silent request renders
// what a reference-following engine would render for that model family. It
// does NOT default to false: gemma-4's reference defaults thinking off and
// Qwen3's defaults it on, so collapsing "unspecified" onto either one would
// silently misrender the other family.
int req_thinking_mode(struct jv *req) {
    if (!req) return THINK_DEFAULT;
    jv *kw = jv_get((jv *)req, "chat_template_kwargs");
    jv *v  = kw ? jv_get(kw, "enable_thinking") : NULL;
    if (!v) v = jv_get((jv *)req, "enable_thinking");
    // absent means DEFAULT, not false: the reference default differs per
    // family, so a silent request must not be forced onto one of them
    if (!v) return THINK_DEFAULT;
    return jv_bool(v, false) ? THINK_ON : THINK_OFF;
}

static void muse_json_string(sbuf *b, const char *s) {
    sb_lit(b, "\"");
    sb_esc(b, s ? s : "", strlen(s ? s : ""));
    sb_lit(b, "\"");
}

static jv *muse_tool_fn(const jv *tool) {
    jv *fn = jv_get((jv *)tool, "function");
    return fn ? fn : (jv *)tool;
}

static bool muse_namespace_seen(const jv *tools, int before,
                                const char *name, size_t nsn) {
    for (int i = 0; i < before; i++) {
        const char *prior = jv_str(jv_get(muse_tool_fn(tools->items[i]),
                                         "name"), "");
        const char *dot = strchr(prior, '.');
        size_t n = dot ? (size_t)(dot - prior) : strlen(prior);
        if (n == nsn && !memcmp(prior, name, n)) return true;
    }
    return false;
}

static void muse_render_tool_defs(const jv *tools, sbuf *b) {
    if (!tools || tools->type != J_ARR || tools->n == 0) return;
    sb_lit(b,
        "In this environment you have access to a set of tools you can use to answer the user's question.\n\n"
        "You can invoke a function by writing a \"<atem:function_calls>\" block like the following:\n"
        "<atem:function_calls>\n<atem:invoke name=\"$FUNCTION_NAME\">\n"
        "<atem:parameter name=\"$PARAMETER_NAME\">$PARAMETER_VALUE</atem:parameter>\n"
        "...\n</atem:invoke>\n</atem:function_calls>\n\n"
        "String and scalar parameters should be specified as is, while lists and objects should use JSON format. Note that spaces for string values are not stripped. The output is not expected to be valid XML and is parsed with regular expressions.\n"
        "Here are the functions available in JSONSchema format:\n// Tool metadata\n");
    for (int i = 0; i < tools->n; i++) {
        jv *fn = muse_tool_fn(tools->items[i]);
        const char *name = jv_str(jv_get(fn, "name"), "");
        const char *dot = strchr(name, '.');
        size_t nsn = dot ? (size_t)(dot - name) : strlen(name);
        if (muse_namespace_seen(tools, i, name, nsn)) continue;
        sb_lit(b, "{\"name\": \""); sb_esc(b, name, nsn);
        sb_lit(b, "\", \"description\": \"\"}\n");
    }
    sb_lit(b, "// Function schemas");
    for (int i = 0; i < tools->n; i++) {
        jv *fn = muse_tool_fn(tools->items[i]);
        sb_lit(b, "\n{\"name\": ");
        muse_json_string(b, jv_str(jv_get(fn, "name"), ""));
        sb_lit(b, ", \"description\": ");
        muse_json_string(b, jv_str(jv_get(fn, "description"), ""));
        sb_lit(b, ", \"parameters\": ");
        jv *params = jv_get(fn, "parameters");
        if (params) jv_dump(params, b); else sb_lit(b, "{}");
        sb_lit(b, "}");
    }
    sb_lit(b,
        "\n\nHere's an example of how to call a function in the tool set:\n"
        "(If the tool namespace is not specified, invoke the function directly as `example_function_name` rather than `example_tool_name.example_function_name`)\n\n"
        "to=example_tool_name.example_function_name\n\n"
        "<atem:function_calls>\n<atem:invoke name=\"example_tool_name.example_function_name\">\n"
        "<atem:parameter name=\"example_parameter_1\">value_1</atem:parameter>\n"
        "<atem:parameter name=\"example_parameter_2\">This is the value for the second parameter\nthat can span\n\"multiple\" lines\n</atem:parameter>\n"
        "</atem:invoke>\n</atem:function_calls>");
}

size_t render_messages_with_tools(int tmpl, const chat_msg *msgs, int n_msgs,
                                  bool add_assistant, int thinking,
                                  const jv *tools,
                                  char *out, size_t cap) {
    size_t off = 0;
    out[0] = 0;
    switch (tmpl) {
    case TMPL_ORNITH:
        for (int i = 0; i < n_msgs; i++) {
            bool tool_response =
                !strcmp(msgs[i].role, "user") &&
                !strncmp(msgs[i].content, "<tool_response>",
                         strlen("<tool_response>"));
            bool prev_tool_response = i > 0 &&
                !strcmp(msgs[i - 1].role, "user") &&
                !strncmp(msgs[i - 1].content, "<tool_response>",
                         strlen("<tool_response>"));
            bool next_tool_response = i + 1 < n_msgs &&
                !strcmp(msgs[i + 1].role, "user") &&
                !strncmp(msgs[i + 1].content, "<tool_response>",
                         strlen("<tool_response>"));
            if (tool_response) {
                if (!prev_tool_response)
                    off = emit(out, cap, off, "<|im_start|>user\n", NULL, NULL);
                else
                    off = emit(out, cap, off, "\n", NULL, NULL);
                off = emit(out, cap, off, "%s", msgs[i].content, NULL);
                if (!next_tool_response)
                    off = emit(out, cap, off, "<|im_end|>\n", NULL, NULL);
            } else {
                off = emit(out, cap, off, "<|im_start|>%s\n%s<|im_end|>\n",
                           msgs[i].role, msgs[i].content);
            }
        }
        if (add_assistant)
            off = emit(out, cap, off, "<|im_start|>assistant\n<think>\n", NULL, NULL);
        break;
    case TMPL_CHATML:
    case TMPL_CHATML_THINK:
        for (int i = 0; i < n_msgs; i++)
            off = emit(out, cap, off, "<|im_start|>%s\n%s<|im_end|>\n",
                       msgs[i].role, msgs[i].content);
        if (add_assistant) {
            off = emit(out, cap, off, "<|im_start|>assistant\n", NULL, NULL);
            // Qwen3's enable_thinking=false branch, verbatim from
            // Qwen/Qwen3-* tokenizer_config.json: an already-closed thought
            // block, which is how the family is asked not to reason. Its
            // reference defaults thinking ON, so this is emitted only when a
            // caller explicitly turns it off -- never for THINK_DEFAULT.
            if (tmpl == TMPL_CHATML_THINK && thinking == THINK_OFF)
                off = emit(out, cap, off, "<think>\n\n</think>\n\n",
                           NULL, NULL);
        }
        break;
    case TMPL_LLAMA3:
        for (int i = 0; i < n_msgs; i++)
            off = emit(out, cap, off,
                       "<|start_header_id|>%s<|end_header_id|>\n\n%s<|eot_id|>",
                       msgs[i].role, msgs[i].content);
        if (add_assistant)
            off = emit(out, cap, off,
                       "<|start_header_id|>assistant<|end_header_id|>\n\n", NULL, NULL);
        break;
    case TMPL_ZEPHYR:
        for (int i = 0; i < n_msgs; i++)
            off = emit(out, cap, off, "<|%s|>\n%s</s>\n",
                       msgs[i].role, msgs[i].content);
        if (add_assistant)
            off = emit(out, cap, off, "<|assistant|>\n", NULL, NULL);
        break;
    case TMPL_PHI3:
        // same <|role|> framing as zephyr, but turns end with <|end|>
        for (int i = 0; i < n_msgs; i++)
            off = emit(out, cap, off, "<|%s|>\n%s<|end|>\n",
                       msgs[i].role, msgs[i].content);
        if (add_assistant)
            off = emit(out, cap, off, "<|assistant|>\n", NULL, NULL);
        break;
    case TMPL_APERTUS: {
        // apertus (reference: swiss-ai/Apertus-8B-Instruct-2509
        // chat_template.jinja): <|role_start|>CONTENT<|role_end|> per turn,
        // with a native system role that must come first.
        //
        // Between the system turn and the first real turn the reference always
        // emits a developer block carrying two switches. Runner supports
        // neither thinking mode nor Apertus-native tool declarations here, so
        // it emits the both-disabled constant -- which is exactly what the
        // reference produces for a plain chat, not an approximation of it.
        //
        // Two deliberate omissions, both matching decisions already made for
        // other families: when there is no system message the reference
        // substitutes a default one containing strftime_now('%Y-%m-%d'), and
        // runner does not embed a live date in a prompt (as with Llama-3.2's
        // "Cutting Knowledge Date" header); and BOS is added by the tokenizer,
        // not spelled into the template.
        const char *sys = NULL;
        int first = 0;
        if (n_msgs > 0 && !strcmp(msgs[0].role, "system")) {
            sys = msgs[0].content;
            first = 1;
        }
        if (sys)
            off = emit(out, cap, off, "<|system_start|>%s<|system_end|>", sys, NULL);
        off = emit(out, cap, off,
                   "<|developer_start|>Deliberation: disabled\n"
                   "Tool Capabilities: disabled<|developer_end|>", NULL, NULL);
        for (int i = first; i < n_msgs; i++) {
            off = emit(out, cap, off, "<|%s_start|>%s",
                       msgs[i].role, msgs[i].content);
            off = emit(out, cap, off, "<|%s_end|>", msgs[i].role, NULL);
        }
        if (add_assistant)
            off = emit(out, cap, off, "<|assistant_start|>", NULL, NULL);
        break;
    }
    case TMPL_MUSE: {
        // muse-glimmer (reference: the model's OWN tokenizer.chat_template,
        // read from the official meta-models GGUF, not a summary):
        // <|start|>ROLE<|message|>CONTENT<|eot|> per turn; assistant turns
        // carry a recipient — ` to=user` for answers, ` to=self` for
        // reasoning turns, which is what the think_open/close pair splits.
        // When no system message is present the reference injects a default
        // one whose constant parts are reproduced here; its current-date
        // line is conditional in the template (`current_date is defined`)
        // and omitted here — a live date would also defeat the prefix cache.
        // Structured tools are rendered below with the model's native atem
        // declaration macro; the legacy wrapper passes NULL and stays plain.
        sbuf muse_tail = {0};
        sb_lit(&muse_tail, "\n\nReasoning strength: high.");
        if (tools && tools->type == J_ARR && tools->n) {
            sb_lit(&muse_tail, "\n\n");
            muse_render_tool_defs(tools, &muse_tail);
        }
        sb_lit(&muse_tail, "\n\n# Valid recipients: \"self\"");
        if (tools && tools->type == J_ARR) {
            for (int i = 0; i < tools->n; i++) {
                jv *fn = muse_tool_fn(tools->items[i]);
                const char *name = jv_str(jv_get(fn, "name"), "");
                const char *dot = strchr(name, '.');
                size_t nsn = dot ? (size_t)(dot - name) : strlen(name);
                if (muse_namespace_seen(tools, i, name, nsn)) continue;
                sb_lit(&muse_tail, ", \""); sb_put(&muse_tail, name, nsn);
                sb_lit(&muse_tail, ".*\"");
            }
        }
        sb_lit(&muse_tail, ", \"user\".<|eot|>");
        bool has_system = false;
        for (int i = 0; i < n_msgs; i++)
            if (!strcmp(msgs[i].role, "system")) has_system = true;
        if (!has_system) {
            off = emit(out, cap, off,
                       "<|start|>system<|message|>You are a helpful AI "
                       "assistant.\nKnowledge cutoff: 2026-01-04.", NULL, NULL);
            off = emit(out, cap, off, "%s", muse_tail.s, NULL);
        }
        for (int i = 0; i < n_msgs; i++) {
            const chat_msg *mm = &msgs[i];
            if (!strcmp(mm->role, "system")) {
                off = emit(out, cap, off, "<|start|>system<|message|>%s",
                           mm->content, NULL);
                off = emit(out, cap, off, "%s", muse_tail.s, NULL);
            } else if (!strcmp(mm->role, "assistant")) {
                off = emit(out, cap, off, "<|start|>assistant to=%s<|message|>",
                           mm->name ? mm->name : "user", NULL);
                off = emit(out, cap, off, "%s<|eot|>", mm->content, NULL);
            } else if (!strcmp(mm->role, "tool") && mm->name) {
                off = emit(out, cap, off, "<|start|>tool %s<|message|>",
                           mm->name, NULL);
                off = emit(out, cap, off, "<tool_output name=\"%s\">\n",
                           mm->name, NULL);
                off = emit(out, cap, off, "%s\n</tool_output><|eot|>",
                           mm->content, NULL);
            } else {
                // user and anything else (a tool result arrives as its own
                // named role in the reference) render as a plain named turn
                off = emit(out, cap, off, "<|start|>%s<|message|>",
                           mm->role, NULL);
                off = emit(out, cap, off, "%s<|eot|>", mm->content, NULL);
            }
        }
        // The reference generation prompt is the bare header: the model then
        // chooses its recipient, opening ` to=self` when it wants a reasoning
        // turn. An explicit THINK_ON starts that self-addressed turn in the
        // prompt, which makes reasoning-before-tool deterministic. With tools,
        // THINK_OFF still leaves the recipient to the constrained header (it
        // cannot pin `user`, because a required tool must remain reachable).
        if (add_assistant) {
            bool have_tools = tools && tools->type == J_ARR && tools->n;
            const char *head = thinking == THINK_ON
                         ? "<|start|>assistant to=self<|message|>"
                         : thinking == THINK_OFF && !have_tools
                         ? "<|start|>assistant to=user<|message|>"
                         : "<|start|>assistant";
            off = emit(out, cap, off, "%s", head, NULL);
        }
        free(muse_tail.s);
        break;
    }
    case TMPL_GRANITE:
        // granite 4.1 (reference: the model's OWN tokenizer.chat_template):
        // <|start_of_role|>ROLE<|end_of_role|>CONTENT<|end_of_text|>\n per
        // turn, assistant included. Tool declarations and the Hermes-style
        // <tool_call> JSON blocks the template also defines are not rendered
        // here — granite tool calling is unimplemented, not approximated.
        for (int i = 0; i < n_msgs; i++) {
            off = emit(out, cap, off, "<|start_of_role|>%s<|end_of_role|>",
                       msgs[i].role, NULL);
            off = emit(out, cap, off, "%s<|end_of_text|>\n",
                       msgs[i].content, NULL);
        }
        if (add_assistant)
            off = emit(out, cap, off, "<|start_of_role|>assistant<|end_of_role|>",
                       NULL, NULL);
        break;
    case TMPL_GEMMA4:
        // gemma4 (reference: llama.cpp models/templates/google-gemma-4-31B-it
        // .jinja): <|turn>role\n CONTENT <turn|>\n per turn, a native system
        // role (unlike gemma1-3), assistant role named "model"
        for (int i = 0; i < n_msgs; i++) {
            const char *role = !strcmp(msgs[i].role, "assistant") ? "model"
                                                                  : msgs[i].role;
            off = emit(out, cap, off, "<|turn>%s\n", role, NULL);
            off = emit(out, cap, off, "%s<turn|>\n", msgs[i].content, NULL);
        }
        // Generation prompt, read from the MODEL'S OWN chat template
        // (gguf tokenizer.chat_template on gemma-4-E2B), not from a summary
        // of llama.cpp's copy:
        //
        //   {%- if add_generation_prompt -%}
        //     {%- if ns.prev_message_type != 'tool_response'
        //            and ns.prev_message_type != 'tool_call' -%}
        //         {{- '<|turn>model\n' -}}
        //     {%- elif ns.prev_message_type == 'tool_response'
        //              and enable_thinking -%}
        //         {{- '<|channel>thought\n' -}}
        //     {%- endif -%}
        //   {%- endif -%}
        //
        // There is NO empty-thought-block pre-seed here, in either thinking
        // mode. On 2026-08-08 this code grew one — `<|channel>thought\n`
        // immediately closed by `<channel|>` — from a web summary of the
        // llama.cpp template rather than the model's own. That construct
        // exists in the real template only when re-rendering a PRIOR
        // assistant message that actually contained thinking text; handing it
        // to the model as an empty block at generation time is a state it was
        // never trained on. Measured cost: gemma-4-E2B's planning score fell
        // 0.575 -> 0.300 and it emitted raw reasoning prose as its visible
        // answer. The original unconditional `<|turn>model\n` was right.
        //
        // Thinking is NOT selected here for this family. The template sets
        // `enable_thinking | default(false)` and, when true, injects
        // `<|think|>\n` at the top of the FIRST SYSTEM TURN — a different
        // place entirely. That path is not implemented, so THINK_ON is
        // accepted and ignored rather than approximated; see the note in
        // template.h.
        if (add_assistant) {
            bool after_tool = n_msgs > 0 &&
                              !strcmp(msgs[n_msgs - 1].role, "tool");
            if (!after_tool)
                off = emit(out, cap, off, "<|turn>model\n", NULL, NULL);
            else if (thinking == THINK_ON)
                off = emit(out, cap, off, "<|channel>thought\n", NULL, NULL);
            // prev was a tool response and thinking is off: emit nothing,
            // which is what the template does and what the unconditional
            // header used to get wrong.
        }
        break;
    case TMPL_GEMMA: {
        // gemma has no system role: fold a system message into the first
        // user turn; the assistant role is named "model"
        const char *gsys = NULL;
        for (int i = 0; i < n_msgs; i++) {
            const chat_msg *m = &msgs[i];
            if (!strcmp(m->role, "system")) { gsys = m->content; continue; }
            const char *role = !strcmp(m->role, "assistant") ? "model" : m->role;
            if (gsys && !strcmp(m->role, "user")) {
                off = emit(out, cap, off, "<start_of_turn>%s\n%s\n\n", role, gsys);
                off = emit(out, cap, off, "%s<end_of_turn>\n", m->content, NULL);
                gsys = NULL;
            } else {
                off = emit(out, cap, off, "<start_of_turn>%s\n", role, NULL);
                off = emit(out, cap, off, "%s<end_of_turn>\n", m->content, NULL);
            }
        }
        if (add_assistant)
            off = emit(out, cap, off, "<start_of_turn>model\n", NULL, NULL);
        break;
    }
    case TMPL_LLAMA2:
    case TMPL_MISTRAL: {
        // fold an initial system message into the first user turn
        const char *sys = NULL;
        for (int i = 0; i < n_msgs; i++) {
            const chat_msg *m = &msgs[i];
            if (!strcmp(m->role, "system")) { sys = m->content; continue; }
            if (!strcmp(m->role, "user")) {
                if (sys) {
                    // mistral has no <<SYS>> block; its template accepts only
                    // user and assistant, so the system text leads the turn
                    off = tmpl == TMPL_MISTRAL
                        ? emit(out, cap, off, "[INST] %s\n\n", sys, NULL)
                        : emit(out, cap, off, "[INST] <<SYS>>\n%s\n<</SYS>>\n\n", sys, NULL);
                    off = emit(out, cap, off, "%s [/INST]", m->content, NULL);
                    sys = NULL;
                } else {
                    off = emit(out, cap, off, "[INST] %s [/INST]", m->content, NULL);
                }
            } else { // assistant
                off = emit(out, cap, off, " %s </s>", m->content, NULL);
            }
        }
        break;
    }
    default: // TMPL_RAW: concatenate contents
        for (int i = 0; i < n_msgs; i++)
            off = emit(out, cap, off, "%s", msgs[i].content, NULL);
        break;
    }
    return off;
}

size_t render_messages(int tmpl, const chat_msg *msgs, int n_msgs,
                       bool add_assistant, int thinking,
                       char *out, size_t cap) {
    return render_messages_with_tools(tmpl, msgs, n_msgs, add_assistant,
                                      thinking, NULL, out, cap);
}

// ------------------------------------------------- thinking-tag splitter

// Some models interleave thinking blocks with plain text anywhere in a response
// (thought . answer . thought . ...), so the splitter scans the whole stream:
// bytes between open and close tags are reasoning, the rest is content. The
// last strlen(tag)-1 bytes are held back while scanning so a tag split across
// chunk boundaries still resolves.

enum { TS_CONTENT, TS_WS, TS_THINK };

void think_init(think_split *t, const char *open, const char *close) {
    memset(t, 0, sizeof(*t));
    t->open  = open;
    t->close = close;
}

void think_init_reasoning(think_split *t, const char *open, const char *close) {
    think_init(t, open, close);
    if (open) t->state = TS_THINK;
}

void think_free(think_split *t) {
    free(t->buf);
    t->buf = NULL;
    t->n = t->cap = 0;
}

static void ts_keep(think_split *t, const char *b, int n) {
    // b always points into t->buf and n <= what feed() already grew cap to,
    // so no realloc may happen here — it would free the memory b points into
    memmove(t->buf, b, n);
    t->n = n;
}

static const char *ts_find(const char *p, int len, const char *tag, int tl) {
    for (int i = 0; i + tl <= len; i++)
        if (memcmp(p + i, tag, tl) == 0) return p + i;
    return NULL;
}

int think_feed(think_split *t, const char *bytes, int n, think_cb cb, void *ud) {
    if (!t->open) return n > 0 ? cb(ud, 0, bytes, n) : 0;

    if (t->n + n > t->cap) {
        int ncap = t->n + n + 64;
        // realloc into a temp: on failure the old buffer is still valid and must
        // not be leaked or written through a NULL pointer. Abort cleanly (a
        // nonzero return stops the generation stream, like the g->dead path).
        char *nb = realloc(t->buf, (size_t)ncap);
        if (!nb) return 1;
        t->buf = nb;
        t->cap = ncap;
    }
    memcpy(t->buf + t->n, bytes, n);
    t->n += n;
    const char *p = t->buf;
    int len = t->n;
    int ol = (int)strlen(t->open), cl = (int)strlen(t->close);

    for (;;) {
        if (t->state == TS_WS) { // drop whitespace after an open tag
            while (len > 0 && (*p == ' ' || *p == '\n' || *p == '\t' || *p == '\r')) {
                p++; len--;
            }
            if (len == 0) { t->n = 0; return 0; }
            t->state = TS_THINK;
        }
        bool     think = t->state == TS_THINK;
        const char *tag = think ? t->close : t->open;
        int          tl = think ? cl : ol;
        const char   *q = ts_find(p, len, tag, tl);
        if (!q) {
            int hold = tl - 1 < len ? tl - 1 : len;   // possible partial tag
            int rc = len - hold > 0 ? cb(ud, think, p, len - hold) : 0;
            ts_keep(t, p + (len - hold), hold);
            return rc;
        }
        int rc = q > p ? cb(ud, think, p, (int)(q - p)) : 0;
        if (rc) { ts_keep(t, q + tl, len - (int)(q - p) - tl); return rc; }
        len -= (int)(q - p) + tl;
        p = q + tl;
        t->state = think ? TS_CONTENT : TS_WS;
    }
}

int think_finish(think_split *t, think_cb cb, void *ud) {
    if (!t->open || t->n == 0) { t->n = 0; return 0; }
    // a held partial tag never completed: it is literal text of whichever
    // section we were in (whitespace after an open tag is dropped)
    int rc = t->state == TS_WS ? 0
           : cb(ud, t->state == TS_THINK, t->buf, t->n);
    t->n = 0;
    return rc;
}

// ------------------------------------------------- tool-call convention

// parse gemma4 tool-call blocks — <|tool_call>call:NAME{json}<tool_call|> —
// out of the content, appending OpenAI tool_calls items to tc. Returns the
// number of calls; content is compacted in place.
// OpenAI "tools" declarations rendered as a system turn, teaching the model
// the call syntax tool_calls_parse reads back. The trained template's native
// declaration macros are more elaborate; this works in practice.
void tools_render(const jv *tools, sbuf *out) {
    if (!tools || tools->type != J_ARR || tools->n == 0) return;
    sb_lit(out, "You have these tools available. To call one, reply with "
                "exactly <|tool_call>call:NAME{json arguments}<tool_call|> "
                "and nothing else. Tools:\n");
    jv_dump(tools, out);
}

void tools_render_for(int tmpl, const jv *tools, sbuf *out) {
    if (tmpl != TMPL_ORNITH) {
        tools_render(tools, out);
        return;
    }
    if (!tools || tools->type != J_ARR || tools->n == 0) return;
    sb_lit(out, "# Tools\n\nYou have access to the following functions:\n\n<tools>");
    for (int i = 0; i < tools->n; i++) {
        sb_lit(out, "\n");
        jv_dump(tools->items[i], out);
    }
    sb_lit(out,
        "\n</tools>\n\nIf you choose to call a function ONLY reply in the "
        "following format with NO suffix:\n\n<tool_call>\n"
        "<function=example_function_name>\n"
        "<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
        "<parameter=example_parameter_2>\nvalue_2\n</parameter>\n"
        "</function>\n</tool_call>\n\n<IMPORTANT>\n"
        "Function calls MUST use a function block nested inside tool_call. "
        "Required parameters MUST be specified. Reasoning may precede the "
        "tool call, but no text may follow it.\n</IMPORTANT>");
}

const char *tool_result_name(const jv *messages, int message_index) {
    if (!messages || messages->type != J_ARR || message_index < 0 ||
        message_index >= messages->n) return NULL;
    jv *result = messages->items[message_index];
    const char *name = jv_str(jv_get(result, "name"), NULL);
    if (name && name[0]) return name;
    const char *id = jv_str(jv_get(result, "tool_call_id"), NULL);
    if (!id) return NULL;
    for (int i = 0; i < message_index; i++) {
        jv *calls = jv_get(messages->items[i], "tool_calls");
        if (!calls || calls->type != J_ARR) continue;
        for (int k = 0; k < calls->n; k++) {
            const char *candidate = jv_str(jv_get(calls->items[k], "id"), NULL);
            if (!candidate || strcmp(candidate, id)) continue;
            jv *fn = jv_get(calls->items[k], "function");
            return jv_str(jv_get(fn, "name"), NULL);
        }
    }
    return id[0] ? id : NULL;
}

void tool_history_render_for(int tmpl, const jv *calls, sbuf *out) {
    if (!calls || calls->type != J_ARR) return;
    int muse_calls = 0;
    for (int i = 0; i < calls->n; i++) {
        jv *fn = jv_get(calls->items[i], "function");
        const char *name = jv_str(jv_get(fn, "name"), NULL);
        const char *args = jv_str(jv_get(fn, "arguments"), "{}");
        if (!name) continue;
        if (tmpl != TMPL_ORNITH && tmpl != TMPL_MUSE) {
            sb_fmt(out, "<|tool_call>call:%s%s<tool_call|>", name, args);
            continue;
        }
        jv *obj = json_parse(args, strlen(args));
        if (tmpl == TMPL_MUSE) {
            if (muse_calls++)
                sb_fmt(out, "<|eom|><|start|>assistant to=%s<|message|>",
                       name);
            sb_fmt(out, "<atem:function_calls>\n<atem:invoke name=\"%s\">\n",
                   name);
            if (obj && obj->type == J_OBJ) {
                for (int k = 0; k < obj->n; k++) {
                    sb_fmt(out, "<atem:parameter name=\"%s\">", obj->keys[k]);
                    if (obj->items[k]->type == J_STR)
                        sb_put(out, obj->items[k]->str,
                               strlen(obj->items[k]->str));
                    else
                        jv_dump(obj->items[k], out);
                    sb_lit(out, "</atem:parameter>\n");
                }
            }
            sb_lit(out, "</atem:invoke>\n</atem:function_calls>");
            jv_free(obj);
            continue;
        }
        sb_fmt(out, "<tool_call>\n<function=%s>\n", name);
        if (obj && obj->type == J_OBJ) {
            for (int k = 0; k < obj->n; k++) {
                sb_fmt(out, "<parameter=%s>\n", obj->keys[k]);
                if (obj->items[k]->type == J_STR)
                    sb_put(out, obj->items[k]->str, strlen(obj->items[k]->str));
                else
                    jv_dump(obj->items[k], out);
                sb_lit(out, "\n</parameter>\n");
            }
        }
        sb_lit(out, "</function>\n</tool_call>");
        jv_free(obj);
    }
}

// ------------------------------------------- strict tool-call envelope

// The discriminator value of the no-call branch. It shares a namespace with
// the declared tool names, so a tool actually called "final" is rejected
// rather than silently shadowed.
#define FINAL_BRANCH "final"

// what the final branch accepts when the caller asked for plain text
#define FINAL_TEXT_SCHEMA \
    "{\"type\":\"object\",\"properties\":{\"content\":{\"type\":\"string\"}}," \
    "\"required\":[\"content\"]}"

static void envelope_branch(sbuf *s, bool first, const char *name,
                            const jv *args_schema, const char *args_literal) {
    if (!first) sb_lit(s, ",");
    // property order matters: schema.c dispatches the union on `tool`, so the
    // discriminator must be decided before `args` is sampled
    sb_lit(s, "{\"type\":\"object\",\"properties\":{\"tool\":{\"const\":\"");
    sb_esc(s, name, strlen(name));
    sb_lit(s, "\"},\"args\":");
    if (args_schema) jv_dump(args_schema, s);
    else             sb_lit(s, args_literal);
    sb_lit(s, "},\"required\":[\"tool\",\"args\"]}");
}

// tools[i].function.name, or NULL when the declaration is unusable
static const char *tool_name_of(jv *tool, char *err, int errcap) {
    if (!tool || tool->type != J_OBJ) {
        snprintf(err, errcap, "each tools[] entry must be an object");
        return NULL;
    }
    const char *type = jv_str(jv_get(tool, "type"), "function");
    if (strcmp(type, "function") != 0) {
        snprintf(err, errcap, "tools[].type must be \"function\"");
        return NULL;
    }
    jv *fn = jv_get(tool, "function");
    if (!fn || fn->type != J_OBJ) {
        snprintf(err, errcap, "tools[].function must be an object");
        return NULL;
    }
    const char *name = jv_str(jv_get(fn, "name"), NULL);
    if (!name || !name[0]) {
        snprintf(err, errcap, "tools[].function.name must be a non-empty string");
        return NULL;
    }
    if (!strcmp(name, FINAL_BRANCH)) {
        snprintf(err, errcap,
                 "tool name \"" FINAL_BRANCH "\" is reserved for the no-call branch");
        return NULL;
    }
    return name;
}

static int tool_choice_kind(jv *choice, const char **named, char *err, int errcap) {
    *named = NULL;
    if (!choice || choice->type == J_NULL) return TCH_AUTO;
    if (choice->type == J_STR) {
        if (!strcmp(choice->str, "auto"))     return TCH_AUTO;
        if (!strcmp(choice->str, "none"))     return TCH_NONE;
        if (!strcmp(choice->str, "required")) return TCH_REQUIRED;
    } else if (choice->type == J_OBJ) {
        const char *type = jv_str(jv_get(choice, "type"), "function");
        const char *name = jv_str(jv_get(jv_get(choice, "function"), "name"), NULL);
        if (!strcmp(type, "function") && name && name[0]) {
            *named = name;
            return TCH_NAMED;
        }
    }
    snprintf(err, errcap, "tool_choice must be \"auto\", \"none\", \"required\" "
                          "or {\"type\":\"function\",\"function\":{\"name\":...}}");
    return -1;
}

int tool_envelope_build(jv *tools, jv *choice, jv *final_schema,
                        tool_envelope *out, char *err, int errcap) {
    return tool_envelope_build_ex(tools, choice, final_schema, false,
                                  out, err, errcap);
}

// Bound on a parallel turn's call count. An unbounded array under a token
// budget is a truncation waiting to happen, and sval_close would have to
// close it mid-call; a small cap keeps every legal document completable.
#define PARALLEL_MAX_CALLS 8

int tool_envelope_build_ex(jv *tools, jv *choice, jv *final_schema,
                           bool parallel, tool_envelope *out,
                           char *err, int errcap) {
    memset(out, 0, sizeof(*out));
    err[0] = 0;

    const char *named = NULL;
    int kind = tool_choice_kind(choice, &named, err, errcap);
    if (kind < 0) return -1;

    bool have_tools = tools && tools->type == J_ARR && tools->n > 0;
    if (!have_tools && tools && tools->type != J_ARR && tools->type != J_NULL) {
        snprintf(err, errcap, "tools must be an array");
        return -1;
    }
    if (!have_tools) {
        // asking for a tool call with nothing to call is a contradiction, not
        // a request to answer normally
        if (kind == TCH_REQUIRED || kind == TCH_NAMED) {
            snprintf(err, errcap, "tool_choice requires a non-empty tools array");
            return -1;
        }
        return 0;
    }
    if (kind == TCH_NONE) return 0;   // not a union this engine can express
    // schema.c caps a discriminated union at 60 branches; the final branch
    // takes one of them under "auto"
    if (tools->n > 59) {
        snprintf(err, errcap, "too many tools (max 59)");
        return -1;
    }

    out->kind = kind;
    if (kind == TCH_NAMED) {
        out->named = strdup(named);
        if (!out->named) { snprintf(err, errcap, "out of memory building tool choice"); return -1; }
    }
    out->final_is_text = final_schema == NULL;
    out->parallel = parallel;
    out->max_calls = parallel ? PARALLEL_MAX_CALLS : 1;

    sbuf schema = { 0 }, turn = { 0 };
    if (parallel)
        // One uniform array: a direct answer is a single-element array
        // holding the final branch, so the model never has to choose between
        // two document shapes — only how many entries to emit.
        sb_fmt(&schema, "{\"type\":\"object\",\"properties\":{\"calls\":"
                        "{\"type\":\"array\",\"minItems\":1,\"maxItems\":%d,"
                        "\"items\":", PARALLEL_MAX_CALLS);
    sb_lit(&schema, "{\"oneOf\":[");
    // The wording matters as much as the schema: sampling is constrained to
    // the union either way, but a model that does not understand the choice
    // spends its one branch on a tool call for a question that needed none.
    // So the no-call branch is stated FIRST and named as the default.
    sb_lit(&turn, "Reply with exactly one JSON object and nothing else.\n");
    if (parallel) {
        // The wording carries as much weight as the schema: sampling is
        // constrained either way, but a model that misreads the shape spends
        // its calls badly. State the container once, then the entry forms.
        sb_fmt(&turn, "The object is {\"calls\": [ ... ]} — a list of 1 to %d "
                      "entries, each one of the forms below. Put every tool "
                      "call you want made this turn in that list.\n",
               PARALLEL_MAX_CALLS);
    }
    if (kind == TCH_AUTO) {
        sb_lit(&turn, parallel ? "To answer the user directly, use one entry:\n"
                               : "To answer the user directly, reply:\n");
        if (out->final_is_text)
            sb_lit(&turn, "  {\"tool\": \"" FINAL_BRANCH "\", "
                          "\"args\": {\"content\": \"<your answer>\"}}\n");
        else
            sb_lit(&turn, "  {\"tool\": \"" FINAL_BRANCH "\", "
                          "\"args\": <the JSON object you were asked for>}\n");
        sb_lit(&turn, parallel ? "To call tools instead, use one entry each:\n"
                               : "To call a tool instead, reply:\n");
        sb_lit(&turn, "  {\"tool\": \"<tool name>\", \"args\": {<arguments>}}\n"
                      "Call a tool only when it is needed to answer; "
                      "otherwise answer directly.\n");
        if (parallel)
            sb_lit(&turn, "Do not mix an answer entry with tool-call entries; "
                          "either answer or call tools.\n");
    } else {
        sb_lit(&turn, "You must call a tool. Reply:\n"
                      "  {\"tool\": \"<tool name>\", \"args\": {<arguments>}}\n");
        if (kind == TCH_NAMED)
            sb_lit(&turn, "You must call the tool named below.\n");
    }
    sb_lit(&turn, "Available tools:\n");

    int emitted = 0;
    for (int i = 0; i < tools->n; i++) {
        const char *name = tool_name_of(tools->items[i], err, errcap);
        if (!name) goto bad;
        for (int j = 0; j < i; j++) {
            const char *prev = jv_str(jv_get(jv_get(tools->items[j], "function"),
                                             "name"), NULL);
            if (prev && !strcmp(prev, name)) {
                snprintf(err, errcap, "duplicate tool name '%.60s'", name);
                goto bad;
            }
        }
        if (kind == TCH_NAMED && strcmp(name, named) != 0) continue;

        jv *fn = jv_get(tools->items[i], "function");
        jv *params = jv_get(fn, "parameters");
        if (params && params->type != J_OBJ && params->type != J_NULL) {
            snprintf(err, errcap, "tools[].function.parameters must be an object");
            goto bad;
        }
        // no parameters declared: any object, which is as tight as this
        // compiler can express without a properties map
        envelope_branch(&schema, emitted == 0, name,
                        params && params->type == J_OBJ ? params : NULL,
                        "{\"type\":\"object\"}");
        emitted++;

        jv_dump(tools->items[i], &turn);
        sb_lit(&turn, "\n");
    }
    if (kind == TCH_NAMED && emitted == 0) {
        snprintf(err, errcap, "tool_choice names a tool that is not declared");
        goto bad;
    }
    if (kind == TCH_AUTO)
        envelope_branch(&schema, emitted == 0, FINAL_BRANCH, final_schema,
                        FINAL_TEXT_SCHEMA);
    sb_lit(&schema, "]}");
    if (parallel)
        sb_lit(&schema, "}},\"required\":[\"calls\"]}");

    if (schema.failed || turn.failed || !schema.s || !turn.s) {
        snprintf(err, errcap, "out of memory building the tool envelope");
        goto bad;
    }
    out->schema_src = schema.s;
    out->system_turn = turn.s;
    return 1;
bad:
    free(schema.s);
    free(turn.s);
    free(out->named);
    memset(out, 0, sizeof(*out));
    return -1;
}

void tool_envelope_free(tool_envelope *e) {
    if (!e) return;
    free(e->schema_src);
    free(e->system_turn);
    free(e->named);
    e->schema_src = e->system_turn = e->named = NULL;
}

// Map ONE envelope entry. Shared by the single-call document and each element
// of the parallel form so the two cannot drift in how they render a call.
// `index` numbers the call ids; returns 1 when a call was appended, 0 for the
// final branch, -1 when the entry is malformed.
static int envelope_entry_map(const tool_envelope *e, jv *v, int index,
                              sbuf *content, sbuf *tc) {
    if (!v || v->type != J_OBJ) return -1;
    const char *tool = jv_str(jv_get(v, "tool"), NULL);
    jv *args = jv_get(v, "args");
    if (!tool) return -1;

    if (!strcmp(tool, FINAL_BRANCH)) {
        if (e->final_is_text) {
            const char *text = jv_str(jv_get(args, "content"), "");
            sb_put(content, text, strlen(text));
        } else if (args) {
            jv_dump(args, content);
        }
        return 0;
    }

    sbuf a = { 0 };
    if (args) jv_dump(args, &a);
    else      sb_lit(&a, "{}");
    sb_fmt(tc, "{\"id\":\"call_%d\",\"type\":\"function\","
               "\"function\":{\"name\":\"", index);
    sb_esc(tc, tool, strlen(tool));
    sb_lit(tc, "\",\"arguments\":\"");
    sb_esc(tc, a.s ? a.s : "{}", a.s ? a.n : 2);
    sb_lit(tc, "\"}}");
    if (a.failed) tc->failed = true;
    free(a.s);
    return 1;
}

static const char *atem_find(const char *p, const char *end,
                             const char *needle) {
    size_t n = strlen(needle);
    if (p > end || n > (size_t)(end - p)) return NULL;
    const char *last = end - n;
    for (; p <= last; p++)
        if (!memcmp(p, needle, n)) return p;
    return NULL;
}

static const char *atem_attr(const char *p, const char *end, const char *key,
                             const char **value_end) {
    size_t kn = strlen(key);
    const char *k = atem_find(p, end, key);
    if (!k || k >= end || k + kn >= end || k[kn] != '"') return NULL;
    const char *v = k + kn + 1;
    const char *q = memchr(v, '"', (size_t)(end - v));
    if (!q) return NULL;
    *value_end = q;
    return v;
}

static jv *atem_param_schema(const tool_envelope *e,
                             const char *tool, size_t tool_n,
                             const char *param, size_t param_n) {
    if (!e || !e->tools || e->tools->type != J_ARR) return NULL;
    for (int i = 0; i < e->tools->n; i++) {
        jv *fn = jv_get(e->tools->items[i], "function");
        if (!fn) fn = e->tools->items[i];
        const char *name = jv_str(jv_get(fn, "name"), "");
        if (strlen(name) != tool_n || memcmp(name, tool, tool_n)) continue;
        jv *props = jv_get(jv_get(fn, "parameters"), "properties");
        if (!props || props->type != J_OBJ) return NULL;
        for (int k = 0; k < props->n; k++)
            if (strlen(props->keys[k]) == param_n &&
                !memcmp(props->keys[k], param, param_n)) return props->items[k];
    }
    return NULL;
}

static const char *atem_enum_recovery(jv *choices, const char *raw,
                                      size_t raw_n) {
    const char *best = jv_str(choices->items[0], "");
    size_t best_prefix = 0;
    for (int i = 0; i < choices->n; i++) {
        const char *candidate = jv_str(choices->items[i], NULL);
        if (!candidate) continue;
        size_t candidate_n = strlen(candidate), prefix = 0;
        size_t limit = raw_n < candidate_n ? raw_n : candidate_n;
        while (prefix < limit && raw[prefix] == candidate[prefix]) prefix++;
        if (prefix > best_prefix) {
            best = candidate;
            best_prefix = prefix;
        }
    }
    return best;
}

static void atem_number_recovery(sbuf *out, jv *schema, bool integer) {
    double value = 0.0;
    jv *bound = jv_get(schema, "minimum");
    if (bound && bound->type == J_NUM && value < bound->num)
        value = integer ? ceil(bound->num) : bound->num;
    bound = jv_get(schema, "exclusiveMinimum");
    // HUGE_VAL, not the INFINITY macro: the build carries
    // -Werror=nan-infinity-disabled under -ffast-math (Apple Clang enforces
    // it; GCC has no such warning, which is how this slipped through the
    // Linux-only night gates).
    if (bound && bound->type == J_NUM && value <= bound->num)
        value = integer ? floor(bound->num) + 1.0
                        : nextafter(bound->num, HUGE_VAL);
    bound = jv_get(schema, "maximum");
    if (bound && bound->type == J_NUM && value > bound->num)
        value = integer ? floor(bound->num) : bound->num;
    bound = jv_get(schema, "exclusiveMaximum");
    if (bound && bound->type == J_NUM && value >= bound->num)
        value = integer ? ceil(bound->num) - 1.0
                        : nextafter(bound->num, -HUGE_VAL);
    sb_fmt(out, integer ? "%.0f" : "%.17g", value);
}

static int atem_map(const tool_envelope *e, const char *doc, size_t n,
                    sbuf *content, sbuf *tc) {
    const char *p = doc, *end = doc + n;
    int calls = 0;
    while (p < end) {
        const char *inv = atem_find(p, end, "<atem:invoke name=\"");
        if (!inv || inv >= end) break;
        const char *name_end = NULL;
        const char *name = atem_attr(inv, end, "name=", &name_end);
        if (!name) return -1;
        const char *open_end = atem_find(name_end, end, ">");
        const char *inv_end = open_end
                            ? atem_find(open_end, end, "</atem:invoke>") : NULL;
        if (!open_end || !inv_end || inv_end > end) return -1;

        sbuf args = {0};
        sb_lit(&args, "{");
        int params = 0;
        const char *q = open_end + 1;
        while (q < inv_end) {
            const char *par = atem_find(q, inv_end,
                                        "<atem:parameter name=\"");
            if (!par || par >= inv_end) break;
            const char *pn_end = NULL;
            const char *pn = atem_attr(par, inv_end, "name=", &pn_end);
            const char *val = pn_end ? atem_find(pn_end, inv_end, ">") : NULL;
            if (!pn || !val || val >= inv_end) { free(args.s); return -1; }
            val++;
            const char *close = atem_find(val, inv_end, "</atem:parameter>");
            if (!close || close > inv_end) { free(args.s); return -1; }
            if (params++) sb_lit(&args, ",");
            sb_lit(&args, "\""); sb_esc(&args, pn, (size_t)(pn_end - pn));
            sb_lit(&args, "\":");
            size_t vn = (size_t)(close - val);
            jv *ps = atem_param_schema(e, name, (size_t)(name_end - name),
                                      pn, (size_t)(pn_end - pn));
            const char *pt = jv_str(jv_get(ps, "type"), "string");
            bool json_value = !strcmp(pt, "object") || !strcmp(pt, "array") ||
                              !strcmp(pt, "integer") || !strcmp(pt, "number") ||
                              !strcmp(pt, "boolean") || !strcmp(pt, "null");
            jv *structured = json_value ? json_parse(val, vn) : NULL;
            if (structured) { jv_dump(structured, &args); jv_free(structured); }
            else if (json_value) {
                // A token-budget close may leave a raw scalar prefix that is
                // not valid JSON (empty, `-`, `tru`, ...). The native atem
                // value itself is untyped text; use the declared parameter
                // type to keep the recovered OpenAI arguments executable.
                if (!strcmp(pt, "boolean")) sb_lit(&args, "false");
                else if (!strcmp(pt, "null")) sb_lit(&args, "null");
                else if (!strcmp(pt, "object")) sb_lit(&args, "{}");
                else if (!strcmp(pt, "array")) sb_lit(&args, "[]");
                else atem_number_recovery(&args, ps, !strcmp(pt, "integer"));
            } else {
                const char *sv = val;
                size_t sn = vn;
                jv *choices = jv_get(ps, "enum");
                if (choices && choices->type == J_ARR && choices->n) {
                    bool member = false;
                    for (int z = 0; z < choices->n; z++) {
                        const char *candidate = jv_str(choices->items[z], NULL);
                        if (candidate && strlen(candidate) == vn &&
                            !memcmp(candidate, val, vn)) member = true;
                    }
                    if (!member) {
                        const char *fallback = atem_enum_recovery(
                            choices, val, vn);
                        sv = fallback;
                        sn = strlen(fallback);
                    }
                }
                sb_lit(&args, "\""); sb_esc(&args, sv, sn); sb_lit(&args, "\"");
            }
            q = close + strlen("</atem:parameter>");
        }
        sb_lit(&args, "}");
        if (args.failed) { free(args.s); return -1; }
        if (calls) sb_lit(tc, ",");
        sb_fmt(tc, "{\"id\":\"call_%d\",\"type\":\"function\","
                   "\"function\":{\"name\":\"", calls);
        sb_esc(tc, name, (size_t)(name_end - name));
        sb_lit(tc, "\",\"arguments\":\"");
        sb_esc(tc, args.s, args.n);
        sb_lit(tc, "\"}}");
        free(args.s);
        calls++;
        p = inv_end + strlen("</atem:invoke>");
    }
    if (calls) return calls;

    // A native direct answer is addressed to the user. Keep the recipient
    // header out of content just as the JSON envelope syntax is kept out.
    const char *u = atem_find(doc, end, " to=user<|message|>");
    if (!u) return -1;
    u += strlen(" to=user<|message|>");
    const char *stop = atem_find(u, end, "<|eot|>");
    sb_put(content, u, stop && stop <= end ? (size_t)(stop - u)
                                           : (size_t)(end - u));
    return 0;
}

bool muse_user_payload_strip(sbuf *payload) {
    if (!payload || !payload->s) return false;
    const char *message = strstr(payload->s, "<|message|>");
    if (!message || message >= payload->s + payload->n) return false;
    size_t off = (size_t)(message - payload->s) + strlen("<|message|>");
    memmove(payload->s, payload->s + off, payload->n - off);
    payload->n -= off;
    payload->s[payload->n] = 0;
    return true;
}

int tool_envelope_map(const tool_envelope *e, const char *doc, size_t n,
                      sbuf *content, sbuf *tc) {
    if (e && e->atem) return atem_map(e, doc, n, content, tc);
    if (e && e->muse_user_header) {
        const char *message = strstr(doc, "<|message|>");
        if (!message || message >= doc + n) return -1;
        message += strlen("<|message|>");
        n -= (size_t)(message - doc);
        doc = message;
    }
    jv *v = json_parse(doc, n);
    if (!v || v->type != J_OBJ) { jv_free(v); return -1; }

    if (e->parallel) {
        jv *calls = jv_get(v, "calls");
        if (!calls || calls->type != J_ARR) { jv_free(v); return -1; }
        int emitted = 0;
        for (int i = 0; i < calls->n; i++) {
            if (emitted) sb_lit(tc, ",");
            int rc = envelope_entry_map(e, calls->items[i], emitted, content, tc);
            if (rc < 0) { jv_free(v); return -1; }
            // a final branch contributes content and no comma-separated item,
            // so the separator above must key off what was actually emitted
            if (rc == 0 && emitted) tc->n -= 1;
            emitted += rc;
        }
        jv_free(v);
        return emitted;
    }

    const char *tool = jv_str(jv_get(v, "tool"), NULL);
    jv *args = jv_get(v, "args");
    if (!tool) { jv_free(v); return -1; }

    if (!strcmp(tool, FINAL_BRANCH)) {
        if (e->final_is_text) {
            const char *text = jv_str(jv_get(args, "content"), "");
            sb_put(content, text, strlen(text));
        } else if (args) {
            // the caller asked for a schema-shaped answer: hand back its own
            // document verbatim, not a field lifted out of it
            jv_dump(args, content);
        }
        jv_free(v);
        return 0;
    }

    // OpenAI carries arguments as a JSON *string*, so the document is dumped
    // and then escaped into the field
    sbuf a = { 0 };
    if (args) jv_dump(args, &a);
    else      sb_lit(&a, "{}");
    sb_lit(tc, "{\"id\":\"call_0\",\"type\":\"function\",\"function\":{\"name\":\"");
    sb_esc(tc, tool, strlen(tool));
    sb_lit(tc, "\",\"arguments\":\"");
    sb_esc(tc, a.s ? a.s : "{}", a.s ? a.n : 2);
    sb_lit(tc, "\"}}");
    if (a.failed) tc->failed = true;
    free(a.s);
    jv_free(v);
    return 1;
}

// ------------------------------------- streaming envelope demultiplexer
//
// The buffered mapper above parses a finished document. A stream has no
// finished document to parse, so the same decision is made incrementally:
// hold bytes back until the discriminator is known, then forward everything
// after it to the channel that branch selected. Holding back is what keeps
// envelope syntax out of the client's `content` — by the time a byte is
// forwarded, it is already known to be assistant text or tool arguments.

enum { TS_TOOL, TS_ARGS, TS_FINAL_KEY, TS_FINAL_STR, TS_VALUE, TS_ATEM,
       TS_MUSE_HEADER, TS_MUSE_CONTENT, TS_DONE };

static void head_put(tool_stream *s, const char *b, size_t n) {
    if (s->head_n + n + 1 > s->head_cap) {
        size_t cap = s->head_cap ? s->head_cap * 2 : 128;
        while (cap < s->head_n + n + 1) cap *= 2;
        char *tmp = realloc(s->head, cap);
        if (!tmp) { s->state = TS_DONE; return; }
        s->head = tmp;
        s->head_cap = cap;
    }
    memcpy(s->head + s->head_n, b, n);
    s->head_n += n;
    s->head[s->head_n] = 0;
}

static void head_drop(tool_stream *s, size_t upto) {
    memmove(s->head, s->head + upto, s->head_n - upto);
    s->head_n -= upto;
    if (s->head) s->head[s->head_n] = 0;
}

static bool ts_ws(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

// index just past `"key" :` within head, or -1 while it has not arrived
static long head_after_key(const tool_stream *s, const char *quoted_key) {
    if (!s->head) return -1;
    const char *at = strstr(s->head, quoted_key);
    if (!at) return -1;
    size_t i = (size_t)(at - s->head) + strlen(quoted_key);
    while (i < s->head_n && ts_ws(s->head[i])) i++;
    if (i >= s->head_n) return -1;
    if (s->head[i] != ':') return -1;
    return (long)(i + 1);
}

// with head[i] at an opening quote, the index just past the closing one
static long head_str_end(const tool_stream *s, size_t i) {
    for (size_t j = i + 1; j < s->head_n; j++) {
        if (s->head[j] == '\\') { j++; continue; }
        if (s->head[j] == '"') return (long)(j + 1);
    }
    return -1;
}

// forward raw JSON text of the selected value, dropping insignificant
// whitespace so the concatenated deltas are the same document the buffered
// path re-serializes
static int ts_value(tool_stream *s, const char *b, int n) {
    int rc = 0, run = -1;
    int (*emit)(void *, const char *, int) =
        s->called ? s->sink.call_args : s->sink.content;
    #define TS_FLUSH(upto) do { \
        if (run >= 0 && (upto) > run && emit) \
            rc = emit(s->sink.ud, b + run, (upto) - run); \
        run = -1; \
    } while (0)

    for (int i = 0; i < n && !rc; i++) {
        char c = b[i];
        if (s->in_str) {
            if (s->esc)            s->esc = false;
            else if (c == '\\')    s->esc = true;
            else if (c == '"') {
                s->in_str = false;
                if (s->depth == 0) { // a bare string value ends here
                    if (run < 0) run = i;
                    TS_FLUSH(i + 1);
                    s->state = TS_DONE;
                    return rc;
                }
            }
            if (run < 0) run = i;
            continue;
        }
        if (ts_ws(c)) { TS_FLUSH(i); continue; }
        if (s->depth == 0 && s->started) {
            // a scalar value has begun and this byte closes our parent
            if (c == ',' || c == '}' || c == ']') {
                TS_FLUSH(i);
                s->state = TS_DONE;
                return rc;
            }
        }
        s->started = true;
        if (c == '"')                    s->in_str = true;
        else if (c == '{' || c == '[')   s->depth++;
        else if (c == '}' || c == ']') {
            if (--s->depth <= 0) {
                if (run < 0) run = i;
                TS_FLUSH(i + 1);
                s->state = TS_DONE;
                return rc;
            }
        }
        if (run < 0) run = i;
    }
    TS_FLUSH(n);
    #undef TS_FLUSH
    return rc;
}

// forward the `final` branch's content string, unescaped, so a streamed
// answer is byte-identical to the buffered one
static int ts_final_str(tool_stream *s, const char *b, int n) {
    int rc = 0;
    for (int i = 0; i < n && !rc; i++) {
        char c = b[i];
        if (s->n_pend) {
            if (s->n_pend < (int)sizeof(s->pend)) s->pend[s->n_pend++] = c;
            char out[4];
            int outn = 0;
            int used = json_unescape(s->pend, (size_t)s->n_pend, out, &outn);
            if (used == 0) continue;             // still ambiguous
            if (used < 0) { s->n_pend = 0; continue; }  // drop a bad escape
            if (outn && s->sink.content)
                rc = s->sink.content(s->sink.ud, out, outn);
            // json_unescape may have declined a trailing byte (the lookahead
            // that proved there was no surrogate pair): replay it
            int left = s->n_pend - used;
            memmove(s->pend, s->pend + used, (size_t)left);
            s->n_pend = 0;
            if (left > 0 && !rc) {
                char tail[sizeof(s->pend)];
                memcpy(tail, s->pend, (size_t)left);
                rc = ts_final_str(s, tail, left);
                if (s->state != TS_FINAL_STR) return rc;
            }
            continue;
        }
        if (c == '"') { s->state = TS_DONE; return rc; }
        if (c == '\\') { s->pend[0] = c; s->n_pend = 1; continue; }
        int run = i;
        while (i < n && b[i] != '"' && b[i] != '\\') i++;
        if (s->sink.content) rc = s->sink.content(s->sink.ud, b + run, i - run);
        i--;
    }
    return rc;
}

// advance through the undecided prefix; returns non-zero when a sink asked to
// stop. Runs until it needs more input or the branch has been resolved.
static int ts_resolve(tool_stream *s) {
    for (;;) {
        switch (s->state) {
        case TS_TOOL: {
            long p = head_after_key(s, "\"tool\"");
            if (p < 0) return 0;
            size_t i = (size_t)p;
            while (i < s->head_n && ts_ws(s->head[i])) i++;
            if (i >= s->head_n) return 0;
            if (s->head[i] != '"') { s->state = TS_DONE; return 0; }
            long e = head_str_end(s, i);
            if (e < 0) return 0;
            size_t len = (size_t)e - i - 2;
            s->name = malloc(len + 1);
            if (!s->name) { s->state = TS_DONE; return 0; }
            memcpy(s->name, s->head + i + 1, len);
            s->name[len] = 0;
            s->called = strcmp(s->name, FINAL_BRANCH) != 0;
            head_drop(s, (size_t)e);
            s->state = TS_ARGS;
            // announce the call the moment the discriminator resolves: that
            // is the earliest point at which it is known, and telling the
            // client early is the reason to stream at all
            if (s->called && s->sink.call_begin) {
                int rc = s->sink.call_begin(s->sink.ud, s->name);
                if (rc) return rc;
            }
            break;
        }
        case TS_ARGS: {
            long p = head_after_key(s, "\"args\"");
            if (p < 0) return 0;
            size_t i = (size_t)p;
            while (i < s->head_n && ts_ws(s->head[i])) i++;
            if (i >= s->head_n) return 0;   // wait for the first value byte
            head_drop(s, i);
            if (s->called) {
                s->state = TS_VALUE;
            } else if (s->env && s->env->final_is_text) {
                s->state = TS_FINAL_KEY;
            } else {
                // the caller asked for a schema-shaped answer: its own
                // document is the reply, forwarded verbatim
                s->state = TS_VALUE;
            }
            break;
        }
        case TS_FINAL_KEY: {
            long p = head_after_key(s, "\"content\"");
            if (p < 0) return 0;
            size_t i = (size_t)p;
            while (i < s->head_n && ts_ws(s->head[i])) i++;
            if (i >= s->head_n) return 0;
            if (s->head[i] != '"') { s->state = TS_DONE; return 0; }
            head_drop(s, i + 1);
            s->state = TS_FINAL_STR;
            break;
        }
        default: {
            // a streaming state: drain whatever is still buffered into it
            if (!s->head_n) return 0;
            char *buf = s->head;
            size_t n = s->head_n;
            s->head = NULL;
            s->head_n = s->head_cap = 0;
            int rc = tool_stream_feed(s, buf, (int)n);
            free(buf);
            return rc;
        }
        }
    }
}

void tool_stream_init(tool_stream *s, const tool_envelope *e,
                      const tool_stream_sink *sink) {
    memset(s, 0, sizeof(*s));
    s->env = e;
    if (sink) s->sink = *sink;
    s->state = e && e->atem ? TS_ATEM
             : e && e->muse_plain_payload ? TS_MUSE_HEADER : TS_TOOL;
}

static int ts_muse_header(tool_stream *s, const char *bytes, int n) {
    head_put(s, bytes, (size_t)n);
    if (s->state == TS_DONE) return 0;
    const char *message = s->head ? strstr(s->head, "<|message|>") : NULL;
    if (!message) return 0;
    size_t off = (size_t)(message - s->head) + strlen("<|message|>");
    head_drop(s, off);
    s->state = TS_MUSE_CONTENT;
    int rc = s->head_n && s->sink.content
               ? s->sink.content(s->sink.ud, s->head, (int)s->head_n) : 0;
    s->head_n = 0;
    if (s->head) s->head[0] = 0;
    return rc;
}

static int ts_atem(tool_stream *s, const char *bytes, int n) {
    head_put(s, bytes, (size_t)n);
    if (s->state == TS_DONE) return 0;
    for (;;) {
        const char *inv = s->head ? strstr(s->head, "<atem:invoke name=\"") : NULL;
        if (inv) {
            const char *close = strstr(inv, "</atem:invoke>");
            if (!close) return 0;
            size_t upto = (size_t)(close - s->head) + strlen("</atem:invoke>");
            sbuf content = {0}, tc = {0}, wrapped = {0};
            int mapped = atem_map(s->env, inv, upto - (size_t)(inv - s->head),
                                  &content, &tc);
            sb_lit(&wrapped, "["); sb_put(&wrapped, tc.s, tc.n); sb_lit(&wrapped, "]");
            jv *arr = mapped == 1 ? json_parse(wrapped.s, wrapped.n) : NULL;
            jv *fn = arr && arr->type == J_ARR && arr->n == 1
                       ? jv_get(arr->items[0], "function") : NULL;
            const char *name = jv_str(jv_get(fn, "name"), NULL);
            const char *args = jv_str(jv_get(fn, "arguments"), NULL);
            int rc = 0;
            if (!name || !args) rc = 0;
            else {
                s->called = true;
                if (s->sink.call_begin) rc = s->sink.call_begin(s->sink.ud, name);
                if (!rc && s->sink.call_args)
                    rc = s->sink.call_args(s->sink.ud, args, (int)strlen(args));
                if (!rc && s->sink.call_end) rc = s->sink.call_end(s->sink.ud);
            }
            jv_free(arr); free(content.s); free(tc.s); free(wrapped.s);
            head_drop(s, upto);
            if (rc) return rc;
            continue;
        }
        const char *user = s->head ? strstr(s->head, " to=user<|message|>") : NULL;
        const char *eot = user ? strstr(user, "<|eot|>") : NULL;
        if (user && eot) {
            user += strlen(" to=user<|message|>");
            int rc = s->sink.content
                       ? s->sink.content(s->sink.ud, user, (int)(eot - user)) : 0;
            s->state = TS_DONE;
            return rc;
        }
        return 0;
    }
}

int tool_stream_feed(tool_stream *s, const char *bytes, int n) {
    if (n <= 0) return 0;
    switch (s->state) {
    case TS_DONE:      return 0;   // trailing envelope syntax: not the client's
    case TS_ATEM:      return ts_atem(s, bytes, n);
    case TS_MUSE_HEADER:return ts_muse_header(s, bytes, n);
    case TS_MUSE_CONTENT:
        return s->sink.content ? s->sink.content(s->sink.ud, bytes, n) : 0;
    case TS_VALUE:     return ts_value(s, bytes, n);
    case TS_FINAL_STR: return ts_final_str(s, bytes, n);
    default:
        head_put(s, bytes, (size_t)n);
        return ts_resolve(s);
    }
}

bool tool_stream_called(const tool_stream *s) { return s->called; }

void tool_stream_free(tool_stream *s) {
    if (!s) return;
    free(s->head);
    free(s->name);
    s->head = NULL;
    s->name = NULL;
    s->head_n = s->head_cap = 0;
}

int tool_calls_parse(sbuf *content, sbuf *tc) {
    if (!content->s) return 0;
    static const char OPEN[] = "<|tool_call>call:";
    static const char CLOSE[] = "<tool_call|>";
    int n_calls = 0;
    char *w = content->s; // write cursor for the compacted content
    const char *p = content->s, *end = content->s + content->n;
    while (p < end) {
        const char *o = strstr(p, OPEN);
        if (!o) {
            memmove(w, p, end - p);
            w += end - p;
            break;
        }
        memmove(w, p, o - p);
        w += o - p;
        const char *name = o + sizeof(OPEN) - 1;
        const char *brace = name;
        while (brace < end && *brace != '{' && *brace != '<' && brace - name < 128)
            brace++;
        if (brace >= end || *brace != '{') { p = name; continue; } // not a call
        // brace-match the args object (string- and escape-aware)
        const char *q = brace;
        int depth = 0;
        bool in_str = false;
        for (; q < end; q++) {
            if (in_str) {
                if (*q == '\\') q++;
                else if (*q == '"') in_str = false;
            } else if (*q == '"') in_str = true;
            else if (*q == '{') depth++;
            else if (*q == '}' && --depth == 0) { q++; break; }
        }
        if (depth != 0) { p = name; continue; } // truncated: leave as content
        sb_fmt(tc, "%s{\"id\":\"call_%d\",\"type\":\"function\",\"function\":"
                   "{\"name\":\"", n_calls ? "," : "", n_calls);
        sb_esc(tc, name, (int)(brace - name));
        sb_lit(tc, "\",\"arguments\":\"");
        sb_esc(tc, brace, (int)(q - brace));
        sb_lit(tc, "\"}}");
        n_calls++;
        p = q;
        if (end - p >= (int)sizeof(CLOSE) - 1 &&
            memcmp(p, CLOSE, sizeof(CLOSE) - 1) == 0)
            p += sizeof(CLOSE) - 1;
    }
    if (n_calls) content->n = (int)(w - content->s);
    return n_calls;
}

static const char *trim_left(const char *p, const char *end) {
    while (p < end && (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n')) p++;
    return p;
}

static const char *trim_right(const char *p, const char *end) {
    while (end > p && (end[-1] == ' ' || end[-1] == '\t' ||
                       end[-1] == '\r' || end[-1] == '\n')) end--;
    return end;
}

// Qwen3.5/Ornith's qwen3_xml dialect. Parameter bodies are JSON when they
// parse as JSON; ordinary text is preserved as a JSON string.
static int ornith_tool_calls_parse(sbuf *content, sbuf *tc) {
    if (!content->s) return 0;
    static const char OPEN[] = "<tool_call>";
    static const char FN[] = "<function=";
    static const char FN_END[] = "</function>";
    static const char CLOSE[] = "</tool_call>";
    static const char PARAM[] = "<parameter=";
    static const char PARAM_END[] = "</parameter>";
    char *w = content->s;
    const char *p = content->s, *end = content->s + content->n;
    int n_calls = 0;
    while (p < end) {
        const char *o = strstr(p, OPEN);
        if (!o || o >= end) {
            memmove(w, p, (size_t)(end - p));
            w += end - p;
            break;
        }
        const char *f = trim_left(o + sizeof(OPEN) - 1, end);
        if (end - f < (int)sizeof(FN) - 1 ||
            memcmp(f, FN, sizeof(FN) - 1) != 0) {
            memmove(w, p, (size_t)(o + 1 - p));
            w += o + 1 - p;
            p = o + 1;
            continue;
        }
        const char *name = f + sizeof(FN) - 1;
        const char *name_end = memchr(name, '>', (size_t)(end - name));
        const char *fn_end = name_end ? strstr(name_end + 1, FN_END) : NULL;
        const char *close = fn_end ? trim_left(fn_end + sizeof(FN_END) - 1, end) : NULL;
        if (!name_end || !fn_end || !close ||
            end - close < (int)sizeof(CLOSE) - 1 ||
            memcmp(close, CLOSE, sizeof(CLOSE) - 1) != 0) {
            memmove(w, p, (size_t)(o + 1 - p));
            w += o + 1 - p;
            p = o + 1;
            continue;
        }
        memmove(w, p, (size_t)(o - p));
        w += o - p;
        sb_fmt(tc, "%s{\"id\":\"call_%d\",\"type\":\"function\",\"function\":"
                   "{\"name\":\"", n_calls ? "," : "", n_calls);
        sb_esc(tc, name, (int)(name_end - name));
        sb_lit(tc, "\",\"arguments\":\"{");
        const char *a = name_end + 1;
        bool first = true;
        while (a < fn_end) {
            const char *po = strstr(a, PARAM);
            if (!po || po >= fn_end) break;
            const char *pn = po + sizeof(PARAM) - 1;
            const char *pn_end = memchr(pn, '>', (size_t)(fn_end - pn));
            const char *pv_end = pn_end ? strstr(pn_end + 1, PARAM_END) : NULL;
            if (!pn_end || !pv_end || pv_end > fn_end) break;
            const char *pv = trim_left(pn_end + 1, pv_end);
            const char *pe = trim_right(pv, pv_end);
            sb_lit(tc, first ? "\\\"" : ",\\\"");
            sb_esc(tc, pn, (int)(pn_end - pn));
            sb_lit(tc, "\\\":");
            jv *value = json_parse(pv, (size_t)(pe - pv));
            sbuf encoded = {0};
            if (value) {
                jv_dump(value, &encoded);
                jv_free(value);
            } else {
                sb_lit(&encoded, "\"");
                sb_esc(&encoded, pv, (int)(pe - pv));
                sb_lit(&encoded, "\"");
            }
            sb_esc(tc, encoded.s, encoded.n);
            free(encoded.s);
            first = false;
            a = pv_end + sizeof(PARAM_END) - 1;
        }
        sb_lit(tc, "}\"}}");
        n_calls++;
        p = close + sizeof(CLOSE) - 1;
    }
    if (n_calls) content->n = (int)(w - content->s);
    return n_calls;
}

int tool_calls_parse_for(int tmpl, sbuf *content, sbuf *tc) {
    return tmpl == TMPL_ORNITH ? ornith_tool_calls_parse(content, tc)
                               : tool_calls_parse(content, tc);
}
