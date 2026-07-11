// Chat template detection and rendering (ChatML, Llama-2/3, Zephyr).
#include "runner.h"

#include <stdio.h>
#include <string.h>

int template_detect(const char *meta_tmpl, tokenizer *tok) {
    if (meta_tmpl) {
        if (strstr(meta_tmpl, "<|im_start|>"))        return TMPL_CHATML;
        if (strstr(meta_tmpl, "<|start_header_id|>")) return TMPL_LLAMA3;
        if (strstr(meta_tmpl, "<|user|>"))            return TMPL_ZEPHYR;
        if (strstr(meta_tmpl, "<start_of_turn>"))     return TMPL_GEMMA;
        if (strstr(meta_tmpl, "[INST]"))              return TMPL_LLAMA2;
    }
    if (tok_find(tok, "<|im_start|>") >= 0)        return TMPL_CHATML;
    if (tok_find(tok, "<|start_header_id|>") >= 0) return TMPL_LLAMA3;
    if (tok_find(tok, "<|user|>") >= 0)            return TMPL_ZEPHYR;
    if (tok_find(tok, "<start_of_turn>") >= 0)     return TMPL_GEMMA;
    return TMPL_LLAMA2;
}

int template_from_name(const char *name) {
    if (!strcmp(name, "chatml")) return TMPL_CHATML;
    if (!strcmp(name, "llama2")) return TMPL_LLAMA2;
    if (!strcmp(name, "llama3")) return TMPL_LLAMA3;
    if (!strcmp(name, "zephyr")) return TMPL_ZEPHYR;
    if (!strcmp(name, "gemma"))  return TMPL_GEMMA;
    if (!strcmp(name, "raw"))    return TMPL_RAW;
    return -1;
}

const char *template_name(int t) {
    switch (t) {
        case TMPL_CHATML: return "chatml";  case TMPL_LLAMA2: return "llama2";
        case TMPL_LLAMA3: return "llama3";  case TMPL_ZEPHYR: return "zephyr";
        case TMPL_GEMMA:  return "gemma";
        default: return "raw";
    }
}

static size_t emit(char *out, size_t cap, size_t off, const char *fmt,
                   const char *a, const char *b) {
    if (off >= cap) return off;
    int n = snprintf(out + off, cap - off, fmt, a, b);
    return n > 0 ? off + (size_t)n : off;
}

size_t render_messages(int tmpl, const chat_msg *msgs, int n_msgs,
                       bool add_assistant, char *out, size_t cap) {
    size_t off = 0;
    out[0] = 0;
    switch (tmpl) {
    case TMPL_CHATML:
        for (int i = 0; i < n_msgs; i++)
            off = emit(out, cap, off, "<|im_start|>%s\n%s<|im_end|>\n",
                       msgs[i].role, msgs[i].content);
        if (add_assistant)
            off = emit(out, cap, off, "<|im_start|>assistant\n", NULL, NULL);
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
    case TMPL_LLAMA2: {
        // fold an initial system message into the first user turn
        const char *sys = NULL;
        for (int i = 0; i < n_msgs; i++) {
            const chat_msg *m = &msgs[i];
            if (!strcmp(m->role, "system")) { sys = m->content; continue; }
            if (!strcmp(m->role, "user")) {
                if (sys) {
                    off = emit(out, cap, off, "[INST] <<SYS>>\n%s\n<</SYS>>\n\n", sys, NULL);
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
