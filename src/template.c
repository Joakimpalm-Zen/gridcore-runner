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
        // Harmony (gpt-oss): <|channel|> plus <|return|> is the pair no other
        // family carries. Checked before muse because both use <|start|>role
        // <|message|> framing; muse additionally requires <|eot|>, which
        // Harmony does not have, so the two cannot claim each other.
        if (strstr(meta_tmpl, "<|channel|>") && strstr(meta_tmpl, "<|return|>"))
            return TMPL_HARMONY;
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
    if (tok_find(tok, "<|channel|>") >= 0 && tok_find(tok, "<|return|>") >= 0)
        return TMPL_HARMONY;
    if (tok_find(tok, "<|eot|>") >= 0 && tok_find(tok, "<|message|>") >= 0)
        return TMPL_MUSE;
    if (tok_find(tok, "<|start_of_role|>") >= 0)   return TMPL_GRANITE;
    if (tok_find(tok, "<|im_start|>") >= 0)        return TMPL_CHATML;
    if (tok_find(tok, "<|start_header_id|>") >= 0) return TMPL_LLAMA3;
    if (tok_find(tok, "<|user|>") >= 0)
        return tok_find(tok, "<|end|>") >= 0 ? TMPL_PHI3 : TMPL_ZEPHYR;
    if (tok_find(tok, "<|turn>") >= 0)             return TMPL_GEMMA4;
    if (tok_find(tok, "<start_of_turn>") >= 0)     return TMPL_GEMMA;

    // Nothing matched. The fallback still renders llama2 — changing what
    // unrecognised models render is a behavioural decision, not a bug fix —
    // but it must not be SILENT. Diagnosed 2026-08-13b on gpt-oss-20b, whose
    // Harmony template (`<|start|>` / `<|channel|>` / `<|message|>` /
    // `<|return|>`) has no branch here: chat mode reported
    // "template: llama2" and fed the model `[INST]` / `<<SYS>>` markup that
    // appears nowhere in its own template. The model then emitted its
    // analysis-channel reasoning as visible output and ran past every stop
    // condition, because llama2's stops are not Harmony's. That is the same
    // class of error this project already refused for granite tool calling:
    // unimplemented, not approximated. Say so.
    if (meta_tmpl && *meta_tmpl) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            fprintf(stderr,
                "warning: this model ships a chat template this build does "
                "not recognise; falling back to llama2 markup, which the "
                "model was not trained on. Expect wrong turn framing and "
                "missed stops. Use --chat-template to name a supported one, "
                "or `raw` for no framing at all.\n");
        }
    }
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
    if (!strcmp(name, "harmony")) return TMPL_HARMONY;
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
        case TMPL_HARMONY: return "harmony";
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

// OpenAI Harmony's native function declaration syntax. The reference
// renderer converts the OpenAI JSON Schema parameter object to a compact
// TypeScript type inside a `functions` namespace; reproducing that shape is
// part of the prompt contract, not presentation sugar.
static bool harmony_required(jv *schema, const char *name) {
    jv *req = jv_get(schema, "required");
    if (!req || req->type != J_ARR) return false;
    for (int i = 0; i < req->n; i++)
        if (req->items[i]->type == J_STR && !strcmp(req->items[i]->str, name))
            return true;
    return false;
}

// THE COMMENTING RULE, in one place so nobody has to rediscover it site by
// site: the TOOL-level description splits into one "// " comment per line
// (harmony_comment_lines, below); EVERY other description or title in the
// namespace takes a single "// " prefix and is not split
// (harmony_comment_once). Those other four sites are the object schema's own
// description, a property's title, a property's description, and the
// description a oneOf property lifts above its own name.
//
// The reference is what makes the rule, not taste: openai-harmony abd677f7
// splits only at encoding.rs:775-777 and uses a single format! at
// encoding.rs:486-488, 508-510, 518 and 565. A multi-line value at any of the
// four therefore drops its continuation into the type body as a bare,
// uncommented, unindented line. That is invalid TypeScript and it is
// deliberately reproduced: the goal is one prompt per tool schema across
// engines rather than a runner-specific spelling, so conformance is measured
// against openai-harmony, the protocol's reference implementation, and not
// against what would read better as TypeScript. Commenting the continuations
// would make runner the odd engine out.
static void harmony_comment_lines(sbuf *out, const char *indent,
                                  const char *text) {
    if (!text || !*text) return;
    const char *p = text;
    for (;;) {
        const char *nl = strchr(p, '\n');
        sb_fmt(out, "%s// ", indent);
        sb_put(out, p, nl ? (size_t)(nl - p) : strlen(p));
        sb_lit(out, "\n");
        if (!nl) break;
        p = nl + 1;
    }
}

// One "// " prefix for the whole text, newlines and all — the non-tool half of
// the commenting rule above. Only the marker line is indented, matching the
// reference's single format!.
static void harmony_comment_once(sbuf *out, const char *indent,
                                 const char *text) {
    if (!text || !*text) return;
    sb_fmt(out, "%s// %s\n", indent, text);
}

static void harmony_schema_ts(jv *schema, const char *indent, sbuf *out);

// indent widened by `extra` spaces; NULL on allocation failure, which callers
// degrade to the unwidened indent rather than losing the type entirely.
static char *harmony_indent(const char *indent, size_t extra) {
    size_t il = strlen(indent);
    char *w = malloc(il + extra + 1);
    if (!w) return NULL;
    memcpy(w, indent, il);
    memset(w + il, ' ', extra);
    w[il + extra] = 0;
    return w;
}

static bool harmony_has_enum(jv *schema) {
    jv *e = jv_get(schema, "enum");
    return e && e->type == J_ARR && e->n > 0;
}

// `default: <value>`, without the comment marker. The reference prints a
// string default bare when the schema is an enum — the value is already one of
// the quoted alternatives — and JSON-encoded otherwise. Inside a top-level
// oneOf the reference skips that enum case, so `enum_bare` selects which of
// the two spellings this call site uses.
static void harmony_default_text(sbuf *out, jv *dflt, jv *owner, bool enum_bare) {
    sb_lit(out, "default: ");
    if (enum_bare && dflt->type == J_STR && harmony_has_enum(owner))
        sb_fmt(out, "%s", dflt->str);
    else
        jv_dump(dflt, out);
}

// A `nullable: true` schema gains " | null", but only when the rendered type
// does not already offer null: type ["string","null"] with nullable set is one
// union in the reference, not "string | null | null".
static void harmony_type_nullable(jv *schema, const char *indent, sbuf *out) {
    size_t mark = out->n;
    harmony_schema_ts(schema, indent, out);
    if (!jv_bool(jv_get(schema, "nullable"), false)) return;
    if (out->failed || !out->s) return;
    if (strstr(out->s + mark, "null")) return;
    sb_lit(out, " | null");
}

// The trailing `// <description> default: <value>` that follows a oneOf
// alternative. `desc` is passed in because the property-level form suppresses
// it in cases the alternative itself cannot see.
static void harmony_variant_comment(sbuf *out, jv *variant, const char *desc,
                                    bool enum_bare) {
    jv *dflt = jv_get(variant, "default");
    if (!desc && !dflt) return;
    sb_lit(out, " // ");
    if (desc) sb_fmt(out, "%s", desc);
    if (desc && dflt) sb_lit(out, " ");
    if (dflt) harmony_default_text(out, dflt, variant, enum_bare);
}

static void harmony_schema_ts(jv *schema, const char *indent, sbuf *out) {
    if (!schema || schema->type != J_OBJ) { sb_lit(out, "any"); return; }
    jv *one = jv_get(schema, "oneOf");
    if (one && one->type == J_ARR && one->n) {
        // Every alternative, the first one included, is introduced by
        // "\n<indent> | ", so the union opens with a leading pipe on its own
        // line. Verified against the reference renderer, which writes the same
        // prefix on the first iteration as on the rest.
        char *vi = harmony_indent(indent, 3);
        for (int i = 0; i < one->n; i++) {
            jv *v = one->items[i];
            sb_fmt(out, "\n%s | ", indent);
            harmony_type_nullable(v, vi ? vi : indent, out);
            harmony_variant_comment(out, v,
                                    jv_str(jv_get(v, "description"), NULL),
                                    false);
        }
        free(vi);
        return;
    }
    jv *tv = jv_get(schema, "type");
    if (tv && tv->type == J_ARR) {
        bool any = false;
        for (int i = 0; i < tv->n; i++) {
            if (tv->items[i]->type != J_STR) continue;
            if (any) sb_lit(out, " | ");
            const char *t = tv->items[i]->str;
            sb_lit(out, !strcmp(t, "integer") ? "number" : t);
            any = true;
        }
        if (!any) sb_lit(out, "any");
        return;
    }
    const char *type = jv_str(tv, "any");
    if (!strcmp(type, "object")) {
        const char *desc = jv_str(jv_get(schema, "description"), NULL);
        if (desc) harmony_comment_once(out, indent, desc);
        sb_lit(out, "{\n");
        jv *props = jv_get(schema, "properties");
        if (props && props->type == J_OBJ) {
            size_t il = strlen(indent);
            char *child = malloc(il + 5);
            if (!child) { sb_lit(out, "}"); return; }
            memcpy(child, indent, il);
            memcpy(child + il, "    ", 5);
            for (int i = 0; i < props->n; i++) {
                jv *p = props->items[i];
                jv *pone = jv_get(p, "oneOf");
                const char *opt =
                    harmony_required(schema, props->keys[i]) ? "" : "?";
                const char *title = jv_str(jv_get(p, "title"), NULL);
                if (title) {
                    harmony_comment_once(out, indent, title);
                    sb_fmt(out, "%s//\n", indent);
                }
                const char *pd = jv_str(jv_get(p, "description"), NULL);
                if (pd && !pone)
                    harmony_comment_once(out, indent, pd);
                jv *examples = jv_get(p, "examples");
                if (examples && examples->type == J_ARR && examples->n) {
                    sb_fmt(out, "%s// Examples:\n", indent);
                    for (int k = 0; k < examples->n; k++) {
                        // the reference lists string examples only, but still
                        // opens the block for a non-string-only list
                        if (examples->items[k]->type != J_STR) continue;
                        sb_fmt(out, "%s// - ", indent);
                        jv_dump(examples->items[k], out);
                        sb_lit(out, "\n");
                    }
                }
                jv *dflt = jv_get(p, "default");
                if (pone && pone->type == J_ARR) {
                    // A oneOf property is laid out vertically: comments first,
                    // then the bare `name:` line, one alternative per line, and
                    // a comma alone on the closing line.
                    const char *v0d =
                        pone->n ? jv_str(jv_get(pone->items[0], "description"),
                                         NULL)
                                : NULL;
                    bool desc_above = pd && !(v0d && !strcmp(pd, v0d));
                    if (desc_above) harmony_comment_once(out, indent, pd);
                    if (dflt) {
                        sb_fmt(out, "%s// ", indent);
                        harmony_default_text(out, dflt, p, true);
                        sb_lit(out, "\n");
                    }
                    sb_fmt(out, "%s%s%s:\n", indent, props->keys[i], opt);
                    char *vi = harmony_indent(indent, 3);
                    for (int k = 0; k < pone->n; k++) {
                        jv *v = pone->items[k];
                        sb_fmt(out, "%s | ", indent);
                        harmony_type_nullable(v, vi ? vi : indent, out);
                        // the first alternative says nothing the property
                        // description above it has not already said
                        const char *vd = jv_str(jv_get(v, "description"), NULL);
                        if ((k == 0 && desc_above) ||
                            (vd && pd && !strcmp(vd, pd)))
                            vd = NULL;
                        harmony_variant_comment(out, v, vd, true);
                        sb_lit(out, "\n");
                    }
                    free(vi);
                    sb_fmt(out, "%s,\n", indent);
                    continue;
                }
                sb_fmt(out, "%s%s%s: ", indent, props->keys[i], opt);
                harmony_type_nullable(p, child, out);
                sb_lit(out, ",");
                if (dflt && !pone) {
                    sb_lit(out, " // ");
                    harmony_default_text(out, dflt, p, true);
                }
                sb_lit(out, "\n");
            }
            free(child);
        }
        sb_fmt(out, "%s}", indent);
    } else if (!strcmp(type, "string")) {
        jv *vals = jv_get(schema, "enum");
        bool any = false;
        if (vals && vals->type == J_ARR) {
            // string alternatives only: the reference drops the rest, and an
            // enum with nothing left to offer falls back to plain string
            for (int i = 0; i < vals->n; i++) {
                if (vals->items[i]->type != J_STR) continue;
                if (any) sb_lit(out, " | ");
                jv_dump(vals->items[i], out);
                any = true;
            }
        }
        if (!any) sb_lit(out, "string");
    } else if (!strcmp(type, "integer") || !strcmp(type, "number")) {
        sb_lit(out, "number");
    } else if (!strcmp(type, "boolean")) {
        sb_lit(out, "boolean");
    } else if (!strcmp(type, "array")) {
        jv *items = jv_get(schema, "items");
        if (items) { harmony_schema_ts(items, indent, out); sb_lit(out, "[]"); }
        else sb_lit(out, "Array<any>");
    } else {
        // every remaining type name, "null" among them, is `any`: the
        // reference has no TypeScript spelling for a null-only parameter
        sb_lit(out, "any");
    }
}

static void harmony_render_tool_defs(const jv *tools, sbuf *out) {
    if (!tools || tools->type != J_ARR || tools->n == 0) return;
    sb_lit(out, "# Tools\n\n## functions\n\nnamespace functions {\n");
    for (int i = 0; i < tools->n; i++) {
        jv *fn = jv_get(tools->items[i], "function");
        const char *name = jv_str(jv_get(fn, "name"), "");
        const char *desc = jv_str(jv_get(fn, "description"), "");
        sb_lit(out, "\n");
        harmony_comment_lines(out, "", desc);
        sb_fmt(out, "type %s = ", name);
        jv *params = jv_get(fn, "parameters");
        if (params) {
            sb_lit(out, "(_: ");
            harmony_schema_ts(params, "", out);
            sb_lit(out, ") => any;\n");
        } else {
            sb_lit(out, "() => any;\n");
        }
    }
    sb_lit(out, "\n} // namespace functions");
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
    case TMPL_HARMONY: {
        // gpt-oss / OpenAI Harmony. Reference: the model's OWN
        // tokenizer.chat_template in the official GGUF. Turn framing is
        // <|start|>ROLE<|message|>CONTENT<|end|>; the assistant additionally
        // names a channel, <|channel|>analysis for reasoning and
        // <|channel|>final for the answer, and closes its final message with
        // <|return|> when generating.
        //
        // The system turn is the format's own preamble (identity, reasoning
        // effort, channel declaration) — NOT the caller's system message. A
        // caller system message is a DEVELOPER turn in Harmony, which is the
        // one structural thing that surprises people porting from ChatML.
        //
        // The preamble carries the knowledge cutoff but NOT the current date.
        // The reference emits both, adjacent, which makes them look like one
        // decision; they are two. `Current date:` is strftime_now() — a live
        // wall-clock value, so the prompt prefix would change at midnight and
        // evict the whole prefix cache for no gain (the same call llama3,
        // apertus and muse make). The cutoff is a FROZEN literal: byte-
        // identical on every render, nothing to interpolate, no cache cost, and
        // it is a fact the model is trained to have. Omitting it was a
        // side-effect of omitting the date, not a decision.
        bool have_tools = tools && tools->type == J_ARR && tools->n;
        sbuf system = {0};
        sb_lit(&system, "<|start|>system<|message|>You are ChatGPT, a large "
                        "language model trained by OpenAI.\n"
                        "Knowledge cutoff: 2024-06\n\nReasoning: medium");
        // The channel list is a CONSTANT in the reference, not a function of
        // the declared tools: SystemContent::default() (openai-harmony
        // abd677f7, chat.rs) always requires analysis, commentary, final, and
        // the model's own GGUF jinja template writes the same three as a
        // literal. commentary is where the model puts a user-visible preamble
        // before a tool call, so it is legal even with nothing to call.
        sb_lit(&system, "\n\n# Valid channels: analysis, commentary, final. "
                        "Channel must be included for every message.");
        // Declared function tools add one routing line to the SYSTEM turn.
        // The reference gates it on the CONVERSATION carrying a developer
        // `functions` namespace with at least one tool, not on the system
        // turn's own contents (encoding.rs:175-194 feeds
        // RenderOptions::conversation_has_function_tools, consumed at :996).
        if (have_tools)
            sb_lit(&system, "\nCalls to these tools must go to the commentary "
                            "channel: 'functions'.");
        sb_lit(&system, "<|end|>");
        off = emit(out, cap, off, "%s", system.s, NULL);
        free(system.s);

        // The DEVELOPER turn carries the caller's instructions and the tool
        // namespace, in that order, separated by a blank line, and it sits
        // immediately after the system preamble. Both halves are optional and
        // the turn exists if either does — the reference builds it from
        // DeveloperContent's two fields and joins the present ones with "\n\n"
        // (openai-harmony abd677f7, encoding.rs:1012-1037), and the model's own
        // GGUF jinja gates the whole turn on `developer_message or tools`.
        //
        // The namespace goes HERE, not in the system turn. openai-harmony has a
        // second slot that renders identical bytes into the system turn
        // (SystemContent::with_tools), but that one is for the model's BUILT-IN
        // tools; OpenAI function tools, which is all Runner ever declares, are
        // developer content. Only the position differs, so a golden taken from
        // the wrong slot looks perfect and is wrong.
        sbuf dev = {0};
        for (int i = 0; i < n_msgs; i++) {
            if (strcmp(msgs[i].role, "system")) continue;
            // fold every system message into one instructions block, in order
            sb_lit(&dev, dev.n ? "\n\n" : "# Instructions\n\n");
            sb_fmt(&dev, "%s", msgs[i].content);
        }
        if (have_tools) {
            if (dev.n) sb_lit(&dev, "\n\n");
            harmony_render_tool_defs(tools, &dev);
        }
        if (dev.n) {
            off = emit(out, cap, off, "<|start|>developer<|message|>",
                       NULL, NULL);
            off = emit(out, cap, off, "%s<|end|>", dev.s, NULL);
        }
        free(dev.s);

        for (int i = 0; i < n_msgs; i++) {
            const chat_msg *mm = &msgs[i];
            if (!strcmp(mm->role, "system")) continue;  // already in developer
            if (!strcmp(mm->role, "assistant") && mm->name && mm->name[0]) {
                const char *prefix = !strncmp(mm->name, "functions.", 10)
                                   ? "" : "functions.";
                off = emit(out, cap, off, "<|start|>assistant to=%s%s",
                           prefix, mm->name);
                off = emit(out, cap, off,
                           "<|channel|>commentary <|constrain|>json<|message|>%s"
                           "<|call|>", mm->content, NULL);
            } else if (!strcmp(mm->role, "tool") && mm->name && mm->name[0]) {
                const char *prefix = !strncmp(mm->name, "functions.", 10)
                                   ? "" : "functions.";
                // ` to=assistant` is load-bearing, not decoration. The
                // reference resolves the author token BEFORE it consumes the
                // channel, and a namespaced author ("functions.NAME") matches
                // no known role; it is accepted as Role::Tool only via the
                // recipient fallback branch. Drop the recipient and the whole
                // prompt fails to parse with "Unknown role: functions.NAME".
                off = emit(out, cap, off,
                           "<|start|>%s%s to=assistant<|channel|>commentary",
                           prefix, mm->name);
                off = emit(out, cap, off, "<|message|>%s<|end|>",
                           mm->content, NULL);
            } else if (!strcmp(mm->role, "assistant") && mm->channel &&
                       (!strcmp(mm->channel, "analysis") ||
                        !strcmp(mm->channel, "commentary"))) {
                off = emit(out, cap, off,
                           "<|start|>assistant<|channel|>%s<|message|>",
                           mm->channel, NULL);
                off = emit(out, cap, off, "%s<|end|>", mm->content, NULL);
            } else if (!strcmp(mm->role, "assistant")) {
                // history carries answers only: a past turn's analysis is not
                // replayed, which is what the reference template does too
                off = emit(out, cap, off,
                           "<|start|>assistant<|channel|>final<|message|>",
                           NULL, NULL);
                off = emit(out, cap, off, "%s<|end|>", mm->content, NULL);
            } else {
                off = emit(out, cap, off, "<|start|>%s<|message|>",
                           mm->role, NULL);
                off = emit(out, cap, off, "%s<|end|>", mm->content, NULL);
            }
        }
        // Generation prompt. The bare header lets the model pick its own
        // channel, which is what it is trained to do and what the splitter
        // expects by default; THINK_ON/THINK_OFF prime a channel explicitly,
        // the same lever muse pulls with ` to=self` / ` to=user`.
        if (add_assistant) {
            // Unlike muse, the DEFAULT primes the analysis channel rather
            // than leaving the header bare. Two reasons, both practical: the
            // model opens analysis on its own anyway (it is a reasoning
            // model), and the decoded open marker is the bare word
            // "analysis", which is far too common to hunt for in a free
            // stream. Priming lets the splitter start already inside
            // reasoning and never search for it. THINK_OFF primes final and
            // gets no reasoning at all.
            const char *head = have_tools
                    ? "<|start|>assistant"
                : thinking == THINK_OFF
                    ? "<|start|>assistant<|channel|>final<|message|>"
                    : "<|start|>assistant<|channel|>analysis<|message|>";
            off = emit(out, cap, off, "%s", head, NULL);
        }
        break;
    }
    case TMPL_MUSE: {
        // muse-glimmer (reference: the model's OWN tokenizer.chat_template,
        // read from the official meta-models GGUF, not a summary):
        // <|start|>ROLE<|message|>CONTENT<|eot|> per turn; assistant turns
        // carry a recipient — ` to=user` for answers, ` to=self` for
        // reasoning turns, which is what the think_open/close pair splits.
        // When no system message is present the reference injects a default
        // one whose constant parts are reproduced here. Its current-date line
        // is omitted for one reason only: a live wall-clock date changes the
        // prompt prefix at midnight and evicts the prefix cache, which is the
        // same call llama3, apertus and harmony make. It is NOT omitted
        // because the reference makes it optional — an earlier version of this
        // comment claimed the template gates it on `current_date is defined`,
        // which is false: that branch falls through to `elif strftime_now is
        // defined`, so under transformers the line always renders.
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
    case TMPL_GEMMA4: {
        // gemma4 (reference: llama.cpp models/templates/google-gemma-4-31B-it
        // .jinja): <|turn>role\n CONTENT <turn|>\n per turn, a native system
        // role (unlike gemma1-3), assistant role named "model"
        // Thinking is selected in the FIRST SYSTEM TURN, not at the
        // generation prompt: the model's own template (read from
        // tokenizer.chat_template on gemma-4-E2B, 2026-08-12) opens that turn
        // when `enable_thinking or tools or messages[0] is system/developer`
        // and injects `<|think|>` at the very top of it, above any system
        // text. `<|think|>` is a real token in that vocabulary. With no
        // system message the turn is created anyway, carrying only the
        // marker — which is why this cannot be folded into the loop below.
        // (A `developer` first message is rendered as its own turn here, as
        // it always has been; the template folds it into the system turn.
        // That divergence predates this path and is untouched by it.)
        int g4_first = 0;
        if (thinking == THINK_ON) {
            off = emit(out, cap, off, "<|turn>system\n<|think|>\n", NULL, NULL);
            if (n_msgs > 0 && !strcmp(msgs[0].role, "system")) {
                off = emit(out, cap, off, "%s<turn|>\n", msgs[0].content, NULL);
                g4_first = 1;
            } else {
                off = emit(out, cap, off, "<turn|>\n", NULL, NULL);
            }
        }
        for (int i = g4_first; i < n_msgs; i++) {
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
        // Thinking is not selected HERE for this family — it was selected in
        // the first system turn above, which is where the template puts it.
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
    }
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
    // Harmony history is one native recipient turn per call. The server
    // expands those turns before rendering; concatenating the generic marker
    // into assistant content would teach a second, conflicting protocol.
    if (tmpl == TMPL_HARMONY) return;
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
    if (e->owns_tools) jv_free(e->tools);
    e->tools = NULL;
    e->owns_tools = false;
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

static bool harmony_declares(const tool_envelope *e, const char *name,
                             size_t name_n) {
    if (!e->tools || e->tools->type != J_ARR) return false;
    for (int i = 0; i < e->tools->n; i++) {
        jv *fn = jv_get(e->tools->items[i], "function");
        const char *declared = jv_str(jv_get(fn, "name"), NULL);
        if (declared && strlen(declared) == name_n &&
            !memcmp(declared, name, name_n)) return true;
    }
    return false;
}

static int harmony_map(const tool_envelope *e, const char *doc, size_t n,
                       sbuf *reasoning, sbuf *content, sbuf *tc) {
    const char *p = doc, *end = doc + n;
    static const char analysis[] = "<|channel|>analysis<|message|>";
    static const char commentary[] = "<|channel|>commentary<|message|>";
    static const char final[] = "<|channel|>final<|message|>";
    static const char handoff_text[] = "<|end|><|start|>assistant";
    // The analysis region is bounded in schema.c by the LONGER form, the one
    // that already names a channel. Searching for the short form here would
    // stop at the first "<|end|><|start|>assistant" the model typed as body
    // text — legal under the schema, since inside a free-text region the
    // control tokens are masked and the handoff can only be spelled out
    // byte-by-byte — and then fail every following compare. Only the search
    // uses the longer form; p still advances by the short one, so "<|channel|>"
    // is left in place for the final/call_prefix compares below.
    static const char handoff_channel[] = "<|end|><|start|>assistant<|channel|>";
    static const char call_prefix[] =
        "<|channel|>commentary to=functions.";
    static const char call_header[] = "<|constrain|>json<|message|>";
    if ((size_t)(end - p) >= sizeof(analysis) - 1 &&
        !memcmp(p, analysis, sizeof(analysis) - 1)) {
        p += sizeof(analysis) - 1;
        const char *handoff = atem_find(p, end, handoff_channel);
        if (!handoff) return -1;
        if (reasoning) sb_put(reasoning, p, (size_t)(handoff - p));
        p = handoff + sizeof(handoff_text) - 1;
    }
    if ((size_t)(end - p) >= sizeof(final) - 1 &&
        !memcmp(p, final, sizeof(final) - 1)) {
        p += sizeof(final) - 1;
        const char *body_end = end;
        static const char ret[] = "<|return|>";
        if ((size_t)(body_end - p) >= sizeof(ret) - 1 &&
            !memcmp(body_end - (sizeof(ret) - 1), ret, sizeof(ret) - 1))
            body_end -= sizeof(ret) - 1;
        if (e->final_is_text) {
            sb_put(content, p, (size_t)(body_end - p));
        } else {
            jv *answer = json_parse(p, (size_t)(body_end - p));
            if (!answer) return -1;
            jv_dump(answer, content);
            jv_free(answer);
        }
        return content->failed ? -1 : 0;
    }
    if ((size_t)(end - p) >= sizeof(call_prefix) - 1 &&
        !memcmp(p, call_prefix, sizeof(call_prefix) - 1)) {
        p += sizeof(call_prefix) - 1;
    } else {
        if ((size_t)(end - p) < sizeof(commentary) - 1 ||
            memcmp(p, commentary, sizeof(commentary) - 1)) return -1;
        p += sizeof(commentary) - 1;
        const char *handoff = atem_find(p, end, handoff_text);
        if (!handoff) return -1;
        sb_put(content, p, (size_t)(handoff - p));
        p = handoff + sizeof(handoff_text) - 1;
        if ((size_t)(end - p) < sizeof(call_prefix) - 1 ||
            memcmp(p, call_prefix, sizeof(call_prefix) - 1)) return -1;
        p += sizeof(call_prefix) - 1;
    }
    const char *header = atem_find(p, end, call_header);
    if (!header || header == p ||
        !harmony_declares(e, p, (size_t)(header - p)))
        return -1;
    const char *args_text = header + sizeof(call_header) - 1;
    jv *args = json_parse(args_text, (size_t)(end - args_text));
    if (!args) return -1;
    sbuf canonical = {0};
    jv_dump(args, &canonical);
    jv_free(args);
    sb_lit(tc, "{\"id\":\"call_0\",\"type\":\"function\","
               "\"function\":{\"name\":\"");
    sb_esc(tc, p, (size_t)(header - p));
    sb_lit(tc, "\",\"arguments\":\"");
    sb_esc(tc, canonical.s ? canonical.s : "{}",
           canonical.s ? canonical.n : 2);
    sb_lit(tc, "\"}}");
    bool failed = canonical.failed || tc->failed;
    free(canonical.s);
    return failed ? -1 : 1;
}

int tool_envelope_map_channels(const tool_envelope *e, const char *doc,
                               size_t n, sbuf *reasoning, sbuf *content,
                               sbuf *tc) {
    if (!e || !doc || !content || !tc) return -1;
    if (e->harmony) return harmony_map(e, doc, n, reasoning, content, tc);
    return tool_envelope_map(e, doc, n, content, tc);
}

int tool_envelope_map(const tool_envelope *e, const char *doc, size_t n,
                      sbuf *content, sbuf *tc) {
    if (!e || !doc || !content || !tc) return -1;
    if (e->harmony) return harmony_map(e, doc, n, NULL, content, tc);
    if (e->atem) return atem_map(e, doc, n, content, tc);
    if (e->muse_user_header) {
        const char *end = doc + n;
        const char *message = atem_find(doc, end, "<|message|>");
        if (!message) return -1;
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
       TS_HARMONY,
       TS_MUSE_HEADER, TS_MUSE_CONTENT,
       // parallel_tool_calls: between two entries of {"calls":[...]}. A
       // value ending inside TS_VALUE/TS_FINAL_STR leaves the entry's own
       // closing '}' unread (they only consume the value itself), so this
       // state looks for it, then for the ',' that starts another entry or
       // the ']' that ends the array.
       TS_ENTRY_SEP,
       TS_DONE };

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

// An entry's value (a tool's args, or the final branch's content) just
// finished. For the plain single-call envelope that is the whole document:
// TS_DONE swallows whatever trails it, exactly as before this function
// existed. For the parallel {"calls":[...]} document it is only ONE array
// element: emit call_end for a tool branch, reset the per-entry parsing
// state, and hand off whatever of the input chunk is left over. That
// leftover may already hold the entry's closing '}' and the next entry
// besides — a real token stream, and especially the truncation closer, can
// deliver all of that in one piece.
static int ts_entry_close(tool_stream *s, const char *rest, int rest_n, int rc) {
    if (rc) { s->state = TS_DONE; return rc; }
    if (!(s->env && s->env->parallel)) { s->state = TS_DONE; return 0; }
    bool was_called = s->called;
    free(s->name);
    s->name = NULL;
    s->called = false;
    s->depth = 0;
    s->started = false;
    s->in_str = false;
    s->esc = false;
    s->n_pend = 0;
    s->state = TS_ENTRY_SEP;
    if (was_called && s->sink.call_end) {
        rc = s->sink.call_end(s->sink.ud);
        if (rc) { s->state = TS_DONE; return rc; }
    }
    return rest_n > 0 ? tool_stream_feed(s, rest, rest_n) : 0;
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
                    return ts_entry_close(s, b + i + 1, n - (i + 1), rc);
                }
            }
            if (run < 0) run = i;
            continue;
        }
        if (ts_ws(c)) { TS_FLUSH(i); continue; }
        if (s->depth == 0 && s->started) {
            // a scalar value has begun and this byte closes our parent; it
            // is not consumed here (a bare scalar has no delimiter of its
            // own), so it is still the first byte of what is left over
            if (c == ',' || c == '}' || c == ']') {
                TS_FLUSH(i);
                return ts_entry_close(s, b + i, n - i, rc);
            }
        }
        s->started = true;
        if (c == '"')                    s->in_str = true;
        else if (c == '{' || c == '[')   s->depth++;
        else if (c == '}' || c == ']') {
            if (--s->depth <= 0) {
                if (run < 0) run = i;
                TS_FLUSH(i + 1);
                return ts_entry_close(s, b + i + 1, n - (i + 1), rc);
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
        if (c == '"') return ts_entry_close(s, b + i + 1, n - (i + 1), rc);
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
            if (s->called) s->any_called = true;
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
        case TS_ENTRY_SEP: {
            // the value just forwarded leaves its entry's own closing '}'
            // unread; find it, then a ',' before another entry or the ']'
            // that ends the calls array. Peek only -- nothing is dropped
            // until the outcome is known, so a split chunk just waits for
            // more bytes and re-scans the same prefix next time.
            if (!s->head) return 0;
            size_t i = 0;
            while (i < s->head_n && ts_ws(s->head[i])) i++;
            if (i >= s->head_n) return 0;
            if (s->head[i] != '}') { s->state = TS_DONE; return 0; }
            i++;
            while (i < s->head_n && ts_ws(s->head[i])) i++;
            if (i >= s->head_n) return 0;
            if (s->head[i] == ']') {
                head_drop(s, i + 1);
                s->state = TS_DONE;
                return 0;
            }
            if (s->head[i] != ',') { s->state = TS_DONE; return 0; }
            i++;
            head_drop(s, i);
            s->state = TS_TOOL;
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
    s->state = e && e->harmony ? TS_HARMONY
             : e && e->atem ? TS_ATEM
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
                s->any_called = true;
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
    case TS_HARMONY:
        head_put(s, bytes, (size_t)n);
        return 0;
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

int tool_stream_finish(tool_stream *s) {
    if (!s || s->state != TS_HARMONY) return 0;
    sbuf reasoning = {0}, content = {0}, tc = {0}, wrapped = {0};
    int mapped = tool_envelope_map_channels(
        s->env, s->head ? s->head : "", s->head_n,
        &reasoning, &content, &tc);
    int rc = 0;
    if (mapped < 0) { rc = -1; goto done; }
    if (reasoning.n && s->sink.reasoning)
        rc = s->sink.reasoning(s->sink.ud, reasoning.s, (int)reasoning.n);
    if (!rc && content.n && s->sink.content)
        rc = s->sink.content(s->sink.ud, content.s, (int)content.n);
    if (!rc && mapped == 1) {
        sb_lit(&wrapped, "["); sb_put(&wrapped, tc.s, tc.n); sb_lit(&wrapped, "]");
        jv *arr = json_parse(wrapped.s, wrapped.n);
        jv *fn = arr && arr->type == J_ARR && arr->n == 1
                   ? jv_get(arr->items[0], "function") : NULL;
        const char *name = jv_str(jv_get(fn, "name"), NULL);
        const char *args = jv_str(jv_get(fn, "arguments"), NULL);
        if (!name || !args) rc = -1;
        else {
            s->called = s->any_called = true;
            if (s->sink.call_begin) rc = s->sink.call_begin(s->sink.ud, name);
            if (!rc && s->sink.call_args)
                rc = s->sink.call_args(s->sink.ud, args, (int)strlen(args));
            if (!rc && s->sink.call_end) rc = s->sink.call_end(s->sink.ud);
        }
        jv_free(arr);
    }
done:
    free(reasoning.s); free(content.s); free(tc.s); free(wrapped.s);
    s->state = TS_DONE;
    return rc;
}

bool tool_stream_called(const tool_stream *s) { return s->any_called; }

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
