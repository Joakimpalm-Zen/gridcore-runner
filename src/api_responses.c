// OpenAI Responses request -> chat. Lifted out of server.c (RNR-019); see api.h.
#include "api.h"
#include "compat.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ------------------------------------------------ Responses request → chat
//
// The inbound half of the translation. A Responses request says the same
// things a chat request does in a different vocabulary, so it is rewritten
// into that vocabulary once, here, and everything downstream is the path
// /v1/chat/completions already takes. Nothing below generates or samples; if
// it did, there would be two engines to keep honest instead of one.

// Responses declares a tool flat — {"type":"function","name":...,"parameters":
// ...} — where chat nests it under "function". Rather than teach the envelope
// compiler a second shape (and risk the chat path with it), the flat form is
// re-serialised into the nested one and re-parsed. Returns an owned jv the
// caller frees, or NULL with err set.
static jv *responses_tools(jv *tools, char *err, int errcap, bool *oom) {
    if (!tools || tools->type == J_NULL) return NULL;
    if (tools->type != J_ARR) {
        snprintf(err, errcap, "tools must be an array");
        return NULL;
    }
    sbuf b = {0};
    sb_lit(&b, "[");
    int emitted = 0;
    // one level of `namespace` nesting is flattened, so the loop walks the
    // outer list and, for a namespace, its inner list
    for (int i = 0; i < tools->n; i++) {
        jv *outer = tools->items[i];
        if (!outer || outer->type != J_OBJ) {
            snprintf(err, errcap, "each tools[] entry must be an object");
            free(b.s);
            return NULL;
        }
        const char *otype = jv_str(jv_get(outer, "type"), "function");
        jv *group = NULL;
        if (!strcmp(otype, "namespace")) {
            // Codex groups related function tools under a namespace entry. A
            // namespace is a container, not a tool, so it is flattened rather
            // than refused: every leaf is a local function after all.
            group = jv_get(outer, "tools");
            if (!group || group->type != J_ARR) {
                snprintf(err, errcap,
                         "tools[].type \"namespace\" must carry a tools array");
                free(b.s);
                return NULL;
            }
        }
        int inner_n = group ? group->n : 1;
        for (int k = 0; k < inner_n; k++) {
            jv *t = group ? group->items[k] : outer;
            if (!t || t->type != J_OBJ) {
                snprintf(err, errcap, "each tools[] entry must be an object");
                free(b.s);
                return NULL;
            }
            const char *type = jv_str(jv_get(t, "type"), "function");
            if (strcmp(type, "function") != 0) {
                // A hosted tool the client itself marked unavailable is not a
                // request for anything, so dropping it misleads nobody. One
                // that is actually asked for is a capability this runtime does
                // not have, and saying so beats leaving the caller waiting for
                // a call that can never come.
                jv *web = jv_get(t, "external_web_access");
                if (web && web->type == J_BOOL && !web->b) continue;
                snprintf(err, errcap,
                         "tools[].type \"%.40s\" is not supported; "
                         "only \"function\" tools can run locally", type);
                free(b.s);
                return NULL;
            }
            // already nested (a client reusing its chat tool definitions)
            jv *nested = jv_get(t, "function");
            if (emitted++) sb_lit(&b, ",");
            sb_lit(&b, "{\"type\":\"function\",\"function\":");
            if (nested && nested->type == J_OBJ) {
                jv_dump(nested, &b);
            } else {
                sb_lit(&b, "{\"name\":");
                jv *nm = jv_get(t, "name");
                if (nm) jv_dump(nm, &b); else sb_lit(&b, "null");
                jv *desc = jv_get(t, "description");
                if (desc) { sb_lit(&b, ",\"description\":"); jv_dump(desc, &b); }
                jv *params = jv_get(t, "parameters");
                if (params) { sb_lit(&b, ",\"parameters\":"); jv_dump(params, &b); }
                sb_lit(&b, "}");
            }
            sb_lit(&b, "}");
        }
    }
    sb_lit(&b, "]");
    if (b.failed || !b.s) {
        snprintf(err, errcap, "out of memory translating tools");
        *oom = true;
        free(b.s);
        return NULL;
    }
    jv *out = json_parse(b.s, b.n);
    free(b.s);
    if (!out) snprintf(err, errcap, "tools did not translate to a valid shape");
    return out;
}

// tool_choice, likewise: the named form is flat here and nested in chat.
static jv *responses_tool_choice(jv *tc, char *err, int errcap, bool *oom) {
    if (!tc || tc->type != J_OBJ) return NULL; // strings pass through unchanged
    const char *name = jv_str(jv_get(tc, "name"), NULL);
    if (!name) {
        snprintf(err, errcap,
                 "tool_choice object must be {\"type\":\"function\",\"name\":...}");
        return NULL;
    }
    sbuf b = {0};
    sb_lit(&b, "{\"type\":\"function\",\"function\":{\"name\":\"");
    sb_esc(&b, name, strlen(name));
    sb_lit(&b, "\"}}");
    jv *out = b.failed || !b.s ? NULL : json_parse(b.s, b.n);
    free(b.s);
    // The buffer this re-parses is one this function serialised from a single
    // escaped name, so it round-trips by construction: a NULL here is the
    // allocator, not the caller's tool_choice.
    if (!out) {
        snprintf(err, errcap, "out of memory translating tool_choice");
        *oom = true;
    }
    return out;
}

// `text.format` is the Responses spelling of `response_format`. Returns the
// schema to constrain to, or NULL; *bad is set when the field is malformed.
static jv *responses_schema(jv *req, bool *bad, char *err, int errcap) {
    *bad = false;
    jv *text = jv_get(req, "text");
    if (!text || text->type == J_NULL) return NULL;
    if (text->type != J_OBJ) {
        snprintf(err, errcap, "text must be an object");
        *bad = true;
        return NULL;
    }
    jv *fmt = jv_get(text, "format");
    if (!fmt || fmt->type == J_NULL) return NULL;
    if (fmt->type != J_OBJ) {
        snprintf(err, errcap, "text.format must be an object");
        *bad = true;
        return NULL;
    }
    const char *type = jv_str(jv_get(fmt, "type"), "");
    if (!strcmp(type, "text")) return NULL;
    if (!strcmp(type, "json_object")) return NULL; // handled as json mode
    if (strcmp(type, "json_schema") != 0) {
        snprintf(err, errcap,
                 "text.format.type must be text, json_object or json_schema");
        *bad = true;
        return NULL;
    }
    // Responses puts the schema directly on the format object rather than
    // under a json_schema wrapper
    jv *sch = jv_get(fmt, "schema");
    if (!sch || sch->type != J_OBJ) {
        snprintf(err, errcap, "text.format.schema must be an object");
        *bad = true;
        return NULL;
    }
    return sch;
}

// Flatten one `input` item to prompt text, appending it as a chat turn.
// Returns the role to file it under, or NULL when the item carries nothing.
//
// `call_name` is the function a `function_call` item was made under, already
// resolved by the caller -- which is also where a call that cannot be named is
// refused. It is resolved out there rather than in here because Harmony needs
// the same name on the turn itself, and because a name this function cannot
// find is a reason to answer 400, not to return NULL and be skipped.
// *oom separates the two NULLs. An item that carries nothing renderable is
// skipped by the caller, which is right; an item whose text could not be
// ASSEMBLED used to take the same exit, so an allocation failure dropped a turn
// out of the conversation and the request still answered 200.
static char *responses_item_text(jv *item, int tmpl, const char **role,
                                 const char *call_name, bool *oom) {
    const char *type = jv_str(jv_get(item, "type"), NULL);
    sbuf b = {0};
    // a tool result the caller is feeding back: this is the tool loop
    if (type && !strcmp(type, "function_call_output")) {
        jv *out = jv_get(item, "output");
        if (out && out->type == J_STR) sb_put(&b, out->str, strlen(out->str));
        else if (out) jv_dump(out, &b);
        if (b.failed) { free(b.s); *oom = true; return NULL; }
        // ornith frames a result as a <tool_response> block in a USER turn;
        // its own render loop keys on that content prefix. Every other family
        // carries the result plain under role "tool" (chatml wraps it in the
        // template, gemma4/muse name it on the turn header) -- the SAME
        // framing the chat surface produces via tool_result_wrap.
        if (tmpl == TMPL_ORNITH) {
            sbuf w = {0};
            *role = tool_result_wrap(tmpl, b.s ? b.s : "", &w);
            free(b.s);
            if (!w.s) { *oom = true; return NULL; } // ornith wrap is non-empty
            return w.s;
        }
        *role = "tool";
        if (b.s) return b.s;
        char *empty = strdup("");
        if (!empty) *oom = true;
        return empty;
    }
    // the assistant's own earlier call, replayed: serialized in the family's
    // native protocol -- the SAME assistant_calls_render path the chat surface
    // takes -- so a gemma4/ornith/muse history reads like what the model was
    // trained to emit rather than runner's generic call syntax.
    if (type && !strcmp(type, "function_call")) {
        *role = "assistant";
        const char *args = jv_str(jv_get(item, "arguments"), "{}");
        // Harmony's call turn is authored elsewhere (the caller replaces this
        // text with the raw arguments and sets the recipient name), so leave a
        // valid placeholder rather than route it through the generic serializer.
        if (tmpl == TMPL_HARMONY) {
            sb_fmt(&b, "<|tool_call>call:%s%s<tool_call|>", call_name, args);
            if (b.failed) { free(b.s); *oom = true; return NULL; }
            return b.s;
        }
        jv *calls = tool_call_synth(call_name, args);
        if (!calls) { *oom = true; return NULL; }
        assistant_calls_render(tmpl, "", calls, &b, NULL);
        jv_free(calls);
        if (!b.s || b.failed) { free(b.s); *oom = true; return NULL; }
        return b.s;
    }
    *role = jv_str(jv_get(item, "role"), "user");
    // "developer" is the Responses spelling of a system turn; chat templates
    // know the latter
    if (!strcmp(*role, "developer")) *role = "system";
    jv *content = jv_get(item, "content");
    if (content && content->type == J_STR) {
        sb_put(&b, content->str, strlen(content->str));
    } else if (content && content->type == J_ARR) {
        for (int i = 0; i < content->n; i++) {
            jv *part = content->items[i];
            const char *pt = jv_str(jv_get(part, "type"), "");
            // input_text / output_text are the Responses spellings; "text" is
            // accepted too because clients reusing chat parts send it
            if (strcmp(pt, "input_text") && strcmp(pt, "output_text") &&
                strcmp(pt, "text"))
                continue; // images and files have no local renderer
            const char *txt = jv_str(jv_get(part, "text"), NULL);
            if (!txt) continue;
            if (b.n) sb_lit(&b, "\n");
            sb_put(&b, txt, strlen(txt));
        }
    }
    if (b.failed) { free(b.s); *oom = true; return NULL; }
    return b.s;
}

static const char *responses_call_name(jv *items, int before, const char *id) {
    if (!items || items->type != J_ARR || !id) return NULL;
    for (int i = 0; i < before; i++) {
        jv *item = items->items[i];
        if (strcmp(jv_str(jv_get(item, "type"), ""), "function_call")) continue;
        const char *call_id = jv_str(jv_get(item, "call_id"), NULL);
        if (call_id && !strcmp(call_id, id))
            return jv_str(jv_get(item, "name"), NULL);
    }
    return NULL;
}

// Stateful Responses features this runtime has no store behind. Refusing them
// is the project invariant: a client that asked the server to remember a turn
// and got a 200 would believe it did.
static bool responses_reject_stateful(sock_t fd, jv *req) {
    jv *v = jv_get(req, "previous_response_id");
    if (v && v->type != J_NULL) {
        send_error(fd, 400,
                   "previous_response_id is not supported: this runtime is "
                   "stateless and stores no conversation. Send the prior turns "
                   "in `input` instead.");
        return true;
    }
    v = jv_get(req, "store");
    if (v && v->type != J_NULL) {
        if (v->type != J_BOOL) {
            send_error(fd, 400, "store must be a boolean");
            return true;
        }
        if (v->b) {
            send_error(fd, 400,
                       "store:true is not supported: this runtime is stateless "
                       "and cannot retrieve a stored response. Use store:false.");
            return true;
        }
    }
    v = jv_get(req, "background");
    if (v && v->type != J_NULL) {
        if (v->type != J_BOOL) {
            send_error(fd, 400, "background must be a boolean");
            return true;
        }
        if (v->b) {
            send_error(fd, 400,
                       "background:true is not supported: there is no response "
                       "store to poll. Use a streaming or buffered request.");
            return true;
        }
    }
    v = jv_get(req, "conversation");
    if (v && v->type != J_NULL) {
        send_error(fd, 400,
                   "conversation is not supported: this runtime is stateless "
                   "and stores no conversation.");
        return true;
    }
    // "truncation":"auto" asks the server to silently drop history to fit; a
    // caller told 200 would never learn its context had been edited
    const char *tr = jv_str(jv_get(req, "truncation"), NULL);
    if (tr && strcmp(tr, "disabled") != 0) {
        send_error(fd, 400,
                   "truncation:\"auto\" is not supported; a prompt that exceeds "
                   "the context window is rejected rather than silently cut");
        return true;
    }
    // `include` asks for extra output payloads (logprobs, image URLs, encrypted
    // reasoning) none of which this runtime can produce
    v = jv_get(req, "include");
    if (v && v->type == J_ARR && v->n > 0) {
        send_error(fd, 400,
                   "include[] is not supported; no additional output payloads "
                   "are available from this runtime");
        return true;
    }
    return false;
}

void handle_responses(slot_t *s, sock_t fd, jv *req) {
    if (responses_reject_stateful(fd, req)) return;

    jv *input = jv_get(req, "input");
    if (!input || input->type == J_NULL) {
        send_error(fd, 400, "missing input");
        return;
    }
    if (input->type != J_STR && input->type != J_ARR) {
        send_error(fd, 400, "input must be a string or an array of items");
        return;
    }
    // reasoning is accepted and echoed back rather than rejected: `effort` and
    // `summary` are hints about how much thinking to do, not guarantees about
    // the response document, and a local model's thinking channel is already
    // reported as a reasoning item. A malformed one is still an error.
    jv *reasoning = jv_get(req, "reasoning");
    if (reasoning && reasoning->type != J_NULL && reasoning->type != J_OBJ) {
        send_error(fd, 400, "reasoning must be an object");
        return;
    }

    char terr[224];
    // Set where the refusal is the SERVER running out of memory rather than
    // the request being wrong; a 400 invalid_request_error there tells the
    // caller to fix something they did not get wrong.
    bool oom = false;
    bool bad_fmt = false;
    jv *final_schema = responses_schema(req, &bad_fmt, terr, sizeof(terr));
    if (bad_fmt) { send_error(fd, 400, terr); return; }

    jv *tools = responses_tools(jv_get(req, "tools"), terr, sizeof(terr), &oom);
    if (jv_get(req, "tools") && jv_get(req, "tools")->type != J_NULL && !tools) {
        send_error(fd, oom ? 500 : 400, terr);
        return;
    }
    jv *choice_raw = jv_get(req, "tool_choice");
    jv *choice_owned = NULL;
    if (choice_raw && choice_raw->type == J_OBJ) {
        choice_owned = responses_tool_choice(choice_raw, terr, sizeof(terr),
                                             &oom);
        if (!choice_owned) {
            jv_free(tools);
            send_error(fd, oom ? 500 : 400, terr);
            return;
        }
    }

    tool_envelope env = {0};
    int rc = tool_envelope_build(tools, choice_owned ? choice_owned : choice_raw,
                                 final_schema, &env, terr, sizeof(terr));
    if (rc < 0) {
        jv_free(tools);
        jv_free(choice_owned);
        send_error(fd, 400, terr);
        return;
    }
    bool strict = rc == 1;
    // Teach declared tools in the family's model-native declaration syntax --
    // the SAME helper the chat surface uses -- instead of the generic block, so
    // a gemma4 or muse model offered tools here sees the format it was trained
    // on. env->tools aliases the heap `tools`, freed once at the tail after
    // run_completion; owns_tools stays false.
    bool native_decl = false;
    const jv *native_tools = tool_decl_native(s->tmpl, strict,
                                              /*atem_tool_calling=*/true, tools,
                                              &env, &native_decl);
    if (strict) {
        bool parallel = false;
        if (!request_bool(req, "parallel_tool_calls", false, &parallel)) {
            tool_envelope_free(&env);
            jv_free(tools);
            jv_free(choice_owned);
            send_error(fd, 400, "parallel_tool_calls must be a boolean");
            return;
        }
        if (parallel) {
            tool_envelope_free(&env);
            jv_free(tools);
            jv_free(choice_owned);
            send_error(fd, 400,
                       "parallel_tool_calls:true is not supported yet; "
                       "one call per turn");
            return;
        }
    }

    // assemble the turns: tool system turn, then `instructions` as a system
    // message, then the input items in order
    int n_items = input->type == J_ARR ? input->n : 1;
    sbuf ts = {0};
    if (strict && !native_decl)
        sb_put(&ts, env.system_turn, strlen(env.system_turn));
    else if (!native_decl)
        tools_render(tools, &ts);
    // The tool turn is content too: a builder that ran out here left `ts` short
    // or empty and the prompt would go out without the declarations the caller
    // sent, asking the model to call tools it was never shown.
    chat_msg *cm = NULL;
    char **owned = NULL;
    if (!ts.failed) {
        cm = malloc(sizeof(chat_msg) * (size_t)(n_items + 2));
        owned = malloc(sizeof(char *) * (size_t)n_items);
    }
    // client-controlled size: a NULL here would be indexed below. Fail cleanly.
    if (!cm || !owned) {
        free(cm); free(owned); free(ts.s);
        tool_envelope_free(&env);
        jv_free(tools);
        jv_free(choice_owned);
        send_error(fd, 500, "out of memory building responses prompt");
        return;
    }
    size_t total = ts.n + 128;
    int n_cm = 0, n_own = 0;
    if (ts.n)
        cm[n_cm++] = (chat_msg){ .role = "system", .content = ts.s };
    const char *instructions = jv_str(jv_get(req, "instructions"), NULL);
    if (instructions && instructions[0]) {
        cm[n_cm++] = (chat_msg){
            .role = "system", .content = instructions,
        };
        total += strlen(instructions) + 64;
    }
    if (input->type == J_STR) {
        cm[n_cm++] = (chat_msg){ .role = "user", .content = input->str };
        total += strlen(input->str) + 64;
    } else {
        for (int i = 0; i < input->n; i++) {
            const char *role = "user";
            const char *type = jv_str(jv_get(input->items[i], "type"), "");
            bool is_call = !strcmp(type, "function_call");
            // The function a replayed call was made under. It used to be read
            // inside responses_item_text(), which returned NULL when the item
            // carried no `name` -- and a NULL there is `continue`d, so the
            // assistant's own call left the conversation while the request
            // still answered 200. The model was then shown a tool result for a
            // call it never made, with nothing in the response saying so.
            //
            // Same rule as the unattributable tool RESULT below, for the same
            // reason: deduce the name when exactly one function is declared
            // (there is nothing to choose between), and otherwise refuse with
            // the field that would fix it. Never invent one, and never drop
            // the turn.
            const char *call_name = NULL;
            if (is_call) {
                call_name = jv_str(jv_get(input->items[i], "name"), NULL);
                if (!call_name || !call_name[0])
                    call_name = sole_tool_name(tools);
                if (!call_name) {
                    for (int k = 0; k < n_own; k++) free(owned[k]);
                    free(owned); free(cm); free(ts.s);
                    tool_envelope_free(&env);
                    jv_free(tools);
                    jv_free(choice_owned);
                    send_error(fd, 400,
                               "function_call cannot be attributed to a tool: "
                               "the item carries no `name`, and `tools` does "
                               "not declare exactly one function to deduce it "
                               "from. Send `name` on the function_call item.");
                    return;
                }
            }
            bool oom = false;
            char *text = responses_item_text(input->items[i], s->tmpl, &role,
                                             call_name, &oom);
            if (env.harmony && is_call && text) {
                free(text);
                text = strdup(jv_str(jv_get(input->items[i], "arguments"), "{}"));
                if (!text) oom = true;
            }
            if (oom) {
                for (int k = 0; k < n_own; k++) free(owned[k]);
                free(owned); free(cm); free(ts.s);
                tool_envelope_free(&env);
                jv_free(tools);
                jv_free(choice_owned);
                send_error(fd, 500, "out of memory building responses prompt");
                return;
            }
            if (!text) continue;
            owned[n_own++] = text;
            const char *name = NULL;
            bool is_output = !strcmp(type, "function_call_output");
            // muse addresses its assistant call turn ` to=NAME`, exactly as
            // harmony's recipient does, so both carry the name on the turn.
            if (is_call && (env.harmony || s->tmpl == TMPL_MUSE))
                name = call_name;
            // gemma4 and muse also NAME the RESULT on its turn header. Resolve
            // it the same way harmony does, but do NOT refuse when it cannot be
            // found: these families render an unresolved result under the
            // template's own fallback, matching the chat surface (which uses
            // tool_result_name and never refuses for them).
            else if (is_output &&
                     (s->tmpl == TMPL_GEMMA4 || s->tmpl == TMPL_MUSE)) {
                const char *cid = jv_str(jv_get(input->items[i], "call_id"),
                                         NULL);
                name = responses_call_name(input, i, cid);
                if (!name) name = sole_tool_name(tools);
            } else if (env.harmony && is_output) {
                const char *cid = jv_str(jv_get(input->items[i], "call_id"),
                                         NULL);
                name = responses_call_name(input, i, cid);
                if (!name) name = sole_tool_name(tools);
                // Harmony has no way to say "a tool returned this" without
                // saying WHICH: the turn is authored by the function itself.
                // With nothing to name it, the choices are an off-protocol
                // turn the model has never seen or a fabricated function name,
                // and both are answered 200 as if the request had been
                // understood. So it is refused instead, with the one thing the
                // caller can act on.
                if (!name) {
                    for (int k = 0; k < n_own; k++) free(owned[k]);
                    free(owned); free(cm); free(ts.s);
                    tool_envelope_free(&env);
                    jv_free(tools);
                    jv_free(choice_owned);
                    // its own buffer: `terr` is sized for the envelope
                    // compiler's messages, and this one quotes an id
                    char e[288];
                    snprintf(e, sizeof(e),
                             "function_call_output cannot be attributed to a "
                             "tool: %s%.40s%s. This runtime is stateless, so "
                             "send the matching function_call item in `input` "
                             "alongside its output.",
                             cid ? "no function_call in `input` carries "
                                   "call_id \"" : "it carries no call_id",
                             cid ? cid : "", cid ? "\"" : "");
                    send_error(fd, 400, e);
                    return;
                }
            }
            cm[n_cm++] = (chat_msg){ .role = role, .content = text,
                                    .name = name };
            total += strlen(role) + strlen(text) + 64;
        }
    }
    if (n_cm == 0) {
        free(owned); free(cm); free(ts.s);
        tool_envelope_free(&env);
        jv_free(tools);
        jv_free(choice_owned);
        send_error(fd, 400, "no input content");
        return;
    }
    // `total` is the opening guess only -- under Harmony the tool namespace
    // it never counted is rendered into the prompt too. render_prompt_alloc
    // measures the real size and grows to it.
    char *prompt = render_prompt_alloc(s->tmpl, cm, n_cm, true,
                                       req_thinking_mode(req),
                                       native_tools, total + 256);
    if (!prompt) {
        for (int i = 0; i < n_own; i++) free(owned[i]);
        free(owned); free(cm); free(ts.s);
        tool_envelope_free(&env);
        jv_free(tools);
        jv_free(choice_owned);
        send_error(fd, 500, "out of memory building responses prompt");
        return;
    }
    run_completion(s, fd, prompt, API_RESPONSES, req, strict ? &env : NULL);
    free(prompt);
    for (int i = 0; i < n_own; i++) free(owned[i]);
    free(owned);
    free(cm);
    free(ts.s);
    tool_envelope_free(&env);
    jv_free(tools);
    jv_free(choice_owned);
}
