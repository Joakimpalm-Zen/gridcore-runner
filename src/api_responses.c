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
static jv *responses_tools(jv *tools, char *err, int errcap) {
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
        free(b.s);
        return NULL;
    }
    jv *out = json_parse(b.s, b.n);
    free(b.s);
    if (!out) snprintf(err, errcap, "tools did not translate to a valid shape");
    return out;
}

// tool_choice, likewise: the named form is flat here and nested in chat.
static jv *responses_tool_choice(jv *tc, char *err, int errcap) {
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
    if (!out) snprintf(err, errcap, "out of memory translating tool_choice");
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
static char *responses_item_text(jv *item, const char **role) {
    const char *type = jv_str(jv_get(item, "type"), NULL);
    sbuf b = {0};
    // a tool result the caller is feeding back: this is the tool loop
    if (type && !strcmp(type, "function_call_output")) {
        *role = "tool";
        jv *out = jv_get(item, "output");
        if (out && out->type == J_STR) sb_put(&b, out->str, strlen(out->str));
        else if (out) jv_dump(out, &b);
        return b.s ? b.s : strdup("");
    }
    // the assistant's own earlier call, replayed: rendered in runner's call
    // syntax so the history reads like what the model actually emitted
    if (type && !strcmp(type, "function_call")) {
        *role = "assistant";
        const char *name = jv_str(jv_get(item, "name"), NULL);
        const char *args = jv_str(jv_get(item, "arguments"), "{}");
        if (!name) { free(b.s); return NULL; }
        sb_fmt(&b, "<|tool_call>call:%s%s<tool_call|>", name, args);
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
    return b.s;
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
    bool bad_fmt = false;
    jv *final_schema = responses_schema(req, &bad_fmt, terr, sizeof(terr));
    if (bad_fmt) { send_error(fd, 400, terr); return; }

    jv *tools = responses_tools(jv_get(req, "tools"), terr, sizeof(terr));
    if (jv_get(req, "tools") && jv_get(req, "tools")->type != J_NULL && !tools) {
        send_error(fd, 400, terr);
        return;
    }
    jv *choice_raw = jv_get(req, "tool_choice");
    jv *choice_owned = NULL;
    if (choice_raw && choice_raw->type == J_OBJ) {
        choice_owned = responses_tool_choice(choice_raw, terr, sizeof(terr));
        if (!choice_owned) { jv_free(tools); send_error(fd, 400, terr); return; }
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
    if (strict) sb_put(&ts, env.system_turn, strlen(env.system_turn));
    else        tools_render(tools, &ts);
    chat_msg *cm = malloc(sizeof(chat_msg) * (size_t)(n_items + 2));
    char **owned = malloc(sizeof(char *) * (size_t)n_items);
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
    if (ts.n) cm[n_cm++] = (chat_msg){ "system", ts.s };
    const char *instructions = jv_str(jv_get(req, "instructions"), NULL);
    if (instructions && instructions[0]) {
        cm[n_cm++] = (chat_msg){ "system", instructions };
        total += strlen(instructions) + 64;
    }
    if (input->type == J_STR) {
        cm[n_cm++] = (chat_msg){ "user", input->str };
        total += strlen(input->str) + 64;
    } else {
        for (int i = 0; i < input->n; i++) {
            const char *role = "user";
            char *text = responses_item_text(input->items[i], &role);
            if (!text) continue;
            owned[n_own++] = text;
            cm[n_cm++] = (chat_msg){ role, text };
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
    char *prompt = malloc(total + 256);
    if (!prompt) {
        for (int i = 0; i < n_own; i++) free(owned[i]);
        free(owned); free(cm); free(ts.s);
        tool_envelope_free(&env);
        jv_free(tools);
        jv_free(choice_owned);
        send_error(fd, 500, "out of memory building responses prompt");
        return;
    }
    render_messages(s->tmpl, cm, n_cm, true, req_enable_thinking(req),
                    prompt, total + 256);
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
