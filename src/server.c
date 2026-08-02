// HTTP server: OpenAI-compatible API with N parallel inference slots.
//
//   POST /v1/chat/completions   messages, sampling params, stream (SSE),
//                               response_format {"type":"json_object"}
//   POST /v1/completions        raw prompt completion
//   POST /v1/embeddings         mean-pooled L2-normed embeddings
//   GET  /v1/models             the loaded model
//   GET  /v1/capabilities       registry + feature discovery
//   GET  /health                liveness
//
// Swap-mode request bodies may carry "keep_alive" (seconds of idle before
// the model unloads; 0 = unload now, negative = keep forever).
//
// Each slot owns a full inference context (KV cache + thread pool); model
// weights are shared between slots through the page cache (mmap).
#include "runner.h"
#include "json.h"
#include "compat.h"
#include "http.h"
#include "server_int.h"
#include "scheduler.h"
#include "completion.h"

#include <errno.h>
#include <pthread.h>
#include <stdarg.h>
#include <limits.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <stdatomic.h>




// ---------------------------------------------------------------- routes

// flatten one OpenAI message to plain text: string content passes through;
// the AI-SDK part-array form (Cline et al.) concatenates its text parts;
// assistant tool_calls render in runner's own call syntax so replayed
// history reads like what the model actually emitted. Returns a heap
// string, or NULL when the message carries nothing usable.
static char *message_text(jv *msg, int tmpl) {
    jv *content = jv_get(msg, "content");
    sbuf b = {0};
    const char *role = jv_str(jv_get(msg, "role"), "user");
    const char *reason = jv_str(jv_get(msg, "reasoning_content"), NULL);
    if (tmpl == TMPL_ORNITH && !strcmp(role, "assistant")) {
        sb_lit(&b, "<think>\n");
        if (reason) sb_put(&b, reason, strlen(reason));
        sb_lit(&b, "\n</think>\n\n");
    }
    if (content && content->type == J_STR) {
        sb_put(&b, content->str, strlen(content->str));
    } else if (content && content->type == J_ARR) {
        for (int i = 0; i < content->n; i++) {
            const char *type = jv_str(jv_get(content->items[i], "type"), "");
            const char *text = jv_str(jv_get(content->items[i], "text"), NULL);
            if (strcmp(type, "text") != 0 || !text) continue; // images etc.
            if (b.n) sb_lit(&b, "\n");
            sb_put(&b, text, strlen(text));
        }
    }
    jv *calls = jv_get(msg, "tool_calls");
    tool_history_render_for(tmpl, calls, &b);
    if (tmpl == TMPL_ORNITH && !strcmp(role, "tool")) {
        sbuf wrapped = {0};
        sb_lit(&wrapped, "<tool_response>\n");
        if (b.s) sb_put(&wrapped, b.s, b.n);
        sb_lit(&wrapped, "\n</tool_response>");
        free(b.s);
        b = wrapped;
    }
    return b.s;
}

static void handle_chat(slot_t *s, sock_t fd, jv *req) {
    jv *msgs = jv_get(req, "messages");
    if (!msgs || msgs->type != J_ARR || msgs->n == 0) {
        send_error(fd, 400, "missing messages");
        return;
    }
    // OpenAI "tools" become a leading system turn (template.c owns the syntax).
    //
    // Strict mode compiles them into a discriminated union that constrains
    // sampling, so the model cannot name an undeclared tool or malform its
    // arguments. It applies to streamed requests too: the envelope is
    // demultiplexed as it is generated (tool_stream) rather than parsed
    // afterward, so both paths reach the same call from the same guarantee.
    jv *tools = jv_get(req, "tools");
    tool_envelope env = {0};
    // parallel_tool_calls is read BEFORE the envelope is built, because it
    // changes the envelope's shape. Silently ignoring a request for several
    // calls would leave the caller expecting calls it never gets.
    bool parallel = false;
    if (!request_bool(req, "parallel_tool_calls", false, &parallel)) {
        send_error(fd, 400, "parallel_tool_calls must be a boolean");
        return;
    }
    bool want_stream = false;
    if (!request_bool(req, "stream", false, &want_stream)) {
        send_error(fd, 400, "stream must be a boolean");
        return;
    }
    if (parallel && want_stream) {
        // The streaming demultiplexer tracks one call per turn; emitting
        // several would need per-index delta state it does not have. Refuse
        // rather than quietly downgrade to one call.
        send_error(fd, 400,
                   "parallel_tool_calls:true is not supported with stream:true; "
                   "use a buffered request");
        return;
    }
    char terr[224];
    int rc = tool_envelope_build_ex(tools, jv_get(req, "tool_choice"),
                                    request_schema(req), parallel, &env,
                                    terr, sizeof(terr));
    if (rc < 0) { send_error(fd, 400, terr); return; }
    // Ornith is specifically trained on qwen3_xml. Keep its native protocol
    // instead of forcing the model into runner's generic JSON envelope.
    bool strict = rc == 1 && s->tmpl != TMPL_ORNITH;
    // When the strict envelope does not apply — no tools declared, or the
    // ornith template's native protocol — the flag is vacuous and stays
    // TOLERATED, exactly as before: ordinary OpenAI-shaped traffic sends
    // parallel_tool_calls alongside requests that will never call anything,
    // and rejecting those would break it.
    sbuf ts = {0};
    if (strict) sb_put(&ts, env.system_turn, strlen(env.system_turn));
    else        tools_render_for(s->tmpl, tools, &ts);
    bool ornith_merged_system = false;
    if (s->tmpl == TMPL_ORNITH && ts.n && msgs->n > 0 &&
        !strcmp(jv_str(jv_get(msgs->items[0], "role"), ""), "system")) {
        char *system = message_text(msgs->items[0], s->tmpl);
        if (system && system[0]) {
            sb_lit(&ts, "\n\n");
            sb_put(&ts, system, strlen(system));
        }
        free(system);
        ornith_merged_system = true;
    }
    chat_msg *cm = malloc(sizeof(chat_msg) * (msgs->n + 1));
    char **owned = malloc(sizeof(char *) * msgs->n);
    // client-controlled size (a 32MB body of tiny messages): a NULL here would
    // be indexed below. Fail the request cleanly instead of crashing.
    if (!cm || !owned) {
        free(cm); free(owned); free(ts.s);
        tool_envelope_free(&env);
        send_error(fd, 500, "out of memory building chat prompt");
        return;
    }
    size_t total = ts.n + 64;
    int n_cm = 0, n_own = 0;
    if (ts.n) cm[n_cm++] = (chat_msg){ "system", ts.s };
    for (int i = 0; i < msgs->n; i++) {
        if (i == 0 && ornith_merged_system) continue;
        const char *role = jv_str(jv_get(msgs->items[i], "role"), "user");
        char *content = message_text(msgs->items[i], s->tmpl);
        if (!content) continue;
        owned[n_own++] = content;
        if (s->tmpl == TMPL_ORNITH && !strcmp(role, "tool")) role = "user";
        cm[n_cm++] = (chat_msg){ role, content };
        total += strlen(role) + strlen(content) + 64;
    }
    if (n_cm == 0) {
        free(owned);
        free(cm);
        free(ts.s);
        tool_envelope_free(&env);
        send_error(fd, 400, "no message content");
        return;
    }
    char *prompt = malloc(total + 256);
    if (!prompt) {
        for (int i = 0; i < n_own; i++) free(owned[i]);
        free(owned); free(cm); free(ts.s);
        tool_envelope_free(&env);
        send_error(fd, 500, "out of memory building chat prompt");
        return;
    }
    render_messages(s->tmpl, cm, n_cm, true, prompt, total + 256);
    run_completion(s, fd, prompt, API_CHAT, req, strict ? &env : NULL);
    free(prompt);
    for (int i = 0; i < n_own; i++) free(owned[i]);
    free(owned);
    free(cm);
    free(ts.s);
    tool_envelope_free(&env);
}

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

static void handle_responses(slot_t *s, sock_t fd, jv *req) {
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
    render_messages(s->tmpl, cm, n_cm, true, prompt, total + 256);
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

// ----------------------------------------- Anthropic Messages request → chat
//
// The inbound half of the third translation, and the same move
// handle_responses makes: an Anthropic request says the same things a chat
// request does in a different vocabulary, so it is rewritten into that
// vocabulary once, here, and everything downstream is the path
// /v1/chat/completions already takes. Nothing below generates or samples.

// A growable list of rendered chat turns.
//
// An Anthropic message does not map one-to-one onto a chat turn: a single user
// message can carry several tool_result blocks *and* text, and the chat
// vocabulary the templates speak files a tool result as its own turn. So turns
// are appended as they are produced rather than indexed by message.
typedef struct {
    chat_msg *cm;
    char    **owned;     // heap turns to free; borrowed ones are not listed
    int       n, n_own, cap;
    size_t    total;     // rendered-size estimate for the prompt buffer
    bool      failed;
} turnbuf;

static void turn_add_borrowed(turnbuf *t, const char *role, const char *text) {
    if (t->n >= t->cap) { t->failed = true; return; }
    t->cm[t->n++] = (chat_msg){ role, text };
    t->total += strlen(role) + strlen(text) + 64;
}

// takes ownership of `text` on every path, including failure
static void turn_add(turnbuf *t, const char *role, char *text) {
    if (!text) { t->failed = true; return; }
    if (t->n >= t->cap) { free(text); t->failed = true; return; }
    t->owned[t->n_own++] = text;
    turn_add_borrowed(t, role, text);
}

static void turnbuf_free(turnbuf *t) {
    for (int i = 0; i < t->n_own; i++) free(t->owned[i]);
    free(t->owned);
    free(t->cm);
}

// The text of one tool_result block. Its `content` is a string or a block
// list, and `is_error` is the one bit of the result the model needs to see
// that plain text would otherwise lose.
static char *anth_tool_result_text(jv *b) {
    sbuf r = {0};
    if (jv_bool(jv_get(b, "is_error"), false)) sb_lit(&r, "error: ");
    jv *c = jv_get(b, "content");
    if (c && c->type == J_STR) {
        sb_put(&r, c->str, strlen(c->str));
    } else if (c && c->type == J_ARR) {
        for (int i = 0; i < c->n; i++) {
            const char *txt = jv_str(jv_get(c->items[i], "text"), NULL);
            if (!txt) continue;
            if (r.n) sb_lit(&r, "\n");
            sb_put(&r, txt, strlen(txt));
        }
    } else if (c && c->type != J_NULL) {
        jv_dump(c, &r);
    }
    return r.s ? r.s : strdup("");
}

// Flatten one message's content into turns. Returns false with err set when it
// carries something this runtime cannot render: dropping an image would answer
// a question about content the model never saw, which is exactly the silent
// success this project refuses.
static bool anth_blocks(jv *msg, const char *role, turnbuf *t,
                        char *err, int errcap) {
    jv *content = jv_get(msg, "content");
    if (content && content->type == J_STR) {
        turn_add(t, role, strdup(content->str));
        return true;
    }
    if (!content || content->type != J_ARR) {
        snprintf(err, errcap,
                 "messages[].content must be a string or an array of blocks");
        return false;
    }
    sbuf body = {0};
    for (int i = 0; i < content->n; i++) {
        jv *b = content->items[i];
        if (!b || b->type != J_OBJ) {
            snprintf(err, errcap,
                     "each messages[].content block must be an object");
            free(body.s);
            return false;
        }
        const char *bt = jv_str(jv_get(b, "type"), "");
        if (!strcmp(bt, "text")) {
            const char *txt = jv_str(jv_get(b, "text"), NULL);
            if (!txt) {
                snprintf(err, errcap, "a text block must carry a text string");
                free(body.s);
                return false;
            }
            if (body.n) sb_lit(&body, "\n");
            sb_put(&body, txt, strlen(txt));
        } else if (!strcmp(bt, "tool_use")) {
            // the assistant's own earlier call, replayed: rendered in runner's
            // call syntax so the history reads like what the model emitted
            const char *name = jv_str(jv_get(b, "name"), NULL);
            if (!name) {
                snprintf(err, errcap, "a tool_use block must carry a name");
                free(body.s);
                return false;
            }
            jv *input = jv_get(b, "input");
            sb_fmt(&body, "<|tool_call>call:%s", name);
            if (input && input->type != J_NULL) jv_dump(input, &body);
            else                                sb_lit(&body, "{}");
            sb_lit(&body, "<tool_call|>");
        } else if (!strcmp(bt, "tool_result")) {
            // the tool loop closing: a result is its own turn in the chat
            // vocabulary, so it is emitted ahead of whatever text accompanies
            // it in the same Anthropic message
            turn_add(t, "tool", anth_tool_result_text(b));
        } else if (!strcmp(bt, "thinking") || !strcmp(bt, "redacted_thinking")) {
            // Replayed reasoning. Anthropic wants it back so *it* can verify a
            // signature; there is nothing to verify locally, and it is the
            // model's own scratch work rather than anything the user said, so
            // it is not put back into the prompt.
            continue;
        } else {
            snprintf(err, errcap,
                     "messages[].content block type \"%.40s\" is not supported; "
                     "this runtime renders text, tool_use and tool_result "
                     "blocks only", bt);
            free(body.s);
            return false;
        }
    }
    if (body.n) turn_add(t, role, body.s);
    else        free(body.s);
    return true;
}

// `system` is a string or a list of text blocks. Returns an owned string, or
// NULL when there is no system content (which is not an error).
static char *anth_system_text(jv *system, char *err, int errcap) {
    if (!system || system->type == J_NULL) return NULL;
    if (system->type == J_STR)
        return system->str[0] ? strdup(system->str) : NULL;
    if (system->type != J_ARR) {
        snprintf(err, errcap,
                 "system must be a string or an array of text blocks");
        return NULL;
    }
    sbuf b = {0};
    for (int i = 0; i < system->n; i++) {
        const char *txt = jv_str(jv_get(system->items[i], "text"), NULL);
        if (!txt) continue;
        if (b.n) sb_lit(&b, "\n");
        sb_put(&b, txt, strlen(txt));
    }
    return b.s;
}

// Anthropic declares a tool as {name, description, input_schema} where chat
// nests it under "function" and calls the schema "parameters". Rather than
// teach the envelope compiler a third shape (and risk the two paths already
// using it), the Anthropic form is re-serialised into the nested one and
// re-parsed — exactly what responses_tools does. Returns an owned jv.
static jv *anth_tools(jv *tools, char *err, int errcap) {
    if (!tools || tools->type == J_NULL) return NULL;
    if (tools->type != J_ARR) {
        snprintf(err, errcap, "tools must be an array");
        return NULL;
    }
    sbuf b = {0};
    sb_lit(&b, "[");
    for (int i = 0; i < tools->n; i++) {
        jv *t = tools->items[i];
        if (!t || t->type != J_OBJ) {
            snprintf(err, errcap, "each tools[] entry must be an object");
            free(b.s);
            return NULL;
        }
        // A server tool (web_search_*, computer_*, bash_*, text_editor_*) is
        // named by its `type`; a client function tool has no type at all.
        // Saying so beats leaving the caller waiting for a call that can never
        // come from a runtime with no such capability.
        const char *type = jv_str(jv_get(t, "type"), NULL);
        if (type && strcmp(type, "custom") != 0 && strcmp(type, "function") != 0) {
            snprintf(err, errcap,
                     "tools[].type \"%.40s\" is a server-side tool and is not "
                     "supported; only client tools can run locally", type);
            free(b.s);
            return NULL;
        }
        if (i) sb_lit(&b, ",");
        sb_lit(&b, "{\"type\":\"function\",\"function\":{\"name\":");
        jv *nm = jv_get(t, "name");
        if (nm) jv_dump(nm, &b); else sb_lit(&b, "null");
        jv *desc = jv_get(t, "description");
        if (desc) { sb_lit(&b, ",\"description\":"); jv_dump(desc, &b); }
        jv *params = jv_get(t, "input_schema");
        if (params) { sb_lit(&b, ",\"parameters\":"); jv_dump(params, &b); }
        sb_lit(&b, "}}");
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

// tool_choice is an object in every Anthropic form; chat spells three of the
// four as bare strings. Returns an owned jv, or NULL with err set.
static jv *anth_tool_choice(jv *tc, char *err, int errcap) {
    if (!tc || tc->type == J_NULL) return NULL;
    if (tc->type != J_OBJ) {
        snprintf(err, errcap, "tool_choice must be an object");
        return NULL;
    }
    // "don't disable parallel use" is a request to allow several calls in one
    // turn. The envelope is one call per turn on every surface, so this is
    // refused here exactly as parallel_tool_calls:true is on the other two,
    // rather than answered with a single call the caller cannot distinguish
    // from a considered choice.
    jv *par = jv_get(tc, "disable_parallel_tool_use");
    if (par && par->type != J_NULL) {
        if (par->type != J_BOOL) {
            snprintf(err, errcap,
                     "tool_choice.disable_parallel_tool_use must be a boolean");
            return NULL;
        }
        if (!par->b) {
            snprintf(err, errcap,
                     "tool_choice.disable_parallel_tool_use:false is not "
                     "supported yet; this runtime emits one tool call per turn. "
                     "Omit it or send true.");
            return NULL;
        }
    }
    const char *type = jv_str(jv_get(tc, "type"), NULL);
    const char *mapped = NULL;
    if (!type) {
        snprintf(err, errcap, "tool_choice.type is required");
        return NULL;
    }
    if (!strcmp(type, "auto")) mapped = "\"auto\"";
    else if (!strcmp(type, "none")) mapped = "\"none\"";
    else if (!strcmp(type, "any")) mapped = "\"required\"";  // any one tool
    else if (strcmp(type, "tool") != 0) {
        snprintf(err, errcap,
                 "tool_choice.type must be \"auto\", \"any\", \"tool\" or \"none\"");
        return NULL;
    }
    sbuf b = {0};
    if (mapped) {
        sb_lit(&b, mapped);
    } else {
        const char *name = jv_str(jv_get(tc, "name"), NULL);
        if (!name || !name[0]) {
            snprintf(err, errcap,
                     "tool_choice.type \"tool\" requires a name");
            return NULL;
        }
        sb_lit(&b, "{\"type\":\"function\",\"function\":{\"name\":\"");
        sb_esc(&b, name, strlen(name));
        sb_lit(&b, "\"}}");
    }
    jv *out = b.failed || !b.s ? NULL : json_parse(b.s, b.n);
    free(b.s);
    if (!out) snprintf(err, errcap, "out of memory translating tool_choice");
    return out;
}

// Features with no local implementation. Refusing them is the project
// invariant: a client told 200 believes the thing it asked for happened.
static bool anth_reject_unsupported(slot_t *s, sock_t fd, jv *req) {
    jv *v = jv_get(req, "mcp_servers");
    if (v && v->type == J_ARR && v->n > 0) {
        send_error(fd, 400,
                   "mcp_servers is not supported: this runtime cannot reach "
                   "remote MCP servers on your behalf. Run the tools locally "
                   "and declare them in tools[].");
        return true;
    }
    v = jv_get(req, "container");
    if (v && v->type != J_NULL) {
        send_error(fd, 400,
                   "container is not supported: there is no code-execution "
                   "container behind this runtime.");
        return true;
    }
    v = jv_get(req, "metadata");
    if (v && v->type != J_NULL && v->type != J_OBJ) {
        send_error(fd, 400, "metadata must be an object");
        return true;
    }
    // `thinking` promises the turn will carry thinking blocks. Whether it can
    // is a property of the resident model — a thinking-tagged one separates
    // its reasoning channel already — so it is answered honestly per model
    // rather than accepted and quietly not done.
    v = jv_get(req, "thinking");
    if (v && v->type != J_NULL) {
        if (v->type != J_OBJ) {
            send_error(fd, 400, "thinking must be an object");
            return true;
        }
        const char *type = jv_str(jv_get(v, "type"), NULL);
        if (!type || (strcmp(type, "enabled") && strcmp(type, "disabled") &&
                      strcmp(type, "adaptive"))) {
            send_error(fd, 400,
                       "thinking.type must be \"enabled\", \"disabled\" or "
                       "\"adaptive\"");
            return true;
        }
        if (!strcmp(type, "enabled") && !s->m->think_open) {
            send_error(fd, 400,
                       "thinking:enabled is not supported by the resident "
                       "model: it has no reasoning channel to separate, so no "
                       "thinking block could be returned.");
            return true;
        }
    }
    return false;
}

// Build the prompt one Messages request asks for. Shared by /v1/messages and
// /v1/messages/count_tokens so the count is necessarily the count of the
// prompt the real request would have run — the two cannot drift.
//
// Returns a heap prompt on success (caller frees) with *strict/*env set, or
// NULL having already answered `fd` with the error.
static char *messages_prompt(slot_t *s, sock_t fd, jv *req, tool_envelope *env,
                             bool *strict) {
    char terr[224];
    *strict = false;
    memset(env, 0, sizeof(*env));

    jv *msgs = jv_get(req, "messages");
    if (!msgs || msgs->type != J_ARR || msgs->n == 0) {
        send_error(fd, 400, "missing messages");
        return NULL;
    }

    jv *tools = anth_tools(jv_get(req, "tools"), terr, sizeof(terr));
    jv *raw_tools = jv_get(req, "tools");
    if (raw_tools && raw_tools->type != J_NULL && !tools) {
        send_error(fd, 400, terr);
        return NULL;
    }
    jv *choice = anth_tool_choice(jv_get(req, "tool_choice"), terr, sizeof(terr));
    jv *raw_choice = jv_get(req, "tool_choice");
    if (raw_choice && raw_choice->type != J_NULL && !choice) {
        jv_free(tools);
        send_error(fd, 400, terr);
        return NULL;
    }

    // the same envelope compiler, from the same declarations, as both OpenAI
    // surfaces: this is what makes an Anthropic tool call and a chat tool call
    // the same internal agent action
    int rc = tool_envelope_build(tools, choice, NULL, env, terr, sizeof(terr));
    if (rc < 0) {
        jv_free(tools);
        jv_free(choice);
        send_error(fd, 400, terr);
        return NULL;
    }
    *strict = rc == 1;

    sbuf ts = {0};
    if (*strict) sb_put(&ts, env->system_turn, strlen(env->system_turn));
    else         tools_render(tools, &ts);

    // upper bound on turns: the tool system turn, the system turn, and for
    // each message its own turn plus one per tool_result block it carries
    int cap = 2 + msgs->n;
    for (int i = 0; i < msgs->n; i++) {
        jv *c = jv_get(msgs->items[i], "content");
        if (c && c->type == J_ARR) cap += c->n;
    }
    turnbuf t = { .cm = malloc(sizeof(chat_msg) * (size_t)cap),
                  .owned = malloc(sizeof(char *) * (size_t)cap),
                  .cap = cap, .total = ts.n + 128 };
    if (!t.cm || !t.owned) t.failed = true;

    char *sys = NULL;
    if (!t.failed) {
        if (ts.n) turn_add_borrowed(&t, "system", ts.s);
        terr[0] = 0;
        sys = anth_system_text(jv_get(req, "system"), terr, sizeof(terr));
        if (terr[0]) {
            turnbuf_free(&t);
            free(ts.s);
            tool_envelope_free(env);
            jv_free(tools);
            jv_free(choice);
            send_error(fd, 400, terr);
            return NULL;
        }
        if (sys) turn_add(&t, "system", sys);
    }

    bool ok = !t.failed;
    for (int i = 0; ok && i < msgs->n; i++) {
        jv *msg = msgs->items[i];
        const char *role = jv_str(jv_get(msg, "role"), NULL);
        if (!role || (strcmp(role, "user") && strcmp(role, "assistant") &&
                      strcmp(role, "system"))) {
            snprintf(terr, sizeof(terr),
                     "messages[].role must be \"user\", \"assistant\" or "
                     "\"system\"");
            ok = false;
            break;
        }
        ok = anth_blocks(msg, role, &t, terr, sizeof(terr));
    }
    if (ok && t.failed) {
        snprintf(terr, sizeof(terr), "out of memory building the prompt");
        ok = false;
    }
    if (ok && t.n == 0) {
        snprintf(terr, sizeof(terr), "no message content");
        ok = false;
    }
    char *prompt = NULL;
    if (ok) {
        prompt = malloc(t.total + 256);
        if (prompt) render_messages(s->tmpl, t.cm, t.n, true, prompt,
                                    t.total + 256);
        else ok = false;
    }
    turnbuf_free(&t);
    free(ts.s);
    jv_free(tools);
    jv_free(choice);
    if (!ok) {
        tool_envelope_free(env);
        free(prompt);
        send_error(fd, 400, terr[0] ? terr : "cannot build prompt");
        return NULL;
    }
    return prompt;
}

static void handle_messages(slot_t *s, sock_t fd, jv *req) {
    if (anth_reject_unsupported(s, fd, req)) return;
    // max_tokens is required on this surface, unlike the OpenAI ones where it
    // defaults. A caller that forgot it wants a cap, not the server's.
    jv *mt = jv_get(req, "max_tokens");
    if (!mt || mt->type == J_NULL) {
        send_error(fd, 400, "max_tokens is required");
        return;
    }
    tool_envelope env;
    bool strict = false;
    char *prompt = messages_prompt(s, fd, req, &env, &strict);
    if (!prompt) return;
    run_completion(s, fd, prompt, API_MESSAGES, req, strict ? &env : NULL);
    free(prompt);
    tool_envelope_free(&env);
}

// POST /v1/messages/count_tokens: how many input tokens this exact request
// would have cost. It runs the whole inbound translation and stops before
// generation, so the answer is the prompt the request would really have used.
static void handle_count_tokens(slot_t *s, sock_t fd, jv *req) {
    if (anth_reject_unsupported(s, fd, req)) return;
    tool_envelope env;
    bool strict = false;
    char *prompt = messages_prompt(s, fd, req, &env, &strict);
    if (!prompt) return;
    size_t cap = strlen(prompt) + 16;
    int32_t *toks = malloc(sizeof(int32_t) * cap);
    int n = toks ? tok_encode(s->tok, prompt, toks, (int)cap, true, true) : -1;
    free(toks);
    free(prompt);
    tool_envelope_free(&env);
    if (n < 0) { send_error(fd, 500, "out of memory tokenizing prompt"); return; }
    if (n == 0) { send_error(fd, 400, "empty prompt"); return; }
    sbuf r = {0};
    sb_fmt(&r, "{\"input_tokens\":%d}", n);
    send_built(fd, &r);
    free(r.s);
}

static void handle_completion(slot_t *s, sock_t fd, jv *req) {
    const char *prompt = jv_str(jv_get(req, "prompt"), NULL);
    if (!prompt) { send_error(fd, 400, "missing prompt"); return; }
    run_completion(s, fd, prompt, API_TEXT, req, NULL);
}

static void handle_embeddings(slot_t *s, sock_t fd, jv *req) {
    jv *input = jv_get(req, "input");
    const char *one = jv_str(input, NULL);
    int n_in = one ? 1 : (input && input->type == J_ARR ? input->n : 0);
    if (n_in == 0) { send_error(fd, 400, "missing input"); return; }

    model_t *m = s->m;
    jv *encoding = jv_get(req, "encoding_format");
    if (!absent(encoding) &&
        (encoding->type != J_STR || strcmp(encoding->str, "float") != 0)) {
        send_error(fd, 400, "encoding_format must be float");
        return;
    }
    jv *dimensions = jv_get(req, "dimensions");
    if (!absent(dimensions) &&
        (dimensions->type != J_NUM || !isfinite(dimensions->num) ||
         dimensions->num != m->n_embd)) {
        send_error(fd, 400, "dimensions must equal the model embedding size");
        return;
    }
    float *emb = malloc(sizeof(float) * m->n_embd);
    if (!emb) { send_error(fd, 500, "out of memory"); return; }  // RNS-3
    sbuf r = {0};
    sb_lit(&r, "{\"object\":\"list\",\"data\":[");
    int total = 0;
    bool ok = true;
    // RNS-1: an error must be reported to the client only AFTER the device turn
    // is released. send_error() does a blocking socket write bounded by the 30s
    // SO_SNDTIMEO; doing it under dev_mu would stall decode_worker — and thereby
    // every other in-flight generation — behind one slow/dead embeddings client.
    // So the loop records (err_code, err_msg) and breaks; the send happens below,
    // off-lock, mirroring run_completion.
    int err_code = 0;
    const char *err_msg = NULL;
    // model_embed launches kernels and engine_reset touches this slot's KV —
    // the same device work every other handler serializes under dev_mu. Take the
    // device turn so an embeddings request cannot launch while decode_worker is
    // capturing a CUDA graph (hazard at dev_mu, ~line 596). Every exit from the
    // loop below is a break, so the single sched_prefill_end() after the loop
    // always runs — the lock is never left held.
    sched_prefill_begin();
    for (int k = 0; k < n_in && ok; k++) {
        const char *txt = one ? one : jv_str(input->items[k], NULL);
        if (!txt) { err_code = 400; err_msg = "input must be strings"; ok = false; break; }
        size_t cap = strlen(txt) + 16;
        int32_t *toks = malloc(sizeof(int32_t) * cap);
        int n = toks ? tok_encode(s->tok, txt, toks, (int)cap, true, true) : -1;
        if (n < 0) {
            free(toks);
            err_code = 500; err_msg = "out of memory tokenizing input";
            ok = false;
            break;
        }
        if (n == 0 || !model_embed(m, toks, n, emb)) {
            free(toks);
            err_code = 400;
            err_msg = n == 0 ? "empty input" : "input exceeds context window";
            ok = false;
            break;
        }
        free(toks);
        total += n;
        sb_fmt(&r, "%s{\"object\":\"embedding\",\"index\":%d,\"embedding\":[",
               k ? "," : "", k);
        for (int j = 0; j < m->n_embd; j++)
            sb_fmt(&r, "%s%.7g", j ? "," : "", emb[j]);
        sb_lit(&r, "]}");
    }
    // model_embed overwrote this slot's KV cache — invalidate the prefix cache
    engine_reset(&s->e);
    s->e.pos = 0;
    sched_prefill_end();
    if (!ok) {
        // deferred from the loop — sent here, off the device lock (RNS-1)
        send_error(fd, err_code, err_msg);
    } else {
        sb_lit(&r, "],\"model\":\"");
        sb_esc(&r, SV.model_name, strlen(SV.model_name));
        sb_fmt(&r, "\",\"usage\":{\"prompt_tokens\":%d,\"total_tokens\":%d}}",
               total, total);
        send_built(fd, &r);
        fprintf(stderr, "[slot %d] embeddings: %d input(s), %d tok\n",
                s->id, n_in, total);
    }
    free(r.s);
    free(emb);
}

// ---------------------------------------------------------------- http

// /health and /v1/models read only startup-immutable strings plus an atomic
// resident snapshot, so they are safe to answer from the accept thread with no lock
static void send_health(sock_t fd) {
    char b[384];
    int n, res = resident_load();
    if (SV.n_reg > 0 && res >= 0) {
        char esc[192];
        json_escape(SV.reg[res].name, strlen(SV.reg[res].name), esc, sizeof(esc));
        n = snprintf(b, sizeof(b), "{\"status\":\"ok\",\"resident\":\"%s\"}", esc);
    } else if (SV.n_reg > 0) {
        n = snprintf(b, sizeof(b), "{\"status\":\"ok\",\"resident\":null}");
    } else {
        n = snprintf(b, sizeof(b), "{\"status\":\"ok\"}");
    }
    send_response(fd, 200, "application/json", b, n);
}

static void send_models(sock_t fd) {
    sbuf r = {0};
    sb_lit(&r, "{\"object\":\"list\",\"data\":[");
    if (SV.n_reg > 0) {
        for (int i = 0; i < SV.n_reg; i++) {
            char esc[192];
            json_escape(SV.reg[i].name, strlen(SV.reg[i].name), esc, sizeof(esc));
            sb_fmt(&r, "%s{\"id\":\"%s\",\"object\":\"model\","
                       "\"owned_by\":\"runner\"}", i ? "," : "", esc);
        }
    } else {
        char esc[256];
        json_escape(SV.model_name, strlen(SV.model_name), esc, sizeof(esc));
        sb_fmt(&r, "{\"id\":\"%s\",\"object\":\"model\",\"owned_by\":\"runner\"}", esc);
    }
    sb_lit(&r, "]}");
    send_built(fd, &r);
    free(r.s);
}

// Server-wide prefix-cache telemetry. Per-request telemetry says what one
// request saved; this says whether the cache is earning its memory —
// hit rate, resident bytes against the budget, and the prefill time it has
// avoided so far.
static void send_prefix_cache(sock_t fd) {
    prefix_cache_stats st;
    prefix_cache_stats_get(&st);
    char b[640];
    int n = snprintf(b, sizeof(b),
        "{\"object\":\"runner.prefix_cache\","
        "\"enabled\":%s,\"entries\":%d,\"bytes\":%llu,\"budget_bytes\":%llu,"
        "\"ttl_seconds\":%.1f,\"hits\":%llu,\"misses\":%llu,\"stores\":%llu,"
        "\"evictions\":%llu,\"tokens_reused\":%llu,"
        "\"saved_prefill_seconds\":%.6f,\"prefill_seconds_per_token\":%.9f}",
        st.budget ? "true" : "false", st.entries,
        (unsigned long long)st.bytes, (unsigned long long)st.budget, st.ttl,
        (unsigned long long)st.hits, (unsigned long long)st.misses,
        (unsigned long long)st.stores, (unsigned long long)st.evictions,
        (unsigned long long)st.tokens_reused,
        st.saved_prefill_s, st.cost_per_token_s);
    send_response(fd, 200, "application/json", b, n);
}

static void send_capabilities(sock_t fd) {
    sbuf r = {0};
    int res = resident_load();
    sb_lit(&r, "{\"object\":\"runner.capabilities\",\"swap\":");
    sb_lit(&r, SV.n_reg > 0 && !SV.single ? "true" : "false");
    sb_lit(&r, ",\"resident\":");
    if (SV.n_reg > 0 && res >= 0) {
        sb_lit(&r, "\"");
        sb_esc(&r, SV.reg[res].name, strlen(SV.reg[res].name));
        sb_lit(&r, "\"");
    } else {
        sb_lit(&r, "null");
    }
    sb_fmt(&r, ",\"context\":%d,\"models\":[", context_load());
    if (SV.n_reg > 0) {
        for (int i = 0; i < SV.n_reg; i++) {
            if (i) sb_lit(&r, ",");
            sb_lit(&r, "{\"id\":\"");
            sb_esc(&r, SV.reg[i].name, strlen(SV.reg[i].name));
            sb_fmt(&r, "\",\"resident\":%s}", i == res ? "true" : "false");
        }
    } else {
        sb_lit(&r, "{\"id\":\"");
        sb_esc(&r, SV.model_name, strlen(SV.model_name));
        sb_lit(&r, "\",\"resident\":true}");
    }
    model_t *pm = res >= 0 ? SV.slots[0].m : NULL;
    sb_lit(&r, "],\"agent_profile\":");
    if (!pm || !pm->agent_profile) {
        sb_lit(&r, "null");
    } else {
        sb_fmt(&r, "{\"protocol_version\":%u,\"tokenizer_version\":%u,\"schema_id\":\"",
               pm->agent_protocol_version, pm->agent_tokenizer_version);
        sb_esc(&r, pm->agent_schema_id, strlen(pm->agent_schema_id));
        sb_lit(&r, "\",\"schema_digest\":\"");
        sb_esc(&r, pm->agent_schema_digest, strlen(pm->agent_schema_digest));
        sb_lit(&r, "\",\"required_features\":[");
        for (uint64_t i = 0; i < pm->n_agent_required_features; i++) {
            if (i) sb_lit(&r, ",");
            sb_lit(&r, "\"");
            sb_esc(&r, pm->agent_required_features[i].s,
                   pm->agent_required_features[i].n);
            sb_lit(&r, "\"");
        }
        sb_lit(&r, "]}");
    }
    // Admitted-but-unconsumed MTP heads: a controller can see that the export
    // carries predictor blocks and that this build excluded them, rather than
    // inferring it from a layer count that silently differs from the card.
    sb_fmt(&r, ",\"mtp\":{\"declared_layers\":%d,\"consumed\":false}",
           pm ? pm->mtp_layers : 0);
    sb_lit(&r, ",\"sampling\":{\"preset\":");
    if (SV.preset_name) {
        sb_lit(&r, "\"");
        sb_esc(&r, SV.preset_name, strlen(SV.preset_name));
        sb_lit(&r, "\"");
    } else {
        sb_lit(&r, "null");  // swap mode before the first model is resident
    }
    {
        const sampler *d = &SV.slots[0].smp_base;
        sb_fmt(&r, ",\"temperature\":%.2f,\"top_p\":%.2f,\"top_k\":%d,"
                   "\"min_p\":%.2f,\"repeat_penalty\":%.2f}",
               (double)d->temp, (double)d->top_p, d->top_k,
               (double)d->min_p, (double)d->repeat_penalty);
    }
    sb_lit(&r, ",\"features\":{"
               "\"responses_api\":true,"
               "\"messages_api\":true,"
               "\"json_object\":true,"
               "\"json_schema\":true,"
               "\"stop_sequences\":true,"
               "\"schema_conditionals\":true,"
               "\"schema_string_bounds\":true,"
               "\"schema_integer_bounds\":true,"
               "\"request_telemetry\":true,"
               "\"prefix_cache\":true,"
               "\"prefix_cache_controls\":true,"
               "\"shared_prefix_cache\":true,"
               "\"forkable_prefixes\":true,"
               "\"repeat_penalty\":true,"
               "\"family_sampling_presets\":true}}");
    send_built(fd, &r);
    free(r.s);
}

static void handle_conn(slot_t *s, sock_t fd) {
    // a stalled or dead client must not pin an inference slot: the whole
    // request (header + body) has to arrive within this budget. Generation
    // time stays unbounded — the deadline only covers reading the request.
    //
    // The write side needs its own bound. A client that sends a valid request
    // and then stops reading fills the socket buffer, and an unbounded
    // blocking write parks the slot forever: the read deadline is already
    // satisfied and never fires again, so the slot is never returned.
    sock_send_timeout(fd, 30.0);
    double deadline = now_s() + 10.0;
    char hdr[16384];
    size_t got = 0;
    char *body_start = NULL;
    while (got < sizeof(hdr) - 1) {
        double remaining = deadline - now_s();
        if (remaining <= 0) { send_error(fd, 408, "request read timed out"); return; }
        sock_recv_timeout(fd, remaining);
        int r = sock_recv(fd, hdr + got, sizeof(hdr) - 1 - got);
        if (r <= 0) {
            // r == 0: orderly close, client is gone. r < 0: timeout (or a
            // socket error, where the 408 write fails harmlessly).
            if (r < 0) send_error(fd, 408, "request read timed out");
            return;
        }
        got += (size_t)r;
        hdr[got] = 0;
        if ((body_start = strstr(hdr, "\r\n\r\n")) != NULL) break;
    }
    if (!body_start) { send_error(fd, 400, "bad request"); return; }
    char *header_end = body_start;
    body_start += 4;

    char method[8] = {0}, path[256] = {0};
    char *first_header = NULL;
    if (!parse_request_line(hdr, method, path, &first_header)) {
        send_error(fd, 400, "malformed request line");
        return;
    }
    // Route on the path component. SDKs use query parameters for protocol
    // feature selection — Claude Code, for example, sends
    // `/v1/messages?beta=true`. The query does not rename the resource and is
    // deliberately not interpreted by this stateless inference boundary.
    char *query = strchr(path, '?');
    if (query) *query = 0;

    size_t content_length = 0;
    if (!parse_request_framing(first_header, header_end, &content_length)) {
        send_error(fd, 400, "invalid request framing");
        return;
    }
    if (content_length > 32u * 1024 * 1024) {
        send_error(fd, 400, "body too large");
        return;
    }

    char *body = NULL;
    if (content_length > 0) {
        body = malloc(content_length + 1);
        if (!body) { send_error(fd, 500, "cannot allocate request body"); return; }
        size_t have = got - (size_t)(body_start - hdr);
        if (have > content_length) have = content_length;
        memcpy(body, body_start, have);
        while (have < content_length) {
            double remaining = deadline - now_s();
            if (remaining <= 0) {
                free(body);
                send_error(fd, 408, "request read timed out");
                return;
            }
            sock_recv_timeout(fd, remaining);
            int r = sock_recv(fd, body + have, content_length - have);
            if (r <= 0) {
                free(body);
                if (r < 0) send_error(fd, 408, "request read timed out");
                return;
            }
            have += (size_t)r;
        }
        body[content_length] = 0;
    }

    if (!strcmp(method, "GET") && !strcmp(path, "/unload")) {
        // llama-swap-compatible: free the resident model's memory. Normally
        // answered straight from the accept loop (see accept_fastpath); this
        // path still serves a request that slipped past it.
        handle_unload(fd);
    } else if (!strcmp(method, "GET") &&
               !strcmp(path, "/v1/runner/prefix-cache")) {
        send_prefix_cache(fd);
    } else if (!strcmp(method, "POST") &&
               !strcmp(path, "/v1/runner/prefix-cache/clear")) {
        // Explicit release, for benchmarks that need a cold cache and for
        // operators reclaiming the memory without unloading the model.
        prefix_cache_clear();
        send_prefix_cache(fd);
    } else if (!strcmp(method, "GET") && !strcmp(path, "/health")) {
        send_health(fd);
    } else if (!strcmp(method, "GET") && !strcmp(path, "/v1/models")) {
        send_models(fd);
    } else if (!strcmp(method, "GET") && !strcmp(path, "/v1/capabilities")) {
        send_capabilities(fd);
    } else if (!strcmp(method, "POST") &&
               (!strcmp(path, "/v1/chat/completions") ||
                !strcmp(path, "/v1/responses") ||
                !strcmp(path, "/v1/messages") ||
                !strcmp(path, "/v1/messages/count_tokens") ||
                !strcmp(path, "/v1/completions") ||
                !strcmp(path, "/v1/embeddings"))) {
        jv *req = body ? json_parse(body, content_length) : NULL;
        if (!req) {
            send_error(fd, 400, "invalid JSON body");
        } else {
            bool has_keep_alive = false;
            int keep_alive = 0;
            if (!request_keep_alive(req, &has_keep_alive, &keep_alive)) {
                send_error(fd, 400, "keep_alive out of range");
                jv_free(req);
                free(body);
                return;
            }
            atomic_fetch_add(&SV.active_requests, 1);
            bool ok = true;
            if (SV.n_reg > 0) {
                int sw = swap_to(jv_str(jv_get(req, "model"), NULL));
                if (sw == SWAP_LOAD_FAILED) {
                    send_error(fd, 500,
                               "model failed to load (registered but broken; see server log)");
                    ok = false;
                } else if (sw == SWAP_ABORTED) {
                    send_error(fd, 503,
                               "model load abandoned (unload or shutdown requested; retry)");
                    ok = false;
                } else if (sw < 0) {
                    send_error_detail(fd, 404,
                                      "unknown model (see /v1/models)",
                                      "model", "model_not_found");
                    ok = false;
                }
            } else if (!validate_single_model_request(fd, req)) {
                ok = false;
            }
            if (ok) {
                if (strcmp(path, "/v1/chat/completions") == 0) handle_chat(s, fd, req);
                else if (strcmp(path, "/v1/responses") == 0) handle_responses(s, fd, req);
                else if (strcmp(path, "/v1/messages") == 0) handle_messages(s, fd, req);
                else if (strcmp(path, "/v1/messages/count_tokens") == 0)
                    handle_count_tokens(s, fd, req);
                else if (strcmp(path, "/v1/embeddings") == 0) handle_embeddings(s, fd, req);
                else handle_completion(s, fd, req);
                // Ollama-style keep_alive: seconds of idle before the model
                // unloads (swap mode) — 0 unloads now, negative pins forever
                if (has_keep_alive && SV.n_reg > 0) {
                    pthread_mutex_lock(&SV.swap_mu);
                    if (keep_alive == 0) unload_resident();
                    else SV.ttl = keep_alive < 0 ? 0 : keep_alive;
                    pthread_mutex_unlock(&SV.swap_mu);
                }
            }
            // Drop the request from the count BEFORE the bookkeeping lock:
            // an /unload that saw this request active left pending_unload for
            // us, and the count must already be zero when we honour it.
            atomic_fetch_sub(&SV.active_requests, 1);
            if (SV.n_reg > 0) {
                bool unloaded = false;
                pthread_mutex_lock(&SV.swap_mu);
                SV.last_used = now_s();
                if (SV.pending_unload && !SV.loading &&
                    !atomic_load(&SV.active_requests)) {
                    unload_draft();
                    unload_resident();
                    SV.pending_unload = false;
                    unloaded = true;
                }
                pthread_mutex_unlock(&SV.swap_mu);
                if (unloaded) prefix_cache_clear();
            }
            jv_free(req);
        }
    } else {
        send_error(fd, 404, "not found");
    }
    free(body);
}

static void *slot_worker(void *arg) {
    slot_t *s = arg;
    for (;;) {
        sock_t fd = q_pop();
        if (fd == SOCK_INVALID) return NULL;
        handle_conn(s, fd);
        sock_close(fd);
    }
}

// answer tiny GETs from the accept loop: single-slot serving means one long
// generation used to block /health until the gridcore watchdog declared a
// live runner "unhealthy: timed out". /unload is answered here too — it never
// frees anything a slot is using (handle_unload defers under an active load
// or generation), and an operator reclaiming memory must not queue behind
// the very work that holds it. POSTs are handed to a slot untouched.
static bool accept_fastpath(sock_t fd) {
#ifndef _WIN32
    // POSIX fd_set is a fixed-size bitmask indexed by fd value; FD_SET on an
    // fd >= FD_SETSIZE is undefined behavior (out-of-bounds write). Windows
    // fd_set is a count-based array instead, so it isn't affected.
    if (fd >= FD_SETSIZE) return false;
#endif
    fd_set rs;
    struct timeval tv = { 0, 250000 }; // loopback data lands in <1ms
    FD_ZERO(&rs);
    FD_SET(fd, &rs);
    if (select(fd + 1, &rs, NULL, NULL, &tv) != 1) return false;
    char hdr[2048];
    int n = sock_peek(fd, hdr, 64);
    if (n <= 0) { sock_close(fd); return true; } // died before speaking
    hdr[n] = 0;
    // match the path plus the space HTTP/1.x always puts before the version,
    // so "GET /healthzzz" falls through to the slot path instead of being
    // misrouted here. Bare HTTP/0.9 "GET /health\r\n" (no version) won't
    // match, but neither curl nor the gridcore watchdog send that, so it's
    // not worth the extra branch.
    bool health = !strncmp(hdr, "GET /health ", 12);
    bool models = !strncmp(hdr, "GET /v1/models ", 15);
    bool caps = !strncmp(hdr, "GET /v1/capabilities ", 21);
    bool unload = !strncmp(hdr, "GET /unload ", 12);
    if (!health && !models && !caps && !unload) return false;
    // Drain the request before replying: closing with unread bytes can RST
    // the connection and discard our response. But the accept thread must
    // never block indefinitely on a single connection — it's the only thing
    // calling accept(), so a stalled client here would queue every later
    // connection behind it, reproducing the watchdog-timeout bug this
    // fastpath exists to fix. Re-select before each recv with a short
    // timeout and cap the total drain time; if a client dribbles bytes too
    // slowly to finish the header in the budget, answer anyway (these GETs
    // are tiny and read-only, so a reply is always correct) and move on.
    double deadline = now_s() + 0.5;
    size_t got = 0;
    while (got < sizeof(hdr) - 1 && !strstr(hdr, "\r\n\r\n")) {
        double remaining = deadline - now_s();
        if (remaining <= 0) break;
        struct timeval dtv;
        dtv.tv_sec = 0;
        dtv.tv_usec = (long)(remaining * 1e6);
        if (dtv.tv_usec > 100000) dtv.tv_usec = 100000; // poll in <=100ms steps
        FD_ZERO(&rs);
        FD_SET(fd, &rs);
        if (select(fd + 1, &rs, NULL, NULL, &dtv) != 1) break;
        int r = sock_recv(fd, hdr + got, sizeof(hdr) - 1 - got);
        if (r <= 0) break;
        got += (size_t)r;
        hdr[got] = 0;
    }
    char *header_end = strstr(hdr, "\r\n\r\n");
    if (!header_end) {
        send_error(fd, 400, "bad request");
        sock_close(fd);
        return true;
    }
    char method[8] = {0}, path[256] = {0};
    char *first_header = NULL;
    size_t content_length = 0;
    if (!parse_request_line(hdr, method, path, &first_header) ||
        !parse_request_framing(first_header, header_end, &content_length) ||
        content_length > 32u * 1024 * 1024) {
        send_error(fd, 400, "invalid request framing");
        sock_close(fd);
        return true;
    }
    if (health) send_health(fd);
    else if (models) send_models(fd);
    else if (unload) handle_unload(fd);
    else        send_capabilities(fd);
    sock_close(fd);
    return true;
}

// ---------------------------------------------------------------- entry

#ifndef _WIN32
static volatile sig_atomic_t stop_requested;
static volatile sig_atomic_t listener_fd = -1;

static void stop_handler(int sig) {
    (void)sig;
    // A second signal is the operator overruling the drain: exit now. _exit is
    // async-signal-safe (128+SIGINT — the shell's convention for a Ctrl-C kill),
    // where the alternative on a pinned drain was reaching for SIGKILL.
    if (stop_requested) _exit(130);
    stop_requested = 1;
    int fd = (int)listener_fd;
    if (fd >= 0) {
        listener_fd = -1;
        close(fd); // async-signal-safe; wakes accept()
    }
}

static void install_stop_handlers(void) {
    struct sigaction sa = {0};
    sa.sa_handler = stop_handler;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);
}

// Between the handlers installing and the listener publishing there is nothing
// a signal can close, so startup polls this at its long stops (model loads)
// and abandons the launch instead of serving a request nobody wants anymore.
static bool stop_was_requested(void) { return stop_requested != 0; }
#else
// Windows port of the same design: stop flag + listener close + drain +
// second-signal escalation, via SetConsoleCtrlHandler. The handler runs on
// its own thread (not an async-signal context), so plain volatile stores and
// closesocket() are safe here.
static volatile LONG win_stop_requested;
static volatile SOCKET win_listener_socket = INVALID_SOCKET;

static BOOL WINAPI win_stop_handler(DWORD ctrl_type) {
    switch (ctrl_type) {
    case CTRL_C_EVENT:
    case CTRL_BREAK_EVENT:
    case CTRL_CLOSE_EVENT:
    case CTRL_SHUTDOWN_EVENT: {
        // A second Ctrl-C is the operator overruling the drain: exit now,
        // with the shell's 128+SIGINT convention for parity with POSIX.
        if (InterlockedExchange(&win_stop_requested, 1)) _exit(130);
        SOCKET s = win_listener_socket;
        if (s != INVALID_SOCKET) {
            win_listener_socket = INVALID_SOCKET;
            closesocket(s); // wakes accept()
        }
        if (ctrl_type == CTRL_CLOSE_EVENT || ctrl_type == CTRL_SHUTDOWN_EVENT) {
            // Returning from these lets Windows terminate the process
            // immediately; hold the handler thread briefly so the drain in
            // main gets its window (Windows grants ~5s on close).
            Sleep(4000);
        }
        return TRUE; // handled: keep running so the drain can finish
    }
    default:
        return FALSE;
    }
}

static void install_stop_handlers(void) {
    win_stop_requested = 0;
    win_listener_socket = INVALID_SOCKET;
    SetConsoleCtrlHandler(win_stop_handler, TRUE);
}

static bool stop_was_requested(void) { return win_stop_requested != 0; }
#endif

int server_run(model_t *base, tokenizer *tok, const char *model_path,
               const model_params *mp, sampler defaults,
               const sampler_override *ov, int port, int parallel,
               int n_threads, int ttl, const char *draft_path, int draft_k) {
    sock_init();
#ifndef _WIN32
    stop_requested = 0;
    listener_fd = -1;
#endif
    install_stop_handlers(); // resets the stop flag + listener on both platforms
    if (ov) SV.ov = *ov;
    // `defaults` arrives already resolved against the preloaded model; in swap
    // mode there is no model yet and swap_to resolves per load
    if (base) {
        char ident[256];
        sampler_ident(gguf_get_str(&base->gf, "general.name", NULL),
                      base->path, ident, sizeof(ident));
        SV.preset_name = sampler_preset_for(base->arch, ident)->name;
    } else {
        SV.preset_name = NULL;
    }
    if (parallel < 1) parallel = 1;
    if (parallel > 16) parallel = 16;
    bool swap_mode = strchr(model_path, '=') != NULL;
    if (ttl < 0) ttl = swap_mode ? 300 : 0; // single-model default: never unload
    if (draft_path && swap_mode) {
        fprintf(stderr, "note: --draft needs a single served model — "
                "ignoring it in swap mode\n");
        draft_path = NULL;
    }

    // "name=path,name2=path2" enables swap mode: one resident model,
    // loaded per request's "model" field, unloaded after ttl idle seconds
    if (strchr(model_path, '=')) {
        // Validate every limit and reject the whole spec with an exact reason
        // rather than silently truncating a name/path (which can collide or
        // select the wrong file) or dropping entries past the cap (RNR-014).
        const int max_reg = (int)(sizeof(SV.reg) / sizeof(SV.reg[0]));
        char tmp[4096];
        if (strlen(model_path) >= sizeof(tmp)) {
            fprintf(stderr, "error: -m registry spec is too long (max %zu bytes)\n",
                    sizeof(tmp) - 1);
            return 1;
        }
        snprintf(tmp, sizeof(tmp), "%s", model_path);
        for (char *tk = strtok(tmp, ","); tk; tk = strtok(NULL, ",")) {
            if (SV.n_reg >= max_reg) {
                fprintf(stderr, "error: too many models in -m (max %d)\n", max_reg);
                return 1;
            }
            char *eq = strchr(tk, '=');
            if (!eq) { fprintf(stderr, "error: bad registry entry '%s' (want name=path)\n", tk); return 1; }
            *eq = 0;
            const char *nm = tk, *pth = eq + 1;
            if (!*nm || !*pth) {
                fprintf(stderr, "error: registry entry '%s=%s' has an empty name or path\n", nm, pth);
                return 1;
            }
            if (strlen(nm) >= sizeof(SV.reg[0].name)) {
                fprintf(stderr, "error: model name '%s' is too long (max %zu chars)\n",
                        nm, sizeof(SV.reg[0].name) - 1);
                return 1;
            }
            if (strlen(pth) >= sizeof(SV.reg[0].path)) {
                fprintf(stderr, "error: model path for '%s' is too long (max %zu chars)\n",
                        nm, sizeof(SV.reg[0].path) - 1);
                return 1;
            }
            for (int j = 0; j < SV.n_reg; j++)
                if (!strcmp(SV.reg[j].name, nm)) {
                    fprintf(stderr, "error: duplicate model name '%s' in -m\n", nm);
                    return 1;
                }
            snprintf(SV.reg[SV.n_reg].name, sizeof(SV.reg[0].name), "%s", nm);
            snprintf(SV.reg[SV.n_reg].path, sizeof(SV.reg[0].path), "%s", pth);
            if (!plat_file_readable(SV.reg[SV.n_reg].path)) {
                fprintf(stderr, "error: cannot read %s\n", SV.reg[SV.n_reg].path);
                return 1;
            }
            SV.n_reg++;
        }
        if (parallel > 1) {
            fprintf(stderr, "note: model swapping uses a single inference slot; "
                    "ignoring --parallel %d\n", parallel);
        }
        parallel = SV.n_reg > 0 ? 1 : parallel;
        resident_store(-1);
    }

    int threads_per_slot = n_threads / parallel;
    if (threads_per_slot < 1) threads_per_slot = 1;

    const char *name = strrchr(model_path, '/');
    const char *bsname = strrchr(model_path, '\\'); // Windows path separator
    if (bsname && (!name || bsname > name)) name = bsname;
    SV.model_name = SV.n_reg > 0 ? SV.reg[0].name : (name ? name + 1 : model_path);
    SV.n_predict_cap = 1024;
    SV.q.limit = (int)(sizeof(SV.q.fds) / sizeof(sock_t));
    // a queue bound may only lower the fixed fd capacity, never raise it
    SV.q.limit = (int)env_i64("RUNNER_MAX_QUEUE", 1, SV.q.limit, SV.q.limit);
    // a bad timeout must not silently become 0 (which disables the deadline)
    SV.req_timeout = env_f64("RUNNER_REQUEST_TIMEOUT", 0.0, 1e9, SV.req_timeout);
    // Shared, forkable prompt prefixes. Sized in host RAM rather than as a
    // fraction of anything, because it is the one cache whose useful size is
    // set by the *traffic* (how many distinct system/tool/schema blocks the
    // agents on this box use) and not by the model.
    {
        uint64_t mb  = env_u64("RUNNER_PREFIX_CACHE_MB", 0, 1u << 20, 512);
        double   ttl = env_f64("RUNNER_PREFIX_CACHE_TTL", 0.0, 1e9, 600.0);
        prefix_cache_configure((size_t)mb * 1024 * 1024, ttl);
    }
    context_store(base ? base->n_ctx : mp->n_ctx);
    SV.n_slots = parallel;
    SV.slots = calloc(parallel, sizeof(slot_t));
    if (!SV.slots) {
        fprintf(stderr, "error: cannot allocate server slots\n");
        return 1;
    }
    if (pthread_mutex_init(&SV.q.mu, NULL) != 0) {
        fprintf(stderr, "error: cannot initialize server queue mutex\n");
        return 1;
    }
    if (pthread_cond_init(&SV.q.cv, NULL) != 0) {
        fprintf(stderr, "error: cannot initialize server queue condition\n");
        pthread_mutex_destroy(&SV.q.mu);
        return 1;
    }

    if (SV.n_reg > 0) {
        // swap mode: models are loaded on demand
        slot_t *s = &SV.slots[0];
        s->id = 0;
        s->smp = defaults;
        s->smp_base = defaults;
        if (!init_swap_runtime(mp, n_threads, ttl)) return 1;
    } else {
        int tmpl = template_detect(gguf_get_str(&base->gf, "tokenizer.chat_template", NULL),
                                   tok);
        model_params slot_mp = *mp;
        slot_mp.verbose = false;
        slot_mp.n_threads = threads_per_slot;
        // each slot gets its share of the CPU-forced fallback cap too
        slot_mp.cpu_fallback_threads = mp->cpu_fallback_threads / parallel;

        for (int i = 0; i < parallel; i++) {
            // a Ctrl-C during a multi-slot load means "don't start": honour it
            // between loads rather than serving with the flag silently set
            if (stop_was_requested()) {
                fprintf(stderr, "shutdown requested during startup — exiting\n");
                return 0;
            }
            slot_t *s = &SV.slots[i];
            s->id = i;
            s->tok = tok;
            s->tmpl = tmpl;
            s->smp = defaults;
            s->smp.rng = defaults.rng ^ (0x9E3779B97F4A7C15ull * (unsigned)(i + 1));
            s->smp_base = s->smp;
            if (i == 0) {
                s->m = base;
                // mirror model_load's CPU-forced bump: this replacement pool
                // would otherwise silently undo it for slot 0
                int slot_threads = threads_per_slot;
                if (base->qwen35 &&
                    slot_mp.cpu_fallback_threads > slot_threads)
                    slot_threads = slot_mp.cpu_fallback_threads;
                tpool *replacement = tpool_create(slot_threads);
                if (!replacement) {
                    fprintf(stderr, "error: cannot create pool for slot 0\n");
                    return 1;
                }
                tpool_destroy(base->tp); // replace the single-thread load pool
                base->tp = replacement;
            } else {
                s->m = calloc(1, sizeof(model_t));
                if (!s->m) {
                    fprintf(stderr, "error: cannot allocate slot %d model\n", i);
                    return 1;
                }
                if (!model_load(s->m, model_path, &slot_mp)) {
                    fprintf(stderr, "error: failed to load slot %d\n", i);
                    return 1;
                }
            }
            if (!engine_init(&s->e, s->m, s->tok, &s->smp)) {
                fprintf(stderr, "error: out of memory initializing slot %d engine\n", i);
                return 1;
            }
            if (draft_path) {
                // per-slot draft context: each slot owns a full draft KV;
                // weights dedupe through the page cache like slot models
                s->e.dm = spec_draft_load(draft_path, s->m, &slot_mp);
                if (s->e.dm) s->e.draft_k = draft_k;
            }
        }

        if (parallel == 1) {
            // join the registry machinery so GET /unload frees the resident
            // model (the next request lazily reloads it) and --ttl works.
            // slot 0's containers are the caller's; borrowed avoids freeing
            // them on the first unload
            snprintf(SV.reg[0].name, sizeof(SV.reg[0].name), "%s", SV.model_name);
            snprintf(SV.reg[0].path, sizeof(SV.reg[0].path), "%s", model_path);
            SV.reg[0].tmpl = tmpl;
            SV.model_name = SV.reg[0].name;
            SV.n_reg = 1;
            SV.single = true;
            SV.borrowed = true;
            resident_store(0);
            SV.last_used = now_s();
            SV.draft = SV.slots[0].e.dm;
            SV.draft_k = draft_k;
            // a draft the gates rejected at startup stays rejected: only a
            // draft that actually served is worth reloading after /unload
            if (SV.draft) SV.draft_path = draft_path;
            if (!init_swap_runtime(mp, threads_per_slot, ttl)) return 1;
        }
    }

    // last long stop is behind us; a signal from here on is either caught
    // right now or by the published listener below
    if (stop_was_requested()) {
        fprintf(stderr, "shutdown requested during startup — exiting\n");
        return 0;
    }

    sock_t lfd = socket(AF_INET, SOCK_STREAM, 0);
    if (lfd == SOCK_INVALID) {
        fprintf(stderr, "error: cannot create server socket\n");
        return 1;
    }
    int one = 1;
    setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, (const char *)&one, sizeof(one));
    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons((uint16_t)port);
    if (bind(lfd, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
        fprintf(stderr, "error: cannot bind 127.0.0.1:%d (%s)\n", port, sock_errstr());
        sock_close(lfd);
        return 1;
    }
    if (listen(lfd, 64) != 0) {
        fprintf(stderr, "error: cannot listen on 127.0.0.1:%d (%s)\n", port,
                sock_errstr());
        sock_close(lfd);
        return 1;
    }
#ifndef _WIN32
    listener_fd = lfd;
#else
    win_listener_socket = lfd;
#endif

    // Every slot's model exists now, which is the only precondition the batch
    // has. Declining is not an error: sched_generate then runs the untouched
    // solo path, which is what a single slot, swap mode, or a backend without
    // batched kernels all get.
    bool batched = sched_start();

    for (int i = 0; i < parallel; i++) {
        if (pthread_create(&SV.slots[i].th, NULL, slot_worker, &SV.slots[i]) != 0) {
            fprintf(stderr, "error: cannot start server slot %d\n", i);
            sock_close(lfd);
            return 1;
        }
    }

    if (SV.n_reg > 0 && !SV.single)
        fprintf(stderr,
                "server listening on http://127.0.0.1:%d — %d models, swap on demand"
                " (ttl %ds)\n"
                "  POST /v1/chat/completions | POST /v1/responses | POST /v1/completions\n"
                "  GET /v1/models | GET /v1/capabilities | GET /health\n",
                port, SV.n_reg, SV.ttl);
    else
        fprintf(stderr,
                "server listening on http://127.0.0.1:%d — %d slot%s x %d threads%s\n"
                "  POST /v1/chat/completions | POST /v1/responses | POST /v1/completions\n"
                "  GET /v1/models | GET /v1/capabilities | GET /health\n",
                port, parallel, parallel > 1 ? "s" : "", threads_per_slot,
                batched ? ", continuous batching" : "");

    for (;;) {
        // covers the race where the signal landed after the socket-creation
        // check but before listener_fd published: the handler had no fd to
        // close, so accept() would block forever with the flag already set.
        // A signal landing after this check finds listener_fd published and
        // closes it, so accept() fails; no window remains.
        if (stop_was_requested()) break;
        sock_t cfd = accept(lfd, NULL, NULL);
        if (cfd == SOCK_INVALID) {
#ifndef _WIN32
            if (stop_requested) break;
#else
            if (win_stop_requested) break;
#endif
            if (errno == EINTR) continue;
            break;
        }
        if (!accept_fastpath(cfd)) q_push(cfd);
    }
    // Stop admission first, fail work that has not started, then allow active
    // requests to finish before dismantling the scheduler and model state.
#ifndef _WIN32
    if (listener_fd >= 0) {
        listener_fd = -1;
        sock_close(lfd);
    }
#else
    if (win_listener_socket != INVALID_SOCKET) {
        win_listener_socket = INVALID_SOCKET;
        sock_close(lfd);
    }
#endif
    queue_shutdown();
    // A worker parked in a --wait-for-vram queue would pin the joins below
    // for up to the full wait; the load gives up at its next poll instead.
    atomic_store(&SV.load_cancel, 1);
    for (int i = 0; i < parallel; i++) pthread_join(SV.slots[i].th, NULL);
    sched_shutdown();
    atomic_store(&SV.shutdown, true);
    if (SV.reaper_started) pthread_join(SV.reaper_th, NULL);

    for (int i = 0; i < parallel; i++) free(SV.slots[i].e.hist);
    if (SV.n_reg > 0) {
        pthread_mutex_lock(&SV.swap_mu);
        unload_draft();
        unload_resident();
        pthread_mutex_unlock(&SV.swap_mu);
        pthread_mutex_destroy(&SV.swap_mu);
    } else {
        tokenizer_free(tok);
        for (int i = 0; i < parallel; i++) {
            model_t *draft = SV.slots[i].e.dm;
            if (draft) { model_free(draft); free(draft); }
            model_free(SV.slots[i].m);
            if (i > 0) free(SV.slots[i].m);
        }
    }
    prefix_cache_clear();
    pthread_cond_destroy(&SV.q.cv);
    pthread_mutex_destroy(&SV.q.mu);
    free(SV.slots);
    return 0;
}
