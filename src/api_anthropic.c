// Anthropic Messages request -> chat. Lifted out of server.c (RNR-019); see api.h.
#include "api.h"
#include "compat.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
        if (prompt) render_messages(s->tmpl, t.cm, t.n, true,
                                    req_enable_thinking(req), prompt,
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

void handle_messages(slot_t *s, sock_t fd, jv *req) {
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
void handle_count_tokens(slot_t *s, sock_t fd, jv *req) {
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

