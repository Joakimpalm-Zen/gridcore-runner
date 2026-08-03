// Vercel AI SDK against runner's OpenAI-compatible surface.
//
// This is the client most local-model tooling actually reaches for (Cline,
// Continue, and anything built on Next.js), and it is not the OpenAI SDK: it
// has its own request builder and its own stream parser, and it sends field
// combinations the official SDK never does. `test_openai_sdk.py` proves the
// official client works; this proves a second, independent one does.
//
// Run by tests/conformance/test_ai_sdk.py, which skips when node_modules is
// absent so the C suite never depends on npm. Results go to stdout as one JSON
// document: {"ok": [...], "fail": [{"name","error"}]}. Exit code is the number
// of failures, capped at 100.
//
//   node smoke.mjs http://127.0.0.1:PORT [model]

import { createOpenAI } from '@ai-sdk/openai';
import { generateText, streamText, generateObject, tool, stepCountIs } from 'ai';
import { z } from 'zod';

const baseURL = (process.argv[2] || 'http://127.0.0.1:8080').replace(/\/$/, '') + '/v1';
const provider = createOpenAI({ baseURL, apiKey: 'not-used' });

// The served id, read from the server. A hardcoded name would make every case
// below pass or fail for a reason unrelated to the SDK.
async function servedModel() {
  const res = await fetch(`${baseURL}/models`);
  if (!res.ok) throw new Error(`GET /v1/models -> ${res.status}`);
  const body = await res.json();
  const id = body?.data?.[0]?.id;
  if (!id) throw new Error(`no model in /v1/models: ${JSON.stringify(body).slice(0, 200)}`);
  return id;
}

const ok = [];
const fail = [];
async function check(name, fn) {
  try {
    await fn();
    ok.push(name);
  } catch (e) {
    fail.push({ name, error: String(e?.message || e).slice(0, 400) });
  }
}

const modelId = process.argv[3] || (await servedModel());
// .chat() pins the chat-completions route. The bare provider call would use the
// Responses API, which is a different surface with its own tests below.
const chat = provider.chat(modelId);

await check('generateText', async () => {
  const r = await generateText({
    model: chat, prompt: 'hello', maxOutputTokens: 16, temperature: 0,
  });
  if (typeof r.text !== 'string' || r.text.length === 0)
    throw new Error(`no text: ${JSON.stringify(r.text)}`);
  if (!r.usage || typeof r.usage.inputTokens !== 'number')
    throw new Error(`no usage: ${JSON.stringify(r.usage)}`);
  if (!['stop', 'length', 'tool-calls', 'content-filter', 'other', 'unknown'].includes(r.finishReason))
    throw new Error(`bad finishReason: ${r.finishReason}`);
});

await check('streamText', async () => {
  const r = streamText({
    model: chat, prompt: 'hello', maxOutputTokens: 16, temperature: 0,
  });
  let text = '';
  // The SDK's own stream parser is the check here: a misordered or misnamed
  // chunk throws inside this loop rather than yielding wrong text.
  for await (const delta of r.textStream) text += delta;
  if (!text) throw new Error('stream produced no text');
  const finish = await r.finishReason;
  if (!finish) throw new Error('stream never reported a finishReason');
});

await check('streamText usage', async () => {
  const r = streamText({
    model: chat, prompt: 'hello', maxOutputTokens: 16, temperature: 0,
  });
  for await (const _ of r.textStream) { /* drain */ }
  const usage = await r.usage;
  if (!usage || typeof usage.totalTokens !== 'number')
    throw new Error(`stream reported no usage: ${JSON.stringify(usage)}`);
});

await check('tool call', async () => {
  const r = await generateText({
    model: chat,
    maxOutputTokens: 64,
    temperature: 0,
    // stopWhen at one step keeps this a pure call check: the SDK would
    // otherwise execute the tool and loop, which the next case covers.
    stopWhen: stepCountIs(1),
    toolChoice: 'required',
    tools: {
      get_weather: tool({
        description: 'Look up the weather for a city',
        inputSchema: z.object({ city: z.string() }),
      }),
    },
    prompt: 'weather in Oslo?',
  });
  if (!r.toolCalls?.length)
    throw new Error(`no toolCalls: ${JSON.stringify(r.toolCalls)}`);
  const call = r.toolCalls[0];
  if (call.toolName !== 'get_weather')
    throw new Error(`wrong tool: ${call.toolName}`);
  // The SDK parses arguments against inputSchema; a malformed document would
  // have thrown before reaching here.
  if (typeof call.input?.city !== 'string')
    throw new Error(`arguments did not parse: ${JSON.stringify(call.input)}`);
});

await check('tool round trip', async () => {
  // Two model turns with an executed tool between them. This is where an
  // OpenAI-shaped history is easiest to reject: the SDK replays an assistant
  // message carrying tool-call parts and no text, then a tool result keyed by
  // call id.
  const r = await generateText({
    model: chat,
    maxOutputTokens: 64,
    temperature: 0,
    stopWhen: stepCountIs(3),
    tools: {
      get_weather: tool({
        description: 'Look up the weather for a city',
        inputSchema: z.object({ city: z.string() }),
        execute: async ({ city }) => ({ city, tempC: 12, sky: 'raining' }),
      }),
    },
    prompt: 'weather in Oslo? Use the tool.',
  });
  if (r.steps.length < 2)
    throw new Error(`the tool result was never sent back: ${r.steps.length} step(s)`);
});

await check('generateObject', async () => {
  const r = await generateObject({
    model: chat,
    maxOutputTokens: 96,
    temperature: 0,
    schema: z.object({ name: z.string(), age: z.number().int() }),
    prompt: 'invent a person',
  });
  // generateObject validates against the zod schema itself, so reaching here
  // means the document parsed and typechecked.
  if (typeof r.object?.name !== 'string' || typeof r.object?.age !== 'number')
    throw new Error(`object did not typecheck: ${JSON.stringify(r.object)}`);
});

await check('embeddings', async () => {
  // This SDK sends `encoding_format: "float"`, so it is NOT a second guard on
  // the base64 defect the OpenAI SDK exposed — test_structured_output.py owns
  // that one. What it checks is that the batch comes back whole and usable
  // through a different client's decoder.
  const { embedMany } = await import('ai');
  const r = await embedMany({
    model: provider.textEmbeddingModel(modelId),
    values: ['a car', 'a banana'],
  });
  if (r.embeddings.length !== 2)
    throw new Error(`wrong count: ${r.embeddings.length}`);
  const widths = new Set(r.embeddings.map((v) => v.length));
  if (widths.size !== 1) throw new Error(`ragged widths: ${[...widths]}`);
  const norm = Math.sqrt(r.embeddings[0].reduce((a, x) => a + x * x, 0));
  // runner documents L2-normalised vectors; a decode that got the byte order
  // or the width wrong would land nowhere near 1.
  if (Math.abs(norm - 1) > 1e-3)
    throw new Error(`decoded vector is not unit length: ${norm}`);
});

await check('error is typed, not a parse failure', async () => {
  try {
    await generateText({
      model: provider.chat('definitely-not-served'),
      prompt: 'hi', maxOutputTokens: 8,
    });
  } catch (e) {
    const status = e?.statusCode ?? e?.data?.statusCode ?? e?.cause?.statusCode;
    if (status === 400 || status === 404) return;
    throw new Error(`unknown model raised ${e?.name} status=${status}`);
  }
  throw new Error('an unknown model was accepted');
});

process.stdout.write(JSON.stringify({ model: modelId, ok, fail }, null, 2) + '\n');
process.exit(Math.min(fail.length, 100));
