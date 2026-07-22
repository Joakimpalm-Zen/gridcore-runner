import OpenAI from "openai";
import Anthropic from "@anthropic-ai/sdk";

const base = process.argv[2];
if (!base) throw new Error("usage: smoke.mjs http://127.0.0.1:PORT");

const openai = new OpenAI({baseURL: `${base}/v1`, apiKey: "compat-test"});
const chat = await openai.chat.completions.create({
  model: "runner", messages: [{role: "user", content: "Say OK"}],
  max_tokens: 4, temperature: 0,
});
if (!chat.choices?.[0]?.message) throw new Error("OpenAI chat parse failed");

const response = await openai.responses.create({
  model: "runner", input: "Say OK", max_output_tokens: 4,
});
if (!response.id || !Array.isArray(response.output)) {
  throw new Error("OpenAI Responses parse failed");
}

const anthropic = new Anthropic({baseURL: base, apiKey: "compat-test"});
const message = await anthropic.messages.create({
  model: "runner", max_tokens: 4,
  messages: [{role: "user", content: "Say OK"}],
});
if (!message.id || !Array.isArray(message.content)) {
  throw new Error("Anthropic Messages parse failed");
}

console.log(JSON.stringify({openai: "pass", anthropic: "pass"}));
