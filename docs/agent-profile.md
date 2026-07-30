# Gridcore agent profile metadata

Runner admits ordinary GGUF files without an agent profile exactly as before.
An export opts into the versioned contract by declaring any `gridcore.agent.*`
key; once opted in, all keys below are required and validated before model
state or tensor buffers are allocated.

| GGUF key | Type | Admitted value |
|---|---|---|
| `gridcore.agent.protocol_version` | integer | `1` |
| `gridcore.agent.tokenizer_version` | integer | `1` |
| `gridcore.agent.schema_id` | string | non-empty canonical schema ID |
| `gridcore.agent.schema_digest` | string | 64 lowercase hexadecimal characters |
| `gridcore.agent.required_features` | array of strings | known features listed below |

The protocol-1 runtime features are `dense`, `json_schema`,
`continuous_batching`, `prefix_cache`, and `spec_decode`. Unknown required
features fail closed with `unsupported required agent feature '<name>'`; they
are never silently ignored. Features are runtime mechanisms, not policy or
execution authority.

The CLI prints the admitted versions, schema identity, digest, and feature
count after load. Server clients receive the full admitted object as
`agent_profile` from `GET /v1/capabilities`; it is `null` for an ordinary GGUF.
