"""Legacy Completions logprobs used by reference-differential tooling."""


def test_text_completion_returns_requested_logprobs(client):
    response = client.completion({
        "prompt": "hello",
        "max_tokens": 3,
        "temperature": 0,
        "logprobs": 5,
    }, name="completion_logprobs")

    choice = response.json["choices"][0]
    logprobs = choice["logprobs"]
    assert len(logprobs["tokens"]) == 3
    assert len(logprobs["token_logprobs"]) == 3
    assert len(logprobs["top_logprobs"]) == 3
    assert len(logprobs["text_offset"]) == 3
    assert all(len(row) <= 5 for row in logprobs["top_logprobs"])
    assert choice["text"] == "".join(logprobs["tokens"])
