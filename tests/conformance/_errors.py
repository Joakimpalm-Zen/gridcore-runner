"""Failure taxonomy for the conformance harness.

Phase 0 exit criterion: "Failures distinguish protocol, schema, model-quality,
and transport errors." Every assertion in this suite raises one of these, so a
red run says *what kind* of thing broke without reading the traceback.

  TransportError   the bytes did not arrive intact: refused/reset connection,
                   read timeout, truncated or unframed HTTP response. Nothing
                   can be said about the API from such a run.
  ProtocolError    bytes arrived fine but the API contract was broken: wrong
                   status code, missing OpenAI-required field, malformed SSE
                   framing, an invariant like "unsupported field must 400".
  SchemaError      the response parsed and was well-formed OpenAI, but its
                   shape did not match the declared/expected structure — e.g.
                   response_format json_schema output violating its schema.
  ModelQualityError  everything protocol-level was correct; only the generated
                   *content* was unsatisfying. With the synthetic test.gguf this
                   is expected and is recorded in the report rather than failing
                   the run (see harness.quality_note).
"""


class ConformanceError(AssertionError):
    """Base class. Subclasses AssertionError so pytest renders it as a failure."""

    category = "unknown"

    def __init__(self, message, **context):
        self.context = context
        detail = "".join(f"\n    {k} = {v!r}" for k, v in context.items())
        super().__init__(f"[{self.category}] {message}{detail}")


class TransportError(ConformanceError):
    category = "transport"


class ProtocolError(ConformanceError):
    category = "protocol"


class SchemaError(ConformanceError):
    category = "schema"


class ModelQualityError(ConformanceError):
    category = "model_quality"


CATEGORIES = ("transport", "protocol", "schema", "model_quality")


def categorize(exc):
    """Best-effort category for any exception raised during a test."""
    if isinstance(exc, ConformanceError):
        return exc.category
    if isinstance(exc, (OSError, TimeoutError)):
        return "transport"
    return "unknown"
