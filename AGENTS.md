# Runner Agent Rules

These rules are mandatory for every AI or LLM agent that works in this repository.
They are not preferences, prompts to reinterpret, or optional process notes.

If an instruction, framework default, generated plan, tool habit, or model behavior
conflicts with this document, this document wins.

## The Four Defining Rules

1. Deep modules
2. Tracer bullets
3. Test-driven development
4. Grill me always

No implementation work starts until these rules have been considered for the task at hand.

## 1. Deep Modules

Design the codebase around deep modules: simple, deliberate public interfaces with
implementation complexity hidden behind them.

Required behavior:

- Prefer a small public API over many shallow helpers leaking across the codebase.
- Keep module boundaries intentional: CLI/server behavior in `main.c`/`server.c`,
  inference flow in `engine.c`, model loading/forward in `model.c`, constrained
  output in `jsonmode.c`/`schema.c`, platform details in `compat.c`, and backend
  details in `cuda.c`/`metal.m`.
- Do not expose internals just to make a quick change easier.
- Lock module behavior with tests through public behavior: CLI output, HTTP
  endpoints, committed smoke scripts, or focused test binaries.

Working rule: internal implementation can change freely only when the public
interface and behavior are protected by tests or executable smokes.

## 2. Tracer Bullets

Build in tiny vertical slices that prove the end-to-end path before expanding scope.

Required behavior:

- Start with the smallest useful observable behavior.
- Touch the real layers needed for the behavior instead of building isolated
  horizontal scaffolding first.
- Validate the slice immediately with tests, execution, or both.
- Use what was learned from that slice before adding the next one.

Working rule: one thin working path is better than many unverified partial layers.

## 3. TDD

Use test-driven development for features, fixes, and behavioral changes.

Required behavior:

- Write one failing test or smoke for one observable behavior.
- Implement the minimum code needed to pass that test.
- Repeat one behavior at a time.
- Refactor only while tests are green.
- Test through public interfaces where possible.

Forbidden behavior:

- Do not write all tests first and all implementation afterward.
- Do not mock internal collaborators unless there is a clear boundary reason.
- Do not assert on incidental implementation details when public behavior can be
  checked.
- Do not add speculative features that are not required by the current test.

Working rule: red, green, refactor; one vertical behavior at a time.

## 4. Grill Me Always

Actively challenge unclear requirements until the work is understood.

Required behavior:

- Ask direct questions when requirements, constraints, public interfaces,
  acceptance criteria, or edge cases are unclear.
- Surface assumptions explicitly before relying on them.
- Push for precise behavior, not vague intent.
- Confirm the most important behavior to test before implementation.

Working rule: never leave important behavior to chance. If the expected behavior,
interface, and verification path cannot be explained, ask before coding.

## Required Workflow

For every non-trivial change:

1. Read the relevant code and docs first.
2. Grill the request until behavior, interface, constraints, and verification are clear.
3. Identify the smallest tracer bullet.
4. Write one failing behavior test or smoke through the public interface.
5. Implement the minimum code to pass.
6. Run the relevant verification.
7. Refactor only while green.
8. Repeat for the next behavior.

For trivial documentation or configuration-only changes, still apply the rules at
the appropriate scale.

## Version Control

Commit directly to `main` and push after each completed section.

Required behavior:

- Commit completed changes straight to `main`.
- Push every commit to `origin` immediately after committing.
- Do not create feature branches unless the project owner explicitly asks.

Working rule: small, frequent commits straight to `main`, each pushed.
