# Mutability Fuzzing Sketch

This note documents a practical first step toward mutability fuzzing without needing a full language-level validity oracle.

## Why template fuzzing first

A full grammar fuzzer would need to decide whether a generated program should typecheck, which is hard while semantics are evolving.

Instead, start with **template-generated programs** whose expected outcome is known by construction:

- same shape, different root mutability (`var` vs `let`),
- direct vs aliased vs generic-identity transport paths,
- expected result decided from the root mutability and pointer mutability semantics.

This catches regressions like mutability laundering through inference paths, while staying cheap to maintain.

## Example scaffold

- `examples/mutability_fuzz_sketch.rs` runs a small oracle-backed matrix.
- It prints whether each case matches the expected accept/reject result.
- Use strict mode in CI-like runs:

```bash
cargo run --example mutability_fuzz_sketch -- --strict
```

## Suggested next expansion

- Add combinatorial generators over:
  - root binding kind (`let` / `var`),
  - transport path (`&x`, `&*p`, `id(&x)`, tuple/struct wrappers),
  - write site (`*p = ...`, `(*p).field = ...`, index writes).
- Keep each generated case small (single function) and label the expected oracle rule for easy triage.
- Store mismatching programs to disk so they can be promoted to unit tests.
