# CLI/REPL Debugging Guide (`src/main.rs`)

## Why use the CLI first

When behavior looks wrong (parser shape, inferred types, or reported errors), start with the REPL from `cargo run`.

It is often the quickest way to inspect what the compiler currently thinks, without adding temporary logging.

## What the REPL gives you

- Parsed AST view:
  - each entered statement/expression is printed as a pretty tree,
  - this is the fastest way to verify parse precedence/grouping and macro-expanded shape.
- Immediate typechecking feedback:
  - diagnostics are reported with source spans,
  - mismatch errors include expected/found information.
- Type dump helpers:
  - `:types` dumps solved type info for the current program,
  - `:types-of <name>` dumps the solved type region for one definition,
  - `:type <name...>` prints one-line type info for selected names.

## Practical debugging pattern

1. Run `cargo run`.
2. Paste the suspicious expression/function exactly as written.
3. Check the printed tree first (confirm AST shape before changing inference code).
4. If you need a targeted mismatch, force an expectation with an annotation, for example:

```txt
(complex_expression_here) :void;
```

This intentionally creates a type obligation and usually surfaces what the checker inferred versus what was required.

5. Use `:types-of <name>` to inspect only one function/type definition when a full dump is noisy.

## Related commands

- `:reset` clears REPL state.
- `:load <path...>` loads source files into the current session.
- `:types` / `:types-of <name>` / `:type <name...>` for solved type introspection.
- `quit` or `exit` to leave.
