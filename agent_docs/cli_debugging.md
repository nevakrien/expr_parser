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
  - `:types-of <Struct.method>` also works for struct member methods,
  - `:type <name...>` prints one-line type info for selected names.
  - Type dump now also annotates member-access sites with `member access implicit deref chain: ...`, listing each implicit dereference step plus the final resolved base type used for member lookup.

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

## Debugger-first pipeline example

When you need debugger breakpoints, assertions, or a reproducible one-file harness, use `examples/debuger.rs` instead of the interactive REPL.

- Run it with `cargo run --example debuger`.
- Put `dbg!`, `assert!`, or temporary prints directly in the example.
- For debugger stepping, launch `gdb --args target/debug/examples/debuger` after building.
- The example already runs parse + lower + typecheck and wires diagnostics through `ErrorReporter`, so it is a good template for targeted investigations.

## Related commands

- `:reset` clears REPL state.
- `:load <path...>` loads source files into the current session.
- `:types` / `:types-of <name>` / `:type <name...>` for solved type introspection.
- `quit` or `exit` to leave.
