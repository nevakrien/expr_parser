# Macro System Notes (`src/macros.rs`)

This subsystem implements simple unhygienic source-level macros.

Macros are expanded on parser `Expr` trees before definition gathering/lowering.

## Definition Shape

Macro definitions are global assignments:

- `name = macro(params...) body`

`Macro::new` expects at least two arguments:

1. parameter list expression of form `(a, b, ...)`
2. body expression

Validation errors:

- missing body -> `ERR_MACRO_NEEDS_BODY`
- malformed parameter list -> `ERR_MACRO_SIGNATURE`
- non-identifier parameter -> `ERR_MACRO_PARAM_IDENT`

The stored macro payload is:

- declaration location
- ordered parameter names (`Vec<String>`)
- raw body `Expr`

## Expansion Entry Point

`expand_macros_recursive(expr, program)` performs expansion in two phases:

1. Repeatedly expand the current node while it is a direct macro call.
2. Recurse into children.

This allows nested expansion and expansion-to-expansion chaining at one site.

## What Counts as a Macro Call

Only this parser shape is treated as a call candidate:

- `Expr::Postfix("(", args)`
- first arg is callee expression
- callee must be `Expr::Atom(Token::Ident(name))`
- macro lookup is global (`program.get_macro(name)`)

No scope/hygiene lookup is performed for macros.

## Substitution Semantics

Substitution is purely syntactic and unhygienic:

- Identifier atom equal to a macro parameter is replaced with provided argument expression clone.
- Other identifiers are left as-is.
- Tree shape (`Atom/Bin/Prefix/Postfix`) is preserved recursively.

All rebuilt nodes use the call-site location, including operator tokens in rebuilt nodes.

Consequence: diagnostics from expanded code usually point to call site, not original macro definition body.

## Arity Behavior

During apply:

- argument count must match parameter count exactly
- mismatch raises `CompileError::Arity { call_name: "Macro expansion", ... }`

## Known Limitations (By Design)

- Unhygienic: introduced names can capture/be captured.
- No local macro scoping; lookup is effectively global.
- No quoting/escaping model; everything is plain tree substitution.
- No token-level span preservation from macro body.

For this codebase, that is intentional and currently closer to C-style macro behavior than hygienic macro systems.
