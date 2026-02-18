# IR Lowering Guide (`src/ir.rs`)

This file defines the main front-end IR shape (`Value`, `Pattern`, `TypeExpr`) and the syntax lowering pass from parser `Expr` trees into that IR.

## What Lives Here

- Core IDs: `ValId`, `PatId`, `TExpId`, `NameId`, `LabelId`.
- Arena spans: `ValueSpan`, `PatternSpan`, `TypeExprSpan`.
- Runtime/value IR: `Value`.
- Pattern IR: `Pattern`.
- Type-syntax IR: `TypeExpr`.
- Lowering methods on `Program`:
  - `lower_value*`
  - `lower_pattern*`
  - `lower_type_expr*`

Lowering writes into `Program` arenas; the arenas themselves are owned by `src/program.rs`.

## Arena/Span Model (Important)

The lowering pipeline reserves contiguous slots before lowering subexpressions:

- `reserve_value_span(n)` pre-fills `n` `Value::Literal(Void)` sentinels.
- `reserve_pattern_span(n)` pre-fills wildcard pattern sentinels.
- `reserve_type_expr_span(n)` pre-fills wildcard type sentinels.

Then lowering fills those exact indices with `set_*` calls. This avoids temporary vectors and keeps stable IDs during recursive lowering.

Practical invariant: a span is always contiguous and indexable by arithmetic (`start + i`).

## `Value` Design Notes

`Value` is expression-oriented but makes effects explicit.

- Pure operations stay pure:
  - `Value::BinOp` and `Value::UnOp` are side-effect free.
- Effect/control operations are separate variants:
  - `Assign`, `Let`, `Block`, `If`, `While`, `Match`, `Return`, `Goto`, `Break`, `Continue`.
- Call-ish forms share `Call` payload:
  - `Call`, `Index`, `Construct`.
  - Each carries `base`, `args`, and `named_args_start` for positional/named split.

### Named arguments invariant

Lowering allows named arguments (`a = 1`) only as a suffix of the argument list. A positional argument after a named one triggers `ERR_POS_ARG_AFTER_NAMED`.

## Entry Points and Dispatch

- `lower_value(expr)` allocates one output slot and delegates to `lower_value_into`.
- `lower_value_into` special-cases `goto` first, then calls `lower_value_inner`.
- `lower_value_inner` pattern-matches parser `Expr` and picks the lowering helper.

Rough dispatch order:

1. Atom/blocks/tuples/call-like forms.
2. Assignment/cast/access/type/let/match/if/while/jump/function forms.
3. Generic prefix/postfix/bin-op fallbacks.

## Control-Flow and Scope Behavior

### Blocks

- Lowered with `with_scope_value` (same push/pop invariant as `with_scope`, but for infallible lowering paths that accumulate diagnostics instead of returning `Result`).
- All but last item become `statements`.
- Final item becomes `return_value` unless it is a standalone `;` atom.
- Label declarations (`` `name ``) inside blocks become `Value::LabelDecl` and are only legal inside function bodies.

### Functions

- Lowered from `fn` / `cfn` into `Value::Func`.
- Optional generic list: `fn[T](...)`.
- Signature may have output arrow: `(params) -> out`.
- Body is optional (`body: None` for declaration-style forms).
- Function lowering wraps work in:
  - `with_function_labels_value` (label namespace and unresolved-label checks that are appended to `Program::lowering_errors`)
  - `with_scope_value` (local bindings)

### `goto` and labels

`goto` lowering is patched through `Program` label machinery:

- Forward goto initially stores `LabelId::PENDING`.
- When label is defined later, pending goto sites are rewritten to final `LabelId`.
- At function end, any still-pending labels become `CompileError::UnresolvedLabel`.

## Pattern and Type-Expr Lowering

### Patterns

Supported today:

- bind (`x`), wildcard (`_`), tuple patterns, mutability wrappers (`mut`/`const`), and pattern annotations (`pat : type`).

Notably, pattern binds allocate fresh `NameId` into current scope immediately.

### Type expressions

Supported today:

- names and `_`
- tuples
- specialization/index syntax: `Base[T, U]`
- pointer/reference forms via prefix `*` and `&` with optional `mut`/`const`
- inline `struct`/`cstruct`/`enum`/`union` forms through `StructLike`

Unsupported forms route through `ERR_UNSUPPORTED_TYPE_EXPR`.

## Operator Lowering Contract

- Assignment operators (`=`, `+=`, ...) lower to `Value::Assign` with `AssignOp` flavor.
- Logical ops (`&&`, `||`) lower to `Value::LogicOp` (kept separate from pure binops).
- Pipe (`|>`) rewrites to call form by inserting lhs as first argument after callee.
- Member-access operators (`.`, `::`, `->`) lower to:
  - `Value::Access` when RHS is an identifier,
  - `Value::IntAccess` when RHS is an integer literal (tuple-style access such as `x.0` / `x->0`).

## Known Fragile/Incomplete Areas

- Several comments mark future refactors toward flatter/value-pattern unified IR.
- Some parse-shape assumptions are guarded by `debug_assert!` and `unreachable!` rather than user-facing diagnostics.
- Type-only and value-only contexts still overlap in a few places and are intentionally strict with unsupported-form errors.

When extending syntax, start by checking `lower_value_inner`, `lower_pattern_inner`, and `lower_type_expr_inner`; these three are the central dispatch hubs.
