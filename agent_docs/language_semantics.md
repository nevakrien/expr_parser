# Language Semantics Notes

This document captures language-level behavior that spans parsing/lowering/type-inference boundaries.
Use this as the first reference for syntax/meaning questions; use `agent_docs/type_inference.md` for solver internals.

## Function Syntax and Calling Convention

- `fn(...)` defines a language-native (`Hot`) function.
- `cfn(...)` defines a C-calling-convention (`C`) function.
- A missing body (`f = fn(...) -> T;` or `f = cfn(...) -> T;`) is a signature-only external declaration.
  - Lowering stores this as `Value::Func { body: None, ... }`.
  - Type inference checks only the signature and skips body constraints.
- A global function name now tracks two groups:
  - declarations (`body: None`), and
  - implementations (`body: Some(...)`).

Global signature compatibility rules:

- all later declarations must exactly match the first declaration signature,
- at most one implementation is allowed,
- if an implementation exists, its signature must exactly match the declared/reference signature.

These same declaration/implementation compatibility rules also apply to struct member methods; lowering stores each `Struct.method` entry as a `FunctionSet` rather than a single value.

Statement boundary note:

- Top-level statements are expression statements with optional `;` parsing support.
- In multiline source blobs, declaration-only forms like `name = fn(... )` should usually end with `;` to avoid being parsed as part of a following expression.

Comments:

- The lexer skips `//` line comments (from `//` to end-of-line) as trivia.

Null-pointer literals:

- `null` and `nil` are aliases for the same literal value.
- They represent a nullable raw pointer literal (the `*T` / `*const T` / `*mut T` family), not a normal reference.
- Their target pointee type and mutability are inferred from context (for example assignment/annotation).

Type-level representation:

- Function types carry calling convention and generic arity in `TypeValue::Func { calling_convention, generics, ... }`.
- Known conventions print as `fn(...) -> ...` or `cfn(...) -> ...`.
- Unknown convention placeholders print as `fn?(...) -> ...`.

Calling-convention unification behavior:

- `Unknown` can merge with either `Hot` or `C`.
- `Hot` and `C` do not unify and surface as a regular type mismatch with both function signatures in clash payloads.

Closure support:

- Function literals used as local expression values (closures) are currently not supported.
- Type inference emits a direct error: `sorry we dont support closures`.
- Top-level `name = fn(...) { ... }` / `name = cfn(...) { ... }` definitions remain supported.

## Struct Syntax and Layout Marker

- `struct { ... }` uses language-native layout marker (`Hot`).
- `cstruct { ... }` uses C layout marker (`C`).
- Current inference/layout code records the marker in `StructRep.layout`.
- Today this mainly preserves intent; future passes can branch on it (for example field-reordering only for non-C layout types).

## Array Type Expressions

- Type expressions accept bracket array forms:
  - `[T; N]` for sized arrays.
  - `[T]` parses as an unsized array form.
- Current typechecking support:
  - `[T; N]` resolves to `TypeValue::Array(T, ArrayType::Sized(N))`.
  - `[T]` resolves in type expressions to `TypeValue::Array(T, ArrayType::Unsized)`.
- Value/index semantics are still array-focused and evolving; unsized arrays are currently a type-level form, not a fully general runtime container model.

## Generic Declaration `where` Clauses (Lowering Shape)

- Generic declaration brackets on `fn`/`struct`-like forms now allow an inline `where` split point:
  - Example: `fn['a, 'b, T, where T<'a, T<'b](...) { ... }`.
- Lowering stores declaration parts in `GenDec` as:
  - lifetime/generic declaration patterns (`parts` + `lifetime_end`), and
  - `where` constraints as a dedicated type-expression span (`where_clause`).
- Parsing detail:
  - The first `where ...` item inside the bracket list starts the where-clause section.
  - Any remaining comma-separated items after that are also lowered as where constraints.

## Type-Expression `<` Constraints

- Type expressions now accept binary `<` in lowering and represent it as `TypeExpr::Lt { lhs, rhs }`.
- This is primarily intended for generic where-constraint forms (for example `T<'a`).
- Other binary operators in type expressions remain unsupported and continue to emit lowering errors.

## Lifetimes and Reference Kinds (Refactor Contract)

The active implementation is expected to move toward the clean-room type-system
refactor described in `agent_docs/type_inference.md` and
`agent_docs/lifetimes_plan.md`. Treat older detailed lifetime behavior as
legacy reference material.

Language-level intent to preserve:

- Normal references are lifetime-checked borrows (`&'a T`, `&mut 'a T`).
- Raw/reference style is distinct from normal safe references and should not be
  silently upgraded or downgraded by shape solving.
- Reborrows and implicit deref/member/index projections should record
  obligations for later lifetime/provenance checking.
- Lifetime ordering such as `'a < 'b` is not type-shape equality; it belongs to a
  later graph/order phase.
- Unnamed/elided lifetime behavior is still an open design area for the new
  solver and should not be copied from old finalization hacks without review.

Parser note:

- Lifetime tokens in reference syntax are parsed atomically (`&'a T` keeps `'a`
  as just the lifetime name).
- Postfix/path continuations are not consumed as part of the lifetime node.

## Member Access Semantics (`.`, `::`, `->`)

Lowering maps syntax to `AccessKind`:

- `a.b` -> `AccessKind::Dot`
- `A::b` -> `AccessKind::Static`
- `p->b` -> `AccessKind::Ptr`
- `t.0` / `p->0` lower as integer member access (`Value::IntAccess`) with the same `AccessKind` rules.

Type-inference behavior:

- `.` allows at most one implicit deref step.
- `->` can chain implicit pointer-like deref steps (with a bounded search) until member lookup resolves.
- For smart-pointer-like structs, lookup prefers direct members first; only then falls back to `__deref`/`__deref_mut` targets.
- Integer member access is reserved for tuples: the index is resolved against tuple arity after implicit deref.
- `::` is not valid for tuple integer member access (`tuple element access does not support \`::\``).
- Implicit deref hops are recorded in solved data as a chain of `(KindId, Projection)` entries; this keeps the old per-hop chain model while preserving which projection kind produced each step.

Method access/currying:

- `obj.method` resolves as a callable where `self` is already consumed from the parameter list when applicable.
- The access node type is the curried callable type.
- The full uncurried method signature is tracked separately in `SolvedTypes.member_method_types`.

## Special Member Methods (Quick Reference)

- Reserved internal names begin with `__` (except names ending with `_` are not treated as reserved internals).
- Implemented special member signatures include operator overloads (`__add`, `__neg`, ...), `__deref`, `__deref_mut`, and `__free`.
- `__size_of` and `__align_of` are builtin member methods available on any type and return `usize`.
- `__size_of` is modeled as a reference receiver (`&self`, with future intent to also allow `&'raw self` explicitly) so unsized values can report runtime size from metadata.
- `__free` is also available as a builtin member method on any type (`&mut self -> void`), while user structs may additionally define a checked `Struct.__free` member override.
- `__user_free` is still reserved and not a recognized special member method.
- Signature checks for these happen during global signature inference.

## Labels and `goto`

- Backtick is prefix label syntax: `` `name ``.
- A statement-form label declaration is a standalone backtick expression statement, for example `` `err; ``.
- `goto` is a prefix control-flow form with one argument and currently expects direct label syntax (`goto `err;`).
- Label resolution is function-local and lazy during lowering:
  - Forward jumps like `goto `err; `err;` are valid.
  - Labels are isolated per function body; same label text in different functions does not share a target.
  - If a function uses a label that is never declared in that function, lowering emits `label \`X\` was used but never defined`.
- Label declarations and `goto` both type as `Never`, same as `break`/`continue`.
