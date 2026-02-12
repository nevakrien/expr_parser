# Language Semantics Notes

This document captures language-level behavior that spans parsing/lowering/type-inference boundaries.
Use this as the first reference for syntax/meaning questions; use `agent_docs/type_infrence.md` for solver internals.

## Function Syntax and Calling Convention

- `fn(...)` defines a language-native (`Hot`) function.
- `cfn(...)` defines a C-calling-convention (`C`) function.
- A missing body (`f = fn(...) -> T;` or `f = cfn(...) -> T;`) is a signature-only external declaration.
  - Lowering stores this as `Value::Func { body: None, ... }`.
  - Type inference checks only the signature and skips body constraints.

Type-level representation:

- Function types carry calling convention in `TypeValue::Func { calling_convention, ... }`.
- Known conventions print as `fn(...) -> ...` or `cfn(...) -> ...`.
- Unknown convention placeholders print as `fn?(...) -> ...`.

Calling-convention unification behavior:

- `Unknown` can merge with either `Hot` or `C`.
- `Hot` and `C` do not unify and surface as a regular type mismatch with both function signatures in clash payloads.

## Struct Syntax and Layout Marker

- `struct { ... }` uses language-native layout marker (`Hot`).
- `cstruct { ... }` uses C layout marker (`C`).
- Current inference/layout code records the marker in `StructRep.layout`.
- Today this mainly preserves intent; future passes can branch on it (for example field-reordering only for non-C layout types).

## Member Access Semantics (`.`, `::`, `->`)

Lowering maps syntax to `AccessKind`:

- `a.b` -> `AccessKind::Dot`
- `A::b` -> `AccessKind::Static`
- `p->b` -> `AccessKind::Ptr`

Type-inference behavior:

- `.` allows at most one implicit deref step.
- `->` can chain implicit pointer-like deref steps (with a bounded search) until member lookup resolves.
- For smart-pointer-like structs, lookup prefers direct members first; only then falls back to `__deref`/`__deref_mut` targets.
- Implicit deref hops are recorded in `SolvedTypes.member_access_implicit_derefs` for later lowering/rewrite stages.

Method access/currying:

- `obj.method` resolves as a callable where `self` is already consumed from the parameter list when applicable.
- The access node type is the curried callable type.
- The full uncurried method signature is tracked separately in `SolvedTypes.member_method_types`.

## Special Member Methods (Quick Reference)

- Reserved internal names begin with `__` (except names ending with `_` are not treated as reserved internals).
- Implemented special signatures include operator overloads (`__add`, `__neg`, ...), `__free`, `__deref`, and `__deref_mut`.
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
