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
  - implementations/specializations (`body: Some(...)`).

Global signature compatibility rules:

- all later declarations must be the same as or a specialization of the first declaration,
- each implementation must be a specialization of the first declaration type,
- duplicate implementation specializations (same concrete function type) are rejected.

These same declaration/implementation compatibility rules also apply to struct member methods; lowering stores each `Struct.method` entry as a `FunctionSet` rather than a single value.

Statement boundary note:

- Top-level statements are expression statements with optional `;` parsing support.
- In multiline source blobs, declaration-only forms like `name = fn(... )` should usually end with `;` to avoid being parsed as part of a following expression.

Comments:

- The lexer skips `//` line comments (from `//` to end-of-line) as trivia.

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
- Current typechecking support is intentionally limited:
  - only `[T; N]` is accepted as a concrete type and maps to the existing sized `Array` type shape,
  - `[T]` currently reports a type error (`unsized array types are not supported yet`) and is reserved for future work.

## Lifetimes and Reference Kinds (Planned Contract)

This is the intended language contract for upcoming lifetime-aware typing and borrow checking.

- Normal references are lifetime-checked borrows (`&'a T`, `&mut 'a T`).
- `'raw` is a separate lifetime state representing non-null pointer-style access (`&'raw T`, `&mut 'raw T`).
- `&'a T` and `&'raw T` are distinct; type inference must not silently upgrade/downgrade between them.

Downcasting/reborrowing:

- No implicit lifetime downcast for user-level references.
- Users must spell reborrow/downcast explicitly (for example `&*var`).
- Reborrow relationships are tracked as lifetime bounds (for example `'b < 'a`) and validated later in borrow analysis.

Smart-pointer method signatures:

- Safe/tied deref shape: fn['a](&'a self)->&'a out.
- Raw receiver deref shape: fn['a](&'raw self)->&'a out.
  - This allows producing arbitrary output lifetimes from a raw receiver.
- Address exposure shape: fn['a](&'raw self)->&'raw out.
  - This intentionally disables normal borrow guarantees along that path.

Implicit references created by desugaring:

- Member access, index access, and deref-chain resolution can synthesize fresh reference temporaries.
- Those temporaries include implicit lifetime casts and must be recorded in solved metadata for later borrow-check pass consumption.
- Temporary policy: these implicit casts may target any lifetime (for example: a -> 'raw, a -> 'static).
- Future policy: borrow analysis will reject illegal casts and enforce the real lattice.

Early vs deferred lifetime errors:

- Immediate typecheck rejection is allowed when constraints are directly contradictory in one signature/body typing step.
  - Example: f(x:&'a t)->&'b t{x} is immediately invalid if no reborrow relation justifies 'b.
- Constraints introduced by explicit reborrows should remain recorded for borrow analysis (not necessarily rejected in the first typing phase).

Unnamed lifetimes:

- In global signatures:
  - unnamed input-side lifetimes are treated as independent fresh lifetimes,
  - unnamed output-side lifetimes are intended as joins over input lifetimes (for example: 'a+'b+...).
  - temporary implementation rule: if exactly one input lifetime exists, pick it; otherwise emit `not implemented yet`.
- In bodies:
  - every unresolved/unnamed lifetime site mints a fresh lifetime id,
  - minted ids are tracked explicitly so later borrow analysis can index per-lifetime data in dense vectors.

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
- Implemented special member signatures include operator overloads (`__add`, `__neg`, ...), `__deref`, and `__deref_mut`.
- Destruction hooks (`__free`, `__user_free`) are global predeclared function families, not member methods.
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
