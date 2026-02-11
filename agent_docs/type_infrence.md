# Type Inference System Summary (`src/type_inference.rs`)

## High-Level Philosophy

The implementation is intentionally constraint-first and error-tolerant:

1. Make straightforward inference obvious (`let`, annotations, function signatures, literals).
2. Keep going after clashes so multiple diagnostics are emitted in one run.
3. Leave room for future features (overloads/lifetimes), even if parts are TODO.

There is also an explicit engineering style used in the file: many internal "workhorse" helpers take *parts* of `InferState` (`store`, `parent`, `cluster`, `func_defs`, `struct_infers`, etc.) instead of taking `&mut InferState` wholesale. This is deliberate to enable complex borrow patterns and avoid borrow-checker dead ends during recursive unification and deferred solving.

## The Workhorses (Read These First)

These functions are the center of the entire file and appear all over inference:

- `unify_clusters(...)`:
  - union-find merge operation with two-way absorb attempt,
  - understands weak states (`IntLike`, `FloatLike`) and deferred structural states (`Func`, `Struct`, `Ptr`),
  - on failure returns `TypeClash` without collapsing the two clusters so inference can continue.
- `force_type(...)`:
  - pins one cluster to a concrete `TypeId`,
  - validates compatibility with unresolved/weak/deferred states,
  - delegates to specific helpers when target is `Func`/`Struct`/`Ptr` placeholders.
- `specialize_type(...)`:
  - substitutes generic placeholders (`TypeValue::Generic(GenId)`) with concrete local clusters,
  - recursively specializes function/struct/pointer shapes,
  - this is the mechanism that prevents generic-id scope collisions from leaking across call sites.
- `unify_if_distinct(...)`:
  - utility wrapper used heavily in operator resolution,
  - avoids redundant unify calls when both clusters already share one root,
  - returns "did progress happen" for the solver fixpoint loop.

## Unification and Clash Semantics

### How `unify_clusters` decides

`unify_clusters` tries merge in two directions (`found <- wanted`, then reversed). Internally `_try_absorb` handles cases like:

- `Nothing` can absorb most things.
- `Solved(t1)` with `Solved(t2)` succeeds only if `t1 == t2`.
- `Solved` with `IntLike`/`FloatLike` succeeds only if builtin category matches.
- `Func` with `Func` unifies each param/output cluster pair.
- `Struct` with `Struct` requires same struct id + same generic arity, then unifies generic clusters.
- pointer states merge only when `raw`/`mutable` flags are compatible and targets unify.

If both directions fail, `TypeClash` is produced with best-effort `found`/`wanted` mock types (`extract_bad_type` + `make_*_mock` helpers).

### How clashes become user errors

`TypeClash` is the payload embedded in many concrete `TypeError` variants:

- `ValuesContradict` (most expression-level mismatches),
- `AnnotationMismatch` / `PatternAnnotationMismatch`,
- `TypeClashBeforeMentioned` (typedef-related),
- `FieldTypeMismatch`, `TypeDefPatternMismatch`, etc.

Important behavior: a clash does **not** terminate inference globally. Clusters stay separate and inference continues, which allows collecting additional unrelated hard errors.

### Where `force_type` is used

`force_type` appears at places with one-way requirements (instead of symmetric equality):

- type-def patterns must be `Type` (`Value::TypeDef`),
- `if` / `while` conditions must be `bool`,
- constructor fields in non-generic structs are forced to declared field types,
- plus internal helpers like `unify_func_with_type`, `unify_struct_with_type`, `unify_ptr_with_type` recursively force components.

## Where `unify`/`force` Are Called in the Front-End

Main `ctx.unify(...)` call sites are concentrated in:

- global typedef reconciliation (`infer_global_types`),
- global signature consistency check (`infer_value_internals` known type vs gathered),
- `let`, `let-else`, assignment, and branch unification,
- value and pattern annotations,
- call-site function-shape matching,
- generic constructor field matching,
- function body vs declared output matching,
- pending specialization output reconciliation.

Main `ctx.force_type(...)` call sites are concentrated in:

- type-def pattern kind check (`Type`),
- `if`/`while` condition bool checks,
- non-generic constructor field typing.

## Generic and Specialization Model (Critical)

### Where generics are allowed to appear

Generic binders are intentionally constrained because generic IDs are local/sequential (`GenId(0..n)`) and not globally normalized.

- Function generics:
  - introduced by function generic pattern list,
  - effectively allowed only in top-level generic function signatures (`gather_func_signature::<true>` in global pass),
  - recorded and wrapped as `TypeValue::WithGenerics { count, body }` during `finalize`.
- Struct generics:
  - introduced in top-level struct type definitions (`compile_struct_type::<true>` in global typedef pass),
  - stored as `TypeValue::Struct { id, generics: [Generic(GenId(...))] }` template shape.
- Generic references in type expressions:
  - may appear where local generic names resolve through `local_types` mapping.
- Explicit specialization syntax:
  - `TypeExpr::Index { base, args }` creates specialization requests,
  - may become deferred (`pending_specializations`) until base typedef is known.

### Why specialization is mandatory

Because `GenId` is sequential inside each binder, "raw" generic ids from one scope cannot be safely reused in another scope. Without specialization, a generic from one function/type could alias a different binder's `GenId` accidentally.

Specialization avoids that by replacing generic placeholders with fresh local clusters each time:

- `global_to_specialized_local` for referencing generic global functions,
- constructor path for generic structs (`Construct` branch),
- `resolve_pending_specializations` for deferred type-level specializations.

### Core specialization flow

1. Start from template type (`WithGenerics.body` or generic struct template).
2. Allocate fresh cluster list for generic arguments.
3. Run `specialize_type` recursively to substitute every `TypeValue::Generic` occurrence.
4. Unify/force substituted result against call-site or annotation constraints.

## Core Type Model

### Stable IDs and sentinels

- `TypeId`, `GenId`, `StructId`: core ids.
- `UNKNOWN_TYPE`, `UNKNOWN_INT_SIZE`, `UNKNOWN_FLOAT_SIZE`: unresolved/weak placeholders for diagnostics.
- `BadTypeId`: wrapper used when diagnostics can include unresolved internals.

### Type shapes and storage

- `BuiltinType`: primitive set (`int`, sized ints, floats, `bool`, `str`, `void`, `Type`).
- `TypeValue`: builtins, tuple, array, function, pointer, generic binder (`WithGenerics`), generic param (`Generic`), struct instance (`Struct`).
- `TypeStore`: interned type arena + struct table.
  - builtins are interned first,
  - structural equality is intern identity,
  - helper predicates for int/float classes and pretty-printing.

### Struct representation

- `StructRep` contains optional name, field list, and generic count.
- Recursive structs are supported by creating struct ids early, then resolving field types in `finalize`.

## Internal Constraint Engine

### Union-find state

- Cluster id: `CId`.
- `ResolveKind` cluster states:
  - `Solved(TypeId)`, `Nothing`,
  - weak literals: `IntLike`, `FloatLike`,
  - deferred structures: `Func(FuncInferId)`, `Struct(StructInferId)`, `Ptr { ... }`.
- `InferState` holds:
  - IR-node -> cluster bindings,
  - union-find arrays (`parent`, `cluster`),
  - deferred operator sites,
  - function/struct placeholder metadata,
  - pending specialization queue,
  - error sink + solved output sink.

### Diagnostic mock types

When unresolved clusters must still be printed in errors, mock builders (`mock_type_from_cluster`, `make_func_mock`, `make_struct_mock`, `make_ptr_mock`) synthesize best-effort type shapes.

## Inference Pipeline

Main orchestration is two-phase:

1. `infer_global_types`
   - resolves typedefs/structs,
   - resolves function signatures (without body internals),
   - validates special member method signatures (`__add`, unary overload names, `__free`) during member-method signature gather,
   - supports recursive typedef + deferred specialization setup.
2. `infer_value_internals`
  - resolves function body internals or arbitrary value internals,
  - reconciles with known global signatures when present.

`run_typechecker` runs global pass, then member methods, then global functions, reporting through `ErrorReporter` and returning solved data or error count.

## Constraint Gathering

### Expressions (`gather_constraints`)

This is the AST/IR-to-typechecker bridge and one of the most important maintenance hotspots. Most new language features need to add or adjust logic here.

High-level behavior:

- literals create weak (`IntLike`/`FloatLike`) or concrete builtin clusters,
- `let`, assignment, and branch joins create equality constraints with `ctx.unify`,
- one-way obligations use `ctx.force_type` (`if`/`while` condition bool, etc.),
- calls/construction build deferred structural constraints (`Func`/`Struct`) then resolve via unification,
- operators are queued as deferred sites and reprocessed in solver rounds.

Important fragile/unfinished expression areas:

- `Value::NameRef`:
  - may resolve to local bound cluster, global function, or type name value,
  - generic globals are specialized on access (`global_to_specialized_local`),
  - assumptions around global name resolution and overload sets are explicitly incomplete (`todo!`).
- `Value::Call`:
  - positional calls are implemented via function-placeholder unification,
  - named-argument calls are mostly TODO and depend on resolving exact callable identity/signature,
  - this area is sensitive because argument mapping is an implicit conversion/rewrite step.
- `Value::Construct`:
  - validates constructor base is a global type and specifically a struct,
  - supports mixed positional/named fields with duplicate/unknown/missing-field checks,
  - generic structs trigger per-call specialization of field types before field unification,
  - contains many assumptions and error paths that are easy to break when extending syntax.
- `Value::AddrOf`:
  - creates pointer placeholder state (`ResolveKind::Ptr`) with partially-known flags,
  - later pointer solving depends on unification + deferred resolve,
  - future `Deref` support (not implemented yet) will need to integrate here and in operator/access paths.
- `Value::Access`:
  - partial support for struct field lookup from solved or deferred struct states,
  - member-method route is largely TODO,
  - this is a likely extension point for method dispatch and implicit receiver conversions.

Other notable implemented branches:

- `Value::TypeAnnotation` and pattern annotation branches are separate but both unify annotation type with constrained subject.
- `Value::Cast` intentionally does not require source/target equality (it gives target type identity).
- `Value::TypeDef` ensures pattern has type `Type` and registers local typedef clusters.
- `Value::Func` uses signature gather + body gather and unifies body with output cluster.

### Patterns and type expressions

- `gather_pattern_constraints*` handles bind/wildcard/annotated patterns and binds names to clusters.
- `gather_generic_constraints` maps generic parameter bind names to `TypeValue::Generic(GenId)` and records them in both value-name and type-name local maps.
- `compile_type_expr` lowers type syntax to clusters (name refs, wildcard, inline struct defs, pointers, specialization index).

Critical type-expression fragility points:

- `TypeExpr::NameRef`:
  - may resolve through local generic/type bindings or global typedef/builtin lookup,
  - unresolved globals can intentionally leave placeholder clusters for later solving.
- `TypeExpr::Index` (specialization):
  - one of the most delicate paths,
  - validates base is a global type and currently expects struct specialization,
  - may enqueue `pending_specializations` if base typedef is not yet solved,
  - easy place to introduce generic-id/scope bugs if specialization flow is modified.
- pointer type expressions (`TypeExpr::Ptr`) feed directly into deferred pointer cluster states, so pointer semantics changes usually require touching both gather and deferred resolution helpers.

Maintenance note: this whole gather layer is intentionally unfinished in places. Treat `NameRef`, `Call`, `Construct`, `TypeExpr::Index`, `AddrOf` (and future `Deref`) as priority review zones whenever adding type-system features, implicit conversions, or dispatch behavior.

## Middle Solver and Finalization

`main_solver` iterates until fixpoint:

1. `resolve_operator_types`
2. `resolve_deferred_types`
3. `resolve_pending_specializations`

Then `finalize`:

- commits solved typedef/value/pattern types into `SolvedTypes`,
- writes finalized struct field types and fills missing struct names,
- wraps generic function values into `WithGenerics`,
- emits unresolved errors once per unresolved root (to reduce duplicate noise).

## Operator Resolution Notes

- Operators are deferred as `BinOpSite` / `UnOpSite` and revisited during solver iterations.
- `unify_if_distinct` is the main operator-resolution merge primitive.
- Builtin legality checks are tri-state (`true` / `false` / `unknown`) to avoid premature hard errors.
- User-struct overload lookup is wired for detection but actual overload execution is still mostly TODO.
- Even though runtime/operator-call dispatch is still TODO, signature validation now enforces shape rules for special member names:
  - binary overload names (`__add`, `__sub`, etc.): `self`-like first parameter + exactly one extra parameter,
  - unary overload names (`__neg`, `__not`, `__bitnot`): `self`-like first parameter + no extra parameters,
  - `__free`: first parameter must be `&mut self`, no extra parameters, return type `void`; checks short-circuit on the first failing `__free` requirement so one root error is emitted.
- The validation is now based on solved global function type signatures (`TypeValue::Func` / `TypeValue::WithGenerics` body), not raw signature clusters.
- Member method names that start with `__` and do not end with `_` are treated as reserved builtin names; unknown reserved names emit a dedicated type error.

## Error and Test Philosophy

- `TypeError` is intentionally detailed and source-oriented; clashes carry `found`/`wanted` type payloads.
- Engine prefers collecting multiple deterministic hard errors over bailing early.
- Inline tests cover happy paths and representative failure modes (unresolveds, annotation/branch mismatch, generic specialization, recursive structs, operator legality).
