# Type Inference System Summary (`src/type_inference.rs`, `src/global_type_inference.rs`, `src/local_type_inference.rs`)

## File Layout Update

Type inference is no longer a single monolithic file. The implementation is now split by phase:

- `src/global_type_inference.rs`: global typedef + function signature solving.
- `src/local_type_inference.rs`: function body / local value inference.
- `src/type_inference.rs`: shared type model, union-find machinery, common helpers, error types, and orchestration.

When reading call paths, prefer starting from `infer_global_types` (global pass) and `infer_value_internals` (local pass), then jump into shared helpers in `type_inference.rs`.

## High-Level Philosophy

The implementation is intentionally constraint-first and error-tolerant:

1. Make straightforward inference obvious (`let`, annotations, function signatures, literals).
2. Keep going after clashes so multiple diagnostics are emitted in one run.
3. Leave room for future features (overloads/lifetimes), even if parts are TODO.

There is also an explicit engineering style used in the file: many internal "workhorse" helpers take *parts* of `InferState` (`store`, `parent`, `cluster`, `func_defs`, `struct_infers`, etc.) instead of taking `&mut InferState` wholesale. This is deliberate to enable complex borrow patterns and avoid borrow-checker dead ends during recursive unification and deferred solving.

Recent lifetime/deref work added two important implementation details:

- `TypeState` now carries a lightweight lifetime union-find (`LId` parent + origin site storage) used by implicit deref chains.
- Implicit deref resolution threads shared chain state (`shared_lid`, chain mutability) through `resolve_struct_deref_target`, so multiple deref hops can share one lifetime identity and defer mutability collapse until enough information is known.
- Finalization currently applies a temporary hack that normalizes unresolved pointer lifetime kinds (`SafeRef`/`SomeRef`) to `Ref(Unknown)` right before `finalize`.
- Unresolved finalization diagnostics (`Unresolved`, `UnresolvedPattern`, `UnresolvedTypeExpr`) now carry a best-effort mock type string for richer error output.

For syntax-level semantics that span parser/lowering/typecheck (for example `fn` vs `cfn`, `struct` vs `cstruct`, and `. / :: / ->` behavior), see `agent_docs/language_semantics.md`.

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
  - substitutes generic placeholders (`TypeValue::Generic(GenId, TraitInfo)`) with concrete local clusters,
  - recursively specializes function/struct/pointer shapes,
  - this is the mechanism that prevents generic-id scope collisions from leaking across call sites.
- `specialize_type(...)`:
  - now always wraps specialization in a shared specialization context,
  - guarantees lifetime specialization is applied together with generic specialization,
  - maps solved/global lifetimes used in specialized signatures to fresh local unresolved `LId`s per specialization call.
- `unify_if_distinct(...)`:
  - utility wrapper used heavily in operator resolution,
  - avoids redundant unify calls when both clusters already share one root,
  - returns "did progress happen" for the solver fixpoint loop.

## Local Solver Order Fuzzing

- Solver order planning now lives in `src/local_solver_order.rs`; `local_solver` in `src/local_type_inference.rs` consumes that planner.
- `src/local_solver_order.rs` is module-gated in `src/lib.rs` behind `#[cfg(feature = "solver_order_fuzz")]`, so it is not compiled in stable/non-fuzz builds.
- `local_solver` itself is split into two cfg-separated implementations in `src/local_type_inference.rs`:
  - non-fuzz builds use a direct fixed pass sequence with no random planner construction,
  - fuzz builds use planner-driven randomized pass order.
- Fuzzing is enabled with standard feature cfg (`#[cfg(feature = "solver_order_fuzz")]`), no build-script cfg wiring.
- Features `solver_order_fuzz` and `determinism` are intentionally incompatible (compile-time `compile_error!`).
- Seed control:
  - set `EXPR_SOLVER_ORDER_SEED=<u64>` to reproduce one exact schedule,
  - if unset, a random seed is generated via `rand::random::<u64>()` and printed to stderr as `[solver-order-fuzz] seed=...`.
- The fuzzing schedule randomizes:
  - pass order among `resolve_operator_types`, pending-index/member/int access, specializations, derefs,
  - a two-mode deferred strategy switch:
    - mode A: include `resolve_deferred_types` in the main loop (including stall retry), and skip `full_resolve_deferred_types` at the end,
    - mode B: do not run `resolve_deferred_types` in the main loop, and run `full_resolve_deferred_types` at finalize.
- Invariant goal: successful resolution should remain solver-order independent; this mode is intended to expose hidden order dependencies.

## Unification and Clash Semantics

### How `unify_clusters` decides

`unify_clusters` tries merge in two directions (`found <- wanted`, then reversed). Internally `_try_absorb` handles cases like:

- `Nothing` can absorb most things.
- `Solved(t1)` with `Solved(t2)` succeeds only if `t1 == t2`.
- `Solved` with `IntLike`/`FloatLike` succeeds only if builtin category matches.
- `Func` with `Func` unifies each param/output cluster pair.
- `Struct` with `Struct` requires same struct id + same generic arity, then unifies generic clusters.
- pointer states merge only when `raw`/`mutable` flags are compatible and targets unify.

If both directions fail, `TypeClash` is produced with best-effort `found`/`wanted` type strings generated directly from unresolved cluster structure (`write_*_mock_string_inner` + `extract_clash_type_string`).

### How clashes become user errors

`TypeClash` is the payload embedded in many concrete `TypeError` variants:

- `ValuesContradict` (most expression-level mismatches),
- `AnnotationMismatch` / `PatternAnnotationMismatch`,
- `FunctionOutputAnnotationMismatch` (explicit `fn(..)->T` output annotation vs inferred body result),
- `TypeClashBeforeMentioned` (typedef-related),
- `FieldTypeMismatch`, `TypeDefPatternMismatch`, etc.

Important behavior: a clash does **not** terminate inference globally. Clusters stay separate and inference continues, which allows collecting additional unrelated hard errors.

Orientation note: clash payloads should consistently mean `found = actual/inferred shape` and `wanted = required constraint`. In particular, `force_type`/`unify_*_with_type` paths now keep this same orientation for deferred placeholders (`Func`/`Struct`/`Tuple`/`Array`/`Ptr`) instead of flipping sides when a concrete type requirement fails.

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


# IMPORTANT Operation Order
It's important that inference remain order-independent. 
Errors emitted can be order-dependent, but whether or not a solve is reached must be independent.

This means that unification of types must happen only when they MUST absolutely be exactly equal and under no circumstances.
It's also critical that explicit checks over ResolveKind treat states like Unknown correctly.

For example, when trying to check if a type is a pointer:
```rust
if let ResolveKind::Ptr{..} = state {
  // Good: try to find a pointer
}else{
  // Bad: assume it's never a pointer
}
```

This pattern is wrong if:
1. The state happens to be `Solved` with some pointer type.
2. The state happens to be `Unsolved` and will later solve to a pointer.
3. Depending on why we check, a struct solved to a user type that implements deref can also be correct.

To cover unsolved cases, it's a good idea to push the current information into some sort of queue in ReqState, and then later in the solve loop check whether the unsolved has been resolved.

# IMPORTANT Correctness
It's important to note that ONLY once all requirements have been checked against the CURRENT state can we say that there are no hard errors.
Suppose we have:
```
x+y
```
in our code when both are currently unsolved.
Later we solve x as `fn()->int` and y as `usize`.
The bin_op resolution needs to check this expression so that we get the error.

So it is vital that all methods in the main solver verify anything they need.
It's also critical that new requirements must be put into the Reqs struct and only be pulled out once verified.

## Generic and Specialization Model (Critical)

### Where generics are allowed to appear

Generic binders are intentionally constrained because generic IDs are local/sequential (`GenId(0..n)`) and not globally normalized.

- Function generics:
  - introduced by function generic pattern list,
  - effectively allowed only in top-level generic function signatures (`gather_func_signature::<true>` in global pass),
  - recorded directly on the function type as `TypeValue::Func { generics: Vec<TraitInfo>, ... }`.
- Struct generics:
  - introduced in top-level struct type definitions (`compile_struct_type::<true>` in global typedef pass),
  - stored as `TypeValue::Struct { id, generics: [Generic(GenId(...), TraitInfo)] }` template shape,
  - and the canonical struct rep stores the declared metadata in `StructRep.gen_info: Vec<TraitInfo>`.
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

1. Start from template type (`TypeValue::Func` with non-empty generic metadata, or generic struct template).
2. Allocate fresh cluster list for generic arguments.
3. Run `specialize_type` recursively to substitute every `TypeValue::Generic` occurrence.
4. Unify/force substituted result against call-site or annotation constraints.

Lifetime invariant: every call-site specialization now carries a lifetime map as well. Even when type-generic arity is zero, function signatures are still specialized so solved/global lifetime identities do not leak directly into local inference clusters.

Pointer note: specialization must recurse through `TypeValue::Ptr` as well as function and struct shapes. If pointer wrappers are left unspecialized, methods like `__deref` / `__deref_mut` can incorrectly keep `T` as a global generic id instead of binding it from the receiver (for example `&Box[T] -> &T` called with `Box[int]`), which then produces spurious receiver/type-clash diagnostics.

## Core Type Model

### Stable IDs and sentinels

- `TypeId`, `GenId`, `StructId`: core ids.
- `UNKNOWN_TYPE`, `UNKNOWN_INT_SIZE`, `UNKNOWN_FLOAT_SIZE`: unresolved/weak placeholders for diagnostics.
- `BadTypeId`: legacy wrapper still used by some non-clash diagnostics; clash payloads now use strings directly.

## Place Mutability and Origin Constraints

Current local inference mutability is driven by a single pending-constraint queue:

- `ReqState.pending_mutability_matches` stores `PendingMutabilityMatchRequirement` entries.
- Each entry points at a single projected origin (`projected_origin`); the parent origin is looked up from the origin graph at resolve-time.
- The rule is monotone: if a projected origin is required mutable, its parent origin must also be mutable.

### Constraint sources

- Writable-place checks (`assign`, `&mut`) now enqueue mutability-match requirements instead of using a dedicated writable-place pending queue.
- Casts to mutable pointer/reference targets (`value as &mut T`, raw-mutable pointer forms) now impose a directional mutability requirement on pointer-like sources: source mutability can stay mutable or become more permissive (`mut -> const`), but immutable pointer/reference sources are rejected for `-> mut` casts.
- Projection origin creation for explicit mutable projections marks that origin as requiring mutability.
- Projection origin creation now goes through `new_suborigin(...)`; mutable projections immediately run `mutability_subtype(...)` with `WritablePlaceContext::OriginProjection` so parent/child mutability consistency is checked at construction time.
- Implicit `__deref_mut` checks keep pointer-chain validation and enqueue pending mutability requirements when pointer mutability is unresolved.

### Resolution model

- `resolve_pending_mutability_matches` runs in both stable and fuzzed local solver loops.
- New mutability requirements are stepped immediately via `PendingMutabilityMatchRequirement::step`; only requirements that return `retain` are enqueued. The queue remains the fallback for unresolved constraints rather than the default path.
- Pending mutability checks now also enforce origin/pointer consistency in the opposite direction: when an origin is known immutable, any associated unresolved pointer cluster is pushed to `mutable: Some(false)`, and a hard writable-place error is emitted if that associated pointer is already known mutable.
- If parent mutability is already known mutable, a weak/unknown projected side can resolve immediately (no retain), because the directional constraint is already satisfied.
- Place mutability is still computed via `PlaceKind { access_kind, mutable }` and chain-aware checks (`implicit_deref_sites`), but pending constraints themselves are always origin-to-origin.
- Unknown pointer mutability can be promoted to mutable, but only when constraints permit it:
  - never when strong origin is known immutable,
  - for plain deref writes, only when strong origin is explicitly mutable,
  - for non-deref projection writes (`->`, index, dot with chain), unresolved strong origin may still allow promotion.

### Origin mutability behavior

- `origin_mutability_from_ancestry` was removed.
- Effective origin mutability is cached per `OriginNode` (`effective_mutability`) and read via `SearchState::origin_mutability(origin)`:
  - binding roots that alias another origin (`BindingRoot` with parent) do not override parent mutability,
  - mutable binding roots do not get parent-linked in let-alias flow.
- `SearchState::new_origin(...)` computes `effective_mutability` for the appended node directly instead of triggering whole-graph recomputation on every insertion.
- `AddrOf` now records projection mutability as `Option<bool>` directly (plain `&` no longer forces `false`), so later constraints decide whether mutability can be promoted.
- Origin mutability promotion (`TypeState::set_origin_mutable_if_unknown`) now returns a success flag; local mutability-subtype enforcement uses that result to emit a writable-place diagnostic immediately when a required mutable mark cannot be applied (instead of silently no-op'ing on immutable origins).
- Writable-place diagnostics now try to include an origin cause label (`TypeError::SimpleRelated`) when mutability fails: assignment/`&mut` error sites point at the immutable origin declaration site when available (for example immutable `let` binding that a later deref write flows from).

### Practical pitfalls

- `ResolveKind::Nothing` still means unresolved, not immutable.
- Keep `resolve_pending_mutability_matches` before fallback `force_unresolved_ptr_mutability_to_immut` in solver order.
- For flaky tests that include helper functions plus target `f`, prefer deterministic function selection (tests now prefer function named `f`).

### Type shapes and storage

- `BuiltinType`: primitive set (`int`, sized ints, floats, `bool`, `str`, `void`, `Type`).
- `TypeValue`: builtins, tuple, array, function, pointer, generic param (`Generic`), struct instance (`Struct`).
  - function types now carry an explicit calling convention (`Hot`, `C`, `Unknown`), so diagnostics can print `fn`, `cfn`, or `fn?`.
  - function types also carry declaration-side where-clause metadata for both
    lifetime ordering edges (`'a < 'b`) and generic-lifetime requirements
    (`T<'a`), used today for storage/printing and unused-parameter checks.
- `TypeStore`: interned type arena + struct table.
  - builtins are interned first,
  - structural equality is intern identity,
  - caches per-function unused generic parameter indexes keyed by function `TypeId`,
  - helper predicates for int/float classes and pretty-printing.

### Struct representation

- `StructRep` contains optional name, field list, generic count, and a layout spec (`Hot` vs `C` for `cstruct`).
- `StructRep` also stores declaration-side where-clause metadata for both
  lifetime-order edges and `T<'a`-style generic lifetime requirements.
- Typedef-driven struct names are now assigned during typedef compilation (`do_typedef`), by checking the compiled cluster for either `ResolveKind::Struct` or a solved `TypeValue::Struct` and setting `StructRep.name` only when currently unset.
- Recursive structs are supported by creating struct ids early, then resolving field types in `finalize`.

## Internal Constraint Engine

### Union-find state

- Cluster id: `CId`.
- `ResolveKind` cluster states:
  - `Solved(TypeId)`, `Nothing`, `Never`,
  - weak literals: `IntLike`, `FloatLike`,
  - deferred structures: `Func(FuncInferId)`, `Struct(StructInferId)`, `Tuple(TupleInferId)`, `Array { element, len }`, `Ptr { ... }`.
- `InferState` holds:
  - IR-node -> cluster bindings,
  - union-find arrays (`parent`, `cluster`),
  - deferred operator sites,
  - function/struct placeholder metadata,
  - pending specialization queue,
  - error sink + solved output sink.

### Diagnostic mock types

When unresolved clusters must still be printed in errors, writers now stream best-effort type strings into `&mut String` (`write_mock_type_from_cluster` and shape-specific `write_*_mock_string_inner` helpers), instead of interning temporary mock `TypeId`s for clash payloads.

User-facing clash/type strings now use a function-context name renderer when available:

- During function signature/body gathering, inference installs a per-function `GenLifeNameRender::TextNames` context built from that function's `GenDec`.
- In that context, generic/lifetime placeholders prefer user-written names (`T`, `'a`) and use numeric fallback lifetimes (`'0`, `'1`, ...) for implicit slots.
- Outside function context (global typedef/struct flow), renderer stays `Generate` and uses fallback generated names.
- Error-path struct display no longer appends debug subscript suffixes (typedump/debug formatting still uses existing `TypeStore` formatting paths).

## Inference Pipeline


Main orchestration is two-phase:

1. `infer_global_types`
   - resolves typedefs/structs,
   - resolves function signatures (without body internals),
  - reports unused function generic slots and unused function lifetime slots from solved signature types,
    - examples: `fn[T, U](x:T)->T` reports unused `U`; `fn['a](x:int)->int` reports unused `'a`,
  - reports unused struct generic/lifetime slots from struct field signatures in global typedefs.
     - performs a single per-function-set pass that validates declaration/implementation grouping for both global functions and member methods:
      - when declarations exist, the first declaration is the only reference signature,
      - if that first declaration is unsolved, compatibility checks for that set are skipped,
      - later declarations (`body: None`) must exactly match the first declaration,
      - at most one implementation (`body: Some`) is allowed,
      - if an implementation exists, it must exactly match the reference signature,
    - inserts `SolvedTypes.function_types` (keyed by `NameId`) during that same pass as a single reference entry (`reference_type` + first decl/impl sites),
    - validates and inserts special member overloads (`__add`, unary overload names, `__deref`, `__deref_mut`) in one pass per member method reference type,
    - while inserting deref methods, `__deref`/`__deref_mut` pair compatibility is checked immediately when the second method is seen (instead of a separate post-pass),
    - builds `TypeStore.struct_overloads` inline while walking member method sets so body inference does not repeatedly rescan/reshape member overload declarations at each use site,
   - supports recursive typedef + deferred specialization setup.
2. `infer_value_internals`
  - resolves function body internals or arbitrary value internals,
  - reconciles with known global signatures when present.
  - for function values, local signature/body gathering now anchors to the already-solved global signature before body constraints by unifying directly with the solved `TypeValue::Func` shape.

`run_typechecker` runs global pass, then member methods, then global functions, reporting through `ErrorReporter` and returning solved data or error count. Internal C-side perf hooks are gated via helper wrappers (`perf_*_if_enabled`) and can be disabled at runtime with `EXPR_PARSER_DISABLE_INTERNAL_PERF=1` or forced off in-source with `FORCE_DISABLE_INTERNAL_PERF`.

## Constraint Gathering

### Expressions (`gather_constraints`)

This is the AST/IR-to-typechecker bridge and one of the most important maintenance hotspots. Most new language features need to add or adjust logic here.

High-level behavior:

- literals create weak (`IntLike`/`FloatLike`) or concrete builtin clusters,
- `null` / `nil` literals create deferred pointer clusters with nullable raw-pointer kind (`PtrKind::Solved(PointerStyle::Raw(Nullable::Yes))`) and unknown pointee/mutability until constrained,
- tuple values (`Value::Tuple`) gather element clusters and produce deferred tuple clusters,
- array values (`Value::Array`) unify element clusters and produce deferred array clusters with known length,
- `let`, assignment, and branch joins create equality constraints with `ctx.unify`,
- one-way obligations use `ctx.force_type` (`if`/`while` condition bool, etc.),
- `return` is checked against the current function output cluster (threaded as `Option<CId>` through `gather_constraints`; `None` means outermost/non-function context),
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
  - later pointer solving depends on unification + deferred resolve.
- `Value::Deref`:
  - supports builtin pointer dereference,
  - also supports struct-based smart-pointer style dereference through member methods `__deref` and `__deref_mut`,
  - when both methods exist, dereference target types are constrained to agree,
  - global overload metadata now stores deref method sites as one merged entry (`deref_style`) and records mutability mode as:
    - `Some(false)` when only `__deref` exists,
    - `Some(true)` when only `__deref_mut` exists,
    - `None` when both exist,
  - write-style place checks now consume origin + deref-chain mutability metadata during local inference: `*p = _` requires mutable pointer/reference provenance, and `->` assignment rejects chains containing immutable deref hops,
  - the deref expression now owns a dedicated output cluster and immediately tries a `pointee -> output` unification when the source is already resolvable,
  - when deref starts from an unresolved `Nothing` source, it records a pending pointer-like constraint (`source -> target`) and resolves it in the middle solver instead of eagerly forcing the source to pointer.
- `Value::Access`:
  - supports struct field lookup from solved and deferred struct states,
  - unresolved member-access receivers are deferred into a pending queue (similar to pointer-like deferred solving) and retried in the main solver instead of erroring early,
  - `.` member access performs at most one implicit dereference step (`(*x).field` behavior),
  - `->` member access can chain implicit pointer-like dereference steps (with a safety cap) until lookup resolves,
  - when member lookup has already traversed at least one implicit deref hop and still fails, diagnostics now prefer `UnknownField` over the generic "requires a struct or pointer-like base" message,
- smart-pointer access tries direct member lookup on the current struct first, and only falls back to `__deref`/`__deref_mut` target lookup when direct lookup misses,
- smart-deref steps now track an optional "source" pointer provenance (`PendingImplicitDeref.source: Option<CId>`): pointer-like hops keep/update it, while struct-smart-deref hops may clear it and rely on recorded receiver-chain pointers,
- when a smart-deref hop needs `__deref_mut`, local inference now checks pointer mutability provenance (source pointer or most recent pointer-like chain receiver) and emits a hard error if the chain would require upgrading immutable to mutable,
- all implicit deref hops used by member access and indexing are tracked in `SolvedTypes.implicit_derefs` so later IR lowering can materialize the exact implicit dereference chain,
- writable-place checks for `.` member/tuple writes now also consult those recorded implicit-deref receiver chains (and pending member/int-access queues), so `&mut` / `&'raw mut` bases reached via one implicit dot deref are accepted while immutable bases still produce the dedicated member-access diagnostics,
- local pending metadata now stores implicit-deref receiver chains keyed by expression site id (`ValId -> Vec<CId>`) rather than append-only vectors, so writable-place checks and finalize lookup do O(1)-ish site lookup without reverse linear scans,
- for smart-deref struct hops, the chain records the full step path: pre-deref value type, synthesized self-reference input type (`&self`/`&mut self` shape), deref-method result pointer/reference type, then the pointee target,
  - if a field is not found, member methods are resolved from `program.member_methods`,
  - method access now supports implicit receiver currying: `obj.method` becomes a closure where `self` is already unified/applied when the first parameter is self-like (`self`, `&self`, `&mut self`),
  - important binding detail: inference binds the `Value::Access` node to the **curried** callable type used by the call site, while tracking the full called method signature (`self` still present) in `SolvedTypes.member_method_types`.
  - see `agent_docs/language_semantics.md` for the language-level contract for `.`, `::`, and `->`.
- `Value::Index`:
  - currently only the builtin array case is implemented (no overload/slice semantics yet),
  - gather enqueues indexing sites into a deferred queue (same queue+`retain_mut` pattern used by operators/member access),
  - solver only requires index operand `usize` once base resolves to an actual indexable array-like target,
  - index-base resolution follows pointer-like and smart-deref chains (`struct -> __deref/__deref_mut -> &array -> array`) with a recursion safety cap,
  - if base resolves to `[T; N]`, `[T]`, or pointer/reference to array (`&[T; N]`, `*[T; N]`, etc.), result is constrained to `T`,
  - unresolved bases stay pending; non-array/non-pointer-to-array bases emit a hard error.

Other notable implemented branches:

- `Value::TypeAnnotation` and pattern annotation branches are separate but both unify annotation type with constrained subject.
- `Value::Cast` intentionally does not require source/target equality (it gives target type identity).
- `Value::TypeDef` ensures pattern has type `Type` and registers local typedef clusters.
- `Value::Func` uses signature gather + body gather and unifies body with output cluster.
  - when `fn` has an explicit output type (`-> T`) and body/result clashes with it, inference reports a dedicated annotation-style error (`FunctionOutputAnnotationMismatch`) instead of a generic `ValuesContradict`.
  - closure literals in local/body expression position are intentionally rejected with `sorry we dont support closures`; only top-level function definitions/declarations are supported.
- `Value::Goto`, `Value::Break`, `Value::Continue`, `Value::LabelDecl`:
  - all produce `ResolveKind::Never` clusters,
  - `Never` absorbs into any other type during unification, allowing inference to continue past control flow.

### Patterns and type expressions

- `gather_pattern_constraints*` now threads an optional `parent_origin` through
  recursive pattern gather, so immutable binding roots can inherit provenance
  directly instead of relying only on later patch-up.
- Mutable bindings intentionally do not inherit that parent origin; they remain
  fresh writable roots and only pick up provenance through the older guarded
  post-pass when appropriate.
- `gather_generic_constraints` maps generic parameter bind names to `TypeValue::Generic(GenId, TraitInfo)` and records them in both value-name and type-name local maps.
  - Current bound parsing supports `:dsize` and stores it as `TraitInfo { sized: false }`; unannotated generics use `TraitInfo { sized: true }`.
  - Type-string rendering shows `:dsize` on function signature binders (`fn[T:dsize](...)`) and keeps generic uses as plain `T` (`(T) -> T`) so bounds are displayed at declaration sites.
- Sizedness enforcement now uses the same immediate-check + pending-queue pattern as other deferred constraints:
  - Immediate pass calls `require_sized_or_enqueue(...)`.
  - If sizedness is known now, emit/accept immediately.
  - If cluster is unresolved (`ResolveKind::Nothing` or unknown sentinel types), enqueue `PendingSizedRequirement`.
  - Verification pass runs after deferred type resolution (`resolve_pending_sized_requirements`), outside the main solve loop.
  - This avoids solver-state noise (no dedicated `ResolveKind` for sizedness).
- Current enforced sizedness sites:
  - struct fields must be sized,
  - tuple element types must be sized,
  - generic arguments for sized generic parameters must be sized,
  - function parameter/return types must be sized,
  - local `let` binding type and assignment target type must be sized.
- `compile_type_expr` lowers type syntax to clusters (name refs, wildcard, tuple, inline struct defs, pointers, specialization index).

Critical type-expression fragility points:

- `TypeExpr::NameRef`:
  - may resolve through local generic/type bindings or global typedef/builtin lookup,
  - unresolved globals can intentionally leave placeholder clusters for later solving.
- `TypeExpr::Index` (specialization):
  - one of the most delicate paths,
  - validates base is a global type and currently expects struct specialization,
  - may enqueue `pending_specializations` if base typedef is not yet solved,
  - easy place to introduce generic-id/scope bugs if specialization flow is modified.
- `TypeExpr::Array`:
  - parser/lowering now produce `TypeExpr::Array(element, len)` from bracket type syntax,
  - sized form `[T; N]` lowers to deferred `ResolveKind::Array { element, size: ArrayType::Sized(N) }`,
  - unsized form `[T]` now lowers to deferred `ResolveKind::Array { element, size: ArrayType::Unsized }`.
- inline struct type definitions now reject elided reference lifetimes in fields (for example `inner:&[int;2]`), and require those lifetimes to be declared in the struct lifetime parameter list.
  - implementation detail: struct-field type compilation now uses a dedicated `TypeExprCompileMode::Struct` branch so `&T`/`&'_ T` in struct fields emit a direct hard error instead of silently acting like function-signature elision.
- pointer type expressions (`TypeExpr::Ptr`) feed directly into deferred pointer cluster states, so pointer semantics changes usually require touching both gather and deferred resolution helpers.

Maintenance note: this whole gather layer is intentionally unfinished in places. Treat `NameRef`, `Call`, `Construct`, `TypeExpr::Index`, `AddrOf`, and `Deref` paths as priority review zones whenever adding type-system features, implicit conversions, or dispatch behavior.

## Middle Solver and Finalization

There are now two solver entrypoints:

- `global_solver`: used in global signature/type-def solving; runs only global-safe deferred steps, then `finalize_global`.
- `local_solver`: used for function internals/value solving; runs operator/member/index/deref local queues, then `finalize_local`.

`local_solver` iterates until fixpoint:

1. `resolve_operator_types`
2. `resolve_deferred_types`
3. `resolve_pending_indexes`
4. `resolve_pending_member_accesses`
5. `resolve_pending_int_accesses`
6. `resolve_pending_specializations`
7. `resolve_pending_derefs` (only after the previous passes stop making progress)
8. `finalize_unresolved_lifetimes_as_unknown` (only after deferred queues stall)

It's important that these updates remain order-independent.
Errors emitted can be order-dependent, but whether or not a solve is reached must be independent.

Then `finalize_local`:

- commits solved local value/pattern/member-access data into per-function `InnerFunctionTypes`,
- writes finalized struct field types,
- keeps generic function values as `TypeValue::Func` with explicit per-generic `TraitInfo` metadata,
- emits unresolved errors once per unresolved root (to reduce duplicate noise),
- finalizes member/operator call-site method metadata and implicit deref chains into `InnerFunctionTypes`,
- suppresses duplicate unresolved reporting between curried call-site values and unresolved full member signatures, preferring unresolved receiver/reference value sites when only the full member signature remains unresolved.

`finalize_global` is separate and only commits global products (typedefs, struct field types, solved function signatures and signature metadata).

## SolvedTypes Shape Update

`SolvedTypes` is now split by scope:

- Global data stays global (`typedef_types`, named function/member function signature maps).
- Function internals (`val_types`, `pat_types`, member method call-site types, implicit deref chains) live in `InnerFunctionTypes`, attached to `SolvedFunctionTypes.inner` for each function.

`SolvedFunctionTypes` now carries signature metadata captured during global resolution, including:

- solved function type id,
- impl site,
- declaration sites,
- argument tuples `(PatId, Option<NameId>, TypeId)`,
- generic parameter list,
- lifetime parameter list,
- optional `inner` local-inference results.

This enables local inference to load pre-solved signature info directly instead of re-gathering signatures from syntax.

Named function-set maps now store canonical `ValId` links (not duplicated full solved payloads). The canonical entry in `function_values` is updated with function-set metadata (`impl_site` + `declaration_sites`), and name/member lookups resolve through that id.

## Operator Resolution Notes

- Operators are deferred as `BinOpSite` / `UnOpSite` and revisited during solver iterations.
- Assignment operators are now also deferred for operator-driven cases:
  - `a <op>= b` (`AssignOp::Bin`) is modeled as the same binary operator site as `a <op> b` with output constrained to `a`.
  - `++a` / `a++` / `--a` / `a--` are solved through assignment-op sites that (a) prefer dedicated overload names, then (b) fall back to `__add` / `__sub` with an implicit int-like rhs and output constrained to the target.
  - all assignment forms now gate on mutable-place checks before operator resolution (`var`/`let mut` locals, mutable deref origins, mutable autoderef chains).
- `unify_if_distinct` is the main operator-resolution merge primitive.
- Builtin legality checks are tri-state (`true` / `false` / `unknown`) to avoid premature hard errors.
- Builtin bitwise bool support:
  - `bool & bool`, `bool | bool`, and `bool ^ bool` are valid and preserve `bool`.
  - Compound assignment forms (`&=`, `|=`, `^=`) reuse the same deferred binary-operator resolution path, so bool support applies there too.
- Builtin binary pointer arithmetic now supports raw pointers only: `*T` / `*const T` can do `ptr + int`, `int + ptr`, and `ptr - int` (result keeps pointer type), plus `ptr - ptr` (both operands must be compatible raw pointers, result is `isize`).
- Non-raw references (`&T`, `&mut T`) are intentionally rejected for builtin pointer arithmetic and still produce overload-not-found diagnostics.
- User-struct operator overloads are now enforced through solved member-method signatures:
  - Resolver looks up the method (`__add`, `__neg`, etc.) from `TypeStore.struct_overloads` (populated in global pass).
  - It reads the method type from `SolvedTypes`, and when the solved function type has non-zero `generics`, specializes it into fresh local clusters (`solved_type_to_specialized_local`) before unifying against the expected function shape for the operator site.
  - Receiver currying is centralized in `make_member_closure`: it unifies `self` (including `&self` / `&mut self` via explicit `ResolveKind::Ptr` clusters) and returns a closure-like function cluster with `self` removed from the parameter list.
  - Both binary and unary operator resolution now reuse this same closure helper, then unify that closure against an expected function shape for the operator site.
  - This means operator overload resolution now constrains lhs/rhs/output directly from method signatures (instead of only reporting overload presence).
  - On successful resolution, operator sites are also recorded in `SolvedTypes.member_method_types` so tooling can recover the selected member name and full (uncurried) method signature.
- Deferred operator queues are now drained with `retain_mut`: resolved sites (including successful overload application and hard errors) are removed, while only truly pending/unknown sites are retained for future solver rounds.
- Smart deref target resolution now prefers cached global overload metadata (`TypeStore.struct_overloads`) and no longer re-checks `__deref`/`__deref_mut` target compatibility per deref/member/index callsite; compatibility is validated once during global signature checks.
- Deferred deref/member/index result unifications now use `actual_result -> constrained_output` ordering so clash payloads read naturally as `found <actual>, expected <constraint>`.
- Operator overload resolution assumes global signatures are already solved before function-body inference (`infer_global_types` first); missing method type/function-shape in this stage is treated as an internal-invariant violation (`unreachable!`).
- Even though runtime/operator-call dispatch is still TODO, signature validation now enforces shape rules for special member names:
  - binary overload names (`__add`, `__sub`, etc.): `self`-like first parameter + exactly one extra parameter,
  - unary overload names (`__neg`, `__not`, `__bitnot`, `__pre_inc`, `__post_inc`, `__pre_dec`, `__post_dec`): `self`-like first parameter + no extra parameters,
  - `__deref`: first parameter must be `&self`, no extra parameters, and return type must be a non-raw shared reference (`&T`).
  - `__deref_mut`: first parameter must be `&mut self`, no extra parameters, and return type must be a non-raw mutable reference (`&mut T`).
  - if both `__deref` and `__deref_mut` exist on the same struct, both must dereference to the same `T` target.
  - `__free` / `__user_free`: first parameter must be exactly `&mut Struct[T0, T1, ...]` (all struct generics free and in declaration order), no extra parameters, and return type must be `void`.
  - `__size_of`: first parameter must be a reference receiver (`&self`, with future `&'raw self` intent), no extra parameters, and return type must be `usize`.
  - `__align_of`: first parameter must be `self`-like, no extra parameters, and return type must be `usize`.
  - builtin fallback methods now exist for **any receiver type** (including unresolved/generic/builtin/reference forms):
    - `x.__free()` is available with fallback shape `fn[T](&mut T)->void`,
    - `x.__size_of()` is available with fallback shape `fn[T](&T)->usize`,
    - `x.__align_of()` is available with fallback shape `fn[T](T)->usize`,
    - struct-defined member methods still win when present for that struct+name.
  - `__user_free` is still reserved and treated as an unknown builtin member name when used as a member method.
- The validation is now based on solved global function type signatures (`TypeValue::Func`), not raw signature clusters.
- Member method names that start with `__` and do not end with `_` are treated as reserved builtin names; unknown reserved names emit a dedicated type error.

## Near-Term Roadmap / Intent

- Current status: we have first IR lowering and type inference metadata, but semantic materialization is still split into later phases.
- Next phase intent: run value/ownership analysis that decides where values are consumed, borrowed, or mutated, then inserts implicit operations (`__free`, implicit member method calls, implicit deref steps) explicitly.
- The inferred implicit deref/member metadata (`SolvedTypes.member_method_types`, `SolvedTypes.implicit_derefs`) is intentionally staged for that pass; it is not the final execution-level rewrite yet.
- Medium-term intent: introduce a new IR tier dedicated to post-typecheck value semantics (ownership/borrows/destruction + explicit implicit-op insertion), so later backend/codegen phases do not rely on typechecker-only side channels.
- Smart-pointer target: make a `Box`-style type a first-class validation case for deref/member flows; this mostly needs external-type integration and ABI/foreign-call tagging.
- External/ABI syntax/semantics notes now live in `agent_docs/language_semantics.md`.

## Lifetime Inference Plan (Design Notes)

This section records the intended implementation shape for adding lifetimes to typing/inference before full borrow checking.

### Core semantic constraints

- No automatic user-level lifetime downcast: if code wants a shorter/derived borrow, it must express reborrow explicitly (`&*x` style).
- `'raw` is a first-class and distinct lifetime state, not a fallback variant of regular lifetimes.
- Inference must not coerce `&'a T` into `&'raw T` by unification side effects.

### Smart-pointer and deref signatures

- Safe deref remains tied: fn['a](&'a self)->&'a out.
- Raw receiver deref is allowed: fn['a](&'raw self)->&'a out.
- Raw address exposure is allowed: fn['a](&'raw self)->&'raw out.
- `&mut 'raw` must remain distinct from normal noalias mutable borrows in later ownership/borrow phases.

### Implicit cast recording requirements

- Access/index/deref-chain logic synthesizes fresh references; these are implicit cast/reborrow sites.
- Type inference should record each cast edge with enough metadata for borrow analysis to revisit legality.
- Temporary policy: inferred implicit casts may target any lifetime (including 'raw or 'static).
- Borrow analysis will become the enforcing stage that accepts/rejects these recorded casts.

### Immediate typecheck rejections vs deferred borrow checks

- Hard reject in typecheck when there is direct lifetime contradiction with no reborrow relation.
  - Canonical example: f(x:&'a t)->&'b t{x} should fail immediately.
- Reborrow-driven relations (for example generated 'b < 'a) should be preserved as constraints and deferred to borrow checking.

### Unnamed lifetime policy

- Global signatures:
  - explicit function lifetime generics are accepted and mapped to local signature lifetime slots (they are not overwritten by elision inference),
  - input-side unnamed lifetimes => fresh independent named/bound slots,
  - output-side unnamed lifetime => intended join over input lifetimes.
  - implementation detail: signature compile now tracks lid ranges for input/output compilation, assigns concrete external lifetimes to new input-elided lids, then applies output elision by directly inspecting newly-created output lids (no output-type rewalk).
  - temporary implementation fallback: when exactly one implicit input lifetime exists, rewrite all output-elided lifetime slots to that lifetime; otherwise emit a type error and keep unresolved output lifetime slots so inference can continue.
- Function bodies:
  - always mint fresh lifetime ids for unnamed/unconstrained lifetimes,
  - store all minted ids so borrow checking can allocate dense per-lifetime vectors indexed by id.
  - pre-finalize fallback now resolves unresolved lifetime roots (by iterating lifetime union-find roots) to `Unknown`, rather than scanning type clusters.

### Suggested implementation staging

1. Extend type model with lifetime ids/variables and raw-vs-nonraw reference distinction preserved through unification.
2. Teach gather layer to mint body lifetimes, instantiate signature lifetimes, and emit cast/reborrow metadata for implicit receiver/index/deref steps.
3. Add immediate contradiction checks in local body typing for impossible returns/signature equalities (without waiting for borrow pass).
4. Thread lifetime metadata into `SolvedTypes` (or adjacent solved artifact) so later borrow checker can consume a stable, fully-indexed graph.
5. Implement borrow-analysis phase that validates cast edges, reborrow bounds, and raw-specific escape rules.

## Error and Test Philosophy

- `TypeError` is intentionally detailed and source-oriented; clashes carry `found`/`wanted` type payloads.
- Engine prefers collecting multiple deterministic hard errors over bailing early.
- Inline tests cover happy paths and representative failure modes (unresolveds, annotation/branch mismatch, generic specialization, recursive structs, operator legality).

## InferState Refactoring (In Progress)

### Goal
The refactoring goal was to reduce the number of arguments in internal helper functions by grouping related state parts into structs that can be passed as a single `&mut` reference instead of many individual `&mut` parts.

### State Structure (Current)
```
InferState
├── ExternState   (store, program, errors, ans)
├── SearchState   (val_cluster, pat_cluster, typedef_cluster, local_types, names{name -> (cid, kind)}; kind -> derived Origin)
├── TypeState
│   ├── TypeCore  (parent, cluster) - union-find
│   └── TypeExtra (func_defs, struct_defs, struct_infers, tuple_infers)
└── ReqState      (bin_op_sites, un_op_sites, pending_specializations, member_method_type_sites, etc.)
```
