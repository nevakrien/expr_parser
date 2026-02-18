# Type Inference System Summary (`src/type_inference.rs`)

## High-Level Philosophy

The implementation is intentionally constraint-first and error-tolerant:

1. Make straightforward inference obvious (`let`, annotations, function signatures, literals).
2. Keep going after clashes so multiple diagnostics are emitted in one run.
3. Leave room for future features (overloads/lifetimes), even if parts are TODO.

There is also an explicit engineering style used in the file: many internal "workhorse" helpers take *parts* of `InferState` (`store`, `parent`, `cluster`, `func_defs`, `struct_infers`, etc.) instead of taking `&mut InferState` wholesale. This is deliberate to enable complex borrow patterns and avoid borrow-checker dead ends during recursive unification and deferred solving.

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
  - recorded directly on the function type as `TypeValue::Func { generics, ... }`.
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

1. Start from template type (`TypeValue::Func` with non-zero `generics`, or generic struct template).
2. Allocate fresh cluster list for generic arguments.
3. Run `specialize_type` recursively to substitute every `TypeValue::Generic` occurrence.
4. Unify/force substituted result against call-site or annotation constraints.

Pointer note: specialization must recurse through `TypeValue::Ptr` as well as function and struct shapes. If pointer wrappers are left unspecialized, methods like `__deref` / `__deref_mut` can incorrectly keep `T` as a global generic id instead of binding it from the receiver (for example `&Box[T] -> &T` called with `Box[int]`), which then produces spurious receiver/type-clash diagnostics.

## Core Type Model

### Stable IDs and sentinels

- `TypeId`, `GenId`, `StructId`: core ids.
- `UNKNOWN_TYPE`, `UNKNOWN_INT_SIZE`, `UNKNOWN_FLOAT_SIZE`: unresolved/weak placeholders for diagnostics.
- `BadTypeId`: wrapper used when diagnostics can include unresolved internals.

### Type shapes and storage

- `BuiltinType`: primitive set (`int`, sized ints, floats, `bool`, `str`, `void`, `Type`).
- `TypeValue`: builtins, tuple, array, function, pointer, generic param (`Generic`), struct instance (`Struct`).
  - function types now carry an explicit calling convention (`Hot`, `C`, `Unknown`), so diagnostics can print `fn`, `cfn`, or `fn?`.
- `TypeStore`: interned type arena + struct table.
  - builtins are interned first,
  - structural equality is intern identity,
  - helper predicates for int/float classes and pretty-printing.

### Struct representation

- `StructRep` contains optional name, field list, generic count, and a layout spec (`Hot` vs `C` for `cstruct`).
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

When unresolved clusters must still be printed in errors, mock builders (`mock_type_from_cluster`, `make_func_mock`, `make_struct_mock`, `make_ptr_mock`) synthesize best-effort type shapes.

## Inference Pipeline


Main orchestration is two-phase:

1. `infer_global_types`
   - resolves typedefs/structs,
   - resolves function signatures (without body internals),
    - performs a single per-function-set pass that validates declaration/implementation grouping for both global functions and member methods:
      - when declarations exist, the first declaration is the only reference signature,
      - if that first declaration is unsolved, compatibility checks for that set are skipped,
      - later declarations (`body: None`) must exactly match the first declaration,
      - at most one implementation (`body: Some`) is allowed,
      - if an implementation exists, it must exactly match the reference signature,
    - inserts `SolvedTypes.function_types` (keyed by `NameId`) during that same pass as a single reference entry (`reference_type` + first decl/impl sites),
   - validates special member method signatures (`__add`, unary overload names, `__deref`, `__deref_mut`) against each method set reference type,
   - builds `TypeStore.struct_overloads` inline while walking member method sets (validated `__deref` / `__deref_mut` and operator overload entries) so body inference does not repeatedly rescan/reshape member overload declarations at each use site,
   - supports recursive typedef + deferred specialization setup.
2. `infer_value_internals`
  - resolves function body internals or arbitrary value internals,
  - reconciles with known global signatures when present.
  - for function values, local signature/body gathering now anchors to the already-solved global signature before body constraints by unifying directly with the solved `TypeValue::Func` shape.

`run_typechecker` runs global pass, then member methods, then global functions, reporting through `ErrorReporter` and returning solved data or error count.

## Constraint Gathering

### Expressions (`gather_constraints`)

This is the AST/IR-to-typechecker bridge and one of the most important maintenance hotspots. Most new language features need to add or adjust logic here.

High-level behavior:

- literals create weak (`IntLike`/`FloatLike`) or concrete builtin clusters,
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
  - the deref expression now owns a dedicated output cluster and immediately tries a `pointee -> output` unification when the source is already resolvable,
  - when deref starts from an unresolved `Nothing` source, it records a pending pointer-like constraint (`source -> target`) and resolves it in the middle solver instead of eagerly forcing the source to pointer.
- `Value::Access`:
  - supports struct field lookup from solved and deferred struct states,
  - unresolved member-access receivers are deferred into a pending queue (similar to pointer-like deferred solving) and retried in the main solver instead of erroring early,
  - `.` member access performs at most one implicit dereference step (`(*x).field` behavior),
  - `->` member access can chain implicit pointer-like dereference steps (with a safety cap) until lookup resolves,
  - smart-pointer access tries direct member lookup on the current struct first, and only falls back to `__deref`/`__deref_mut` target lookup when direct lookup misses,
  - all implicit deref hops used by member access and indexing are tracked in `SolvedTypes.implicit_derefs` so later IR lowering can materialize the exact implicit dereference chain,
  - for smart-deref struct hops, the chain records both the pre-deref receiver type and the deref-method result pointer/reference type (before the pointee target),
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

- `gather_pattern_constraints*` handles bind/wildcard/annotated patterns and binds names to clusters.
- `gather_generic_constraints` maps generic parameter bind names to `TypeValue::Generic(GenId)` and records them in both value-name and type-name local maps.
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
- pointer type expressions (`TypeExpr::Ptr`) feed directly into deferred pointer cluster states, so pointer semantics changes usually require touching both gather and deferred resolution helpers.

Maintenance note: this whole gather layer is intentionally unfinished in places. Treat `NameRef`, `Call`, `Construct`, `TypeExpr::Index`, `AddrOf` (and future `Deref`) as priority review zones whenever adding type-system features, implicit conversions, or dispatch behavior.

## Middle Solver and Finalization

`main_solver` iterates until fixpoint:

1. `resolve_operator_types`
2. `resolve_deferred_types`
3. `resolve_pointer_likes`
4. `resolve_pending_indexes`
5. `resolve_pending_member_accesses`
6. `resolve_pending_specializations`

It's important that these updates remain order-independent.
Errors emitted can be order-dependent, but whether or not a solve is reached must be independent.

Then `finalize`:

- commits solved typedef/value/pattern types into `SolvedTypes`,
- writes finalized struct field types,
- keeps generic function values as `TypeValue::Func` with a non-zero `generics` count,
- emits unresolved errors once per unresolved root (to reduce duplicate noise),
- finalizes `SolvedTypes.member_method_types` from deferred member/operator call sites,
- finalizes `SolvedTypes.implicit_derefs` for value sites that used implicit dereference hops (member access + index),
- suppresses duplicate unresolved reporting between curried call-site values and unresolved full member signatures, preferring unresolved receiver/reference value sites when only the full member signature remains unresolved.

## Operator Resolution Notes

- Operators are deferred as `BinOpSite` / `UnOpSite` and revisited during solver iterations.
- Assignment operators are now also deferred for operator-driven cases:
  - `a <op>= b` (`AssignOp::Bin`) is modeled as the same binary operator site as `a <op> b` with output constrained to `a`.
  - `++a` / `a++` / `--a` / `a--` are solved through assignment-op sites that (a) prefer dedicated overload names, then (b) fall back to `__add` / `__sub` with an implicit int-like rhs and output constrained to the target.
- `unify_if_distinct` is the main operator-resolution merge primitive.
- Builtin legality checks are tri-state (`true` / `false` / `unknown`) to avoid premature hard errors.
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
  - `__free`: first parameter must be `&mut self`, no extra parameters, and return type must be `void`.
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
  - input-side unnamed lifetimes => fresh independent named/bound slots,
  - output-side unnamed lifetime => intended join over input lifetimes.
  - temporary implementation fallback: when exactly one input lifetime exists, use it; otherwise emit `not implemented yet`.
- Function bodies:
  - always mint fresh lifetime ids for unnamed/unconstrained lifetimes,
  - store all minted ids so borrow checking can allocate dense per-lifetime vectors indexed by id.

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
├── SearchState   (val_cluster, pat_cluster, typedef_cluster, local_types, names)
├── TypeState
│   ├── TypeCore  (parent, cluster) - union-find
│   └── TypeExtra (func_defs, struct_defs, struct_infers, tuple_infers)
└── ReqState      (bin_op_sites, un_op_sites, pending_specializations, member_method_type_sites, etc.)
```
