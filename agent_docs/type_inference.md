# Type System Direction

The active type-system code lives under `src/type_system/`. Treat the old hybrid
type/kind/lifetime implementation as legacy reference material only; do not copy
its architecture forward unless a task explicitly asks for a behavior port.

## Current Shape

`src/type_system/mod.rs` is the public re-export facade. The main modules are:

- `kinds.rs`: ids, primitive kind enums, pointer/lifetime shape enums,
  `TypeKind`, builtin `KindId` constants, and naming helpers.
- `solving.rs`: `TypeUniverse`, intern/storage/lookup state, struct metadata,
  mutability solver data, solved-type records, origin records, and kind display
  helpers that need solver state.
- `operator_solver.rs`: projections and operator/member-access helpers.
- `origin.rs`: origin/provenance ids and nodes.
- `lifetime_solve.rs`: experimental lifetime graph/order helpers.
- `errors.rs`: typechecker diagnostic payloads.

The core shape id is `KindId`; a solved-enough `TypeKind`/`KindId` is the type
the rest of the compiler prints and stores. Do not reintroduce the old
`TypeId`/`UNKNOWN_TYPE` sentinel scheme into the new code.

`TypeUniverse` owns two deliberately separate halves:

- `KindStorage`: immutable/interned shape storage and per-id payload stores.
- `KindLookUp`: union-finds for `KindId`/`PtrId`/lifetime equality, lifetime
  ordering, and the mutability implication solver.

This split lets code borrow storage while mutably path-compressing lookup state.
Prefer signatures that take `&TypeUniverse.storage` plus `&mut TypeUniverse.look`
or narrower field borrows when possible.

## Interning And Builtins

`TypeIntern` is structural hash-consing over concrete immutable `TypeKind`s:

```text
HashMap<TypeKind, KindId>
IndexVec<KindId, Option<TypeKind>>
```

`None` slots are unsolved kinds and must not participate in structural
interning. Partial builtin shapes may be interned only when they are monotone
solver facts that can refine toward a more precise shape. Hardcoded builtin
`KindId`s must be fully known because they are global language constants.

`TypeUniverse::new()` pre-interns `HARD_CODED_BUILTIN_KINDS` before allocating
unknown/user kinds. `KindId` associated constants rely on that fixed order.
Source-level builtin aliases should resolve through `BUILTINS` rather than being
added as separate hardcoded shapes.

Variable-length payloads use `IndexSpan<KindId>` / `IndexSpan<LifeId>` rows.
`TypeUniverse::intern_kind_span` and `TypeUniverse::intern_life_span` first reuse
existing contiguous rows and allocate alias rows only when needed.

Pointer styles must be allocated through `TypeUniverse::add_ptr_style`, not by
directly pushing into storage. This keeps `KindStorage` pointer-style rows and
`KindLookUp.ptr` in lockstep.

## Mutability

Pointer shapes carry a `MutId`; the meaning of that id lives in
`KindLookUp.mutable` (`MutInfo`). The fixed roots are `MutId::FALSE` and
`MutId::TRUE`; other nodes are unknowns with implication edges.

Display uses `KindStorage::kind_to_string(&mut KindLookUp, ...)` for final
defaulted output. Diagnostics that run before defaulting can use
`kind_to_string_with_mut_guess(..., MutGuessMode::UnknownAsUnknown)` to render
unresolved pointer mutability as `?mut`.

Pending mutability implication edges are deterministic `BTreeSet<MutId>`
destinations. Diagnostic reasons are node-owned; each node keeps at most one
reason path, preferring lower-depth paths to avoid duplicate reports.

## Solved Data

`SolvedTypes` is the result shape used by diagnostics and debug dumps:
`function_values`, inner value/pattern types, member method accesses, implicit
deref records, and origin dump data. Type ids are `KindId`s and printing goes
through root-compressing `KindStorage::kind_to_string(&mut KindLookUp, ...)`.

Local dump/reporting paths must render `InnerFunctionTypes` ids against that
function's `my_universe`; only external function signatures and global
definitions live in the outer/global `TypeUniverse`.

Implicit deref records store each hop as `(KindId, Projection)` in `SolvedTypes`.
Projection definitions live in `src/type_system/operator_solver.rs`.

## Equality Solver

Shape solving should answer questions like:

- is this an integer, function, struct, tuple, array, pointer, or generic shape?
- what are the function parameters and output shape?
- what struct/member/operator candidate applies?
- where are implicit derefs inserted?
- what shape constraints do annotations and calls impose?

Shape solving should not prove borrow legality, lifetime ordering, or mutable
access permission.

Baseline `unify` is equality over `KindId`/kind data only. It should recursively
merge equal shape and delegate non-shape equality to the relevant variable
systems. Pointer mutability should merge through `MutInfo`; lifetime equality
should merge through lifetime solver state.

`TypeUniverse::unify(found, wanted)` canonicalizes both `KindId`s with the kind
union-find, tries to absorb `found` into `wanted`, then tries the reverse before
reporting a `TypeClash`. Unknown kind slots absorb concrete shapes. Concrete
shapes recursively unify tuples, structs, functions, arrays, and pointers.
Partial builtin integer, float, array-size, and raw-pointer nullability fields
are refined by strict `Option` merging.

`unify` is intentionally non-transactional. A failed equality may leave useful
monotone refinements behind, and same-shape recursive unification should keep
walking compatible child positions before returning failure. Do not add rollback
or short-circuiting unless solver policy changes deliberately.

Directional relations should not be folded into baseline `unify`. A
subtyping/coercion operation should be a separate mutating constraint method,
named `require_subtype` rather than `subtype_of`, because it can emit ordered
mutability/lifetime constraints such as `actual <= expected` instead of simply
answering a predicate.

## Pending Requirements And Obligations

Calls, operators, member access, indexing, deref, and specialization should be
represented as pending requirements. A requirement becomes runnable when its
discriminator type has enough shape information.

Examples of discriminator choices:

- `a + b`: discriminate on `a`.
- `a.b`: discriminate on `a`.
- `a[b]`: discriminate on `a`.
- `f(x)`: discriminate on `f`.
- unary operators: discriminate on the operand.

When a pending requirement fires, it should emit shape equalities, emit more
pending requirements, emit non-shape obligations, resolve to a unique
implementation/result, stay pending, or report an error at finalization.

Keep watcher edges small by watching only discriminator type variables, not all
inputs of a requirement. Design for deterministic, mostly linear behavior over
normal AST constraint graphs, while accepting that adversarial dense graphs can
be worse.

Non-shape facts should be recorded as obligations and checked after shape
solving reaches a fixed point. Examples include writable/readable place checks,
mutable access requirements, safe-reference requirements, raw-pointer access,
implicit deref/reborrow facts, declared lifetime where-clauses, and body-induced
outlives relationships.

Late defaulting belongs with obligations. For example, unconstrained pointer
style may choose a minimal/default lattice value after all shape uses are known,
as long as the choice is consistent across linked variables and does not change
whether shape solving succeeds.

## Current Local Constraint Scaffold

`src/type_system/local_inference.rs` currently contains an intentionally
incomplete `gather_constraints` sketch whose main job is to show where facts
should be collected, not to fully typecheck the language yet.

- The recursive value walker returns `(KindId, Option<OriginId>)` today.
- That uses `OriginId` instead of raw `Origin` because derived provenance nodes
  need stable parent ids (`OriginKind::Derived { parent, ... }`).
- Global function signatures are no longer reused by raw `KindId` inside local
  inference. Bodies and external calls re-lower/specialize signatures into the
  local `TypeUniverse`, because global and local universes have unrelated ids.
- This is not just a conceptual distinction: the same raw `KindId` number can
  legitimately refer to different shapes in the global and local universes, so
  specialization must rebuild shape structure locally instead of forwarding ids.
- Signature gathering is a two-phase pass: gather every global/member function
  signature first, then walk implementation bodies. This keeps local call
  specialization stable even when a body references a later declaration.
- `UniversalLifeId::STATIC` is reserved as id `0`; declared function lifetime
  parameters start at `1` and are tracked by vector position
  (`IndexVec<UniversalLifeId, Option<...>>` with `None` at slot `0`). Lifetime
  where-clauses like
  `'a < 'b` are recorded as ordered universal-lifetime edges.
- Specializing an external function call replays those ordered lifetime edges
  into the caller's local universe at the same time the callee signature is
  lowered there.
- Function-local `Value::Func` still follows the old behavior and emits the
  direct closure error `sorry we dont support closures`.
- Borrow/address-of currently preserves the underlying place origin instead of
  inventing a fresh derived node; later obligation plumbing should record the
  borrow operation itself.
- Function calls currently produce a transient result origin and mark the call
  site as the future hook for callable-shape requirements.
- Function body result vs declared output already reports
  `FunctionOutputAnnotationMismatch` in the current scaffold, so direct
  monomorphic calls lowered into the local universe can surface wrong-return-type
  errors even before the richer pending-requirement machinery exists.
- Access, index, deref, casts, lets, matches, and pattern binding all have
  explicit slots where future equality constraints and later origin/lifetime
  obligations should be attached.

## Lifetime Boundary

Lifetime variables may use union-find for equality discovered through SCC
collapse, but lifetime ordering is graph/lattice logic, not shape equality.

The lifetime phase should consume solved shape data plus obligations, build an
origin/order graph, collapse SCCs, validate equality components, and check
directed outlives edges. This likely wants a middle IR before it becomes a real
borrow checker.

Do not port old `CId`/`LId` coupling forward as a design requirement.
