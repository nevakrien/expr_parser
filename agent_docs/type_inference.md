# Type System Refactor Direction

The existing type-inference implementation is being treated as legacy reference
material. Do not extend the old hybrid type/kind/lifetime solver unless the
task is explicitly to preserve or extract behavior before the clean-room
refactor.

A local detached snapshot of the current GitHub `main` branch at the time it
was created exists at:

```text
/home/user/Desktop/rust_stuff/expr_parser/expr_parse_pre
```

Use that reference tree for old implementation details, historical behavior, and
test comparison. It is meant to stay frozen on this machine until someone
explicitly refreshes it. The active branch should move toward the design below.

## Refactor Summary

The new type system should split solving into distinct phases instead of using
one mixed solver for type equality, kind checks, pointer style, mutability,
lifetimes, overloads, and borrow-like relationships.

Target phase boundary:

1. Shape solving: equality over type shape only.
2. Pointer-style obligations: mutability/rawness/reference capability checks and
   late defaulting.
3. Lifetime solving: origin/order graph extraction, SCC collapse, and outlives
   validation.
4. Borrow/storage checking: later, likely over a middle IR rather than the AST.

The most important architectural rule is that union-find is for equality.
Outlives constraints, mutability permissions, raw/safe conversions, and borrow
legality are not shape equality.

## Current Scaffold

The active crate now has the first clean-room scaffold in `src/type_kinds.rs`.
The core shape id is `KindId`; a solved-enough `TypeKind`/`KindId` is the type
the rest of the compiler prints and stores. Do not reintroduce the old
`TypeId`/`UNKNOWN_TYPE` sentinel scheme.

`TypeUniverse` owns two deliberately separate halves:

- `KindStorage`: immutable/interned shape storage and per-id payload stores.
- `KindLookUp`: union-finds for `KindId`/`PtrId`, lifetime ordering, and the
  mutability implication solver.

This split is important for Rust borrowing: common code should be able to hold a
reference into storage while mutably path-compressing a union-find, using
signatures like `&TypeUniverse.storage` plus `&mut TypeUniverse.look` or narrower
field borrows.

Mutability is no longer stored as `Option<bool>` in `KindStorage`. Pointer shapes
still carry a `MutId`, but the meaning of that id lives in `KindLookUp.mutable`
(`MutInfo`): `MutId::FALSE`, `MutId::TRUE`, or an unknown node with implication
edges. `TypeUniverse::kind_to_string` defaults unresolved mutability to const for
final display, while `kind_to_string_with_mut_guess(...,
MutGuessMode::UnknownAsUnknown)` renders unresolved pointer mutability as `?mut`
for diagnostics that run before all limitations/defaults have been inserted.

`TypeIntern` is structural hash-consing:

```text
HashMap<TypeKind, KindId>
IndexVec<KindId, Option<TypeKind>>
```

Equal immutable concrete shapes intern to the same `KindId`. `None` slots are
allowed in storage for completely unsolved kinds, but those unknown slots do not
participate in structural interning because one unknown may later resolve to many
contradictory concrete shapes. If solving refines shape, it should intern the
refined shape and connect ids through union-find rather than mutating a hashmap
key in place.

`SolvedTypes` intentionally preserves the old result shape used by diagnostics
and debug dumps: `function_values`, inner value/pattern types, member method
accesses, implicit deref records, and origin dump data remain present, but all
old type ids are now `KindId`s and printing goes through `TypeUniverse::kind_to_string`.
Implicit derefs currently use the old solved-data model again: each recorded hop
is stored as `(KindId, Projection)` in `SolvedTypes`, preserving the deref-chain
list while attaching the projection kind for each step. The projection data model
now lives in `src/operator_solver.rs` rather than in `TypeKind`.

## Shape Solver

Shape solving should answer questions like:

- is this an integer, function, struct, tuple, array, pointer, or generic shape?
- what are the function parameters and output shape?
- what struct/member/operator candidate applies?
- where are implicit derefs inserted?
- what shape constraints do annotations and calls impose?

Shape solving should not prove borrow legality, lifetime ordering, or mutable
access permission.

A compact shape state is preferred, with larger payloads behind arena ids:

```text
ShapeState:
  Unknown
  Error
  Bool / Void / Never
  Int(...)
  Float(...)
  Func(FuncShapeId)
  Struct(StructId, generic_type_vars)
  Tuple(TupleShapeId)
  Array(ArrayShapeId)
  Ptr(PtrShapeId)
  Generic(...)
```

Pointer shapes should reference pointer info instead of putting lifetime,
mutability, and rawness fields on every type:

```text
PtrShape:
  target: TypeVarId
  info: PtrInfoId

PtrInfo:
  lifetime: LifetimeVarId
  mutability: MutabilityVarId
  rawness: RawnessVarId
```

`ThinVec` is worth considering for fields that are usually empty or tiny, such
as function params, tuple fields, generic args, lifetime args, where-clause
edges, and watcher lists. The bigger win is still keeping pointer/lifetime data
off non-pointer shapes entirely.

## Pending Requirements

Calls, operators, member access, indexing, deref, and specialization should be
represented as pending requirements. A requirement becomes runnable when its
discriminator type has enough shape information.

Examples of discriminator choices:

- `a + b`: discriminate on `a`.
- `a.b`: discriminate on `a`.
- `a[b]`: discriminate on `a`.
- `f(x)`: discriminate on `f`.
- unary operators: discriminate on the operand.

The one-discriminator rule is an intentional simplification. It avoids
multi-parameter overload resolution and keeps pending edges sparse in normal
programs.

When a pending requirement fires, it should either:

- emit shape equalities,
- emit more pending requirements,
- emit non-shape obligations,
- resolve to a unique implementation/result,
- stay pending because the discriminator is still too unknown,
- or report an error at finalization if it cannot be resolved.

Readiness must be explicit. Do not treat arbitrary partial information as proof
that a requirement is ready. For example, an addition requirement can fire when
the left side is known as int-like, float-like, pointer-like, struct-like, or a
generic with an applicable operator bound. It should not fire just because some
lifetime or mutability field is known.

## Complexity Notes

A global fixed-point scan is simple but repeatedly touches every pending
requirement. Watcher lists are not asymptotically magic: if one variable watches
all requirements, waking it can still be expensive. They are still preferred in
practice because ordinary programs should have sparse expression/constraint
graphs.

Keep watcher edges small by watching only discriminator type variables, not all
inputs of a requirement. For `a + b`, the pending addition watches `a`; after
`a` resolves to an int-like shape, it emits equality constraints tying `b` and
the output to `a` and then leaves the pending set.

Design for deterministic, mostly linear behavior over normal AST constraint
graphs, while accepting that adversarial dense graphs can be worse.

## Obligations

Non-shape facts should be recorded as obligations and checked after shape
solving reaches a fixed point.

Examples:

- requires writable place,
- requires readable place,
- requires mutable access,
- requires safe reference,
- permits raw pointer access,
- records implicit deref/reborrow facts,
- records declared lifetime where-clause requirements,
- records body-induced outlives relationships.

Late defaulting belongs here. For example, an unconstrained pointer style such
as `&?raw ?mut int` can choose a minimal/default lattice value after all shape
uses are known, as long as the choice is consistent across the linked variables
and does not change whether shape solving succeeds.

## Lifetime Direction

Lifetime variables may use union-find for equality discovered through SCC
collapse, but lifetime ordering is graph/lattice logic, not shape equality.

The long-term lifetime phase should consume solved shape data plus obligations,
build an origin/order graph, collapse SCCs, validate equality components, and
check directed outlives edges. This likely wants a middle IR before it becomes a
real borrow checker.

Do not port old `CId`/`LId` coupling forward as a design requirement. If old
behavior is needed, inspect the reference tree.
