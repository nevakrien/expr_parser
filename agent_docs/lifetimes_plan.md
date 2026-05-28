# Lifetime Plan (Origin-Threaded Lifetime Model)

This is the primary lifetime-planning document.
It describes the lifetime model centered on **origin-threaded provenance** and
late lifetime validation / SCC solving.

The lifetime system operates in stages:

1. **Declaration-time well-formedness recording**
2. **Local lifetime solve** (per function body)
3. **Global lifetime composition** (across functions)

The declaration stage records lifetime structure and declared ordering facts
from signatures / typedefs / structs. The local stage constructs additional
body-induced ordering constraints within a single function and detects purely
local contradictions. The global stage later composes summaries across
functions.

This document now focuses on the **origin-threaded representation** needed to
support both declared well-formedness requirements and body-induced borrow
constraints.

---

# Problem Summary

Current local inference still falls back to assigning `LifeTime::Unknown`
when lifetime roots remain unresolved near the end of solving.

While this is useful for debugging, it destroys ordering information and
prevents detection of invalid lifetime relationships.

The missing behaviors today are:

- nested reference types should enforce well-formedness requirements such as
  `&'a &'b T` requiring `'a <= 'b`
- struct and signature declarations should preserve declared lifetime-ordering
  requirements as metadata
- reference-producing operations should generate ordering constraints
- local variable storage should bound lifetimes
- equality relationships should be detected through SCC collapse
- invalid merges between lifetimes should be detected explicitly
- local results should be exportable to a global lifetime solve

Instead of erasing information by assigning unknown lifetimes early, we keep
the provenance structure that created each reference and analyze lifetime
relationships structurally.

---

# Core Model

The lifetime system is represented by two connected layers:

- **Origin trees** that describe where references come from, including nested
  reference structure in declarations and value formation.
- **Late lifetime graphs** built from those origins plus `LId` data near the end
  of solving.

The key design point is that gather/type-compile code should thread parent
origin information directly instead of retroactively reconstructing parentage
through side maps.

A constraint `L_a <= L_b` means lifetime `L_b` outlives lifetime `L_a`.

Edges therefore represent **upper-bound relationships**.

After the relevant graph is constructed:

1. **Strongly connected components (SCCs) are collapsed**
2. **SCC contents are validated**
3. **The SCC DAG becomes the lifetime ordering structure**

Each SCC represents a group of lifetimes forced to be equal.

---

# Constraint Sources

Lifetime relationships come from two distinct sources.

1. **Declared requirements**
   - signature/type-expression nested-reference well-formedness
   - struct field requirements that all instances inherit
   - eventually type aliases / typedef bodies as well

2. **Body-induced requirements**
   - borrows, reborrows, deref/projection chains, storage bounds
   - these must later be checked against what declarations allow

Both should be derived from origin structure when practical, rather than by a
late recursive walk over solved types.

---

# Origin-Threaded Gathering

The current direction is to thread `parent_origin: Option<OriginId>` through
gather / type compilation so nested reference structure is created in-order at
the point syntax is visited.

This replaces the current model where parent/provenance relationships are often
recovered later through `value_origins` / `pattern_origins` hash maps and other
retroactive attachment.

Planned consequences:

- values, patterns, and type expressions should all participate in the same
  parent-threaded origin construction scheme where feasible
- empty / low-value origins are acceptable if they buy consistency
- origin lookup tables, if still needed, should move toward dense/vector-backed
  storage rather than ad-hoc hash maps
- the long-term goal is to abolish the current hash-map-style origin side tables
  as the primary representation

This means some origins will exist for types/values that ultimately do not
contribute ordering edges. That is acceptable; sparse conditional origin
construction is more error-prone for lifetime work.

# Declared Well-Formedness Requirements

The first missing well-formedness rule is:

`&'a &'b T` requires `'a <= 'b`.

Important distinction:

- this requirement is allowed when it is part of declared type structure
  (signatures, struct fields, later typedef/type aliases)
- the same shape arising from body pressure can be illegal if it forces new
  ordering between external lifetimes that the declaration did not permit

---

Declared requirements therefore need an explicit representation separate from
the body-origin graph, even if both are stored using origin-local references.
During function-body checking, the function's own declared `where` lifetime
edges are treated as allowed orderings for global-lifetime validation, not as
imported obligations. Imported struct/callee requirements remain separate so a
call or construction cannot satisfy itself by adding the very ordering it needs.

One required form is:

- `origin_a <= origin_b` (often thought of as `origin_b >= origin_a`)

More requirement kinds may be needed later, especially for typedef/type-alias
validation where the declaration itself may need to be rejected before any
instantiation exists.

## Struct Metadata

If a struct field contains nested references such as `&'a &'b T`, every
instance of that struct inherits the requirement `'a <= 'b`.

This means struct metadata must store declared lifetime-order requirements so
specialization/instantiation can replay them.

Function signatures need analogous declared-order metadata.

## Type Aliases / Typedefs

Type aliases such as `type A = &'a &'b T;` likely need direct declaration-time
validation too. This may require type-level origins or an equivalent structural
requirement representation even when no value-level origin exists.

This part is still open, but the design should avoid painting us into a corner
where only value origins can express requirements.

# Reference-Producing Operations

Operations that produce references in function bodies create ordering
constraints between the produced reference lifetime and the lifetime of the
value it originates from.

Conceptually:

`L_result <= L_source`

meaning the produced reference cannot outlive the source it was derived from.

Typical cases include:

- address-of (`AddrOf`)
- reborrow
- dereference
- member projection
- indexing
- casts preserving reference provenance

These relationships are discovered by following the origin provenance chain
until the next relevant pointer-producing origin is encountered.

---

# Local Variable Storage

Local variables introduce storage lifetimes.

Each variable has a lifetime:

`VarLifetime(var_id)`

Any value stored in that variable must satisfy:

`L_value <= VarLifetime(var_id)`

This introduces ordering constraints between value lifetimes and the
storage lifetime of the variable.

These constraints connect the lifetime graph to later borrow checking, which
reasons about variable storage duration.

This remains separate from declared nested-reference well-formedness, but both
systems want the same consistent origin-threaded representation.

---

# Raw Pointer Conversions

When a reference is derived from a raw pointer, its lifetime cannot be
proven by the compiler.

Example:

`let r = unsafe { &*raw_ptr };`

In this situation the resulting lifetime is classified as:

`LifeTime::Unknown`

This does **not represent inference failure**.

Instead it encodes that the reference originates from raw pointer
provenance and therefore has no statically provable ordering guarantees.

Unknown lifetimes may exist in the graph but should not introduce
ordering constraints between unrelated lifetimes.

`RawRoot` should be reserved for raw-pointer provenance boundaries.
Regular place-based roots (for example simple `let p = &x` style borrows)
should use a non-raw root kind (currently `PlaceRoot`) so provenance walks
do not accidentally treat them as raw-pointer boundaries.

---

# Lifetime Classes

After SCC collapse each lifetime node is assigned a class.

## Local

A lifetime created entirely within the function.

Local lifetimes may merge freely with other local lifetimes.

---

## External(i)

A lifetime originating outside the function.

Examples include:

- function parameters
- captured references
- references passed into the function

Two distinct external lifetimes must never appear in the same SCC unless a
future rule explicitly models declared equality (currently not planned).

Such a merge would require them to be equal, which is invalid.

---

## Static

Represents the `'static` lifetime.

This may be implemented either as:

- a dedicated root node dominating the graph, or
- a special class with known ordering rules.

---

## Unknown

Represents lifetimes derived from raw pointer provenance.

Unknown lifetimes acknowledge the existence of a reference but encode
that the compiler cannot reason about its safety relationships.

They are valid nodes in the graph and do not represent inference failure.

---

# SCC Validation

After SCC collapse we validate the contents of each component.

Invalid configurations include:

### Multiple Externals

`External(A) == External(B)`

This means the function requires two distinct external lifetimes to be equal.

This is always an error.

---

### Illegal Body-Induced External Ordering

Some external lifetime orderings are legal only because they were declared by a
signature / struct requirement.

Example direction:

- signature contains `&'a &'b T`, therefore `'a <= 'b` is declared and allowed
- body later tries to force some new external ordering not implied by the
  declaration, which should be rejected

This distinction is one of the main reasons declared requirements need to be
tracked explicitly, not reconstructed from body edges later.

---

### Invalid Local–External Equality

Some merges between local and external lifetimes may be invalid.

The exact rule depends on escape policy. Generally a local lifetime may
merge with an external only if it originates from that external through
valid reference derivation.

---

# Solver Outline

The declaration and local solve pipeline should proceed roughly as follows:

1. Gather declarations while threading parent origins through nested type
   structure.
2. Record declared well-formedness requirements from nested refs.
3. Store declared requirements in function/struct metadata.
4. Gather function bodies with the same origin-threaded approach.
5. Collect all lifetime nodes (`LId` roots).
6. Gather body ordering constraints from:
   - origin-derived reference operations
   - local variable storage bounds
7. Build the late lifetime constraint graph.
8. Compute strongly connected components.
9. Validate SCC composition.
10. Check body-induced external requirements against the declared allowlist.
11. Construct the SCC DAG representing lifetime ordering.

Nodes that are not constrained by the graph remain valid local lifetimes.

`Unknown` lifetimes remain only for raw-pointer-derived references.

---

# Exported Representation

The result of the local solve is a borrow-checker-facing structure:

`SolvedLifetimeGraph`

This graph contains:

- dense lifetime node IDs
- ordering edges between SCC nodes
- node class information
- optional diagnostic anchors

The representation must **not expose internal inference IDs**.

---

# Local → Global Pipeline

The local lifetime solve produces information used by a later
global lifetime solve.

Local stage responsibilities:

- detect purely local lifetime contradictions
- construct lifetime ordering DAG
- classify lifetime nodes
- export a stable representation

Global stage responsibilities:

- compose lifetime summaries across function boundaries
- detect cycles or equality requirements introduced by composition
- produce cross-function diagnostics

At the local graph layer, diagnostics should prefer reporting the offending
ordering edges themselves when a constraint is illegal (especially for distinct
global/external lifetimes). SCC/cycle machinery is still useful for preventing
bad equalization, but user-facing errors should stay focused on the specific
unauthorized outlives requirements rather than on the existence of a cycle.

---

# Integration Point

The lifetime graph solve still occurs late in local inference.

However, declared well-formedness recording occurs earlier during declaration /
signature compilation, before body solving.

Suggested ordering:

1. normal deferred solving passes
2. pointer-style fallback passes
3. local lifetime graph construction
4. SCC solve and validation
5. finalize lifetime classifications
6. export solved graph

This ensures ordering information is preserved before finalization.

---

# Incremental Milestones

1. Refactor gather/type compilation to thread parent origins directly.
2. Reduce / remove hash-map-style origin side tables as the primary model.
3. Record declared nested-reference well-formedness requirements for global
   signatures.
4. Store declared lifetime-order metadata on structs.
5. Extend declared checks to typedef/type-alias forms.
6. Gather body origin-derived ordering edges from the new representation.
7. Add variable-storage lifetime constraints.
8. Validate body-induced external ordering against declared requirements.
9. Export `SolvedLifetimeGraph`.
10. Integrate with borrow checking.
11. Add global lifetime composition stage.

---

# Open Questions

1. Exact representation for declared requirements beyond simple
   `origin_a <= origin_b`.
2. Whether typedef/type-alias validation needs first-class type-level origins.
3. Exact rules for when body-induced external ordering is considered allowed.
4. Exact rules for when local and external lifetimes may merge.
5. Whether `'static` should be modeled as a root node or a class.
6. How much edge metadata is needed for useful diagnostics.
7. Exact form of summaries exported for the global solve.
