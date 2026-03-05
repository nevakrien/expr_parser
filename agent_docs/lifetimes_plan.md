# Lifetime Plan (Local Graph First)

This is the primary lifetime-planning document.
It describes the local-first lifetime model based on a constraint graph solved
with **SCC (strongly connected component) collapse**.

The lifetime system operates in two stages:

1. **Local lifetime solve** (per function)
2. **Global lifetime solve** (across functions)

The local stage constructs lifetime ordering constraints within a single
function and detects purely local contradictions. The global stage later
composes summaries across functions.

This document focuses on the **local lifetime graph**, which is the current
implementation priority and the basis for later global composition.

---

# Problem Summary

Current local inference still falls back to assigning `LifeTime::Unknown`
when lifetime roots remain unresolved near the end of solving.

While this is useful for debugging, it destroys ordering information and
prevents detection of invalid lifetime relationships.

The missing behaviors today are:

- reference-producing operations should generate ordering constraints
- local variable storage should bound lifetimes
- equality relationships should be detected through SCC collapse
- invalid merges between lifetimes should be detected explicitly
- local results should be exportable to a global lifetime solve

Instead of erasing information by assigning unknown lifetimes early,
we build a **lifetime constraint graph** and analyze it structurally.

---

# Core Model

The lifetime system is represented as a directed graph.

Nodes represent **lifetime variables (`LId`)**.

Edges represent **ordering constraints**.

A constraint `L_a <= L_b` means lifetime `L_b` outlives lifetime `L_a`.

Edges therefore represent **upper-bound relationships**.

After the graph is constructed:

1. **Strongly connected components (SCCs) are collapsed**
2. **SCC contents are validated**
3. **The SCC DAG becomes the lifetime ordering structure**

Each SCC represents a group of lifetimes forced to be equal.

---

# Constraint Sources

Lifetime edges are produced from semantic lifetime relationships
observed during local inference.

Edge gathering should be **origin-driven**, reusing provenance data
already recorded in `OriginNode`.

The lifetime graph therefore reflects the **actual reference provenance
relationships recorded during inference**, rather than reconstructing
relationships from types alone.

---

# Origin-Driven Constraint Extraction

The existing `OriginNode` graph already records the provenance chain
for values derived from references.

Each `OriginNode` has a kind describing the projection or transformation
that produced it.

Many origin kinds are associated with a pointer-producing `CId`.

The existing helper used by the codebase is:

`OriginKind::associated_pointer() -> Option<CId>`

Important detail: the `CId` returned by this function is always the
**resulting pointer produced by the origin node**, not the source.

Example origin kinds that return a `CId` include:

- `ArgumentRoot`
- `CallReturnRoot`
- `RawRoot`
- `Reborrow`
- `Deref`
- `CastProjection`

Member and index projections instead inherit pointer provenance from
their parent origin.

Lifetime relationships should therefore be derived primarily by
**walking origin parent chains**, rather than attempting to interpret
individual `CId` values.

---

# Reference-Producing Operations

Operations that produce references create ordering constraints between
the produced reference lifetime and the lifetime of the value it
originates from.

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

These relationships are discovered by following the **origin provenance
chain** until the next pointer-producing origin is encountered.

---

# Local Variable Storage

Local variables introduce storage lifetimes.

Each variable has a lifetime:

`VarLifetime(var_id)`

Any value stored in that variable must satisfy:

`L_value <= VarLifetime(var_id)`

This introduces ordering constraints between value lifetimes and the
storage lifetime of the variable.

These constraints connect the lifetime graph to later borrow checking,
which reasons about variable storage duration.

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

Two distinct external lifetimes must never appear in the same SCC.

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

### Invalid Local–External Equality

Some merges between local and external lifetimes may be invalid.

The exact rule depends on escape policy. Generally a local lifetime may
merge with an external only if it originates from that external through
valid reference derivation.

---

# Local Solver Outline

The local lifetime solve proceeds as follows:

1. Collect all lifetime nodes (`LId` roots).
2. Gather ordering constraints from:
   - origin-derived reference operations
   - local variable storage bounds
3. Build the lifetime constraint graph.
4. Compute strongly connected components.
5. Validate SCC composition.
6. Construct the SCC DAG representing lifetime ordering.

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

---

# Integration Point

The lifetime graph solve occurs late in local inference.

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

1. Introduce lifetime edge recording in local inference.
2. Gather edges from origin-derived reference operations.
3. Add variable-storage lifetime constraints.
4. Implement SCC-based lifetime solve.
5. Validate external lifetime conflicts.
6. Export `SolvedLifetimeGraph`.
7. Integrate with borrow checking.
8. Add global lifetime composition stage.

---

# Open Questions

1. Exact rules for when local and external lifetimes may merge.
2. Whether `'static` should be modeled as a root node or a class.
3. How much edge metadata is needed for useful diagnostics.
4. Exact form of summaries exported for the global solve.
