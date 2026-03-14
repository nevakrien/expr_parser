# Lifetime Implementation Status

This file tracks what is already implemented in the lifetime/type-inference
pipeline vs what is still planned.

Use this with:

- `agent_docs/lifetimes_plan.md` for the current main plan (local graph + SCC).
- `agent_docs/type_inference.md` for detailed solver behavior.
- `agent_docs/language_semantics.md` for user-facing language contract.

## Implemented Today

- Function lifetime generics are accepted and kept distinct from elided/inferred
  lifetimes during signature compilation.
- Input-side elided lifetimes are minted as fresh independent lifetime slots.
- Output-side elided lifetimes use the current temporary rule:
  - if there is exactly one implicit input lifetime, output-elided slots are
    rewritten to that lifetime,
  - otherwise the compiler emits a type error and leaves output slots unresolved
    so inference can continue.
- Local inference carries lifetime identities (`LId`) and lifetime union-find
  state to connect lifetime constraints through solving.
- Function/struct specialization includes lifetime specialization plumbing so
  solved/global lifetimes are remapped to fresh local unresolved lifetimes per
  specialization site.
- Local solve now runs a lifetime-graph pass that seeds origin nodes with
  associated `LId`s, extracts origin-order constraints, runs SCC collapse, and
  unifies `LId`s inside cycle components.
- Lifetime graph seeding now also canonicalizes origin `lifetime_seed`s to
  union-find roots, unifies seed `LId`s with associated pointer `LId`s when
  available, and mints missing `BindingRoot` seeds as fresh
  `LifeTime::Local` lifetimes.
- `let`-introduced binding roots now seed required local lifetimes directly on
  their origin-attached `LId`s during gather, so truly local storage origins are
  explicitly represented before graph solving.
- After graph solve, unresolved lifetime roots are finalized by constraint
  context: roots that are constrained (directly or transitively) by `u <= l`
  where `l` is local are promoted to fresh local lifetimes; other unresolved
  roots are assigned fresh unknown lifetimes (`LifeTime::Unknown`).
- Struct field signatures now reject elided reference lifetimes in inline struct
  definitions when lifetime params are required.

## Partially Implemented / Transitional

- Lifetime-aware local inference exists, with unresolved/non-required lifetime
  roots now converging to `Unknown` while `let`-binding roots keep explicit
  required-local seeds.
- Lifetime contradiction handling is split:
  - some direct contradictions are rejected during typecheck,
  - reborrow/order-sensitive checks are deferred.
- Implicit deref and pointer-style operations thread lifetime/mutability state,
  but borrow-checker-grade ordering validation is not complete.
- `src/lifetime_graph.rs` now has an origin-parent-based ordering extractor that
  walks `OriginNode.parent` chains, treats binding aliases / member / index
  projections as transparent ancestry when needed, and emits graph-local
  ordering edges (`LifetimeGraphId <= LifetimeGraphId`) from per-origin
  lifetime seeds.
- Lifetime-order edge extraction treats `RawRoot` as a provenance boundary in
  parent-chain walks, so orderings are emitted only for non-raw ancestry
  relationships instead of relying on cached per-origin raw flags.
- Origin roots now distinguish place-based roots (`PlaceRoot`) from true raw
  provenance boundaries (`RawRoot`), so raw-boundary handling does not apply to
  ordinary `let`/borrow provenance.
- `src/lifetime_graph.rs` now includes a reusable SCC solve pass over lifetime
  ordering edges. It uses Tarjan DFS to produce SCC components and the local
  solver uses SCC membership to drive `LId`-level unification checks.
- SCC-driven unification now runs per component (not per in-component edge), so
  every `LId` inside an equality component is attempted against a single leader
  and diagnostics can anchor to representative origins when a known-lifetime
  merge is incompatible.
- Lifetime-graph diagnostics now use a dedicated lifetime error path instead of
  generic simple errors. Ordering failures report the concrete lifetime names
  involved (for example `this reborrow requires lifetime 'a to outlive 'b`),
  and invalid global-vs-global outlives requirements are now reported at the
  individual offending edge sites instead of being surfaced primarily as SCC
  cycle diagnostics.
- Local lifetime graph solving now also validates directed ordering edges when
  both sides already have known lifetimes; impossible known orderings (for
  example requiring an external lifetime to be shorter than a local lifetime)
  now emit a direct local diagnostic instead of silently passing.
- Origin seeding now prefers the associated-pointer lifetime when an existing
  origin seed conflicts with that pointer lifetime, avoiding stale/conflicting
  seeds on pointer-associated origins.
- Current origin attachment still relies heavily on `value_origins` /
  `pattern_origins` side tables and retroactive parent relationships. The new
  direction is to replace this with parent-origin threading during gather/type
  compilation so nested reference structure is built in syntax order.
- Pattern gathering now accepts a threaded `parent_origin: Option<OriginId>` and
  uses it for immutable binding roots / nested annotated patterns, while mutable
  bindings still intentionally avoid inheriting parent provenance so local
  mutability is not over-constrained.

## Not Implemented Yet

- Declaration-time nested-reference well-formedness checking for global
  signatures (`&'a &'b T` requiring `'a <= 'b`).
- Enforcement/propagation of declared lifetime-order metadata through
  specialization and local solving.
- Typedef/type-alias declaration validation for nested-reference
  well-formedness.
- Validation that body-induced external ordering requirements are a subset of
  declared/allowed requirements.
- Full local lifetime graph construction from origin/provenance edges integrated
  into local inference with per-origin `LId` seeding.
- Stable exported `SolvedLifetimeGraph` artifact for downstream borrow checking.
- Global lifetime composition across function boundaries.
- Complete borrow-check pass enforcing deferred lifetime ordering legality.

## Next Implementation Slice: Declared Where-Clause Ordering DAGs

The next lifetime task is to make explicit declaration where clauses such as
`['a, 'b where 'a < 'b]` participate in typechecking instead of being parser/IR
only metadata.

Target behavior for this slice:

- Every struct declaration and solved function signature should own a validated
  declaration-local lifetime ordering DAG derived from its where clause.
- The global solve phase should build and validate that DAG once, then store it
  on declaration metadata so local/body inference does not recompute it.
- When a global item is specialized, its declaration DAG must be remapped onto
  the specialized lifetime ids at the use site. No local graph edge should ever
  directly reference the original declaration lifetime ids.
- This remapping must be used consistently for all specialization paths,
  including struct construction, member-method currying, operator/member
  overload resolution, and smart-deref helper paths.

Planned representation direction:

- Store declaration edges in terms of declaration lifetime parameter indexes
  (for example `shorter: usize, longer: usize`) rather than global/local `LId`s.
- After global validation, store the transitive-reduced or otherwise validated
  DAG on `StructRep` and on solved function-signature metadata.
- At specialization time, translate those declaration indexes through the
  specialization lifetime argument list into concrete specialized `LId` pairs,
  then insert those edges into the local lifetime graph as where-clause edges.

Solver constraints for this slice:

- Local lifetime solving needs two edge phases:
  - phase 1: declaration/where-clause edges, inserted without the current
    global-order legality rejection,
  - phase 2: inference/body edges, where illegal newly-required global ordering
    still reports errors.
- The SCC/Tarjan solve remains the same core algorithm. The implementation must
  preserve O(1) amortized insertion per edge and avoid per-edge global scans or
  graph recomputation.

Important invariants to preserve while implementing:

- declaration lifetime ids stay declaration-local and never leak into local
  solve graphs,
- specialization always remaps declaration DAGs through specialized lifetime
  ids,
- illegal ordering errors should come from inference/body pressure that exceeds
  the already-established declaration relationships, not from loading the
  declaration relationships themselves.

## Current Status Update

The parser/IR where-clause metadata is now stored on solved global declaration
types in plain sorted/deduped forms:

- `TypeValue::Func` stores `lifetime_orderings`, canonicalized during interning
  so equivalent function signatures share the same `TypeId` regardless of edge
  insertion order or duplicates.
- `TypeValue::Func` also stores `generic_lifetime_requirements` for
  `T<'a`-style where-clause requirements.
- `StructRep` stores both `lifetime_orderings` and
  `generic_lifetime_requirements`, canonicalized at construction time.
- Type printers now render both lifetime-order edges and `T<'a`-style
  requirements using the language syntax shape, including the `, where ...`
  separator inside the generic/lifetime list.

This slice only stores and prints the declaration metadata. It does not yet
enforce those edges during specialization or lifetime solving.

## Practical Constraints For Contributors

- Treat `Unknown` as a temporary representation boundary, not final semantics.
- Preserve lifetime specialization at every generic specialization call site.
- Do not erase origin/provenance information needed by planned graph extraction.
- Keep diagnostics stable when adding lifetime checks (`found` vs `wanted`
  orientation should stay consistent).

## Migration Direction

Current direction is:

1. Keep existing lifetime identity plumbing intact.
2. Move origin construction toward threaded parent origins instead of hash-map
   side tables.
3. Record declared well-formedness requirements during declaration/signature
   compilation.
4. Replace fallback-heavy unresolved handling with explicit local graph edges.
5. Solve local lifetimes via SCC collapse and validate SCC contents.
6. Export a CId-free solved lifetime graph for borrow-check consumption.
7. Layer global lifetime composition on top of local solved summaries.
