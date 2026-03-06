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
- Finalization has a fallback that resolves unresolved lifetime roots to
  `Unknown` so type solving can complete.
- Struct field signatures now reject elided reference lifetimes in inline struct
  definitions when lifetime params are required.

## Partially Implemented / Transitional

- Lifetime-aware local inference exists, but unresolved regions still collapse to
  `Unknown` in fallback paths.
- Lifetime contradiction handling is split:
  - some direct contradictions are rejected during typecheck,
  - reborrow/order-sensitive checks are deferred.
- Implicit deref and pointer-style operations thread lifetime/mutability state,
  but borrow-checker-grade ordering validation is not complete.
- `src/lifetime_graph.rs` now has an origin-parent-based ordering extractor that
  walks `OriginNode.parent` chains, treats binding aliases / member / index
  projections as transparent ancestry when needed, and emits graph-local
  ordering edges (`LifetimeGraphId <= LifetimeGraphId`) through a
  caller-provided pointer-to-lifetime resolver that maps inference lifetimes
  into the graph namespace.
- Lifetime-order edge extraction now ignores edges where either endpoint is
  marked raw-provenance, so raw-pointer flows still participate in mutability
  tracking without introducing lifetime outlives constraints.
- Origin roots now distinguish place-based roots (`PlaceRoot`) from true raw
  provenance boundaries (`RawRoot`), so raw-boundary handling does not apply to
  ordinary `let`/borrow provenance.
- `src/lifetime_graph.rs` now includes a reusable SCC solve pass over lifetime
  ordering edges. It now uses a Tarjan-style DFS that validates cycle-forming
  edges as they are discovered and immediately severs disallowed edges instead
  of rebuilding SCCs from scratch. Component lifetime unification uses
  `LifeTime` directly and treats `LifeTime::Unknown(_)` as an upgradeable
  placeholder.
- `InnerFunctionTypes` now records `lifetime_unknown_count`, i.e. how many
  fallback `LifeTime::Unknown` lifetimes were minted during local solving for
  that function.

## Not Implemented Yet

- Full local lifetime graph construction from origin/provenance edges integrated
  into local inference.
- SCC collapse + validation pass fully integrated into local inference
  finalization (currently available as `lifetime_graph` solve utility but not
  yet wired into the main local solver pipeline).
- Stable exported `SolvedLifetimeGraph` artifact for downstream borrow checking.
- Global lifetime composition across function boundaries.
- Complete borrow-check pass enforcing deferred lifetime ordering legality.

## Practical Constraints For Contributors

- Treat `Unknown` as a temporary representation boundary, not final semantics.
- Preserve lifetime specialization at every generic specialization call site.
- Do not erase origin/provenance information needed by planned graph extraction.
- Keep diagnostics stable when adding lifetime checks (`found` vs `wanted`
  orientation should stay consistent).

## Migration Direction

Current direction is:

1. Keep existing lifetime identity plumbing intact.
2. Replace fallback-heavy unresolved handling with explicit local graph edges.
3. Solve local lifetimes via SCC collapse and validate SCC contents.
4. Export a CId-free solved lifetime graph for borrow-check consumption.
5. Layer global lifetime composition on top of local solved summaries.
