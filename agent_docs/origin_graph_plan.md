# Origin Graph Rewrite Plan (Local Type Inference)

This is the authoritative plan for replacing late/reconstructed place origin logic with explicit provenance tracked during gather.

The goal is to make mutability + lifetime provenance deterministic, debuggable, and reusable by future borrow checking.

## Current implementation status (Feb 2026)

- Core provenance storage is now in place (`OriginId` arena + `ValId`/`PatId` origin maps), and writable checks no longer use `value_origin` reconstruction.
- `InnerFunctionTypes` now persists origin artifacts for downstream stages.
- Gather currently records provenance as side data while still returning `CId`; the planned `(CId, Option<OriginId>)` signature refactor is still pending.

## Why This Rewrite Is Required

- Current writable checks can reconstruct origin from expression shape (`value_origin`) after the fact.
- Reconstruction loses information about reborrows/casts/call returns and can misclassify writable state.
- The known bug class is immutable-local escaping through unresolved-pointer promotion:
  - immutable `x` -> `p = &x` -> `*p = ...` accepted in some paths.
- Existing implicit-deref vectors are not provenance; they are resolution breadcrumbs.

Hard requirement: provenance must be created once during gather and then only referenced by id.

## Non-Negotiable Constraints

- `value_origin` must be removed, not "mostly unused".
- `gather_constraints` must return a tuple `(CId, Option<OriginId>)`.
- Origins must be arena ids with links (`OriginId`), not vectors/chains assembled ad hoc.
- Every derived pointer/reference value gets a new origin node, even when tied to same lifetime.
- Function return values like `f(&'a T)->&'a U` are never the same origin as the argument; they are new nodes that reference prior provenance.

## Origin Creation Rules (Only 4 Root Creators)

All origins are projections from exactly four root-creation mechanisms:

1. Binding root
   - Introduced by local `let`/`var` declaration sites.
   - Store declaration `PatId` and declared mutability.
2. Argument root
   - Introduced by function parameter bindings.
   - Store parameter `PatId` and parameter mutability semantics.
3. Call-return root
   - Introduced by function call result values.
   - Even if return lifetime is tied to an input (`&'a -> &'a`), return origin is a new node.
4. Raw/provenance-breaking root
   - Introduced when creating pointer-like provenance from non-provenance data (e.g. int->ptr cast).
   - Treated as fresh provenance for borrow/lifetime tracking.

Everything else is a projection/derived node from one or more prior origins.

## Origin Node Shape

Use id-backed arena in local inference state, persisted to solved output.

Minimum node payload:

- `kind`: root kind or projection kind.
- `parent`: optional parent origin id (single-parent chain, no per-node heap allocation).
- `decl_site`: `PatId` and/or `ValId` for diagnostics.
- `declared_mutability`: `Option<bool>`.
- `lifetime_seed`: placeholder/metadata for later borrow checking.

Projection kinds to model explicitly:

- reborrow (`&*x`, explicit/implicit borrow-induced projection),
- deref,
- member/index projection,
- cast projection (provenance-preserving cast),
- raw-deref lifetime projection (new lifetime identity, mutability still constrained by ancestry).

## Cast Semantics (Critical)

This area is easy to get wrong and must be explicit in code.

- Provenance-preserving casts keep ancestry via a projection node.
  - Example intent: `let x = &mut y; let x2 = x as &mut _;` should preserve mutability provenance expectations.
- Provenance-breaking casts create a fresh raw root.
  - Example intent: int->pointer or equivalent "from nowhere" pointer construction.
- Never silently collapse cast result to parent identity; always new node + link (or fresh root if broken).

## Mutability Decision Rules

Writable checks must consult origin ancestry, not just pointer cluster mutability state.

- If ancestry reaches a known immutable root, mutable-place requirements fail.
- If ancestry reaches mutable-legal root and type mutability is compatible, writable can pass.
- If type mutability unresolved, keep pending checks, but provenance constraints remain fixed.
- Pointer-cluster promotion (`None -> Some(true)`) is allowed only when provenance does not forbid mutable access.

This prevents unresolved-type fallback from upgrading immutable provenance.

## Lifetime Decision Rules (Forward-Compatible)

- Raw-pointer deref introduces a fresh lifetime-facing projection node.
- Reborrows/projections record lifetime relationship edges for later borrow analysis.
- Lifetime-equal signatures (`&'a -> &'a`) still produce distinct origins linked by projection ancestry.

## Required Code Changes

## 1) Data Model (`src/type_inference.rs`)

- Add `OriginId` and origin node enums/structs.
- Extend `SearchState` with:
  - origin arena,
  - `ValId -> OriginId` mapping,
  - optional `PatId -> OriginId` mapping.
- Extend `InnerFunctionTypes` to persist provenance artifacts:
  - origin arena snapshot,
  - value/pattern origin maps.

## 2) Gather Rewrite (`src/local_type_inference.rs`)

- Change `gather_constraints` signature to return `(CId, Option<OriginId>)`.
- Thread tuple through all callers.
- In every `Value::*` branch, explicitly define origin behavior.
- Remove `value_origin` function and all uses.

## 3) Writable Checks (`src/local_type_inference.rs`)

- Rewrite `require_place_writable` and related helpers to consume gathered origin ids.
- Add ancestry walk helpers for mutability/lifetime provenance queries.
- Keep pending-writable queue behavior, but base decisions on provenance + types, never reconstruction.

## 4) Finalization (`src/local_type_inference.rs`)

- In `finalize_local`, copy origin artifacts into `InnerFunctionTypes`.
- Keep existing type finalization behavior unchanged.

## 5) API Exposure (`src/type_inference.rs`)

- Add accessor(s) on solved function internals so later passes/tests can inspect origins.
- Avoid exposing mutable internals directly; read-only query API is preferred.

## Migration Order (Do In This Order)

1. Add data types + storage fields (compiles, unused initially).
2. Change gather return type + thread tuple through compile path.
3. Create origins in gather for all value forms.
4. Switch writable checks to new provenance query path.
5. Delete old reconstruction path (`value_origin`).
6. Persist origins in `InnerFunctionTypes` during finalize.
7. Add tests, then update docs.

Do not skip step 5. Leaving both systems active causes divergence bugs.

## Test Matrix (Must Add/Keep)

- immutable local through shared ref assignment is rejected.
- reborrow chain preserves root mutability (`p1=&y; p2=&*p1`).
- cast-preserving reborrow semantics behave as expected.
- provenance-breaking cast to pointer creates fresh raw root.
- call return produces a distinct origin node even with tied lifetime (`&'a -> &'a`).
- existing mutable/deref/member/index tests remain green.

## Known Failure Modes If Incomplete

- Keeping `value_origin` anywhere in writable path reintroduces bug class.
- Treating `&'a -> &'a` returns as same origin collapses alias/projection distinctions.
- Modeling casts without explicit preserve-vs-break split causes unsound upgrades.
- Storing origins as vectors/chains again prevents stable ids for diagnostics and borrow checker.

## Files Expected To Change

- `src/local_type_inference.rs`
- `src/type_inference.rs`
- `agent_docs/type_inference.md`
- tests in `src/type_inference.rs`

Optional follow-up doc: add a dedicated provenance query guide once implementation lands.
