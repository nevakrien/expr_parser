# ResolveKind solved-semantics refactor log

This note tracks the migration from `ResolveKind::Solved(TypeId)` to:

- `Cluster { state: ResolveKind, solved_ty: Option<TypeId> }`
- solved writes through `set_cluster_solved(store, cid, ty)`
- solved state construction through `make_resolve_kind(types, store, ty)`

It is intentionally exhaustive for old solved semantics, based on old `HEAD` grep results.

## Invariant

- A cluster is solved iff `cluster.solved_ty.is_some()`.
- When solved, `cluster.state` must be the shape returned by `make_resolve_kind(...)` for that same `TypeId`.
- `set_cluster_state` must not mutate solved clusters.
- setting solved on an already solved cluster is not progress (`false`), and must not overwrite.

## Why merged logic is correct (`__try_absorb`)

Old logic had separate branches for:

- `(Solved, Solved)`
- `(Solved, IntLike)` / `(Solved, FloatLike)`
- `(Solved, Func/Struct/Ptr/Tuple/Array)`

New logic merges this by pre-checking solved-ness first:

1. `dst_solved && src_solved`: compare exact concrete types (same behavior as old `(Solved, Solved)`).
2. `dst_solved && !src_solved`: validate/absorb unresolved shape into known concrete type (same behavior as old `(Solved, X)` branches).
3. `!dst_solved && src_solved`: `force_type(dst, src_ty)` (this is the symmetric case that old code got via second absorb direction; now explicit and clearer).
4. neither solved: run unresolved-state merge logic.

This is correct for both old solved and unresolved cases because solved handling is now orthogonal to shape (`state`) and no branch relies on "kind implies unsolved".

## Full old solved-semantics inventory and migration

The entries below are all old solved-semantic references from `HEAD` in
`src/type_inference.rs`, `src/local_type_inference.rs`, `src/global_type_inference.rs`.

### A) Solved reads (`if let ResolveKind::Solved(...)` etc.)

- `src/global_type_inference.rs`:
  - all old solved checks were migrated to `cluster_solved_type(...)` pre-checks,
  - solved-sensitive branches now do solved checks before unresolved-state `match` paths.
- `src/local_type_inference.rs`:
  - old solved checks were replaced by `cluster_solved_type(...)` / `cluster[root].solved_ty` reads,
  - sites that classify pointers/operands now do solved pre-checks before state-based classification.
- `src/type_inference.rs`:
  - fast solved-path checks now use `cluster_solved_type(root)`,
  - readers for func params/return, struct generics, tuple items, array element, and ptr target now pull from `solved_ty`,
  - solved pre-checks were added to type-display and class helpers (`cluster_is_int_like`, `cluster_is_float_like`, `cluster_is_bool`).

### B) Solved writes (`state = ResolveKind::Solved(...)`)

- All solved writes now go through `set_cluster_solved(...)` or `new_solved(...)`.
- `src/type_inference.rs` updates:
  - builtin defaults,
  - `new_solved` implementation,
  - post-resolve promotions,
  - force-type solved writes,
  - specialization solved cluster construction,
  - deferred resolver solved writes (progress only when newly solved).
- `src/local_type_inference.rs` updates:
  - local solved cluster creation paths use `new_solved(ex.store, ...)`.

### C) `__try_absorb` branch merge (old explicit solved pattern arms)

- In `src/type_inference.rs`, explicit solved-pattern arms were consolidated into solved-first pre-checks that preserve prior behavior for:
  - solved-vs-solved compare,
  - solved-vs-`IntLike`/`FloatLike` compatibility,
  - solved-vs-`Func`/`Struct`/`Ptr`/`Tuple`/`Array` via unify-with-type paths.

And explicitly added the symmetric solved-source path:

- `src/type_inference.rs` current `if let Some(t) = src_solved { force_type(dst, t)?; ... }`

### D) Constructor/API changes needed to preserve invariant

- `TypeState::new` now takes `&TypeStore` so defaults are solved via `set_cluster_solved`.
- `TypeState::clear_local_state` now takes `&TypeStore` for same reason.
- `InferState::new` and `InferState::clear_local_state` updated accordingly.
- `TypeState::new_solved` now requires `&TypeStore` and always uses `set_cluster_solved`.
- Added `make_resolve_kind(types, store, ty)` and routed solved construction through it.

### E) Safety guards against old "kind means unsolved" assumption

- `set_cluster_state` now refuses to mutate solved clusters (debug-assert equal state).
- `copy_cluster_state` now preserves solved invariant and avoids solved-dst overwrite.
- `set_cluster_solved` returns `false` if already solved (no fake progress).

## Not part of this semantic migration

- `PtrKind::Solved(...)` references are pointer-style metadata and are unrelated to removed `ResolveKind::Solved(TypeId)`.

## Post-migration correction

- `extract_clash_type_string` must not treat `ResolveKind::Nothing` as unresolved by itself.
- After this migration, solved builtin/generic clusters intentionally have `state = ResolveKind::Nothing` and `solved_ty = Some(...)`.
- The unresolved fast-path now checks both fields: only `solved_ty.is_none()` **and** `state == Nothing` is unresolved.
