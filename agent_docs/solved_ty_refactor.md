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

- `HEAD:src/global_type_inference.rs:52` -> `cluster_solved_type(...)` check.
- `HEAD:src/global_type_inference.rs:207` -> `cluster_solved_type(...)` check.
- `HEAD:src/global_type_inference.rs:228` -> `cluster_solved_type(...)` check.
- `HEAD:src/global_type_inference.rs:244` -> `cluster_solved_type(...)` check.
- `HEAD:src/global_type_inference.rs:256` -> `cluster_solved_type(...)` check.
- `HEAD:src/global_type_inference.rs:732` -> solved pre-check + concrete type inspection.
- `HEAD:src/global_type_inference.rs:1542` -> solved pre-check moved before `match cluster_state`.

- `HEAD:src/local_type_inference.rs:281` -> `cluster_solved_type(...)` check.
- `HEAD:src/local_type_inference.rs:296` -> `cluster_solved_type(...)` check.
- `HEAD:src/local_type_inference.rs:308` -> `cluster_solved_type(...)` check.
- `HEAD:src/local_type_inference.rs:331` -> `cluster_solved_type(...).is_none()`.
- `HEAD:src/local_type_inference.rs:384` -> read `cluster[root].solved_ty`.
- `HEAD:src/local_type_inference.rs:1608` -> solved pre-check branch (outside state-match).
- `HEAD:src/local_type_inference.rs:2041` -> solved pre-check branch (outside state-match).
- `HEAD:src/local_type_inference.rs:2185` -> solved pre-check branch.
- `HEAD:src/local_type_inference.rs:2504` -> solved pre-check branch.
- `HEAD:src/local_type_inference.rs:3116` -> solved pre-check branch (outside state-match).
- `HEAD:src/local_type_inference.rs:3225` -> solved pre-check branch (outside state-match).
- `HEAD:src/local_type_inference.rs:3522` -> solved pre-check before classification match.
- `HEAD:src/local_type_inference.rs:3551` -> solved pre-check before classification match.
- `HEAD:src/local_type_inference.rs:3631` -> solved pre-check before ptr-parts match.

- `HEAD:src/type_inference.rs:2767` -> `cluster_solved_type(root)` fast path.
- `HEAD:src/type_inference.rs:2792` + `:2794` -> replaced by solved fast path before `match state`.
- `HEAD:src/type_inference.rs:3269` -> read `solved_ty` for func params.
- `HEAD:src/type_inference.rs:3279` -> read `solved_ty` for func return.
- `HEAD:src/type_inference.rs:3306` -> read `solved_ty` for struct generics.
- `HEAD:src/type_inference.rs:3344` -> read `solved_ty` for tuple items.
- `HEAD:src/type_inference.rs:3361` -> `cluster_solved_type` for array element.
- `HEAD:src/type_inference.rs:3389` -> `cluster_solved_type` for ptr target.
- `HEAD:src/type_inference.rs:3615` -> solved pre-check in type display.
- `HEAD:src/type_inference.rs:4283` -> solved pre-check in `cluster_is_int_like`.
- `HEAD:src/type_inference.rs:4304` -> solved pre-check in `cluster_is_float_like`.
- `HEAD:src/type_inference.rs:4325` -> solved pre-check in `cluster_is_bool`.

### B) Solved writes (`state = ResolveKind::Solved(...)`)

- `HEAD:src/type_inference.rs:2102` builtin defaults -> `set_cluster_solved(store, id, builtin_ty)`.
- `HEAD:src/type_inference.rs:2153` `new_solved` -> `set_cluster_solved(store, id, t)`.
- `HEAD:src/type_inference.rs:2596`/`:2645`/`:2708`/`:2741` post-try-resolve promotion -> `set_cluster_solved(...)`.
- `HEAD:src/type_inference.rs:2788`/`:2803`/`:2814`/`:2820`/`:2826`/`:2832`/`:2838`/`:2844` force-type writes -> `set_cluster_solved(...)`.
- `HEAD:src/type_inference.rs:3955` specialization solved cluster creation -> `new_solved(ex.store, ty)`.
- `HEAD:src/type_inference.rs:4064` specialization builtin solved creation -> `new_solved(ex.store, ty)`.
- `HEAD:src/type_inference.rs:4410` deferred resolver write -> `set_cluster_solved(...)` and progress only if newly solved.
- `HEAD:src/local_type_inference.rs:1720` solved cluster creation -> `new_solved(ex.store, t)`.

### C) `__try_absorb` branch merge (old explicit solved pattern arms)

- `HEAD:src/type_inference.rs:2518` `(Solved, Solved)` -> solved/solved pre-check compare.
- `HEAD:src/type_inference.rs:2532` `(Solved, IntLike)` -> solved-dst + unresolved-src compatibility check.
- `HEAD:src/type_inference.rs:2542` `(Solved, FloatLike)` -> same.
- `HEAD:src/type_inference.rs:2602` `(Solved, Func(call))` -> solved-dst + unify-with-type path.
- `HEAD:src/type_inference.rs:2651` `(Solved, Struct(call))` -> same.
- `HEAD:src/type_inference.rs:2659` `(Solved, Ptr {..})` -> same.
- `HEAD:src/type_inference.rs:2714` `(Solved, Tuple(...))` -> same.
- `HEAD:src/type_inference.rs:2747` `(Solved, Array {..})` -> same.

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
