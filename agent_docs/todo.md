# Agent TODO Notes

## Active Type-Inference Work

- Specialization must always include lifetime specialization, not just generic type substitution.
- Every specialization call should register global signature lifetimes as fresh local unresolved `LId`s and preserve equality links through substitution.
- Fix/verify struct elided-lifetime rejection (`S=struct{p:&int}` must error with the declared-lifetime message).

## Type Inference Architecture Follow-up

- Future direction (requested by user): consider splitting current `main_solver` usage by context.
- Struct global-range resolution likely needs a mostly gather-first path with only a narrow loop over selected resolve passes (instead of full solver behavior).
- Function body inference may be able to stay gather-only for many cases.
- This is a design follow-up note only; no behavior change is implied by this note.

## Known Temporary Failing Tests

- `type_inference::type_infer_tests::generic_box_array_index_chain_includes_box_step`
- `type_inference::type_infer_tests::member_access_curried_ref_self_and_tracks_full_signature`
- `type_inference::type_infer_tests::struct_deref_to_array_index_expression_typechecks`

These currently fail during the specialization/lifetime refactor because assertions still expect pre-refactor lifetime display/resolution behavior.
