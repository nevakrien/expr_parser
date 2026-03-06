# Agent TODO Notes

## Nice To Have
currently TypeValue is expensive to clone making it not usable with .iter().map

causing a lot of explicit vec construction and pushing with
```
if let SomePattern{vars,..} = ex.store.type_value(ty) else {
	unreachble!()
}
```
in id based loops to avoid the issue.
if we moved all the expensive fields to be Rc<[]> that would allow us to fixup this code to be more readble

## Current Failing Tests

As of March 2026, the following 4 tests fail:

1. **`place_is_checked_delayed`** (`src/type_inference.rs:5347`)
   - Expected: Error "cannot assign through immutable dereference"
   - Actual: Test passes when it should fail (inference succeeds)
   - Reason: The delayed place checking for immutable references is not detecting the error

2. **`pending_ptr_member_method_call_rejects_immutable_to_mut_hop_after_type_is_known`** (`src/type_inference.rs:8261`)
   - Expected: Error when ptr member method call goes through shared-to-mutable deref chain after type is known
   - Actual: Inference succeeds when it should fail
   - Reason: Mutable receiver method calls through `->` allow shared-to-mutable deref hop incorrectly

3. **`nested_box_mut_addr_of_member_uses_mut_deref_chain`** (`src/type_inference.rs:8334`)
   - Expected: Mutable reference step in autoderef chain (`&'a0 mut Box[Box[S]]`)
   - Actual: Gets `&'idk0 Box[Box[S]]` (shared reference) instead
   - Reason: The mut addr_of member access is using shared deref chain for nested Box

4. **`ptr_member_method_call_rejects_chain_with_immutable_deref_hop`** (`src/type_inference.rs:8237`)
   - Expected: Error "implicit `__deref_mut` step requires mutable source"
   - Actual: Inference succeeds when it should fail
   - Reason: Ptr member method call through shared deref chain incorrectly allowed

## Previous Work (may be stale)

- Specialization must always include lifetime specialization, not just generic type substitution.
- Every specialization call should register global signature lifetimes as fresh local unresolved `LId`s and preserve equality links through substitution.
- Fix/verify struct elided-lifetime rejection (`S=struct{p:&int}` must error with the declared-lifetime message).
- Mutable receiver method calls through `->` currently allow a shared-to-mutable deref hop in some paths; see failing tests `ptr_member_method_call_rejects_chain_with_immutable_deref_hop` and `pending_ptr_member_method_call_rejects_immutable_to_mut_hop_after_type_is_known` in `src/type_inference.rs`.
- Call-return provenance currently drops mutability from `var` roots when a shared reference crosses function boundaries; new regression tests `assignment_through_generic_identity_shared_reference_from_var_binding_is_allowed`, `assignment_through_non_generic_ref_identity_from_var_binding_is_allowed`, and `assignment_through_nested_generic_identity_from_var_binding_is_allowed` in `src/type_inference.rs` are expected to fail until `CallReturnRoot` ancestry preserves writable provenance.

## Fuzzing Follow-up

- `examples/mutability_fuzz_sketch.rs` is a template-oracle scaffold for mutability regression hunting.
- Design/usage notes are in `agent_docs/mutability_fuzzing.md`.

## Type Inference Architecture Follow-up

- `main_solver` splitting is done: global uses `global_solver`, locals use `local_solver`.
- Struct global-range resolution likely needs a mostly gather-first path with only a narrow loop over selected resolve passes (instead of full solver behavior).
- Function body inference may be able to stay gather-only for many cases.
- This is a design follow-up note only; no behavior change is implied by this note.

## Origin Provenance Rewrite

- Detailed implementation plan is tracked in `agent_docs/origin_graph_plan.md`.
- Provenance regression tests for shared-reference assignment now pass under the gathered-origin writable checks.
