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

## Active Type-Inference Work

- Specialization must always include lifetime specialization, not just generic type substitution.
- Every specialization call should register global signature lifetimes as fresh local unresolved `LId`s and preserve equality links through substitution.
- Fix/verify struct elided-lifetime rejection (`S=struct{p:&int}` must error with the declared-lifetime message).
- Mutable receiver method calls through `->` currently allow a shared-to-mutable deref hop in some paths; see failing tests `ptr_member_method_call_rejects_chain_with_immutable_deref_hop` and `pending_ptr_member_method_call_rejects_immutable_to_mut_hop_after_type_is_known` in `src/type_inference.rs`.

## Type Inference Architecture Follow-up

- `main_solver` splitting is done: global uses `global_solver`, locals use `local_solver`.
- Struct global-range resolution likely needs a mostly gather-first path with only a narrow loop over selected resolve passes (instead of full solver behavior).
- Function body inference may be able to stay gather-only for many cases.
- This is a design follow-up note only; no behavior change is implied by this note.

## Origin Provenance Rewrite

- Detailed implementation plan is tracked in `agent_docs/origin_graph_plan.md`.
- Provenance regression tests for shared-reference assignment now pass under the gathered-origin writable checks.
