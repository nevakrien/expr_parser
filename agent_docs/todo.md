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

## Lifetimes
- Declared where-clause lifetime orderings are still stored as explicit edges
  only. Before adding transitive closure at interning/struct construction time,
  rework declaration-side `missing where-clause requirement` diagnostics so they
  do not report every implied edge separately.
- Local/body lifetime solving still needs a path to consume declared ordering
  DAGs directly instead of relying on only origin-derived edges.
