# Origin / Provenance Refactor Direction

The previous origin graph plan targeted the old local type-inference data model.
It has been replaced by this high-level direction for the clean-room type-system
rewrite. The old plan remains available in the local detached snapshot of
GitHub `main`:

```text
/home/user/Desktop/rust_stuff/expr_parser/expr_parse_pre
```

## Goal

Origins should explain where pointer/reference values come from so later phases
can check mutability, lifetime ordering, and borrow/storage legality. Origins
should be id-backed and stable enough for diagnostics and downstream IR passes.

## Boundary With Type Solving

Shape solving may create pointer shapes and attach `PtrInfoId`s. It should record
origin/provenance obligations when an operation creates or transforms a
reference-like value, but it should not walk old AST side tables to prove borrow
legality.

## Desired Properties

- Origin records are arena ids, not reconstructed ad hoc chains.
- Pointer/reference-producing operations record provenance at the operation site.
- Reborrows, derefs, member projections, index projections, and casts should be
  distinguishable.
- Provenance-preserving casts and provenance-breaking raw roots should remain
  distinct.
- Function returns should get their own origin even when their lifetime is tied
  to an input.

## Deferred Until New Core Exists

- Exact origin node enum.
- Exact storage location in solved function metadata.
- How much origin data is attached during type solving vs during middle-IR
  lowering.
- Final borrow-check interface.

Avoid reintroducing detailed dependencies on legacy `CId`-returning gather code.
