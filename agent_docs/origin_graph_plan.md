# Origin / Provenance Direction

The old origin graph plan targeted the legacy local type-inference data model.
Treat that detail as reference material only.

## Goal

Origins should explain where pointer/reference values come from so later phases
can check mutability, lifetime ordering, and borrow/storage legality. Origins
should be id-backed and stable enough for diagnostics and downstream IR passes.

## Boundary With Type Solving

Shape solving may create pointer shapes and attach `PtrInfoId`s. It should record
origin/provenance obligations when an operation creates or transforms a
reference-like value, but it should not prove borrow legality.

## Desired Properties

- Origin records are arena ids, not reconstructed ad hoc chains.
- Pointer/reference-producing operations record provenance at the operation site.
- Reborrows, derefs, member projections, index projections, and casts should be
  distinguishable.
- Provenance-preserving casts and provenance-breaking raw roots should remain
  distinct.
- Function returns should get their own origin even when their lifetime is tied
  to an input.

## Deferred Details

- Exact origin node enum.
- Exact storage location in solved function metadata.
- How much origin data is attached during type solving vs during middle-IR
  lowering.
- Final borrow-check interface.

## Current Scaffold Note

`OriginId`/`Origin` are defined in `src/type_system/origin.rs`. Current origin
nodes can represent function arguments, locals, globals, transients, and derived
values with a `Projection`; each origin also carries effective mutability.

`InnerFunctionTypes` has `value_origins` and `pattern_origins` maps for solved
function metadata. Detailed validation and final borrow/lifetime checks are
still deferred.
