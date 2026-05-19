# Lifetime Implementation Status

The old lifetime implementation status has been intentionally gutted because it
described the legacy hybrid solver. For old details, use the local detached
snapshot of GitHub `main`:

```text
/home/user/Desktop/rust_stuff/expr_parser/expr_parse_pre
```

## Current Active Status

- The active branch is expected to move toward a clean-room type-system rewrite.
- Old `CId`/`LId` coupling, finalization hacks, origin side tables, and
  declaration-edge plumbing are reference material, not the desired endpoint.
- New implementation work should first establish shape solving, pending
  requirements, and obligation recording.
- Real lifetime validation should come after the new shape/pointer obligation
  boundary is stable.

## Preserve From The Old System

- Tests are valuable behavioral references.
- Type dumps/debug output are valuable and should be recreated early.
- Existing language syntax around lifetime parameters and where clauses remains
  useful, even if the internal representation changes.

## Do Not Preserve As Architecture

- Mixed kind/type solving.
- Lifetime ordering inside normal type unification.
- Pointer mutability/rawness decisions in the shape equality solver.
- Long solver functions that combine gather, resolution, defaulting, and
  diagnostics.
