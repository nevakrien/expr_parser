# Mutability Testing Direction

The previous mutability fuzzing sketch was tied to the old solver. Keep the
testing idea, but do not treat the old implementation details as current
architecture.

## Useful Idea To Preserve

Template-generated programs are still a good way to test mutability behavior
without needing a full semantic oracle.

Useful axes after the refactor:

- root binding kind: immutable vs mutable storage,
- pointer style: shared, mutable, raw,
- transport path: direct borrow, reborrow, function identity, struct/tuple wrap,
- write site: deref write, field write, index write,
- expected result: accepted, rejected by obligation check, or rejected by borrow
  checking.

## Refactor Boundary

Mutability should be checked as pointer-style/provenance obligations after shape
solving, not by forcing shape unification to carry mutable-place semantics.

Recreate fuzzing once the new obligation layer exists.
