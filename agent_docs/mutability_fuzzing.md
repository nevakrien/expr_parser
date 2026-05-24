# Mutability Testing Direction

Keep the template-generated testing idea, but do not copy legacy solver details
into the current architecture.

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

The active scaffold now keeps pointer mutability in `KindLookUp.mutable`
(`MutInfo`) rather than `KindStorage`. Unknown mutability can be printed as
`?mut` during early diagnostics with `MutGuessMode::UnknownAsUnknown`, then
treated as const for final/defaulted display with `MutGuessMode::UnknownAsConst`.
Pending mutability implications are a deterministic `BTreeSet<MutId>` of target
nodes, not edge-owned reason records. Reasons are stored on nodes, with implied
paths pointing at the parent node and recording depth; a node keeps only one
reason, preferring lower depth, to avoid duplicate reporting. Conflicts should
preserve both sides: the reason something had to be mutable and the reason it had
to be const.

Recreate fuzzing once the new obligation layer exists.
