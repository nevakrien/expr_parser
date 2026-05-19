# Lifetime Refactor Direction

The previous detailed lifetime plan described the old hybrid implementation and
should not be used as the active architecture. The old version is available in
the local detached snapshot of GitHub `main`:

```text
/home/user/Desktop/rust_stuff/expr_parser/expr_parse_pre
```

The new direction is to make lifetimes a separate graph/order phase after type
shape solving, not a set of special cases inside the shape solver.

## Target Boundary

Shape solving should produce pointer/reference shapes and record lifetime-facing
obligations. It should not decide whether an outlives relationship is legal.

The lifetime phase should consume:

- solved shape graph,
- pointer info ids,
- origin/provenance records,
- explicit outlives obligations,
- declaration where-clause obligations,
- implicit deref/reborrow/member/index obligations.

It should produce:

- collapsed lifetime equality components,
- a directed lifetime-order DAG,
- diagnostics for impossible orderings,
- metadata usable by a future borrow/storage checker.

## Core Rule

Use union-find only for equality. A relationship like `'a < 'b` or
`'a <= 'b` is an edge in a graph until SCC solving proves equality. It is not a
normal type-unification constraint.

## Phases

1. Record lifetime variables on pointer info.
2. Record outlives and provenance obligations while resolving expressions.
3. After shape solving, build a lifetime graph from obligations and origins.
4. Run SCC collapse to find equality components.
5. Validate known-lifetime merges and directed ordering edges.
6. Export a solved graph for later borrow checking.

This phase may eventually require a real middle IR. Avoid adding more AST-side
retroactive reconstruction unless it is temporary scaffolding for the refactor.

## Open Design Points

- Exact origin representation after the clean-room type-system rewrite.
- How declaration where-clauses are replayed through generic specialization.
- Which implicit casts/reborrows remain type obligations vs borrow-check facts.
- What metadata the future middle IR needs from type solving.

Keep this document high level until the new type-system core exists.
