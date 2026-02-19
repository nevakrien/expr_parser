# Lifetime Semantics and Implementation Plan

This document is a focused planning note for introducing lifetime-aware typing and borrow checking.
It complements:

- `agent_docs/language_semantics.md` for language-level contract.
- `agent_docs/type_inference.md` for solver/dataflow details.

## Target semantics (agreed)

- No implicit user-facing lifetime downcast. Explicit reborrow syntax (`&*x`) is required.
- `'raw` is a distinct lifetime state with non-null pointer semantics.
- `&'a T` is never inferred as `&'raw T`.
- Smart-pointer APIs can choose between:
  - tied safe deref: fn['a](&'a self)->&'a out
  - raw receiver deref: fn['a](&'raw self)->&'a out
  - raw address exposure: fn['a](&'raw self)->&'raw out
- `&mut 'raw` is not treated as noalias.

## Typecheck-stage behavior split

Immediate in typecheck:

- reject trivially contradictory lifetime equalities in one typing step.
- example: f(x:&'a t)->&'b t { x } fails immediately when no reborrow relation can connect 'a to 'b.

Deferred to borrow checker:

- constraints introduced by explicit/implicit reborrows (for example `'b < 'a`).
- legality of implicit cast edges introduced by desugaring.
- raw-specific aliasing and escape checks.

## Implicit cast/reborrow sites to record

These sites can create fresh references not directly written by the user:

- member access (`.` and `->`) through deref chains,
- index access,
- method receiver adaptation and currying,
- other implicit deref steps already tracked today.

Each site should record at least:

- source reference/lifetime,
- target reference/lifetime,
- expression/value id and span,
- reason tag (member/index/deref/call-receiver),
- whether this came from explicit syntax (`&*`) or compiler synthesis.

## Unnamed lifetime policy

Global signatures:

- explicit function lifetime generics are now accepted in typechecking and stay distinct from implicit/elided lifetime inference.
- unnamed input lifetimes => fresh independent binders.
- unnamed output lifetime => join of all input lifetimes (future).
- temporary behavior: if exactly one implicit input lifetime exists, assign that to the elided output; otherwise emit a type error and leave output lifetime as `Unknown` so inference can continue.

Function bodies:

- unnamed/inferred lifetime occurrences always mint a fresh lifetime id.
- keep all minted ids in stable order for dense `Vec`-indexed borrow-check data.

## Suggested implementation phases

1. **Representation phase**
   - Add lifetime ids/vars in type model for references.
   - Encode `raw` as explicit distinct lifetime state.

2. **Constraint-gather phase**
   - Parse/compile lifetime binders/mentions.
   - Mint body lifetimes.
   - Emit reborrow/cast constraints and record implicit cast edges.

3. **Inference phase**
   - Preserve `raw` distinction in unification.
   - Implement immediate contradiction checks.
   - Keep deferred lifetime inequalities for borrow analysis.

4. **Solved artifact phase**
   - Expose lifetime ids, bounds, and cast edges in solved output.
   - Ensure stable ids for downstream borrow checker indexing.

5. **Borrow-check phase**
   - Validate inequalities/regions.
   - Validate implicit cast legality.
   - Enforce raw vs non-raw alias/escape rules.

## Risks and gotchas

- Implicit receiver transformations currently happen in several places; missing one can make borrow checking unsound.
- Generic specialization must duplicate lifetime params per use site just like type generics, or constraints will leak between calls.
- Error orientation (`found` vs `wanted`) should stay consistent for lifetime clashes too.
