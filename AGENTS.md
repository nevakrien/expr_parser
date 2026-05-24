# Repository Guidelines

## Agent Responsibility
- If a test fails you may not change it just so it works without getting explicit permission from a human
  adding new tests is fine but existing tests should be kept as is
- Do Not add inline(always) unless you have a specific reason 
- Do Not add clones to vectors or similar large structures unless there is no better way, if u do comment explaining why there isnt s better way.
- Agents should keep `agent_docs/` up to date when behavior, architecture, or important APIs change.
- If an agent touches a complex subsystem, it should also update the relevant doc in `agent_docs/` as part of the same task when practical.
- Treat stale docs in `agent_docs/` as a maintenance issue and refresh them proactively.

## Agent Docs Usage
- Before changing a subsystem, agents should check `agent_docs/` for the relevant module summary and read it first.
- `agent_docs/` exists to capture architecture, invariants, known fragile areas, and feature-extension notes that are easy to miss in code-only scans.
- If docs and code disagree and the code was not written by the current agent in this task, treat code as source-of-truth and update docs to match.
- Agents should only change code to match docs when the agent itself introduced the mismatch in the current task.
- When adding a new complex subsystem, add a new `agent_docs/<subsystem>.md` file so future agents have a focused starting point.

## Type-System Solver Direction
- The new type-system code should operate on `KindId`/kinds directly. Do not reintroduce the old duplicated split where types and kinds had parallel unification paths.
- Baseline `unify` means full equality merge: kind shape, pointer info, mutability variables, and lifetime variables should be merged through their own dedicated merge mechanisms rather than manually special-cased at every use site.
- If a relation is directional, keep it separate from `unify`. Name the mutating constraint operation `require_subtype`, not a predicate-style name like `subtype_of`, because the operation will likely emit ordered mutability and lifetime constraints instead of just returning a boolean.

### Current docs index
- `agent_docs/type_inference.md`: active type-system data model, solver rules, and refactor direction.
- `agent_docs/cli_debugging.md`: practical guide to using the REPL/CLI to inspect parsed AST shape, inferred types, and typechecker diagnostics while investigating behavior.
- `agent_docs/language_semantics.md`: language-level behavior across parser/lowering/type inference, including `fn` vs `cfn`, `struct` vs `cstruct`, and `. / :: / ->` member-access semantics.
- `agent_docs/ir_lowering.md`: IR data model and lowering behavior in `src/ir.rs`, including span arenas, expression lowering rules, and control-flow edge cases.
- `agent_docs/program_model.md`: `src/program.rs` architecture notes covering definition gathering, name resolution, scopes, pending-name flow, and label patching.
- `agent_docs/macros.md`: unhygienic macro subsystem behavior in `src/macros.rs`, including expansion order and known limitations.
- `agent_docs/string_intern_and_ids.md`: intern table internals and `IdHashMap`/`IdentityHasher` conventions for ID-keyed maps.
- `agent_docs/lifetimes_plan.md`: high-level lifetime phase direction for the clean-room type-system rewrite.
- `agent_docs/origin_graph_plan.md`: high-level origin/provenance direction for the clean-room rewrite.
- `agent_docs/mutability_fuzzing.md`: high-level mutability testing direction after the new obligation layer exists.

## Commands
- First step for quick compile/typecheck checks: run `scripts/repl_batch.sh` with stdin + EOF.

```bash
scripts/repl_batch.sh <<'EOF'
main = fn() {};
EOF
```

- Add `--type-dump` when you need solved types:

```bash
scripts/repl_batch.sh --type-dump <<'EOF'
main = fn() {};
EOF
```

- `cargo run` runs the interactive CLI/REPL when you want manual exploration.
- See `agent_docs/cli_debugging.md` for practical debugging workflows and REPL command details.
- `cargo test` runs the full test suite and should be used routinely.
- `cargo fmt` formats Rust code; run before committing style changes.
- `cargo clippy` runs lint checks; fix warnings before submitting.

## Benchmarks
- Benchmark are build using `cargo build/run --release --example bench_name`
- Benchmark binaries are built into `target/release/examples/`.
- Use `perf stat` or `perf record` on the example binaries (for example `target/release/examples/no_macros_benchmark`).

## Tests
- `cargo test` should be used by agents after every change that can be tested.
- It's a good idea to also run `cargo test --features solver_order_fuzz` when touching type-inference solver logic; randomized pass order helps catch hidden order-dependence bugs early.
- New tests should conform to the current style, usually covering more than one thing in a single test.
- some tests should ideally check for error cases and for the information in the error to be correct (including spans)
- tests should generally prefer unwrap to except because unwrap has more usefull debug info
