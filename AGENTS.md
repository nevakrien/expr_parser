# Repository Guidelines

## Project Structure & Module Organization
- `src/main.rs` hosts the REPL entry point and pretty-printing helpers.
- `src/parsing.rs` contains tokens, AST types, lexer, and parser logic.
- `src/error_reporting.rs` centralizes error formatting with `ariadne`.
- `src/macros.rs` implements unhygienic macro system with parameter substitution.
- `src/program.rs` provides Program struct and high-level parsing with macro expansion.
- `src/ir.rs` contains  first IR design (not full implemented yet).
- `src/struct_layout.rs` calculates target-aware struct layouts from type info.
- `src/string_intern.rs` implements the string interner.
- `src/type_inference.rs` contains the type inference sketch and tests.
- `src/lib.rs` is the library entry point.
- `agent_docs/` contains agent-maintained technical documentation and deep module summaries.
- `target/` is build output and should not be edited.

## Agent Documentation Responsibility
- If a test fails you may not change it just so it works without getting explicit permission from a human
  adding new tests is fine but existing tests should be kept as is
- Do Not add inline(always) unless you have a specific reason 
- Do Not add clones to vectors or similar large structures unless there is no better way, if u do comment explaining why there isnt better ways.
- Agents should keep `agent_docs/` up to date when behavior, architecture, or important APIs change.
- If an agent touches a complex subsystem, it should also update the relevant doc in `agent_docs/` as part of the same task when practical.
- Treat stale docs in `agent_docs/` as a maintenance issue and refresh them proactively.

## Agent Docs Usage (Read This First)
- Before changing a subsystem, agents should check `agent_docs/` for the relevant module summary and read it first.
- `agent_docs/` exists to capture architecture, invariants, known fragile areas, and feature-extension notes that are easy to miss in code-only scans.
- If docs and code disagree and the code was not written by the current agent in this task, treat code as source-of-truth and update docs to match.
- Agents should only change code to match docs when the agent itself introduced the mismatch in the current task.
- When adding a new complex subsystem, add a new `agent_docs/<subsystem>.md` file so future agents have a focused starting point.

### Current docs index
- `agent_docs/type_inference.md`: detailed guide to `src/type_inference.rs`, including unification/clash behavior, specialization, generic-scope risks, and gather-layer fragility/extension points.
- `agent_docs/cli_debugging.md`: practical guide to using the REPL/CLI to inspect parsed AST shape, inferred types, and typechecker diagnostics while investigating behavior.
- `agent_docs/language_semantics.md`: language-level behavior across parser/lowering/type inference, including `fn` vs `cfn`, `struct` vs `cstruct`, and `. / :: / ->` member-access semantics.
- `agent_docs/ir_lowering.md`: IR data model and lowering behavior in `src/ir.rs`, including span arenas, expression lowering rules, and control-flow edge cases.
- `agent_docs/program_model.md`: `src/program.rs` architecture notes covering definition gathering, name resolution, scopes, pending-name flow, and label patching.
- `agent_docs/macros.md`: unhygienic macro subsystem behavior in `src/macros.rs`, including expansion order and known limitations.
- `agent_docs/struct_layout.md`: layout computation model in `src/struct_layout.rs`, including recursion detection and generic specialization handling.
- `agent_docs/string_intern_and_ids.md`: intern table internals and `IdHashMap`/`IdentityHasher` conventions for ID-keyed maps.
- `agent_docs/low_ir_sketch.md`: current status and intent for the unfinished low-level IR sketch in `src/low_ir.rs`.
- `agent_docs/lifetimes_plan.md`: primary lifetime plan focused on local lifetime ordering via origin-derived constraints, SCC solving, raw-vs-ref handling, and a CId-free solved graph boundary for borrow checking.
- `agent_docs/lifetime_impl_status.md`: current implementation status for lifetime/type-inference work (implemented pieces, transitional behavior, and remaining stages).
- `agent_docs/todo.md`: short-lived implementation notes (including active refactor items and known temporary failures).

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

### Known Temporary Failures
- Current expected failing tests (as of March 2026):
  - `type_inference::type_infer_tests::dot_member_and_tuple_writes_on_const_and_raw_const_refs_emit_one_error_per_function`
  - `type_inference::type_infer_tests::place_is_checked_delayed`
  - `type_inference::type_infer_tests::pending_ptr_member_method_call_rejects_immutable_to_mut_hop_after_type_is_known`
  - `type_inference::type_infer_tests::nested_box_mut_addr_of_member_uses_mut_deref_chain`
  - `type_inference::type_infer_tests::ptr_member_assignment_rejects_chain_with_immutable_deref_hop`
  - `type_inference::type_infer_tests::ptr_member_method_call_rejects_chain_with_immutable_deref_hop`
- Keep this list in sync with `agent_docs/todo.md`.
