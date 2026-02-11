# Repository Guidelines

## Project Structure & Module Organization
- `src/main.rs` hosts the REPL entry point and pretty-printing helpers.
- `src/parsing.rs` contains tokens, AST types, lexer, and parser logic.
- `src/error_reporting.rs` centralizes error formatting with `ariadne`.
- `src/error_messages.rs` provides shared error message constants.
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
- `agent_docs/type_infrence.md`: detailed guide to `src/type_inference.rs`, including unification/clash behavior, specialization, generic-scope risks, and gather-layer fragility/extension points.

## Commands
- `cargo run` runs a repl that can be used for testing current behivior and errors.
- `cargo test` runs the full test suite and should be used routinely.
- `cargo fmt` formats Rust code; run before committing style changes.
- `cargo clippy` runs lint checks; fix warnings before submitting.

## Benchmarks
- Benchmark are build using `cargo build/run --release --example bench_name`
- Benchmark binaries are built into `target/release/examples/`.
- Use `perf stat` or `perf record` on the example binaries (for example `target/release/examples/no_macros_benchmark`).

## Tests
- `cargo test` should be used by agents after every change that can be tested.
- New tests should conform to the current style, usually covering more than one thing in a single test.
- some tests should ideally check for error cases and for the information in the error to be correct (including spans)
- tests should generally prefer unwrap to except because unwrap has more usefull debug info
