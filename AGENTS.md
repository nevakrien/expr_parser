# Repository Guidelines

## Project Structure & Module Organization
- `src/main.rs` hosts the REPL entry point and pretty-printing helpers.
- `src/parsing.rs` contains tokens, AST types, lexer, and parser logic.
- `src/error_reporting.rs` centralizes error formatting with `ariadne`.
- `src/macros.rs` implements unhygienic macro system with parameter substitution.
- `src/program.rs` provides Program struct and high-level parsing with macro expansion.
- `src/ir.rs` contains  first IR design (not full implemented yet).
- `src/lib.rs` is the library entry point.
- `target/` is build output and should not be edited.

## Commands
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

## Tasks
1. adding basic type infrence (types are represented as values)
2. adding a multi error for reporting things that go wrong independently.