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
- `target/` is build output and should not be edited.

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

## TODO
- Member methods: parse `StructName.method = fn ...` in `src/ir.rs` and represent as syntax-level member method definitions (no type checks yet); document parse rules and expected AST/IR shape.
- Pointer/reference types + mutability: add syntax for `*T`, `&T`, `&mut T` and ensure mutability is represented; document how these are parsed and surfaced in IR. note that we already have the start of mutability on variables, and its purposfully not tested yet as that goes into borrow_checking and not general type chekc. however &const x vs &mut x vs &x is a diffrent story. we currently just have address_of and not more info than that.
- Member access + member calls: support `value.field` and `value.method(...)` in typecheck. we need to resolve MemberAccess nodes properly. this is especially tricky with methods as those would have to be added later. it should be possible to save methods into the SolvedTypes and use that. or even have them in TypeStore somewhere maybe in StructRep or a similar struct related field there.
