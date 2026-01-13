# Repository Guidelines

## Project Structure & Module Organization
- `src/main.rs` hosts the REPL entry point and pretty-printing helpers.
- `src/parsing.rs` contains tokens, AST types, lexer, and parser logic.
- `src/error_reporting.rs` centralizes error formatting with `ariadne`.
- `src/macros.rs` implements unhygienic macro system with parameter substitution.
- `src/program.rs` provides Program struct and high-level parsing with macro expansion.
- `src/ir.rs` contains commented-out first IR design (not yet implemented).
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
- `cargo test` should be used routinely to verify new changes.
- New tests should conform to the current style, usually covering more than one thing in a single test.
- some tests should ideally check for error cases and for the information in the error to be correct (including spans)
- tests should generally prefer unwrap to except because unwrap has more usefull debug info

## Agent-Specific Instructions
- Keep contributor docs concise and focused on this repo’s parser/lexer workflow.
- Never change the AST structs unless explicitly instructed by the user.
- Keep the language style in mind when adding features.
- ";" "," and "(" are almost completely optional to the grammar try and make sure it stays this way.
- Run fmt and test before commiting and consider also runing clippy
- Dont change unrelated code


# Plan
The general plan is to grow this project into a language. It is not realistic for an agent to finish everything in one go; this section is mainly a reference for upcoming work and to answer questions.

The pipeline should look like this:

lexer/parser (done) -> macros (implemented) -> name resolution (planned) -> first IR (sketched)
-> type inference (?) -> run/LLVM (?)

We do not have a plan for everything yet, but the first few stages are sketched out.

## macros
Macros are explicitly unhygienic and run before variable binding. They are order-dependent and should perform basic expression replacement, similar to C but with a bit more structure.

We may add macros in other languages as extensions in the future, which is one of the main reasons the AST is so simple.

Macros should resolve recursively starting from the outermost call, repeatedly expanding until the result is not a macro call. Then we descend into the AST to resolve inner macros.

## first ir
The first IR is desugared so `a && b` and
"if (a as bool) { b as bool } else false"
are equivalent.

The first IR should have variables already resolved to IDs. This is trickier than it seems because of patterns: expressions like
"let Some(x) = 5" need to know that `x` is a new binding, while `Some` is a reference to an existing name.

There is an open design decision here: either treat constructor names as known because they are already defined (making match validity depend on prior definitions), or infer constructor/pattern names by convention (for example, leading-capital identifiers). The former introduces a context-dependent element to the grammar that can make for confusing error messages when moving struct/enum definitions. The latter is more ergonomic but introduces language and Unicode considerations for non-capitalized scripts.

##
