# Repository Guidelines

## Project Structure & Module Organization
- `src/main.rs` hosts the REPL entry point and pretty-printing helpers.
- `src/parsing.rs` contains tokens, AST types, lexer, and parser logic.
- `src/error_reporting.rs` centralizes error formatting with `ariadne`.
- `target/` is build output and should not be edited.

## Commands
- `cargo test` runs the full test suite and should be used routinely.
- `cargo fmt` formats Rust code; run before committing style changes.
- `cargo clippy` runs lint checks; fix warnings before submitting.

## Tests
- `cargo test` should be used routinely to verify new changes.
- New tests should conform to the current style, usually covering more than one thing in a single test.
- some tests should ideally check for error cases and for the information in the error to be correct (including spans)

## Agent-Specific Instructions
- Keep contributor docs concise and focused on this repo’s parser/lexer workflow.
- Never change the AST structs unless explicitly instructed by the user.
- Keep the language style in mind when adding features.
- ";" "," and "(" are almost completely optional to the grammar try and make sure it stays this way.
- Run fmt and test before commiting and consider also runing clippy