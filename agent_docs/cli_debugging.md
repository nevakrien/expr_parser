# CLI/REPL Debugging Guide (`src/main.rs`)

## First command to run (agents)

Use the batch wrapper first. It is the default low-noise path:

```bash
scripts/repl_batch.sh <<'EOF'
main = fn() {};
EOF
```

Use this when you want solved types:

```bash
scripts/repl_batch.sh --type-dump <<'EOF'
main = fn() {};
EOF
```

## Why this path

- Reads full stdin until EOF, then compiles once.
- Suppresses warning spam (`RUSTFLAGS=-Awarnings`).
- Keeps output focused on diagnostics and optional type dump.

Use this when you want solved origins (currently mainly structure + mutability;
adding lifetime info to the dump is a planned debugging improvement and should
be kept in mind while refactoring origins):

```bash
scripts/repl_batch.sh --origin-dump <<'EOF'
main = fn() {};
EOF
```

## Interactive REPL (only when needed)

If you need an interactive session, run `cargo run`.

Runtime toggles:

- `:show-ast <on|off>`
- `:type-dump <on|off>`
- `:origin-dump <on|off>`
- `:modes`
- `:types` / `:types-of <name>` / `:type <name...>`
- `:origins` / `:origins-of <name>`
- `:load <path...>` / `:reset` / `:quit`
