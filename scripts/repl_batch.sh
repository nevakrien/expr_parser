#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

if [[ -n "${RUSTFLAGS-}" ]]; then
  export RUSTFLAGS="$RUSTFLAGS -Awarnings"
else
  export RUSTFLAGS="-Awarnings"
fi

cd "$REPO_ROOT"
exec cargo run --quiet -- --stdin-batch "$@"
