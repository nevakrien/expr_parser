#!/usr/bin/env bash
set -euo pipefail

cargo run --release --example file_typecheck_benchmark
python3 scripts/typecheck_core_sweep.py

