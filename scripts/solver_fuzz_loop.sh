#!/usr/bin/env bash

set -u -o pipefail

# Optional: pass max iterations as first argument for bounded runs.
# Default is infinite loop.
max_iterations="${1:-}"

if [[ -n "$max_iterations" && ! "$max_iterations" =~ ^[0-9]+$ ]]; then
    echo "Usage: $0 [max_iterations]"
    exit 2
fi

iterations=0

echo "Starting solver-order fuzz test loop"
echo "Command: cargo test --features solver_order_fuzz -- -q"
if [[ -n "$max_iterations" ]]; then
    echo "Max iterations: $max_iterations"
else
    echo "Max iterations: infinite"
fi

while true; do
    if [[ -n "$max_iterations" && "$iterations" -ge "$max_iterations" ]]; then
        echo "Reached max iterations ($max_iterations) with no failures."
        exit 0
    fi

    iterations=$((iterations + 1))
    seed="$(od -An -N8 -tu8 /dev/urandom | tr -d ' ')"
    log_file="/tmp/expr_solver_fuzz_${iterations}_$$.log"

    echo "Iteration $iterations | seed=$seed"

    if EXPR_SOLVER_ORDER_SEED="$seed" cargo test --features solver_order_fuzz -- -q >"$log_file" 2>&1; then
        echo "  pass"
        rm -f "$log_file"
        continue
    fi

    echo
    echo "Crash/failure detected."
    echo "Iterations completed before failure: $((iterations - 1))"
    echo "Failing iteration: $iterations"
    echo "Seed: $seed"

    mapfile -t failed_tests < <(
        awk '
            /^test .* \.\.\. FAILED$/ {
                line=$0
                sub(/^test /, "", line)
                sub(/ \.\.\. FAILED$/, "", line)
                failed[line]=1
            }
            /^failures:$/ {
                in_failures=1
                next
            }
            in_failures && /^    / {
                line=$0
                sub(/^    /, "", line)
                if (line != "") {
                    failed[line]=1
                }
            }
            in_failures && /^$/ {
                in_failures=0
            }
            END {
                for (name in failed) {
                    print name
                }
            }
        ' "$log_file" | sort
    )

    if [[ ${#failed_tests[@]} -gt 0 ]]; then
        echo "Crashing/failing tests:"
        for test_name in "${failed_tests[@]}"; do
            echo "  - $test_name"
        done
    else
        echo "Crashing/failing tests: (none parsed from cargo output)"
        echo "See full log: $log_file"
        exit 1
    fi

    echo "Full log saved at: $log_file"
    exit 1
done
