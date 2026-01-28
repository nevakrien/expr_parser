#!/bin/bash

echo "Running benchmark on old algorithm (10 times)..."
echo "Iteration,Time_s,Statements_per_sec,Microseconds_per_statement,Cache_miss_rate,Branch_miss_rate,Instructions,B,Cycles"

cd /home/user/Desktop/rust_stuff/expr_parser
git checkout benchmark-comparison >/dev/null 2>&1
cargo build --release --example file_lower_benchmark >/dev/null 2>&1

for i in {1..10}; do
    echo -n "$i,"
    
    # Run benchmark and extract timing info
    result=$(./target/release/examples/file_lower_benchmark lower_benchmark_data.txt 2>/dev/null | grep -E "(Time:|Statements per second:|Microseconds per statement:)")
    time=$(echo "$result" | grep "Time:" | sed 's/.*Time: \([0-9.]*\)s.*/\1/')
    stmts_per_sec=$(echo "$result" | grep "Statements per second:" | sed 's/.*Statements per second: \([0-9.]*\).*/\1/')
    microsecs=$(echo "$result" | grep "Microseconds per statement:" | sed 's/.*Microseconds per statement: \([0-9.]*\).*/\1/')
    
    # Run perf and extract key metrics
    perf_result=$(perf stat -e cache-misses,cache-references,branch-misses,branches,instructions,cycles ./target/release/examples/file_lower_benchmark lower_benchmark_data.txt 2>&1 | grep -E "(cache-misses|cache-references|branch-misses|branches|instructions|cycles)" | tail -6)
    
    cache_misses=$(echo "$perf_result" | grep "cache-misses" | awk '{print $1}' | head -1)
    cache_refs=$(echo "$perf_result" | grep "cache-references" | awk '{print $1}' | head -1)
    branch_misses=$(echo "$perf_result" | grep "branch-misses" | awk '{print $1}' | head -1)
    branches=$(echo "$perf_result" | grep "branches" | awk '{print $1}' | head -1)
    instructions=$(echo "$perf_result" | grep "instructions" | awk '{print $1}' | head -1)
    cycles=$(echo "$perf_result" | grep "cycles" | awk '{print $1}' | head -1)
    
    # Calculate percentages using Python
    stats=$(python3 <<EOF
import sys
cache_misses = int("$cache_misses".replace(",", ""))
cache_refs = int("$cache_refs".replace(",", ""))
branch_misses = int("$branch_misses".replace(",", ""))
branches = int("$branches".replace(",", ""))
cache_miss_rate = (cache_misses / cache_refs) * 100 if cache_refs > 0 else 0
branch_miss_rate = (branch_misses / branches) * 100 if branches > 0 else 0
print(f"{cache_miss_rate:.2f}%,{branch_miss_rate:.2f}%,{instructions},{cycles}")
EOF
)
    
    echo "$time,$stmts_per_sec,$microsecs,$stats"
done