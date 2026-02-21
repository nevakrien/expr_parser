#!/usr/bin/env python3
import argparse
import os
import re
import statistics
import subprocess
import time

import pandas as pd


MICROSECONDS_PER_LINE_RE = re.compile(r"Microseconds per line: ([0-9.]+)")


def parse_cpu_list(cpu_list: str) -> list[int]:
    cpus: list[int] = []
    for chunk in cpu_list.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo, hi = chunk.split("-", 1)
            cpus.extend(range(int(lo), int(hi) + 1))
        else:
            cpus.append(int(chunk))
    return sorted(set(cpus))


def spin_for(seconds: float) -> None:
    if seconds <= 0:
        return
    end = time.perf_counter() + seconds
    x = 0
    while time.perf_counter() < end:
        x = (x + 1) ^ (x << 1)
    if x == -1:
        print("spin")


def run_once(binary: str, input_file: str, cpu: int, env: dict[str, str]) -> float:
    output = subprocess.check_output(
        ["taskset", "-c", str(cpu), binary, input_file],
        text=True,
        env=env,
    )
    match = MICROSECONDS_PER_LINE_RE.search(output)
    if not match:
        raise RuntimeError(f"could not parse output for cpu {cpu}")
    return float(match.group(1))


def classify_cpu(cpu: int) -> str:
    return "P" if cpu <= 15 else "E"


def print_human_table(rows: list[dict[str, float | int | str]]) -> None:
    df = pd.DataFrame(rows)
    display_df = df.copy()

    for col in ["mean_us_per_line", "stddev", "cv_percent", "min", "max"]:
        display_df[col] = display_df[col].map(lambda value: f"{value:.4f}")

    print("\nPer-CPU results")
    print(display_df.to_string(index=False))

    summary = (
        df.groupby("class", as_index=False)
        .agg(
            cpu_count=("cpu", "count"),
            mean_us_per_line=("mean_us_per_line", "mean"),
            stddev_of_cpu_means=("mean_us_per_line", "std"),
            best_cpu_us=("mean_us_per_line", "min"),
            worst_cpu_us=("mean_us_per_line", "max"),
        )
        .sort_values("class")
    )

    for col in [
        "mean_us_per_line",
        "stddev_of_cpu_means",
        "best_cpu_us",
        "worst_cpu_us",
    ]:
        summary[col] = summary[col].map(lambda value: f"{value:.4f}")

    print("\nBy core class")
    print(summary.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep file_typecheck_benchmark across CPUs")
    parser.add_argument("--binary", default="./target/release/examples/file_typecheck_benchmark")
    parser.add_argument("--input", default="typecheck_benchmark_data.txt")
    parser.add_argument("--cpus", default="0-31")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--spin-ms", type=int, default=0)
    parser.add_argument("--keep-internal-perf", action="store_true")
    args = parser.parse_args()

    cpus = parse_cpu_list(args.cpus)
    if not cpus:
        raise SystemExit("no cpus selected")

    env = os.environ.copy()
    if not args.keep_internal_perf:
        env["EXPR_PARSER_DISABLE_INTERNAL_PERF"] = "1"

    spin_seconds = args.spin_ms / 1000.0

    rows: list[dict[str, float | int | str]] = []
    for cpu in cpus:
        values = []
        for _ in range(args.runs):
            spin_for(spin_seconds)
            values.append(run_once(args.binary, args.input, cpu, env))

        mean = statistics.mean(values)
        stddev = statistics.pstdev(values)
        cv_percent = (stddev / mean) * 100.0 if mean else 0.0

        rows.append(
            {
                "cpu": cpu,
                "class": classify_cpu(cpu),
                "mean_us_per_line": mean,
                "stddev": stddev,
                "cv_percent": cv_percent,
                "min": min(values),
                "max": max(values),
            }
        )

    print_human_table(rows)

    print("\nCSV")
    print(pd.DataFrame(rows).to_csv(index=False).strip())


if __name__ == "__main__":
    main()
