use expr_parser::error_reporting::ErrorReporter;
use expr_parser::parsing::Parser;
use expr_parser::program::Program;
use expr_parser::type_system::run_typechecker;
use std::time::{Duration, Instant};

const ITERATIONS: usize = 20;

static FILE_CONTENT: &str = include_str!("../typecheck_benchmark_data.txt");

fn run_once(file_content: &str, reporter: &mut ErrorReporter) -> (Duration, usize, usize, usize) {
    let start = Instant::now();

    let mut program = Program::new();
    let mut parser = Parser::new(file_content, 0);

    let mut compile_errors = 0usize;
    if let Err(errs) = program.lower_all(&mut parser) {
        compile_errors = errs.len();
        for err in errs {
            let _ = reporter.report_compile_error(&err);
        }
    }

    let mut type_errors = 0usize;
    let mut checked = 0usize;

    if compile_errors == 0 {
        let (r, c) = run_typechecker(&program, reporter).unwrap();
        checked = c;
        if let Err(ec) = r {
            type_errors += ec;
        }
    }

    let duration = start.elapsed();
    (duration, checked, compile_errors, type_errors)
}

fn main() {
    println!(
        "Running typecheck benchmark ({} measured iterations + 1 warmup)",
        ITERATIONS
    );

    let file_content = FILE_CONTENT;
    let line_count = file_content.lines().count();

    // Build reporter once
    let mut reporter = ErrorReporter::new();
    reporter.add_source(0, file_content.to_string());

    // -------- Warmup (NOT measured) --------
    println!("Warmup run...");
    let _ = run_once(file_content, &mut reporter);
    println!("runing...");

    // -------- Measured runs --------
    let mut durations = Vec::with_capacity(ITERATIONS);
    let mut functions_checked = 0usize;
    let mut compile_error_count = 0usize;
    let mut type_error_count = 0usize;

    for _ in 0..ITERATIONS {
        let (duration, checked, compile_errors, type_errors) =
            run_once(file_content, &mut reporter);

        durations.push(duration);

        functions_checked = checked;
        compile_error_count = compile_errors;
        type_error_count = type_errors;
    }

    // -------- Statistics --------
    let times: Vec<f64> = durations.iter().map(|d| d.as_secs_f64()).collect();

    let n = times.len() as f64;
    let mean = times.iter().sum::<f64>() / n;

    let variance = if n > 1.0 {
        times.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / (n - 1.0)
    } else {
        0.0
    };

    let stddev = variance.sqrt();

    let lines_per_second = line_count as f64 / mean;
    let funcs_per_second = functions_checked as f64 / mean;

    let total_errors = compile_error_count + type_error_count;

    println!("\nResults ({} iterations):", ITERATIONS);
    println!("  Lines in file: {}", line_count);
    println!("  Functions checked: {}", functions_checked);
    println!("  Compile errors: {}", compile_error_count);
    println!("  Type errors: {}", type_error_count);
    println!("  Total errors: {}", total_errors);

    println!("\nTiming:");
    println!("  Mean time: {:.6} sec", mean);
    println!("  Sample stddev: {:.6} sec", stddev);
    println!("  Relative stddev: {:.2} %", (stddev / mean) * 100.0);

    println!("\nThroughput (based on mean):");
    println!("  Lines per second: {:.2}", lines_per_second);
    println!("  Functions per second: {:.2}", funcs_per_second);
    println!(
        "  Microseconds per line: {:.2}",
        (mean * 1_000_000.0) / line_count as f64
    );
}
