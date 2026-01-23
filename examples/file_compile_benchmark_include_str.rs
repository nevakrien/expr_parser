use expr_parser::error_reporting::ErrorReporter;
use expr_parser::parsing::Parser;
use expr_parser::program::Program;
use std::time::Instant;

fn main() {
    println!("Running file-based compile benchmark (include_str)");

    // NOTE: This keeps UTF-8 validation out of the runtime benchmark, but it
    // bakes the file into the binary and may not represent real-world usage.
    // Prefer the mmap-backed benchmark for end-to-end measurement.
    let file_content = include_str!("../compile_benchmark_data.txt");

    let statement_count = file_content.lines().count();
    let start = Instant::now();
    let mut program = Program::new();
    let mut parser = Parser::new(file_content, 0);
    let mut reporter = ErrorReporter::new();
    reporter.add_source(0, file_content.to_string());

    let mut compiled_count = 0;
    let mut error_count = 0;
    if let Err(err) = program.compile_all(&mut parser) {
        error_count = 1;
        let _ = reporter.report_compile_error(&err);
    } else {
        compiled_count = statement_count;
    }

    let duration = start.elapsed();
    let total_iterations = compiled_count + error_count;
    let exprs_per_second = total_iterations as f64 / duration.as_secs_f64();

    println!("Results:");
    println!("  Statements in file: {}", statement_count);
    println!("  Successfully compiled: {}", compiled_count);
    println!("  Errors: {}", error_count);
    println!("  Total iterations: {}", total_iterations);
    println!("  Time: {:?}", duration);
    println!("  Statements per second: {:.2}", exprs_per_second);
    println!(
        "  Microseconds per statement: {:.2}",
        duration.as_micros() as f64 / total_iterations as f64
    );
    println!(
        "  Milliseconds per statement: {:.4}",
        duration.as_secs_f64() * 1000.0 / total_iterations as f64
    );
}
