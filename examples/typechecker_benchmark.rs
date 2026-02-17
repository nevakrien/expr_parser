use expr_parser::ErrorReporter;
use expr_parser::parsing::Parser;
use expr_parser::program::Program;
use expr_parser::type_inference::run_typechecker;
use std::time::Instant;

const ITERATIONS: usize = 40000;

const SOURCE: &str = r#"
f = fn() -> i32 {
    let a: i32 = 1 + 2;
    let b = a + 3;
    let c = b + 4;
    c
}
"#;

fn main() {
    println!("Running typechecker benchmark (single simple function)");

    let start = Instant::now();
    let mut error_count = 0usize;
    let mut reporter = ErrorReporter::new();
    reporter.add_source(0, SOURCE.to_string());

    for _ in 0..ITERATIONS {
        let mut program = Program::new();
        let mut parser = Parser::new(SOURCE, 0);
        if let Err(errs) = program.lower_all(&mut parser) {
            error_count += errs.len();
            for err in errs {
                let _ = reporter.report_compile_error(&err);
            }
            continue;
        }

        if let (Err(ec), _) = run_typechecker(&program, &mut reporter).unwrap() {
            error_count += ec;
        }
    }

    let duration = start.elapsed();
    let exprs_per_second = ITERATIONS as f64 / duration.as_secs_f64();

    println!("Results:");
    println!("  Errors: {}", error_count);
    println!("  Total iterations: {}", ITERATIONS);
    println!("  Time: {:?}", duration);
    println!("  Iterations per second: {:.2}", exprs_per_second);
    println!(
        "  Microseconds per iteration: {:.2}",
        duration.as_micros() as f64 / ITERATIONS as f64
    );
}
