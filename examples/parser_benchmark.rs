use expr_parser::parsing::Parser;
use std::time::Instant;

fn main() {
    println!("Running in-memory benchmark (cached expressions)");

    // Benchmark parsing with cached expressions
    let start = Instant::now();
    let mut parsed_count = 0;
    let mut error_count = 0;

    for _ in 0..100000 {
        // Create expressions fresh each iteration (like file allocation)
        let lines = vec![
            "1 + 2 * 3",
            "(1 + 2) * 3",
            "1 + 2 + 3 + 4 + 5",
            "1 * 2 * 3 * 4 * 5",
            "1 + 2 * 3 - 4 / 5",
            "2 ^ 3 ^ 2",
            "sin(1) + cos(2)",
            "max(1, 2, 3, 4)",
            "a + b * c - d / e",
            "foo(1, 2, bar(3, 4))",
            "((a + b) * (c - d)) / e",
            "1 2 3 4 5",
            "a b c",
            "a, b, c",
            "a; b; c",
            "f(x) g(y) h(z)",
            "sum 1 2 3 4",
            "1 + -2 * +3",
            "1 + (2 * (3 + (4 * 5)))",
            "1 + (2 * (3 + (4 * (5 + 6))))",
        ];

        for line in &lines {
            let mut parser = Parser::new(line, 0);
            if parser.consume_expr().is_ok() {
                parsed_count += 1;
            } else {
                error_count += 1;
            }
        }
    }

    let duration = start.elapsed();
    let total_iterations = parsed_count + error_count;
    let lines_per_second = total_iterations as f64 / duration.as_secs_f64();

    println!("Results:");
    println!("  Successfully parsed: {}", parsed_count);
    println!("  Errors: {}", error_count);
    println!("  Total iterations: {}", total_iterations);
    println!("  Time: {:?}", duration);
    println!("  Expressions per second: {:.2}", lines_per_second);
    println!(
        "  Microseconds per expression: {:.2}",
        duration.as_micros() as f64 / total_iterations as f64
    );
}
