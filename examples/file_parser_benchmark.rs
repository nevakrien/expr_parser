use expr_parser::parsing::Parser;
use std::fs;
use std::time::Instant;

fn main() {
    println!("Running file-based benchmark (reading file each iteration)");

    // Benchmark parsing with file I/O each iteration
    let start = Instant::now();
    let mut parsed_count = 0;
    let mut error_count = 0;

    for _ in 0..10000 {
        // Read file fresh each iteration
        let file_content = match fs::read_to_string("benchmark_data.txt") {
            Ok(content) => content,
            Err(e) => {
                eprintln!("Error reading benchmark_data.txt: {}", e);
                return;
            }
        };

        let mut parser = Parser::new(&file_content, 0);
        while !parser.is_empty() {
            match parser.consume_stmt() {
                Ok(_) => parsed_count += 1,
                Err(_) => {
                    error_count += 1;
                    break;
                }
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
