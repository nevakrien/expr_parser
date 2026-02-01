use expr_parser::parsing::Parser;
use std::time::Instant;

mod mapped_file;
use mapped_file::MappedFile;

fn main() {
    println!("Running file-based benchmark (single large file)");

    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "benchmark_data.txt".to_string());

    let mapping = match MappedFile::map(&path, "parser_input") {
        Ok(mapping) => mapping,
        Err(message) => {
            eprintln!("{message}");
            return;
        }
    };

    let file_content = match mapping.as_str() {
        Ok(content) => content,
        Err(message) => {
            eprintln!("{message}");
            return;
        }
    };

    let start = Instant::now();
    let mut parsed_count = 0;
    let mut error_count = 0;
    let mut expressions = Vec::new();

    let mut parser = Parser::new(file_content, 0);
    while !parser.is_empty() {
        match parser.consume_stmt() {
            Ok(expr) => {
                expressions.push(expr);
                parsed_count += 1;
            }
            Err(_) => {
                error_count += 1;
                break;
            }
        }
    }

    let duration = start.elapsed();
    let total_iterations = parsed_count + error_count;
    let exprs_per_second = total_iterations as f64 / duration.as_secs_f64();

    println!("Results:");
    println!("  Successfully parsed: {}", parsed_count);
    println!("  Errors: {}", error_count);
    println!("  Total iterations: {}", total_iterations);
    println!("  Time: {:?}", duration);
    println!("  Expressions per second: {:.2}", exprs_per_second);
    println!("  Stored expressions: {}", expressions.len());
    println!(
        "  Microseconds per expression: {:.2}",
        duration.as_micros() as f64 / total_iterations as f64
    );
}
