use expr_parser::parsing::Parser;
use expr_parser::program::Program;
use std::time::Instant;

const ITERATIONS: usize = 60000;

const SOURCE: &str = r#"
add = fn(a, b) { a + b }
sub = fn(a, b) { a - b }
mul = fn(a, b) { a * b }
div = fn(a, b) { a / b }
neg = fn(x) { -x }

add(1, 2)
sub(10, 3)
mul(4, 5)
div(20, 4)
neg(7)

(1 + 2) * (3 + 4)
1 + 2 * 3 - 4 / 5
2 ^ 3 ^ 2

sin(1) + cos(2)
max(1, 2, 3, 4)
min(4, 3, 2, 1)

foo(1, 2, bar(3, 4))
((a + b) * (c - d)) / e

{ add(1, 2) sub(3, 4) mul(5, 6) }
{ neg(1) neg(2) neg(3) }

sum 1 2 3 4
1 2 3 4 5
f(x) g(y) h(z)

1 + (2 * (3 + (4 * 5)))
1 + (2 * (3 + (4 * (5 + 6))))
"#;

fn main() {
    println!("Running normal program benchmark (no macros)");

    let start = Instant::now();
    let mut parsed_count = 0usize;
    let mut error_count = 0usize;

    for _ in 0..ITERATIONS {
        let mut program = Program::new();
        let mut parser = Parser::new(SOURCE, 0);

        while !parser.is_empty() {
            match parser.compile_expr(&mut program, &mut |_| {}) {
                Ok(consumed) => {
                    if consumed {
                        parsed_count += 1;
                    } else {
                        break;
                    }
                }
                Err(_) => {
                    error_count += 1;
                    break;
                }
            }
        }
    }

    let duration = start.elapsed();
    let total_iterations = parsed_count + error_count;
    let exprs_per_second = total_iterations as f64 / duration.as_secs_f64();

    println!("Results:");
    println!("  Parsed expressions: {}", parsed_count);
    println!("  Errors: {}", error_count);
    println!("  Total iterations: {}", total_iterations);
    println!("  Time: {:?}", duration);
    println!("  Expressions per second: {:.2}", exprs_per_second);
    println!(
        "  Microseconds per expression: {:.2}",
        duration.as_micros() as f64 / total_iterations as f64
    );
}
